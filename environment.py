# environment.py
"""
Advanced Environment Wrapper for Revolutionary AI Pipeline
Supports multi-action execution and sophisticated reward shaping
"""

import gymnasium as gym  # MODIFIED: Explicitly import gymnasium
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union  # MODIFIED: Added Union
from collections import deque

from config import ENV_CONFIG, DEVICE, TRAINING_CONFIG
from utils import compute_intrinsic_reward, create_env_description


class MultiActionEnvironment:
    """
    Wrapper that enables multi-action execution in single timesteps
    """

    def __init__(self, env_name: str, num_envs: int = 1):
        self.env_name = env_name
        self.num_envs = max(1, num_envs)  # Ensure num_envs is at least 1

        # Create vectorized environments
        # Gymnasium's make_vec handles num_envs=1 by returning a "vectorized" wrapper around a single env.
        self.envs = gym.make_vec(env_name, num_envs=self.num_envs, vectorization_mode="sync")

        # Properties from the single underlying environment
        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space

        # Overall observation and action space for the vectorized environment
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space

        if hasattr(self.single_observation_space, 'shape') and self.single_observation_space.shape is not None:
            self.state_dim = self.single_observation_space.shape[0] if len(
                self.single_observation_space.shape) > 0 else 1
        elif hasattr(self.single_observation_space, 'n'):
            self.state_dim = self.single_observation_space.n  # Or handle as 1 if one-hot encoding is not done by default
        else:
            print(
                f"Warning: Could not determine state_dim from single_observation_space: {self.single_observation_space}. Defaulting to 1.")
            self.state_dim = 1

        if hasattr(self.single_action_space, 'n'):
            self.num_actions = self.single_action_space.n
        elif hasattr(self.single_action_space, 'shape'):
            self.num_actions = self.single_action_space.shape[0]
        else:
            print(
                f"Warning: Could not determine num_actions from single_action_space: {self.single_action_space}. Defaulting to 1.")
            self.num_actions = 1

        self.max_actions_per_step = 3
        self.action_execution_mode = 'sequential'

        self.env_description = create_env_description(env_name)

        # MODIFIED: Ensure episode_steps and episode_rewards are always numpy arrays
        self.episode_steps = np.zeros(self.num_envs, dtype=int)
        self.episode_rewards = np.zeros(self.num_envs, dtype=float)
        # END MODIFIED

        self.total_episodes = 0
        self.intrinsic_reward_scale = 0.1
        self.state_history: deque = deque(maxlen=100)  # Type hint for clarity

    def reset(self, seed: Optional[int] = None) -> Tuple[
        np.ndarray, Union[List[Dict[Any, Any]], Dict[Any, Any]]]:  # MODIFIED: Return type hint
        """Reset environment(s)"""
        # For gym.make_vec, a single seed will appropriately seed the sub-environments.
        observations, infos = self.envs.reset(seed=seed)

        self.episode_steps.fill(0)
        self.episode_rewards.fill(0.0)
        self.state_history.clear()
        if observations is not None:
            self.state_history.append(observations.copy())

        # make_vec for num_envs=1 still returns infos in a specific structure, often a dict with list-like access for final_info
        # For consistency, we aim to return a list of info dicts if possible, or the direct info if it's for a single env not truly vectorized.
        # However, the gym.vector.SyncVectorEnv returns a dict of arrays for infos.
        # The 'final_info' key in this dict contains a list of dicts for terminated environments.
        # For simplicity in the return type, we pass `infos` directly. `_update_tracking` will handle its structure.
        return observations, infos

    def step(self, actions: List[List[int]],
             uncertainties: Optional[List[float]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[
            List[Dict[Any, Any]], Dict[Any, Any]]]:  # MODIFIED: Return type hint for infos
        """
        Execute multi-actions in environment
        """
        processed_actions_np = self._process_multi_actions(actions)

        observations, rewards, terminated, truncated, infos_from_gym = self.envs.step(processed_actions_np)

        # infos_from_gym from SyncVectorEnv is a dict where values are arrays/lists.
        # Example: infos_from_gym['final_info'] is a list where each element is a dict for a terminated env or None.
        # We will pass infos_from_gym directly to _update_tracking, which will populate individual info dicts.

        if uncertainties is not None and len(uncertainties) == self.num_envs:
            intrinsic_rewards = self._compute_intrinsic_rewards(observations, uncertainties)
            rewards = rewards + intrinsic_rewards * self.intrinsic_reward_scale

        # We need to construct a list of info dicts from infos_from_gym for consistent processing later if needed,
        # or rely on _update_tracking to correctly populate episode stats into infos_from_gym structure.
        # For now, _update_tracking will modify/use infos_from_gym which is passed along.
        self._update_tracking(rewards, terminated, truncated, infos_from_gym)

        if observations is not None:
            self.state_history.append(observations.copy())

        return observations, rewards, terminated, truncated, infos_from_gym

    def _process_multi_actions(self, actions: List[List[int]]) -> np.ndarray:
        # Assumes self.single_action_space.dtype exists and is suitable for discrete actions
        action_dtype = self.single_action_space.dtype if hasattr(self.single_action_space, 'dtype') else np.int32
        processed_np = np.zeros(self.num_envs, dtype=action_dtype)

        for i, action_list_for_env_i in enumerate(actions):
            if not action_list_for_env_i:  # Default action if empty list
                processed_np[i] = 0  # Or some other valid default action index
                continue

            # Current logic: takes the first action from the list for each environment
            processed_np[i] = action_list_for_env_i[0]
            # More complex multi-action processing (e.g., voting, sequential execution within a step)
            # would require significant changes here and in how the environment handles actions.
            # For now, this matches the original intent of supporting a list but using the primary.
        return processed_np

    def _compute_intrinsic_rewards(self, current_observations_np: np.ndarray,
                                   uncertainties: List[float]) -> np.ndarray:
        if len(self.state_history) < 2 or current_observations_np is None:
            return np.zeros(self.num_envs)

        prev_observations_np = self.state_history[-2]  # Get previous state for all envs
        intrinsic_rewards_np = np.zeros(self.num_envs)

        for i in range(self.num_envs):
            # Ensure indexing is correct for observations_np which is (num_envs, obs_dim)
            prev_state_tensor_single = torch.tensor(prev_observations_np[i], dtype=torch.float32, device=DEVICE)
            curr_state_tensor_single = torch.tensor(current_observations_np[i], dtype=torch.float32, device=DEVICE)
            intrinsic_rewards_np[i] = compute_intrinsic_reward(prev_state_tensor_single, curr_state_tensor_single,
                                                               uncertainties[i])
        return intrinsic_rewards_np

    def _update_tracking(self, rewards_np: np.ndarray, terminated_np: np.ndarray,
                         truncated_np: np.ndarray,
                         infos_from_gym: Dict[str, Any]):  # MODIFIED: infos_from_gym is a dict
        """
        Updates episode rewards and lengths.
        Modifies infos_from_gym to include 'episode_reward' and 'episode_length'
        if an episode terminated or was truncated.
        """
        self.episode_rewards += rewards_np
        self.episode_steps += 1  # This increments step count for all active envs

        # gymnasium.vector.SyncVectorEnv puts final info in infos_from_gym['_final_info'] and infos_from_gym['final_info']
        # _final_info is a boolean array indicating which envs have final_info
        # final_info is an array of dicts, with actual info for terminated envs

        final_info_flags = infos_from_gym.get('_final_info', np.array([False] * self.num_envs))
        actual_final_infos = infos_from_gym.get('final_info', [{} for _ in range(self.num_envs)])

        for i in range(self.num_envs):
            if terminated_np[i] or truncated_np[i]:
                # Try to get the episode stats from the final_info if available
                env_final_info = {}
                if final_info_flags[i] and i < len(actual_final_infos) and actual_final_infos[i] is not None:
                    env_final_info = actual_final_infos[i]

                # If 'episode' key already exists (e.g. from RecordEpisodeStatistics wrapper), use it
                if 'episode' in env_final_info and isinstance(env_final_info['episode'], dict):
                    # Gymnasium's RecordEpisodeStatistics provides 'r', 'l', 't'
                    infos_from_gym.setdefault('episode_reward', np.zeros(self.num_envs))[i] = env_final_info['episode'][
                        'r']
                    infos_from_gym.setdefault('episode_length', np.zeros(self.num_envs, dtype=int))[i] = \
                    env_final_info['episode']['l']
                else:
                    # Fallback: use our tracked rewards and steps
                    infos_from_gym.setdefault('episode_reward', np.zeros(self.num_envs))[i] = self.episode_rewards[i]
                    infos_from_gym.setdefault('episode_length', np.zeros(self.num_envs, dtype=int))[i] = \
                    self.episode_steps[i]

                # Also, ensure the main 'final_info' list contains the episode stats if it was populated for this env
                if final_info_flags[i] and i < len(actual_final_infos) and actual_final_infos[i] is not None:
                    if 'episode' not in actual_final_infos[i]:
                        actual_final_infos[i]['episode'] = {}
                    actual_final_infos[i]['episode']['r'] = self.episode_rewards[i]
                    actual_final_infos[i]['episode']['l'] = self.episode_steps[i]
                elif final_info_flags[i]:  # if flag is true but actual_final_infos[i] was None or not dict
                    # This case might happen if the underlying wrapper stack is different.
                    # We should ensure actual_final_infos[i] is a dict before assigning to it.
                    if i < len(actual_final_infos):  # check bounds
                        if actual_final_infos[i] is None: actual_final_infos[i] = {}
                        if 'episode' not in actual_final_infos[i]: actual_final_infos[i]['episode'] = {}
                        actual_final_infos[i]['episode']['r'] = self.episode_rewards[i]
                        actual_final_infos[i]['episode']['l'] = self.episode_steps[i]

                self.episode_rewards[i] = 0.0  # Reset for the next episode for this env
                self.episode_steps[i] = 0  # Reset for the next episode for this env
                self.total_episodes += 1

    def get_state_tensor(self, observations_np: np.ndarray) -> torch.Tensor:
        # observations_np from make_vec is already (num_envs, obs_dim)
        # or (obs_dim,) if num_envs=1 AND the wrapper for num_envs=1 doesn't batch it.
        # However, make_vec(num_envs=1) usually still gives (1, obs_dim).
        # If it's (obs_dim,), we need to unsqueeze.
        if self.num_envs == 1 and observations_np.ndim == len(
                self.single_observation_space.shape if hasattr(self.single_observation_space,
                                                               'shape') and self.single_observation_space.shape else [
                    1]):
            observations_np = np.expand_dims(observations_np, axis=0)  # Ensure batch dim for single env
        return torch.tensor(observations_np, dtype=torch.float32).to(DEVICE)

    def close(self):
        if hasattr(self.envs, 'close'):
            self.envs.close()


class AdvancedRewardShaper:
    def __init__(self, env_name: str, num_envs: int = 1):
        self.env_name = env_name
        self.num_envs = max(1, num_envs)  # Ensure num_envs is at least 1
        # MODIFIED: Initialize prev_shaping_potential as a numpy array always
        self.prev_shaping_potential = np.zeros(self.num_envs)
        self.needs_reset_potential = np.ones(self.num_envs,
                                             dtype=bool)  # Track if potential needs reset (e.g. after episode done)

    def shape_reward(self, state_np_single_env: np.ndarray, action: int, reward: float,
                     done: bool, info: Dict, env_idx: int = 0) -> float:
        """Apply environment-specific reward shaping for a specific environment step."""
        if self.env_name == 'CartPole-v1':
            return self._cartpole_shaping(state_np_single_env, action, reward, done, env_idx)
        return reward

    def _cartpole_shaping(self, state_np_single_env: np.ndarray, action: int,
                          original_reward: float, done: bool, env_idx: int) -> float:
        if len(state_np_single_env) < 4: return original_reward  # Should not happen for CartPole

        pos, _, angle, _ = state_np_single_env  # vel, ang_vel not used in this simple potential

        # Potential based on how upright the pole is and how centered the cart is
        # More negative potential for worse states
        angle_potential = -abs(angle) * 5.0  # Penalize larger angles
        position_potential = -abs(pos) * 0.5  # Penalize being off-center

        current_potential_for_env = angle_potential + position_potential

        shaped_reward_component = 0.0
        gamma = TRAINING_CONFIG.get('gamma', 0.99)

        if self.needs_reset_potential[env_idx]:  # First step of a new episode for this env
            self.prev_shaping_potential[env_idx] = current_potential_for_env  # Initialize potential
            self.needs_reset_potential[env_idx] = False
            # No shaping reward on the very first step after reset
        else:
            shaped_reward_component = gamma * current_potential_for_env - self.prev_shaping_potential[env_idx]
            self.prev_shaping_potential[env_idx] = current_potential_for_env

        if done:  # Episode ended for this env
            self.needs_reset_potential[env_idx] = True  # Mark for reset on next call for this env_idx
            # Optionally, add a large penalty if it was a failure state for CartPole
            if original_reward < ENV_CONFIG.get('max_episode_steps', 500) - 5:  # Heuristic for failure vs success
                shaped_reward_component -= 1.0  # Extra penalty for failure

        return original_reward + shaped_reward_component


def create_environment(env_name: str = None, num_envs: int = None) -> MultiActionEnvironment:
    resolved_env_name = env_name if env_name is not None else ENV_CONFIG['name']
    resolved_num_envs = num_envs if num_envs is not None else ENV_CONFIG['num_envs']
    return MultiActionEnvironment(resolved_env_name, resolved_num_envs)