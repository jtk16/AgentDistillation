# environment.py
"""
Advanced Environment Wrapper for Revolutionary AI Pipeline
Supports multi-action execution and sophisticated reward shaping
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from config import ENV_CONFIG, DEVICE
from utils import compute_intrinsic_reward, create_env_description


class MultiActionEnvironment:
    """
    Wrapper that enables multi-action execution in single timesteps
    """

    def __init__(self, env_name: str, num_envs: int = 1):
        self.env_name = env_name
        self.num_envs = num_envs

        # Create vectorized environments
        if num_envs > 1:
            self.envs = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync")
            self.single_observation_space = self.envs.single_observation_space
            self.single_action_space = self.envs.single_action_space
        else:
            self.envs = gym.make(env_name)
            self.single_observation_space = self.envs.observation_space
            self.single_action_space = self.envs.action_space

        # Environment properties (now derived from single_*)
        self.observation_space = self.envs.observation_space  # This is the space for the (potentially) vectorized env
        self.action_space = self.envs.action_space  # This is the space for the (potentially) vectorized env

        # state_dim and num_actions should reflect a single environment
        if hasattr(self.single_observation_space, 'shape') and self.single_observation_space.shape is not None:
            self.state_dim = self.single_observation_space.shape[0] if len(
                self.single_observation_space.shape) > 0 else 1
        elif hasattr(self.single_observation_space, 'n'):  # Discrete observation space
            self.state_dim = self.single_observation_space.n
        else:
            print(
                f"Warning: Could not determine state_dim reliably from single_observation_space: {self.single_observation_space}")
            self.state_dim = 1

        if hasattr(self.single_action_space, 'n'):
            self.num_actions = self.single_action_space.n
        elif hasattr(self.single_action_space, 'shape'):  # Continuous
            self.num_actions = self.single_action_space.shape[0]
        else:
            print(
                f"Warning: Could not determine num_actions reliably from single_action_space: {self.single_action_space}")
            self.num_actions = 1

        # Multi-action settings
        self.max_actions_per_step = 3
        self.action_execution_mode = 'sequential'  # 'sequential', 'parallel', 'weighted'

        # Environment description for mentor
        self.env_description = create_env_description(env_name)

        # Tracking
        self.episode_steps = np.zeros(num_envs, dtype=int) if num_envs > 1 else 0
        self.episode_rewards = np.zeros(num_envs, dtype=float) if num_envs > 1 else 0.0
        self.total_episodes = 0

        # Advanced features
        self.intrinsic_reward_scale = 0.1
        self.state_history = deque(maxlen=100)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment(s)"""
        if self.num_envs > 1:
            # For make_vec, seed might need to be a list if you want different seeds per sub-env,
            # or a single int to seed them based on this. Gymnasium handles this.
            observations, infos = self.envs.reset(seed=seed)
            self.episode_steps.fill(0)
            self.episode_rewards.fill(0.0)
        else:
            # Ensure seed is passed correctly for single env
            # Gymnasium's single env reset takes a single seed.
            observations, info = self.envs.reset(seed=seed)
            infos = {
                0: info}  # Keep a similar structure for infos if possible, though not strictly necessary for single env
            self.episode_steps = 0
            self.episode_rewards = 0.0

        self.state_history.clear()
        if observations is not None:
            self.state_history.append(observations.copy())  # Store a copy

        return observations, infos

    def step(self, actions: List[List[int]],
             uncertainties: Optional[List[float]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute multi-actions in environment
        Args:
            actions: List of action lists, one per environment. E.g. [[0], [1]] for 2 envs.
                     For single env, it would be [[action_for_env0]].
            uncertainties: Optional uncertainty values for intrinsic reward, one per env.
        Returns:
            observations, rewards, terminated, truncated, infos (List of Dicts, one per env)
        """
        processed_actions_np = self._process_multi_actions(actions)

        if self.num_envs > 1:
            observations, rewards, terminated, truncated, infos_from_gym = self.envs.step(processed_actions_np)
            # Construct a list of info dicts for consistency
            infos_list = [{} for _ in range(self.num_envs)]
            for i in range(self.num_envs):
                # Gymnasium make_vec often packs final info into the main infos_from_gym dict
                if terminated[i] or truncated[i]:
                    if infos_from_gym.get("_final_observation", [False] * self.num_envs)[i]:
                        infos_list[i]["final_observation"] = infos_from_gym["final_observation"][i]
                    if infos_from_gym.get("_final_info", [False] * self.num_envs)[i]:
                        infos_list[i].update(infos_from_gym["final_info"][i])
        else:
            obs_single, reward_single, term_single, trunc_single, info_single = self.envs.step(processed_actions_np[0])
            observations = np.array([obs_single])
            rewards = np.array([reward_single])
            terminated = np.array([term_single])
            truncated = np.array([trunc_single])
            infos_list = [info_single]

        if uncertainties is not None and len(uncertainties) == self.num_envs:
            intrinsic_rewards = self._compute_intrinsic_rewards(observations, uncertainties)
            rewards = rewards + intrinsic_rewards * self.intrinsic_reward_scale

        self._update_tracking(rewards, terminated, truncated, infos_list)

        if observations is not None:
            self.state_history.append(observations.copy())  # Store a copy

        return observations, rewards, terminated, truncated, infos_list

    def _process_multi_actions(self, actions: List[List[int]]) -> np.ndarray:
        processed_np = np.zeros(self.num_envs, dtype=self.single_action_space.dtype if hasattr(self.single_action_space,
                                                                                               'dtype') else np.int32)

        for i, action_list_for_env_i in enumerate(actions):
            if not action_list_for_env_i:
                processed_np[i] = 0  # Default action
                continue

            if len(action_list_for_env_i) == 1:
                processed_np[i] = action_list_for_env_i[0]
            else:
                if self.action_execution_mode == 'sequential':
                    processed_np[i] = action_list_for_env_i[0]
                elif self.action_execution_mode == 'weighted':
                    processed_np[i] = action_list_for_env_i[0]  # Simplified for discrete
                elif self.action_execution_mode == 'parallel':
                    action_counts = np.bincount(action_list_for_env_i, minlength=self.num_actions)
                    processed_np[i] = np.argmax(action_counts)
                else:
                    processed_np[i] = action_list_for_env_i[0]
        return processed_np

    def _compute_intrinsic_rewards(self, current_observations_np: np.ndarray,
                                   uncertainties: List[float]) -> np.ndarray:
        if len(self.state_history) < 2 or current_observations_np is None:
            return np.zeros(self.num_envs)

        prev_observations_np = self.state_history[-2]
        intrinsic_rewards_np = np.zeros(self.num_envs)

        for i in range(self.num_envs):
            prev_state_tensor_single = torch.tensor(prev_observations_np[i], dtype=torch.float32, device=DEVICE)
            curr_state_tensor_single = torch.tensor(current_observations_np[i], dtype=torch.float32, device=DEVICE)
            intrinsic_rewards_np[i] = compute_intrinsic_reward(prev_state_tensor_single, curr_state_tensor_single,
                                                               uncertainties[i])
        return intrinsic_rewards_np

    def _update_tracking(self, rewards_np: np.ndarray, terminated_np: np.ndarray,
                         truncated_np: np.ndarray, infos_list: List[Dict]):
        self.episode_rewards += rewards_np
        self.episode_steps += 1

        for i in range(self.num_envs):
            if terminated_np[i] or truncated_np[i]:
                if 'episode' not in infos_list[i]:  # Gymnasium often uses 'episode' key in info for final stats
                    infos_list[i]['episode'] = {}
                infos_list[i]['episode']['r'] = self.episode_rewards[i]  # Total reward for the episode
                infos_list[i]['episode']['l'] = self.episode_steps[i]  # Total length of the episode
                # Store for logger if not already there
                infos_list[i]['episode_reward'] = self.episode_rewards[i]
                infos_list[i]['episode_length'] = self.episode_steps[i]

                self.episode_rewards[i] = 0.0
                self.episode_steps[i] = 0
                self.total_episodes += 1

    def get_state_tensor(self, observations_np: np.ndarray) -> torch.Tensor:
        if self.num_envs == 1 and observations_np.ndim == len(
                self.single_observation_space.shape if hasattr(self.single_observation_space, 'shape') else [1]):
            observations_np = np.expand_dims(observations_np, axis=0)
        return torch.tensor(observations_np, dtype=torch.float32).to(DEVICE)

    def close(self):
        if hasattr(self.envs, 'close'):
            self.envs.close()


class AdvancedRewardShaper:
    def __init__(self, env_name: str, num_envs: int = 1):  # Added num_envs
        self.env_name = env_name
        self.num_envs = num_envs
        # Initialize prev_shaping_potential based on num_envs
        self.prev_shaping_potential = np.zeros(num_envs) if num_envs > 1 else None

    def shape_reward(self, state_np: np.ndarray, action: int, reward: float,
                     done: bool, info: Dict, env_idx: int = 0) -> float:  # Added env_idx
        """Apply environment-specific reward shaping for a specific environment step."""
        if self.env_name == 'CartPole-v1':
            return self._cartpole_shaping(state_np, action, reward, done, env_idx)
        return reward

    def _cartpole_shaping(self, state_np: np.ndarray, action: int,
                          reward: float, done: bool, env_idx: int) -> float:
        if len(state_np) < 4: return reward

        pos, vel, angle, ang_vel = state_np
        angle_potential = -abs(angle) * 10.0
        position_potential = -abs(pos) * 2.0
        current_potential = angle_potential + position_potential

        shaped_reward_component = 0.0
        gamma = TRAINING_CONFIG.get('gamma', 0.99)  # Get gamma from config

        prev_potential_for_env = None
        if self.num_envs > 1:
            prev_potential_for_env = self.prev_shaping_potential[env_idx]
        else:
            prev_potential_for_env = self.prev_shaping_potential

        if prev_potential_for_env is not None:  # Check if it was set (not first step of episode)
            shaped_reward_component = gamma * current_potential - prev_potential_for_env

        # Store current potential for next step
        if self.num_envs > 1:
            self.prev_shaping_potential[env_idx] = current_potential
        else:
            self.prev_shaping_potential = current_potential

        if done:
            if self.num_envs > 1:
                self.prev_shaping_potential[env_idx] = 0.0  # Reset specific env's potential
            else:
                self.prev_shaping_potential = None
        return reward + shaped_reward_component


def create_environment(env_name: str = None, num_envs: int = None) -> MultiActionEnvironment:
    if env_name is None:
        env_name = ENV_CONFIG['name']
    if num_envs is None:
        num_envs = ENV_CONFIG['num_envs']
    return MultiActionEnvironment(env_name, num_envs)