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
            self.envs = gym.vector.make(env_name, num_envs=num_envs)
        else:
            self.envs = gym.make(env_name)

        # Environment properties
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space

        if hasattr(self.observation_space, 'shape'):
            self.state_dim = self.observation_space.shape[0]
        else:
            self.state_dim = self.observation_space.n

        if hasattr(self.action_space, 'n'):
            self.num_actions = self.action_space.n
        else:
            self.num_actions = self.action_space.shape[0]

        # Multi-action settings
        self.max_actions_per_step = 3
        self.action_execution_mode = 'sequential'  # 'sequential', 'parallel', 'weighted'

        # Environment description for mentor
        self.env_description = create_env_description(env_name)

        # Tracking
        self.episode_steps = np.zeros(num_envs) if num_envs > 1 else 0
        self.episode_rewards = np.zeros(num_envs) if num_envs > 1 else 0
        self.total_episodes = 0

        # Advanced features
        self.intrinsic_reward_scale = 0.1
        self.state_history = deque(maxlen=100)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment(s)"""
        if self.num_envs > 1:
            observations, infos = self.envs.reset(seed=seed)
            self.episode_steps = np.zeros(self.num_envs)
            self.episode_rewards = np.zeros(self.num_envs)
        else:
            observations, info = self.envs.reset(seed=seed)
            infos = [info]
            self.episode_steps = 0
            self.episode_rewards = 0

        self.state_history.clear()
        self.state_history.append(observations)

        return observations, infos

    def step(self, actions: List[List[int]],
             uncertainties: Optional[List[float]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute multi-actions in environment

        Args:
            actions: List of action lists, one per environment
            uncertainties: Optional uncertainty values for intrinsic reward

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        # Process multi-actions into executable format
        processed_actions = self._process_multi_actions(actions)

        # Execute in environment
        if self.num_envs > 1:
            observations, rewards, terminated, truncated, infos = self.envs.step(processed_actions)
        else:
            obs, reward, term, trunc, info = self.envs.step(processed_actions[0])
            observations = np.array([obs])
            rewards = np.array([reward])
            terminated = np.array([term])
            truncated = np.array([trunc])
            infos = [info]

        # Add intrinsic rewards
        if uncertainties is not None:
            intrinsic_rewards = self._compute_intrinsic_rewards(observations, uncertainties)
            rewards = rewards + intrinsic_rewards * self.intrinsic_reward_scale

        # Update tracking
        self._update_tracking(rewards, terminated, truncated, infos)

        # Store state history
        self.state_history.append(observations)

        return observations, rewards, terminated, truncated, infos

    def _process_multi_actions(self, actions: List[List[int]]) -> np.ndarray:
        """Process multi-actions based on execution mode"""
        if self.num_envs > 1:
            processed = np.zeros(self.num_envs, dtype=np.int32)
        else:
            processed = np.zeros(1, dtype=np.int32)

        for i, action_list in enumerate(actions):
            if len(action_list) == 1:
                # Single action
                processed[i] = action_list[0]
            else:
                # Multi-action processing
                if self.action_execution_mode == 'sequential':
                    # Execute primary action (first in list)
                    processed[i] = action_list[0]
                elif self.action_execution_mode == 'weighted':
                    # Weighted combination (for continuous actions)
                    processed[i] = action_list[0]  # Simplified for discrete
                elif self.action_execution_mode == 'parallel':
                    # For discrete actions, use majority vote
                    action_counts = np.bincount(action_list, minlength=self.num_actions)
                    processed[i] = np.argmax(action_counts)

        return processed

    def _compute_intrinsic_rewards(self, observations: np.ndarray,
                                   uncertainties: List[float]) -> np.ndarray:
        """Compute intrinsic motivation rewards"""
        if len(self.state_history) < 2:
            return np.zeros(len(uncertainties))

        prev_obs = self.state_history[-2]
        curr_obs = observations

        intrinsic_rewards = []
        for i in range(len(uncertainties)):
            if self.num_envs > 1:
                prev_state = torch.tensor(prev_obs[i], dtype=torch.float32)
                curr_state = torch.tensor(curr_obs[i], dtype=torch.float32)
            else:
                prev_state = torch.tensor(prev_obs, dtype=torch.float32)
                curr_state = torch.tensor(curr_obs, dtype=torch.float32)

            intrinsic = compute_intrinsic_reward(prev_state, curr_state, uncertainties[i])
            intrinsic_rewards.append(intrinsic)

        return np.array(intrinsic_rewards)

    def _update_tracking(self, rewards: np.ndarray, terminated: np.ndarray,
                         truncated: np.ndarray, infos: List[Dict]):
        """Update episode tracking"""
        if self.num_envs > 1:
            self.episode_rewards += rewards
            self.episode_steps += 1

            # Check for episode completions
            for i in range(self.num_envs):
                if terminated[i] or truncated[i]:
                    infos[i]['episode_reward'] = self.episode_rewards[i]
                    infos[i]['episode_length'] = self.episode_steps[i]
                    self.episode_rewards[i] = 0
                    self.episode_steps[i] = 0
                    self.total_episodes += 1
        else:
            self.episode_rewards += rewards[0]
            self.episode_steps += 1

            if terminated[0] or truncated[0]:
                infos[0]['episode_reward'] = self.episode_rewards
                infos[0]['episode_length'] = self.episode_steps
                self.episode_rewards = 0
                self.episode_steps = 0
                self.total_episodes += 1

    def get_state_tensor(self, observations: np.ndarray) -> torch.Tensor:
        """Convert observations to tensor format"""
        if self.num_envs > 1:
            return torch.tensor(observations, dtype=torch.float32).to(DEVICE)
        else:
            return torch.tensor(observations, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    def close(self):
        """Close environment"""
        if hasattr(self.envs, 'close'):
            self.envs.close()


class AdvancedRewardShaper:
    """
    Sophisticated reward shaping for enhanced learning
    """

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.step_count = 0
        self.prev_shaping = None

    def shape_reward(self, state: np.ndarray, action: int, reward: float,
                     done: bool, info: Dict) -> float:
        """Apply environment-specific reward shaping"""

        if self.env_name == 'CartPole-v1':
            return self._cartpole_shaping(state, action, reward, done)

        return reward

    def _cartpole_shaping(self, state: np.ndarray, action: int,
                          reward: float, done: bool) -> float:
        """CartPole-specific reward shaping"""
        if len(state) < 4:
            return reward

        pos, vel, angle, ang_vel = state[:4]

        # Potential-based shaping (maintains optimal policy)
        angle_potential = -abs(angle) * 10  # Encourage upright position
        position_potential = -abs(pos) * 2  # Encourage center position
        velocity_potential = -abs(vel) * 0.1  # Encourage stability

        current_shaping = angle_potential + position_potential + velocity_potential

        if self.prev_shaping is not None:
            shaped_reward = reward + (current_shaping - self.prev_shaping)
        else:
            shaped_reward = reward

        self.prev_shaping = current_shaping

        # Reset shaping on episode end
        if done:
            self.prev_shaping = None

        return shaped_reward


def create_environment(env_name: str = None, num_envs: int = None) -> MultiActionEnvironment:
    """Factory function to create configured environment"""
    if env_name is None:
        env_name = ENV_CONFIG['name']
    if num_envs is None:
        num_envs = ENV_CONFIG['num_envs']

    return MultiActionEnvironment(env_name, num_envs)