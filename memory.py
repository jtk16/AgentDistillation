# memory.py
"""
Experience Replay Buffer and Trajectory Memory for Revolutionary Pipeline
"""

import torch
import numpy as np
from collections import deque, namedtuple
from typing import List, Optional, Tuple, Dict, Any
import heapq
import random

from config import DEVICE, MEMORY_CONFIG

# Experience tuple
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done',
    'log_prob', 'value', 'uncertainty', 'mentor_advice'
])


class PrioritizedReplayBuffer:
    """Prioritized experience replay for efficient learning"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.00001

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience with priority"""
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if self.size == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6


class TrajectoryBuffer:
    """Stores complete trajectories for learning from successful episodes"""

    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.success_threshold = MEMORY_CONFIG['min_trajectory_reward']

        # Statistics
        self.total_trajectories = 0
        self.successful_trajectories = 0

    def add_trajectory(self, experiences: List[Experience], total_reward: float):
        """Add a complete trajectory"""
        self.total_trajectories += 1

        if total_reward >= self.success_threshold:
            self.successful_trajectories += 1
            trajectory = {
                'experiences': experiences,
                'total_reward': total_reward,
                'length': len(experiences),
                'success': True,
            }
            self.trajectories.append(trajectory)

    def get_best_trajectories(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get k best trajectories by reward"""
        if not self.trajectories:
            return []

        sorted_trajectories = sorted(
            self.trajectories,
            key=lambda x: x['total_reward'],
            reverse=True
        )
        return sorted_trajectories[:k]

    def sample_trajectory(self) -> Optional[Dict[str, Any]]:
        """Sample a random successful trajectory"""
        if not self.trajectories:
            return None
        return random.choice(self.trajectories)

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if not self.trajectories:
            return {'success_rate': 0.0, 'avg_reward': 0.0, 'avg_length': 0.0}

        rewards = [t['total_reward'] for t in self.trajectories]
        lengths = [t['length'] for t in self.trajectories]

        return {
            'success_rate': self.successful_trajectories / max(1, self.total_trajectories),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
        }


class ExperienceCollector:
    """Collects and processes experiences during rollouts"""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        """Reset collectors for new rollout"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.uncertainties = []
        self.mentor_advices = []

        # Per-environment episode tracking
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        self.current_episodes = [[] for _ in range(self.num_envs)]

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
            next_state: np.ndarray, done: np.ndarray, log_prob: torch.Tensor,
            value: torch.Tensor, uncertainty: Dict[str, float],
            mentor_advice: Optional[Any] = None):
        """Add step data from vectorized environment"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.uncertainties.append(uncertainty)
        self.mentor_advices.append(mentor_advice)

        # Track episodes
        for i in range(self.num_envs):
            exp = Experience(
                state=torch.tensor(state[i], dtype=torch.float32),
                action=action[i],
                reward=reward[i],
                next_state=torch.tensor(next_state[i], dtype=torch.float32),
                done=done[i],
                log_prob=log_prob[i] if isinstance(log_prob, torch.Tensor) else log_prob,
                value=value[i] if isinstance(value, torch.Tensor) else value,
                uncertainty=uncertainty,
                mentor_advice=mentor_advice
            )

            self.current_episodes[i].append(exp)
            self.episode_rewards[i] += reward[i]
            self.episode_lengths[i] += 1

    def get_completed_episodes(self) -> List[Tuple[List[Experience], float]]:
        """Get completed episodes and reset them"""
        completed = []

        for i in range(self.num_envs):
            if self.dones[-1][i]:  # Episode completed
                completed.append((
                    self.current_episodes[i].copy(),
                    self.episode_rewards[i]
                ))

                # Reset this environment's tracking
                self.current_episodes[i] = []
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        return completed

    def get_batch_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert collected data to tensors for training"""
        if not self.states:
            return {}

        # Stack all data
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(self.next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32).to(DEVICE)

        # Handle variable-length tensor data
        if isinstance(self.log_probs[0], torch.Tensor):
            log_probs = torch.stack(self.log_probs).to(DEVICE)
        else:
            log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(DEVICE)

        if isinstance(self.values[0], torch.Tensor):
            values = torch.stack(self.values).to(DEVICE)
        else:
            values = torch.tensor(self.values, dtype=torch.float32).to(DEVICE)

        return {
            'states': states.view(-1, states.shape[-1]),
            'actions': actions.view(-1),
            'rewards': rewards.view(-1),
            'next_states': next_states.view(-1, next_states.shape[-1]),
            'dones': dones.view(-1),
            'log_probs': log_probs.view(-1),
            'values': values.view(-1),
        }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation"""
    device = rewards.device
    batch_size = rewards.shape[0]

    advantages = torch.zeros_like(rewards).to(device)
    returns = torch.zeros_like(rewards).to(device)

    # Bootstrap from last value
    next_value = values[-1]
    gae = 0

    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t]
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns