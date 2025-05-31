# utils.py
"""
Utilities, logging, and helper functions for the Revolutionary Pipeline
"""

import torch
import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Optional
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """Comprehensive logging for training metrics"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.start_time = time.time()
        self.metrics_history = {
            'timesteps': [],
            'episodes': [],
            'rewards': [],
            'mentor_queries': [],
            'uncertainties': [],
            'losses': {},
        }

        # Running averages
        self.reward_window = deque(maxlen=100)
        self.query_window = deque(maxlen=100)

        # Log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")

    def log_step(self, timestep: int, metrics: Dict[str, Any]):
        """Log training step metrics"""
        self.metrics_history['timesteps'].append(timestep)

        # Process different metric types
        for key, value in metrics.items():
            if key == 'episode_reward':
                self.reward_window.append(value)
                self.metrics_history['rewards'].append(value)
            elif key == 'mentor_queries':
                self.query_window.append(value)
                self.metrics_history['mentor_queries'].append(value)
            elif key == 'uncertainty':
                self.metrics_history['uncertainties'].append(value)
            elif 'loss' in key:
                if key not in self.metrics_history['losses']:
                    self.metrics_history['losses'][key] = []
                self.metrics_history['losses'][key].append(value)

    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {
            'elapsed_time': time.time() - self.start_time,
            'total_episodes': len(self.metrics_history['rewards']),
            'avg_reward_100': np.mean(self.reward_window) if self.reward_window else 0,
            'max_reward': max(self.metrics_history['rewards']) if self.metrics_history['rewards'] else 0,
            'avg_queries_100': np.mean(self.query_window) if self.query_window else 0,
        }

        # Add loss statistics
        for loss_name, values in self.metrics_history['losses'].items():
            if values:
                stats[f'avg_{loss_name}'] = np.mean(values[-100:])

        return stats

    def save_checkpoint(self, models: Dict[str, torch.nn.Module],
                        optimizers: Dict[str, torch.optim.Optimizer],
                        timestep: int):
        """Save training checkpoint"""
        checkpoint = {
            'timestep': timestep,
            'metrics': self.metrics_history,
            'models': {},
            'optimizers': {},
        }

        for name, model in models.items():
            checkpoint['models'][name] = model.state_dict()

        for name, optimizer in optimizers.items():
            checkpoint['optimizers'][name] = optimizer.state_dict()

        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_{timestep}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.log(f"Saved checkpoint at timestep {timestep}")

    def plot_training_curves(self):
        """Generate training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        if self.metrics_history['rewards']:
            axes[0, 0].plot(self.metrics_history['rewards'], alpha=0.3)
            axes[0, 0].plot(np.convolve(self.metrics_history['rewards'],
                                        np.ones(100) / 100, mode='valid'),
                            label='100-ep avg')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()

        # Mentor queries
        if self.metrics_history['mentor_queries']:
            axes[0, 1].plot(self.metrics_history['mentor_queries'])
            axes[0, 1].set_title('Mentor Queries per Episode')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Queries')

        # Uncertainties
        if self.metrics_history['uncertainties']:
            axes[1, 0].plot(self.metrics_history['uncertainties'])
            axes[1, 0].set_title('Average Uncertainty')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Uncertainty')

        # Losses
        if self.metrics_history['losses']:
            for loss_name, values in self.metrics_history['losses'].items():
                if len(values) > 0:
                    axes[1, 1].plot(values, label=loss_name)
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()


class CurriculumScheduler:
    """Manages curriculum learning progression"""

    def __init__(self, stages: List[Dict[str, float]]):
        self.stages = stages
        self.current_stage = 0
        self.stage_start_time = time.time()

    def get_current_config(self, avg_reward: float) -> Dict[str, float]:
        """Get current curriculum configuration"""
        # Progress through stages based on performance
        while (self.current_stage < len(self.stages) - 1 and
               avg_reward >= self.stages[self.current_stage + 1]['min_reward']):
            self.current_stage += 1
            self.stage_start_time = time.time()

        return self.stages[self.current_stage]

    def should_query_mentor(self, uncertainty: float, base_threshold: float) -> bool:
        """Determine if mentor should be queried based on curriculum"""
        config = self.stages[self.current_stage]
        query_prob = config.get('mentor_query_prob', 0.5)

        # Combine uncertainty threshold with curriculum probability
        adjusted_threshold = base_threshold * (1.0 + query_prob)

        return uncertainty > adjusted_threshold or np.random.random() < query_prob * 0.1


class ActionProcessor:
    """Processes multi-action outputs for environment execution"""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def process_multi_actions(self, actions: List[List[int]],
                              coordination_weights: np.ndarray) -> np.ndarray:
        """
        Process multiple actions per agent into executable format
        For environments that don't support multi-action, combines them intelligently
        """
        batch_size = len(actions)
        processed_actions = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            agent_actions = actions[i]
            weights = coordination_weights[i] if i < len(coordination_weights) else [1.0]

            if len(agent_actions) == 1:
                # Single action
                processed_actions[i] = agent_actions[0]
            else:
                # Multiple actions - use weighted voting
                action_votes = np.zeros(self.num_actions)
                for action, weight in zip(agent_actions, weights):
                    if action < self.num_actions:
                        action_votes[action] += weight

                processed_actions[i] = np.argmax(action_votes)

        return processed_actions


def create_env_description(env_name: str) -> List[str]:
    """Generate textual description of environment for mentor"""
    descriptions = {
        'CartPole-v1': [
            "Cart-pole balancing task with 4D state space",
            "State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]",
            "Actions: 0=push_left, 1=push_right",
            "Goal: Keep pole upright (|angle| < 0.2095) and cart in bounds (|pos| < 2.4)",
            "Physics: Gravity=9.8, pole_mass=0.1, cart_mass=1.0",
            "Strategy: Apply force opposite to pole lean direction",
            "Critical angles require immediate correction",
            "Small angles allow for predictive control based on angular velocity",
        ],
    }

    return descriptions.get(env_name, ["Generic environment"])


def compute_intrinsic_reward(state: torch.Tensor, next_state: torch.Tensor,
                             uncertainty: float) -> float:
    """Compute intrinsic motivation reward based on learning progress"""
    # State difference as novelty measure
    state_change = torch.norm(next_state - state, dim=-1).mean().item()

    # Uncertainty-based exploration bonus
    exploration_bonus = uncertainty * 0.1

    # Learning progress reward
    intrinsic = state_change * 0.05 + exploration_bonus

    return min(intrinsic, 1.0)  # Cap intrinsic reward


def analyze_mentor_student_agreement(mentor_action: int, student_actions: List[int]) -> float:
    """Analyze agreement between mentor and student decisions"""
    if not student_actions:
        return 0.0

    # Check if mentor action appears in student's action list
    if mentor_action in student_actions:
        # Higher agreement if it's the primary action
        return 1.0 if student_actions[0] == mentor_action else 0.5

    return 0.0