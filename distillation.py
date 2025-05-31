# distillation.py
"""
Progressive Knowledge Distillation from Mentor to Student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import DEVICE, DISTILLATION_CONFIG, TRAINING_CONFIG, STUDENT_CONFIG


class FeatureProjector(nn.Module):
    """Projects mentor features to student feature space"""

    def __init__(self, mentor_dim: int, student_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(mentor_dim, (mentor_dim + student_dim) // 2),
            nn.LayerNorm((mentor_dim + student_dim) // 2),
            nn.GELU(),
            nn.Linear((mentor_dim + student_dim) // 2, student_dim),
            nn.LayerNorm(student_dim),
        )

    def forward(self, mentor_features: torch.Tensor) -> torch.Tensor:
        return self.projector(mentor_features)


class ProgressiveDistillationLoss(nn.Module):
    """Progressive distillation loss that adapts over training"""

    def __init__(self, initial_temp: float = 4.0):
        super().__init__()
        self.temperature = initial_temp
        self.min_temperature = 1.0
        self.decay_rate = 0.99995

    def forward(self, student_logits: torch.Tensor, mentor_logits: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled KL divergence"""
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        mentor_probs = F.softmax(mentor_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(student_log_probs, mentor_probs, reduction='batchmean')

        # Scale by temperature squared (as per Hinton et al.)
        return kl_loss * (self.temperature ** 2)

    def update_temperature(self):
        """Decay temperature over time"""
        self.temperature = max(self.min_temperature, self.temperature * self.decay_rate)


class DistillationTrainer:
    """
    Handles knowledge distillation from mentor to student with dark knowledge transfer
    """

    def __init__(self, mentor: nn.Module, student: nn.Module,
                 mentor_hidden_dim: int, student_hidden_dim: int):
        self.mentor = mentor
        self.student = student

        # Feature projector for dimension matching
        self.feature_projector = FeatureProjector(mentor_hidden_dim, student_hidden_dim).to(DEVICE)
        self.feature_optimizer = torch.optim.Adam(self.feature_projector.parameters(), lr=1e-4)

        # Progressive distillation loss
        self.distillation_loss = ProgressiveDistillationLoss(DISTILLATION_CONFIG['temperature'])

        # Tracking
        self.distillation_steps = 0
        self.alpha = DISTILLATION_CONFIG['alpha']  # Balance between RL and distillation

    def compute_distillation_loss(self, states: torch.Tensor,
                                  student_outputs: Dict[str, torch.Tensor],
                                  mentor_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute various distillation losses"""
        losses = {}

        # Get mentor outputs if not provided
        if mentor_outputs is None:
            with torch.no_grad():
                mentor_outputs = self.mentor(states)

        # 1. Policy distillation (dark knowledge)
        policy_loss = self.distillation_loss(
            student_outputs['primary_logits'],
            mentor_outputs['policy_logits']
        )
        losses['policy_distill'] = policy_loss

        # 2. Feature matching loss
        mentor_features = mentor_outputs['features']
        projected_features = self.feature_projector(mentor_features)
        feature_loss = F.mse_loss(student_outputs['features'], projected_features)
        losses['feature_match'] = feature_loss * DISTILLATION_CONFIG['feature_matching_weight']

        # 3. Value function distillation
        value_loss = F.mse_loss(
            student_outputs['value'],
            mentor_outputs['value']
        )
        losses['value_distill'] = value_loss * 0.5

        # 4. Causal understanding transfer (if available)
        if 'causal_features' in mentor_outputs:
            # Project causal features
            causal_projected = self.feature_projector(mentor_outputs['causal_features'])
            causal_loss = F.mse_loss(student_outputs['features'], causal_projected)
            losses['causal_transfer'] = causal_loss * 0.1

        # Total distillation loss
        losses['total_distill'] = sum(losses.values())

        return losses

    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, advantages: torch.Tensor,
                   old_log_probs: torch.Tensor, values: torch.Tensor,
                   mentor_advice: Optional[List] = None) -> Dict[str, float]:
        """Single training step combining RL and distillation"""

        # Get mentor guidance
        with torch.no_grad():
            mentor_outputs = self.mentor(states)

        # Student forward pass
        student_outputs = self.student(states, mentor_features=mentor_outputs['features'])

        # RL losses (PPO)
        rl_losses = self._compute_rl_losses(
            student_outputs, actions, rewards, advantages, old_log_probs, values
        )

        # Distillation losses
        distill_losses = self.compute_distillation_loss(states, student_outputs, mentor_outputs)

        # Progressive weighting
        beta = DISTILLATION_CONFIG['progressive_beta'] ** self.distillation_steps

        # Combined loss
        total_loss = (
                self.alpha * rl_losses['total_rl'] +
                (1 - self.alpha) * beta * distill_losses['total_distill']
        )

        # Optimize student
        self.student.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), TRAINING_CONFIG['max_grad_norm'])
        self.student.optimizer.step()

        # Optimize feature projector
        self.feature_optimizer.zero_grad()
        distill_losses['feature_match'].backward(retain_graph=True)
        self.feature_optimizer.step()

        # Update temperature
        self.distillation_loss.update_temperature()
        self.distillation_steps += 1

        # Collect metrics
        metrics = {
            'total_loss': total_loss.item(),
            'rl_loss': rl_losses['total_rl'].item(),
            'distill_loss': distill_losses['total_distill'].item(),
            'policy_loss': rl_losses['policy_loss'].item(),
            'value_loss': rl_losses['value_loss'].item(),
            'entropy': rl_losses['entropy'].item(),
            'distill_weight': beta * (1 - self.alpha),
            'temperature': self.distillation_loss.temperature,
        }

        return metrics

    def _compute_rl_losses(self, outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, rewards: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute PPO losses"""
        # Policy loss
        dist = torch.distributions.Categorical(logits=outputs['primary_logits'])
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - TRAINING_CONFIG['clip_ratio'],
                            1 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        values = outputs['value'].squeeze()
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -TRAINING_CONFIG['clip_ratio'],
            TRAINING_CONFIG['clip_ratio']
        )
        value_loss_unclipped = F.mse_loss(values, rewards)
        value_loss_clipped = F.mse_loss(values_clipped, rewards)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        # Entropy bonus
        entropy = dist.entropy().mean()

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss * TRAINING_CONFIG['value_coef'],
            'entropy': -entropy * STUDENT_CONFIG['entropy_coef'],
            'total_rl': policy_loss + value_loss * TRAINING_CONFIG['value_coef'] -
                        entropy * STUDENT_CONFIG['entropy_coef'],
        }

    def adaptive_guidance(self, student_uncertainty: float, episode_reward: float) -> float:
        """Adaptively adjust mentor guidance based on student performance"""
        # High uncertainty or low reward -> more mentor guidance
        guidance_weight = min(1.0, student_uncertainty + (100 - episode_reward) / 200)

        # Decay over time
        guidance_weight *= (0.99 ** (self.distillation_steps / 1000))

        return guidance_weight
