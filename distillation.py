# distillation.py
"""
Progressive Knowledge Distillation from Mentor to Student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from config import DEVICE, DISTILLATION_CONFIG, TRAINING_CONFIG, STUDENT_CONFIG, REVOLUTIONARY_FEATURES


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        input_dim = max(1, int(input_dim))
        output_dim = max(1, int(output_dim))
        mid_dim = max(1, (input_dim + output_dim) // 2)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)


class ProgressiveDistillationLoss(nn.Module):
    def __init__(self, initial_temp: float = 4.0):
        super().__init__()
        self.temperature = initial_temp
        self.min_temperature = 1.0
        self.decay_rate = 0.99995

    def forward(self, student_logits: torch.Tensor, mentor_logits: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        mentor_probs = F.softmax(mentor_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, mentor_probs, reduction='batchmean', log_target=False)
        return kl_loss * (self.temperature ** 2)

    def update_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature * self.decay_rate)


class DistillationTrainer(nn.Module):  # MODIFIED: Inherit from nn.Module
    def __init__(self, mentor: nn.Module, student: nn.Module,  # student is not a child module here
                 mentor_hidden_dim: int, student_hidden_dim: int):
        super().__init__()  # MODIFIED: Call super constructor
        self.mentor = mentor
        self.student_ref = student  # Keep a reference to the student, not as a child module
        self.student_hidden_dim = student_hidden_dim

        self.feature_projector = FeatureProjector(mentor_hidden_dim, student_hidden_dim)
        self.distillation_loss_fn = ProgressiveDistillationLoss(DISTILLATION_CONFIG['temperature'])

        self.distillation_steps = 0
        self.alpha = DISTILLATION_CONFIG['alpha']

        self.student_aux_sources: List[Dict[str, Any]] = []
        self.aux_feature_projectors = nn.ModuleDict()  # Child modules

    def set_student_aux_sources(self, student_aux_sources: List[Dict[str, Any]], student_target_feature_dim: int):
        self.student_aux_sources = student_aux_sources
        keys_to_remove = list(self.aux_feature_projectors.keys())
        for key in keys_to_remove: del self.aux_feature_projectors[key]

        for i, source_config in enumerate(self.student_aux_sources):
            name = source_config.get('name', f"aux_source_{i}")
            proj_module_name = f"aux_proj_{name.replace(' ', '_').replace('.', '_')}"
            if 'features' in source_config.get('transfer_targets', []) and \
                    source_config.get('feature_dim') != student_target_feature_dim:
                aux_feature_dim = source_config.get('feature_dim')
                if not isinstance(aux_feature_dim, int) or aux_feature_dim <= 0:
                    print(
                        f"Warning (DistTrainer): Invalid 'feature_dim' ({aux_feature_dim}) for aux source {name}. Skipping projector.")
                    continue
                self.aux_feature_projectors[proj_module_name] = FeatureProjector(aux_feature_dim,
                                                                                 student_target_feature_dim)
                print(f"DistillationTrainer: Initialized FeatureProjector '{proj_module_name}' for '{name}'.")
        # Ensure this module and its children are on the correct device
        # This should be handled by the parent pipeline that owns this DistillationTrainer instance.
        # self.to(DEVICE) # If DistillationTrainer is used standalone, otherwise main pipeline moves it.

    def compute_distillation_loss_components(self, states: torch.Tensor,
                                             student_outputs: Dict[str, torch.Tensor],
                                             mentor_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[
        str, torch.Tensor]:
        # This method computes individual loss components. The sum is handled in train_step.
        losses: Dict[str, torch.Tensor] = {}
        if mentor_outputs is None:
            with torch.no_grad():
                mentor_outputs = self.mentor(states)

        losses['policy_distill_main'] = self.distillation_loss_fn(
            student_outputs['primary_logits'], mentor_outputs['policy_logits']
        )

        if 'features' in student_outputs and 'features' in mentor_outputs:
            projected_mentor_features = self.feature_projector(mentor_outputs['features'])
            losses['feature_match_main'] = F.mse_loss(student_outputs['features'], projected_mentor_features) * \
                                           DISTILLATION_CONFIG['feature_matching_weight']
        else:
            losses['feature_match_main'] = torch.tensor(0.0, device=states.device)

        if 'value' in student_outputs and 'value' in mentor_outputs:
            losses['value_distill_main'] = F.mse_loss(student_outputs['value'], mentor_outputs['value']) * \
                                           DISTILLATION_CONFIG.get('value_distill_weight', 0.5)
        else:
            losses['value_distill_main'] = torch.tensor(0.0, device=states.device)

        if 'causal_features' in mentor_outputs and 'features' in student_outputs:
            projected_causal_features = self.feature_projector(mentor_outputs['causal_features'])
            losses['causal_transfer_main'] = F.mse_loss(student_outputs['features'], projected_causal_features) * 0.1
        else:
            losses['causal_transfer_main'] = torch.tensor(0.0, device=states.device)

        total_aux_loss_terms: List[torch.Tensor] = []
        if hasattr(self, 'student_aux_sources') and self.student_aux_sources:
            for i, aux_source_config in enumerate(self.student_aux_sources):
                aux_model = aux_source_config['model'];
                aux_weight = aux_source_config['weight']
                aux_targets = aux_source_config.get('transfer_targets', [])
                name = aux_source_config.get('name', f"aux_source_{i}")
                proj_name = f"aux_proj_{name.replace(' ', '_').replace('.', '_')}"

                with torch.no_grad():
                    aux_outputs = aux_model(states)

                current_aux_loss = torch.tensor(0.0, device=states.device)
                if 'policy_logits' in aux_targets and 'policy_logits' in aux_outputs:
                    current_aux_loss += self.distillation_loss_fn(student_outputs['primary_logits'],
                                                                  aux_outputs['policy_logits'])
                if 'features' in aux_targets and 'features' in aux_outputs and 'features' in student_outputs:
                    aux_proj = self.aux_feature_projectors.get(proj_name)
                    s_feats = aux_outputs['features'];
                    t_s_feats = student_outputs['features']
                    proj_aux_feats = aux_proj(s_feats) if aux_proj else (
                        s_feats if s_feats.shape == t_s_feats.shape else None)
                    if proj_aux_feats is not None:
                        current_aux_loss += F.mse_loss(t_s_feats, proj_aux_feats) * DISTILLATION_CONFIG.get(
                            'feature_matching_weight', 0.1)
                if 'value' in aux_targets and 'value' in aux_outputs and 'value' in student_outputs:
                    current_aux_loss += F.mse_loss(student_outputs['value'],
                                                   aux_outputs['value']) * DISTILLATION_CONFIG.get(
                        'value_distill_weight', 0.5)

                if aux_weight > 0 and current_aux_loss.requires_grad:  # Only add if weight is positive and loss is computed
                    total_aux_loss_terms.append(current_aux_loss * aux_weight)

        if total_aux_loss_terms:
            losses['aux_distill_combined'] = torch.stack(total_aux_loss_terms).sum()
        else:
            losses['aux_distill_combined'] = torch.tensor(0.0, device=states.device)

        return losses

    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   returns: torch.Tensor, advantages: torch.Tensor,
                   old_log_probs: torch.Tensor, old_values: torch.Tensor,
                   mentor_advice: Optional[List] = None) -> Dict[str, Any]:  # Return Any for the loss tensor

        with torch.no_grad():  # Mentor is not trained here
            mentor_outputs = self.mentor(states)

        # Student forward pass - student is optimized by an external optimizer
        student_features_input = mentor_outputs.get('features') if REVOLUTIONARY_FEATURES.get(
            'student_uses_mentor_features', False) else None
        student_outputs = self.student_ref(states, mentor_features=student_features_input)  # Use student_ref

        rl_losses = self._compute_rl_losses(student_outputs, actions, returns, advantages, old_log_probs, old_values)

        # Compute all distillation loss components (main mentor + auxiliary)
        # These losses involve parameters from self.feature_projector and self.aux_feature_projectors
        distill_loss_components = self.compute_distillation_loss_components(states, student_outputs, mentor_outputs)

        # Combine all distillation losses
        total_distillation_loss = torch.tensor(0.0, device=states.device)
        for key, loss_val in distill_loss_components.items():
            if isinstance(loss_val, torch.Tensor):  # Check if it's a computed loss tensor
                total_distillation_loss += loss_val

        beta = DISTILLATION_CONFIG['progressive_beta'] ** self.distillation_steps

        # Final combined loss for this training step
        # This total_loss will have gradients flowing to student parameters AND projector parameters
        # (feature_projector, aux_feature_projectors) because they are used in computing distill_loss_components.
        total_loss = (
                self.alpha * rl_losses['total_rl'] +
                (1 - self.alpha) * beta * total_distillation_loss
        )

        # The .backward() and optimizer.step() for the student (which now includes projector params)
        # will be handled by the main training loop in EnhancedRevolutionaryPipeline.

        self.distillation_loss_fn.update_temperature()
        self.distillation_steps += 1

        metrics = {
            'total_loss': total_loss.item(),
            'rl_loss_total': rl_losses['total_rl'].item(),
            'distill_loss_total_combined': total_distillation_loss.item(),  # Log the sum of all distill parts
            'ppo_policy_loss': rl_losses['policy_loss'].item(),
            'ppo_value_loss': rl_losses['value_loss'].item(),
            'ppo_entropy': rl_losses['entropy'].item(),
            'main_mentor_policy_distill': distill_loss_components.get('policy_distill_main', torch.tensor(0.0)).item(),
            'main_mentor_feature_match': distill_loss_components.get('feature_match_main', torch.tensor(0.0)).item(),
            'main_mentor_value_distill': distill_loss_components.get('value_distill_main', torch.tensor(0.0)).item(),
            'aux_sources_distill_total': distill_loss_components.get('aux_distill_combined', torch.tensor(0.0)).item(),
            'distill_progressive_weight': beta * (1 - self.alpha),
            'temperature': self.distillation_loss_fn.temperature,
            '_total_loss_tensor_for_backward': total_loss  # Pass the tensor for backward pass
        }
        return metrics

    def _compute_rl_losses(self, outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=outputs['primary_logits'])
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TRAINING_CONFIG['clip_ratio'],
                            1.0 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values_pred = outputs['value'].squeeze(-1)
        old_values_squeezed = old_values.squeeze(
            -1) if old_values.ndim > returns.ndim and old_values.ndim > 1 else old_values

        if TRAINING_CONFIG.get('clip_value_loss', True):
            values_clipped = old_values_squeezed + torch.clamp(values_pred - old_values_squeezed,
                                                               -TRAINING_CONFIG['clip_ratio'],
                                                               TRAINING_CONFIG['clip_ratio'])
            vf_loss1 = F.mse_loss(values_pred, returns)
            vf_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(vf_loss1, vf_loss2)
        else:
            value_loss = F.mse_loss(values_pred, returns)

        entropy = dist.entropy().mean()

        total_rl_loss = policy_loss + \
                        value_loss * STUDENT_CONFIG['value_coef'] - \
                        entropy * STUDENT_CONFIG['entropy_coef']
        return {
            'policy_loss': policy_loss, 'value_loss': value_loss,
            'entropy': entropy, 'total_rl': total_rl_loss,
        }

    def adaptive_guidance(self, student_uncertainty: float, episode_reward: float) -> float:
        guidance_weight = min(1.0, student_uncertainty + (100 - episode_reward) / 200)
        guidance_weight *= (0.99 ** (self.distillation_steps / 1000))
        return guidance_weight