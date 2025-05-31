# student.py
"""
Student Agent with Parallel Reasoning and Multi-Action Capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import DEVICE, STUDENT_CONFIG


class ParallelReasoningModule(nn.Module):
    """Parallel reasoning threads that can operate independently"""

    def __init__(self, input_dim: int, hidden_dim: int, num_threads: int):
        super().__init__()
        self.num_threads = num_threads
        self.diversity_factor = STUDENT_CONFIG.get('reasoning_thread_diversity_factor', 0.05)

        # Each thread has its own reasoning pathway
        self.reasoning_threads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_threads)
        ])

        # Thread aggregation with attention
        self.thread_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Confidence estimation for each thread
        self.thread_confidence = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_threads)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process input through parallel reasoning threads"""
        batch_size = x.shape[0]

        # Run parallel threads
        thread_outputs = []
        thread_confidences = []

        for i, (thread, conf_head) in enumerate(zip(self.reasoning_threads, self.thread_confidence)):
            # Add slight noise for diversity, configurable via STUDENT_CONFIG
            noisy_x = x + torch.randn_like(x) * self.diversity_factor
            thread_out = thread(noisy_x)
            thread_outputs.append(thread_out)

            # Thread confidence
            confidence = torch.sigmoid(conf_head(thread_out))
            thread_confidences.append(confidence)

        # Stack thread outputs
        thread_tensor = torch.stack(thread_outputs, dim=1)  # [B, num_threads, hidden_dim]
        confidences = torch.cat(thread_confidences, dim=1)  # [B, num_threads]

        # Aggregate with attention
        # Ensure query, key, value have shape (N, L, E) or (L, N, E) if batch_first=False
        # Here thread_tensor is [B, num_threads, hidden_dim], so L=num_threads, N=B, E=hidden_dim
        # This is correct for batch_first=True
        aggregated, attention_weights = self.thread_attention(
            thread_tensor, thread_tensor, thread_tensor
        )  # aggregated is [B, num_threads, hidden_dim]

        # Weighted combination based on confidence (or use attention weights directly?)
        # Current: weighted average of attention-aggregated outputs using thread confidences.
        # This might be double-weighting. A common pattern is to use attention output directly,
        # or use confidences to weight the *original* thread_outputs before attention/aggregation.
        # For now, keeping existing logic but noting it.
        # Let's assume aggregated is the result we want, then sum it up or pick one.
        # The original code sums the attention output based on confidence weights.

        # Summing over num_threads dim after weighting by confidence
        confidence_weights = F.softmax(confidences, dim=1).unsqueeze(-1)  # [B, num_threads, 1]
        # Element-wise multiply the aggregated output of each thread by its confidence weight, then sum.
        # `aggregated` is [B, num_threads, hidden_dim].
        final_output = (aggregated * confidence_weights).sum(dim=1)  # [B, hidden_dim]

        info = {
            'thread_outputs': thread_tensor,  # [B, num_threads, hidden_dim]
            'thread_confidences': confidences,  # [B, num_threads]
            'attention_weights': attention_weights,
            # Shape depends on MHA impl, often [B, L, S] or [B, num_heads, L, S]
        }

        return final_output, info


class MultiActionHead(nn.Module):
    """Generates multiple coordinated actions in a single timestep"""

    def __init__(self, hidden_dim: int, num_actions: int, num_action_heads: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_heads = num_action_heads  # Total number of heads including primary

        # Primary action head
        self.primary_head = nn.Linear(hidden_dim, num_actions)

        # Secondary action heads (e.g., if num_action_heads = 3, we have 1 primary + 2 secondary)
        self.secondary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + num_actions, hidden_dim // 2),  # Condition on primary_probs
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_actions),
            )
            for _ in range(max(0, num_action_heads - 1))  # Ensure non-negative range
        ])

        # Action coordination network (outputs weights for ALL action heads, including primary)
        self.coordination = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_action_heads),  # Outputs one weight per head
            nn.Sigmoid(),  # Weights between 0 and 1
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate multiple coordinated actions. features is [B, hidden_dim]"""
        # Primary action
        primary_logits = self.primary_head(features)  # [B, num_actions]
        primary_probs = F.softmax(primary_logits, dim=-1)  # [B, num_actions]

        # Secondary actions conditioned on primary
        secondary_logits_list = []
        if self.secondary_heads:  # Only if there are secondary heads
            conditioned_input = torch.cat([features, primary_probs], dim=-1)  # [B, hidden_dim + num_actions]
            for secondary_head_module in self.secondary_heads:
                secondary_logits_list.append(secondary_head_module(conditioned_input))
        # secondary_logits_list is a list of tensors, each [B, num_actions]

        # Coordination weights (for all heads, including primary)
        coordination_weights = self.coordination(features)  # [B, num_action_heads]

        return {
            'primary_logits': primary_logits,
            'secondary_logits': secondary_logits_list,  # List of [B, num_actions] tensors
            'coordination_weights': coordination_weights,  # [B, num_action_heads]
        }


class UncertaintyEstimator(nn.Module):
    """Estimates both epistemic and aleatoric uncertainty"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.epistemic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.aleatoric_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        epistemic = self.epistemic_net(features)  # [B, 1]
        aleatoric = self.aleatoric_net(features)  # [B, 1]
        # Total uncertainty: commonly sqrt(ep^2 + al^2) or other combinations
        total = torch.sqrt(epistemic.pow(2) + aleatoric.pow(2))  # [B, 1]
        return {'total': total, 'epistemic': epistemic, 'aleatoric': aleatoric}


class StudentAgent(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = STUDENT_CONFIG['hidden_dim']

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.parallel_reasoning = ParallelReasoningModule(
            self.hidden_dim, self.hidden_dim, STUDENT_CONFIG['num_reasoning_threads'])
        self.multi_action_head = MultiActionHead(
            self.hidden_dim, num_actions, STUDENT_CONFIG['num_action_heads'])
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1))
        self.uncertainty_estimator = UncertaintyEstimator(self.hidden_dim)
        self.meta_context = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.meta_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.GELU())
        self.to(DEVICE)

    def forward(self, state: torch.Tensor, mentor_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = state.shape[0]
        if state.dim() == 1: state = state.unsqueeze(0)

        state_features = self.state_encoder(state)
        meta_context_expanded = self.meta_context.expand(batch_size, -1)
        adapted_features = self.meta_adapter(torch.cat([state_features, meta_context_expanded], dim=-1))
        reasoned_features, reasoning_info = self.parallel_reasoning(adapted_features)

        if mentor_features is not None:
            if mentor_features.shape[0] != batch_size:  # Ensure mentor_features match batch size if provided
                mentor_features = mentor_features.expand(batch_size, -1) if mentor_features.shape[
                                                                                0] == 1 else mentor_features[
                                                                                             :batch_size]
            reasoned_features = reasoned_features + 0.3 * mentor_features

        action_outputs = self.multi_action_head(reasoned_features)
        value = self.value_head(reasoned_features)
        uncertainty = self.uncertainty_estimator(reasoned_features)

        return {
            'primary_logits': action_outputs['primary_logits'],  # [B, num_actions]
            'secondary_logits': action_outputs['secondary_logits'],  # List of Tensors, each [B, num_actions]
            'coordination_weights': action_outputs['coordination_weights'],  # [B, num_action_heads]
            'value': value,  # [B, 1]
            'uncertainty': uncertainty,  # Dict of Tensors, each [B, 1]
            'features': reasoned_features,  # [B, hidden_dim]
            'reasoning_info': reasoning_info,  # Dict of Tensors from ParallelReasoning
        }

    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[List[List[int]], List[Dict[str, any]]]:
        """ Select actions for a batch of states.
            Returns:
                actions_batch: List[List[int]], outer list for batch, inner for multi-actions per env.
                info_batch: List[Dict[str, any]], one info dict per env in batch.
        """
        with torch.no_grad():
            outputs = self.forward(state)  # Process batch of states
            batch_size = state.shape[0]

            primary_logits = outputs['primary_logits']  # [B, num_actions]
            if deterministic:
                primary_action_batch = torch.argmax(primary_logits, dim=-1)  # [B]
            else:
                primary_dist = Categorical(logits=primary_logits)
                primary_action_batch = primary_dist.sample()  # [B]

            actions_batch = [[pa.item()] for pa in primary_action_batch]  # Initialize with primary action for each env

            coordination_weights = outputs['coordination_weights']  # [B, num_action_heads]
            secondary_logits_list = outputs['secondary_logits']  # List of Tensors, each [B, num_actions]

            num_total_action_heads = STUDENT_CONFIG['num_action_heads']
            num_secondary_action_heads = len(secondary_logits_list)

            # Iterate through secondary action heads
            for head_idx in range(num_secondary_action_heads):
                # +1 because coordination_weights[0] is for primary, [1] for first secondary, etc.
                current_head_coordination_weights = coordination_weights[:,
                                                    head_idx + 1]  # [B], weights for this secondary head
                current_head_secondary_logits = secondary_logits_list[head_idx]  # [B, num_actions]

                for env_idx in range(batch_size):  # Iterate through environments in the batch
                    if current_head_coordination_weights[env_idx] > 0.5:
                        env_specific_logits = current_head_secondary_logits[env_idx]  # [num_actions]
                        if deterministic:
                            secondary_action = torch.argmax(env_specific_logits, dim=-1)
                        else:
                            secondary_dist = Categorical(logits=env_specific_logits)
                            secondary_action = secondary_dist.sample()
                        actions_batch[env_idx].append(secondary_action.item())

            info_batch = []
            for i in range(batch_size):
                epistemic_unc = outputs['uncertainty']['epistemic'][i].item()
                info_item = {
                    'value': outputs['value'][i].item(),
                    'uncertainty': {
                        'total': outputs['uncertainty']['total'][i].item(),
                        'epistemic': epistemic_unc,
                        'aleatoric': outputs['uncertainty']['aleatoric'][i].item(),
                    },
                    'should_query_mentor': epistemic_unc > STUDENT_CONFIG['uncertainty_threshold'],
                    'num_actions': len(actions_batch[i]),
                    'coordination_weights': coordination_weights[i].cpu().numpy(),
                    'reasoning_confidence': outputs['reasoning_info']['thread_confidences'][i].mean().item(),
                }
                info_batch.append(info_item)

            return actions_batch, info_batch

    def update_meta_context(self, performance_delta: float):
        with torch.no_grad():
            update = torch.randn_like(self.meta_context) * 0.01 * performance_delta
            self.meta_context.data += update
            self.meta_context.data = torch.clamp(self.meta_context.data, -1, 1)