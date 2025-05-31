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
            # Add slight noise for diversity
            noisy_x = x + torch.randn_like(x) * 0.05
            thread_out = thread(noisy_x)
            thread_outputs.append(thread_out)

            # Thread confidence
            confidence = torch.sigmoid(conf_head(thread_out))
            thread_confidences.append(confidence)

        # Stack thread outputs
        thread_tensor = torch.stack(thread_outputs, dim=1)  # [B, num_threads, hidden_dim]
        confidences = torch.cat(thread_confidences, dim=1)  # [B, num_threads]

        # Aggregate with attention
        aggregated, attention_weights = self.thread_attention(
            thread_tensor, thread_tensor, thread_tensor
        )

        # Weighted combination based on confidence
        confidence_weights = F.softmax(confidences, dim=1).unsqueeze(-1)
        final_output = (aggregated * confidence_weights).sum(dim=1)

        info = {
            'thread_outputs': thread_tensor,
            'thread_confidences': confidences,
            'attention_weights': attention_weights,
        }

        return final_output, info


class MultiActionHead(nn.Module):
    """Generates multiple coordinated actions in a single timestep"""

    def __init__(self, hidden_dim: int, num_actions: int, num_action_heads: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_heads = num_action_heads

        # Primary action head
        self.primary_head = nn.Linear(hidden_dim, num_actions)

        # Secondary action heads (conditioned on primary)
        self.secondary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + num_actions, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_actions),
            )
            for _ in range(num_action_heads - 1)
        ])

        # Action coordination network
        self.coordination = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_action_heads),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate multiple coordinated actions"""
        # Primary action
        primary_logits = self.primary_head(features)
        primary_probs = F.softmax(primary_logits, dim=-1)

        # Secondary actions conditioned on primary
        secondary_logits = []
        for secondary_head in self.secondary_heads:
            # Condition on primary action distribution
            conditioned_input = torch.cat([features, primary_probs], dim=-1)
            secondary_logits.append(secondary_head(conditioned_input))

        # Coordination weights (which actions to actually execute)
        coordination_weights = self.coordination(features)

        return {
            'primary_logits': primary_logits,
            'secondary_logits': secondary_logits,
            'coordination_weights': coordination_weights,
        }


class UncertaintyEstimator(nn.Module):
    """Estimates both epistemic and aleatoric uncertainty"""

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Epistemic uncertainty (model uncertainty)
        self.epistemic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Aleatoric uncertainty (inherent randomness)
        self.aleatoric_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        epistemic = self.epistemic_net(features)
        aleatoric = self.aleatoric_net(features)
        total = torch.sqrt(epistemic ** 2 + aleatoric ** 2)

        return {
            'total': total,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
        }


class StudentAgent(nn.Module):
    """
    Revolutionary Student Agent with Parallel Reasoning and Multi-Action Execution
    """

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = STUDENT_CONFIG['hidden_dim']

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Parallel reasoning module
        self.parallel_reasoning = ParallelReasoningModule(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_threads=STUDENT_CONFIG['num_reasoning_threads'],
        )

        # Multi-action generation
        self.multi_action_head = MultiActionHead(
            hidden_dim=self.hidden_dim,
            num_actions=num_actions,
            num_action_heads=STUDENT_CONFIG['num_action_heads'],
        )

        # Value estimation
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(self.hidden_dim)

        # Meta-learning components
        self.meta_context = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.meta_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
        )

        self.to(DEVICE)

    def forward(self, state: torch.Tensor, mentor_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional mentor guidance"""
        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Encode state
        state_features = self.state_encoder(state)

        # Apply meta-learning context
        meta_context = self.meta_context.expand(batch_size, -1)
        adapted_features = self.meta_adapter(torch.cat([state_features, meta_context], dim=-1))

        # Parallel reasoning
        reasoned_features, reasoning_info = self.parallel_reasoning(adapted_features)

        # If mentor features provided, integrate them
        if mentor_features is not None:
            # Simple feature fusion (could be more sophisticated)
            reasoned_features = reasoned_features + 0.3 * mentor_features

        # Generate actions
        action_outputs = self.multi_action_head(reasoned_features)

        # Value estimation
        value = self.value_head(reasoned_features)

        # Uncertainty estimation
        uncertainty = self.uncertainty_estimator(reasoned_features)

        outputs = {
            'primary_logits': action_outputs['primary_logits'],
            'secondary_logits': action_outputs['secondary_logits'],
            'coordination_weights': action_outputs['coordination_weights'],
            'value': value,
            'uncertainty': uncertainty,
            'features': reasoned_features,
            'reasoning_info': reasoning_info,
        }

        return outputs

    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[List[int], Dict[str, any]]:
        """
        Select actions using parallel reasoning and multi-action capability
        Returns multiple actions that can be executed in parallel
        """
        with torch.no_grad():
            outputs = self.forward(state)

            # Primary action
            primary_logits = outputs['primary_logits']
            if deterministic:
                primary_action = torch.argmax(primary_logits, dim=-1)
            else:
                primary_dist = Categorical(logits=primary_logits)
                primary_action = primary_dist.sample()

            # Secondary actions based on coordination weights
            actions = [primary_action.item() if primary_action.dim() == 0 else primary_action[0].item()]

            coordination_weights = outputs['coordination_weights'].squeeze()
            for i, (secondary_logits, weight) in enumerate(zip(outputs['secondary_logits'], coordination_weights[1:])):
                if weight > 0.5:  # Execute this secondary action
                    if deterministic:
                        secondary_action = torch.argmax(secondary_logits, dim=-1)
                    else:
                        secondary_dist = Categorical(logits=secondary_logits)
                        secondary_action = secondary_dist.sample()

                    actions.append(
                        secondary_action.item() if secondary_action.dim() == 0 else secondary_action[0].item())

            # Collect info
            uncertainty = outputs['uncertainty']
            should_query_mentor = uncertainty['epistemic'].item() > STUDENT_CONFIG['uncertainty_threshold']

            info = {
                'value': outputs['value'].item(),
                'uncertainty': {
                    'total': uncertainty['total'].item(),
                    'epistemic': uncertainty['epistemic'].item(),
                    'aleatoric': uncertainty['aleatoric'].item(),
                },
                'should_query_mentor': should_query_mentor,
                'num_actions': len(actions),
                'coordination_weights': outputs['coordination_weights'].squeeze().cpu().numpy(),
                'reasoning_confidence': outputs['reasoning_info']['thread_confidences'].mean().item(),
            }

            return actions, info

    def update_meta_context(self, performance_delta: float):
        """Update meta-learning context based on performance"""
        # Simple gradient-free meta update
        with torch.no_grad():
            update = torch.randn_like(self.meta_context) * 0.01 * performance_delta
            self.meta_context.data += update
            self.meta_context.data = torch.clamp(self.meta_context.data, -1, 1)