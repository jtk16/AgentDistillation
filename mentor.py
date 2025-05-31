# mentor.py
"""
Multimodal Mentor with textual environment understanding and causal reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import DEVICE, MENTOR_CONFIG


@dataclass
class MentorAdvice:
    """Structured advice from the mentor"""
    actions: List[int]  # Sequence of actions
    confidence: float
    reasoning: List[str]
    causal_effects: Dict[str, float]  # Predicted effects of actions
    strategy: str


class CausalAttention(nn.Module):
    """Attention mechanism that learns causal relationships"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.causal_proj = nn.Linear(dim, dim)  # For causal relationships

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with causal awareness
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        # Causal projection
        causal_features = self.causal_proj(x)

        return x, causal_features


class MultimodalMentor(nn.Module):
    """
    Revolutionary Multimodal Mentor that understands environment through text and state
    """

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = MENTOR_CONFIG['hidden_dim']

        # Environmental knowledge embeddings (textual understanding)
        self.env_knowledge = nn.Parameter(
            torch.randn(MENTOR_CONFIG['num_knowledge_tokens'], self.hidden_dim) * 0.02
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Causal transformer layers
        self.causal_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CausalAttention(self.hidden_dim, MENTOR_CONFIG['num_attention_heads']),
                'norm1': nn.LayerNorm(self.hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                ),
                'norm2': nn.LayerNorm(self.hidden_dim),
            })
            for _ in range(MENTOR_CONFIG['num_transformer_layers'])
        ])

        # Action understanding and prediction
        self.action_embeddings = nn.Embedding(num_actions, self.hidden_dim)
        self.causal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, state_dim),  # Predict state changes
        )

        # Output heads
        self.policy_head = nn.Linear(self.hidden_dim, num_actions)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Multi-step planning head
        self.planning_horizon = 5
        self.planning_head = nn.Linear(self.hidden_dim, num_actions * self.planning_horizon)

        # Initialize with environment-specific knowledge
        self._initialize_knowledge()

        self.to(DEVICE)

    def _initialize_knowledge(self):
        """Initialize knowledge tokens with environment-specific information"""
        # This would be customized per environment
        # For CartPole: physics concepts, balance strategies, etc.
        with torch.no_grad():
            # Example: First few tokens represent physics concepts
            self.env_knowledge[0] = torch.randn(self.hidden_dim) * 0.1  # Gravity
            self.env_knowledge[1] = torch.randn(self.hidden_dim) * 0.1  # Momentum
            self.env_knowledge[2] = torch.randn(self.hidden_dim) * 0.1  # Balance
            # etc...

    def forward(self, state: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with causal reasoning"""
        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Encode state
        state_encoding = self.state_encoder(state)  # [B, hidden_dim]

        # Combine with knowledge tokens
        knowledge = self.env_knowledge.unsqueeze(0).expand(batch_size, -1, -1)
        state_token = state_encoding.unsqueeze(1)  # [B, 1, hidden_dim]

        # Concatenate state with knowledge
        x = torch.cat([state_token, knowledge], dim=1)  # [B, 1+K, hidden_dim]

        # Apply causal transformer layers
        attention_maps = []
        causal_features_list = []

        for layer in self.causal_layers:
            # Self-attention with causal reasoning
            attn_out, causal_feat = layer['attention'](layer['norm1'](x))
            x = x + attn_out
            causal_features_list.append(causal_feat)

            # FFN
            x = x + layer['ffn'](layer['norm2'](x))

            if return_attention:
                attention_maps.append(attn_out)

        # Extract final representation
        final_features = x[:, 0]  # Use the state token

        # Generate outputs
        policy_logits = self.policy_head(final_features)
        value = self.value_head(final_features)
        confidence = self.confidence_head(final_features)
        planning_logits = self.planning_head(final_features)

        outputs = {
            'policy_logits': policy_logits,
            'value': value,
            'confidence': confidence,
            'planning_logits': planning_logits.view(batch_size, self.planning_horizon, self.num_actions),
            'features': final_features,
            'causal_features': torch.stack(causal_features_list, dim=1).mean(dim=1)[:, 0],
        }

        if return_attention:
            outputs['attention_maps'] = attention_maps

        return outputs

    def predict_action_effects(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Predict the causal effects of an action"""
        state_encoding = self.state_encoder(state)
        action_encoding = self.action_embeddings(torch.tensor([action], device=DEVICE))

        combined = torch.cat([state_encoding, action_encoding], dim=-1)
        predicted_change = self.causal_predictor(combined)

        return predicted_change

    def get_advice(self, state: torch.Tensor, verbose: bool = False) -> MentorAdvice:
        """Generate sophisticated advice with causal reasoning"""
        with torch.no_grad():
            outputs = self.forward(state, return_attention=True)

            # Extract key information
            policy_probs = F.softmax(outputs['policy_logits'], dim=-1)
            confidence = outputs['confidence'].item()
            planning_logits = outputs['planning_logits']

            # Get immediate action
            immediate_action = torch.argmax(policy_probs, dim=-1).item()

            # Get action sequence from planning
            plan_probs = F.softmax(planning_logits, dim=-1)
            planned_actions = torch.argmax(plan_probs, dim=-1).squeeze().cpu().numpy()

            # Predict causal effects
            causal_effects = {}
            for action in range(self.num_actions):
                effect = self.predict_action_effects(state, action)
                causal_effects[f'action_{action}'] = effect.squeeze().cpu().numpy()

            # Generate reasoning based on state analysis
            state_np = state.cpu().numpy().flatten()
            reasoning = self._generate_reasoning(state_np, immediate_action, causal_effects)

            # Determine strategy
            strategy = self._determine_strategy(state_np, planned_actions)

            return MentorAdvice(
                actions=planned_actions.tolist()[:3],  # Next 3 actions
                confidence=confidence,
                reasoning=reasoning,
                causal_effects={k: float(np.mean(np.abs(v))) for k, v in causal_effects.items()},
                strategy=strategy,
            )

    def _generate_reasoning(self, state: np.ndarray, action: int, effects: Dict) -> List[str]:
        """Generate reasoning steps based on state and causal understanding"""
        reasoning = []

        # State analysis
        if len(state) >= 4:  # CartPole
            pos, vel, angle, ang_vel = state[:4]
            reasoning.append(f"State: pos={pos:.3f}, vel={vel:.3f}, angle={angle:.3f}, ang_vel={ang_vel:.3f}")

            # Physics reasoning
            if abs(angle) > 0.1:
                reasoning.append(f"Critical: Pole tilting {'left' if angle < 0 else 'right'} at {abs(angle):.3f} rad")

            # Causal prediction
            effect_strength = effects[f'action_{action}']
            reasoning.append(f"Action {action} predicted effect magnitude: {effect_strength:.3f}")

        return reasoning

    def _determine_strategy(self, state: np.ndarray, actions: np.ndarray) -> str:
        """Determine high-level strategy"""
        if len(state) >= 4:  # CartPole specific
            angle = state[2]
            if abs(angle) > 0.15:
                return "EMERGENCY_RECOVERY: Aggressive correction needed"
            elif abs(angle) > 0.05:
                return "ACTIVE_BALANCING: Moderate corrections"
            else:
                return "FINE_TUNING: Maintain equilibrium"
        return "EXPLORATORY: Learning environment dynamics"