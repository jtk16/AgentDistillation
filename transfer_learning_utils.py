# transfer_learning_utils.py
"""
Utilities for transfer learning from existing lightweight multimodal agents.
"""
import torch
import torch.nn as nn
import os
from typing import Dict, Optional, Any

# It's better to get these from config at runtime if possible,
# but for a standalone util, we might need to pass them or use defaults.
# For now, we'll pass them to load_transfer_agent.

class DummyTransferAgent(nn.Module):
    """
    A dummy agent to simulate a pre-trained lightweight multimodal agent.
    It produces outputs with expected keys and random tensor values.
    """
    def __init__(self, state_dim: int, num_actions: int, output_feature_dim: int = 64, agent_type: str = "student", name: str = "Dummy"):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.output_feature_dim = output_feature_dim
        self.agent_type = agent_type
        self.name = name

        # A minimal layer to make it a valid nn.Module and ensure parameters exist
        self.fc_dummy = nn.Linear(state_dim if state_dim > 0 else 1, num_actions if num_actions > 0 else 1)

        print(f"Initialized DummyTransferAgent: '{self.name}' (type: {self.agent_type}) with "
              f"state_dim={state_dim}, num_actions={num_actions}, output_feature_dim={output_feature_dim}")

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = state.shape[0]
        device = state.device

        if self.num_actions <= 0: # Should not happen with proper config
            raise ValueError("DummyTransferAgent num_actions must be > 0")

        # Simulate outputs
        if self.agent_type == "mentor":
            policy_logits = torch.randn(batch_size, self.num_actions, device=device)
            value = torch.randn(batch_size, 1, device=device)
            features = torch.randn(batch_size, self.output_feature_dim, device=device)
            causal_features = torch.randn(batch_size, self.output_feature_dim, device=device)
            planning_logits_horizon = 5 # Example
            planning_logits = torch.randn(batch_size, planning_logits_horizon, self.num_actions, device=device)
            return {
                'policy_logits': policy_logits,
                'value': value,
                'features': features,
                'causal_features': causal_features,
                'planning_logits': planning_logits,
                'confidence': torch.rand(batch_size, 1, device=device)
            }
        else: # student or generic
            primary_logits = torch.randn(batch_size, self.num_actions, device=device)
            value = torch.randn(batch_size, 1, device=device)
            features = torch.randn(batch_size, self.output_feature_dim, device=device)
            uncertainty_total = torch.rand(batch_size, 1, device=device)
            # Ensure 'policy_logits' is also available for consistent distillation calls
            return {
                'primary_logits': primary_logits,
                'policy_logits': primary_logits, # Duplicate for compatibility
                'value': value,
                'features': features,
                'uncertainty': {'total': uncertainty_total} # Matching student's output structure
            }

def load_transfer_agent(agent_config: Dict[str, Any],
                        env_state_dim: int,
                        env_num_actions: int,
                        device: torch.device) -> Optional[nn.Module]:
    """
    Loads a pre-trained agent based on the provided configuration.
    Currently uses DummyTransferAgent. Replace with actual loading logic.
    """
    name = agent_config.get('name', 'UnnamedAgent')
    path = agent_config.get('path', '')
    arch_type = agent_config.get('architecture_type', 'DummyArch')
    # Use feature_dim from agent_config, fallback to a default if not present
    output_feature_dim = agent_config.get('feature_dim', 64) # Default for dummy
    agent_role = agent_config.get('role', 'student') # Default role for dummy

    model: Optional[nn.Module] = None
    print(f"Attempting to load transfer agent: '{name}' of type '{arch_type}' from '{path}' (role: {agent_role})")

    # --- Actual loading logic would go here based on arch_type ---
    # Example structure:
    # if arch_type == 'MyKnownAgentArchitectureV1':
    #     model = MyKnownAgentArchitectureV1(state_dim=env_state_dim, num_actions=env_num_actions, ...)
    #     if os.path.exists(path):
    #         try:
    #             model.load_state_dict(torch.load(path, map_location=device))
    #             print(f"Successfully loaded weights for '{name}' from {path}")
    #         except Exception as e:
    #             print(f"Error loading weights for '{name}' from {path}: {e}. Model will have initial weights.")
    #     else:
    #         print(f"Warning: Path not found for '{name}': {path}. Model will have initial weights.")
    # elif arch_type == 'AnotherKnownArch':
    #     # ...
    # else:
    #     print(f"Warning: Architecture type '{arch_type}' not recognized for '{name}'. Using DummyTransferAgent.")
    #     model = DummyTransferAgent(state_dim=env_state_dim, num_actions=env_num_actions,
    #                                output_feature_dim=output_feature_dim, agent_type=agent_role, name=name)

    # For this exercise, always instantiate DummyTransferAgent if no specific logic matches
    if not model:
        print(f"Using DummyTransferAgent for '{name}' as no specific loading logic matched '{arch_type}'.")
        model = DummyTransferAgent(state_dim=env_state_dim, num_actions=env_num_actions,
                                   output_feature_dim=output_feature_dim, agent_type=agent_role, name=name)
        if path:
            if os.path.exists(path):
                print(f"Note: Path '{path}' exists but DummyTransferAgent is used for '{name}'. Implement actual loading for '{arch_type}'.")
            else:
                 print(f"Note: Path '{path}' not found. DummyTransferAgent is used for '{name}'.")

    if model:
        model.to(device)
        model.eval()  # Pre-trained agents should be in evaluation mode by default
        return model
    else:
        # This case should ideally not be reached if DummyTransferAgent is the ultimate fallback
        print(f"Critical Error: Failed to load or instantiate any agent for: '{name}'")
        return None