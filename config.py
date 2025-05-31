# config.py
"""
Configuration and hyperparameters for the Revolutionary AI Pipeline
"""

import torch
import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment configuration
ENV_CONFIG = {
    'name': 'CartPole-v1',
    'num_envs': 4,
    'max_episode_steps': 500,
}

# Mentor configuration
MENTOR_CONFIG = {
    'hidden_dim': 512,
    'num_knowledge_tokens': 64,  # Textual environment knowledge
    'num_attention_heads': 8,
    'num_transformer_layers': 4,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'causal_prediction_weight': 0.3,  # For learning causal relationships
}

# Student configuration
STUDENT_CONFIG = {
    'hidden_dim': 256,
    'num_reasoning_threads': 4,  # Parallel reasoning
    'num_action_heads': 3,  # Multi-action capability
    'uncertainty_threshold': 0.4,  # When to query mentor
    'learning_rate': 3e-4,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
}

# Distillation configuration
DISTILLATION_CONFIG = {
    'temperature': 4.0,  # For soft targets
    'alpha': 0.7,  # Balance between RL and distillation
    'feature_matching_weight': 0.1,
    'progressive_beta': 0.9,  # For progressive distillation
}

# Memory configuration
MEMORY_CONFIG = {
    'trajectory_buffer_size': 10000,
    'prioritized_replay': True,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'min_trajectory_reward': 50,
}

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': 500000,
    'rollout_steps': 512,
    'num_ppo_epochs': 4,
    'batch_size': 64,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_grad_norm': 0.5,
    'target_kl': 0.01,
}

# Curriculum learning configuration
CURRICULUM_CONFIG = {
    'enabled': True,
    'stages': [
        {'min_reward': 50, 'mentor_query_prob': 0.8},
        {'min_reward': 150, 'mentor_query_prob': 0.5},
        {'min_reward': 300, 'mentor_query_prob': 0.3},
        {'min_reward': 450, 'mentor_query_prob': 0.1},
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    'log_interval': 1000,
    'eval_interval': 5000,
    'save_interval': 10000,
    'verbose': True,
}

# Revolutionary features flags
REVOLUTIONARY_FEATURES = {
    'multimodal_mentor': True,
    'parallel_reasoning': True,
    'multi_action_execution': True,
    'active_querying': True,
    'progressive_distillation': True,
    'causal_understanding': True,
    'meta_learning': True,
}