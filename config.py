# demo_config.py
"""
Simplified configuration for demo to ensure everything works
"""

import torch
import numpy as np

# Device configuration - simplified for demo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment configuration - simplified
ENV_CONFIG = {
    'name': 'CartPole-v1',
    'num_envs': 1,  # Single environment for demo
    'max_episode_steps': 500,
}

# Mentor configuration - reduced for demo
MENTOR_CONFIG = {
    'hidden_dim': 64,  # Reduced from 128
    'num_knowledge_tokens': 8,  # Reduced from 16
    'num_attention_heads': 2,
    'num_transformer_layers': 1,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'causal_prediction_weight': 0.3,
    'MENTOR_KNOWLEDGE_CONCEPTS': {
        "physics_concepts": [0, 1],
        "balance_strategy": [2, 3],
        "critical_states": [4, 5]
    },
    'bc_epochs': 5  # Reduced from 10
}

# Student configuration - reduced for demo
STUDENT_CONFIG = {
    'hidden_dim': 32,  # Reduced from 64
    'num_reasoning_threads': 2,  # Reduced from default
    'num_action_heads': 2,  # Reduced from default
    'uncertainty_threshold': 0.4,
    'learning_rate': 3e-4,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'reasoning_thread_diversity_factor': 0.05
}

# Distillation configuration - simplified
DISTILLATION_CONFIG = {
    'temperature': 4.0,
    'alpha': 0.7,
    'feature_matching_weight': 0.1,
    'value_distill_weight': 0.5,
    'progressive_beta': 0.9,
}

# Memory configuration - reduced for demo
MEMORY_CONFIG = {
    'trajectory_buffer_size': 500,  # Reduced from 2000
    'prioritized_replay': True,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'min_trajectory_reward': 50,
}

# Training configuration - extended for better demo
TRAINING_CONFIG = {
    'total_timesteps': 25000,  # Increased for longer demo
    'rollout_steps': 32,  # Reduced from 128
    'num_ppo_epochs': 2,
    'batch_size': 8,  # Reduced from 16
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_grad_norm': 0.5,
    'target_kl': 0.01,
    'clip_value_loss': True,
    'human_cloning_steps': 2000,  # Increased for demo
    'focused_distillation_steps': 8000,  # Increased for demo
    'demo_processing_chunk_size': 3  # Reduced from 5
}

# Curriculum learning configuration - disabled for demo
CURRICULUM_CONFIG = {
    'enabled': False,
    'stages': [
        {'min_reward': 50, 'mentor_query_prob': 0.8},
        {'min_reward': 100, 'mentor_query_prob': 0.5},
        {'min_reward': 150, 'mentor_query_prob': 0.3},
        {'min_reward': 200, 'mentor_query_prob': 0.1},
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    'log_interval': 100,  # Reduced from 500
    'eval_interval': 500,  # Reduced from 2500
    'save_interval': 1000,  # Reduced from 5000
    'verbose': True,
    'num_eval_episodes': 3  # Reduced from 5
}

# Transfer Learning Configuration - disabled for demo
TRANSFER_LEARNING_CONFIG = {
    'enabled': False,
    'mentor_transfer_sources': [],
    'student_transfer_sources': [],
}

# Revolutionary features flags - simplified for demo
REVOLUTIONARY_FEATURES = {
    'multimodal_mentor': True,
    'parallel_reasoning': True,
    'multi_action_execution': True,
    'active_querying': True,
    'progressive_distillation': True,
    'causal_understanding': True,
    'meta_learning': False,  # Disabled for demo stability
}

print(f"Demo configuration loaded for device: {DEVICE}")