# config.py
"""
Configuration and hyperparameters for the Revolutionary AI Pipeline
(Further Adjusted for RTX 2060 Super - OOM Mitigation & Graph Rework)
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
    'num_envs': 1,
    'max_episode_steps': 500,
}

# Mentor configuration
MENTOR_CONFIG = {
    'hidden_dim': 128,
    'num_knowledge_tokens': 16,
    'num_attention_heads': 2,
    'num_transformer_layers': 1,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'causal_prediction_weight': 0.3,
    'MENTOR_KNOWLEDGE_CONCEPTS': {
        "physics_concepts": [0, 1, 2, 3],
        "balance_strategy": list(range(4, 8)),
        "critical_states": list(range(8,12))
    },
    'bc_epochs': 10
}

# Student configuration
STUDENT_CONFIG = {
    'hidden_dim': 64,
    'num_reasoning_threads': 1,
    'num_action_heads': 1,
    'uncertainty_threshold': 0.4,
    'learning_rate': 3e-4,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'reasoning_thread_diversity_factor': 0.05
}

# Distillation configuration
DISTILLATION_CONFIG = {
    'temperature': 4.0,
    'alpha': 0.7,
    'feature_matching_weight': 0.1,
    'value_distill_weight': 0.5,
    'progressive_beta': 0.9,
}

# Memory configuration
MEMORY_CONFIG = {
    'trajectory_buffer_size': 2000,
    'prioritized_replay': True,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'min_trajectory_reward': 50,
}

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': 100000,
    'rollout_steps': 128,
    'num_ppo_epochs': 2,
    'batch_size': 16, # PPO batch size
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_grad_norm': 0.5,
    'target_kl': 0.01,
    'clip_value_loss': True,
    'human_cloning_steps': 5000,
    'focused_distillation_steps': 20000,
    # NEW: Chunk size for processing demos during pathway analysis phase
    'demo_processing_chunk_size': 5 # Number of demonstrations to process in one go for graph building
}

# Curriculum learning configuration
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
    'log_interval': 500,
    'eval_interval': 2500,
    'save_interval': 5000,
    'verbose': True,
    'num_eval_episodes': 5
}

# Transfer Learning Configuration
TRANSFER_LEARNING_CONFIG = {
    'enabled': False,
    'mentor_transfer_sources': [],
    'student_transfer_sources': [],
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