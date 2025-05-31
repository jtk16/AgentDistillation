# llm_config.py
"""
Enhanced configuration for LLM-powered Revolutionary AI Pipeline
Optimized for RTX 2060S (8GB VRAM)
"""

import torch
import numpy as np

# Device configuration with LLM optimization
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Memory management for 2060S
if torch.cuda.is_available():
    # Reserve memory more carefully for LLM + training
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster but less reproducible

# Seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment configuration
ENV_CONFIG = {
    'name': 'CartPole-v1',
    'num_envs': 1,  # Single environment for better LLM performance
    'max_episode_steps': 500,
}

# LLM Mentor configuration
LLM_MENTOR_CONFIG = {
    # Model selection (in order of preference for 2060S)
    'model_name': 'microsoft/Phi-3-mini-4k-instruct',  # Primary choice
    'fallback_models': [
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'Qwen/Qwen2-0.5B-Instruct'
    ],

    # Memory optimization
    'use_4bit_quantization': True,
    'max_context_length': 2048,
    'max_new_tokens': 256,

    # Generation parameters
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,

    # Caching and optimization
    'enable_response_cache': True,
    'cache_size_limit': 100,
    'enable_kv_cache': True,

    # Integration settings
    'query_frequency': 'adaptive',  # 'always', 'adaptive', or 'minimal'
    'confidence_threshold': 0.7,  # Only use LLM advice above this confidence
    'fallback_to_neural': True,  # Fallback to neural layers on LLM failure

    # Performance monitoring
    'log_llm_performance': True,
    'memory_cleanup_interval': 100,  # Clean memory every N queries
}

# Updated Mentor configuration (maintained for compatibility)
MENTOR_CONFIG = {
    'hidden_dim': 64,  # Reduced for memory efficiency
    'num_knowledge_tokens': 8,
    'num_attention_heads': 2,
    'num_transformer_layers': 1,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'causal_prediction_weight': 0.3,
    'bc_epochs': 3,  # Reduced since LLM provides better guidance

    # LLM integration
    'use_llm_mentor': True,
    'llm_config': LLM_MENTOR_CONFIG,
    'hybrid_mode': True,  # Use both LLM and neural components
}

# Student configuration (optimized for LLM distillation)
STUDENT_CONFIG = {
    'hidden_dim': 32,  # Reduced for memory efficiency
    'num_reasoning_threads': 2,
    'num_action_heads': 2,
    'uncertainty_threshold': 0.4,
    'learning_rate': 3e-4,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'reasoning_thread_diversity_factor': 0.05,

    # LLM distillation settings
    'llm_query_probability': 0.3,  # Query LLM 30% of the time initially
    'llm_query_decay': 0.999,  # Reduce queries over time as student improves
    'min_llm_query_probability': 0.05,  # Minimum query rate
}

# Enhanced distillation configuration for LLM
DISTILLATION_CONFIG = {
    'temperature': 4.0,
    'alpha': 0.8,  # Higher weight on distillation from LLM
    'feature_matching_weight': 0.2,
    'value_distill_weight': 0.5,
    'progressive_beta': 0.9,

    # LLM-specific distillation
    'llm_reasoning_weight': 0.3,  # Weight for reasoning consistency
    'llm_confidence_weight': 0.2,  # Weight for confidence alignment
    'text_reasoning_loss': True,  # Enable text-based reasoning loss
}

# Memory configuration (optimized for LLM)
MEMORY_CONFIG = {
    'trajectory_buffer_size': 300,  # Reduced for memory efficiency
    'prioritized_replay': True,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,
    'min_trajectory_reward': 50,
}

# Training configuration (adjusted for LLM integration)
TRAINING_CONFIG = {
    'total_timesteps': 20000,  # Reduced due to improved LLM guidance
    'rollout_steps': 32,
    'num_ppo_epochs': 2,
    'batch_size': 8,  # Small batch for memory efficiency
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'max_grad_norm': 0.5,
    'target_kl': 0.01,
    'clip_value_loss': True,
    'human_cloning_steps': 1000,  # Reduced due to LLM guidance
    'focused_distillation_steps': 5000,
    'demo_processing_chunk_size': 2,

    # LLM-specific training
    'llm_warm_start': True,  # Use LLM heavily at start
    'llm_annealing': True,  # Reduce LLM usage over time
    'memory_cleanup_frequency': 50,  # Clean GPU memory every N steps
}

# Curriculum learning (enhanced with LLM)
CURRICULUM_CONFIG = {
    'enabled': True,
    'stages': [
        {'min_reward': 50, 'mentor_query_prob': 0.8, 'llm_query_prob': 0.6},
        {'min_reward': 100, 'mentor_query_prob': 0.5, 'llm_query_prob': 0.4},
        {'min_reward': 150, 'mentor_query_prob': 0.3, 'llm_query_prob': 0.2},
        {'min_reward': 200, 'mentor_query_prob': 0.1, 'llm_query_prob': 0.1},
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    'log_interval': 50,
    'eval_interval': 200,
    'save_interval': 500,
    'verbose': True,
    'num_eval_episodes': 3,

    # LLM-specific logging
    'log_llm_queries': True,
    'log_llm_reasoning': True,
    'save_llm_conversations': True,  # Save interesting LLM interactions
}

# Transfer Learning Configuration (disabled for LLM focus)
TRANSFER_LEARNING_CONFIG = {
    'enabled': False,
    'mentor_transfer_sources': [],
    'student_transfer_sources': [],
}

# Revolutionary features flags (enhanced for LLM)
REVOLUTIONARY_FEATURES = {
    'multimodal_mentor': True,
    'llm_powered_mentor': True,  # NEW: Enable LLM mentor
    'parallel_reasoning': True,
    'multi_action_execution': True,
    'active_querying': True,
    'progressive_distillation': True,
    'causal_understanding': True,
    'text_based_reasoning': True,  # NEW: Enable text reasoning
    'adaptive_llm_queries': True,  # NEW: Adaptive LLM query frequency
    'hybrid_neural_llm': True,  # NEW: Hybrid neural + LLM approach
    'meta_learning': False,  # Disabled for memory efficiency
}

# Performance optimization flags
OPTIMIZATION_CONFIG = {
    'enable_mixed_precision': True,
    'gradient_checkpointing': True,
    'compile_models': False,  # Disabled for compatibility
    'use_cpu_offload': False,  # Keep everything on GPU for speed
    'memory_efficient_attention': True,
    'dynamic_batch_sizing': False,  # Keep fixed for stability
}

# Hardware-specific optimizations for 2060S
HARDWARE_CONFIG = {
    'target_gpu': 'RTX_2060_SUPER',
    'vram_limit_gb': 8,
    'expected_llm_usage_gb': 3.5,  # Phi-3 Mini 4-bit quantized
    'reserved_training_gb': 4.0,  # For student training and activations
    'safety_margin_gb': 0.5,  # Safety buffer

    # Memory monitoring
    'enable_memory_monitoring': True,
    'memory_warning_threshold': 0.85,  # Warn at 85% VRAM usage
    'emergency_cleanup_threshold': 0.95,  # Emergency cleanup at 95%
}


# Validation that config is feasible for hardware
def validate_config_for_hardware():
    """Validate that the configuration is feasible for the target hardware"""
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, falling back to CPU")
        return False

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    if total_vram < HARDWARE_CONFIG['vram_limit_gb'] * 0.8:
        print(f"⚠️  Warning: Available VRAM ({total_vram:.1f}GB) may be insufficient")
        print(f"   Target configuration requires ~{HARDWARE_CONFIG['vram_limit_gb']}GB")
        return False

    expected_usage = (HARDWARE_CONFIG['expected_llm_usage_gb'] +
                      HARDWARE_CONFIG['reserved_training_gb'] +
                      HARDWARE_CONFIG['safety_margin_gb'])

    if expected_usage > total_vram:
        print(f"⚠️  Warning: Expected usage ({expected_usage:.1f}GB) exceeds available VRAM")
        return False

    print(f"✅ Configuration validated for {total_vram:.1f}GB VRAM")
    return True


# Auto-adjust configuration based on available memory
def auto_adjust_config():
    """Automatically adjust configuration based on available GPU memory"""
    if not torch.cuda.is_available():
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    if total_vram < 6:  # Low VRAM
        print("🔧 Auto-adjusting for low VRAM...")
        LLM_MENTOR_CONFIG['model_name'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        LLM_MENTOR_CONFIG['max_context_length'] = 1024
        TRAINING_CONFIG['batch_size'] = 4
        STUDENT_CONFIG['hidden_dim'] = 24
        MENTOR_CONFIG['hidden_dim'] = 48

    elif total_vram > 10:  # High VRAM
        print("🚀 Auto-adjusting for high VRAM...")
        LLM_MENTOR_CONFIG['max_context_length'] = 4096
        TRAINING_CONFIG['batch_size'] = 16
        STUDENT_CONFIG['hidden_dim'] = 48

    print(f"✅ Configuration auto-adjusted for {total_vram:.1f}GB VRAM")


# Initialize configuration
print(f"🔧 LLM-Enhanced Revolutionary AI Pipeline Configuration")
print(f"Device: {DEVICE}")

if validate_config_for_hardware():
    auto_adjust_config()
    print(f"🎯 Target LLM: {LLM_MENTOR_CONFIG['model_name']}")
    print(
        f"💾 Expected VRAM usage: ~{HARDWARE_CONFIG['expected_llm_usage_gb'] + HARDWARE_CONFIG['reserved_training_gb']:.1f}GB")
else:
    print("⚠️  Configuration may need manual adjustment")