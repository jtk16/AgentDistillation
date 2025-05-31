# test_pipeline.py
"""
Test script to verify Revolutionary AI Pipeline components work correctly
"""

import torch
import numpy as np
import time
from typing import Dict, Any
from collections import namedtuple

from config import *  # Import all config variables
from environment import create_environment
from mentor import MultimodalMentor
from student import StudentAgent
from distillation import DistillationTrainer
from memory import PrioritizedReplayBuffer, TrajectoryBuffer

# Experience is defined below if not imported from memory.py (which is typical)

# Define Experience namedtuple if it's used by memory components being tested
# This should ideally match the definition in memory.py if it exists there
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done',
    'log_prob', 'value', 'uncertainty', 'mentor_advice'
])


def test_environment():
    """Test environment functionality"""
    print("🧪 Testing Environment...")

    env = create_environment()  # Uses num_envs from ENV_CONFIG
    print(f"✅ Environment created: {ENV_CONFIG['name']}")
    print(f"   State dim (single env): {env.state_dim}")
    print(f"   Action space (single env): {env.num_actions}")  # This should be 2 for CartPole
    print(f"   Num envs in vector: {env.num_envs}")

    obs_np, info_dict = env.reset(seed=42)
    print(f"✅ Environment reset successful")
    print(f"   Observation shape from reset: {obs_np.shape}")  # (num_envs, state_dim)

    # Test step with multi-actions
    # Actions: List[List[int]], one inner list per env, each inner list contains action(s) for that env
    test_actions_list = [[np.random.randint(0, env.num_actions)] for _ in range(env.num_envs)]
    uncertainties_list = [0.5] * env.num_envs

    next_obs_np, rewards_np, terminated_np, truncated_np, infos_list = env.step(test_actions_list, uncertainties_list)
    print(f"✅ Environment step successful")
    print(f"   Rewards shape: {rewards_np.shape}, Rewards: {rewards_np}")
    print(f"   Next obs shape: {next_obs_np.shape}")

    env.close()
    print("✅ Environment test completed\n")


def test_mentor():
    """Test mentor functionality"""
    print("🧠 Testing Mentor...")

    env = create_environment()  # num_envs from ENV_CONFIG
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    print(f"✅ Mentor created with {sum(p.numel() for p in mentor.parameters()):,} parameters")

    obs_np, _ = env.reset(seed=42)
    state_tensor_batch = env.get_state_tensor(obs_np)  # (num_envs, state_dim)

    with torch.no_grad():
        outputs_batch = mentor(state_tensor_batch)

    print(f"✅ Mentor forward pass successful")
    print(f"   Policy logits shape: {outputs_batch['policy_logits'].shape}")  # (num_envs, num_actions)
    print(f"   Value shape: {outputs_batch['value'].shape}")  # (num_envs, 1)
    # Confidence is (num_envs, 1), print for first env in batch
    print(f"   Confidence (first in batch): {outputs_batch['confidence'][0].item():.3f}")

    # Test advice generation - get_advice expects a single state tensor (1, state_dim) or (state_dim,)
    single_state_for_advice = state_tensor_batch[0].unsqueeze(0)  # Take first env's state, ensure batch dim of 1
    advice = mentor.get_advice(single_state_for_advice)
    print(f"✅ Mentor advice generation successful")
    print(f"   Recommended actions: {advice.actions}")
    print(f"   Confidence (advice): {advice.confidence:.3f}")
    print(f"   Strategy: {advice.strategy}")
    print(f"   Reasoning: {advice.reasoning[:2] if advice.reasoning else 'N/A'}")

    env.close()
    print("✅ Mentor test completed\n")


def test_student():
    """Test student functionality"""
    print("🎓 Testing Student...")

    env = create_environment()  # num_envs from ENV_CONFIG
    student = StudentAgent(env.state_dim, env.num_actions)
    print(f"✅ Student created with {sum(p.numel() for p in student.parameters()):,} parameters")

    obs_np, _ = env.reset(seed=42)
    state_tensor_batch = env.get_state_tensor(obs_np)  # (num_envs, state_dim)

    with torch.no_grad():
        outputs_batch = student(state_tensor_batch)

    print(f"✅ Student forward pass successful")
    print(f"   Primary logits shape: {outputs_batch['primary_logits'].shape}")  # (num_envs, num_actions)
    secondary_logits_list = outputs_batch['secondary_logits']
    print(f"   Secondary logits count: {len(secondary_logits_list)}")
    if secondary_logits_list:
        print(f"   Secondary logits[0] shape: {secondary_logits_list[0].shape}")  # (num_envs, num_actions)
    print(
        f"   Coordination weights shape: {outputs_batch['coordination_weights'].shape}")  # (num_envs, num_action_heads)

    # Uncertainty outputs are dicts of tensors [num_envs, 1]
    print(f"   Uncertainty - total (first): {outputs_batch['uncertainty']['total'][0].item():.3f}")
    print(f"   Uncertainty - epistemic (first): {outputs_batch['uncertainty']['epistemic'][0].item():.3f}")

    # Test action selection (batched)
    actions_batch_list, info_batch_list = student.act(state_tensor_batch)  # Returns List[List[int]], List[Dict]

    print(f"✅ Student action selection successful for batch")
    if actions_batch_list and info_batch_list:  # Check if not empty
        print(f"   Selected actions (env 0): {actions_batch_list[0]}")
        print(f"   Number of actions (env 0): {info_batch_list[0]['num_actions']}")
        print(f"   Should query mentor (env 0): {info_batch_list[0]['should_query_mentor']}")
        print(f"   Reasoning confidence (env 0): {info_batch_list[0]['reasoning_confidence']:.3f}")
    else:
        print("   Warning: student.act returned empty lists for actions/info.")

    env.close()
    print("✅ Student test completed\n")


def test_distillation():
    """Test distillation functionality"""
    print("🔄 Testing Distillation...")

    env = create_environment()
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    student = StudentAgent(env.state_dim, env.num_actions)

    student.optimizer = torch.optim.Adam(student.parameters(), lr=STUDENT_CONFIG['learning_rate'])
    # mentor.optimizer = torch.optim.Adam(mentor.parameters(), lr=MENTOR_CONFIG['learning_rate']) # Mentor usually not trained here

    distillation_trainer = DistillationTrainer(
        mentor, student,
        MENTOR_CONFIG['hidden_dim'],
        STUDENT_CONFIG['hidden_dim']
    )
    print(f"✅ Distillation trainer created")

    obs_np, _ = env.reset(seed=42)
    state_tensor_batch = env.get_state_tensor(obs_np)

    with torch.no_grad():  # Mentor and student outputs for loss computation only
        mentor_outputs = mentor(state_tensor_batch)
        student_outputs = student(state_tensor_batch)

    distill_losses = distillation_trainer.compute_distillation_loss(
        state_tensor_batch, student_outputs, mentor_outputs
    )

    print(f"✅ Distillation loss computation successful")
    print(f"   Policy distillation loss: {distill_losses['policy_distill'].item():.4f}")
    print(f"   Feature matching loss: {distill_losses['feature_match'].item():.4f}")
    print(f"   Value distillation loss: {distill_losses['value_distill'].item():.4f}")
    if 'causal_transfer' in distill_losses:
        print(f"   Causal transfer loss: {distill_losses['causal_transfer'].item():.4f}")
    print(f"   Total distillation loss: {distill_losses['total_distill'].item():.4f}")

    env.close()
    print("✅ Distillation test completed\n")


def test_memory():
    """Test memory components"""
    print("💾 Testing Memory...")

    # Create a single environment instance for getting state/action dimensions
    env_for_dims = create_environment(num_envs=1)

    replay_buffer = PrioritizedReplayBuffer(
        capacity=MEMORY_CONFIG['trajectory_buffer_size'],
        alpha=MEMORY_CONFIG['priority_alpha'],
        beta=MEMORY_CONFIG['priority_beta']
    )
    print(f"✅ Replay buffer created")

    trajectory_buffer = TrajectoryBuffer(
        max_trajectories=MEMORY_CONFIG['trajectory_buffer_size'] // 100
    )
    print(f"✅ Trajectory buffer created")

    for i in range(10):
        exp = Experience(
            state=torch.randn(env_for_dims.state_dim, device=DEVICE),
            action=np.random.randint(0, env_for_dims.num_actions),
            reward=np.random.random(),
            next_state=torch.randn(env_for_dims.state_dim, device=DEVICE),
            done=False,
            log_prob=torch.tensor(-0.5, device=DEVICE),
            value=torch.tensor(0.5, device=DEVICE),
            uncertainty={'total': 0.3, 'epistemic': 0.2, 'aleatoric': 0.1},
            mentor_advice=None
        )
        replay_buffer.add(exp, priority=1.0)

    env_for_dims.close()  # Close the temporary environment

    print(f"✅ Added experiences to replay buffer")
    print(f"   Buffer size: {replay_buffer.size}")

    if replay_buffer.size > 0:
        sample_batch_size = min(5, replay_buffer.size)
        experiences, indices, weights = replay_buffer.sample(sample_batch_size)
        print(f"✅ Sampling from replay buffer successful")
        print(f"   Sampled {len(experiences)} experiences")
        if weights is not None and len(weights) > 0:  # weights could be None or empty if sample fails
            print(f"   Importance weights shape: {weights.shape}")
        else:
            print(f"   Importance weights: None or empty")

    print("✅ Memory test completed\n")


def test_integration():
    """Test full integration for a single step in a single environment"""
    print("🔧 Testing Integration (single env focus for step logic)...")

    env_single = create_environment(num_envs=1)
    mentor_single = MultimodalMentor(env_single.state_dim, env_single.num_actions)
    student_single = StudentAgent(env_single.state_dim, env_single.num_actions)

    obs_single_np, _ = env_single.reset(seed=42)
    state_tensor_single_batch = env_single.get_state_tensor(obs_single_np)  # (1, state_dim)

    # student.act returns List[List[int]], List[Dict]
    actions_batch_list, info_batch_list = student_single.act(state_tensor_single_batch)

    if not actions_batch_list or not info_batch_list:
        print("❌ Student.act returned empty lists in integration test. Skipping further checks.")
        env_single.close()
        raise AssertionError("Student.act returned empty lists in integration test.")

    student_actions_for_env0 = actions_batch_list[0]  # List[int] for the first (only) env
    student_info_for_env0 = info_batch_list[0]  # Dict for the first (only) env

    print(f"✅ Student action list (env 0): {student_actions_for_env0}")

    if student_info_for_env0.get('should_query_mentor', False):
        advice = mentor_single.get_advice(state_tensor_single_batch)  # Pass (1, state_dim)
        print(f"✅ Mentor advice obtained: {advice.actions[:2] if advice.actions else 'N/A'}")

    # Env step expects actions as List[List[int]]
    actions_for_env_step = [student_actions_for_env0]  # Wrap: [[act1, act2]]
    uncertainty_for_env_step = [student_info_for_env0.get('uncertainty', {}).get('total', 0.5)]

    next_obs_np, rewards_np, term_np, trunc_np, infos_list = env_single.step(
        actions_for_env_step, uncertainty_for_env_step
    )

    print(f"✅ Environment step completed")
    print(f"   Reward: {rewards_np[0]:.3f}")
    print(f"   Done: {term_np[0] or trunc_np[0]}")

    env_single.close()
    print("✅ Integration test completed\n")


def test_performance():
    """Test performance characteristics using batched environment"""
    print("⚡ Testing Performance (batched)...")

    env_batched = create_environment()  # Uses num_envs from ENV_CONFIG
    mentor_batched = MultimodalMentor(env_batched.state_dim, env_batched.num_actions)
    student_batched = StudentAgent(env_batched.state_dim, env_batched.num_actions)

    obs_batched_np, _ = env_batched.reset(seed=42)
    state_tensor_batched = env_batched.get_state_tensor(obs_batched_np)

    num_iterations = 100

    # Student inference (batched)
    start_time_student = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            student_actions_batch, student_info_batch = student_batched.act(state_tensor_batched)
    student_time = (time.time() - start_time_student) / num_iterations

    # Mentor inference (batched forward)
    start_time_mentor_fwd = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            mentor_outputs = mentor_batched(state_tensor_batched)
    mentor_fwd_time = (time.time() - start_time_mentor_fwd) / num_iterations

    # Mentor advice (single instance)
    single_state_for_advice = state_tensor_batched[0].unsqueeze(0)
    start_time_mentor_advice = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            advice = mentor_batched.get_advice(single_state_for_advice)
    mentor_advice_time = (time.time() - start_time_mentor_advice) / num_iterations

    print(f"✅ Performance benchmarking completed")
    print(f"   Student batched 'act': {student_time * 1000:.2f}ms (for {env_batched.num_envs} envs)")
    print(f"   Mentor batched 'forward': {mentor_fwd_time * 1000:.2f}ms (for {env_batched.num_envs} envs)")
    print(f"   Mentor single 'get_advice': {mentor_advice_time * 1000:.2f}ms")
    print(f"   Student parameters: {sum(p.numel() for p in student_batched.parameters()):,}")
    print(f"   Mentor parameters: {sum(p.numel() for p in mentor_batched.parameters()):,}")

    env_batched.close()
    print("✅ Performance test completed\n")


def main():
    """Run all tests"""
    print("🚀 Revolutionary AI Pipeline - Component Testing\n")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(DEVICE)}")
        except Exception as e:
            print(f"Could not get CUDA device name: {e}")
    print()

    all_tests_passed_flag = True
    tests_to_run = [
        test_environment, test_mentor, test_student,
        test_distillation, test_memory, test_integration, test_performance
    ]

    for test_func in tests_to_run:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} FAILED with error: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            all_tests_passed_flag = False
        print("-" * 30)  # Separator

    if all_tests_passed_flag:
        print("🎉 All tests passed! Revolutionary AI Pipeline is ready for training.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    return all_tests_passed_flag


if __name__ == "__main__":
    main()