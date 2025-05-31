# test_pipeline.py
"""
Test script to verify Revolutionary AI Pipeline components work correctly
"""

import torch
import numpy as np
import time
from typing import Dict, Any

from config import *
from environment import create_environment
from mentor import MultimodalMentor
from student import StudentAgent
from distillation import DistillationTrainer
from memory import PrioritizedReplayBuffer, TrajectoryBuffer
from utils import Logger


def test_environment():
    """Test environment functionality"""
    print("🧪 Testing Environment...")

    env = create_environment()
    print(f"✅ Environment created: {ENV_CONFIG['name']}")
    print(f"   State dim: {env.state_dim}")
    print(f"   Action space: {env.num_actions}")
    print(f"   Num envs: {ENV_CONFIG['num_envs']}")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")

    # Test step with multi-actions
    test_actions = [[0] for _ in range(ENV_CONFIG['num_envs'])]
    uncertainties = [0.5] * ENV_CONFIG['num_envs']

    next_obs, rewards, terminated, truncated, infos = env.step(test_actions, uncertainties)
    print(f"✅ Environment step successful")
    print(f"   Rewards: {rewards}")
    print(f"   Next obs shape: {next_obs.shape}")

    env.close()
    print("✅ Environment test completed\n")


def test_mentor():
    """Test mentor functionality"""
    print("🧠 Testing Mentor...")

    env = create_environment()
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    print(f"✅ Mentor created with {sum(p.numel() for p in mentor.parameters()):,} parameters")

    # Test forward pass
    obs, _ = env.reset(seed=42)
    state_tensor = env.get_state_tensor(obs)

    with torch.no_grad():
        outputs = mentor(state_tensor)

    print(f"✅ Mentor forward pass successful")
    print(f"   Policy logits shape: {outputs['policy_logits'].shape}")
    print(f"   Value shape: {outputs['value'].shape}")
    print(f"   Confidence: {outputs['confidence'].item():.3f}")

    # Test advice generation
    advice = mentor.get_advice(state_tensor[0])
    print(f"✅ Mentor advice generation successful")
    print(f"   Recommended actions: {advice.actions}")
    print(f"   Confidence: {advice.confidence:.3f}")
    print(f"   Strategy: {advice.strategy}")
    print(f"   Reasoning: {advice.reasoning[:2]}")  # First 2 reasoning steps

    env.close()
    print("✅ Mentor test completed\n")


def test_student():
    """Test student functionality"""
    print("🎓 Testing Student...")

    env = create_environment()
    student = StudentAgent(env.state_dim, env.num_actions)
    print(f"✅ Student created with {sum(p.numel() for p in student.parameters()):,} parameters")

    # Test forward pass
    obs, _ = env.reset(seed=42)
    state_tensor = env.get_state_tensor(obs)

    with torch.no_grad():
        outputs = student(state_tensor)

    print(f"✅ Student forward pass successful")
    print(f"   Primary logits shape: {outputs['primary_logits'].shape}")
    print(f"   Secondary logits count: {len(outputs['secondary_logits'])}")
    print(f"   Coordination weights shape: {outputs['coordination_weights'].shape}")
    print(f"   Uncertainty - total: {outputs['uncertainty']['total'].item():.3f}")
    print(f"   Uncertainty - epistemic: {outputs['uncertainty']['epistemic'].item():.3f}")

    # Test action selection
    actions, info = student.act(state_tensor)
    print(f"✅ Student action selection successful")
    print(f"   Selected actions: {actions}")
    print(f"   Number of actions: {info['num_actions']}")
    print(f"   Should query mentor: {info['should_query_mentor']}")
    print(f"   Reasoning confidence: {info['reasoning_confidence']:.3f}")

    env.close()
    print("✅ Student test completed\n")


def test_distillation():
    """Test distillation functionality"""
    print("🔄 Testing Distillation...")

    env = create_environment()
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    student = StudentAgent(env.state_dim, env.num_actions)

    # Add optimizers
    student.optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    mentor.optimizer = torch.optim.Adam(mentor.parameters(), lr=1e-4)

    distillation_trainer = DistillationTrainer(
        mentor, student,
        MENTOR_CONFIG['hidden_dim'],
        STUDENT_CONFIG['hidden_dim']
    )
    print(f"✅ Distillation trainer created")

    # Test distillation loss computation
    obs, _ = env.reset(seed=42)
    state_tensor = env.get_state_tensor(obs)

    with torch.no_grad():
        mentor_outputs = mentor(state_tensor)
        student_outputs = student(state_tensor)

    distill_losses = distillation_trainer.compute_distillation_loss(
        state_tensor, student_outputs, mentor_outputs
    )

    print(f"✅ Distillation loss computation successful")
    print(f"   Policy distillation loss: {distill_losses['policy_distill'].item():.4f}")
    print(f"   Feature matching loss: {distill_losses['feature_match'].item():.4f}")
    print(f"   Value distillation loss: {distill_losses['value_distill'].item():.4f}")
    print(f"   Total distillation loss: {distill_losses['total_distill'].item():.4f}")

    env.close()
    print("✅ Distillation test completed\n")


def test_memory():
    """Test memory components"""
    print("💾 Testing Memory...")

    # Test replay buffer
    replay_buffer = PrioritizedReplayBuffer(1000)
    print(f"✅ Replay buffer created")

    # Test trajectory buffer
    trajectory_buffer = TrajectoryBuffer(100)
    print(f"✅ Trajectory buffer created")

    # Add some dummy experiences
    from memory import Experience

    for i in range(10):
        exp = Experience(
            state=torch.randn(4),
            action=np.random.randint(0, 2),
            reward=np.random.random(),
            next_state=torch.randn(4),
            done=False,
            log_prob=torch.tensor(-0.5),
            value=torch.tensor(0.5),
            uncertainty={'total': 0.3, 'epistemic': 0.2, 'aleatoric': 0.1},
            mentor_advice=None
        )
        replay_buffer.add(exp, priority=1.0)

    print(f"✅ Added experiences to replay buffer")
    print(f"   Buffer size: {replay_buffer.size}")

    # Test sampling
    if replay_buffer.size > 0:
        experiences, indices, weights = replay_buffer.sample(5)
        print(f"✅ Sampling from replay buffer successful")
        print(f"   Sampled {len(experiences)} experiences")
        print(f"   Importance weights shape: {weights.shape}")

    print("✅ Memory test completed\n")


def test_integration():
    """Test full integration"""
    print("🔧 Testing Integration...")

    # Create all components
    env = create_environment()
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    student = StudentAgent(env.state_dim, env.num_actions)

    # Test full pipeline step
    obs, _ = env.reset(seed=42)
    state_tensor = env.get_state_tensor(obs)

    # Student acts
    actions, student_info = student.act(state_tensor)
    print(f"✅ Student action: {actions}")

    # Query mentor if needed
    if student_info['should_query_mentor']:
        advice = mentor.get_advice(state_tensor[0])
        print(f"✅ Mentor advice obtained: {advice.actions[:2]}")

    # Environment step
    if not isinstance(actions[0], list):
        action_lists = [actions]
    else:
        action_lists = actions

    uncertainties = [student_info['uncertainty']['total']]
    next_obs, rewards, terminated, truncated, infos = env.step(action_lists, uncertainties)

    print(f"✅ Environment step completed")
    print(f"   Reward: {rewards[0]:.3f}")
    print(f"   Done: {terminated[0] or truncated[0]}")

    env.close()
    print("✅ Integration test completed\n")


def test_performance():
    """Test performance characteristics"""
    print("⚡ Testing Performance...")

    env = create_environment()
    mentor = MultimodalMentor(env.state_dim, env.num_actions)
    student = StudentAgent(env.state_dim, env.num_actions)

    obs, _ = env.reset(seed=42)
    state_tensor = env.get_state_tensor(obs)

    # Benchmark inference times
    num_iterations = 100

    # Student inference
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            actions, info = student.act(state_tensor)
    student_time = (time.time() - start_time) / num_iterations

    # Mentor inference
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            advice = mentor.get_advice(state_tensor[0])
    mentor_time = (time.time() - start_time) / num_iterations

    print(f"✅ Performance benchmarking completed")
    print(f"   Student inference: {student_time * 1000:.2f}ms per step")
    print(f"   Mentor inference: {mentor_time * 1000:.2f}ms per step")
    print(f"   Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"   Mentor parameters: {sum(p.numel() for p in mentor.parameters()):,}")

    env.close()
    print("✅ Performance test completed\n")


def main():
    """Run all tests"""
    print("🚀 Revolutionary AI Pipeline - Component Testing\n")

    # Set device info
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    try:
        test_environment()
        test_mentor()
        test_student()
        test_distillation()
        test_memory()
        test_integration()
        test_performance()

        print("🎉 All tests passed! Revolutionary AI Pipeline is ready for training.")
        print("\nTo start training, run:")
        print("  python main.py")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()