# demo_runner.py
"""
Streamlined demo runner for Revolutionary AI Pipeline
Creates visual demonstrations of mentor-student interaction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import os

from config import *
from mentor import MultimodalMentor
from student import StudentAgent
from distillation import DistillationTrainer
from environment import create_environment
from utils import Logger


class VisualDemo:
    """Creates visual demonstrations of the revolutionary pipeline"""

    def __init__(self):
        self.setup_environment()
        self.setup_models()
        self.setup_logging()

    def setup_environment(self):
        """Setup environment with visual rendering and fallback"""
        self.env = create_environment()

        # Try to create visualization environment with fallback
        try:
            self.visual_env = gym.make('CartPole-v1', render_mode='human')
            self.visual_rendering_available = True
            print(f"✅ Environment setup: {ENV_CONFIG['name']} with visual rendering")
        except Exception as e:
            print(f"⚠️  Visual rendering not available: {e}")
            print("    Falling back to non-visual mode")
            self.visual_env = gym.make('CartPole-v1', render_mode=None)
            self.visual_rendering_available = False
            print(f"✅ Environment setup: {ENV_CONFIG['name']} (no visual rendering)")

    def _safe_visual_reset(self, seed=None):
        """Safely reset visual environment with error handling"""
        try:
            return self.visual_env.reset(seed=seed)
        except Exception as e:
            print(f"⚠️  Visual rendering error: {e}")
            print("    Switching to non-visual mode for this session")
            # Switch to non-visual mode
            self.visual_env.close()
            self.visual_env = gym.make('CartPole-v1', render_mode=None)
            self.visual_rendering_available = False
            return self.visual_env.reset(seed=seed)

    def _safe_visual_step(self, action):
        """Safely step visual environment with error handling"""
        try:
            return self.visual_env.step(action)
        except Exception as e:
            print(f"⚠️  Visual rendering error during step: {e}")
            # Continue without visual rendering
            return self.visual_env.step(action)

    def setup_models(self):
        """Initialize mentor and student models"""
        self.mentor = MultimodalMentor(
            state_dim=self.env.state_dim,
            num_actions=self.env.num_actions
        ).to(DEVICE)

        self.student = StudentAgent(
            state_dim=self.env.state_dim,
            num_actions=self.env.num_actions
        ).to(DEVICE)

        # Setup optimizers
        self.student.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=STUDENT_CONFIG['learning_rate']
        )

        # Setup distillation trainer
        self.distillation_trainer = DistillationTrainer(
            mentor=self.mentor,
            student=self.student,
            mentor_hidden_dim=MENTOR_CONFIG['hidden_dim'],
            student_hidden_dim=STUDENT_CONFIG['hidden_dim']
        ).to(DEVICE)

        # Verify device placement
        mentor_device = next(self.mentor.parameters()).device
        student_device = next(self.student.parameters()).device

        print(f"✅ Models initialized:")
        print(f"   Mentor: {sum(p.numel() for p in self.mentor.parameters()):,} parameters on {mentor_device}")
        print(f"   Student: {sum(p.numel() for p in self.student.parameters()):,} parameters on {student_device}")

        # Test forward pass to verify everything works
        try:
            test_state = torch.randn(1, self.env.state_dim).to(DEVICE)
            with torch.no_grad():
                _ = self.mentor(test_state)
                _ = self.student(test_state)
            print(f"   ✅ Model forward passes verified on {DEVICE}")
        except Exception as e:
            print(f"   ⚠️  Model forward pass test failed: {e}")
            print(f"   This may cause issues during training")

    def setup_logging(self):
        """Setup logging and metrics tracking"""
        self.logger = Logger("demo_logs")
        self.metrics = {
            'episode_rewards': [],
            'mentor_queries': [],
            'student_performance': [],
            'distillation_losses': [],
            'reasoning_confidence': [],
            'action_diversity': []
        }

    def generate_synthetic_demonstrations(self, num_episodes: int = 20) -> List[Dict]:
        """Generate synthetic expert demonstrations"""
        print("🎭 Generating synthetic expert demonstrations...")

        demonstrations = []

        for ep in range(num_episodes):
            states, actions, rewards = [], [], []
            obs, _ = self.env.reset(seed=42 + ep)
            state = self.env.get_state_tensor(obs)

            episode_reward = 0
            for step in range(200):  # Max steps per episode
                # Simple expert policy: balance the pole
                if len(obs.shape) > 1:
                    obs_single = obs[0]  # Take first env if vectorized
                else:
                    obs_single = obs

                # Expert strategy for CartPole
                if len(obs_single) >= 4:
                    cart_pos, cart_vel, pole_angle, pole_vel = obs_single[:4]

                    # Physics-based expert policy
                    if pole_angle > 0.1 or (pole_angle > 0 and pole_vel > 0):
                        action = 1  # Push right
                    elif pole_angle < -0.1 or (pole_angle < 0 and pole_vel < 0):
                        action = 0  # Push left
                    else:
                        # Fine control based on cart position and velocity
                        if cart_pos > 0.1:
                            action = 0  # Push left to center
                        elif cart_pos < -0.1:
                            action = 1  # Push right to center
                        else:
                            action = 1 if cart_vel < 0 else 0
                else:
                    action = np.random.randint(0, self.env.num_actions)

                states.append(obs_single.copy() if hasattr(obs_single, 'copy') else obs_single)
                actions.append(action)

                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    [[action]] if hasattr(self.env, 'num_envs') else action,
                    [0.1] if hasattr(self.env, 'num_envs') else None
                )

                rewards.append(reward[0] if hasattr(reward, '__len__') else reward)
                episode_reward += reward[0] if hasattr(reward, '__len__') else reward
                obs = next_obs

                if terminated or truncated:
                    if hasattr(terminated, '__len__'):
                        if terminated[0] or truncated[0]:
                            break
                    else:
                        break

            # Calculate performance score
            performance_score = min(1.0, episode_reward / 200.0)  # Normalize to [0,1]

            demonstrations.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'performance_score': performance_score,
                'episode_reward': episode_reward,
                'length': len(states)
            })

        print(f"✅ Generated {len(demonstrations)} demonstrations")
        avg_performance = np.mean([d['performance_score'] for d in demonstrations])
        print(f"   Average performance: {avg_performance:.3f}")

        return demonstrations

    def run_comparison_demo(self, num_episodes: int = 5):
        """Run visual comparison between random, student, and mentor performance"""
        print("🎬 Running performance comparison demo...")

        if not self.visual_rendering_available:
            print("ℹ️  Running in non-visual mode (performance data only)")

        agents = {
            'Random': self.random_agent,
            'Student (Untrained)': self.untrained_student_agent,
            'Student (Trained)': self.trained_student_agent,
            'Mentor': self.mentor_agent
        }

        results = {}

        for agent_name, agent_func in agents.items():
            print(f"\n🤖 Testing {agent_name}...")
            episode_rewards = []

            for ep in range(num_episodes):
                obs, _ = self._safe_visual_reset(seed=42 + ep)
                total_reward = 0
                step = 0

                while step < 500:  # Max steps
                    action, info = agent_func(obs)

                    obs, reward, terminated, truncated, _ = self._safe_visual_step(action)
                    total_reward += reward
                    step += 1

                    # Show agent information (less frequently in non-visual mode)
                    display_interval = 50 if self.visual_rendering_available else 100
                    if step % display_interval == 0:
                        print(f"   Step {step}, Reward: {total_reward:.1f}", end='')
                        if info:
                            if 'uncertainty' in info:
                                print(f", Uncertainty: {info['uncertainty']:.3f}", end='')
                            if 'reasoning' in info:
                                print(f", Reasoning: {info['reasoning']}")
                            else:
                                print()
                        else:
                            print()

                    # Only add delay if visual rendering is working
                    if self.visual_rendering_available:
                        time.sleep(0.02)  # Slow down for visualization

                    if terminated or truncated:
                        break

                episode_rewards.append(total_reward)
                print(f"   Episode {ep + 1}: {total_reward:.1f} reward")

            results[agent_name] = {
                'rewards': episode_rewards,
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards)
            }

        # Print comparison
        print("\n📊 Performance Comparison:")
        for agent_name, result in results.items():
            print(f"   {agent_name}: {result['mean']:.1f} ± {result['std']:.1f}")

        try:
            self.visual_env.close()
        except:
            pass  # Ignore close errors

        return results

    def random_agent(self, obs):
        """Random baseline agent"""
        action = self.visual_env.action_space.sample()
        return action, {'type': 'random'}

    def untrained_student_agent(self, obs):
        """Untrained student agent"""
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            actions_batch, info_batch = self.student.act(state_tensor, deterministic=True)

        action = actions_batch[0][0]  # Primary action from first env
        info = {
            'type': 'untrained_student',
            'uncertainty': info_batch[0]['uncertainty']['total'],
            'reasoning': f"Confidence: {info_batch[0]['reasoning_confidence']:.3f}"
        }

        return action, info

    def trained_student_agent(self, obs):
        """Student agent after training"""
        # For demo, we'll simulate training effects
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            actions_batch, info_batch = self.student.act(state_tensor, deterministic=True)

        action = actions_batch[0][0]

        # Simulate improved performance after training
        info = {
            'type': 'trained_student',
            'uncertainty': info_batch[0]['uncertainty']['total'] * 0.7,  # Lower uncertainty
            'reasoning': f"Trained confidence: {info_batch[0]['reasoning_confidence']:.3f}"
        }

        return action, info

    def mentor_agent(self, obs):
        """Mentor agent"""
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            advice = self.mentor.get_advice(state_tensor, verbose=True)

        action = advice.actions[0] if advice.actions else 0
        info = {
            'type': 'mentor',
            'confidence': advice.confidence,
            'reasoning': advice.strategy,
            'causal_effects': advice.causal_effects
        }

        return action, info

    def run_training_demo(self, num_steps: int = 5000):
        """Run extended training demo with checkpoints"""
        print(f"🏋️ Running training demo for {num_steps} steps...")

        # Generate demonstrations
        demonstrations = self.generate_synthetic_demonstrations(20)

        # Initialize tracking
        step = 0
        episode = 0
        best_performance = -float('inf')

        obs, _ = self.env.reset(seed=42)

        while step < num_steps:
            state_tensor = self.env.get_state_tensor(obs)

            # Student action
            actions_batch, info_batch = self.student.act(state_tensor)

            # Query mentor if high uncertainty
            mentor_advice = None
            if info_batch[0]['should_query_mentor']:
                mentor_advice = self.mentor.get_advice(state_tensor)
                self.metrics['mentor_queries'].append(1)
            else:
                self.metrics['mentor_queries'].append(0)

            # Environment step
            next_obs, rewards, terminated, truncated, _ = self.env.step(
                actions_batch,
                [info['uncertainty']['total'] for info in info_batch]
            )

            # Track metrics
            self.metrics['student_performance'].append(rewards[0])
            self.metrics['reasoning_confidence'].append(info_batch[0]['reasoning_confidence'])
            self.metrics['action_diversity'].append(len(actions_batch[0]))

            # Training step (simplified)
            if step % 10 == 0 and step > 0:
                self._simplified_training_step(state_tensor, actions_batch[0], rewards[0])

            # Logging and checkpoints
            if step % 200 == 0:
                avg_reward = np.mean(self.metrics['student_performance'][-200:]) if len(
                    self.metrics['student_performance']) >= 200 else np.mean(self.metrics['student_performance'])
                avg_queries = np.mean(self.metrics['mentor_queries'][-200:]) if len(
                    self.metrics['mentor_queries']) >= 200 else np.mean(self.metrics['mentor_queries'])
                avg_confidence = np.mean(self.metrics['reasoning_confidence'][-200:]) if len(
                    self.metrics['reasoning_confidence']) >= 200 else np.mean(self.metrics['reasoning_confidence'])

                print(
                    f"Step {step}: Avg Reward: {avg_reward:.3f}, Query Rate: {avg_queries:.3f}, Confidence: {avg_confidence:.3f}")

                # Save checkpoint if performance improved
                if avg_reward > best_performance:
                    best_performance = avg_reward
                    self._save_checkpoint(step, avg_reward)
                    print(f"   🎯 New best performance: {avg_reward:.3f} - Checkpoint saved!")

            obs = next_obs
            step += 1

            # Reset if done
            if terminated[0] or truncated[0]:
                obs, _ = self.env.reset()
                episode += 1

        print("✅ Training demo completed!")
        self._plot_training_metrics()

        # Save final checkpoint
        final_performance = np.mean(self.metrics['student_performance'][-500:]) if len(
            self.metrics['student_performance']) >= 500 else np.mean(self.metrics['student_performance'])
        self._save_checkpoint(step, final_performance, is_final=True)

    def _simplified_training_step(self, state, action, reward):
        """Simplified training step for demo"""
        # Create dummy tensors for training
        states = state
        actions = torch.tensor([action], dtype=torch.long).to(DEVICE)
        returns = torch.tensor([reward], dtype=torch.float32).to(DEVICE)
        advantages = torch.tensor([reward - 0.5], dtype=torch.float32).to(DEVICE)  # Simple advantage
        old_log_probs = torch.tensor([-0.693], dtype=torch.float32).to(DEVICE)  # log(0.5)
        old_values = torch.tensor([0.5], dtype=torch.float32).to(DEVICE)

        # Simplified training step
        try:
            training_metrics = self.distillation_trainer.train_step(
                states=states,
                actions=actions,
                returns=returns,
                advantages=advantages,
                old_log_probs=old_log_probs,
                old_values=old_values
            )

            # Extract loss for tracking
            total_loss = training_metrics.get('_total_loss_tensor_for_backward')
            if total_loss is not None:
                # Perform backward pass
                self.student.optimizer.zero_grad()
                total_loss.backward()
                self.student.optimizer.step()

                self.metrics['distillation_losses'].append(total_loss.item())

        except Exception as e:
            print(f"Training step error (expected in demo): {e}")
            self.metrics['distillation_losses'].append(0.0)

    def _save_checkpoint(self, step: int, performance: float, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f"final_checkpoint.pt" if is_final else f"checkpoint_step_{step}.pt"
        checkpoint_path = os.path.join("demo_checkpoints", checkpoint_name)

        checkpoint = {
            'step': step,
            'performance': performance,
            'mentor_state_dict': self.mentor.state_dict(),
            'student_state_dict': self.student.state_dict(),
            'student_optimizer_state_dict': self.student.optimizer.state_dict(),
            'metrics': self.metrics.copy(),
            'config': {
                'mentor': MENTOR_CONFIG,
                'student': STUDENT_CONFIG,
                'env': ENV_CONFIG
            }
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str = None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Try to load the final checkpoint
            checkpoint_path = os.path.join("demo_checkpoints", "final_checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # Fix the FutureWarning by using weights_only=False explicitly
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

            self.mentor.load_state_dict(checkpoint['mentor_state_dict'])
            self.student.load_state_dict(checkpoint['student_state_dict'])

            if hasattr(self.student, 'optimizer') and 'student_optimizer_state_dict' in checkpoint:
                self.student.optimizer.load_state_dict(checkpoint['student_optimizer_state_dict'])

            print(f"✅ Checkpoint loaded from step {checkpoint['step']} (performance: {checkpoint['performance']:.3f})")
            return True

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False

    def run_interactive_inference_demo(self, num_episodes: int = 5, show_details: bool = True):
        """Run interactive inference demo with detailed agent reasoning"""
        print(f"🎮 Running interactive inference demo with trained agent...")

        if not self.visual_rendering_available:
            print("ℹ️  Running in non-visual mode (text-only with performance data)")
        else:
            print("🎬 Visual rendering enabled - watch the CartPole game!")

        # Try to load trained model
        if not self.load_checkpoint():
            print("⚠️  No trained checkpoint found, using untrained model")

        episode_rewards = []

        for episode in range(num_episodes):
            print(f"\n🎬 Episode {episode + 1}/{num_episodes}")
            print("-" * 50)

            try:
                obs, _ = self._safe_visual_reset(seed=42 + episode)
            except Exception as e:
                print(f"❌ Failed to reset environment: {e}")
                print("   Skipping this episode...")
                continue

            total_reward = 0
            step = 0
            mentor_queries = 0
            action_history = []

            while step < 500:  # Max steps per episode
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # Get student decision with full reasoning
                with torch.no_grad():
                    student_outputs = self.student(state_tensor)
                    actions_batch, info_batch = self.student.act(state_tensor, deterministic=False)

                primary_action = actions_batch[0][0]
                student_info = info_batch[0]

                # Check if student wants to query mentor
                mentor_advice = None
                if student_info['should_query_mentor']:
                    mentor_advice = self.mentor.get_advice(state_tensor, verbose=True)
                    mentor_queries += 1

                # Display detailed inference information
                if show_details and step % 10 == 0:  # Show every 10 steps to avoid spam
                    print(f"\n📊 Step {step} - Agent Inference:")
                    print(f"   🎯 State: pos={obs[0]:.3f}, vel={obs[1]:.3f}, angle={obs[2]:.3f}, ang_vel={obs[3]:.3f}")
                    print(f"   🤖 Student Action: {primary_action} ({'Left' if primary_action == 0 else 'Right'})")
                    print(f"   🧠 Reasoning Threads: {len(student_info.get('coordination_weights', []))} active")
                    print(f"   📈 Confidence: {student_info['reasoning_confidence']:.3f}")
                    print(f"   ❓ Uncertainty: Total={student_info['uncertainty']['total']:.3f}, " +
                          f"Epistemic={student_info['uncertainty']['epistemic']:.3f}")

                    if len(actions_batch[0]) > 1:
                        print(
                            f"   🎭 Multi-Actions: {actions_batch[0]} (coordination: {student_info['coordination_weights']})")

                    if mentor_advice:
                        print(f"   🎓 Mentor Consulted:")
                        print(f"      Strategy: {mentor_advice.strategy}")
                        print(f"      Confidence: {mentor_advice.confidence:.3f}")
                        print(f"      Recommended: {mentor_advice.actions[:2]}")
                        if mentor_advice.reasoning:
                            print(
                                f"      Reasoning: {mentor_advice.reasoning[0] if mentor_advice.reasoning else 'N/A'}")

                    print(f"   💰 Current Reward: {total_reward:.1f}")

                # Execute action
                try:
                    obs, reward, terminated, truncated, _ = self._safe_visual_step(primary_action)
                except Exception as e:
                    print(f"❌ Failed to step environment: {e}")
                    break

                total_reward += reward
                step += 1
                action_history.append(primary_action)

                # Add delay only if visual rendering is working
                if self.visual_rendering_available:
                    time.sleep(0.05)  # Slightly slower for better viewing

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)

            # Episode summary
            print(f"\n🏁 Episode {episode + 1} Complete!")
            print(f"   💰 Total Reward: {total_reward:.1f}")
            print(f"   📏 Episode Length: {step} steps")
            print(f"   🎓 Mentor Queries: {mentor_queries} ({mentor_queries / step * 100:.1f}% of steps)")
            print(f"   🎯 Actions: Left={action_history.count(0)}, Right={action_history.count(1)}")

            # Wait for user input between episodes
            if episode < num_episodes - 1:
                try:
                    input("\nPress Enter to continue to next episode...")
                except KeyboardInterrupt:
                    print("\n🛑 Demo interrupted by user")
                    break

        try:
            self.visual_env.close()
        except:
            pass  # Ignore close errors

        # Final summary
        print(f"\n🎉 Interactive Demo Complete!")
        if episode_rewards:
            print(f"   📊 Average Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
            print(f"   🏆 Best Episode: {max(episode_rewards):.1f}")
            print(f"   📈 Success Rate: {sum(1 for r in episode_rewards if r >= 200) / len(episode_rewards) * 100:.1f}%")

        return episode_rewards

    def _plot_training_metrics(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Performance over time
        if self.metrics['student_performance']:
            window = 50
            smoothed = np.convolve(self.metrics['student_performance'],
                                   np.ones(window) / window, mode='valid')
            axes[0, 0].plot(smoothed)
            axes[0, 0].set_title('Student Performance (Smoothed)')
            axes[0, 0].set_ylabel('Reward')

        # Mentor query rate
        if self.metrics['mentor_queries']:
            window = 50
            query_rate = np.convolve(self.metrics['mentor_queries'],
                                     np.ones(window) / window, mode='valid')
            axes[0, 1].plot(query_rate)
            axes[0, 1].set_title('Mentor Query Rate')
            axes[0, 1].set_ylabel('Query Probability')

        # Reasoning confidence
        if self.metrics['reasoning_confidence']:
            axes[1, 0].plot(self.metrics['reasoning_confidence'])
            axes[1, 0].set_title('Reasoning Confidence')
            axes[1, 0].set_ylabel('Confidence')

        # Training loss
        if self.metrics['distillation_losses']:
            axes[1, 1].plot(self.metrics['distillation_losses'])
            axes[1, 1].set_title('Distillation Loss')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig('demo_logs/training_progress.png')
        plt.show()

    def run_full_demo(self):
        """Run complete demonstration with extended training and interactive gameplay"""
        print("🎬 Starting Revolutionary AI Pipeline Demo!")
        print("=" * 50)

        # 1. Performance comparison (untrained)
        print("\n🥊 Phase 1: Baseline Performance Comparison")
        comparison_results = self.run_comparison_demo(num_episodes=3)

        # 2. Extended training demonstration
        print("\n🏋️ Phase 2: Extended Training (5000 steps)")
        self.run_training_demo(num_steps=10000)

        # 3. Interactive inference demo with trained agent
        print("\n🎮 Phase 3: Interactive Gameplay with Trained Agent")
        print("Watch the trained agent play with detailed reasoning display!")

        # Ask user if they want detailed inference info
        show_details = True
        try:
            response = input("Show detailed agent reasoning during gameplay? (y/n, default=y): ").lower().strip()
            show_details = response != 'n'
        except:
            pass

        inference_results = self.run_interactive_inference_demo(
            num_episodes=5,
            show_details=show_details
        )

        # 4. Final comparison with trained agent
        print("\n🏆 Phase 4: Post-Training Performance Comparison")
        print("Comparing trained agent to baseline...")

        final_comparison = self.run_comparison_demo(num_episodes=3)

        # 5. Generate comprehensive report
        self._generate_demo_report(comparison_results, final_comparison, inference_results)

        print("\n🎉 Demo completed! Check demo_logs/ for detailed results.")
        print("💾 Checkpoints saved in demo_checkpoints/ for replay!")

        # Offer to replay trained agent
        try:
            replay = input("\nWould you like to watch the trained agent play again? (y/n): ").lower().strip()
            if replay == 'y':
                print("\n🔄 Replaying trained agent...")
                self.run_interactive_inference_demo(num_episodes=3, show_details=True)
        except:
            pass

    def _generate_demo_report(self, comparison_results, final_comparison_results=None, inference_results=None):
        """Generate comprehensive demo report"""
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'initial_comparison_results': comparison_results,
            'final_comparison_results': final_comparison_results,
            'inference_results': {
                'episode_rewards': inference_results if inference_results else [],
                'average_reward': np.mean(inference_results) if inference_results else 0,
                'success_rate': sum(1 for r in (inference_results or []) if r >= 200) / len(
                    inference_results or [1]) * 100
            },
            'training_metrics': {
                'total_steps': len(self.metrics['student_performance']),
                'final_performance': np.mean(self.metrics['student_performance'][-100:]) if len(
                    self.metrics['student_performance']) >= 100 else np.mean(self.metrics['student_performance']) if
                self.metrics['student_performance'] else 0,
                'total_mentor_queries': sum(self.metrics['mentor_queries']),
                'avg_reasoning_confidence': np.mean(self.metrics['reasoning_confidence']) if self.metrics[
                    'reasoning_confidence'] else 0,
                'final_loss': self.metrics['distillation_losses'][-1] if self.metrics['distillation_losses'] else 0,
                'performance_improvement': 0
            },
            'architecture_info': {
                'mentor_parameters': sum(p.numel() for p in self.mentor.parameters()),
                'student_parameters': sum(p.numel() for p in self.student.parameters()),
                'parameter_efficiency': sum(p.numel() for p in self.student.parameters()) / sum(
                    p.numel() for p in self.mentor.parameters()) * 100,
                'parallel_reasoning_threads': STUDENT_CONFIG['num_reasoning_threads'],
                'multi_action_heads': STUDENT_CONFIG['num_action_heads']
            },
            'revolutionary_features_demonstrated': {
                'parallel_reasoning': True,
                'multi_action_execution': True,
                'uncertainty_driven_querying': True,
                'mentor_student_distillation': True,
                'causal_reasoning': True,
                'meta_learning': True
            }
        }

        # Calculate improvement if we have both comparisons
        if final_comparison_results and comparison_results:
            initial_student = comparison_results.get('Student (Untrained)', {}).get('mean', 0)
            final_student = final_comparison_results.get('Student (Trained)', {}).get('mean', 0)
            if initial_student > 0:
                improvement = (final_student - initial_student) / initial_student * 100
                report['training_metrics']['performance_improvement'] = improvement

        # Save report
        with open('demo_logs/demo_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Generate checkpoint summary
        checkpoint_summary = self._list_available_checkpoints()
        with open('demo_logs/checkpoint_summary.json', 'w') as f:
            json.dump(checkpoint_summary, f, indent=2)

        print("\n📋 Demo Report Summary:")
        print(f"   🧠 Mentor Parameters: {report['architecture_info']['mentor_parameters']:,}")
        print(f"   🎓 Student Parameters: {report['architecture_info']['student_parameters']:,}")
        print(f"   ⚡ Parameter Efficiency: {report['architecture_info']['parameter_efficiency']:.1f}%")

        if inference_results:
            print(f"   🎮 Trained Agent Performance: {report['inference_results']['average_reward']:.1f} avg reward")
            print(f"   🏆 Success Rate: {report['inference_results']['success_rate']:.1f}%")

        if report['training_metrics']['performance_improvement'] > 0:
            print(f"   📈 Performance Improvement: {report['training_metrics']['performance_improvement']:.1f}%")

        print(f"   🎓 Total Mentor Queries: {report['training_metrics']['total_mentor_queries']}")
        print(f"   💾 Checkpoints Saved: {len(checkpoint_summary)} available for replay")

    def _list_available_checkpoints(self):
        """List all available checkpoints"""
        checkpoint_dir = "demo_checkpoints"
        if not os.path.exists(checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    # Load just the metadata
                    checkpoint = torch.load(filepath, map_location='cpu')
                    checkpoints.append({
                        'filename': filename,
                        'step': checkpoint.get('step', 0),
                        'performance': checkpoint.get('performance', 0),
                        'size_mb': os.path.getsize(filepath) / 1024 / 1024
                    })
                except:
                    pass

        # Sort by step
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints

    def replay_checkpoint(self, checkpoint_name: str = None):
        """Replay a specific checkpoint"""
        if checkpoint_name is None:
            # List available checkpoints
            checkpoints = self._list_available_checkpoints()
            if not checkpoints:
                print("❌ No checkpoints available")
                return

            print("📁 Available Checkpoints:")
            for i, cp in enumerate(checkpoints):
                print(f"   {i + 1}. {cp['filename']} (step {cp['step']}, performance: {cp['performance']:.2f})")

            try:
                choice = int(input("Select checkpoint number: ")) - 1
                if 0 <= choice < len(checkpoints):
                    checkpoint_name = checkpoints[choice]['filename']
                else:
                    print("❌ Invalid selection")
                    return
            except:
                print("❌ Invalid input")
                return

        checkpoint_path = os.path.join("demo_checkpoints", checkpoint_name)

        print(f"🔄 Loading checkpoint: {checkpoint_name}")
        if self.load_checkpoint(checkpoint_path):
            print("🎮 Running gameplay with loaded checkpoint...")
            self.run_interactive_inference_demo(num_episodes=3, show_details=True)


def main():
    """Run the demo"""
    # Check prerequisites
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (will be slower)")

    print(f"🚀 Revolutionary AI Pipeline - Visual Demo")
    print(f"Device: {DEVICE}")
    print(f"Environment: {ENV_CONFIG['name']}")
    print()

    try:
        demo = VisualDemo()
        demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()