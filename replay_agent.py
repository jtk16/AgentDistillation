# replay_agent.py
"""
Standalone script to replay saved agent checkpoints
Run this after training to watch your trained agents play!
"""

import torch
import numpy as np
import gymnasium as gym
import time
import json
import os
from datetime import datetime

# Import your demo components
try:
    from demo_runner import VisualDemo
    from config import *
except ImportError:
    print("❌ Error: Make sure demo_runner.py and demo_config.py are in the current directory")
    exit(1)


class AgentReplayer:
    """Standalone agent replay system"""

    def __init__(self):
        self.setup_environment()
        self.demo = None

    def setup_environment(self):
        """Setup environment for replay with fallback"""
        try:
            self.env = gym.make('CartPole-v1', render_mode='human')
            self.visual_rendering_available = True
            print(f"✅ Environment setup: CartPole-v1 with visual rendering")
        except Exception as e:
            print(f"⚠️  Visual rendering not available: {e}")
            print("    Falling back to non-visual mode")
            self.env = gym.make('CartPole-v1', render_mode=None)
            self.visual_rendering_available = False
            print(f"✅ Environment setup: CartPole-v1 (no visual rendering)")

    def _safe_reset(self, seed=None):
        """Safely reset environment with error handling"""
        try:
            return self.env.reset(seed=seed)
        except Exception as e:
            print(f"⚠️  Rendering error: {e}")
            print("    Switching to non-visual mode")
            self.env.close()
            self.env = gym.make('CartPole-v1', render_mode=None)
            self.visual_rendering_available = False
            return self.env.reset(seed=seed)

    def _safe_step(self, action):
        """Safely step environment with error handling"""
        try:
            return self.env.step(action)
        except Exception as e:
            print(f"⚠️  Rendering error during step: {e}")
            return self.env.step(action)

    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoint_dir = "demo_checkpoints"

        if not os.path.exists(checkpoint_dir):
            print("❌ No checkpoint directory found. Run the demo first!")
            return []

        checkpoints = []
        print("\n📁 Available Checkpoints:")
        print("-" * 60)

        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    # Load checkpoint metadata with explicit weights_only=False
                    checkpoint_data = torch.load(filepath, map_location='cpu', weights_only=False)

                    step = checkpoint_data.get('step', 0)
                    performance = checkpoint_data.get('performance', 0)
                    size_mb = os.path.getsize(filepath) / 1024 / 1024

                    # Get timestamp from file
                    timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))

                    checkpoints.append({
                        'filename': filename,
                        'filepath': filepath,
                        'step': step,
                        'performance': performance,
                        'size_mb': size_mb,
                        'timestamp': timestamp
                    })

                    print(f"{len(checkpoints):2d}. {filename}")
                    print(f"     Training Step: {step:,}")
                    print(f"     Performance: {performance:.2f}")
                    print(f"     Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"     Size: {size_mb:.1f} MB")
                    print()

                except Exception as e:
                    print(f"⚠️  Could not read {filename}: {e}")

        if not checkpoints:
            print("❌ No valid checkpoints found. Run the demo first!")

        return checkpoints

    def load_and_replay(self, checkpoint_path: str, num_episodes: int = 3, show_details: bool = True):
        """Load checkpoint and replay agent"""

        # Initialize demo system if needed
        if self.demo is None:
            print("🔄 Initializing demo system...")
            self.demo = VisualDemo()

        # Load the checkpoint
        print(f"📥 Loading checkpoint: {os.path.basename(checkpoint_path)}")

        if not self.demo.load_checkpoint(checkpoint_path):
            print("❌ Failed to load checkpoint")
            return

        print(f"✅ Checkpoint loaded successfully!")

        if not self.visual_rendering_available:
            print("ℹ️  Running in non-visual mode (text-only with performance data)")
        else:
            print("🎬 Visual rendering enabled - watch the CartPole game!")

        print(f"🎮 Starting replay with {num_episodes} episodes...")
        print()

        episode_rewards = []

        for episode in range(num_episodes):
            print(f"🎬 Episode {episode + 1}/{num_episodes}")
            print("-" * 40)

            try:
                obs, _ = self._safe_reset(seed=42 + episode)
            except Exception as e:
                print(f"❌ Failed to reset environment: {e}")
                print("   Skipping this episode...")
                continue

            total_reward = 0
            step = 0
            mentor_queries = 0
            decisions = []

            while step < 500:  # Max 500 steps
                # Convert observation to tensor
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # Get agent decision
                with torch.no_grad():
                    actions_batch, info_batch = self.demo.student.act(state_tensor, deterministic=False)

                action = actions_batch[0][0]  # Primary action
                info = info_batch[0]
                decisions.append(action)

                # Query mentor if needed
                mentor_consulted = False
                if info['should_query_mentor']:
                    mentor_advice = self.demo.mentor.get_advice(state_tensor, verbose=False)
                    mentor_queries += 1
                    mentor_consulted = True

                # Show detailed info every 25 steps
                if show_details and step % 25 == 0:
                    print(f"Step {step:3d}: Action={action} ({'←' if action == 0 else '→'}), " +
                          f"Conf={info['reasoning_confidence']:.2f}, " +
                          f"Unc={info['uncertainty']['total']:.2f}, " +
                          f"Reward={total_reward:5.1f}" +
                          (" 🎓" if mentor_consulted else ""))

                # Execute action
                try:
                    obs, reward, terminated, truncated, _ = self._safe_step(action)
                except Exception as e:
                    print(f"❌ Failed to step environment: {e}")
                    break

                total_reward += reward
                step += 1

                # Add delay only if visual rendering is working
                if self.visual_rendering_available:
                    time.sleep(0.03)

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)

            # Episode summary
            print(f"\n🏁 Episode {episode + 1} Results:")
            print(f"   💰 Total Reward: {total_reward:.1f}")
            print(f"   📏 Steps: {step}")
            if step > 0:
                print(f"   🎓 Mentor Queries: {mentor_queries} ({mentor_queries / step * 100:.1f}%)")
                print(f"   🎯 Action Balance: Left={decisions.count(0)}, Right={decisions.count(1)}")
            print(f"   ⭐ Success: {'Yes' if total_reward >= 200 else 'No'}")

            # Brief pause between episodes
            if episode < num_episodes - 1:
                print("\n⏳ Next episode in 3 seconds...")
                time.sleep(3)

        try:
            self.env.close()
        except:
            pass  # Ignore close errors

        # Final statistics
        if episode_rewards:
            print(f"\n🎉 Replay Complete!")
            print(f"   📊 Average Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
            print(f"   🏆 Best Episode: {max(episode_rewards):.1f}")
            print(f"   📈 Success Rate: {sum(1 for r in episode_rewards if r >= 200) / len(episode_rewards) * 100:.1f}%")

        return episode_rewards

    def interactive_mode(self):
        """Interactive checkpoint selection and replay"""
        print("🎮 Interactive Agent Replay Mode")
        print("=" * 40)

        while True:
            checkpoints = self.list_checkpoints()

            if not checkpoints:
                break

            print("\nOptions:")
            print("  1-N: Select checkpoint number to replay")
            print("  R: Refresh checkpoint list")
            print("  Q: Quit")

            try:
                choice = input("\nEnter your choice: ").strip().upper()

                if choice == 'Q':
                    print("👋 Goodbye!")
                    break
                elif choice == 'R':
                    continue
                else:
                    try:
                        checkpoint_idx = int(choice) - 1
                        if 0 <= checkpoint_idx < len(checkpoints):
                            checkpoint = checkpoints[checkpoint_idx]

                            # Ask for episode count
                            try:
                                episodes = int(input("How many episodes to replay? (default=3): ") or "3")
                                episodes = max(1, min(10, episodes))  # Limit 1-10
                            except:
                                episodes = 3

                            # Ask for detail level
                            details = input("Show detailed reasoning? (y/n, default=y): ").lower().strip() != 'n'

                            print()
                            self.load_and_replay(
                                checkpoint['filepath'],
                                num_episodes=episodes,
                                show_details=details
                            )

                        else:
                            print("❌ Invalid checkpoint number")
                    except ValueError:
                        print("❌ Invalid input")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


def main():
    """Main replay function"""
    print("🎬 Revolutionary AI Pipeline - Agent Replay")
    print("=" * 50)
    print("Watch your trained agents play CartPole with detailed inference!")
    print()

    try:
        replayer = AgentReplayer()
        replayer.interactive_mode()

    except KeyboardInterrupt:
        print("\n👋 Replay interrupted by user")
    except Exception as e:
        print(f"❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()