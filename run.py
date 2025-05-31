# run.py
"""
Convenient launcher for Revolutionary AI Pipeline with preset configurations
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_test():
    """Run component tests"""
    print("🧪 Running pipeline tests...")
    result = subprocess.run([sys.executable, "test_pipeline.py"], capture_output=False)
    return result.returncode == 0


def run_training(args):
    """Run training with specified configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build command
    cmd = [sys.executable, "main.py"]

    # Add log directory with timestamp
    log_dir = f"logs/run_{timestamp}"
    cmd.extend(["--log_dir", log_dir])

    # Add other arguments
    if args.env_name:
        cmd.extend(["--env_name", args.env_name])

    if args.load_checkpoint:
        cmd.extend(["--load_checkpoint", args.load_checkpoint])

    if args.eval_only:
        cmd.append("--eval_only")

    print(f"🚀 Starting training...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log directory: {log_dir}")
    print(f"   Environment: {args.env_name or 'CartPole-v1'}")

    # Run training
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Main launcher"""
    parser = argparse.ArgumentParser(description='Revolutionary AI Pipeline Launcher')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['test', 'train', 'eval'],
                        default='train', help='Run mode')

    # Training arguments
    parser.add_argument('--env_name', type=str, help='Environment name')
    parser.add_argument('--load_checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')

    # Quick presets
    parser.add_argument('--preset', type=str, choices=['cartpole', 'lunar', 'custom'],
                        help='Use preset configuration')

    args = parser.parse_args()

    # Handle presets
    if args.preset == 'cartpole':
        args.env_name = 'CartPole-v1'
        print("🎯 Using CartPole preset")
    elif args.preset == 'lunar':
        args.env_name = 'LunarLander-v2'
        print("🌙 Using LunarLander preset")

    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in virtual environment")
        print("   Recommended: source .venv/bin/activate")
        print()

    # Run based on mode
    if args.mode == 'test':
        success = run_test()
        if success:
            print("\n✅ All tests passed! Ready for training.")
        else:
            print("\n❌ Tests failed. Please check the output above.")
            sys.exit(1)

    elif args.mode == 'train':
        success = run_training(args)
        if not success:
            print("\n❌ Training failed. Check logs for details.")
            sys.exit(1)

    elif args.mode == 'eval':
        args.eval_only = True
        success = run_training(args)
        if not success:
            print("\n❌ Evaluation failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    print("🚀 Revolutionary AI Pipeline Launcher")
    print("=====================================\n")

    main()