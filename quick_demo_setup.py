# quick_demo_setup.py
"""
Quick setup script to prepare and run the Revolutionary AI Pipeline demo
"""

import subprocess
import sys
import os
import torch


def install_dependencies():
    """Install required packages for the demo"""
    print("📦 Installing demo dependencies...")

    required_packages = [
        "torch",
        "gymnasium[classic_control]",
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "networkx",
        "plotly"
    ]

    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False

    return True


def check_system_requirements():
    """Check if system meets requirements for demo"""
    print("🔍 Checking system requirements...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (demo will be slower)")

    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        print(f"✅ Available RAM: {memory_gb:.1f} GB")
        if memory_gb < 4:
            print("⚠️  Low memory detected, demo may be slower")
    except ImportError:
        print("ℹ️  Could not check memory usage")

    return True


def create_demo_structure():
    """Create necessary directories for demo"""
    print("📁 Creating demo directory structure...")

    directories = [
        "demo_logs",
        "demo_results",
        "demo_checkpoints"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")


def run_system_test():
    """Run a quick system test"""
    print("🧪 Running system test...")

    try:
        import torch
        import numpy as np
        import gymnasium as gym

        # Test basic torch operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print("✅ PyTorch operations working")

        # Test gymnasium
        env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("✅ Gymnasium environment working")

        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def run_demo():
    """Launch the demo"""
    print("🚀 Launching Revolutionary AI Pipeline Demo...")

    try:
        # Import and run demo
        from demo_runner import main as demo_main
        demo_main()
    except ImportError:
        print("❌ Demo files not found. Make sure demo_runner.py is in the current directory.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main setup and demo launcher"""
    print("🎬 Revolutionary AI Pipeline - Quick Demo Setup")
    print("=" * 50)
    print("This script will set up and run a visual demonstration of the")
    print("Revolutionary AI Pipeline featuring mentor-student distillation,")
    print("parallel reasoning, and multi-action capabilities.")
    print()

    # Step 1: Check system
    if not check_system_requirements():
        print("❌ System requirements not met")
        return

    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return

    # Step 3: Create structure
    create_demo_structure()

    # Step 4: System test
    if not run_system_test():
        print("❌ System test failed")
        return

    print("✅ Setup completed successfully!")
    print()

    # Step 5: Ask user if they want to run demo
    response = input("🎮 Run the demo now? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        print()
        run_demo()
    else:
        print("Demo setup complete! Run 'python demo_runner.py' when ready.")


if __name__ == "__main__":
    main()