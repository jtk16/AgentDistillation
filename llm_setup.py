# llm_setup.py
"""
Setup and test script for LLM-Enhanced Revolutionary AI Pipeline
Handles dependencies, memory testing, and initial LLM verification
"""

import subprocess
import sys
import os
import torch
import json
from typing import Dict, List


def check_gpu_capability():
    """Check GPU specifications and VRAM availability"""
    print("🔍 Checking GPU capability...")

    if not torch.cuda.is_available():
        print("❌ CUDA not available. This pipeline requires a CUDA-capable GPU.")
        return False

    device_count = torch.cuda.device_count()
    print(f"✅ Found {device_count} CUDA device(s)")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9

        print(f"   GPU {i}: {props.name}")
        print(f"   VRAM: {vram_gb:.1f}GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")

        if "2060" in props.name or "RTX 2060" in props.name:
            print("   🎯 RTX 2060 series detected - optimal for this pipeline!")
        elif vram_gb >= 6:
            print("   ✅ Sufficient VRAM for LLM operation")
        else:
            print("   ⚠️  Limited VRAM - may need smaller LLM model")

    return True


def install_llm_dependencies():
    """Install additional dependencies for LLM functionality"""
    print("📦 Installing LLM dependencies...")

    llm_packages = [
        "transformers>=4.36.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",  # For 4-bit quantization
        "sentencepiece>=0.1.99",  # For tokenization
        "protobuf>=3.20.0", # Often a dependency for transformers or sentencepiece
        "huggingface_hub>=0.19.0"
    ]

    llm_packages = []

    for package in llm_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred while installing {package}: {e}")
            return False


    return True


def test_llm_loading():
    """Test LLM loading with different model sizes"""
    print("🧪 Testing LLM loading capabilities...")

    # Import after installing dependencies
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print("✅ Transformers library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False

    # Test models in order of preference
    test_models = [
        {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "size": "3.8B",
            "expected_vram": 3.5,
            "attn_implementation": "eager"  # Phi-3 often more stable with eager
        },
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B",
            "expected_vram": 1.0,
            "attn_implementation": "sdpa" # Default for others
        },
        {
            "name": "Qwen/Qwen2-0.5B-Instruct",
            "size": "0.5B",
            "expected_vram": 0.6,
            "attn_implementation": "sdpa"
        }
    ]

    available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"Available VRAM: {available_vram:.1f}GB")

    successful_models = []

    for model_info in test_models:
        model_name = model_info["name"]
        expected_vram = model_info["expected_vram"]
        attn_impl = model_info.get("attn_implementation", "sdpa")


        if not torch.cuda.is_available() and expected_vram > 0:
             print(f"⏭️  Skipping {model_name} - CUDA not available for GPU models.")
             continue
        if expected_vram > available_vram * 0.9:  # Leave 10% buffer, was 0.8
            print(f"⏭️  Skipping {model_name} - requires {expected_vram:.1f}GB (insufficient VRAM, needs > {expected_vram / 0.9:.1f}GB total with buffer)")
            continue

        print(f"�� Testing {model_name} ({model_info['size']}) using attn_implementation='{attn_impl}'...")

        try:
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # Test tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"   ✅ Tokenizer loaded")

            # Test model loading with memory monitoring
            initial_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto", # Automatically handles CPU/GPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl
            )

            model.eval()
            final_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            memory_used = final_memory - initial_memory

            print(f"   ✅ Model loaded successfully")
            print(f"   📊 Memory used: {memory_used:.1f}GB")

            # Test generation
            test_prompt = "The optimal action in CartPole when the pole is leaning right is:"
            # Ensure model is on a device, and inputs are on that device
            target_device = model.device if hasattr(model, 'device') else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            inputs = tokenizer(test_prompt, return_tensors="pt").to(target_device)


            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"   ✅ Generation test: '{response.strip()}'")

            successful_models.append({
                "name": model_name,
                "memory_used": memory_used,
                "size": model_info["size"]
            })

            # Cleanup
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()


            print(f"   ✅ {model_name} - WORKING")
            break  # Use first working model

        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue

    if successful_models:
        best_model = successful_models[0]
        print(f"\n🎯 Recommended model: {best_model['name']}")
        print(f"   Size: {best_model['size']}")
        print(f"   Memory usage: {best_model['memory_used']:.1f}GB")
        return best_model["name"]
    else:
        print("\n❌ No LLM models could be loaded successfully")
        return None


def create_optimized_config(recommended_model: str):
    """Create optimized configuration file"""
    print("⚙️ Creating optimized configuration...")

    available_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0


    config = {
        "# LLM-Enhanced Pipeline Configuration": "Auto-generated",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "recommended_llm_model": recommended_model,
        "available_vram_gb": round(available_vram, 1),

        "training_config": {
            "batch_size": 8 if available_vram >= 8 else 4,
            "rollout_steps": 32 if available_vram >= 8 else 16,
            "total_timesteps": 20000,
            "memory_cleanup_frequency": 50
        },

        "llm_config": {
            "model_name": recommended_model,
            "use_4bit_quantization": True if torch.cuda.is_available() else False, # Only quantize on GPU
            "max_context_length": 2048 if available_vram >= 8 else 1024,
            "enable_response_cache": True,
            "query_frequency": "adaptive"
        },

        "memory_optimization": {
            "enable_mixed_precision": True if torch.cuda.is_available() else False,
            "gradient_checkpointing": True if torch.cuda.is_available() else False,
            "memory_efficient_attention": True if torch.cuda.is_available() else False,
        }
    }

    # Save configuration
    with open("llm_pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration saved to llm_pipeline_config.json")


def run_quick_demo():
    """Run a quick demonstration of the LLM pipeline"""
    print("🎮 Running quick LLM pipeline demo...")

    try:
        # Import the LLM components
        from llm_mentor import create_llm_mentor # This will use the corrected LLMMentor
        from llm_config import LLM_MENTOR_CONFIG, DEVICE # Ensure DEVICE is imported and used

        print("🤖 Creating LLM mentor...")
        # Use the recommended_model from the JSON config if it exists,
        config_file_path = "llm_pipeline_config.json"
        llm_model_to_use = LLM_MENTOR_CONFIG['model_name'] # Default
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    pipeline_config = json.load(f)
                    llm_model_to_use = pipeline_config.get("recommended_llm_model", llm_model_to_use)
                    print(f"   Using model from llm_pipeline_config.json: {llm_model_to_use}")
            except Exception as e:
                print(f"   Could not read llm_pipeline_config.json, using default model from llm_config.py. Error: {e}")
        else:
             print(f"   llm_pipeline_config.json not found, using default model from llm_config.py: {llm_model_to_use}")


        mentor = create_llm_mentor(state_dim=4, num_actions=2, model_name=llm_model_to_use)
        mentor.to(DEVICE) # Ensure mentor and its submodules are on the correct device

        print("🧪 Testing mentor with sample CartPole state...")
        # Test state: cart at center, moving right, pole leaning right, angular velocity positive
        test_state = torch.tensor([0.1, 0.5, 0.15, 1.0], device=DEVICE).unsqueeze(0)


        advice = mentor.get_advice(test_state, verbose=True)

        print(f"\n📋 LLM Mentor Advice:")
        print(f"   Actions: {advice.actions}")
        print(f"   Confidence: {advice.confidence:.3f}")
        print(f"   Strategy: {advice.strategy}")
        print(f"   Reasoning: {advice.reasoning[0] if advice.reasoning else 'N/A'}")

        # Test neural forward pass
        print("\n🧠 Testing neural integration...")
        with torch.no_grad(): # Ensure no gradients are computed during inference
            outputs = mentor(test_state)
        print(f"   Policy logits: {outputs['policy_logits'].cpu().numpy()}")
        print(f"   Value: {outputs['value'].item():.3f}")

        # Performance stats
        stats = mentor.get_performance_stats()
        print(f"\n📊 Performance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n✅ LLM pipeline demo completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup and test workflow"""
    print("🚀 LLM-Enhanced Revolutionary AI Pipeline Setup")
    print("=" * 60)
    print("This script will:")
    print("1. Check GPU capability and VRAM")
    print("2. Install LLM dependencies")
    print("3. Test LLM model loading")
    print("4. Create optimized configuration")
    print("5. Run quick demonstration")
    print()

    # Step 1: Check GPU
    if not check_gpu_capability():
        print("\n❌ GPU check failed. Please ensure you have a CUDA-capable GPU for full functionality.")
        # Allow to proceed for CPU testing if user desires, but LLM loading will be skipped for GPU models.

    # Step 2: Install dependencies
    if not install_llm_dependencies():
        print("\n❌ Dependency installation failed.")
        return

    # Step 3: Test LLM loading
    recommended_model = None
    if torch.cuda.is_available(): # Only test loading if GPU is there, as models are GPU-heavy
        recommended_model = test_llm_loading()
        if not recommended_model:
            print("\n⚠️ No suitable LLM model could be loaded on GPU. Demo will likely fail or run on CPU if LLM is required.")
            # Provide a CPU-compatible default if all GPU attempts fail
            recommended_model = " TinyLlama/TinyLlama-1.1B-Chat-v1.0" # A smaller model that might work on CPU (though slowly)
            print(f"   Falling back to default CPU-testable model: {recommended_model} for config generation.")
    else:
        print("\nℹ️ CUDA not available. Skipping LLM loading test. Configuration will be for CPU.")
        # Provide a CPU-compatible default
        recommended_model = " TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Example, or make it None if CPU LLM not supported by pipeline

    if not recommended_model: # If still None after GPU check and CPU fallback
        print("\n❌ No recommended LLM model could be determined. Cannot proceed with demo config.")
        return


    # Step 4: Create configuration
    create_optimized_config(recommended_model)

    # Step 5: Run demo
    demo_success = run_quick_demo()

    # Final summary
    print("\n" + "=" * 60)
    if demo_success:
        print("🎉 Setup completed successfully!")
        print(f"🎯 Recommended LLM for config: {recommended_model}")
        print("🚀 Ready to run: python llm_main.py")
        print("\nNext steps:")
        print("1. Review 'llm_pipeline_config.json' - it has been created/updated.")
        print("2. Run: python llm_main.py --memory_test  # Test memory usage (if GPU available)")
        print("3. Run: python llm_main.py                # Start training")
        if torch.cuda.is_available():
            print("4. Monitor GPU memory with: nvidia-smi")
    else:
        print("❌ Setup completed with issues. Check the logs above.")
        print("💡 Troubleshooting:")
        print("- Ensure sufficient VRAM (6GB+ recommended for smaller LLMs on GPU)")
        print("- If on CPU, expect very slow LLM performance if used.")
        print("- Check internet connection for model downloads.")
        print("- Ensure all dependencies from `install_llm_dependencies` installed correctly.")

    print("=" * 60)


if __name__ == "__main__":
    main()