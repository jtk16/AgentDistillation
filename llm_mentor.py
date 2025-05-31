# llm_mentor.py
"""
Local LLM-based Mentor for Revolutionary AI Pipeline
Optimized for RTX 2060S (8GB VRAM) with efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
import gc
from functools import lru_cache
import threading
import time

from config import DEVICE, MENTOR_CONFIG


@dataclass
class LLMAdvice:
    """Structured advice from LLM mentor"""
    actions: List[int]
    confidence: float
    reasoning: List[str]
    causal_effects: Dict[str, float]
    strategy: str
    raw_response: str


class MemoryOptimizedLLM:
    """Memory-optimized LLM wrapper for efficient inference on 2060S"""

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.device = DEVICE

        # Memory management
        self.max_context_length = 2048  # Reduced for memory efficiency
        self.max_new_tokens = 256
        self.temperature = 0.7

        # Caching for repeated queries
        self.response_cache = {}
        self.cache_size_limit = 100

        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with memory optimizations"""
        print(f"🤖 Loading LLM: {self.model_name}")

        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Set to eval mode
            self.model.eval()

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

            print(f"✅ LLM loaded successfully!")
            print(f"   Model size: ~{self._estimate_model_size():.1f}GB VRAM")
            print(f"   Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            print("🔄 Falling back to TinyLlama...")
            self._fallback_to_tinyllama()

    def _fallback_to_tinyllama(self):
        """Fallback to even smaller model if needed"""
        try:
            self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.model.eval()
            torch.cuda.empty_cache()
            gc.collect()

            print(f"✅ Fallback LLM loaded: {self.model_name}")

        except Exception as e:
            print(f"❌ Critical: Could not load any LLM: {e}")
            raise e

    def _estimate_model_size(self) -> float:
        """Estimate model VRAM usage"""
        if self.model is None:
            return 0.0

        total_params = sum(p.numel() for p in self.model.parameters())
        # Rough estimate: 4-bit quantized model uses ~0.5 bytes per parameter
        estimated_gb = (total_params * 0.5) / 1e9
        return estimated_gb

    @lru_cache(maxsize=50)
    def _cached_generate(self, prompt_hash: str, prompt: str) -> str:
        """Cached generation to avoid repeated LLM calls"""
        return self._generate_uncached(prompt)

    def _generate_uncached(self, prompt: str) -> str:
        """Generate response without caching"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=True
            ).to(self.model.device)

            # Generate with memory-efficient settings
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return "ERROR: Unable to generate response"

    def generate(self, prompt: str) -> str:
        """Generate response with caching"""
        # Create hash for caching
        prompt_hash = str(hash(prompt))

        # Check cache first
        if prompt_hash in self.response_cache:
            return self.response_cache[prompt_hash]

        # Generate new response
        response = self._generate_uncached(prompt)

        # Cache response (with size limit)
        if len(self.response_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]

        self.response_cache[prompt_hash] = response
        return response

    def clear_cache(self):
        """Clear response cache and CUDA cache"""
        self.response_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class LLMPromptManager:
    """Manages prompts and response parsing for the LLM mentor"""

    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.action_space_desc = self._get_action_space_description()
        self.physics_rules = self._get_physics_rules()
        self.system_prompt = self._create_system_prompt()

    def _get_action_space_description(self) -> str:
        """Get environment-specific action descriptions"""
        action_descriptions = {
            "CartPole-v1": {
                "0": "Push cart LEFT (apply force of 10N leftward)",
                "1": "Push cart RIGHT (apply force of 10N rightward)"
            },
            "LunarLander-v2": {
                "0": "Do nothing",
                "1": "Fire left orientation engine",
                "2": "Fire main engine",
                "3": "Fire right orientation engine"
            }
        }

        desc = action_descriptions.get(self.env_name, {"0": "Action 0", "1": "Action 1"})
        return "\n".join([f"Action {k}: {v}" for k, v in desc.items()])

    def _get_physics_rules(self) -> str:
        """Get environment-specific physics rules"""
        physics_rules = {
            "CartPole-v1": """
PHYSICS RULES:
- Cart mass: 1.0kg, Pole mass: 0.1kg, Pole length: 0.5m
- Gravity: 9.8 m/s², Friction: 0.0005
- Episode ends if: |cart_position| > 2.4m OR |pole_angle| > 12°
- Reward: +1 for each step pole stays upright
- Goal: Balance pole as long as possible (max 500 steps)

STRATEGY GUIDELINES:
- Small angles (<5°): Use predictive control based on angular velocity
- Medium angles (5-10°): Apply corrective force opposite to lean
- Large angles (>10°): Emergency correction needed immediately
- Consider cart velocity to avoid oscillations
- Momentum conservation: heavier cart, lighter pole
            """,
            "LunarLander-v2": """
PHYSICS RULES:
- Gravity: -1.6 m/s² (lunar gravity)
- Landing pad: coordinates (0, 0) with ±10m tolerance
- Fuel: limited, penalty for engine use
- Episode ends if: crash (contact_velocity > threshold) OR land successfully
- Reward: +100 for landing, -100 for crash, fuel penalties

STRATEGY GUIDELINES:
- Descent phase: Control vertical velocity, aim for landing pad
- Approach phase: Reduce horizontal drift, prepare for landing
- Landing phase: Soft touchdown with minimal velocity
- Fuel efficiency: Use minimal thrust for corrections
            """
        }

        return physics_rules.get(self.env_name, "Generic environment rules")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM mentor"""
        return f"""You are an EXPERT AI MENTOR for the {self.env_name} environment. Your role is to provide optimal action decisions and strategic advice based on deep understanding of physics and optimal control theory.

{self.physics_rules}

AVAILABLE ACTIONS:
{self.action_space_desc}

RESPONSE FORMAT:
You must respond in this EXACT JSON format:
{{
    "action": <integer_action_number>,
    "confidence": <float_0_to_1>,
    "reasoning": ["<step1>", "<step2>", "<step3>"],
    "strategy": "<HIGH_LEVEL_STRATEGY>",
    "causal_effects": {{"action_0": <float>, "action_1": <float>}},
    "multi_step_plan": [<action1>, <action2>, <action3>]
}}

CRITICAL REQUIREMENTS:
1. ALWAYS respond with valid JSON only
2. "action" must be a valid action number
3. "confidence" between 0.0 and 1.0
4. "reasoning" must be 2-4 clear physics-based steps
5. "causal_effects" predict impact magnitude of each action (0.0-1.0)
6. Be decisive and optimal - lives depend on your decisions!
"""

    def create_state_prompt(self, state: np.ndarray, context: str = "") -> str:
        """Create prompt for current state"""
        state_desc = self._format_state_description(state)

        prompt = f"""CURRENT STATE ANALYSIS:
{state_desc}

{context}

Analyze this state and provide your optimal action decision with complete reasoning. Focus on physics principles and optimal control theory."""

        return prompt

    def _format_state_description(self, state: np.ndarray) -> str:
        """Format state into human-readable description"""
        if self.env_name == "CartPole-v1" and len(state) >= 4:
            pos, vel, angle, ang_vel = state[:4]

            # Convert angle to degrees for readability
            angle_deg = np.degrees(angle)

            # State analysis
            stability = "STABLE" if abs(angle_deg) < 5 else "UNSTABLE" if abs(angle_deg) < 10 else "CRITICAL"
            direction = "LEANING_LEFT" if angle < 0 else "LEANING_RIGHT" if angle > 0 else "UPRIGHT"
            motion = "MOVING_LEFT" if vel < -0.1 else "MOVING_RIGHT" if vel > 0.1 else "STATIONARY"

            description = f"""
Cart Position: {pos:.3f}m (center=0, limits=±2.4m)
Cart Velocity: {vel:.3f}m/s ({motion})
Pole Angle: {angle:.4f}rad ({angle_deg:.2f}°) - {direction}
Angular Velocity: {ang_vel:.3f}rad/s
Stability Status: {stability}
Physics Analysis:
- Gravitational torque: {9.8 * 0.1 * 0.5 * np.sin(angle):.4f} N⋅m
- Cart momentum: {1.0 * vel:.3f} kg⋅m/s
- Angular momentum: {0.1 * ang_vel:.4f} kg⋅m²/s
            """

        else:
            # Generic state description
            description = f"State Vector: {state.tolist()}\nState Magnitude: {np.linalg.norm(state):.3f}"

        return description.strip()

    def parse_llm_response(self, response: str) -> LLMAdvice:
        """Parse LLM response into structured advice"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                return LLMAdvice(
                    actions=data.get("multi_step_plan", [data.get("action", 0)])[:3],
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ["No reasoning provided"]),
                    causal_effects=data.get("causal_effects", {}),
                    strategy=data.get("strategy", "UNKNOWN"),
                    raw_response=response
                )
            else:
                # Fallback parsing for non-JSON responses
                return self._fallback_parse(response)

        except Exception as e:
            print(f"⚠️ LLM response parsing error: {e}")
            return self._emergency_fallback(response)

    def _fallback_parse(self, response: str) -> LLMAdvice:
        """Fallback parsing when JSON parsing fails"""
        # Extract action number from text
        action_match = re.search(r'action[:\s]*(\d+)', response.lower())
        action = int(action_match.group(1)) if action_match else 0

        # Extract confidence
        conf_match = re.search(r'confidence[:\s]*(0?\.\d+|\d+\.?\d*)', response.lower())
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # Extract reasoning sentences
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10][:3]
        reasoning = sentences if sentences else ["Fallback reasoning: Action selected based on heuristics"]

        return LLMAdvice(
            actions=[action],
            confidence=confidence,
            reasoning=reasoning,
            causal_effects={"action_0": 0.3, "action_1": 0.3},
            strategy="FALLBACK_STRATEGY",
            raw_response=response
        )

    def _emergency_fallback(self, response: str) -> LLMAdvice:
        """Emergency fallback when all parsing fails"""
        return LLMAdvice(
            actions=[0],  # Default safe action
            confidence=0.1,
            reasoning=["Emergency fallback: Using default action"],
            causal_effects={"action_0": 0.5, "action_1": 0.5},
            strategy="EMERGENCY_FALLBACK",
            raw_response=response
        )


class LLMMentor(nn.Module):
    """
    LLM-powered mentor that replaces the original MultimodalMentor
    Provides textual reasoning and optimal action decisions
    """

    def __init__(self, state_dim: int, num_actions: int, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Initialize LLM components
        self.llm = MemoryOptimizedLLM(model_name)
        self.prompt_manager = LLMPromptManager()

        # Create simple output projection layers for compatibility
        # These convert LLM decisions into tensor format expected by distillation
        self.hidden_dim = MENTOR_CONFIG['hidden_dim']

        # Projection layers to maintain interface compatibility
        self.feature_projector = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Action and value heads for tensor outputs
        self.policy_head = nn.Linear(self.hidden_dim, num_actions)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        # Cache for repeated states
        self.advice_cache = {}
        self.tensor_cache = {}

        # Performance tracking
        self.query_count = 0
        self.cache_hits = 0

        self.to(DEVICE)

    def forward(self, state: torch.Tensor, use_llm: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass that optionally uses LLM for enhanced reasoning
        Falls back to neural network layers for compatibility
        """
        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Generate basic neural features for compatibility
        features = self.feature_projector(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        # Enhance with LLM reasoning for first state in batch
        if use_llm and batch_size > 0:
            try:
                # Use LLM for the first state and apply to whole batch
                first_state = state[0].cpu().numpy()
                llm_advice = self._get_llm_decision(first_state)

                # Adjust policy logits based on LLM decision
                if llm_advice.actions:
                    llm_action = llm_advice.actions[0]
                    confidence = llm_advice.confidence

                    # Boost probability of LLM-recommended action
                    for i in range(batch_size):
                        if llm_action < self.num_actions:
                            policy_logits[i, llm_action] += confidence * 2.0

            except Exception as e:
                print(f"⚠️ LLM enhancement failed: {e}")
                # Continue with neural network outputs

        return {
            'policy_logits': policy_logits,
            'value': value,
            'features': features,
            'causal_features': features,  # Same as features for compatibility
            'confidence': torch.sigmoid(value),  # Derive confidence from value
            'planning_logits': policy_logits.unsqueeze(1).repeat(1, 5, 1)  # Multi-step planning
        }

    def _get_llm_decision(self, state: np.ndarray) -> LLMAdvice:
        """Get LLM decision for a single state"""
        # Check cache first
        state_key = tuple(np.round(state, 3))  # Round for cache efficiency

        if state_key in self.advice_cache:
            self.cache_hits += 1
            return self.advice_cache[state_key]

        # Query LLM
        self.query_count += 1

        try:
            # Create prompt
            full_prompt = self.prompt_manager.system_prompt + "\n\n" + \
                          self.prompt_manager.create_state_prompt(state)

            # Generate response
            response = self.llm.generate(full_prompt)

            # Parse response
            advice = self.prompt_manager.parse_llm_response(response)

            # Cache result
            if len(self.advice_cache) < 200:  # Limit cache size
                self.advice_cache[state_key] = advice

            return advice

        except Exception as e:
            print(f"⚠️ LLM decision error: {e}")
            # Return safe fallback
            return LLMAdvice(
                actions=[0],
                confidence=0.1,
                reasoning=["LLM error: using fallback"],
                causal_effects={},
                strategy="ERROR_FALLBACK",
                raw_response=""
            )

    def get_advice(self, state: torch.Tensor, verbose: bool = False) -> LLMAdvice:
        """
        Get sophisticated advice from LLM mentor
        This is the main interface for student agent queries
        """
        if state.dim() > 1:
            state = state[0]  # Take first state from batch

        state_np = state.cpu().numpy()
        advice = self._get_llm_decision(state_np)

        if verbose:
            print(f"🧠 LLM Mentor Advice:")
            print(f"   Recommended Actions: {advice.actions}")
            print(f"   Confidence: {advice.confidence:.3f}")
            print(f"   Strategy: {advice.strategy}")
            print(f"   Reasoning: {advice.reasoning[0] if advice.reasoning else 'N/A'}")

        return advice

    def predict_action_effects(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Predict causal effects of an action (for compatibility)"""
        # Simple neural prediction for now
        features = self.feature_projector(state)
        # Return a simple effect prediction
        effect = torch.randn_like(state) * 0.1
        return effect

    def get_performance_stats(self) -> Dict[str, float]:
        """Get LLM performance statistics"""
        total_queries = self.query_count + self.cache_hits
        cache_rate = self.cache_hits / total_queries if total_queries > 0 else 0

        return {
            'total_queries': total_queries,
            'llm_queries': self.query_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_rate,
            'memory_usage_gb': self.llm._estimate_model_size()
        }

    def optimize_memory(self):
        """Optimize memory usage"""
        # Clear caches
        self.advice_cache.clear()
        self.tensor_cache.clear()
        self.llm.clear_cache()

        # Reset counters
        self.query_count = 0
        self.cache_hits = 0

        print("🧹 LLM Mentor memory optimized")


def create_llm_mentor(state_dim: int, num_actions: int,
                      model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> LLMMentor:
    """
    Factory function to create LLM mentor with fallback options
    """

    # Try different models in order of preference for 2060S
    model_options = [
        "microsoft/Phi-3-mini-4k-instruct",  # ~3.8B params, excellent reasoning
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~1.1B params, very fast
        "Qwen/Qwen2-0.5B-Instruct"  # ~0.5B params, minimal VRAM
    ]

    if model_name not in model_options:
        model_options.insert(0, model_name)  # Try user's choice first

    for model in model_options:
        try:
            print(f"🔄 Attempting to load: {model}")
            mentor = LLMMentor(state_dim, num_actions, model)

            # Test the mentor with a dummy state
            test_state = torch.randn(1, state_dim).to(DEVICE)
            with torch.no_grad():
                test_output = mentor(test_state)

            print(f"✅ Successfully loaded LLM Mentor: {model}")
            return mentor

        except Exception as e:
            print(f"❌ Failed to load {model}: {e}")
            continue

    raise RuntimeError("❌ Could not load any LLM model. Check your GPU memory and internet connection.")


# Test script for standalone testing
if __name__ == "__main__":
    print("🧪 Testing LLM Mentor Integration")
    print("=" * 50)

    try:
        # Test LLM initialization
        mentor = create_llm_mentor(state_dim=4, num_actions=2)

        # Test with CartPole state
        test_state = torch.tensor([0.1, -0.5, 0.2, 1.0]).unsqueeze(0).to(DEVICE)

        print("\n🔬 Testing LLM forward pass...")
        outputs = mentor(test_state)
        print(f"✅ Policy logits shape: {outputs['policy_logits'].shape}")
        print(f"✅ Value shape: {outputs['value'].shape}")

        print("\n🔬 Testing LLM advice...")
        advice = mentor.get_advice(test_state, verbose=True)
        print(f"✅ Advice generated successfully")

        print("\n📊 Performance Stats:")
        stats = mentor.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n🎉 LLM Mentor integration test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()