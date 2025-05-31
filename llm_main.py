# llm_main.py
"""
LLM-Enhanced Revolutionary AI Pipeline
Seamlessly integrates local LLM mentor with existing architecture
Optimized for RTX 2060S (8GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import psutil
from typing import Dict, List, Any, Optional
import argparse
import os
import gc

import torch.nn.functional as F

# Import LLM components
from llm_config import *
from llm_mentor import create_llm_mentor, LLMMentor

# Import existing components (with compatibility)
from environment import create_environment, AdvancedRewardShaper
from student import StudentAgent
from distillation import DistillationTrainer, FeatureProjector
from memory import PrioritizedReplayBuffer, TrajectoryBuffer, ExperienceCollector, compute_gae
from utils import Logger, CurriculumScheduler, ActionProcessor
from torch.distributions import Categorical

# Import mathematical framework if available
try:
    from mathematical_framework import PathwayImportanceOptimizer, InformationTheoreticAnalyzer
    from activation_distillation import (
        HumanDemonstrationCollector, ActivationTracker,
        CriticalPathwayAnalyzer as MathCriticalPathwayAnalyzer,
        ActivationSignatureExtractor, FocusedDistillationLoss,
        create_activation_based_distillation_pipeline
    )

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    print("⚠️  Advanced activation features not available, using core LLM pipeline")
    ADVANCED_FEATURES_AVAILABLE = False


class LLMMemoryManager:
    """Manages GPU memory efficiently for LLM + training"""

    def __init__(self):
        self.cleanup_counter = 0
        self.memory_warnings = 0

    def check_memory(self, force_cleanup: bool = False) -> Dict[str, float]:
        """Check current memory usage"""
        if not torch.cuda.is_available():
            return {'total': 0, 'used': 0, 'free': 0, 'percent': 0}

        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total - reserved

        stats = {
            'total_gb': total / 1e9,
            'reserved_gb': reserved / 1e9,
            'allocated_gb': allocated / 1e9,
            'free_gb': free / 1e9,
            'percent_used': (reserved / total) * 100
        }

        # Trigger cleanup if needed
        if stats['percent_used'] > HARDWARE_CONFIG['memory_warning_threshold'] * 100:
            self.memory_warnings += 1
            if self.memory_warnings % 10 == 1:  # Log every 10 warnings
                print(f"⚠️  High memory usage: {stats['percent_used']:.1f}%")

        if (stats['percent_used'] > HARDWARE_CONFIG['emergency_cleanup_threshold'] * 100) or force_cleanup:
            self.emergency_cleanup()

        return stats

    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        print("🧹 Emergency GPU memory cleanup...")
        torch.cuda.empty_cache()
        gc.collect()

        # Additional cleanup
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    def periodic_cleanup(self, step: int):
        """Periodic memory cleanup"""
        self.cleanup_counter += 1

        if step % TRAINING_CONFIG['memory_cleanup_frequency'] == 0:
            self.check_memory(force_cleanup=True)
            print(f"🧹 Periodic memory cleanup at step {step}")


class LLMRevolutionaryPipeline:
    """
    Enhanced Revolutionary Pipeline with LLM mentor integration
    Maintains all original capabilities while adding LLM reasoning
    """

    def __init__(self, args):
        self.args = args
        self.logger = Logger(args.log_dir)
        self.memory_manager = LLMMemoryManager()

        # Check initial memory state
        initial_memory = self.memory_manager.check_memory()
        self.logger.log(
            f"Initial VRAM: {initial_memory['free_gb']:.1f}GB free of {initial_memory['total_gb']:.1f}GB total")

        self.logger.log("Initializing LLM-Enhanced Revolutionary AI Pipeline...")

        # Initialize environment
        self.env = create_environment()
        self.reward_shaper = AdvancedRewardShaper(ENV_CONFIG['name'], num_envs=self.env.num_envs)

        # Initialize LLM mentor (this is the key change!)
        self.logger.log("🤖 Initializing LLM Mentor...")
        try:
            self.mentor = create_llm_mentor(
                self.env.state_dim,
                self.env.num_actions,
                LLM_MENTOR_CONFIG['model_name']
            )
            self.llm_enabled = True
            self.logger.log("✅ LLM Mentor initialized successfully")

            # Log LLM performance stats
            llm_stats = self.mentor.get_performance_stats()
            self.logger.log(f"   LLM Memory Usage: {llm_stats['memory_usage_gb']:.1f}GB")

        except Exception as e:
            self.logger.log(f"❌ LLM Mentor initialization failed: {e}", "ERROR")
            self.logger.log("🔄 Falling back to neural mentor...")
            from mentor import MultimodalMentor
            self.mentor = MultimodalMentor(self.env.state_dim, self.env.num_actions).to(DEVICE)
            self.llm_enabled = False

        # Initialize student agent
        self.student = StudentAgent(self.env.state_dim, self.env.num_actions).to(DEVICE)

        # Check memory after model loading
        post_model_memory = self.memory_manager.check_memory()
        self.logger.log(f"Memory after models: {post_model_memory['free_gb']:.1f}GB free")

        # Setup optimizers
        # Mentor optimizer might not be used if LLM is purely for inference or fine-tuned elsewhere
        if hasattr(self.mentor, 'parameters') and any(p.requires_grad for p in self.mentor.parameters()):
            self.mentor.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.mentor.parameters()),
                                               lr=MENTOR_CONFIG['learning_rate'])
        else:
            self.mentor.optimizer = None

        # Collect all student-related parameters
        student_params = list(self.student.parameters())

        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            mentor=self.mentor,
            student=self.student,  # Pass student reference, not as child module for this trainer
            mentor_hidden_dim=MENTOR_CONFIG['hidden_dim'],
            student_hidden_dim=STUDENT_CONFIG['hidden_dim']
        ).to(DEVICE)

        # Add distillation trainer parameters (its own projectors) to student optimizer
        student_params.extend(self.distillation_trainer.parameters())

        # Advanced features if available
        if ADVANCED_FEATURES_AVAILABLE:
            self.logger.log("🧠 Initializing advanced activation analysis...")
            self.math_pathway_optimizer = PathwayImportanceOptimizer()
            self.pathway_analyzer_shared = MathCriticalPathwayAnalyzer({})

            self.activation_pipeline = create_activation_based_distillation_pipeline(
                pathway_optimizer=self.math_pathway_optimizer,
                pathway_analyzer_instance=self.pathway_analyzer_shared
            )

            self.mentor_tracker = ActivationTracker(self.mentor)
            self.student_tracker = ActivationTracker(self.student)

            self.focused_distillation_loss_module = self.activation_pipeline['distillation_loss']
            if isinstance(self.focused_distillation_loss_module, nn.Module):
                self.focused_distillation_loss_module.to(DEVICE)
                student_params.extend(self.focused_distillation_loss_module.parameters())

            self.critical_signatures: List[Any] = []
        else:
            self.activation_pipeline = None
            self.critical_signatures = []
            self.focused_distillation_loss_module = None  # Ensure it's defined

        # Create student optimizer with all parameters
        self.student.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student_params),
            lr=STUDENT_CONFIG['learning_rate']
        )

        # Initialize memory and learning components
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=MEMORY_CONFIG['trajectory_buffer_size'],
            alpha=MEMORY_CONFIG['priority_alpha'],
            beta=MEMORY_CONFIG['priority_beta']
        )
        self.trajectory_buffer = TrajectoryBuffer(max_trajectories=MEMORY_CONFIG['trajectory_buffer_size'] // 100)
        self.experience_collector = ExperienceCollector(ENV_CONFIG['num_envs'])
        self.curriculum = CurriculumScheduler(CURRICULUM_CONFIG['stages'])
        self.action_processor = ActionProcessor(self.env.num_actions)

        # Training state
        self.training_phase = 'llm_warm_start' if TRAINING_CONFIG.get('llm_warm_start',
                                                                      False) and self.llm_enabled else 'human_cloning'

        self.phase_transitions = {
            'llm_warm_start': 500 if TRAINING_CONFIG.get('llm_warm_start', False) and self.llm_enabled else 0,
            'human_cloning': TRAINING_CONFIG.get('human_cloning_steps', 1000),
            'focused_distillation': TRAINING_CONFIG.get('focused_distillation_steps', 5000),
            'standard_training': float('inf')
        }
        if not (TRAINING_CONFIG.get('llm_warm_start', False) and self.llm_enabled):
            self.phase_transitions['human_cloning'] += self.phase_transitions['llm_warm_start']
            self.phase_transitions['focused_distillation'] += self.phase_transitions['llm_warm_start']

        self.timestep = 0
        self.episode_count = 0
        self.best_reward = float('-inf')

        # LLM-specific tracking
        self.llm_query_count = 0
        self.llm_query_probability = STUDENT_CONFIG['llm_query_probability'] if self.llm_enabled else 0
        self.llm_successful_queries = 0

        # Final memory check
        final_memory = self.memory_manager.check_memory()
        self.logger.log(f"Pipeline initialized! Free VRAM: {final_memory['free_gb']:.1f}GB")
        self.logger.log("LLM-Enhanced pipeline initialization complete!")

    def _update_training_phase(self):
        """Update training phase with LLM considerations"""
        # Adjust end points based on whether LLM warm start happened
        llm_warm_start_duration = self.phase_transitions.get('llm_warm_start', 0)
        human_cloning_end = llm_warm_start_duration + self.phase_transitions['human_cloning']
        focused_distill_end = human_cloning_end + self.phase_transitions['focused_distillation']

        if self.training_phase == 'llm_warm_start' and self.timestep >= llm_warm_start_duration:
            self.training_phase = 'human_cloning'
            self.logger.log(f"Transitioning to phase: {self.training_phase}")
        elif self.training_phase == 'human_cloning' and self.timestep >= human_cloning_end:
            can_do_focused = ADVANCED_FEATURES_AVAILABLE and bool(self.critical_signatures)
            self.training_phase = 'focused_distillation' if can_do_focused else 'llm_enhanced_training'
            self.logger.log(f"Transitioning to phase: {self.training_phase}")
        elif self.training_phase == 'focused_distillation' and self.timestep >= focused_distill_end:
            self.training_phase = 'standard_training'
            self.logger.log(f"Transitioning to phase: {self.training_phase}")
        elif self.training_phase == 'llm_enhanced_training' and self.timestep >= focused_distill_end:  # Fallback from focused
            self.training_phase = 'standard_training'
            self.logger.log(f"Transitioning to phase: {self.training_phase}")

        # Update LLM query probability based on phase and curriculum
        if self.llm_enabled and TRAINING_CONFIG.get('llm_annealing', True):
            base_decay = STUDENT_CONFIG['llm_query_decay']
            min_prob = STUDENT_CONFIG['min_llm_query_probability']

            if self.training_phase == 'llm_warm_start':
                self.llm_query_probability = 0.8  # High LLM usage during warm start
            else:
                self.llm_query_probability = max(
                    min_prob,
                    self.llm_query_probability * base_decay
                )

    def _should_query_llm_mentor(self, uncertainty: float, confidence: float) -> bool:
        """Decide whether to query LLM mentor based on multiple factors"""
        if not self.llm_enabled:
            return False

        # Base probability from training schedule
        base_prob = self.llm_query_probability

        # Adjust based on uncertainty and confidence
        uncertainty_factor = uncertainty  # High uncertainty increases query probability
        confidence_factor = 1.0 - confidence  # Low confidence increases query probability

        # Combine factors
        adjusted_prob = base_prob * (1.0 + uncertainty_factor + confidence_factor)
        adjusted_prob = min(0.9, adjusted_prob)  # Cap at 90%

        # Random decision
        return np.random.random() < adjusted_prob

    def _query_llm_mentor(self, state_tensor_single_env: torch.Tensor) -> Any:
        """Query LLM mentor and track statistics"""
        self.llm_query_count += 1

        try:
            with torch.no_grad():  # Ensure LLM is in eval mode if it has its own eval logic
                if hasattr(self.mentor, 'eval'): self.mentor.eval()
                advice = self.mentor.get_advice(state_tensor_single_env, verbose=False)
                if hasattr(self.mentor, 'train') and self.training_phase not in ['eval_only']: self.mentor.train(
                    False)  # Set back to LLM eval

                # Track successful queries
                if advice and advice.confidence > LLM_MENTOR_CONFIG['confidence_threshold']:
                    self.llm_successful_queries += 1

                return advice

        except Exception as e:
            self.logger.log(f"⚠️ LLM query failed: {e}", "WARN")
            return None

    def _train_models_llm_enhanced(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """LLM-enhanced training that combines neural and LLM guidance"""
        batch_data = rollout_data.get('batch_data')
        if not batch_data:
            return {}

        required_keys = ['rewards', 'values', 'dones', 'states', 'actions', 'log_probs', 'old_values']
        if not all(key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].numel() > 0
                   for key in required_keys):
            self.logger.log("Batch data incomplete for LLM-enhanced training.")
            return {}

        advantages, returns = compute_gae(
            rewards=batch_data['rewards'], values=batch_data['values'], dones=batch_data['dones'],
            gamma=TRAINING_CONFIG['gamma'], gae_lambda=TRAINING_CONFIG['gae_lambda']
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        training_metrics_accum: Dict[str, List[float]] = {}

        self.student.train()
        self.distillation_trainer.train()  # Ensure projectors in trainer are in train mode
        if self.llm_enabled and hasattr(self.mentor,
                                        'eval'):  # LLM mentor (if it's an nn.Module) should typically be in eval
            self.mentor.eval()

        for epoch in range(TRAINING_CONFIG['num_ppo_epochs']):
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            current_batch_size = min(batch_size, num_samples)

            if current_batch_size == 0:
                continue

            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, current_batch_size):
                end = min(start + current_batch_size, num_samples)
                if start == end:
                    continue

                batch_indices = indices[start:end]
                mini_batch_states = batch_data['states'][batch_indices]

                # Forward passes
                student_outputs = self.student(mini_batch_states)

                with torch.no_grad():
                    mentor_outputs = self.mentor(mini_batch_states, use_llm=self.llm_enabled)

                # Compute losses
                distill_losses = self._compute_llm_enhanced_distillation_loss(
                    mini_batch_states, student_outputs, mentor_outputs
                )

                # RL losses
                rl_losses = self._compute_rl_losses(
                    student_outputs,
                    actions=batch_data['actions'][batch_indices],
                    returns=returns[batch_indices],
                    advantages=advantages[batch_indices],
                    old_log_probs=batch_data['log_probs'][batch_indices],
                    old_values=batch_data['old_values'][batch_indices]
                )

                # Combine losses
                total_distillation_loss = sum(loss for loss in distill_losses.values()
                                              if isinstance(loss, torch.Tensor))

                total_loss = (
                        DISTILLATION_CONFIG['alpha'] * rl_losses['total_rl'] +
                        (1 - DISTILLATION_CONFIG['alpha']) * total_distillation_loss
                )

                # Backward pass
                self.student.optimizer.zero_grad()
                if total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.student.optimizer.param_groups[0]['params']),
                        TRAINING_CONFIG['max_grad_norm']
                    )
                    self.student.optimizer.step()

                # Track metrics
                for loss_dict, prefix in [(distill_losses, "distill"), (rl_losses, "rl")]:
                    for k, v_tensor in loss_dict.items():
                        metric_key = f"{prefix}_{k}"
                        if metric_key not in training_metrics_accum:
                            training_metrics_accum[metric_key] = []
                        training_metrics_accum[metric_key].append(v_tensor.item())

                if 'total_loss' not in training_metrics_accum:
                    training_metrics_accum['total_loss'] = []
                training_metrics_accum['total_loss'].append(total_loss.item())

        # Periodic memory cleanup
        self.memory_manager.periodic_cleanup(self.timestep)

        final_metrics = {k: np.mean(v) if v else 0.0 for k, v in training_metrics_accum.items()}

        # Add LLM-specific metrics
        if self.llm_enabled:
            query_success_rate = self.llm_successful_queries / max(1,
                                                                   self.llm_query_count) if self.llm_query_count > 0 else 0
            final_metrics.update({
                'llm_query_count': self.llm_query_count,  # Log accumulated count for the step
                'llm_query_success_rate': query_success_rate,
                'llm_query_probability': self.llm_query_probability
            })
            # Reset per-training-step LLM counters if needed, or accumulate globally
            # self.llm_query_count = 0
            # self.llm_successful_queries = 0

        return final_metrics

    def _compute_llm_enhanced_distillation_loss(self, states: torch.Tensor,
                                                student_outputs: Dict[str, torch.Tensor],
                                                mentor_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        device = states.device

        # Standard policy distillation
        if 'primary_logits' in student_outputs and 'policy_logits' in mentor_outputs:
            policy_loss = F.kl_div(
                F.log_softmax(student_outputs['primary_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1),
                F.softmax(mentor_outputs['policy_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1),
                reduction='batchmean', log_target=False
            ) * (DISTILLATION_CONFIG['temperature'] ** 2)
            losses['policy_distill'] = policy_loss
        else:
            losses['policy_distill'] = torch.tensor(0.0, device=device)

        # Value distillation
        if 'value' in student_outputs and 'value' in mentor_outputs:
            losses['value_distill'] = F.mse_loss(student_outputs['value'], mentor_outputs['value'])
        else:
            losses['value_distill'] = torch.tensor(0.0, device=device)

        # Feature matching - CORRECTED
        if 'features' in student_outputs and 'features' in mentor_outputs:
            projected_mentor_features = self.distillation_trainer.feature_projector(mentor_outputs['features'])
            losses['feature_match'] = F.mse_loss(student_outputs['features'], projected_mentor_features)
        else:
            losses['feature_match'] = torch.tensor(0.0, device=device)

        # LLM-specific reasoning consistency loss
        if self.llm_enabled and DISTILLATION_CONFIG.get('text_reasoning_loss', False) and states.shape[0] > 0:
            try:
                # Query LLM for a sample state to get reasoning
                sample_state = states[0].unsqueeze(0)  # Query for the first state in the batch
                llm_advice = self._query_llm_mentor(sample_state)

                if llm_advice and llm_advice.actions and llm_advice.confidence > 0.5:
                    llm_action = llm_advice.actions[0]
                    llm_confidence = llm_advice.confidence

                    # Boost student's probability for LLM-recommended action
                    student_probs_sample = F.softmax(student_outputs['primary_logits'][0],
                                                     dim=-1)  # For the sample state
                    target_prob_sample = torch.zeros_like(student_probs_sample)
                    if 0 <= llm_action < target_prob_sample.shape[-1]:
                        target_prob_sample[llm_action] = llm_confidence

                    reasoning_loss = F.mse_loss(student_probs_sample, target_prob_sample) * DISTILLATION_CONFIG.get(
                        'llm_reasoning_weight', 0.1)
                    losses['llm_reasoning'] = reasoning_loss
                else:
                    losses['llm_reasoning'] = torch.tensor(0.0, device=device)

            except Exception as e:
                self.logger.log(f"Error in LLM reasoning loss: {e}", "WARN")
                losses['llm_reasoning'] = torch.tensor(0.0, device=device)
        else:
            losses['llm_reasoning'] = torch.tensor(0.0, device=device)

        return losses

    def _compute_rl_losses(self, student_outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute RL losses (same as original)"""
        dev = student_outputs[
            'primary_logits'].device if 'primary_logits' in student_outputs else DEVICE  # Fallback to global DEVICE

        if 'primary_logits' not in student_outputs or student_outputs['primary_logits'] is None:
            self.logger.log("Missing 'primary_logits' in student_outputs for RL loss.", level="ERROR")
            return {k: torch.tensor(0.0, device=dev, requires_grad=True)  # Ensure requires_grad for aggregation
                    for k in ['policy_loss', 'value_loss', 'entropy', 'total_rl']}

        dist = Categorical(logits=student_outputs['primary_logits'])
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TRAINING_CONFIG['clip_ratio'],
                            1.0 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        current_values = student_outputs['value'].squeeze(-1)
        old_values_squeezed = old_values.squeeze(-1) if old_values.ndim > returns.ndim else old_values

        if TRAINING_CONFIG.get('clip_value_loss', True):
            values_clipped = old_values_squeezed + torch.clamp(
                current_values - old_values_squeezed,
                -TRAINING_CONFIG['clip_ratio'],
                TRAINING_CONFIG['clip_ratio']
            )
            vf_loss1 = F.mse_loss(current_values, returns)
            vf_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(vf_loss1, vf_loss2)
        else:
            value_loss = F.mse_loss(current_values, returns)

        entropy_bonus = dist.entropy().mean()

        total_rl_loss = (
                policy_loss +
                value_loss * STUDENT_CONFIG['value_coef'] -
                entropy_bonus * STUDENT_CONFIG['entropy_coef']
        )

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy_bonus,
            'total_rl': total_rl_loss,
        }

    def _collect_rollout(self, initial_states_tensor: torch.Tensor) -> Dict[str, Any]:
        """Collect rollout with LLM mentor integration"""
        self.experience_collector.reset()
        current_states_tensor = initial_states_tensor.clone()
        rollout_rewards: List[float] = []
        # Reset per-rollout LLM counters
        current_rollout_llm_queries = 0
        current_rollout_successful_llm_queries = 0

        for step_in_rollout in range(TRAINING_CONFIG['rollout_steps']):
            # Student action
            student_actions_b, student_info_b = self.student.act(current_states_tensor)

            # Query mentors based on uncertainty and phase
            llm_advice_l: List[Optional[Any]] = [None] * self.env.num_envs

            for i in range(self.env.num_envs):
                student_info = student_info_b[i]
                uncertainty = student_info.get('uncertainty', {}).get('total', 0.5)
                confidence = student_info.get('reasoning_confidence', 0.5)

                if self._should_query_llm_mentor(uncertainty, confidence):
                    current_rollout_llm_queries += 1  # Track queries for this rollout
                    advice = self._query_llm_mentor(current_states_tensor[i].unsqueeze(0))
                    llm_advice_l[i] = advice
                    if advice and advice.confidence > LLM_MENTOR_CONFIG['confidence_threshold']:
                        current_rollout_successful_llm_queries += 1

            # Environment step
            uncertainties_step = [info.get('uncertainty', {}).get('total', 0.5) for info in student_info_b]
            next_obs_np, rewards_np, terminated_np, truncated_np, infos_env = self.env.step(
                student_actions_b, uncertainties_step
            )

            # Apply reward shaping
            shaped_rewards_np = np.zeros_like(rewards_np)
            for i in range(self.env.num_envs):
                prim_act = student_actions_b[i][0] if student_actions_b[i] else 0
                current_env_info = {}
                is_done = terminated_np[i] or truncated_np[i]

                if is_done:
                    f_info_arr = infos_env.get('final_info')  # For SyncVectorEnv
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[i] is not None:
                        current_env_info = f_info_arr[i]
                    elif isinstance(infos_env, list) and i < len(infos_env):  # For DummyVecEnv or other structures
                        current_env_info = infos_env[i]

                shaped_rewards_np[i] = self.reward_shaper.shape_reward(
                    next_obs_np[i], prim_act, rewards_np[i], is_done, current_env_info, env_idx=i
                )

            # Collect experience
            with torch.no_grad():
                student_outputs_c = self.student(current_states_tensor)

            prim_acts_l_c = []
            for s_acts in student_actions_b:
                act_v = s_acts[0] if s_acts and len(s_acts) > 0 else (
                    np.random.randint(0, self.env.num_actions) if self.env.num_actions > 0 else 0)

                if not (0 <= act_v < self.env.num_actions) and self.env.num_actions > 0:
                    act_v = np.clip(act_v, 0, self.env.num_actions - 1)
                elif self.env.num_actions == 0:  # Should not happen
                    act_v = 0
                prim_acts_l_c.append(act_v)

            prim_acts_t_c = torch.tensor(prim_acts_l_c, dtype=torch.long, device=DEVICE)
            log_probs_c = Categorical(logits=student_outputs_c['primary_logits']).log_prob(prim_acts_t_c)
            values_c = student_outputs_c['value'].squeeze(-1)

            uncert_l_c = [info.get('uncertainty', {'total': 0.}) for info in student_info_b]
            # Standard mentor_advice is not used if LLM mentor is primary
            # If a hybrid system with another mentor exists, that logic would go here.
            # For now, passing llm_advice as the mentor_advice for simplicity in Experience tuple.
            self.experience_collector.add(
                state=current_states_tensor.cpu().numpy(),
                action=np.array([a[0] if a and len(a) > 0 else 0 for a in student_actions_b]),
                reward=shaped_rewards_np,
                next_state=next_obs_np,
                done=(terminated_np | truncated_np),
                log_prob=log_probs_c,
                value=values_c,
                uncertainty=uncert_l_c,
                mentor_advice=llm_advice_l  # Store LLM advice if collected
            )

            current_states_tensor = self.env.get_state_tensor(next_obs_np)

            # Track episode completions
            for i in range(self.env.num_envs):
                if terminated_np[i] or truncated_np[i]:
                    f_info_arr = infos_env.get('final_info')  # For SyncVectorEnv
                    ep_info = None
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[i] is not None:
                        ep_info = f_info_arr[i].get('episode')
                    elif isinstance(infos_env, list) and i < len(infos_env) and infos_env[
                        i] is not None:  # For DummyVecEnv
                        ep_info = infos_env[i].get('episode')

                    if ep_info is not None:
                        rollout_rewards.append(ep_info['r'])
                        self.episode_count += 1
                        # Update best reward
                        if ep_info['r'] > self.best_reward:
                            self.best_reward = ep_info['r']

            self.timestep += self.env.num_envs

        # Update global LLM counters after the rollout
        self.llm_query_count += current_rollout_llm_queries
        self.llm_successful_queries += current_rollout_successful_llm_queries

        batch_d = self.experience_collector.get_batch_tensors()
        if 'values' in batch_d and batch_d['values'] is not None:
            batch_d['old_values'] = batch_d['values'].clone().detach()
        else:  # Handle case where values might not be collected if rollout is too short / no states
            self.logger.log("Warning: 'values' not found in batch_data from experience_collector or is None.", "WARN")
            num_expected_samples = TRAINING_CONFIG['rollout_steps'] * self.env.num_envs
            batch_d['old_values'] = torch.zeros(num_expected_samples, device=DEVICE) if 'states' not in batch_d or \
                                                                                        batch_d['states'].shape[
                                                                                            0] == 0 else torch.zeros(
                batch_d['states'].shape[0], device=DEVICE)
            if 'values' not in batch_d: batch_d['values'] = batch_d['old_values'].clone()  # ensure values key exists

        return {
            'rollout_rewards': rollout_rewards,
            'mentor_queries': 0,  # Not using the standard mentor querying in this LLM-focused version
            'llm_queries_this_rollout': current_rollout_llm_queries,  # Specific to this rollout
            'batch_data': batch_d
        }

    def _query_mentor(self, state_tensor_single_env: torch.Tensor) -> Any:
        """Query standard mentor (for compatibility if needed, but LLM is primary)"""
        # This might be called if student_info['should_query_mentor'] is true
        # from a non-LLM part of the student logic.
        # If self.mentor is the LLMMentor, this will effectively call the LLM.
        if hasattr(self.mentor, 'get_advice'):
            with torch.no_grad():
                if hasattr(self.mentor, 'eval'): self.mentor.eval()
                advice = self.mentor.get_advice(state_tensor_single_env)
                # Set mentor back to train if it's an nn.Module and not in eval_only phase
                if hasattr(self.mentor, 'train') and isinstance(self.mentor,
                                                                nn.Module) and self.training_phase != 'eval_only':
                    self.mentor.train(False)  # Typically LLM part is eval, neural part might train
                return advice
        return None

    def train(self):
        """Main training loop with LLM integration"""
        self.logger.log("Starting LLM-enhanced training...")

        # Skip demonstration processing for now - focus on LLM guidance
        self.logger.log(f"Initial training phase: {self.training_phase}")
        if self.training_phase == 'human_cloning':
            self.logger.log(
                "Phase 1: Human Behavior Cloning / Synthetic Demo (Skipping actual demo loading for this LLM focus test)")
            # Simplified: assume critical_signatures might be populated by a placeholder or skipped if ADVANCED_FEATURES_AVAILABLE is false
            if ADVANCED_FEATURES_AVAILABLE:
                self.critical_signatures = []  # Placeholder
            self.timestep = self.phase_transitions['human_cloning']  # Fast-forward past this phase for now

        observations, _ = self.env.reset(seed=SEED)
        start_time = time.time()

        while self.timestep < TRAINING_CONFIG['total_timesteps']:
            # Update training phase and LLM parameters
            self._update_training_phase()

            # Collect rollout with LLM integration
            states_tensor = self.env.get_state_tensor(observations)
            rollout_data = self._collect_rollout(states_tensor)

            # observations might be from single or multiple envs based on self.env structure
            if self.experience_collector.next_states and len(self.experience_collector.next_states) > 0:
                # Ensure observations match the structure expected by get_state_tensor
                last_next_states_np = self.experience_collector.next_states[-1]
                if last_next_states_np.ndim == 1 and self.env.num_envs > 1:  # Should be (num_envs, obs_dim)
                    self.logger.log("Warning: observation dimension mismatch from collector. Attempting reshape.",
                                    "WARN")
                    # This case should ideally not happen if collector stores them correctly
                    # For now, we trust get_state_tensor to handle it or log if it's unexpected.
                    observations = last_next_states_np
                elif last_next_states_np.ndim == 2 and last_next_states_np.shape[0] == self.env.num_envs:
                    observations = last_next_states_np
                else:  # Fallback or single env case
                    observations = last_next_states_np

            # Training step
            if self.timestep > TRAINING_CONFIG['batch_size']:
                training_metrics = {}
                if self.training_phase in ['llm_warm_start', 'llm_enhanced_training',
                                           'standard_training']:  # standard_training will use llm_enhanced if llm enabled
                    training_metrics = self._train_models_llm_enhanced(rollout_data)
                elif ADVANCED_FEATURES_AVAILABLE and self.training_phase == 'focused_distillation':
                    # This part needs self.critical_signatures to be populated
                    # For this focused test, assume it might run if ADVANCED_FEATURES_AVAILABLE
                    if not self.critical_signatures:
                        self.logger.log(
                            "Focused distillation phase but no critical signatures. Running LLM enhanced instead.",
                            "WARN")
                        training_metrics = self._train_models_llm_enhanced(rollout_data)
                    # else:
                    #    training_metrics = self._train_models_focused(rollout_data) # Needs to be defined
                else:  # Fallback if no specific phase matches
                    training_metrics = self._train_models_llm_enhanced(rollout_data)

                if training_metrics:
                    self._log_training_metrics(training_metrics)

            # Periodic evaluation
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['eval_interval'] == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation_metrics(eval_metrics)

            # Curriculum learning
            if CURRICULUM_CONFIG['enabled']:
                avg_reward_stats = self.trajectory_buffer.get_statistics()
                avg_reward = avg_reward_stats.get('avg_reward', 0) if isinstance(avg_reward_stats, dict) else 0
                self.curriculum.get_current_config(avg_reward)

            # Checkpointing
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()

            # Progress logging
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['log_interval'] == 0:
                self._log_progress(time.time() - start_time)

                # Log LLM-specific stats
                if self.llm_enabled and hasattr(self.mentor, 'get_performance_stats'):
                    llm_stats = self.mentor.get_performance_stats()
                    self.logger.log(f"LLM Stats: Queries made this session: {self.llm_query_count}, "
                                    f"Cache hit rate: {llm_stats.get('cache_hit_rate', 0):.2%}, "
                                    f"Success rate: {llm_stats.get('llm_successful_query_integration_rate', 0):.2%}")

                # Memory check
                memory_stats = self.memory_manager.check_memory()
                if memory_stats.get('percent_used', 0) > 80:
                    self.logger.log(f"⚠️ High VRAM usage: {memory_stats.get('percent_used', 0):.1f}%")

        self.logger.log("LLM-enhanced training completed!")
        self._final_evaluation()

    def _train_models_standard(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Standard training fallback (can still be LLM enhanced if LLM is on)"""
        self.logger.log("Running _train_models_standard (which might use LLM enhancements if enabled).")
        return self._train_models_llm_enhanced(rollout_data)

    def _train_models_focused(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Focused training with activation analysis (placeholder if not fully implemented)"""
        self.logger.log("Attempting focused training...")
        if not ADVANCED_FEATURES_AVAILABLE or self.focused_distillation_loss_module is None:
            self.logger.log("Focused distillation features not available. Falling back to LLM enhanced.", "WARN")
            return self._train_models_llm_enhanced(rollout_data)

        # This is a simplified version for the purpose of this fix, assuming
        # the FocusedDistillationLoss module is correctly set up and its parameters
        # are part of the student's optimizer.
        batch_data = rollout_data.get('batch_data')
        if not batch_data: return {}

        required_keys = ['rewards', 'values', 'dones', 'states', 'actions', 'log_probs', 'old_values']
        if not all(
                key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].numel() > 0 for key
                in required_keys):
            self.logger.log("Batch data incomplete for focused training.")
            return {}

        advantages, returns = compute_gae(
            rewards=batch_data['rewards'], values=batch_data['values'], dones=batch_data['dones'],
            gamma=TRAINING_CONFIG['gamma'], gae_lambda=TRAINING_CONFIG['gae_lambda']
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        training_metrics_accum: Dict[str, List[float]] = {}

        self.student.train()
        if hasattr(self.focused_distillation_loss_module, 'train'): self.focused_distillation_loss_module.train()
        if self.llm_enabled and hasattr(self.mentor, 'eval'): self.mentor.eval()

        for _ in range(TRAINING_CONFIG['num_ppo_epochs']):
            # ... (data loading, mini-batching as in _train_models_llm_enhanced) ...
            # Simplified for brevity - assume mini_batch_states, etc. are prepared
            num_samples = len(batch_data['states'])
            current_batch_size = min(TRAINING_CONFIG['batch_size'], num_samples)
            if current_batch_size == 0: continue
            indices = torch.randperm(num_samples)

            for start_idx in range(0, num_samples, current_batch_size):
                end_idx = min(start_idx + current_batch_size, num_samples)
                if start_idx == end_idx: continue
                batch_indices = indices[start_idx:end_idx]
                mini_batch_states = batch_data['states'][batch_indices]

                # Critical: Get activations for focused loss
                self.mentor_tracker.clear_cache()
                self.student_tracker.clear_cache()

                with torch.no_grad():
                    mentor_outputs_distill = self.mentor(mini_batch_states,
                                                         use_llm=self.llm_enabled)  # LLM can still guide mentor
                mentor_activations_distill = self.mentor_tracker.get_activations()

                student_outputs_distill_and_rl = self.student(mini_batch_states)
                student_activations_distill_and_rl = self.student_tracker.get_activations()

                focused_losses_dict = self.focused_distillation_loss_module(
                    student_outputs_distill_and_rl, mentor_outputs_distill,
                    student_activations_distill_and_rl, mentor_activations_distill,
                    self.critical_signatures,  # Ensure this is populated
                    states=mini_batch_states
                )

                rl_losses_dict = self._compute_rl_losses(
                    student_outputs_distill_and_rl,
                    actions=batch_data['actions'][batch_indices],
                    returns=returns[batch_indices],
                    advantages=advantages[batch_indices],
                    old_log_probs=batch_data['log_probs'][batch_indices],
                    old_values=batch_data['old_values'][batch_indices]
                )

                total_focused_loss = focused_losses_dict.get('total_focused', torch.tensor(0.0, device=DEVICE))
                total_rl_loss = rl_losses_dict.get('total_rl', torch.tensor(0.0, device=DEVICE))

                # Combine: focused_distillation_weight for the focused part, standard alpha for RL part
                focused_weight = DISTILLATION_CONFIG.get('focused_distill_weight', 0.5)  # Add this to config if needed
                final_loss = focused_weight * total_focused_loss + (1 - focused_weight) * total_rl_loss

                self.student.optimizer.zero_grad()
                if final_loss.requires_grad: final_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.student.optimizer.param_groups[0]['params']),
                    TRAINING_CONFIG['max_grad_norm'])
                self.student.optimizer.step()

                # Accumulate metrics
                for k, v_tensor in focused_losses_dict.items():
                    metric_k = f"focused_{k}";
                    training_metrics_accum.setdefault(metric_k, []).append(v_tensor.item())
                for k, v_tensor in rl_losses_dict.items():
                    metric_k = f"rl_{k}";
                    training_metrics_accum.setdefault(metric_k, []).append(v_tensor.item())
                training_metrics_accum.setdefault('total_loss_focused_phase', []).append(final_loss.item())

        self.memory_manager.periodic_cleanup(self.timestep)
        final_metrics = {k: np.mean(v) if v else 0.0 for k, v in training_metrics_accum.items()}
        return final_metrics

    def _evaluate(self) -> Dict[str, float]:
        """Evaluation with LLM mentor consideration"""
        self.logger.log("Evaluating...")
        if hasattr(self.student, 'eval'): self.student.eval()
        if hasattr(self.mentor, 'eval'): self.mentor.eval()  # LLM should be in eval

        eval_env = create_environment(env_name=ENV_CONFIG['name'], num_envs=1)  # Eval on single env
        rewards_list, lengths_list, llm_queries_eval = [], [], []

        for _ in range(LOGGING_CONFIG.get('num_eval_episodes', 3)):
            obs_np, _ = eval_env.reset()
            done_eval = False
            ep_reward, ep_length, ep_llm_queries = 0.0, 0, 0
            while not done_eval:
                state_tensor = eval_env.get_state_tensor(obs_np)
                actions_b, info_b = self.student.act(state_tensor, deterministic=True)

                # Simulate if LLM would be queried (without actually incurring cost if not desired for eval)
                # Or, allow actual querying to see LLM impact during eval
                if self.llm_enabled and self._should_query_llm_mentor(
                        info_b[0].get('uncertainty', {}).get('total', 0.5), info_b[0].get('reasoning_confidence', 0.5)):
                    ep_llm_queries += 1
                    # Optionally, use LLM advice for action:
                    # advice = self._query_llm_mentor(state_tensor)
                    # if advice and advice.actions: actions_b[0][0] = advice.actions[0]

                next_obs_np, reward_np, term_np, trunc_np, eval_infos = eval_env.step(
                    [actions_b[0]], [info_b[0].get('uncertainty', {}).get('total', 0.5)]
                )
                obs_np = next_obs_np
                done_eval = term_np[0] or trunc_np[0]
                ep_reward += reward_np[0]
                ep_length += 1
            rewards_list.append(ep_reward)
            lengths_list.append(ep_length)
            llm_queries_eval.append(ep_llm_queries)

        eval_env.close()
        if hasattr(self.student, 'train'): self.student.train()  # Set student back to train mode
        # LLM mentor part is generally kept in eval unless explicitly training it.

        return {
            'eval_reward_mean': float(np.mean(rewards_list)) if rewards_list else 0.0,
            'eval_reward_std': float(np.std(rewards_list)) if rewards_list else 0.0,
            'eval_length_mean': float(np.mean(lengths_list)) if lengths_list else 0.0,
            'eval_llm_queries_mean': float(np.mean(llm_queries_eval)) if llm_queries_eval else 0.0,
        }

    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics including LLM stats"""
        for key, value in metrics.items():
            # Ensure value is a scalar float/int for logging
            log_value = value
            if isinstance(value, torch.Tensor):
                log_value = value.item()
            elif isinstance(value, np.generic):
                log_value = value.item()
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                log_value = np.mean(value)
            elif isinstance(value, (list, np.ndarray)) and len(value) == 0:
                log_value = 0.0

            if isinstance(log_value, (float, int)):
                self.logger.log_step(self.timestep, {f"{self.training_phase}_{key.replace('.', '_')}": log_value})
            # else:
            #    self.logger.log(f"Warning: Metric {key} has unloggable type {type(log_value)}", "WARN")

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        for key, value in metrics.items():
            self.logger.log_step(self.timestep, {key.replace('.', '_'): value})

    def _log_progress(self, elapsed_time: float):
        """Log progress with LLM information"""
        stats = self.logger.get_statistics()
        current_loss = 0.0
        loss_keys_in_order = [
            f"{self.training_phase}_total_loss",  # e.g. llm_enhanced_training_total_loss
            f"focused_total_loss_focused_phase",  # if in focused phase
            f"{self.training_phase}_rl_total_rl",
            f"{self.training_phase}_distill_total_loss_combined"  # if only distill part has a total
        ]
        for key in loss_keys_in_order:
            if key in stats and stats[key] != 0:
                current_loss = stats[key]
                break
        if current_loss == 0 and 'losses' in self.logger.metrics_history:  # Fallback to raw history
            for key_suffix in ['total_loss', 'total_focused', 'total_rl', 'distill_loss_total_combined']:
                full_key = f"{self.training_phase}_{key_suffix}"
                if full_key in self.logger.metrics_history['losses'] and self.logger.metrics_history['losses'][
                    full_key]:
                    current_loss = self.logger.metrics_history['losses'][full_key][-1]
                    break
                if current_loss != 0: break

        self.logger.log(
            f"Phase: {self.training_phase} | Timestep: {self.timestep:,} | Episode: {self.episode_count:,} | "
            f"Avg Reward (100): {stats.get('avg_reward_100', 0.0):.2f} | Best Ep Reward: {self.best_reward:.2f} | "
            f"Loss: {current_loss:.3e} | "
            f"LLM Queries (total): {self.llm_query_count} | LLM Query Prob: {self.llm_query_probability:.3f} | "
            f"Time: {elapsed_time:.1f}s"
        )
        if self.timestep > 0 and self.timestep % (LOGGING_CONFIG['log_interval'] * 10) == 0:
            self.logger.plot_training_curves()

    def _save_checkpoint(self):
        """Save checkpoint including LLM state if applicable (LLM itself is usually not saved due to size)"""
        models_state_dict = {'student': self.student.state_dict()}
        optimizers_state_dict = {'student_optimizer': self.student.optimizer.state_dict()}

        # If mentor is an nn.Module and has an optimizer (i.e., not just a pure LLM inference endpoint)
        if isinstance(self.mentor, nn.Module) and hasattr(self.mentor, 'state_dict'):
            models_state_dict['mentor'] = self.mentor.state_dict()
        if hasattr(self.mentor, 'optimizer') and self.mentor.optimizer is not None:
            optimizers_state_dict['mentor_optimizer'] = self.mentor.optimizer.state_dict()

        # Include DistillationTrainer's parameters (projectors)
        # These are already part of student_optimizer if logic is correct, but saving state_dict is fine.
        if hasattr(self.distillation_trainer, 'state_dict'):
            models_state_dict['distillation_trainer_projectors'] = self.distillation_trainer.state_dict()

        additional_data = {
            'critical_signatures': self.critical_signatures if ADVANCED_FEATURES_AVAILABLE else [],
            'training_phase': self.training_phase,
            'best_reward': self.best_reward,
            'episode_count': self.episode_count,
            'llm_query_count': self.llm_query_count,
            'llm_query_probability': self.llm_query_probability,
        }
        if ADVANCED_FEATURES_AVAILABLE and self.activation_pipeline and hasattr(
                self.activation_pipeline.get('pathway_analyzer'), 'model_structure'):
            additional_data['pathway_analyzer_state'] = {
                'model_structure': self.activation_pipeline['pathway_analyzer'].model_structure}

        checkpoint_path = os.path.join(self.logger.log_dir, f'checkpoint_{self.timestep}.pt')
        try:
            torch.save({
                'timestep': self.timestep,
                'metrics_history': self.logger.metrics_history,
                'models_state_dict': models_state_dict,
                'optimizers_state_dict': optimizers_state_dict,
                'additional_pipeline_data': additional_data
            }, checkpoint_path)
            self.logger.log(f"Saved checkpoint at timestep {self.timestep}")
        except Exception as e:
            self.logger.log(f"Error saving checkpoint: {e}", "ERROR")

    def _final_evaluation(self):
        """Final evaluation with comprehensive LLM analysis"""
        self.logger.log("Performing final evaluation...")
        metrics = self._evaluate()  # Use the existing _evaluate method
        self._log_evaluation_metrics(metrics)  # Log them properly

        traj_stats = self.trajectory_buffer.get_statistics()
        avg_traj_reward = traj_stats.get('avg_reward', 0.0) if isinstance(traj_stats, dict) else 0.0

        self.logger.log("=== FINAL LLM-ENHANCED PIPELINE RESULTS ===")
        self.logger.log(f"Total Timesteps: {self.timestep:,}, Total Episodes: {self.episode_count:,}")
        self.logger.log(f"Best Episode Reward Achieved: {self.best_reward:.2f}")
        self.logger.log(
            f"Final Avg Eval Reward: {metrics.get('eval_reward_mean', 0.0):.2f} +/- {metrics.get('eval_reward_std', 0.0):.2f}")
        self.logger.log(f"Final Avg Eval Episode Length: {metrics.get('eval_length_mean', 0.0):.1f}")
        self.logger.log(f"Final Avg Trajectory Buffer Reward: {avg_traj_reward:.2f}")

        if self.llm_enabled:
            self.logger.log(f"Total LLM Queries Made During Training: {self.llm_query_count}")
            if hasattr(self.mentor, 'get_performance_stats'):
                llm_perf_stats = self.mentor.get_performance_stats()
                self.logger.log(f"LLM Cache Hit Rate: {llm_perf_stats.get('cache_hit_rate', 0.0):.2%}")
                self.logger.log(
                    f"LLM Query Success/Integration Rate: {llm_perf_stats.get('llm_successful_query_integration_rate', 0.0):.2%}")
            self.logger.log(f"Final LLM Query Probability: {self.llm_query_probability:.3f}")

        if ADVANCED_FEATURES_AVAILABLE:
            self.logger.log(f"Number of Critical Signatures Identified: {len(self.critical_signatures)}")

        self._save_checkpoint()  # Save final model
        self.logger.plot_training_curves()
        self.logger.log("Final evaluation and logging complete.")


def main():
    """Main function with LLM support"""
    parser = argparse.ArgumentParser(description='LLM-Enhanced Revolutionary AI Pipeline')
    parser.add_argument('--log_dir', type=str, default='logs_llm')  # Changed default log_dir
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--env_name', type=str, default=ENV_CONFIG['name'])
    parser.add_argument('--eval_only', action='store_true')
    # Removed --llm_model and --disable_llm as they are handled by llm_config.py now
    # Kept human_demos_path and skip_behavior_cloning for ADVANCED_FEATURES_AVAILABLE block if needed
    parser.add_argument('--human_demos_path', type=str, default=None,
                        help="Path to human demonstrations for Phase 1 (if ADVANCED_FEATURES_AVAILABLE).")
    parser.add_argument('--skip_behavior_cloning', action='store_true',
                        help="Skip behavior cloning/synthetic demo phase (if ADVANCED_FEATURES_AVAILABLE).")
    parser.add_argument('--memory_test', action='store_true', help='Test memory usage')

    args = parser.parse_args()

    # Update ENV_CONFIG from args if provided, LLM config is now primarily from llm_config.py
    if args.env_name:
        ENV_CONFIG['name'] = args.env_name

    # Memory test mode
    if args.memory_test:
        print("🧪 Running memory test...")
        memory_manager = LLMMemoryManager()
        initial_memory = memory_manager.check_memory()
        print(f"Initial: {initial_memory}")

        try:
            # Try to load the primary LLM from config to test its memory footprint
            test_mentor = create_llm_mentor(4, 2, LLM_MENTOR_CONFIG['model_name'])
            post_llm_memory = memory_manager.check_memory()
            print(f"After LLM ({LLM_MENTOR_CONFIG['model_name']}): {post_llm_memory}")

            del test_mentor
            memory_manager.emergency_cleanup()
            final_memory = memory_manager.check_memory()
            print(f"After cleanup: {final_memory}")

        except Exception as e:
            print(f"Memory test failed: {e}")
            import traceback
            traceback.print_exc()

        return

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Initialize and run pipeline
    try:
        pipeline = LLMRevolutionaryPipeline(args)

        pipeline.logger.log(f"Environment: {ENV_CONFIG['name']}")
        pipeline.logger.log(f"LLM Enabled: {pipeline.llm_enabled}")
        if pipeline.llm_enabled:
            pipeline.logger.log(
                f"LLM Model: {pipeline.mentor.llm.model_name if hasattr(pipeline.mentor, 'llm') else 'N/A'}")

        # Load checkpoint if specified
        if args.load_checkpoint:
            pipeline.logger.log(f"Attempting to load checkpoint: {args.load_checkpoint}")
            if not os.path.exists(args.load_checkpoint):
                pipeline.logger.log(f"Checkpoint file not found: {args.load_checkpoint}", "ERROR")
            else:
                try:
                    ckpt = torch.load(args.load_checkpoint, map_location=DEVICE)

                    models_state_dict = ckpt.get('models_state_dict', {})
                    if 'student' in models_state_dict:
                        pipeline.student.load_state_dict(models_state_dict['student'])
                    if 'mentor' in models_state_dict and isinstance(pipeline.mentor, nn.Module):
                        pipeline.mentor.load_state_dict(models_state_dict['mentor'])
                    if 'distillation_trainer_projectors' in models_state_dict and hasattr(pipeline.distillation_trainer,
                                                                                          'load_state_dict'):
                        pipeline.distillation_trainer.load_state_dict(
                            models_state_dict['distillation_trainer_projectors'])

                    optimizers_state_dict = ckpt.get('optimizers_state_dict', {})
                    if 'student_optimizer' in optimizers_state_dict and hasattr(pipeline.student,
                                                                                'optimizer') and pipeline.student.optimizer:
                        pipeline.student.optimizer.load_state_dict(optimizers_state_dict['student_optimizer'])
                    if 'mentor_optimizer' in optimizers_state_dict and hasattr(pipeline.mentor,
                                                                               'optimizer') and pipeline.mentor.optimizer:
                        pipeline.mentor.optimizer.load_state_dict(optimizers_state_dict['mentor_optimizer'])

                    pipeline.timestep = ckpt.get('timestep', 0)
                    additional_data = ckpt.get('additional_pipeline_data', {})
                    pipeline.episode_count = additional_data.get('episode_count', 0)
                    pipeline.best_reward = additional_data.get('best_reward', float('-inf'))
                    pipeline.llm_query_count = additional_data.get('llm_query_count', 0)
                    pipeline.llm_query_probability = additional_data.get('llm_query_probability', STUDENT_CONFIG[
                        'llm_query_probability'] if pipeline.llm_enabled else 0)

                    if ADVANCED_FEATURES_AVAILABLE:
                        pipeline.critical_signatures = additional_data.get('critical_signatures', [])
                        if pipeline.activation_pipeline and 'pathway_analyzer_state' in additional_data and hasattr(
                                pipeline.activation_pipeline.get('pathway_analyzer'), 'model_structure'):
                            pipeline.activation_pipeline['pathway_analyzer'].model_structure = additional_data[
                                'pathway_analyzer_state'].get('model_structure', {})

                    pipeline.training_phase = additional_data.get('training_phase', 'human_cloning')
                    pipeline.logger.metrics_history = ckpt.get('metrics_history', pipeline.logger.metrics_history)
                    pipeline.logger.log(f"Checkpoint loaded from timestep {pipeline.timestep}")

                except Exception as e:
                    pipeline.logger.log(f"Failed to load checkpoint: {e}", "ERROR")
                    import traceback
                    pipeline.logger.log(traceback.format_exc(), "ERROR")

        if args.eval_only:
            if not args.load_checkpoint:
                pipeline.logger.log("Evaluation mode requires a checkpoint to be loaded.", "ERROR")
                return
            pipeline._final_evaluation()
        else:
            pipeline.train()

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjusted for potentially smaller LLMs
        print("🔧 Set CUDA memory allocation strategy for efficiency (max_split_size_mb:128)")

    main()