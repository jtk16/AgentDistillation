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
        self.mentor.optimizer = optim.Adam(self.mentor.parameters(), lr=MENTOR_CONFIG['learning_rate'])

        # Collect all student-related parameters
        student_params = list(self.student.parameters())

        # Initialize distillation trainer
        self.distillation_trainer = DistillationTrainer(
            mentor=self.mentor,
            student=self.student,
            mentor_hidden_dim=MENTOR_CONFIG['hidden_dim'],
            student_hidden_dim=STUDENT_CONFIG['hidden_dim']
        ).to(DEVICE)

        # Add distillation trainer parameters to student optimizer
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
        self.training_phase = 'llm_warm_start' if TRAINING_CONFIG['llm_warm_start'] else 'human_cloning'
        self.phase_transitions = {
            'llm_warm_start': 500,  # Brief LLM-guided warm start
            'human_cloning': TRAINING_CONFIG.get('human_cloning_steps', 1000),
            'focused_distillation': TRAINING_CONFIG.get('focused_distillation_steps', 5000),
            'standard_training': float('inf')
        }

        self.timestep = 0
        self.episode_count = 0
        self.best_reward = float('-inf')

        # LLM-specific tracking
        self.llm_query_count = 0
        self.llm_query_probability = STUDENT_CONFIG['llm_query_probability']
        self.llm_successful_queries = 0

        # Final memory check
        final_memory = self.memory_manager.check_memory()
        self.logger.log(f"Pipeline initialized! Free VRAM: {final_memory['free_gb']:.1f}GB")
        self.logger.log("LLM-Enhanced pipeline initialization complete!")

    def _update_training_phase(self):
        """Update training phase with LLM considerations"""
        llm_start = self.phase_transitions.get('llm_warm_start', 0)
        h_end = self.phase_transitions['human_cloning']
        f_end = self.phase_transitions['focused_distillation']

        if self.timestep < llm_start:
            self.training_phase = 'llm_warm_start'
        elif self.timestep < h_end:
            self.training_phase = 'human_cloning'
        elif self.timestep < f_end:
            # Check if we can do focused distillation
            can_do_focused = bool(self.critical_signatures) if ADVANCED_FEATURES_AVAILABLE else False
            if can_do_focused:
                self.training_phase = 'focused_distillation'
            else:
                self.training_phase = 'llm_enhanced_training'  # LLM-guided standard training
        else:
            self.training_phase = 'standard_training'

        # Update LLM query probability based on phase and curriculum
        if TRAINING_CONFIG['llm_annealing']:
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
            with torch.no_grad():
                advice = self.mentor.get_advice(state_tensor_single_env, verbose=False)

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
        self.distillation_trainer.train()
        if self.llm_enabled:
            self.mentor.eval()  # LLM mentor should stay in eval mode

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
                if self.llm_enabled and hasattr(self.mentor, 'get_performance_stats'):
                    # Enhanced distillation with LLM reasoning
                    distill_losses = self._compute_llm_enhanced_distillation_loss(
                        mini_batch_states, student_outputs, mentor_outputs
                    )
                else:
                    # Standard distillation
                    distill_losses = self.distillation_trainer.compute_distillation_loss_components(
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
                    for k, v in loss_dict.items():
                        metric_key = f"{prefix}_{k}"
                        if metric_key not in training_metrics_accum:
                            training_metrics_accum[metric_key] = []
                        if isinstance(v, torch.Tensor):
                            training_metrics_accum[metric_key].append(v.item())
                        else:
                            training_metrics_accum[metric_key].append(v)

                if 'total_loss' not in training_metrics_accum:
                    training_metrics_accum['total_loss'] = []
                training_metrics_accum['total_loss'].append(total_loss.item())

        # Periodic memory cleanup
        self.memory_manager.periodic_cleanup(self.timestep)

        final_metrics = {k: np.mean(v) if v else 0.0 for k, v in training_metrics_accum.items()}

        # Add LLM-specific metrics
        if self.llm_enabled:
            query_success_rate = self.llm_successful_queries / max(1, self.llm_query_count)
            final_metrics.update({
                'llm_query_count': self.llm_query_count,
                'llm_query_success_rate': query_success_rate,
                'llm_query_probability': self.llm_query_probability
            })

        return final_metrics

    def _compute_llm_enhanced_distillation_loss(self, states: torch.Tensor,
                                                student_outputs: Dict[str, torch.Tensor],
                                                mentor_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute enhanced distillation loss that incorporates LLM reasoning"""
        losses = {}

        # Standard policy distillation
        policy_loss = F.kl_div(
            F.log_softmax(student_outputs['primary_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1),
            F.softmax(mentor_outputs['policy_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1),
            reduction='batchmean', log_target=False
        ) * (DISTILLATION_CONFIG['temperature'] ** 2)
        losses['policy_distill'] = policy_loss

        # Value distillation
        if 'value' in student_outputs and 'value' in mentor_outputs:
            losses['value_distill'] = F.mse_loss(student_outputs['value'], mentor_outputs['value'])
        else:
            losses['value_distill'] = torch.tensor(0.0, device=states.device)

        # Feature matching
        if 'features' in student_outputs and 'features' in mentor_outputs:
            losses['feature_match'] = F.mse_loss(student_outputs['features'], mentor_outputs['features'])
        else:
            losses['feature_match'] = torch.tensor(0.0, device=states.device)

        # LLM-specific reasoning consistency loss
        if self.llm_enabled and DISTILLATION_CONFIG.get('text_reasoning_loss', False):
            try:
                # Query LLM for a sample state to get reasoning
                sample_state = states[0].unsqueeze(0)
                llm_advice = self._query_llm_mentor(sample_state)

                if llm_advice and llm_advice.confidence > 0.5:
                    # Create reasoning consistency loss based on action probabilities
                    llm_action = llm_advice.actions[0] if llm_advice.actions else 0
                    llm_confidence = llm_advice.confidence

                    # Boost student's probability for LLM-recommended action
                    student_probs = F.softmax(student_outputs['primary_logits'], dim=-1)
                    target_prob = torch.zeros_like(student_probs)
                    target_prob[:, llm_action] = llm_confidence

                    reasoning_loss = F.mse_loss(student_probs[:1], target_prob[:1]) * DISTILLATION_CONFIG[
                        'llm_reasoning_weight']
                    losses['llm_reasoning'] = reasoning_loss
                else:
                    losses['llm_reasoning'] = torch.tensor(0.0, device=states.device)

            except Exception as e:
                losses['llm_reasoning'] = torch.tensor(0.0, device=states.device)
        else:
            losses['llm_reasoning'] = torch.tensor(0.0, device=states.device)

        return losses

    def _compute_rl_losses(self, student_outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute RL losses (same as original)"""
        dev = student_outputs['primary_logits'].device

        if 'primary_logits' not in student_outputs:
            return {k: torch.tensor(0.0, device=dev, requires_grad=True)
                    for k in ['policy_loss', 'value_loss', 'entropy', 'total_rl']}

        dist = Categorical(logits=student_outputs['primary_logits'])
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TRAINING_CONFIG['clip_ratio'],
                            1.0 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        current_values = student_outputs['value'].squeeze(-1)
        old_values_squeezed = old_values.squeeze(-1)

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
        num_mentor_queries = 0
        num_llm_queries = 0

        for step in range(TRAINING_CONFIG['rollout_steps']):
            # Student action
            student_actions_b, student_info_b = self.student.act(current_states_tensor)

            # Query mentors based on uncertainty and phase
            mentor_advice_l: List[Optional[Any]] = [None] * self.env.num_envs
            llm_advice_l: List[Optional[Any]] = [None] * self.env.num_envs

            for i in range(self.env.num_envs):
                student_info = student_info_b[i]
                uncertainty = student_info.get('uncertainty', {}).get('total', 0.5)
                confidence = student_info.get('reasoning_confidence', 0.5)

                # Standard mentor query (existing logic)
                if student_info.get('should_query_mentor', False):
                    num_mentor_queries += 1
                    mentor_advice_l[i] = self._query_mentor(current_states_tensor[i].unsqueeze(0))

                # LLM mentor query (new logic)
                if self._should_query_llm_mentor(uncertainty, confidence):
                    num_llm_queries += 1
                    llm_advice_l[i] = self._query_llm_mentor(current_states_tensor[i].unsqueeze(0))

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
                    f_info_arr = infos_env.get('final_info')
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[i] is not None:
                        current_env_info = f_info_arr[i]

                shaped_rewards_np[i] = self.reward_shaper.shape_reward(
                    next_obs_np[i], prim_act, rewards_np[i], is_done, current_env_info, env_idx=i
                )

            # Collect experience
            with torch.no_grad():
                student_outputs_c = self.student(current_states_tensor)

            prim_acts_l_c = []
            for s_acts in student_actions_b:
                act_v = s_acts[0] if s_acts and len(s_acts) > 0 else 0
                if not (0 <= act_v < self.env.num_actions):
                    act_v = np.clip(act_v, 0, self.env.num_actions - 1)
                prim_acts_l_c.append(act_v)

            prim_acts_t_c = torch.tensor(prim_acts_l_c, dtype=torch.long, device=DEVICE)
            log_probs_c = Categorical(logits=student_outputs_c['primary_logits']).log_prob(prim_acts_t_c)
            values_c = student_outputs_c['value'].squeeze(-1)

            uncert_l_c = [info.get('uncertainty', {'total': 0.}) for info in student_info_b]

            self.experience_collector.add(
                state=current_states_tensor.cpu().numpy(),
                action=np.array([a[0] if a and len(a) > 0 else 0 for a in student_actions_b]),
                reward=shaped_rewards_np,
                next_state=next_obs_np,
                done=(terminated_np | truncated_np),
                log_prob=log_probs_c,
                value=values_c,
                uncertainty=uncert_l_c,
                mentor_advice=mentor_advice_l
            )

            current_states_tensor = self.env.get_state_tensor(next_obs_np)

            # Track episode completions
            for i in range(self.env.num_envs):
                if terminated_np[i] or truncated_np[i]:
                    f_info_arr = infos_env.get('final_info')
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[i] is not None:
                        if 'episode' in f_info_arr[i]:
                            rollout_rewards.append(f_info_arr[i]['episode']['r'])
                            self.episode_count += 1

            self.timestep += self.env.num_envs

        batch_d = self.experience_collector.get_batch_tensors()
        if 'values' in batch_d:
            batch_d['old_values'] = batch_d['values'].clone().detach()
        else:
            batch_d['old_values'] = torch.empty(0, device=DEVICE)

        return {
            'rollout_rewards': rollout_rewards,
            'mentor_queries': num_mentor_queries,
            'llm_queries': num_llm_queries,
            'batch_data': batch_d
        }

    def _query_mentor(self, state_tensor_single_env: torch.Tensor) -> Any:
        """Query standard mentor (for compatibility)"""
        if hasattr(self.mentor, 'get_advice'):
            with torch.no_grad():
                self.mentor.eval()
                return self.mentor.get_advice(state_tensor_single_env)
        return None

    def train(self):
        """Main training loop with LLM integration"""
        self.logger.log("Starting LLM-enhanced training...")

        # Skip demonstration processing for now - focus on LLM guidance
        self.logger.log("Phase 1: LLM-Guided Training (skipping demo processing)")

        observations, _ = self.env.reset(seed=SEED)
        start_time = time.time()

        while self.timestep < TRAINING_CONFIG['total_timesteps']:
            # Update training phase and LLM parameters
            self._update_training_phase()

            # Collect rollout with LLM integration
            states_tensor = self.env.get_state_tensor(observations)
            rollout_data = self._collect_rollout(states_tensor)

            if self.experience_collector.next_states and len(self.experience_collector.next_states) > 0:
                observations = self.experience_collector.next_states[-1]

            # Training step
            if self.timestep > TRAINING_CONFIG['batch_size']:
                if self.training_phase in ['llm_warm_start', 'llm_enhanced_training']:
                    training_metrics = self._train_models_llm_enhanced(rollout_data)
                elif ADVANCED_FEATURES_AVAILABLE and self.training_phase == 'focused_distillation':
                    training_metrics = self._train_models_focused(rollout_data)
                else:
                    training_metrics = self._train_models_standard(rollout_data)

                if training_metrics:
                    self._log_training_metrics(training_metrics)

            # Periodic evaluation
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['eval_interval'] == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation_metrics(eval_metrics)

            # Curriculum learning
            if CURRICULUM_CONFIG['enabled']:
                avg_reward = self.trajectory_buffer.get_statistics().get('avg_reward', 0)
                self.curriculum.get_current_config(avg_reward)

            # Checkpointing
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()

            # Progress logging
            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['log_interval'] == 0:
                self._log_progress(time.time() - start_time)

                # Log LLM-specific stats
                if self.llm_enabled:
                    llm_stats = self.mentor.get_performance_stats()
                    self.logger.log(f"LLM Stats: {self.llm_query_count} queries, "
                                    f"{llm_stats['cache_hit_rate']:.2%} cache rate")

                # Memory check
                memory_stats = self.memory_manager.check_memory()
                if memory_stats['percent_used'] > 80:
                    self.logger.log(f"⚠️ High VRAM usage: {memory_stats['percent_used']:.1f}%")

        self.logger.log("LLM-enhanced training completed!")
        self._final_evaluation()

    # Additional methods (abbreviated for space - would include all original methods)
    def _train_models_standard(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Standard training fallback"""
        # Implementation similar to original but with LLM integration
        pass

    def _train_models_focused(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Focused training with activation analysis"""
        # Implementation from original pipeline
        pass

    def _evaluate(self) -> Dict[str, float]:
        """Evaluation with LLM mentor consideration"""
        # Implementation similar to original but tracking LLM usage
        pass

    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics including LLM stats"""
        # Implementation similar to original plus LLM metrics
        pass

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        # Implementation from original
        pass

    def _log_progress(self, elapsed_time: float):
        """Log progress with LLM information"""
        # Implementation similar to original plus LLM stats
        pass

    def _save_checkpoint(self):
        """Save checkpoint including LLM state"""
        # Implementation similar to original but with LLM considerations
        pass

    def _final_evaluation(self):
        """Final evaluation with comprehensive LLM analysis"""
        # Implementation similar to original plus LLM performance analysis
        pass


def main():
    """Main function with LLM support"""
    parser = argparse.ArgumentParser(description='LLM-Enhanced Revolutionary AI Pipeline')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--env_name', type=str, default=ENV_CONFIG['name'])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--llm_model', type=str, default=LLM_MENTOR_CONFIG['model_name'])
    parser.add_argument('--disable_llm', action='store_true', help='Disable LLM mentor')
    parser.add_argument('--memory_test', action='store_true', help='Test memory usage')

    args = parser.parse_args()

    # Update configs based on args
    if args.env_name:
        ENV_CONFIG['name'] = args.env_name
    if args.llm_model:
        LLM_MENTOR_CONFIG['model_name'] = args.llm_model
    if args.disable_llm:
        MENTOR_CONFIG['use_llm_mentor'] = False

    # Memory test mode
    if args.memory_test:
        print("🧪 Running memory test...")
        memory_manager = LLMMemoryManager()
        initial_memory = memory_manager.check_memory()
        print(f"Initial: {initial_memory}")

        try:
            test_mentor = create_llm_mentor(4, 2, LLM_MENTOR_CONFIG['model_name'])
            post_llm_memory = memory_manager.check_memory()
            print(f"After LLM: {post_llm_memory}")

            del test_mentor
            memory_manager.emergency_cleanup()
            final_memory = memory_manager.check_memory()
            print(f"After cleanup: {final_memory}")

        except Exception as e:
            print(f"Memory test failed: {e}")

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
            pipeline.logger.log(f"LLM Model: {LLM_MENTOR_CONFIG['model_name']}")

        # Load checkpoint if specified
        if args.load_checkpoint:
            # Implementation similar to original
            pass

        if args.eval_only:
            pipeline._final_evaluation()
        else:
            pipeline.train()

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        print("🔧 Set CUDA memory allocation strategy for efficiency")

    main()