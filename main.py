# enhanced_main.py (main.py)
"""
Enhanced Revolutionary AI Pipeline with Activation-Based Distillation
Integrates human behavior cloning with critical pathway analysis
"""

import torch
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Any, Optional
import argparse
import os

import torch.nn.functional as F

from config import *
from environment import create_environment, AdvancedRewardShaper
from mentor import MultimodalMentor
from student import StudentAgent
from distillation import DistillationTrainer
from memory import PrioritizedReplayBuffer, TrajectoryBuffer, ExperienceCollector, compute_gae
from utils import Logger, CurriculumScheduler, ActionProcessor, analyze_mentor_student_agreement
from torch.distributions import Categorical

# Import from mathematical_framework (MODIFIED)
from mathematical_framework import PathwayImportanceOptimizer, \
    InformationTheoreticAnalyzer # Removed CriticalPathwayAnalyzer here

from activation_distillation import (
    HumanDemonstrationCollector, ActivationTracker, CriticalPathwayAnalyzer as MathCriticalPathwayAnalyzer, # ADDED CriticalPathwayAnalyzer and alias here
    ActivationSignatureExtractor, FocusedDistillationLoss, create_activation_based_distillation_pipeline
)


class EnhancedRevolutionaryPipeline:
    """
    Enhanced training pipeline with activation-based knowledge distillation
    """

    def __init__(self, args):
        self.args = args

        # Setup logging
        self.logger = Logger(args.log_dir)
        self.logger.log("Initializing Enhanced Revolutionary AI Pipeline with Activation-Based Distillation...")

        # Create environment
        self.env = create_environment()
        self.reward_shaper = AdvancedRewardShaper(ENV_CONFIG['name'])

        # Initialize models
        self.mentor = MultimodalMentor(
            state_dim=self.env.state_dim,
            num_actions=self.env.num_actions
        )

        self.student = StudentAgent(
            state_dim=self.env.state_dim,
            num_actions=self.env.num_actions
        )

        # Setup optimizers
        self.student.optimizer = optim.Adam(
            self.student.parameters(),
            lr=STUDENT_CONFIG['learning_rate']
        )

        self.mentor.optimizer = optim.Adam(
            self.mentor.parameters(),
            lr=MENTOR_CONFIG['learning_rate']
        )

        # Initialize mathematical components for advanced analysis/optimization
        self.math_pathway_optimizer = PathwayImportanceOptimizer()
        # Use the pathway analyzer from activation_distillation for consistency in graph building
        # self.math_pathway_analyzer = MathCriticalPathwayAnalyzer({}) # Or use this if different logic needed

        # Initialize activation-based distillation pipeline, passing the optimizer
        # The create_activation_based_distillation_pipeline will instantiate its own pathway_analyzer
        # or we can pass one if we want a single shared instance.
        # Let's ensure pathway_analyzer used by signature_extractor is the one we manage here.
        self.pathway_analyzer_shared = MathCriticalPathwayAnalyzer({})  # Instance to be shared

        self.activation_pipeline = create_activation_based_distillation_pipeline(
            pathway_optimizer=self.math_pathway_optimizer,
            pathway_analyzer_instance=self.pathway_analyzer_shared  # Pass our instance
        )

        # Setup activation tracking
        self.mentor_tracker = ActivationTracker(self.mentor)
        self.student_tracker = ActivationTracker(self.student)

        # Enhanced distillation trainer with activation focus
        self.distillation_trainer = DistillationTrainer(
            mentor=self.mentor,
            student=self.student,
            mentor_hidden_dim=MENTOR_CONFIG['hidden_dim'],
            student_hidden_dim=STUDENT_CONFIG['hidden_dim']
        )

        # Replace standard distillation loss with focused version
        self.focused_distillation_loss = self.activation_pipeline['distillation_loss']

        # Memory components
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=MEMORY_CONFIG['trajectory_buffer_size'],
            alpha=MEMORY_CONFIG['priority_alpha'],
            beta=MEMORY_CONFIG['priority_beta']
        )

        self.trajectory_buffer = TrajectoryBuffer(
            max_trajectories=MEMORY_CONFIG['trajectory_buffer_size'] // 100
        )

        self.experience_collector = ExperienceCollector(ENV_CONFIG['num_envs'])

        # Training components
        self.curriculum = CurriculumScheduler(CURRICULUM_CONFIG['stages'])
        self.action_processor = ActionProcessor(self.env.num_actions)

        # Activation-based components
        self.critical_signatures = []
        # Use the shared pathway_analyzer for graph building and pathway identification
        self.pathway_analyzer = self.pathway_analyzer_shared  # Formerly: self.activation_pipeline['pathway_analyzer']
        self.signature_extractor = self.activation_pipeline['signature_extractor']

        # Training phases
        self.training_phase = 'human_cloning'  # 'human_cloning', 'focused_distillation', 'standard_training'
        self.phase_transitions = {
            'human_cloning': 5000,  # First 5k steps: learn from human demos
            'focused_distillation': 20000,  # Next 15k steps: focused distillation
            'standard_training': float('inf')  # Remaining: standard RL
        }

        # Tracking
        self.timestep = 0
        self.episode_count = 0
        self.best_reward = float('-inf')

        self.logger.log("Enhanced pipeline initialization complete!")

    def train(self):
        """Enhanced training loop with three phases"""
        self.logger.log("Starting enhanced training with activation-based distillation...")

        # Phase 1: Human Behavior Cloning
        if self.args.human_demos_path and not self.args.skip_behavior_cloning:
            self.logger.log("Phase 1: Human Behavior Cloning and Critical Pathway Analysis")
            self._phase1_human_behavior_cloning()
        elif not self.args.skip_behavior_cloning:
            self.logger.log("No human demonstrations path provided, generating synthetic demos for Phase 1...")
            self._generate_synthetic_demonstrations()
        else:
            self.logger.log("Skipping Phase 1 (Human Behavior Cloning / Synthetic Demos).")

        # Phase 2: Focused Distillation Training
        self.logger.log("Phase 2: Focused Distillation Training (will start after human_cloning phase timesteps)")
        # self.training_phase = 'focused_distillation' # This is handled by _update_training_phase

        # Phase 3: Standard RL Training
        self.logger.log("Phase 3: Standard RL Training (will start after focused_distillation phase timesteps)")

        # Initial environment reset
        observations, _ = self.env.reset(seed=SEED)
        # states = self.env.get_state_tensor(observations) # This was a bug, get_state_tensor expects numpy array

        start_time = time.time()

        while self.timestep < TRAINING_CONFIG['total_timesteps']:

            # Update training phase
            self._update_training_phase()

            # Convert observations to tensor after potential reset or step
            states_tensor = self.env.get_state_tensor(observations)

            # === ROLLOUT PHASE ===
            rollout_data = self._collect_rollout(states_tensor)  # Pass tensor here
            # Update observations for the next iteration based on the last state in the rollout
            # Assuming _collect_rollout updates self.experience_collector and the last next_state is needed
            if self.experience_collector.next_states:
                observations = self.experience_collector.next_states[-1]  # This gets the numpy array for next obs
            else:  # If no rollouts yet (e.g. very start), keep current observations
                pass

            # === TRAINING PHASE ===
            if self.timestep > TRAINING_CONFIG['batch_size']:  # Ensure enough samples collected
                if self.training_phase == 'focused_distillation' and self.critical_signatures:
                    training_metrics = self._train_models_focused(rollout_data)
                else:  # Also handles human_cloning phase if we want PPO updates, or standard_training
                    training_metrics = self._train_models_standard(rollout_data)

                if training_metrics:  # Log if metrics were produced
                    self._log_training_metrics(training_metrics)

            # === EVALUATION PHASE ===
            if self.timestep % LOGGING_CONFIG['eval_interval'] == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation_metrics(eval_metrics)

            # === CURRICULUM UPDATE ===
            if CURRICULUM_CONFIG['enabled']:
                avg_reward = self.trajectory_buffer.get_statistics().get('avg_reward', 0)
                self.curriculum.get_current_config(avg_reward)

            # === CHECKPOINTING ===
            if self.timestep % LOGGING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()

            # === LOGGING ===
            if self.timestep % LOGGING_CONFIG['log_interval'] == 0:
                self._log_progress(time.time() - start_time)

        self.logger.log("Enhanced training completed!")
        self._final_evaluation()

    def _phase1_human_behavior_cloning(self):
        """Phase 1: Train mentor on human demonstrations and extract critical pathways"""
        self.logger.log("Starting human behavior cloning phase...")

        # Load human demonstrations
        demo_collector = self.activation_pipeline['demonstration_collector']

        if self.args.human_demos_path and os.path.exists(self.args.human_demos_path):
            raw_demonstrations = self._load_human_demonstrations(self.args.human_demos_path)
            # Use the collector to process/store them if needed, or use directly
            for demo_data in raw_demonstrations:
                demo_collector.collect_demonstration(
                    states=demo_data['states'],  # List of np.ndarray
                    actions=demo_data['actions'],  # List of int
                    performance_score=demo_data['performance_score']
                )
            demonstrations_for_bc = demo_collector.get_successful_demonstrations()

        else:
            self.logger.log("No human demonstrations found or path invalid, skipping actual BC training in Phase 1.")
            # Fallback: generate synthetic ones to allow pathway analysis logic to run if needed
            self._generate_synthetic_demonstrations()  # This will populate demo_collector
            demonstrations_for_bc = demo_collector.demonstrations  # Use all synthetic ones

        if not demonstrations_for_bc:
            self.logger.log(
                "No demonstrations available for behavior cloning after loading/synthesis. Skipping pathway analysis.")
            return

        # Train mentor on demonstrations with activation tracking
        # Store sequences of (activations_dict_per_step, performance_score_of_demo)
        activations_sequence_for_graph = []
        importance_scores_sequence_for_graph = []
        performance_scores_for_signatures = []  # List of floats

        for epoch in range(50):  # Behavior cloning epochs
            epoch_total_bc_loss = 0
            num_bc_batches = 0

            for demo_idx, demo in enumerate(demonstrations_for_bc):
                demo_states_np = demo['states']  # List of np.ndarrays
                demo_actions_np = demo['actions']  # List of ints
                demo_performance = demo['performance_score']

                # Process demo step-by-step or as a batch if model supports it
                # For simplicity, let's assume we process each step to get activations
                # This part needs to be robust to how demos are structured and fed.

                # Convert entire demo to tensors
                states_tensor = torch.tensor(np.array(demo_states_np), dtype=torch.float32).to(DEVICE)
                actions_tensor = torch.tensor(np.array(demo_actions_np), dtype=torch.long).to(DEVICE)

                if states_tensor.ndim == 1: states_tensor = states_tensor.unsqueeze(0)  # if single step demo
                if actions_tensor.ndim == 0: actions_tensor = actions_tensor.unsqueeze(0)

                # --- Behavior Cloning Update ---
                self.mentor.train()
                self.mentor_tracker.clear_cache()
                mentor_outputs_bc = self.mentor(states_tensor)  # Get outputs for all steps in demo

                bc_loss = F.cross_entropy(mentor_outputs_bc['policy_logits'], actions_tensor)
                epoch_total_bc_loss += bc_loss.item()
                num_bc_batches += 1

                self.mentor.optimizer.zero_grad()
                bc_loss.backward(retain_graph=True)  # Retain graph for importance computation if needed by analyzer

                # --- Activation and Importance Collection ---
                # Get activations for each step in the demo
                # The current ActivationTracker captures the *last* forward pass's activations.
                # To get per-step activations for a sequence, we'd need to call mentor() per step.
                # For now, let's simplify: use the activations from the batch forward pass of the demo.
                # This means 'activations_for_graph_step' will be a dict of tensors (Batch=DemoLength, Features)

                activations_for_graph_step = self.mentor_tracker.get_activations()  # (Batch=DemoLength, Features) per layer

                # Compute importance (e.g. gradient-based) using the demo's performance score
                # The target_performance for compute_activation_importance expects a float or scalar tensor.
                importance_for_graph_step = self.pathway_analyzer.compute_activation_importance(
                    activations_for_graph_step,  # Dict: layer -> (DemoLength, Features)
                    demo_performance,  # float
                    method='gradient_based'
                )

                self.mentor.optimizer.step()  # Step after getting grads for importance

                # For graph building, we need List[Dict[str, Tensor]] where each Dict is one timestep.
                # Current activations_for_graph_step is {layer: (DemoSteps, Feats)}. We need to unroll this.
                num_steps_in_demo = states_tensor.shape[0]
                for step_idx in range(num_steps_in_demo):
                    step_activations = {layer: act_tensor[step_idx].unsqueeze(0) for layer, act_tensor in
                                        activations_for_graph_step.items() if
                                        act_tensor.numel() > 0}  # Make it (1,Feats)
                    step_importance = {layer: imp_tensor[step_idx].unsqueeze(0) for layer, imp_tensor in
                                       importance_for_graph_step.items() if imp_tensor.numel() > 0}

                    if step_activations:  # only if there are valid activations
                        activations_sequence_for_graph.append(step_activations)
                        importance_scores_sequence_for_graph.append(step_importance)
                        performance_scores_for_signatures.append(demo_performance)  # Corresponding performance

            avg_bc_loss_epoch = epoch_total_bc_loss / num_bc_batches if num_bc_batches > 0 else 0
            if epoch % 10 == 0:
                self.logger.log(f"Behavior cloning epoch {epoch}, avg loss: {avg_bc_loss_epoch:.4f}")

        self.mentor.eval()  # Set mentor to eval mode after BC training

        if not activations_sequence_for_graph or not importance_scores_sequence_for_graph:
            self.logger.log("No valid activation/importance sequences collected from BC. Skipping pathway analysis.")
            return

        # Build activation graph and identify critical pathways
        self.logger.log("Analyzing critical activation pathways from mentor BC...")
        # Use the shared self.pathway_analyzer
        activation_graph = self.pathway_analyzer.build_activation_graph(
            activations_sequence_for_graph, importance_scores_sequence_for_graph
        )

        if activation_graph.number_of_nodes() > 0:
            critical_pathways_nodesets = self.pathway_analyzer.identify_critical_pathways(
                activation_graph, method='spectral_clustering'  # Or other methods
            )
        else:
            self.logger.log("Activation graph has no nodes. Skipping pathway identification.")
            critical_pathways_nodesets = []

        if not critical_pathways_nodesets:
            self.logger.log("No critical pathways identified. Focused distillation may be impacted.")
            self.critical_signatures = []
            return

        # Extract activation signatures using the more principled PathwayImportanceOptimizer
        # The signature_extractor now needs the optimizer and the analyzer.
        # It also needs performance_scores_sequence for the optimizer.
        self.logger.log(f"Extracting {len(critical_pathways_nodesets)} critical pathway signatures...")
        self.critical_signatures = self.signature_extractor.extract_signatures(
            critical_pathways_nodesets,  # List[Set[str]]
            activations_sequence_for_graph,  # List[Dict[str, Tensor (1,Feats)]]
            performance_scores_sequence=performance_scores_for_signatures,  # List[float]
            target_dim=MENTOR_CONFIG.get('num_knowledge_tokens', 64)  # Or specific dim from config
        )

        if self.critical_signatures:
            self.logger.log(f"Identified and extracted {len(self.critical_signatures)} critical activation signatures.")
            avg_importance = np.mean(
                [sig.importance_score for sig in self.critical_signatures if sig]) if self.critical_signatures else 0
            self.logger.log(f"Average signature importance (from optimizer): {avg_importance:.4f}")
        else:
            self.logger.log("No critical signatures were extracted.")

    def _generate_synthetic_demonstrations(self):
        """Generate synthetic demonstrations using random policy (fallback)"""
        self.logger.log("Generating synthetic demonstrations (random policy fallback)...")
        demo_collector = self.activation_pipeline['demonstration_collector']
        demo_collector.demonstrations = []  # Clear any previous

        num_synthetic_demos = 10
        for episode in range(num_synthetic_demos):
            obs_list_np, _ = self.env.reset()  # obs_list_np is (num_envs, state_dim)

            # For synthetic demos, let's use the first env for simplicity if num_envs > 1
            current_obs_np = obs_list_np[0] if self.env.num_envs > 1 else obs_list_np

            states_buffer, actions_buffer = [], []
            episode_reward_sum = 0
            max_steps_per_demo = 200

            for step in range(max_steps_per_demo):
                # Use a single random action for the first environment
                action_for_env0 = np.random.randint(0, self.env.num_actions)

                # Prepare actions list for potentially vectorized env.step
                # If num_envs > 1, other envs also need actions. Let them be random too.
                actions_for_all_envs = [[np.random.randint(0, self.env.num_actions)] for _ in range(self.env.num_envs)]
                actions_for_all_envs[0] = [action_for_env0]  # Set action for our target env

                # Dummy uncertainties
                uncertainties_for_all_envs = [0.5] * self.env.num_envs

                next_obs_list_np, rewards_np, terminated_np, truncated_np, _ = self.env.step(
                    actions_for_all_envs, uncertainties_for_all_envs
                )

                states_buffer.append(current_obs_np.copy())
                actions_buffer.append(action_for_env0)

                reward_env0 = rewards_np[0]
                episode_reward_sum += reward_env0

                current_obs_np = next_obs_list_np[0] if self.env.num_envs > 1 else next_obs_list_np

                if terminated_np[0] or truncated_np[0]:
                    break

            # Normalize performance score (example normalization)
            performance_score = min(1.0, max(0.0, episode_reward_sum / (
                max_steps_per_demo * 0.1 if ENV_CONFIG['name'] == 'CartPole-v1' else 100.0)))
            if states_buffer:  # Only collect if some steps were taken
                demo_collector.collect_demonstration(states_buffer, actions_buffer, performance_score)

        self.logger.log(f"Generated {len(demo_collector.demonstrations)} synthetic demonstrations.")
        # After generating synthetic demos, call phase1 again to process them for pathways
        # This creates a recursive call risk if _phase1_human_behavior_cloning itself calls this.
        # Instead, _phase1_human_behavior_cloning should use the collected demos.
        # The current logic is: if no path, call this, then _phase1 uses demo_collector.demonstrations. This is okay.

    def _update_training_phase(self):
        """Update training phase based on timestep"""
        if self.timestep < self.phase_transitions['human_cloning']:
            self.training_phase = 'human_cloning'  # Or a 'post_bc_pre_focus' phase
        elif self.timestep < self.phase_transitions['focused_distillation']:
            self.training_phase = 'focused_distillation'
        else:
            self.training_phase = 'standard_training'

    def _train_models_focused(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Training with focused distillation using critical pathways"""
        batch_data = rollout_data.get('batch_data')

        if not batch_data or not self.critical_signatures:  # Skip if no data or no signatures
            # self.logger.log("No batch data or no critical signatures for focused training, falling back to standard.")
            return self._train_models_standard(rollout_data)  # Fallback

        # Ensure all required keys are in batch_data
        required_keys = ['rewards', 'values', 'dones', 'states', 'actions', 'log_probs']
        if not all(key in batch_data and batch_data[key].numel() > 0 for key in required_keys):
            # self.logger.log("Batch data incomplete for focused training.")
            return {}

        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards=batch_data['rewards'],
            values=batch_data['values'],
            dones=batch_data['dones'],
            gamma=TRAINING_CONFIG['gamma'],
            gae_lambda=TRAINING_CONFIG['gae_lambda']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        training_metrics = {}
        self.student.train()
        self.mentor.eval()  # Mentor is usually in eval mode during distillation

        # Enhanced student training with focused distillation
        for epoch in range(TRAINING_CONFIG['num_ppo_epochs']):
            # Create mini-batches
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            if num_samples < batch_size:  # Not enough samples for a full batch
                # print(f"Warning: Not enough samples ({num_samples}) for a full batch ({batch_size}) in focused training.")
                # Potentially skip epoch or use smaller batch if robustly handled by loss
                if num_samples == 0: continue  # Skip epoch if no samples
                current_batch_size = num_samples
            else:
                current_batch_size = batch_size

            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, current_batch_size):
                end = min(start + current_batch_size, num_samples)
                if start == end: continue  # Avoid empty batch
                batch_indices = indices[start:end]

                # Extract mini-batch
                mini_batch_states = batch_data['states'][batch_indices]
                mini_batch_actions_rl = batch_data['actions'][batch_indices]
                mini_batch_returns_rl = returns[batch_indices]
                mini_batch_advantages_rl = advantages[batch_indices]
                mini_batch_old_log_probs_rl = batch_data['log_probs'][batch_indices]
                mini_batch_old_values_rl = batch_data['values'][batch_indices]  # For PPO value clipping if used

                # Get outputs with activation tracking
                self.mentor_tracker.clear_cache()
                self.student_tracker.clear_cache()

                with torch.no_grad():  # Mentor forward pass should not require grads here
                    mentor_outputs_distill = self.mentor(mini_batch_states)

                student_outputs_distill_and_rl = self.student(mini_batch_states)

                mentor_activations_distill = self.mentor_tracker.get_activations()  # From mentor pass
                student_activations_distill_and_rl = self.student_tracker.get_activations()  # From student pass

                # Compute focused distillation loss
                # Ensure critical_signatures are valid
                valid_signatures = [cs for cs in self.critical_signatures if cs is not None]
                if not valid_signatures:
                    # print("Warning: No valid critical signatures for focused loss computation.")
                    focused_losses = {'total_focused': torch.tensor(0.0, device=DEVICE)}  # Provide a zero tensor
                else:
                    focused_losses = self.focused_distillation_loss(
                        student_outputs_distill_and_rl, mentor_outputs_distill,
                        student_activations_distill_and_rl, mentor_activations_distill,
                        valid_signatures
                    )

                # Add standard RL losses (PPO)
                rl_losses = self._compute_rl_losses(
                    student_outputs_distill_and_rl,  # Contains primary_logits and value
                    actions=mini_batch_actions_rl,
                    returns=mini_batch_returns_rl,
                    advantages=mini_batch_advantages_rl,
                    old_log_probs=mini_batch_old_log_probs_rl
                    # old_values are not directly used by this _compute_rl_losses version, but good to have if PPO value clipping is added
                )

                # Combined loss
                # Weights for combining distillation and RL losses
                distill_weight = 0.6  # Example
                rl_weight = 0.4  # Example

                total_loss = (
                        distill_weight * focused_losses.get('total_focused', torch.tensor(0.0, device=DEVICE)) +
                        rl_weight * rl_losses.get('total_rl', torch.tensor(0.0, device=DEVICE))
                )

                # Optimize student
                self.student.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), TRAINING_CONFIG['max_grad_norm'])
                self.student.optimizer.step()

                # Collect metrics (ensure keys exist)
                for loss_dict, prefix in [(focused_losses, "focused"), (rl_losses, "rl")]:
                    for key, value in loss_dict.items():
                        metric_key = f"{prefix}_{key}"
                        if metric_key not in training_metrics: training_metrics[metric_key] = []
                        training_metrics[metric_key].append(value.item())

                if 'total_loss' not in training_metrics: training_metrics['total_loss'] = []
                training_metrics['total_loss'].append(total_loss.item())

        # Average metrics
        for key in training_metrics:
            if training_metrics[key]:  # Check if list is not empty
                training_metrics[key] = np.mean(training_metrics[key])
            else:
                training_metrics[key] = 0  # Or some other indicator for no data

        return training_metrics

    def _train_models_standard(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Standard training without focused distillation (PPO + standard KD)"""
        batch_data = rollout_data.get('batch_data')
        if not batch_data:
            # self.logger.log("No batch data for standard training.")
            return {}

        required_keys = ['rewards', 'values', 'dones', 'states', 'actions', 'log_probs']
        if not all(key in batch_data and batch_data[key].numel() > 0 for key in required_keys):
            # self.logger.log("Batch data incomplete for standard training.")
            return {}

        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards=batch_data['rewards'],
            values=batch_data['values'],
            dones=batch_data['dones'],
            gamma=TRAINING_CONFIG['gamma'],
            gae_lambda=TRAINING_CONFIG['gae_lambda']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        training_metrics = {}
        self.student.train()

        # Standard PPO training + standard distillation (via DistillationTrainer)
        for epoch in range(TRAINING_CONFIG['num_ppo_epochs']):
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            if num_samples < batch_size:
                # print(f"Warning: Not enough samples ({num_samples}) for a full batch ({batch_size}) in standard training.")
                if num_samples == 0: continue
                current_batch_size = num_samples
            else:
                current_batch_size = batch_size

            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, current_batch_size):
                end = min(start + current_batch_size, num_samples)
                if start == end: continue
                batch_indices = indices[start:end]

                # Extract mini-batch for DistillationTrainer's train_step
                # This requires states, actions, rewards (which are returns here), advantages, old_log_probs, old_values
                metrics_from_distill_trainer = self.distillation_trainer.train_step(
                    states=batch_data['states'][batch_indices],
                    actions=batch_data['actions'][batch_indices],
                    rewards=returns[batch_indices],  # PPO uses returns (target for value func)
                    advantages=advantages[batch_indices],
                    old_log_probs=batch_data['log_probs'][batch_indices],
                    values=batch_data['values'][batch_indices]
                    # mentor_advice is optional and not directly used by train_step here, but by compute_distillation_loss if mentor_outputs not given
                )

                # Accumulate metrics from DistillationTrainer
                for key, value in metrics_from_distill_trainer.items():
                    if key not in training_metrics: training_metrics[key] = []
                    training_metrics[key].append(value)  # Value is already float from trainer

        # Average metrics
        for key in training_metrics:
            if training_metrics[key]:
                training_metrics[key] = np.mean(training_metrics[key])
            else:
                training_metrics[key] = 0

        return training_metrics

    def _compute_rl_losses(self, student_outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           # `rewards` in PPO context are often `returns` for value loss
                           advantages: torch.Tensor, old_log_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute standard PPO losses. `student_outputs` must contain 'primary_logits' and 'value'."""

        # Policy loss (CLIP)
        dist = Categorical(logits=student_outputs['primary_logits'])
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - TRAINING_CONFIG['clip_ratio'], 1 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        current_values = student_outputs['value'].squeeze(-1)  # Ensure it's [B]
        value_loss = F.mse_loss(current_values, returns)  # Target for value function is empirical returns

        # Entropy bonus
        entropy_bonus = dist.entropy().mean()

        total_rl_loss = (
                policy_loss +
                value_loss * TRAINING_CONFIG['value_coef'] -
                entropy_bonus * STUDENT_CONFIG['entropy_coef']
        )

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,  # Coefficient applied when summing
            'entropy_bonus': entropy_bonus,  # Coefficient applied when summing
            'total_rl': total_rl_loss,
        }

    def _load_human_demonstrations(self, demo_path: str) -> List[Dict]:
        """Load human demonstrations from file"""
        demonstrations = []
        try:
            import pickle
            with open(demo_path, 'rb') as f:
                # Assuming demo_data is a list of dicts,
                # each dict having 'states', 'actions', 'performance_score'
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, list):
                    for demo_item in loaded_data:
                        if isinstance(demo_item, dict) and \
                                'states' in demo_item and \
                                'actions' in demo_item and \
                                'performance_score' in demo_item:
                            demonstrations.append({
                                'states': np.array(demo_item['states']),  # Ensure numpy array
                                'actions': np.array(demo_item['actions']),  # Ensure numpy array
                                'performance_score': float(demo_item['performance_score']),
                                'length': len(demo_item['states'])
                            })
                        else:
                            self.logger.log(f"Skipping invalid demo item: {type(demo_item)}", level="WARN")

            self.logger.log(f"Loaded {len(demonstrations)} human demonstrations from {demo_path}")
        except FileNotFoundError:
            self.logger.log(f"Demonstration file not found: {demo_path}", level="ERROR")
        except Exception as e:
            self.logger.log(f"Error loading demonstrations from {demo_path}: {e}", level="ERROR")
        return demonstrations

    def _collect_rollout(self, initial_states_tensor: torch.Tensor) -> Dict[str, Any]:
        """Enhanced rollout collection"""
        self.experience_collector.reset()
        current_states_tensor = initial_states_tensor.clone()  # current_states_tensor is (num_envs, state_dim)

        rollout_episode_rewards = []  # Rewards for episodes completed *during this rollout*
        num_mentor_queries_this_rollout = 0

        for step_in_rollout in range(TRAINING_CONFIG['rollout_steps']):
            # === STUDENT ACTION SELECTION ===
            # Student.act expects (Batch=NumEnvs, StateDim)
            student_actions_list_of_lists, student_info_list_of_dicts = [], []

            # Assuming student.act is designed for single state input and we call it per env for now
            # Or, if student.act can handle batch, use current_states_tensor directly.
            # Let's assume student.act handles a batch of states (num_envs, state_dim)
            # and returns actions as List[List[int]] and info as List[Dict]

            # Check if student.act can handle batch, current implementation might be for single state
            # For now, let's assume it's modified or we iterate (less efficient)
            # The student.act in student.py seems to process a batch and extracts item() for single outputs.
            # It needs to be made more robust for batch operations if num_envs > 1.
            # Let's proceed assuming student.act handles current_states_tensor (Batch=N_envs, Feats)
            # and returns actions: List[List[int]] and info: List[Dict] (one per env)

            # Temporary fix for student.act if it doesn't fully support batch output for info part:
            all_env_actions = []  # This will store List[int] (actions for one env) for each environment
            all_env_info = []  # This will store Dict (info for one env) for each environment

            if self.env.num_envs > 1:
                for i in range(self.env.num_envs):
                    single_state_tensor = current_states_tensor[i].unsqueeze(0)  # Shape: (1, state_dim)

                    # student.act processes a batch. single_state_tensor is batch_size=1.
                    # actions_batch_for_one_env will be List[List[int]] of length 1, e.g., [[act1, act2,...]]
                    # info_batch_for_one_env will be List[Dict] of length 1, e.g., [{info_dict_for_this_env}]
                    actions_batch_for_one_env, info_batch_for_one_env = self.student.act(single_state_tensor)

                    # Correctly extract the single list of actions and the single info dictionary
                    all_env_actions.append(actions_batch_for_one_env[0])  # Appends List[int]
                    all_env_info.append(info_batch_for_one_env[0])  # Appends Dict
            else:  # Single environment
                # current_states_tensor is (1, state_dim)
                # actions_batch will be List[List[int]] of length 1, e.g., [[[act1, act2]]]
                # info_batch will be List[Dict] of length 1, e.g., [[{info_dict}]]
                actions_batch, info_batch = self.student.act(current_states_tensor)
                all_env_actions.append(actions_batch[0])  # Appends List[int] (actions for the single env)
                all_env_info.append(info_batch[0])

            student_selected_actions = all_env_actions  # Now correctly List[List[int]]
            student_runtime_info = all_env_info  # Now correctly List[Dict]

            # === MENTOR QUERYING ===
            # This part should now work correctly as student_runtime_info[i] will be a Dict
            mentor_advice_list = [None] * self.env.num_envs
            for i in range(self.env.num_envs):
                if student_runtime_info[i].get('should_query_mentor', False):  # This line was causing the error
                    num_mentor_queries_this_rollout += 1
                    mentor_advice_list[i] = self._query_mentor(current_states_tensor[i].unsqueeze(0))

            # === ENVIRONMENT STEP ===
            # Ensure student_selected_actions is List[List[int]] for env.step
            # uncertainties: List[float], one per environment
            uncertainties_for_step = [info.get('uncertainty', {}).get('total', 0.5) for info in student_runtime_info]

            next_obs_np, rewards_np, terminated_np, truncated_np, infos_from_env = self.env.step(
                student_selected_actions, uncertainties_for_step
            )  # these are all np.arrays of shape (num_envs, ...) or List[Dict] for infos_from_env

            # Apply reward shaping
            shaped_rewards_np = np.zeros_like(rewards_np)
            for i in range(self.env.num_envs):
                # reward_shaper.shape_reward expects single obs, action, reward, etc.
                # Use primary action from student_selected_actions[i][0] for shaping.
                primary_action_env_i = student_selected_actions[i][0] if student_selected_actions[i] else 0  # fallback

                # Handle obs for shaping: next_obs_np[i] is the state *after* action
                # For potential-based shaping, often the state *before* action and after action are used.
                # The current shaper might use the resulting state (next_obs_np[i]).
                shaped_rewards_np[i] = self.reward_shaper.shape_reward(
                    next_obs_np[i], primary_action_env_i, rewards_np[i],
                    (terminated_np[i] or truncated_np[i]), infos_from_env[i]
                )

            # === EXPERIENCE COLLECTION ===
            # ExperienceCollector.add expects batch tensors or np arrays.
            # current_states_tensor is (N_envs, StateDim)
            # student_selected_actions is List[List[int]], needs to be np.array (N_envs, NumStudentActions) or (N_envs, 1) if primary
            # For log_probs and values, need to run student model again on current_states_tensor
            with torch.no_grad():
                student_outputs_for_collection = self.student(current_states_tensor)

            primary_actions_for_logprob = torch.tensor([sel_acts[0] for sel_acts in student_selected_actions],
                                                       dtype=torch.long, device=DEVICE)



            dist_for_logprob = Categorical(
                logits=student_outputs_for_collection['primary_logits'])  # This is the failing line
            log_probs_for_collection = dist_for_logprob.log_prob(primary_actions_for_logprob)

            values_for_collection = student_outputs_for_collection['value'].squeeze(-1)  # Ensure (N_envs,)

            # Convert uncertainties to consistent format if student_runtime_info is List[Dict]
            uncertainty_values_for_collection = [info.get('uncertainty', {}) for info in student_runtime_info]

            self.experience_collector.add(
                state=current_states_tensor.cpu().numpy(),  # (N_envs, StateDim)
                action=np.array([sel_acts[0] for sel_acts in student_selected_actions]),  # (N_envs,), primary actions
                reward=shaped_rewards_np,  # (N_envs,)
                next_state=next_obs_np,  # (N_envs, StateDim)
                done=(terminated_np | truncated_np),  # (N_envs,)
                log_prob=log_probs_for_collection,  # (N_envs,)
                value=values_for_collection,  # (N_envs,)
                uncertainty=uncertainty_values_for_collection,  # List[Dict] or Dict
                mentor_advice=mentor_advice_list  # List[MentorAdvice or None]
            )

            # === STATE UPDATE ===
            current_states_tensor = self.env.get_state_tensor(next_obs_np)  # For next iteration

            # Track episode completions from infos_from_env
            for i in range(self.env.num_envs):
                if infos_from_env[i].get('episode_reward') is not None:
                    rollout_episode_rewards.append(infos_from_env[i]['episode_reward'])
                    self.episode_count += 1

            self.timestep += self.env.num_envs

        return {
            'rollout_rewards': rollout_episode_rewards,
            'mentor_queries': num_mentor_queries_this_rollout,
            'batch_data': self.experience_collector.get_batch_tensors()  # This converts collected lists to tensors
        }

    def _collect_experience_step(self, states: torch.Tensor, actions: List[List[int]],
                                 rewards: np.ndarray, next_states_np: np.ndarray,  # Renamed for clarity
                                 terminated: np.ndarray, truncated: np.ndarray,
                                 student_info_list: List[Dict],  # Renamed, assumed List[Dict]
                                 mentor_advice_list: List[Optional[Any]]):  # Renamed, assumed List
        """Collects experiences for each environment within a single step of the rollout."""
        # This method is called from _collect_rollout, which now does most of the work.
        # The ExperienceCollector.add() is called directly in _collect_rollout.
        # This method can be simplified or removed if _collect_rollout handles all logic.
        # For now, let's assume _collect_rollout handles it, so this method is mostly a pass-through
        # or can be refactored into _collect_rollout.

        # The main logic has been moved to _collect_rollout's ExperienceCollector.add call.
        # This method is kept for structural similarity but its direct utility is reduced.

        # Check for completed episodes and add to trajectory buffer
        # This part is still relevant if ExperienceCollector.add doesn't automatically do it.
        # The current ExperienceCollector.get_completed_episodes is called *after* adding.
        completed_episodes_data = self.experience_collector.get_completed_episodes()
        for episode_experiences, episode_reward_value in completed_episodes_data:
            self.trajectory_buffer.add_trajectory(episode_experiences, episode_reward_value)
            if episode_reward_value > self.best_reward:
                self.best_reward = episode_reward_value
                # self.logger.log(f"New best reward from completed episode: {episode_reward_value:.2f}")

    def _query_mentor(self, state_tensor_single_env: torch.Tensor) -> Any:  # state is (1, StateDim)
        """Query mentor for advice for a single environment's state."""
        with torch.no_grad():
            self.mentor.eval()  # Ensure mentor is in eval mode
            advice = self.mentor.get_advice(state_tensor_single_env)  # get_advice expects (1,StateDim) or (StateDim,)
            return advice

    def _evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation"""
        self.logger.log("Running evaluation...")
        self.student.eval()  # Set student to eval mode

        # Create a separate eval environment (typically single, non-vectorized for clarity)
        eval_env_config_name = ENV_CONFIG.get('name', 'CartPole-v1')
        eval_env = create_environment(env_name=eval_env_config_name, num_envs=1)

        eval_rewards_all_eps = []
        eval_lengths_all_eps = []
        eval_mentor_queries_all_eps = []
        # pathway_activations_all_eps = [] # This might be too much data, consider sampling

        num_eval_episodes = 10
        for episode_num in range(num_eval_episodes):
            obs_np, _ = eval_env.reset()  # obs_np is (state_dim,) for single env

            episode_reward = 0
            episode_length = 0
            episode_mentor_queries = 0
            # episode_pathway_activations = []

            done = False
            truncated_episode = False
            while not (done or truncated_episode):
                state_tensor_eval = eval_env.get_state_tensor(obs_np)  # Converts to (1, state_dim)

                # Student action (deterministic for eval)
                actions_list_eval, info_eval = self.student.act(state_tensor_eval, deterministic=True)
                action_to_take_eval = actions_list_eval[0]  # Primary action for single env step

                if info_eval.get('should_query_mentor', False):
                    episode_mentor_queries += 1

                # Pathway activation tracking (optional, can be heavy)
                # if self.training_phase == 'focused_distillation' and self.critical_signatures:
                #     self.student_tracker.clear_cache()
                #     _ = self.student(state_tensor_eval) # Forward pass
                #     current_activations = self.student_tracker.get_activations()
                #     episode_pathway_activations.append(current_activations)

                # Environment step with the chosen primary action
                # Eval env expects List[List[int]] for actions
                next_obs_np, reward_eval, done, truncated_episode, _ = eval_env.step(
                    [[action_to_take_eval]],
                    [info_eval.get('uncertainty', {}).get('total', 0.5)]
                )
                obs_np = next_obs_np  # Update obs for next step

                episode_reward += reward_eval[0]  # reward_eval is np.array([float])
                episode_length += 1

            eval_rewards_all_eps.append(episode_reward)
            eval_lengths_all_eps.append(episode_length)
            eval_mentor_queries_all_eps.append(episode_mentor_queries)
            # if episode_pathway_activations: pathway_activations_all_eps.append(episode_pathway_activations)

        eval_env.close()
        self.student.train()  # Set student back to train mode

        eval_metrics = {
            'eval_reward_mean': np.mean(eval_rewards_all_eps) if eval_rewards_all_eps else 0.0,
            'eval_reward_std': np.std(eval_rewards_all_eps) if eval_rewards_all_eps else 0.0,
            'eval_length_mean': np.mean(eval_lengths_all_eps) if eval_lengths_all_eps else 0.0,
            'eval_mentor_queries_mean': np.mean(eval_mentor_queries_all_eps) if eval_mentor_queries_all_eps else 0.0,
            'current_training_phase_at_eval': self.phase_transitions.get(self.training_phase, -1)
            # Log phase numerically
        }

        # Pathway consistency analysis (if data collected)
        # if pathway_activations_all_eps and self.critical_signatures:
        #     # This analysis needs careful implementation, potentially averaging over episodes/steps
        #     # For now, let's assume a placeholder or skip if too complex for one update
        #     # pathway_consistency = self._analyze_pathway_consistency_eval(pathway_activations_all_eps)
        #     # eval_metrics['eval_pathway_consistency'] = pathway_consistency
        #     pass

        return eval_metrics

    def _analyze_pathway_consistency(self, pathway_activations_list_of_dicts: List[Dict]) -> float:
        """Analyze consistency of critical pathway activations during rollouts/training steps.
           Input: List of activation dicts, each from one student forward pass.
        """
        if not self.critical_signatures or not pathway_activations_list_of_dicts:
            return 0.0

        consistencies_per_signature = []

        for signature in self.critical_signatures:
            if not signature: continue  # Skip None signatures

            # Collect the (compressed) student pathway activations for this signature across all provided steps
            student_pathway_patterns_over_time = []
            for student_batch_activations in pathway_activations_list_of_dicts:
                # Extract raw activations for this signature's pathway from student_batch_activations
                current_raw_pathway_acts = self.focused_distillation_loss._extract_activations_for_signature(
                    student_batch_activations, signature
                )  # (Batch, Num_Pathway_Neurons)

                if current_raw_pathway_acts.numel() > 0:
                    # Compress to match signature.activation_pattern's dimension
                    # Take mean over batch if exists, then compress
                    if current_raw_pathway_acts.dim() > 1:
                        mean_batch_acts = current_raw_pathway_acts.mean(dim=0)
                    else:
                        mean_batch_acts = current_raw_pathway_acts

                    target_dim = signature.activation_pattern.shape[0]
                    compressed_pattern = self.focused_distillation_loss._compress_to_signature_space(
                        mean_batch_acts, target_dim
                    )
                    student_pathway_patterns_over_time.append(compressed_pattern)

            if len(student_pathway_patterns_over_time) > 1:
                patterns_tensor = torch.stack(student_pathway_patterns_over_time)  # (Time, TargetDim)
                # Calculate consistency: 1 - normalized variance or distance from mean signature
                # Higher value means more consistent (closer to the mean pattern, or less variance)
                mean_pattern = patterns_tensor.mean(dim=0)
                # Cosine similarity to the mean pattern
                similarity_to_mean = F.cosine_similarity(patterns_tensor, mean_pattern.unsqueeze(0),
                                                         dim=1).mean().item()
                consistencies_per_signature.append(max(0.0, similarity_to_mean))

        return np.mean(consistencies_per_signature) if consistencies_per_signature else 0.0

    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Enhanced logging with phase information"""
        for key, value in metrics.items():
            # Ensure value is a scalar float, not a tensor or complex object
            log_value = value
            if isinstance(value, torch.Tensor):
                log_value = value.item()
            elif isinstance(value, np.ndarray):
                log_value = value.item()  # Or np.mean(value) if it's an array from somewhere

            if isinstance(log_value, (float, int)):
                self.logger.log_step(self.timestep, {
                    f"{self.training_phase}_{key.replace('.', '_')}": log_value})  # Replace dots for cleaner keys
            # else:
            # print(f"Warning: Metric '{key}' has non-scalar value '{log_value}' of type {type(log_value)}, not logging.")

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Enhanced evaluation logging"""
        for key, value in metrics.items():
            self.logger.log_step(self.timestep, {key.replace('.', '_'): value})

    def _log_progress(self, elapsed_time: float):
        """Enhanced progress logging"""
        stats = self.logger.get_statistics()  # Gets averages etc.

        # Try to get a recent loss if available for logging
        recent_total_loss = 0
        if 'losses' in self.logger.metrics_history and 'total_loss' in self.logger.metrics_history['losses']:
            if self.logger.metrics_history['losses']['total_loss']:
                recent_total_loss = self.logger.metrics_history['losses']['total_loss'][-1]
        elif f"{self.training_phase}_total_loss" in stats:  # Check if it's in averaged stats
            recent_total_loss = stats[f"{self.training_phase}_total_loss"]
        elif f"{self.training_phase}_rl_total_rl" in stats:  # Check for specific phase loss
            recent_total_loss = stats[f"{self.training_phase}_rl_total_rl"]

        self.logger.log(
            f"Phase: {self.training_phase} | "
            f"TS: {self.timestep:,} | Ep: {self.episode_count:,} | "
            f"Rew (100): {stats.get('avg_reward_100', 0):.2f} | "
            f"Best Rew: {self.best_reward:.2f} | "
            # f"Avg Loss: {recent_total_loss:.3e} | " # Can be noisy
            f"CritSigs: {len(self.critical_signatures) if self.critical_signatures else 0} | "
            f"Time: {elapsed_time:.1f}s"
        )

        # Plot training curves less frequently
        if self.timestep > 0 and self.timestep % (
                LOGGING_CONFIG['log_interval'] * 20) == 0:  # e.g. every 20k if interval is 1k
            self.logger.plot_training_curves()

    def _save_checkpoint(self):
        """Enhanced checkpointing with activation signatures"""
        models_state = {
            'mentor': self.mentor.state_dict(),
            'student': self.student.state_dict(),
        }
        optimizers_state = {
            'student': self.student.optimizer.state_dict(),
            'mentor': self.mentor.optimizer.state_dict(),
        }

        # Save additional activation-based data
        # Ensure critical_signatures are serializable (dataclasses are fine with pickle)
        # pathway_analyzer_state: save relevant parts, not entire object if too complex or has unpicklables
        pathway_analyzer_state_to_save = {}
        if self.pathway_analyzer:
            pathway_analyzer_state_to_save = {
                'model_structure': self.pathway_analyzer.model_structure,
                # activation_graph is nx.Graph, picklable by default
                'activation_graph': self.pathway_analyzer.activation_graph,
                'pathway_importance': self.pathway_analyzer.pathway_importance,  # If still used
            }

        additional_data = {
            'critical_signatures': self.critical_signatures,
            'training_phase': self.training_phase,
            'pathway_analyzer_state': pathway_analyzer_state_to_save,
            'best_reward': self.best_reward,
            'episode_count': self.episode_count
        }

        # Use logger's save_checkpoint for models and optimizers
        # self.logger.save_checkpoint(models_state, optimizers_state, self.timestep)
        # The logger method expects nn.Module and optim.Optimizer objects, not state_dicts
        # Let's adapt or save separately. For now, save combined.

        checkpoint_path = os.path.join(self.logger.log_dir, f'checkpoint_{self.timestep}.pt')
        torch.save({
            'timestep': self.timestep,
            'metrics_history': self.logger.metrics_history,  # Logger might save this itself too
            'models_state_dict': models_state,
            'optimizers_state_dict': optimizers_state,
            'additional_pipeline_data': additional_data
        }, checkpoint_path)

        self.logger.log(f"Saved full pipeline checkpoint at timestep {self.timestep} to {checkpoint_path}")

    def _final_evaluation(self):
        """Enhanced final evaluation"""
        self.logger.log("Running final enhanced evaluation...")

        final_metrics = self._evaluate()  # Returns a dict of floats
        trajectory_stats = self.trajectory_buffer.get_statistics()  # Also dict of floats

        self.logger.log("=== ENHANCED FINAL RESULTS ===")
        self.logger.log(f"Final Training Phase Reached: {self.training_phase}")
        self.logger.log(f"Total Timesteps: {self.timestep:,}")
        self.logger.log(f"Total Episodes Trained: {self.episode_count:,}")
        self.logger.log(
            f"Number of Critical Activation Signatures: {len(self.critical_signatures) if self.critical_signatures else 0}")

        self.logger.log(
            f"Final Eval Avg Reward: {final_metrics.get('eval_reward_mean', 0):.2f} +/- {final_metrics.get('eval_reward_std', 0):.2f}")
        self.logger.log(f"Overall Best Episode Reward during Training: {self.best_reward:.2f}")

        if trajectory_stats:
            self.logger.log(f"Trajectory Buffer Success Rate: {trajectory_stats.get('success_rate', 0):.2%}")
            self.logger.log(f"Trajectory Buffer Avg Reward: {trajectory_stats.get('avg_reward', 0):.2f}")

        self.logger.log(f"Final Eval Avg Episode Length: {final_metrics.get('eval_length_mean', 0):.1f}")
        self.logger.log(f"Final Eval Avg Mentor Queries per Ep: {final_metrics.get('eval_mentor_queries_mean', 0):.1f}")

        # if 'eval_pathway_consistency' in final_metrics: # If implemented and added
        #     self.logger.log(f"Final Eval Pathway Consistency Score: {final_metrics['eval_pathway_consistency']:.3f}")

        # Save final models and data
        self._save_checkpoint()  # Save one last time
        self.logger.plot_training_curves()  # Plot final curves
        self.logger.log("Enhanced final evaluation complete. All data saved.")


def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Revolutionary AI Pipeline with Activation-Based Distillation')
    parser.add_argument('--log_dir', type=str, default='logs', help='Logging directory')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--env_name', type=str, default=ENV_CONFIG['name'], help='Environment name (overrides config)')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--human_demos_path', type=str, default=None,
                        help='Path to human demonstration data (e.g., .pkl file)')
    parser.add_argument('--skip_behavior_cloning', action='store_true', help='Skip behavior cloning phase')

    args = parser.parse_args()

    # Override ENV_CONFIG['name'] if arg is provided
    if args.env_name:
        ENV_CONFIG['name'] = args.env_name
        print(f"Overriding environment name to: {args.env_name}")

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create and run enhanced pipeline
    pipeline = EnhancedRevolutionaryPipeline(args)

    if args.load_checkpoint:
        # TODO: Implement enhanced checkpoint loading logic
        # This would involve loading model state_dicts, optimizer state_dicts,
        # and potentially other training states like timestep, critical_signatures, etc.
        pipeline.logger.log(f"Loading checkpoint from: {args.load_checkpoint} (Full loading logic TBD)")
        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=DEVICE)
            pipeline.mentor.load_state_dict(checkpoint['models_state_dict']['mentor'])
            pipeline.student.load_state_dict(checkpoint['models_state_dict']['student'])
            pipeline.mentor.optimizer.load_state_dict(checkpoint['optimizers_state_dict']['mentor'])
            pipeline.student.optimizer.load_state_dict(checkpoint['optimizers_state_dict']['student'])
            pipeline.timestep = checkpoint.get('timestep', 0)
            pipeline.episode_count = checkpoint.get('additional_pipeline_data', {}).get('episode_count', 0)
            pipeline.best_reward = checkpoint.get('additional_pipeline_data', {}).get('best_reward', float('-inf'))
            pipeline.critical_signatures = checkpoint.get('additional_pipeline_data', {}).get('critical_signatures', [])
            # Consider how to restore logger's metrics_history for plotting continuity
            pipeline.logger.metrics_history = checkpoint.get('metrics_history', pipeline.logger.metrics_history)
            pipeline.logger.log(f"Checkpoint loaded successfully from timestep {pipeline.timestep}.")

        except Exception as e:
            pipeline.logger.log(f"Failed to load checkpoint: {e}", level="ERROR")

    if args.eval_only:
        if not args.load_checkpoint:
            pipeline.logger.log("Evaluation only mode requires a checkpoint to load.", level="ERROR")
            return
        pipeline._final_evaluation()
    else:
        pipeline.train()


if __name__ == "__main__":
    main()