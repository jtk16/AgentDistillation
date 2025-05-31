# enhanced_main.py (main.py)
"""
Enhanced Revolutionary AI Pipeline with Activation-Based Distillation
Integrates human behavior cloning with critical pathway analysis
"""

import torch
import torch.nn as nn
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
from distillation import DistillationTrainer, FeatureProjector
from memory import PrioritizedReplayBuffer, TrajectoryBuffer, ExperienceCollector, compute_gae
from utils import Logger, CurriculumScheduler, ActionProcessor, analyze_mentor_student_agreement
from torch.distributions import Categorical

from mathematical_framework import PathwayImportanceOptimizer, \
    InformationTheoreticAnalyzer

from activation_distillation import (
    HumanDemonstrationCollector, ActivationTracker,
    CriticalPathwayAnalyzer as MathCriticalPathwayAnalyzer,
    ActivationSignatureExtractor, FocusedDistillationLoss, create_activation_based_distillation_pipeline
)

from transfer_learning_utils import load_transfer_agent


class EnhancedRevolutionaryPipeline:
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args.log_dir)
        self.logger.log("Initializing Enhanced Revolutionary AI Pipeline...")

        self.env = create_environment()
        self.reward_shaper = AdvancedRewardShaper(ENV_CONFIG['name'], num_envs=self.env.num_envs)

        self.mentor = MultimodalMentor(state_dim=self.env.state_dim, num_actions=self.env.num_actions).to(DEVICE)
        self.student = StudentAgent(state_dim=self.env.state_dim, num_actions=self.env.num_actions).to(DEVICE)

        # MODIFIED: Parameter collection for the student's optimizer
        student_params_to_optimize = list(self.student.parameters())

        self.mentor.optimizer = optim.Adam(self.mentor.parameters(), lr=MENTOR_CONFIG['learning_rate'])

        self.math_pathway_optimizer = PathwayImportanceOptimizer()
        self.pathway_analyzer_shared = MathCriticalPathwayAnalyzer({})

        self.activation_pipeline = create_activation_based_distillation_pipeline(
            pathway_optimizer=self.math_pathway_optimizer,
            pathway_analyzer_instance=self.pathway_analyzer_shared
        )

        self.mentor_tracker = ActivationTracker(self.mentor)
        self.student_tracker = ActivationTracker(self.student)

        # DistillationTrainer is now an nn.Module. Its parameters (projectors) need to be optimized.
        self.distillation_trainer = DistillationTrainer(
            mentor=self.mentor,
            student=self.student,  # Pass student reference
            mentor_hidden_dim=MENTOR_CONFIG['hidden_dim'],
            student_hidden_dim=STUDENT_CONFIG['hidden_dim']
        ).to(DEVICE)  # Move DistillationTrainer and its submodules (projectors) to device
        student_params_to_optimize.extend(self.distillation_trainer.parameters())

        self.focused_distillation_loss_module = self.activation_pipeline['distillation_loss']
        if isinstance(self.focused_distillation_loss_module, nn.Module):
            self.focused_distillation_loss_module.to(DEVICE)
            student_params_to_optimize.extend(self.focused_distillation_loss_module.parameters())

        # Create student optimizer with all relevant parameters (student + projectors from DT + projectors from FDLM)
        self.student.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student_params_to_optimize),
            lr=STUDENT_CONFIG['learning_rate']
        )

        self.mentor_aux_sources: List[Dict[str, Any]] = []
        self.student_aux_sources: List[Dict[str, Any]] = []

        if TRANSFER_LEARNING_CONFIG.get('enabled', False):
            self.logger.log("Initializing transfer learning sources...")
            for m_conf in TRANSFER_LEARNING_CONFIG.get('mentor_transfer_sources', []):
                loaded_agent = load_transfer_agent(m_conf, self.env.state_dim, self.env.num_actions, DEVICE)
                if loaded_agent:
                    m_projector = None
                    self.mentor_aux_sources.append({**m_conf, 'model': loaded_agent, 'projector': m_projector})
                    self.logger.log(f"Loaded auxiliary mentor source: {m_conf['name']}")

            for s_conf in TRANSFER_LEARNING_CONFIG.get('student_transfer_sources', []):
                loaded_agent = load_transfer_agent(s_conf, self.env.state_dim, self.env.num_actions, DEVICE)
                if loaded_agent:
                    self.student_aux_sources.append({**s_conf, 'model': loaded_agent})
                    self.logger.log(f"Loaded auxiliary student source: {s_conf['name']}")

        if hasattr(self.distillation_trainer, 'set_student_aux_sources'):
            self.distillation_trainer.set_student_aux_sources(self.student_aux_sources, STUDENT_CONFIG['hidden_dim'])

        if hasattr(self.focused_distillation_loss_module, 'set_student_aux_sources') and \
                isinstance(self.focused_distillation_loss_module, FocusedDistillationLoss):
            self.focused_distillation_loss_module.set_student_aux_sources(self.student_aux_sources,
                                                                          STUDENT_CONFIG['hidden_dim'])
            # Re-create optimizer if new parameters were added to focused_distillation_loss_module dynamically
            # This is needed because focused_distillation_loss_module creates projectors in set_student_aux_sources
            # It's safer to collect all parameters *after* set_student_aux_sources has been called.
            all_params_for_student_opt = list(self.student.parameters()) + \
                                         list(self.distillation_trainer.parameters()) + \
                                         list(self.focused_distillation_loss_module.parameters())
            self.student.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, all_params_for_student_opt),
                lr=STUDENT_CONFIG['learning_rate']
            )

        self.replay_buffer = PrioritizedReplayBuffer(capacity=MEMORY_CONFIG['trajectory_buffer_size'],
                                                     alpha=MEMORY_CONFIG['priority_alpha'],
                                                     beta=MEMORY_CONFIG['priority_beta'])
        self.trajectory_buffer = TrajectoryBuffer(max_trajectories=MEMORY_CONFIG['trajectory_buffer_size'] // 100)
        self.experience_collector = ExperienceCollector(ENV_CONFIG['num_envs'])
        self.curriculum = CurriculumScheduler(CURRICULUM_CONFIG['stages'])
        self.action_processor = ActionProcessor(self.env.num_actions)
        self.critical_signatures: List[Any] = []
        self.pathway_analyzer = self.activation_pipeline['pathway_analyzer']
        self.signature_extractor = self.activation_pipeline['signature_extractor']
        self.training_phase = 'human_cloning'
        self.phase_transitions = {'human_cloning': TRAINING_CONFIG.get('human_cloning_steps', 5000),
                                  'focused_distillation': TRAINING_CONFIG.get('focused_distillation_steps', 20000),
                                  'standard_training': float('inf')}
        self.timestep = 0;
        self.episode_count = 0;
        self.best_reward = float('-inf')
        self.logger.log("Enhanced pipeline initialization complete!")

    def _process_demonstrations_for_pathways(self, demonstrations_for_bc: List[Dict]): # From previous fix
        if not demonstrations_for_bc:
            self.logger.log("No demonstrations available. Skipping pathway analysis and mentor BC/fine-tuning.")
            self.critical_signatures = []
            return

        bc_epochs = MENTOR_CONFIG.get('bc_epochs', 10)
        demo_chunk_size = TRAINING_CONFIG.get('demo_processing_chunk_size', 5)
        all_extracted_signatures: List[Any] = []

        self.logger.log(f"Starting Mentor Training / Fine-tuning / Pathway Analysis for {bc_epochs} epochs, "
                        f"processing {len(demonstrations_for_bc)} demos in chunks of {demo_chunk_size}.")

        for chunk_start_idx in range(0, len(demonstrations_for_bc), demo_chunk_size):
            chunk_end_idx = min(chunk_start_idx + demo_chunk_size, len(demonstrations_for_bc))
            demo_chunk = demonstrations_for_bc[chunk_start_idx:chunk_end_idx]
            self.logger.log(f"Processing demo chunk: {chunk_start_idx+1}-{chunk_end_idx}...")

            activations_sequence_for_graph_chunk: List[Dict[str, torch.Tensor]] = []
            importance_scores_sequence_for_graph_chunk: List[Dict[str, torch.Tensor]] = []
            performance_scores_for_signatures_chunk: List[float] = []

            for epoch in range(bc_epochs):
                epoch_total_bc_loss = 0.0; epoch_total_mentor_transfer_loss = 0.0; num_bc_batches_in_chunk_epoch = 0
                for _, demo in enumerate(demo_chunk):
                    demo_states_np, demo_actions_np = np.array(demo['states']), np.array(demo['actions'])
                    demo_performance = demo['performance_score']; max_demo_len_for_mentor_pass = 100
                    if len(demo_states_np) > max_demo_len_for_mentor_pass:
                        demo_states_np = demo_states_np[:max_demo_len_for_mentor_pass]; demo_actions_np = demo_actions_np[:max_demo_len_for_mentor_pass]
                    states_tensor = torch.tensor(demo_states_np, dtype=torch.float32).to(DEVICE); actions_tensor = torch.tensor(demo_actions_np, dtype=torch.long).to(DEVICE)
                    if states_tensor.ndim == 1: states_tensor = states_tensor.unsqueeze(0)
                    if actions_tensor.ndim == 0: actions_tensor = actions_tensor.unsqueeze(0)
                    if states_tensor.shape[0] == 0: continue
                    self.mentor.train(); self.mentor_tracker.clear_cache(); mentor_outputs_bc = self.mentor(states_tensor)
                    bc_loss = F.cross_entropy(mentor_outputs_bc['policy_logits'], actions_tensor)
                    current_batch_mentor_transfer_loss = torch.tensor(0.0, device=DEVICE)
                    if TRANSFER_LEARNING_CONFIG.get('enabled', False) and self.mentor_aux_sources:
                        for aux_source_config in self.mentor_aux_sources:
                            aux_model, aux_weight = aux_source_config['model'], aux_source_config['weight']; aux_targets = aux_source_config.get('transfer_targets', [])
                            with torch.no_grad(): aux_mentor_outputs = aux_model(states_tensor)
                            current_aux_loss_term = torch.tensor(0.0, device=DEVICE)
                            if 'policy_logits' in aux_targets and 'policy_logits' in aux_mentor_outputs: current_aux_loss_term += F.kl_div(F.log_softmax(mentor_outputs_bc['policy_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1), F.softmax(aux_mentor_outputs['policy_logits'] / DISTILLATION_CONFIG['temperature'], dim=-1), reduction='batchmean', log_target=False) * (DISTILLATION_CONFIG['temperature']**2)
                            if 'value' in aux_targets and 'value' in aux_mentor_outputs and 'value' in mentor_outputs_bc: current_aux_loss_term += F.mse_loss(mentor_outputs_bc['value'], aux_mentor_outputs['value']) * DISTILLATION_CONFIG.get('value_distill_weight',0.5)
                            current_batch_mentor_transfer_loss += current_aux_loss_term * aux_weight
                    total_mentor_loss = bc_loss + current_batch_mentor_transfer_loss
                    epoch_total_bc_loss += bc_loss.item(); epoch_total_mentor_transfer_loss += current_batch_mentor_transfer_loss.item(); num_bc_batches_in_chunk_epoch += 1
                    self.mentor.optimizer.zero_grad(); total_mentor_loss.backward(retain_graph=True)
                    if epoch == bc_epochs - 1:
                        activations_for_graph_step = self.mentor_tracker.get_activations()
                        if activations_for_graph_step:
                            target_perf_tensor = torch.tensor(demo_performance, device=DEVICE, dtype=torch.float32)
                            importance_for_graph_step = self.pathway_analyzer.compute_activation_importance(activations_for_graph_step, target_perf_tensor, method='gradient_based')
                            num_steps_in_demo = states_tensor.shape[0]
                            for step_idx in range(num_steps_in_demo):
                                step_acts = {k: v[step_idx].unsqueeze(0) for k, v in activations_for_graph_step.items() if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == num_steps_in_demo}
                                step_imps = {k: v[step_idx].unsqueeze(0) for k, v in importance_for_graph_step.items() if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == num_steps_in_demo}
                                if step_acts: activations_sequence_for_graph_chunk.append(step_acts); importance_scores_sequence_for_graph_chunk.append(step_imps); performance_scores_for_signatures_chunk.append(demo_performance)
                    self.mentor.optimizer.step()
                    del states_tensor, actions_tensor, mentor_outputs_bc, total_mentor_loss
                    if 'activations_for_graph_step' in locals(): del activations_for_graph_step
                    if 'importance_for_graph_step' in locals(): del importance_for_graph_step
                    if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()
                avg_bc_loss_ep = epoch_total_bc_loss / num_bc_batches_in_chunk_epoch if num_bc_batches_in_chunk_epoch > 0 else 0.0; avg_mtr_loss_ep = epoch_total_mentor_transfer_loss / num_bc_batches_in_chunk_epoch if num_bc_batches_in_chunk_epoch > 0 else 0.0
                if epoch % max(1, bc_epochs // 2) == 0 : self.logger.log(f"Chunk {chunk_start_idx//demo_chunk_size + 1}, Mentor Epoch {epoch+1}/{bc_epochs}, Avg BC Loss: {avg_bc_loss_ep:.4f}, Avg M.Transfer Loss: {avg_mtr_loss_ep:.4f}")
            self.mentor.eval()
            if not activations_sequence_for_graph_chunk or not importance_scores_sequence_for_graph_chunk: self.logger.log(f"No valid sequences from chunk {chunk_start_idx//demo_chunk_size + 1}."); continue
            self.logger.log(f"Analyzing pathways for demo chunk {chunk_start_idx//demo_chunk_size + 1}..."); activation_graph_chunk = self.pathway_analyzer.build_activation_graph(activations_sequence_for_graph_chunk, importance_scores_sequence_for_graph_chunk)
            critical_pathways_nodesets_chunk = []
            if activation_graph_chunk.number_of_nodes() > 0: critical_pathways_nodesets_chunk = self.pathway_analyzer.identify_critical_pathways(activation_graph_chunk)
            else: self.logger.log(f"Graph for chunk {chunk_start_idx//demo_chunk_size + 1} has no nodes.")
            if not critical_pathways_nodesets_chunk: self.logger.log(f"No critical pathways in chunk {chunk_start_idx//demo_chunk_size + 1}.")
            else:
                self.logger.log(f"Extracting {len(critical_pathways_nodesets_chunk)} sigs from chunk..."); chunk_signatures = self.signature_extractor.extract_signatures(critical_pathways_nodesets_chunk, activations_sequence_for_graph_chunk, performance_scores_sequence=performance_scores_for_signatures_chunk, target_dim=MENTOR_CONFIG.get('num_knowledge_tokens', 16))
                if chunk_signatures: all_extracted_signatures.extend(chunk_signatures); self.logger.log(f"Extracted {len(chunk_signatures)} sigs from chunk.")
            del activations_sequence_for_graph_chunk, importance_scores_sequence_for_graph_chunk, performance_scores_for_signatures_chunk
            if 'activation_graph_chunk' in locals(): del activation_graph_chunk
            if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()
        self.critical_signatures = all_extracted_signatures
        if self.critical_signatures: self.logger.log(f"Total critical sigs: {len(self.critical_signatures)}")
        else: self.logger.log("No critical sigs extracted overall.")

    def train(self):  # Copied from previous full response for completeness
        self.logger.log("Starting enhanced training with activation-based distillation...")
        if self.args.human_demos_path and not self.args.skip_behavior_cloning:
            self.logger.log("Phase 1: Human Behavior Cloning and Critical Pathway Analysis")
            self._phase1_human_behavior_cloning()
        elif not self.args.skip_behavior_cloning:
            self.logger.log("No human demonstrations path provided, generating synthetic demos for Phase 1...")
            self._generate_synthetic_demonstrations()
        else:
            self.logger.log("Skipping Phase 1 (Human Behavior Cloning / Synthetic Demos / Pathway Analysis).")
            self.critical_signatures = []

        self.logger.log("Phase 2: Focused Distillation Training (will start after human_cloning phase timesteps)")
        self.logger.log("Phase 3: Standard RL Training (will start after focused_distillation phase timesteps)")

        observations, _ = self.env.reset(seed=SEED)
        start_time = time.time()

        while self.timestep < TRAINING_CONFIG['total_timesteps']:
            self._update_training_phase()
            states_tensor = self.env.get_state_tensor(observations)
            rollout_data = self._collect_rollout(states_tensor)

            if self.experience_collector.next_states and len(self.experience_collector.next_states) > 0:
                observations = self.experience_collector.next_states[-1]

            if self.timestep > TRAINING_CONFIG['batch_size']:
                if self.training_phase == 'focused_distillation' and (self.critical_signatures or (
                        TRANSFER_LEARNING_CONFIG.get('enabled',
                                                     False) and self.student_aux_sources)):  # MODIFIED: Allow focused if aux sources exist
                    training_metrics = self._train_models_focused(rollout_data)
                else:
                    if self.training_phase == 'focused_distillation':
                        self.logger.log(
                            "In focused_distillation phase but no critical signatures and no student aux sources. Running standard training.")
                    training_metrics = self._train_models_standard(rollout_data)

                if training_metrics: self._log_training_metrics(training_metrics)

            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['eval_interval'] == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation_metrics(eval_metrics)

            if CURRICULUM_CONFIG['enabled']:
                avg_reward = self.trajectory_buffer.get_statistics().get('avg_reward', 0)
                self.curriculum.get_current_config(avg_reward)

            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()

            if self.timestep > 0 and self.timestep % LOGGING_CONFIG['log_interval'] == 0:
                self._log_progress(time.time() - start_time)
                if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()

        self.logger.log("Enhanced training completed!")
        self._final_evaluation()

    def _phase1_human_behavior_cloning(self):
        self.logger.log("Starting human behavior cloning phase...")
        demo_collector = self.activation_pipeline['demonstration_collector']
        raw_demonstrations = []
        if self.args.human_demos_path and os.path.exists(self.args.human_demos_path):
            raw_demonstrations = self._load_human_demonstrations(self.args.human_demos_path)
            for demo_data_item in raw_demonstrations:
                if isinstance(demo_data_item, dict) and all(
                        k in demo_data_item for k in ['states', 'actions', 'performance_score']):
                    demo_collector.collect_demonstration(
                        states=demo_data_item['states'], actions=demo_data_item['actions'],
                        performance_score=demo_data_item['performance_score']
                    )
        else:
            self.logger.log("No human demonstrations found or path invalid.")
        demonstrations_to_process = demo_collector.get_successful_demonstrations()
        if not demonstrations_to_process and demo_collector.demonstrations:
            demonstrations_to_process = demo_collector.demonstrations
        self._process_demonstrations_for_pathways(demonstrations_to_process)

    def _generate_synthetic_demonstrations(self): # As in prior correct version
        self.logger.log("Generating synthetic demonstrations..."); demo_collector = self.activation_pipeline['demonstration_collector']
        demo_collector.demonstrations = [] ; num_episodes = 10; max_steps = TRAINING_CONFIG.get("max_synthetic_demo_collection_steps", 100)
        for _ in range(num_episodes):
            obs_list, _ = self.env.reset(); current_obs_list = obs_list
            ep_s_bufs = [[] for _ in range(self.env.num_envs)]; ep_a_bufs = [[] for _ in range(self.env.num_envs)]; ep_r_sums = np.zeros(self.env.num_envs)
            for _ in range(max_steps):
                acts_all = [[np.random.randint(0, self.env.num_actions)] for _ in range(self.env.num_envs)]; uncerts_all = [0.5]*self.env.num_envs
                next_obs_list, rews, terms, truncs, _ = self.env.step(acts_all, uncerts_all)
                for i in range(self.env.num_envs): ep_s_bufs[i].append(current_obs_list[i].copy()); ep_a_bufs[i].append(acts_all[i][0]); ep_r_sums[i] += rews[i]
                current_obs_list = next_obs_list
                if np.all(terms | truncs): break
            for i in range(self.env.num_envs):
                if ep_s_bufs[i]: max_r = ENV_CONFIG.get('max_episode_steps',500); p_score=np.clip(ep_r_sums[i]/(max_r*0.5),0.,1.); demo_collector.collect_demonstration(ep_s_bufs[i], ep_a_bufs[i], p_score)
        self.logger.log(f"Generated {len(demo_collector.demonstrations)} synthetic demos."); self._process_demonstrations_for_pathways(demo_collector.demonstrations)

    def _update_training_phase(self): # As in prior correct version
        h_end = self.phase_transitions['human_cloning']; f_end = self.phase_transitions['focused_distillation']
        if self.timestep < h_end: self.training_phase = 'human_cloning'
        elif self.timestep < f_end:
            can_do_focused = bool(self.critical_signatures) or (TRANSFER_LEARNING_CONFIG.get('enabled',False) and self.student_aux_sources)
            if not can_do_focused and self.training_phase != 'standard_training_fallback': self.logger.log("No crit_sigs or student_aux for focused. Will behave like standard."); self.training_phase = 'standard_training_fallback' # Special state to log once
            elif can_do_focused : self.training_phase = 'focused_distillation'
            else: self.training_phase = 'standard_training_fallback' # if it was already fallback
        else: self.training_phase = 'standard_training'

    def _train_models_focused(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        batch_data = rollout_data.get('batch_data')
        # Fallback if no critical signatures AND no student_aux_sources if TL enabled
        if not self.critical_signatures and not (
                TRANSFER_LEARNING_CONFIG.get('enabled', False) and self.student_aux_sources):
            self.logger.log(
                "No critical signatures and no student aux sources for focused training, falling back to standard.")
            return self._train_models_standard(rollout_data)

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

        self.student.train()  # Student and its submodules (like focused_distillation_loss_module's projectors)
        if isinstance(self.focused_distillation_loss_module, nn.Module):
            self.focused_distillation_loss_module.train()
        self.mentor.eval()

        for _ in range(TRAINING_CONFIG['num_ppo_epochs']):
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            current_batch_size = num_samples if num_samples < batch_size else batch_size
            if current_batch_size == 0: continue
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, current_batch_size):
                end = min(start + current_batch_size, num_samples)
                if start == end: continue
                batch_indices = indices[start:end]
                mini_batch_states = batch_data['states'][batch_indices]

                self.mentor_tracker.clear_cache();
                self.student_tracker.clear_cache()
                with torch.no_grad():
                    mentor_outputs_distill = self.mentor(mini_batch_states)
                student_outputs_distill_and_rl = self.student(mini_batch_states)  # Student forward pass
                mentor_activations_distill = self.mentor_tracker.get_activations()
                student_activations_distill_and_rl = self.student_tracker.get_activations()

                focused_losses_dict = {}
                valid_signatures = [cs for cs in self.critical_signatures if
                                    cs is not None] if self.critical_signatures else []

                # Ensure there's something to distill from (signatures or aux sources)
                can_do_focused_distill = bool(valid_signatures) or \
                                         (TRANSFER_LEARNING_CONFIG.get('enabled', False) and \
                                          hasattr(self.focused_distillation_loss_module, 'student_aux_sources') and \
                                          self.focused_distillation_loss_module.student_aux_sources)

                if not mentor_activations_distill or not student_activations_distill_and_rl or not can_do_focused_distill:
                    focused_losses_dict = {'total_focused': torch.tensor(0.0, device=DEVICE)}
                elif isinstance(self.focused_distillation_loss_module, FocusedDistillationLoss):
                    focused_losses_dict = self.focused_distillation_loss_module(
                        student_outputs_distill_and_rl, mentor_outputs_distill,
                        student_activations_distill_and_rl, mentor_activations_distill,
                        valid_signatures, states=mini_batch_states
                    )
                else:
                    focused_losses_dict = {'total_focused': torch.tensor(0.0, device=DEVICE)}

                rl_losses_dict = self._compute_rl_losses(
                    student_outputs_distill_and_rl, actions=batch_data['actions'][batch_indices],
                    returns=returns[batch_indices], advantages=advantages[batch_indices],
                    old_log_probs=batch_data['log_probs'][batch_indices],
                    old_values=batch_data['old_values'][batch_indices]
                )

                # These alpha/beta are for balancing RL with the *entire* focused distillation loss
                distill_component_weight = 1.0 - DISTILLATION_CONFIG['alpha']
                rl_component_weight = DISTILLATION_CONFIG['alpha']

                current_total_loss = (
                        distill_component_weight * focused_losses_dict.get('total_focused',
                                                                           torch.tensor(0.0, device=DEVICE)) +
                        rl_component_weight * rl_losses_dict.get('total_rl', torch.tensor(0.0, device=DEVICE))
                )

                self.student.optimizer.zero_grad()
                if current_total_loss.requires_grad: current_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.student.optimizer.param_groups[0]['params']),
                    TRAINING_CONFIG['max_grad_norm']
                )
                self.student.optimizer.step()

                for loss_d, prefix in [(focused_losses_dict, "focused"), (rl_losses_dict, "rl")]:
                    for k, v_tensor in loss_d.items():
                        metric_k = f"{prefix}_{k.replace('.', '_')}"
                        if metric_k not in training_metrics_accum: training_metrics_accum[metric_k] = []
                        training_metrics_accum[metric_k].append(v_tensor.item())
                if 'total_loss' not in training_metrics_accum: training_metrics_accum['total_loss'] = []
                training_metrics_accum['total_loss'].append(current_total_loss.item())

        final_metrics: Dict[str, float] = {k: np.mean(v) if v else 0.0 for k, v in training_metrics_accum.items()}
        return final_metrics

    def _train_models_standard(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        batch_data = rollout_data.get('batch_data')
        if not batch_data: return {}
        # Ensure 'old_values' is present as it's now required by DistillationTrainer._compute_rl_losses
        required_keys = ['rewards', 'values', 'dones', 'states', 'actions', 'log_probs', 'old_values']
        if not all(
                key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].numel() > 0 for key
                in required_keys):
            self.logger.log(
                f"Batch data incomplete for standard training. Missing or empty keys: {[k for k in required_keys if not (k in batch_data and isinstance(batch_data[k], torch.Tensor) and batch_data[k].numel() > 0)]}")
            return {}

        advantages, returns = compute_gae(
            rewards=batch_data['rewards'], values=batch_data['values'],
            dones=batch_data['dones'], gamma=TRAINING_CONFIG['gamma'], gae_lambda=TRAINING_CONFIG['gae_lambda']
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        training_metrics_accum: Dict[str, List[float]] = {}
        self.student.train()
        self.distillation_trainer.train()  # Set DistillationTrainer and its submodules to train mode

        for _ in range(TRAINING_CONFIG['num_ppo_epochs']):
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            current_batch_size = num_samples if num_samples < batch_size else batch_size
            if current_batch_size == 0: continue

            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, current_batch_size):
                end = min(start + current_batch_size, num_samples)
                if start == end: continue
                batch_indices = indices[start:end]

                # DistillationTrainer.train_step computes the loss and returns metrics + loss tensor
                metrics_and_loss = self.distillation_trainer.train_step(
                    states=batch_data['states'][batch_indices],
                    actions=batch_data['actions'][batch_indices],
                    returns=returns[batch_indices],
                    advantages=advantages[batch_indices],
                    old_log_probs=batch_data['log_probs'][batch_indices],
                    old_values=batch_data['old_values'][batch_indices]
                )

                total_loss_tensor = metrics_and_loss.pop('_total_loss_tensor_for_backward', None)
                if total_loss_tensor is not None and total_loss_tensor.requires_grad:
                    self.student.optimizer.zero_grad()
                    total_loss_tensor.backward()  # Single backward pass for student + distillation_trainer params
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.student.optimizer.param_groups[0]['params']),
                        TRAINING_CONFIG['max_grad_norm']
                    )
                    self.student.optimizer.step()
                elif total_loss_tensor is None:
                    self.logger.log("Warning: No loss tensor returned from distillation_trainer.train_step",
                                    level="WARN")

                for key, value in metrics_and_loss.items():  # Log other metrics
                    metric_key = key.replace('.', '_')
                    if metric_key not in training_metrics_accum: training_metrics_accum[metric_key] = []
                    training_metrics_accum[metric_key].append(value)  # value is already float

        final_training_metrics: Dict[str, float] = {
            k: np.mean(v) if v else 0.0 for k, v in training_metrics_accum.items()
        }
        return final_training_metrics

    def _compute_rl_losses(self, student_outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor
                           ) -> Dict[str, torch.Tensor]:
        dev = student_outputs['primary_logits'].device if 'primary_logits' in student_outputs else DEVICE
        if 'primary_logits' not in student_outputs or student_outputs['primary_logits'] is None:
            self.logger.log("Missing 'primary_logits' in student_outputs for RL loss.", level="ERROR")
            return {k: torch.tensor(0.0, device=dev, requires_grad=True) for k in
                    ['policy_loss', 'value_loss', 'entropy', 'total_rl']}

        dist = Categorical(logits=student_outputs['primary_logits'])
        current_log_probs = dist.log_prob(actions)
        ratio = torch.exp(current_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TRAINING_CONFIG['clip_ratio'],
                            1.0 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        current_values = student_outputs['value'].squeeze(-1)
        old_values_squeezed = old_values.squeeze(-1)  # Ensure old_values is also squeezed if it had an extra dim

        if TRAINING_CONFIG.get('clip_value_loss', True):
            values_clipped = old_values_squeezed + torch.clamp(current_values - old_values_squeezed,
                                                               -TRAINING_CONFIG['clip_ratio'],
                                                               TRAINING_CONFIG['clip_ratio'])
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
            'policy_loss': policy_loss, 'value_loss': value_loss,
            'entropy': entropy_bonus, 'total_rl': total_rl_loss,
        }

    def _load_human_demonstrations(self, demo_path: str) -> List[Dict]:
        demonstrations = []
        if not os.path.exists(demo_path):
            self.logger.log(f"Demonstration file not found: {demo_path}", level="ERROR");
            return demonstrations
        try:
            import pickle
            with open(demo_path, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, list):
                for item in loaded_data:
                    if isinstance(item, dict) and all(k in item for k in ['states', 'actions', 'performance_score']):
                        s, a = np.array(item['states']), np.array(item['actions'])
                        if s.ndim >= 1 and a.ndim >= 0 and len(s) == len(a) and len(s) > 0:
                            demonstrations.append(
                                {'states': s, 'actions': a, 'performance_score': float(item['performance_score']),
                                 'length': len(s)})
                        else:
                            self.logger.log(f"Skipping demo: inconsistent states/actions or empty.", level="WARN")
                    else:
                        self.logger.log(f"Skipping invalid demo item format: {type(item)}", level="WARN")
            else:
                self.logger.log(f"Demo file {demo_path} not a list.", level="ERROR")
            self.logger.log(f"Loaded {len(demonstrations)} demos from {demo_path}")
        except Exception as e:
            self.logger.log(f"Error loading demos: {e}", level="ERROR"); import traceback; self.logger.log(
                traceback.format_exc(), level="DEBUG")
        return demonstrations

    def _collect_rollout(self, initial_states_tensor: torch.Tensor) -> Dict[
        str, Any]:  # As in prior correct version (with KeyError fix)
        self.experience_collector.reset();
        current_states_tensor = initial_states_tensor.clone()
        rollout_rewards: List[float] = [];
        num_mentor_queries = 0
        for _ in range(TRAINING_CONFIG['rollout_steps']):
            student_actions_b, student_info_b = self.student.act(current_states_tensor)
            mentor_advice_l: List[Optional[Any]] = [None] * self.env.num_envs
            for i in range(self.env.num_envs):
                if student_info_b[i].get('should_query_mentor', False): num_mentor_queries += 1; mentor_advice_l[
                    i] = self._query_mentor(current_states_tensor[i].unsqueeze(0))
            uncertainties_step = [info.get('uncertainty', {}).get('total', 0.5) for info in student_info_b]
            next_obs_np, rewards_np, terminated_np, truncated_np, infos_env = self.env.step(student_actions_b,
                                                                                            uncertainties_step)
            shaped_rewards_np = np.zeros_like(rewards_np)
            for i in range(self.env.num_envs):
                prim_act = student_actions_b[i][0] if student_actions_b[i] else 0
                current_env_info = {};
                is_done = terminated_np[i] or truncated_np[i]
                if is_done:
                    f_info_arr = infos_env.get('final_info');
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[
                        i] is not None: current_env_info = f_info_arr[i]
                shaped_rewards_np[i] = self.reward_shaper.shape_reward(next_obs_np[i], prim_act, rewards_np[i],
                                                                       is_done, current_env_info, env_idx=i)
            with torch.no_grad():
                student_outputs_c = self.student(current_states_tensor)
            prim_acts_l_c = []
            for s_acts in student_actions_b:
                act_v = s_acts[0] if s_acts and len(s_acts) > 0 else (
                    np.random.randint(0, self.env.num_actions) if self.env.num_actions > 0 else 0)
                if not (0 <= act_v < self.env.num_actions) and self.env.num_actions > 0:
                    act_v = np.clip(act_v, 0, self.env.num_actions - 1)
                elif self.env.num_actions == 0:
                    act_v = 0
                prim_acts_l_c.append(act_v)
            prim_acts_t_c = torch.tensor(prim_acts_l_c, dtype=torch.long, device=DEVICE)
            log_probs_c = Categorical(logits=student_outputs_c['primary_logits']).log_prob(prim_acts_t_c)
            values_c = student_outputs_c['value'].squeeze(-1)
            uncert_l_c = [info.get('uncertainty', {'total': 0.}) for info in student_info_b]
            self.experience_collector.add(state=current_states_tensor.cpu().numpy(), action=np.array(
                [a[0] if a and len(a) > 0 else 0 for a in student_actions_b]), reward=shaped_rewards_np,
                                          next_state=next_obs_np, done=(terminated_np | truncated_np),
                                          log_prob=log_probs_c, value=values_c, uncertainty=uncert_l_c,
                                          mentor_advice=mentor_advice_l)
            current_states_tensor = self.env.get_state_tensor(next_obs_np)
            for i in range(self.env.num_envs):
                if terminated_np[i] or truncated_np[i]:
                    f_info_arr = infos_env.get('final_info')
                    if f_info_arr is not None and i < len(f_info_arr) and f_info_arr[i] is not None and 'episode' in \
                            f_info_arr[i]:
                        rollout_rewards.append(f_info_arr[i]['episode']['r']);
                        self.episode_count += 1
            self.timestep += self.env.num_envs
        batch_d = self.experience_collector.get_batch_tensors()
        if 'values' in batch_d:
            batch_d['old_values'] = batch_d['values'].clone().detach()
        else:
            batch_d['old_values'] = torch.empty(0, device=DEVICE)
        return {'rollout_rewards': rollout_rewards, 'mentor_queries': num_mentor_queries, 'batch_data': batch_d}

    def _query_mentor(self, state_tensor_single_env: torch.Tensor) -> Any:  # As in prior correct version
        with torch.no_grad(): self.mentor.eval(); return self.mentor.get_advice(state_tensor_single_env)

    def _evaluate(self) -> Dict[str, float]:  # As in prior correct version
        self.logger.log("Eval...");
        self.student.eval()
        eval_env = create_environment(env_name=ENV_CONFIG['name'], num_envs=1)
        r_s, l_s, q_s = [], [], []
        for _ in range(LOGGING_CONFIG.get('num_eval_episodes', 10)):
            o, _ = eval_env.reset();
            epr, epl, epq = 0., 0, 0;
            dn = False
            while not dn:
                s_t = eval_env.get_state_tensor(o);
                acts_b, infos_b = self.student.act(s_t, deterministic=True)
                if infos_b[0].get('should_query_mentor', False): epq += 1
                n_o_v, r_v, t_v, tr_v, _ = eval_env.step([acts_b[0]],
                                                         [infos_b[0].get('uncertainty', {}).get('total', 0.5)])
                o, dn = n_o_v[0], t_v[0] or tr_v[0];
                epr += r_v[0];
                epl += 1
            r_s.append(epr);
            l_s.append(epl);
            q_s.append(epq)
        eval_env.close();
        self.student.train()
        return {'eval_reward_mean': float(np.mean(r_s)) if r_s else 0.,
                'eval_reward_std': float(np.std(r_s)) if r_s else 0.,
                'eval_length_mean': float(np.mean(l_s)) if l_s else 0.,
                'eval_mentor_queries_mean': float(np.mean(q_s)) if q_s else 0.}

    def _log_training_metrics(self, metrics: Dict[str, float]):  # As in prior correct version
        for k, v in metrics.items():
            lv = v.item() if isinstance(v, torch.Tensor) else (v.item() if isinstance(v, np.generic) else (
                np.mean(v) if isinstance(v, (list, np.ndarray)) and len(
                    v) > 0 else v))  # handle empty list for mean
            if isinstance(lv, (float, int)): self.logger.log_step(self.timestep, {
                f"{self.training_phase}_{k.replace('.', '_')}": lv})

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):  # As in prior correct version
        for k, v in metrics.items(): self.logger.log_step(self.timestep, {k.replace('.', '_'): v})

    def _log_progress(self, elapsed_time: float):  # As in prior correct version
        stats = self.logger.get_statistics();
        loss = 0.
        keys = [f"{self.training_phase}_total_loss", f"focused_total_focused", "rl_total_rl", "distill_loss_total",
                "total_loss"]
        for k_ in keys:
            if k_ in stats and stats[k_] != 0: loss = stats[k_];break
        if loss == 0. and 'losses' in self.logger.metrics_history:  # check float
            for p_pref in [self.training_phase, "focused", "rl", ""]:
                for l_suf in ['total_loss', 'total_focused', 'total_rl', 'distill_loss_total']:
                    ln = f"{p_pref}_{l_suf}" if p_pref else l_suf;
                    ln = ln.replace("__", "_")
                    if ln in self.logger.metrics_history['losses'] and self.logger.metrics_history['losses'][
                        ln]: loss = self.logger.metrics_history['losses'][ln][-1];break
                if abs(loss) > 1e-9: break  # check float
        self.logger.log(
            f"Phase:{self.training_phase}|TS:{self.timestep:,}|Ep:{self.episode_count:,}|Rew(100):{stats.get('avg_reward_100', 0):.2f}|Best:{self.best_reward:.2f}|Loss:{loss:.3e}|Sigs:{len(self.critical_signatures)}|T:{elapsed_time:.1f}s")
        if self.timestep > 0 and self.timestep % (
                LOGGING_CONFIG['log_interval'] * 10) == 0: self.logger.plot_training_curves()

    def _save_checkpoint(self):  # As in prior correct version
        m_s = {'mentor': self.mentor.state_dict(), 'student': self.student.state_dict()}
        o_s = {'student': self.student.optimizer.state_dict() if hasattr(self.student,
                                                                         'optimizer') and self.student.optimizer else None,
               'mentor': self.mentor.optimizer.state_dict() if hasattr(self.mentor,
                                                                       'optimizer') and self.mentor.optimizer else None}
        # Add main feature projector optimizer from DistillationTrainer if it exists and has state
        if hasattr(self.distillation_trainer, 'feature_projector') and hasattr(
                self.distillation_trainer.feature_projector, 'parameters'):
            # This assumes main_feature_projector_optimizer is part of student optimizer now, so not saved separately
            pass
        o_s = {k: v for k, v in o_s.items() if v is not None}
        aux_po_s = {}  # Aux projectors in DistillationTrainer are now part of student optimizer
        pa_s_tosave = {};
        if self.pathway_analyzer and hasattr(self.pathway_analyzer, 'model_structure'): pa_s_tosave[
            'model_structure'] = self.pathway_analyzer.model_structure
        add_d = {'critical_signatures': self.critical_signatures, 'training_phase': self.training_phase,
                 'pathway_analyzer_state': pa_s_tosave, 'best_reward': self.best_reward,
                 'episode_count': self.episode_count, 'distillation_trainer_aux_proj_optimizers': aux_po_s}
        cp_path = os.path.join(self.logger.log_dir, f'checkpoint_{self.timestep}.pt')
        try:
            torch.save({'timestep': self.timestep, 'metrics_history': self.logger.metrics_history,
                        'models_state_dict': m_s, 'optimizers_state_dict': o_s, 'additional_pipeline_data': add_d},
                       cp_path)
            self.logger.log(f"Saved ckpt @ {self.timestep}")
        except Exception as e:
            self.logger.log(f"Err saving ckpt:{e}", "ERROR")

    def _final_evaluation(self):  # As in prior correct version
        self.logger.log("Final eval...");
        mets = self._evaluate();
        ts = self.trajectory_buffer.get_statistics()
        self.logger.log("=== FINAL RESULTS ===");
        self.logger.log(f"Phase:{self.training_phase},TS:{self.timestep:,},Eps:{self.episode_count:,}")
        self.logger.log(f"Crit Sigs:{len(self.critical_signatures)}");
        self.logger.log(f"Eval Rew:{mets.get('eval_reward_mean', 0):.2f} +/- {mets.get('eval_reward_std', 0):.2f}")
        self.logger.log(f"Best Train Rew:{self.best_reward:.2f}")
        if ts and isinstance(ts, dict): self.logger.log(
            f"Traj Buf SR:{ts.get('success_rate', 0):.2%},AvgRew:{ts.get('avg_reward', 0):.2f}")
        self.logger.log(
            f"Eval EpLen:{mets.get('eval_length_mean', 0):.1f},MentorQ:{mets.get('eval_mentor_queries_mean', 0):.1f}")
        self._save_checkpoint();
        self.logger.plot_training_curves();
        self.logger.log("Final eval complete.")

def main():  # As in prior correct version (checkpoint loading adjusted for new optimizer structure)
    parser = argparse.ArgumentParser(description='Enhanced Revolutionary AI Pipeline')
    parser.add_argument('--log_dir', type=str, default='logs');
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--env_name', type=str, default=ENV_CONFIG['name']);
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--human_demos_path', type=str, default=None);
    parser.add_argument('--skip_behavior_cloning', action='store_true')
    args = parser.parse_args()
    if args.env_name: ENV_CONFIG['name'] = args.env_name
    torch.manual_seed(SEED);
    np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    pipeline = EnhancedRevolutionaryPipeline(args)
    pipeline.logger.log(
        f"Env:{ENV_CONFIG['name']},TL:{TRANSFER_LEARNING_CONFIG.get('enabled', False)},M Aux:{len(pipeline.mentor_aux_sources)},S Aux:{len(pipeline.student_aux_sources)}")
    if args.load_checkpoint:
        pipeline.logger.log(f"Loading ckpt: {args.load_checkpoint}")
        if not os.path.exists(args.load_checkpoint): pipeline.logger.log(f"CKPT NOT FOUND:{args.load_checkpoint}",
                                                                         "ERROR");return
        try:
            ckpt = torch.load(args.load_checkpoint, map_location=DEVICE);
            msd = ckpt.get('models_state_dict', {})
            if 'mentor' in msd: pipeline.mentor.load_state_dict(msd['mentor'])
            if 'student' in msd: pipeline.student.load_state_dict(msd['student'])
            osd = ckpt.get('optimizers_state_dict', {})
            if hasattr(pipeline.mentor,
                       'optimizer') and pipeline.mentor.optimizer and 'mentor' in osd: pipeline.mentor.optimizer.load_state_dict(
                osd['mentor'])
            if hasattr(pipeline.student,
                       'optimizer') and pipeline.student.optimizer and 'student' in osd: pipeline.student.optimizer.load_state_dict(
                osd['student'])
            # Note: Projector optimizers are no longer saved separately, their params are in student.optimizer
            pipeline.timestep = ckpt.get('timestep', 0);
            apd = ckpt.get('additional_pipeline_data', {})
            pipeline.episode_count = apd.get('episode_count', 0);
            pipeline.best_reward = apd.get('best_reward', float('-inf'))
            pipeline.critical_signatures = apd.get('critical_signatures', []);
            pipeline.training_phase = apd.get('training_phase', 'human_cloning')
            pa_s_l = apd.get('pathway_analyzer_state', {});
            if pipeline.pathway_analyzer and hasattr(pipeline.pathway_analyzer,
                                                     'model_structure'): pipeline.pathway_analyzer.model_structure = pa_s_l.get(
                'model_structure', {})
            pipeline.logger.metrics_history = ckpt.get('metrics_history', pipeline.logger.metrics_history);
            pipeline.logger.log(f"CKPT loaded from TS {pipeline.timestep}.")
        except Exception as e:
            pipeline.logger.log(f"Failed to load ckpt:{e}", "ERROR");import traceback;pipeline.logger.log(
                traceback.format_exc(), "ERROR");return
    if args.eval_only:
        if not args.load_checkpoint: pipeline.logger.log("Eval mode needs ckpt.", "ERROR");return
        pipeline._final_evaluation()
    else:
        pipeline.train()

if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ: print(
        "Consider PYTORCH_CUDA_ALLOC_CONF env var for CUDA memory.")
    main()