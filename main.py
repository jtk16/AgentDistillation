# enhanced_main.py
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

from config import *
from environment import create_environment, AdvancedRewardShaper
from mentor import MultimodalMentor
from student import StudentAgent
from distillation import DistillationTrainer
from memory import PrioritizedReplayBuffer, TrajectoryBuffer, ExperienceCollector, compute_gae
from utils import Logger, CurriculumScheduler, ActionProcessor, analyze_mentor_student_agreement
from activation_distillation import (
    HumanDemonstrationCollector, ActivationTracker, CriticalPathwayAnalyzer,
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

        # Initialize activation-based distillation pipeline
        self.activation_pipeline = create_activation_based_distillation_pipeline()

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
        self.pathway_analyzer = self.activation_pipeline['pathway_analyzer']
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
        if self.args.human_demos_path:
            self.logger.log("Phase 1: Human Behavior Cloning and Critical Pathway Analysis")
            self._phase1_human_behavior_cloning()
        else:
            self.logger.log("No human demonstrations provided, generating synthetic demos...")
            self._generate_synthetic_demonstrations()

        # Phase 2: Focused Distillation Training
        self.logger.log("Phase 2: Focused Distillation Training")
        self.training_phase = 'focused_distillation'

        # Phase 3: Standard RL Training
        self.logger.log("Phase 3: Standard RL Training")

        # Initial environment reset
        observations, _ = self.env.reset(seed=SEED)
        states = self.env.get_state_tensor(observations)

        start_time = time.time()

        while self.timestep < TRAINING_CONFIG['total_timesteps']:

            # Update training phase
            self._update_training_phase()

            # === ROLLOUT PHASE ===
            rollout_data = self._collect_rollout(states)

            # === TRAINING PHASE ===
            if self.timestep > TRAINING_CONFIG['batch_size']:
                if self.training_phase == 'focused_distillation':
                    training_metrics = self._train_models_focused(rollout_data)
                else:
                    training_metrics = self._train_models_standard(rollout_data)
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
            demonstrations = self._load_human_demonstrations(self.args.human_demos_path)
        else:
            self.logger.log("No human demonstrations found, skipping Phase 1")
            return

        # Train mentor on demonstrations with activation tracking
        activation_sequences = []
        importance_sequences = []

        for epoch in range(50):  # Behavior cloning epochs
            epoch_activations = []
            epoch_importance = []

            for demo in demonstrations:
                states = torch.tensor(demo['states'], dtype=torch.float32).to(DEVICE)
                actions = torch.tensor(demo['actions'], dtype=torch.long).to(DEVICE)

                # Forward pass with activation tracking
                self.mentor_tracker.clear_cache()
                mentor_outputs = self.mentor(states)
                activations = self.mentor_tracker.get_activations()

                # Compute behavior cloning loss
                bc_loss = F.cross_entropy(mentor_outputs['policy_logits'], actions)

                # Backward pass to get gradients for importance computation
                self.mentor.optimizer.zero_grad()
                bc_loss.backward(retain_graph=True)

                # Compute activation importance
                importance_scores = self.pathway_analyzer.compute_activation_importance(
                    activations, demo['performance_score'], method='gradient_based'
                )

                epoch_activations.append(activations)
                epoch_importance.append(importance_scores)

                # Update mentor
                self.mentor.optimizer.step()

            activation_sequences.extend(epoch_activations)
            importance_sequences.extend(epoch_importance)

            if epoch % 10 == 0:
                avg_loss = sum(F.cross_entropy(
                    self.mentor(torch.tensor(demo['states'], dtype=torch.float32).to(DEVICE))['policy_logits'],
                    torch.tensor(demo['actions'], dtype=torch.long).to(DEVICE)).item()
                               for demo in demonstrations) / len(demonstrations)
                self.logger.log(f"Behavior cloning epoch {epoch}, avg loss: {avg_loss:.4f}")

        # Build activation graph and identify critical pathways
        self.logger.log("Analyzing critical activation pathways...")
        activation_graph = self.pathway_analyzer.build_activation_graph(
            activation_sequences, importance_sequences
        )

        critical_pathways = self.pathway_analyzer.identify_critical_pathways(
            activation_graph, method='spectral_clustering'
        )

        # Extract activation signatures
        self.critical_signatures = self.signature_extractor.extract_signatures(
            critical_pathways, activation_sequences, target_dim=64
        )

        self.logger.log(f"Identified {len(self.critical_signatures)} critical activation signatures")
        self.logger.log(
            f"Average signature importance: {np.mean([sig.importance_score for sig in self.critical_signatures]):.4f}")

    def _generate_synthetic_demonstrations(self):
        """Generate synthetic demonstrations using random policy (fallback)"""
        self.logger.log("Generating synthetic demonstrations...")

        demo_collector = self.activation_pipeline['demonstration_collector']

        for episode in range(10):  # Generate 10 demo episodes
            obs, _ = self.env.reset()
            states, actions = [], []
            total_reward = 0

            for step in range(200):
                action = np.random.randint(0, self.env.num_actions)
                next_obs, reward, terminated, truncated, _ = self.env.step([[action]], [0.5])

                states.append(obs.copy() if hasattr(obs, 'copy') else obs)
                actions.append(action)
                total_reward += reward[0]

                obs = next_obs
                if terminated[0] or truncated[0]:
                    break

            performance_score = min(1.0, total_reward / 100.0)  # Normalize
            demo_collector.collect_demonstration(states, actions, performance_score)

        # Extract critical pathways from synthetic demos (simplified)
        self.critical_signatures = [
            self.signature_extractor.extract_signatures(
                [{'dummy_pathway'}], [{}], target_dim=64
            )[0] if self.signature_extractor.extract_signatures([{'dummy_pathway'}], [{}], target_dim=64) else None
        ]
        self.critical_signatures = [sig for sig in self.critical_signatures if sig is not None]

    def _update_training_phase(self):
        """Update training phase based on timestep"""
        if self.timestep < self.phase_transitions['human_cloning']:
            self.training_phase = 'human_cloning'
        elif self.timestep < self.phase_transitions['focused_distillation']:
            self.training_phase = 'focused_distillation'
        else:
            self.training_phase = 'standard_training'

    def _train_models_focused(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Training with focused distillation using critical pathways"""
        batch_data = rollout_data['batch_data']

        if not batch_data or not self.critical_signatures:
            return self._train_models_standard(rollout_data)

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

        # Enhanced student training with focused distillation
        for epoch in range(TRAINING_CONFIG['num_ppo_epochs']):

            # Create mini-batches
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Extract mini-batch
                mini_batch_states = batch_data['states'][batch_indices]

                # Get outputs with activation tracking
                self.mentor_tracker.clear_cache()
                self.student_tracker.clear_cache()

                mentor_outputs = self.mentor(mini_batch_states)
                student_outputs = self.student(mini_batch_states)

                mentor_activations = self.mentor_tracker.get_activations()
                student_activations = self.student_tracker.get_activations()

                # Compute focused distillation loss
                focused_losses = self.focused_distillation_loss(
                    student_outputs, mentor_outputs,
                    student_activations, mentor_activations,
                    self.critical_signatures
                )

                # Add standard RL losses
                rl_losses = self._compute_rl_losses(
                    student_outputs,
                    batch_data['actions'][batch_indices],
                    returns[batch_indices],
                    advantages[batch_indices],
                    batch_data['log_probs'][batch_indices],
                    batch_data['values'][batch_indices]
                )

                # Combined loss
                total_loss = (
                        0.6 * focused_losses['total_focused'] +
                        0.4 * rl_losses['total_rl']
                )

                # Optimize student
                self.student.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), TRAINING_CONFIG['max_grad_norm'])
                self.student.optimizer.step()

                # Collect metrics
                for key, value in focused_losses.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value.item())

                for key, value in rl_losses.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value.item())

                training_metrics['total_loss'] = training_metrics.get('total_loss', []) + [total_loss.item()]

        # Average metrics
        for key in training_metrics:
            training_metrics[key] = np.mean(training_metrics[key])

        return training_metrics

    def _train_models_standard(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """Standard training without focused distillation"""
        # Use the original training method from main.py
        batch_data = rollout_data['batch_data']

        if not batch_data:
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

        # Standard PPO training
        for epoch in range(TRAINING_CONFIG['num_ppo_epochs']):

            # Create mini-batches
            batch_size = TRAINING_CONFIG['batch_size']
            num_samples = len(batch_data['states'])
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Extract mini-batch
                mini_batch = {
                    'states': batch_data['states'][batch_indices],
                    'actions': batch_data['actions'][batch_indices],
                    'old_log_probs': batch_data['log_probs'][batch_indices],
                    'values': batch_data['values'][batch_indices],
                    'advantages': advantages[batch_indices],
                    'returns': returns[batch_indices]
                }

                # Train with standard distillation
                metrics = self.distillation_trainer.train_step(
                    states=mini_batch['states'],
                    actions=mini_batch['actions'],
                    rewards=mini_batch['returns'],
                    advantages=mini_batch['advantages'],
                    old_log_probs=mini_batch['old_log_probs'],
                    values=mini_batch['values']
                )

                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value)

        # Average metrics
        for key in training_metrics:
            training_metrics[key] = np.mean(training_metrics[key])

        return training_metrics

    def _compute_rl_losses(self, outputs: Dict[str, torch.Tensor],
                           actions: torch.Tensor, returns: torch.Tensor,
                           advantages: torch.Tensor, old_log_probs: torch.Tensor,
                           old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute standard PPO losses"""
        # Policy loss
        dist = torch.distributions.Categorical(logits=outputs['primary_logits'])
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - TRAINING_CONFIG['clip_ratio'],
                            1 + TRAINING_CONFIG['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = outputs['value'].squeeze()
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss * TRAINING_CONFIG['value_coef'],
            'entropy': -entropy * STUDENT_CONFIG['entropy_coef'],
            'total_rl': policy_loss + value_loss * TRAINING_CONFIG['value_coef'] -
                        entropy * STUDENT_CONFIG['entropy_coef'],
        }

    def _load_human_demonstrations(self, demo_path: str) -> List[Dict]:
        """Load human demonstrations from file"""
        # Placeholder implementation - in practice, load from structured format
        demonstrations = []

        try:
            # Example loading logic (adapt based on your data format)
            import pickle
            with open(demo_path, 'rb') as f:
                demo_data = pickle.load(f)

            for demo in demo_data:
                demonstrations.append({
                    'states': demo['states'],
                    'actions': demo['actions'],
                    'performance_score': demo.get('performance_score', 1.0),
                    'length': len(demo['states'])
                })

        except Exception as e:
            self.logger.log(f"Error loading demonstrations: {e}")
            return []

        self.logger.log(f"Loaded {len(demonstrations)} human demonstrations")
        return demonstrations

    def _collect_rollout(self, initial_states: torch.Tensor) -> Dict[str, Any]:
        """Enhanced rollout collection (uses original method with activation tracking when needed)"""
        # Use the original rollout method from main.py but with enhanced logging
        self.experience_collector.reset()
        states = initial_states.clone()

        rollout_rewards = []
        mentor_query_count = 0

        for step in range(TRAINING_CONFIG['rollout_steps']):

            # === STUDENT ACTION SELECTION ===
            student_actions, student_info = self.student.act(states)

            # === MENTOR QUERYING ===
            mentor_advice = None
            should_query_mentor = any(info.get('should_query_mentor', False) for info in student_info) if isinstance(
                student_info, list) else student_info.get('should_query_mentor', False)

            if should_query_mentor:
                mentor_advice = self._query_mentor(states)
                mentor_query_count += 1

                # Analyze agreement
                if mentor_advice:
                    agreement = analyze_mentor_student_agreement(
                        mentor_advice.actions[0] if mentor_advice.actions else 0,
                        student_actions[0] if isinstance(student_actions[0], list) else [student_actions[0]]
                    )

            # === ENVIRONMENT STEP ===
            # Process student actions for environment
            if not isinstance(student_actions[0], list):
                # Single environment case
                action_lists = [student_actions]
            else:
                action_lists = student_actions

            uncertainties = [student_info['uncertainty']['total']] if not isinstance(student_info, list) else [
                info['uncertainty']['total'] for info in student_info]

            next_observations, rewards, terminated, truncated, infos = self.env.step(
                action_lists, uncertainties
            )

            # Apply reward shaping
            shaped_rewards = []
            for i, (obs, act, rew, done, info) in enumerate(
                    zip(next_observations, action_lists, rewards, terminated | truncated, infos)):
                if isinstance(act, list):
                    act = act[0]  # Use primary action for shaping
                shaped_rew = self.reward_shaper.shape_reward(obs, act, rew, done, info)
                shaped_rewards.append(shaped_rew)

            rewards = np.array(shaped_rewards)

            # === EXPERIENCE COLLECTION ===
            self._collect_experience_step(
                states, action_lists, rewards, next_observations,
                terminated, truncated, student_info, mentor_advice
            )

            # === STATE UPDATE ===
            states = self.env.get_state_tensor(next_observations)

            # Track episode completions
            for info in infos:
                if 'episode_reward' in info:
                    rollout_rewards.append(info['episode_reward'])
                    self.episode_count += 1

            self.timestep += ENV_CONFIG['num_envs']

        return {
            'rollout_rewards': rollout_rewards,
            'mentor_queries': mentor_query_count,
            'batch_data': self.experience_collector.get_batch_tensors()
        }

    def _collect_experience_step(self, states: torch.Tensor, actions: List[List[int]],
                                 rewards: np.ndarray, next_states: np.ndarray,
                                 terminated: np.ndarray, truncated: np.ndarray,
                                 student_info: Any, mentor_advice: Any):
        """Enhanced experience collection (uses original method)"""

        # Get student outputs for log probs and values
        with torch.no_grad():
            student_outputs = self.student(states)

        # Extract values and log probs
        if isinstance(student_info, list):
            values = torch.stack([torch.tensor(info['value']) for info in student_info])
            uncertainties = [info['uncertainty'] for info in student_info]
        else:
            values = torch.tensor([student_info['value']])
            uncertainties = [student_info['uncertainty']]

        # Process actions for log prob computation
        processed_actions = []
        log_probs = []

        for i, action_list in enumerate(actions):
            if isinstance(action_list, list):
                primary_action = action_list[0]
            else:
                primary_action = action_list

            processed_actions.append(primary_action)

            # Compute log prob for primary action
            logits = student_outputs['primary_logits'][i] if student_outputs['primary_logits'].dim() > 1 else \
            student_outputs['primary_logits']
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(primary_action))
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)

        # Add to experience collector
        self.experience_collector.add(
            state=states.cpu().numpy(),
            action=np.array(processed_actions),
            reward=rewards,
            next_state=next_states,
            done=terminated | truncated,
            log_prob=log_probs,
            value=values,
            uncertainty=uncertainties[0] if len(uncertainties) == 1 else uncertainties,
            mentor_advice=mentor_advice
        )

        # Check for completed episodes
        completed_episodes = self.experience_collector.get_completed_episodes()
        for episode_experiences, episode_reward in completed_episodes:
            self.trajectory_buffer.add_trajectory(episode_experiences, episode_reward)

            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.logger.log(f"New best reward: {episode_reward:.2f}")

    # Include all other methods from the original main.py
    def _query_mentor(self, states: torch.Tensor) -> Any:
        """Query mentor for advice"""
        with torch.no_grad():
            # Query mentor for the first state in batch
            state = states[0] if states.dim() > 1 else states
            advice = self.mentor.get_advice(state, verbose=False)
            return advice

    def _evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation"""
        self.logger.log("Running evaluation...")

        eval_env = create_environment(num_envs=1)

        eval_rewards = []
        eval_lengths = []
        eval_mentor_queries = []
        pathway_activations = []

        for episode in range(10):  # 10 evaluation episodes
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            mentor_queries = 0

            done = False
            while not done:
                state = eval_env.get_state_tensor(obs)

                # Student action (deterministic)
                actions, info = self.student.act(state, deterministic=True)

                # Query mentor if high uncertainty
                if info['should_query_mentor']:
                    mentor_queries += 1

                # Track activation patterns during evaluation (for analysis)
                if self.training_phase == 'focused_distillation' and self.critical_signatures:
                    self.student_tracker.clear_cache()
                    _ = self.student(state)
                    activations = self.student_tracker.get_activations()
                    pathway_activations.append(activations)

                # Environment step
                obs, reward, terminated, truncated, _ = eval_env.step([actions], [info['uncertainty']['total']])

                episode_reward += reward[0]
                episode_length += 1
                done = terminated[0] or truncated[0]

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_mentor_queries.append(mentor_queries)

        eval_env.close()

        eval_metrics = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_length_mean': np.mean(eval_lengths),
            'eval_mentor_queries_mean': np.mean(eval_mentor_queries),
            'training_phase': self.training_phase,
        }

        # Add pathway-specific metrics if available
        if pathway_activations:
            pathway_consistency = self._analyze_pathway_consistency(pathway_activations)
            eval_metrics['pathway_consistency'] = pathway_consistency

        return eval_metrics

    def _analyze_pathway_consistency(self, pathway_activations: List[Dict]) -> float:
        """Analyze consistency of critical pathway activations"""
        if not self.critical_signatures or not pathway_activations:
            return 0.0

        # Simplified consistency measure
        consistencies = []

        for signature in self.critical_signatures:
            signature_activations = []

            for activations in pathway_activations:
                # Extract pathway activations (simplified)
                pathway_acts = self.focused_distillation_loss._extract_pathway_activations(
                    activations, signature.neuron_indices, signature.layer_id
                )
                if pathway_acts.numel() > 0:
                    signature_activations.append(pathway_acts.flatten()[:16])  # Truncate for consistency

            if len(signature_activations) > 1:
                signature_tensor = torch.stack(signature_activations)
                consistency = 1.0 - torch.std(signature_tensor, dim=0).mean().item()
                consistencies.append(max(0.0, consistency))

        return np.mean(consistencies) if consistencies else 0.0

    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Enhanced logging with phase information"""
        for key, value in metrics.items():
            self.logger.log_step(self.timestep, {f"{self.training_phase}_{key}": value})

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Enhanced evaluation logging"""
        for key, value in metrics.items():
            self.logger.log_step(self.timestep, {key: value})

    def _log_progress(self, elapsed_time: float):
        """Enhanced progress logging"""
        stats = self.logger.get_statistics()

        self.logger.log(
            f"Phase: {self.training_phase} | "
            f"Timestep: {self.timestep:,} | "
            f"Episodes: {self.episode_count:,} | "
            f"Avg Reward (100): {stats['avg_reward_100']:.2f} | "
            f"Best Reward: {self.best_reward:.2f} | "
            f"Critical Signatures: {len(self.critical_signatures)} | "
            f"Time: {elapsed_time:.1f}s"
        )

        # Plot training curves
        if self.timestep % (LOGGING_CONFIG['log_interval'] * 10) == 0:
            self.logger.plot_training_curves()

    def _save_checkpoint(self):
        """Enhanced checkpointing with activation signatures"""
        models = {
            'mentor': self.mentor,
            'student': self.student,
        }

        optimizers = {
            'student': self.student.optimizer,
            'mentor': self.mentor.optimizer,
        }

        # Save additional activation-based data
        additional_data = {
            'critical_signatures': self.critical_signatures,
            'training_phase': self.training_phase,
            'pathway_analyzer_state': self.pathway_analyzer.__dict__ if hasattr(self.pathway_analyzer,
                                                                                '__dict__') else {},
        }

        self.logger.save_checkpoint(models, optimizers, self.timestep)

        # Save activation-specific data
        activation_checkpoint_path = os.path.join(self.logger.log_dir, f'activation_data_{self.timestep}.pt')
        torch.save(additional_data, activation_checkpoint_path)

        self.logger.log(f"Saved activation data at timestep {self.timestep}")

    def _final_evaluation(self):
        """Enhanced final evaluation"""
        self.logger.log("Running final evaluation...")

        final_metrics = self._evaluate()
        trajectory_stats = self.trajectory_buffer.get_statistics()

        self.logger.log("=== ENHANCED FINAL RESULTS ===")
        self.logger.log(f"Final Training Phase: {self.training_phase}")
        self.logger.log(f"Critical Activation Signatures: {len(self.critical_signatures)}")
        self.logger.log(
            f"Final Average Reward: {final_metrics['eval_reward_mean']:.2f} ± {final_metrics['eval_reward_std']:.2f}")
        self.logger.log(f"Best Episode Reward: {self.best_reward:.2f}")
        self.logger.log(f"Success Rate: {trajectory_stats['success_rate']:.2%}")
        self.logger.log(f"Average Episode Length: {final_metrics['eval_length_mean']:.1f}")
        self.logger.log(f"Mentor Queries per Episode: {final_metrics['eval_mentor_queries_mean']:.1f}")

        if 'pathway_consistency' in final_metrics:
            self.logger.log(f"Pathway Consistency Score: {final_metrics['pathway_consistency']:.3f}")

        # Save final models
        self._save_checkpoint()


def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Revolutionary AI Pipeline with Activation-Based Distillation')
    parser.add_argument('--log_dir', type=str, default='logs', help='Logging directory')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--env_name', type=str, default=None, help='Environment name (overrides config)')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--human_demos_path', type=str, default=None, help='Path to human demonstration data')
    parser.add_argument('--skip_behavior_cloning', action='store_true', help='Skip behavior cloning phase')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create and run enhanced pipeline
    pipeline = EnhancedRevolutionaryPipeline(args)

    if args.load_checkpoint:
        # TODO: Implement enhanced checkpoint loading
        pipeline.logger.log(f"Loading checkpoint: {args.load_checkpoint}")

    if args.eval_only:
        pipeline._final_evaluation()
    else:
        pipeline.train()


if __name__ == "__main__":
    main()