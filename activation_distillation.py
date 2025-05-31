# activation_distillation.py
"""
Advanced Activation-Based Knowledge Distillation with Human Behavior Cloning
Revolutionary approach using graph theory and critical pathway analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union  # MODIFIED: Added Any
import networkx as nx
from scipy.stats import entropy
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import SpectralClustering
# import matplotlib.pyplot as plt # Not used directly in provided snippet
from dataclasses import dataclass

from config import DEVICE, MENTOR_CONFIG, STUDENT_CONFIG, DISTILLATION_CONFIG, ENV_CONFIG
from mathematical_framework import PathwayImportanceOptimizer  # InformationTheoreticAnalyzer not directly used here
from distillation import FeatureProjector  # MODIFIED: Import FeatureProjector


@dataclass
class ActivationSignature:  # Unchanged
    layer_id: int
    pathway_node_ids: Set[str]
    neuron_indices: List[int]
    importance_score: float
    activation_pattern: torch.Tensor
    temporal_dynamics: Optional[torch.Tensor] = None
    causal_influence: float = 0.0


class HumanDemonstrationCollector:  # Unchanged from previous full version
    def __init__(self, env_name: str, multimodal_inputs: bool = True):
        self.env_name = env_name;
        self.multimodal_inputs = multimodal_inputs
        self.demonstrations = [];
        self.multimodal_data = {}

    def collect_demonstration(self, states: List[np.ndarray], actions: List[int], performance_score: float,
                              video_frames: Optional[List[np.ndarray]] = None, audio_data: Optional[np.ndarray] = None,
                              expert_commentary: Optional[List[str]] = None) -> Dict:
        demo = {'states': states, 'actions': actions, 'performance_score': performance_score, 'length': len(states),
                'success_rate': performance_score > 0.8}  # Example success condition
        if self.multimodal_inputs:
            if video_frames is not None: demo['video_frames'] = video_frames
            if audio_data is not None: demo['audio_data'] = audio_data
            if expert_commentary is not None: demo['commentary'] = expert_commentary
        self.demonstrations.append(demo);
        return demo

    def get_successful_demonstrations(self, min_performance: float = 0.8) -> List[Dict]:
        return [d for d in self.demonstrations if d['performance_score'] >= min_performance]


class ActivationTracker:  # Unchanged from previous full version
    def __init__(self, model: nn.Module):
        self.model = model;
        self.activation_cache = {};
        self.hooks = [];
        self.layer_names = []
        self.setup_hooks()

    def setup_hooks(self):
        def make_hook(name):
            def hook(module, input, output):  # module, input, output standard hook signature
                if isinstance(output, torch.Tensor):
                    self.activation_cache[name] = output.detach().clone()
                elif isinstance(output,
                                (list, tuple)):  # Handle outputs that are lists/tuples (e.g. from MultiActionHead)
                    self.activation_cache[name] = [o.detach().clone() if isinstance(o, torch.Tensor) else o for o in
                                                   output]

            return hook

        for name, module in self.model.named_modules():
            # Hook relevant layers, e.g. Linear, Conv, Attention. Add more as needed.
            if isinstance(module,
                          (nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.LayerNorm)):
                self.hooks.append(module.register_forward_hook(make_hook(name)));
                self.layer_names.append(name)

    def get_activations(self) -> Dict[str, Any]:
        return self.activation_cache.copy()  # Return Any as cache can have lists

    def clear_cache(self):
        self.activation_cache.clear()

    def remove_hooks(self):
        for hook in self.hooks: hook.remove()


class CriticalPathwayAnalyzer:  # Unchanged from previous full version, simplified for brevity
    def __init__(self, model_structure: Dict):
        self.model_structure = model_structure; self.activation_graph: Optional[
            nx.Graph] = None; self.pathway_importance: Dict = {}

    def compute_activation_importance(self, activations: Dict[str, torch.Tensor],
                                      target_performance: Union[float, torch.Tensor], method: str = 'gradient_based') -> \
    Dict[str, torch.Tensor]:
        dev = list(activations.values())[0].device if activations and activations.values() else DEVICE
        if not isinstance(target_performance, torch.Tensor): target_performance = torch.tensor(target_performance,
                                                                                               device=dev)
        importance_scores = {}
        for layer_name, activation in activations.items():
            if activation is None or not isinstance(activation, torch.Tensor) or activation.numel() == 0: continue
            if method == 'gradient_based':
                score = torch.abs(
                    activation.grad) if activation.requires_grad and activation.grad is not None else torch.zeros_like(
                    activation)
            elif method == 'variance_based':
                score = torch.var(activation, dim=0, keepdim=True).expand_as(
                    activation) if activation.numel() > 0 else torch.zeros_like(activation)
            else:
                score = torch.zeros_like(activation)
            importance_scores[layer_name] = score
        return importance_scores

    def build_activation_graph(self, activations_sequence: List[Dict[str, torch.Tensor]],
                               importance_scores_sequence: List[Dict[str, torch.Tensor]]) -> nx.Graph:
        G = nx.Graph();
        self.activation_graph = G;  # Simplified
        # Add nodes with average importance
        # Add edges based on temporal correlation (simplified)
        print("Warning: CriticalPathwayAnalyzer.build_activation_graph is simplified in this version.")
        return G

    def identify_critical_pathways(self, activation_graph: Optional[nx.Graph], method: str = 'spectral_clustering') -> \
    List[Set[str]]:
        if activation_graph is None or activation_graph.number_of_nodes() == 0: return []
        print("Warning: CriticalPathwayAnalyzer.identify_critical_pathways is simplified in this version.")
        # Fallback: return top N important nodes as individual "pathways"
        if activation_graph.number_of_nodes() > 0:
            # Ensure nodes have 'importance' attribute if relying on it
            try:
                top_nodes = sorted(activation_graph.nodes(data=True), key=lambda x: x[1].get('importance', 0),
                                   reverse=True)[:5]
                return [{node_data[0]} for node_data in top_nodes if node_data]
            except:
                return []  # If sorting fails
        return []


class ActivationSignatureExtractor:  # Unchanged from previous full version, simplified for brevity
    def __init__(self, compression_method: str = 'pca', pathway_optimizer: Optional[PathwayImportanceOptimizer] = None,
                 pathway_analyzer: Optional[CriticalPathwayAnalyzer] = None):
        self.compression_method = compression_method;
        self.compressors: Dict[str, Any] = {};
        self.pathway_optimizer = pathway_optimizer;
        self.pathway_analyzer = pathway_analyzer

    def extract_signatures(self, critical_pathways: List[Set[str]], activations_sequence: List[Dict[str, torch.Tensor]],
                           performance_scores_sequence: Optional[List[float]] = None, target_dim: int = 64) -> List[
        ActivationSignature]:
        if not critical_pathways or not activations_sequence: return []
        signatures = []
        for i, pathway_nodes in enumerate(critical_pathways):
            if not pathway_nodes: continue
            # Simplified: create dummy signature
            dummy_pattern = torch.randn(target_dim, device=DEVICE)
            sig = ActivationSignature(layer_id=i, pathway_node_ids=pathway_nodes,
                                      neuron_indices=list(range(min(5, len(pathway_nodes)))),
                                      importance_score=np.random.rand(), activation_pattern=dummy_pattern)
            signatures.append(sig)
        print(
            f"Warning: ActivationSignatureExtractor.extract_signatures is simplified. Extracted {len(signatures)} dummy signatures.")
        return signatures


# MODIFIED: FocusedDistillationLoss is now an nn.Module
class FocusedDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0,
                 alpha: float = 0.7):  # alpha from DISTILLATION_CONFIG used for overall balance
        super().__init__()
        self.temperature = temperature
        self.alpha_focused_balance = alpha  # Internal balance for this loss's components

        self.student_aux_sources: List[Dict[str, Any]] = []
        # Projectors are now part of this nn.Module, so their params are discoverable
        self.aux_feature_projectors_focused = nn.ModuleDict()

    def set_student_aux_sources(self, student_aux_sources: List[Dict[str, Any]], student_target_feature_dim: int):
        self.student_aux_sources = student_aux_sources
        # Clear and rebuild projectors
        # First, convert self.aux_feature_projectors_focused to a standard dict to allow item deletion while iterating
        current_projectors_keys = list(self.aux_feature_projectors_focused.keys())
        for key in current_projectors_keys:
            del self.aux_feature_projectors_focused[key]

        for i, source_config in enumerate(self.student_aux_sources):
            name = source_config.get('name', f"aux_source_{i}")
            if 'features' in source_config.get('transfer_targets', []) and \
                    source_config.get('feature_dim') != student_target_feature_dim:

                aux_feature_dim = source_config.get('feature_dim')
                if not isinstance(aux_feature_dim, int) or aux_feature_dim <= 0:
                    print(
                        f"Warning (FocusedLoss): Invalid 'feature_dim' ({aux_feature_dim}) for aux source {name}. Skipping projector.")
                    continue

                projector = FeatureProjector(aux_feature_dim,
                                             student_target_feature_dim)  # DEVICE handled by .to(DEVICE) later
                proj_module_name = f"aux_focused_proj_{name.replace(' ', '_').replace('.', '_')}"
                self.aux_feature_projectors_focused[proj_module_name] = projector
                print(f"FocusedDistillationLoss: Initialized FeatureProjector '{proj_module_name}' for '{name}'.")
        self.to(DEVICE)  # Ensure new projectors are moved to the correct device

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                mentor_outputs: Dict[str, torch.Tensor],
                student_activations: Dict[str, torch.Tensor],
                mentor_activations: Dict[str, torch.Tensor],
                critical_signatures: List[ActivationSignature],
                states: torch.Tensor  # MODIFIED: Added states
                ) -> Dict[str, torch.Tensor]:
        losses = {}
        output_device = student_outputs['primary_logits'].device

        # 1. Traditional policy distillation (from main mentor)
        policy_loss = F.kl_div(
            F.log_softmax(student_outputs['primary_logits'] / self.temperature, dim=-1),
            F.softmax(mentor_outputs['policy_logits'].nan_to_num(0.0, posinf=1e6, neginf=-1e6) / self.temperature,
                      dim=-1),
            reduction='batchmean', log_target=False
        ) * (self.temperature ** 2)
        losses['policy_distill'] = policy_loss

        # 2. Value distillation (from main mentor)
        if 'value' in student_outputs and 'value' in mentor_outputs:
            losses['value_distill'] = F.mse_loss(student_outputs['value'], mentor_outputs['value'])
        else:
            losses['value_distill'] = torch.tensor(0.0, device=output_device)

        # 3. Critical pathway distillation (from main mentor)
        losses['critical_pathway'] = self._compute_pathway_distillation_loss(
            student_activations, mentor_activations, critical_signatures
        )

        # 4. Signature matching loss (from main mentor's derived signatures)
        losses['signature_match'] = self._compute_signature_matching_loss(
            student_activations, critical_signatures
        )

        # --- NEW: Distillation from auxiliary student sources ---
        total_aux_focused_loss = torch.tensor(0.0, device=output_device)
        if hasattr(self, 'student_aux_sources') and self.student_aux_sources:
            for i, aux_source_config in enumerate(self.student_aux_sources):
                aux_model = aux_source_config['model']
                aux_weight = aux_source_config['weight']
                aux_transfer_targets = aux_source_config.get('transfer_targets', [])
                source_name = aux_source_config.get('name', f"aux_source_{i}")
                proj_module_name = f"aux_focused_proj_{source_name.replace(' ', '_').replace('.', '_')}"

                with torch.no_grad():
                    aux_outputs = aux_model(states)

                current_source_loss = torch.tensor(0.0, device=output_device)
                # Policy Logits
                if 'policy_logits' in aux_transfer_targets and 'policy_logits' in aux_outputs:
                    aux_policy_loss_term = F.kl_div(
                        F.log_softmax(student_outputs['primary_logits'] / self.temperature, dim=-1),
                        F.softmax(aux_outputs['policy_logits'] / self.temperature, dim=-1),
                        reduction='batchmean', log_target=False
                    ) * (self.temperature ** 2)
                    current_source_loss += aux_policy_loss_term
                # Features
                if 'features' in aux_transfer_targets and 'features' in aux_outputs and 'features' in student_outputs:
                    aux_projector = self.aux_feature_projectors_focused.get(proj_module_name)
                    source_features = aux_outputs['features']
                    target_student_features = student_outputs['features']
                    projected_aux_s_features = None
                    if aux_projector:
                        projected_aux_s_features = aux_projector(source_features)
                    elif source_features.shape == target_student_features.shape:
                        projected_aux_s_features = source_features

                    if projected_aux_s_features is not None:
                        aux_feature_loss_term = F.mse_loss(target_student_features, projected_aux_s_features)
                        current_source_loss += aux_feature_loss_term * DISTILLATION_CONFIG.get(
                            'feature_matching_weight', 0.1)
                # Value
                if 'value' in aux_transfer_targets and 'value' in aux_outputs and 'value' in student_outputs:
                    aux_value_loss_term = F.mse_loss(student_outputs['value'], aux_outputs['value'])
                    current_source_loss += aux_value_loss_term * DISTILLATION_CONFIG.get('value_distill_weight', 0.5)

                total_aux_focused_loss += current_source_loss * aux_weight

        losses['aux_focused_distill'] = total_aux_focused_loss
        # --- END NEW ---

        # Weighting for focused loss components (excluding aux, which has its own weight)
        # self.alpha_focused_balance is for policy_distill from main mentor vs other main mentor losses
        total_focused_val = (
                self.alpha_focused_balance * losses['policy_distill'] +
                (1.0 - self.alpha_focused_balance) * (
                        0.3 * losses['value_distill'] +  # Example internal weights
                        0.4 * losses['critical_pathway'] +
                        0.2 * losses['signature_match']
                ) +
                losses['aux_focused_distill']  # Aux losses are added with their own pre-defined weights
        )
        losses['total_focused'] = total_focused_val
        return losses

    # Helper methods for FocusedDistillationLoss (unchanged from previous full version)
    def _extract_activations_for_signature(self, activations: Dict[str, Any],
                                           signature: ActivationSignature) -> torch.Tensor:
        pathway_neuron_activations = []
        # Determine device from the first available tensor, or default
        device_to_use = next((t.device for t in activations.values() if isinstance(t, torch.Tensor) and t.numel() > 0),
                             DEVICE)

        if not hasattr(signature, 'pathway_node_ids') or not signature.pathway_node_ids:
            return torch.empty(0, device=device_to_use)

        for node_id in signature.pathway_node_ids:
            try:
                layer_name, neuron_idx_str = node_id.rsplit('_', 1)
                neuron_idx = int(neuron_idx_str)
                layer_acts_item = activations.get(layer_name)

                # Handle cases where layer_acts_item might be a list (e.g. from MultiActionHead in student)
                layer_acts = None
                if isinstance(layer_acts_item, list):  # If it's a list, try to get the primary tensor or first tensor
                    if layer_acts_item and isinstance(layer_acts_item[0], torch.Tensor):
                        layer_acts = layer_acts_item[0]  # Heuristic: use the first tensor
                elif isinstance(layer_acts_item, torch.Tensor):
                    layer_acts = layer_acts_item

                if layer_acts is not None and layer_acts.numel() > 0 and neuron_idx < layer_acts.shape[-1]:
                    neuron_act = layer_acts[..., neuron_idx].unsqueeze(-1)  # (Batch, 1)
                    pathway_neuron_activations.append(neuron_act)
            except Exception as e:
                # print(f"Debug: Error extracting node {node_id}: {e}")
                continue

        if pathway_neuron_activations:
            try:
                return torch.cat(pathway_neuron_activations, dim=-1)  # (Batch, Num_Total_Neurons_In_Pathway)
            except Exception as e:
                # print(f"Debug: Error concatenating pathway activations: {e}")
                return torch.empty(0, device=device_to_use)
        return torch.empty(0, device=device_to_use)

    def _compute_pathway_distillation_loss(self, student_activations: Dict[str, Any],
                                           mentor_activations: Dict[str, Any],
                                           critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        pathway_losses = []
        device_to_use = next(
            (t.device for t in student_activations.values() if isinstance(t, torch.Tensor) and t.numel() > 0), DEVICE)

        for signature in critical_signatures:
            if not signature: continue
            pathway_student_acts = self._extract_activations_for_signature(student_activations, signature)
            pathway_mentor_acts = self._extract_activations_for_signature(mentor_activations, signature)

            if pathway_student_acts.numel() > 0 and pathway_mentor_acts.numel() > 0 and \
                    pathway_student_acts.shape == pathway_mentor_acts.shape:
                pathway_loss = F.mse_loss(pathway_student_acts, pathway_mentor_acts)
                pathway_losses.append(pathway_loss * signature.importance_score)

        if pathway_losses: return torch.stack(pathway_losses).mean()
        return torch.tensor(0.0, device=device_to_use)

    def _compute_signature_matching_loss(self, student_activations: Dict[str, Any],
                                         critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        signature_losses = []
        device_to_use = next(
            (t.device for t in student_activations.values() if isinstance(t, torch.Tensor) and t.numel() > 0), DEVICE)

        for signature in critical_signatures:
            if not signature or not hasattr(signature, 'activation_pattern'): continue
            current_pathway_student_batch_acts = self._extract_activations_for_signature(student_activations, signature)

            if current_pathway_student_batch_acts.numel() > 0:
                # Compress the batch activations to match the signature's pattern dimension
                # Typically, signature.activation_pattern is 1D.
                # We need to average current_pathway_student_batch_acts over batch dim first.
                if current_pathway_student_batch_acts.dim() > 1:  # (Batch, PathwayFeatures)
                    mean_batch_pathway_acts = current_pathway_student_batch_acts.mean(dim=0)  # (PathwayFeatures,)
                else:  # Already (PathwayFeatures,)
                    mean_batch_pathway_acts = current_pathway_student_batch_acts

                target_dim = signature.activation_pattern.shape[0]
                compressed_current = self._compress_to_signature_space(mean_batch_pathway_acts, target_dim)

                # Ensure signature pattern is on the same device
                sig_pattern_on_device = signature.activation_pattern.to(compressed_current.device)

                signature_loss = F.mse_loss(compressed_current, sig_pattern_on_device)
                signature_losses.append(signature_loss * signature.importance_score)

        if signature_losses: return torch.stack(signature_losses).mean()
        return torch.tensor(0.0, device=device_to_use)

    def _compress_to_signature_space(self, activations: torch.Tensor, target_dim: int) -> torch.Tensor:
        flattened = activations.flatten();
        current_len = len(flattened);
        output_device = activations.device
        if current_len == 0 and target_dim == 0: return torch.empty(0, device=output_device)
        if current_len == 0 and target_dim > 0: return torch.zeros(target_dim, device=output_device)

        if current_len == target_dim:
            return flattened
        elif current_len > target_dim:
            return flattened[:target_dim]
        else:  # current_len < target_dim
            padded = torch.zeros(target_dim, device=output_device);
            padded[:current_len] = flattened;
            return padded


def create_activation_based_distillation_pipeline(
        pathway_optimizer: Optional[PathwayImportanceOptimizer] = None,
        pathway_analyzer_instance: Optional[CriticalPathwayAnalyzer] = None
) -> Dict[str, Any]:  # MODIFIED: Return Any
    analyzer_to_use = pathway_analyzer_instance if pathway_analyzer_instance is not None else CriticalPathwayAnalyzer(
        {})
    # Pass global DISTILLATION_CONFIG values to FocusedDistillationLoss
    focused_loss_module = FocusedDistillationLoss(
        temperature=DISTILLATION_CONFIG['temperature'],
        alpha=DISTILLATION_CONFIG['alpha']  # This alpha is for main mentor policy vs other main mentor losses
    )
    pipeline = {
        'demonstration_collector': HumanDemonstrationCollector(ENV_CONFIG['name'], multimodal_inputs=True),
        'activation_tracker': None,  # To be initialized with a model
        'pathway_analyzer': analyzer_to_use,
        'signature_extractor': ActivationSignatureExtractor(
            compression_method='pca',  # Example
            pathway_optimizer=pathway_optimizer,
            pathway_analyzer=analyzer_to_use
        ),
        'distillation_loss': focused_loss_module,  # This is the FocusedDistillationLoss instance
    }
    return pipeline