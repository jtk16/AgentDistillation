# activation_distillation.py
"""
Advanced Activation-Based Knowledge Distillation with Human Behavior Cloning
Revolutionary approach using graph theory and critical pathway analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from scipy.stats import entropy
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from dataclasses import dataclass

from config import DEVICE, MENTOR_CONFIG, STUDENT_CONFIG


@dataclass
class ActivationSignature:
    """Critical activation pattern identified from human demonstrations"""
    layer_id: int
    neuron_indices: List[int]
    importance_score: float
    activation_pattern: torch.Tensor
    temporal_dynamics: Optional[torch.Tensor] = None
    causal_influence: float = 0.0


class HumanDemonstrationCollector:
    """Collects and processes human demonstrations for mentor training"""

    def __init__(self, env_name: str, multimodal_inputs: bool = True):
        self.env_name = env_name
        self.multimodal_inputs = multimodal_inputs
        self.demonstrations = []
        self.multimodal_data = {}

    def collect_demonstration(self,
                              states: List[np.ndarray],
                              actions: List[int],
                              performance_score: float,
                              video_frames: Optional[List[np.ndarray]] = None,
                              audio_data: Optional[np.ndarray] = None,
                              expert_commentary: Optional[List[str]] = None) -> Dict:
        """Collect a single human demonstration with multimodal data"""

        demo = {
            'states': states,
            'actions': actions,
            'performance_score': performance_score,
            'length': len(states),
            'success_rate': performance_score > 0.8,  # Threshold for success
        }

        # Multimodal data collection
        if self.multimodal_inputs:
            if video_frames is not None:
                demo['video_frames'] = video_frames
            if audio_data is not None:
                demo['audio_data'] = audio_data
            if expert_commentary is not None:
                demo['commentary'] = expert_commentary

        self.demonstrations.append(demo)
        return demo

    def get_successful_demonstrations(self, min_performance: float = 0.8) -> List[Dict]:
        """Filter for high-quality demonstrations"""
        return [demo for demo in self.demonstrations
                if demo['performance_score'] >= min_performance]


class ActivationTracker:
    """Tracks and analyzes neural activations during behavior cloning"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_cache = {}
        self.hooks = []
        self.layer_names = []
        self.setup_hooks()

    def setup_hooks(self):
        """Setup forward hooks to capture activations"""

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_cache[name] = output.detach().clone()
                elif isinstance(output, (list, tuple)):
                    self.activation_cache[name] = [o.detach().clone() if isinstance(o, torch.Tensor) else o for o in
                                                   output]

            return hook

        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
                self.layer_names.append(name)

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get current activation cache"""
        return self.activation_cache.copy()

    def clear_cache(self):
        """Clear activation cache"""
        self.activation_cache.clear()

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()


class CriticalPathwayAnalyzer:
    """
    Advanced mathematical analysis to identify critical activation pathways
    Uses graph theory, information theory, and spectral analysis
    """

    def __init__(self, model_structure: Dict):
        self.model_structure = model_structure
        self.activation_graph = None
        self.pathway_importance = {}

    def compute_activation_importance(self,
                                      activations: Dict[str, torch.Tensor],
                                      target_performance: float,
                                      method: str = 'gradient_based') -> Dict[str, torch.Tensor]:
        """Compute importance scores for individual activations"""

        importance_scores = {}

        for layer_name, activation in activations.items():
            if method == 'gradient_based':
                # Gradient-based importance: |∇L/∇a_i|
                if activation.requires_grad:
                    score = torch.abs(activation.grad) if activation.grad is not None else torch.zeros_like(activation)
                else:
                    score = torch.zeros_like(activation)

            elif method == 'variance_based':
                # High variance indicates discrimination capability
                score = torch.var(activation, dim=0, keepdim=True).expand_as(activation)

            elif method == 'information_theoretic':
                # Mutual information with task success
                score = self._compute_mutual_information(activation, target_performance)

            elif method == 'causal_influence':
                # Causal effect on downstream layers
                score = self._compute_causal_influence(layer_name, activation)

            else:
                raise ValueError(f"Unknown importance method: {method}")

            importance_scores[layer_name] = score

        return importance_scores

    def _compute_mutual_information(self, activation: torch.Tensor, target: float) -> torch.Tensor:
        """Compute mutual information between activations and target performance"""
        # Discretize activations for MI computation
        activation_np = activation.detach().cpu().numpy()

        # Simple binning approach (can be improved with adaptive methods)
        activation_binned = np.digitize(activation_np, bins=np.linspace(activation_np.min(), activation_np.max(), 10))
        target_binned = int(target * 10)  # Convert performance to discrete bins

        # Compute MI for each neuron
        mi_scores = np.zeros_like(activation_np)
        for i in range(activation_np.shape[-1]):
            neuron_bins = activation_binned[..., i].flatten()
            mi_scores[..., i] = self._mutual_info_discrete(neuron_bins, [target_binned] * len(neuron_bins))

        return torch.tensor(mi_scores, device=activation.device)

    def _mutual_info_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute discrete mutual information"""
        # Joint distribution
        xy = np.histogram2d(x, y, bins=10)[0]
        xy = xy / xy.sum()

        # Marginal distributions
        x_dist = xy.sum(axis=1)
        y_dist = xy.sum(axis=0)

        # MI = H(X) + H(Y) - H(X,Y)
        mi = 0.0
        for i in range(len(x_dist)):
            for j in range(len(y_dist)):
                if xy[i, j] > 0:
                    mi += xy[i, j] * np.log(xy[i, j] / (x_dist[i] * y_dist[j]))

        return mi

    def _compute_causal_influence(self, layer_name: str, activation: torch.Tensor) -> torch.Tensor:
        """Compute causal influence on downstream layers using intervention"""
        # Simplified causal influence computation
        # In practice, this would involve interventional analysis

        # Measure sensitivity of downstream outputs to this layer
        baseline_output = self._get_downstream_output(layer_name, activation)

        # Small perturbation
        perturbed_activation = activation + torch.randn_like(activation) * 0.01
        perturbed_output = self._get_downstream_output(layer_name, perturbed_activation)

        # Causal influence = sensitivity to perturbation
        influence = torch.abs(perturbed_output - baseline_output).mean()

        return torch.full_like(activation, influence.item())

    def _get_downstream_output(self, layer_name: str, activation: torch.Tensor) -> torch.Tensor:
        """Get output from downstream layers (simplified)"""
        # This would require more sophisticated model introspection
        return torch.mean(activation)

    def build_activation_graph(self,
                               activations_sequence: List[Dict[str, torch.Tensor]],
                               importance_scores: List[Dict[str, torch.Tensor]]) -> nx.Graph:
        """Build graph representation of activation dependencies"""

        G = nx.Graph()

        # Add nodes (neurons) with importance as node attributes
        for layer_name in activations_sequence[0].keys():
            layer_importance = torch.stack([scores[layer_name] for scores in importance_scores]).mean(dim=0)

            for neuron_idx in range(layer_importance.shape[-1]):
                node_id = f"{layer_name}_{neuron_idx}"
                importance = layer_importance[..., neuron_idx].mean().item()
                G.add_node(node_id, layer=layer_name, neuron_idx=neuron_idx, importance=importance)

        # Add edges based on correlation/causation
        self._add_graph_edges(G, activations_sequence)

        self.activation_graph = G
        return G

    def _add_graph_edges(self, G: nx.Graph, activations_sequence: List[Dict[str, torch.Tensor]]):
        """Add edges based on activation correlations and layer connectivity"""

        # Compute temporal correlations between neurons
        correlations = self._compute_temporal_correlations(activations_sequence)

        # Add edges for highly correlated neurons
        threshold = 0.7  # Correlation threshold

        for (node1, node2), correlation in correlations.items():
            if abs(correlation) > threshold:
                G.add_edge(node1, node2, weight=abs(correlation), correlation=correlation)

    def _compute_temporal_correlations(self, activations_sequence: List[Dict[str, torch.Tensor]]) -> Dict[
        Tuple[str, str], float]:
        """Compute temporal correlations between all neuron pairs"""
        correlations = {}

        # Flatten activations across time
        flattened_activations = {}
        for layer_name in activations_sequence[0].keys():
            layer_activations = torch.stack([acts[layer_name] for acts in activations_sequence])
            # Shape: [time_steps, batch_size, ...features]
            flattened_activations[layer_name] = layer_activations.flatten(start_dim=1)

        # Compute correlations between neurons
        all_nodes = list(self.activation_graph.nodes()) if self.activation_graph else []

        for i, node1 in enumerate(all_nodes):
            for j, node2 in enumerate(all_nodes[i + 1:], i + 1):
                layer1, neuron1 = node1.rsplit('_', 1)
                layer2, neuron2 = node2.rsplit('_', 1)

                if layer1 in flattened_activations and layer2 in flattened_activations:
                    act1 = flattened_activations[layer1][..., int(neuron1)]
                    act2 = flattened_activations[layer2][..., int(neuron2)]

                    correlation = torch.corrcoef(torch.stack([act1.flatten(), act2.flatten()]))[0, 1].item()
                    correlations[(node1, node2)] = correlation

        return correlations

    def identify_critical_pathways(self,
                                   activation_graph: nx.Graph,
                                   method: str = 'spectral_clustering') -> List[Set[str]]:
        """Identify critical pathways using graph analysis"""

        if method == 'spectral_clustering':
            return self._spectral_pathway_detection(activation_graph)
        elif method == 'centrality_based':
            return self._centrality_pathway_detection(activation_graph)
        elif method == 'minimum_spanning_tree':
            return self._mst_pathway_detection(activation_graph)
        elif method == 'community_detection':
            return self._community_pathway_detection(activation_graph)
        else:
            raise ValueError(f"Unknown pathway detection method: {method}")

    def _spectral_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use spectral clustering to identify activation communities"""

        # Create adjacency matrix
        nodes = list(G.nodes())
        adj_matrix = nx.adjacency_matrix(G, nodelist=nodes).toarray()

        # Apply spectral clustering
        n_clusters = min(8, len(nodes) // 10)  # Adaptive number of clusters
        if n_clusters > 1:
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = clustering.fit_predict(adj_matrix)

            # Group nodes by cluster
            pathways = []
            for cluster_id in range(n_clusters):
                pathway = {nodes[i] for i in range(len(nodes)) if labels[i] == cluster_id}
                if len(pathway) > 2:  # Minimum pathway size
                    pathways.append(pathway)

            return pathways
        else:
            return [{node} for node in nodes[:5]]  # Fallback

    def _centrality_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use centrality measures to identify critical pathways"""

        # Compute various centrality measures
        betweenness = nx.betweenness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        pagerank = nx.pagerank(G)

        # Combine centrality scores
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                    0.4 * betweenness.get(node, 0) +
                    0.3 * eigenvector.get(node, 0) +
                    0.3 * pagerank.get(node, 0)
            )

        # Select top nodes and their neighborhoods
        top_nodes = sorted(combined_centrality.keys(),
                           key=lambda x: combined_centrality[x], reverse=True)[:10]

        pathways = []
        for central_node in top_nodes:
            # Include node and its immediate neighbors
            neighborhood = {central_node} | set(G.neighbors(central_node))
            pathways.append(neighborhood)

        return pathways

    def _mst_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use minimum spanning tree to find essential connections"""

        # Create MST based on importance-weighted edges
        mst = nx.minimum_spanning_tree(G, weight='weight')

        # Find connected components after removing weak edges
        weak_edges = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] < 0.5]
        mst.remove_edges_from(weak_edges)

        # Each connected component is a pathway
        pathways = [set(component) for component in nx.connected_components(mst)]

        return pathways

    def _community_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use community detection algorithms"""

        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G)
            return [set(community) for community in communities]
        except:
            # Fallback to simple connected components
            return [set(component) for component in nx.connected_components(G)]


class ActivationSignatureExtractor:
    """Extract compressed activation signatures using dimensionality reduction"""

    def __init__(self, compression_method: str = 'pca'):
        self.compression_method = compression_method
        self.compressors = {}

    def extract_signatures(self,
                           critical_pathways: List[Set[str]],
                           activations_sequence: List[Dict[str, torch.Tensor]],
                           target_dim: int = 64) -> List[ActivationSignature]:
        """Extract compressed activation signatures for critical pathways"""

        signatures = []

        for pathway_idx, pathway in enumerate(critical_pathways):
            # Collect activations for this pathway
            pathway_activations = self._collect_pathway_activations(pathway, activations_sequence)

            if pathway_activations.numel() == 0:
                continue

            # Apply dimensionality reduction
            compressed_pattern = self._compress_activations(
                pathway_activations, target_dim, f"pathway_{pathway_idx}"
            )

            # Compute importance score for this pathway
            importance = self._compute_pathway_importance(pathway_activations)

            # Extract neuron indices
            neuron_indices = self._extract_neuron_indices(pathway)

            # Analyze temporal dynamics
            temporal_dynamics = self._analyze_temporal_dynamics(pathway_activations)

            signature = ActivationSignature(
                layer_id=pathway_idx,
                neuron_indices=neuron_indices,
                importance_score=importance,
                activation_pattern=compressed_pattern,
                temporal_dynamics=temporal_dynamics,
                causal_influence=0.0  # Will be computed later
            )

            signatures.append(signature)

        return signatures

    def _collect_pathway_activations(self,
                                     pathway: Set[str],
                                     activations_sequence: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Collect activations for neurons in the pathway"""

        pathway_data = []

        for node_id in pathway:
            try:
                layer_name, neuron_idx = node_id.rsplit('_', 1)
                neuron_idx = int(neuron_idx)

                # Collect this neuron's activations across time
                neuron_sequence = []
                for activations in activations_sequence:
                    if layer_name in activations:
                        layer_act = activations[layer_name]
                        if layer_act.dim() >= 2 and neuron_idx < layer_act.shape[-1]:
                            neuron_sequence.append(layer_act[..., neuron_idx].flatten())

                if neuron_sequence:
                    pathway_data.append(torch.stack(neuron_sequence))

            except (ValueError, IndexError):
                continue

        if pathway_data:
            return torch.cat(pathway_data, dim=-1)  # Concatenate all neurons
        else:
            return torch.empty(0)

    def _compress_activations(self, activations: torch.Tensor, target_dim: int, pathway_id: str) -> torch.Tensor:
        """Apply dimensionality reduction to activation patterns"""

        if activations.numel() == 0:
            return torch.zeros(target_dim)

        activations_np = activations.detach().cpu().numpy()

        if self.compression_method == 'pca':
            if pathway_id not in self.compressors:
                self.compressors[pathway_id] = PCA(n_components=min(target_dim, activations_np.shape[-1]))

            compressor = self.compressors[pathway_id]

            if activations_np.shape[0] > 1:  # Need multiple samples for PCA
                compressed = compressor.fit_transform(activations_np)
                # Take mean across time steps
                compressed_mean = compressed.mean(axis=0)
            else:
                compressed_mean = activations_np.flatten()[:target_dim]

        elif self.compression_method == 'nmf':
            if pathway_id not in self.compressors:
                self.compressors[pathway_id] = NMF(n_components=min(target_dim, activations_np.shape[-1]))

            compressor = self.compressors[pathway_id]
            activations_positive = np.abs(activations_np)  # NMF requires non-negative

            if activations_positive.shape[0] > 1:
                compressed = compressor.fit_transform(activations_positive)
                compressed_mean = compressed.mean(axis=0)
            else:
                compressed_mean = activations_positive.flatten()[:target_dim]

        else:
            # Simple truncation fallback
            compressed_mean = activations_np.flatten()[:target_dim]

        # Pad if necessary
        if len(compressed_mean) < target_dim:
            padded = np.zeros(target_dim)
            padded[:len(compressed_mean)] = compressed_mean
            compressed_mean = padded

        return torch.tensor(compressed_mean, dtype=torch.float32)

    def _compute_pathway_importance(self, pathway_activations: torch.Tensor) -> float:
        """Compute overall importance score for a pathway"""
        if pathway_activations.numel() == 0:
            return 0.0

        # Use variance as a proxy for importance
        variance = torch.var(pathway_activations).item()

        # Normalize by typical activation magnitudes
        magnitude = torch.mean(torch.abs(pathway_activations)).item()

        importance = variance / (magnitude + 1e-8)
        return float(importance)

    def _extract_neuron_indices(self, pathway: Set[str]) -> List[int]:
        """Extract neuron indices from pathway node IDs"""
        indices = []
        for node_id in pathway:
            try:
                _, neuron_idx = node_id.rsplit('_', 1)
                indices.append(int(neuron_idx))
            except ValueError:
                continue
        return sorted(indices)

    def _analyze_temporal_dynamics(self, pathway_activations: torch.Tensor) -> torch.Tensor:
        """Analyze temporal dynamics of pathway activations"""
        if pathway_activations.dim() < 2 or pathway_activations.shape[0] < 2:
            return torch.zeros(4)  # Return default dynamics

        # Compute basic temporal statistics
        dynamics = torch.zeros(4)

        # 1. Temporal variance
        dynamics[0] = torch.var(pathway_activations, dim=0).mean()

        # 2. Temporal correlation (autocorrelation at lag 1)
        if pathway_activations.shape[0] > 1:
            dynamics[1] = torch.corrcoef(torch.stack([
                pathway_activations[:-1].flatten(),
                pathway_activations[1:].flatten()
            ]))[0, 1]

        # 3. Trend (linear slope)
        time_steps = torch.arange(pathway_activations.shape[0], dtype=torch.float32)
        mean_activations = pathway_activations.mean(dim=-1)
        if len(time_steps) > 1:
            slope = torch.corrcoef(torch.stack([time_steps, mean_activations]))[0, 1]
            dynamics[2] = slope

        # 4. Stability (inverse of coefficient of variation)
        mean_val = pathway_activations.mean()
        std_val = pathway_activations.std()
        dynamics[3] = mean_val / (std_val + 1e-8)

        return dynamics


class FocusedDistillationLoss(nn.Module):
    """
    Enhanced distillation loss that focuses on critical activation pathways
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                mentor_outputs: Dict[str, torch.Tensor],
                student_activations: Dict[str, torch.Tensor],
                mentor_activations: Dict[str, torch.Tensor],
                critical_signatures: List[ActivationSignature]) -> Dict[str, torch.Tensor]:
        """Compute focused distillation loss"""

        losses = {}

        # 1. Traditional policy distillation
        policy_loss = F.kl_div(
            F.log_softmax(student_outputs['primary_logits'] / self.temperature, dim=-1),
            F.softmax(mentor_outputs['policy_logits'] / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        losses['policy_distill'] = policy_loss

        # 2. Value distillation
        value_loss = F.mse_loss(student_outputs['value'], mentor_outputs['value'])
        losses['value_distill'] = value_loss

        # 3. Critical pathway distillation (NEW)
        pathway_loss = self._compute_pathway_distillation_loss(
            student_activations, mentor_activations, critical_signatures
        )
        losses['critical_pathway'] = pathway_loss

        # 4. Signature matching loss (NEW)
        signature_loss = self._compute_signature_matching_loss(
            student_activations, critical_signatures
        )
        losses['signature_match'] = signature_loss

        # Combined loss with critical pathway emphasis
        total_loss = (
                self.alpha * policy_loss +
                0.3 * value_loss +
                0.4 * pathway_loss +  # Higher weight for critical pathways
                0.2 * signature_loss
        )

        losses['total_focused'] = total_loss

        return losses

    def _compute_pathway_distillation_loss(self,
                                           student_activations: Dict[str, torch.Tensor],
                                           mentor_activations: Dict[str, torch.Tensor],
                                           critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        """Compute distillation loss focused on critical pathways"""

        pathway_losses = []

        for signature in critical_signatures:
            # Extract activations for this pathway's neurons
            pathway_student_acts = self._extract_pathway_activations(
                student_activations, signature.neuron_indices, signature.layer_id
            )
            pathway_mentor_acts = self._extract_pathway_activations(
                mentor_activations, signature.neuron_indices, signature.layer_id
            )

            if pathway_student_acts.numel() > 0 and pathway_mentor_acts.numel() > 0:
                # Weighted MSE loss based on pathway importance
                pathway_loss = F.mse_loss(pathway_student_acts, pathway_mentor_acts)
                weighted_loss = pathway_loss * signature.importance_score
                pathway_losses.append(weighted_loss)

        if pathway_losses:
            return torch.stack(pathway_losses).mean()
        else:
            return torch.tensor(0.0, device=student_activations[list(student_activations.keys())[0]].device)

    def _compute_signature_matching_loss(self,
                                         student_activations: Dict[str, torch.Tensor],
                                         critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        """Compute loss for matching critical activation signatures"""

        signature_losses = []

        for signature in critical_signatures:
            # Extract current student pathway activations
            current_pathway_acts = self._extract_pathway_activations(
                student_activations, signature.neuron_indices, signature.layer_id
            )

            if current_pathway_acts.numel() > 0:
                # Compress to match signature dimensions
                compressed_current = self._compress_to_signature_space(
                    current_pathway_acts, signature.activation_pattern.shape[0]
                )

                # Signature matching loss
                signature_loss = F.mse_loss(compressed_current,
                                            signature.activation_pattern.to(compressed_current.device))
                weighted_loss = signature_loss * signature.importance_score
                signature_losses.append(weighted_loss)

        if signature_losses:
            return torch.stack(signature_losses).mean()
        else:
            return torch.tensor(0.0, device=list(student_activations.values())[0].device)

    def _extract_pathway_activations(self,
                                     activations: Dict[str, torch.Tensor],
                                     neuron_indices: List[int],
                                     layer_id: int) -> torch.Tensor:
        """Extract activations for specific neurons in a pathway"""

        # This is a simplified extraction - in practice, you'd need to map
        # layer_id to actual layer names and handle different architectures
        layer_names = list(activations.keys())
        if layer_id < len(layer_names):
            layer_name = layer_names[layer_id]
            layer_activations = activations[layer_name]

            # Extract specific neurons
            if len(neuron_indices) > 0 and layer_activations.shape[-1] > max(neuron_indices):
                return layer_activations[..., neuron_indices]

        return torch.empty(0)

    def _compress_to_signature_space(self, activations: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Compress activations to match signature dimensionality"""
        flattened = activations.flatten()

        if len(flattened) >= target_dim:
            return flattened[:target_dim]
        else:
            padded = torch.zeros(target_dim, device=activations.device)
            padded[:len(flattened)] = flattened
            return padded


def create_activation_based_distillation_pipeline() -> Dict:
    """Factory function to create the complete activation-based distillation pipeline"""

    pipeline = {
        'demonstration_collector': HumanDemonstrationCollector('CartPole-v1', multimodal_inputs=True),
        'activation_tracker': None,  # Will be initialized with model
        'pathway_analyzer': CriticalPathwayAnalyzer({}),
        'signature_extractor': ActivationSignatureExtractor('pca'),
        'distillation_loss': FocusedDistillationLoss(temperature=4.0, alpha=0.7),
    }

    return pipeline