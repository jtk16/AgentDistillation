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
# Import PathwayImportanceOptimizer from mathematical_framework
from mathematical_framework import PathwayImportanceOptimizer, \
    InformationTheoreticAnalyzer  # Added InformationTheoreticAnalyzer if needed by PathwayImportanceOptimizer indirectly, or for future use


@dataclass
class ActivationSignature:
    """Critical activation pattern identified from human demonstrations"""
    layer_id: int  # Corresponds to pathway_idx in current setup
    pathway_node_ids: Set[str]  # Store the original node IDs for the pathway
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

    def __init__(self, model_structure: Dict):  # model_structure might not be used currently
        self.model_structure = model_structure
        self.activation_graph = None
        self.pathway_importance = {}  # This might be redundant if PathwayImportanceOptimizer is used

    def compute_activation_importance(self,
                                      activations: Dict[str, torch.Tensor],
                                      target_performance: float,  # Should be torch.Tensor for consistency
                                      method: str = 'gradient_based') -> Dict[str, torch.Tensor]:
        """Compute importance scores for individual activations"""
        # Ensure target_performance is a tensor
        if not isinstance(target_performance, torch.Tensor):
            target_performance = torch.tensor(target_performance,
                                              device=list(activations.values())[0].device if activations else DEVICE)

        importance_scores = {}

        for layer_name, activation in activations.items():
            if method == 'gradient_based':
                # Gradient-based importance: |∇L/∇a_i|
                if activation.requires_grad:
                    grad = activation.grad
                    if grad is None and hasattr(activation, 'retains_grad') and not activation.retains_grad:
                        # Try to retain grad if it wasn't already and backward pass is expected
                        # This is tricky; ideally backward() should have been called appropriately
                        # For now, assume grad should exist or we use zeros.
                        # print(f"Warning: Gradient not found for {layer_name}, ensure backward() was called with retain_graph=True if needed.")
                        pass  # Fall through to zeros if no grad
                    score = torch.abs(grad) if grad is not None else torch.zeros_like(activation)

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

    def _compute_mutual_information(self, activation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mutual information between activations and target performance"""
        # Discretize activations for MI computation
        activation_np = activation.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Simple binning approach (can be improved with adaptive methods)
        activation_binned = np.digitize(activation_np, bins=np.linspace(activation_np.min(), activation_np.max(), 10))

        # Ensure target is 1D for MI score calculation with individual neurons
        if target_np.ndim > 1:  # Assuming target is (batch_size, 1) or just (batch_size,)
            target_np = target_np.squeeze()
        if target_np.size == 0:  # Handle empty target
            return torch.zeros_like(activation)

        # If target is a single value, replicate it for MI calculation if activation has a batch dim
        if activation_np.ndim > 1 and target_np.size == 1 and activation_np.shape[0] > 1:
            target_binned = np.full(activation_np.shape[0], int(target_np.item() * 10))
        elif target_np.size > 1:
            target_binned = np.digitize(target_np, bins=np.linspace(target_np.min(), target_np.max(), 10))
        else:  # Single target, single activation (batch_size=1)
            target_binned = np.array([int(target_np.item() * 10)])

        # Compute MI for each neuron
        mi_scores_np = np.zeros_like(activation_np)

        num_features = activation_np.shape[-1] if activation_np.ndim > 0 else 0

        for i in range(num_features):
            neuron_bins = activation_binned[..., i].flatten()

            # Ensure target_binned matches neuron_bins length for MI calculation
            if len(neuron_bins) != len(target_binned) and len(target_binned) == 1:
                target_binned_for_neuron = np.full_like(neuron_bins, target_binned[0])
            elif len(neuron_bins) == len(target_binned):
                target_binned_for_neuron = target_binned
            else:  # Mismatch, cannot compute MI, assign 0
                mi_scores_np[..., i] = 0
                continue

            if len(neuron_bins) > 0:  # Check for empty neuron_bins
                mi_scores_np[..., i] = self._mutual_info_discrete(neuron_bins, target_binned_for_neuron)

        return torch.tensor(mi_scores_np, device=activation.device)

    def _mutual_info_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute discrete mutual information"""
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            return 0.0
        # Joint distribution
        # Ensure bins cover the range of data, especially for discrete/binned data
        x_bins = max(10, len(np.unique(x))) if len(np.unique(x)) > 1 else 2
        y_bins = max(10, len(np.unique(y))) if len(np.unique(y)) > 1 else 2

        try:
            xy_hist = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
        except Exception as e:
            # print(f"Warning: histogram2d failed: {e}. Inputs x: {x[:5]}, y: {y[:5]}")
            return 0.0

        xy_prob = xy_hist / xy_hist.sum() if xy_hist.sum() > 0 else np.zeros_like(xy_hist)

        # Marginal distributions
        x_prob = xy_prob.sum(axis=1)
        y_prob = xy_prob.sum(axis=0)

        # MI = H(X) + H(Y) - H(X,Y) or Sum p(x,y) log (p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(xy_prob.shape[0]):
            for j in range(xy_prob.shape[1]):
                if xy_prob[i, j] > 1e-12 and x_prob[i] > 1e-12 and y_prob[j] > 1e-12:  # check for valid probabilities
                    mi += xy_prob[i, j] * np.log(xy_prob[i, j] / (x_prob[i] * y_prob[j]))
        return max(0.0, mi)  # MI should be non-negative

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
        if baseline_output.numel() > 0 and perturbed_output.numel() > 0:
            influence = torch.abs(perturbed_output - baseline_output).mean()
            return torch.full_like(activation, influence.item())
        else:
            return torch.zeros_like(activation)

    def _get_downstream_output(self, layer_name: str, activation: torch.Tensor) -> torch.Tensor:
        """Get output from downstream layers (simplified)"""
        # This would require more sophisticated model introspection
        # For now, let's assume the mean activation is a proxy for output that influences performance
        if activation.numel() > 0:
            return torch.mean(activation)
        else:
            return torch.tensor(0.0, device=activation.device)

    def build_activation_graph(self,
                               activations_sequence: List[Dict[str, torch.Tensor]],
                               importance_scores_sequence: List[Dict[str, torch.Tensor]]) -> nx.Graph:
        """Build graph representation of activation dependencies"""
        G = nx.Graph()
        if not activations_sequence or not importance_scores_sequence:
            self.activation_graph = G
            return G

        # Add nodes (neurons) with importance as node attributes
        # Average importance scores over the sequence for each neuron
        avg_importance_scores = {}
        if importance_scores_sequence:
            for layer_name in importance_scores_sequence[0].keys():
                # Stack scores for this layer from all timesteps
                layer_scores_list = [scores[layer_name] for scores in importance_scores_sequence if
                                     layer_name in scores and scores[layer_name].numel() > 0]
                if layer_scores_list:
                    # Ensure all tensors have the same shape before stacking if they vary batch-wise, or handle appropriately.
                    # This example assumes they are compatible or represent neuron-wise scores consistently.
                    try:
                        stacked_scores = torch.stack(layer_scores_list)
                        avg_importance_scores[layer_name] = stacked_scores.mean(dim=0)
                    except RuntimeError as e:
                        # print(f"Skipping layer {layer_name} in graph node importance due to shape mismatch: {e}")
                        # Handle cases where dimensions might not match, e.g. due to variable batch sizes not averaged out before this stage.
                        # A simple fix might be to average each tensor individually if shapes differ across sequence but neuron count is fixed.
                        # Or, ensure upstream processing provides consistent shapes.
                        # For now, we'll skip layers that cause errors or try a different aggregation.
                        # Let's try to average the means if stacking fails
                        try:
                            mean_scores = [s.mean(dim=0, keepdim=True) if s.ndim > 1 else s for s in
                                           layer_scores_list]  # Mean over batch
                            # Attempt to stack again after taking means
                            stacked_mean_scores = torch.stack(
                                [m.squeeze() for m in mean_scores if m.numel() == layer_scores_list[0].shape[-1]])
                            avg_importance_scores[layer_name] = stacked_mean_scores.mean(dim=0)

                        except Exception as e_inner:
                            # print(f"Further error processing layer {layer_name} for importance scores: {e_inner}")
                            pass

        for layer_name in activations_sequence[0].keys():
            # Use the pre-computed average importance for this layer
            layer_avg_importance = avg_importance_scores.get(layer_name)
            if layer_avg_importance is None or layer_avg_importance.numel() == 0:
                continue  # Skip if no importance score computed

            num_neurons = layer_avg_importance.shape[-1] if layer_avg_importance.dim() > 0 else (
                1 if layer_avg_importance.numel() > 0 else 0)

            for neuron_idx in range(num_neurons):
                node_id = f"{layer_name}_{neuron_idx}"
                # Importance for the specific neuron
                if layer_avg_importance.dim() > 0:
                    importance = layer_avg_importance[..., neuron_idx].mean().item()
                else:  # scalar importance if layer_avg_importance was 0-dim
                    importance = layer_avg_importance.item()

                G.add_node(node_id, layer=layer_name, neuron_idx=neuron_idx, importance=importance)

        self.activation_graph = G  # Set graph before adding edges to allow _compute_temporal_correlations to use it
        # Add edges based on correlation/causation
        if G.number_of_nodes() > 0:  # Only add edges if nodes were added
            self._add_graph_edges(G, activations_sequence)

        return G

    def _add_graph_edges(self, G: nx.Graph, activations_sequence: List[Dict[str, torch.Tensor]]):
        """Add edges based on activation correlations and layer connectivity"""
        if not activations_sequence:
            return

        # Compute temporal correlations between neurons
        correlations = self._compute_temporal_correlations(activations_sequence)

        # Add edges for highly correlated neurons
        threshold = 0.7  # Correlation threshold

        for (node1, node2), correlation in correlations.items():
            if abs(correlation) > threshold:
                if G.has_node(node1) and G.has_node(node2):  # Ensure nodes exist before adding edge
                    G.add_edge(node1, node2, weight=abs(correlation), correlation=correlation)

    def _compute_temporal_correlations(self, activations_sequence: List[Dict[str, torch.Tensor]]) -> Dict[
        Tuple[str, str], float]:
        """Compute temporal correlations between all neuron pairs"""
        correlations = {}
        if not self.activation_graph or not activations_sequence:  # Ensure graph and sequence exist
            return correlations

        # Flatten activations across time
        flattened_activations = {}
        if not activations_sequence: return correlations

        valid_layer_names = activations_sequence[0].keys()

        for layer_name in valid_layer_names:
            layer_activations_list = [acts[layer_name] for acts in activations_sequence if
                                      layer_name in acts and acts[layer_name].numel() > 0]
            if not layer_activations_list: continue

            # Handle potential variable batch sizes by taking mean over batch for each step
            # Or ensure consistent batch sizes upstream. Here, let's assume consistent features dim.
            # If activations are (batch, features), we want (time, features) after processing.
            # If activations are (batch, H, W, C), we need to flatten features.

            processed_layer_activations = []
            for acts_tensor in layer_activations_list:
                if acts_tensor.dim() > 1:  # e.g. (batch, features) or (batch, H, W, C)
                    # average over batch, then flatten remaining feature dimensions
                    processed_acts = acts_tensor.mean(dim=0).flatten()
                else:  # (features,)
                    processed_acts = acts_tensor.flatten()
                processed_layer_activations.append(processed_acts)

            if not processed_layer_activations: continue

            try:
                # Stack along time dimension; requires all processed_acts to have same shape.
                stacked_acts = torch.stack(processed_layer_activations)  # Shape: [time_steps, num_features_flat]
                flattened_activations[layer_name] = stacked_acts
            except RuntimeError as e:
                # print(f"Skipping layer {layer_name} in temporal correlations due to shape mismatch after processing: {e}")
                # This can happen if num_features_flat is not consistent.
                pass

        # Compute correlations between neurons
        all_nodes = list(self.activation_graph.nodes())

        for i, node1_id in enumerate(all_nodes):
            for j, node2_id in enumerate(all_nodes[i + 1:], i + 1):  # Use i+1 for unique pairs
                layer1, neuron1_idx_str = node1_id.rsplit('_', 1)
                layer2, neuron2_idx_str = node2_id.rsplit('_', 1)

                neuron1_idx = int(neuron1_idx_str)
                neuron2_idx = int(neuron2_idx_str)

                if layer1 in flattened_activations and layer2 in flattened_activations:
                    # flattened_activations[layer] is [time_steps, num_features_flat]
                    if neuron1_idx < flattened_activations[layer1].shape[1] and \
                            neuron2_idx < flattened_activations[layer2].shape[1]:

                        act1_series = flattened_activations[layer1][:, neuron1_idx]
                        act2_series = flattened_activations[layer2][:, neuron2_idx]

                        if act1_series.numel() > 1 and act2_series.numel() > 1:  # Need at least 2 points for corrcoef
                            # Ensure series have same length if they came from different processing pipelines
                            min_len = min(len(act1_series), len(act2_series))
                            act1_series = act1_series[:min_len]
                            act2_series = act2_series[:min_len]

                            if act1_series.std() > 1e-6 and act2_series.std() > 1e-6:  # Avoid NaNs for constant series
                                correlation_matrix = torch.corrcoef(torch.stack([act1_series, act2_series]))
                                correlation_val = correlation_matrix[0, 1].item()
                                if not np.isnan(correlation_val):
                                    correlations[(node1_id, node2_id)] = correlation_val
                            else:
                                correlations[(node1_id, node2_id)] = 0.0  # Or some other indicator for no variance
                        else:
                            correlations[(node1_id, node2_id)] = 0.0
                    else:  # neuron index out of bounds
                        correlations[(node1_id, node2_id)] = 0.0
                else:  # layer not in flattened_activations
                    correlations[(node1_id, node2_id)] = 0.0

        return correlations

    def identify_critical_pathways(self,
                                   activation_graph: nx.Graph,
                                   method: str = 'spectral_clustering') -> List[Set[str]]:
        """Identify critical pathways using graph analysis"""
        if activation_graph is None or activation_graph.number_of_nodes() == 0:
            return []

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
        if G.number_of_nodes() < 2: return [[node] for node in G.nodes()]

        # Create adjacency matrix
        nodes = list(G.nodes())
        adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()
        # Ensure adjacency matrix is symmetric for spectral clustering
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        # Apply spectral clustering
        # Adaptive number of clusters, ensure n_clusters < n_samples
        n_clusters = min(8, G.number_of_nodes() // 10 if G.number_of_nodes() >= 20 else G.number_of_nodes() // 2)
        n_clusters = max(2, n_clusters)  # Need at least 2 clusters for clustering to make sense if possible

        if G.number_of_nodes() > n_clusters and n_clusters > 1:
            try:
                # affinity='precomputed' expects a similarity matrix (non-negative, symmetric)
                # If using correlation as weight, ensure it's properly scaled for similarity (e.g., abs value, or (1+corr)/2 )
                # Our current adj_matrix uses abs(correlation) as weight, which is fine.
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans',
                                                random_state=SEED)
                labels = clustering.fit_predict(adj_matrix)

                # Group nodes by cluster
                pathways = []
                for cluster_id in range(n_clusters):
                    pathway = {nodes[i] for i in range(len(nodes)) if labels[i] == cluster_id}
                    if len(pathway) > 2:  # Minimum pathway size
                        pathways.append(pathway)
                return pathways if pathways else [[node] for node in nodes[:5]]  # Fallback if no pathways meet criteria
            except Exception as e:
                # print(f"Spectral clustering failed: {e}. Falling back to centrality.")
                return self._centrality_pathway_detection(G)  # Fallback to another method

        else:  # Not enough nodes for meaningful clustering
            # Fallback: return top N most important nodes as individual "pathways" or use centrality
            # print(f"Not enough nodes ({G.number_of_nodes()}) for spectral clustering with n_clusters={n_clusters}. Using centrality.")
            return self._centrality_pathway_detection(G)

    def _centrality_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use centrality measures to identify critical pathways"""
        if G.number_of_nodes() == 0: return []

        # Compute various centrality measures
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
            # Eigenvector centrality can fail on graphs with multiple components or be slow
            # Using a try-except block or checking graph connectivity might be needed.
            # Also, max_iter might need adjustment.
            eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
            pagerank = nx.pagerank(G, weight='weight')
        except Exception as e:
            # print(f"Centrality calculation failed: {e}. Using node importance as fallback.")
            # Fallback to just using node importance if centrality fails
            if G.number_of_nodes() == 0: return []
            top_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('importance', 0), reverse=True)[
                        :max(1, min(5, G.number_of_nodes()))]
            return [{node[0]} for node in top_nodes]

        # Combine centrality scores
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                    0.4 * betweenness.get(node, 0) +
                    0.3 * eigenvector.get(node, 0) +
                    0.3 * pagerank.get(node, 0)
            )

        if not combined_centrality:  # if all centrality measures failed or returned empty
            if G.number_of_nodes() == 0: return []
            top_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('importance', 0), reverse=True)[
                        :max(1, min(5, G.number_of_nodes()))]
            return [{node[0]} for node in top_nodes]

        # Select top N nodes based on combined centrality
        num_top_nodes = max(1, min(10, G.number_of_nodes() // 5 if G.number_of_nodes() >= 10 else G.number_of_nodes()))
        top_nodes = sorted(combined_centrality.keys(),
                           key=lambda x: combined_centrality[x], reverse=True)[:num_top_nodes]

        pathways = []
        for central_node in top_nodes:
            # Include node and its immediate neighbors if they exist
            neighborhood = {central_node}
            if G.has_node(central_node):
                neighborhood.update(set(G.neighbors(central_node)))
            if neighborhood:  # Add if not empty
                pathways.append(neighborhood)

        return pathways if pathways else [[node] for node in
                                          G.nodes()[:1]]  # Ensure at least one pathway if graph not empty

    def _mst_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use minimum spanning tree to find essential connections"""
        if G.number_of_nodes() == 0: return []

        # MST works on connected graphs. If G is not connected, operate on largest component.
        if not nx.is_connected(G):
            largest_cc_nodes = max(nx.connected_components(G), key=len, default=None)
            if largest_cc_nodes is None or len(largest_cc_nodes) < 2:
                return [[node] for node in G.nodes()[:1]]  # Fallback for very small/disconnected graphs
            subgraph = G.subgraph(largest_cc_nodes)
        else:
            subgraph = G

        if subgraph.number_of_edges() == 0:  # MST requires edges
            # Fallback: if no edges, return components or important nodes
            return [set(c) for c in nx.connected_components(G) if c] if not nx.is_connected(G) else [[node] for node in
                                                                                                     G.nodes()[:1]]

        # Create MST based on importance-weighted edges
        # MST uses 'weight' attribute. Lower weight is preferred.
        # If 'weight' is correlation (higher is better), invert for MST (e.g., 1 - abs(corr))
        # Or, use maximum_spanning_tree.
        # Assuming G.edges have 'weight' as similarity/correlation here.
        try:
            # We want strong connections, so maximum spanning tree based on correlation strength
            mst = nx.maximum_spanning_tree(subgraph, weight='weight')
        except Exception as e:
            # print(f"MST calculation failed: {e}")
            return [set(c) for c in nx.connected_components(G) if c]  # Fallback

        # Find connected components after removing weak edges from MST
        # Threshold for what constitutes a "weak" edge in the MST
        weak_edge_threshold = 0.5  # Example threshold, might need tuning

        weak_edges = [(u, v) for u, v, d in mst.edges(data=True) if d.get('weight', 0) < weak_edge_threshold]

        # Create a new graph from MST to remove edges
        temp_mst_graph = mst.copy()
        temp_mst_graph.remove_edges_from(weak_edges)

        # Each connected component in the modified MST is a pathway
        pathways = [set(component) for component in nx.connected_components(temp_mst_graph) if component]

        return pathways if pathways else [set(c) for c in nx.connected_components(G) if c]

    def _community_pathway_detection(self, G: nx.Graph) -> List[Set[str]]:
        """Use community detection algorithms"""
        if G.number_of_nodes() == 0: return []

        try:
            import networkx.algorithms.community as nx_comm
            # Using Louvain algorithm (often good for modularity)
            communities = nx_comm.louvain_communities(G, weight='weight', seed=SEED)
            return [set(community) for community in communities if community]  # filter out empty sets
        except ImportError:
            # print("NetworkX community algorithms not fully available or Louvain failed. Falling back to connected components.")
            return [set(component) for component in nx.connected_components(G) if component]
        except Exception as e:  # Catch other errors during community detection
            # print(f"Community detection failed: {e}. Falling back.")
            return [set(component) for component in nx.connected_components(G) if component]


class ActivationSignatureExtractor:
    """Extract compressed activation signatures using dimensionality reduction"""

    def __init__(self, compression_method: str = 'pca', pathway_optimizer: Optional[PathwayImportanceOptimizer] = None,
                 pathway_analyzer: Optional[CriticalPathwayAnalyzer] = None):
        self.compression_method = compression_method
        self.compressors = {}
        self.pathway_optimizer = pathway_optimizer  # New: For principled importance scores
        self.pathway_analyzer = pathway_analyzer  # New: For consistent pathway activation extraction

    def extract_signatures(self,
                           critical_pathways: List[Set[str]],  # List of sets of node_ids
                           activations_sequence: List[Dict[str, torch.Tensor]],
                           performance_scores_sequence: Optional[List[float]] = None,  # New: For importance calculation
                           target_dim: int = 64) -> List[ActivationSignature]:
        """Extract compressed activation signatures for critical pathways"""
        signatures = []
        if not critical_pathways:
            return signatures

        # Aggregate activations_sequence for importance calculation if pathway_optimizer is used
        mean_activations_for_optimizer = {}
        if self.pathway_optimizer and activations_sequence:
            # Assuming activations_sequence is a list of dicts (timesteps of layer_name: activation_tensor)
            # We need one representative Dict[str, torch.Tensor]
            # Example: average activations over the sequence for each layer
            if activations_sequence:
                all_layer_names = set()
                for act_dict in activations_sequence:
                    all_layer_names.update(act_dict.keys())

                for layer_name in all_layer_names:
                    layer_acts_list = [act_dict[layer_name] for act_dict in activations_sequence if
                                       layer_name in act_dict and act_dict[layer_name].numel() > 0]
                    if layer_acts_list:
                        # Assuming batch dimension is first, average over it, then stack and average over time
                        # This part needs careful handling of shapes.
                        # For simplicity, let's assume layer_acts_list contains (Features,) or (Batch, Features)
                        # We want to get a single representative tensor for the layer.
                        try:
                            # If batch dim exists, average over it first
                            processed_acts = [la.mean(dim=0) if la.ndim > 1 else la for la in layer_acts_list]
                            mean_activations_for_optimizer[layer_name] = torch.stack(processed_acts).mean(dim=0)
                        except Exception as e:
                            # print(f"Could not average activations for layer {layer_name} for optimizer: {e}")
                            # Fallback: use the first timestep's activations if averaging fails
                            if layer_name in activations_sequence[0] and activations_sequence[0][
                                layer_name].numel() > 0:
                                act_tensor = activations_sequence[0][layer_name]
                                mean_activations_for_optimizer[layer_name] = act_tensor.mean(
                                    dim=0) if act_tensor.ndim > 1 else act_tensor

        # Calculate importance scores for all pathways at once if optimizer is available
        pathway_importance_scores_map = {}
        if self.pathway_optimizer and mean_activations_for_optimizer and performance_scores_sequence and critical_pathways:
            # Ensure performance_scores_sequence is a tensor
            perf_tensor = torch.tensor(performance_scores_sequence, dtype=torch.float32,
                                       device=DEVICE if mean_activations_for_optimizer else 'cpu')
            # The optimizer expects a single performance score tensor, let's use the mean if it's a sequence
            if perf_tensor.ndim > 0 and len(perf_tensor) > 1:
                perf_tensor_for_opt = perf_tensor.mean()
            elif perf_tensor.ndim == 0:
                perf_tensor_for_opt = perf_tensor
            else:  # Only one score
                perf_tensor_for_opt = perf_tensor[0] if len(perf_tensor) > 0 else torch.tensor(0.0, device=DEVICE)

            # The optimizer might need a performance score per sample if activations are batched.
            # For now, let's assume compute_pathway_importance_matrix can handle a single mean performance score
            # or that mean_activations_for_optimizer are already averaged over batch.
            # Let's assume performance_tensor_for_opt should be a scalar for this specific call.

            # compute_pathway_importance_matrix returns a tensor of weights for the list of pathways
            try:
                # Critical pathways is List[Set[str]]
                # Mean_activations_for_optimizer is Dict[str, Tensor]
                # Perf_tensor_for_opt is scalar Tensor
                # This call might need adjustment based on exact expectation of optimizer's MI calc.
                # Current _compute_pathway_mi in optimizer uses correlation, which needs vector inputs.
                # Let's assume for now the optimizer handles this.
                # The optimizer's _extract_pathway_activations will be used.
                # The performance_scores argument to compute_pathway_importance_matrix should be a 1D tensor
                # that aligns with the "samples" implicitly present in the `activations` dict for MI calculation.
                # If `mean_activations_for_optimizer` is truly mean (no batch/time), MI calc is hard.
                # Let's re-evaluate the input to pathway_optimizer.
                # It needs Dict[str, Tensor (Batch, Features)] and performance_scores (Batch,).
                # For signature extraction, we are doing this once. We can use the first valid item from activations_sequence
                # and its corresponding performance score.
                first_valid_acts = None
                first_valid_perf = None
                if activations_sequence and performance_scores_sequence:
                    for i, acts_dict in enumerate(activations_sequence):
                        if i < len(performance_scores_sequence) and acts_dict:
                            first_valid_acts = acts_dict
                            first_valid_perf = torch.tensor([performance_scores_sequence[i]], dtype=torch.float32,
                                                            device=DEVICE)
                            break

                if first_valid_acts and first_valid_perf is not None:
                    all_pathway_importances = self.pathway_optimizer.compute_pathway_importance_matrix(
                        critical_pathways,  # List[Set[str]]
                        first_valid_acts,  # Dict[str, Tensor (Batch, Features)]
                        first_valid_perf  # Tensor (Batch,)
                    )
                    for i, pathway_nodes in enumerate(critical_pathways):
                        pathway_key = tuple(sorted(list(pathway_nodes)))  # Make hashable key
                        pathway_importance_scores_map[pathway_key] = all_pathway_importances[i].item()
                else:
                    # print("Not enough data for pathway optimizer, using default importance.")
                    pass  # Will use fallback importance

            except Exception as e:
                # print(f"Error using PathwayImportanceOptimizer: {e}. Using default importance.")
                pass  # Fallback to default below

        for pathway_idx, pathway_nodes in enumerate(critical_pathways):  # pathway_nodes is a Set[str]
            # Collect activations for this pathway
            # Use self.pathway_analyzer if available for consistency, else internal method
            if self.pathway_analyzer:
                pathway_activations = self.pathway_analyzer._extract_pathway_activations(pathway_nodes,
                                                                                         activations_sequence[
                                                                                             0] if activations_sequence else {})  # Example: use first timestep
            else:
                pathway_activations = self._collect_pathway_activations(pathway_nodes, activations_sequence)

            if pathway_activations.numel() == 0:
                continue

            # Apply dimensionality reduction
            compressed_pattern = self._compress_activations(
                pathway_activations, target_dim, f"pathway_{pathway_idx}"
            )

            # Get importance score
            pathway_key_for_map = tuple(sorted(list(pathway_nodes)))
            if pathway_key_for_map in pathway_importance_scores_map:
                importance = pathway_importance_scores_map[pathway_key_for_map]
            else:  # Fallback if optimizer wasn't used or failed for this pathway
                importance = self._compute_fallback_pathway_importance(pathway_activations)

            # Extract neuron indices
            neuron_indices = self._extract_neuron_indices(pathway_nodes)

            # Analyze temporal dynamics
            temporal_dynamics = self._analyze_temporal_dynamics(pathway_activations)

            signature = ActivationSignature(
                layer_id=pathway_idx,  # This is more like a "pathway_signature_index"
                pathway_node_ids=pathway_nodes,
                neuron_indices=neuron_indices,
                importance_score=importance,
                activation_pattern=compressed_pattern,
                temporal_dynamics=temporal_dynamics,
                causal_influence=0.0  # Will be computed later if needed
            )
            signatures.append(signature)

        return signatures

    def _collect_pathway_activations(self,
                                     pathway_nodes: Set[str],  # Set of node_ids like "layer_neuron"
                                     activations_sequence: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Collect activations for neurons in the pathway across the sequence."""
        # This method aggregates activations for a pathway over a sequence.
        # For PCA/NMF, we typically want a 2D matrix (samples, features).
        # Here, "samples" could be time steps, and "features" are the concatenated neurons in the pathway.

        pathway_time_series_data = []  # List of tensors, each tensor is for a time step

        if not activations_sequence:
            return torch.empty(0)

        for activations_at_step in activations_sequence:  # For each time step
            step_pathway_neurons_activations = []  # For current time step, collect all relevant neuron acts
            for node_id in pathway_nodes:
                try:
                    layer_name, neuron_idx_str = node_id.rsplit('_', 1)
                    neuron_idx = int(neuron_idx_str)

                    if layer_name in activations_at_step:
                        layer_act_at_step = activations_at_step[layer_name]  # (Batch, Features) or (Features,)

                        # Handle batch dimension: average over it or assume batch_size=1 for sequence items
                        if layer_act_at_step.dim() > 1 and layer_act_at_step.shape[0] > 1:  # Has a batch dim
                            neuron_act_at_step = layer_act_at_step.mean(dim=0)[..., neuron_idx]
                        elif layer_act_at_step.dim() > 0:  # (Features,)
                            neuron_act_at_step = layer_act_at_step[..., neuron_idx]
                        else:  # Scalar tensor
                            neuron_act_at_step = layer_act_at_step if neuron_idx == 0 else torch.empty(0)

                        if neuron_act_at_step.numel() > 0:
                            step_pathway_neurons_activations.append(neuron_act_at_step.flatten())


                except (ValueError, IndexError) as e:
                    # print(f"Skipping node {node_id} in _collect_pathway_activations: {e}")
                    continue

            if step_pathway_neurons_activations:
                # Concatenate all neuron activations for this pathway at this time step
                pathway_time_series_data.append(torch.cat(step_pathway_neurons_activations))

        if pathway_time_series_data:
            # Stack to get (time_steps, num_pathway_features)
            # Ensure all tensors in pathway_time_series_data have the same length
            min_len = min(t.size(0) for t in pathway_time_series_data)
            padded_data = [F.pad(t, (0, max(0, min_len - t.size(0))))[:min_len] for t in pathway_time_series_data]
            try:
                return torch.stack(padded_data)
            except Exception as e:
                # print(f"Could not stack pathway_time_series_data: {e}")
                return torch.empty(0)
        else:
            return torch.empty(0)

    def _compress_activations(self, pathway_activations_seq: torch.Tensor, target_dim: int,
                              pathway_id: str) -> torch.Tensor:
        """Apply dimensionality reduction to activation patterns (sequence).
           pathway_activations_seq is expected to be (time_steps, num_pathway_features)
        """
        if pathway_activations_seq.numel() == 0 or pathway_activations_seq.shape[0] < 1:  # Need at least one sample
            return torch.zeros(target_dim, device=DEVICE)  # Ensure device consistency

        # PCA/NMF expect (n_samples, n_features)
        # Here, n_samples = time_steps, n_features = num_pathway_features
        activations_np = pathway_activations_seq.detach().cpu().numpy()

        actual_features = activations_np.shape[1]
        current_target_dim = min(target_dim, actual_features)  # Cannot extract more components than features

        if current_target_dim == 0:  # No features to compress
            return torch.zeros(target_dim, device=DEVICE)

        if self.compression_method == 'pca':
            if pathway_id not in self.compressors or self.compressors[pathway_id].n_components != current_target_dim:
                self.compressors[pathway_id] = PCA(n_components=current_target_dim, random_state=SEED)

            compressor = self.compressors[pathway_id]

            # Need multiple samples (time steps) for PCA fit, or if only 1 sample, just take features.
            if activations_np.shape[0] > 1:
                try:
                    compressed = compressor.fit_transform(activations_np)  # (time_steps, target_dim)
                    compressed_mean = compressed.mean(axis=0)  # (target_dim,)
                except ValueError:  # e.g. if n_samples < n_components
                    compressed_mean = activations_np.mean(axis=0).flatten()[:current_target_dim]

            else:  # Single time step
                compressed_mean = activations_np.flatten()[:current_target_dim]


        elif self.compression_method == 'nmf':
            if pathway_id not in self.compressors or self.compressors[pathway_id].n_components != current_target_dim:
                self.compressors[pathway_id] = NMF(n_components=current_target_dim, init='random', random_state=SEED,
                                                   max_iter=200)  # Added max_iter

            compressor = self.compressors[pathway_id]
            # NMF requires non-negative. Using absolute values.
            activations_positive = np.abs(activations_np)

            if activations_positive.shape[0] > 1:  # Need multiple samples
                try:
                    compressed = compressor.fit_transform(activations_positive)
                    compressed_mean = compressed.mean(axis=0)
                except ValueError:
                    compressed_mean = activations_positive.mean(axis=0).flatten()[:current_target_dim]

            else:  # Single time step
                compressed_mean = activations_positive.flatten()[:current_target_dim]

        else:
            # Simple truncation fallback: average over time, then truncate/pad features
            compressed_mean = activations_np.mean(axis=0).flatten()[:current_target_dim]

        # Pad if necessary to meet original target_dim (if current_target_dim was smaller)
        if len(compressed_mean) < target_dim:
            padded = np.zeros(target_dim)
            padded[:len(compressed_mean)] = compressed_mean
            compressed_mean = padded
        elif len(compressed_mean) > target_dim:  # Should not happen if current_target_dim was used
            compressed_mean = compressed_mean[:target_dim]

        return torch.tensor(compressed_mean, dtype=torch.float32).to(DEVICE)

    def _compute_fallback_pathway_importance(self, pathway_activations: torch.Tensor) -> float:
        """Fallback: Compute overall importance score for a pathway (e.g., variance-based)"""
        if pathway_activations.numel() == 0:
            return 0.0

        # Use variance as a proxy for importance (variance over time and features in pathway)
        variance = torch.var(pathway_activations).item()

        # Normalize by typical activation magnitudes
        magnitude = torch.mean(torch.abs(pathway_activations)).item()

        importance = variance / (magnitude + 1e-8) if magnitude > 1e-8 else variance
        return float(importance)

    def _extract_neuron_indices(self, pathway_nodes: Set[str]) -> List[int]:
        """Extract neuron indices from pathway node IDs (like 'layername_idx')"""
        indices = set()  # Use set to avoid duplicates if nodes map to same conceptual index
        for node_id in pathway_nodes:
            try:
                # Assuming format "layername_neuronindex"
                neuron_idx_str = node_id.split('_')[-1]
                indices.add(int(neuron_idx_str))
            except (ValueError, IndexError):
                # print(f"Warning: Could not parse neuron index from node_id '{node_id}'")
                continue
        return sorted(list(indices))

    def _analyze_temporal_dynamics(self, pathway_activations: torch.Tensor) -> torch.Tensor:
        """Analyze temporal dynamics of pathway activations (expected input: [time_steps, features])"""
        if pathway_activations.dim() < 2 or pathway_activations.shape[0] < 2:  # Need at least 2 time steps
            return torch.zeros(4, device=DEVICE)  # Return default dynamics on the correct device

        dynamics = torch.zeros(4, device=DEVICE)  # Ensure output is on DEVICE

        # 1. Temporal variance (variance of mean activation over time)
        mean_activation_over_features = pathway_activations.mean(dim=1)  # (time_steps,)
        dynamics[0] = torch.var(mean_activation_over_features)

        # 2. Temporal correlation (autocorrelation at lag 1 of mean activation)
        if pathway_activations.shape[0] > 1:  # Redundant check, but good practice
            series_for_autocorr = mean_activation_over_features
            if series_for_autocorr.std() > 1e-6:  # Check for variance before corrcoef
                # Ensure stack inputs are 1D
                autocorr_stack = torch.stack([series_for_autocorr[:-1].flatten(), series_for_autocorr[1:].flatten()])
                if autocorr_stack.shape[1] > 1:  # Need at least 2 pairs
                    try:
                        dynamics[1] = torch.corrcoef(autocorr_stack)[0, 1]
                    except Exception:  # Catch any error from corrcoef (e.g. if still constant after checks)
                        dynamics[1] = 0.0
                else:
                    dynamics[1] = 0.0  # Not enough points for correlation
            else:
                dynamics[1] = 0.0  # No variance, so no meaningful correlation

        # 3. Trend (linear slope of mean activation over time)
        time_steps = torch.arange(pathway_activations.shape[0], dtype=torch.float32, device=DEVICE)
        # Using mean_activation_over_features from above
        if len(time_steps) > 1 and mean_activation_over_features.std() > 1e-6:  # Check for variance
            # Simplified trend: correlation with time
            trend_stack = torch.stack([time_steps, mean_activation_over_features])
            if trend_stack.shape[1] > 1:  # Need at least 2 points
                try:
                    slope = torch.corrcoef(trend_stack)[0, 1]
                    dynamics[2] = slope if not torch.isnan(slope) else 0.0
                except Exception:
                    dynamics[2] = 0.0
            else:
                dynamics[2] = 0.0
        else:
            dynamics[2] = 0.0

        # 4. Stability (inverse of coefficient of variation of mean activation)
        mean_val = mean_activation_over_features.mean()
        std_val = mean_activation_over_features.std()
        dynamics[3] = mean_val / (std_val + 1e-8) if std_val > 1e-9 else (mean_val / 1e-8 if abs(mean_val) > 0 else 0.0)

        return dynamics.nan_to_num(0.0)  # Replace NaNs with 0 if any slip through


class FocusedDistillationLoss(nn.Module):
    """
    Enhanced distillation loss that focuses on critical activation pathways
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        # self.pathway_optimizer = pathway_optimizer # Store if needed for dynamic weighting per batch
        # self.pathway_analyzer = pathway_analyzer

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                mentor_outputs: Dict[str, torch.Tensor],
                student_activations: Dict[str, torch.Tensor],  # Current batch activations
                mentor_activations: Dict[str, torch.Tensor],  # Current batch activations
                critical_signatures: List[ActivationSignature]) -> Dict[str, torch.Tensor]:
        """Compute focused distillation loss"""
        losses = {}
        output_device = student_outputs['primary_logits'].device

        # 1. Traditional policy distillation
        # Ensure mentor logits are not nan/inf
        valid_mentor_logits = mentor_outputs['policy_logits'].nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)

        policy_loss = F.kl_div(
            F.log_softmax(student_outputs['primary_logits'] / self.temperature, dim=-1),
            F.softmax(valid_mentor_logits / self.temperature, dim=-1),  # Use validated logits
            reduction='batchmean',
            log_target=False  # log_target=False expects probabilities for target
        ) * (self.temperature ** 2)

        losses['policy_distill'] = policy_loss

        # 2. Value distillation
        losses['value_distill'] = F.mse_loss(student_outputs['value'], mentor_outputs['value'])

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
                self.alpha * losses['policy_distill'] +
                0.3 * losses['value_distill'] +  # Example weight
                0.4 * losses['critical_pathway'] +  # Higher weight for critical pathways
                0.2 * losses['signature_match']  # Example weight
        )
        losses['total_focused'] = total_loss

        return losses

    def _extract_activations_for_signature(self,
                                           activations: Dict[str, torch.Tensor],  # layer_name -> activation_tensor
                                           signature: ActivationSignature) -> Optional[torch.Tensor]:
        """Helper to extract and concatenate activations for neurons in a signature's pathway."""
        # This method needs to map signature.layer_id (pathway_idx) and signature.neuron_indices
        # back to specific layer names and neuron indices within those layers.
        # The current ActivationSignature.neuron_indices are global indices from the pathway nodes.
        # We need to know which layer each of these original nodes belonged to.
        # Let's assume signature.pathway_node_ids (Set[str] like "layerName_neuronIdx") is available.

        pathway_neuron_activations = []
        if not hasattr(signature, 'pathway_node_ids') or not signature.pathway_node_ids:
            # Fallback or error if pathway_node_ids is missing
            # This part depends on how layer_id and neuron_indices in ActivationSignature are defined.
            # If layer_id is an index into list(activations.keys()) and neuron_indices are for that layer:
            layer_names = list(activations.keys())
            if signature.layer_id < len(layer_names):
                layer_name = layer_names[signature.layer_id]  # This is a fragile mapping
                layer_activations = activations.get(layer_name)
                if layer_activations is not None and layer_activations.numel() > 0:
                    # Assume signature.neuron_indices are valid for this layer_activations
                    # Ensure neuron_indices are within bounds
                    valid_indices = [idx for idx in signature.neuron_indices if idx < layer_activations.shape[-1]]
                    if valid_indices:
                        # (Batch, Num_Selected_Neurons_in_Layer)
                        return layer_activations[..., valid_indices]
            return torch.empty(0, device=list(activations.values())[0].device if activations else DEVICE)

        # Preferred method using pathway_node_ids
        for node_id in signature.pathway_node_ids:
            try:
                layer_name, neuron_idx_str = node_id.rsplit('_', 1)
                neuron_idx = int(neuron_idx_str)
                layer_acts = activations.get(layer_name)
                if layer_acts is not None and layer_acts.numel() > 0 and neuron_idx < layer_acts.shape[-1]:
                    # Get (Batch, 1) for this neuron
                    neuron_act = layer_acts[..., neuron_idx].unsqueeze(-1)
                    pathway_neuron_activations.append(neuron_act)
            except Exception:
                continue

        if pathway_neuron_activations:
            return torch.cat(pathway_neuron_activations, dim=-1)  # (Batch, Num_Total_Neurons_In_Pathway)
        return torch.empty(0, device=list(activations.values())[0].device if activations else DEVICE)

    def _compute_pathway_distillation_loss(self,
                                           student_activations: Dict[str, torch.Tensor],
                                           mentor_activations: Dict[str, torch.Tensor],
                                           critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        """Compute distillation loss focused on critical pathways (neuron-level)."""
        pathway_losses = []
        output_device = list(student_activations.values())[0].device if student_activations else DEVICE

        for signature in critical_signatures:
            pathway_student_acts = self._extract_activations_for_signature(student_activations, signature)
            pathway_mentor_acts = self._extract_activations_for_signature(mentor_activations, signature)

            if pathway_student_acts.numel() > 0 and pathway_mentor_acts.numel() > 0:
                # Ensure compatible shapes, e.g. by taking mean over batch or ensuring batch sizes match
                # For MSE, shapes must match.
                # Let's assume they do, or apply .mean(dim=0) if batch sizes differ and want to compare means.
                # For now, assume batch sizes match from the input dictionaries.
                if pathway_student_acts.shape == pathway_mentor_acts.shape:
                    pathway_loss = F.mse_loss(pathway_student_acts, pathway_mentor_acts)
                    weighted_loss = pathway_loss * signature.importance_score
                    pathway_losses.append(weighted_loss)

        if pathway_losses:
            return torch.stack(pathway_losses).mean()
        else:
            return torch.tensor(0.0, device=output_device)

    def _compute_signature_matching_loss(self,
                                         student_activations: Dict[str, torch.Tensor],  # Current batch activations
                                         critical_signatures: List[ActivationSignature]) -> torch.Tensor:
        """Compute loss for matching critical activation signatures (compressed representations)."""
        signature_losses = []
        output_device = list(student_activations.values())[0].device if student_activations else DEVICE

        # This requires a way to compress current batch student activations for a pathway
        # to match the pre-computed signature.activation_pattern.
        # We need a temporary compressor like in ActivationSignatureExtractor, or pass it in.
        # For simplicity, let's use a basic compression here (mean over batch, then flatten and truncate/pad).

        for signature in critical_signatures:
            # 1. Extract student activations for the neurons in this signature's pathway for the current batch
            current_pathway_student_batch_acts = self._extract_activations_for_signature(student_activations, signature)

            if current_pathway_student_batch_acts.numel() > 0:
                # 2. Compress these batch activations to match signature.activation_pattern dimensions
                # signature.activation_pattern is (target_dim,)
                # current_pathway_student_batch_acts is (Batch, Num_Pathway_Neurons)

                # Simple compression: mean over batch, then ensure dim matches.
                # This is a very basic stand-in for the PCA/NMF used in signature extraction.
                # A more robust way would be to apply the *fitted* PCA/NMF compressor from signature extraction.
                if current_pathway_student_batch_acts.dim() > 1:
                    mean_batch_pathway_acts = current_pathway_student_batch_acts.mean(dim=0)  # (Num_Pathway_Neurons,)
                else:  # Already 1D (e.g. batch size 1 was squeezed)
                    mean_batch_pathway_acts = current_pathway_student_batch_acts

                target_dim = signature.activation_pattern.shape[0]
                compressed_current = self._compress_to_signature_space(mean_batch_pathway_acts, target_dim)

                # Signature matching loss
                signature_loss = F.mse_loss(compressed_current,
                                            signature.activation_pattern.to(compressed_current.device))
                weighted_loss = signature_loss * signature.importance_score
                signature_losses.append(weighted_loss)

        if signature_losses:
            return torch.stack(signature_losses).mean()
        else:
            return torch.tensor(0.0, device=output_device)

    def _extract_pathway_activations(self,  # This method seems redundant with _extract_activations_for_signature
                                     activations: Dict[str, torch.Tensor],
                                     neuron_indices: List[int],
                                     # These are specific to a layer if layer_id is a layer name/index
                                     layer_id: int) -> torch.Tensor:  # layer_id is pathway_idx in ActivationSignature
        """DEPRECATED LIKELY - Extract activations for specific neurons in a pathway.
           Relies on layer_id mapping to a single layer and neuron_indices being for that layer.
           _extract_activations_for_signature is more robust with pathway_node_ids.
        """
        # This is a simplified extraction - in practice, you'd need to map
        # layer_id (which is pathway_idx) to actual layer names and handle different architectures
        # This was the original problematic implementation.
        # It's better to rely on signature.pathway_node_ids.
        # Keeping it for now if other parts still call it, but it should be phased out.

        layer_names = list(activations.keys())
        # The layer_id in ActivationSignature is currently pathway_idx, not a direct layer map.
        # This method is likely misinterpreting layer_id.
        # Let's assume if called, it means to pick an arbitrary layer if layer_id is out of bounds.
        # This part is a known weakness of the original implementation.
        # For now, we'll return empty if the logic is unclear.
        # If signature.pathway_node_ids is available, that should be used.
        # print(f"Warning: _extract_pathway_activations called with layer_id={layer_id} (pathway_idx), which is ambiguous.")

        # Attempt a guess: pick one layer based on some heuristic if layer_id is pathway_idx
        # This is not robust.
        chosen_layer_name = None
        if layer_id < len(layer_names):  # Simplistic: assume layer_id can map to an index in layer_names
            chosen_layer_name = layer_names[layer_id]
        elif layer_names:  # Fallback to first layer
            chosen_layer_name = layer_names[0]

        if chosen_layer_name:
            layer_activations = activations.get(chosen_layer_name)
            if layer_activations is not None and layer_activations.numel() > 0:
                valid_indices = [idx for idx in neuron_indices if idx < layer_activations.shape[-1]]
                if valid_indices:
                    return layer_activations[..., valid_indices]

        return torch.empty(0, device=list(activations.values())[0].device if activations else DEVICE)

    def _compress_to_signature_space(self, activations: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Compress activations (1D tensor) to match signature dimensionality (target_dim)."""
        flattened = activations.flatten()
        current_len = len(flattened)
        output_device = activations.device

        if current_len == target_dim:
            return flattened
        elif current_len > target_dim:
            return flattened[:target_dim]
        else:  # current_len < target_dim
            padded = torch.zeros(target_dim, device=output_device)
            padded[:current_len] = flattened
            return padded


def create_activation_based_distillation_pipeline(
        pathway_optimizer: Optional[PathwayImportanceOptimizer] = None,
        pathway_analyzer_instance: Optional[CriticalPathwayAnalyzer] = None
) -> Dict:
    """Factory function to create the complete activation-based distillation pipeline."""

    # If pathway_analyzer_instance is not provided, create a default one
    analyzer_to_use = pathway_analyzer_instance if pathway_analyzer_instance is not None else CriticalPathwayAnalyzer(
        {})

    pipeline = {
        'demonstration_collector': HumanDemonstrationCollector('CartPole-v1', multimodal_inputs=True),
        'activation_tracker': None,  # Will be initialized with model
        'pathway_analyzer': analyzer_to_use,  # Use the passed or new instance
        'signature_extractor': ActivationSignatureExtractor(
            compression_method='pca',
            pathway_optimizer=pathway_optimizer,  # Pass the optimizer here
            pathway_analyzer=analyzer_to_use  # Pass analyzer for consistency
        ),
        'distillation_loss': FocusedDistillationLoss(temperature=4.0, alpha=0.7),
    }

    return pipeline