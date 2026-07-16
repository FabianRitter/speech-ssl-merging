import os
import glob
import numpy as np
import torch

from time import time
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from torch import nn
import re

from torchnlp.utils import lengths_to_mask
from graphs.base_graph import NodeType
from metric_calculators import CovarianceMetric, MeanMetric
from matching_functions import match_tensors_permute, match_tensors_permute_symmetric, match_tensors_intra_pairwise, match_tensors_zipit
from matching_functions import compute_correlation
from merging_utils.ties_merger_adaptation_to_corr import TIESMerging
from inspect import getmembers, isfunction
try:
    import ipdb as pdb
except ImportError:
    import pdb

def analyze_zipit_unmerge_matrix(unmerge_matrix, dims_per_graph, node_id="Unknown"):
    """
    Analyzes a ZipIt! unmerge matrix to count within- and between-model merges.

    Args:
        unmerge_matrix (torch.Tensor): The GLOBAL unmerge matrix [TotalDim, FinalDim]
                                       before chunking. Should contain mostly 0s and 1s.
        dims_per_graph (list[int]): List containing the original feature dimensions
                                     of each model (e.g., [dim_hubert, dim_mert]).
                                     Currently assumes len == 2.
        node_id (any): Identifier for the node being processed (for logging).

    Returns:
        dict: Counts of different merge types for the final features.
              Keys: 'final_dim', 'within_model1', 'within_model2',
                    'between_models', 'single_model1', 'single_model2',
                    'total_original_dim1', 'total_original_dim2'.
    """
    if len(dims_per_graph) != 2:
        print(f"Warning: ZipIt analysis currently only supports 2 models. Found {len(dims_per_graph)}. Skipping analysis for node {node_id}.")
        return None

    total_original_dim, final_dim = unmerge_matrix.shape
    dim_model1 = dims_per_graph[0]
    dim_model2 = dims_per_graph[1]

    # --- Use deduced dimensions for the check ---
    # This check is important after the fix for deducing dimensions
    if total_original_dim != dim_model1 + dim_model2:
         # This check might be redundant if dims_per_graph is deduced correctly,
         # but good as a safeguard if it comes from elsewhere.
         print(f"Error Node {node_id}: Dimension mismatch in ZipIt analysis. "
               f"Matrix TotalDim ({total_original_dim}) != "
               f"Provided sum(dims_per_graph) ({dim_model1 + dim_model2}).")
         # Let's trust the matrix shape and proceed if possible, but log warning
         # Re-deduce from matrix shape if inconsistent:
         if total_original_dim % 2 == 0 and len(dims_per_graph) == 2:
              print(f"  Re-deducing dims for analysis: {total_original_dim // 2}")
              dim_model1 = total_original_dim // 2
              dim_model2 = total_original_dim // 2
         else:
              print("  Cannot resolve dimension mismatch. Skipping analysis.")
              return None # Cannot proceed if dimensions don't match

    counts = {
        'final_dim': final_dim,
        'within_model1': 0, # Features merged only from model 1
        'within_model2': 0, # Features merged only from model 2
        'between_models': 0, # Features merged across model 1 and model 2
        'single_model1': 0, # Unmerged features from model 1
        'single_model2': 0, # Unmerged features from model 2
        'total_original_dim1': dim_model1,
        'total_original_dim2': dim_model2,
    }

    # Use a tolerance for floating point comparisons if needed, but ZipIt should be binary
    threshold = 0.5

    for j in range(final_dim): # Iterate through each *final* merged feature column
        contributing_indices = torch.where(unmerge_matrix[:, j] > threshold)[0]
        num_contributors = len(contributing_indices)

        if num_contributors == 0:
            # This shouldn't happen if unmerge is correct, maybe a zero column?
            print(f"Warning Node {node_id}: Final feature column {j} has no contributors.")
            continue
        elif num_contributors == 1:
            # Single feature preserved
            idx = contributing_indices[0].item()
            if idx < dim_model1:
                counts['single_model1'] += 1
            else: # idx must be >= dim_model1
                counts['single_model2'] += 1
        else:
            # Merge occurred
            from_model1 = any(idx < dim_model1 for idx in contributing_indices)
            # Use dim_model1 as the boundary index
            from_model2 = any(idx >= dim_model1 for idx in contributing_indices)

            if from_model1 and from_model2:
                counts['between_models'] += 1
            elif from_model1:
                counts['within_model1'] += 1
            elif from_model2: # Should be only case left
                counts['within_model2'] += 1
            else:
                 # This case should also not happen if logic is correct
                 print(f"Warning Node {node_id}: Merge column {j} has multiple contributors but not classified.")


    # Sanity check: Sum of counts should equal final_dim
    total_counted = (counts['within_model1'] + counts['within_model2'] +
                     counts['between_models'] + counts['single_model1'] +
                     counts['single_model2'])
    if total_counted != final_dim:
        print(f"Warning Node {node_id}: Count mismatch! Counted {total_counted} final features, expected {final_dim}.")
        # print(f"  Counts: {counts}")
        # pdb.set_trace()


    return counts

def contains_name(layer_name, node_list):
    for node in node_list:
        if node in layer_name:
            return True
    return False


def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]



class MergeHandler:
    def __init__(self, graph, merge, unmerge, orig):
        self.graph = graph
        # just store merge and unmerge matrices
        self.orig = orig
        self.merge = merge
        self.unmerge = unmerge
        # Define handlers for different module types.
        # self.module_handlers = {
        #     'Conv1d': self.handle_conv1d,
        #     'Conv2d': self.handle_conv2d,
        #     'Linear': self.handle_linear,
        #     'LayerNorm': self.handle_layernorm,
        #     # add others as needed:
        #     'BatchNorm2d': self.handle_batchnorm2d,
        #     # for activation functions that are parameterless:
        #     'GELU': self.handle_fn,
        #     'ReLU': self.handle_fn,
        #     'Tanh': self.handle_fn,
        #     # fallback handler:
        #     'default': self.handle_default
        # }

class ModelMerge(nn.Module):

    def __init__(self, *graphs, device=0):
        super().__init__()
        
        self.hooks = []
        self.init(graphs, device)

    def init(self, graphs, device):

        # move all graph models to eval
        for g in graphs:
            g.model.to(device).eval()

        self.graphs = graphs
        self.device = device
        self.merged_model = None
        count = 0
        for graph in self.graphs:
            print(count)
            count+=1
            graph.add_hooks(device=device)
    
    def analyze_feature_similarity(self, dataloader, nodes_to_analyze, cost_dict, 
                                 node_scaling_factors, 
                                 num_batches_for_similarity=1): # Removed permutation_intensity_results
        
        # Construct a simple map for display names here or just use node_id
        node_id_to_display_name_map = {}
        for node_id in nodes_to_analyze:
            # Basic display name logic (can be enhanced if needed from graph info)
            info = self.graphs[0].get_node_info(node_id) # Assuming graph[0] exists
            layer_name_from_info = info.get('layer', f'Node_{node_id}')
            if layer_name_from_info:
                 # Basic simplification for logging
                 simplified = layer_name_from_info.replace("model.encoder.layers.", "EncL")
                 simplified = simplified.replace("model.feature_extractor.conv_layers.", "CNN_L")
                 simplified = simplified.replace(".self_attn.out_proj", "_AttnOut")
                 simplified = simplified.replace(".fc2", "_FC2Pre")
                 simplified = simplified.replace("_(PREFIX_TARGET)", "_PFX")
                 node_id_to_display_name_map[node_id] = simplified
            else:
                 node_id_to_display_name_map[node_id] = f"Node_{node_id}"


        print(f"\n--- Starting Feature Similarity Analysis for nodes: {node_id_to_display_name_map.values()} ---")
        
        similarity_stats = {
            node_id: {
                'before_sim_sum': 0.0,
                'after_sim_sum': 0.0,
                'num_samples_compared': 0 
            } for node_id in nodes_to_analyze
        }

        if not self.unmerges:
            print("ERROR: self.unmerges not populated. Cannot perform similarity analysis.")
            return {}

        print("Preparing for feature collection pass (hooks managed by compute_intermediates)...")

        batches_processed = 0
        # Ensure dataloader can be re-iterated or get a new one if it's a one-shot iterator
        # If dataloader is a list of dataloaders, handle appropriately (e.g. use the first one)
        current_dataloader = dataloader[0] if isinstance(dataloader, list) else dataloader

        for batch_idx, (wavs, *others) in enumerate(tqdm(current_dataloader, desc="Feature Similarity Pass")):
            if batches_processed >= num_batches_for_similarity:
                print(f"Processed {num_batches_for_similarity} batches for similarity analysis. Stopping.")
                break
            
            try:
                wavs = [torch.FloatTensor(wav).to(self.device) for wav in wavs]
                current_batch_intermediates = [g.compute_intermediates(wavs, device=self.device) for g in self.graphs]
                current_batch_intermediates = self.remove_pads_dynamic(current_batch_intermediates, wavs, node_scaling_factors)

                for node_id in nodes_to_analyze:
                    display_name_for_node = node_id_to_display_name_map.get(node_id, f"Node_{node_id}")
                    if node_id not in current_batch_intermediates[0] or \
                       node_id not in current_batch_intermediates[1]:
                        print(f"Warning: Node {display_name_for_node} missing in current batch. Skipping.")
                        continue

                    A_H_batch = current_batch_intermediates[0][node_id].float()
                    A_M_batch = current_batch_intermediates[1][node_id].float()

                    if A_H_batch.shape[0] == 0 or A_M_batch.shape[0] == 0:
                        print(f"Warning: Node {display_name_for_node} has empty features. Skipping.")
                        continue
                    if A_H_batch.shape != A_M_batch.shape:
                        print(f"Warning: Shape mismatch Node {display_name_for_node}: H({A_H_batch.shape}), M({A_M_batch.shape}). Skipping.")
                        continue
                    
                    dim_node = A_H_batch.shape[0]
                    P_M = self.unmerges[node_id][1].to(self.device)
                    
                    if P_M.shape[0] != dim_node or P_M.shape[1] != dim_node:
                        print(f"Warning: P_M shape ({P_M.shape}) vs feat dim ({dim_node}) mismatch for {display_name_for_node}. Skipping.")
                        continue

                    A_M_prime_batch = P_M @ A_M_batch
                    A_H_norm = A_H_batch / (torch.norm(A_H_batch, p=2, dim=1, keepdim=True) + 1e-8)
                    A_M_norm = A_M_batch / (torch.norm(A_M_batch, p=2, dim=1, keepdim=True) + 1e-8)
                    A_M_prime_norm = A_M_prime_batch / (torch.norm(A_M_prime_batch, p=2, dim=1, keepdim=True) + 1e-8)

                    cos_sim_before_channels = (A_H_norm * A_M_norm).sum(dim=1)
                    avg_cos_sim_before_batch = cos_sim_before_channels.mean().item()
                    
                    cos_sim_after_channels = (A_H_norm * A_M_prime_norm).sum(dim=1)
                    avg_cos_sim_after_batch = cos_sim_after_channels.mean().item()
                    
                    if num_batches_for_similarity == 1:
                        similarity_stats[node_id]['avg_cos_sim_before_perm'] = avg_cos_sim_before_batch
                        similarity_stats[node_id]['avg_cos_sim_after_perm'] = avg_cos_sim_after_batch
                    else:
                        similarity_stats[node_id]['before_sim_sum'] += avg_cos_sim_before_batch * dim_node
                        similarity_stats[node_id]['after_sim_sum'] += avg_cos_sim_after_batch * dim_node
                    similarity_stats[node_id]['num_samples_compared'] += dim_node


            except Exception as e:
                print(f"Error processing batch {batch_idx} in similarity analysis: {e}")
                import traceback; traceback.print_exc()
                continue
            batches_processed += 1

        final_similarity_results_with_names = {}
        for node_id, stats in similarity_stats.items():
            display_name = node_id_to_display_name_map.get(node_id, f"Node_{node_id}")
            percent_perm_val = float('nan') # Placeholder if Exp1 results not available here
            # If you pass permutation_intensity_results (Exp1) to this func, you can get it:
            # if permutation_intensity_results:
            #   percent_perm_val = permutation_intensity_results["mert"].get(display_name, float('nan'))

            if num_batches_for_similarity > 1 and stats['num_samples_compared'] > 0:
                avg_before = stats['before_sim_sum'] / stats['num_samples_compared']
                avg_after = stats['after_sim_sum'] / stats['num_samples_compared']
            elif num_batches_for_similarity == 1:
                 avg_before = stats.get('avg_cos_sim_before_perm', float('nan'))
                 avg_after = stats.get('avg_cos_sim_after_perm', float('nan'))
            else:
                avg_before, avg_after = float('nan'), float('nan')
            
            final_similarity_results_with_names[display_name] = {
                "avg_cos_sim_before_perm": avg_before,
                "avg_cos_sim_after_perm": avg_after,
                "cost_from_algo": cost_dict.get(node_id, float('nan')), # cost_dict is from compute_transformations
                "percent_permuted": percent_perm_val # Needs Exp1 results if you want it here
            }
        
        print("--- Feature Similarity Analysis Finished ---")
        return final_similarity_results_with_names
    
    def _get_layer_idx_from_node_info(self, node_info):
        """
        Parses the transformer layer index from the node's layer name.
        Returns the layer index (int) or None if not found/applicable.
        Example layer_name: 'model.encoder.layers.5.self_attn.out_proj' -> returns 5
        """
        layer_name = node_info.get('layer')
        if not layer_name or 'encoder.layers.' not in layer_name:
            return None

        # Use regex to find the number after 'layers.'
        match = re.search(r'encoder\.layers\.(\d+)', layer_name)
        if match:
            return int(match.group(1))
        else:
            # Fallback split logic (less robust)
            parts = layer_name.split('.')
            try:
                layers_idx = parts.index('layers')
                if layers_idx + 1 < len(parts) and parts[layers_idx + 1].isdigit():
                    return int(parts[layers_idx + 1])
            except ValueError:
                pass # 'layers' not found
        return None # Index not found

    # helper function to collect hiddens. Do not recommend using for large FractionalDataloader
    def get_hiddens(self, dataloaders):
        data_stores = [defaultdict(lambda: None) for g in self.graphs]

        with torch.no_grad():
            for dataloader in dataloaders:
                for x, _ in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):
                    x = x.to(self.device)
                    intermediates = [g.compute_intermediates(x) for g in self.graphs] # shape [feat_dim, num_tokens]
                    nodes = list(intermediates[0].keys())
                    for node in nodes:
                        intermeds_float = [i[node][:,1:-1].float().detach() for i in intermediates] # len = num_graphs
                        if data_stores[0][node] == None:
                            for i in range(len(self.graphs)):
                                data_stores[i][node] = intermeds_float[i]
                        else:
                            for i in range(len(self.graphs)):
                                data_stores[i][node] = torch.cat((data_stores[i][node], intermeds_float[i]), 1)
        return data_stores

    # get average variance across features for each node
    def compute_variances(self, dataloaders):
        data_stores = self.get_hiddens(dataloaders)
        nodes = list(data_stores[0].keys())
                
        for node in nodes:
            for i in range(len(self.graphs)):
                data_stores[i][node] = torch.mean(torch.var(data_stores[i][node], dim=1))
        return data_stores

    # for investigating representations b/w two models 
    def compute_rep_distances(self, dataloaders):
        node_dists = []
        data_stores = self.get_hiddens(dataloaders)
        nodes = list(data_stores[0].keys())

        for node in nodes:
            x = data_stores[0][node]  # shape [feat_dim, num_tokens]
            y = data_stores[1][node]  # shape [feat_dim, num_tokens]
            dists = (x - y).pow(2).sum(0).sqrt()
            node_dists.append(torch.mean(dists))
        return node_dists

    def remove_pads_dynamic(self, intermediates, wavs, node_scaling, final_downsample=320):
        """
        Dynamically removes padding from the concatenated intermediate features using
        per-sample waveform lengths and node-specific scaling factors.
        
        Args:
            intermediates: List (per graph) of dictionaries mapping node ids to feature tensors.
                        Each tensor has shape [feat_dim, total_tokens] where tokens from all samples are concatenated.
            wavs: List of input waveform tensors (one per sample).
            node_scaling: Dictionary mapping node identifiers (e.g. 4, 7, 11, etc.) to a scaling factor.
                        For example:
                        {
                            4: 64, 7: 32, 11: 16, 15: 8, 19: 4, 23: 2, 27: 1,
                            # transformer nodes can default to 1
                            37: 1, 42: 1, ...
                        }
            final_downsample: The overall downsampling factor for the final CNN output (e.g. 320).
        
        Returns:
            Updated intermediates with the padded tokens removed (concatenation reversed).
        """
        bsz = len(wavs)
        # Compute effective "final" lengths for each sample based on the final downsampling.
        # For each waveform, the expected final token count is wav_length // final_downsample.
        final_lengths = [int(wav.shape[-1] // final_downsample) for wav in wavs]
        
        for g_idx in range(len(self.graphs)):
            for node in list(intermediates[0].keys()):
                # Determine the node-specific scaling factor (default is 1)
                factor = node_scaling.get(node, 1)
                # For each sample in the batch, compute its expected token count at this node.
                effective_lengths = [length * factor for length in final_lengths]
                tensor_to_edit = intermediates[g_idx][node]  # Shape: [feat_dim, total_tokens]
                list_of_tensors = []
                start_idx = 0
                # For each sample, slice the concatenated tensor with its effective length.
                for eff_len in effective_lengths:
                    end_idx = start_idx + eff_len
                    list_of_tensors.append(tensor_to_edit[:, start_idx:end_idx])
                    start_idx = end_idx
                # Concatenate back along the token dimension to get a [feat_dim, sum(effective_lengths)] tensor.
                intermediates[g_idx][node] = torch.cat(list_of_tensors, dim=1)
        return intermediates



    def remove_pads(self, intermediates, input, lens):
        """
        Removes padding from HuBERT intermediates based on downsampled feature lengths.

        Args:
            intermediates: List of feature maps or node representations.
            input: Tensor input to the Transformer.
            lens: List of downsampled lengths for each sequence in the batch.

        Returns:
            intermediates: Updated intermediates with padding removed.
        """
        bsz = len(input)  # Batch size
        
        
        for g_idx in range(len(self.graphs)):
            for node in list(intermediates[0].keys()):
                tensor_to_edit = intermediates[g_idx][node]  # shape: [feat_dim, total_features]
                # Split concatenated tensor based on `lens`
                list_of_tensors = []
                start_idx = 0
                for i in range(bsz):
                    end_idx = start_idx + lens[i]
                    list_of_tensors.append(tensor_to_edit[:, start_idx:end_idx])
                    start_idx = end_idx

                # Concatenate valid features back
                new_tensor = torch.cat(list_of_tensors, dim=1)
                intermediates[g_idx][node] = new_tensor

        return intermediates


    def load_toks(self, saved_path):
        filenames = glob.glob(os.path.join(saved_path, 'toks', '*.pt'))
        num_files = len(filenames)
        tok_ids_all = []
        for i in range(num_files):
            tok_ids_all.append(torch.load(f'{saved_path}/toks/{i}.pt'))
        return torch.cat(tok_ids_all)

    def sent_rep(self, intermediates, node, sentence_level, lens, special_toks=False):
        # already shape [feat_dim, bsz]
        if sentence_level == 'cls':
            return [intermediates[i][node] for i in range(len(self.graphs))]
        bsz = len(lens)
        intermeds_float = []

        if intermediates[0][node].shape[-1] == len(lens): #bsz
            intermeds_float = [intermediates[0][node], intermediates[1][node]]
            return intermeds_float
        for g_idx in range(len(self.graphs)):
            sent_levels = []
            last_idx = 0
            for senlen in lens:
                actual_len = senlen 
                if special_toks == False:
                    actual_len = senlen - 2
                sent_levels.append(intermediates[g_idx][node][:,last_idx:last_idx + actual_len])
                last_idx += actual_len
            if sentence_level == 'maxpool':
                try:
                    sent_avgs = [torch.amax(sent_levels[i].float(), 1).unsqueeze(1) for i in range(bsz)]
                except:
                    breakpoint()
            elif sentence_level == 'avgpool':
                sent_avgs = [torch.mean(sent_levels[i].float(), 1).unsqueeze(1) for i in range(bsz)]
            intermeds_float.append(torch.hstack(sent_avgs)) # list of [[dim, bsz], [dim, bsz]]
        return intermeds_float


    def compute_metrics(self, dataloader, metric_classes, sentence_level=None, special_toks=False,
                    print_featnorms=False):

        # Keep node_scaling as it is
        node_scaling = { 4: 64, 7: 32, 11: 16, 15: 8, 19: 4, 23: 2, 27: 1, # CNN Layers
                     **{node_id: 1 for node_id in range(28, 250)} # Default transformer nodes to 1
                    }

        self.metrics = None

        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        
        # Ensure hooks are added ONCE before the loop
        print("Adding hooks before starting dataloader loop...")
        for g in self.graphs:
            g.add_hooks(device=self.device)
            print(f"  Graph {self.graphs.index(g)} hooks: {len(g.hooks)}")

        # downsampling_rate = self.graphs[0].get_downsample_rates(key="encoder") # Keep if needed elsewhere
        
        for dataloader_idx, dataloader in enumerate(dataloader_list):
            print(f"Processing Dataloader {dataloader_idx + 1}/{len(dataloader_list)}")
            for batch_idx, (wavs, *others) in enumerate(tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics")):
                try:
                    # load batch & track number of elements
                    wavs = [torch.FloatTensor(wav).to(self.device) for wav in wavs]

                    # --- Intermediate Capture ---
                    # Ensure hooks are active for *this* batch computation
                    # (Usually done once in init, but double-check if models are changing)
                    # Example: force re-adding hooks if needed:
                    # for g in self.graphs: g.add_hooks(device=self.device)

                    intermediates = [g.compute_intermediates(wavs) for g in self.graphs] # shape [feat_dim, num_tokens]

                    # --- Padding Removal ---
                    # Important: Verify node_scaling keys match your actual graph PREFIX node IDs
                    # Add debug prints here if intermediates seem incorrect for transformer nodes
                    # print(f"Batch {batch_idx}, Intermediates keys: {list(intermediates[0].keys())}")
                    # for node_id in intermediates[0].keys():
                    #      if node_id > 30: # Check a transformer node
                    #          print(f"  Node {node_id} shape before pad removal: {intermediates[0][node_id].shape}")

                    intermediates = self.remove_pads_dynamic(intermediates, wavs, node_scaling)

                    # --- Metric Initialization ---
                    # Get nodes *after* potential errors in compute_intermediates or padding removal
                    current_nodes = list(intermediates[0].keys())
                    if not current_nodes:
                        print(f"Warning: No intermediates captured for batch {batch_idx}. Skipping metric update.")
                        continue

                    if self.metrics is None:
                        # Initialize metrics only for nodes present in the *first successful* batch
                        self.metrics = {n: {k: v() for k, v in metric_classes.items()} for n in current_nodes}
                        print(f"Initialized metrics for nodes: {list(self.metrics.keys())}")


                    # --- Metric Update ---
                    # Iterate through nodes *present in the current batch's intermediates*
                    # And *also* present in the initialized self.metrics keys
                    valid_nodes_for_update = set(current_nodes) & set(self.metrics.keys())

                    if not valid_nodes_for_update:
                        print(f"Warning: No valid nodes for metric update in batch {batch_idx}. Intermediates: {current_nodes}, Metrics: {list(self.metrics.keys())}")
                        continue
                        

                    for node in valid_nodes_for_update:
                        # node should be a PREFIX node ID (int)
                        if not isinstance(node, int): continue # Skip non-integer keys if any creep in

                        node_metrics = self.metrics[node] # Get metrics specific to this node
                        for metric in node_metrics.values():
                            try:
                                # Extract features for the current node from all graphs
                                intermeds_float = [i[node].float().detach() for i in intermediates] # List of [feat_dim, num_tokens_for_node]
                                # Ensure tensors are not empty
                                if any(t.numel() == 0 for t in intermeds_float):
                                    print(f"Warning: Empty intermediate tensor for node {node} in batch {batch_idx}. Skipping metric update for this node.")
                                    continue
                                # --- START: Per-Feature Standardization ---
                                intermeds_standardized = []
                                for model_feats in intermeds_float:
                                    # model_feats shape: [feat_dim_per_model, num_tokens_for_node]
                                    if model_feats.shape[1] > 1: # Need at least 2 vectors for std dev
                                        mean_per_feature = torch.mean(model_feats, dim=1, keepdim=True)
                                        std_per_feature = torch.std(model_feats, dim=1, keepdim=True)
                                        # Add epsilon to std_per_feature to prevent division by zero
                                        standardized_feats = (model_feats - mean_per_feature) / (std_per_feature + 1e-7)
                                        intermeds_standardized.append(standardized_feats)
                                    elif model_feats.shape[1] == 1: # Only one vector, cannot compute std. Center it.
                                        mean_per_feature = torch.mean(model_feats, dim=1, keepdim=True)
                                        centered_feats = model_feats - mean_per_feature
                                        intermeds_standardized.append(centered_feats)
                                    else: # Should be caught by numel() == 0 check, but as a safeguard
                                        intermeds_standardized.append(model_feats) # Append as is if 0 vectors
                                metric.update(len(wavs), *intermeds_standardized)
                            except KeyError:
                                print(f"Warning: Node {node} not found in intermediates for batch {batch_idx}. Skipping metric update.")
                            except Exception as e:
                                print(f"Error updating metric for node {node}, batch {batch_idx}: {e}")
                                # import traceback; traceback.print_exc() # More detailed error
                                # pdb.set_trace()

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    import traceback; traceback.print_exc()
                    # Decide whether to continue or stop
                    # continue

        # --- Metric Finalization ---
        if self.metrics is None:
            print("Error: No metrics were computed. Check intermediate capture and update steps.")
            return None, None

        finalized_metrics = {}
        print(f"Finalizing metrics for nodes: {list(self.metrics.keys())}")
        for node, node_metrics in self.metrics.items():
            finalized_node_metrics = {}
            # Finalize metrics for *all* nodes that were initialized and potentially updated
            print(f"  Finalizing node {node}...")
            for metric_name, metric in node_metrics.items():
                try:
                    finalized_node_metrics[metric_name] = metric.finalize(print_featnorms=print_featnorms)
                    if finalized_node_metrics[metric_name] is None:
                        print(f"    Metric {metric_name} for node {node} failed finalization (returned None).")
                    else:
                        print(f"    Metric {metric_name} finalized.")
                except Exception as e:
                    print(f"    Error finalizing metric {metric_name} for node {node}: {e}")
                    # Decide how to handle - skip metric, skip node, error out?
                    # Store None or skip the metric for this node
                    finalized_node_metrics[metric_name] = None # Example: store None
            finalized_metrics[node] = finalized_node_metrics

        self.metrics = finalized_metrics # Update self.metrics with finalized versions

        # Check if any metrics are None after finalization
        for node, metrics in self.metrics.items():
            if any(v is None for v in metrics.values()):
                print(f"Warning: Node {node} has one or more metrics that failed to finalize.")


        return self.metrics, None # Return finalized metrics              

    def save_features(self, dataloader, sentence_level=False, special_toks=False, 
                        save_feats=False, save_dir=None):
    
        self.metrics = None
        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader
        
        numel = 0
        if save_feats:
            tok_indices = []
            feats = [defaultdict(list),defaultdict(list)]

        for dataloader in dataloader_list:
            batch_count = 0
            for x, lens in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics: "):

                # load batch & track element numbers 
                x = x.to(self.device)
                if sentence_level != None:
                    numel_local = x.shape[0]
                else:
                    numel_local =  sum(lens)
                    if special_toks == False:
                        numel_local =- 2*x.shape[0] # num tokens - BOS/EOS toks 
                numel += numel_local
                    
                # get intermediates and remove padding idxs 
                if 'Bert' in type(self.graphs[0].model).__name__:
                    attn_mask = lengths_to_mask(list(lens))  
                    intermediates =  [g.compute_intermediates(x, attn_mask=attn_mask.long().to(self.device)) for g in self.graphs] # shape [feat_dim, num_tokens]
                else:
                    intermediates = [g.compute_intermediates(x) for g in self.graphs] # shape [feat_dim, num_tokens]
                intermediates = self.remove_pads(intermediates, x, lens, sentence_level, special_toks)

                # store intermediates 
                nodes = list(intermediates[0].keys())
                if save_feats:
                    batch_tok_indices = x.flatten()[torch.argwhere(x.flatten() != 0)].squeeze().detach().cpu()
                    tok_indices.append(batch_tok_indices)
                    for node in nodes:
                        feats[0][node].append(intermediates[0][node].detach().cpu())
                        feats[1][node].append(intermediates[1][node].detach().cpu())

                    # if big enough accumulation, save to file
                    batch_count += 1
                    if batch_count % 1000 == 0:
                        num = batch_count // 1000
                        if batch_count * 8 > 100000:
                            return  
                        print(f'saving features {num}')
                        with open(f'{save_dir}/toks/{num}.pt', 'wb+') as tok_out:
                            torch.save(torch.cat(tok_indices), tok_out) #write toks
                            tok_indices = [] # release memory
                        for model_no in [0, 1]:
                            with open(f'{save_dir}/feats_{model_no}/{num}.pt', 'wb+') as model_out:
                                for node in feats[model_no].keys():
                                    feats[model_no][node] = torch.cat(feats[model_no][node], dim=1)
                                torch.save(feats[model_no], model_out) #write feats
                                feats[model_no] = defaultdict(list) # release memory
                
            if save_feats:
                print('saving last batch')
                num = batch_count // 1000 + 1
                for model_no in [0, 1]:
                    with open(f'{save_dir}/feats_{model_no}/{num}.pt', 'wb+') as model_out:
                        for node in feats[model_no].keys():
                            feats[model_no][node] = torch.cat(feats[model_no][node], dim=1)
                        torch.save(feats[model_no], model_out)
                        feats[model_no] = defaultdict(list)
                with open(f'{save_dir}/toks/{num}.pt', 'wb+') as toks_out:
                    tok_indices = torch.cat(tok_indices)
                    torch.save(tok_indices, toks_out)
                print('finished saving features to file')
        return None, None
    
    ### HELPER FUNCTIONS FOR CORRELATIONS ###

    def compute_np_corr(self, X,Y):
        feats_concat = torch.cat((X.to('cpu'),Y.to('cpu'))).type(torch.float32)
        corr = np.corrcoef(feats_concat)
        corr = np.nan_to_num(corr)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        return corr

    def compute_np_cov(self, X, Y):
        feats_concat = torch.cat((X.to('cpu'),Y.to('cpu'))).type(torch.float32)
        feats_concat =  feats_concat - feats_concat.mean(dim=1)[:,None]
        cov = (feats_concat @ feats_concat.T).div(feats_concat.shape[1])
        return cov
        
    def cov_to_corr(self, cov, no_corr=False):
        if no_corr == True:
            return cov 
        std = torch.diagonal(cov).sqrt()
        corr = cov / (torch.clamp(torch.nan_to_num(torch.outer(std, std)),min=1e-7))
        return corr

    def separate_res_nodes(self, nodes):
        resnodes = []
        non_resnodes = []
        for node in nodes:
            if self.graphs[0].get_node_info(node)['type'] == NodeType.POSTFIX:
                prev_node_info = self.graphs[0].get_node_info(node-1)['layer']
                if ((self.graphs[0].modules['q'] in prev_node_info) or 
                                            (self.graphs[0].modules['k'] in prev_node_info)):
                    #non_resnodes.append(node) # this is a qk node
                    continue
                else:
                    resnodes.append(node) # all res keys are postfixes by design
            else:
                non_resnodes.append(node)
        return resnodes, non_resnodes


    # load certain number of saved feats
    def load_features(self, saved_path, num, res='first', total_num=10):
        filenames = glob.glob(os.path.join(saved_path, f'feats_{num}', '*.pt'))
        filenames = filenames[:total_num]
        feats_final = {}

        print('loading feats')
        for filename in tqdm(filenames):
            try:
                feats = torch.load(filename)
            except RuntimeError:
                continue
            
            # sort nodes by res or non-res
            resnodes, non_resnodes = self.separate_res_nodes(list(feats.keys()))

            # keep resnodes of interest only
            if res == 'first':
                res_keys_used = [resnodes[0]]
            elif res == 'last':
                res_keys_used = [resnodes[-1]]
            elif res == 'all':
                res_keys_used = resnodes 
            elif res == 'sep':
                res_keys_used = resnodes
            elif res == 'none':
                res_keys_used = []

            # go through non resnodes, and get features ready
            for node in non_resnodes:
                if node not in feats_final:
                    feats_final[node] = feats[node]
                else:
                    feats_final[node] = torch.cat([feats_final[node], feats[node]], dim=1)

            for node in res_keys_used:
                if node not in feats_final:
                    feats_final[node] = feats[node]
                else:
                    feats_final[node] = torch.cat([feats_final[node], feats[node]], dim=1)
                 
            for key in resnodes:
                if key not in feats_final:
                    feats_final[key] = []
        return feats_final

    def compute_corrs(self, nodes, feats_0, feats_1, res='first'):
        corrs = {}

        resnodes, non_resnodes = self.separate_res_nodes(nodes)

        for node in tqdm(non_resnodes):
            if feats_0[node] != []:
                corrs[node] = torch.Tensor(self.compute_np_corr(feats_0[node], feats_1[node]))    
            
        if res == 'first':
            resnode = resnodes[0]
            corrs['res'] = torch.Tensor(self.compute_np_corr(feats_0[resnode], feats_1[resnode]))    
        elif res == 'last':
            resnode = resnodes[-1]
            corrs['res'] = torch.Tensor(self.compute_np_corr(feats_0[resnode], feats_1[resnode]))    
        elif res == 'all':
            node = resnodes[0]
            cov = torch.Tensor(self.compute_np_cov(feats_0[node], feats_1[node]))    
            for node in resnodes[1:]:
                cov += torch.Tensor(self.compute_np_cov(feats_0[node], feats_1[node]))    
            cov /= len(resnodes)
            corrs['res'] = torch.Tensor(self.cov_to_corr(cov))
        elif res == 'sep':
            for node in resnodes:
                corrs[node] = torch.Tensor(self.compute_np_corr(feats_0[node], feats_1[node]))
        # not handling 'none' case for now

        return corrs

    def compute_metric_corrs(self, nodes, res='first', no_corr=False, qk=False):
        corrs = {}
        resnodes, non_resnodes = self.separate_res_nodes(nodes)

        
        for node in tqdm(non_resnodes):
            corrs[node] = self.cov_to_corr(self.metrics[node]['covariance'], no_corr)
        
        if resnodes == []:
            return corrs
        if res == 'first':
            resnode = resnodes[0]
            corrs['res'] = self.cov_to_corr(self.metrics[resnode]['covariance'], no_corr=no_corr)
        elif res == 'last':
            resnode = resnodes[-1]
            corrs['res'] = self.cov_to_corr(self.metrics[resnode]['covariance'], no_corr=no_corr)
        elif res == 'all':
            node = resnodes[0]
            cov = self.metrics[node]['covariance']
            for node in resnodes[1:]:
                cov += self.metrics[node]['covariance']
            cov /= len(resnodes)
            corrs['res'] =self.cov_to_corr(cov, no_corr=no_corr)
        elif res == 'sep':
            for node in resnodes:
                corrs[node] = self.cov_to_corr(self.metrics[node]['covariance'], no_corr=no_corr)
        
        return corrs

    ### END HELPER FUNCTIONS FOR CORRELATIONS ###

    def compute_transformations(self, transform_fn, reduce_ratio=.5, permute_heads=False, 
                                ignore_heads=False, print_costs=False, no_absval=False,
                                saved_features=None, res='first',
                                no_corr=False, layer_weights=None, alpha=1.0, enable_weighted_alignment=False, **kwargs):
        
        is_zipit_merge_global = 'zipit' in transform_fn.__name__ # Check global setting
        zipit_applied_nodes = set() # <<< Store nodes where ZipIt! was actually applied

        start_time = time()
        self.merges = {}
        self.unmerges = {}
        cost_dict = {}
        zipit_analysis_results = {} # <<< Initialize dictionary to store analysis results
        global_res_merge= None
        global_res_unmerge = None
        # Define layers that might indicate a special context AFTER the PREFIX node
        # We will check the successor node's layer name against these
        attn_output_indicator = self.graphs[0].modules.get('lin_attn', 'self_attn.out_proj') # Get the actual name used
        qkv_input_indicators = [
            self.graphs[0].modules.get('q', 'self_attn.q_proj'),
            self.graphs[0].modules.get('k', 'self_attn.k_proj'),
            self.graphs[0].modules.get('v', 'self_attn.v_proj')
        ]
        ff_output_indicator = self.graphs[0].modules.get('fc2', 'fc2') # Usually fc2


        if saved_features:
            feats_0 = self.load_features(saved_features, 0, res=res)
            feats_1 = self.load_features(saved_features, 1, res=res)
            nodes = list(feats_0.keys())
            nodes.sort()
            print('computing corrs')
            corrs = self.compute_corrs(nodes, feats_0, feats_1, res=res)
        else:
            nodes = list(self.metrics.keys())
            nodes.sort()
            print(f'Computing correlations from computed metrics for nodes: {nodes}')
            corrs = self.compute_metric_corrs(nodes, res=res, no_corr=no_corr)

        
        # corrs has all nonres nodes & the one res node. Unless this is sep, then it has all nodes
        # Nodes to iterate over are those for which we have correlations
        # Separate residual nodes conceptually first
        all_int_nodes = sorted([k for k in corrs.keys() if isinstance(k, int)])
        resnodes, non_resnodes = self.separate_res_nodes(all_int_nodes)
        nodes_to_process_main = non_resnodes # Process non-residuals first

        print(f"Nodes for main processing loop: {nodes_to_process_main}")
        print(f"Residual nodes to process later (if res='sep' or globally): {resnodes}")
        if 'res' in corrs:
            print("Global residual correlation ('res') key found.")
        
        # --- Determine if using ZipIt! ---
        is_zipit_merge = 'zipit' in transform_fn.__name__
        if is_zipit_merge:
            print("Using ZipIt! style merging. MHA-specific permutation logic will be skipped.")


        # --- Main Transformation Loop (Non-Residual Nodes) ---
        for node in tqdm(nodes_to_process_main, desc="Computing non-residual transformations"):
            if node not in corrs or corrs[node] is None:
                print(f"Warning: Skipping node {node}, no correlation found.")
                continue

            correlation_matrix = corrs[node]
            info = self.graphs[0].get_node_info(node) # PREFIX node info
            print(f"\nProcessing Node {node}: {info}")

            # ******************************************************************
            # ***** START: Weighted Alignment Modification *****
            # ******************************************************************
            current_correlation_matrix = correlation_matrix # Default to original
            layer_idx_successor = None # Layer index of the module *after* the PREFIX node
            scale_applied = 1.0

            if enable_weighted_alignment and layer_weights and info['type'] == NodeType.PREFIX:
                # Identify the layer index relevant to this PREFIX node
                layer_idx_successor = self._get_layer_idx_from_node_info(info)

                if layer_idx_successor is not None:
                    # The PREFIX before layer L+1 aligns the output of layer L
                    print(f"layer_weights: {layer_weights}")
                    layer_idx_output = layer_idx_successor - 1

                    importance = layer_weights.get(layer_idx_output, 0.0) # Get importance, default 0

                    if importance > 0:
                        scale = (1.0 + alpha * importance)
                        print(f"  Applying weighted alignment scaling for layer {layer_idx_output} output.")
                        print(f"    Node {node}, Layer Index Output: {layer_idx_output}, Importance: {importance:.4f}, Alpha: {alpha}, Scale: {scale:.4f}")

                        # Clone and scale the cross-correlation block
                        current_correlation_matrix = correlation_matrix.clone()
                        Om = current_correlation_matrix.shape[0] // 2 # Assuming 2 models
                        # Scale the block used by match_tensors_permute (HuBERT vs MERT)
                        current_correlation_matrix[:Om, Om:] *= scale
                        # Optional: scale the symmetric block too?
                        # current_correlation_matrix[Om:, :Om] *= scale
                        scale_applied = scale # For logging cost later if needed
                    else:
                         print(f"  No scaling for Node {node} (Layer Index Output: {layer_idx_output}, Importance: 0)")

                else:
                     print(f"  Skipping scaling for Node {node}: Not identified as transformer layer PREFIX or failed to parse index.")




            # --- Determine Context based on Successor ---
            call_mha_variant = False # Default to False
            if not is_zipit_merge: # Only check context if NOT using ZipIt
                successor_module_node = None
                successor_layer_name = None
                try: # Use try-except for robustness in graph traversal
                    queue = list(self.graphs[0].succs(node))
                    visited = set(queue); visited.add(node)
                    while queue:
                        current_succ = queue.pop(0)
                        succ_info = self.graphs[0].get_node_info(current_succ)
                        if succ_info['type'] == NodeType.MODULE:
                            successor_module_node = current_succ
                            successor_layer_name = succ_info['layer']
                            print(f"  Successor module found: Node {successor_module_node}, Layer: {successor_layer_name}")
                            break
                        for s in self.graphs[0].succs(current_succ):
                            if s not in visited: visited.add(s); queue.append(s)
                except Exception as e:
                    print(f"  Error finding successor module for node {node}: {e}")
                    successor_layer_name = None

                if successor_layer_name is None:
                    print(f"Warning: No MODULE successor found for PREFIX node {node}. Applying default transform.")
                    # Decide default behavior - apply base transform?
                    call_mha_variant = False
                else:
                    # Check if the successor indicates an MHA context (Attn Output or QKV Input)
                    is_attn_out = attn_output_indicator in successor_layer_name
                    is_qkv_in = any(qkv_ind in successor_layer_name for qkv_ind in qkv_input_indicators)

                    call_mha_variant = (is_attn_out or is_qkv_in) and \
                                    'match_tensors_permute' in transform_fn.__name__ and \
                                    not ignore_heads

            
            # --- Select and Compute Transformation ---
            merge, unmerge, extra_info, cost = None, None, None, None
            global_unmerge_matrix_for_analysis = None # <<< To store the global matrix for analysis

            try:
                actual_transform_fn = transform_fn # Default to the main function
                current_kwargs = {**kwargs, 'correlation_matrix': current_correlation_matrix, 'print_costs': print_costs, 'no_absval': no_absval}

                if call_mha_variant and not is_zipit_merge: # Use MHA variant only if applicable and not ZipIt:
                    context_type = "Attention Output" if is_attn_out else "QKV Input"
                    print(f"  Using MHA-specific transform for {context_type}...")
                    n_heads = self.graphs[0].num_heads
                    mha_transform_fn_name = transform_fn.__name__.replace('_MHA','') + '_MHA'
                    try:
                        actual_transform_fn = get_merging_fn(mha_transform_fn_name)
                        print(f"    Calling: {mha_transform_fn_name}")
                        mha_kwargs = {'n_heads': n_heads, 'permute_heads': permute_heads, **kwargs}
                        merge, unmerge, attn_head_perm, cost = actual_transform_fn(
                            r=reduce_ratio,
                            correlation_matrix=current_correlation_matrix,
                            print_costs=print_costs, no_absval=no_absval, **mha_kwargs
                        )
                        extra_info = attn_head_perm
                    except KeyError:
                        print(f"    Warning: MHA variant '{mha_transform_fn_name}' not found. Falling back.")
                        call_mha_variant = False # Force fallback

                else:
                    if is_zipit_merge:
                         print(f"  Using ZipIt! transform '{actual_transform_fn.__name__}'...")
                         zipit_applied_nodes.add(node)
                         zipit_metric_dict = {'covariance': corrs[node]} # Reconstruct metric dict if needed
                         # Check if 'mean' metric exists for this node
                         if 'mean' in self.metrics.get(node, {}):
                            zipit_metric_dict['mean'] = self.metrics[node]['mean']
                         # Check if 'magnitudes' are expected/computed (ZipIt uses this)
                         # If MeanMetric computes magnitudes, add them:
                         # if 'magnitudes' in self.metrics.get(node, {}).get('mean', {}): # Hypothetical
                         #     zipit_metric_dict['magnitudes'] = self.metrics[node]['mean']['magnitudes']

                         # Call ZipIt! - it expects 'metric' as first arg
                         # Note: ZipIt returns merge.T, unmerge. Adjust if needed.
                         # merge_T, unmerge = actual_transform_fn(
                         #     zipit_metric_dict, r=reduce_ratio, **kwargs # Pass original ZipIt args if needed
                         # )
                         # merge = merge_T.T # Transpose back if needed

                         # OR If your match_tensors_zipit is adapted for correlation_matrix kwarg:
                         # Check if it returns cost (the provided one doesn't explicitly)
                         # Let's assume it returns (merge.T, unmerge, maybe_merge_value)
                         zipit_kwargs = {
                                'metric': zipit_metric_dict,
                                'r': reduce_ratio,
                                'a': kwargs.get('a', 0.3), # Use provided or default
                                'b': kwargs.get('b', 0.125), # Use provided or default
                                'print_merges': kwargs.get('print_merges', False), # Allow passing other ZipIt args
                                'get_merge_value': kwargs.get('get_merge_value', False),
                                'add_bias': kwargs.get('add_bias', False)
                            }
                         merge_T, global_unmerge_matrix_for_analysis, _, cost = actual_transform_fn(
                              metric=zipit_metric_dict, # Pass the metric dict
                              r=reduce_ratio,
                              # Pass other ZipIt specific hypers like a, b if needed via **kwargs
                              **kwargs
                         )
                         merge = merge_T
                         unmerge = global_unmerge_matrix_for_analysis # Use captured
                         # ZipIt cost isn't directly returned, maybe calculate from merge_value?
                         print(f"    ZipIt Merge shape: {merge.shape}, Unmerge shape: {unmerge.shape}")
                    else:
                        context_type = f"FF/CNN/Other (Successor: {successor_layer_name or 'Unknown'})"
                        print(f"  Using base transform '{transform_fn.__name__}' for {context_type}...")
                        actual_transform_fn = transform_fn
                        # Ensure reduce_ratio is passed correctly (might be positional or keyword)
                        # Check transform_fn signature if needed
                        merge, unmerge, _, cost = actual_transform_fn(
                            reduce_ratio, # Assuming it's the first arg after potential self/cls
                            correlation_matrix=current_correlation_matrix,
                            print_costs=print_costs, no_absval=no_absval, **kwargs
                        )

                # --- Store Results ---
                if merge is None or unmerge is None: raise ValueError("Transform returned None.")

                # <<< Perform ZipIt Analysis (if applicable) >>>
                if is_zipit_merge and unmerge is not None and len(self.graphs) == 2:
                    num_graphs = len(self.graphs)
                    total_dim_analysis = unmerge.shape[0]
                    if total_dim_analysis % num_graphs == 0:
                         dim_per_graph_analysis = total_dim_analysis // num_graphs
                         dims_per_graph_analysis = [dim_per_graph_analysis] * num_graphs
                         analysis = analyze_zipit_unmerge_matrix(unmerge, dims_per_graph_analysis, node)
                         if analysis:
                             zipit_analysis_results[node] = analysis # <<< Store analysis
                             print(f"  ZipIt Analysis Node {node}: {analysis}")
                    else:
                         print(f"Warning Node {node}: Cannot perform ZipIt analysis - dimensions not divisible ({total_dim_analysis}/{num_graphs})")


                num_graphs = len(self.graphs)
                # Check if dimensions are compatible before chunking
                print(f"Node {node}: Raw Merge shape: {merge.shape}, Raw Unmerge shape: {unmerge.shape}")
                print(f"           Expected input dim for merge: {merge.shape[1]}, Expected output dim for unmerge: {unmerge.shape[0]}")

                total_dim = merge.shape[1] # Get TotalOriginalDim from rows of merge/unmerge

                # Verify total_dim matches sum(dims_per_graph) - needed for chunking
                if total_dim % num_graphs != 0:
                     print(f"ERROR Node {node}: Total dimension ({total_dim}) is not divisible by number of graphs ({num_graphs}).")
                     # This indicates a mismatch between graph info and computed correlation matrix size
                     # Check metric computation and graph construction.
                     raise ValueError("Dimension mismatch in compute_transformations")


                # Chunk the global ZipIt matrices:
                # merge chunks: merge_i = merge[:, start_i:end_i] -> shape [FinalDim, dim_i]
                # unmerge chunks: unmerge_i = unmerge[start_i:end_i, :] -> shape [dim_i, FinalDim]
                dim_per_graph = total_dim // num_graphs
                dims_to_split_by = [dim_per_graph] * num_graphs # Create the list
                # --- Chunk the matrices ---
                try:
                    if is_zipit_merge:
                        # Split BOTH merge and unmerge along dim 0 (TotalOriginalDim)
                        # merge_chunks_i shape: [dim_i, FinalDim]
                        merge_chunks = torch.split(merge_T, dims_to_split_by, dim=1)
                        # unmerge_chunks_i shape: [dim_i, FinalDim]
                        unmerge_chunks = torch.split(unmerge, dims_to_split_by, dim=0)

                        print(f"    Chunked Merge shapes: {[c.shape for c in merge_chunks]}")
                        print(f"    Chunked Unmerge shapes: {[c.shape for c in unmerge_chunks]}")
                    else:
                        print("  Chunking Permutation matrices...")
                        merge_chunks = torch.split(merge, dims_to_split_by, dim=1)
                        # Split unmerge along TotalDim (dim 0) -> chunks [dim_i, Om]
                        unmerge_chunks = torch.split(unmerge, dims_to_split_by, dim=0)
                        print(f"    Permutation Chunked Merge shapes: {[c.shape for c in merge_chunks]}")
                        print(f"    Permutation Chunked Unmerge shapes: {[c.shape for c in unmerge_chunks]}")

                except RuntimeError as e:
                     print(f"ERROR Node {node}: Failed to split matrices.")
                     print(f"  Merge shape: {merge_T.shape}, Unmerge shape: {unmerge.shape}")
                     print(f"  Trying to split dim 0 ({total_dim}) with sizes: {dims_to_split_by}") # Verify sizes here
                     pdb.set_trace() # Break again if it still fails
                     raise e
    

                # Check chunk shapes:
                # print(f"    Merge chunk shapes: {[c.shape for c in merge_chunks]}")
                # print(f"    Unmerge chunk shapes: {[c.shape for c in unmerge_chunks]}")

                # Store the chunked matrices
                self.merges[node] = merge_chunks     # List of tensors, one per graph [FinalDim, dim_i]
                self.unmerges[node] = unmerge_chunks # List of tensors, one per graph [dim_i, FinalDim]

                # For permutation methods, the structure was different:
                # merge was [TotalDim, Om], unmerge was [TotalDim, Om]
                # merge chunks were [dim_i, Om], unmerge chunks were [dim_i, Om]
                # This seems inconsistent with how merge/unmerge are applied.

                # Let's stick to the ZipIt interpretation for now:
                # merge_i: [FinalDim, dim_i]
                # unmerge_i: [dim_i, FinalDim]

                # Re-check application logic in merge_node/unmerge_node if issues arise.

                cost_dict[node] = cost if cost is not None else float('nan') # Use NaN if cost not available
                print(f"  Transformation computed for node {node}. Cost: {cost_dict[node]:.4f}")

            except Exception as e:
                print(f"Error computing transformation for node {node}: {e}")
                import traceback; traceback.print_exc()
                cost_dict[node] = float('nan')
                # pdb.set_trace() # Optional breakpoint


        # --- Handle Residual Nodes ---
        print("\nProcessing Residual Nodes...")
        # Residual node logic seems compatible with ZipIt, as it also operates on correlations
        if res == 'sep':
            # Process each residual node individually
            for node in tqdm(resnodes, desc="Computing separate residual transformations"):
                 if node not in corrs or corrs[node] is None:
                      print(f"Warning: Skipping residual node {node}, no correlation found.")
                      cost_dict[node] = float('nan')
                      continue
                 correlation_matrix = corrs[node]
                 info = self.graphs[0].get_node_info(node)
                 print(f"Processing Residual Node {node} separately: {info}")
                 try:
                      current_kwargs = {**kwargs, 'correlation_matrix': correlation_matrix, 'print_costs': print_costs, 'no_absval': no_absval}
                      if is_zipit_merge:
                           zipit_metric_dict = {'covariance': corrs[node]}
                           # Add mean/magnitudes if available/needed
                           merge_T, global_unmerge_res_for_analysis, _, cost = actual_transform_fn(zipit_metric_dict, r=reduce_ratio, **kwargs)
                           unmerge = global_unmerge_res_for_analysis
                           merge = merge_T
                           print(f"    ZipIt Res Merge shape: {merge.shape}, Unmerge shape: {unmerge.shape}")
                      else:
                           merge, unmerge, _, cost = transform_fn(reduce_ratio, **current_kwargs)

                      if merge is None or unmerge is None: raise ValueError("Transform returned None.")

                      # <<< Analyze residual ZipIt merge >>>
                      if is_zipit_merge and global_unmerge_res_for_analysis is not None and len(self.graphs) == 2:
                           num_graphs_res = len(self.graphs)
                           total_dim_res_analysis = global_unmerge_res_for_analysis.shape[0]
                           if total_dim_res_analysis % num_graphs_res == 0:
                                dim_per_graph_res_analysis = total_dim_res_analysis // num_graphs_res
                                dims_per_graph_res_analysis = [dim_per_graph_res_analysis] * num_graphs_res
                                analysis_res = analyze_zipit_unmerge_matrix(global_unmerge_res_for_analysis, dims_per_graph_res_analysis, f"ResNode {node}")
                                if analysis_res:
                                     zipit_analysis_results[node] = analysis_res # <<< Store analysis
                                     print(f"  ZipIt Analysis ResNode {node}: {analysis_res}")
                           else:
                                print(f"Warning ResNode {node}: Cannot perform ZipIt analysis - dimensions not divisible ({total_dim_res_analysis}/{num_graphs_res})")

                      # Chunk residual matrices like non-residual ones
                      num_graphs_res = len(self.graphs)
                      total_dim_res = merge.shape[1]


                      if total_dim_res % num_graphs_res != 0:
                            print(f"ERROR ResNode {node or 'Global'}: Total dimension ({total_dim_res}) not divisible by num graphs ({num_graphs_res}).")
                            raise ValueError("Dimension mismatch")
                      dim_per_graph_res = total_dim_res // num_graphs_res
                      dims_per_graph_res = [dim_per_graph_res] * num_graphs_res
                      
                      if is_zipit_merge:
                        self.merges[node] = torch.split(merge_T, dims_per_graph_res, dim=0)
                        self.unmerges[node] = torch.split(unmerge, dims_per_graph_res, dim=0)
                      else:
                          self.merges[node] = torch.split(merge, dims_per_graph_res, dim=0)
                          self.unmerges[node] = torch.split(unmerge, dims_per_graph_res, dim=0)
                        
                      cost_dict[node] = cost if cost is not None else float('nan')
                      print(f"  Transformation computed for residual node {node}. Cost: {cost_dict[node]:.4f}")
                 except Exception as e:
                      print(f"Error computing transformation for residual node {node}: {e}")
                      cost_dict[node] = float('nan')
                      # pdb.set_trace()


        elif 'res' in corrs and corrs['res'] is not None:
            # Use the global 'res' correlation
            # --- Variables to store results of computation ---
            global_res_merge = None
            global_res_unmerge = None
            global_unmerge_gres_for_analysis = None # For ZipIt analysis

            # --- Compute Global Residual Transformation (only once) ---
            # Check if already computed (e.g. if this code runs multiple times, though unlikely)
            if 'res' not in self.merges and 'res' not in cost_dict:
                print(f"Computing global residual transformation using 'res' correlation matrix...")
                correlation_matrix_res = corrs['res']
                try:
                    current_kwargs = {**kwargs, 'correlation_matrix': correlation_matrix_res, 'print_costs': print_costs, 'no_absval': no_absval}

                    if is_zipit_merge:
                        print(f"  Applying ZipIt! to global residual...")
                        zipit_metric_dict = {'covariance': corrs['res']}
                        # Add mean/magnitudes if needed...
                        merge_T, global_unmerge_gres_for_analysis, _ , cost = actual_transform_fn( # Capture global unmerge
                            metric=zipit_metric_dict,
                            r=reduce_ratio,
                            **kwargs
                        )
                        global_res_merge = merge_T
                        global_res_unmerge = global_unmerge_gres_for_analysis # Use captured
                        cost_res = None # ZipIt doesn't return cost directly
                        print(f"    ZipIt Global Res Merge shape: {global_res_merge.shape}, Unmerge shape: {global_res_unmerge.shape}")
                    else:
                        print(f"  Applying {transform_fn.__name__} to global residual...")
                        # Standard transform call
                        global_res_merge, global_res_unmerge, _, cost_res = transform_fn(
                            reduce_ratio,
                            **current_kwargs # Pass correlation_matrix etc.
                        )

                    # Check if computation was successful
                    if global_res_merge is not None and global_res_unmerge is not None:
                        cost_dict['res'] = cost_res if cost_res is not None else float('nan')
                        print(f"  Global residual transformation computed. Cost: {cost_dict['res']:.4f}")

                        # --- Perform ZipIt Analysis for Global Residual ---
                        if is_zipit_merge and global_unmerge_gres_for_analysis is not None and len(self.graphs) == 2:
                             # Deduce dimensions for analysis (assuming consistency)
                             num_graphs_gres = len(self.graphs)
                             total_dim_gres_analysis = global_unmerge_gres_for_analysis.shape[0] # Total original dim

                             if total_dim_gres_analysis % num_graphs_gres == 0:
                                 dim_per_graph_gres_analysis = total_dim_gres_analysis // num_graphs_gres
                                 dims_per_graph_gres_analysis = [dim_per_graph_gres_analysis] * num_graphs_gres
                                 analysis_gres = analyze_zipit_unmerge_matrix(global_unmerge_gres_for_analysis, dims_per_graph_gres_analysis, "GlobalRes")
                                 if analysis_gres:
                                     zipit_analysis_results['res'] = analysis_gres # Store analysis
                                     print(f"  ZipIt Analysis GlobalRes: {analysis_gres}")
                             else:
                                 print(f"Warning: Cannot perform ZipIt analysis on GlobalRes - dimensions not divisible ({total_dim_gres_analysis} / {num_graphs_gres})")

                    else:
                        print("Warning: Global residual transformation computation returned None.")
                        cost_dict['res'] = float('nan')
                        # Ensure variables are None if computation failed
                        global_res_merge, global_res_unmerge, global_unmerge_gres_for_analysis = None, None, None

                except Exception as e:
                    print(f"Error computing global residual transformation: {e}")
                    import traceback; traceback.print_exc()
                    cost_dict['res'] = float('nan')
                    # Ensure variables are None on error
                    global_res_merge, global_res_unmerge, global_unmerge_gres_for_analysis = None, None, None

            # --- Assign Global Transformations to Residual Nodes ---
            if global_res_merge is not None and global_res_unmerge is not None:
                 print(f"Assigning global residual transform to nodes: {resnodes}")
                 # <<< Deduce dimensions and chunk (using deduced dimensions) >>>
                 num_graphs_gres = len(self.graphs)
                 total_dim_gres = global_res_merge.shape[1]
                 if total_dim_gres % num_graphs_gres != 0:
                     print(f"ERROR GlobalRes Assign: Cannot chunk.")
                 elif not resnodes:
                     print("Warning: No residual nodes found.")
                 else:
                     dim_per_graph_gres = total_dim_gres // num_graphs_gres
                     dims_per_graph_gres = [dim_per_graph_gres] * num_graphs_gres
                     print(f"GlobalRes Assign: Deduced DimPerGraph={dim_per_graph_gres}")
                     try:
                         global_merge_chunks = torch.split(global_res_merge.T, dims_per_graph_gres, dim=0) # changed for .T for zipit consistency?
                         global_unmerge_chunks = torch.split(global_res_unmerge, dims_per_graph_gres, dim=0)
                         assigned_count = 0
                         for resnode in resnodes:
                             if resnode not in self.merges:
                                 self.merges[resnode] = global_merge_chunks
                                 self.unmerges[resnode] = global_unmerge_chunks
                                 cost_dict[resnode] = cost_dict.get('res', float('nan'))
                                 assigned_count += 1
                         print(f"Assigned global residual transform to {assigned_count} nodes.")
                     except Exception as e:
                         print(f"Error during chunking/assignment of global residual transform: {e}")
            else:
                 print("Skipping assignment of global residual transform.")
        
        else:
            # This case means 'res' key didn't exist in corrs or was None
            print("Skipping global residual processing: 'res' key not found in correlations or correlation is None.")


        # --- Final Steps (Timing, Return) ---
        self.compute_transform_time = time() - start_time
        print(f"Transformation computation finished in {self.compute_transform_time:.2f} seconds.")
        print(f"Computed transformations (merge/unmerge matrices stored) for nodes: {sorted(list(self.merges.keys()))}")
        self.zipit_applied_nodes = zipit_applied_nodes



        return self.merges, self.unmerges, cost_dict, zipit_analysis_results

    
    def find_previous_weighted_node(self, node, graph):
        """Find the nearest predecessor node with trainable parameters (e.g., Conv1d, GroupNorm with affine=True)."""
        current = node
        while True:
            preds = graph.preds(current)
            if not preds:
                return None
            current = preds[0]  # Assuming single predecessor for simplicity
            info = graph.get_node_info(current)
            if info['type'] == NodeType.MODULE:
                module = graph.get_module(info['layer'])
                if isinstance(module, (nn.Conv1d, nn.Linear)) or (isinstance(module, nn.GroupNorm) and module.affine):
                    return preds

        
    def find_next_parameterized_successor(self, node, graph):
        current = node
        while True:
            succs = graph.succs(current)
            if not succs:
                return None
            current = succs[0]  # Assuming single successor
            info = graph.get_node_info(current)
            module = graph.get_module(info['layer'])
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.GroupNorm)) and hasattr(module, 'weight'):
                return succs[0]
    
    def find_closest_parameterized_predecessor(self, graph, start_node):
        """Traverses backwards from start_node to find the nearest MODULE node with parameters."""
        queue = list(graph.preds(start_node))
        visited = set(queue)
        visited.add(start_node)

        while queue:
            current_node = queue.pop(0)
            info = graph.get_node_info(current_node)

            if info['type'] == NodeType.MODULE:
                module = graph.get_module(info['layer'])
                # Check if the module has trainable parameters we care about merging/unmerging
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) or \
                (isinstance(module, (nn.LayerNorm, nn.GroupNorm)) and hasattr(module, 'weight') and module.weight is not None):
                    # Check specifically for affine=True in GroupNorm if needed
                    if isinstance(module, nn.GroupNorm) and not module.affine:
                        pass # Skip non-affine GroupNorm
                    else:
                        return current_node # Found parameterized module

            # Continue BFS backwards
            for pred in graph.preds(current_node):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return None # No parameterized predecessor found

    def find_closest_parameterized_successor(self, graph, start_node):
        """Traverses forwards from start_node to find the nearest MODULE node with parameters."""
        queue = list(graph.succs(start_node))
        visited = set(queue)
        visited.add(start_node)

        while queue:
            current_node = queue.pop(0)
            info = graph.get_node_info(current_node)

            if info['type'] == NodeType.MODULE:
                try: # Handle cases where layer might not exist if graph is incomplete
                    module = graph.get_module(info['layer'])
                    # Check if the module has trainable parameters
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) or \
                    (isinstance(module, (nn.LayerNorm, nn.GroupNorm)) and hasattr(module, 'weight') and module.weight is not None):
                        if isinstance(module, nn.GroupNorm) and not module.affine:
                            pass # Skip non-affine GroupNorm
                        else:
                            return current_node # Found parameterized module
                except KeyError:
                    print(f"Warning: Layer {info['layer']} not found in named_modules during successor search from node {start_node}.")


            # Continue BFS forwards
            for succ in graph.succs(current_node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return None # No parameterized successor found



    def merge_node(self, node, merger):
        # merger contains: graph, merge chunk [FinalDim, OrigDim_i], unmerge chunk [OrigDim_i, FinalDim], orig_node_id

        # --- Prepare for Debugging ---
        # Need access to the global merging function name used
        # Assuming self.transform_fn holds the function object
        is_zipit_merge = 'zipit' in self.transform_fn.__name__ if hasattr(self, 'transform_fn') else False
        # Check if the *original* PREFIX node that triggered this application matches the debug target
        # We also check is_zipit_merge to only debug during ZipIt runs
        do_debug = True

        # --- Standard merge_node logic ---
        info = merger.graph.get_node_info(node)
        graph = merger.graph
        if info['type'] != NodeType.MODULE: return # Skip non-modules
        try:
            module = graph.get_module(info['layer'])
            layer_name = info['layer']
            layer_type = type(module).__name__

            # <<< DEBUG Step 4: Pre-Application Checks >>>
            if do_debug:
                print(f"\n[DEBUG Step 4 @ PredNode {node}] Preparing MERGE:")
                print(f"  Target Layer: {layer_name} (Type: {layer_type})")
                print(f"  Merger.merge chunk shape: {merger.merge.shape}") # Expected: [FinalDim, OrigDim_i]

            # --- Check and Log Shapes for Weight ---
            if hasattr(module, 'weight') and module.weight is not None:
                target_weight_shape = module.weight.shape
                merger_merge_shape = merger.merge.shape
                if do_debug:
                     print(f"  Target weight shape: {target_weight_shape}")
                # Compatibility Check (Merge operation specific)
                compatible_weight = False
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.GroupNorm)):
                    # Check: merge @ weight -> merger.merge.shape[1] == weight.shape[0]
                    if merger_merge_shape[1] == target_weight_shape[0]:
                        compatible_weight = True
                    else:
                         if do_debug: print(f"  ERROR: Shape mismatch for WEIGHT merge! {merger_merge_shape[1]} != {target_weight_shape[0]}")
                else:
                     if do_debug: print(f"  Skipping compatibility check for unhandled weight type {layer_type}")

                if do_debug and not compatible_weight and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.GroupNorm)):
                    pdb.set_trace() # Break if incompatible for handled types

                # --- Prepare for Step 5 ---
                weight_before = module.weight.data.clone().detach()
                norm_before = torch.linalg.norm(weight_before).item() if weight_before.numel() > 0 else 0.0
            else:
                weight_before = None

            # --- Check and Log Shapes for Bias ---
            if hasattr(module, 'bias') and module.bias is not None:
                target_bias_shape = module.bias.shape
                merger_merge_shape = merger.merge.shape
                if do_debug:
                     print(f"  Target bias shape: {target_bias_shape}")
                # Compatibility Check: merge @ bias -> merger.merge.shape[1] == bias.shape[0]
                compatible_bias = False
                if merger_merge_shape[1] == target_bias_shape[0]:
                     compatible_bias = True
                else:
                     if do_debug: print(f"  ERROR: Shape mismatch for BIAS merge! {merger_merge_shape[1]} != {target_bias_shape[0]}")

                if do_debug and not compatible_bias:
                    pdb.set_trace()

                # --- Prepare for Step 5 ---
                bias_before = module.bias.data.clone().detach()
                bias_norm_before = torch.linalg.norm(bias_before).item() if bias_before.numel() > 0 else 0.0
            else:
                bias_before = None
            # <<< End DEBUG Step 4 >>>

            # --- Actual Transformation (using original weights for safety in debug) ---
            if isinstance(module, nn.Linear):
                if weight_before is not None and compatible_weight: module.weight.data = merger.merge @ weight_before
                if bias_before is not None and compatible_bias: module.bias.data = merger.merge @ bias_before
            elif isinstance(module, nn.Conv1d):
                 if weight_before is not None and compatible_weight: module.weight.data = torch.einsum('UO,OIK->UIK', merger.merge, weight_before)
                 if bias_before is not None and compatible_bias: module.bias.data = merger.merge @ bias_before
            elif isinstance(module, nn.Conv2d):
                 if weight_before is not None and compatible_weight: module.weight.data = torch.einsum('UO,OIHW->UIHW', merger.merge, weight_before)
                 if bias_before is not None and compatible_bias: module.bias.data = merger.merge @ bias_before
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                 if hasattr(module, 'weight') and weight_before is not None and compatible_weight: module.weight.data = merger.merge @ weight_before
                 if hasattr(module, 'bias') and bias_before is not None and compatible_bias: module.bias.data = merger.merge @ bias_before
            else:
                 # Only print warning if debugging is active for this node trigger
                 if do_debug: print(f"  Skipping merge application for unhandled layer type: {layer_type}")

            # <<< DEBUG Step 5: Post-Application Checks >>>
            if do_debug:
                print(f"[DEBUG Step 5 @ PredNode {node}] Post-MERGE Checks:")
                if weight_before is not None:
                    weight_after = module.weight.data.clone().detach()
                    norm_after = torch.linalg.norm(weight_after).item() if weight_after.numel() > 0 else 0.0
                    changed = not torch.equal(weight_before, weight_after)
                    print(f"  Weight changed? {changed} (Norm: {norm_before:.4f} -> {norm_after:.4f})")
                    if torch.isnan(weight_after).any(): print("  ERROR: Weight contains NaN after merge!")
                    if not changed and compatible_weight and norm_before > 1e-6 : print("  WARNING: Weight merge applied but tensor didn't change significantly.")
                if bias_before is not None:
                     bias_after = module.bias.data.clone().detach()
                     bias_norm_after = torch.linalg.norm(bias_after).item() if bias_after.numel() > 0 else 0.0
                     changed = not torch.equal(bias_before, bias_after)
                     print(f"  Bias changed? {changed} (Norm: {bias_norm_before:.4f} -> {bias_norm_after:.4f})")
                     if torch.isnan(bias_after).any(): print("  ERROR: Bias contains NaN after merge!")
                     if not changed and compatible_bias and bias_norm_before > 1e-6 : print("  WARNING: Bias merge applied but tensor didn't change significantly.")
                # pdb.set_trace() # Optional breakpoint after merge application
            # <<< End DEBUG Step 5 >>>

        except KeyError:
            # Only print warning if debugging, otherwise KeyErrors might be expected if graph is imperfect
            if do_debug: print(f"Warning: Layer {info.get('layer', 'N/A')} for node {node} not found in named_modules during merge.")
        except Exception as e:
            print(f"ERROR during merge_node for node {node}, layer {info.get('layer', 'N/A')}: {e}")
            import traceback; traceback.print_exc()
            raise
    
    def unmerge_node(self, node, merger):
        # merger contains: graph, merge chunk [FinalDim, OrigDim_i], unmerge chunk [OrigDim_i, FinalDim], orig_node_id

        # --- Prepare for Debugging ---
        is_zipit_merge = 'zipit' in self.transform_fn.__name__ if hasattr(self, 'transform_fn') else False
        #DEBUG_NODE_ID = getattr(self, '_DEBUG_NODE_ID_INTERNAL', None)
        do_debug = True

        # --- Standard unmerge_node logic ---
        graph = merger.graph
        info = merger.graph.get_node_info(node)
        if info['type'] != NodeType.MODULE: return
        try:
            module = graph.get_module(info['layer'])
            layer_name = info['layer']
            layer_type = type(module).__name__

            # <<< DEBUG Step 4: Pre-Application Checks >>>
            if do_debug:
                 print(f"\n[DEBUG Step 4 @ SuccNode {node}] Preparing UNMERGE:")
                 print(f"  Target Layer: {layer_name} (Type: {layer_type})")
                 print(f"  Merger.unmerge chunk shape: {merger.unmerge.shape}") # Expected: [OrigDim_i, FinalDim]

            # --- Skip Norm Layers Directly ---
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if do_debug: print(f"  Skipping parameter unmerge for Norm layer: {layer_name}")
                # Optional: Implement propagation logic here if needed, call unmerge_node recursively
                # succ_after_norm = self.find_closest_parameterized_successor(graph, node)
                # if succ_after_norm and succ_after_norm != node:
                #    if do_debug: print(f"  Propagating unmerge past Norm to Node {succ_after_norm}")
                #    self.unmerge_node(succ_after_norm, merger) # Recursive call
                return # Stop processing this Norm node

            # --- Check and Log Shapes for Weight ---
            if hasattr(module, 'weight') and module.weight is not None:
                target_weight_shape = module.weight.shape
                merger_unmerge_shape = merger.unmerge.shape
                if do_debug:
                    print(f"  Target weight shape: {target_weight_shape}")
                # Compatibility Check (Unmerge operation specific)
                compatible_weight = False
                if isinstance(module, nn.Linear):
                    # Check: weight @ unmerge -> weight.shape[1] == unmerge.shape[0]
                    if target_weight_shape[1] == merger_unmerge_shape[0]: compatible_weight = True
                    else:
                        if do_debug: print(f"  ERROR: Shape mismatch for Linear WEIGHT unmerge! {target_weight_shape[1]} != {merger_unmerge_shape[0]}")
                elif isinstance(module, nn.Conv1d):
                    # Check: einsum('OIK,IU->OUK', W, U) -> W.shape[1] == U.shape[0]
                    if target_weight_shape[1] == merger_unmerge_shape[0]: compatible_weight = True
                    else:
                        if do_debug: print(f"  ERROR: Shape mismatch for Conv1D WEIGHT unmerge! {target_weight_shape[1]} != {merger_unmerge_shape[0]}")
                elif isinstance(module, nn.Conv2d):
                     # Check: einsum('OIHW,IU->OUHW', W, U) -> W.shape[1] == U.shape[0]
                     if target_weight_shape[1] == merger_unmerge_shape[0]: compatible_weight = True
                     else:
                         if do_debug: print(f"  ERROR: Shape mismatch for Conv2D WEIGHT unmerge! {target_weight_shape[1]} != {merger_unmerge_shape[0]}")
                else:
                     if do_debug: print(f"  Skipping compatibility check for unhandled weight type {layer_type}")

                if do_debug and not compatible_weight and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    pdb.set_trace()

                # --- Prepare for Step 5 ---
                weight_before = module.weight.data.clone().detach()
                norm_before = torch.linalg.norm(weight_before).item() if weight_before.numel() > 0 else 0.0
            else:
                weight_before = None
            # <<< End DEBUG Step 4 >>>

            # --- Actual Transformation ---
            if isinstance(module, nn.Linear):
                if weight_before is not None and compatible_weight: module.weight.data = weight_before @ merger.unmerge
                # Bias is usually applied *after* the linear transform, so it's not affected by input unmerge
            elif isinstance(module, nn.Conv1d):
                 if weight_before is not None and compatible_weight: module.weight.data = torch.einsum('OIK,IU->OUK', weight_before, merger.unmerge)
            elif isinstance(module, nn.Conv2d):
                 if weight_before is not None and compatible_weight: module.weight.data = torch.einsum('OIHW,IU->OUHW', weight_before, merger.unmerge)
            else:
                # Attempt propagation only if not a Norm layer (handled above)
                if do_debug: print(f"  Skipping unmerge application for unhandled layer type: {layer_type}. Attempting propagation.")
                succ = self.find_closest_parameterized_successor(graph, node)
                if succ and succ != node: # Avoid infinite loops
                    if do_debug: print(f"  Propagating unmerge to next successor: Node {succ}")
                    self.unmerge_node(succ, merger) # Recursive call
                else:
                    if do_debug: print(f"  No successor found for propagation.")
                return # Stop processing current node after attempting propagation


            # <<< DEBUG Step 5: Post-Application Checks >>>
            if do_debug and weight_before is not None:
                print(f"[DEBUG Step 5 @ SuccNode {node}] Post-UNMERGE Checks:")
                weight_after = module.weight.data.clone().detach()
                norm_after = torch.linalg.norm(weight_after).item() if weight_after.numel() > 0 else 0.0
                changed = not torch.equal(weight_before, weight_after)
                print(f"  Weight changed? {changed} (Norm: {norm_before:.4f} -> {norm_after:.4f})")
                if torch.isnan(weight_after).any(): print("  ERROR: Weight contains NaN after unmerge!")
                if not changed and compatible_weight and norm_before > 1e-6 : print("  WARNING: Weight unmerge applied but tensor didn't change significantly.")
                # pdb.set_trace() # Optional breakpoint after unmerge application
            # <<< End DEBUG Step 5 >>>

        except KeyError:
            if do_debug: print(f"Warning: Layer {info.get('layer', 'N/A')} for node {node} not found in named_modules during unmerge.")
        except Exception as e:
             print(f"ERROR during unmerge_node for node {node}, layer {info.get('layer', 'N/A')}: {e}")
             import traceback; traceback.print_exc()
             raise



    def harden_merge_matrix(soft_merge: torch.Tensor) -> torch.Tensor:
        """
        Converts a soft merge matrix (each column sums to 1) into a hard permutation matrix.
        For each column, the entry with the highest value is set to 1 and all others to 0.
        
        Args:
            soft_merge (torch.Tensor): Soft merge matrix of shape (d, d)
        
        Returns:
            torch.Tensor: Hard permutation matrix of shape (d, d)
        """
        # Get the dimension of the merge matrix.
        d = soft_merge.size(0)
        # For each column, find the index of the maximum value.
        _, max_indices = torch.max(soft_merge, dim=0)
        # Create a zero matrix of the same shape.
        hard_merge = torch.zeros_like(soft_merge)
        # For each column, set the maximum index to 1.
        hard_merge[max_indices, torch.arange(d)] = 1.0
        return hard_merge

    # adding custom transformations here, for more control
    def apply_transformations_custom(self, merge_cls=False):
        qk_flag = False
        qk_nodes = [self.graphs[0].modules[name] for name in ['q', 'k']]


        # Sanity check: ensure merge/unmerge dicts are populated
        if not hasattr(self, 'merges') or not self.merges:
            print("Warning: No merge transformations found. Skipping application.")
            return None


        
        final_merger = None
        graph_device = "cuda"
        processed_nodes = set() # Keep track of nodes already merged/unmerged to avoid double application

        last_merger_for_graph = [None] * len(self.graphs)
        sorted_merge_nodes = sorted(self.merges.keys()) # Process in graph order
        
        for node_idx, node in enumerate(sorted_merge_nodes):
            if node not in self.merges or node not in self.unmerges:
                print(f"Warning: Missing merge/unmerge matrix for node {node}. Skipping.")
                continue

            merges_for_node = self.merges[node]
            unmerges_for_node = self.unmerges[node]

            # Apply transformations for each graph
            for graph_idx, graph in enumerate(self.graphs):
                merge_matrix = merges_for_node[graph_idx].to(graph_device)
                unmerge_matrix = unmerges_for_node[graph_idx].to(graph_device)

                # Create a temporary handler object (optional, but keeps pattern)
                merger = MergeHandler(graph, merge_matrix, unmerge_matrix, node)

                # --- Apply Merge (Upstream) ---
                # Find the closest parameterized predecessor module whose output space needs merging.
                predecessor_node = self.find_closest_parameterized_predecessor(graph, node)

                if predecessor_node is not None:
                    # Only apply if not already processed by a *later* merge operation
                    # (This check might need refinement depending on graph complexity)
                    if predecessor_node not in processed_nodes:
                         print(f"Graph {graph_idx}, Node {node}: Applying MERGE to predecessor {predecessor_node} ({graph.get_node_info(predecessor_node)['layer']})")
                         self.merge_node(predecessor_node, merger)
                         # Mark as processed for merging (output space aligned)
                         graph.merged.add(predecessor_node)
                         processed_nodes.add(predecessor_node) # Avoid re-applying merge later if it's a predecessor for multiple PREFIX nodes
                    else:
                         print(f"Graph {graph_idx}, Node {node}: Predecessor {predecessor_node} already processed by a later merge. Skipping merge.")

                else:
                    print(f"Warning: Graph {graph_idx}, Node {node}: No parameterized predecessor found to apply MERGE.")

                # --- Apply Unmerge (Downstream) ---
                # Find the closest parameterized successor module whose input space needs unmerging.
                successor_node = self.find_closest_parameterized_successor(graph, node)


                if successor_node is not None:
                     print(f"Graph {graph_idx}, Node {node}: Applying UNMERGE to successor {successor_node} ({graph.get_node_info(successor_node)['layer']})")
                     self.unmerge_node(successor_node, merger)
                     # Mark as processed for unmerging (input space aligned)
                     graph.unmerged.add(successor_node)
                     # Note: Don't add to processed_nodes here, as a successor might need multiple unmerges
                     # if it follows multiple PREFIX nodes (less common). The unmerge operation should be cumulative.
                else:
                    # If no direct successor, maybe apply to a final classification/projection head?
                    # This part is tricky and depends heavily on the graph structure *after* the last transformer layer.
                    # The original code had specific checks for 'cls.predictions', 'pooler', 'classification_heads'.
                    # You might need similar logic if your graph includes these and they aren't found by the successor search.
                    print(f"Warning: Graph {graph_idx}, Node {node}: No parameterized successor found to apply UNMERGE directly.")
                    # Consider applying to graph output/head if this is the last merge node?
                    if node_idx == len(sorted_merge_nodes) - 1: # If it's the last merge node in the sorted list
                        # Attempt to find and unmerge the final projection/head layer
                        output_head_node = self._find_final_head_node(graph) # You'd need to implement this helper
                        if output_head_node:
                             print(f"Graph {graph_idx}, Node {node}: Applying final UNMERGE to head node {output_head_node}")
                             self.unmerge_node(output_head_node, merger)
                             graph.unmerged.add(output_head_node)
                        else:
                             print(f"Graph {graph_idx}, Node {node}: Could not find final head node for UNMERGE.")


                # Keep track of the last merger applied for potential final unmerge outside the loop
                last_merger_for_graph[graph_idx] = merger


        # --- Optional: Handle final embedding/head unmerging ---
        # The original code unmerged embeddings based on the 'final_layer_norm' merge.
        # This graph-aware approach doesn't have that specific anchor.
        # If you need to unmerge embeddings or a final output layer *after* all other transforms,
        # you might use the 'last_merger_for_graph' saved above.
        # Example (needs careful adaptation to your graph):
        # for graph_idx, graph in enumerate(self.graphs):
        #     final_merger = last_merger_for_graph[graph_idx]
        #     if final_merger:
        #         # Find embedding node(s)
        #         embedding_nodes = [n for n, info in graph.G.nodes(data=True) if info['type'] == NodeType.EMBEDDING]
        #         for emb_node in embedding_nodes:
        #              print(f"Graph {graph_idx}: Applying final UNMERGE to embedding node {emb_node}")
        #              self.unmerge_node(emb_node, final_merger) # unmerge_node needs to handle embeddings correctly
        #              graph.unmerged.add(emb_node)
        #
        #         # Find final output/head node if not handled above
        #         # ... apply unmerge ...


        # The original function returned one 'final_merger', perhaps for unmerging something outside the graph loop.
        # Returning None for now, as the graph-aware approach should handle most cases within the loop.
        # If you need the last transformation for external use, return last_merger_for_graph.
        return last_merger_for_graph # Or return last_merger_for_graph if needed
    
    def _find_final_head_node(self, graph):
        """Helper to find a potential final classification/projection head node."""
        # This needs to be specific to your model's output structure.
        # Look for nodes connected to the OUTPUT node or specific named layers.
        output_nodes = [n for n, info in graph.G.nodes(data=True) if info['type'] == NodeType.OUTPUT]
        if not output_nodes: return None

        queue = list(graph.preds(output_nodes[0]))
        visited = set(queue)
        visited.add(output_nodes[0])

        while queue:
            current_node = queue.pop(0)
            info = graph.get_node_info(current_node)

            # Add conditions to identify your head layer(s)
            # Example: Check layer name if graph includes it
            if info['type'] == NodeType.MODULE:
                # Check for common head names or if it's the last parameterized layer before output
                if 'final_proj' in info.get('layer','') or 'classifier' in info.get('layer','') or 'lm_head' in info.get('layer', ''):
                    module = graph.get_module(info['layer'])
                    if isinstance(module, nn.Linear): # Check if it's a likely head layer type
                        return current_node

            for pred in graph.preds(current_node):
                if pred not in visited:
                    # Limit backward search depth if needed
                    visited.add(pred)
                    queue.append(pred)
        return None



        
    def get_merged_state_dict(self, interp_w=None, save_both=False, use_ties=False, quantile=0.8, maintain_hubert_behavior=True):
        """
        Post transformations, obtain state dictionary for merged model by linearly interpolating between 
        transformed models in each graph. By default all parameters are averaged, but if given an interp_w 
        weight, will be weightedly averaged instead.
        - interp_w (Optional): If None, all parameters of each model is averaged for merge. Otherwise, 
        interp_w is a list of len(num_models_to_merge), with weights bearing the importance of incorporating 
        features from each model into the merged result.
        use_ties (bool): If True, applies TIES-Merging.
        quantile (float): Quantile for trimming in TIES-Merging.
        Returns: state dict of merged model.
        """
        if save_both:
            merged_state_dict1 = self.graphs[0].model.state_dict().copy()
            merged_state_dict2 = self.graphs[1].model.state_dict().copy()
            return [merged_state_dict1, merged_state_dict2]
        else:
            if use_ties:
                hubert_dict = self.graphs[0].model.state_dict()
                mert_dict = self.graphs[1].model.state_dict()
                # Use TIESMerging class
                ties_merger = TIESMerging(hubert_dict, mert_dict, quantile=quantile, interp_w=interp_w, maintain_hubert_behavior=maintain_hubert_behavior)
                state_dict = ties_merger.merge()
            else:
                state_dict = {}
                merged_state_dict = self.merged_model.state_dict()
                keys = list(self.graphs[0].model.state_dict().keys())
                try:
                    for key in keys:
                        if key in merged_state_dict:
                            param = self.graphs[0].model.state_dict()[key]
                            if interp_w is not None and param.shape == merged_state_dict[key].shape:
                                print(f"merging params with interp_w: {interp_w}")
                                new_value = sum(graph.model.state_dict()[key] * w for graph, w in zip(self.graphs, interp_w))
                            else:
                                new_value = sum(graph.model.state_dict()[key] for graph in self.graphs) / len(self.graphs)
                            state_dict[key] = new_value
                except RuntimeError as e:
                    # Only catch runtime errors about tensor sizes, we need to be able to add models with diff heads together
                    if 'size' not in str(e):
                        raise e
            return state_dict
        


    def clear_hooks(self):
        """ Clears all hooks from graphs. """
        for g in self.graphs:
            g.clear_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []  

    def transform_single_model(self, model,
                  dataloader,
                  sentence_level=None,
                  special_toks=False,
                  transform_fn=match_tensors_intra_pairwise,
                  metric_classes=(CovarianceMetric, MeanMetric),
                  save_both=False,
                  permute_heads=True,
                  ignore_heads=False,
                  no_absval=False,
                  merge_cls=False,
                  saved_features=None,
                  res_type='none',
                  merge_cnn=True,  # NEW FLAG but not being used here to be hoinest
                  interp_w=None,
                  merge_type='ff+attn',
                  **transform_kwargs
                  ):
        """ Note: this consumes the models given to the graphs. Do not modify the models you give this. """
        
        self.merged_model = model.to(self.device).eval() # same arch as graph models , this model is already a HUBERT model.
                
        
        
        if not isinstance(metric_classes, dict):
            metric_classes = { x.name: x for x in metric_classes }
        
        self.metric_classes = metric_classes
        self.transform_fn = transform_fn

        _, vars = self.compute_metrics(dataloader, 
                            metric_classes=metric_classes, 
                            sentence_level=sentence_level,
                            special_toks=special_toks)


        _, _, cost_dict = self.compute_transformations(transform_fn, reduce_ratio=1 - 1. / len(self.graphs),
                                    permute_heads=permute_heads,
                                    ignore_heads=ignore_heads,
                                    no_absval=no_absval, 
                                    saved_features=saved_features,
                                    res=res_type,
                                    **transform_kwargs
                                    )
        


        final_merger = self.apply_transformations_custom(merge_cls=merge_cls)

        if save_both:
            #self.merged_model1 = deepcopy(self.graphs[0].model).to(self.device)
            def load_hubert_base(model_name="hubert_base"):
                # Load HuBERT base model from s3prl
                #os.environ["TORCH_HOME"] = "/workspace/s3prl/s3prl/cache"
                #base_model = torch.hub.load("s3prl/s3prl", model_name).cuda()
                import s3prl.hub as hub
                model = getattr(hub, 'hubert_base')()
                device = 'cuda'  # or cpu
                model = model.to(device)
                model.model.encoder.layerdrop = 0  # Ensure no dropout in encoder layers
                return model

            self.merged_model1 = load_hubert_base(model_name="hubert_base").to(self.device)
            self.merged_model1.load_state_dict(self.graphs[0].model.state_dict())
            #self.merged_model2 = deepcopy(self.graphs[1].model).to(self.device)
            self.merged_model2 = load_hubert_base(model_name="hubert_base").to(self.device)
            self.merged_model2.load_state_dict(self.graphs[1].model.state_dict())

        if save_both:
            merged_dicts = self.get_merged_state_dict(save_both=True, interp_w=interp_w,use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"])
            self.merged_model1.load_state_dict(merged_dicts[0])
            self.merged_model2.load_state_dict(merged_dicts[1])
            self.merged_model.load_state_dict(self.get_merged_state_dict(interp_w=interp_w, save_both=False, use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"] ), strict=False)
        else:
            self.merged_model.load_state_dict(self.get_merged_state_dict(interp_w=interp_w, save_both=False, use_ties=transform_kwargs["use_ties"], quantile=transform_kwargs["quantile"], maintain_hubert_behavior=transform_kwargs["maintain_hubert_behavior"] ), strict=False)
        self.add_hooks()

        if final_merger == None:
            unmerge = None
        else:
            unmerge = final_merger.unmerge

        return unmerge, cost_dict 
    
    def _get_affected_layer_names(self, graph, node_id):
        """
        Finds the layer names of the closest parameterized predecessor and successor
        for a given PREFIX node ID. Returns a set of layer names.
        """
        affected_layers = set()
        # Find predecessor
        pred_node = self.find_closest_parameterized_predecessor(graph, node_id)
        if pred_node:
            pred_info = graph.get_node_info(pred_node)
            if pred_info and 'layer' in pred_info and pred_info['layer']:
                affected_layers.add(pred_info['layer'])

        # Find successor
        succ_node = self.find_closest_parameterized_successor(graph, node_id)
        if succ_node:
            succ_info = graph.get_node_info(succ_node)
            if succ_info and 'layer' in succ_info and succ_info['layer']:
                affected_layers.add(succ_info['layer'])

        # Optional: Consider layers *between* pred and succ if graph is detailed enough?
        # This basic version only gets immediate parameterized neighbors.

        return affected_layers
    
    def _get_affected_layer_names_for_conditional_interpolation(self, graph, zipit_node_id):
        """
        Finds the layer name of the closest parameterized SUCCESSOR
        for a given PREFIX node ID where ZipIt! was applied.
        This successor layer's parameters are modified by the 'unmerge' matrix
        and should receive the special interpolation weights.
        Returns a set containing zero or one layer name.
        """
        affected_layers = set()

        # Find successor (the layer whose INPUT weights are modified by unmerge)
        succ_node = self.find_closest_parameterized_successor(graph, zipit_node_id)
        if succ_node:
            succ_info = graph.get_node_info(succ_node)
            if succ_info and 'layer' in succ_info and succ_info['layer']:
                layer_name = succ_info['layer']
                affected_layers.add(layer_name)
                print(f"    Node {zipit_node_id}: Identified SUCCESSOR for conditional interp: {layer_name} (Node {succ_node})")
            # else:
                 # Optional: Log if successor found but no layer name
                 # print(f"    Node {zipit_node_id}: Successor node {succ_node} found, but no layer name in info.")
        # else:
             # Optional: Log if no successor found
             # print(f"    Node {zipit_node_id}: No parameterized successor found.")

        return affected_layers

    def interpolate_state_dicts(self, state_dict_1, state_dict_2, interp_w,
                                use_ties=False, quantile=0.8, maintain_hubert_behavior=True):
        """
        Interpolates two state dictionaries based on provided weights.
        Handles both simple weighted averaging and TIES-Merging.

        Args:
            state_dict_1: The state dictionary of the first transformed model.
            state_dict_2: The state dictionary of the second transformed model.
            interp_w (list/tuple): Weights for interpolation (e.g., [0.5, 0.5]).
            use_ties (bool): If True, applies TIES-Merging.
            quantile (float): Quantile for trimming in TIES-Merging.
            maintain_hubert_behavior (bool): TIES sign resolution strategy.

        Returns:
            dict: The interpolated state dictionary.
        """
        if len(interp_w) != 2:
             raise ValueError("interp_w must contain exactly two weights.")
        # Ensure weights sum close to 1, normalize if not (optional, depends on strictness)
        # if not np.isclose(sum(interp_w), 1.0):
        #     print(f"Warning: Interpolation weights {interp_w} do not sum to 1. Normalizing.")
        #     total = sum(interp_w)
        #     interp_w = [w / total for w in interp_w]
        zipit_affected_layers = set()
        if hasattr(self, 'zipit_applied_nodes') and self.zipit_applied_nodes:
            print("  Identifying layers affected by ZipIt! nodes...")
            # We need a reference graph to find layers (use graph 0)
            ref_graph = self.graphs[0]
            for node_id in self.zipit_applied_nodes:
                layers = self._get_affected_layer_names_for_conditional_interpolation(ref_graph, node_id)
                zipit_affected_layers.update(layers)
            print(f"  Layers identified as affected by ZipIt!: {zipit_affected_layers}")


        if use_ties:
            print(f"  Applying TIES-Merging with quantile={quantile}, weights={interp_w}")
            # Ensure TIESMerging class is imported
            from merging_utils.ties_merger_adaptation_to_corr import TIESMerging
            # Pass state dicts directly
            ties_merger = TIESMerging(state_dict_1, state_dict_2,
                                      quantile=quantile, interp_w=interp_w,
                                      maintain_hubert_behavior=maintain_hubert_behavior)
            final_state_dict = ties_merger.merge()
        else:
            print(f"  Applying simple weighted averaging with conditional weights...")
            final_state_dict = {}
            keys = state_dict_1.keys()
            default_w1, default_w2 = interp_w
            zipit_w1, zipit_w2 = 0.5, 0.5

            num_zipit_weighted = 0
            num_default_weighted = 0
            num_skipped = 0

            for key in keys:
                current_w1, current_w2 = default_w1, default_w2
                is_affected_by_zipit = any(key.startswith(prefix) for prefix in zipit_affected_layers)

                if is_affected_by_zipit:
                    current_w1, current_w2 = zipit_w1, zipit_w2
                    num_zipit_weighted += 1
                else:
                    num_default_weighted += 1

                # Perform Averaging
                if key in state_dict_2:
                    param1 = state_dict_1[key]
                    param2 = state_dict_2[key]
                    if param1.shape == param2.shape:
                        final_state_dict[key] = param1 * current_w1 + param2 * current_w2
                    else:
                         # print(f"Warning: Shape mismatch for key '{key}'. Using Model 1's value.") # Reduce verbosity
                         final_state_dict[key] = param1
                         num_skipped += 1
                else:
                    # print(f"Warning: Key '{key}' not found in second state dict. Using Model 1's value.") # Reduce verbosity
                    final_state_dict[key] = state_dict_1[key]
                    num_skipped += 1

            print(f"  Finished simple averaging.")
            print(f"    Params weighted: ZipIt(50/50)={num_zipit_weighted}, Default({default_w1:.1f}/{default_w2:.1f})={num_default_weighted}, Skipped={num_skipped}")
            return final_state_dict 

              
    def transform(self, model,
                  dataloader,
                  sentence_level=None,
                  special_toks=False,
                  transform_fn=match_tensors_permute,
                  metric_classes=(CovarianceMetric, MeanMetric),
                  save_both=False,
                  permute_heads=False,
                  ignore_heads=False,
                  no_absval=False,
                  merge_cls=False,
                  saved_features=None,
                  res_type='none',
                  merge_cnn=True,  # NEW FLAG but not being used here to be hoinest
                  interp_w=None,
                  layer_weights=None, 
                  alpha=1.0,
                  enable_weighted_alignment=False,
                  run_feature_similarity_analysis=False, # NEW FLAG
                  num_batches_for_similarity=1,       # NEW FLAG
                  nodes_for_similarity_count=3,       # NEW FLAG
                  **transform_kwargs
                  ):
        """ Note: this consumes the models given to the graphs. Do not modify the models you give this. """
        print(f"Starting transform process with function: {transform_fn.__name__}")
        self.zipit_analysis = None # <<< Initialize analysis storage attribute
        self.merged_model = model.to(self.device).eval() # same arch as graph models , this model is already a HUBERT model.
                
        if not isinstance(metric_classes, dict):
            metric_classes = { x.name: x for x in metric_classes }
        
        self.metric_classes = metric_classes
        self.transform_fn = transform_fn

        # if we did not pre-save features, compute them here:
        if saved_features == None:
            _, vars = self.compute_metrics(dataloader, 
                                metric_classes=metric_classes, 
                                sentence_level=sentence_level,
                                special_toks=special_toks)
            

        _, _, cost_dict, zipit_analysis_results = self.compute_transformations(transform_fn, reduce_ratio=1 - 1. / len(self.graphs),
                                    permute_heads=permute_heads,
                                    ignore_heads=ignore_heads,
                                    no_absval=no_absval, 
                                    saved_features=saved_features,
                                    res=res_type,
                                    layer_weights=layer_weights, alpha=alpha,
                                    enable_weighted_alignment=enable_weighted_alignment,
                                    **transform_kwargs
                                    )
  

        self.zipit_analysis = zipit_analysis_results # <<< Store analysis results
        print(f"self.zipit_analysis = {self.zipit_analysis}")

        self.feature_similarity_results = None # Initialize
        
        if run_feature_similarity_analysis:
            if not self.unmerges:
                print("WARNING: Cannot run feature similarity analysis because self.unmerges is not populated.")
            else:
                all_prefix_nodes_with_unmerges = sorted([
                    node_id for node_id in self.unmerges.keys()
                    if isinstance(self.unmerges[node_id], tuple) and len(self.unmerges[node_id]) == 2 and node_id != 'res'
                ])
                
                # Select nodes based on nodes_for_similarity_count
                nodes_to_analyze_sim = all_prefix_nodes_with_unmerges[:nodes_for_similarity_count]
                
                node_scaling_factors = { 4: 64, 7: 32, 11: 16, # ... etc ... 
                                           **{node_id: 1 for node_id in range(28, 250)} }

                if nodes_to_analyze_sim:
                    # Ensure dataloader can be re-iterated or get a new one
                    # For simplicity, assume `dataloader` can be re-iterated. If it's a one-shot
                    # iterator from `get_dataloader`, you'd need to call `get_dataloader` again here.
                    # E.g., sim_dataloader = get_dataloader(transform_kwargs.get('data_config_for_sim_pass'), split="train")
                    sim_dataloader = dataloader # ASSUMING RE-ITERABLE or you handle re-init externally

                    self.feature_similarity_results = self.analyze_feature_similarity(
                        dataloader=sim_dataloader,
                        nodes_to_analyze=nodes_to_analyze_sim,
                        cost_dict=cost_dict,
                        node_scaling_factors=node_scaling_factors,
                        num_batches_for_similarity=num_batches_for_similarity
                    )
                    print("\nFeature Similarity Analysis Results (Experiment 2 - from ModelMerge.transform):")
                    for layer_name, res_sim in self.feature_similarity_results.items():
                        print(f"  Layer: {layer_name}")
                        print(f"    Avg CosSim Before: {res_sim.get('avg_cos_sim_before_perm', float('nan')):.4f}")
                        print(f"    Avg CosSim After : {res_sim.get('avg_cos_sim_after_perm', float('nan')):.4f}")
                        print(f"    Algorithm Cost   : {res_sim.get('cost_from_algo', float('nan')):.4f}")
                    
                    return None, cost_dict
                    
                    # --- OPTION TO STOP SCRIPT HERE FOR ANALYSIS ---
                    # You can add a global flag or another arg to `transform` like `exit_after_similarity_analysis`
                    # if transform_kwargs.get("exit_after_similarity_analysis", False):
                    #    print("Exiting after feature similarity analysis as requested.")
                    #    # Return something to indicate to main() to exit, or just sys.exit() if appropriate for your workflow
                    #    return None, cost_dict # Or a special marker

                else:
                    print("No valid nodes found to analyze for feature similarity.")


        _ = self.apply_transformations_custom(merge_cls=merge_cls)
        print("Transformations applied.")


        return None, cost_dict
    
    def add_hooks(self):
        """ Add hooks at zip start or stop at locations for merged model and base models. """
        # Remove the hooks from the models to add or own
        self.clear_hooks()
        

