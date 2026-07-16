import os
import yaml
import glob
import torch
import random
import argparse
import logging
import torchaudio
import numpy as np
from argparse import Namespace
from torch.distributed import is_initialized, get_world_size
from transformers import AutoModel, AutoConfig
from metric_calculators import get_metric_fns
from merging_utils.dataset import get_dataloader
from enum import Enum

from copy import deepcopy
import json
try:
    import ipdb as pdb
except ImportError:
    import pdb
import sys

from s3prl import hub
from graphs.hubert_graph_complete_merge import HuBERTGraph
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout
from s3prl.utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override
import umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import importlib
from huggingface_hub import HfApi, HfFolder
from inspect import getmembers, isfunction

from merging_utils.model_merger_new import ModelMerge
from merging_utils.merging_within_models_and_between import ModelMerge as ModelMergeComplete

from graphs.base_graph import NodeType

import debugpy



import torch.nn as nn

def collect_raw_features_for_nodes(
    graphs, 
    dataloader, 
    node_ids_to_collect, 
    num_batches_to_collect,
    device,
    node_scaling_factors,
    final_downsample_rate=320 # Make this explicit
):
    logging.info(f"Collecting raw features for nodes {node_ids_to_collect} over {num_batches_to_collect} batches...")
    
    for g in graphs:
        g.model.eval() # Ensure models are in eval mode
        g.clear_hooks() 
        g.add_hooks(device=device) 

    collected_features_A = {node_id: [] for node_id in node_ids_to_collect}
    collected_features_B = {node_id: [] for node_id in node_ids_to_collect}

    batches_processed = 0
    with torch.no_grad():
        for batch_idx, (wavs, *others) in enumerate(dataloader):
            if batches_processed >= num_batches_to_collect:
                logging.info(f"Collected features from {batches_processed} batches. Stopping.")
                break
            
            wavs_gpu = [torch.FloatTensor(wav).to(device) for wav in wavs]
            
            try:
                intermediates_batch = [g.compute_intermediates(wavs_gpu, device=device) for g in graphs]
            except Exception as e:
                logging.error(f"Error in compute_intermediates for batch {batch_idx}: {e}")
                continue # Skip this batch

            final_lengths_batch = [int(wav.shape[-1] // final_downsample_rate) for wav in wavs_gpu]

            padded_intermediates_A = intermediates_batch[0]
            padded_intermediates_B = intermediates_batch[1]
            
            unpadded_A_batch = {}
            unpadded_B_batch = {}

            for node_id in node_ids_to_collect:
                # Process Model A
                if node_id in padded_intermediates_A and padded_intermediates_A[node_id] is not None:
                    factor = node_scaling_factors.get(node_id, 1)
                    effective_lengths_node = [length * factor for length in final_lengths_batch]
                    
                    tensor_A = padded_intermediates_A[node_id]
                    list_A = []; start_idx_A = 0
                    current_total_len_A = 0
                    for eff_len in effective_lengths_node:
                        list_A.append(tensor_A[:, start_idx_A : start_idx_A + eff_len])
                        start_idx_A += eff_len
                        current_total_len_A += eff_len
                    
                    if tensor_A.shape[1] < current_total_len_A:
                        logging.warning(f"Batch {batch_idx}, Node {node_id} (Model A): Tensor length {tensor_A.shape[1]} < expected sum of effective lengths {current_total_len_A}. Truncating.")
                        # This indicates an issue with length calculation or unexpected tensor shapes from hooks.
                        # For robustness, we might skip or adjust, but for now, this will likely lead to mismatched data.
                        # It's better to ensure the lengths match. If they don't, it implies a deeper problem.
                        # Let's assume lengths will match from correct hook outputs + padding removal.
                        # If not, we should ensure we don't go out of bounds.
                        # The current slicing `start_idx_A + eff_len` might go out of bounds if `tensor_A` is shorter than expected.
                        # The original `remove_pads_dynamic` implies the input tensor is already a concatenation of max-length sequences.
                        # Let's re-evaluate the padding removal here for `collect_raw_features`.
                        # The core issue might be that `g.compute_intermediates` returns features that are ALREADY effectively unpadded
                        # or concatenated differently than `ModelMerge.remove_pads_dynamic` expects.
                        # For now, let's assume the logic is okay and `tensor_A.shape[1]` is the total length of concatenated max-padded sequences for the batch.
                        # `remove_pads_dynamic` in ModelMerge would then un-concatenate and re-concatenate *only* the valid parts.

                    if list_A: 
                        unpadded_A_batch[node_id] = torch.cat(list_A, dim=1)
                    else: # Should not happen if node_id in padded_intermediates_A
                        unpadded_A_batch[node_id] = torch.empty((0,0), device='cpu')


                # Process Model B
                if node_id in padded_intermediates_B and padded_intermediates_B[node_id] is not None:
                    factor = node_scaling_factors.get(node_id, 1) # Recalculate just in case, though same for A and B
                    effective_lengths_node = [length * factor for length in final_lengths_batch]

                    tensor_B = padded_intermediates_B[node_id] # CORRECTED LINE
                    list_B = []; start_idx_B = 0
                    for eff_len in effective_lengths_node:
                        list_B.append(tensor_B[:, start_idx_B : start_idx_B + eff_len])
                        start_idx_B += eff_len
                    if list_B: 
                        unpadded_B_batch[node_id] = torch.cat(list_B, dim=1)
                    else: # Should not happen
                        unpadded_B_batch[node_id] = torch.empty((0,0), device='cpu')


            for node_id_collect in node_ids_to_collect: # Use a different loop variable
                if node_id_collect in unpadded_A_batch and unpadded_A_batch[node_id_collect].numel() > 0 :
                    collected_features_A[node_id_collect].append(unpadded_A_batch[node_id_collect].cpu())
                if node_id_collect in unpadded_B_batch and unpadded_B_batch[node_id_collect].numel() > 0:
                    collected_features_B[node_id_collect].append(unpadded_B_batch[node_id_collect].cpu())
            
            batches_processed += 1

    final_features_A = {nid: (torch.cat(flist, dim=1) if flist else torch.empty((0,0))) for nid, flist in collected_features_A.items()}
    final_features_B = {nid: (torch.cat(flist, dim=1) if flist else torch.empty((0,0))) for nid, flist in collected_features_B.items()}
    
    for g_clean in graphs: 
        g_clean.clear_hooks()
        
    logging.info(f"Finished collecting raw features. Example shapes for collected nodes:")
    for nid_log in node_ids_to_collect:
        shape_a_log = final_features_A.get(nid_log, torch.empty(0)).shape
        shape_b_log = final_features_B.get(nid_log, torch.empty(0)).shape
        logging.info(f"  Node {nid_log}: A: {shape_a_log}, B: {shape_b_log}")
    return final_features_A, final_features_B

# --- Plotting Function (Enhanced) ---
def plot_correlation_matrices_for_node(
    corr_matrix_before, 
    corr_matrix_after,  
    node_id,
    display_node_name,
    save_dir,
    num_heads=0, 
    max_dim_to_plot=769, 
    vmin=0.0,
    vmax=0.7 
):
    logging.info(f"Plotting for Node {node_id} ('{display_node_name}'), max_dim={max_dim_to_plot}, heads={num_heads}, vmin={vmin}, vmax={vmax}")
    abs_corr_before = np.abs(corr_matrix_before)
    abs_corr_after = np.abs(corr_matrix_after)

    dim_a_orig, dim_b_orig = abs_corr_before.shape
    dim_a_after, dim_b_after = abs_corr_after.shape # Should be dim_a_orig x dim_a_orig if properly aligned

    if dim_a_orig == 0 or dim_b_orig == 0 or dim_a_after == 0 or dim_b_after == 0:
        logging.warning(f"Node {node_id}: Empty correlation matrix. Before: {abs_corr_before.shape}, After: {abs_corr_after.shape}. Skipping plot.")
        return

    plot_dim_a = min(dim_a_orig, max_dim_to_plot)
    plot_dim_b_orig_plot = min(dim_b_orig, max_dim_to_plot)
    plot_dim_b_after_plot = min(dim_b_after, max_dim_to_plot) 

    abs_corr_before_plot = abs_corr_before[:plot_dim_a, :plot_dim_b_orig_plot]
    abs_corr_after_plot = abs_corr_after[:plot_dim_a, :plot_dim_b_after_plot]

    diff_norm_plotted = np.linalg.norm(abs_corr_before_plot - abs_corr_after_plot)
    logging.info(f"Node {node_id} ('{display_node_name}'): Norm of difference between *plotted* abs_corr matrices: {diff_norm_plotted:.4f}")
    if diff_norm_plotted < 1e-3 and plot_dim_a > 0: # If very small difference, log more info
        logging.warning(f"Node {node_id}: Plotted matrices are very similar (diff_norm={diff_norm_plotted:.4E}). Max abs diff: {np.max(np.abs(abs_corr_before_plot - abs_corr_after_plot)):.4E}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Cross-Correlation: {display_node_name} (dim={plot_dim_a})", fontsize=13, fontweight='bold')

    cmap_to_use = 'viridis'

    # Auto-scale vmax per panel if the data range is much smaller than the fixed vmax
    data_max = max(abs_corr_before_plot.max(), abs_corr_after_plot.max())
    effective_vmax = min(vmax, max(data_max * 1.1, 0.15))  # At least 0.15 to avoid pure-noise scaling
    effective_vmin = vmin

    im1 = axes[0].imshow(abs_corr_before_plot, cmap=cmap_to_use, vmin=effective_vmin, vmax=effective_vmax, aspect='auto', interpolation='nearest')
    axes[0].set_title("Before Permutation", fontsize=11)
    axes[0].set_xlabel("MERT Channel", fontsize=10)
    axes[0].set_ylabel("HuBERT Channel", fontsize=10)
    axes[0].tick_params(axis='both', which='major', labelsize=8)

    im2 = axes[1].imshow(abs_corr_after_plot, cmap=cmap_to_use, vmin=effective_vmin, vmax=effective_vmax, aspect='auto', interpolation='nearest')
    axes[1].set_title("After Permutation", fontsize=11)
    axes[1].set_xlabel("MERT Channel (Permuted)", fontsize=10)
    axes[1].tick_params(axis='both', which='major', labelsize=8)

    # Hardcode 12 attention heads for pre-trained HuBERT/MERT (768-dim)
    effective_num_heads = num_heads if num_heads > 0 else (12 if plot_dim_a in [768, 769] else 0)
    if effective_num_heads > 0 and plot_dim_a > 0 and plot_dim_a % effective_num_heads == 0:
        head_dim_size = plot_dim_a // effective_num_heads
        if head_dim_size > 2:
            grid_ticks = np.arange(head_dim_size, plot_dim_a, head_dim_size)
            for ax_plt in axes:
                ax_plt.set_xticks(grid_ticks - 0.5, minor=True)
                ax_plt.set_yticks(grid_ticks - 0.5, minor=True)
                ax_plt.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.6)
    else:
         logging.debug(f"Node {node_id}: Not adding head grid lines (num_heads={effective_num_heads}, plot_dim_a={plot_dim_a}).")

    fig.subplots_adjust(right=0.88, bottom=0.15, top=0.88, wspace=0.25) 
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7]) 
    cbar = fig.colorbar(im1, cax=cbar_ax) 
    cbar.set_label("Absolute Correlation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    safe_display_name = re.sub(r'[^\w\s-]', '', display_node_name).strip().replace(' ', '_')
    plot_filename = os.path.join(save_dir, f"correlation_heatmap_node_{node_id}_{safe_display_name}.png")
    try:
        plt.savefig(plot_filename, dpi=300) 
        logging.info(f"Saved correlation heatmap to {plot_filename}")
    except Exception as e:
        logging.error(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

# --- Main experiment function for correlation heatmap visualization (Corrected and enhanced) ---
def run_correlation_heatmap_experiment(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"Starting correlation heatmap experiment with args: {args}")

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('soundfile')
    hack_isinstance()

    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    # --- Load models for ModelMerge instance ---
    logging.info("Loading base models for ModelMerge...")
    model1_for_merge = load_hubert_base(model_name="hubert_base")
    model2_for_merge_mapped = load_hubert_base(model_name="hubert_base") 

    temp_config_mert = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True, local_files_only=False)
    temp_config_mert.output_hidden_states = True
    if not hasattr(temp_config_mert, 'conv_pos_batch_norm'):
        temp_config_mert.conv_pos_batch_norm = False  # Fix for newer transformers versions
    model2_mert_orig_for_merge = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config_mert, trust_remote_code=True, local_files_only=False).to(args.device)
    disable_MERT_encoder_dropout(model2_mert_orig_for_merge)

    new_state_dict_mert_for_merge = map_mert_to_hubert(model2_mert_orig_for_merge.state_dict())
    wrapped_weights_mert_for_merge = wrap_mert_weights(new_state_dict_mert_for_merge)
    model2_for_merge_mapped.load_state_dict(wrapped_weights_mert_for_merge, strict=False)
    del model2_mert_orig_for_merge # Clean up

    with open("merging_utils/data_config.yaml", "r") as file:
        data_config = yaml.safe_load(file)
    
    # Dataloader for metrics and feature collection
    # Adjust num_batches_for_metric based on how many batches you want for metrics vs. viz features
    metrics_dataloader = get_dataloader(data_config, split="train")
    feature_collection_dataloader = get_dataloader(data_config, split="train") # Can be same or different

    # --- Initialize ModelMerge with its own set of models ---
    logging.info("Initializing ModelMerge instance...")
    graph1_for_merge = HuBERTGraph(model1_for_merge, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
    graph2_for_merge = HuBERTGraph(model2_for_merge_mapped, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
    Merge = ModelMerge(graph1_for_merge, graph2_for_merge, device=args.device)
    
    merging_metric_fns = get_metric_fns(["covariance", "mean"]) 

    logging.info("Computing metrics for ModelMerge...")
    Merge.compute_metrics(
        metrics_dataloader,
        metric_classes=merging_metric_fns,
        sentence_level=None, 
        special_toks=False
    )
    if Merge.metrics is None:
        logging.error("Metrics computation failed. Aborting heatmap visualization.")
        return

    logging.info("Computing transformations (permutation matrices) using ModelMerge...")
    merging_function = get_merging_fn(args.merging_algorithm)
    Merge.compute_transformations( 
        transform_fn=merging_function,
        reduce_ratio=0.5, 
        permute_heads=(args.merge_type == 'qkv+attn' or args.merge_type == 'ff+attn' or args.merge_type == 'all'),
        ignore_heads=False,
        no_absval=True,
        saved_features=None, 
        res='sep', 
        layer_weights=args.layer_weights, 
        alpha=args.alpha, 
        enable_weighted_alignment=args.enable_weighted_alignment,
        a=args.zipit_a, 
        b=args.zipit_b,
        interp_w=args.interp_weights,
        use_ties=args.use_ties, 
        quantile=args.quantile,
        maintain_hubert_behavior=args.maintain_hubert_behavior,
        merge_type=args.merge_type
    )

    if not hasattr(Merge, 'unmerges') or not Merge.unmerges:
        logging.error("Permutation matrices (unmerges) not computed by ModelMerge. Aborting heatmap visualization.")
        return

    # --- Node Selection (same as before) ---
    valid_prefix_nodes = sorted([
        node_id for node_id in Merge.unmerges.keys()
        if isinstance(node_id, int) and 
           isinstance(Merge.unmerges[node_id], (list, tuple)) and len(Merge.unmerges[node_id]) == 2 
    ])
    if not valid_prefix_nodes:
        logging.warning("No valid PREFIX nodes found with unmerge matrices for visualization.")
        return
    nodes_to_visualize_ids = []
    if args.correlation_viz_nodes:
        nodes_to_visualize_ids = [nid for nid in args.correlation_viz_nodes if nid in Merge.unmerges and nid in valid_prefix_nodes]
        if not nodes_to_visualize_ids:
             logging.warning(f"None of the specified --correlation_viz_nodes ({args.correlation_viz_nodes}) are valid. Selecting defaults.")
             nodes_to_visualize_ids = valid_prefix_nodes[:min(3, len(valid_prefix_nodes))]
        elif len(nodes_to_visualize_ids) != len(args.correlation_viz_nodes):
            logging.warning("Some specified --correlation_viz_nodes were invalid or missing data.")
    else:
        nodes_to_visualize_ids = valid_prefix_nodes[:min(3, len(valid_prefix_nodes))] 
    if not nodes_to_visualize_ids:
        logging.warning("No nodes selected for correlation heatmap visualization.")
        return
    logging.info(f"Selected nodes for heatmap visualization: {nodes_to_visualize_ids}")

    expdir_viz = os.path.join(args.expdir, "correlation_heatmaps_recomputed_no_deepcopy") 
    os.makedirs(expdir_viz, exist_ok=True)
    
    node_scaling_factors_viz = { 4: 64, 7: 32, 11: 16, 15: 8, 19: 4, 23: 2, 27: 1,
                              **{node_id_scale: 1 for node_id_scale in range(28, 250)} }

    # --- Load FRESH, ORIGINAL models for raw feature collection ---
    logging.info("Loading fresh, original models for raw feature collection...")
    model1_base_for_features = load_hubert_base(model_name="hubert_base")
    model2_mapped_for_features = load_hubert_base(model_name="hubert_base")
    
    # Re-apply MERT mapping to the second fresh model
    temp_config_mert_feat = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True, local_files_only=False)
    temp_config_mert_feat.output_hidden_states = True
    if not hasattr(temp_config_mert_feat, 'conv_pos_batch_norm'):
        temp_config_mert_feat.conv_pos_batch_norm = False  # Fix for newer transformers versions
    model2_mert_orig_feat = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config_mert_feat, trust_remote_code=True, local_files_only=False).to(args.device)
    disable_MERT_encoder_dropout(model2_mert_orig_feat)
    new_state_dict_mert_feat = map_mert_to_hubert(model2_mert_orig_feat.state_dict())
    wrapped_weights_mert_feat = wrap_mert_weights(new_state_dict_mert_feat)
    model2_mapped_for_features.load_state_dict(wrapped_weights_mert_feat, strict=False)
    del model2_mert_orig_feat # Clean up

    graph_A_for_features = HuBERTGraph(model1_base_for_features, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
    graph_B_for_features = HuBERTGraph(model2_mapped_for_features, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()

    raw_features_A, raw_features_B = collect_raw_features_for_nodes(
        graphs=[graph_A_for_features, graph_B_for_features], 
        dataloader=feature_collection_dataloader, # Use the appropriate dataloader
        node_ids_to_collect=nodes_to_visualize_ids,
        num_batches_to_collect=args.sim_analysis_batches, 
        device=args.device,
        node_scaling_factors=node_scaling_factors_viz
    )
    
    # --- Node display name mapping (using Merge's graphs for info) ---
    node_display_name_map_viz = {}
    for node_id_map in nodes_to_visualize_ids:
        # Get node info from the graphs used by ModelMerge, as these define the PREFIX nodes
        node_info_prefix = Merge.graphs[0].get_node_info(node_id_map) 
        display_layer_name_map = f"PREFIX_Node_{node_id_map}" 
        if node_info_prefix and node_info_prefix.get('layer'): # Check if node_info_prefix is not None
            display_layer_name_map = f"{node_info_prefix['layer']}_(PFX_TARGET)"
        elif node_info_prefix: # Check if node_info_prefix is not None before accessing successors
            q_map = list(Merge.graphs[0].succs(node_id_map))
            visited_map = set(q_map); visited_map.add(node_id_map)
            found_module_map = False
            while q_map:
                curr_map = q_map.pop(0)
                succ_info_map = Merge.graphs[0].get_node_info(curr_map)
                if succ_info_map and succ_info_map['type'] == NodeType.MODULE and succ_info_map.get('layer'):
                    display_layer_name_map = f"InputTo_{succ_info_map['layer']}"
                    found_module_map = True; break
                if succ_info_map: # Check if succ_info_map is not None before accessing successors
                    for s_next_map in Merge.graphs[0].succs(curr_map):
                        if s_next_map not in visited_map: visited_map.add(s_next_map); q_map.append(s_next_map)
        node_display_name_map_viz[node_id_map] = simplify_display_name(display_layer_name_map, node_id_map)

    num_attention_heads = 0
    # Try to get from model1_for_merge (used by Merge)
    if hasattr(model1_for_merge.model, 'config') and hasattr(model1_for_merge.model.config, 'encoder_attention_heads'):
        num_attention_heads = model1_for_merge.model.config.encoder_attention_heads # Corrected path
        logging.info(f"Using num_attention_heads = {num_attention_heads} for plotting grid lines.")
    else:
        logging.warning("Could not determine num_attention_heads from model config. Grid lines for heads might be skipped.")

    # --- Plotting loop (same as before) ---
    for node_id_plot in nodes_to_visualize_ids:
        logging.info(f"Generating recomputed plot for node {node_id_plot}...")
        
        if node_id_plot not in Merge.unmerges or Merge.unmerges[node_id_plot] is None or len(Merge.unmerges[node_id_plot]) < 2 :
            logging.warning(f"Node {node_id_plot}: Unmerge matrix missing or invalid for Model B. Skipping.")
            continue
        P_B_tensor = Merge.unmerges[node_id_plot][1] 
        if P_B_tensor is None:
            logging.warning(f"Node {node_id_plot}: P_B is None. Skipping.")
            continue
        
        A_raw_node = raw_features_A.get(node_id_plot) 
        B_raw_node = raw_features_B.get(node_id_plot)

        


        if A_raw_node is None or B_raw_node is None or A_raw_node.numel() == 0 or B_raw_node.numel() == 0:
            logging.warning(f"Node {node_id_plot}: Raw features missing or empty. A_shape: {A_raw_node.shape if A_raw_node is not None else 'None'}, B_shape: {B_raw_node.shape if B_raw_node is not None else 'None'}. Skipping.")
            continue
        
        if A_raw_node.shape[0] != B_raw_node.shape[0]: 
            logging.warning(f"Node {node_id_plot}: Feature dimension mismatch in raw features. A_dim: {A_raw_node.shape[0]}, B_dim: {B_raw_node.shape[0]}. Skipping.")
            continue
        if A_raw_node.shape[1] == 0 or B_raw_node.shape[1] == 0: 
             logging.warning(f"Node {node_id_plot}: Zero tokens collected for A ({A_raw_node.shape[1]}) or B ({B_raw_node.shape[1]}). Skipping.")
             continue
        
        D_model_features = A_raw_node.shape[0]
        if P_B_tensor.shape[0] != D_model_features or P_B_tensor.shape[1] != D_model_features:
            logging.warning(f"Node {node_id_plot}: P_B shape {P_B_tensor.shape} mismatch with raw feature dim {D_model_features}. Skipping.")
            continue

        A_raw_np = A_raw_node.float().numpy()
        B_raw_np = B_raw_node.float().numpy()
        P_B_np = P_B_tensor.cpu().float().numpy() 

        identity_check_np = np.eye(D_model_features, dtype=P_B_np.dtype)
        is_identity_np = np.allclose(P_B_np, identity_check_np, atol=1e-5) 
        logging.info(f"Node {node_id_plot}: Is P_B_np an identity matrix? {is_identity_np}")
        if not is_identity_np:
            # Calculate how many diagonal elements are NOT close to 1 (for a near-permutation matrix P, P @ P.T approx I)
            # A more direct way for a permutation matrix P: D - trace(P) if it's truly a permutation of Identity.
            # Or, how many rows are not identity rows.
            permuted_channels_count_np = 0
            for r_idx in range(D_model_features):
                if not np.allclose(P_B_np[r_idx, :], identity_check_np[r_idx, :], atol=1e-5):
                    permuted_channels_count_np += 1
            logging.info(f"Node {node_id_plot}: Approx. permuted channels by P_B_np: {int(permuted_channels_count_np)}/{D_model_features}")

        combined_before_np = np.vstack((A_raw_np, B_raw_np))
        # Ensure sufficient data for correlation
        if combined_before_np.shape[1] < 2:
            logging.warning(f"Node {node_id_plot}: Insufficient observations ({combined_before_np.shape[1]}) for correlation. Skipping.")
            continue
        corr_matrix_full_before = np.corrcoef(combined_before_np)
        if np.isnan(corr_matrix_full_before).any():
            logging.warning(f"Node {node_id_plot}: NaN found in 'before' correlation matrix. Filling NaNs with 0.")
            corr_matrix_full_before = np.nan_to_num(corr_matrix_full_before)
        C_AB_before_recomputed = corr_matrix_full_before[:D_model_features, D_model_features:]

        B_prime_raw_np = P_B_np @ B_raw_np 

        combined_after_np = np.vstack((A_raw_np, B_prime_raw_np))
        if combined_after_np.shape[1] < 2: # Check again for safety, though should be same as before
            logging.warning(f"Node {node_id_plot}: Insufficient observations ({combined_after_np.shape[1]}) for 'after' correlation. Skipping.")
            continue
        corr_matrix_full_after = np.corrcoef(combined_after_np)
        if np.isnan(corr_matrix_full_after).any():
            logging.warning(f"Node {node_id_plot}: NaN found in 'after' correlation matrix. Filling NaNs with 0.")
            corr_matrix_full_after = np.nan_to_num(corr_matrix_full_after)
        C_AB_after_recomputed = corr_matrix_full_after[:D_model_features, D_model_features:]
        
        current_display_name = node_display_name_map_viz.get(node_id_plot, f"Node_{node_id_plot}")
        
        logging.info(f"--- NEIGHBOR DEBUG for PREFIX Node {node_id_plot} ('{current_display_name}') ---")
    
        # Use the graphs from ModelMerge to find neighbors, as they have the module info
        graph_ref = Merge.graphs[0] # Use graph A as reference for structure

        # Find preceding module that was ACTUALLY modified by the 'merge' part of the transform
        # The 'merge' part of apply_transformations_custom calls self.merge_node on a predecessor.
        # ModelMerge.find_closest_parameterized_predecessor(graph, start_node_for_prefix_logic)
        preceding_module_node_id = Merge.find_closest_parameterized_predecessor(graph_ref, node_id_plot)
        preceding_module_layer_name = "N/A"
        if preceding_module_node_id is not None:
            pred_info = graph_ref.get_node_info(preceding_module_node_id)
            if pred_info and pred_info.get('layer'):
                preceding_module_layer_name = pred_info['layer']
        logging.info(f"  Preceding parameterized module (output affected by P_B.T): Node {preceding_module_node_id}, Layer: {preceding_module_layer_name}")

        # Find succeeding module that was ACTUALLY modified by the 'unmerge' part of the transform
        succeeding_module_node_id = Merge.find_closest_parameterized_successor(graph_ref, node_id_plot)
        succeeding_module_layer_name = "N/A"
        if succeeding_module_node_id is not None:
            succ_info = graph_ref.get_node_info(succeeding_module_node_id)
            if succ_info and succ_info.get('layer'):
                succeeding_module_layer_name = succ_info['layer']
        logging.info(f"  Succeeding parameterized module (input affected by P_B): Node {succeeding_module_node_id}, Layer: {succeeding_module_layer_name}")


        if Merge.metrics and node_id_plot in Merge.metrics and 'covariance' in Merge.metrics[node_id_plot]:
            cov_from_merge_metrics_A = Merge.metrics[node_id_plot]['covariance'][:D_model_features, :D_model_features]
            cov_from_merge_metrics_B = Merge.metrics[node_id_plot]['covariance'][D_model_features:, D_model_features:]
            cross_cov_from_merge_metrics = Merge.metrics[node_id_plot]['covariance'][:D_model_features, D_model_features:]
            
            # Convert cross-covariance to cross-correlation
            std_A_metric = torch.sqrt(torch.diag(cov_from_merge_metrics_A) + 1e-9)
            std_B_metric = torch.sqrt(torch.diag(cov_from_merge_metrics_B) + 1e-9)
            corr_AB_from_merge_metrics = cross_cov_from_merge_metrics / (torch.outer(std_A_metric, std_B_metric) + 1e-9)
            corr_AB_from_merge_metrics_np = corr_AB_from_merge_metrics.cpu().numpy()

            diff_corr_metric_vs_raw = C_AB_before_recomputed - corr_AB_from_merge_metrics_np
            norm_diff_corr_metric_vs_raw = np.linalg.norm(diff_corr_metric_vs_raw)
            logging.info(f"  Norm of difference (Corr_AB_from_raw_features - Corr_AB_from_Merge.metrics): {norm_diff_corr_metric_vs_raw:.4f}")
            if norm_diff_corr_metric_vs_raw > 0.1: # Arbitrary threshold for "significant difference"
                logging.warning(f"  Node {node_id_plot}: Correlation matrix from raw features differs notably from the one used by Merge algorithm. This could be a source of discrepancy if P_B is based on different stats.")
                # This could happen if dataloaders/num_batches for metric calculation vs. raw feature collection are different.
        else:
            logging.info(f"  Node {node_id_plot}: Could not retrieve covariance from Merge.metrics to compare correlation source.")
        
        plot_correlation_matrices_for_node(
            C_AB_before_recomputed,
            C_AB_after_recomputed,
            node_id_plot,
            current_display_name,
            expdir_viz,
            num_heads=num_attention_heads,
            max_dim_to_plot=args.correlation_viz_max_dim,
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax
        )

        # --- DUMP raw correlation matrices for downstream per-channel plots ---
        npz_out_path = os.path.join(
            expdir_viz,
            f"correlation_matrices_node_{node_id_plot}_{simplify_display_name(current_display_name, node_id_plot)}.npz"
        )
        # Extract matched column indices from P_B (P_B @ B reorders B's rows, so col ordering is P_B.argmax(axis=1))
        # For a permutation matrix, col_ind[i] is the column index where row i is 1.
        try:
            col_ind_from_P = np.argmax(P_B_np, axis=1)
        except Exception:
            col_ind_from_P = None
        np.savez_compressed(
            npz_out_path,
            C_AB_before=C_AB_before_recomputed.astype(np.float32),
            C_AB_after=C_AB_after_recomputed.astype(np.float32),
            col_ind=col_ind_from_P if col_ind_from_P is not None else np.array([]),
            node_id=node_id_plot,
            display_name=current_display_name,
        )
        logging.info(f"Saved correlation matrices npz: {npz_out_path}")

    logging.info(f"Recomputed correlation heatmap experiment finished. Plots saved in {expdir_viz}")
    if hasattr(Merge, 'clear_hooks'): # Ensure Merge has this method
      Merge.clear_hooks()
    else: # Fallback for graphs used for feature collection
        graph_A_for_features.clear_hooks()
        graph_B_for_features.clear_hooks()






# def collect_raw_features_for_nodes(
#     graphs, # List of [graph_A, graph_B]
#     dataloader, # Dataloader to iterate over
#     node_ids_to_collect, # List of integer node IDs
#     num_batches_to_collect,
#     device,
#     node_scaling_factors # From ModelMerge.remove_pads_dynamic
# ):
#     logging.info(f"Collecting raw features for nodes {node_ids_to_collect} over {num_batches_to_collect} batches...")
    
#     # Ensure hooks are active for feature collection on the original models in the graphs
#     for g in graphs:
#         g.clear_hooks() # Clear any previous
#         g.add_hooks(device=device) # Add fresh hooks

#     collected_features_A = {node_id: [] for node_id in node_ids_to_collect}
#     collected_features_B = {node_id: [] for node_id in node_ids_to_collect}

#     batches_processed = 0
#     # Create a dummy ModelMerge instance just for remove_pads_dynamic if needed,
#     # or ensure graphs[0].model contains what remove_pads_dynamic expects for final_downsample
#     # For simplicity here, assuming node_scaling_factors are passed correctly.
#     # A better way might be to have a simpler remove_pads accessible.
#     # The main `Merge` object already has remove_pads_dynamic.

#     with torch.no_grad():
#         for batch_idx, (wavs, *others) in enumerate(dataloader):
#             if batches_processed >= num_batches_to_collect:
#                 break
            
#             wavs_gpu = [torch.FloatTensor(wav).to(device) for wav in wavs]
            
#             # Get intermediates from original models in the graphs
#             # graph.compute_intermediates uses the model stored within the graph
#             intermediates_batch = [g.compute_intermediates(wavs_gpu, device=device) for g in graphs]
            
#             # Need a way to call remove_pads_dynamic effectively.
#             # Let's assume a simplified version or that `graphs` has a `remove_pads_dynamic` like method
#             # For now, this is a placeholder for robust padding removal
#             # This is tricky as remove_pads_dynamic is a method of ModelMerge.
#             # We might need to pass the Merge object or replicate the padding logic.
#             # Let's try to replicate its core idea if Merge object isn't easily available here.
            
#             # Simplified padding removal for feature collection context:
#             # This assumes final_downsample = 320 (hardcoded, could be passed)
#             final_downsample_rate = 320 # Typically for HuBERT CNN output
#             final_lengths_batch = [int(wav.shape[-1] // final_downsample_rate) for wav in wavs_gpu]

#             padded_intermediates_A = intermediates_batch[0]
#             padded_intermediates_B = intermediates_batch[1]
            
#             unpadded_A_batch = {}
#             unpadded_B_batch = {}

#             for node_id in node_ids_to_collect:
#                 if node_id in padded_intermediates_A and node_id in padded_intermediates_B:
#                     factor = node_scaling_factors.get(node_id, 1)
#                     effective_lengths_node = [length * factor for length in final_lengths_batch]
                    
#                     tensor_A = padded_intermediates_A[node_id]
#                     list_A = []; start_idx = 0
#                     for eff_len in effective_lengths_node:
#                         list_A.append(tensor_A[:, start_idx : start_idx + eff_len])
#                         start_idx += eff_len
#                     if list_A: unpadded_A_batch[node_id] = torch.cat(list_A, dim=1)

#                     tensor_B = padded_intermediates_B[node_id]
#                     list_B = []; start_idx = 0
#                     for eff_len in effective_lengths_node:
#                         list_B.append(tensor_B[:, start_idx : start_idx + eff_len])
#                         start_idx += eff_len
#                     if list_B: unpadded_B_batch[node_id] = torch.cat(list_B, dim=1)

#             for node_id in node_ids_to_collect:
#                 if node_id in unpadded_A_batch:
#                     collected_features_A[node_id].append(unpadded_A_batch[node_id].cpu())
#                 if node_id in unpadded_B_batch:
#                     collected_features_B[node_id].append(unpadded_B_batch[node_id].cpu())
            
#             batches_processed += 1

#     final_features_A = {nid: (torch.cat(flist, dim=1) if flist else torch.empty(0)) for nid, flist in collected_features_A.items()}
#     final_features_B = {nid: (torch.cat(flist, dim=1) if flist else torch.empty(0)) for nid, flist in collected_features_B.items()}
    
#     for g in graphs: # Clean up hooks from original models
#         g.clear_hooks()
        
#     logging.info(f"Finished collecting raw features. Shapes (example for node {node_ids_to_collect[0] if node_ids_to_collect else 'N/A'}): "
#                  f"A: {final_features_A.get(node_ids_to_collect[0] if node_ids_to_collect else None, torch.empty(0)).shape}, "
#                  f"B: {final_features_B.get(node_ids_to_collect[0] if node_ids_to_collect else None, torch.empty(0)).shape}")
#     return final_features_A, final_features_B



# def run_correlation_heatmap_experiment(args):
#     logging.basicConfig(level=logging.INFO)
#     logging.info(f"Starting correlation heatmap experiment with args: {args}")

#     torch.multiprocessing.set_sharing_strategy('file_system')
#     torchaudio.set_audio_backend('soundfile')
#     hack_isinstance()

#     if args.cache_dir is not None:
#         torch.hub.set_dir(args.cache_dir)

#     # --- Model Loading and Preparation (similar to main()) ---
#     model1_base = load_hubert_base(model_name="hubert_base")
#     model2_mapped = load_hubert_base(model_name="hubert_base") 

#     temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
#     temp_config.output_hidden_states = True
#     model2_mert_orig = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(args.device)
#     disable_MERT_encoder_dropout(model2_mert_orig)

#     new_state_dict_mert = map_mert_to_hubert(model2_mert_orig.state_dict())
#     wrapped_weights_mert = wrap_mert_weights(new_state_dict_mert)
#     model2_mapped.load_state_dict(wrapped_weights_mert, strict=False)
#     del model2_mert_orig

#     # --- Dataloader ---
#     with open("merging_utils/data_config.yaml", "r") as file:
#         data_config = yaml.safe_load(file) # Use safe_load
    
#     # Potentially use fewer samples for this visualization to speed it up
#     # data_config_viz = deepcopy(data_config)
#     # data_config_viz["train_batch_size"] = 4 # Example: smaller batch for viz
#     # data_config_viz["num_workers"] = 2
#     # For Librispeech subset, maybe limit number of files in get_dataloader if possible or use a smaller dataset config.
#     # For now, using the same dataloader as in main(), ensure it's not too large.
#     # You might want to use args.sim_analysis_batches for the number of batches.
#     dataloader = get_dataloader(data_config, split="train")


#     # --- Graph and Merger Initialization ---
#     graph1 = HuBERTGraph(model1_base, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
#     graph2 = HuBERTGraph(model2_mapped, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()

#     Merge = ModelMerge(graph1, graph2, device=args.device)
#     merging_metric_fns = get_metric_fns(["covariance", "mean"]) # Ensure 'mean' is available if ZipIt needs it

#     # --- Compute Metrics (Populates Merge.metrics) ---
#     logging.info("Computing metrics for correlation visualization...")
#     Merge.compute_metrics(
#         dataloader,
#         metric_classes=merging_metric_fns,
#         sentence_level=None, # Assuming frame/token level for PREFIX nodes
#         special_toks=False
#     )
#     if Merge.metrics is None:
#         logging.error("Metrics computation failed. Aborting heatmap visualization.")
#         return

#     # --- Compute Transformations (Populates Merge.merges, Merge.unmerges) ---
#     # This also computes 'corrs' internally based on Merge.metrics
#     logging.info("Computing transformations to get permutation matrices...")
#     merging_function = get_merging_fn(args.merging_algorithm)
#     _, _, _, _ = Merge.compute_transformations( # We don't need the returned dicts here directly
#         transform_fn=merging_function,
#         reduce_ratio=0.5, # Or 1 - 1./len(Merge.graphs)
#         permute_heads=(args.merge_type == 'qkv+attn' or args.merge_type == 'ff+attn' or args.merge_type == 'all'),
#         ignore_heads=False,
#         no_absval=True,
#         saved_features=None, # We are computing metrics live
#         res='sep', # 'sep' to ensure all residual nodes also have corrs if needed
#         layer_weights=args.layer_weights, 
#         alpha=args.alpha, 
#         enable_weighted_alignment=args.enable_weighted_alignment,
#         # Pass other kwargs from args if your transform_fn needs them
#         a=args.zipit_a, 
#         b=args.zipit_b,
#         interp_w=args.interp_weights, #Though not directly used by permute, good practice if fn expects
#         use_ties=args.use_ties, 
#         quantile=args.quantile,
#         maintain_hubert_behavior=args.maintain_hubert_behavior,
#         merge_type=args.merge_type
#     )

#     if not hasattr(Merge, 'unmerges') or not Merge.unmerges:
#         logging.error("Permutation matrices (unmerges) not computed. Aborting heatmap visualization.")
#         return

#     # --- Re-compute Correlation Matrices (as done inside compute_transformations) ---
#     # `compute_metric_corrs` uses `Merge.metrics`.
#     # We need all PREFIX nodes for which an unmerge (permutation) matrix was computed.
    
#     valid_prefix_nodes = sorted([
#         node_id for node_id in Merge.unmerges.keys()
#         if isinstance(node_id, int) and # Ensure it's a node ID, not 'res'
#            isinstance(Merge.unmerges[node_id], (list, tuple)) and len(Merge.unmerges[node_id]) == 2 # For 2 models
#     ])

#     if not valid_prefix_nodes:
#         logging.warning("No valid PREFIX nodes found with unmerge matrices for visualization.")
#         return
        
#     corrs_for_viz = Merge.compute_metric_corrs(nodes=valid_prefix_nodes, res='sep', no_corr=False) # Get cov converted to corr

#     # --- Node Selection for Visualization ---
#     nodes_to_visualize_ids = []
#     if args.correlation_viz_nodes:
#         nodes_to_visualize_ids = [nid for nid in args.correlation_viz_nodes if nid in Merge.unmerges and nid in corrs_for_viz]
#         if len(nodes_to_visualize_ids) != len(args.correlation_viz_nodes):
#             logging.warning("Some specified --correlation_viz_nodes were invalid or missing data.")
#     else:
#         # Default: visualize a few, e.g., the first few non-residual PREFIX nodes
#         # Or, you could try to find nodes corresponding to specific layers like "7th attention layer"
#         # For now, let's take up to 3 valid prefix nodes
#         nodes_to_visualize_ids = valid_prefix_nodes[:min(3, len(valid_prefix_nodes))]

#     if not nodes_to_visualize_ids:
#         logging.warning("No nodes selected for correlation heatmap visualization.")
#         return

#     logging.info(f"Selected nodes for heatmap visualization: {nodes_to_visualize_ids}")

#     # --- Generate and Save Plots ---
#     os.makedirs(args.expdir, exist_ok=True)
#     graph_A_orig = HuBERTGraph(model1_base, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
#     graph_B_orig = HuBERTGraph(model2_mapped, merge_type=args.merge_type, merge_cnn=args.merge_cnn).graphify()
    
#     # Define node_scaling_factors as used in ModelMerge.compute_metrics
#     # This might need to be retrieved or defined consistently.
#     # From ModelMerge.compute_metrics:
#     node_scaling_factors_viz = { 4: 64, 7: 32, 11: 16, 15: 8, 19: 4, 23: 2, 27: 1,
#                               **{node_id_scale: 1 for node_id_scale in range(28, 250)} }


#     # Dataloader for collecting features for visualization (can be smaller)
#     viz_feature_dataloader = get_dataloader(data_config, split="train")

#     raw_features_A, raw_features_B = collect_raw_features_for_nodes(
#         graphs=[graph_A_orig, graph_B_orig], # Use graphs with original models
#         dataloader=viz_feature_dataloader,
#         node_ids_to_collect=nodes_to_visualize_ids,
#         num_batches_to_collect=args.sim_analysis_batches, # Use existing arg
#         device=args.device,
#         node_scaling_factors=node_scaling_factors_viz
#     )
#     # --- Generate and Save Plots ---
#     expdir_viz = os.path.join(args.expdir, "correlation_heatmaps_recomputed") # New folder
#     os.makedirs(expdir_viz, exist_ok=True)
#     node_display_name_map_viz = { # Create display names (same logic as before)
#         # ... (copy your existing display name generation logic here) ...
#         node_id: simplify_display_name(f"Node_{node_id}", node_id) # Placeholder
#         for node_id in nodes_to_visualize_ids
#     }

#     for node_id in nodes_to_visualize_ids:
#         logging.info(f"Generating recomputed plot for node {node_id}...")
        
#         P_B_tensor = Merge.unmerges[node_id][1] # This is P_B for Model B, should be on CUDA
#         if P_B_tensor is None:
#             logging.warning(f"Node {node_id}: P_B is None. Skipping.")
#             continue
        
#         A_raw_node = raw_features_A.get(node_id) # Already on CPU from collect_raw_features
#         B_raw_node = raw_features_B.get(node_id)

#         if A_raw_node is None or B_raw_node is None or A_raw_node.numel() == 0 or B_raw_node.numel() == 0:
#             logging.warning(f"Node {node_id}: Raw features missing or empty. A: {A_raw_node.shape if A_raw_node is not None else 'None'}, B: {B_raw_node.shape if B_raw_node is not None else 'None'}. Skipping.")
#             continue
        
#         if A_raw_node.shape[0] != B_raw_node.shape[0]:
#             logging.warning(f"Node {node_id}: Feature dimension mismatch in raw features. A: {A_raw_node.shape[0]}, B: {B_raw_node.shape[0]}. Skipping.")
#             continue
        
#         D_model_features = A_raw_node.shape[0]
#         if P_B_tensor.shape[0] != D_model_features or P_B_tensor.shape[1] != D_model_features:
#             logging.warning(f"Node {node_id}: P_B shape {P_B_tensor.shape} mismatch with raw feature dim {D_model_features}. Skipping.")
#             continue

#         # Ensure features are float for corrcoef and P_B is on the same device for matmul
#         A_raw_np = A_raw_node.float().numpy()
#         B_raw_np = B_raw_node.float().numpy()
#         P_B_np = P_B_tensor.cpu().float().numpy() # P_B to CPU and numpy for matmul with numpy B_raw

#         # Debug print P_B (already added this based on previous step's output)
#         identity_check_np = np.eye(D_model_features)
#         is_identity_np = np.allclose(P_B_np, identity_check_np)
#         logging.info(f"Node {node_id}: (Recheck) Is P_B_np an identity matrix? {is_identity_np}")
#         permuted_channels_count_np = np.sum(~np.all(P_B_np == identity_check_np, axis=1))
#         logging.info(f"Node {node_id}: (Recheck) Permuted channels by P_B_np: {permuted_channels_count_np}/{D_model_features}")


#         # C_AB_before: Corr(A_raw, B_raw)
#         # np.corrcoef expects rows to be variables, columns observations. Our features are D x N.
#         # Concatenate for np.corrcoef: make a (2D) x N matrix
#         combined_before_np = np.vstack((A_raw_np, B_raw_np))
#         corr_matrix_full_before = np.corrcoef(combined_before_np)
#         # Check for NaNs which can happen if variance is zero for some features
#         if np.isnan(corr_matrix_full_before).any():
#             logging.warning(f"Node {node_id}: NaN found in 'before' correlation matrix. Filling NaNs with 0.")
#             corr_matrix_full_before = np.nan_to_num(corr_matrix_full_before)
#         C_AB_before_recomputed = corr_matrix_full_before[:D_model_features, D_model_features:]

#         # B_prime_raw = P_B @ B_raw
#         B_prime_raw_np = P_B_np @ B_raw_np # DxD @ DxN -> DxN

#         # C_AB_after: Corr(A_raw, B_prime_raw)
#         combined_after_np = np.vstack((A_raw_np, B_prime_raw_np))
#         corr_matrix_full_after = np.corrcoef(combined_after_np)
#         if np.isnan(corr_matrix_full_after).any():
#             logging.warning(f"Node {node_id}: NaN found in 'after' correlation matrix. Filling NaNs with 0.")
#             corr_matrix_full_after = np.nan_to_num(corr_matrix_full_after)
#         C_AB_after_recomputed = corr_matrix_full_after[:D_model_features, D_model_features:]
        
#         current_display_name = node_display_name_map_viz.get(node_id, f"Node_{node_id}")
        
#         # Logging norm of difference for the recomputed matrices
#         diff_norm_recomputed = np.linalg.norm(np.abs(C_AB_before_recomputed) - np.abs(C_AB_after_recomputed))
#         logging.info(f"Node {node_id}: Norm of difference between RECOMPUTED abs_corr plots: {diff_norm_recomputed}")

#         plot_correlation_matrices_for_node(
#             C_AB_before_recomputed,
#             C_AB_after_recomputed,
#             node_id,
#             current_display_name,
#             expdir_viz, # Save to new subfolder
#             max_dim_to_plot=args.correlation_viz_max_dim
#         )

#     logging.info("Recomputed correlation heatmap experiment finished.")
#     Merge.clear_hooks() # Clear hooks from the Merge instance's graphs


    # # Create simplified display name map
    # node_display_names = {}
    # for node_id_viz in nodes_to_visualize_ids:
    #     # Use your existing simplify_display_name or a similar logic
    #     # For simplicity, let's try to get layer name from graph info
    #     node_info_prefix = Merge.graphs[0].get_node_info(node_id_viz)
    #     display_name = f"PREFIX_Node_{node_id_viz}" # Fallback
    #     if node_info_prefix.get('layer'):
    #         display_name = f"{node_info_prefix['layer']}_(PFX_TARGET)"
    #     else: # Try successor
    #         q_s = list(Merge.graphs[0].succs(node_id_viz))
    #         visited_s = set(q_s)
    #         found_module_s = False
    #         while q_s:
    #             curr_s = q_s.pop(0)
    #             s_info = Merge.graphs[0].get_node_info(curr_s)
    #             if s_info['type'] == NodeType.MODULE and s_info.get('layer'):
    #                 display_name = f"InputTo_{s_info['layer']}"
    #                 found_module_s = True; break
    #             for s_next_s in Merge.graphs[0].succs(curr_s):
    #                 if s_next_s not in visited_s: visited_s.add(s_next_s); q_s.append(s_next_s)
    #     node_display_names[node_id_viz] = simplify_display_name(display_name, node_id_viz)


    # for node_id in nodes_to_visualize_ids:
    #     if node_id not in corrs_for_viz or corrs_for_viz[node_id] is None:
    #         logging.warning(f"Node {node_id}: Correlation matrix not found or is None. Skipping plot.")
    #         continue
    #     if node_id not in Merge.unmerges or Merge.unmerges[node_id] is None or len(Merge.unmerges[node_id]) < 2:
    #         logging.warning(f"Node {node_id}: Permutation matrix for Model B not found. Skipping plot.")
    #         continue

    #     full_corr_matrix_node = corrs_for_viz[node_id].cpu().numpy()
    #     # P_A is unmerges[node_id][0], P_B is unmerges[node_id][1]
    #     # We assume P_A is identity. P_B is the permutation for MERT (Model B)
    #     P_B = Merge.unmerges[node_id][1].cpu().numpy() 

    #     dim_total = full_corr_matrix_node.shape[0]
    #     if dim_total % 2 != 0:
    #         logging.warning(f"Node {node_id}: Total dimension of correlation matrix ({dim_total}) is not even. Cannot split for A/B. Skipping.")
    #         continue
        
    #     D_model = dim_total // 2 # Dimension of each model's features at this node

    #     if P_B.shape[0] != D_model or P_B.shape[1] != D_model:
    #         logging.warning(f"Node {node_id}: Permutation matrix P_B shape {P_B.shape} does not match model feature dimension {D_model}. Skipping.")
    #         continue

    #     # Extract C_AB (HuBERT vs MERT channels correlation)
    #     # C_AB = corr(Features_A, Features_B_original)
    #     C_AB_before = full_corr_matrix_node[:D_model, D_model:]

    #     # Calculate C_AB_after = corr(Features_A, P_B @ Features_B_original)
    #     # This is equivalent to C_AB_before @ P_B_transpose
    #     C_AB_after = C_AB_before @ P_B.T
        
    #     current_display_name = node_display_names.get(node_id, f"Node_{node_id}")
        
    #     logging.info(f"Node {node_id}: Inspecting P_B (shape {P_B.shape})")
    #     if D_model > 0 : # D_model_features is derived from full_corr_matrix_node.shape[0] // 2
    #         identity_check = np.eye(D_model)
    #         is_identity = np.allclose(P_B, identity_check)
    #         logging.info(f"Node {node_id}: Is P_B an identity matrix? {is_identity}")
    #         if not is_identity:
    #             num_diff_rows = np.sum(np.abs(P_B - identity_check).sum(axis=1) > 1e-5)
    #             logging.info(f"Node {node_id}: Number of rows in P_B differing from identity: {num_diff_rows} / {D_model}")
    #             # For a permutation matrix P, number of elements permuted from identity position
    #             # is D - trace(P), assuming P is binary and rows/cols sum to 1.
    #             # Or more simply, count how many rows are not identity rows.
    #             permuted_channels_count = 0
    #             for r in range(P_B.shape[0]):
    #                 if not np.allclose(P_B[r, :], identity_check[r, :]):
    #                     permuted_channels_count +=1
    #             logging.info(f"Node {node_id}: Number of channels permuted by P_B from identity: {permuted_channels_count} / {D_model}")

        # plot_correlation_matrices_for_node(
        #     C_AB_before,
        #     C_AB_after,
        #     node_id,
        #     current_display_name,
        #     args.expdir,
        #     max_dim_to_plot=args.correlation_viz_max_dim
        # )

    # logging.info("Correlation heatmap experiment finished.")
    # Merge.clear_hooks()

# def plot_correlation_matrices_for_node(
#     corr_matrix_before, # D_A x D_B
#     corr_matrix_after,  # D_A x D_B_permuted (effectively D_A x D_A after alignment)
#     node_id,
#     display_node_name,
#     save_dir,
#     num_heads=0, 
#     max_dim_to_plot=769, # Default updated
#     vmin=0.0,
#     vmax=0.7 # Default updated
# ):
#     """
#     Plots two correlation matrices (before and after permutation) side-by-side.
#     Uses 'hot' colormap and a shared colorbar.
#     """
#     abs_corr_before = np.abs(corr_matrix_before)
#     abs_corr_after = np.abs(corr_matrix_after)

#     dim_a, dim_b_orig = abs_corr_before.shape
#     _, dim_b_after = abs_corr_after.shape 

#     # Crop matrices if they are too large for clear visualization
#     plot_dim_a = min(dim_a, max_dim_to_plot)
#     plot_dim_b_orig = min(dim_b_orig, max_dim_to_plot)
#     # For the 'after' plot, if it's aligned, Model B's permuted dimension should match Model A's
#     plot_dim_b_after = min(dim_b_after, max_dim_to_plot) 

#     abs_corr_before_plot = abs_corr_before[:plot_dim_a, :plot_dim_b_orig]
#     abs_corr_after_plot = abs_corr_after[:plot_dim_a, :plot_dim_b_after]

#     # This norm is useful for debugging if the permutation had *any* visual effect on the plotted submatrix
#     diff_norm_plotted = np.linalg.norm(abs_corr_before_plot - abs_corr_after_plot)
#     logging.info(f"Node {node_id} ('{display_node_name}'): Norm of difference between *plotted* abs_corr matrices: {diff_norm_plotted:.4f}")

#     fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0)) # Adjusted for shared colorbar
#     fig.suptitle(f"Channel Correlation: {display_node_name}\n(Node {node_id}, Plotted up to {plot_dim_a}x{max(plot_dim_b_orig, plot_dim_b_after)} dims)", fontsize=10)

#     cmap_to_use = 'hot' # Changed to 'hot'
#     # cmap_to_use = 'YlOrRd' # Another good alternative from the paper

#     # Before Permutation
#     im1 = axes[0].imshow(abs_corr_before_plot, cmap=cmap_to_use, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
#     axes[0].set_title("Before Permutation", fontsize=9)
#     axes[0].set_xlabel("Model B (MERT) Channel Index", fontsize=8)
#     axes[0].set_ylabel("Model A (HuBERT) Channel Index", fontsize=8)
#     axes[0].tick_params(axis='both', which='major', labelsize=7)

#     # After Permutation
#     im2 = axes[1].imshow(abs_corr_after_plot, cmap=cmap_to_use, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
#     axes[1].set_title("After Permutation", fontsize=9)
#     axes[1].set_xlabel("Model B (MERT) Channel Index (Permuted)", fontsize=8)
#     # axes[1].set_ylabel("Model A (HuBERT) Channel Index", fontsize=8) # Y-axis is the same
#     axes[1].tick_params(axis='both', which='major', labelsize=7)

#     # Add gridlines for attention heads if applicable
#     # Grid lines are useful if plot_dim_a == plot_dim_b_after (square matrix after perm)
#     if num_heads > 0 and plot_dim_a == plot_dim_b_after and plot_dim_a > 0 and plot_dim_a % num_heads == 0 :
#         head_dim_size = plot_dim_a // num_heads
#         # Avoid drawing too many grid lines if head_dim_size is very small
#         if head_dim_size > 2: # Only draw if heads are at least a few pixels wide
#             grid_ticks = np.arange(head_dim_size, plot_dim_a, head_dim_size)
#             for ax_plt in axes: # Apply to both subplots
#                 ax_plt.set_xticks(grid_ticks - 0.5, minor=True)
#                 ax_plt.set_yticks(grid_ticks - 0.5, minor=True)
#                 ax_plt.grid(which='minor', color='cyan', linestyle=':', linewidth=0.7, alpha=0.5) # Changed color for visibility
                
#                 # Optional: Add major ticks at head boundaries for clarity if not too crowded
#                 if plot_dim_a <= 384 and head_dim_size >= 16 : # Heuristics for readability
#                     major_tick_locs = grid_ticks - head_dim_size / 2
#                     try: # Try setting major ticks, might fail if too dense
#                         ax_plt.set_xticks(major_tick_locs)
#                         ax_plt.set_xticklabels([f"H{i+1}" for i in range(num_heads)], rotation=45, ha="right", fontsize=6)
#                         ax_plt.set_yticks(major_tick_locs)
#                         ax_plt.set_yticklabels([f"H{i+1}" for i in range(num_heads)], fontsize=6)
#                     except Exception: # Fallback if setting labels fails
#                         ax_plt.tick_params(axis='x', labelrotation=0) # Keep default numeric ticks
#                         ax_plt.tick_params(axis='y')
#                 else: 
#                     ax_plt.tick_params(axis='x', labelrotation=0)
#     else:
#          logging.info(f"Node {node_id}: Not adding head grid lines (num_heads={num_heads}, plot_dim_a={plot_dim_a}, plot_dim_b_after={plot_dim_b_after}).")


#     # Shared colorbar
#     fig.subplots_adjust(right=0.88, bottom=0.1) # Make space for colorbar and x-labels
#     cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7]) # [left, bottom, width, height]
#     cbar = fig.colorbar(im1, cax=cbar_ax) # Use im1 for the colorbar mapping (could be im2 too)
#     cbar.set_label("Absolute Correlation", fontsize=8)
#     cbar.ax.tick_params(labelsize=7)

#     # plt.tight_layout(rect=[0, 0.03, 0.88, 0.95]) # tight_layout often fights with add_axes. Manual adjustment is sometimes better.
#     # Instead of tight_layout after add_axes, adjust subplot parameters before or use fig.subplots_adjust carefully.
    
#     safe_display_name = re.sub(r'[^\w\s-]', '', display_node_name).strip().replace(' ', '_')
#     plot_filename = os.path.join(save_dir, f"correlation_heatmap_node_{node_id}_{safe_display_name}.png")
#     plt.savefig(plot_filename, dpi=300) # bbox_inches='tight' can be added if needed
#     logging.info(f"Saved correlation heatmap to {plot_filename}")
#     plt.close(fig)


# def plot_correlation_matrices_for_node(
#     corr_matrix_before, # D_A x D_B
#     corr_matrix_after,  # D_A x D_B_permuted (effectively D_A x D_A after alignment)
#     node_id,
#     display_node_name,
#     save_dir,
#     max_dim_to_plot=256
# ):
#     """
#     Plots two correlation matrices (before and after permutation) side-by-side.
#     """
#     abs_corr_before = np.abs(corr_matrix_before)
#     abs_corr_after = np.abs(corr_matrix_after)

#     dim_a, dim_b_orig = abs_corr_before.shape
#     _, dim_b_after = abs_corr_after.shape # Should be same as dim_a if properly aligned

#     # Crop matrices if they are too large for clear visualization
#     plot_dim_a = min(dim_a, max_dim_to_plot)
#     plot_dim_b_orig = min(dim_b_orig, max_dim_to_plot)
#     plot_dim_b_after = min(dim_b_after, max_dim_to_plot) # Should ideally be plot_dim_a

#     abs_corr_before_plot = abs_corr_before[:plot_dim_a, :plot_dim_b_orig]
#     abs_corr_after_plot = abs_corr_after[:plot_dim_a, :plot_dim_b_after]

#     diff_norm = np.linalg.norm(abs_corr_before_plot - abs_corr_after_plot); logging.info(f"Node {node_id}: Norm of difference between abs_corr plots: {diff_norm}")

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     fig.suptitle(f"Channel Correlation: {display_node_name} (Node {node_id})\n(Showing up to {max_dim_to_plot}x{max_dim_to_plot} dims)", fontsize=10)

#     # Before Permutation
#     im1 = axes[0].imshow(abs_corr_before_plot, cmap='viridis', vmin=0, vmax=0.6, aspect='auto')
#     axes[0].set_title("Before Permutation", fontsize=9)
#     axes[0].set_xlabel("Model B (MERT) Channel Index", fontsize=8)
#     axes[0].set_ylabel("Model A (HuBERT) Channel Index", fontsize=8)

#     # After Permutation
#     im2 = axes[1].imshow(abs_corr_after_plot, cmap='viridis', vmin=0, vmax=0.6, aspect='auto')
#     axes[1].set_title("After Permutation", fontsize=9)
#     axes[1].set_xlabel("Model B (MERT) Channel Index (Permuted)", fontsize=8)
#     axes[1].set_ylabel("Model A (HuBERT) Channel Index", fontsize=8)

#     fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04, label="Absolute Correlation")
#     fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04, label="Absolute Correlation")

#     plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
#     plot_filename = os.path.join(save_dir, f"correlation_heatmap_node_{node_id}_{display_node_name.replace(' ', '_')}.png")
#     plt.savefig(plot_filename, dpi=300)
#     print(f"Saved correlation heatmap to {plot_filename}")
#     plt.close(fig)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class PermuteAndUnpermute(nn.Module):
    def __init__(self, target_module, layer_name, seed=42, unpermute_output=True):
        super().__init__()
        self.target_module = target_module
        self.unpermute_output = unpermute_output
        self.layer_name = layer_name  # Store the original name
        # Set seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        # Get the weight of the target module
        weight = target_module.weight.data.clone().cuda()
        # Create a random permutation along the output dimension (assumed dim 0)
        self.perm = torch.randperm(weight.size(0), device=weight.device)
        # Compute the inverse permutation
        self.inv_perm = torch.argsort(self.perm)
        # Permute the weights and bias of the target module
        target_module.weight.data.copy_(weight[self.perm, :])
        if target_module.bias is not None:
            target_module.bias.data.copy_(target_module.bias.data[self.perm])
        print(f"Applied permutation in {target_module}: {self.perm.tolist()}")

    def forward(self, x):
        # Get the output from the target module
        out = self.target_module(x)
        if self.unpermute_output:
            # If unpermute_output flag is True, restore the original order.
            return out.index_select(1, self.inv_perm)
        else:
            # Otherwise, return the permuted output.
            return out
    
    def state_dict(self, *args, **kwargs):
        # Return the underlying module's state dict.
        # This makes the wrapper transparent to merging code that expects the same key names.
        return self.target_module.state_dict(*args, **kwargs)

    def __getattr__(self, name):
        # If an attribute isn't found on this wrapper, try the target module.
        # This allows external code to access attributes like 'weight' on this object.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.target_module, name)
    
    @property
    def layer(self):
        # Expose the original layer name so that the graph can use it.
        return self.layer_name

def replace_module(model, target_module_name, new_module):
    names = target_module_name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    setattr(parent, names[-1], new_module)


def clear_hooks_and_prepare(model):
    for module in model.modules():
        module._backward_hooks = {}
        module._forward_hooks = {}
        module._forward_pre_hooks = {}

def assemble_new_checkpoint(averaged_state_dict, model_mode='teacher', source_config_dict=None):
    """
    Assemble a checkpoint from the merged state dict.

    Args:
        averaged_state_dict: The merged model state dict.
        model_mode: 'teacher' for HuBERT Base format, 'distilled' for multi_distiller format.
        source_config_dict: For distilled mode, the original Config dict from the source checkpoint.
    """
    if model_mode == 'distilled':
        # Distilled model checkpoint format (compatible with multi_distiller_local upstream)
        new_checkpoint = {
            'Distiller': averaged_state_dict,
            'Config': source_config_dict if source_config_dict else {},
        }
        return new_checkpoint
    else:
        # Teacher model checkpoint format (HuBERT Base)
        new_checkpoint = {}
        new_checkpoint['model_cfg'] = {'_name': 'hubert', 'label_rate': 50.0, 'extractor_mode': 'default', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'layer_type': 'transformer', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'untie_final_proj': False, 'layer_norm_first': False, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 0.1, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': False, 'checkpoint_activations': False, 'required_seq_len_multiple': 2, 'depthwise_conv_kernel_size': 31, 'attn_type': '', 'pos_enc_type': 'abs', 'fp16': True}
        new_checkpoint["dictionaries_symbols"] = [[str(i) for i in range(500)]]
        new_checkpoint['model_weight'] = averaged_state_dict
        new_checkpoint["task_cfg"] = {'_name': 'hubert_pretraining', 'data': '', 'fine_tuning': False, 'labels': ['layer6.km500'], 'label_dir': None, 'label_rate': 50.0, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_keep_size': None, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False}
        return new_checkpoint


def remove_model_prefix(state_dict):
    """Removes the 'model.' prefix from state_dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[len("model."):]  # Remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

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

def wrap_mert_weights(mapped_state_dict):
    """Wrap MERT weights with 'model.' prefix to match HuBERT's architecture."""
    wrapped_state_dict = {}
    for key, value in mapped_state_dict.items():
        wrapped_key = f"model.{key}" if not key.startswith("model.") else key
        wrapped_state_dict[wrapped_key] = value
    return wrapped_state_dict


def map_mert_to_hubert(state_dict):
    """Map MERT's keys to match HuBERT's key structure."""
    mapped_state_dict = {}

    for key, value in state_dict.items():
        # Handle convolutional layers
        if key.startswith("feature_extractor.conv_layers"):
            parts = key.split('.')
            layer_idx = parts[2]
            if "conv" in parts:
                new_key = f"feature_extractor.conv_layers.{layer_idx}.0.{'.'.join(parts[4:])}"
            elif "layer_norm" in parts:
                new_key = f"feature_extractor.conv_layers.{layer_idx}.2.{'.'.join(parts[4:])}"
            else:
                new_ley = key  # Skip irrelevant keys like activations
            mapped_state_dict[new_key] = value

        # Handle feature projection
        elif key.startswith("feature_projection"):
            if "projection" in key:
                new_key = key.replace("feature_projection.projection", "post_extract_proj")
            elif "layer_norm" in key:
                new_key = key.replace("feature_projection.layer_norm", "post_extract_proj.layer_norm")
            elif "dropout" in key:
                # Optional: Dropout handling if required
                continue  # Dropout layers might not need to be mapped, as they do not have weights
            mapped_state_dict[new_key] = value

        # Handle encoder layers
        elif key.startswith("encoder.layers"):
            parts = key.split('.')
            layer_idx = parts[2]
            submodule = parts[3]
            rest = parts[4:]
            if submodule == "attention":
                if "q_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.q_proj.{'.'.join(rest[1:])}"
                elif "k_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.k_proj.{'.'.join(rest[1:])}"
                elif "v_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.v_proj.{'.'.join(rest[1:])}"
                elif "out_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.out_proj.{'.'.join(rest[1:])}"
            elif submodule == "layer_norm":
                new_key = f"encoder.layers.{layer_idx}.self_attn_layer_norm.{'.'.join(rest)}"
            elif submodule == "final_layer_norm":
                new_key = f"encoder.layers.{layer_idx}.final_layer_norm.{'.'.join(rest)}"
            elif submodule == "feed_forward":
                if "intermediate_dense" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.fc1.{'.'.join(rest[1:])}"
                elif "output_dense" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.fc2.{'.'.join(rest[1:])}"
            mapped_state_dict[new_key] = value

        
        # Handle positional convolutions (MERT → HuBERT)
        elif key.startswith("encoder.pos_conv_embed.conv"):
            if "bias" in key:
                new_key = "encoder.pos_conv.0.bias"
                # Ensure shape consistency
                if value.shape != torch.Size([768]):
                    value = value.view(768)  # Reshape if necessary
            elif "weight_g" in key:
                new_key = "encoder.pos_conv.0.weight_g"
            elif "weight_v" in key:
                new_key = "encoder.pos_conv.0.weight_v"
            else:
                print(f"Skipping unrecognized key in positional convolution: {key}")
                continue  # Ignore unknown keys

            mapped_state_dict[new_key] = value
        
        elif key == "masked_spec_embed":
            new_key = "model.mask_emb"
            mapped_state_dict[new_key] = value
        
        # # Handle missing layer_norm mapping for MERT
        # elif key == "encoder.layer_norm.weight":
        #     new_key = "model.layer_norm.weight"
        #     mapped_state_dict[new_key] = value
        # elif key == "encoder.layer_norm.bias":
        #     new_key = "model.layer_norm.bias"
        #     mapped_state_dict[new_key] = value

        # Handle other keys (if necessary)
        else:
            print(f"Unrecognized key: {key}")
            mapped_state_dict[key] = value
    
    return mapped_state_dict

import matplotlib.pyplot as plt

def override_config_for_music4all(data_config):
    # Override training dataset and libri_root for music4all
    data_config["datarc"]["train"] = ["music4all_16khz-5000-samples"]
    data_config["datarc"]["libri_root"] = os.environ.get("S3PRL_MUSIC4ALL_ROOT", "/path/to/music4all")
    return data_config

def compare_permutation_matrices(mat1, mat2):
    # Compute cosine similarity and normalized Frobenius norm difference
    flat1 = mat1.flatten()
    flat2 = mat2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(flat1, flat2, dim=0)
    frob_diff = (mat1 - mat2).norm() / (mat1.norm() + 1e-8)
    return cos_sim.item(), frob_diff.item()


def extract_perm_vectors_from_merge(merge_obj):
    """
    Extract MERT permutation vectors (argmax of the permutation matrix) from a
    completed ModelMerge object.  Returns {node_id: perm_vector (int tensor)}.
    """
    perm_vectors = {}
    if not (hasattr(merge_obj, 'merges') and merge_obj.merges):
        return perm_vectors
    for node_id, tup in merge_obj.merges.items():
        if not (isinstance(tup, tuple) and len(tup) == 2):
            continue
        p_mert = tup[1]
        if p_mert is None or p_mert.numel() == 0:
            continue
        if p_mert.ndim != 2 or p_mert.shape[0] != p_mert.shape[1]:
            continue
        perm_vectors[node_id] = torch.argmax(p_mert.cpu(), dim=0)  # shape (D,)
    return perm_vectors


def compute_perm_delta(perm_pass_a, perm_pass_b, perm_pass_identity=None):
    """
    Compare two sets of permutation vectors (output of extract_perm_vectors_from_merge).

    Returns per-node stats:
      {node_id: {
          'n_total': int,
          'n_changed_ab': int,   # neurons whose assignment changed between pass a and b
          'frac_changed_ab': float,
          'n_restored_to_identity': int,   # neurons that were != identity in pass a but == identity in pass b
          'n_newly_moved_from_identity': int,  # neurons that were == identity in pass a but != identity in pass b
      }}
    perm_pass_identity: if None, uses the integer identity permutation [0,1,...,D-1].
    """
    node_ids = set(perm_pass_a.keys()) & set(perm_pass_b.keys())
    stats = {}
    for node_id in sorted(node_ids):
        va = perm_pass_a[node_id]  # (D,)
        vb = perm_pass_b[node_id]  # (D,)
        D = va.shape[0]
        identity = torch.arange(D)

        n_changed_ab = int((va != vb).sum().item())
        frac_changed_ab = n_changed_ab / D if D > 0 else 0.0

        # How many neurons were permuted away from identity in pass a, but restored to identity in pass b
        was_permuted_a = (va != identity)
        restored = (was_permuted_a & (vb == identity))
        n_restored = int(restored.sum().item())

        # How many neurons were at identity in pass a, but moved away in pass b (new permutations)
        was_identity_a = (va == identity)
        newly_moved = (was_identity_a & (vb != identity))
        n_newly_moved = int(newly_moved.sum().item())

        stats[node_id] = {
            'n_total': D,
            'n_changed_ab': n_changed_ab,
            'frac_changed_ab': frac_changed_ab,
            'n_restored_to_identity': n_restored,
            'n_newly_moved_from_identity': n_newly_moved,
        }
    return stats


def log_perm_delta(stats, pass_a, pass_b, save_path=None):
    """Pretty-print per-layer permutation delta and optionally save to JSON."""
    print(f"\n{'='*64}")
    print(f"  Permutation Delta: Pass {pass_a} → Pass {pass_b}")
    print(f"{'='*64}")
    total_changed = 0
    total_restored = 0
    total_newly_moved = 0
    total_neurons = 0
    for node_id, s in stats.items():
        D = s['n_total']
        nc = s['n_changed_ab']
        nr = s['n_restored_to_identity']
        nm = s['n_newly_moved_from_identity']
        frac = s['frac_changed_ab']
        print(f"  node {node_id:3d} (D={D:4d}): "
              f"{nc:4d}/{D} changed ({frac:5.1%})  "
              f"restored={nr:4d}  newly_moved={nm:4d}")
        total_changed += nc
        total_restored += nr
        total_newly_moved += nm
        total_neurons += D
    overall_frac = total_changed / total_neurons if total_neurons > 0 else 0
    print(f"  {'─'*60}")
    print(f"  OVERALL: {total_changed}/{total_neurons} changed ({overall_frac:.2%})"
          f"  restored={total_restored}  newly_moved={total_newly_moved}")
    print(f"{'='*64}")

    if save_path:
        serialisable = {
            str(nid): s for nid, s in stats.items()
        }
        serialisable['_summary'] = {
            'pass_a': pass_a, 'pass_b': pass_b,
            'total_neurons': total_neurons,
            'total_changed': total_changed,
            'overall_frac_changed': overall_frac,
            'total_restored_to_identity': total_restored,
            'total_newly_moved_from_identity': total_newly_moved,
        }
        with open(save_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"  Delta saved → {save_path}")
    return overall_frac


def load_randomized_hubert():
    """
    Load a HuBERT model and randomize its parameters.
    """
    # Load the pretrained HuBERT base model
    hubert_model = getattr(hub, 'hubert_base')()

    # Reinitialize the parameters with random values
    for param in hubert_model.parameters():
        if param.requires_grad:
            torch.nn.init.normal_(param, mean=0, std=0.02)  # Random Gaussian initialization

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hubert_model = hubert_model.to(device)

    return hubert_model

def permute_single_layer(model, target_module_name, seed=42):
    """
    Permute only the target module's weights along the output dimension.
    Returns a dictionary mapping the module name to its permutation vector.
    """
    permutation_dict = {}
    torch.manual_seed(seed)
    random.seed(seed)
    for name, module in model.named_modules():
        if name == target_module_name:
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d):
                weight = module.weight.data.clone()
                # Assume permutation along the output dimension (dim 0)
                perm = torch.randperm(weight.size(0))
                permutation_dict[name] = perm
                module.weight.data.copy_(weight[perm, :])
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.copy_(module.bias.data[perm])
                print(f"Permuted layer: {name} with permutation: {perm.tolist()}")
                return permutation_dict
            else:
                print(f"Module {name} is not Linear/Conv1d; skipping.")
    print(f"Target module {target_module_name} not found!")
    return permutation_dict


def extract_computed_permutation(merge_matrix):
    """
    Given a computed merge matrix (assumed to be a permutation matrix or close),
    extract the permutation as a vector by taking argmax along rows.
    """
    return torch.argmax(merge_matrix, dim=1)

def compare_permutations(ground_truth_perm, computed_perm):
    """
    Compare two permutation vectors by computing the fraction of matching indices.
    """
    return (ground_truth_perm == computed_perm).float().mean().item()

def plot_permutation_accuracy(accuracy_dict, filter_none=True):
    """
    Plot a bar chart of permutation recovery accuracy per module.
    """
    if filter_none:
        filtered = {k: v for k, v in accuracy_dict.items() if v is not None}
    else:
        filtered = accuracy_dict
    
    modules = list(filtered.keys())
    accuracies = [filtered[m] for m in modules]

    
    plt.figure(figsize=(10, 6))
    plt.bar(modules, accuracies)
    plt.xlabel("Module")
    plt.ylabel("Permutation Recovery Accuracy")
    plt.title("Comparison of Ground Truth vs. Computed Permutations per Module")
    plt.xticks(rotation=45, ha="right")
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(f'plot_permutation_accuracy_for_{modules[-1]}.png', dpi=500)
    plt.show()

def compare_model_weights(model_a, model_b):
    """
    Compare two model state dicts by computing a relative L2 norm difference per parameter.
    Returns a dictionary of differences.
    """
    differences = {}
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    for key, param_a in state_a.items():
        if key in state_b:
            param_b = state_b[key]
            # Only compare if shapes match
            if param_a.shape == param_b.shape:
                diff = (param_a - param_b).norm() / (param_a.norm() + 1e-8)
                differences[key] = diff.item()
    return differences



def map_node_to_module(graph, node):
    """Map a PREFIX node to the succeeding MODULE node."""
    info = graph.get_node_info(node)
    if info['layer'] is not None:
        if info['type'] == NodeType.PREFIX:
            return info['layer']
        else:
            return None
    else:
        preds = graph.preds(node)
        if len(preds) == 1:
            pred_info = graph.get_node_info(preds[0])
            if pred_info['layer'] is not None:
                return pred_info['layer']
        return f"{info['type'].name}_{node}"




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


class DistilledModelWrapper(torch.nn.Module):
    """
    Wraps a MultiDistillerModel so its forward() signature matches what the graph
    expects: forward(wavs) instead of forward(wave, pad_mask).
    The pad_mask is computed automatically from the input waveforms.

    Named modules are delegated to the inner model so that graph module lookups
    (e.g., 'encoder.layers.0.self_attn.q_proj') resolve correctly without any prefix.
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model

    def forward(self, x, pad_mask=None):
        # x can be a list of waveforms (s3prl style) or a single tensor
        if isinstance(x, (list, tuple)):
            # Pad to same length and create pad_mask
            max_len = max(wav.shape[-1] for wav in x)
            padded = torch.zeros(len(x), max_len, device=x[0].device)
            mask = torch.zeros(len(x), max_len, dtype=torch.bool, device=x[0].device)
            for i, wav in enumerate(x):
                length = wav.shape[-1]
                padded[i, :length] = wav
                mask[i, length:] = True  # True = padded positions
            return self.inner(padded, mask, no_pred=True)
        else:
            # Single tensor input
            if pad_mask is None:
                pad_mask = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
            return self.inner(x, pad_mask, no_pred=True)

    # Delegate named_modules etc. to inner model so graph lookups work
    def named_modules(self, *args, **kwargs):
        # Yield own name first, then yield inner model's modules WITHOUT 'inner.' prefix
        yield '', self
        for name, module in self.inner.named_modules(*args, **kwargs):
            if name:  # skip empty string (the inner model itself)
                yield name, module

    def named_parameters(self, *args, **kwargs):
        for name, param in self.inner.named_parameters(*args, **kwargs):
            yield name, param

    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.inner.load_state_dict(*args, **kwargs)

    @property
    def encoder(self):
        return self.inner.encoder

    @property
    def feature_extractor(self):
        return self.inner.feature_extractor


def load_distilled_model(ckpt_path, device='cuda'):
    """
    Load a distilled model (e.g. hubert_l1_wide, mert_l1_wide) from a checkpoint file.
    Returns (model, config_dict) where model is a DistilledModelWrapper around
    MultiDistillerModel, and config_dict is the upstream config from the checkpoint.

    The returned model has NO 'model.' prefix in its state dict keys — use model_type='distilled'
    when constructing the HuBERTGraph.
    """
    from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract config — check Config first, then Up_Config for Barlow-style checkpoints
    config_dict = ckpt.get('Config', {})
    upstream_key = None
    upstream_config = None
    for candidate in ['multi_distiller', 'distiller']:
        if candidate in config_dict:
            cfg = config_dict[candidate]
            # Barlow checkpoints have 'distiller' in Config but it only contains
            # {'self_correlation': True}, not the full model config. The full config
            # lives in Up_Config.distiller. Detect this by checking for encoder_layers.
            if isinstance(cfg, dict) and 'encoder_layers' in cfg:
                upstream_key = candidate
                upstream_config = cfg
                break
    if upstream_config is None:
        # Fallback: check Up_Config (Barlow-style checkpoints)
        up_config = ckpt.get('Up_Config', {})
        for candidate in ['distiller', 'multi_distiller']:
            if candidate in up_config:
                cfg = up_config[candidate]
                if isinstance(cfg, dict) and 'encoder_layers' in cfg:
                    upstream_key = candidate
                    upstream_config = cfg
                    print(f"[load_distilled_model] Found config in Up_Config.{candidate}")
                    break
    if upstream_config is None:
        raise ValueError(f"Cannot find upstream config with encoder_layers in checkpoint. "
                         f"Config keys: {list(config_dict.keys())}, "
                         f"Up_Config keys: {list(ckpt.get('Up_Config', {}).keys())}")

    # Ensure config_dict has the full model config under the upstream key.
    # For Barlow checkpoints where config came from Up_Config, patch config_dict
    # so the merged checkpoint will have the correct architecture info.
    if upstream_key and (upstream_key not in config_dict or
            not isinstance(config_dict.get(upstream_key), dict) or
            'encoder_layers' not in config_dict.get(upstream_key, {})):
        config_dict[upstream_key] = upstream_config
        print(f"[load_distilled_model] Patched config_dict['{upstream_key}'] with full model config")

    # Build MultiDistillerConfig from the dict
    model_config = MultiDistillerConfig(upstream_config)

    # Build model
    inner_model = MultiDistillerModel(model_config)

    # Load state dict — checkpoint stores model weights under 'Distiller' or upstream_key
    state_dict = None
    for sd_key in ['Distiller', 'multi_distiller', upstream_key]:
        if sd_key in ckpt:
            state_dict = ckpt[sd_key]
            break
    if state_dict is None:
        raise ValueError(f"Cannot find model state dict in checkpoint. Keys: {list(ckpt.keys())}")

    # Load weights (strict=False to allow missing output_layer keys if pruned)
    missing, unexpected = inner_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_distilled_model] Missing keys (expected if output heads excluded): {missing}")
    if unexpected:
        print(f"[load_distilled_model] Unexpected keys: {unexpected}")

    # Wrap in adapter so forward(wavs) works without explicit pad_mask
    model = DistilledModelWrapper(inner_model)
    model = model.to(device)
    model.encoder.layerdrop = 0  # No dropout during merging
    model.eval()

    print(f"[load_distilled_model] Loaded from {ckpt_path}")
    print(f"  encoder_layers={upstream_config['encoder_layers']}, "
          f"encoder_embed_dim={upstream_config['encoder_embed_dim']}, "
          f"encoder_attention_heads={upstream_config['encoder_attention_heads']}")

    return model, config_dict

def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # upstream settings
    parser.add_argument('--hub', default="torch", choices=["torch", "huggingface"],
        help='The model Hub used to retrieve the upstream model.')

    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    parser.add_argument('--experiment', 
                        choices=['main'], 
                        default='main', 
                        help='Select which experiment to run. Only main allowed in this code version.')
    
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('--enable_weighted_alignment', action='store_true',
                        help='Enable scaling of correlations based on layer importance for alignment.')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')
    parser.add_argument('--upstream_model_name', default="model.pt", help='The name of the model file in the HuggingFace Hub repo.')
    parser.add_argument('--upstream_revision', help="The commit hash of the specified HuggingFace Repository")
    parser.add_argument('-x', '--fix_feature_len', action='store_true', help="Fix the feature length")
    parser.add_argument('--merge_cnn', action='store_true', help='Include CNN feature extractor in merging')
    parser.add_argument('--use_ties', action='store_true', help="Enable sign interference check and handling in TIES merging.")
    parser.add_argument('--quantile', type=float, default=0.8, help='Quantile for trimming in TIES-Merging')
    parser.add_argument('-q', '--ignore_length_dif', action='store_true', help="Fix the feature length")
    parser.add_argument("--interp_weights", type=float, nargs=2, default=[0.5, 0.5],
                    help="Weights for interpolating the merging of two models.")
    parser.add_argument('--json_file', type=str, default=None,
                        help='Optional Google service-account JSON for experiment logging to Google Sheets; omit to disable.')
    parser.add_argument('--logfile', type=str,default=None)#i can reuse my same json file.
    parser.add_argument('-d', '--downstream', help='Typically downstream dataset need manual preparation. Please check downstream/README.md for details')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname',default="baseline", help='Save experiment at result/merged_upstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')
    parser.add_argument('--disable_cudnn', action='store_true', help='Disable CUDNN')
    parser.add_argument("--maintain_hubert_behavior", type=bool, default=True, help="If True, prioritize HuBERT behavior; if False, use sign of largest magnitude.")
    parser.add_argument('--features_path', help='The option for online feature saving.', type=str)
    parser.add_argument('--pre_extract_dir', help='The path for using offline feature preextracted by utility/feature_extractor.', type=str)
    parser.add_argument('--merging_strategy', type=str, default='uniform',
                    choices=['uniform', 'hybrid_zipit_cnn_permute_transformer'],
                    help='Strategy for applying merging algorithms. "uniform" uses --merging_algorithm everywhere. '
                         '"hybrid_zipit_cnn_permute_transformer" uses ZipIt! for CNNs and Permutation for Transformers.')
    parser.add_argument("--merging_algorithm", type=str, default="match_tensors_permute",help="function used for merging the models.")
    parser.add_argument('--merge_type', type=str, default='ff+attn',
                        choices=['ff_only', 'ff+attn', 'qkv', 'qkv+attn', 'qkv+ff', 'all', 'none'], # Add more as needed 
                        help='Defines which parts of the transformer layers to align (controls PREFIX node placement).')
    parser.add_argument('--zipit_a', type=float, default=0.3, help='ZipIt alpha hyperparameter (similarity decay)')
    parser.add_argument('--zipit_b', type=float, default=0.125, help='ZipIt beta hyperparameter (within-model budget control)')
    parser.add_argument('--pitch_important_layers', type=int, nargs='+', default=None,
                    help='List of 0-based layer indices deemed important for PitchID (from MERT analysis).')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for PitchID layer importance')
    parser.add_argument('--make_permutation_analysis', action='store_true', help='If this flag is set, the permutation analysis will be performed and there wont be any saved merged model, only permutation info.')
    parser.add_argument('--run_feature_similarity', action='store_true', 
                    help='Run feature similarity analysis (Experiment 2).')
    parser.add_argument('--sim_analysis_batches', type=int, default=16,
                        help='Number of batches for feature similarity analysis.')
    parser.add_argument('--sim_analysis_nodes_count', type=int, default=3,
                        help='Number of leading PREFIX nodes for feature similarity analysis.')
    parser.add_argument('--run_correlation_heatmap_viz', action='store_true',
                        help='Run channel correlation heatmap visualization experiment.')
    parser.add_argument('--correlation_viz_nodes', type=int, nargs='+', default=None,
                        help='List of specific PREFIX node IDs to visualize for correlation heatmaps. '
                             'If None, a few default nodes might be chosen.')
    parser.add_argument('--correlation_viz_max_dim', type=int, default=769,
                        help='Maximum dimension to plot for correlation heatmaps (sub-matrix if original is larger).')
    parser.add_argument('--exit_after_similarity', action='store_true',
                        help='Exit script after feature similarity analysis (if run).')
    parser.add_argument('--heatmap_vmin', type=float, default=0.0, help='Min value for heatmap color scale.')
    parser.add_argument('--heatmap_vmax', type=float, default=0.7, help='Max value for heatmap color scale.')

    # === Model mode: teacher (HuBERT/MERT base) vs distilled (hubert_l1_wide, mert_l1_wide, etc.) ===
    parser.add_argument('--model_mode', type=str, default='teacher',
                        choices=['teacher', 'distilled'],
                        help='"teacher" merges full HuBERT Base + MERT teacher models (original behavior). '
                             '"distilled" merges two distilled model checkpoints.')
    parser.add_argument('--model1_ckpt', type=str, default=None,
                        help='Path to first model checkpoint (required for --model_mode distilled).')
    parser.add_argument('--model2_ckpt', type=str, default=None,
                        help='Path to second model checkpoint (required for --model_mode distilled).')
    parser.add_argument('--interpolation_weights', type=float, nargs='+', default=None,
                        help='List of HuBERT/model1 weights for interpolation (e.g., 0.9 0.8 0.5 0.2 0.1). '
                             'Model2 weight is computed as 1 - w1. If not provided, uses default set.')
    parser.add_argument('--num_passes', type=int, default=1,
                        help='Number of permutation alignment passes. 1=G1 (single-pass), 2=G2, 5=G3. '
                             'Each pass aligns the previously-permuted MERT toward HuBERT again. '
                             'Permutation delta is logged between consecutive passes.')
    # --- G4 Sinkhorn hyperparameters ---
    parser.add_argument('--sinkhorn_iters', type=int, default=20,
                        help='Number of Sinkhorn row/column normalization iterations.')
    parser.add_argument('--sinkhorn_opt_steps', type=int, default=300,
                        help='Number of gradient descent steps for Sinkhorn optimization.')
    parser.add_argument('--sinkhorn_lr', type=float, default=0.01,
                        help='Learning rate for Sinkhorn logit optimization.')
    parser.add_argument('--sinkhorn_tau_max', type=float, default=1.0,
                        help='Starting (high) temperature for Sinkhorn annealing.')
    parser.add_argument('--sinkhorn_tau_min', type=float, default=0.01,
                        help='Final (low) temperature for Sinkhorn annealing.')
    parser.add_argument('--sinkhorn_e2e', action='store_true',
                        help='Use G4.2 end-to-end Sinkhorn (soft permutations in forward pass). '
                             'Requires --merging_algorithm match_tensors_sinkhorn_cc. '
                             'If not set, G4.1 (pre-collected features Sinkhorn) is used when '
                             '--merging_algorithm is match_tensors_sinkhorn_cc.')





    
    args = parser.parse_args()
    backup_files = []

    # Example weights (absolute values)
    raw_layer_importance = {
         0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, # Assuming layers 0-3 are not critical
         4: 1.5895, 5: 1.9505, 6: 1.8311, # Your critical layers
         7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0 # Assuming layers 7-11 are not critical
    }
    # Normalize (optional)
    #max_importance = max(layer_importance.values()) if layer_importance else 1.0
    args.layer_weights = raw_layer_importance # {l: w / max_importance for l, w in layer_importance.items()}
    #print(f"Layer importance weights: {args.layer_weights}")
    print(f"Using RAW ABSOLUTE layer importance weights: {args.layer_weights}")


    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"args: {args}")
    if args.expdir is None:
        args.expdir = f'result/merged_pretrain_upstream/permutation-covariance/{args.expname}_merge_cnn_{args.merge_cnn}_use_ties_{args.use_ties}_quantile_{args.quantile}_maintain_hubert_behavior_{args.maintain_hubert_behavior}'
    
    
    print('[Runner] - Start a new experiment')
    os.makedirs(args.expdir, exist_ok=True)


    # we do not need this for now.
    # if args.upstream_model_config is not None and os.path.isfile(args.upstream_model_config):
    #     backup_files.append(args.upstream_model_config)

    # if args.override is not None and args.override.lower() != "none":
    #     override(args.override, args, config)
    #     os.makedirs(args.expdir, exist_ok=True)
        
    return args


def find_closest_prefix_node(graph, target_module_name):
    """
    Given a graph and a target module name, return the PREFIX node that comes
    immediately before the node whose layer equals target_module_name.
    If an exact match is not found, return the PREFIX node with the highest id
    that is less than the id of any node with layer==target_module_name.
    """
    # Find all nodes that have layer equal to target_module_name.
    target_nodes = [node for node in graph.G.nodes() 
                    if graph.get_node_info(node)['layer'] == target_module_name]
    if target_nodes:
        target_node = max(target_nodes)  # assume later node is closer
        # Now among all PREFIX nodes with id < target_node, choose the one with maximum id.
        prefix_candidates = [node for node in graph.G.nodes() 
                             if graph.get_node_info(node)['type'] == NodeType.PREFIX and node < target_node]
        if prefix_candidates:
            return max(prefix_candidates)
    # Otherwise, try a fallback: return any PREFIX node whose successor has the target name.
    for node in sorted(graph.G.nodes()):
        info = graph.get_node_info(node)
        if info['type'] == NodeType.PREFIX:
            succs = graph.succs(node)
            for s in succs:
                succ_info = graph.get_node_info(s)
                if succ_info['layer'] == target_module_name:
                    return node
    return None

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces

def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)

from s3prl.upstream.interfaces import Featurizer

def _get_featurizer(model,args):
        model = Featurizer(
            upstream = model.model,
            feature_selection = args.upstream_feature_selection,
            layer_selection = args.upstream_layer_selection,
            upstream_device = "cuda",
            normalize = args.upstream_feature_normalize,
            fixed_length = args.fix_feature_len,
            ignore_length_dif = args.ignore_length_dif
        ).to("cuda")

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )

def _get_downstream(featurizer, model,config,args):
    expert = importlib.import_module(f"s3prl.downstream.{args.downstream}.expert")
    Downstream = getattr(expert, "DownstreamExpert")
    model = Downstream(
        upstream_dim=featurizer.model.output_dim,
        upstream_rate=featurizer.model.downsample_rate,
        **dict(config, sample_rate=model.model.sample_rate),
        **vars(args)
    ).to("cuda")
    return _init_model(
        model=model,
        name='Downstream',
        trainable=True,
        interfaces=['get_dataloader', 'log_records']
    )

def collect_features(model, dataloader, layer_names):
    if hasattr(model, 'remove_all_hooks'):
        model.remove_all_hooks()


    averaged_features_collect = {layer: [] for layer in layer_names}
    all_labels = []
    hooks = []
    
    # Verify layer names exist in the model
    named_modules = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in named_modules:
            raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {list(named_modules.keys())}")

    # Register hooks with proper scoping
    for layer_name in layer_names:
        module = named_modules[layer_name]
        # Use a named function to avoid lambda scoping issues
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # Check dimensions and decide which dimension is time:
                if isinstance(output, torch.Tensor):
                    if output.dim() == 3:
                        # For conv layers, assume [B, C, T]; for transformer layers, assume [B, T, C].
                        # You can check the layer name: if it contains "conv_layers", then assume conv shape.
                        if "conv_layers" in layer_name:
                            # Average over time dimension (dim=2) for conv layers
                            reduced = torch.mean(output.detach(), dim=2)
                        else:
                            # Otherwise, for transformer layers assume [B, T, C] and average over time (dim=1)
                            reduced = torch.mean(output.detach(), dim=1)
                    else:
                        # For outputs not 3D, just take them as is.
                        reduced = output.detach()
                    averaged_features_collect[layer_name].append(reduced.cpu())
                else:
                    print(f"Warning: {layer_name} output is not a tensor, got {type(output)}")
            return hook_fn

        
        hook = module.register_forward_hook(make_hook(layer_name))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for wavs, labels, *others  in dataloader:
            wavs = [torch.FloatTensor(wav).to("cuda") for wav in wavs]
            _ = model(wavs)
            all_labels.append(list(labels))
    
    for hook in hooks:
        hook.remove()
    

    # Concatenate results
    #all_labels = torch.cat(all_labels, dim=0)
    for layer in layer_names:
        if not averaged_features_collect[layer]:
            raise RuntimeError(f"No features collected for layer '{layer}'. Check model forward pass.")
        averaged_features_collect[layer] = torch.cat(averaged_features_collect[layer], dim=0)  # Shape: (total_samples, feature_dim, time)

    return averaged_features_collect, all_labels

def average_over_time(features):
    return {layer: feat.mean(dim=2) for layer, feat in features.items()}  # Shape: (total_samples, feature)

def compute_avg_cosine_similarity(feat1, feat2):
    feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
    feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
    similarity = (feat1 * feat2).sum(dim=1)
    return similarity.mean().item()


def to_serializable(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    elif isinstance(x, (list, tuple)):
        return [to_serializable(item) for item in x]
    else:
        return x

import re
# --- Helper function for shorter display names ---
def simplify_display_name(original_name, node_id):
    # Try to parse from model.feature_extractor.conv_layers.X.Y_...
    match_cnn_prefix = re.match(r"model\.feature_extractor\.conv_layers\.(\d+)\.(\d+)_.*", original_name)
    if match_cnn_prefix:
        block, stage = match_cnn_prefix.groups()
        return f"CNN_L{block}_S{stage}"

    # Try to parse from InputTo_model.feature_extractor.conv_layers.X.Y
    match_cnn_input = re.match(r"InputTo_model\.feature_extractor\.conv_layers\.(\d+)\.(\d+)", original_name)
    if match_cnn_input:
        block, stage = match_cnn_input.groups()
        return f"CNN_L{block}_S{stage}_In" # Less likely to be the target for PREFIX

    # Try to parse from model.encoder.layers.L.self_attn.out_proj_...
    match_enc_attn = re.match(r"model\.encoder\.layers\.(\d+)\.self_attn\.out_proj_.*", original_name)
    if match_enc_attn:
        layer_idx = match_enc_attn.group(1)
        return f"Enc{layer_idx}_AttnOut"
    
    # Try to parse from InputTo_model.encoder.layers.L.self_attn.out_proj
    match_enc_attn_input = re.match(r"InputTo_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj", original_name)
    if match_enc_attn_input:
            layer_idx = match_enc_attn_input.group(1)
            return f"Enc{layer_idx}_AttnOut_In" # Usually what a PREFIX before out_proj means

    # Try to parse from model.encoder.layers.L.fc2_...
    match_enc_fc2 = re.match(r"model\.encoder\.layers\.(\d+)\.fc2_.*", original_name)
    if match_enc_fc2:
        layer_idx = match_enc_fc2.group(1)
        return f"Enc{layer_idx}_FC2_Pre"
    
    # Try to parse from InputTo_model.encoder.layers.L.fc2
    match_enc_fc2_input = re.match(r"InputTo_model\.encoder\.layers\.(\d+)\.fc2", original_name)
    if match_enc_fc2_input:
        layer_idx = match_enc_fc2_input.group(1)
        return f"Enc{layer_idx}_FC2_In" # Usually what a PREFIX before fc2 means

    # Fallback if using InputTo_ structure and it's not caught above
    if original_name.startswith("InputTo_"):
        # A simple fallback for InputTo_
        simplified = original_name.replace("InputTo_model.encoder.layers.", "Enc")
        simplified = simplified.replace(".self_attn.out_proj", "_AttnIn")
        simplified = simplified.replace(".fc2", "_FC2In")
        simplified = simplified.replace("InputTo_model.feature_extractor.conv_layers.", "CNN")
        simplified = simplified.replace(".", "_") # Replace remaining dots
        if len(simplified) > 25: # Truncate if still too long
            simplified = simplified[:12] + "..." + simplified[-10:]
        return simplified

    # Fallback for _(PREFIX_TARGET)
    if "_(PREFIX_TARGET)" in original_name:
        simplified = original_name.replace("_(PREFIX_TARGET)", "_PFX")
        simplified = simplified.replace("model.encoder.layers.", "Enc")
        simplified = simplified.replace(".self_attn.out_proj", "_AttnOut")
        simplified = simplified.replace(".fc2", "_FC2Pre")
        simplified = simplified.replace("model.feature_extractor.conv_layers.", "CNN_")
        simplified = simplified.replace(".", "_") # Replace remaining dots
        if len(simplified) > 25: # Truncate
            simplified = simplified[:12] + "..." + simplified[-10:]
        return simplified

    return f"Node{node_id}" # Absolute fallback



def main(args):
    logging.basicConfig(level=logging.INFO)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('soundfile')
    hack_isinstance()

    # Build interpolation schemes from CLI or defaults
    if args.interpolation_weights is not None:
        interpolation_schemes = [(w, round(1.0 - w, 4)) for w in args.interpolation_weights]
    else:
        interpolation_schemes = [
            (0.0, 1.0),
            (0.1, 0.9),
            (0.2, 0.8),
            (0.5, 0.5),
            (0.8, 0.2),
            (0.9, 0.1),
            (1.0, 0.0),
        ]
    print(f"Will generate merged models for interpolation schemes: {interpolation_schemes}")

    # get config and arguments
    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    if sum(args.interp_weights) != 1.0:
        print("Warning: Interpolation weights do not sum to 1. Not Normalizing...")

    # --- Model Loading: branch on model_mode ---
    source_config_dict = None  # Only used for distilled mode

    if args.model_mode == 'distilled':
        # === Distilled model path ===
        if not args.model1_ckpt or not args.model2_ckpt:
            raise ValueError("--model1_ckpt and --model2_ckpt are required for --model_mode distilled")

        print(f"[Distilled Mode] Loading model1 from: {args.model1_ckpt}")
        model1_for_graph, config_dict1 = load_distilled_model(args.model1_ckpt, device=args.device)
        print(f"[Distilled Mode] Loading model2 from: {args.model2_ckpt}")
        model2_for_graph, config_dict2 = load_distilled_model(args.model2_ckpt, device=args.device)

        # Use config from model1 as the base for the merged checkpoint
        source_config_dict = config_dict1

        graph_model_type = 'distilled'
        model1_name = os.path.basename(os.path.dirname(args.model1_ckpt))
        model2_name = os.path.basename(os.path.dirname(args.model2_ckpt))

    else:
        # === Teacher model path (original behavior) ===
        model1_base = load_hubert_base(model_name="hubert_base")
        model2_mapped = load_hubert_base(model_name="hubert_base") # Start with HuBERT architecture

        temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
        temp_config.output_hidden_states = True
        if not hasattr(temp_config, 'conv_pos_batch_norm'):
            temp_config.conv_pos_batch_norm = False  # Fix for newer transformers versions
        model2_mert = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(args.device)
        disable_MERT_encoder_dropout(model2_mert)

        print("Mapping MERT weights to HuBERT structure...")
        new_state_dict = map_mert_to_hubert(model2_mert.state_dict())
        wrapped_weights = wrap_mert_weights(new_state_dict)
        missing_keys, unexpected_keys = model2_mapped.load_state_dict(wrapped_weights, strict=False)
        print("MERT Mapping - Missing keys:", missing_keys)
        print("MERT Mapping - Unexpected keys:", unexpected_keys)

        print("Loading fresh model instances for merging...")
        model1_for_graph = load_hubert_base(model_name="hubert_base")
        model2_for_graph = load_hubert_base(model_name="hubert_base")
        model2_for_graph.load_state_dict(wrapped_weights, strict=False)
        del model2_mert

        graph_model_type = 'hubert'
        model1_name = "hubert_base"
        model2_name = "mert_base"

    # --- Dataloader ---
    with open("merging_utils/data_config.yaml", "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    print(f"[DATA INFO] data config is {data_config}")
    dataloader = get_dataloader(data_config, split="train")

    # --- Prepare for multi-pass permutation (G1=1 pass, G2=2 passes, G3=5 passes) ---
    # Save the original model1 state dict so each pass can reload a fresh copy.
    # model2 state is updated after each pass to the permuted weights.
    from upstream.multi_distiller.model import MultiDistillerModel, MultiDistillerConfig
    if args.model_mode == 'distilled':
        model1_original_state = {k: v.clone() for k, v in model1_for_graph.state_dict().items()}
        model2_current_state = {k: v.clone() for k, v in model2_for_graph.state_dict().items()}
        upstream_key = next(k for k in source_config_dict if k in ('multi_distiller', 'distiller'))

        # Build a randomised model_to_merge once (used as ZipIt budget dummy)
        fresh_config_dummy = MultiDistillerConfig(source_config_dict[upstream_key])
        fresh_inner_dummy = MultiDistillerModel(fresh_config_dummy)
        for param in fresh_inner_dummy.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=0.02)
        model_to_merge = DistilledModelWrapper(fresh_inner_dummy).to(args.device)
    else:
        model_to_merge = load_randomized_hubert()

    merging_function = get_merging_fn(args.merging_algorithm)
    merging_metric = get_metric_fns(["covariance"])

    # =====================================================================
    # G4.2 END-TO-END SINKHORN BRANCH
    # If --sinkhorn_e2e is set, bypass the multi-pass graph-based loop and
    # use the PermutedModelWrapper for global end-to-end optimization.
    # After optimization, inject hard permutations into the graph infrastructure
    # for weight application and merging.
    # =====================================================================
    if args.sinkhorn_e2e:
        print(f"\n{'='*64}")
        print(f"  G4.2 End-to-End Sinkhorn Optimization")
        print(f"{'='*64}")

        # logging already imported at module level (line 7)
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        from sinkhorn_e2e import PermutedModelWrapper, optimize_sinkhorn_e2e, build_node_id_to_name_mapping

        # --- Detect model config ---
        mdm = model2_for_graph.inner if hasattr(model2_for_graph, 'inner') else model2_for_graph
        num_layers = len(mdm.encoder.layers)
        embed_dim = mdm.config.encoder_embed_dim if hasattr(mdm.config, 'encoder_embed_dim') else 768
        ffn_dim = mdm.config.encoder_ffn_embed_dim if hasattr(mdm.config, 'encoder_ffn_embed_dim') else 3072
        num_heads = mdm.config.encoder_attention_heads if hasattr(mdm.config, 'encoder_attention_heads') else 12
        print(f"  Model config: {num_layers} layers, embed_dim={embed_dim}, ffn_dim={ffn_dim}, heads={num_heads}")

        # --- Create PermutedModelWrapper for model B ---
        permuted_b = PermutedModelWrapper(
            inner_model=model2_for_graph,
            num_layers=num_layers, embed_dim=embed_dim, ffn_dim=ffn_dim,
            num_heads=num_heads, merge_cnn=args.merge_cnn,
            sinkhorn_iters=args.sinkhorn_iters, device=args.device,
        )

        # --- Build graph for model A (for feature collection via hooks) ---
        graph_a = HuBERTGraph(model1_for_graph, merge_type=args.merge_type,
                              merge_cnn=args.merge_cnn,
                              model_type=graph_model_type).graphify()

        # --- Run E2E optimization ---
        hard_perms = optimize_sinkhorn_e2e(
            model_a_graph=graph_a,
            permuted_model_b=permuted_b,
            dataloader=dataloader,
            num_opt_steps=args.sinkhorn_opt_steps,
            lr=args.sinkhorn_lr,
            tau_max=args.sinkhorn_tau_max,
            tau_min=args.sinkhorn_tau_min,
            device=args.device,
        )

        # --- Inject hard permutations into graph infrastructure ---
        # Build graph for model B and create ModelMerge for weight application
        graph_b = HuBERTGraph(model2_for_graph, merge_type=args.merge_type,
                              merge_cnn=args.merge_cnn,
                              model_type=graph_model_type).graphify()

        Merge = ModelMerge(graph_a, graph_b, device=args.device)
        Merge.zipit_analysis = {}  # Not used in E2E mode

        # Build name→node_id reverse mapping from graph A
        name_to_node = {}
        node_to_name = build_node_id_to_name_mapping(graph_a)
        for nid, name in node_to_name.items():
            name_to_node[name] = nid

        # Inject permutations: for each merge point, set the merge matrices
        # Format: merges[node_id] = (P_model_a, P_model_b)
        # P_model_a = identity, P_model_b = the hard permutation
        Om_map = {}  # cache dims per merge point
        Merge.merges = {}
        Merge.unmerges = {}
        for name, P in hard_perms.items():
            if name not in name_to_node:
                print(f"  Warning: merge point '{name}' not found in graph, skipping")
                continue
            nid = name_to_node[name]
            D = P.shape[0]
            # P from extract_hard_permutations is [D, D] with P[i] = one-hot for mapped index
            # The graph expects (merge_A, merge_B) where merge_B is transposed
            # For orthogonal permutations: merge = unmerge = P.T
            identity = torch.eye(D, device=args.device)
            P_t = P.T.to(args.device)
            Merge.merges[nid] = (identity, P_t)
            Merge.unmerges[nid] = (identity, P_t)
            Om_map[name] = D
            print(f"  Injected permutation for {name} (node {nid}): {D}×{D}")

        # Apply permutations to model B weights using existing graph infrastructure
        print(f"\n  Applying E2E permutations to model weights...")
        Merge.apply_transformations_custom()

        graphs = [[Merge.graphs[0], Merge.graphs[1]]]
        cost_dict = {}  # E2E branch: costs tracked internally by optimization loop
        all_perm_vectors = []
        all_cost_dicts = [cost_dict]
        print(f"  G4.2 E2E optimization complete. Proceeding to merge/interpolation...")

    else:
        # =====================================================================
        # STANDARD GRAPH-BASED MULTI-PASS LOOP (G1/G2/G3/G4.1)
        # =====================================================================

        # Multi-pass loop
        all_perm_vectors = []   # one dict per pass: {node_id -> perm_vector}
        all_cost_dicts = []

        for pass_num in range(1, args.num_passes + 1):
            print(f"\n{'='*64}")
            print(f"  Permutation Pass {pass_num}/{args.num_passes}")
            print(f"{'='*64}")

        # --- Load fresh models for this pass ---
        if args.model_mode == 'distilled':
            # model1: always the original HuBERT/speech model
            fresh_cfg1 = MultiDistillerConfig(source_config_dict[upstream_key])
            model1_pass = DistilledModelWrapper(MultiDistillerModel(fresh_cfg1)).to(args.device)
            model1_pass.load_state_dict(model1_original_state)

            # model2: original MERT for pass 1, permuted state for subsequent passes.
            # Use strict=False because the MERT checkpoint uses output_layer.* (single-teacher)
            # but the fresh MultiDistillerModel expects output_layers.{teacher}.* (multi-teacher).
            # The backbone/encoder keys always match; only output projections differ and are irrelevant.
            fresh_cfg2 = MultiDistillerConfig(source_config_dict[upstream_key])
            model2_pass = DistilledModelWrapper(MultiDistillerModel(fresh_cfg2)).to(args.device)
            missing, unexpected = model2_pass.inner.load_state_dict(model2_current_state, strict=False)
            if missing or unexpected:
                backbone_missing = [k for k in missing if not k.startswith('output_layer')]
                if backbone_missing:
                    raise RuntimeError(f"Critical backbone keys missing from model2 state dict: {backbone_missing}")
                print(f"  [Pass {pass_num}] model2 loaded (strict=False): "
                      f"{len(missing)} missing (output_layer keys), {len(unexpected)} unexpected — OK")
        else:
            # Teacher-mode: pass 1 only (iterative not implemented for teacher mode)
            model1_pass = model1_for_graph
            model2_pass = model2_for_graph

        # --- Build graphs ---
        graph1 = HuBERTGraph(model1_pass, merge_type=args.merge_type, merge_cnn=args.merge_cnn,
                             model_type=graph_model_type).graphify()
        graph2 = HuBERTGraph(model2_pass, merge_type=args.merge_type, merge_cnn=args.merge_cnn,
                             model_type=graph_model_type).graphify()

        if pass_num == 1:
            for node in graph1.G.nodes:
                info = graph1.get_node_info(node)
                print(f"Node {node}: {info}")

        # --- Run transform ---
        Merge = ModelMerge(graph1, graph2, device="cuda")
        unmerge, cost_dict = Merge.transform(
            model_to_merge,
            dataloader,
            sentence_level=None,
            special_toks=False,
            transform_fn=merging_function,
            metric_classes=merging_metric,
            permute_heads=(args.merge_type == 'qkv+attn' or args.merge_type == 'ff+attn' or args.merge_type == 'all'),
            ignore_heads=False,
            save_both=False,
            merge_cls=False,
            no_absval=True,
            saved_features=None,
            res_type="first",
            interp_w=[0.5, 0.5],
            quantile=args.quantile,
            use_ties=args.use_ties,
            maintain_hubert_behavior=args.maintain_hubert_behavior,
            merge_type=args.merge_type,
            layer_weights=args.layer_weights,
            alpha=args.alpha,
            enable_weighted_alignment=args.enable_weighted_alignment,
            run_feature_similarity_analysis=args.run_feature_similarity,
            num_batches_for_similarity=args.sim_analysis_batches,
            nodes_for_similarity_count=args.sim_analysis_nodes_count,
            # G4 Sinkhorn hyperparameters (ignored by G1/G2 matching functions via **kwargs)
            sinkhorn_iters=args.sinkhorn_iters,
            num_opt_steps=args.sinkhorn_opt_steps,
            lr=args.sinkhorn_lr,
            tau_max=args.sinkhorn_tau_max,
            tau_min=args.sinkhorn_tau_min,
        )
        all_cost_dicts.append(cost_dict)

        # --- Extract permutation vectors and compute deltas ---
        perm_vecs_this_pass = extract_perm_vectors_from_merge(Merge)
        all_perm_vectors.append(perm_vecs_this_pass)

        if pass_num > 1 and len(all_perm_vectors) >= 2:
            delta_stats = compute_perm_delta(all_perm_vectors[-2], all_perm_vectors[-1])
            os.makedirs(args.expdir, exist_ok=True)
            delta_path = os.path.join(
                args.expdir,
                f"perm_delta_pass{pass_num-1}_to_pass{pass_num}.json"
            )
            overall_delta = log_perm_delta(delta_stats, pass_num - 1, pass_num,
                                           save_path=delta_path)
            print(f"  Overall permutation delta (pass {pass_num-1}→{pass_num}): {overall_delta:.2%}")

        # --- Update model2 state for next pass (if any) ---
        if pass_num < args.num_passes and args.model_mode == 'distilled':
            model2_current_state = {k: v.clone()
                                    for k, v in Merge.graphs[1].model.state_dict().items()}
            print(f"  Pass {pass_num} done — model2 updated with permuted state for pass {pass_num+1}.")

        # 'Merge' now refers to the final pass's ModelMerge object
        # 'cost_dict' is the final pass's cost dict
        graphs = [[Merge.graphs[0], Merge.graphs[1]]]

    # --- Both branches (E2E and standard) set 'graphs' and 'Merge' ---

    

    if args.make_permutation_analysis:
        DEBUG_PERMUTATION_DETAILS = True # Set to True to see detailed matrix info
        import sys
        # --- Experiment: Quantify Permutation Intensity ---
        print("\n--- Running Permutation Intensity Experiment ---")
        permutation_intensity_results = {
            "hubert": {},
            "mert": {},
            "node_layer_map": {}
        }

        if hasattr(Merge, 'merges') and Merge.merges:
            print(f"starting inside analusis..")
            # Assuming HuBERT is graph 0, MERT is graph 1
            # And Merge.unmerges[node_id][1] is the P matrix for MERT
            # And Merge.unmerges[node_id][0] is the P matrix for HuBERT (should be Identity)

            sorted_prefix_nodes = sorted([
                node_id for node_id in Merge.merges.keys()
                if isinstance(Merge.merges[node_id], tuple) and len(Merge.merges[node_id]) == 2
            ])

            for node_id in sorted_prefix_nodes:
                unmerge_hubert = Merge.merges[node_id][0].cpu() # P_hubert
                unmerge_mert = Merge.merges[node_id][1].cpu()   # P_mert
                
                node_info_prefix = Merge.graphs[0].get_node_info(node_id)
                display_layer_name = f"PREFIX_Node_{node_id}" # Fallback

                # Option 1: Use the 'layer' attribute of the PREFIX node itself if set meaningfully
                if node_info_prefix.get('layer'):
                    display_layer_name = f"{node_info_prefix['layer']}_(PREFIX_TARGET)"
                else:
                # Option 2: Try to find the first *module* successor
                    q = list(Merge.graphs[0].succs(node_id))
                    visited_succ = set(q)
                    found_module_succ = False
                    while q:
                        curr_succ = q.pop(0)
                        succ_info = Merge.graphs[0].get_node_info(curr_succ)
                        if succ_info['type'] == NodeType.MODULE and succ_info.get('layer'):
                            display_layer_name = f"InputTo_{succ_info['layer']}"
                            found_module_succ = True
                            break
                        for s_next in Merge.graphs[0].succs(curr_succ):
                            if s_next not in visited_succ:
                                visited_succ.add(s_next)
                                q.append(s_next)
                    if not found_module_succ:
                        # Option 3: Try to find the first *module* predecessor if it's for output alignment
                        # This depends on how your graph defines PREFIX vs POSTFIX effects.
                        # For now, the successor logic is more common for `unmerge`.
                        pass


                permutation_intensity_results["node_layer_map"][node_id] = display_layer_name
                if DEBUG_PERMUTATION_DETAILS:
                    print(f"\n--- Node ID: {node_id} ({display_layer_name}) ---")
                    print(f"  HuBERT unmerge matrix shape: {unmerge_hubert.shape}")
                    print(f"  MERT   unmerge matrix shape: {unmerge_mert.shape}")

                for model_name, p_matrix in [("hubert", unmerge_hubert), ("mert", unmerge_mert)]:
                    if p_matrix is None or p_matrix.numel() == 0:
                        print(f"Warning: Empty permutation matrix for {model_name} at node {node_id} ({display_layer_name}). Skipping.")
                        permutation_intensity_results[model_name][display_layer_name] = None
                        continue

                    if p_matrix.ndim != 2 or p_matrix.shape[0] != p_matrix.shape[1]:
                        print(f"Warning: Permutation matrix for {model_name} at node {node_id} ({display_layer_name}) is not square 2D ({p_matrix.shape}). Skipping.")
                        permutation_intensity_results[model_name][display_layer_name] = None
                        continue
                    
                    D = p_matrix.shape[0]
                    if D == 0:
                        permutation_intensity_results[model_name][display_layer_name] = 0.0
                        continue
                    
                    if DEBUG_PERMUTATION_DETAILS:
                        print(f"\n  Inspecting matrix for {model_name} (Node {node_id}, Dim {D}):")
                        is_binary = torch.all((p_matrix == 0) | (p_matrix == 1))
                        row_sums = p_matrix.sum(dim=1)
                        col_sums = p_matrix.sum(dim=0)
                        is_row_stochastic = torch.allclose(row_sums, torch.ones(D), atol=1e-5)
                        is_col_stochastic = torch.allclose(col_sums, torch.ones(D), atol=1e-5)
                        print(f"    Is binary (0 or 1 values only)? {is_binary}")
                        print(f"    All row sums close to 1? {is_row_stochastic} (Example sum: {row_sums[0].item() if D > 0 else 'N/A'})")
                        print(f"    All col sums close to 1? {is_col_stochastic} (Example sum: {col_sums[0].item() if D > 0 else 'N/A'})")
                        if D > 0: print(f"    Matrix (first 5x5 if D>=5):\n{p_matrix[:min(5,D), :min(5,D)]}")
                        # Check if it's an identity matrix
                        if D > 0:
                            is_identity = torch.allclose(p_matrix, torch.eye(D, device=p_matrix.device), atol=1e-5)
                            print(f"    Is identity matrix? {is_identity}")
                    
                    perm_indices = torch.argmax(p_matrix, dim=0)
                    identity_indices = torch.arange(D)
                    num_permuted_channels = torch.sum(perm_indices != identity_indices).item()
                    percent_permuted = (num_permuted_channels / D) * 100 if D > 0 else 0
                    permutation_intensity_results[model_name][display_layer_name] = percent_permuted

                    if DEBUG_PERMUTATION_DETAILS and D > 0:
                        print(f"    Calculated perm_indices (first 10): {perm_indices[:10].tolist()}")
                        print(f"    Identity indices (first 10)   : {identity_indices[:10].tolist()}")
                        print(f"    Number of permuted channels: {num_permuted_channels} / {D}")
                        print(f"    Percentage permuted: {percent_permuted:.2f}%")
                    
                    if model_name == "hubert" and percent_permuted > 1.0:
                        print(f"SANITY CHECK: HuBERT shows {percent_permuted:.2f}% permuted channels at {display_layer_name} (Node {node_id}). Expected ~0%.")

            # --- Plotting (MERT only) ---
            if permutation_intensity_results["mert"]:
                simplified_labels_map = {}
                for node_id_key, old_label in permutation_intensity_results["node_layer_map"].items():
                    simplified_labels_map[old_label] = simplify_display_name(old_label, node_id_key)
                    
                # Prepare data for MERT plot, filtering out None values
                mert_data_for_plot = {}
                for old_label, perc in permutation_intensity_results["mert"].items():
                    if perc is not None:
                        simplified_label = simplified_labels_map.get(old_label, old_label) # Use simplified if available
                        mert_data_for_plot[simplified_label] = perc

                # mert_data_for_plot = {
                #     label: perc 
                #     for label, perc in permutation_intensity_results["mert"].items() 
                #     if perc is not None
                # }
                
                if not mert_data_for_plot:
                    print("No valid MERT permutation data to plot after filtering.")
                else:
                    plot_labels = []
                    plot_percentages = []
                    
                    # Use sorted_prefix_nodes to determine the order
                    for node_id in sorted_prefix_nodes:
                        original_label_for_node = permutation_intensity_results["node_layer_map"].get(node_id)
                        if original_label_for_node:
                            simplified_label = simplified_labels_map.get(original_label_for_node, original_label_for_node)
                            percentage = permutation_intensity_results["mert"].get(original_label_for_node)
                            if percentage is not None:
                                plot_labels.append(simplified_label)
                                plot_percentages.append(percentage)
                    
                    if not plot_labels: # Fallback if the above logic fails
                        plot_labels = list(mert_data_for_plot.keys())
                        plot_percentages = list(mert_data_for_plot.values())


                    # labels = list(mert_data_for_plot.keys())
                    # mert_percentages = list(mert_data_for_plot.values())

                    x = np.arange(len(plot_labels))
                    fig_width = max(15, len(plot_labels) * 0.55) # Dynamically adjust width

                    
                    fig, ax = plt.subplots(figsize=(fig_width, 7))
                    
                    # Plot only MERT percentages
                    ax.bar(x, plot_percentages, width=0.7, label='MERT: Permuted Channels', color='deepskyblue', edgecolor='black', linewidth=0.5)

                    ax.set_ylabel('Permuted Channels in MERT (%)', fontsize=12)
                    ax.set_title(f'MERT Permutation Intensity by Layer\n(Merge: {args.merging_algorithm}, Type: {args.merge_type}, CNN: {args.merge_cnn})', fontsize=14)
                    ax.set_xticks(x)
                    ax.set_xticklabels(plot_labels, rotation=60, ha="right", fontsize=9) # Rotate more for clarity
                    ax.legend(loc='upper left', fontsize=10) # Move legend
                    ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='gray')
                    ax.set_ylim(0, 100.5)
                    ax.tick_params(axis='y', labelsize=10)
                    # Add percentage text on top of bars if not too cluttered
                    if len(plot_labels) < 30: # Only add text if few bars
                        for i, v in enumerate(plot_percentages):
                            if v > 5: # Only label significant bars
                                ax.text(i, v + 1.5, f"{v:.0f}%", color='black', ha='center', va='bottom', fontsize=7)
                    
                    plt.subplots_adjust(bottom=0.25, top=0.9) # Adjust margins
                    fig.tight_layout()
                    save_dir = args.expdir
                    os.makedirs(save_dir, exist_ok=True)
                    merging_fn_name = args.merging_algorithm
                    base_exp_tag = f"{merging_fn_name}_{args.merge_type}_mergecnn_{args.merge_cnn}_hubert_base_mert_base"
                    plot_filename = f'{save_dir}/mert_permutation_intensity_{base_exp_tag}_cleaned.png' # Updated filename
                    plt.savefig(plot_filename, dpi=400)
                    print(f"Saved MERT permutation intensity plot to {plot_filename}")
                    plt.close(fig)
                    results_data = {
                        "args": {k: str(v) if isinstance(v, Namespace) else v for k, v in vars(args).items()}, # Convert Namespace if present
                        "data_config": data_config if data_config else "Dummy", # Handle dummy data
                        "cost_dict": {str(k): float(v) if v is not None else 'NaN' for k, v in cost_dict.items()}, # Convert keys & handle NaN
                        "zipit_analysis": Merge.zipit_analysis # Save the analysis results
                    }
                    args_and_results_path = f'{save_dir}/{base_exp_tag}.args_results.json' # Use JSON
                # Save HuBERT permutation data for internal checks
                    # Add to existing JSON results (still good to save HuBERT's data for internal checks)
                    if "results_data" in locals() and isinstance(results_data, dict):
                        results_data["permutation_intensity_analysis"] = permutation_intensity_results
                        try:
                            with open(args_and_results_path, 'w+') as f_out:
                                json.dump(results_data, f_out, indent=4, cls=NpEncoder) # Added NpEncoder for robustness
                            print(f"Updated args, costs, and permutation analysis to {args_and_results_path}")
                        except Exception as e:
                            print(f"Error updating JSON results with permutation analysis: {e}")
                    else:
                        print("Warning: `results_data` dictionary not found. Permutation analysis not saved to JSON.")
            else:
                print("No MERT permutation data collected. Skipping intensity plot.")
        else:
            print("No permutation matrices found in Merge.unmerges. Skipping intensity experiment.")

        print("--- Permutation Intensity Experiment Finished ---")

        # Save raw permutation matrices for spy plots (Exp 1B)
        perm_matrix_save_dir = os.path.join("result/analysis/chapter5/perm_matrices")
        os.makedirs(perm_matrix_save_dir, exist_ok=True)
        if hasattr(Merge, 'merges') and Merge.merges:
            matrices_to_save = {}
            for node_id in sorted_prefix_nodes:
                if isinstance(Merge.merges[node_id], tuple) and len(Merge.merges[node_id]) == 2:
                    matrices_to_save[node_id] = {
                        'hubert': Merge.merges[node_id][0].cpu(),
                        'mert': Merge.merges[node_id][1].cpu(),
                    }
            save_path = os.path.join(perm_matrix_save_dir, f"{args.expname}_perm_matrices.pt")
            torch.save(matrices_to_save, save_path)
            print(f"Saved {len(matrices_to_save)} permutation matrices to {save_path}")

        Merge.clear_hooks()  # For example
        print("Analysis complete, exiting...")
        sys.exit(0)
    

    if args.run_feature_similarity:
        import sys
        Merge.clear_hooks()  # For example
        sys.exit(0)
        
    
    
    save_dir = args.expdir
    # model1_name and model2_name are already set above based on model_mode
    merging_fn_name = args.merging_algorithm
    merge_type_name = args.merge_type # Ensure merge_type is passed or defined

    # <<< Add ZipIt! hypers to filename if applicable >>>
    zipit_hypers_str = ""
    if 'zipit' in merging_fn_name and hasattr(args, 'zipit_a') and hasattr(args, 'zipit_b'):
        zipit_hypers_str = f"_a{args.zipit_a}_b{args.zipit_b}"
    # <<< >>>

    os.makedirs(save_dir, exist_ok=True)

    # Save args and config once
    base_exp_tag = f"{merging_fn_name}{zipit_hypers_str}_{merge_type_name}_mergecnn_{args.merge_cnn}_{model1_name}_{model2_name}"

    args_and_results_path = f'{save_dir}/{base_exp_tag}.args_results.json' # Use JSON

    results_data = {
        "args": {k: str(v) if isinstance(v, Namespace) else v for k, v in vars(args).items()}, # Convert Namespace if present
        "data_config": data_config if data_config else "Dummy", # Handle dummy data
        "cost_dict": {str(k): float(v) if v is not None else 'NaN' for k, v in cost_dict.items()}, # Convert keys & handle NaN
        "zipit_analysis": Merge.zipit_analysis # Save the analysis results
    }


    try:
        with open(args_and_results_path, 'w+') as f_out:
             json.dump(results_data, f_out, indent=4)
        print(f"Saved args, costs, and analysis to {args_and_results_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        # Fallback to simple text file?
        fallback_path = f'{save_dir}/{base_exp_tag}.args_results.txt'
        with open(fallback_path, 'w+') as f_args:
            f_args.write(str(vars(args)) + '\n')
            f_args.write(str(data_config) + '\n')
            f_args.write(f"Cost Dict: {json.dumps(results_data['cost_dict'])}\n")
            f_args.write(f"ZipIt Analysis: {json.dumps(results_data['zipit_analysis'])}\n")
        print(f"Saved fallback results to {fallback_path}")
    
    print("\nPerforming interpolation and saving...")
    # Get the state dicts of the *transformed* models ONCE
    transformed_state_dict_hubert = Merge.graphs[0].model.state_dict()
    transformed_state_dict_mert = Merge.graphs[1].model.state_dict()

    for interp_w in interpolation_schemes:
        w1, w2 = interp_w
        print(f"  Interpolating with weights: {interp_w}...")

        # Perform interpolation using the transformed states
        interpolated_state_dict = Merge.interpolate_state_dicts(
            transformed_state_dict_hubert,
            transformed_state_dict_mert,
            interp_w=interp_w,
            use_ties=args.use_ties,
            quantile=args.quantile,
            maintain_hubert_behavior=args.maintain_hubert_behavior
        )

        # Assemble the checkpoint structure
        if args.model_mode == 'distilled':
            # Distilled models have no 'model.' prefix, no need to remove it
            new_checkpoint = assemble_new_checkpoint(
                interpolated_state_dict,
                model_mode='distilled',
                source_config_dict=source_config_dict
            )
        else:
            cleaned_state_dict = remove_model_prefix(interpolated_state_dict)
            new_checkpoint = assemble_new_checkpoint(cleaned_state_dict, model_mode='teacher')

        # Create a specific filename for this interpolation
        param_tail = f'interp_{w1:.1f}_{w2:.1f}' # Format weights in filename
        save_path = f'{save_dir}/merged_{base_exp_tag}_{param_tail}.ckpt'

        print(f"  Saving merged model to: '{save_path}'")
        torch.save(new_checkpoint, save_path)

    print("\nFinished saving all interpolation schemes.")

    print(f"cost in each node for the merging is:")
    print(cost_dict)
    if Merge.zipit_analysis:
        print("\nZipIt Analysis Results:")
        print(json.dumps(Merge.zipit_analysis, indent=2))
    
    Merge.clear_hooks()
    print("Hooks cleared.")
    
    
if __name__ == '__main__':
    # here we can read argparse and then call either main function or experiment1 depending if an optional experiment1 flag is passed.
    # Allow other computers to attach to debugpy on this IP address and port.
    args = get_downstream_args()
    print(f"Running with args: {args}")
    if args.run_correlation_heatmap_viz:
        run_correlation_heatmap_experiment(args)
    elif args.make_permutation_analysis: # Keep your existing analysis
        # The provided main() function has a sys.exit(0) if make_permutation_analysis is true.
        # So, we can keep it as is: main will handle its modes, and this new mode is separate.
         main(args) # This will run the permutation intensity if args.make_permutation_analysis is true
    else:
        main(args) # Default run
