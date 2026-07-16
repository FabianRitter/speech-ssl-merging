"""
Merge two wide distilled models using task arithmetic.

Uses theta_base_wide.pt (saved by create_wide_base_model.py) as the base model,
instead of recreating it from HuBERT (which would fail due to dimension mismatch).

Usage:
    # Merge L1 wide models at all weight combinations:
    python merge_wide_models.py --loss_type l1

    # Merge Barlow wide models at all weight combinations:
    python merge_wide_models.py --loss_type barlow

    # Merge at a specific weight combination:
    python merge_wide_models.py --loss_type l1 --lambda1 0.5 --lambda2 0.5

    # Custom checkpoint paths:
    python merge_wide_models.py \
        --speech_ckpt result/pretrain/hubert_l1_wide/states-300000.ckpt \
        --music_ckpt result/pretrain/mert_l1_wide/states-300000.ckpt \
        --base_state result/pretrain/theta_base_wide/theta_base_wide.pt \
        --lambda1 0.5 --lambda2 0.5
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel


# --- Reuse core functions from merge_by_addition-both-weighting.py ---

class TIESMerging:
    def __init__(self, k=None, enable_sign_interference=False, sign_as_in_ties=False):
        self.k = k
        self.enable_sign_interference = enable_sign_interference
        self.sign_as_in_ties = sign_as_in_ties

    def keep_topk_reset_rest_to_zero(self, tensor, k):
        if k is None or k >= 1.0:
            return tensor
        num_elements_to_keep = int(k * tensor.numel())
        if num_elements_to_keep == 0:
            return torch.zeros_like(tensor)
        topk_values, _ = torch.topk(tensor.abs().flatten(), num_elements_to_keep)
        threshold = topk_values[-1]
        tensor = torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
        return tensor

    def resolve_sign_conflicts(self, tensor_1, tensor_2):
        global_sign = torch.sign(tensor_1 + tensor_2)
        resolved_tensor = torch.where(
            torch.sign(tensor_2) != global_sign,
            -tensor_2,
            tensor_2
        )
        return resolved_tensor

    def sign_interference_check(self, tensor_1, tensor_2):
        return torch.sign(tensor_1) != torch.sign(tensor_2)

    def merge(self, base_state_dict, model_speech, model_music, speech_weight, music_weight):
        merged_model = {}
        if speech_weight > music_weight:
            main_model, secondary_model = model_speech, model_music
        else:
            main_model, secondary_model = model_music, model_speech

        for key in base_state_dict.keys():
            base_param = base_state_dict[key]
            main_param = main_model.get(key, torch.zeros_like(base_param))
            secondary_param = secondary_model.get(key, torch.zeros_like(base_param))
            pruned_secondary_param = self.keep_topk_reset_rest_to_zero(secondary_param, self.k)
            if self.enable_sign_interference and not self.sign_as_in_ties:
                sign_interference = self.sign_interference_check(main_param, pruned_secondary_param)
                pruned_secondary_param = torch.where(
                    sign_interference, torch.zeros_like(pruned_secondary_param), pruned_secondary_param
                )
            if self.enable_sign_interference and self.sign_as_in_ties:
                pruned_secondary_param = self.resolve_sign_conflicts(main_param, pruned_secondary_param)
            merged_param = base_param + speech_weight * main_param + music_weight * pruned_secondary_param
            merged_model[key] = merged_param
        return merged_model


def compute_task_vector(base_state_dict, target_state_dict):
    task_vector = {}
    for key in base_state_dict.keys():
        if key in target_state_dict:
            if base_state_dict[key].shape == target_state_dict[key].shape:
                task_vector[key] = target_state_dict[key] - base_state_dict[key]
            else:
                print(f"Skipping {key} due to shape mismatch: "
                      f"{base_state_dict[key].shape} vs {target_state_dict[key].shape}")
    return task_vector


def apply_task_vector(base_state_dict, task_vector, weight=1):
    combined_state_dict = {}
    for key in base_state_dict.keys():
        combined_state_dict[key] = base_state_dict[key] + weight * task_vector.get(key, torch.zeros_like(base_state_dict[key]))
    return combined_state_dict


def assemble_new_checkpoint(state_dict, config, args=None):
    new_checkpoint = {}
    new_checkpoint['Distiller'] = state_dict
    new_checkpoint['Config'] = config
    new_checkpoint['Optimizer'] = None
    new_checkpoint['Step'] = 0
    if args is not None:
        new_checkpoint['Args'] = args
    return new_checkpoint


def build_wide_config():
    """Build the config dict for the wide merged model (used at downstream inference)."""
    config = {
        'multi_distiller': {
            'extractor_mode': 'default',
            'extractor_conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2',
            'extractor_dropout': 0.0,
            'feature_grad_mult': 0.1,
            'conv_pos': 128,
            'conv_pos_groups': 16,
            'encoder_layers': 2,
            'encoder_embed_dim': 1536,
            'encoder_ffn_embed_dim': 3072,
            'encoder_attention_heads': 6,
            'activation_fn': 'gelu',
            'layer_norm_first': False,
            'attention_type': 'original',
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'activation_dropout': 0.1,
            'encoder_layerdrop': 0.0,
            'final_dim': 768,
            'out_layer_type': 'expand-last',
            'n_tasks': 3,
            'task_emb_type': 'expand-last',
            'loss_type': 'l1',
            'feat_pen_loss': 0.0,
            'cosine_loss': 1.0,
            'pred_layer_id': [4, 8, 12],
            'init_teacher_conv_layers': False,
            'init_teacher_encoder_layers': False,
            'teacher_names': ['hubert_base'],
            'initialize_from': ['hubert_base'],
            'use_feat_translator': False,
            'translator_type': 'avgpool',
            'translator_kwargs': {'hidden_size_factor': 1.0},
        },
        'teacher': {
            'models': ['hubert_base'],
            'n_layers': 12
        },
        'task': {
            'sequence_length': 250000
        },
        'audio': {
            'target_level': None
        }
    }
    return config


def merge_at_weights(base_state_dict, speech_vector, music_vector, lambda1, lambda2,
                     config, save_dir, loss_type, use_ties=False, k=0.3,
                     enable_sign_interference=False, sign_as_in_ties=False):
    """Merge at a specific weight combination and save."""
    print(f"\n--- Merging: speech={lambda1}, music={lambda2} ---")

    if use_ties:
        merger = TIESMerging(k=k, enable_sign_interference=enable_sign_interference,
                             sign_as_in_ties=sign_as_in_ties)
        combined = merger.merge(base_state_dict, speech_vector, music_vector, lambda1, lambda2)
    else:
        combined = apply_task_vector(base_state_dict, speech_vector, lambda1)
        combined = apply_task_vector(combined, music_vector, lambda2)

    save_name = (f"task_vector_wide_{loss_type}_hubert_weight_{lambda1}"
                 f"_mert_weight_{lambda2}")
    save_path = os.path.join(save_dir, save_name, "learning_by_addition.ckpt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_checkpoint = assemble_new_checkpoint(combined, config)
    torch.save(new_checkpoint, save_path)
    print(f"Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Merge wide distilled models via task arithmetic")
    parser.add_argument('--loss_type', choices=['l1', 'barlow'], default=None,
                        help="Loss type preset (auto-fills checkpoint paths)")
    parser.add_argument('--speech_ckpt', default=None, help="Path to speech (HuBERT) model checkpoint")
    parser.add_argument('--music_ckpt', default=None, help="Path to music (MERT) model checkpoint")
    parser.add_argument('--base_state', default='result/pretrain/theta_base_wide/theta_base_wide.pt',
                        help="Path to theta_base_wide.pt")
    parser.add_argument('--save_dir', default='result/pretrain',
                        help="Directory to save merged checkpoints")
    parser.add_argument('--lambda1', type=float, default=None,
                        help="Weight for speech task vector (if not set, runs all 5 combinations)")
    parser.add_argument('--lambda2', type=float, default=None,
                        help="Weight for music task vector")
    parser.add_argument('--speech_step', type=int, default=300000,
                        help="Training step of the speech checkpoint to load")
    parser.add_argument('--music_step', type=int, default=300000,
                        help="Training step of the music checkpoint to load")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--use_ties', action='store_true')
    parser.add_argument('--k', type=float, default=0.3)
    parser.add_argument('--enable_sign_interference', action='store_true')
    parser.add_argument('--sign_as_in_ties', action='store_true')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve checkpoint paths from loss_type preset
    if args.loss_type and not args.speech_ckpt:
        if args.loss_type == 'l1':
            args.speech_ckpt = f'result/pretrain/hubert_l1_wide/states-{args.speech_step}.ckpt'
            args.music_ckpt = f'result/pretrain/mert_l1_wide/states-{args.music_step}.ckpt'
        elif args.loss_type == 'barlow':
            args.speech_ckpt = f'result/pretrain/hubert_barlow_wide/states-{args.speech_step}.ckpt'
            args.music_ckpt = f'result/pretrain/mert_barlow_wide/states-{args.music_step}.ckpt'

    if not args.speech_ckpt or not args.music_ckpt:
        parser.error("Must provide --loss_type or both --speech_ckpt and --music_ckpt")

    loss_label = args.loss_type or "custom"

    print(f"=== Wide Model Task Arithmetic Merge ({loss_label}) ===")
    print(f"Base state:  {args.base_state}")
    print(f"Speech ckpt: {args.speech_ckpt}")
    print(f"Music ckpt:  {args.music_ckpt}")

    # Load base state dict (theta_base_wide)
    print("\nLoading theta_base_wide...")
    base_state_dict = torch.load(args.base_state, map_location='cpu')
    print(f"Base state dict keys: {len(base_state_dict)}")

    # Load trained checkpoints
    print("Loading speech checkpoint...")
    speech_ckpt = torch.load(args.speech_ckpt, map_location='cpu')
    speech_state_dict = speech_ckpt['Distiller']

    print("Loading music checkpoint...")
    music_ckpt = torch.load(args.music_ckpt, map_location='cpu')
    music_state_dict = music_ckpt['Distiller']

    # Compute task vectors
    print("\nComputing task vectors...")
    speech_vector = compute_task_vector(base_state_dict, speech_state_dict)
    music_vector = compute_task_vector(base_state_dict, music_state_dict)
    print(f"Speech vector keys: {len(speech_vector)}, Music vector keys: {len(music_vector)}")

    # Build config for merged checkpoint
    config = build_wide_config()

    # Weight combinations
    if args.lambda1 is not None and args.lambda2 is not None:
        weight_combos = [(args.lambda1, args.lambda2)]
    else:
        weight_combos = [
            (0.9, 0.1),
            (0.8, 0.2),
            (0.5, 0.5),
            (0.2, 0.8),
            (0.1, 0.9),
        ]

    # Merge at each weight combination
    saved_paths = []
    for l1, l2 in weight_combos:
        path = merge_at_weights(
            base_state_dict, speech_vector, music_vector,
            l1, l2, config, args.save_dir, loss_label,
            use_ties=args.use_ties, k=args.k,
            enable_sign_interference=args.enable_sign_interference,
            sign_as_in_ties=args.sign_as_in_ties
        )
        saved_paths.append(path)

    print(f"\n=== Done! Saved {len(saved_paths)} merged checkpoints ===")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
