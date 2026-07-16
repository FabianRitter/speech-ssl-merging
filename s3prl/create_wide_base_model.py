"""
Create and save the base model state dict for wide model task arithmetic.

Both HuBERT and MERT distillation experiments MUST use this same base to ensure
identical initialization, which is required for valid task vector computation.

Usage:
    conda activate s3prl_multidistiller
    python create_wide_base_model.py

This saves: result/pretrain/theta_base_wide/theta_base_wide.pt
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel


def main():
    # Wide model architecture config (shared by all 4 experiments)
    config_dict = {
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
    }

    config = MultiDistillerConfig(config_dict)

    # Create model with random init
    print("Creating wide MultiDistillerModel (embed_dim=1536, heads=6, FFN=3072, 2 layers)...")
    model = MultiDistillerModel(config)

    # Load HuBERT conv layers (512-dim, compatible with wide model)
    print("Loading HuBERT base for conv layer initialization...")
    hubert = torch.hub.load("s3prl/s3prl", "hubert_base")
    model.feature_extractor.load_state_dict(
        hubert.model.feature_extractor.state_dict()
    )
    print("Loaded HuBERT conv layers into wide model")

    # Save base state dict
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result", "pretrain", "theta_base_wide")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "theta_base_wide.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved theta_base_wide to: {save_path}")

    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Verify the state dict can be loaded back
    model2 = MultiDistillerModel(config)
    model2.load_state_dict(torch.load(save_path, map_location='cpu'))
    print("Verification: state dict loaded back successfully")


if __name__ == "__main__":
    main()
