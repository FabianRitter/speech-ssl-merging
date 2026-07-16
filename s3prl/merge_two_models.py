"""
Merge two distilled models using task arithmetic.
Wrapper around merge_by_addition-both-weighting.py that accepts model paths as arguments.

The key difference from calling merge_by_addition-both-weighting.py directly is that
this wrapper preserves the base model's teacher_names in the config so the merged
checkpoint's state_dict keys match the model architecture at load time.

Usage:
    python merge_two_models.py \
        --music_ckpt result/pretrain/MODEL_MUSIC/states-220000.ckpt \
        --speech_ckpt result/pretrain/MODEL_SPEECH/states-220000.ckpt \
        --save_path result/pretrain/merged_model_name/learning_by_addition.ckpt \
        --lambda1 0.9 --lambda2 0.1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random
import yaml

from s3prl.upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel


def modify_config_preserve_teachers(config, base_config_path):
    """
    Modify config for the merged checkpoint, preserving teacher_names
    from the base config so the model architecture matches the state_dict.
    """
    base_config = yaml.load(open(base_config_path, "r"), Loader=yaml.FullLoader)
    base_teacher_names = base_config["multi_distiller"]["teacher_names"]
    base_teacher_models = base_config["teacher"]["models"]

    # Ensure config has multi_distiller key
    if 'distiller' in config and 'multi_distiller' not in config:
        config['multi_distiller'] = config.pop('distiller')

    # Preserve teacher_names from base config (matches state_dict keys)
    config['multi_distiller']['teacher_names'] = base_teacher_names
    config['multi_distiller']['initialize_from'] = ['hubert_base']

    # Preserve teacher models from base config
    config['teacher'] = {'models': base_teacher_models, 'n_layers': 12}

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--music_ckpt', required=True, help="Path to music model checkpoint")
    parser.add_argument('--speech_ckpt', required=True, help="Path to speech model checkpoint")
    parser.add_argument('--save_path', required=True, help="Path to save merged checkpoint")
    parser.add_argument('--lambda1', type=float, required=True, help="Weight for speech task vector")
    parser.add_argument('--lambda2', type=float, required=True, help="Weight for music task vector")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--use_ties', action='store_true')
    parser.add_argument('--k', type=float, default=0.3)
    parser.add_argument('--enable_sign_interference', action='store_true')
    parser.add_argument('--sign_as_in_ties', action='store_true')
    parser.add_argument('--base_config', default=None,
                        help="Path to base model config. Default: pretrain/multi_distiller/config_model.yaml")
    args = parser.parse_args()

    # Fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import functions from the original merge script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "merge_module",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_by_addition-both-weighting.py")
    )
    merge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge_module)

    # Resolve base config path
    base_config_path = args.base_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pretrain", "multi_distiller", "config_model.yaml"
    )

    model_paths = [args.music_ckpt, args.speech_ckpt]

    # Create output directory
    directory = os.path.dirname(args.save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    print(f"Using lambdas: speech={args.lambda1}, music={args.lambda2}")
    print(f"Music ckpt:  {args.music_ckpt}")
    print(f"Speech ckpt: {args.speech_ckpt}")
    print(f"Base config: {base_config_path}")
    print(f"Save path:   {args.save_path}")

    # Run the merge (task_arithmetic internally calls modify_config, but we'll
    # override the config in the saved checkpoint afterwards)
    # Load checkpoints manually to control config
    theta_m_checkpoint = merge_module.load_checkpoint(model_paths[0])
    theta_m_state_dict = theta_m_checkpoint['Distiller']

    theta_h_checkpoint = merge_module.load_checkpoint(model_paths[1])
    theta_h_state_dict = theta_h_checkpoint['Distiller']

    # Build base model using the base config (same as task_arithmetic does)
    hubert_model = merge_module.load_hubert_base()
    upstream_config = yaml.load(open(base_config_path, "r"), Loader=yaml.FullLoader)

    datarc = {
        "num_workers": 24,
        "train_batch_size": 24,
        "dev_batch_size": 24,
        "max_timestep": 0,
        "libri_root": os.environ.get("S3PRL_LIBRI_ROOT", "/path/to/LibriSpeech"),
        "file_path": os.environ.get("S3PRL_BUCKET_PATH", "./data/len_for_bucket"),
        "sets": ["train-clean-100"],
        "devsets": ["dev-clean"],
        "data_stats": {
            "fbank_mean": [-8.202537669959721],
            "fbank_std": [4.238643955336016],
            "wav_mean": [8.7623e-13],
            "wav_std": [0.0608]
        }
    }

    model_config = MultiDistillerConfig(upstream_config["multi_distiller"], **datarc)
    distiller_model = MultiDistillerModel(model_config)
    distiller_model = merge_module.initialize_distiller_from_hubert(distiller_model, hubert_model)
    base_state_dict = distiller_model.state_dict()

    # Compute task vectors
    music_vector = merge_module.compute_task_vector(base_state_dict, theta_m_state_dict)
    speech_vector = merge_module.compute_task_vector(base_state_dict, theta_h_state_dict)

    # Combine task vectors
    if args.use_ties:
        merger = merge_module.TIESMerging(
            k=args.k,
            enable_sign_interference=args.enable_sign_interference,
            sign_as_in_ties=args.sign_as_in_ties
        )
        combined_state_dict = merger.merge(
            base_state_dict, speech_vector, music_vector, args.lambda1, args.lambda2
        )
    else:
        combined_state_dict = merge_module.apply_task_vector(base_state_dict, speech_vector, args.lambda1)
        combined_state_dict = merge_module.apply_task_vector(combined_state_dict, music_vector, args.lambda2)

    # Use our fixed config modifier that preserves teacher_names
    config = modify_config_preserve_teachers(theta_m_checkpoint["Config"], base_config_path)
    print(f"Config teacher_names: {config['multi_distiller']['teacher_names']}")

    # Assemble and save checkpoint
    new_checkpoint = merge_module.assemble_new_checkpoint(combined_state_dict, config, args=None)
    torch.save(new_checkpoint, args.save_path)
    print(f"Task arithmetic model checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
