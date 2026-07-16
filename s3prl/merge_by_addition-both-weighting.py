import torch
import os
from s3prl.upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel
import yaml
import numpy as np
import random
import argparse


class TIESMerging:
    def __init__(self, k=None, enable_sign_interference=False, sign_as_in_ties=False):
        self.k = k
        self.enable_sign_interference = enable_sign_interference
        self.sign_as_in_ties = sign_as_in_ties

    def keep_topk_reset_rest_to_zero(self, tensor, k):
        if k is None or k >= 1.0:
            return tensor

        # Calculate the number of elements to keep based on the percentage
        num_elements_to_keep = int(k * tensor.numel())
        if num_elements_to_keep == 0:
            return torch.zeros_like(tensor)

        topk_values, _ = torch.topk(tensor.abs().flatten(), num_elements_to_keep)
        threshold = topk_values[-1]
        tensor = torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
        return tensor

    def sign_interference_check(self, tensor_1, tensor_2):
        return torch.sign(tensor_1) != torch.sign(tensor_2)
    
    def resolve_sign_conflicts(self, tensor_1, tensor_2):
        # Compute global sign vector as the sum of weighted signs
        global_sign = torch.sign(tensor_1 + tensor_2)
        # Align the secondary tensor's signs to the global sign
        resolved_tensor = torch.where(
            torch.sign(tensor_2) != global_sign,
            -tensor_2,  # Flip the sign to align with the global sign
            tensor_2
        )
        return resolved_tensor

    def merge(self, base_state_dict, model_speech, model_music, speech_weight, music_weight):
        assert 0 <= speech_weight <= 1, "Speech weight must be between 0 and 1"
        assert 0 <= music_weight <= 1, "Music weight must be between 0 and 1"
        #assert speech_weight + music_weight == 1, "Speech and music weights must sum to 1"

        merged_model = {}

        # Determine main model based on the larger weight
        if speech_weight > music_weight:
            main_model = model_speech
            secondary_model = model_music
        else:
            main_model = model_music
            secondary_model = model_speech

        for key in base_state_dict.keys():
            base_param = base_state_dict[key]
            main_param = main_model.get(key, torch.zeros_like(base_param))
            secondary_param = secondary_model.get(key, torch.zeros_like(base_param))

            # Step 1: Prune secondary model weights
            pruned_secondary_param = self.keep_topk_reset_rest_to_zero(secondary_param, self.k)

            # Optional: Step 2 - Check for sign interference
            if self.enable_sign_interference and not self.sign_as_in_ties:
                sign_interference = self.sign_interference_check(main_param, pruned_secondary_param)
                pruned_secondary_param = torch.where(
                    sign_interference,
                    torch.zeros_like(pruned_secondary_param),
                    pruned_secondary_param
                )
            if self.enable_sign_interference and self.sign_as_in_ties:
                pruned_secondary_param = self.resolve_sign_conflicts(main_param, pruned_secondary_param)

            # Merge parameters
            merged_param = (
                base_param + 
                speech_weight * main_param + 
                music_weight * pruned_secondary_param
            )

            merged_model[key] = merged_param

        return merged_model



def load_hubert_base(model_name="hubert_base"):
    # Load HuBERT base model from s3prl
    base_model = torch.hub.load("s3prl/s3prl", model_name).cuda()
    base_model.model.encoder.layerdrop = 0  # Ensure no dropout in encoder layers
    return base_model

def initialize_distiller_from_hubert(distiller_model, hubert_model):
    """Initialize a distiller model with weights from a HuBERT model."""
    
    # Initialize feature extractor from HuBERT model
    distiller_model.feature_extractor.load_state_dict(hubert_model.model.feature_extractor.state_dict())
    print("[Distiller Initialization] Feature extractor initialized from HuBERT")

    # Initialize post-extract projection if it exists
    if hasattr(distiller_model, 'post_extract_proj') and distiller_model.post_extract_proj is not None:
        distiller_model.post_extract_proj.load_state_dict(hubert_model.model.post_extract_proj.state_dict())
        print("[Distiller Initialization] Post-extract projection initialized from HuBERT")
    else:
        print("[Distiller Initialization] Post-extract projection not loaded SOMETHING IS WRONG!")
        exit

    # Initialize encoder layers
    distiller_model.encoder.pos_conv.load_state_dict(hubert_model.model.encoder.pos_conv.state_dict())
    print("[Distiller Initialization] Positional convolution initialized from HuBERT")

    for l in range(len(distiller_model.encoder.layers)):
        distiller_model.encoder.layers[l].load_state_dict(hubert_model.model.encoder.layers[l].state_dict())
        print(f"[Distiller Initialization] Encoder layer {l} initialized from HuBERT")

    return distiller_model

def create_theta_0(distiller_model, save_path, hubert_model_name="hubert_base"):
    # Load HuBERT base model as source of initial weights
    hubert_model = load_hubert_base(model_name=hubert_model_name)

    # Initialize the distiller model with HuBERT weights
    initialized_distiller = initialize_distiller_from_hubert(distiller_model, hubert_model)

    return initialized_distiller

def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint

def compute_task_vector(base_state_dict, target_state_dict):
    """Compute the task vector as the difference between target and base model."""
    task_vector = {}
    for key in base_state_dict.keys():
        if key in target_state_dict:
            # Ensure tensors are the same size before subtraction
            if base_state_dict[key].shape == target_state_dict[key].shape:
                task_vector[key] = target_state_dict[key] - base_state_dict[key]
            else:
                print(f"Skipping {key} due to shape mismatch: "
                      f"{base_state_dict[key].shape} vs {target_state_dict[key].shape}")
    return task_vector

def apply_task_vector(base_state_dict, task_vector, weight=1):
    """Apply a task vector to a base state dict by adding the vector."""
    combined_state_dict = {}
    for key in base_state_dict.keys():
        combined_state_dict[key] = base_state_dict[key] + weight * task_vector.get(key, torch.zeros_like(base_state_dict[key]))
    return combined_state_dict

# def modify_config(config):
#     # Modify 'teacher_names' and 'teacher': 'models' to reflect task combination
#     if 'multi_distiller' in config and 'teacher_names' in config['multi_distiller']:
#         config['multi_distiller']['teacher_names'] = ['hubert_base']
#     else:
#         print(f"multi_distiller not in cofig file, something happened , please check! config is {config}")
#     if 'teacher' in config and 'models' in config['teacher']:
#         config['teacher']['models'] = ['hubert_base']
#     return config

def modify_config(config):
    # Check for 'multi_distiller' and modify its content
    if 'multi_distiller' in config:
        if 'teacher_names' in config['multi_distiller']:
            config['multi_distiller']['teacher_names'] = ['hubert_base']
        else:
            config['multi_distiller']['teacher_names'] = ['hubert_base']
    elif 'distiller' in config:
        # Rename 'distiller' to 'multi_distiller' and keep its content
        config['multi_distiller'] = config.pop('distiller')
        config['multi_distiller']['teacher_names'] = ['hubert_base']
    else:
        print(f"Neither 'multi_distiller' nor 'distiller' found in config! Please check the config structure. Config: {config}")

    # Modify or add the 'teacher' key
    if 'teacher' in config:
        if 'models' in config['teacher']:
            config['teacher']['models'] = ['hubert_base']
        else:
            config['teacher']['models'] = ['hubert_base']
    else:
        config['teacher'] = {'models': ['hubert_base']}
    
    # Add 'initialize_from' key to 'multi_distiller'
    if 'multi_distiller' in config:
        config['multi_distiller']['initialize_from'] = ['hubert_base']

    return config


def assemble_new_checkpoint(state_dict, config, args=None):
    new_checkpoint = {}
    new_checkpoint['Distiller'] = state_dict
    new_checkpoint['Config'] = config
    new_checkpoint['Optimizer'] = None
    new_checkpoint['Step'] = 0
    if args is not None:
        new_checkpoint['Args'] = args
    return new_checkpoint

def task_arithmetic(args, model_paths, save_path):
    # Load base (Theta 0) model
    
    # Load Theta M and Theta H models
    theta_m_checkpoint = load_checkpoint(model_paths[0])
    theta_m_state_dict = theta_m_checkpoint['Distiller']
    
    theta_h_checkpoint = load_checkpoint(model_paths[1])
    theta_h_state_dict = theta_h_checkpoint['Distiller']
    config = theta_h_checkpoint['Config']
    #import pdb
    #pdb.set_trace()
    ##### base model from HuBERT in this case.
    hubert_model = load_hubert_base()
    
    # this is bad coding practivce and I should give this as an argument.
    upstream_config = yaml.load(open("./pretrain/multi_distiller/config_model.yaml","r"),Loader=yaml.FullLoader)
    
    datarc = {
    "num_workers": 24,
    "train_batch_size": 24,  # The train batch size is too small... this will make it ultra noisy
    "dev_batch_size": 24,
    "max_timestep": 0,
    "libri_root": os.environ.get("S3PRL_LIBRI_ROOT", "/path/to/LibriSpeech"),
    "file_path": os.environ.get("S3PRL_BUCKET_PATH", "./data/len_for_bucket"),
    "sets": ["train-clean-100"],  # , 'train-clean-360', 'train-other-500'
    "devsets": ["dev-clean"],
    "data_stats": {  # Add new fields for mean and std
        "fbank_mean": [-8.202537669959721],
        "fbank_std": [4.238643955336016],
        "wav_mean": [8.7623e-13],
        "wav_std": [0.0608]
    }
    }
    #import pdb
    #pdb.set_trace()   
    
    model_config = MultiDistillerConfig(upstream_config["multi_distiller"],**datarc)
    distiller_model = MultiDistillerModel(model_config)
    distiller_model = initialize_distiller_from_hubert(distiller_model, hubert_model)
    base_state_dict = distiller_model.state_dict()
    
    
    # Compute task vectors
    music_vector = compute_task_vector(base_state_dict, theta_m_state_dict)
    speech_vector = compute_task_vector(base_state_dict, theta_h_state_dict)
    
    # Combine task vectors with base model
    if args.use_ties:
        merger = TIESMerging(k=args.k, enable_sign_interference=args.enable_sign_interference, sign_as_in_ties = args.sign_as_in_ties)
        combined_state_dict = merger.merge(base_state_dict, speech_vector, music_vector, args.lambda1, args.lambda2)
    else:
        combined_state_dict = apply_task_vector(base_state_dict, speech_vector, args.lambda1)
        combined_state_dict = apply_task_vector(combined_state_dict, music_vector, args.lambda2)

    
    # Modify config to reflect the new teacher combination
    config = modify_config(theta_m_checkpoint["Config"])

    print(f"checking this config:  {config}")
    # Assemble the new checkpoint
    new_checkpoint = assemble_new_checkpoint(combined_state_dict, config, args=None)
    
    # Save the new checkpoint
    torch.save(new_checkpoint, save_path)
    print(f"Task arithmetic model checkpoint saved to {save_path}")

if __name__ == "__main__":
    # Fix seed and make backends deterministic
    # run example: python merge_by_addition-both-weighting.py --lambda1 0.9 --lambda2 0.1
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--lambda1', default=0.5, type=float, help="Weight for the speech task vector.")
    parser.add_argument('--lambda2', default=0.5, type=float, help="Weight for the music task vector.")
    parser.add_argument('--k', default=0.3, type=float, help="Percentage of top weights to keep in TIES merging.")
    parser.add_argument('--enable_sign_interference', action='store_true', help="Enable sign interference check and handling in TIES merging.")
    parser.add_argument('--sign_as_in_ties', action='store_true', help="Enable sign global resolution as in TIES.")
    parser.add_argument('--use_ties', action='store_true', help="Enable sign interference check and handling in TIES merging.")
    args = parser.parse_args()
    args = parser.parse_args()

    #assert args.lambda1 + args.lambda2 == 1, "Error: lambda1 and lambda2 must sum to 1!"
    print(f"using lambdas: speech: {args.lambda1}  music: {args.lambda2}")

    if args.use_ties:
        print(f"USING TIES MERGING STRATEGY WITH ENABLE SIGN INTERFERENCE SET UP AS {args.enable_sign_interference}")

    
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Paths to your two models for Theta M and Theta H
    model_paths = [
        'result/pretrain/distilmert_music4all_and_ls960_2layers/states-epoch-18.ckpt',  # MUSIC
        'result/pretrain/distilhubert_music4all_and_ls960_2layers/states-epoch-18.ckpt',  # SPEECH
    ]

    # result/pretrain/DistilHuBERT_100hrs_libri_l1_cos/dev-dis-best-168-epoch-exp.ckpt
    # distilhubert-init-hubert-libri-960/distilhubert_ls960_4-8-12.ckpt
    # distilhubert-ls960-own
    # distill_only_mert-init-weight-from-hubert_base-models-simple-avg-pool-for-teacher-train-libri-960

    # Path to the base (Theta 0) model
    #base_model_path = 'path/to/initial_hubert_base_model.ckpt' ## missing this part..........
    # Path to save the combined model
    if args.use_ties:
        save_path = f'result/pretrain/task_vector_dhubert_ls_960_weight_{args.lambda1}_and_mert_ls_960_weight_{args.lambda2}_both_init_hubert_both_same_seed_{args.use_ties}_ties_weight_{args.k}_enable_sign_interference_{args.enable_sign_interference}_as_ties_{args.sign_as_in_ties}/learning_by_addition.ckpt'
    else:
        save_path = f'result/pretrain/task_vector_dhubert_ls_960_and_music4all_weight_{args.lambda1}_and_mert_ls_960_and_music4all_weight_{args.lambda2}_both_init_hubert_both_same_seed/learning_by_addition.ckpt'

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    # Run task arithmetic
    task_arithmetic(args, model_paths, save_path)

