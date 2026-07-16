import torch
import os
from s3prl.upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel
import yaml
import numpy as np
import random
import argparse

def load_hubert_base(model_name="hubert_base"):
    # Load HuBERT base model from s3prl (set TORCH_HOME to control the cache location)
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
    checkpoint = torch.load(model_path, map_location='cpu')
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

def modify_config(config):
    # Modify 'teacher_names' and 'teacher': 'models' to reflect task combination
    if 'multi_distiller' in config and 'teacher_names' in config['multi_distiller']:
        config['multi_distiller']['teacher_names'] = ['hubert_base']
    if 'teacher' in config and 'models' in config['teacher']:
        config['teacher']['models'] = ['hubert_base']
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
    if args.learn_which_modality == "music":
        print(f"giving the speech distilled model music abilities.")
        combined_state_dict = apply_task_vector(theta_h_state_dict, music_vector, weight= args.weight)
    elif args.learn_which_modality == "speech":
        print(f"Giving the music distilled model speech abilities.")
        combined_state_dict = apply_task_vector(theta_m_state_dict, speech_vector, weight= args.weight)
    else:
        print(f"Unrecognized modality: {args.learn_which_modality}")
    
    # Modify config to reflect the new teacher combination
    config = modify_config(theta_m_checkpoint["Config"])
    # Assemble the new checkpoint
    new_checkpoint = assemble_new_checkpoint(combined_state_dict, config, args=None)
    
    # Save the new checkpoint
    torch.save(new_checkpoint, save_path)
    print(f"Task arithmetic model checkpoint saved to {save_path}")

if __name__ == "__main__":
    # Fix seed and make backends deterministic
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--weight', default=1, type=float, help="the weight to add for the second task vector.")
    parser.add_argument('--learn_which_modality', choices=['music', 'speech'] ,default="speech", type=str, help="what to learn: either speech or music.")
    args = parser.parse_args()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Paths to your two models for Theta M and Theta H
    model_paths = [
        'result/pretrain/distill_only_mert-init-weight-from-hubert_base-models-simple-avg-pool-for-teacher-train-libri-960/states-epoch-25.ckpt',  # Theta M
        'result/pretrain/distilhubert-ls960-own/states-epoch-25.ckpt',  # Theta H
    ]


    # Path to the base (Theta 0) model
    #base_model_path = 'path/to/initial_hubert_base_model.ckpt' ## missing this part..........
    # Path to save the combined model
    save_path = f'result/pretrain/task_vector_dhubert_ls960_and_mert_ls960_both_init_hubert_{args.learn_which_modality}_tsv_weight_{args.weight}/learning_by_addition.ckpt'
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    # Run task arithmetic
    task_arithmetic(args, model_paths, save_path)
