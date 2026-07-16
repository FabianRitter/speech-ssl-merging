import torch
import os

def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint

def average_distiller_state_dicts(checkpoints):
    distiller_state_dicts = [ckpt['Distiller'] for ckpt in checkpoints]
    param_keys = [set(state_dict.keys()) for state_dict in distiller_state_dicts]
    if not all(keys == param_keys[0] for keys in param_keys):
        raise ValueError("Distiller state dicts have different parameter keys!")

    averaged_state_dict = {}
    for key in param_keys[0]:
        params = torch.stack([state_dict[key] for state_dict in distiller_state_dicts], dim=0)
        averaged_param = params.mean(dim=0)
        averaged_state_dict[key] = averaged_param
    return averaged_state_dict

def modify_config(config):
    # Modify 'teacher_names' to ['hubert_base']
    if 'multi_distiller' in config and 'teacher_names' in config['multi_distiller']:
        config['multi_distiller']['teacher_names'] = ['hubert_base']
    else:
        print("Warning: 'teacher_names' not found in 'multi_distiller' config.")

    # Modify 'teacher': 'models' to ['hubert_base']
    if 'teacher' in config and 'models' in config['teacher']:
        config['teacher']['models'] = ['hubert_base']
    else:
        print("Warning: 'models' not found in 'teacher' config.")

    return config

def assemble_new_checkpoint(averaged_state_dict, config, args=None):
    new_checkpoint = {}
    new_checkpoint['Distiller'] = averaged_state_dict
    new_checkpoint['Config'] = config
    new_checkpoint['Optimizer'] = None  # Set to None or omit if not needed
    new_checkpoint['Step'] = 0  # Reset step to 0
    if args is not None:
        new_checkpoint['Args'] = args
    return new_checkpoint

def average_models(model_paths, save_path):
    checkpoints = [load_checkpoint(path) for path in model_paths]
    # Average the 'Distiller' state dicts
    averaged_state_dict = average_distiller_state_dicts(checkpoints)

    # Use the 'Config' from the first checkpoint and modify it
    config = checkpoints[0]['Config']
    config = modify_config(config)

    # Optionally, collect 'Args' from the first checkpoint
    args = checkpoints[0].get('Args', None)

    # Assemble the new checkpoint
    new_checkpoint = assemble_new_checkpoint(averaged_state_dict, config, args=args)

    # Save the new checkpoint
    torch.save(new_checkpoint, save_path)
    print(f"Averaged model checkpoint saved to {save_path}")


if __name__ == "__main__":
    # Paths to your three distilled models
    model_paths = [
        'result/pretrain/distill_mert_init_mert_music4all_avgpool/states-epoch-65.ckpt',
        'result/pretrain/distilhubert-init-hubert-libri-960/distilhubert_ls960_4-8-12.ckpt',
    ]
    #         'result/pretrain/distill_only_ssast-legit-ssast-init-weight-from-hubert-simple-avg-pool-for-teacher-train-libri-960/states-260000.ckpt',

    # Path to save the averaged model
    save_path = 'result/pretrain/mert-from-music4all-and-mert-init-distilhubert-from-hubert-and-ls960/averaged_model.ckpt'

    # Ensure all model paths exist
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

    average_models(model_paths, save_path)
