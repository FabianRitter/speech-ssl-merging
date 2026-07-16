import os
import argparse
import torch
from tqdm import tqdm
from s3prl import hub
from s3prl.util.audio_utils import load_audio, find_audios
import yaml
from s3prl.downstream.runner import ModelEntry

def init_model(model, name, trainable, interfaces=None):
    for interface in interfaces or []:
        assert hasattr(model, interface), interface
    return ModelEntry(model, name, trainable, interfaces)

def get_upstream(args):
    Upstream = getattr(hub, args.upstream)
    if args.local_path:
        model = Upstream(args.local_path)
    else:
        model = Upstream()
    return init_model(
        model = model,
        name = 'Upstream',
        trainable = False,
        interfaces = ["get_downsample_rates"]
    )

def extend_args(config, args):
    with open(config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    args.overwrite = config['downstream_expert']['datarc']['pre_extract'].get('overwrite', False)
    args.audio_dir = config['downstream_expert']['datarc']['pre_extract']['audio_dir']
    args.keep_folder_structure = config['downstream_expert']['datarc']['pre_extract'].get('keep_folder_structure', True)
    args.is_mono = config['downstream_expert']['datarc']['pre_extract']['audio_loader'].get('is_mono', True)
    args.is_normalize = config['downstream_expert']['datarc']['pre_extract']['audio_loader'].get('is_normalize', False)
    args.crop_to_length_in_sec = config['downstream_expert']['datarc']['pre_extract']['audio_loader'].get('crop_to_length_in_sec', None)
    args.sliding_window_size_in_sec = config['downstream_expert']['datarc']['pre_extract']['audio_loader'].get('sliding_window_size_in_sec', None)
    args.sliding_window_overlap_in_percent = config['downstream_expert']['datarc']['pre_extract']['audio_loader'].get('sliding_window_overlap_in_percent', 0)
    args.reduction = config['downstream_expert']['datarc']['pre_extract']['feature_extractor'].get('reduction', False)
    return args

def process_audio_files(audio_files, FeatureExtractor, args, device):
    FeatureExtractor.model.to(device)
    FeatureExtractor.model.eval()

    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            valid_files = []

            # Load and preprocess the audio
            try:
                waveform = load_audio(
                    audio_file,
                    target_sr=args.target_sr,
                    is_mono=args.is_mono,
                    is_normalize=args.is_normalize,
                    crop_to_length_in_sec=args.crop_to_length_in_sec,
                    device=device,
                ).squeeze(0)
                valid_files.append(audio_file)
            except Exception as e:
                print(f"skip audio {audio_file} because of {e}")
                continue

            # Pad the waveform and move it to the device
            waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
            # waveform = torch.nn.functional.pad(waveform, (0, 0))

            all_features_per_audio = []  # Store chunk-wise features
            chunk_lengths = []  # Store the number of features in each chunk
            if args.sliding_window_size_in_sec:
                assert args.sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
                overlap_in_sec = args.sliding_window_size_in_sec * args.sliding_window_overlap_in_percent / 100
                chunk_size = int(args.target_sr * args.sliding_window_size_in_sec)  # Window size in samples
                step_size = int(args.target_sr * (args.sliding_window_size_in_sec - overlap_in_sec))  # Step size for sliding
                num_chunks = (waveform.shape[1] // step_size) + int(waveform.shape[1] % step_size != 0)

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * step_size
                    if (chunk_idx == num_chunks-1):
                        if waveform.shape[1] - start_idx < args.target_sr*0.1:
                            break
                        chunk = waveform[:, start_idx: ]
                    else:
                        chunk = waveform[:, start_idx: start_idx + chunk_size]

                    # Pass chunk through the model and extract features
                    features = FeatureExtractor.model(chunk)["hidden_states"]
                    num_layers = len(features)
                    if len(all_features_per_audio) == 0:
                        all_features_per_audio = [[] for _ in range(num_layers)]
                    
                    # Store chunk-wise features
                    for layer_idx in range(num_layers):
                        chunk_feature = features[layer_idx][0]  # No batch dimension

                        # If reduction is "mean", store mean of chunk over time, else store full chunk feature
                        if args.reduction == "mean":
                            mean_chunk_feature = torch.mean(chunk_feature, dim=0)  # Mean over time
                            all_features_per_audio[layer_idx].append(mean_chunk_feature)  # Append mean feature
                            chunk_lengths.append(chunk_feature.shape[0])  # Store the length of each chunk
                        else:
                            all_features_per_audio[layer_idx].append(chunk_feature)  # Store full chunk features
                            # If reduction is "mean", calculate weighted mean of chunk features
                
                if args.reduction == "mean":
                    total_length = sum(chunk_lengths)  # Calculate the total length (sum of all chunk lengths)
                    final_weighted_feature = []
                    for layer_idx in range(num_layers):
                        weighted_sum = 0
                        for chunk_idx, chunk_feature in enumerate(all_features_per_audio[layer_idx]):
                            chunk_weight = chunk_lengths[chunk_idx] / total_length  # Weight for the current chunk
                            weighted_sum += chunk_weight * chunk_feature  # Weighted sum
                        final_weighted_feature.append(weighted_sum)
                    final_features = final_weighted_feature
                else:
                    final_features = [torch.cat(layer_features, dim=1).cpu() for layer_features in all_features_per_audio]  # Concatenate over time dimension
            
            else:
                # Pass chunk through the model and extract features
                features = FeatureExtractor.model(waveform)["hidden_states"]
                num_layers = len(features)
                if args.reduction == "mean":
                    final_features = [torch.squeeze(torch.mean(f, 1)).cpu() for f in features]
                else:
                    final_features = [torch.squeeze(f).cpu() for f in features]
            
            # Save the final features
            output_file = get_output_file_path(args, audio_file)
            if os.path.exists(output_file) and not args.overwrite:
                continue
            torch.save(final_features, output_file)

def get_output_file_path(args, audio_file):
    if args.keep_folder_structure:
        output_file = os.path.splitext(os.path.join(
            args.output_dir,
            os.path.relpath(audio_file, args.audio_dir)
        ))[0] + ".pt"
    else:
        output_file = os.path.splitext(os.path.join(
            args.output_dir,
            os.path.basename(audio_file)
        ))[0] + ".pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Downstream task configuration file", default="")
    parser.add_argument("-u", "--upstream", type=str, help="Upstream model to extract feature from", default="")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory of features")
    parser.add_argument("-t", "--target_sr", type=int, help="Target sample rate of audio encoder model", default=16000)
    parser.add_argument("-l", "--local_path", type=str, help="Local path to upstream model", default=None)

    args = parser.parse_args()

    args = extend_args(args.config, args)
    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir, feature_dir=args.output_dir, keep_folder_structure=args.keep_folder_structure)
    print(f'Found {len(audio_files)} audio files')

    FeatureExtractor = get_upstream(args)

    process_audio_files(audio_files, FeatureExtractor, args, device)

if __name__ == '__main__':
    main()
