import os
import torchaudio
import torchaudio.transforms as T
from joblib import Parallel, delayed
import argparse

# Function to resample a single audio file
def resample_audio(file_path, target_sample_rate=16000, output_dir='resampled_audios'):
    try:
        # Load the audio file
        waveform, original_sample_rate = torchaudio.load(file_path)

        # Only resample if the original sample rate differs from the target
        if original_sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the resampled audio
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name)
        torchaudio.save(output_file_path, waveform, target_sample_rate)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to process the dataset in parallel
def process_dataset_parallel(dataset_dir, target_sample_rate=16000, output_dir='resampled_audios', n_jobs=-1):
    audio_extensions = ['.mp3', '.wav', '.flac']  # Supported audio formats
    todo = []

    # Collect all audio file paths
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if any(file.endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(root, file)
                todo.append(file_path)

    # Process files in parallel
    Parallel(n_jobs=n_jobs)(delayed(resample_audio)(file_path, target_sample_rate, output_dir) for file_path in todo)

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Resample audio files to a common sample rate.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--target_sample_rate", type=int, default=16000, help="Target sample rate (default: 16000 Hz)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (default: all cores)")
    parser.add_argument("--output_dir", type=str, default="resampled_audios", help="Directory to save resampled files")
    return parser.parse_args()

# Main function to execute the resampling
def main():
    args = parse_args()

    # Process dataset with the provided arguments
    process_dataset_parallel(args.dataset_dir, args.target_sample_rate, args.output_dir, n_jobs=args.n_jobs)

if __name__ == "__main__":
    main()
