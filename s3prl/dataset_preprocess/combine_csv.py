import pandas as pd
import os
import argparse

# Function to combine CSV files
def combine_csv_files(csv1_path, csv2_path, csv3_path, output_path, sample_size=15000):
    # Load the three CSV files
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    csv3 = pd.read_csv(csv3_path)

    # Modify the file paths based on the directory
    csv1['file_path'] = csv1['file_path'].apply(lambda x: os.path.join('AudioSet', x) if x.startswith('train/') else x)
    csv2['file_path'] = csv2['file_path'].apply(lambda x: os.path.join('LibriSpeech', x) if x.startswith('train-clean-100/') else x)
    csv3['file_path'] = csv3['file_path'].apply(lambda x: os.path.join('music4all', x) if x.startswith('audios/') else x)

    # Select a sample from each CSV
    csv1_sample = csv1.sample(n=sample_size, random_state=42)
    csv2_sample = csv2.sample(n=sample_size, random_state=42)
    csv3_sample = csv3.sample(n=sample_size, random_state=42)

    # Combine the three samples
    combined_sample = pd.concat([csv1_sample, csv2_sample, csv3_sample], ignore_index=True)

    # Re-label the index column correctly
    combined_sample.reset_index(drop=True, inplace=True)

    # Save the new combined DataFrame to a new CSV file
    combined_sample.to_csv(output_path, index=False)
    print(f'New combined CSV file saved as {output_path}')

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Combine multiple CSV files and select a sample from each.")
    parser.add_argument('--csv1', type=str, required=True, help="Path to the first CSV file (e.g., audioset.csv)")
    parser.add_argument('--csv2', type=str, required=True, help="Path to the second CSV file (e.g., librispeech.csv)")
    parser.add_argument('--csv3', type=str, required=True, help="Path to the third CSV file (e.g., music4all.csv)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the combined CSV file (e.g., combined_dataset.csv)")
    parser.add_argument('--sample_size', type=int, default=15000, help="Number of rows to sample from each CSV (default: 15000)")
    return parser.parse_args()

# Main function to execute the CSV combination
def main():
    args = parse_args()
    combine_csv_files(args.csv1, args.csv2, args.csv3, args.output, args.sample_size)

if __name__ == "__main__":
    main()
