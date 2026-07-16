import os
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from ..distortions import DistortionFactory

class GtzanDataset(Dataset):
    def __init__(self, split, root_dir, target_sr=24000, **kwargs):
        """
        Initialize the GTZAN dataset for a specific split.
        
        Args:
            split (str): One of 'train', 'valid', or 'test'.
            root_dir (str): Root directory of the GTZAN dataset.
            **kwargs: Additional arguments (e.g., from config).
        """
        self.split = split
        self.root_dir = root_dir
        self.target_sample_rate = target_sr
        # Fixed list of genres for consistent label mapping
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.label_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}

        # Determine the split file
        split_files = {
            'train': 'train_filtered.txt',
            'valid': 'valid_filtered.txt',
            'test': 'test_filtered.txt'
        }
        if split not in split_files:
            print(f"your split is {split}")
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'.")
        
        
        filtered_file = os.path.join(root_dir, split_files[split])
        with open(filtered_file, 'r') as f:
            self.file_list = [line.strip() for line in f]

        # Build data list with full paths and labels
        distortion_mode = kwargs.get('distortion_mode', None)
        if distortion_mode:
            self.distortion = DistortionFactory(
                distortion_types=kwargs['distortion_types'],
                distortion_config=kwargs['distortion_config'],
            )
        else:
            self.distortion = None

        self.data = []
        for rel_path in self.file_list:
            genre = rel_path.split('/')[0]  # e.g., 'rock' from 'rock/rock.00070.wav'
            full_path = os.path.join(root_dir, 'Data', 'genres_original', rel_path)
            self.data.append({'wav_path': full_path, 'label': self.label_to_idx[genre]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Load and return a single data sample.
        
        Returns:
            dict: Contains 'x' (waveform tensor) and 'label' (integer label).
        """
        item = self.data[index]
        wav, sample_rate = torchaudio.load(item['wav_path'])  # Shape: (channels, seq_len)
        if sample_rate != self.target_sample_rate:
            wav = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(wav)
        #assert sample_rate == 22050, f"Unexpected sample rate: {sample_rate}"
        # GTZAN files are mono; squeeze channels dimension
        wav = wav.squeeze(0).numpy()  # Shape: (seq_len,)
        if self.distortion is not None:
            wav = self.distortion.add_distortion(wav, self.target_sample_rate)
        label = item['label']
        filename = os.path.basename(item['wav_path'])
        return wav, label, filename
    
    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     Collate function to be used in the DataLoader.
    #     Args:
    #         batch (list): List of tuples (waveform, label)
    #     Returns:
    #         tuple: (list of waveforms, list of labels)
    #     """
    #     waveforms, labels, filenames = zip(*batch)
    #     return list(waveforms), list(labels), list(filenames)
    
    @staticmethod
    def collate_fn(batch):
        waveforms, labels, filenames = zip(*batch)
        # Convert to tensors and pad to the longest sequence
        max_len = max(wav.shape[0] for wav in waveforms)
        padded_wavs = [
            torch.nn.functional.pad(torch.from_numpy(wav).float(), (0, max_len - wav.shape[0]))
            for wav in waveforms
        ]
        waveforms = torch.stack(padded_wavs).numpy()  # Shape: (batch_size, max_len)
        return list(waveforms), list(labels), list(filenames)