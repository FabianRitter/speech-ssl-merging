import torchaudio
import numpy as np
import torch
import os
import pandas as pd
from torch.utils import data
import torch.nn.functional as F
import json
from glob import glob
import csv
from ..distortions import DistortionFactory

torchaudio.set_audio_backend("soundfile")
CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
print(f"torchaudio.list_audio_backends()  {torchaudio.list_audio_backends()}   ")

class ESC50Dataset(data.Dataset):
    def __init__(self, root_dir, folds, sample_rate=16000, return_audio_path=True, **kwargs):
        """
        Args:
            root_dir (str): Root directory containing all audio files and labels.
            folds (list): List of fold names to include in this dataset (e.g., ["fold00", "fold01"]).
            sample_rate (int): Target sample rate for resampling the audio.
            return_audio_path (bool): If True, __getitem__ will return the audio path as well.
        """
        self.root_dir = root_dir
        self.folds = folds
        self.sample_rate = sample_rate
        self.return_audio_path = return_audio_path
        self.data = []
        # Load label vocabulary
        self.class2id, self.id2class = self._load_label_vocabulary(os.path.join(root_dir, "labelvocabulary.csv"))
        self.num_classes = len(self.id2class)
        self.norm_mean = -6.627
        self.norm_std = 5.359
        
        # Load data
        for fold in folds:
            fold_path = os.path.join(root_dir, f"{fold}.json")
            audio_dir = os.path.join(root_dir, "48000", fold)
            
            with open(fold_path, 'r') as f:
                labels = json.load(f)
            
            # Append each file path and its label to the dataset list
            for filename, label in labels.items():
                audio_path = os.path.join(audio_dir, filename)
                if os.path.exists(audio_path):  # Ensure file exists
                    self.data.append((audio_path, self.class2id[label[0]]))  # label[0] to get the int label

        distortion_mode = kwargs.get('distortion_mode', None)
        if distortion_mode:
            self.distortion = DistortionFactory(
                distortion_types=kwargs['distortion_types'],
                distortion_config=kwargs['distortion_config'],
            )
        else:
            self.distortion = None

    def _load_label_vocabulary(self, vocab_path):
        """
        Load the label vocabulary from a CSV file and create mappings for class2id and id2class.
        Args:
            vocab_path (str): Path to the CSV file containing the label vocabulary.

        Returns:
            class2id (dict): Mapping from class name to class ID.
            id2class (dict): Mapping from class ID to class name.
        """
        class2id = {}
        id2class = {}
        with open(vocab_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row['idx'])
                class_name = row['label']
                class2id[class_name] = class_id
                id2class[class_id] = class_name
        return class2id, id2class
    
    def label2class(self, id_list):
        """
        Convert a list of class IDs to their corresponding class names.

        Args:
            id_list (list): List of integer class IDs.

        Returns:
            list: List of class names corresponding to the input IDs.
        """
        return [self.id2class[class_id] for class_id in id_list]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(audio_path)
        
        # Resample to the desired sample rate and remove extra dimension
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()
        
        # Return based on whether audio path should be included
        audio_np = audio.numpy()
        if self.distortion is not None:
            audio_np = self.distortion.add_distortion(audio_np, self.sample_rate)
        if self.return_audio_path:
            return audio_np, label, audio_path
        else:
            return audio_np, label

    def collate_fn(self, samples):
        return zip(*samples)


class ESC50FeatureDataset(data.Dataset):
    def __init__(self, root_dir, feature_dir, folds, sample_rate=16000, return_audio_path=True):
        """
        Args:
            root_dir (str): Root directory containing all audio files and labels.
            folds (list): List of fold names to include in this dataset (e.g., ["fold00", "fold01"]).
            sample_rate (int): Target sample rate for resampling the audio.
            return_audio_path (bool): If True, __getitem__ will return the audio path as well.
        """
        self.root_dir = root_dir
        self.feature_dir = feature_dir
        self.folds = folds
        self.sample_rate = sample_rate
        self.return_audio_path = return_audio_path
        self.data = []
        # Load label vocabulary
        self.class2id, self.id2class = self._load_label_vocabulary(os.path.join(root_dir, "labelvocabulary.csv"))
        self.num_classes = len(self.id2class)
        
        # Load data
        for fold in folds:
            fold_path = os.path.join(root_dir, f"{fold}.json")
            audio_dir = os.path.join(feature_dir, fold)
            
            with open(fold_path, 'r') as f:
                labels = json.load(f)
            
            # Append each file path and its label to the dataset list
            for filename, label in labels.items():
                audio_path = os.path.join(audio_dir, filename.replace('.wav', '.pt'))
                if os.path.exists(audio_path):  # Ensure file exists
                    self.data.append((audio_path, self.class2id[label[0]]))  # label[0] to get the int label
    
    def _load_label_vocabulary(self, vocab_path):
        """
        Load the label vocabulary from a CSV file and create mappings for class2id and id2class.
        Args:
            vocab_path (str): Path to the CSV file containing the label vocabulary.

        Returns:
            class2id (dict): Mapping from class name to class ID.
            id2class (dict): Mapping from class ID to class name.
        """
        class2id = {}
        id2class = {}
        with open(vocab_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row['idx'])
                class_name = row['label']
                class2id[class_name] = class_id
                id2class[class_id] = class_name
        return class2id, id2class
    
    def label2class(self, id_list):
        """
        Convert a list of class IDs to their corresponding class names.

        Args:
            id_list (list): List of integer class IDs.

        Returns:
            list: List of class names corresponding to the input IDs.
        """
        return [self.id2class[class_id] for class_id in id_list]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        
        feature = torch.load(audio_path, map_location="cpu")
        if len(feature[0].shape) == 1:
            feature = [f.unsqueeze(0).unsqueeze(0) for f in feature]
        elif len(feature[0].shape) == 2:
            feature = [f.unsqueeze(0) for f in feature]
        
        # Return based on whether audio path should be included
        if self.return_audio_path:
            return feature, label, audio_path
        else:
            return feature, label
        
    def collate_fn(self, samples):
        zipped = list(zip(*samples))
        
        batch_size = len(zipped[0])
        num_layers = len(zipped[0][0])
        
        # Initialize a list to hold the final output for each layer
        output_list = []
        for layer_idx in range(num_layers):
            # Collect all batch elements for the current layer
            layer_tensors = [zipped[0][batch_idx][layer_idx].squeeze(0) for batch_idx in range(batch_size)]
            # Stack tensors from all batches along the 0th dimension to form [batch, 1, hidden]
            stacked_tensor = torch.stack(layer_tensors, dim=0)
            output_list.append(stacked_tensor)
        
        zipped[0] = output_list
        
        return zipped