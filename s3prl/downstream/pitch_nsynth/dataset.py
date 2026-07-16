import numpy as np 
import os 
import torchaudio
import torch

torchaudio.set_audio_backend("sox_io")
CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
print(f"torchaudio.list_audio_backends()  {torchaudio.list_audio_backends()}   ")
current_backend = torchaudio.get_audio_backend()
print(f"Current audio backend: {current_backend}")

import numpy as np
from torch.utils import data
import pandas as pd
import torch.nn.functional as F
import json
from ..distortions import DistortionFactory

class PitchClassiDataset(data.Dataset):
    def __init__(self, metadata_dir, split, sample_duration=None, return_audio_path=True, **kwargs):
        # self.cfg = cfg
        self.metadata_dir = os.path.join(metadata_dir, f'nsynth-{split}/examples.json')
        self.metadata = json.load(open(self.metadata_dir,'r'))
        self.metadata = [(k + '.wav', v['pitch']) for k, v in self.metadata.items()]

        self.audio_dir = os.path.join(metadata_dir, f'nsynth-{split}')
        self.return_audio_path = return_audio_path
        self.sample_rate = kwargs['sample_rate']
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']
        distortion_mode = kwargs.get('distortion_mode', None)
        if distortion_mode:
            self.distortion = DistortionFactory(
                distortion_types=kwargs['distortion_types'],
                distortion_config=kwargs['distortion_config'],
            )
        else:
            self.distortion = None

    def label2class(self, id_list):
        return [ id+9 for id in id_list]

    def __getitem__(self, index):
        audio_path = self.metadata[index][0]
        label = self.metadata[index][1] - 9

        wav, sr = torchaudio.load(os.path.join(self.audio_dir, "audio", audio_path))
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()

        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[0] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[0]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[random_start:random_start+self.sample_duration]
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{audio_path.replace('/','-')}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True
        if self.return_audio_path:
            audio_np = audio.numpy()
            if self.distortion is not None:
                audio_np = self.distortion.add_distortion(audio_np, self.sample_rate)
            return audio_np, label, audio_path.replace("/","-")
        audio_np = audio.numpy()
        if self.distortion is not None:
            audio_np = self.distortion.add_distortion(audio_np, self.sample_rate)
        return audio_np, label

    def __len__(self):
        return len(self.metadata)

    def collate_fn(self, samples):
        return zip(*samples)

class PitchClassiFeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, sample_duration=None, return_audio_path=True, **kwargs):
        # self.cfg = cfg
        self.split = split
        self.metadata_dir = os.path.join(metadata_dir, f'nsynth-{split}/examples.json')
        self.metadata = json.load(open(self.metadata_dir,'r'))
        self.metadata = [(k + '.wav', v['pitch']) for k, v in self.metadata.items()]

        self.feature_dir = feature_dir
        self.return_audio_path = return_audio_path
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']
    
    def label2class(self, id_list):
        return [ id+9 for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.metadata[index][0]
        label = self.metadata[index][1] - 9
        
        feature = torch.load(os.path.join(self.feature_dir, f"nsynth-{self.split}", "audio", audio_path.replace(".wav", ".pt")), map_location="cpu")
        if len(feature[0].shape) == 1:
            feature = [f.unsqueeze(0).unsqueeze(0) for f in feature]
        elif len(feature[0].shape) == 2:
            feature = [f.unsqueeze(0) for f in feature]
        
        if self.return_audio_path:
            return feature, label, audio_path.replace("/","-")
        return feature, label

    def __len__(self):
        return len(self.metadata)
    
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