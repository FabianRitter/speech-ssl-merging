import torchaudio
import numpy as np
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os
import pandas as pd
import torch
from torch.utils import data
import torch.nn.functional as F
from ..distortions import DistortionFactory

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


class VocalTechniqueDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=None, return_audio_path=True, **kwargs): # Singer trim?
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_t.txt'), 
                                    names = ['audio_path'])
        self.audio_dir = audio_dir
        self.class2id = {'belt': 0, 'breathy': 1, 'inhaled': 2, 'lip_trill': 3, 'spoken': 4, 'straight': 5, 'trill': 6, 'trillo': 7, 'vibrato': 8, 'vocal_fry': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = kwargs["sample_rate"]
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

    def label2tech(self, id_list):
        return [self.id2class[id] for id in id_list]

    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(os.path.join(self.audio_dir, "audio", audio_path), backend="soundfile")
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

        # # convert
        # audio_features = self.processor(audio, return_tensors="pt", sampling_rate=self.cfg.target_sr, padding=True).input_values[0]
        
        label = self.class2id[audio_path.split('/')[0]]
                
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{audio_path.replace('/', '-')}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True
        if self.return_audio_path:
            audio_np = audio.numpy()
            if self.distortion is not None:
                audio_np = self.distortion.add_distortion(audio_np, self.sample_rate)
            return audio_np, label, audio_path.replace('/', '-')

        audio_np = audio.numpy()
        if self.distortion is not None:
            audio_np = self.distortion.add_distortion(audio_np, self.sample_rate)
        return audio_np, label

    def __len__(self):
        return len(self.metadata)
    
    def collate_fn(self, samples):
        return zip(*samples)
    
class VocalTechniqueFeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, sample_duration=None, return_audio_path=True, **kwargs): # Singer trim?
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_t.txt'), 
                                    names = ['audio_path'])
        self.feature_dir = feature_dir
        self.class2id = {'belt': 0, 'breathy': 1, 'inhaled': 2, 'lip_trill': 3, 'spoken': 4, 'straight': 5, 'trill': 6, 'trillo': 7, 'vibrato': 8, 'vocal_fry': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']

    def label2tech(self, id_list):
        return [self.id2class[id] for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        
        feature = torch.load(os.path.join(self.feature_dir, "audio", audio_path.replace(".wav", ".pt")), map_location="cpu")
        if len(feature[0].shape) == 1:
            feature = [f.unsqueeze(0).unsqueeze(0) for f in feature]
        elif len(feature[0].shape) == 2:
            feature = [f.unsqueeze(0) for f in feature]
        
        label = self.class2id[audio_path.split('/')[0]]

        if self.return_audio_path:
            return feature, label, audio_path.replace('/', '-')
        print(feature[0].shape)
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