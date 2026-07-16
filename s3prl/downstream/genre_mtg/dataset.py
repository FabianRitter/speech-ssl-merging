import torchaudio
import numpy as np 
import torch
import os 
import pandas as pd
from torch.utils import data
import torch.nn.functional as F

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


class MTGGenreAudioDataset(data.Dataset):
    def __init__(self, audio_dir, datarc, split, return_audio_path=True, split_version=0, low_quality_source=True): # Singer trim?
        # self.cfg = cfg
        if split == 'dev':
            split = 'validation'
        self.metadata_dir = os.path.join(datarc['meta_data'], f'data/splits/split-{split_version}/autotagging_genre-{split}.tsv')
        self.audio_dir = audio_dir
        self.split_version = split_version
        self.low_quality_source = low_quality_source
        self.metadata = open(self.metadata_dir, 'r').readlines()[1:]

        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]

        assert len(self.all_paths) == len(self.all_tags) == len(self.metadata)
        # read class2id
        self.class2id = self.read_class2id(self.metadata_dir, split_version)
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        
    def label2class(self, id_list):
        return [self.id2class[id] for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.all_paths[index]
        if self.low_quality_source:
            audio_path = audio_path.replace('.mp3', '.low.mp3')
        class_name = self.all_tags[index]
        
        wav, sr = torchaudio.load(os.path.join(self.audio_dir, "audio-low", audio_path))
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()
        
        label = torch.zeros(len(self.class2id))  # TODO: how to deal with this?
        for c in class_name:
            label[self.class2id[c.strip()]] = 1
            
        if self.return_audio_path:
            return audio.numpy(), label, audio_path
        return audio.numpy(), label
    
    def read_class2id(self, metadata_dir, split_version):
        class2id = {}
        for split in ['train', 'validation', 'test']:
            data = open(os.path.join(metadata_dir, f'data/splits/split-{split_version}/autotagging_genre-{split}.tsv'), "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
        return class2id

    def __len__(self):
        return len(self.metadata)
    
    def collate_fn(self, samples):
        return zip(*samples)
    

class MTGGenreFeatureDataset(data.Dataset):
    def __init__(self, feature_dir, datarc, split, return_audio_path=True, split_version=0, low_quality_source=True): # Singer trim?
        # self.cfg = cfg
        if split == 'dev':
            split = 'validation'
        self.metadata_dir = os.path.join(datarc['meta_data'], f'data/splits/split-{split_version}/autotagging_genre-{split}.tsv')
        self.feature_dir = feature_dir
        self.split_version = split_version
        self.metadata = open(self.metadata_dir, 'r').readlines()[1:]
        self.low_quality_source = low_quality_source
        
        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]

        assert len(self.all_paths) == len(self.all_tags) == len(self.metadata)
        # read class2id
        self.class2id = self.read_class2id(self.metadata_dir, split_version)
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        
    def label2class(self, id_list):
        return [self.id2class[id] for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.all_paths[index]
        if self.low_quality_source:
            audio_path = audio_path.replace('.mp3', '.low.pt')
        else:
            audio_path = audio_path.replace('.mp3', '.pt')
        class_name = self.all_tags[index]
        
        feature = torch.load(os.path.join(self.feature_dir, audio_path), map_location="cpu")
        if len(feature[0].shape) == 1:
            feature = [f.unsqueeze(0).unsqueeze(0) for f in feature]
        elif len(feature.shape) == 2:
            feature = [f.unsqueeze(0) for f in feature]
        
        label = torch.zeros(len(self.class2id))  # TODO: how to deal with this?
        for c in class_name:
            label[self.class2id[c.strip()]] = 1
            
        if self.return_audio_path:
            return feature, label, audio_path
        return feature, label
    
    def read_class2id(self, metadata_dir, split_version):
        class2id = {}
        for split in ['train', 'validation', 'test']:
            data = open(metadata_dir, "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
        return class2id

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