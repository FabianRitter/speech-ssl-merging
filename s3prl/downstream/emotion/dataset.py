# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from ..distortions import DistortionFactory

SAMPLE_RATE = 16000


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True, **kwargs):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()
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

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{self.meta_data[idx]['path'].replace('/', '-')}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True
        audio = wav.numpy()
        if self.distortion is not None:
            audio = self.distortion.add_distortion(audio, SAMPLE_RATE)
        return audio, label, self.meta_data[idx]['path'].replace('/', '-') #Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
