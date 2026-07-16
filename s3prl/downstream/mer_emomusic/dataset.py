import numpy as np 
import os 
import torchaudio

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')

import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch.nn.functional as F

class EMOMusicDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=None, return_audio_path=True):
        self.metadata = os.path.join(metadata_dir, 'meta.json')
        with open(self.metadata) as f:
            self.metadata = json.load(f)
        self.audio_dir = audio_dir
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.classes = """arousal, valence""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path

        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None

    def label2class(self, id_list):
        return [self.id2class[id] for id in id_list]

    def __getitem__(self, index):
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'
        audio = load_audio(os.path.join(self.audio_dir, "wav", audio_path))
        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
        
        audio = self.process_wav(audio)
        # label = self.metadata[audio_name_without_ext]['y']
        label = np.array(self.metadata[audio_name_without_ext]['y'], dtype=np.float32)

        if self.return_audio_path:
            return audio, label, audio_path
        return audio, label

    def __len__(self):
        return len(self.audio_names_without_ext)
    
    def collate_fn(self, samples):
        return zip(*samples)

