"""
    Dataset for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from pretrain.bucket_dataset import WaveDataset
import torch.nn.functional as F 


class OnlineWaveDataset(WaveDataset):
    """Online waveform dataset"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        max_timestep=0,
        libri_root=None,
        target_level=-25,
        **kwargs
    ):
        super().__init__(
            task_config,
            bucket_size,
            file_path,
            sets,
            max_timestep,
            libri_root,
            **kwargs
        )
        self.target_level = target_level
        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
        if wav.shape[0] == 2:  # Check if stereo (2 channels)
            wav = wav.mean(dim=0, keepdim=True)  # Average across channels to get mono
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav  # Add channel dimension if missing


        return wav.squeeze()  # (seq_len)


    def __getitem__(self, index):
        try:
            x_batch = []

            for x_file in self.X[index]:
                # Load the waveform
                waveform = self._sample(self._load_feat(x_file))
                
                # Handle multi-channel (stereo) audio: downmix to mono if more than 1 channel
                if waveform.ndim == 2 and waveform.shape[0] > 1:  # Check if multi-channel
                    print(f"Multi-channel audio detected: {waveform.shape}, downmixing to mono.", index)
                    waveform = torch.mean(waveform, dim=0, keepdim=True)  # Downmix to mono
                    waveform = waveform.squeeze(0)

                #     waveform = waveform.unsqueeze(0)  # Add channel dimension if it's missing

                # Ensure waveform length consistency: pad to 160000 samples if needed
                if waveform.shape[-1] < 160000:  # If waveform is shorter than 160000
                    waveform = F.pad(waveform, (0, 160000 - waveform.shape[-1]))
                # Ensure the waveform has at least 2 dimensions (for compatibility with Conv1d)


                x_batch.append(waveform - waveform.mean()) ## modified to maintain setup of AST and newer audio processing task. This normalization is useful.

            # Pad the sequences to the length of the longest sequence
            x_pad_batch = pad_sequence(x_batch, batch_first=True)

            # Create the padding mask
            x_lens = torch.LongTensor([wave.shape[-1] for wave in x_batch])  # Sequence lengths
            pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool)  # (batch_size, seq_len)
            for idx in range(x_pad_batch.shape[0]):
                pad_mask[idx, x_lens[idx]:] = 0  # Mask out padding regions with zeros

            return [x_pad_batch, x_batch, x_lens, pad_mask]

        except Exception as e:
            print(f"Error occurred at index {index}: {e}")
            raise e  # Re-raise the exception to stop the process
