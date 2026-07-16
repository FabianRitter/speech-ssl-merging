# -*- coding: utf-8 -*- #
"""
Lightweight distortion module for noisy evaluation.
Supports CHiME3 additive noise and Gaussian noise.
No augment/pyroomacoustics dependencies — only numpy, random, glob, yaml, torchaudio.
"""

import glob
import random
import numpy as np
import yaml
import torchaudio


class DistortionFactory:
    """Apply additive noise distortions to audio signals."""

    def __init__(self, distortion_types, distortion_config, snr_range=(10, 20)):
        """
        Args:
            distortion_types: list of noise type strings, e.g. ['chime'] or ['g']
            distortion_config: path to YAML file mapping noise names to directories
            snr_range: (min_snr, max_snr) in dB for mixing
        """
        self.distortion_types = distortion_types
        self.snr_range = snr_range

        with open(distortion_config, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Normalize config keys to lowercase for case-insensitive lookup
        self.config = {k.lower(): v for k, v in raw_config.items()}

        # Pre-glob noise file lists for real-noise types
        self.noise_files = {}
        for noise_type in distortion_types:
            if noise_type == 'g':
                continue  # Gaussian noise is generated on the fly
            noise_dir = self.config.get(noise_type.lower())
            if noise_dir is None:
                raise ValueError(f"No directory configured for noise type '{noise_type}' in {distortion_config}")
            files = sorted(glob.glob(f"{noise_dir}/*.wav"))
            if not files:
                raise ValueError(f"No .wav files found in {noise_dir} for noise type '{noise_type}'")
            self.noise_files[noise_type] = files
            print(f"[DistortionFactory] Loaded {len(files)} noise files for '{noise_type}' from {noise_dir}")

    @staticmethod
    def _load_wav(filepath, target_sr):
        """Load a wav file and resample to target_sr."""
        wav, sr = torchaudio.load(filepath)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        return wav.squeeze(0).numpy()

    @staticmethod
    def _snr_coeff(snr_db, signal, noise):
        """Compute the scaling coefficient for the noise given a target SNR in dB."""
        sig_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return 0.0
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        return np.sqrt(target_noise_power / noise_power)

    @staticmethod
    def _add_real_noise(signal, noise, snr_db):
        """Tile/trim noise to match signal length, then mix at the given SNR."""
        sig_len = len(signal)
        noise_len = len(noise)

        if noise_len < sig_len:
            # Tile noise to cover the signal
            reps = (sig_len // noise_len) + 1
            noise = np.tile(noise, reps)
        # Random offset crop
        max_start = len(noise) - sig_len
        start = random.randint(0, max(0, max_start))
        noise = noise[start:start + sig_len]

        coeff = DistortionFactory._snr_coeff(snr_db, signal, noise)
        return signal + coeff * noise

    def add_distortion(self, signal, sample_rate):
        """
        Apply a random distortion from self.distortion_types to the signal.

        Args:
            signal: numpy array of shape (num_samples,)
            sample_rate: sample rate of the signal

        Returns:
            Distorted signal as numpy array.
        """
        noise_type = random.choice(self.distortion_types)
        snr_db = random.uniform(*self.snr_range)

        if noise_type == 'g':
            # Gaussian white noise
            noise = np.random.randn(*signal.shape).astype(signal.dtype)
            return self._add_real_noise(signal, noise, snr_db)
        else:
            # Real noise (CHiME, etc.)
            noise_file = random.choice(self.noise_files[noise_type])
            noise = self._load_wav(noise_file, sample_rate)
            return self._add_real_noise(signal, noise, snr_db)
