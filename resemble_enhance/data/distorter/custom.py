import logging
import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

from ..utils import walk_paths
from .base import Effect

_logger = logging.getLogger(__name__)


@dataclass
class RandomRIR(Effect):
    rir_dir: Path | None
    rir_rate: int = 44_000  # Default/fallback rate
    rir_suffix: str = ".npy"
    deterministic: bool = False

    @cached_property
    def rir_paths(self):
        if self.rir_dir is None:
            return []
        return list(walk_paths(self.rir_dir, self.rir_suffix))
    
    @cached_property
    def rir_dataset_rates(self):
        """Cache sample rates for each RIR dataset directory"""
        if self.rir_dir is None:
            return {}
        
        dataset_rates = {}
        
        # Scan each subdirectory for sample_rate.txt
        for subdir in self.rir_dir.iterdir():
            if subdir.is_dir():
                sample_rate_file = subdir / "sample_rate.txt"
                if sample_rate_file.exists():
                    try:
                        with open(sample_rate_file, 'r') as f:
                            rate = int(f.read().strip())
                        dataset_rates[subdir.name] = rate
                        _logger.info(f"Loaded sample rate {rate} Hz for RIR dataset: {subdir.name}")
                    except (ValueError, IOError) as e:
                        _logger.warning(f"Could not read sample rate from {sample_rate_file}: {e}")
                        dataset_rates[subdir.name] = self.rir_rate  # Use fallback
                else:
                    _logger.warning(f"No sample_rate.txt found in {subdir}, using default {self.rir_rate} Hz")
                    dataset_rates[subdir.name] = self.rir_rate  # Use fallback
        
        _logger.info(f"Cached sample rates for {len(dataset_rates)} RIR datasets")
        return dataset_rates
    
    def _get_rir_native_rate(self, rir_path: Path):
        """Get the native sample rate for a specific RIR file"""
        # Find which dataset this RIR belongs to
        for parent in rir_path.parents:
            if parent.parent == self.rir_dir:  # This is a dataset directory
                dataset_name = parent.name
                return self.rir_dataset_rates.get(dataset_name, self.rir_rate)
        
        # Fallback if we can't determine the dataset
        return self.rir_rate

    def _sample_rir(self):
        if len(self.rir_paths) == 0:
            return None, None

        if self.deterministic:
            rir_path = self.rir_paths[0]
        else:
            rir_path = random.choice(self.rir_paths)

        rir = np.squeeze(np.load(rir_path))
        assert isinstance(rir, np.ndarray)
        
        # Get the native sample rate for this RIR
        native_rate = self._get_rir_native_rate(rir_path)

        return rir, native_rate

    def apply(self, wav, sr):
        # ref: https://github.com/haoheliu/voicefixer_main/blob/b06e07c945ac1d309b8a57ddcd599ca376b98cd9/dataloaders/augmentation/magical_effects.py#L158

        if len(self.rir_paths) == 0:
            return wav

        length = len(wav)

        # Sample RIR and get its native sample rate
        rir, rir_native_rate = self._sample_rir()
        
        if rir is None:
            return wav
        
        # Use the actual native sample rate of the RIR instead of assuming self.rir_rate
        _logger.debug(f"Using RIR with native sample rate: {rir_native_rate} Hz")
        
        # Resample audio to match RIR's native sample rate
        wav = librosa.resample(wav, orig_sr=sr, target_sr=rir_native_rate, res_type="kaiser_fast")
        
        # Ensure both wav and rir are 1D arrays for convolution
        if rir is not None:
            # If RIR is multi-channel, take the first channel or average them
            if rir.ndim > 1:
                if rir.shape[1] == 1:
                    rir = rir.squeeze()
                else:
                    # For stereo/multi-channel RIR, take the first channel
                    # or average channels - using first channel for simplicity
                    rir = rir[:, 0] if rir.shape[0] > rir.shape[1] else rir[0, :]
                    _logger.debug(f"Multi-channel RIR detected, using first channel. Original shape: {rir.shape}")
        
        # Ensure wav is also 1D
        if wav.ndim > 1:
            wav = wav.squeeze()
            if wav.ndim > 1:
                wav = wav[:, 0] if wav.shape[0] > wav.shape[1] else wav[0, :]

        # Apply RIR convolution
        wav = signal.convolve(wav, rir, mode="same")

        # Prevent clipping
        actlev = np.max(np.abs(wav))
        if actlev > 0.99:
            wav = (wav / actlev) * 0.98

        # Resample back to target sample rate
        wav = librosa.resample(wav, orig_sr=rir_native_rate, target_sr=sr, res_type="kaiser_fast")

        # Ensure output length matches input length
        if abs(length - len(wav)) > 10:
            _logger.warning(f"length mismatch: {length} vs {len(wav)}")

        if length > len(wav):
            wav = np.pad(wav, (0, length - len(wav)))
        elif length < len(wav):
            wav = wav[:length]

        return wav


class RandomGaussianNoise(Effect):
    def __init__(self, alpha_range=(0.8, 1)):
        super().__init__()
        self.alpha_range = alpha_range

    def apply(self, wav, sr):
        noise = np.random.randn(*wav.shape)
        noise_energy = np.sum(noise**2)
        wav_energy = np.sum(wav**2)
        noise = noise * np.sqrt(wav_energy / noise_energy)
        alpha = random.uniform(*self.alpha_range)
        return wav * alpha + noise * (1 - alpha)