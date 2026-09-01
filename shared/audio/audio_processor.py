"""
Audio processor for preprocessing and postprocessing audio data.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor for handling audio input/output.
    
    Provides methods for:
    - Preprocessing raw audio waveforms
    - Extracting features (e.g., Mel spectrograms)
    - Postprocessing model outputs
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio processor.
        
        Args:
            config: Configuration dict with keys:
                - sample_rate: Audio sample rate (default: 16000)
                - n_mels: Number of Mel bands (default: 80)
                - n_fft: FFT size (default: 1024)
                - hop_length: Hop length (default: 256)
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_mels = config.get("n_mels", 80)
        self.n_fft = config.get("n_fft", 1024)
        self.hop_length = config.get("hop_length", 256)
        
        # Initialize Mel spectrogram transform
        self.mel_transform = self._create_mel_transform()
        
        logger.info(
            f"AudioProcessor initialized: sample_rate={self.sample_rate}, "
            f"n_mels={self.n_mels}"
        )
    
    def _create_mel_transform(self) -> nn.Module:
        """Create Mel spectrogram transform."""
        try:
            import torchaudio.transforms as T
            
            return T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        except ImportError:
            logger.warning("torchaudio not available, using fallback Mel transform")
            return None
    
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw audio waveform.
        
        Args:
            audio: Raw audio waveform [batch, time] or [time]
        
        Returns:
            Preprocessed audio features
        """
        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize audio
        audio = self._normalize(audio)
        
        # Extract Mel spectrogram
        if self.mel_transform is not None:
            mel_spec = self.mel_transform(audio)
            # Convert to log scale
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        else:
            # Fallback: simple FFT-based features
            mel_spec = self._fallback_feature_extraction(audio)
        
        return mel_spec
    
    def postprocess(self, features: torch.Tensor) -> torch.Tensor:
        """
        Postprocess audio features back to waveform.
        
        Args:
            features: Audio features [batch, n_mels, time]
        
        Returns:
            Reconstructed audio waveform
        """
        # Convert from log scale
        mel_spec = torch.exp(features)
        
        # Inverse Mel transform (simplified)
        # In practice, you'd use a vocoder like HiFi-GAN
        audio = self._inverse_mel(mel_spec)
        
        return audio
    
    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _fallback_feature_extraction(self, audio: torch.Tensor) -> torch.Tensor:
        """Fallback feature extraction when torchaudio is not available."""
        # Simple FFT-based features
        fft = torch.fft.rfft(audio, n=self.n_fft)
        magnitude = torch.abs(fft)
        
        # Downsample to n_mels
        if magnitude.size(-1) > self.n_mels:
            magnitude = torch.nn.functional.adaptive_avg_pool1d(
                magnitude.unsqueeze(1), self.n_mels
            ).squeeze(1)
        
        return magnitude
    
    def _inverse_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Simplified inverse Mel transform."""
        # This is a placeholder - in practice, use a proper vocoder
        # For now, just return a dummy waveform
        batch_size = mel_spec.size(0)
        time_steps = mel_spec.size(-1)
        
        # Generate dummy audio
        audio = torch.randn(batch_size, time_steps * self.hop_length)
        
        return audio
