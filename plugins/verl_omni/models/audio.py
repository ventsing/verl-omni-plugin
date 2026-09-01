"""
Audio models for verl-omni.

Provides audio processing head, encoder, and decoder.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from shared.audio import AudioProcessor

logger = logging.getLogger(__name__)


class AudioHead(nn.Module):
    """
    Audio processing head for encoding and decoding audio.
    
    This module provides:
    - Audio encoding: waveform -> features
    - Audio decoding: features -> waveform
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio head.
        
        Args:
            config: Configuration dict with keys:
                - sample_rate: Audio sample rate (default: 16000)
                - n_mels: Number of Mel bands (default: 80)
                - hidden_size: Model hidden size
        """
        super().__init__()
        self.config = config
        
        # Initialize encoder and decoder
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
        
        # Audio processor
        self.processor = AudioProcessor(config)
        
        logger.info("AudioHead initialized")
    
    def forward(self, audio_input: torch.Tensor, mode: str = "encode") -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio_input: Audio input (waveform or features)
            mode: "encode" or "decode"
        
        Returns:
            Encoded features or decoded waveform
        """
        if mode == "encode":
            return self.encoder(audio_input)
        elif mode == "decode":
            return self.decoder(audio_input)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class AudioEncoder(nn.Module):
    """
    Audio encoder that converts waveform to features.
    
    Architecture:
    - Conv1d layers for feature extraction
    - Layer normalization
    - Adaptive pooling
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio encoder.
        
        Args:
            config: Configuration dict
        """
        super().__init__()
        self.config = config
        
        # Get dimensions
        n_mels = config.get("n_mels", 80)
        hidden_size = config.get("hidden_size", 512)
        
        # Build encoder layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Conv1d(512, hidden_size, kernel_size=3, padding=1),
        )
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        logger.info(f"AudioEncoder initialized: n_mels={n_mels}, hidden_size={hidden_size}")
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features.
        
        Args:
            audio_features: Input features [batch, n_mels, time]
        
        Returns:
            Encoded features [batch, hidden_size]
        """
        # Pass through conv layers
        x = self.conv_layers(audio_features)
        
        # Pool to fixed size
        x = self.pooling(x).squeeze(-1)  # [batch, hidden_size]
        
        return x


class AudioDecoder(nn.Module):
    """
    Audio decoder that converts features back to waveform.
    
    Architecture:
    - Linear layers for feature transformation
    - Output Mel spectrogram
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio decoder.
        
        Args:
            config: Configuration dict
        """
        super().__init__()
        self.config = config
        
        # Get dimensions
        hidden_size = config.get("hidden_size", 512)
        n_mels = config.get("n_mels", 80)
        audio_length = config.get("audio_length", 100)
        
        # Build decoder layers
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_mels * audio_length),
        )
        
        self.n_mels = n_mels
        self.audio_length = audio_length
        
        logger.info(f"AudioDecoder initialized: hidden_size={hidden_size}, n_mels={n_mels}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode audio features.
        
        Args:
            features: Input features [batch, hidden_size]
        
        Returns:
            Decoded audio [batch, n_mels, audio_length]
        """
        # Pass through decoder layers
        x = self.decoder_layers(features)
        
        # Reshape to [batch, n_mels, audio_length]
        x = x.view(-1, self.n_mels, self.audio_length)
        
        return x
