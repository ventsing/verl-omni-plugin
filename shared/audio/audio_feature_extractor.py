"""
Audio feature extractor for extracting various audio features.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioFeatureExtractor(nn.Module):
    """
    Neural network module for extracting audio features.
    
    Can be used as part of a larger model for audio processing.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dict with keys:
                - input_dim: Input feature dimension (default: 80)
                - hidden_dim: Hidden layer dimension (default: 256)
                - output_dim: Output feature dimension (default: 512)
                - num_layers: Number of convolutional layers (default: 3)
        """
        super().__init__()
        
        self.input_dim = config.get("input_dim", 80)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", 512)
        self.num_layers = config.get("num_layers", 3)
        
        # Build convolutional layers
        self.layers = self._build_layers()
        
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        logger.info(
            f"AudioFeatureExtractor initialized: input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )
    
    def _build_layers(self) -> nn.ModuleList:
        """Build convolutional layers."""
        layers = nn.ModuleList()
        
        # First layer
        layers.append(nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Output layer
        layers.append(nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=3, padding=1))
        
        return layers
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features.
        
        Args:
            features: Input features [batch, input_dim, time]
        
        Returns:
            Extracted features [batch, output_dim]
        """
        x = features
        
        # Pass through convolutional layers
        for layer in self.layers[:-1]:
            x = layer(x)
        
        # Final convolution
        x = self.layers[-1](x)
        
        # Pool to fixed size
        x = self.pooling(x).squeeze(-1)  # [batch, output_dim]
        
        return x
