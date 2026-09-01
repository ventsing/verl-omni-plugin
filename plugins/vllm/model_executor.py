"""
Model executor extensions for vllm.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VllmAudioEncoder(nn.Module):
    """
    Optimized audio encoder for vllm inference.
    
    Uses vllm-optimized operators for fast inference.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio encoder.
        
        Args:
            config: Configuration dict
        """
        super().__init__()
        self.config = config
        
        # Build optimized layers
        self.layers = self._build_optimized_layers(config)
        
        logger.info("VllmAudioEncoder initialized")
    
    def _build_optimized_layers(self, config: dict[str, Any]) -> nn.ModuleList:
        """Build optimized encoder layers."""
        n_mels = config.get("n_mels", 80)
        hidden_size = config.get("hidden_size", 512)
        
        # Use vllm-optimized layers when available
        try:
            from vllm.model_executor.layers import Linear, LayerNorm
            
            layers = nn.ModuleList([
                Linear(n_mels, 256),
                LayerNorm(256),
                nn.ReLU(),
                Linear(256, 512),
                LayerNorm(512),
                nn.ReLU(),
                Linear(512, hidden_size),
            ])
        except ImportError:
            # Fallback to standard PyTorch layers
            layers = nn.ModuleList([
                nn.Linear(n_mels, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, hidden_size),
            ])
        
        return layers
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features.
        
        Args:
            audio_features: Input features [batch, n_mels, time]
        
        Returns:
            Encoded features [batch, hidden_size]
        """
        x = audio_features
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Pool to fixed size
        x = x.mean(dim=-1)  # [batch, hidden_size]
        
        return x
