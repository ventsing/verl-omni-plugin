"""
Audio reward manager for verl-omni.

Provides audio quality assessment and reward computation.
"""

import logging
from typing import Any

from shared.audio import AudioQualityModel

logger = logging.getLogger(__name__)


class AudioRewardManager:
    """
    Audio reward manager that computes rewards based on audio quality.
    
    Uses audio quality metrics:
    - MCD (Mel Cepstral Distortion)
    - F0 correlation
    - Spectral loss
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize audio reward manager.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        
        # Initialize audio quality model
        self.audio_quality_model = AudioQualityModel(config.get("audio", {}))
        
        logger.info("AudioRewardManager initialized")
    
    def compute_reward(
        self,
        outputs: dict[str, Any],
        targets: dict[str, Any],
    ) -> float:
        """
        Compute audio reward.
        
        Args:
            outputs: Model outputs
            targets: Target outputs
        
        Returns:
            Audio reward value
        """
        # Extract audio from outputs and targets
        audio_output = outputs.get("audio")
        audio_target = targets.get("audio")
        
        if audio_output is None or audio_target is None:
            logger.warning("Audio not found in outputs or targets")
            return 0.0
        
        # Compute audio quality metrics
        metrics = self.audio_quality_model.evaluate(audio_output, audio_target)
        
        # Return overall quality score
        reward = metrics["overall"]
        
        logger.debug(f"Audio reward: {reward:.4f}, metrics: {metrics}")
        
        return reward
