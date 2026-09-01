"""
Custom platform for vllm.
"""

import logging

logger = logging.getLogger(__name__)


class VllmCustomPlatform:
    """Custom platform for vllm with audio support."""
    
    def __init__(self):
        logger.info("VllmCustomPlatform initialized")
    
    def get_device_name(self) -> str:
        """Get device name."""
        return "custom"
    
    def apply_audio_optimizations(self):
        """Apply audio-specific optimizations."""
        logger.info("Applying audio optimizations")
