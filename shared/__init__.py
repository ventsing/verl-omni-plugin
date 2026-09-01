"""
Shared utilities and tools for verl-omni-plugin.

This package contains tools that are shared across multiple plugins:
- patch_manager: Unified monkey-patch management
- audio: Audio processing utilities
- utils: Common utilities
"""

__version__ = "0.1.0"

from shared.patch_manager import BasePatchManager
from shared.audio import AudioProcessor, AudioFeatureExtractor
from shared.utils import setup_logger, load_config

__all__ = [
    "BasePatchManager",
    "AudioProcessor",
    "AudioFeatureExtractor",
    "setup_logger",
    "load_config",
]
