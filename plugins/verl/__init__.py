"""
verl plugin - Extensions for verl reinforcement learning framework.

This plugin provides:
- Custom platform support
- Enhanced async and full-duplex trainers
- Multimodal worker extensions
- Distributed communication optimizations
- Data processing enhancements
- Reward framework extensions
"""

__version__ = "0.1.0"

from plugins.verl.platform import CustomPlatform
from plugins.verl.trainer import FullDuplexTrainer, AsyncTrainerEnhanced
from plugins.verl.workers import EnhancedEngineWorkerGroup
from plugins.verl.reward import MultimodalRewardManager

__all__ = [
    "CustomPlatform",
    "FullDuplexTrainer",
    "AsyncTrainerEnhanced",
    "EnhancedEngineWorkerGroup",
    "MultimodalRewardManager",
]
