"""
verl-omni plugin - Extensions for multimodal RL training with diffusion models.

This plugin provides:
- Audio model support (audio head, encoder, decoder)
- Omni-model extensions (multimodal fusion)
- Audio Flow-GRPO pipeline
- Full-duplex omni training
- Audio reward management
- Multimodal agent loops
"""

__version__ = "0.1.0"

from plugins.verl_omni.models.audio import AudioHead, AudioEncoder
from plugins.verl_omni.models.omni import CustomOmniModelAdapter
from plugins.verl_omni.reward import AudioRewardManager

__all__ = [
    "AudioHead",
    "AudioEncoder",
    "CustomOmniModelAdapter",
    "AudioRewardManager",
]
