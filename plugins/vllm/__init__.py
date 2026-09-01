"""
vllm plugin - Extensions for vllm inference engine.

This plugin provides:
- Custom platform support
- Audio model inference
- Multimodal attention
- Distributed communication optimizations
"""

__version__ = "0.1.0"

from plugins.vllm.platform import VllmCustomPlatform
from plugins.vllm.model_executor import VllmAudioEncoder

__all__ = [
    "VllmCustomPlatform",
    "VllmAudioEncoder",
]
