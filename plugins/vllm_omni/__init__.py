"""
vllm-omni plugin - Extensions for multimodal inference.

This plugin provides:
- Audio inference pipeline
- Streaming audio inference
- Full-duplex omni inference
"""

__version__ = "0.1.0"

from plugins.vllm_omni.pipelines import AudioInferencePipeline, AudioStreamingPipeline

__all__ = [
    "AudioInferencePipeline",
    "AudioStreamingPipeline",
]
