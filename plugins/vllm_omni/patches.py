"""
Monkey-patch management for vllm-omni plugin.
"""

import logging

from shared.patch_manager import BasePatchManager

logger = logging.getLogger(__name__)


class VllmOmniPatchManager(BasePatchManager):
    """Patch manager for vllm-omni repository."""
    
    @classmethod
    def register_all_patches(cls):
        """Register all vllm-omni patches."""
        cls._register_pipeline_patches()
        
        logger.info(f"Registered {len(cls.get_registered_patches())} vllm-omni patches")
    
    @classmethod
    def _register_pipeline_patches(cls):
        """Register pipeline-related patches."""
        # Patch: Audio inference pipeline
        cls.register_patch(
            name="audio_inference_pipeline",
            target_module="vllm_omni.pipeline.base",
            target_attr="InferencePipeline",
            replacement_fn="plugins.vllm_omni.pipelines:AudioInferencePipeline",
            version_check=lambda: cls._check_version("vllm_omni", ">=0.1.0"),
            description="Audio inference pipeline",
        )
