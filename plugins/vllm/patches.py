"""
Monkey-patch management for vllm plugin.
"""

import logging

from shared.patch_manager import BasePatchManager

logger = logging.getLogger(__name__)


class VllmPatchManager(BasePatchManager):
    """Patch manager for vllm repository."""
    
    @classmethod
    def register_all_patches(cls):
        """Register all vllm patches."""
        cls._register_model_patches()
        
        logger.info(f"Registered {len(cls.get_registered_patches())} vllm patches")
    
    @classmethod
    def _register_model_patches(cls):
        """Register model-related patches."""
        # Patch: Audio encoder
        cls.register_patch(
            name="audio_encoder",
            target_module="vllm.model_executor.models.registry",
            target_attr="ModelRegistry",
            replacement_fn="plugins.vllm.model_executor:VllmAudioEncoder",
            version_check=lambda: cls._check_version("vllm", ">=0.6.0"),
            description="Audio encoder for inference",
        )
