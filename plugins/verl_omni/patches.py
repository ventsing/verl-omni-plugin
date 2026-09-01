"""
Monkey-patch management for verl-omni plugin.
"""

import logging

from shared.patch_manager import BasePatchManager

logger = logging.getLogger(__name__)


class VerlOmniPatchManager(BasePatchManager):
    """Patch manager for verl-omni repository."""
    
    @classmethod
    def register_all_patches(cls):
        """Register all verl-omni patches."""
        cls._register_model_patches()
        cls._register_reward_patches()
        
        logger.info(f"Registered {len(cls.get_registered_patches())} verl-omni patches")
    
    @classmethod
    def _register_model_patches(cls):
        """Register model-related patches."""
        # Patch 1: Audio head
        cls.register_patch(
            name="audio_head",
            target_module="verl_omni.models.transformers.qwen3_omni_thinker",
            target_attr="Qwen3OmniThinker",
            replacement_fn="plugins.verl_omni.models.audio:AudioHead",
            version_check=lambda: cls._check_version("verl_omni", ">=0.2.0"),
            description="Audio head processing",
        )
        
        # Patch 2: Omni model adapter
        cls.register_patch(
            name="omni_model_adapter",
            target_module="verl_omni.pipelines.model_base",
            target_attr="OmniModelBase",
            replacement_fn="plugins.verl_omni.models.omni:CustomOmniModelAdapter",
            version_check=lambda: cls._check_version("verl_omni", ">=0.2.0"),
            description="Custom omni model with multimodal fusion",
        )
    
    @classmethod
    def _register_reward_patches(cls):
        """Register reward-related patches."""
        # Patch: Audio reward manager
        cls.register_patch(
            name="audio_reward_manager",
            target_module="verl_omni.reward_loop.reward_manager.multi",
            target_attr="MultiVisualRewardManager",
            replacement_fn="plugins.verl_omni.reward:AudioRewardManager",
            version_check=lambda: cls._check_version("verl_omni", ">=0.2.0"),
            description="Audio quality assessment and reward",
        )
