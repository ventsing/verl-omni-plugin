"""
Monkey-patch management for verl plugin.
"""

import logging

from shared.patch_manager import BasePatchManager

logger = logging.getLogger(__name__)


class VerlPatchManager(BasePatchManager):
    """Patch manager for verl repository."""
    
    @classmethod
    def register_all_patches(cls):
        """Register all verl patches."""
        cls._register_trainer_patches()
        cls._register_worker_patches()
        cls._register_reward_patches()
        
        logger.info(f"Registered {len(cls.get_registered_patches())} verl patches")
    
    @classmethod
    def _register_trainer_patches(cls):
        """Register trainer-related patches."""
        # Patch 1: Full-duplex trainer
        cls.register_patch(
            name="full_duplex_trainer",
            target_module="verl.trainer.ppo.v1.trainer_base",
            target_attr="BaseTrainer",
            replacement_fn="plugins.verl.trainer:FullDuplexTrainer",
            version_check=lambda: cls._check_version("verl", ">=0.6.0"),
            description="Full-duplex training support",
        )
        
        # Patch 2: Enhanced async trainer
        cls.register_patch(
            name="enhanced_async_trainer",
            target_module="verl.experimental.fully_async_policy.fully_async_trainer",
            target_attr="FullyAsyncTrainer",
            replacement_fn="plugins.verl.trainer:AsyncTrainerEnhanced",
            version_check=lambda: cls._check_version("verl", ">=0.6.0"),
            description="Enhanced asynchronous training",
        )
    
    @classmethod
    def _register_worker_patches(cls):
        """Register worker-related patches."""
        # Patch: Enhanced engine worker
        cls.register_patch(
            name="enhanced_engine_worker",
            target_module="verl.workers.engine_workers",
            target_attr="EngineWorkerGroup",
            replacement_fn="plugins.verl.workers:EnhancedEngineWorkerGroup",
            version_check=lambda: cls._check_version("verl", ">=0.6.0"),
            description="Enhanced engine worker with multimodal support",
        )
    
    @classmethod
    def _register_reward_patches(cls):
        """Register reward-related patches."""
        # Patch: Multimodal reward manager
        cls.register_patch(
            name="multimodal_reward_manager",
            target_module="verl.experimental.reward_loop.reward_manager.base",
            target_attr="BaseRewardManager",
            replacement_fn="plugins.verl.reward:MultimodalRewardManager",
            version_check=lambda: cls._check_version("verl", ">=0.6.0"),
            description="Multimodal reward computation",
        )
