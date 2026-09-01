"""
Plugin framework for verl, verl-omni, vllm, and vllm-omni.

This package provides extensions for:
- verl: Reinforcement learning training framework
- verl-omni: Multimodal RL training with diffusion models
- vllm: High-performance LLM inference engine
- vllm-omni: Multimodal inference with omni-modality models
"""

__version__ = "0.1.0"

import logging

logger = logging.getLogger(__name__)


def register_vllm_plugin():
    """Entry point for vLLM plugin registration."""
    logger.info("Registering verl-omni-plugin with vLLM")
    
    # Apply vLLM patches
    try:
        from plugins.vllm.patches import VllmPatchManager
        VllmPatchManager.register_all_patches()
        VllmPatchManager.apply_all()
        logger.info("Successfully applied vLLM patches")
    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}", exc_info=True)


def register_verl_plugin():
    """Entry point for verl plugin registration."""
    logger.info("Registering verl-omni-plugin with verl")
    
    # Apply verl patches
    try:
        from plugins.verl.patches import VerlPatchManager
        VerlPatchManager.register_all_patches()
        VerlPatchManager.apply_all()
        logger.info("Successfully applied verl patches")
    except Exception as e:
        logger.error(f"Failed to apply verl patches: {e}", exc_info=True)


def register_verl_omni_plugin():
    """Entry point for verl-omni plugin registration."""
    logger.info("Registering verl-omni-plugin with verl-omni")
    
    # Apply verl-omni patches
    try:
        from plugins.verl_omni.patches import VerlOmniPatchManager
        VerlOmniPatchManager.register_all_patches()
        VerlOmniPatchManager.apply_all()
        logger.info("Successfully applied verl-omni patches")
    except Exception as e:
        logger.error(f"Failed to apply verl-omni patches: {e}", exc_info=True)


def register_vllm_omni_plugin():
    """Entry point for vllm-omni plugin registration."""
    logger.info("Registering verl-omni-plugin with vllm-omni")
    
    # Apply vllm-omni patches
    try:
        from plugins.vllm_omni.patches import VllmOmniPatchManager
        VllmOmniPatchManager.register_all_patches()
        VllmOmniPatchManager.apply_all()
        logger.info("Successfully applied vllm-omni patches")
    except Exception as e:
        logger.error(f"Failed to apply vllm-omni patches: {e}", exc_info=True)


# Auto-register all plugins on import
def _auto_register():
    """Automatically register all available plugins."""
    import os
    
    # Check environment variables to control which plugins to register
    enable_all = os.getenv("VERL_OMNI_PLUGIN_ENABLE_ALL", "0") == "1"
    
    if enable_all or os.getenv("VERL_OMNI_PLUGIN_ENABLE_VERL", "1") == "1":
        try:
            register_verl_plugin()
        except ImportError:
            logger.debug("verl not available, skipping")
    
    if enable_all or os.getenv("VERL_OMNI_PLUGIN_ENABLE_VERL_OMNI", "1") == "1":
        try:
            register_verl_omni_plugin()
        except ImportError:
            logger.debug("verl-omni not available, skipping")
    
    if enable_all or os.getenv("VERL_OMNI_PLUGIN_ENABLE_VLLM", "1") == "1":
        try:
            register_vllm_plugin()
        except ImportError:
            logger.debug("vllm not available, skipping")
    
    if enable_all or os.getenv("VERL_OMNI_PLUGIN_ENABLE_VLLM_OMNI", "1") == "1":
        try:
            register_vllm_omni_plugin()
        except ImportError:
            logger.debug("vllm-omni not available, skipping")


# Uncomment to enable auto-registration on import
# _auto_register()


__all__ = [
    "register_vllm_plugin",
    "register_verl_plugin",
    "register_verl_omni_plugin",
    "register_vllm_omni_plugin",
]
