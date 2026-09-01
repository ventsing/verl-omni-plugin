"""
Custom platform implementation for verl.

Extends the PlatformBase to support custom hardware and optimizations.
"""

import logging
from typing import Any, Optional

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


@PlatformRegistry.register(platform="custom")
class CustomPlatform(PlatformBase):
    """
    Custom platform with audio and multimodal support.
    
    This platform extends the base platform to provide:
    - Audio processing optimizations
    - Multimodal input handling
    - Custom memory management
    """
    
    def __init__(self):
        super().__init__()
        self._device_name = "custom_device"
        self._vendor_name = "custom_vendor"
        
        logger.info("CustomPlatform initialized")
    
    @property
    def device_name(self) -> str:
        """Return device type string."""
        return self._device_name
    
    @property
    def vendor_name(self) -> str:
        """Return hardware vendor name."""
        return self._vendor_name
    
    @property
    def device_module(self):
        """Return torch device module."""
        import torch
        return torch.cuda  # Fallback to CUDA, override as needed
    
    def is_available(self) -> bool:
        """Check if platform is available."""
        # Custom detection logic
        return True
    
    def current_device(self) -> int:
        """Return current device index."""
        return 0
    
    def device_count(self) -> int:
        """Return number of available devices."""
        return 1
    
    def set_device(self, device_index: int) -> None:
        """Set current device."""
        pass
    
    def synchronize(self, device_index: Optional[int] = None) -> None:
        """Synchronize device."""
        pass
    
    def manual_seed(self, seed: int) -> None:
        """Seed device RNG."""
        import torch
        torch.manual_seed(seed)
    
    def manual_seed_all(self, seed: int) -> None:
        """Seed all devices' RNG."""
        import torch
        torch.manual_seed(seed)
    
    def set_allocator_settings(self, settings: str) -> None:
        """Configure memory allocator."""
        pass
    
    def empty_cache(self) -> None:
        """Release cached memory."""
        pass
    
    def get_device_capability(self, device_index: int = 0):
        """Return device capability."""
        return (8, 0)  # Example capability
    
    def communication_backend_name(self) -> str:
        """Return communication backend name."""
        return "nccl"
    
    def visible_devices_envvar(self) -> str:
        """Return visible devices environment variable."""
        return "CUDA_VISIBLE_DEVICES"
    
    def nvtx_range(self, msg: str):
        """NVTX range context manager."""
        import contextlib
        
        @contextlib.contextmanager
        def _range():
            yield
        
        return _range()
    
    def profiler_start(self) -> None:
        """Start profiler."""
        pass
    
    def profiler_stop(self) -> None:
        """Stop profiler."""
        pass
    
    def ray_resource_name(self) -> str:
        """Return Ray resource name."""
        return "GPU"
    
    def ray_noset_envvars(self) -> list[str]:
        """Return Ray no-set environment variables."""
        return []
    
    def is_ipc_supported(self) -> bool:
        """Check if IPC is supported."""
        return True
    
    def cudart(self):
        """Return CUDA runtime API."""
        return None
    
    def apply_model_patches(self, model_type: str) -> None:
        """
        Apply platform-specific model patches.
        
        Args:
            model_type: Type of model to patch
        """
        logger.info(f"Applying model patches for {model_type}")
        
        if model_type == "audio_model":
            self._apply_audio_patches()
        elif model_type == "omni_model":
            self._apply_omni_patches()
    
    def _apply_audio_patches(self):
        """Apply audio-specific optimizations."""
        logger.info("Applying audio patches")
        # Add audio-specific patches here
    
    def _apply_omni_patches(self):
        """Apply omni-model-specific optimizations."""
        logger.info("Applying omni patches")
        # Add omni-specific patches here
