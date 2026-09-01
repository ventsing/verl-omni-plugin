"""
Base patch manager implementation.

Provides core functionality for registering and applying monkey-patches.
"""

import logging
import sys
from typing import Any, Callable, Optional

from shared.patch_manager.version_check import VersionChecker

logger = logging.getLogger(__name__)


class BasePatchManager:
    """
    Base class for managing monkey-patches across repositories.
    
    Usage:
        class MyPatchManager(BasePatchManager):
            @classmethod
            def register_all_patches(cls):
                cls.register_patch(
                    name="my_patch",
                    target_module="some.module",
                    target_attr="SomeClass",
                    replacement_fn="my_plugin.module:ReplacementClass",
                    version_check=lambda: cls._check_version("package", ">=1.0.0"),
                    description="My custom patch"
                )
    """
    
    _patches: dict[str, dict[str, Any]] = {}
    _applied_patches: set[str] = set()
    
    @classmethod
    def register_patch(
        cls,
        name: str,
        target_module: str,
        target_attr: str,
        replacement_fn: str,
        version_check: Optional[Callable[[], bool]] = None,
        description: str = "",
    ) -> None:
        """
        Register a monkey-patch.
        
        Args:
            name: Unique name for the patch
            target_module: Module path to patch (e.g., "verl.trainer.ppo.v1.trainer_base")
            target_attr: Attribute name to patch (e.g., "BaseTrainer")
            replacement_fn: Replacement function/class in format "module.path:ClassName"
            version_check: Optional function to check version compatibility
            description: Human-readable description of the patch
        """
        cls._patches[name] = {
            "module": target_module,
            "attr": target_attr,
            "replacement_fn": replacement_fn,
            "version_check": version_check,
            "description": description,
            "original": None,
        }
        
        logger.info(f"Registered patch: {name} - {description}")
    
    @classmethod
    def apply_all(cls) -> None:
        """Apply all registered patches."""
        for name, patch_info in cls._patches.items():
            if name in cls._applied_patches:
                logger.warning(f"Patch {name} already applied, skipping")
                continue
            
            # Check version compatibility
            if patch_info["version_check"]:
                if not patch_info["version_check"]():
                    logger.warning(f"Version check failed for {name}, skipping")
                    continue
            
            # Apply the patch
            try:
                cls._apply_patch(name, patch_info)
                cls._applied_patches.add(name)
                logger.info(f"Successfully applied patch: {name}")
            except Exception as e:
                logger.error(f"Failed to apply patch {name}: {e}", exc_info=True)
    
    @classmethod
    def _apply_patch(cls, name: str, patch_info: dict[str, Any]) -> None:
        """Apply a single patch."""
        # Import target module
        module = sys.modules.get(patch_info["module"])
        if module is None:
            __import__(patch_info["module"])
            module = sys.modules[patch_info["module"]]
        
        # Save original implementation
        patch_info["original"] = getattr(module, patch_info["attr"])
        
        # Load replacement
        module_path, func_name = patch_info["replacement_fn"].rsplit(":", 1)
        replacement_module = __import__(module_path, fromlist=[func_name])
        replacement = getattr(replacement_module, func_name)
        
        # Apply replacement
        setattr(module, patch_info["attr"], replacement)
    
    @classmethod
    def unpatch_all(cls) -> None:
        """Revert all applied patches."""
        for name in list(cls._applied_patches):
            patch_info = cls._patches[name]
            
            try:
                module = sys.modules[patch_info["module"]]
                setattr(module, patch_info["attr"], patch_info["original"])
                cls._applied_patches.remove(name)
                logger.info(f"Successfully reverted patch: {name}")
            except Exception as e:
                logger.error(f"Failed to revert patch {name}: {e}", exc_info=True)
    
    @classmethod
    def get_applied_patches(cls) -> list[str]:
        """Get list of applied patch names."""
        return list(cls._applied_patches)
    
    @classmethod
    def get_registered_patches(cls) -> list[str]:
        """Get list of registered patch names."""
        return list(cls._patches.keys())
    
    @classmethod
    def _check_version(cls, package: str, required: str) -> bool:
        """
        Check if a package meets version requirements.
        
        Args:
            package: Package name (e.g., "verl", "vllm")
            required: Version requirement (e.g., ">=0.6.0")
        
        Returns:
            True if version requirement is met
        """
        return VersionChecker.check(package, required)
