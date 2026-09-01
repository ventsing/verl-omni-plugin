"""
Unified patch manager for all plugins.

Provides a centralized way to register, apply, and manage monkey-patches
across different repositories (verl, verl-omni, vllm, vllm-omni).
"""

from shared.patch_manager.base import BasePatchManager
from shared.patch_manager.version_check import VersionChecker

__all__ = ["BasePatchManager", "VersionChecker"]
