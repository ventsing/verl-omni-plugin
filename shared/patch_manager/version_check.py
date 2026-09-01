"""
Version checking utilities for patch compatibility.
"""

import logging
from typing import Optional

from packaging import version

logger = logging.getLogger(__name__)


class VersionChecker:
    """Utility class for checking package versions."""
    
    @staticmethod
    def check(package: str, required: str) -> bool:
        """
        Check if a package meets version requirements.
        
        Args:
            package: Package name
            required: Version requirement (e.g., ">=0.6.0", "==1.0.0")
        
        Returns:
            True if requirement is met, False otherwise
        """
        try:
            # Parse requirement
            if required.startswith(">="):
                min_version = required[2:]
                current = VersionChecker._get_version(package)
                if current is None:
                    return False
                return version.parse(current) >= version.parse(min_version)
            
            elif required.startswith("=="):
                exact_version = required[2:]
                current = VersionChecker._get_version(package)
                if current is None:
                    return False
                return version.parse(current) == version.parse(exact_version)
            
            elif required.startswith("<="):
                max_version = required[2:]
                current = VersionChecker._get_version(package)
                if current is None:
                    return False
                return version.parse(current) <= version.parse(max_version)
            
            else:
                logger.warning(f"Unsupported version requirement format: {required}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to check version for {package}: {e}")
            return False
    
    @staticmethod
    def _get_version(package: str) -> Optional[str]:
        """
        Get the installed version of a package.
        
        Args:
            package: Package name
        
        Returns:
            Version string or None if not found
        """
        try:
            if package == "verl":
                import verl
                return verl.__version__
            elif package == "verl_omni":
                import verl_omni
                return verl_omni.__version__
            elif package == "vllm":
                import vllm
                return vllm.__version__
            elif package == "vllm_omni":
                import vllm_omni
                return vllm_omni.__version__
            else:
                # Try to import and get version
                module = __import__(package)
                return getattr(module, "__version__", None)
        except ImportError:
            logger.debug(f"Package {package} not installed")
            return None
    
    @staticmethod
    def check_all(requirements: dict[str, str]) -> bool:
        """
        Check multiple package version requirements.
        
        Args:
            requirements: Dict mapping package names to version requirements
        
        Returns:
            True if all requirements are met
        """
        for package, required in requirements.items():
            if not VersionChecker.check(package, required):
                return False
        return True
