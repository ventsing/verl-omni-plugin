"""
Common utilities shared across plugins.
"""

from shared.utils.logging import setup_logger
from shared.utils.config import load_config, save_config

__all__ = [
    "setup_logger",
    "load_config",
    "save_config",
]
