"""
Configuration utilities for plugins.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dict
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dict
        config_path: Path to save config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            yaml.safe_dump(config, f, default_flow_style=False)
        elif config_path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.info(f"Saved config to {config_path}")
