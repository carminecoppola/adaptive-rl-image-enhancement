"""
Configuration utilities.

This module loads YAML configuration files and expands environment
variables defined in the `.env` file.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file and expand environment variables.

    Example:
        "${DATA_ROOT}" becomes the value of DATA_ROOT from `.env`.
    """
    load_dotenv()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        raw_config = file.read()

    expanded_config = os.path.expandvars(raw_config)

    return yaml.safe_load(expanded_config)


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """
    Load all YAML configs from the config directory.
    """
    config_dir = Path(config_dir)

    configs = {}

    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(config_file)

    return configs