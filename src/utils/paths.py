"""
Path utilities.

This module centralizes project paths loaded from configuration files.
"""

from pathlib import Path

from src.utils.config import load_config


def get_paths(config_path: str = "configs/paths.yaml") -> dict[str, Path]:
    """
    Load project paths from paths.yaml and convert them to Path objects.
    """
    config = load_config(config_path)

    raw_paths = config["paths"]

    return {
        key: Path(value)
        for key, value in raw_paths.items()
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    """
    Create all configured directories if they do not exist.
    """
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)