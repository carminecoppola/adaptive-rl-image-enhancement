"""Dataset loading and synthetic degradation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.data.load_uieb import PairedUIEBDataset


def get_dataset_name(config: dict[str, Any]) -> str:
    """Return the normalized dataset identifier from a config mapping."""
    return str(config.get("name", "uieb")).strip().lower()


def get_effective_image_size(config: dict[str, Any]) -> int:
    """Return and validate the square image size used by the policy."""
    image_size = int(config.get("image_size", 128))
    if image_size <= 0:
        raise ValueError("dataset.image_size must be positive")
    return image_size


def _resolve_uieb_root(config: dict[str, Any], dataset_root: str | Path) -> Path:
    configured_path = config.get("path")
    if configured_path:
        configured = Path(str(configured_path)).expanduser()
        if configured.exists():
            return configured

    root = Path(dataset_root).expanduser()
    if (root / "raw").is_dir() or (root / "raw-890").is_dir():
        return root
    return root / "UIEB"


def load_train_dataset(
    config: dict[str, Any],
    dataset_root: str | Path,
) -> Dataset:
    """Load the paired dataset used by training and evaluation scripts."""
    dataset_name = get_dataset_name(config)
    if dataset_name != "uieb":
        raise ValueError(f"Unsupported dataset {dataset_name!r}; only 'uieb' is available")

    return PairedUIEBDataset(root_path=_resolve_uieb_root(config, dataset_root))


__all__ = [
    "PairedUIEBDataset",
    "get_dataset_name",
    "get_effective_image_size",
    "load_train_dataset",
]
