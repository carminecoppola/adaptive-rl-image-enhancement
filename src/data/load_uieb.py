"""Load paired images from the Underwater Image Enhancement Benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test", "all"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class UIEBPair:
    """Paths for one degraded/reference image pair."""

    image_id: str
    degraded_path: Path
    reference_path: Path


def _find_existing_directory(root: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    expected = ", ".join(str(root / name) for name in candidates)
    raise FileNotFoundError(f"None of the expected UIEB directories exists: {expected}")


def _discover_pairs(root: Path) -> list[UIEBPair]:
    """Match each degraded image to its reference by filename stem.

    UIEB stores `raw/` and `reference/` as separate directories with the
    same filenames on both sides. Matching by stem (not by directory order)
    guarantees pair integrity even if one side is missing files or sorted
    differently — a silent off-by-one pairing here would corrupt every
    downstream PSNR/SSIM value without raising an error.
    """
    raw_dir = _find_existing_directory(root, ("raw", "raw-890"))
    reference_dir = _find_existing_directory(root, ("reference", "reference-890"))

    reference_by_stem = {
        path.stem: path
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    pairs = [
        UIEBPair(path.stem, path, reference_by_stem[path.stem])
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem in reference_by_stem
    ]
    if not pairs:
        raise FileNotFoundError(f"No matching degraded/reference image pairs found under {root}")
    return pairs


def _split_indices(size: int, split: SplitName, seed: int) -> list[int]:
    if split == "all":
        return list(range(size))
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test, all")

    indices = list(range(size))
    # A locally-seeded RNG (not the global one) makes the split reproducible
    # across processes/runs regardless of what else consumes randomness
    # before this call — the same seed always yields the same train/val/test
    # partition, which is required for comparing checkpoints across runs.
    random.Random(seed).shuffle(indices)
    train_end = max(1, int(size * 0.8))
    val_end = max(train_end + 1, int(size * 0.9)) if size > 1 else train_end
    val_end = min(val_end, size)

    split_indices = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }[split]
    return sorted(split_indices)


def _load_rgb(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return rgb.copy()


def _pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class UIEBDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Tensor-oriented UIEB dataset used by tests and batch analysis."""

    def __init__(
        self,
        root_path: str | Path,
        image_size: int = 128,
        split: SplitName = "train",
        subset_size: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root_path = Path(root_path).expanduser()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"UIEB root not found: {self.root_path}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive when provided")

        self.image_size = image_size
        self.pairs = _discover_pairs(self.root_path)
        self.indices = _split_indices(len(self.pairs), split, seed)
        if subset_size is not None:
            self.indices = self.indices[:subset_size]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, str]:
        pair = self.pairs[self.indices[item]]
        degraded = _load_rgb(pair.degraded_path, self.image_size)
        reference = _load_rgb(pair.reference_path, self.image_size)
        return _pil_to_tensor(degraded), _pil_to_tensor(reference), pair.image_id


class PairedUIEBDataset(Dataset[tuple[Image.Image, dict[str, Any]]]):
    """PIL-oriented UIEB dataset consumed by the sequential environment."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path).expanduser()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"UIEB root not found: {self.root_path}")
        self.pairs = _discover_pairs(self.root_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> tuple[Image.Image, dict[str, Any]]:
        pair = self.pairs[item]
        with Image.open(pair.degraded_path) as degraded_source:
            degraded = degraded_source.convert("RGB").copy()
        with Image.open(pair.reference_path) as reference_source:
            reference = reference_source.convert("RGB").copy()
        return degraded, {"reference_pil": reference, "image_id": pair.image_id}


def load_uieb_dataset(
    image_size: int = 128,
    subset_size: int | None = None,
    split: SplitName = "train",
    seed: int = 42,
    root_path: str | Path = "data/UIEB",
) -> tuple[Tensor, Tensor, list[str]]:
    """Load a UIEB split into stacked tensors and image identifiers."""
    dataset = UIEBDataset(
        root_path=root_path,
        image_size=image_size,
        split=split,
        subset_size=subset_size,
        seed=seed,
    )
    if len(dataset) == 0:
        empty = torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
        return empty, empty.clone(), []

    degraded, references, image_ids = zip(
        *(dataset[index] for index in range(len(dataset))), strict=True
    )
    return torch.stack(degraded), torch.stack(references), list(image_ids)


__all__ = ["PairedUIEBDataset", "UIEBDataset", "load_uieb_dataset"]
