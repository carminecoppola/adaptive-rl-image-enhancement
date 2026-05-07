#!/usr/bin/env python3
"""Compare intrinsic visual quality across candidate datasets.

Outputs a console table with:
- native resolution statistics
- edge-detail proxy (Laplacian variance)
- practical recommendation for RL image-enhancement experiments
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, STL10, OxfordIIITPet, Food101

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.splits import sample_indices


@dataclass
class DatasetResult:
    name: str
    status: str
    count: int
    mean_width: float
    mean_height: float
    mean_laplacian_var: float
    note: str


Loader = Callable[[str, bool], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare visual quality of candidate datasets.")
    parser.add_argument("--num-images", type=int, default=200, help="Number of samples per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "stl10", "oxford_pets", "food101"],
        help="Datasets to evaluate: cifar10 stl10 oxford_pets food101",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow torchvision datasets download if missing locally.",
    )
    return parser.parse_args()


def laplacian_variance_uint8(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def to_rgb_pil(sample) -> Image.Image:
    if isinstance(sample, tuple):
        image = sample[0]
    else:
        image = sample

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # Numpy fallback
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def load_cifar10(root: str, allow_download: bool):
    return CIFAR10(root=root, train=True, download=allow_download)


def load_stl10(root: str, allow_download: bool):
    return STL10(root=root, split="train", download=allow_download)


def load_oxford_pets(root: str, allow_download: bool):
    return OxfordIIITPet(root=root, split="trainval", download=allow_download)


def load_food101(root: str, allow_download: bool):
    return Food101(root=root, split="train", download=allow_download)


LOADERS: dict[str, Loader] = {
    "cifar10": load_cifar10,
    "stl10": load_stl10,
    "oxford_pets": load_oxford_pets,
    "food101": load_food101,
}


def evaluate_dataset(name: str, root: str, num_images: int, seed: int, allow_download: bool) -> DatasetResult:
    if name not in LOADERS:
        return DatasetResult(name=name, status="ERROR", count=0, mean_width=0.0, mean_height=0.0, mean_laplacian_var=0.0, note="unknown dataset key")

    try:
        ds = LOADERS[name](root, allow_download)
    except Exception as exc:
        return DatasetResult(name=name, status="MISSING", count=0, mean_width=0.0, mean_height=0.0, mean_laplacian_var=0.0, note=str(exc).splitlines()[0][:180])

    if len(ds) == 0:
        return DatasetResult(name=name, status="EMPTY", count=0, mean_width=0.0, mean_height=0.0, mean_laplacian_var=0.0, note="dataset length is 0")

    idxs = sample_indices(list(range(len(ds))), k=min(num_images, len(ds)), seed=seed)
    widths: list[int] = []
    heights: list[int] = []
    lap_vars: list[float] = []

    for idx in idxs:
        rgb_pil = to_rgb_pil(ds[idx])
        widths.append(rgb_pil.width)
        heights.append(rgb_pil.height)
        lap_vars.append(laplacian_variance_uint8(np.asarray(rgb_pil, dtype=np.uint8)))

    return DatasetResult(
        name=name,
        status="OK",
        count=len(idxs),
        mean_width=float(np.mean(widths)),
        mean_height=float(np.mean(heights)),
        mean_laplacian_var=float(np.mean(lap_vars)),
        note="",
    )


def recommend(results: list[DatasetResult]) -> str:
    ok = [r for r in results if r.status == "OK"]
    if not ok:
        return "No candidate dataset available locally. Re-run with --allow-download or prepare datasets manually."

    # Prefer higher resolution first, then detail proxy.
    ranked = sorted(ok, key=lambda r: (r.mean_width * r.mean_height, r.mean_laplacian_var), reverse=True)
    best = ranked[0]

    if best.name == "cifar10":
        return "Only CIFAR-10 is currently usable; keep it for policy logic tests, but visual quality remains limited."

    if best.name == "stl10":
        return "Recommended next step: STL10. It is much higher resolution than CIFAR-10 while remaining lightweight for fast RL iteration."

    if best.name == "oxford_pets":
        return "Recommended next step: Oxford-IIIT Pet. Good visual quality and semantic consistency; suitable for clearer enhancement demos."

    if best.name == "food101":
        return "Recommended next step: Food101. High visual richness, but may increase domain complexity for policy learning."

    return f"Recommended next step: {best.name}."


def main() -> None:
    args = parse_args()
    data_root = os.getenv("DATASET_ROOT")
    if not data_root:
        raise ValueError("DATASET_ROOT is not set. Please load .env first.")

    results = [
        evaluate_dataset(
            name=name,
            root=data_root,
            num_images=args.num_images,
            seed=args.seed,
            allow_download=args.allow_download,
        )
        for name in args.datasets
    ]

    print("=" * 110)
    print("DATASET VISUAL QUALITY COMPARISON")
    print("=" * 110)
    print(f"data_root: {data_root}")
    print(f"samples per dataset (max): {args.num_images}")
    print()

    header = f"{'dataset':15s} {'status':10s} {'n':>5s} {'mean_w':>8s} {'mean_h':>8s} {'lap_var':>12s}  note"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:15s} {r.status:10s} {r.count:5d} {r.mean_width:8.1f} {r.mean_height:8.1f} {r.mean_laplacian_var:12.2f}  {r.note}"
        )

    print()
    print("Recommendation:")
    print(f"- {recommend(results)}")

    print()
    print("Interpretation rule of thumb:")
    print("- CIFAR-10 (32x32) is ideal for quick policy-intelligence checks but weak for visual-quality demos.")
    print("- For better visual readability, prioritize datasets >= 96x96.")
    print("=" * 110)


if __name__ == "__main__":
    main()
