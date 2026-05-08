"""
Compare baseline methods on the current branch setup.

This script evaluates:
1. Classical method baselines
2. Fixed policy baselines
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.data import get_dataset_name, load_train_dataset
from src.data.degradation import degrade_image
from src.evaluation.baselines import evaluate_all_baselines, evaluate_all_method_baselines
from src.utils import load_config, sample_indices


def choose_degradation_type(default_type: str, candidate_types: list[str], key: int) -> str:
    if default_type != "mixed":
        return default_type
    if not candidate_types:
        return "gaussian_noise"
    return candidate_types[key % len(candidate_types)]


def compare_methods_on_dataset(
    dataset_name: str,
    dataset_root: str,
    num_samples: int,
) -> dict[str, Any]:
    dataset_cfg_all = load_config("configs/dataset.yaml")
    dataset_cfg = dataset_cfg_all.get("dataset", {})
    degradation_cfg = dataset_cfg_all.get("degradation", {})
    dataset_cfg["name"] = dataset_name

    default_degradation_type = str(degradation_cfg.get("type", "gaussian_noise"))
    candidate_degradation_types = degradation_cfg.get("candidate_types", [])
    if not isinstance(candidate_degradation_types, list):
        candidate_degradation_types = []

    dataset = load_train_dataset(dataset_cfg, dataset_root=dataset_root)
    normalized_name = get_dataset_name(dataset_cfg)
    sample_idxs = sample_indices(range(len(dataset)), k=min(num_samples, len(dataset)), seed=42)

    results: dict[str, Any] = {
        "dataset": normalized_name,
        "num_samples": len(sample_idxs),
        "methods": {},
        "per_sample": [],
    }

    method_metrics: dict[str, dict[str, list[float]]] = {}
    for method in [
        "identity",
        "histogram",
        "dcp",
        "policy_input_only",
        "policy_brightness_contrast_sharpen",
        "policy_brightness_gamma_contrast",
        "policy_denoise_sharpen",
    ]:
        method_metrics[method] = {"delta_psnr": [], "delta_ssim": []}

    for offset, sample_idx in enumerate(sample_idxs):
        clean_image, _ = dataset[sample_idx]
        clean_image = clean_image.convert("RGB")
        degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=sample_idx + offset + 42,
        )
        degraded_image = degrade_image(
            clean_image,
            degradation_type=degradation_type,
            noise_std=float(degradation_cfg.get("noise_std", 0.1)),
            brightness_factor=float(degradation_cfg.get("brightness_factor", 0.6)),
            contrast_factor=float(degradation_cfg.get("contrast_factor", 0.6)),
            blur_radius=float(degradation_cfg.get("blur_radius", 1.0)),
        )

        sample_result: dict[str, Any] = {
            "sample_idx": int(sample_idx),
            "degradation_type": degradation_type,
            "methods": {},
        }

        method_results = evaluate_all_method_baselines(clean_image, degraded_image)
        for method_name, metrics in method_results.items():
            sample_result["methods"][method_name] = {
                "delta_psnr": metrics.get("delta_psnr", 0.0),
                "delta_ssim": metrics.get("delta_ssim", 0.0),
            }
            method_metrics[method_name]["delta_psnr"].append(metrics.get("delta_psnr", 0.0))
            method_metrics[method_name]["delta_ssim"].append(metrics.get("delta_ssim", 0.0))

        policy_results = evaluate_all_baselines(clean_image, degraded_image)
        for policy_name, metrics in policy_results.items():
            prefixed_name = f"policy_{policy_name}"
            sample_result["methods"][prefixed_name] = {
                "delta_psnr": metrics.get("delta_psnr", 0.0),
                "delta_ssim": metrics.get("delta_ssim", 0.0),
            }
            method_metrics[prefixed_name]["delta_psnr"].append(metrics.get("delta_psnr", 0.0))
            method_metrics[prefixed_name]["delta_ssim"].append(metrics.get("delta_ssim", 0.0))

        results["per_sample"].append(sample_result)

    for method_name, metrics in method_metrics.items():
        if not metrics["delta_psnr"]:
            continue
        results["methods"][method_name] = {
            "mean_delta_psnr": float(np.mean(metrics["delta_psnr"])),
            "std_delta_psnr": float(np.std(metrics["delta_psnr"])),
            "mean_delta_ssim": float(np.mean(metrics["delta_ssim"])),
            "std_delta_ssim": float(np.std(metrics["delta_ssim"])),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline methods on the current branch setup.")
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "STL10"], default="CIFAR10")
    parser.add_argument("--dataset-root", type=str, help="Root directory for datasets.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/comparison")
    args = parser.parse_args()

    dataset_root = args.dataset_root or os.getenv("DATASET_ROOT", "datasets")
    results = compare_methods_on_dataset(
        dataset_name=args.dataset,
        dataset_root=dataset_root,
        num_samples=args.num_samples,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"comparison_{args.dataset}.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")
    for method_name, metrics in sorted(results["methods"].items()):
        print(
            f"{method_name:35s} "
            f"PSNR {metrics['mean_delta_psnr']:+.3f} +/- {metrics['std_delta_psnr']:.3f} | "
            f"SSIM {metrics['mean_delta_ssim']:+.4f} +/- {metrics['std_delta_ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
