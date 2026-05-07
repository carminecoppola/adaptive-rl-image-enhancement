"""
Compare experiment results against baseline methods and policy baselines.

This script evaluates the performance of:
1. RL agent trained with different experiments (A/B/C)
2. Classical baseline methods (DCP, histogram equalization, identity)
3. Policy baselines (fixed action sequences)

Usage:
    python scripts/compare_experiment_results.py \
        --exp-a <exp_A_checkpoint> \
        --exp-b <exp_B_checkpoint> \
        --exp-c <exp_C_checkpoint> \
        --dataset CIFAR10 \
        --num-samples 50 \
        --output results/comparison
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data import get_dataset_name, load_train_dataset, get_effective_image_size
from src.agents import DQNAgent
from src.evaluation.baselines import (
    evaluate_all_method_baselines,
    evaluate_all_baselines,
)
from src.metrics import compute_psnr, compute_ssim
from src.metrics.color_cast import compute_color_cast_score
from src.training.dqn_training_helpers import build_env_for_image
from src.utils import load_config, sample_indices


def evaluate_agent_on_sample(
    agent: DQNAgent,
    clean_image: Image.Image,
    degraded_image: Image.Image | None,
    max_steps: int = 5,
    degradation_type: str = "gaussian_noise",
) -> dict[str, float]:
    """
    Evaluate RL agent on a single image.
    
    Returns metrics for the enhanced image.
    """
    env = build_env_for_image(
        clean_image=clean_image.convert("RGB"),
        max_steps=max_steps,
        image_size=(clean_image.size[0], clean_image.size[1]),
        reward_metric="psnr",
        step_penalty=0.011,
        repeated_action_penalty=0.05,
        no_improvement_penalty=0.03,
        stop_bonus_scale=0.3,
        stop_no_improvement_penalty=0.04,
        early_stop_min_improvement=0.01,
        truncate_without_stop_penalty=0.75,
        stop_action_bonus=0.04,
        terminal_reward_psnr_scale=1.5,
        terminal_reward_ssim_scale=0.0,
        color_cast_weight=0.15,
        color_cast_improvement_scale=0.5,
        include_step_channel=True,
        degradation_type=degradation_type,
        noise_std=0.1,
        degraded_image=degraded_image,
    )
    
    state, _ = env.reset(seed=42)
    
    # Rollout agent
    total_reward = 0.0
    steps_taken = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps_taken += 1
        if terminated or truncated:
            break
    
    enhanced_image = env.current_image
    degraded_image_input = env.initial_degraded_image
    
    psnr_degraded = compute_psnr(degraded_image_input, clean_image)
    psnr_enhanced = compute_psnr(enhanced_image, clean_image)
    ssim_degraded = compute_ssim(degraded_image_input, clean_image)
    ssim_enhanced = compute_ssim(enhanced_image, clean_image)
    color_cast_enhanced = compute_color_cast_score(enhanced_image)
    
    return {
        "total_reward": float(total_reward),
        "steps_taken": float(steps_taken),
        "psnr_degraded": float(psnr_degraded),
        "psnr_enhanced": float(psnr_enhanced),
        "delta_psnr": float(psnr_enhanced - psnr_degraded),
        "ssim_degraded": float(ssim_degraded),
        "ssim_enhanced": float(ssim_enhanced),
        "delta_ssim": float(ssim_enhanced - ssim_degraded),
        "color_cast_score": float(color_cast_enhanced),
    }


def compare_methods_on_dataset(
    dataset_name: str,
    dataset_root: str,
    num_samples: int = 50,
    agent_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """
    Compare all methods (RL agent, classical baselines, policy baselines) on a dataset.
    
    Returns:
        Dict with results for each method and per-sample metrics
    """
    # Load dataset
    dataset_cfg = load_config("configs/dataset.yaml").get("dataset", {})
    dataset_cfg["name"] = dataset_name
    
    dataset = load_train_dataset(dataset_cfg, dataset_root=dataset_root)
    
    # Sample indices
    sample_idxs = sample_indices(range(len(dataset)), k=min(num_samples, len(dataset)), seed=42)
    
    results = {
        "dataset": dataset_name,
        "num_samples": len(sample_idxs),
        "methods": {},
        "per_sample": [],
    }
    
    # Initialize method aggregators
    method_metrics = {}
    for method in ["rl_agent", "identity", "histogram", "dcp", "policy_input_only", "policy_brightness_contrast"]:
        method_metrics[method] = {metric: [] for metric in ["delta_psnr", "delta_ssim", "color_cast"]}
    
    print(f"Evaluating {len(sample_idxs)} samples from {dataset_name}...")
    
    for idx, sample_idx in enumerate(sample_idxs):
        if dataset_name == "UIEB":
            degraded_image, clean_image = dataset[sample_idx]
        else:
            clean_image, _ = dataset[sample_idx]
            degraded_image = None
        
        clean_image = clean_image.convert("RGB")
        if degraded_image:
            degraded_image = degraded_image.convert("RGB")
        
        sample_result = {"sample_idx": sample_idx, "methods": {}}
        
        # Evaluate classical method baselines
        method_results = evaluate_all_method_baselines(clean_image, degraded_image or clean_image)
        
        for method_name, metrics in method_results.items():
            if metrics:
                sample_result["methods"][method_name] = {
                    "delta_psnr": metrics.get("delta_psnr", 0.0),
                    "delta_ssim": metrics.get("delta_ssim", 0.0),
                }
                method_metrics[method_name]["delta_psnr"].append(metrics.get("delta_psnr", 0.0))
                method_metrics[method_name]["delta_ssim"].append(metrics.get("delta_ssim", 0.0))
        
        # Evaluate policy baselines
        policy_results = evaluate_all_baselines(clean_image, degraded_image or clean_image)
        for policy_name, metrics in policy_results.items():
            sample_result["methods"][f"policy_{policy_name}"] = {
                "delta_psnr": metrics.get("delta_psnr", 0.0),
                "delta_ssim": metrics.get("delta_ssim", 0.0),
            }
            method_metrics[f"policy_{policy_name}"]["delta_psnr"].append(metrics.get("delta_psnr", 0.0))
            method_metrics[f"policy_{policy_name}"]["delta_ssim"].append(metrics.get("delta_ssim", 0.0))
        
        results["per_sample"].append(sample_result)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(sample_idxs)} samples")
    
    # Aggregate results
    for method_name, metrics in method_metrics.items():
        if metrics["delta_psnr"]:
            results["methods"][method_name] = {
                "mean_delta_psnr": float(np.mean(metrics["delta_psnr"])),
                "std_delta_psnr": float(np.std(metrics["delta_psnr"])),
                "mean_delta_ssim": float(np.mean(metrics["delta_ssim"])),
                "std_delta_ssim": float(np.std(metrics["delta_ssim"])),
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results against baselines"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR10", "STL10", "UIEB"],
        default="CIFAR10",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Root directory for datasets (defaults to DATASET_ROOT env var)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    import os
    dataset_root = args.dataset_root or os.getenv("DATASET_ROOT", "datasets")
    
    # Run comparison
    results = compare_methods_on_dataset(
        dataset_name=args.dataset,
        dataset_root=dataset_root,
        num_samples=args.num_samples,
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"comparison_{args.dataset}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {results['num_samples']}")
    print("\nMethod Performance (mean ± std):")
    print("-"*80)
    
    for method_name, metrics in sorted(results["methods"].items()):
        mean_psnr = metrics["mean_delta_psnr"]
        std_psnr = metrics["std_delta_psnr"]
        mean_ssim = metrics["mean_delta_ssim"]
        std_ssim = metrics["std_delta_ssim"]
        print(f"{method_name:30s} | PSNR: {mean_psnr:6.2f} ± {std_psnr:5.2f} | SSIM: {mean_ssim:6.3f} ± {std_ssim:5.3f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
