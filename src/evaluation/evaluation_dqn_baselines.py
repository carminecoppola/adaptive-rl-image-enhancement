import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

# Allow direct execution (python src/evaluation/evaluation_dqn_baselines.py) by adding
# project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.filters import ImageAction
from src.agents import DQNAgent
from src.data.degradation import degrade_image
from src.envs.image_enhancement_env import ImageEnhancementEnv
from src.metrics import compute_psnr, compute_ssim
from src.evaluation.baselines import BASELINE_POLICIES, evaluate_baseline_policy
from src.utils import load_config, sample_indices, build_train_eval_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN against classical baselines on fixed CIFAR-10 split.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (defaults to latest best).")
    parser.add_argument("--num-images", type=int, default=50, help="Number of eval images sampled from checkpoint eval split.")
    return parser.parse_args()


def resolve_checkpoint(explicit: str, checkpoint_roots: list[Path]) -> Path:
    if explicit:
        cp = Path(explicit)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        return cp

    candidates: list[Path] = []
    for root in checkpoint_roots:
        candidates.extend(root.glob("dqn/*/dqn_best_policy_net.pt"))
        candidates.extend(root.glob("dqn/dqn_best_policy_net.pt"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        searched = ", ".join(str(root / "dqn") for root in checkpoint_roots)
        raise FileNotFoundError(f"No checkpoint found under: {searched}")
    return candidates[-1]


def build_env_for_image(clean_image, max_steps, reward_metric, reward_cfg, degradation_cfg):
    degraded = degrade_image(
        clean_image,
        degradation_type=degradation_cfg.get("type", "gaussian_noise"),
        noise_std=float(degradation_cfg.get("noise_std", 0.1)),
    )
    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded,
        max_steps=max_steps,
        image_size=(128, 128),
        reward_metric=reward_metric,
        step_penalty=float(reward_cfg.get("step_penalty", 0.01)),
        repeated_action_penalty=float(reward_cfg.get("repeated_action_penalty", 0.0)),
        no_improvement_penalty=float(reward_cfg.get("no_improvement_penalty", 0.0)),
        stop_bonus_scale=float(reward_cfg.get("stop_bonus_scale", 0.0)),
        stop_no_improvement_penalty=float(reward_cfg.get("stop_no_improvement_penalty", 0.0)),
        early_stop_min_improvement=float(reward_cfg.get("early_stop_min_improvement", 0.0)),
    )


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["psnr_enhanced", "ssim_enhanced", "delta_psnr", "delta_ssim"]
    out: dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in rows]
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"std_{key}"] = float(np.std(values))
    return out


def main() -> None:
    args = parse_args()

    dataset_cfg = load_config("configs/dataset.yaml")
    env_all = load_config("configs/environment.yaml")
    train_all = load_config("configs/training.yaml")

    env_cfg = env_all.get("environment", {})
    reward_cfg = env_all.get("reward", {})
    train_cfg = train_all.get("training", {})
    degradation_cfg = dataset_cfg.get("degradation", {})

    max_steps = int(env_cfg.get("max_steps", 5))
    reward_metric = "psnr" if bool(reward_cfg.get("use_psnr", True)) else "ssim"
    seed = int(train_cfg.get("seed", 42))
    collapse_threshold = float(train_cfg.get("action_collapse_threshold", 0.70))
    min_stop_rate = float(train_cfg.get("min_stop_rate", 0.10))

    checkpoint_root = Path(os.getenv("CHECKPOINT_ROOT", "checkpoints"))
    local_checkpoint_root = PROJECT_ROOT / "checkpoints"
    checkpoint_roots = [checkpoint_root]
    if local_checkpoint_root != checkpoint_root:
        checkpoint_roots.append(local_checkpoint_root)
    logs_root = Path(os.getenv("LOGS_ROOT", "logs"))
    checkpoint_path = resolve_checkpoint(args.checkpoint, checkpoint_roots)

    dataset_root = os.getenv("DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("DATASET_ROOT is not defined in .env")

    dataset = CIFAR10(root=dataset_root, train=True, download=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_indices = checkpoint.get("eval_indices")
    if not eval_indices:
        _, eval_indices = build_train_eval_indices(
            dataset_size=len(dataset),
            eval_pool_size=int(train_cfg.get("eval_pool_size", 500)),
            seed=seed,
        )

    eval_subset = sample_indices(eval_indices, k=args.num_images, seed=seed + 1234)

    sample_img, _ = dataset[eval_subset[0]]
    sample_env = build_env_for_image(sample_img.convert("RGB"), max_steps, reward_metric, reward_cfg, degradation_cfg)

    agent = DQNAgent(num_actions=sample_env.action_space.n, epsilon=0.0)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.epsilon = 0.0

    per_policy_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    for offset, idx in enumerate(eval_subset):
        clean_image, _ = dataset[idx]
        clean_image = clean_image.convert("RGB")

        env = build_env_for_image(
            clean_image,
            max_steps,
            reward_metric,
            reward_cfg,
            degradation_cfg,
        )
        state, _ = env.reset(seed=seed + 30000 + offset)
        clean_eval = env.clean_image
        degraded_eval = env.initial_degraded_image.copy()

        # Run DQN policy.
        current_state = state
        dqn_enhanced = degraded_eval
        for _ in range(max_steps):
            action = agent.select_action(current_state)
            next_state, _, terminated, truncated, _ = env.step(action)
            current_state = next_state
            if terminated or truncated:
                break
        dqn_enhanced = env.current_image.copy()

        psnr_degraded = compute_psnr(degraded_eval, clean_eval)
        ssim_degraded = compute_ssim(degraded_eval, clean_eval)
        psnr_dqn = compute_psnr(dqn_enhanced, clean_eval)
        ssim_dqn = compute_ssim(dqn_enhanced, clean_eval)

        per_policy_rows["dqn"].append(
            {
                "psnr_enhanced": psnr_dqn,
                "ssim_enhanced": ssim_dqn,
                "delta_psnr": psnr_dqn - psnr_degraded,
                "delta_ssim": ssim_dqn - ssim_degraded,
            }
        )

        for baseline_name, baseline_actions in BASELINE_POLICIES.items():
            metrics = evaluate_baseline_policy(
                clean_image=clean_eval,
                degraded_image=degraded_eval,
                actions=baseline_actions,
            )
            per_policy_rows[baseline_name].append(
                {
                    "psnr_enhanced": metrics["psnr_enhanced"],
                    "ssim_enhanced": metrics["ssim_enhanced"],
                    "delta_psnr": metrics["delta_psnr"],
                    "delta_ssim": metrics["delta_ssim"],
                }
            )

    aggregated = {name: aggregate_metrics(rows) for name, rows in per_policy_rows.items()}
    dqn_metrics = aggregated.get("dqn", {})
    input_only_metrics = aggregated.get("input_only", {})

    acceptance_checks = {
        "mean_delta_psnr_positive": bool(dqn_metrics.get("mean_delta_psnr", 0.0) > 0.0),
        "dqn_not_worse_than_input_psnr": bool(
            dqn_metrics.get("mean_psnr_enhanced", -float("inf")) >= input_only_metrics.get("mean_psnr_enhanced", float("inf"))
        ),
    }

    action_analysis_file = logs_root / "dqn" / checkpoint.get("run_id", checkpoint_path.parent.name) / "action_analysis.json"
    if action_analysis_file.exists():
        with open(action_analysis_file, "r") as f:
            action_analysis = json.load(f)
        acceptance_checks["dominant_action_share_ok"] = bool(
            float(action_analysis.get("dominant_action_share", 1.0)) <= collapse_threshold
        )
        acceptance_checks["stop_rate_ok"] = bool(
            float(action_analysis.get("stop_rate", 0.0)) >= min_stop_rate
        )
    else:
        acceptance_checks["dominant_action_share_ok"] = None
        acceptance_checks["stop_rate_ok"] = None

    available_checks = [v for v in acceptance_checks.values() if isinstance(v, bool)]
    acceptance_passed = bool(available_checks) and all(available_checks)

    print("\nEvaluation Results (Fixed CIFAR-10 Eval Split)")
    print("=" * 90)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval subset size: {len(eval_subset)}")

    header = f"{'Policy':30s} {'PSNR(mean±std)':22s} {'SSIM(mean±std)':22s} {'ΔPSNR':8s} {'ΔSSIM':8s}"
    print("\n" + header)
    print("-" * len(header))

    for name in sorted(aggregated.keys()):
        m = aggregated[name]
        print(
            f"{name:30s} "
            f"{m['mean_psnr_enhanced']:.3f}±{m['std_psnr_enhanced']:.3f}   "
            f"{m['mean_ssim_enhanced']:.3f}±{m['std_ssim_enhanced']:.3f}   "
            f"{m['mean_delta_psnr']:+.3f}   "
            f"{m['mean_delta_ssim']:+.3f}"
        )

    print("\nAcceptance checks:")
    for name, value in acceptance_checks.items():
        print(f"  {name:35s}: {value}")
    print(f"  {'acceptance_passed':35s}: {acceptance_passed}")

    run_id = checkpoint.get("run_id", checkpoint_path.parent.name)
    out_dir = logs_root / "dqn" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_baselines.json"

    with open(out_file, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "run_id": run_id,
                "eval_subset_size": len(eval_subset),
                "aggregated": aggregated,
                "acceptance_checks": acceptance_checks,
                "acceptance_passed": acceptance_passed,
            },
            f,
            indent=2,
        )

    print(f"\nSaved evaluation report: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
