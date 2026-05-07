import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from typing import cast

# Allow direct execution (python src/evaluation/evaluation_dqn_baselines.py) by adding
# project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.filters import ImageAction
from src.agents import DQNAgent
from src.data import get_dataset_name, get_effective_image_size, load_train_dataset
from src.data.degradation import degrade_image
from src.envs.env import ImageEnhancementEnv
from src.metrics import compute_psnr, compute_ssim
from src.evaluation.baselines import BASELINE_POLICIES, evaluate_baseline_policy
from src.evaluation.eval_types import AcceptanceChecks, PolicyMetrics, PolicyRow
from src.utils import load_config, sample_indices, build_train_eval_indices, apply_subset_limits


def choose_degradation_type(default_type: str, candidate_types: list[str], key: int) -> str:
    if default_type != "mixed":
        return default_type
    if not candidate_types:
        return "gaussian_noise"
    return candidate_types[key % len(candidate_types)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN against classical baselines on fixed eval split.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (defaults to latest best).")
    parser.add_argument("--num-images", type=int, default=50, help="Number of eval images sampled from checkpoint eval split.")
    parser.add_argument("--output-name", type=str, default="evaluation_baselines.json", help="Output filename under run log dir.")
    parser.add_argument("--action-analysis-file", type=str, default="action_analysis.json", help="Action analysis file name under run log dir.")
    parser.add_argument("--degradation-type", type=str, default="", help="Override degradation type (e.g. gaussian_noise, combined).")
    parser.add_argument("--noise-std", type=float, default=-1.0, help="Override degradation noise std.")
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


def infer_use_dueling_from_checkpoint(checkpoint: dict) -> bool:
    state_dict = checkpoint.get("policy_net_state_dict", {})
    if not isinstance(state_dict, dict):
        return False
    return any(k.startswith("value_head.") or k.startswith("advantage_head.") for k in state_dict.keys())


def build_env_for_image(
    clean_image,
    max_steps,
    image_size: tuple[int, int],
    reward_metric,
    reward_cfg,
    degradation_type: str,
    noise_std: float,
):
    degraded = degrade_image(
        clean_image,
        degradation_type=degradation_type,
        noise_std=noise_std,
    )
    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded,
        max_steps=max_steps,
        image_size=image_size,
        reward_metric=reward_metric,
        step_penalty=float(reward_cfg.get("step_penalty", 0.01)),
        repeated_action_penalty=float(reward_cfg.get("repeated_action_penalty", 0.0)),
        no_improvement_penalty=float(reward_cfg.get("no_improvement_penalty", 0.0)),
        stop_bonus_scale=float(reward_cfg.get("stop_bonus_scale", 0.0)),
        stop_no_improvement_penalty=float(reward_cfg.get("stop_no_improvement_penalty", 0.0)),
        early_stop_min_improvement=float(reward_cfg.get("early_stop_min_improvement", 0.0)),
    )


def aggregate_metrics(rows: list[PolicyRow]) -> PolicyMetrics:
    keys = ["psnr_enhanced", "ssim_enhanced", "delta_psnr", "delta_ssim"]
    out: dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in rows]
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"std_{key}"] = float(np.std(values))
    return {
        "mean_psnr_enhanced": out["mean_psnr_enhanced"],
        "std_psnr_enhanced": out["std_psnr_enhanced"],
        "mean_ssim_enhanced": out["mean_ssim_enhanced"],
        "std_ssim_enhanced": out["std_ssim_enhanced"],
        "mean_delta_psnr": out["mean_delta_psnr"],
        "std_delta_psnr": out["std_delta_psnr"],
        "mean_delta_ssim": out["mean_delta_ssim"],
        "std_delta_ssim": out["std_delta_ssim"],
    }


def main() -> None:
    args = parse_args()

    dataset_cfg = load_config("configs/dataset.yaml")
    env_all = load_config("configs/environment.yaml")
    train_all = load_config("configs/training.yaml")

    env_cfg = env_all.get("environment", {})
    reward_cfg = env_all.get("reward", {})
    train_cfg = train_all.get("training", {})
    dataset_core_cfg = dataset_cfg.get("dataset", {})
    degradation_cfg = dataset_cfg.get("degradation", {})
    default_degradation_type = degradation_cfg.get("type", "gaussian_noise")
    candidate_degradation_types = degradation_cfg.get("candidate_types", [])
    if not isinstance(candidate_degradation_types, list):
        candidate_degradation_types = []
    if args.degradation_type:
        default_degradation_type = args.degradation_type
        candidate_degradation_types = [args.degradation_type]
    noise_std = float(args.noise_std if args.noise_std >= 0.0 else degradation_cfg.get("noise_std", 0.1))

    max_steps = int(env_cfg.get("max_steps", 5))
    dataset_image_size = get_effective_image_size(dataset_core_cfg)
    image_size = (dataset_image_size, dataset_image_size)
    reward_metric = "psnr" if bool(reward_cfg.get("use_psnr", True)) else "ssim"
    seed = int(train_cfg.get("seed", 42))
    collapse_threshold = float(train_cfg.get("action_collapse_threshold", 0.70))
    min_stop_rate = float(train_cfg.get("min_stop_rate", 0.10))
    eval_subset_size_cfg = int(dataset_core_cfg.get("eval_subset_size", 0) or 0)

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

    dataset_name = get_dataset_name(dataset_core_cfg)
    dataset = load_train_dataset(dataset_core_cfg, dataset_root=dataset_root)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eval_indices = checkpoint.get("eval_indices")
    if not eval_indices:
        train_indices_auto, eval_indices_auto = build_train_eval_indices(
            dataset_size=len(dataset),
            eval_pool_size=int(train_cfg.get("eval_pool_size", 500)),
            seed=seed,
        )
        _, eval_indices = apply_subset_limits(
            train_indices=train_indices_auto,
            eval_indices=eval_indices_auto,
            train_subset_size=int(dataset_core_cfg.get("train_subset_size", 0) or 0),
            eval_subset_size=eval_subset_size_cfg,
            seed=seed,
        )

    eval_subset = sample_indices(eval_indices, k=args.num_images, seed=seed + 1234)
    if not eval_subset:
        raise RuntimeError("Empty eval subset: check eval indices and --num-images.")

    sample_img, _ = dataset[eval_subset[0]]
    sample_degradation_type = choose_degradation_type(
        default_type=default_degradation_type,
        candidate_types=candidate_degradation_types,
        key=eval_subset[0] + seed + 1234,
    )
    sample_env = build_env_for_image(
        sample_img.convert("RGB"),
        max_steps,
        image_size,
        reward_metric,
        reward_cfg,
        sample_degradation_type,
        noise_std,
    )

    use_dueling_dqn = bool(checkpoint.get("use_dueling_dqn", infer_use_dueling_from_checkpoint(checkpoint)))
    action_space = cast(spaces.Discrete, sample_env.action_space)
    obs_shape = sample_env.observation_space.shape
    if obs_shape is None:
        raise RuntimeError("Observation space shape is None; cannot infer in_channels.")

    agent = DQNAgent(
        num_actions=int(action_space.n),
        in_channels=int(obs_shape[-1]),
        use_dueling_dqn=use_dueling_dqn,
        epsilon=0.0,
    )
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.epsilon = 0.0

    per_policy_rows: dict[str, list[PolicyRow]] = defaultdict(list)

    for offset, idx in enumerate(eval_subset):
        clean_image, _ = dataset[idx]
        clean_image = clean_image.convert("RGB")
        degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=idx + seed + 30000 + offset,
        )

        env = build_env_for_image(
            clean_image,
            max_steps,
            image_size,
            reward_metric,
            reward_cfg,
            degradation_type,
            noise_std,
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
        final_image = env.current_image
        if final_image is None:
            raise RuntimeError("Environment returned None current_image during evaluation rollout.")
        dqn_enhanced = final_image.copy()

        psnr_degraded = compute_psnr(degraded_eval, clean_eval)
        ssim_degraded = compute_ssim(degraded_eval, clean_eval)
        psnr_dqn = compute_psnr(dqn_enhanced, clean_eval)
        ssim_dqn = compute_ssim(dqn_enhanced, clean_eval)

        per_policy_rows["dqn"].append(
            PolicyRow(
                {
                "psnr_enhanced": psnr_dqn,
                "ssim_enhanced": ssim_dqn,
                "delta_psnr": psnr_dqn - psnr_degraded,
                "delta_ssim": ssim_dqn - ssim_degraded,
                }
            )
        )

        for baseline_name, baseline_actions in BASELINE_POLICIES.items():
            metrics = evaluate_baseline_policy(
                clean_image=clean_eval,
                degraded_image=degraded_eval,
                actions=baseline_actions,
            )
            per_policy_rows[baseline_name].append(
                PolicyRow(
                    {
                    "psnr_enhanced": metrics["psnr_enhanced"],
                    "ssim_enhanced": metrics["ssim_enhanced"],
                    "delta_psnr": metrics["delta_psnr"],
                    "delta_ssim": metrics["delta_ssim"],
                    }
                )
            )

    aggregated: dict[str, PolicyMetrics] = {name: aggregate_metrics(rows) for name, rows in per_policy_rows.items()}
    dqn_metrics = aggregated.get("dqn")
    input_only_metrics = aggregated.get("input_only")
    if dqn_metrics is None or input_only_metrics is None:
        raise RuntimeError("Missing required policy metrics for dqn/input_only.")

    acceptance_checks: AcceptanceChecks = {
        "baseline_report_generated": True,
        "mean_delta_psnr_positive": bool(dqn_metrics["mean_delta_psnr"] > 0.0),
        "output_psnr_ge_input_psnr": bool(
            dqn_metrics["mean_psnr_enhanced"] >= input_only_metrics["mean_psnr_enhanced"]
        ),
        "dominant_action_share_ok": False,
        "stop_rate_ok": False,
        "action_analysis_available": False,
    }

    action_analysis_file = logs_root / "dqn" / checkpoint.get("run_id", checkpoint_path.parent.name) / args.action_analysis_file
    stop_rate = None
    dominant_action_share = None
    avg_episode_length = None
    if action_analysis_file.exists():
        with open(action_analysis_file, "r") as f:
            action_analysis = json.load(f)
        stop_rate = float(action_analysis.get("stop_rate", 0.0))
        dominant_action_share = float(action_analysis.get("dominant_action_share", 1.0))
        episode_length = action_analysis.get("episode_length", {})
        avg_episode_length = float(episode_length.get("avg", 0.0)) if isinstance(episode_length, dict) else None
        acceptance_checks["dominant_action_share_ok"] = bool(
            dominant_action_share <= collapse_threshold
        )
        acceptance_checks["stop_rate_ok"] = bool(
            stop_rate >= min_stop_rate
        )
        acceptance_checks["action_analysis_available"] = True

    acceptance_passed = all(bool(v) for v in acceptance_checks.values())

    print("\nEvaluation Results (Fixed Eval Split)")
    print("=" * 90)
    print(f"Dataset: {dataset_name} | image_size={image_size[0]}x{image_size[1]}")
    if eval_subset_size_cfg > 0:
        print(f"Configured eval_subset_size: {eval_subset_size_cfg}")
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
    out_file = out_dir / args.output_name

    with open(out_file, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "run_id": run_id,
                "eval_subset_size": len(eval_subset),
                "degradation": {
                    "type": default_degradation_type,
                    "candidate_types": candidate_degradation_types,
                    "noise_std": noise_std,
                },
                "input_psnr": input_only_metrics["mean_psnr_enhanced"],
                "output_psnr": dqn_metrics["mean_psnr_enhanced"],
                "mean_delta_psnr": dqn_metrics["mean_delta_psnr"],
                "input_ssim": input_only_metrics["mean_ssim_enhanced"],
                "output_ssim": dqn_metrics["mean_ssim_enhanced"],
                "mean_delta_ssim": dqn_metrics["mean_delta_ssim"],
                "stop_rate": stop_rate,
                "dominant_action_share": dominant_action_share,
                "avg_episode_length": avg_episode_length,
                "gate_metrics": {
                    "input_psnr": input_only_metrics["mean_psnr_enhanced"],
                    "output_psnr": dqn_metrics["mean_psnr_enhanced"],
                    "mean_delta_psnr": dqn_metrics["mean_delta_psnr"],
                    "input_ssim": input_only_metrics["mean_ssim_enhanced"],
                    "output_ssim": dqn_metrics["mean_ssim_enhanced"],
                    "mean_delta_ssim": dqn_metrics["mean_delta_ssim"],
                    "stop_rate": stop_rate,
                    "dominant_action_share": dominant_action_share,
                    "avg_episode_length": avg_episode_length,
                },
                "action_analysis_file": args.action_analysis_file,
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
