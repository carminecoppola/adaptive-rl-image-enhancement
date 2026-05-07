import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from typing import cast

# Allow direct execution (python src/evaluation/analyze_dqn_actions.py) by adding
# the project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.filters import ImageAction, get_action_name
from src.agents import DQNAgent
from src.data import get_dataset_name, get_effective_image_size, load_train_dataset
from src.data.degradation import degrade_image
from src.envs.image_enhancement_env import ImageEnhancementEnv
from src.evaluation.eval_types import SampleActionRecord
from src.metrics import compute_psnr, compute_ssim
from src.utils import load_config, sample_indices, build_train_eval_indices, apply_subset_limits


def choose_degradation_type(default_type: str, candidate_types: list[str], key: int) -> str:
    if default_type != "mixed":
        return default_type
    if not candidate_types:
        return "gaussian_noise"
    return candidate_types[key % len(candidate_types)]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DQN action behavior on fixed eval subset.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (defaults to latest run best checkpoint).")
    parser.add_argument("--num-images", type=int, default=50, help="Number of eval images sampled from fixed eval split.")
    parser.add_argument("--output-name", type=str, default="action_analysis.json", help="Output filename under run log dir.")
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

    eval_subset = sample_indices(eval_indices, k=args.num_images, seed=seed + 999)
    if not eval_subset:
        raise RuntimeError("Empty eval subset: check eval indices and --num-images.")

    sample_img, _ = dataset[eval_subset[0]]
    sample_degradation_type = choose_degradation_type(
        default_type=default_degradation_type,
        candidate_types=candidate_degradation_types,
        key=eval_subset[0] + seed + 999,
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

    action_counter = Counter()
    position_counter = defaultdict(Counter)
    sequence_counter = Counter()
    episode_lengths = []
    stop_count = 0
    per_sample: list[SampleActionRecord] = []

    for offset, idx in enumerate(eval_subset):
        clean, _ = dataset[idx]
        degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=idx + seed + 20000 + offset,
        )
        env = build_env_for_image(
            clean.convert("RGB"),
            max_steps,
            image_size,
            reward_metric,
            reward_cfg,
            degradation_type,
            noise_std,
        )
        state, _ = env.reset(seed=seed + 20000 + offset)
        clean_eval = env.clean_image
        degraded_eval = env.initial_degraded_image.copy()
        psnr_in = compute_psnr(degraded_eval, clean_eval)
        ssim_in = compute_ssim(degraded_eval, clean_eval)

        sequence = []
        for step in range(max_steps):
            action = agent.select_action(state)
            action_name = get_action_name(action)

            if action == int(ImageAction.STOP):
                stop_count += 1

            sequence.append(action_name)
            action_counter[action_name] += 1
            position_counter[step + 1][action_name] += 1

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break

        final_image = env.current_image
        if final_image is None:
            raise RuntimeError("Environment returned None current_image during action analysis rollout.")
        enhanced_eval = final_image.copy()
        psnr_out = compute_psnr(enhanced_eval, clean_eval)
        ssim_out = compute_ssim(enhanced_eval, clean_eval)
        episode_lengths.append(len(sequence))
        sequence_counter[tuple(sequence)] += 1
        per_sample.append(
            SampleActionRecord(
                {
                "sample_index": int(idx),
                "degradation_type": degradation_type,
                "sequence": sequence,
                "episode_length": int(len(sequence)),
                "input_psnr": float(psnr_in),
                "output_psnr": float(psnr_out),
                "delta_psnr": float(psnr_out - psnr_in),
                "input_ssim": float(ssim_in),
                "output_ssim": float(ssim_out),
                "delta_ssim": float(ssim_out - ssim_in),
                }
            )
        )

    total_actions = sum(action_counter.values())
    dominant_action, dominant_count = action_counter.most_common(1)[0]
    dominant_share = dominant_count / total_actions if total_actions else 0.0
    stop_rate = stop_count / total_actions if total_actions else 0.0

    print("\nDQN Action Analysis")
    print("=" * 80)
    print(f"Dataset: {dataset_name} | image_size={image_size[0]}x{image_size[1]}")
    if eval_subset_size_cfg > 0:
        print(f"Configured eval_subset_size: {eval_subset_size_cfg}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint selection metric: {checkpoint.get('best_by_metric', 'n/a')}")
    if "best_delta_psnr" in checkpoint:
        print(f"Checkpoint best delta PSNR: {checkpoint['best_delta_psnr']:+.4f}")
    print(f"Eval subset size: {len(eval_subset)}")

    print("\nOverall action frequency:")
    for action, count in action_counter.most_common():
        print(f"  {action:25s}: {count}")

    print("\nAction frequency by step position:")
    for step in sorted(position_counter.keys()):
        print(f"\nStep {step}:")
        for action, count in position_counter[step].most_common():
            print(f"  {action:25s}: {count}")

    print("\nMost common action sequences:")
    for sequence, count in sequence_counter.most_common(10):
        print(f"  [{count:02d}x] {' -> '.join(sequence)}")

    avg_length = sum(episode_lengths) / len(episode_lengths)
    print("\nEpisode length:")
    print(f"  Average length: {avg_length:.2f}")
    print(f"  Min length    : {min(episode_lengths)}")
    print(f"  Max length    : {max(episode_lengths)}")

    print("\nAcceptance checks:")
    print(f"  Dominant action share: {dominant_share:.3f} (threshold <= {collapse_threshold:.3f})")
    print(f"  Stop rate            : {stop_rate:.3f} (target >= {min_stop_rate:.3f})")

    passed = dominant_share <= collapse_threshold and stop_rate >= min_stop_rate
    print(f"  PASS                 : {passed}")

    run_id = checkpoint.get("run_id", checkpoint_path.parent.name)
    out_dir = logs_root / "dqn" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.output_name
    top_sequences = [
        {"sequence": list(seq), "count": int(cnt)}
        for seq, cnt in sequence_counter.most_common(10)
    ]
    position_counts = {
        str(step): dict(counter)
        for step, counter in sorted(position_counter.items())
    }
    best_samples = sorted(per_sample, key=lambda x: x["delta_psnr"], reverse=True)[:5]
    worst_samples = sorted(per_sample, key=lambda x: x["delta_psnr"])[:5]

    with open(out_file, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "run_id": run_id,
                "eval_subset_size": len(eval_subset),
                "dominant_action": dominant_action,
                "dominant_action_share": dominant_share,
                "stop_rate": stop_rate,
                "collapse_threshold": collapse_threshold,
                "min_stop_rate": min_stop_rate,
                "checkpoint_best_by_metric": checkpoint.get("best_by_metric"),
                "checkpoint_best_delta_psnr": checkpoint.get("best_delta_psnr"),
                "passed": passed,
                "action_counter": dict(action_counter),
                "action_counter_by_step": position_counts,
                "top_sequences": top_sequences,
                "episode_length": {
                    "avg": float(avg_length),
                    "min": int(min(episode_lengths)),
                    "max": int(max(episode_lengths)),
                },
                "per_sample": per_sample,
                "best_samples_by_delta_psnr": best_samples,
                "worst_samples_by_delta_psnr": worst_samples,
                "degradation": {
                    "type": default_degradation_type,
                    "candidate_types": candidate_degradation_types,
                    "noise_std": noise_std,
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved action analysis report: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
