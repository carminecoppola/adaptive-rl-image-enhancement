import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

# Allow direct execution (python src/evaluation/analyze_dqn_actions.py) by adding
# the project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.filters import ImageAction, get_action_name
from src.agents import DQNAgent
from src.data.degradation import degrade_image
from src.envs.image_enhancement_env import ImageEnhancementEnv
from src.utils import load_config, sample_indices, build_train_eval_indices


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DQN action behavior on fixed CIFAR-10 eval subset.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (defaults to latest run best checkpoint).")
    parser.add_argument("--num-images", type=int, default=50, help="Number of eval images sampled from fixed eval split.")
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

    eval_subset = sample_indices(eval_indices, k=args.num_images, seed=seed + 999)

    sample_img, _ = dataset[eval_subset[0]]
    sample_env = build_env_for_image(sample_img.convert("RGB"), max_steps, reward_metric, reward_cfg, degradation_cfg)
    agent = DQNAgent(num_actions=sample_env.action_space.n, epsilon=0.0)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.epsilon = 0.0

    action_counter = Counter()
    position_counter = defaultdict(Counter)
    sequence_counter = Counter()
    episode_lengths = []
    stop_count = 0

    for offset, idx in enumerate(eval_subset):
        clean, _ = dataset[idx]
        env = build_env_for_image(clean.convert("RGB"), max_steps, reward_metric, reward_cfg, degradation_cfg)
        state, _ = env.reset(seed=seed + 20000 + offset)

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

        episode_lengths.append(len(sequence))
        sequence_counter[tuple(sequence)] += 1

    total_actions = sum(action_counter.values())
    dominant_action, dominant_count = action_counter.most_common(1)[0]
    dominant_share = dominant_count / total_actions if total_actions else 0.0
    stop_rate = stop_count / total_actions if total_actions else 0.0

    print("\nDQN Action Analysis")
    print("=" * 80)
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
    out_file = out_dir / "action_analysis.json"

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
            },
            f,
            indent=2,
        )

    print(f"\nSaved action analysis report: {out_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
