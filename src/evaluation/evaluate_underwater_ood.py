from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from PIL import Image
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import DQNAgent
from src.evaluation.baselines_underwater import BASELINE_IDS, get_baseline
from src.evaluation.run_context import load_run_config_bundle
from src.metrics.underwater_no_reference import compute_uciqe, compute_uiqm_proxy
from src.training.dqn_training_helpers import build_env_for_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate underwater checkpoint on UIEB challenging-60 without references.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--challenge-dir",
        default="",
        help="Optional override for UIEB challenging-60 directory. Defaults to DATASET_ROOT/UIEB/challenging-60.",
    )
    parser.add_argument(
        "--output-name",
        default="evaluation_ood_challenging60.json",
        help="Output filename under run log dir.",
    )
    return parser.parse_args()


def infer_use_dueling_from_checkpoint(checkpoint: dict) -> bool:
    state_dict = checkpoint.get("policy_net_state_dict", {})
    if not isinstance(state_dict, dict):
        return False
    return any(k.startswith("value_head.") or k.startswith("advantage_head.") for k in state_dict.keys())


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    metrics = ["uciqe", "uiqm_proxy", "delta_uciqe", "delta_uiqm_proxy"]
    out: dict[str, float] = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        out[f"mean_{metric}"] = float(np.mean(values))
        out[f"std_{metric}"] = float(np.std(values))
    return out


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_id = checkpoint.get("run_id", checkpoint_path.parent.name)
    dataset_cfg, env_all, train_all, run_log_dir = load_run_config_bundle(checkpoint_path, run_id)

    env_cfg = env_all.get("environment", {})
    reward_cfg = env_all.get("reward", {})
    train_cfg = train_all.get("training", {})
    action_set_name = str(env_cfg.get("action_set", "general"))
    if action_set_name != "underwater_v1":
        raise ValueError(f"OOD challenging-60 evaluation requires underwater_v1 action set, got: {action_set_name}")

    dataset_root = os.getenv("DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("DATASET_ROOT is not defined in .env")

    challenge_dir = Path(args.challenge_dir) if args.challenge_dir else Path(dataset_root) / "UIEB" / "challenging-60"
    if not challenge_dir.exists():
        raise FileNotFoundError(f"challenging-60 directory not found: {challenge_dir}")

    image_paths = sorted(
        [path for path in challenge_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not image_paths:
        raise RuntimeError(f"No challenge images found in {challenge_dir}")

    max_steps = int(env_cfg.get("max_steps", 5))
    image_size_val = int(dataset_cfg.get("dataset", {}).get("image_size", 128))
    image_size = (image_size_val, image_size_val)
    use_dueling_dqn = bool(checkpoint.get("use_dueling_dqn", infer_use_dueling_from_checkpoint(checkpoint)))

    sample_image = Image.open(image_paths[0]).convert("RGB")
    sample_env = build_env_for_image(
        clean_image=sample_image,
        max_steps=max_steps,
        image_size=image_size,
        reward_metric="psnr",
        step_penalty=float(reward_cfg.get("step_penalty", 0.01)),
        repeated_action_penalty=float(reward_cfg.get("repeated_action_penalty", 0.0)),
        no_improvement_penalty=float(reward_cfg.get("no_improvement_penalty", 0.0)),
        stop_bonus_scale=float(reward_cfg.get("stop_bonus_scale", 0.0)),
        stop_no_improvement_penalty=float(reward_cfg.get("stop_no_improvement_penalty", 0.0)),
        early_stop_min_improvement=float(reward_cfg.get("early_stop_min_improvement", 0.0)),
        truncate_without_stop_penalty=float(reward_cfg.get("truncate_without_stop_penalty", 0.0)),
        stop_action_bonus=float(reward_cfg.get("stop_action_bonus", 0.0)),
        terminal_reward_psnr_scale=float(reward_cfg.get("terminal_reward_psnr_scale", 0.0)),
        terminal_reward_ssim_scale=float(reward_cfg.get("terminal_reward_ssim_scale", 0.0)),
        include_step_channel=bool(env_cfg.get("include_step_channel", True)),
        action_set_name=action_set_name,
        degradation_type="none",
        noise_std=0.0,
        degraded_image=sample_image,
    )
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

    per_policy_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_image_results: list[dict[str, object]] = []
    action_counter = Counter()

    for offset, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        env = build_env_for_image(
            clean_image=image,
            max_steps=max_steps,
            image_size=image_size,
            reward_metric="psnr",
            step_penalty=float(reward_cfg.get("step_penalty", 0.01)),
            repeated_action_penalty=float(reward_cfg.get("repeated_action_penalty", 0.0)),
            no_improvement_penalty=float(reward_cfg.get("no_improvement_penalty", 0.0)),
            stop_bonus_scale=float(reward_cfg.get("stop_bonus_scale", 0.0)),
            stop_no_improvement_penalty=float(reward_cfg.get("stop_no_improvement_penalty", 0.0)),
            early_stop_min_improvement=float(reward_cfg.get("early_stop_min_improvement", 0.0)),
            truncate_without_stop_penalty=float(reward_cfg.get("truncate_without_stop_penalty", 0.0)),
            stop_action_bonus=float(reward_cfg.get("stop_action_bonus", 0.0)),
            terminal_reward_psnr_scale=float(reward_cfg.get("terminal_reward_psnr_scale", 0.0)),
            terminal_reward_ssim_scale=float(reward_cfg.get("terminal_reward_ssim_scale", 0.0)),
            include_step_channel=bool(env_cfg.get("include_step_channel", True)),
            action_set_name=action_set_name,
            degradation_type="none",
            noise_std=0.0,
            degraded_image=image,
        )
        state, _ = env.reset(seed=int(train_cfg.get("seed", 42)) + 40000 + offset)
        sequence: list[int] = []

        for _ in range(max_steps):
            action = agent.select_action(state)
            sequence.append(int(action))
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break

        enhanced = env.current_image
        if enhanced is None:
            raise RuntimeError("Environment returned None current_image during OOD rollout.")

        input_uciqe = compute_uciqe(image)
        output_uciqe = compute_uciqe(enhanced)
        input_uiqm = compute_uiqm_proxy(image)
        output_uiqm = compute_uiqm_proxy(enhanced)

        per_policy_rows["dqn"].append(
            {
                "uciqe": output_uciqe,
                "uiqm_proxy": output_uiqm,
                "delta_uciqe": output_uciqe - input_uciqe,
                "delta_uiqm_proxy": output_uiqm - input_uiqm,
            }
        )

        for baseline_id in BASELINE_IDS:
            baseline = get_baseline(baseline_id)
            enhanced_tensor = baseline(pil_to_tensor(image))
            enhanced_image = tensor_to_pil(enhanced_tensor)
            baseline_uciqe = compute_uciqe(enhanced_image)
            baseline_uiqm = compute_uiqm_proxy(enhanced_image)
            per_policy_rows[baseline_id].append(
                {
                    "uciqe": baseline_uciqe,
                    "uiqm_proxy": baseline_uiqm,
                    "delta_uciqe": baseline_uciqe - input_uciqe,
                    "delta_uiqm_proxy": baseline_uiqm - input_uiqm,
                }
            )

        for action in sequence:
            action_counter[str(action)] += 1

        per_image_results.append(
            {
                "image_name": image_path.name,
                "dqn_sequence": sequence,
                "input_uciqe": input_uciqe,
                "output_uciqe": output_uciqe,
                "delta_uciqe": output_uciqe - input_uciqe,
                "input_uiqm_proxy": input_uiqm,
                "output_uiqm_proxy": output_uiqm,
                "delta_uiqm_proxy": output_uiqm - input_uiqm,
            }
        )

    aggregated = {policy: aggregate_rows(rows) for policy, rows in per_policy_rows.items()}
    dqn_metrics = aggregated["dqn"]
    input_metrics = aggregated["input_only"]

    out_dir = run_log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.output_name
    with open(out_file, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "run_id": run_id,
                "challenge_dir": str(challenge_dir),
                "num_images": len(image_paths),
                "aggregated": aggregated,
                "summary": {
                    "dqn_mean_delta_uciqe": dqn_metrics["mean_delta_uciqe"],
                    "dqn_mean_delta_uiqm_proxy": dqn_metrics["mean_delta_uiqm_proxy"],
                    "input_mean_uciqe": input_metrics["mean_uciqe"],
                    "input_mean_uiqm_proxy": input_metrics["mean_uiqm_proxy"],
                },
                "per_image": per_image_results,
                "action_counter": dict(action_counter),
            },
            f,
            indent=2,
        )

    print(f"Saved OOD challenging-60 report: {out_file}")


if __name__ == "__main__":
    main()
