from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.training.dqn_types import EpisodeSummaryRow, EvalHistoryRow, ResolvedConfig, RunMeta


def build_checkpoint_payload(
    *,
    agent,
    num_actions: int,
    best_eval_reward: float,
    best_delta_psnr: float,
    best_eval_episode: int,
    best_eval_subset: list[int],
    use_double_dqn: bool,
    use_dueling_dqn: bool,
    seed: int,
    train_indices: list[int],
    eval_indices: list[int],
    run_id: str,
    image_size: tuple[int, int] | None = None,
    episode: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "policy_net_state_dict": agent.policy_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "epsilon": agent.epsilon,
        "num_actions": num_actions,
        "best_eval_reward": best_eval_reward,
        "best_delta_psnr": best_delta_psnr,
        "best_by_metric": "mean_delta_psnr",
        "best_eval_episode": best_eval_episode,
        "best_eval_subset": best_eval_subset,
        "use_double_dqn": use_double_dqn,
        "use_dueling_dqn": use_dueling_dqn,
        "seed": seed,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "run_id": run_id,
    }
    if image_size is not None:
        payload["image_size"] = [int(image_size[0]), int(image_size[1])]
    if episode is not None:
        payload["episode"] = episode
    return payload


def write_best_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def write_final_artifacts(
    *,
    run_log_dir: Path,
    run_ckpt_dir: Path,
    final_checkpoint_payload: dict[str, Any],
    episode_summary_rows: list[EpisodeSummaryRow],
    eval_history: list[EvalHistoryRow],
    seed: int,
    train_indices: list[int],
    eval_indices: list[int],
    dataset_config: dict[str, Any],
    env_config_all: dict[str, Any],
    training_config_all: dict[str, Any],
    resolved: ResolvedConfig,
    run_meta: RunMeta,
) -> Path:
    final_checkpoint_path = run_ckpt_dir / "dqn_final_policy_net.pt"
    torch.save(final_checkpoint_payload, final_checkpoint_path)

    episode_csv = run_log_dir / "episode_summary.csv"
    with open(episode_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "image_idx",
                "reward",
                "avg_loss",
                "epsilon",
                "steps",
                "action_entropy",
                "action_repeat_ratio",
                "stop_used",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_summary_rows)

    eval_json = run_log_dir / "eval_summary.json"
    with open(eval_json, "w") as f:
        json.dump(eval_history, f, indent=2)

    split_json = run_log_dir / "dataset_split.json"
    with open(split_json, "w") as f:
        json.dump({"seed": seed, "train_indices": train_indices, "eval_indices": eval_indices}, f)

    effective_config_json = run_log_dir / "effective_config.json"
    with open(effective_config_json, "w") as f:
        json.dump(
            {
                "dataset": dataset_config,
                "environment": env_config_all,
                "training": training_config_all,
                "resolved": resolved,
            },
            f,
            indent=2,
        )

    run_meta_json = run_log_dir / "run_meta.json"
    with open(run_meta_json, "w") as f:
        json.dump(run_meta, f, indent=2)

    return final_checkpoint_path
