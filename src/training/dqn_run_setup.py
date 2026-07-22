from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class DQNRunPaths:
    run_id: str
    run_log_dir: Path
    run_ckpt_dir: Path


def create_run_paths(logs_root: Path, checkpoint_root: Path) -> DQNRunPaths:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
    run_id = os.getenv("RUN_ID", f"dqn_{timestamp}_{slurm_job_id}")

    run_log_dir = logs_root / "dqn" / run_id
    run_ckpt_dir = checkpoint_root / "dqn" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    return DQNRunPaths(
        run_id=run_id,
        run_log_dir=run_log_dir,
        run_ckpt_dir=run_ckpt_dir,
    )


def create_debug_writer(run_log_dir: Path) -> tuple[csv.DictWriter | None, TextIO | None, int]:
    debug_reward = os.getenv("DQN_DEBUG_REWARD", "0").strip() == "1"
    debug_episodes = int(os.getenv("DQN_DEBUG_EPISODES", "5"))
    if not debug_reward:
        return None, None, debug_episodes

    debug_path = run_log_dir / "reward_debug.csv"
    debug_file_handle = open(debug_path, "w", newline="")
    debug_writer = csv.DictWriter(
        debug_file_handle,
        fieldnames=[
            "episode",
            "step",
            "action",
            "action_name",
            "reward",
            "delta_quality",
            "step_penalty_applied",
            "repeated_penalty_applied",
            "no_improvement_penalty_applied",
            "stop_bonus_applied",
            "stop_no_improvement_penalty_applied",
            "previous_quality",
            "quality",
            "psnr",
            "ssim",
            "terminated",
            "truncated",
            "epsilon",
        ],
    )
    debug_writer.writeheader()
    print(f"[DEBUG] Reward diagnostics enabled -> {debug_path}")
    return debug_writer, debug_file_handle, debug_episodes
