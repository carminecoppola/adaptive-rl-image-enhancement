from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_run_log_dir(checkpoint_path: Path, run_id: str) -> Path:
    candidates = []
    env_logs = os.getenv("LOGS_ROOT")
    if env_logs:
        candidates.append(Path(env_logs) / "dqn" / run_id)
    try:
        root = checkpoint_path.parents[2].parent
        candidates.append(root / "logs" / "dqn" / run_id)
    except IndexError:
        pass
    candidates.append(PROJECT_ROOT / "logs" / "dqn" / run_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_run_config_bundle(
    checkpoint_path: Path,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """
    Load dataset/environment/training config bundle for a checkpointed run.

    Preference order:
    1. effective_config.json stored with the run
    2. current canonical underwater experiment config
    """
    run_log_dir = resolve_run_log_dir(checkpoint_path, run_id)
    effective_config_path = run_log_dir / "effective_config.json"
    if effective_config_path.exists():
        with effective_config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        canonical = load_config("configs/experiments/underwater_dqn_v1.yaml")
        return (
            payload.get(
                "dataset",
                {
                    "dataset": canonical.get("dataset", {}),
                    "degradation": canonical.get("degradation", {}),
                },
            ),
            payload.get("environment", canonical),
            payload.get("training", canonical),
            run_log_dir,
        )

    canonical = load_config("configs/experiments/underwater_dqn_v1.yaml")
    return (
        {"dataset": canonical.get("dataset", {}), "degradation": canonical.get("degradation", {})},
        canonical,
        canonical,
        run_log_dir,
    )
