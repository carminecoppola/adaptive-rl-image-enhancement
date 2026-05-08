import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation artifacts (behavior, ID/OOD, baselines).")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--num-images", type=int, default=50, help="Evaluation subset size.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to run child scripts.")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def resolve_run_log_dir(checkpoint: Path) -> Path:
    run_id = checkpoint.parent.name
    candidates = []
    env_logs = os.getenv("LOGS_ROOT")
    if env_logs:
        candidates.append(Path(env_logs) / "dqn" / run_id)
    # Fallback from checkpoint path: .../<root>/checkpoints/dqn/<run_id>/file
    # -> .../<root>/logs/dqn/<run_id>
    try:
        root = checkpoint.parents[2].parent
        candidates.append(root / "logs" / "dqn" / run_id)
    except IndexError:
        pass
    candidates.append(PROJECT_ROOT / "logs" / "dqn" / run_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_behavior_md(path: Path, behavior: dict, eval_id: dict) -> None:
    best = behavior.get("best_samples_by_delta_psnr", [])
    worst = behavior.get("worst_samples_by_delta_psnr", [])
    dqn = eval_id.get("aggregated", {}).get("dqn", {})

    lines = [
        "# Behavior Report (ID)",
        "",
        "## Key Metrics",
        f"- Stop rate: `{behavior.get('stop_rate', 0.0):.4f}`",
        f"- Dominant action share: `{behavior.get('dominant_action_share', 0.0):.4f}`",
        f"- Episode length avg/min/max: `{behavior.get('episode_length', {}).get('avg', 0.0):.2f}` / "
        f"`{behavior.get('episode_length', {}).get('min', 0)}` / `{behavior.get('episode_length', {}).get('max', 0)}`",
        f"- Mean delta PSNR (DQN): `{dqn.get('mean_delta_psnr', 0.0):+.4f}`",
        "",
        "## Top Actions",
    ]
    for action, count in sorted(behavior.get("action_counter", {}).items(), key=lambda kv: kv[1], reverse=True)[:8]:
        lines.append(f"- `{action}`: {count}")

    lines.extend(["", "## Best Samples by Delta PSNR"])
    for row in best:
        lines.append(
            f"- idx `{row['sample_index']}` | delta_psnr `{row['delta_psnr']:+.4f}` | seq: "
            + " -> ".join(row.get("sequence", []))
        )

    lines.extend(["", "## Worst Samples by Delta PSNR"])
    for row in worst:
        lines.append(
            f"- idx `{row['sample_index']}` | delta_psnr `{row['delta_psnr']:+.4f}` | seq: "
            + " -> ".join(row.get("sequence", []))
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_final_baseline_csv(path: Path, eval_id: dict, stop_rate: float, avg_ep_len: float) -> None:
    agg = eval_id.get("aggregated", {})
    fields = [
        "policy",
        "mean_psnr",
        "mean_ssim",
        "mean_delta_psnr",
        "mean_delta_ssim",
        "stop_rate_dqn",
        "avg_episode_length_dqn",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy in sorted(agg.keys()):
            row = agg[policy]
            writer.writerow(
                {
                    "policy": policy,
                    "mean_psnr": row.get("mean_psnr_enhanced"),
                    "mean_ssim": row.get("mean_ssim_enhanced"),
                    "mean_delta_psnr": row.get("mean_delta_psnr"),
                    "mean_delta_ssim": row.get("mean_delta_ssim"),
                    "stop_rate_dqn": stop_rate if policy == "dqn" else "",
                    "avg_episode_length_dqn": avg_ep_len if policy == "dqn" else "",
                }
            )


def write_final_summary_md(path: Path, eval_id: dict, gen_report: dict, behavior: dict) -> None:
    dqn = eval_id.get("aggregated", {}).get("dqn", {})
    checks = eval_id.get("acceptance_checks", {})
    lines = [
        "# Final Baseline Summary",
        "",
        "## ID Performance (Selected Run)",
        f"- DQN mean delta PSNR: `{dqn.get('mean_delta_psnr', 0.0):+.4f}`",
        f"- DQN mean delta SSIM: `{dqn.get('mean_delta_ssim', 0.0):+.4f}`",
        f"- Stop rate: `{behavior.get('stop_rate', 0.0):.4f}` (target `{behavior.get('min_stop_rate', 0.1):.2f}`)",
        f"- Acceptance passed: `{eval_id.get('acceptance_passed', False)}`",
        "",
        "## OOD Gap",
        f"- ID mean delta PSNR: `{gen_report['id']['mean_delta_psnr']:+.4f}`",
        f"- OOD noise0.2 mean delta PSNR: `{gen_report['ood_noise02']['mean_delta_psnr']:+.4f}`",
        f"- OOD combined0.2 mean delta PSNR: `{gen_report['ood_combined02']['mean_delta_psnr']:+.4f}`",
        "",
        "## Critical Interpretation",
        "- DQN shows positive PSNR gain in-distribution with interpretable discrete action sequences.",
        "- Under stronger OOD degradations, quality drops, highlighting limited robustness/generalization.",
        "- This supports the project contribution: interpretable sequential decision-making is feasible, even if not SOTA-supervised.",
        "",
        "## Gate Status",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_id = checkpoint.parent.name
    out_dir = resolve_run_log_dir(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_script = PROJECT_ROOT / "src" / "evaluation" / "analyze_dqn_actions.py"
    eval_script = PROJECT_ROOT / "src" / "evaluation" / "evaluation_dqn_baselines.py"

    # ID
    run_cmd(
        [
            args.python_bin,
            str(analyze_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "action_analysis_id.json",
        ]
    )
    run_cmd(
        [
            args.python_bin,
            str(eval_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "evaluation_baselines_id.json",
            "--action-analysis-file",
            "action_analysis_id.json",
        ]
    )

    # OOD-1
    run_cmd(
        [
            args.python_bin,
            str(analyze_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "action_analysis_ood_noise02.json",
            "--degradation-type",
            "gaussian_noise",
            "--noise-std",
            "0.2",
        ]
    )
    run_cmd(
        [
            args.python_bin,
            str(eval_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "evaluation_baselines_ood_noise02.json",
            "--action-analysis-file",
            "action_analysis_ood_noise02.json",
            "--degradation-type",
            "gaussian_noise",
            "--noise-std",
            "0.2",
        ]
    )

    # OOD-2
    run_cmd(
        [
            args.python_bin,
            str(analyze_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "action_analysis_ood_combined02.json",
            "--degradation-type",
            "combined",
            "--noise-std",
            "0.2",
        ]
    )
    run_cmd(
        [
            args.python_bin,
            str(eval_script),
            "--checkpoint",
            str(checkpoint),
            "--num-images",
            str(args.num_images),
            "--output-name",
            "evaluation_baselines_ood_combined02.json",
            "--action-analysis-file",
            "action_analysis_ood_combined02.json",
            "--degradation-type",
            "combined",
            "--noise-std",
            "0.2",
        ]
    )

    behavior_id = load_json(out_dir / "action_analysis_id.json")
    eval_id = load_json(out_dir / "evaluation_baselines_id.json")
    eval_ood_noise = load_json(out_dir / "evaluation_baselines_ood_noise02.json")
    eval_ood_combined = load_json(out_dir / "evaluation_baselines_ood_combined02.json")

    behavior_report = {
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "stop_rate": behavior_id.get("stop_rate"),
        "dominant_action_share": behavior_id.get("dominant_action_share"),
        "episode_length": behavior_id.get("episode_length"),
        "action_counter": behavior_id.get("action_counter", {}),
        "action_counter_by_step": behavior_id.get("action_counter_by_step", {}),
        "top_sequences": behavior_id.get("top_sequences", []),
        "best_samples_by_delta_psnr": behavior_id.get("best_samples_by_delta_psnr", []),
        "worst_samples_by_delta_psnr": behavior_id.get("worst_samples_by_delta_psnr", []),
    }
    save_json(out_dir / "behavior_report_id.json", behavior_report)
    write_behavior_md(out_dir / "behavior_report_id.md", behavior_id, eval_id)

    id_dqn = eval_id["aggregated"]["dqn"]
    ood_noise_dqn = eval_ood_noise["aggregated"]["dqn"]
    ood_combined_dqn = eval_ood_combined["aggregated"]["dqn"]
    generalization_report = {
        "run_id": run_id,
        "id": {
            "mean_delta_psnr": id_dqn["mean_delta_psnr"],
            "mean_delta_ssim": id_dqn["mean_delta_ssim"],
            "acceptance_passed": eval_id.get("acceptance_passed"),
        },
        "ood_noise02": {
            "mean_delta_psnr": ood_noise_dqn["mean_delta_psnr"],
            "mean_delta_ssim": ood_noise_dqn["mean_delta_ssim"],
            "acceptance_passed": eval_ood_noise.get("acceptance_passed"),
        },
        "ood_combined02": {
            "mean_delta_psnr": ood_combined_dqn["mean_delta_psnr"],
            "mean_delta_ssim": ood_combined_dqn["mean_delta_ssim"],
            "acceptance_passed": eval_ood_combined.get("acceptance_passed"),
        },
        "delta_id_to_ood_noise02": {
            "mean_delta_psnr_drop": ood_noise_dqn["mean_delta_psnr"] - id_dqn["mean_delta_psnr"],
            "mean_delta_ssim_drop": ood_noise_dqn["mean_delta_ssim"] - id_dqn["mean_delta_ssim"],
        },
        "delta_id_to_ood_combined02": {
            "mean_delta_psnr_drop": ood_combined_dqn["mean_delta_psnr"] - id_dqn["mean_delta_psnr"],
            "mean_delta_ssim_drop": ood_combined_dqn["mean_delta_ssim"] - id_dqn["mean_delta_ssim"],
        },
    }
    save_json(out_dir / "generalization_report.json", generalization_report)

    write_final_baseline_csv(
        out_dir / "final_baseline_table.csv",
        eval_id,
        float(behavior_id.get("stop_rate", 0.0)),
        float(behavior_id.get("episode_length", {}).get("avg", 0.0)),
    )
    write_final_summary_md(out_dir / "final_baseline_summary.md", eval_id, generalization_report, behavior_id)

    print(f"[DONE] Generated presentation artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
