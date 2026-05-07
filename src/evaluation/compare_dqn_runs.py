import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DQN runs from log artifacts.")
    parser.add_argument(
        "--logs-root",
        type=str,
        default=os.getenv("LOGS_ROOT", "logs"),
        help="Root logs directory containing dqn/<run_id>/ artifacts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path. Defaults to <logs-root>/dqn/run_comparison.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    dqn_root = logs_root / "dqn"
    if not dqn_root.exists():
        raise FileNotFoundError(f"DQN logs folder not found: {dqn_root}")

    rows = []
    for run_dir in sorted(dqn_root.glob("dqn_*")):
        if not run_dir.is_dir():
            continue

        run_meta = load_json(run_dir / "run_meta.json")
        action = load_json(run_dir / "action_analysis.json")
        baseline = load_json(run_dir / "evaluation_baselines.json")
        acceptance_checks = baseline.get("acceptance_checks", {})
        gate_metrics = baseline.get("gate_metrics", {})

        aggregated = baseline.get("aggregated", {})
        dqn_metrics = aggregated.get("dqn", {})
        input_metrics = aggregated.get("input_only", {})

        rows.append(
            {
                "run_id": run_dir.name,
                "best_eval_reward": run_meta.get("best_eval_reward"),
                "best_delta_psnr": run_meta.get("best_delta_psnr"),
                "best_by_metric": run_meta.get("best_by_metric"),
                "action_analysis_passed": action.get("passed"),
                "dominant_action_share": gate_metrics.get("dominant_action_share", action.get("dominant_action_share")),
                "stop_rate": gate_metrics.get("stop_rate", action.get("stop_rate")),
                "avg_episode_length": gate_metrics.get("avg_episode_length"),
                "mean_delta_psnr": gate_metrics.get("mean_delta_psnr", dqn_metrics.get("mean_delta_psnr")),
                "mean_delta_ssim": gate_metrics.get("mean_delta_ssim", dqn_metrics.get("mean_delta_ssim")),
                "input_psnr": gate_metrics.get("input_psnr", input_metrics.get("mean_psnr_enhanced")),
                "output_psnr": gate_metrics.get("output_psnr", dqn_metrics.get("mean_psnr_enhanced")),
                "input_ssim": gate_metrics.get("input_ssim", input_metrics.get("mean_ssim_enhanced")),
                "output_ssim": gate_metrics.get("output_ssim", dqn_metrics.get("mean_ssim_enhanced")),
                "output_psnr_ge_input_psnr": acceptance_checks.get("output_psnr_ge_input_psnr"),
                "stop_rate_ok": acceptance_checks.get("stop_rate_ok"),
                "dominant_action_share_ok": acceptance_checks.get("dominant_action_share_ok"),
                "acceptance_passed": baseline.get("acceptance_passed"),
            }
        )

    rows.sort(key=lambda r: (r.get("mean_delta_psnr") is not None, r.get("mean_delta_psnr", -1e9)), reverse=True)

    print(
        "run_id\tin_psnr\tout_psnr\tdelta_psnr\tin_ssim\tout_ssim\tdelta_ssim\tstop_rate\tdom_share\tavg_len\toutput_psnr_ge_input_psnr\t"
        "stop_rate_ok\tdominant_action_share_ok\taction_pass\tacceptance_pass"
    )
    for row in rows:
        print(
            f"{row['run_id']}\t"
            f"{row.get('input_psnr')}\t"
            f"{row.get('output_psnr')}\t"
            f"{row.get('mean_delta_psnr')}\t"
            f"{row.get('input_ssim')}\t"
            f"{row.get('output_ssim')}\t"
            f"{row.get('mean_delta_ssim')}\t"
            f"{row.get('stop_rate')}\t"
            f"{row.get('dominant_action_share')}\t"
            f"{row.get('avg_episode_length')}\t"
            f"{row.get('output_psnr_ge_input_psnr')}\t"
            f"{row.get('stop_rate_ok')}\t"
            f"{row.get('dominant_action_share_ok')}\t"
            f"{row.get('action_analysis_passed')}\t"
            f"{row.get('acceptance_passed')}"
        )

    output_path = Path(args.output) if args.output else dqn_root / "run_comparison.json"
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved run comparison: {output_path}")


if __name__ == "__main__":
    main()
