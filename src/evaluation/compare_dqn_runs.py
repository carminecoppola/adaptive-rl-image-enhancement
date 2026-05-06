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
                "dominant_action_share": action.get("dominant_action_share"),
                "stop_rate": action.get("stop_rate"),
                "mean_delta_psnr": dqn_metrics.get("mean_delta_psnr"),
                "mean_delta_ssim": dqn_metrics.get("mean_delta_ssim"),
                "mean_psnr_dqn": dqn_metrics.get("mean_psnr_enhanced"),
                "mean_psnr_input": input_metrics.get("mean_psnr_enhanced"),
                "acceptance_passed": baseline.get("acceptance_passed"),
            }
        )

    rows.sort(key=lambda r: (r.get("mean_delta_psnr") is not None, r.get("mean_delta_psnr", -1e9)), reverse=True)

    print("run_id\tbest_delta_psnr\tmean_delta_psnr\taction_pass\tacceptance_pass")
    for row in rows:
        print(
            f"{row['run_id']}\t"
            f"{row.get('best_delta_psnr')}\t"
            f"{row.get('mean_delta_psnr')}\t"
            f"{row.get('action_analysis_passed')}\t"
            f"{row.get('acceptance_passed')}"
        )

    output_path = Path(args.output) if args.output else dqn_root / "run_comparison.json"
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved run comparison: {output_path}")


if __name__ == "__main__":
    main()
