from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical underwater results report from run artifacts.")
    parser.add_argument("--run-dir", required=True, help="Run log directory containing evaluation artifacts.")
    parser.add_argument(
        "--output-markdown",
        default="underwater_results.md",
        help="Markdown report filename inside run-dir.",
    )
    parser.add_argument(
        "--output-json",
        default="underwater_results_summary.json",
        help="JSON summary filename inside run-dir.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_metric(payload: dict, *keys: str, default: float | None = None):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_meta = load_json(run_dir / "run_meta.json")
    best_eval = load_json(run_dir / "evaluation_baselines_best.json")
    final_eval = load_json(run_dir / "evaluation_baselines_final.json")
    ood_eval = load_json(run_dir / "evaluation_ood_challenging60.json")

    summary = {
        "run_id": run_meta["run_id"],
        "best_checkpoint": {
            "episode": run_meta["best_eval_episode"],
            "tracking_delta_psnr": run_meta["best_delta_psnr"],
            "id_mean_delta_psnr": safe_metric(best_eval, "aggregated", "dqn", "mean_delta_psnr"),
            "id_mean_delta_ssim": safe_metric(best_eval, "aggregated", "dqn", "mean_delta_ssim"),
            "acceptance_passed": best_eval.get("acceptance_passed"),
        },
        "final_checkpoint": {
            "id_mean_delta_psnr": safe_metric(final_eval, "aggregated", "dqn", "mean_delta_psnr"),
            "id_mean_delta_ssim": safe_metric(final_eval, "aggregated", "dqn", "mean_delta_ssim"),
            "acceptance_passed": final_eval.get("acceptance_passed"),
        },
        "ood_challenging60": {
            "mean_delta_uciqe": safe_metric(ood_eval, "aggregated", "dqn", "mean_delta_uciqe"),
            "mean_delta_uiqm_proxy": safe_metric(ood_eval, "aggregated", "dqn", "mean_delta_uiqm_proxy"),
            "num_images": ood_eval.get("num_images"),
        },
        "bologna_reference": {
            "psnr": 15.47,
            "ssim": 0.628,
            "notes": "Reported DDQN benchmark from Bologna reference analysis doc.",
        },
    }

    lines = [
        "# Underwater RL Results",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Best checkpoint episode: `{summary['best_checkpoint']['episode']}`",
        f"- Best-on-fixed-tracking-set delta PSNR: `{summary['best_checkpoint']['tracking_delta_psnr']:+.4f}`",
        f"- Best checkpoint ID mean delta PSNR: `{summary['best_checkpoint']['id_mean_delta_psnr']:+.4f}`",
        f"- Final checkpoint ID mean delta PSNR: `{summary['final_checkpoint']['id_mean_delta_psnr']:+.4f}`",
        f"- OOD challenging-60 mean delta UCIQE: `{summary['ood_challenging60']['mean_delta_uciqe']:+.4f}`",
        f"- OOD challenging-60 mean delta UIQM proxy: `{summary['ood_challenging60']['mean_delta_uiqm_proxy']:+.4f}`",
        "",
        "## Bologna Comparison",
        "",
        "| Metric | Bologna DDQN | Ours Best ID | Ours Final ID | Ours OOD |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| PSNR / delta PSNR | `15.47 dB` absolute | `{summary['best_checkpoint']['id_mean_delta_psnr']:+.4f}` delta | `{summary['final_checkpoint']['id_mean_delta_psnr']:+.4f}` delta | `n/a` |",
        f"| SSIM / delta SSIM | `0.628` absolute | `{summary['best_checkpoint']['id_mean_delta_ssim']:+.4f}` delta | `{summary['final_checkpoint']['id_mean_delta_ssim']:+.4f}` delta | `n/a` |",
        f"| OOD no-reference | `n/a` | `n/a` | `n/a` | `UCIQE {summary['ood_challenging60']['mean_delta_uciqe']:+.4f}`, `UIQM proxy {summary['ood_challenging60']['mean_delta_uiqm_proxy']:+.4f}` |",
        "",
        "## Notes",
        "",
        "- `best-on-fixed-tracking-set` and `final-stable-checkpoint` are reported separately by design.",
        "- ID uses paired-reference metrics (PSNR/SSIM deltas).",
        "- OOD challenging-60 has no references, so it is reported with no-reference quality metrics.",
        "- Bologna values are reported as absolute metrics from the reference document; ours are mostly delta metrics, so direct comparison must be interpreted carefully.",
    ]

    with open(run_dir / args.output_markdown, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(run_dir / args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved markdown report: {run_dir / args.output_markdown}")
    print(f"Saved JSON summary: {run_dir / args.output_json}")


if __name__ == "__main__":
    main()
