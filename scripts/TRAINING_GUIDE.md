# Training and Evaluation Guide

## Prepare the environment

From the repository root:

```bash
bash scripts/setup_env.sh
source venv/bin/activate
cp .env.example .env
```

Edit `.env` so `DATASET_ROOT`, `LOGS_ROOT`, and `CHECKPOINT_ROOT` point to
writable locations. The canonical UIEB dataset must contain `raw`, `reference`,
and `challenging-60` directories.

## Local training

```bash
python -m src.training.train \
  --experiment underwater_dqn_v1 \
  --phase full_training
```

## Slurm training

Submit from the repository root so Slurm can resolve the log paths:

```bash
mkdir -p logs
sbatch scripts/train_underwater.sbatch
```

The launcher derives the project root from `SLURM_SUBMIT_DIR`; it no longer
depends on a user-specific absolute path. Override `PROJECT_ROOT` or
`EXPERIMENT` through exported environment variables when necessary.

## Automated workflow

The Slurm launcher performs:

1. full DDQN training;
2. action analysis for best and final checkpoints;
3. paired baseline evaluation for both checkpoints;
4. OOD evaluation on `UIEB/challenging-60`;
5. canonical Markdown and JSON report generation.

## Expected artifacts

`${LOGS_ROOT}/dqn/<RUN_ID>/`:

- `effective_config.json`
- `dataset_split.json`
- `episode_summary.csv`
- `eval_summary.json`
- `run_meta.json`
- `action_analysis_best.json`
- `action_analysis_final.json`
- `evaluation_baselines_best.json`
- `evaluation_baselines_final.json`
- `evaluation_ood_challenging60.json`
- `underwater_results.md`
- `underwater_results_summary.json`

`${CHECKPOINT_ROOT}/dqn/<RUN_ID>/`:

- `dqn_best_policy_net.pt`
- `dqn_final_policy_net.pt`

## Manual evaluation

```bash
python -m src.evaluation.analyze_dqn_actions \
  --checkpoint /path/to/dqn_best_policy_net.pt \
  --num-images 50

python -m src.evaluation.evaluation_dqn_baselines \
  --checkpoint /path/to/dqn_best_policy_net.pt \
  --num-images 50

python -m src.evaluation.evaluate_underwater_ood \
  --checkpoint /path/to/dqn_best_policy_net.pt
```

The analysis notebook must consume generated run artifacts; it must not
recompute results with a separate evaluation path.
