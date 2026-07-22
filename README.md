# Adaptive Reinforcement Learning for Underwater Image Enhancement

This repository implements a reproducible Double DQN (DDQN) workflow that
learns short, interpretable enhancement sequences for paired underwater
images. The canonical policy operates directly on the current image and
chooses among four deterministic actions:

- `white_balance`
- `contrast_up`
- `sharpen`
- `stop`

The project is an independent implementation inspired by earlier underwater
reinforcement-learning work. It is not a fork of the University of Bologna
notebook or codebase.

## Canonical result

The official v4.0 run is `dqn_underwater_full_20260510_165955_1494`.

| Metric | Best checkpoint |
|---|---:|
| Episode | 1,540 |
| Mean in-domain ΔPSNR | **+1.5492 dB** |
| Output PSNR | **18.7157 dB** |
| Output SSIM | **0.8275** |
| Acceptance suite | **Passed** |
| OOD ΔUCIQE (`challenging-60`) | **-0.1707** |
| OOD ΔUIQM proxy | **-0.0119** |

The paired in-domain result is positive. The negative no-reference OOD deltas
show that robustness to shifted underwater conditions remains unresolved.

## How the system works

1. Load a degraded/reference UIEB pair.
2. Convert the current image and normalized step into a four-channel state.
3. Predict one Q-value for each action with a convolutional network.
4. Apply the selected deterministic operator, or terminate with `stop`.
5. Compute reference-based quality change and behavioral shaping terms.
6. Store the transition in replay memory and optimize the DDQN policy.
7. Select checkpoints by mean ΔPSNR and enforce behavioral acceptance gates.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component boundaries and
[docs/underwater_results.md](docs/underwater_results.md) for the experiment
history.

## Repository layout

```text
configs/experiments/     Canonical experiment configuration
src/actions/             Deterministic enhancement operators and registries
src/agents/              Q-network, replay buffer, and DDQN agent
src/data/                UIEB pairing, tensor/PIL adapters, and degradations
src/envs/                Gymnasium image-enhancement environment
src/evaluation/          Baselines, action analysis, OOD evaluation, reports
src/metrics/             Paired and no-reference metrics
src/training/            Training loop, tracking, artifacts, and run setup
src/utils/               Configuration and deterministic split helpers
tests/                   Unit tests for actions, environment, DDQN, and baselines
docs/                    Architecture, results, and comparison notes
```

## Setup

```bash
bash scripts/setup_env.sh
source venv/bin/activate
cp .env.example .env
```

Set the dataset, log, and checkpoint roots in `.env`. The expected dataset
layout is:

```text
UIEB/
  raw/
  reference/
  challenging-60/
```

## Train

Local execution:

```bash
python -m src.training.train \
  --experiment underwater_dqn_v1 \
  --phase full_training
```

Slurm execution:

```bash
sbatch scripts/train_underwater.sbatch
```

## Evaluate

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

Generate the canonical per-run report with:

```bash
python -m src.evaluation.generate_underwater_report \
  --run-dir /path/to/logs/dqn/<RUN_ID>
```

## Quality checks

Install development dependencies, then run:

```bash
python -m pytest
ruff check .
ruff format --check .
python -m compileall -q src tests
```

## Reproducibility

Each official run stores its resolved configuration, split metadata,
checkpoints, episode summaries, evaluation history, action analysis, baseline
evaluation, OOD evaluation, and generated report under its run-scoped output
directory. Local datasets, checkpoints, logs, generated reports, and
presentations are intentionally ignored by Git.
