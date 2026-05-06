# AGENT_CONTEXT.md

## Project Identity
- **Title:** Adaptive Image Enhancement via Deep Reinforcement Learning
- **Domain:** Reinforcement Learning + Image Processing
- **Primary Goal:** Train an RL agent to improve degraded images by applying a sequence of image-processing actions.

## Problem Formulation (RL)
- **State:** current image (degraded or partially enhanced)
- **Action:** one filter operation from a discrete action set
- **Reward:** quality improvement (PSNR/SSIM) with optional step penalty
- **Episode End:** `stop` action or `max_steps` reached

## Current Scope
This repository is in progressive implementation mode:
- Build and validate modules step-by-step
- Keep code modular and testable
- Avoid big-bang implementation

## Action Space (current config)
Defined in `configs/environment.yaml`:
- increase_brightness
- decrease_brightness
- increase_contrast
- decrease_contrast
- gaussian_denoise
- sharpen
- gamma_up
- gamma_down
- stop

## Dataset Policy (Important)
- **Do not store datasets inside the repository.**
- Datasets and heavy artifacts are stored on HPC disk:
  - `/storage/internal_02`
- Project-specific storage root:
  - `/storage/internal_02/adaptive_rl_image_enhancement`

## Environment Variables (.env)
Key variables (see `.env.example`):
- `HPC_STORAGE_ROOT`
- `DATA_ROOT`
- `DATASET_ROOT`
- `PROCESSED_DATA_ROOT`
- `CHECKPOINT_ROOT`
- `RESULTS_ROOT`
- `LOGS_ROOT`
- `HF_HOME`
- `TORCH_HOME`

## Local Development Rules
- Virtual environment name: `venv`
- Activate with: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Never commit `.env`, `venv/`, datasets, checkpoints, or large result artifacts

## Repository Structure
- `configs/` -> YAML configuration files
- `src/data/` -> dataset download, degradation, dataset wrappers
- `src/actions/` -> image action/filter operators
- `src/envs/` -> RL environment definition
- `src/agents/` -> RL agent logic (DQN)
- `src/metrics/` -> PSNR/SSIM utilities
- `src/training/` -> training entrypoints/pipelines
- `src/evaluation/` -> evaluation and baseline comparison
- `src/utils/` -> config and path utilities
- `scripts/` -> operational shell scripts only
- `tests/` -> test scripts (single place for tests)

## Project Phases
1. Environment and project scaffold
2. Dataset setup (CIFAR-10 initially)
3. Define reward
4. Implement Gymnasium environment
5. Test random episode
6. Implement classical baselines
7. Implement DQN
8. Run small training
9. Evaluation
10. Final report

## Current Status
- Core RL pipeline implemented end-to-end (train, analyze, evaluate)
- CIFAR-10 training/eval split is deterministic and run-traceable
- DQN checkpoint selection is now PSNR-first (`mean_delta_psnr`), with eval-reward tie-break
- Per-run configuration snapshot is persisted in `effective_config.json`
- Action-behavior diagnostics and baseline comparison are both run-scoped
- Working with CIFAR-10 + synthetic degradations for initial validation

## How to Run Quick Checks
From project root:
- `python3 tests/test_degradation.py`
- `python3 tests/test_actions.py`
- `python3 tests/test_metrics.py`

## Stabilization Protocol (DQN)
1. Launch training with Slurm:
- `sbatch scripts/train_dqn.sbatch`
2. Analyze policy behavior on fixed eval split:
- `python3 src/evaluation/analyze_dqn_actions.py`
3. Evaluate DQN vs baselines on same fixed eval split:
- `python3 src/evaluation/evaluation_dqn_baselines.py`
4. Compare multiple runs in one table (tracking PSNR-first alignment):
- `python3 src/evaluation/compare_dqn_runs.py`

Artifacts are run-scoped by `RUN_ID`:
- `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/...`
- `${LOGS_ROOT}/dqn/<RUN_ID>/...`

Pass/Fail criteria:
- Dominant action share <= `training.action_collapse_threshold`
- Stop rate >= `training.min_stop_rate`
- `mean_delta_psnr > 0` on fixed eval subset
- DQN mean PSNR not worse than `input_only`
- Baseline comparison report generated without errors

## Key Artifacts (per run)
- `episode_summary.csv` -> train dynamics by episode
- `eval_summary.json` -> eval metrics over training checkpoints
- `action_analysis.json` -> action collapse / stop-rate diagnostics
- `evaluation_baselines.json` -> DQN vs baselines + acceptance checks
- `effective_config.json` -> frozen run configuration (dataset/env/training)
- `run_meta.json` -> best metric, selection mode, run metadata

## Known Difficulties (Current Phase)
- Reward and final quality can diverge if selection is based only on eval reward
- High run-to-run variance remains even with fixed seeds
- Policy can pass action-behavior checks while still degrading PSNR
- Generalization beyond synthetic CIFAR-10 degradations is still open

## Conventions for Future Changes
- Keep modules small and explicit
- Add tests when adding features
- Prefer config-driven behavior over hardcoded paths
- Preserve HPC storage policy
- Keep this file updated when architecture or workflow changes
