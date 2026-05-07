# AGENT_CONTEXT.md

## Project Identity
- **Title:** Adaptive Image Enhancement via Deep Reinforcement Learning
- **Domain:** Reinforcement Learning + Image Processing
- **Primary Goal:** Learn an interpretable sequence of classical image-processing actions that improves degraded images.

## Scientific Positioning
- This project does **not** try to prove RL always beats supervised restoration (e.g., U-Net).
- The target contribution is: image enhancement as an **interpretable sequential decision process** with DQN choosing discrete operators.

## Current Technical Setup
- **Dataset:** CIFAR-10 train split as clean source images (native `32x32`).
- **Degradations:** synthetic (`gaussian_noise`, `low_brightness`, `low_contrast`, `blur`, `combined`, and `mixed` policy).
- **Environment:** Gymnasium (`ImageEnhancementEnv`) with `max_steps`, `stop` action, optional step-channel.
- **Action space:**
  - `increase_brightness`
  - `decrease_brightness`
  - `increase_contrast`
  - `decrease_contrast`
  - `gaussian_denoise`
  - `sharpen`
  - `gamma_up`
  - `gamma_down`
  - `stop`
- **Agent:** DQN with target network and replay buffer.
- **Checkpoint selection policy:** PSNR-first (`mean_delta_psnr`), tie-break on `mean_eval_reward`.

## Infrastructure and Storage Policy
- Heavy artifacts/datasets stay on HPC storage (`/storage/internal_02/...`), not in git.
- Run artifacts are scoped by run id:
  - `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/...`
  - `${LOGS_ROOT}/dqn/<RUN_ID>/...`

## Phase Status (Requested Roadmap)

### Phase 1 - Robust Evaluation Criteria
**Status:** Implemented and validated.

Implemented items:
- Acceptance gate in `evaluation_dqn_baselines.py` with hard-fail checks:
  - `baseline_report_generated`
  - `mean_delta_psnr_positive` (`mean_delta_psnr > 0`)
  - `output_psnr_ge_input_psnr`
  - `stop_rate_ok` (>= `training.min_stop_rate`)
  - `dominant_action_share_ok` (<= `training.action_collapse_threshold`)
  - `action_analysis_available` (missing analysis file => fail)
- `evaluation_baselines.json` now exports explicit gate metrics:
  - `input_psnr`, `output_psnr`, `mean_delta_psnr`
  - `input_ssim`, `output_ssim`, `mean_delta_ssim`
  - `stop_rate`, `dominant_action_share`, `avg_episode_length`
  - plus grouped `gate_metrics` block.
- `compare_dqn_runs.py` updated to show gate-oriented metrics directly in run comparison.

Validation evidence:
- Historical bad runs are rejected when quality checks fail.
- `run_comparison.json` consolidates multi-run gate outcomes.

### Phase 2 - DQN Stability Upgrades
**Status:** Partially implemented.

Implemented:
- **Double DQN:** available and active in controlled experiments (`use_double_dqn=true`).

Implemented but not yet fully validated as final setting:
- **Dueling DQN:** code path exists and has been tested in A/B runs.

Not implemented yet:
- **Prioritized Experience Replay (PER)**.

Notes:
- Significant run-to-run variance still observed.
- Some runs remain quality-negative even with behavior checks passing.

### Phase 3 - Reward Refinement
**Status:** In progress (Phase 3A completed as controlled tuning).

What was done:
- Controlled reward tuning experiments with same training budget and architecture.
- Three key runs:
  - `dqn_phase3a_control_20260506_122118`
  - `dqn_phase3a_treatment_20260506_123200`
  - `dqn_phase3a_treatment2_20260506_124100`

Observed trend:
- `stop_rate`: `0.024 -> 0.041 -> 0.055` (improving, still below threshold `0.10`).
- `mean_delta_psnr`: `-1.512 -> -0.990 -> +0.703` (alignment improved in treatment2).

Current best trade-off run:
- `dqn_phase3a_treatment2_20260506_124100`
- `mean_delta_psnr = +0.7033`
- `output_psnr_ge_input_psnr = true`
- `stop_rate = 0.0546` (gate still fails only on stop-rate)

### Phase 4 - Policy Behavior Analysis
**Status:** Implemented baseline tooling; expanded with visual diagnostics.

Implemented:
- `analyze_dqn_actions.py` logs:
  - action frequencies
  - per-step action frequencies
  - most common sequences
  - stop-rate
  - episode length stats
  - best/worst samples by delta PSNR
- Visual inspection artifacts generated:
  - original CIFAR-10 sample grid
  - native-vs-zoom comparison
  - step-by-step agent rollout with per-step PSNR/SSIM

Latest visual artifacts:
- `${LOGS_ROOT}/dqn/visual_inspection_20260506/`

### Phase 5 - Generalization / OOD
**Status:** Partially implemented.

Implemented:
- OOD stress evaluation protocol with stronger degradations (e.g., noise std 0.2).

Evidence:
- ID (mixed, noise 0.1): positive (`+0.7033` on best run)
- OOD (gaussian noise 0.2): negative (`-1.9320`)

Interpretation:
- In-distribution improvement is possible.
- Generalization remains weak under stronger/no-shifted degradations.

### Phase 6 - Baseline Comparison
**Status:** Implemented for classical baselines, not closed for final scientific conclusion.

Implemented:
- `evaluation_dqn_baselines.py` compares DQN against:
  - `input_only`
  - fixed classical heuristic pipelines
- Unified eval subset and run-scoped JSON reports.

Still missing for final closure:
- Stable, repeated wins under gate constraints.
- Optional supervised lightweight baseline (if decided later).

## Important Recent Engineering Change (Image Sharpness)
Problem reported:
- "Clean" and output images looked blurry.

Root cause found:
- CIFAR-10 is native `32x32`; previous `32 -> 128` resizing introduced visible smoothing.

Applied fix:
- Training/eval/action-analysis now use YAML-driven image size aligned to dataset (`32x32`).

Compatibility note:
- Existing DQN conv stack was built around `128x128` feature extraction.
- For backward compatibility with legacy checkpoints, network forward now upsamples tensor inputs internally when needed.
- This keeps old checkpoints runnable while preserving dataset-faithful env images.

## What Has Been Tried So Far
- PSNR-first checkpoint selection (instead of reward-only selection).
- Strict acceptance gating and mandatory action-analysis availability.
- Double DQN activation and A/B runs.
- Dueling variant exploratory run(s).
- Reward shaping iterations (control/treatment protocol).
- ID vs OOD evaluation runs.
- Visual inspection of dataset quality and policy trajectories.

## Open Problems (As of 2026-05-06)
1. **Stop-rate remains below threshold** in otherwise promising runs.
2. **Run instability**: outcomes vary substantially between runs.
3. **Reward-quality mismatch risk** still exists in some settings.
4. **OOD robustness is insufficient** (quality drops under stronger degradations).
5. **Dataset ceiling**: CIFAR-10 low native resolution limits visual fidelity and realism of enhancement claims.

## Recommended Immediate Next Steps
1. Consolidate Phase 3 reward tuning to raise stop-rate without sacrificing delta PSNR.
2. Run a controlled Double-vs-Double+Dueling matrix with fixed seeds and equal budgets.
3. Decide whether to keep CIFAR-10 for method validation only and introduce a higher-resolution dataset for stronger visual claims.
4. If stability remains poor, implement PER as next incremental Phase 2 extension.

## How to Run Core Checks
From project root (with `venv`):
- `./venv/bin/python src/evaluation/analyze_dqn_actions.py --checkpoint <ckpt>`
- `./venv/bin/python src/evaluation/evaluation_dqn_baselines.py --checkpoint <ckpt>`
- `./venv/bin/python src/evaluation/compare_dqn_runs.py`

## Key Artifacts
Per run:
- `episode_summary.csv`
- `eval_summary.json`
- `action_analysis.json`
- `evaluation_baselines.json`
- `effective_config.json`
- `run_meta.json`

Global:
- `${LOGS_ROOT}/dqn/run_comparison.json`
- `${LOGS_ROOT}/dqn/visual_inspection_20260506/`

## Conventions
- Keep modules explicit and testable.
- Prefer config-driven behavior over hardcoded constants.
- Avoid storing heavy artifacts in repository.
- Update this file whenever workflow logic, acceptance policy, or experimental conclusions change.
