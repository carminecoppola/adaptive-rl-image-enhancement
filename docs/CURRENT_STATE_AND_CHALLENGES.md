# Adaptive RL Image Enhancement: Current State and Challenges

## Core Contribution Statement
The main contribution of this project is not to outperform fully supervised restoration networks, but to investigate whether image enhancement can be modeled as an interpretable sequential decision-making process, where a DQN-based agent adaptively selects classical image-processing operations according to the current degradation state.

## 1. Project Snapshot
This project trains a DQN agent to improve degraded images through a short sequence of discrete image-processing actions.

Current setup:
- Dataset default: CIFAR-10 (clean) with synthetic degradations
- Dataset visual-safe mode: STL-10 (native 96x96) with controlled subset
- Environment: Gymnasium image enhancement with `max_steps` and `stop` action
- Reward: step-wise quality delta (PSNR/SSIM) plus penalties/bonuses
- Agent: DQN with replay buffer and target network

## 2. What Is Working
- End-to-end pipeline is available:
  - training: `src/training/train_dqn.py`
  - action diagnostics: `src/evaluation/analyze_dqn_actions.py`
  - quality comparison vs baselines: `src/evaluation/evaluation_dqn_baselines.py`
- Run-scoped artifacts are consistently generated under `${LOGS_ROOT}/dqn/<RUN_ID>/`.
- Checkpoint selection now follows a PSNR-first policy:
  - primary metric: `mean_delta_psnr`
  - tie-break: `mean_eval_reward`
- Run reproducibility improved via:
  - deterministic split
  - saved eval subset per evaluation point
  - `effective_config.json` snapshot

## 3. Main Difficulties Encountered

### 3.1 Reward/Objective Misalignment
Observed issue:
- Some runs achieved high eval reward while still decreasing final PSNR against degraded input.

Why it matters:
- The model can optimize proxy reward terms without delivering the true objective (image quality gain).

Mitigation already introduced:
- PSNR-first checkpoint selection and explicit acceptance gates on delta PSNR.

### 3.2 Training Instability Across Runs
Observed issue:
- Significant variance in outcomes between runs with similar settings.

Why it matters:
- Makes results fragile and harder to defend in final evaluation/presentation.

Mitigation already introduced:
- richer logging, run comparison tooling, and stricter acceptance criteria.

### 3.3 Behavioral Checks Not Sufficient Alone
Observed issue:
- Passing action-collapse and stop-rate checks does not guarantee quality improvement.

Why it matters:
- Risk of false positives in “stable” runs that are still suboptimal.

Mitigation already introduced:
- combined acceptance:
  - action behavior checks
  - positive `mean_delta_psnr`
  - DQN PSNR not worse than `input_only`

### 3.4 Domain Gap and Generalization Risk
Observed issue:
- Training/evaluation is on CIFAR-10 + synthetic degradations; transfer to other domains remains unproven.

Why it matters:
- Real-world robustness is not yet guaranteed.

Mitigation path:
- controlled expansion of degradation space and out-of-distribution validation.

## 4. Why Results Are Not Yet Consistently Good
Root causes are likely multi-factor:
- reward shaping sensitivity
- DQN variance/instability
- limited action expressivity for some degradation modes
- narrow training domain (CIFAR-10 synthetic)

This suggests the bottleneck is **not only** “more training epochs” or “more data volume”, but the interaction between optimization target, policy stability, and domain design.

## 5. Current Acceptance Framework
A run is considered healthy only if all key checks pass:
- `dominant_action_share <= threshold`
- `stop_rate >= min_stop_rate`
- `mean_delta_psnr > 0`
- DQN mean PSNR `>= input_only` mean PSNR

Output location:
- `evaluation_baselines.json` contains `acceptance_checks` and `acceptance_passed`.

## 6. Suggested Narrative for Presentation

### Slide track (recommended)
1. Problem framing and RL formulation
2. Baseline pipeline and action space
3. Failure analysis: reward vs true quality mismatch
4. Engineering fixes introduced (PSNR-first, run artifacts, acceptance gates)
5. Current outcomes and remaining risks
6. Next technical step (stability/generalization-focused)

### Key message
The project progressed from a “working RL prototype” to a “measurable and auditable training system”, where failures are now detectable and comparable across runs, enabling informed iteration.

## 7. Next Step Before Final Resolution
Run the A/B/C experimental protocol with the new acceptance framework:
- Phase A: short sanity runs
- Phase B: medium stability runs
- Phase C: long final runs

Then select the final model only from runs that satisfy the full acceptance suite.

## STL-10 Safe Configuration Update (2026-05-07)

What was added:
- Alternative dataset config `configs/dataset_stl10_safe.yaml`:
  - `name=STL10`
  - `image_size=96`
  - `train_subset_size=5000`
  - `eval_subset_size=500`
- Split helper now supports deterministic subset limiting for train/eval pools.
- Train/eval scripts apply subset limits from dataset config when specified.
- Environment now skips resize when input image already matches target size to avoid unnecessary interpolation.

Validation completed (no long training launched):
- STL-10 load smoke test: sample size confirmed at `96x96`.
- Visual/degradation smoke test:
  - 5 clean STL-10 samples
  - 5 degraded STL-10 samples
  - PSNR/SSIM degraded vs clean computed
  - Output grid stored at:
    `${LOGS_ROOT}/dqn/stl10_safe_validation_20260507/stl10_clean_degraded_grid.png`

Notebook status:
- `visual_policy_analysis.ipynb` executes end-to-end without runtime cell errors (validated through `nbconvert --execute`).
- Jupyter runtime dependencies were installed in `venv` (`notebook`, `nbformat`, etc.).

## Step 1 Closure (Robust Evaluation Gate)

What was fixed:
- Strict acceptance gate enforced in evaluation with mandatory checks:
  - `baseline_report_generated`
  - `mean_delta_psnr_positive`
  - `output_psnr_ge_input_psnr`
  - `stop_rate_ok`
  - `dominant_action_share_ok`
  - `action_analysis_available`
- Missing `action_analysis.json` now causes mandatory failure.
- Evaluation scripts aligned with training for `mixed` degradation and observation channels.
- Gate report now exports normalized summary metrics in `gate_metrics`:
  - `input_psnr`, `output_psnr`, `mean_delta_psnr`
  - `input_ssim`, `output_ssim`, `mean_delta_ssim`
  - `stop_rate`, `dominant_action_share`, `avg_episode_length`
- Environment image size is now config-driven from dataset YAML (CIFAR-10 default `32x32`) to avoid artificial blur from `32->128` upscaling in clean/evaluated images.

Validation evidence:
- Per-run pass/fail is stored in each `evaluation_baselines.json`.
- Cross-run summary is stored in `${LOGS_ROOT}/dqn/run_comparison.json`.
- Known bad checkpoints are correctly rejected when PSNR delta is negative.

Validated examples (Step 1 closure batch):
- `dqn_1444_20260506_105528`: behavior checks pass, quality checks fail (`acceptance_passed=false`).
- `dqn_1441_20260506_103453`: behavior checks pass, quality checks fail (`acceptance_passed=false`).
- `dqn_1443_20260506_105047`: quality checks pass, stop-rate check fails (`acceptance_passed=false`).
- Regression strictness test with isolated `LOGS_ROOT` (missing diagnostics): `action_analysis_available=false` and `acceptance_passed=false`.

Current known limitation:
- Even with a correct gate, many runs can still fail if policy quality is poor; this is expected and indicates the next work should target training stability/reward refinement, not gate logic changes.

## Phase 3A Progress (Reward Tuning, Double DQN Fixed)

Controlled runs were executed with identical training settings (`use_double_dqn=true`, `use_dueling_dqn=false`, 260 episodes) and reward-only changes:
- `dqn_phase3a_control_20260506_122118` (control)
- `dqn_phase3a_treatment_20260506_123200` (stop-aware conservative v1)
- `dqn_phase3a_treatment2_20260506_124100` (stop-aware conservative v2)

Observed trend:
- `stop_rate`: `0.024` -> `0.041` -> `0.055` (improving but still below threshold `0.10`)
- `mean_delta_psnr`: `-1.512` -> `-0.990` -> `+0.703` (quality alignment recovered in v2)

Current interpretation:
- Reward tuning is moving behavior in the expected direction.
- The latest run is quality-positive and interpretable, but full gate pass is still blocked by stop-rate.

## Behavior + OOD Evidence for Presentation

Run used for evidence:
- `dqn_phase3a_treatment2_20260506_124100`

In-distribution (ID) result:
- `mean_delta_psnr = +0.7033`
- `output_psnr_ge_input_psnr = true`
- `stop_rate = 0.0546`

Out-of-distribution (OOD) stress check:
- Protocol: evaluation-only with stronger noise (`gaussian_noise`, `noise_std=0.2`), same checkpoint.
- `mean_delta_psnr = -1.9320`
- `stop_rate = 0.0925`

Takeaway:
- The agent can improve images in-distribution but does not yet generalize robustly under stronger degradations.
- This supports the project narrative: the work now provides measurable policy behavior and failure visibility, enabling targeted iteration.
