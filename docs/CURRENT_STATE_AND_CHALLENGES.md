# Adaptive RL Image Enhancement: Current State and Challenges

## Core Contribution Statement
The main contribution of this project is not to outperform fully supervised restoration networks, but to investigate whether image enhancement can be modeled as an interpretable sequential decision-making process, where a DQN-based agent adaptively selects classical image-processing operations according to the current degradation state.

## 1. Project Snapshot
This project trains a DQN agent to improve degraded images through a short sequence of discrete image-processing actions.

Current setup:
- Dataset: CIFAR-10 (clean) with synthetic degradations
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
