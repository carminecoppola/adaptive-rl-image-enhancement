# Current Project State

Reference date: 2026-05-10
Canonical configuration: `underwater_dqn_v1` v4.0

## Final configuration

| Component | Value |
|---|---|
| Dataset | UIEB, 890 paired images |
| Agent | Double DQN |
| Observation | 128×128 RGB image + normalized step channel |
| Actions | white balance, contrast up, sharpen, STOP |
| Maximum episode length | 5 decisions |
| Reward quality term | ΔPSNR + 10 × ΔSSIM |
| Training episodes | 5,000 |
| Best checkpoint | Episode 1,540 |

## Official run

Run ID: `dqn_underwater_full_20260510_165955_1494`

| Metric | Best checkpoint | Final checkpoint |
|---|---:|---:|
| Mean in-domain ΔPSNR | **+1.5492 dB** | +1.0060 dB |
| Output PSNR | **18.7157 dB** | 18.1724 dB |
| Output SSIM | **0.8275** | 0.8035 |
| Acceptance suite | Passed | Passed |

OOD evaluation on `challenging-60`:

| Metric | Change |
|---|---:|
| Mean ΔUCIQE | **-0.1707** |
| Mean ΔUIQM proxy | **-0.0119** |

## Interpretation

The final policy produces a measurable improvement on the paired in-domain
evaluation split. The best checkpoint is materially stronger than the final
checkpoint, so training quality is not monotonic and checkpoint tracking is
required. Negative OOD deltas prevent a general robustness claim.

## Canonical workflow

1. `python -m src.training.train --experiment underwater_dqn_v1 --phase full_training`
2. `python -m src.evaluation.analyze_dqn_actions`
3. `python -m src.evaluation.evaluation_dqn_baselines`
4. `python -m src.evaluation.evaluate_underwater_ood`
5. `python -m src.evaluation.generate_underwater_report`

The Slurm entrypoint `scripts/train_underwater.sbatch` executes the full
workflow and writes all artifacts under the run-specific log directory.

## Open limitation

The main unresolved issue is distribution shift between paired UIEB training
images and `challenging-60`. The extended action-set ablation improved OOD
metrics but lost too much paired performance; LAB statistics preserved paired
performance but worsened OOD results.
