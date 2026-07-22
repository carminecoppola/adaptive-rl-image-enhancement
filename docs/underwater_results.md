# Underwater Experiment Results

This document is the canonical repository summary for consolidated underwater
experiments. Per-run JSON and Markdown artifacts remain the source of truth for
individual executions.

## Evaluation protocol

Each official run includes:

1. full DDQN training;
2. action analysis for the best checkpoint;
3. paired baseline evaluation for best and final checkpoints;
4. no-reference OOD evaluation on `challenging-60`;
5. a generated human-readable and machine-readable report.

Paired results are reported as changes relative to the degraded input. OOD
images have no paired reference, so UCIQE and a UIQM proxy are reported instead.

## Consolidated experiments

| Run | Isolated change | Best ID ΔPSNR | OOD ΔUCIQE | Decision |
|---|---|---:|---:|---|
| `1490` | Restored curated baseline | +1.5089 dB | -0.2504 | Baseline |
| `1491` | Increase horizon from 3 to 5 | **+1.5622 dB** | -0.2600 | Strong ID candidate |
| `1492` | Extended 8-action set | +0.8762 dB | **-0.1677** | Rejected: excessive ID loss |
| `1493` | Add global LAB statistics | +1.5573 dB | -0.3724 | Rejected: OOD regression |
| `1494` | Final v4.0 configuration | **+1.5492 dB** | -0.1707 | Official presentation run |

## Official v4.0 run

Run ID: `dqn_underwater_full_20260510_165955_1494`

### Best checkpoint

- episode: `1540`
- mean ΔPSNR: `+1.5492 dB`
- output PSNR: `18.7157 dB`
- output SSIM: `0.8275`
- acceptance suite: passed

### Final checkpoint

- mean ΔPSNR: `+1.0060 dB`
- output PSNR: `18.1724 dB`
- output SSIM: `0.8035`
- acceptance suite: passed

### OOD `challenging-60`

- mean ΔUCIQE: `-0.1707`
- mean ΔUIQM proxy: `-0.0119`

## Decision rationale

The final configuration keeps the curated four-action policy and five-step
horizon without LAB statistics. It stays close to the best paired result,
improves OOD performance relative to the restored baseline, and remains easier
to inspect than the extended action set.

The ablations do not solve OOD generalization. The next experiment should
prioritize broader training domains and perceptual/no-reference validation
rather than merely increasing the number of episodes or actions.
