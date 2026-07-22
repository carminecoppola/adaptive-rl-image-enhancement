# Bologna 2022 Comparison

Reference implementation: `sissaNassir/Underwater-Image_Enhancement` (2022).

## Scope

The Bologna project established the central idea of selecting underwater image
enhancement operations with DDQN. This repository reimplements that idea with
an image-based state, a smaller curated action space, modular training and
evaluation code, deterministic configuration, acceptance gates, ablations,
and a separate OOD evaluation.

The comparison is indicative rather than controlled: training details and
evaluation protocol are not identical. Absolute PSNR and SSIM values can be
reported side by side, but they do not prove superiority across studies.

## Reported comparison

| Property | Bologna 2022 | This project |
|---|---:|---:|
| DDQN output PSNR | 15.47 dB | **18.7157 dB** |
| DDQN output SSIM | 0.628 | **0.8275** |
| Training episodes | 20,000 | **5,000** |
| Policy actions | 20 | **4** |
| Explicit STOP behavior analysis | Limited | **Yes** |
| Deterministic configuration and artifacts | Limited | **Yes** |
| OOD evaluation | Not reported | **Yes** |

## What was retained

- sequential enhancement as a reinforcement-learning problem;
- discrete, interpretable image-processing actions;
- reference-based quality feedback;
- Double DQN for value estimation.

## What changed

- the policy observes the current image directly instead of a handcrafted
  CIELAB/VGG feature vector;
- the canonical action space is reduced to white balance, contrast, sharpen,
  and STOP;
- training, action analysis, baseline comparison, OOD evaluation, and report
  generation are separate modules;
- checkpoint selection is based on paired ΔPSNR and guarded by behavioral
  acceptance checks;
- all effective parameters and run artifacts are stored per run.

## Correct conclusion

The current project reports stronger paired absolute metrics with fewer
episodes and a smaller action space, but the protocols are not identical. The
defensible contribution is the compact interpretable policy and the stricter,
reproducible evaluation workflow—not a universal claim of better underwater
enhancement.
