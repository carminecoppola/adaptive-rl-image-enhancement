# Architecture

## Design objective

The system learns *which* deterministic enhancement operation to apply and
*when* to stop. It does not generate replacement pixels with a neural image
model. This keeps every rollout short, reproducible, and inspectable.

## Data flow

```text
UIEB pair
  degraded image ──> ImageEnhancementEnv ──> observation (RGB + step)
  reference image ─────────────────────────> reward metrics only

observation ──> QNetwork ──> action ──> deterministic operator
     ^                                      │
     └──────────── updated image ───────────┘

transition ──> ReplayBuffer ──> DQNAgent.optimize_model()
```

The reference image never enters the policy observation. It is retained by the
environment only to compute PSNR, SSIM, and terminal quality terms.

## Component boundaries

### Actions

`src/actions/underwater_v1.py` contains pure tensor transformations.
`src/actions/__init__.py` exposes named action sets and maps action identifiers
to operators. The canonical set contains white balance, contrast, sharpening,
and STOP.

### Environment

`src/envs/env.py` owns episode state, reward decomposition, termination, and
diagnostic information. STOP preserves the current image; the step limit
truncates an episode and may add a penalty.

### Agent and network

`src/agents/q_network.py` defines the convolutional Q-network.
`src/agents/dqn_agent.py` owns epsilon-greedy action selection, replay-based
optimization, Double DQN targets, gradient clipping, and target synchronization.

### Training

`src/training/train.py` composes the dataset, deterministic split, environment,
agent, replay memory, checkpoint selection, and run artifacts. Supporting
modules keep tracking, serialization, and run-path concerns out of the main
loop.

### Evaluation

The evaluation package has three independent responsibilities:

1. inspect learned action usage and STOP behavior;
2. compare DDQN against fixed enhancement pipelines on the paired split;
3. measure no-reference change on `challenging-60`.

Checkpoint acceptance requires positive paired improvement, output quality not
below the degraded input, non-collapsed action usage, sufficient STOP behavior,
and complete analysis artifacts.

## Compatibility

The public alias `DQN` remains available for older notebooks, but new code
should import `QNetwork`. Renaming the Python class does not change checkpoint
parameter keys, so existing state dictionaries remain loadable.
