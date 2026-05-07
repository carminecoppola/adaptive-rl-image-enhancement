# Adaptive Image Enhancement via Deep Reinforcement Learning

## Project Overview

This project implements **Adaptive Image Enhancement via Deep Reinforcement Learning**, with specialized support for **underwater image restoration**. An RL agent learns to enhance degraded images by applying a sequence of image-processing actions.

### Key Innovation: Underwater Domain Adaptation

The project extends standard image enhancement with underwater-specific capabilities:
- **Physics-based degradation simulation**: Wavelength absorption, backscatter veil, contrast loss
- **Underwater action space**: 4 specialized filters (red channel boost, LAB color balance, CLAHE, saturation)
- **Color cast detection**: LAB-space metric for underwater color correction
- **UIEB dataset support**: Real underwater images with reference clean versions

---

## RL Formulation

| Component | Details |
|-----------|---------|
| **State** | Current enhanced image + episode progress channel |
| **Action Space** | 13 discrete actions (9 standard + 4 underwater-specific) |
| **Reward** | PSNR improvement + color cast improvement + terminal bonuses |
| **Agent** | Double DQN (DDQN) with replay buffer and target network |

---

## Quick Start

### 1. Environment Setup
```bash
bash scripts/setup_env.sh
source venv/bin/activate
```

### 2. Download Datasets
```bash
# For CIFAR10/STL10 (automatic)
bash scripts/download_dataset.sh

# For UIEB (manual - see docs/CURRENT_STATE_AND_CHALLENGES.md)
# Dataset requires manual download due to licensing
```

### 3. Run Experiments

Three experiment configurations with increasing complexity:

```bash
# Experiment A: CIFAR10 baseline (standard actions only, ~15 min)
python src/training/train.py --experiment exp_A_cifar_baseline

# Experiment B: CIFAR10 with underwater actions (~40 min)
python src/training/train.py --experiment exp_B_cifar_underwater

# Experiment C: UIEB real underwater images (full setup, ~150 min)
python src/training/train.py --experiment exp_C_uieb_full
```

### 4. Evaluate & Compare Baselines

```bash
# Compare RL agent vs classical methods (DCP, histogram, identity)
python scripts/compare_experiment_results.py \
    --dataset CIFAR10 \
    --num-samples 50 \
    --output results/comparison
```

### 5. Run Tests

```bash
# All tests
pytest tests/ -v --tb=short

# Specific module
pytest tests/test_baselines.py -v
```

---

## Project Architecture

### Core Components

**1. RL Agent (src/agents/)**
- `dqn.py`: Double DQN network architecture
- `dqn_agent.py`: Agent orchestration (training, action selection)
- `replay_buffer.py`: Experience storage and sampling

**2. Environment (src/envs/)**
- `image_enhancement_env.py`: Gymnasium-compatible environment
- Episode state: Image + step progress
- Reward: PSNR improvement, color cast correction, terminal bonuses

**3. Actions (src/actions/)**
- `filters.py`: 13-action discrete space
  - Standard: brightness, contrast, gamma, denoise, sharpen, blur
  - Underwater: red boost, color balance, CLAHE, saturation

**4. Metrics (src/metrics/)**
- `psnr.py`, `ssim.py`: Standard quality metrics
- `color_cast.py`: Underwater-specific LAB-space color balance metric

**5. Data Processing (src/data/)**
- `dataset.py`: CIFAR10, STL10, UIEB loaders
- `degradation.py`: Synthetic degradation + physics-based underwater simulation
- `download_dataset.py`: Automatic dataset download

**6. Training (src/training/)**
- `train.py`: Main training loop with experiment framework
- `dqn_training_helpers.py`: Environment building, evaluation utilities
- `dqn_artifacts.py`: Checkpoint management

**7. Evaluation (src/evaluation/)**
- `baselines.py`: Classical methods (DCP, histogram, identity)
- `evaluate_agent.py`: Agent performance evaluation
- `compare_dqn_runs.py`: Multi-experiment comparison

### Configuration System

YAML-based hierarchical configs in `configs/`:

```
configs/
├── dataset.yaml          # Dataset paths & properties
├── environment.yaml      # RL env params (action space, rewards)
├── training.yaml         # Training settings (episodes, learning rate)
├── evaluation.yaml       # Eval settings
├── paths.yaml            # Storage locations
└── experiments/
    ├── exp_A_cifar_baseline.yaml      # CIFAR10, standard actions
    ├── exp_B_cifar_underwater.yaml    # CIFAR10, full actions + color cast
    └── exp_C_uieb_full.yaml          # UIEB, all features
```

**Experiment Override**: `--experiment exp_B_cifar_underwater` merges experiment config with base configs via `deep_merge_dicts()`

---

## Datasets Supported

| Dataset | Resolution | Samples | Training Use | Notes |
|---------|------------|---------|--------------|-------|
| **CIFAR10** | 32×32 | 50,000 | Standard degradation | Automatic download |
| **STL10** | 96×96 | 5,000 | Standard degradation | Automatic download |
| **UIEB** | 128×128 | 890 | Real underwater images | Manual download required |

### UIEB Dataset Access

The UIEB dataset requires manual download due to licensing. See [docs/CURRENT_STATE_AND_CHALLENGES.md](docs/CURRENT_STATE_AND_CHALLENGES.md) for instructions.

---

## Underwater Enhancement Details

### Degradation Model

**Physics-Based Simulation** (`src/data/degradation.py`):
```python
# Wavelength absorption (depth-dependent)
r_new = r * (1 - 0.8 * depth_factor)      # Red absorbs most
g_new = g * (1 - 0.6 * depth_factor)      # Green moderate
b_new = b * (1 - 0.3 * depth_factor)      # Blue absorbs least

# Backscatter veil (turbidity × depth)
backscatter = bluish_haze * turbidity * depth

# Contrast reduction
contrast_factor = 0.7 - turbidity * 0.2
```

### Enhancement Actions

**Underwater-Specific (ImageAction 9-12)**:
1. **RED_CHANNEL_BOOST** (ImageAction.RED_CHANNEL_BOOST): `R *= 1.3`
2. **LAB_COLOR_BALANCE** (ImageAction.LAB_COLOR_BALANCE): Neutralize a*/b* toward 128
3. **CLAHE** (ImageAction.CLAHE): Adaptive histogram on L channel
4. **SATURATION_BOOST** (ImageAction.SATURATION_BOOST): PIL color enhancement

### Color Cast Metric

**LAB-Space Deviation** (`src/metrics/color_cast.py`):
```
score = (|a_mean - 128| + |b_mean - 128|) / 256
# Lower = better (neutral white point is a=128, b=128)
```

Integrated into reward: `color_cast_reward = weight × improvement_scale × Δ_color_cast`

---

## Classical Baselines

Implemented in `src/evaluation/baselines.py`:

| Method | Category | Use Case |
|--------|----------|----------|
| **Identity** | Zero baseline | Establishes "no processing" reference |
| **Histogram Equalization** | Global contrast | Simple brightness/contrast fix |
| **Dark Channel Prior** | Haze removal | State-of-art for underwater/foggy images |

**Dark Channel Prior Algorithm**:
1. Compute dark channel via morphological erosion (minimum within patches)
2. Estimate transmission map: `t = 1 - ω × (dark_channel / max)` where ω=0.95
3. Estimate atmospheric light from brightest dark pixels
4. Restore: `I_restored = (I_degraded - A) / max(t, 0.1) + A`

---

## Testing & Validation

**Comprehensive Test Suite**: 54+ passing tests

```bash
pytest tests/ -v
```

| Module | Tests | Status |
|--------|-------|--------|
| Agents (DDQN) | 9 | ✅ |
| Environment | 6 | ✅ |
| Actions | 10 | ✅ |
| Metrics | 7 | ✅ |
| Degradation | 8 | ✅ |
| Datasets | 6 | ✅ |
| Baselines | 8 | ✅ |

---

## File Structure

```
adaptive-rl-image-enhancement/
├── README.md                          # This file
├── AGENT_CONTEXT.md                   # Implementation details
├── main.py                            # Entry point
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── .gitignore
│
├── src/                               # Main codebase
│   ├── __init__.py
│   ├── actions/
│   │   ├── __init__.py
│   │   └── filters.py                 # 13-action space
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn.py                     # DDQN network
│   │   ├── dqn_agent.py               # Agent class
│   │   └── replay_buffer.py           # Experience storage
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # CIFAR10/STL10/UIEB loaders
│   │   ├── datasets.py                # Dataset utilities
│   │   ├── degradation.py             # Synthetic + underwater
│   │   └── download_dataset.py        # Auto-download
│   ├── envs/
│   │   ├── __init__.py
│   │   └── image_enhancement_env.py   # Gymnasium environment
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── baselines.py               # DCP, histogram, identity
│   │   ├── evaluate_agent.py          # Agent evaluation
│   │   ├── compare_dqn_runs.py        # Multi-run comparison
│   │   └── analyze_dqn_actions.py     # Action analysis
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   └── color_cast.py              # Underwater metric
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                   # Main training loop
│   │   ├── dqn_training_helpers.py    # Utilities
│   │   ├── dqn_artifacts.py           # Checkpointing
│   │   ├── dqn_tracking.py            # Logging
│   │   └── dqn_types.py               # Type definitions
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Config loading
│       ├── paths.py                   # Path utilities
│       └── splits.py                  # Train/test splits
│
├── configs/                           # YAML configurations
│   ├── dataset.yaml
│   ├── environment.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   ├── paths.yaml
│   └── experiments/
│       ├── exp_A_cifar_baseline.yaml
│       ├── exp_B_cifar_underwater.yaml
│       └── exp_C_uieb_full.yaml
│
├── scripts/                           # Utility scripts
│   ├── setup_env.sh                   # Environment setup
│   ├── download_dataset.sh            # Dataset download
│   ├── train.sh                       # Training runner
│   ├── evaluate.sh                    # Evaluation runner
│   ├── train_dqn.sbatch               # HPC batch script
│   └── compare_experiment_results.py  # Baseline comparison
│
├── tests/                             # Test suite
│   ├── test_ddqn.py
│   ├── test_env.py
│   ├── test_actions.py
│   ├── test_color_cast.py
│   ├── test_degradation.py
│   ├── test_uieb_dataset.py
│   └── test_baselines.py
│
├── notebooks/
│   ├── visual_policy_analysis.ipynb
│   └── visual_policy_analysis.executed.ipynb
│
├── checkpoints/                       # Model checkpoints
│   └── dqn/
│       ├── dqn_best_policy_net.pt
│       ├── dqn_final_policy_net.pt
│       └── dqn_policy_net.pt
│
├── logs/                              # Training logs
│   └── dqn/
│       └── dqn_phase3a_treatment2_.../
│
└── docs/
    └── CURRENT_STATE_AND_CHALLENGES.md # Full technical details
```

---

## Experiments & Results

### Experiment A: CIFAR10 Baseline
- **Dataset**: CIFAR10 (32×32, 50K samples)
- **Actions**: Standard only (9 actions)
- **Reward**: PSNR + terminal bonus
- **Episodes**: 300
- **Expected Runtime**: 15 minutes
- **Purpose**: Establish baseline performance

### Experiment B: CIFAR10 Underwater
- **Dataset**: CIFAR10 with synthetic underwater degradation
- **Actions**: Full action set (13 actions)
- **Reward**: PSNR + color cast improvement + terminal bonus
- **Episodes**: 500
- **Expected Runtime**: 40 minutes
- **Purpose**: Demonstrate underwater-specific enhancement

### Experiment C: UIEB Real Underwater
- **Dataset**: UIEB real underwater images (128×128, 890 pairs)
- **Actions**: Full action set (13 actions)
- **Reward**: Complete reward stack
- **Episodes**: 1000
- **Expected Runtime**: 150 minutes
- **Purpose**: Validate on real underwater imagery

---

## Storage Architecture

Datasets and artifacts are stored on HPC storage (not in repo):

```
/storage/internal_02/ccoppola/
├── datasets/              # CIFAR10, STL10, UIEB
├── checkpoints/           # Model weights
├── logs/                  # Training logs (tensorboard)
├── results/               # Evaluation results
└── torch_home/            # PyTorch cache
```

Configure via `.env`:
```bash
DATASET_ROOT=/storage/internal_02/ccoppola/datasets
CHECKPOINT_ROOT=/storage/internal_02/ccoppola/checkpoints
LOGS_ROOT=/storage/internal_02/ccoppola/logs
RESULTS_ROOT=/storage/internal_02/ccoppola/results
```

---

## Environment Variables

```bash
# Required paths
DATASET_ROOT=/storage/internal_02/ccoppola/datasets
CHECKPOINT_ROOT=/storage/internal_02/ccoppola/checkpoints
LOGS_ROOT=/storage/internal_02/ccoppola/logs
RESULTS_ROOT=/storage/internal_02/ccoppola/results

# Optional caches
TORCH_HOME=/storage/internal_02/ccoppola/torch_home
HF_HOME=/storage/internal_02/ccoppola/huggingface_home
```

Create `.env` file in project root, or use `.env.example` as template.

---

## Development Commands

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test module
pytest tests/test_baselines.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Train a specific experiment
python src/training/train.py --experiment exp_A_cifar_baseline

# Evaluate trained agent
python src/evaluation/evaluate_agent.py --checkpoint checkpoints/dqn/best.pt

# Compare methods
python scripts/compare_experiment_results.py --dataset CIFAR10 --num-samples 50
```

---

## Known Limitations

1. **UIEB Manual Download**: Licensing requires manual setup
2. **HPC-Specific**: Paths assume `/storage/internal_02/` availability
3. **Computational**: Exp C requires ~150 minutes on GPU
4. **Color Cast Metric**: LAB deviation is simplified (ignores perceptual uniformity edge cases)
5. **Baseline DCP**: Transmission smoothing is heuristic-based

See [docs/CURRENT_STATE_AND_CHALLENGES.md](docs/CURRENT_STATE_AND_CHALLENGES.md) for full technical details, future improvements, and references.

---

## References

- **Double DQN**: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
- **Dark Channel Prior**: He et al., "Single Image Haze Removal using Dark Channel Prior" (CVPR 2009)
- **UIEB Dataset**: Li et al., "An Underwater Image Enhancement Benchmark Dataset and Beyond" (TNNLS 2020)

---

## Project Status

✅ **9/10 Tasks Complete**:
- ✅ Task 0: Project restructuring
- ✅ Task 1: DDQN implementation
- ✅ Task 2: Terminal reward system
- ✅ Task 3: UIEB dataset integration
- ✅ Task 4: Underwater actions
- ✅ Task 5: Color cast detection
- ✅ Task 6: Underwater degradation
- ✅ Task 7: Experiment framework
- ✅ Task 8: Classical baselines
- ⏳ Task 9: Final documentation (in progress)

See [docs/CURRENT_STATE_AND_CHALLENGES.md](docs/CURRENT_STATE_AND_CHALLENGES.md) for complete implementation details.
