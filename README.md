# Adaptive Image Enhancement via Deep Reinforcement Learning

## Project Goal
This project explores reinforcement learning for image enhancement. An RL agent will learn to improve degraded images by applying a sequence of image-processing actions.

## RL Formulation (High Level)
- **State:** current degraded/enhanced image
- **Action:** one image-processing operation (brightness, contrast, denoise, sharpen, gamma, stop)
- **Reward:** improvement in image quality metrics (PSNR/SSIM), optionally with a step penalty

## Project Phases
1. Dataset download and organization
2. Degradation generation pipeline
3. RL environment definition
4. Baseline fixed-filter pipelines
5. DQN training
6. Evaluation and reporting

## Environment Setup
```bash
cd adaptive-rl-image-enhancement
bash scripts/setup_env.sh
```

## Dataset Storage Policy
- Datasets and large artifacts are stored under `/storage/internal_02`.
- Datasets are **not** stored inside this repository.
- Configurable roots are managed with `.env` variables and YAML config files in `configs/`.

## Common Commands
```bash
bash scripts/download_dataset.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

## Notes
- Current codebase is a scaffold with placeholders and TODOs.
- ML training logic and environment dynamics are intentionally not implemented yet.
