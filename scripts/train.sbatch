#!/usr/bin/env bash
#SBATCH --job-name=adaptive_dqn_debug
#SBATCH --output=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/logs/slurm/%x_%j.out
#SBATCH --error=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/logs/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_ROOT="/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement"
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
  echo "venv not found. Run scripts/setup_env.sh first."
  exit 1
fi

source venv/bin/activate

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Copy .env.example to .env first."
  exit 1
fi

mkdir -p "${LOGS_ROOT}/slurm" "${LOGS_ROOT}/dqn" "${CHECKPOINT_ROOT}"

# Diagnostic logging for reward analysis.
export DQN_DEBUG_REWARD=1
export DQN_DEBUG_EPISODES=10
export RUN_ID="dqn_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

python3 src/training/train_dqn.py
