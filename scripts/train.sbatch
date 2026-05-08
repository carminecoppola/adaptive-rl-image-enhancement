#!/usr/bin/env bash
#SBATCH --job-name=adaptive_dqn
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_ROOT="/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement"
cd "$PROJECT_ROOT"

EXPERIMENT="${1:-${EXPERIMENT:-}}"
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: sbatch scripts/train.sbatch <experiment_name>"
  echo "or:    sbatch --export=ALL,EXPERIMENT=phase_a_sanity scripts/train.sbatch"
  exit 1
fi

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

if [ ! -f "configs/experiments/${EXPERIMENT}.yaml" ]; then
  echo "Experiment config not found: configs/experiments/${EXPERIMENT}.yaml"
  exit 1
fi

mkdir -p "${LOGS_ROOT}/dqn" "${CHECKPOINT_ROOT}"

python3 src/training/train.py --experiment "$EXPERIMENT"
