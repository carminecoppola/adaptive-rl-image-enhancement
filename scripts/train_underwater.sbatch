#!/bin/bash
#SBATCH --job-name=underwater_training
#SBATCH --output=logs/sbatch_training_%j.log
#SBATCH --error=logs/sbatch_training_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=12

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

mkdir -p "${LOGS_ROOT}" "${LOGS_ROOT}/dqn" "${CHECKPOINT_ROOT}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
job_id="${SLURM_JOB_ID:-local}"
full_run_id="dqn_underwater_full_${timestamp}_${job_id}"

full_best_ckpt="${CHECKPOINT_ROOT}/dqn/${full_run_id}/dqn_best_policy_net.pt"
full_final_ckpt="${CHECKPOINT_ROOT}/dqn/${full_run_id}/dqn_final_policy_net.pt"
full_log="${LOGS_ROOT}/full_training_${job_id}.log"

echo "=========================================="
echo "Underwater RL Consolidation Pipeline"
echo "=========================================="
echo "Job ID: ${job_id}"
echo "Project root: ${PROJECT_ROOT}"
echo "Full run id : ${full_run_id}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "LOGS_ROOT=${LOGS_ROOT}"
echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "=========================================="

echo ""
echo "[1/3] Full training"
RUN_ID="${full_run_id}" python3 src/training/train.py \
  --experiment underwater_dqn_v1 \
  --phase full_training \
  2>&1 | tee -a "${full_log}"

if [ ! -f "${full_best_ckpt}" ]; then
  echo "Full best checkpoint not found: ${full_best_ckpt}"
  exit 1
fi
if [ ! -f "${full_final_ckpt}" ]; then
  echo "Full final checkpoint not found: ${full_final_ckpt}"
  exit 1
fi

echo ""
echo "[2/3] Full evaluation artifacts"
python3 src/evaluation/analyze_dqn_actions.py \
  --checkpoint "${full_best_ckpt}" \
  --num-images 50 \
  --output-name action_analysis_best.json

python3 src/evaluation/evaluation_dqn_baselines.py \
  --checkpoint "${full_best_ckpt}" \
  --num-images 50 \
  --output-name evaluation_baselines_best.json \
  --action-analysis-file action_analysis_best.json

python3 src/evaluation/analyze_dqn_actions.py \
  --checkpoint "${full_final_ckpt}" \
  --num-images 50 \
  --output-name action_analysis_final.json

python3 src/evaluation/evaluation_dqn_baselines.py \
  --checkpoint "${full_final_ckpt}" \
  --num-images 50 \
  --output-name evaluation_baselines_final.json \
  --action-analysis-file action_analysis_final.json

python3 src/evaluation/evaluate_underwater_ood.py \
  --checkpoint "${full_best_ckpt}" \
  --output-name evaluation_ood_challenging60.json

python3 src/evaluation/generate_underwater_report.py \
  --run-dir "${LOGS_ROOT}/dqn/${full_run_id}"

echo ""
echo "[3/3] Pipeline complete"
echo "Run directory: ${LOGS_ROOT}/dqn/${full_run_id}"
echo "Best checkpoint: ${full_best_ckpt}"
echo "Final checkpoint: ${full_final_ckpt}"
echo "Report: ${LOGS_ROOT}/dqn/${full_run_id}/underwater_results.md"
