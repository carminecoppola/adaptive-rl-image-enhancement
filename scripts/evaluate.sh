#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-}"

source venv/bin/activate

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Copy .env.example to .env and update values first."
  exit 1
fi

if [ -n "$CHECKPOINT" ]; then
  python src/evaluation/evaluation_dqn_baselines.py --checkpoint "$CHECKPOINT"
else
  python src/evaluation/evaluation_dqn_baselines.py
fi
