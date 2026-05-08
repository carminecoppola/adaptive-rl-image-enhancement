#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-}"
PHASE="${2:-}"

source venv/bin/activate

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Copy .env.example to .env and update values first."
  exit 1
fi

cmd=(python src/training/train.py)
if [ -n "$EXPERIMENT" ]; then
  cmd+=(--experiment "$EXPERIMENT")
fi
if [ -n "$PHASE" ]; then
  cmd+=(--phase "$PHASE")
fi

"${cmd[@]}"
