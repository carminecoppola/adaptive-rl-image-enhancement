#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-}"

source venv/bin/activate

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Copy .env.example to .env and update values first."
  exit 1
fi

if [ -n "$EXPERIMENT" ]; then
  python src/training/train.py --experiment "$EXPERIMENT"
else
  python src/training/train.py
fi
