#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment.
source venv/bin/activate

# Load environment variables.
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Copy .env.example to .env and update values first."
  exit 1
fi

# Placeholder evaluation invocation.
python -m src.evaluation.evaluate_agent
