#!/usr/bin/env bash
set -euo pipefail

# Create a local Python environment and install runtime dependencies.
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "${1:-}" == "--dev" ]]; then
  python -m pip install -r requirements-dev.txt
fi
