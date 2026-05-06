#!/usr/bin/env bash
set -euo pipefail

# Setup local Python virtual environment and install dependencies.
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
