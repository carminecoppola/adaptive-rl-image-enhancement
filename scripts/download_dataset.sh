#!/bin/bash

set -e

source venv/bin/activate

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

mkdir -p "$DATASET_ROOT"

python -m src.data.download_dataset