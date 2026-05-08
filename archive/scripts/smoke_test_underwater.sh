#!/bin/bash
# Smoke Test for Underwater Image Enhancement DQN
# Validates entire pipeline before full training
# Usage: ./scripts/smoke_test_underwater.sh

set -e  # Exit on error

echo "=========================================="
echo "🏊 Underwater DQN: Smoke Test (Phase 12)"
echo "=========================================="
echo ""

# Set environment variables
export UIEB_ROOT="${UIEB_ROOT:-/mnt/uieb}"
export DATASET_ROOT="${DATASET_ROOT:-/mnt/datasets}"
export LOGS_ROOT="${LOGS_ROOT:-./logs}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-./checkpoints}"

# Create required directories
mkdir -p "$LOGS_ROOT"
mkdir -p "$CHECKPOINTS_ROOT"

echo "Environment variables:"
echo "  UIEB_ROOT=$UIEB_ROOT"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  LOGS_ROOT=$LOGS_ROOT"
echo "  CHECKPOINTS_ROOT=$CHECKPOINTS_ROOT"
echo ""

# Check if UIEB dataset exists
if [ ! -d "$UIEB_ROOT" ]; then
    echo "⚠ WARNING: UIEB_ROOT not found at $UIEB_ROOT"
    echo "  Dataset may not be available. Training may fail."
    echo "  Set UIEB_ROOT environment variable to the UIEB dataset path."
    echo ""
fi

# Activate venv if available
if [ -d "./venv" ]; then
    echo "Activating Python venv..."
    source ./venv/bin/activate
    echo "✓ venv activated"
    echo ""
fi

# Print Python info
echo "Python environment:"
python --version
echo "  Path: $(which python)"
echo ""

# Run smoke test training
echo "Starting smoke test training..."
echo "Config: configs/experiments/underwater_dqn_v1.yaml"
echo "Parameters:"
echo "  - Episodes: 1000 (smoke test)"
echo "  - Dataset: UIEB"
echo "  - Subset: 500 training images"
echo "  - Seed: 42 (deterministic)"
echo ""

python src/training/train.py \
    --experiment underwater_dqn_v1 \
    2>&1 | tee "${LOGS_ROOT}/smoke_test.log"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Smoke test training completed successfully!"
else
    echo ""
    echo "✗ Smoke test training failed with exit code $TRAIN_EXIT_CODE"
    echo "  See logs at: ${LOGS_ROOT}/smoke_test.log"
    exit $TRAIN_EXIT_CODE
fi

# Validation checks
echo ""
echo "=========================================="
echo "📋 Smoke Test Validation"
echo "=========================================="
echo ""

# Check if log directory exists
LATEST_LOG=$(ls -td "${LOGS_ROOT}"/dqn_underwater_v1* 2>/dev/null | head -n1)

if [ -z "$LATEST_LOG" ]; then
    echo "✗ No training log directory found!"
    exit 1
fi

echo "Found training log: $(basename "$LATEST_LOG")"

# Run validation checks (Python script)
python << 'EOF'
import json
import os
from pathlib import Path

LOGS_ROOT = os.getenv('LOGS_ROOT', './logs')
LATEST_LOG = sorted(Path(LOGS_ROOT).glob('dqn_underwater_v1*'), key=os.path.getmtime, reverse=True)

if not LATEST_LOG:
    print("✗ No training logs found")
    exit(1)

log_dir = LATEST_LOG[0]
print(f"Validating: {log_dir.name}")

# Success criteria
criteria = {
    'training_ran': False,
    'has_artifacts': False,
    'has_metrics': False,
    'mean_delta_psnr_positive': None,
    'stop_rate_positive': None,
}

# Check training artifacts
if (log_dir / 'training_metadata.json').exists():
    criteria['has_artifacts'] = True
    print("✓ Training metadata found")

if (log_dir / 'episode_summary.csv').exists():
    criteria['has_metrics'] = True
    print("✓ Episode metrics found")
    
    # Parse metrics
    import pandas as pd
    df = pd.read_csv(log_dir / 'episode_summary.csv')
    
    if 'mean_delta_psnr' in df.columns:
        mean_delta = df['mean_delta_psnr'].mean()
        criteria['mean_delta_psnr_positive'] = mean_delta > 0
        print(f"  Mean PSNR delta: {mean_delta:.4f} {'✓' if mean_delta > 0 else '✗'}")
    
    if 'stop_rate' in df.columns:
        stop_rate = df['stop_rate'].mean()
        criteria['stop_rate_positive'] = stop_rate > 0.01
        print(f"  Mean stop rate: {stop_rate:.4f} {'✓' if stop_rate > 0.01 else '✗'}")

# Summary
print("")
print("Validation summary:")
passed = sum(1 for v in criteria.values() if v is True)
failed = sum(1 for v in criteria.values() if v is False)

for key, value in criteria.items():
    status = "✓" if value is True else ("✗" if value is False else "?")
    print(f"  {status} {key}")

if failed == 0 and passed >= 2:
    print("")
    print("✓✓✓ SMOKE TEST PASSED ✓✓✓")
    print("Ready for full training!")
    exit(0)
else:
    print("")
    print("⚠ Smoke test validation incomplete or failed")
    exit(1)
EOF

exit $?
