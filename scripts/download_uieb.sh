#!/bin/bash
# Download and setup UIEB (Underwater Image Enhancement Benchmark) dataset
# 
# IMPORTANT: The UIEB dataset requires manual download and license acceptance.
# Visit: https://li-chongyi.github.io/proj_benchmark.html
#
# This script provides setup instructions and directory creation.

set -e

# Get dataset root from environment or use default
DATASET_ROOT="${DATASET_ROOT:-.}"
UIEB_DIR="$DATASET_ROOT/UIEB"

echo "=========================================="
echo "UIEB Dataset Setup"
echo "=========================================="
echo ""
echo "Dataset root: $DATASET_ROOT"
echo "UIEB directory: $UIEB_DIR"
echo ""

# Check if already downloaded
if [ -d "$UIEB_DIR/raw-890" ] && [ -d "$UIEB_DIR/reference-890" ]; then
    echo "✓ UIEB dataset already found at $UIEB_DIR"
    echo "  Checking file counts..."
    RAW_COUNT=$(find "$UIEB_DIR/raw-890" -type f | wc -l)
    REF_COUNT=$(find "$UIEB_DIR/reference-890" -type f | wc -l)
    echo "  - raw-890/: $RAW_COUNT files"
    echo "  - reference-890/: $REF_COUNT files"
    echo ""
    exit 0
fi

# Create directories
echo "Creating UIEB directories..."
mkdir -p "$UIEB_DIR/raw-890"
mkdir -p "$UIEB_DIR/reference-890"
echo ""

# Print download instructions
echo "=========================================="
echo "MANUAL DOWNLOAD REQUIRED"
echo "=========================================="
echo ""
echo "The UIEB dataset requires manual download due to licensing restrictions."
echo ""
echo "Steps:"
echo "1. Visit: https://li-chongyi.github.io/proj_benchmark.html"
echo "2. Accept the dataset terms and conditions"
echo "3. Download the UIEB dataset (890 pairs)"
echo "4. Extract the archive to:"
echo "   $UIEB_DIR/"
echo ""
echo "Expected structure after extraction:"
echo "  $UIEB_DIR/"
echo "  ├── raw-890/         (890 degraded underwater images)"
echo "  └── reference-890/   (890 reference high-quality images)"
echo ""
echo "File naming convention:"
echo "  raw-890/123.png      (input: degraded image)"
echo "  reference-890/123.png (reference: clean image)"
echo ""
echo "After download, verify with:"
echo "  find $UIEB_DIR/raw-890 -type f | wc -l"
echo "  find $UIEB_DIR/reference-890 -type f | wc -l"
echo ""
echo "Both should show ~890 files."
echo "=========================================="
