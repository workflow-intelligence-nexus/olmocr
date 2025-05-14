#!/bin/bash

# Check if PDF file argument is provided
if [ $# -lt 1 ]; then
    echo "Error: No PDF file specified"
    echo "Usage: $0 <path_to_pdf> [uid] [target_image_dim]"
    echo "  uid: Optional. Unique identifier for the output file"
    echo "  target_image_dim: Optional. Target longest image dimension (default: 1600)"
    exit 1
fi

PDF_PATH="$1"
CUSTOM_ID="${2:-}"
TARGET_DIM="${3:-1600}"  # Default to 1600 if not provided

# Get absolute path to PDF
PDF_ABSOLUTE_PATH=$(realpath "$PDF_PATH")

# Extract filename without extension for output naming
PDF_FILENAME=$(basename "$PDF_PATH")
PDF_NAME_NO_EXT="${PDF_FILENAME%.*}"

# Verify PDF file exists
if [ ! -f "$PDF_ABSOLUTE_PATH" ]; then
    echo "Error: PDF file not found: $PDF_PATH"
    exit 1
fi

# Source conda.sh to enable conda command
source ~/miniconda/etc/profile.d/conda.sh

# Activate conda environment
conda activate olmocr311

# Set CUDA environment for Ubuntu 24.04
export CUDA_VISIBLE_DEVICES=1  # Use the RTX A6000 (GPU 1)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr  # The base directory where CUDA is installed on Ubuntu 24.04

# Create a temporary directory for custom output
TMP_WORKSPACE="./localworkspace_tmp"
mkdir -p "$TMP_WORKSPACE/results"

# Run pipeline with target dimension parameter
python -m olmocr.pipeline "$TMP_WORKSPACE" --pdfs "$PDF_ABSOLUTE_PATH" --target_longest_image_dim $TARGET_DIM

# Find the generated output file
OUTPUT_FILE=$(find "$TMP_WORKSPACE/results" -name "*.jsonl" | head -n 1)

if [ -n "$OUTPUT_FILE" ]; then
    # Create the final output filename
    if [ -n "$CUSTOM_ID" ]; then
        FINAL_FILENAME="./localworkspace/results/${PDF_NAME_NO_EXT}_${CUSTOM_ID}.jsonl"
    else
        FINAL_FILENAME="./localworkspace/results/${PDF_NAME_NO_EXT}.jsonl"
    fi
    
    # Copy the file to the desired location with the new name
    mkdir -p "./localworkspace/results"
    cp "$OUTPUT_FILE" "$FINAL_FILENAME"
    echo "Output saved to: $FINAL_FILENAME"
    
    # Clean up temporary workspace
    rm -rf "$TMP_WORKSPACE"
fi

# Deactivate conda environment when done
conda deactivate

exit $?