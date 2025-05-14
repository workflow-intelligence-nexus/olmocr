#!/bin/bash

# Check if PDF file argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No PDF file specified"
    echo "Usage: $0 <path_to_pdf> [target_image_dim]"
    echo "  target_image_dim: Optional. Target longest image dimension (default: 1600)"
    exit 1
fi

PDF_PATH="$1"
TARGET_DIM="${2:-1600}"  # Default to 1600 if not provided

# Get absolute path to PDF
PDF_ABSOLUTE_PATH=$(realpath "$PDF_PATH")

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

# Run pipeline with target dimension parameter
python -m olmocr.pipeline ./localworkspace --pdfs "$PDF_ABSOLUTE_PATH" --target_longest_image_dim $TARGET_DIM

# Deactivate conda environment when done
conda deactivate

exit $?