#!/bin/bash

# Check if PDF file argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No PDF file specified"
    echo "Usage: $0 <path_to_pdf>"
    exit 1
fi

PDF_PATH="$1"

# Verify PDF file exists
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF file not found: $PDF_PATH"
    exit 1
fi

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

# Run pipeline with A6000 GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m olmocr.pipeline ./localworkspace --pdfs "$PDF_PATH"

exit $?
