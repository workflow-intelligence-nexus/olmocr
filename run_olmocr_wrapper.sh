#!/bin/bash

# Set up WSL2-specific paths
export WSL_INTEROP=/run/WSL/1
export WSL_DISTRO_NAME=Ubuntu-24.04
export WSLENV=WSL_INTEROP:WSL_DISTRO_NAME

# Set up CUDA paths from your Windows installation
export PATH="$PATH:/usr/lib/wsl/lib:/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"

# Source conda
source ~/miniconda/etc/profile.d/conda.sh

# Activate conda environment
conda activate olmocr311

# Debug: Log the environment
{
    echo "=== Environment ==="
    env
    echo -e "\n=== GPU Info ==="
    nvidia-smi
    echo -e "\n=== CUDA Check ==="
    which nvcc 2>&1
    nvcc --version 2>&1
} > /mnt/f/GithubToolRepos/olmocrLinux/wrapper_debug.log 2>&1

# Change to script directory
cd /mnt/f/GithubToolRepos/olmocrLinux

# Run the script with the current environment
exec ./run_olmocr_background.sh