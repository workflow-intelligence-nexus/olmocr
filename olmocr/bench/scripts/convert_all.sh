#!/bin/bash

set -e

# Function to create conda environment if it doesn't exist
create_conda_env() {
    env_name=$1
    python_version=$2
    
    # Check if environment exists
    if conda info --envs | grep -q "^$env_name "; then
        echo "Environment $env_name already exists, using it."
    else
        echo "Creating conda environment: $env_name"
        conda create -y -n $env_name python=$python_version
    fi
}

# # Create and activate olmocr environment
# create_conda_env "olmocr" "3.11"
# source $(conda info --base)/etc/profile.d/conda.sh
# source activate olmocr

# # Run olmocr benchmarks
# echo "Running olmocr benchmarks..."
# python -m olmocr.bench.convert olmocr --repeats 5

# # Install marker-pdf and run benchmarks
# echo "Installing marker-pdf and running benchmarks..."
# pip install marker-pdf
# python -m olmocr.bench.convert marker

# # Install verovio and run benchmarks
# echo "Installing verovio and running benchmarks..."
# pip install verovio
# python -m olmocr.bench.convert gotocr

# # Run chatgpt benchmarks
# echo "Running chatgpt benchmarks..."
# python -m olmocr.bench.convert chatgpt

# Create and activate mineru environment
create_conda_env "mineru" "3.11"
source activate mineru

# Install magic-pdf and run benchmarks
# TODO: Fix this, I was not able to get it to all install successfully
# echo "Installing magic-pdf and running mineru benchmarks..."
# pip install -U "magic-pdf[full]==1.2.2" --extra-index-url https://wheels.myhloli.com
# python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# pip install huggingface_hub Pillow paddleocr ultralytics doclayout-yolo pycocotools 
# wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
# python download_models_hf.py
# python -m olmocr.bench.convert mineru

echo "All benchmarks completed successfully."