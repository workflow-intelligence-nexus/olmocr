#!/bin/bash
# Start SGLang and HTTP service for n8n integration

# Get WSL IP address
WSL_IP=$(ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "WSL IP address: $WSL_IP"

# Set up environment
cd /mnt/f/GithubToolRepos/olmocrLinux
source ~/miniconda/etc/profile.d/conda.sh
conda activate olmocr311
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr

# Start SGLang server in background
echo "Starting SGLang server..."
python -m sglang.launch_server \
  --model-path allenai/olmOCR-7B-0225-preview \
  --port 30024 \
  --device cuda \
  --dtype half \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  --host 0.0.0.0 &

# Wait for SGLang server to initialize
echo "Waiting for SGLang server to initialize..."
sleep 30

# Start HTTP service with correct SGLang URL
echo "Starting HTTP service..."
SGLANG_SERVER_URL="http://$WSL_IP:30024" uvicorn http_service:app --host 0.0.0.0 --port 8000