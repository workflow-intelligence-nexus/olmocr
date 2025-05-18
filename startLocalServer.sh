#!/bin/bash
# Enhanced startup script with health checks and pipeline integration

# --- Configuration ---
SGLANG_PORT=30024 # Default SGLang port
HTTP_PORT=8000    # Port for our FastAPI service
NUM_WORKERS=4     # Number of Uvicorn workers
MAX_SGLANG_ATTEMPTS=300 # Max 5 minutes for SGLang to start

# --- Environment Setup ---
WSL_IP=$(ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
if [ -z "$WSL_IP" ]; then
    echo "Warning: Could not determine WSL IP address for eth0. Falling back to localhost."
    WSL_IP="localhost"
fi
echo "WSL IP determined as: $WSL_IP"

cd /mnt/f/GithubToolRepos/olmocrLinux # Ensure we are in the correct directory
source ~/miniconda/etc/profile.d/conda.sh
conda activate olmocr311 # Activate the correct conda environment

# Environment variables required by olmocr.pipeline or http_service
export CUDA_VISIBLE_DEVICES=1 # Corrected to 1 as per user feedback (A6000)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_HOME=/usr # This might be system-specific or set elsewhere

# Environment variables that might be used by http_service.py or pipeline.py
export SGLANG_SERVER_PORT=$SGLANG_PORT
# Add other necessary exports from pipeline.py if http_service.py relies on them
# export MODEL_MAX_CONTEXT=8192
# export TARGET_ANCHOR_TEXT_LEN=6000
# export MAX_PAGE_RETRIES=8

# --- Cleanup Handler ---
SGLANG_PID=""
HTTP_PID=""

cleanup() {
    echo "Stopping services..."
    if [ ! -z "$HTTP_PID" ]; then
        kill $HTTP_PID 2>/dev/null
    fi
    if [ ! -z "$SGLANG_PID" ]; then
        kill $SGLANG_PID 2>/dev/null
    fi
    echo "Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Install Dependencies ---
echo "Installing API dependencies (fastapi, uvicorn, python-multipart, httpx)..."
pip install fastapi uvicorn python-multipart httpx
if [ $? -ne 0 ]; then
    echo "Failed to install API dependencies. Exiting."
    exit 1
fi

# --- Ensure Port is Free ---
echo "Attempting to free port $SGLANG_PORT if it's in use..."
fuser -k -n tcp $SGLANG_PORT || echo "Port $SGLANG_PORT was already free or fuser command failed (non-critical)."
sleep 1 # Give a moment for the port to be released

# --- Start SGLang Server ---
echo "Starting SGLang server on port $SGLANG_PORT..."
python -m sglang.launch_server \
  --model-path allenai/olmOCR-7B-0225-preview \
  --port $SGLANG_PORT \
  --device cuda \
  --dtype auto \
  --chat-template qwen2-vl \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 &
SGLANG_PID=$!

# --- SGLang Health Check ---
echo "Waiting for SGLang server to become ready (max ${MAX_SGLANG_ATTEMPTS}s)..."
ATTEMPT=1
SGLANG_READY_URL="http://localhost:$SGLANG_PORT/v1/models"
while [ $ATTEMPT -le $MAX_SGLANG_ATTEMPTS ]; do
    # Check if SGLang process is still running
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "SGLang server process terminated unexpectedly. Check SGLang logs."
        exit 1
    fi

    # Attempt to connect
    if curl --silent --fail --max-time 2 $SGLANG_READY_URL -o /dev/null; then
        echo "SGLang server is ready after $ATTEMPT seconds."
        break
    fi
    
    # Optional: print a waiting message less frequently
    if [ $((ATTEMPT % 10)) -eq 0 ]; then
        echo "Still waiting for SGLang server... ($ATTEMPT/$MAX_SGLANG_ATTEMPTS attempts)"
    fi
    
    ATTEMPT=$((ATTEMPT+1))
    sleep 1
done

if [ $ATTEMPT -gt $MAX_SGLANG_ATTEMPTS ]; then
    echo "ERROR: SGLang server did not become ready after $MAX_SGLANG_ATTEMPTS seconds."
    echo "Attempted to reach $SGLANG_READY_URL"
    kill $SGLANG_PID 2>/dev/null # Ensure SGLang is killed if it didn't start properly
    exit 1
fi

# --- Start HTTP Service ---
echo "Starting HTTP service (FastAPI with Uvicorn) on port $HTTP_PORT..."
# The SGLANG_SERVER_URL might be used by http_service.py if it needs to make direct calls,
# though the current http_service.py uses localhost and SGLANG_SERVER_PORT env var.
# For clarity, we can set it.
export SGLANG_SERVER_URL="http://$WSL_IP:$SGLANG_PORT" 

uvicorn http_service:app --host 0.0.0.0 --port $HTTP_PORT --workers $NUM_WORKERS --timeout-keep-alive 300 &
HTTP_PID=$!

echo "--- Server Status ---"
echo "SGLang Server PID: $SGLANG_PID (Port: $SGLANG_PORT)"
echo "HTTP Service PID: $HTTP_PID (URL: http://$WSL_IP:$HTTP_PORT)"
echo "API Endpoints:"
echo "  POST /process-pdf"
echo "  GET  /queue-status"
echo "  GET  /system-health"
echo "Press Ctrl+C to stop all services."

# Wait for PIDs to exit. If one exits, cleanup will be called.
wait $SGLANG_PID $HTTP_PID
# If wait returns (e.g., if one process exits cleanly not via SIGINT/SIGTERM), call cleanup.
cleanup
