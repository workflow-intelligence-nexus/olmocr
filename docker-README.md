# OLMoCR Docker Deployment for Windows 11

This guide explains how to containerize and run the OLMoCR application with FastAPI on Windows 11 to make it accessible from your n8n environment.

## Prerequisites

- Docker Desktop for Windows installed
- WSL2 enabled on Windows 11
- NVIDIA Container Toolkit for Windows installed
- NVIDIA GPU drivers installed on Windows
- A6000 GPU available on the host machine

## Building and Running the Container

### Option 1: Using Docker Compose (Recommended)

1. Build and start the container:

```bash
docker-compose up -d
```

2. Check if the container is running:

```bash
docker-compose ps
```

3. View logs:

```bash
docker-compose logs -f
```

4. Stop the container:

```bash
docker-compose down
```

### Option 2: Using Docker Directly

1. Build the Docker image:

```bash
docker build -t olmocr-api:latest .
```

2. Run the container:

```bash
docker run -d --name olmocr-api \
  --gpus '"device=1"' \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  olmocr-api:latest
```

## Accessing the API

Once the container is running, you can access the FastAPI service:

- API Documentation: http://localhost:8000/docs
- API Endpoint for PDF processing: http://localhost:8000/process-pdf

## Connecting to n8n

To connect your n8n environment to the OLMoCR API:

1. Make sure both n8n and OLMoCR containers are on the same Docker network.
   - If using the provided docker-compose.yml, the network is named `olmocr-network`
   - If your n8n is running in a different Docker Compose setup, add the following to your n8n's docker-compose.yml:

```yaml
networks:
  default:
    external:
      name: olmocr-network
```

2. In n8n, use the HTTP Request node to interact with the OLMoCR API:
   - URL: `http://olmocr-api:8000/process-pdf`
   - Method: POST
   - Binary Data: Enable this option and select your PDF file
   - Headers: `Content-Type: multipart/form-data`

## Windows 11 Specific Setup

### Setting up NVIDIA Container Toolkit on Windows

1. Install the latest NVIDIA drivers for your GPU from the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page.

2. Install Docker Desktop for Windows and ensure WSL2 is enabled:
   - In PowerShell (as Administrator), run: `wsl --set-default-version 2`
   - Verify with: `wsl -l -v`

3. Install NVIDIA Container Toolkit for Windows:
   - Download and install [NVIDIA Container Toolkit for Windows](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-nvidia-drivers)
   - Follow the instructions for WSL2 GPU support

4. Configure Docker Desktop:
   - Open Docker Desktop → Settings → Resources → WSL Integration
   - Enable integration with your Ubuntu distribution
   - In Settings → Docker Engine, add the following to your configuration:
   ```json
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```
   - Click "Apply & Restart"

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues:

1. Verify NVIDIA drivers are properly installed on Windows:
```powershell
nvidia-smi
```

2. Check if Docker can access the GPU:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

3. Ensure the correct GPU (A6000) is being used by setting `CUDA_VISIBLE_DEVICES=1` in the environment variables.

### Connection Issues

If n8n cannot connect to the OLMoCR API:

1. Verify both containers are running:
```bash
docker ps
```

2. Check they're on the same network:
```bash
docker network inspect olmocr-network
```

3. Test the API is accessible from within the n8n container:
```bash
docker exec -it [n8n-container-id] curl -v http://olmocr-api:8000/health
```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /process-pdf`: Process a PDF file
  - Request: Multipart form with a file field
  - Response: JSON with processing results
