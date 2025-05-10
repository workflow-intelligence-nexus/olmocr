# Technical Context: OLMoCR

## Core Technologies
1. **Python** (primary implementation language)
2. **PyTorch** (machine learning components)
3. **PDF processing libraries**:
   - pdf2image
   - pdfminer
   - pypdf
4. **Computer Vision**:
   - OpenCV
   - PIL/Pillow

## Development Environment
1. **Project Structure**:
   - Python package (`olmocr/`)
   - Documentation (`docs/`)
   - Tests (`tests/`)
   - Scripts (`scripts/`)

2. **Build System**:
   - Makefile for common tasks
   - pyproject.toml for Python packaging
   - Docker containerization with docker-compose.yml
   - Windows 11 compatible Docker configuration

3. **Dependencies**:
   - Managed via pyproject.toml
   - Specific requirements in gantry-requirements.txt

## Key Development Patterns
1. **Testing**:
   - Unit tests in `tests/`
   - Test PDFs in `tests/gnarly_pdfs/`
   - Integration test script available

2. **Documentation**:
   - Sphinx-based documentation
   - ReadTheDocs integration
   - CHANGELOG.md for version history

3. **Interface Patterns**:
   - Async/Sync wrapper pattern for HTTP service
   - process_pdf_file_async() async function for FastAPI integration
   - process_pdf_file() sync wrapper for CLI and non-async contexts

3. **CI/CD**:
   - Docker build scripts
   - Beaker scripts for GPU execution
   - Release process documented in RELEASE_PROCESS.md

## Performance Considerations
1. **GPU Acceleration**:
   - Available for ML components
   - Docker configurations provided
   - Benchmarks in `scripts/benchmark_throughput.py`
   - Specific A6000 GPU configuration (local environment):
     - CUDA_HOME=/usr/local/cuda-12.8
     - CUDA_DEVICE_ORDER=PCI_BUS_ID
     - CUDA_VISIBLE_DEVICES=1
     - SGLang server with half precision (--dtype half)
   - Docker container GPU configuration:
     - CUDA 11.8 with cuDNN 8
     - NumPy 1.24.3 for compatibility
     - PyTorch 2.1.2 with CUDA 11.8 support
     - WSL2 integration for Windows 11

2. **Parallel Processing**:
   - Work queue implementation (`work_queue.py`)
   - S3 utilities for distributed processing

## Docker Deployment
1. **Container Configuration**:
   - Base image: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
   - FastAPI service exposed on port 8000
   - GPU passthrough via NVIDIA Container Toolkit

2. **Windows 11 Integration**:
   - WSL2 backend for Docker Desktop
   - NVIDIA GPU drivers on Windows host
   - Port forwarding from Windows host to container
   - Network configuration for n8n integration
