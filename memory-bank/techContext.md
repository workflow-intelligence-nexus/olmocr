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
   - Docker support available

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
   - Specific A6000 GPU configuration:
     - CUDA_HOME=/usr/local/cuda-12.8
     - CUDA_DEVICE_ORDER=PCI_BUS_ID
     - CUDA_VISIBLE_DEVICES=1
     - SGLang server with half precision (--dtype half)

2. **Parallel Processing**:
   - Work queue implementation (`work_queue.py`)
   - S3 utilities for distributed processing
