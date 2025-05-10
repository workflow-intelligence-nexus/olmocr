FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=1

# Install system dependencies
RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update

# Install PDF-specific dependencies
RUN apt-get update && apt-get -y install python3-apt
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update -y && apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

# Install Python and development tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    ca-certificates \
    build-essential \
    curl \
    unzip

# Set up Python
RUN rm -rf /var/lib/apt/lists/* \
    && unlink /usr/bin/python3 \
    && ln -s /usr/bin/python3.11 /usr/bin/python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && pip3 install -U pip

# Install uv package manager for faster installations
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml pyproject.toml
COPY olmocr/version.py olmocr/version.py
COPY requirements-http.txt requirements-http.txt
COPY gantry-requirements.txt gantry-requirements.txt

# Install dependencies with specific NumPy version for compatibility
RUN /root/.local/bin/uv pip install --system --no-cache numpy==1.24.3
RUN /root/.local/bin/uv pip install --system --no-cache -e .
RUN /root/.local/bin/uv pip install --system --no-cache -r requirements-http.txt

# Install SGLang with compatible dependencies for CUDA 11.8
RUN /root/.local/bin/uv pip install --system --no-cache torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN /root/.local/bin/uv pip install --system --no-cache "sglang[all]==0.4.2" --no-deps
RUN /root/.local/bin/uv pip install --system --no-cache sgl-kernel==0.0.3.post1 --no-deps

# Install additional dependencies that SGLang needs
RUN /root/.local/bin/uv pip install --system --no-cache transformers accelerate sentencepiece protobuf

# Copy the rest of the application
COPY olmocr olmocr
COPY http_service.py http_service.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "http_service:app", "--host", "0.0.0.0", "--port", "8000"]
