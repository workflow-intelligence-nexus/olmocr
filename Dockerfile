FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Use the correct CUDA version that matches the base image
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
# Ensure we use the A6000 GPU (device 1) with 48GB VRAM instead of the RTX 5090
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=1

# Install system dependencies
RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update

# Install PDF and font dependencies
RUN apt-get update && apt-get -y install \
    python3-apt \
    poppler-utils \
    ttf-mscorefonts-installer \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    fontconfig \
    libgl1-mesa-glx

# Accept font EULA and rebuild font cache
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN fc-cache -f -v

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

# Install core PDF processing dependencies
RUN /root/.local/bin/uv pip install --system --no-cache \
    numpy==1.24.3 \
    pdf2image==1.17.0 \
    pdfminer.six==20231228 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.2.0

# Install project dependencies
RUN /root/.local/bin/uv pip install --system --no-cache -e .
RUN /root/.local/bin/uv pip install --system --no-cache -r requirements-http.txt

# Install SGLang with compatible dependencies for CUDA 12.x
RUN /root/.local/bin/uv pip install --system --no-cache torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
RUN /root/.local/bin/uv pip install --system --no-cache "sglang[all]==0.4.2" --no-deps
RUN /root/.local/bin/uv pip install --system --no-cache sgl-kernel==0.0.3.post1 --no-deps

# Install all dependencies that SGLang needs to avoid further errors
RUN /root/.local/bin/uv pip install --system --no-cache \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    ipython \
    aiohttp \
    pydantic \
    tqdm \
    psutil \
    numpy \
    packaging \
    typing-extensions \
    requests \
    huggingface-hub \
    safetensors \
    tokenizers \
    einops \
    websockets \
    setproctitle \
    pyzmq \
    vllm==0.4.0  # Pin to a specific version compatible with SGLang 0.4.2

# Copy the rest of the application
COPY olmocr olmocr
COPY http_service.py http_service.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "http_service:app", "--host", "0.0.0.0", "--port", "8000"]
