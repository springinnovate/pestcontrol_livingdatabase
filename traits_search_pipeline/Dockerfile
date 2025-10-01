# docker build -t pcld_trait_search_env:latest . && docker run --rm -it -v "%CD%":/app -w /app pcld_trait_search_env:latest

# GPU-enabled container with CUDA + cuDNN and Python 3.11 for HF/Transformers
# Build:
#   docker build -t pcld_trait_search_env:latest .
# Run (requires NVIDIA Container Toolkit):
#   docker run --rm -it --gpus all -v "%CD%":/app -w /app pcld_trait_search_env:latest
#
# Quick test inside container:
#   python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Base OS deps + Python 3.11 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git build-essential \
    procps psmisc htop less vim-tiny \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Make python/pip default to 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# (Optional) Speed up pip a bit and avoid warning about no distutils
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
               --extra-index-url https://pypi.org/simple \
               torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt

# If your requirements.txt does NOT pin torch+CUDA, uncomment one of the lines below:
# CUDA 12.1 wheels:
# RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# Or rely on your requirements.txt (recommended if already pinned):
RUN pip install -r requirements.txt

# Playwright browser deps + Chromium
RUN python -m pip install playwright && python -m playwright install --with-deps chromium

CMD ["bash"]
