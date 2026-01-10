# FLUX Image Generation Docker - Gradio Service
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="flux-serverless"
LABEL description="FLUX Image Generation Service with Gradio UI"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# CUDA architecture list for torch compilation
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0"

# Set up paths for model caching
ENV HF_HOME=/home/caches
ENV TRANSFORMERS_CACHE=/home/caches
ENV HUGGINGFACE_HUB_CACHE=/home/caches
ENV SAFETENSORS_CACHE_DIR=/home/caches
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace token - pass via environment variable at runtime
# ENV HF_TOKEN=your_token_here

# Gradio settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set aliases for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

# Install additional required packages
RUN pip install --no-cache-dir \
    gradio \
    diffusers \
    transformers \
    accelerate \
    huggingface_hub \
    peft \
    safetensors \
    sentencepiece \
    python-dotenv \
    Pillow \
    einops \
    omegaconf

# Install xformers for memory efficiency
RUN pip install --no-cache-dir xformers==0.0.24 || true

# Create cache directories
RUN mkdir -p /home/caches && \
    chmod -R 777 /home/caches

# Copy application files
COPY automation_12.py /app/automation_12.py

# Expose Gradio port
EXPOSE 7860

# Set the working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Gradio application
CMD ["python", "-u", "automation_12.py"]
