# DeepSeek-OCR Dockerfile
# Requires NVIDIA GPU and nvidia-docker runtime

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set TRITON_PTXAS_PATH for CUDA 11.8
ENV TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
ENV VLLM_USE_V1=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Download and install vLLM wheel for CUDA 11.8
RUN wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp312-abi3-manylinux1_x86_64.whl && \
    pip install vllm-0.8.5+cu118-cp312-abi3-manylinux1_x86_64.whl && \
    rm vllm-0.8.5+cu118-cp312-abi3-manylinux1_x86_64.whl

# Install other requirements
RUN pip install -r requirements.txt

# Install flash-attention (this may take a while)
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Copy the application code
COPY DeepSeek-OCR-master/ /app/DeepSeek-OCR-master/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set the working directory to vllm
WORKDIR /app/DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Set CUDA device (can be overridden)
ENV CUDA_VISIBLE_DEVICES=0

# Default command - run bash for interactive use
CMD ["/bin/bash"]
