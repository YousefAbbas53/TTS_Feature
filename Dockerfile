# Use CUDA 12.4 base image - matches torch 2.5.1+cu124 used in production
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set non-interactive to avoid prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, ffmpeg, git and audio dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch 2.5.1 with CUDA 12.4 (matches tested production environment)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Set the working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the RunPod serverless handler
CMD ["python3", "-u", "handler.py"]
