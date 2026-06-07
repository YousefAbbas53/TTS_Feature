# Use RunPod's official PyTorch base image (Python 3.11 + CUDA 12.4 + PyTorch 2.4.0 pre-installed)
# This avoids CUDA/torch install issues and matches the tested production environment
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set non-interactive to avoid prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Accept Coqui TOS license agreement (required to download XTTS model)
ENV COQUI_TOS_AGREED=1

# Install system audio libraries required by coqui-tts and ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Upgrade PyTorch to 2.7.0 with CUDA 12.8 (required for Blackwell GPUs - RTX PRO 6000 Series)
# PyTorch 2.4.0 in the base image does NOT have SM_100 (Blackwell) kernels
# Available cu128 versions: 2.7.0, 2.7.1, 2.8.0+ (2.6.0 does NOT exist in cu128)
RUN pip install --no-cache-dir --upgrade \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install coqui-tts and its dependencies first (heavy package - separate layer for caching)
RUN pip install --no-cache-dir \
    "coqui-tts>=0.24.2,<0.28" \
    coqpit-config \
    "transformers>=4.40.0,<5.0" \
    tokenizers

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default speaker WAV so workers don't need internet at startup
RUN mkdir -p /app/models && \
    curl -fsSL -o /app/models/default_speaker.wav \
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav" || \
    echo "WARNING: Speaker WAV download failed, will retry at runtime"

# Copy the preload script and run it to bake model weights into the image
COPY preload_model.py .
RUN python3 preload_model.py

# Copy the rest of the application code
COPY . .

# Run the RunPod serverless handler
CMD ["python3", "-u", "handler.py"]
