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
    wget -q -O /app/models/default_speaker.wav \
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav" || \
    echo "WARNING: Speaker WAV download failed, will retry at runtime"

# Pre-download the XTTS v2 model weights into the image (avoids cold-start downloads)
RUN python3 -c "
import os, traceback
os.environ['COQUI_TOS_AGREED'] = '1'
print('Loading TTS model...')
try:
    from TTS.api import TTS
    model = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
    print('Model loaded successfully!')
except Exception as e:
    print('ERROR loading model:', e)
    traceback.print_exc()
    raise
"

# Copy the rest of the application code
COPY . .

# Run the RunPod serverless handler
CMD ["python3", "-u", "handler.py"]
