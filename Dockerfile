# Use an official NVIDIA CUDA base image with Ubuntu 22.04 to ensure Vast.ai compatibility and GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive to avoid prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, ffmpeg, and libsndfile1 for audio processing stability
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip for better dependency resolution
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch explicitly for CUDA 11.8 to ensure full RTX 3090 / Vast.ai compatibility
# (Done before other requirements to ensure correct indexing)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
