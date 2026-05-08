# Lightweight Python image for Hugging Face Spaces (no GPU needed for edge-tts)
FROM python:3.10-slim

# Set non-interactive to avoid prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg and libsndfile1 for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create required directories
RUN mkdir -p outputs temp models

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run the FastAPI application on port 7860 (Hugging Face requirement)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
