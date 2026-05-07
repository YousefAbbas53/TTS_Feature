# Text-to-Speech (TTS) FastAPI Service

This is a production-ready Text-to-Speech API built with FastAPI and `edge-tts`. It is optimized for asynchronous generation, handles long texts by chunking them, and is structured for deployment on GPU-enabled platforms like Vast.ai.

## Project Structure

```
project/
├── app.py                 # FastAPI application
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker configuration
├── .dockerignore          # Docker ignore file
├── models/                # Directory for local models (if added later)
├── outputs/               # Temporary output directory for generated audio
├── temp/                  # Temporary chunk processing directory
├── utils/                 # Utility modules
│   ├── config.py          # Configuration and settings
│   ├── text_utils.py      # Text cleaning and chunking logic
│   └── audio_utils.py     # TTS generation and audio concatenation logic
└── README.md              # Documentation
```

## Features
- **FastAPI / Uvicorn**: High-performance asynchronous API.
- **Robust Chunking**: Automatically chunks long text inputs without breaking sentences.
- **Edge-TTS Integration**: Uses Microsoft Edge's neural TTS service.
- **GPU Readiness**: Includes `torch` and memory cleanup (`torch.cuda.empty_cache()`) for compatibility with Vast.ai templates, allowing easy drop-in of local GPU models later.
- **Temp File Cleanup**: Automatically cleans up temporary chunks and output files after they are served.

## How to Run Locally

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system.

### Steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Access the API documentation at `http://localhost:8000/docs`.

## How to Build and Run the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t tts-fastapi .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 --gpus all tts-fastapi
   ```
   *(Note: `--gpus all` is optional if you are strictly using `edge-tts` but recommended if you are deploying to a GPU-enabled instance and plan to use local PyTorch models).*

## How to Deploy on Vast.ai

1. Spin up an instance on Vast.ai.
2. Under "Template Configuration", select a base image such as `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` or simply check the box for "Use Custom Image" and enter `python:3.10` or a custom Docker Hub image if you have pushed yours.
3. If using an unconfigured instance, SSH in and run:
   ```bash
   git clone <your-repo-url>
   cd tts-repo
   docker build -t tts-fastapi .
   docker run -d -p 8000:8000 --gpus all tts-fastapi
   ```
4. Ensure port `8000` is mapped and exposed in your Vast.ai instance settings so you can reach the API externally.

## API Usage

### `POST /generate`

**Endpoint**: `/generate`
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "text": "Hello! This is a test of the Text to Speech API. It can handle very long text by splitting it appropriately.",
  "lang": "en",
  "voice_id": "preset_1"
}
```

**Parameters**:
- `text` (string, required): The text to synthesize.
- `lang` (string, optional): Language code (e.g., `"en"`, `"ar"`). Defaults to `"en"`.
- `voice_id` (string, optional): A preset voice from the registry (`preset_1` to `preset_5`). Defaults to `"preset_1"`.
- `custom_voice_name` (string, optional): A specific Edge-TTS voice name (e.g., `"en-US-AriaNeural"`).

**Response**:
Returns the generated audio file (`audio/wav`).

### Example using `curl`

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text":"Welcome to the text to speech service.","voice_id":"preset_2"}' \
     --output generated_speech.wav
```
