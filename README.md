# Text-to-Speech (TTS) FastAPI Service

A production-ready Text-to-Speech API built with FastAPI and Coqui XTTS v2. It handles long texts via automatic chunking, supports voice cloning, and is deployed on RunPod GPU instances.

## Project Structure

```
project/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── .dockerignore          # Docker ignore file
├── models/                # Local model/speaker wav files
├── outputs/               # Temporary output directory for generated audio
├── temp/                  # Temporary chunk processing directory
├── utils/                 # Utility modules
│   ├── config.py          # Configuration and settings
│   ├── text_utils.py      # Text cleaning and chunking logic
│   ├── audio_utils.py     # TTS generation and audio concatenation logic
│   └── file_utils.py      # File loading utilities (TXT, PDF, EPUB)
└── README.md              # Documentation
```

## Features
- **FastAPI / Uvicorn**: High-performance asynchronous API.
- **Coqui XTTS v2**: Multilingual zero-shot voice cloning TTS model.
- **Robust Chunking**: Automatically chunks long text inputs without breaking sentences.
- **GPU Acceleration**: Full CUDA 11.8 support, model runs on GPU via RunPod.
- **Voice Cloning**: Upload any `.wav` voice file to clone it for the output.
- **Multi-format Input**: Accepts `.txt`, `.pdf`, `.epub`, and `.md` book files.
- **Temp File Cleanup**: Automatically cleans up temporary chunks and output files after serving.

## How to Run Locally

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system.
- NVIDIA GPU with CUDA 11.8 (recommended) or CPU fallback.

### Steps
1. Install dependencies:
   ```bash
   pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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

## How to Deploy on RunPod

1. **Push your Docker image** to Docker Hub or GitHub Container Registry:
   ```bash
   docker build -t <your-dockerhub-username>/tts-fastapi:latest .
   docker push <your-dockerhub-username>/tts-fastapi:latest
   ```

2. **Create a Pod on RunPod**:
   - Go to [RunPod.io](https://runpod.io) → **Pods** → **+ New Pod**.
   - Select a GPU template (e.g., RTX 3090 or A40 for XTTS v2).
   - Under **Container Image**, enter your Docker Hub image: `<your-dockerhub-username>/tts-fastapi:latest`.
   - Set **Container Port** to `8000`.
   - Under **Environment Variables**, add any needed env vars.

3. **Access your API**:
   - Once the pod starts, RunPod provides a proxy URL like:  
     `https://<pod-id>-8000.proxy.runpod.net`
   - Visit `https://<pod-id>-8000.proxy.runpod.net/docs` for the interactive Swagger UI.

> **Note**: On first start, Coqui XTTS v2 will download the model (~2GB) from HuggingFace.  
> Subsequent restarts will use the cached model.

## API Usage

### `GET /`
Health check endpoint.

**Response**:
```json
{"status": "success", "message": "TTS API is running perfectly on RunPod!"}
```

---

### `POST /generate`
Generate speech from raw text.

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "text": "Hello! This is a test of the Text to Speech API.",
  "lang": "en",
  "voice_id": "preset_1"
}
```

**Parameters**:
- `text` (string, required): The text to synthesize.
- `lang` (string, optional): Language code (e.g., `"en"`, `"ar"`). Defaults to `"en"`.
- `voice_id` (string, optional): A preset voice (`preset_1` to `preset_5`). Defaults to `"preset_1"`.
- `custom_voice_name` (string, optional): Ignored for XTTS (use `voice_file` in `/generate-from-file` instead).

**Response**: Returns the generated audio file (`audio/wav`).

**Example (curl)**:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text":"Welcome to the text to speech service.","lang":"en"}' \
     --output generated_speech.wav
```

---

### `POST /generate-from-file`
Generate speech from an uploaded document (TXT, PDF, EPUB) with optional voice cloning.

**Content-Type**: `multipart/form-data`

**Form Fields**:
- `file` (required): The document file (`.txt`, `.pdf`, `.epub`, `.md`).
- `voice_file` (optional): A `.wav` or `.mp3` file to clone the voice from.
- `lang` (optional): Language code. Defaults to `"en"`.
- `voice_id` (optional): Preset voice ID. Defaults to `"preset_1"`.

**Example (curl)**:
```bash
curl -X POST "http://localhost:8000/generate-from-file" \
     -F "file=@mybook.pdf" \
     -F "lang=en" \
     --output audiobook.wav
```
