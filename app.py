from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
import os
import shutil
import asyncio
from typing import Optional
from pathlib import Path

# GPU Optimization Imports (as per user request)
import torch

from utils.config import (
    DIR_TEMP, DIR_OUTPUT, DEFAULT_LANG, DEFAULT_VOICE_ID, 
    resolve_voice_name, MAX_CHARS, MIN_CHARS
)
from utils.text_utils import chunk_text
from utils.audio_utils import tts_to_wav, concat_wavs_by_timeline

app = FastAPI(
    title="TTS FastAPI Service",
    description="Production-ready Text-to-Speech API using Edge-TTS",
    version="1.0.0"
)

# Startup event to clear temp and output directories and initialize GPU
@app.on_event("startup")
async def startup_event():
    # Cleanup previous runs
    for directory in [DIR_TEMP, DIR_OUTPUT]:
        for item in directory.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    # Optional GPU Warmup/Initialization
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Running on CPU (edge-tts is API based).")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    lang: str = Field(DEFAULT_LANG, description="Language code (e.g., 'en', 'ar')")
    voice_id: str = Field(DEFAULT_VOICE_ID, description="Voice ID from registry")
    custom_voice_name: Optional[str] = Field(None, description="Direct Voice name, e.g. 'en-US-GuyNeural'")

def remove_file(path: Path):
    """Background task to remove a file after returning it."""
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"Failed to delete {path}: {e}")

@app.post("/generate", summary="Generate TTS Audio from Text")
async def generate_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        voice_name = resolve_voice_name(request.voice_id, request.custom_voice_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    req_id = uuid.uuid4().hex[:8]
    req_temp_dir = DIR_TEMP / req_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)
    
    output_wav = DIR_OUTPUT / f"output_{req_id}.wav"

    try:
        # Wrap inference simulation in no_grad for GPU memory optimization requirements
        with torch.no_grad():
            # 1. Chunk Text
            chunks = chunk_text(request.text, lang=request.lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid text found after cleaning.")

            # 2. Generate Audio for each chunk asynchronously
            tasks = []
            chunk_wav_paths = []
            
            for i, chunk in enumerate(chunks):
                chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
                chunk_wav_paths.append(chunk_wav)
                tasks.append(tts_to_wav(chunk, chunk_wav, voice_name))
            
            # Run all edge-tts requests concurrently
            await asyncio.gather(*tasks)

            # 3. Concatenate
            if len(chunk_wav_paths) == 1:
                # If only one chunk, just move it to output
                shutil.move(str(chunk_wav_paths[0]), str(output_wav))
            else:
                await concat_wavs_by_timeline(chunk_wav_paths, output_wav)

        # Ensure CUDA cache is cleared after generation to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Schedule cleanup of the output file after it's returned
        background_tasks.add_task(remove_file, output_wav)
        
        # Cleanup the temp directory for this request
        shutil.rmtree(req_temp_dir, ignore_errors=True)

        return FileResponse(
            path=output_wav, 
            media_type="audio/wav", 
            filename="generated_speech.wav"
        )

    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        if output_wav.exists():
            output_wav.unlink()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        raise HTTPException(status_code=500, detail=str(e))
