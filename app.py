from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
import os
import shutil
import asyncio
from typing import Optional
from pathlib import Path

import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# GPU logic removed as edge-tts operates via API

from utils.config import (
    DIR_TEMP, DIR_OUTPUT, DEFAULT_LANG, DEFAULT_VOICE_ID, 
    resolve_voice_name, MAX_CHARS, MIN_CHARS
)
from utils.text_utils import chunk_text
from utils.audio_utils import tts_to_wav, concat_wavs_by_timeline, DEFAULT_SPEAKER_PATH
from utils.file_utils import load_book_text

# Global lock to ensure only one TTS chunk is processed by the GPU at a time
gpu_tts_lock = asyncio.Lock()

app = FastAPI(
    title="TTS FastAPI Service",
    description="Production-ready Text-to-Speech API using Edge-TTS",
    version="1.0.0"
)

# Startup event to clear temp and output directories and ensure they exist
@app.on_event("startup")
async def startup_event():
    # Ensure directories always exist
    for directory in [DIR_TEMP, DIR_OUTPUT]:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Cleanup previous runs safely
    for directory in [DIR_TEMP, DIR_OUTPUT]:
        for item in directory.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logging.error(f"Cleanup error on {item}: {e}")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    lang: str = Field(DEFAULT_LANG, description="Language code (e.g., 'en', 'ar')")
    voice_id: str = Field(DEFAULT_VOICE_ID, description="Voice ID from registry")
    custom_voice_name: Optional[str] = Field(None, description="Direct Voice name, e.g. 'en-US-GuyNeural'")

async def remove_file(path: Path):
    """Background task to safely remove a file after returning it with a 30-second delay."""
    await asyncio.sleep(30)  # Safe async delay to guarantee response finishes streaming before deletion
    try:
        if path.exists():
            path.unlink()
            logging.info(f"Safely deleted output file: {path}")
    except Exception as e:
        logging.error(f"Failed to delete {path}: {e}")

@app.get("/", summary="Health Check")
async def root():
    return {"status": "success", "message": "TTS API is running perfectly on Hugging Face Spaces!", "endpoints": ["POST /generate"]}

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
        # 1. Chunk Text
        chunks = chunk_text(request.text, lang=request.lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text found after cleaning.")

        # 2. Generate Audio for each chunk sequentially using a global lock to prevent GPU OOM
        async def safe_tts(chunk_text, chunk_wav_path, voice, lang):
            async with gpu_tts_lock:
                return await tts_to_wav(chunk_text, chunk_wav_path, voice, lang)

        tasks = []
        chunk_wav_paths = []
        
        for i, chunk in enumerate(chunks):
            chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
            chunk_wav_paths.append(chunk_wav)
            tasks.append(safe_tts(chunk, chunk_wav, str(DEFAULT_SPEAKER_PATH), request.lang))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_wav_paths = []
        for path, result in zip(chunk_wav_paths, results):
            if isinstance(result, Exception):
                logging.error(f"Failed to generate TTS for chunk {path}: {result}")
            else:
                successful_wav_paths.append(path)

        if not successful_wav_paths:
            logging.error(f"All TTS chunks failed for request {req_id}")
            raise HTTPException(
                status_code=500, 
                detail="All TTS generation chunks failed due to API errors. Please verify the text content or try again later."
            )

        # 3. Concatenate
        if len(successful_wav_paths) == 1:
            # If only one chunk, just move it to output
            shutil.move(str(successful_wav_paths[0]), str(output_wav))
        else:
            await concat_wavs_by_timeline(successful_wav_paths, output_wav)

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
            
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")

@app.post("/generate-from-file", summary="Generate TTS Audio from an uploaded File (TXT, PDF, EPUB) and optional Voice")
async def generate_audio_from_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The book file to process (.txt, .pdf, .epub)"),
    voice_file: UploadFile = File(None, description="Optional custom voice reference (.wav, .mp3) for zero-shot cloning"),
    lang: str = Form(DEFAULT_LANG, description="Language code (e.g., 'en', 'ar')"),
    voice_id: str = Form(DEFAULT_VOICE_ID, description="Voice ID from registry"),
    custom_voice_name: Optional[str] = Form(None, description="Direct Voice name, e.g. 'en-US-GuyNeural'")
):
    """
    Accepts a document file, extracts the text, and processes it into speech.
    If voice_file is provided, clones that specific voice for the entire book.
    WARNING: For large books, this process may take a long time and could timeout
    if the HTTP client doesn't wait long enough.
    """
    
    try:
        voice_name = resolve_voice_name(voice_id, custom_voice_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    req_id = uuid.uuid4().hex[:8]
    req_temp_dir = DIR_TEMP / req_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded book file temporarily
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".txt", ".pdf", ".epub", ".md"]:
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file_ext}'. Allowed: .txt, .pdf, .epub")
        
    temp_file_path = req_temp_dir / f"uploaded_book{file_ext}"
    
    speaker_wav_path = DEFAULT_SPEAKER_PATH
    if voice_file:
        voice_ext = Path(voice_file.filename).suffix.lower()
        custom_voice_path = req_temp_dir / f"custom_voice{voice_ext}"
        try:
            with custom_voice_path.open("wb") as buffer:
                shutil.copyfileobj(voice_file.file, buffer)
            speaker_wav_path = custom_voice_path
        except Exception as e:
            logging.error(f"Failed to save custom voice file: {e}")
            
    try:
        # Write uploaded file to disk
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract and clean text using the book parser
        extracted_text = load_book_text(temp_file_path, lang=lang)
        
        if not extracted_text:
             raise ValueError("No text could be extracted from the file.")
             
        # Cleanup the temporary uploaded file now that we have the text
        temp_file_path.unlink()
        
    except Exception as e:
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
        
    output_wav = DIR_OUTPUT / f"output_file_{req_id}.wav"

    try:
        # 1. Chunk Text
        chunks = chunk_text(extracted_text, lang=lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text found after cleaning.")

        # 2. Generate Audio for each chunk sequentially using a global lock to prevent GPU OOM
        async def safe_tts(chunk_text, chunk_wav_path, voice, language):
            async with gpu_tts_lock:
                return await tts_to_wav(chunk_text, chunk_wav_path, voice, language)

        tasks = []
        chunk_wav_paths = []
        
        for i, chunk in enumerate(chunks):
            chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
            chunk_wav_paths.append(chunk_wav)
            tasks.append(safe_tts(chunk, chunk_wav, str(speaker_wav_path), lang))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_wav_paths = []
        for path, result in zip(chunk_wav_paths, results):
            if isinstance(result, Exception):
                logging.error(f"Failed to generate TTS for chunk {path}: {result}")
            else:
                successful_wav_paths.append(path)

        if not successful_wav_paths:
            logging.error(f"All TTS chunks failed for request {req_id}")
            raise HTTPException(
                status_code=500, 
                detail="All TTS generation chunks failed due to API errors. Please verify the file content or try again later."
            )

        # 3. Concatenate
        if len(successful_wav_paths) == 1:
            shutil.move(str(successful_wav_paths[0]), str(output_wav))
        else:
            await concat_wavs_by_timeline(successful_wav_paths, output_wav)

        # Schedule cleanup of the output file after it's returned
        background_tasks.add_task(remove_file, output_wav)
        
        # Cleanup the temp directory for this request
        shutil.rmtree(req_temp_dir, ignore_errors=True)

        return FileResponse(
            path=output_wav, 
            media_type="audio/wav", 
            filename=f"generated_book_speech.wav"
        )

    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        if output_wav.exists():
            output_wav.unlink()
            
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")
