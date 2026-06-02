from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
import os
import shutil
import asyncio
import time
import logging
from typing import Optional
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from utils.config import (
    DIR_TEMP, DIR_OUTPUT, DEFAULT_LANG, DEFAULT_VOICE_ID,
    resolve_voice_name, MAX_CHARS, MIN_CHARS
)
from utils.text_utils import chunk_text
from utils.audio_utils import tts_to_wav, concat_wavs_by_timeline, DEFAULT_SPEAKER_PATH
from utils.file_utils import load_book_text


# ─── Global State ─────────────────────────────────────────────────────────────

# Global GPU lock: ensures only one TTS inference runs at a time
gpu_tts_lock = asyncio.Lock()

# In-memory job store (MVP: no Redis/Celery needed)
# Structure per job:
#   status      : "queued" | "processing" | "completed" | "failed"
#   progress    : 0-100 (percentage)
#   done_chunks : number of chunks processed
#   total_chunks: total chunks in the document
#   message     : human-readable status string
#   output_path : str path to final WAV (when completed)
#   temp_dir    : str path to temp directory for this job
#   error       : error message (when failed)
#   created_at  : unix timestamp (for TTL cleanup)
jobs: dict[str, dict] = {}

JOB_TTL_SECONDS = 2 * 60 * 60  # 2 hours before auto-cleanup


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="TTS FastAPI Service",
    description="Async Job-Based Text-to-Speech API using Coqui XTTS v2. "
                "Submit a document, get a job_id, poll for progress, download when done.",
    version="2.0.0"
)


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Ensure directories exist and start background cleanup scheduler."""
    for directory in [DIR_TEMP, DIR_OUTPUT]:
        directory.mkdir(parents=True, exist_ok=True)
    # Launch the cleanup scheduler as a long-running background coroutine
    asyncio.create_task(_cleanup_scheduler())
    logging.info("TTS API started. Background cleanup scheduler is running.")


# ─── Background Cleanup ───────────────────────────────────────────────────────

async def _cleanup_scheduler():
    """Runs every 30 minutes and removes expired jobs + their files."""
    while True:
        await asyncio.sleep(30 * 60)
        await _cleanup_expired_jobs()


async def _cleanup_expired_jobs():
    now = time.time()
    expired_ids = [
        jid for jid, job in list(jobs.items())
        if now - job["created_at"] > JOB_TTL_SECONDS
    ]
    for jid in expired_ids:
        job = jobs.pop(jid, None)
        if not job:
            continue
        # Remove output file
        if job.get("output_path"):
            out = Path(job["output_path"])
            try:
                if out.exists():
                    out.unlink()
            except Exception as e:
                logging.warning(f"[Cleanup] Could not delete output for job {jid}: {e}")
        # Remove temp dir
        if job.get("temp_dir"):
            shutil.rmtree(job["temp_dir"], ignore_errors=True)
        logging.info(f"[Cleanup] Expired job {jid} removed (TTL exceeded).")


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    lang: str = Field(DEFAULT_LANG, description="Language code (e.g., 'en', 'ar')")
    voice_id: str = Field(DEFAULT_VOICE_ID, description="Preset voice ID from registry (preset_1 to preset_5)")
    custom_voice_name: Optional[str] = Field(
        None,
        description="Custom speaker wav filename inside the models/ directory (e.g. 'my_voice.wav')"
    )


class JobResponse(BaseModel):
    job_id: str
    message: str
    status_url: str
    download_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    done_chunks: int
    total_chunks: int
    message: str
    error: Optional[str] = None


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_job(temp_dir: Optional[Path] = None) -> dict:
    """Creates a fresh job dict."""
    return {
        "status": "queued",
        "progress": 0,
        "done_chunks": 0,
        "total_chunks": 0,
        "message": "Job queued, waiting to start...",
        "output_path": None,
        "temp_dir": str(temp_dir) if temp_dir else None,
        "error": None,
        "created_at": time.time(),
    }


async def _remove_file_delayed(path: Path, delay: int = 30):
    """Deletes a file after a delay (used after serving synchronous responses)."""
    await asyncio.sleep(delay)
    try:
        if path.exists():
            path.unlink()
            logging.info(f"Deleted temporary output file: {path}")
    except Exception as e:
        logging.error(f"Failed to delete {path}: {e}")


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health Check")
async def root():
    active = sum(1 for j in jobs.values() if j["status"] in ("queued", "processing"))
    return {
        "status": "success",
        "message": "TTS API is running perfectly on RunPod!",
        "version": "2.0.0",
        "active_jobs": active,
        "endpoints": {
            "short_text_tts": "POST /generate",
            "submit_document_job": "POST /generate-from-file",
            "check_job_status": "GET /job-status/{job_id}",
            "download_result": "GET /download/{job_id}",
            "list_jobs": "GET /jobs",
        }
    }


@app.get("/jobs", summary="List all jobs (debug)")
async def list_jobs():
    """Returns a summary of all tracked jobs."""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": jid,
                "status": j["status"],
                "progress": j["progress"],
                "message": j["message"],
            }
            for jid, j in jobs.items()
        ]
    }


# ─── /generate  (Synchronous – Short Text) ────────────────────────────────────

@app.post("/generate", summary="Generate TTS Audio from Text (synchronous, short texts only)")
async def generate_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Synchronous endpoint for short text-to-speech.
    Suitable for texts that can be processed in under 90 seconds.
    For full books/documents, use POST /generate-from-file instead.
    """
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
        chunks = chunk_text(request.text, lang=request.lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text found after cleaning.")

        successful_wav_paths = []
        for i, chunk in enumerate(chunks):
            chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
            try:
                async with gpu_tts_lock:
                    await tts_to_wav(chunk, chunk_wav, voice_name, request.lang)
                successful_wav_paths.append(chunk_wav)
                logging.info(f"[{req_id}] Chunk {i + 1}/{len(chunks)} done.")
            except Exception as e:
                logging.error(f"[{req_id}] Chunk {i} failed: {e}")

        if not successful_wav_paths:
            raise HTTPException(
                status_code=500,
                detail="All TTS chunks failed. Please verify the text content or try again."
            )

        if len(successful_wav_paths) == 1:
            shutil.move(str(successful_wav_paths[0]), str(output_wav))
        else:
            await concat_wavs_by_timeline(successful_wav_paths, output_wav)

        background_tasks.add_task(_remove_file_delayed, output_wav)
        shutil.rmtree(req_temp_dir, ignore_errors=True)

        return FileResponse(
            path=output_wav,
            media_type="audio/wav",
            filename="generated_speech.wav"
        )

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        if output_wav.exists():
            output_wav.unlink()
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")


# ─── /generate-from-file  (Async Job – Books/Documents) ──────────────────────

@app.post(
    "/generate-from-file",
    summary="Submit TTS Job for Book/Document (async, returns immediately)",
    response_model=JobResponse,
)
async def generate_audio_from_file(
    file: UploadFile = File(..., description="Book file (.txt, .pdf, .epub, .md)"),
    voice_file: UploadFile = File(None, description="Optional voice reference (.wav, .mp3) for zero-shot cloning"),
    lang: str = Form(DEFAULT_LANG, description="Language code (e.g., 'en', 'ar')"),
    voice_id: str = Form(DEFAULT_VOICE_ID, description="Preset voice ID (preset_1 to preset_5)"),
    custom_voice_name: Optional[str] = Form(
        None,
        description="Custom speaker wav filename inside models/ directory (overrides voice_id if set)"
    ),
):
    """
    Submits a TTS job for a document. Returns a job_id **immediately** (< 1 second).

    Workflow:
    1. POST /generate-from-file  →  { "job_id": "abc123" }
    2. GET  /job-status/abc123   →  { "status": "processing", "progress": 45 }
    3. GET  /download/abc123     →  audiobook.wav  (when status = "completed")
    """

    # Validate voice
    try:
        voice_name = resolve_voice_name(voice_id, custom_voice_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate book file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".txt", ".pdf", ".epub", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: .txt, .pdf, .epub, .md"
        )

    # Create unique job ID and temp directory
    job_id = uuid.uuid4().hex[:12]
    req_temp_dir = DIR_TEMP / job_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)

    # Save book file to disk BEFORE returning (so background task has it)
    book_path = req_temp_dir / f"book{file_ext}"
    try:
        with book_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Save voice reference file (if provided)
    speaker_wav_path = DEFAULT_SPEAKER_PATH
    if voice_file and voice_file.filename:
        voice_ext = Path(voice_file.filename).suffix.lower()
        if voice_ext not in [".wav", ".mp3", ".ogg", ".flac"]:
            shutil.rmtree(req_temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice file type '{voice_ext}'. Allowed: .wav, .mp3, .ogg, .flac"
            )
        custom_voice_path = req_temp_dir / f"voice{voice_ext}"
        try:
            with custom_voice_path.open("wb") as buf:
                shutil.copyfileobj(voice_file.file, buf)
            speaker_wav_path = custom_voice_path
            logging.info(f"[{job_id}] Using uploaded voice file: {custom_voice_path.name}")
        except Exception as e:
            logging.error(f"[{job_id}] Failed to save voice file, using default: {e}")

    # Register job in store
    jobs[job_id] = _make_job(temp_dir=req_temp_dir)

    # Launch background processing (non-blocking)
    asyncio.create_task(
        _process_tts_job(
            job_id=job_id,
            book_path=book_path,
            speaker_wav_path=speaker_wav_path,
            lang=lang,
            req_temp_dir=req_temp_dir,
        )
    )

    logging.info(f"[{job_id}] Job created and queued for file: {file.filename}")
    return JobResponse(
        job_id=job_id,
        message="Job submitted successfully. Poll the status URL for progress.",
        status_url=f"/job-status/{job_id}",
        download_url=f"/download/{job_id}",
    )


# ─── Background TTS Processing ────────────────────────────────────────────────

async def _process_tts_job(
    job_id: str,
    book_path: Path,
    speaker_wav_path: Path,
    lang: str,
    req_temp_dir: Path,
):
    """
    Background coroutine that processes a full TTS job.
    Updates jobs[job_id] incrementally so the client can poll progress.
    """
    job = jobs[job_id]
    output_wav = DIR_OUTPUT / f"output_job_{job_id}.wav"

    try:
        # ── Step 1: Extract text ────────────────────────────────────────────
        job["status"] = "processing"
        job["message"] = "Extracting text from document..."
        logging.info(f"[{job_id}] Extracting text from {book_path.name}...")

        extracted_text = load_book_text(book_path, lang=lang)
        if not extracted_text:
            raise ValueError("No text could be extracted from the document.")

        # Remove uploaded book file now that we have the text
        try:
            book_path.unlink()
        except Exception:
            pass

        # ── Step 2: Chunk text ──────────────────────────────────────────────
        job["message"] = "Chunking and cleaning text..."
        chunks = chunk_text(extracted_text, lang=lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
        if not chunks:
            raise ValueError("No valid text chunks found after cleaning.")

        total = len(chunks)
        job["total_chunks"] = total
        job["message"] = f"Starting TTS generation for {total} chunks..."
        logging.info(f"[{job_id}] Text split into {total} chunks. Starting TTS inference...")

        # ── Step 3: Generate audio per chunk ───────────────────────────────
        successful_wav_paths = []

        for i, chunk in enumerate(chunks):
            chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
            try:
                async with gpu_tts_lock:
                    await tts_to_wav(chunk, chunk_wav, str(speaker_wav_path), lang)
                successful_wav_paths.append(chunk_wav)
            except Exception as e:
                logging.error(f"[{job_id}] Chunk {i} failed: {e}")

            # Update progress after each chunk
            done = i + 1
            job["done_chunks"] = done
            job["progress"] = int(done / total * 100)
            job["message"] = f"Processing chunk {done}/{total}"

        if not successful_wav_paths:
            raise RuntimeError(
                "All TTS chunks failed to generate. "
                "Check that the document contains valid text and try again."
            )

        # ── Step 4: Concatenate all chunks ─────────────────────────────────
        job["message"] = "Concatenating audio chunks into final file..."
        logging.info(f"[{job_id}] Concatenating {len(successful_wav_paths)} chunks...")

        if len(successful_wav_paths) == 1:
            shutil.move(str(successful_wav_paths[0]), str(output_wav))
        else:
            await concat_wavs_by_timeline(successful_wav_paths, output_wav)

        # ── Done! ──────────────────────────────────────────────────────────
        job["status"] = "completed"
        job["progress"] = 100
        job["done_chunks"] = total
        job["output_path"] = str(output_wav)
        job["message"] = "Processing complete! Download your file from /download/{job_id}"
        logging.info(f"[{job_id}] ✅ Job completed successfully. Output: {output_wav}")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Processing failed: {str(e)}"
        if output_wav.exists():
            try:
                output_wav.unlink()
            except Exception:
                pass
        logging.error(f"[{job_id}] ❌ Job failed: {e}")

    finally:
        # Always clean up temp dir
        shutil.rmtree(req_temp_dir, ignore_errors=True)


# ─── /job-status ──────────────────────────────────────────────────────────────

@app.get(
    "/job-status/{job_id}",
    summary="Get TTS Job Status",
    response_model=JobStatusResponse,
)
async def get_job_status(job_id: str):
    """
    Poll this endpoint to track job progress.

    Possible statuses:
    - **queued**: Job registered, waiting for GPU to become available
    - **processing**: TTS inference running (check progress field)
    - **completed**: Done! Download from /download/{job_id}
    - **failed**: Something went wrong (check error field)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        done_chunks=job["done_chunks"],
        total_chunks=job["total_chunks"],
        message=job["message"],
        error=job.get("error"),
    )


# ─── /download ────────────────────────────────────────────────────────────────

@app.get("/download/{job_id}", summary="Download Completed TTS Audio")
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    """
    Returns the final WAV file once the job is completed.
    The file and job are automatically cleaned up 60 seconds after download.
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have already been downloaded and cleaned up."
        )

    job = jobs[job_id]

    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=202,
            detail=f"Job is still processing. Progress: {job['progress']}%. "
                   f"Current status: {job['message']}"
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )

    if not job.get("output_path"):
        raise HTTPException(status_code=500, detail="Output path missing for completed job.")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(
            status_code=410,
            detail="Output file is no longer available. It may have been cleaned up by the TTL scheduler."
        )

    # Schedule cleanup: delete file + remove job entry after 60 seconds
    async def _cleanup_after_download():
        await asyncio.sleep(60)
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass
        jobs.pop(job_id, None)
        logging.info(f"[{job_id}] Cleaned up after successful download.")

    background_tasks.add_task(_cleanup_after_download)

    logging.info(f"[{job_id}] Serving download: {output_path.name}")
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"audiobook_{job_id}.wav"
    )
