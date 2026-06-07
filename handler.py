import os
import asyncio
import logging
import urllib.request
import requests
import shutil
import uuid
from pathlib import Path
from typing import Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import system configuration and utilities
from utils.config import (
    DIR_TEMP, DIR_OUTPUT, DEFAULT_LANG, DEFAULT_VOICE_ID,
    resolve_voice_name, MAX_CHARS, MIN_CHARS
)
from utils.text_utils import chunk_text
from utils.audio_utils import (
    tts_to_wav, concat_wavs_by_timeline, wav_duration_seconds, DEFAULT_SPEAKER_PATH
)
from utils.file_utils import load_book_text

# Import runpod SDK
import runpod
import boto3
from botocore.config import Config

# Ensure required directories exist
for directory in [DIR_TEMP, DIR_OUTPUT]:
    directory.mkdir(parents=True, exist_ok=True)


# ─── S3 Upload Utility ────────────────────────────────────────────────────────

def upload_to_s3(file_path: Path, object_name: str) -> Optional[str]:
    """
    Uploads a file to an S3-compatible cloud storage bucket.
    Reads credentials from standard environment variables:
      - S3_BUCKET_NAME (Required)
      - AWS_ACCESS_KEY_ID (Required)
      - AWS_SECRET_ACCESS_KEY (Required)
      - AWS_DEFAULT_REGION (Optional, defaults to us-east-1)
      - S3_ENDPOINT_URL (Optional, for Cloudflare R2, Backblaze B2, etc.)
    """
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not bucket_name or not access_key or not secret_key:
        logging.warning(
            "S3 credentials not fully configured (S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY). Skipping cloud upload."
        )
        return None

    try:
        # Standardize endpoint url if provided
        endpoint = None
        if endpoint_url and endpoint_url.strip():
            endpoint = endpoint_url.strip()

        # Initialize boto3 S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            region_name=region_name,
            config=Config(signature_version="s3v4")
        )
        
        logging.info(f"Uploading {file_path.name} to bucket {bucket_name} as {object_name}...")
        
        # Upload file (WAV audio type)
        s3_client.upload_file(
            Filename=str(file_path),
            Bucket=bucket_name,
            Key=object_name,
            ExtraArgs={"ContentType": "audio/wav"}
        )
        
        # Build the final public download URL
        public_url_prefix = os.environ.get("S3_PUBLIC_URL_PREFIX") or os.environ.get("PUBLIC_URL_PREFIX")
        if public_url_prefix:
            public_url = f"{public_url_prefix.rstrip('/')}/{object_name}"
        elif not endpoint:
            # Standard AWS S3 URL
            public_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        else:
            # Clean base endpoint for custom storage provider (R2/B2/Spaces)
            base_endpoint = endpoint.rstrip("/")
            if bucket_name in base_endpoint:
                public_url = f"{base_endpoint}/{object_name}"
            else:
                public_url = f"{base_endpoint}/{bucket_name}/{object_name}"
                
        logging.info(f"Upload complete. Public URL: {public_url}")
        return public_url

    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}", exc_info=True)
        return None


# ─── File Download Utility ───────────────────────────────────────────────────

def download_file(url: str, dest_path: Path) -> Path:
    """Downloads a file from a URL to a local path."""
    logging.info(f"Downloading file from URL: {url}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)
        logging.info(f"Download complete: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return dest_path
    except Exception as e:
        logging.error(f"Failed to download file from {url}: {e}")
        raise RuntimeError(f"Could not download file from URL: {url}. Details: {e}")


# ─── RunPod Serverless Handler ───────────────────────────────────────────────

async def handler(job):
    """
    RunPod Serverless Handler function.
    Processes the request, generates the TTS audio, uploads to S3, and returns the result.
    """
    job_id = job["id"]
    job_input = job.get("input", {})

    logging.info(f"─── Starting Job {job_id} ───")

    # 1. Parse Input Parameters
    book_url = job_input.get("book_url")
    book_text = job_input.get("book_text")
    voice_url = job_input.get("voice_url")
    voice_id = job_input.get("voice_id", DEFAULT_VOICE_ID)
    lang = job_input.get("lang", DEFAULT_LANG)
    custom_voice_name = job_input.get("custom_voice_name")

    # Validate inputs
    if not book_url and not book_text:
        return {"error": "Either 'book_url' or 'book_text' must be provided in the input.", "status": "failed"}

    try:
        voice_name = resolve_voice_name(voice_id, custom_voice_name)
    except ValueError as e:
        return {"error": str(e), "status": "failed"}

    # Set up temporary directories for this job execution
    req_temp_dir = DIR_TEMP / job_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)
    output_wav = DIR_OUTPUT / f"output_job_{job_id}.wav"

    s3_url = None
    try:
        # 2. Extract Text (either from direct text input or by downloading the book file)
        text = ""
        if book_text and book_text.strip():
            logging.info(f"Processing direct text input ({len(book_text)} characters)...")
            from utils.text_utils import clean_text
            text = clean_text(book_text, lang=lang)
        elif book_url:
            file_ext = Path(book_url).suffix.lower()
            # Handle URLs without extensions
            if not file_ext or file_ext not in [".txt", ".pdf", ".epub", ".md"]:
                file_ext = ".txt"
            
            book_path = req_temp_dir / f"book{file_ext}"
            download_file(book_url, book_path)
            
            logging.info(f"Extracting text from downloaded file {book_path.name}...")
            text = load_book_text(book_path, lang=lang)

        if not text or not text.strip():
            raise ValueError("No valid text could be processed or extracted from the input.")

        # 3. Resolve Speaker Voice File
        speaker_wav_path = DEFAULT_SPEAKER_PATH
        if voice_url:
            voice_ext = Path(voice_url).suffix.lower()
            if not voice_ext or voice_ext not in [".wav", ".mp3", ".ogg", ".flac"]:
                voice_ext = ".wav"
            custom_voice_path = req_temp_dir / f"voice{voice_ext}"
            download_file(voice_url, custom_voice_path)
            speaker_wav_path = custom_voice_path
        else:
            speaker_wav_path = Path(voice_name)

        # 4. Chunk text into sentences/phrases
        logging.info("Chunking text...")
        chunks = chunk_text(text, lang=lang, max_chars=MAX_CHARS, min_chars=MIN_CHARS)
        if not chunks:
            raise ValueError("No valid text chunks found after cleaning and preprocessing.")

        total_chunks = len(chunks)
        logging.info(f"Split text into {total_chunks} chunks. Starting TTS generation...")

        # 5. Synthesize TTS chunks sequentially (XTTS requires single-process GPU access)
        successful_wav_paths = []
        for i, chunk in enumerate(chunks):
            chunk_wav = req_temp_dir / f"chunk_{i:04d}.wav"
            try:
                # Synthesize chunk
                await tts_to_wav(chunk, chunk_wav, str(speaker_wav_path), lang)
                successful_wav_paths.append(chunk_wav)
                logging.info(f"Chunk {i + 1}/{total_chunks} synthesized successfully.")
            except Exception as chunk_err:
                logging.error(f"Failed to synthesize chunk {i + 1}: {chunk_err}")

        if not successful_wav_paths:
            raise RuntimeError("All TTS audio chunks failed to generate.")

        # 6. Concatenate all chunks into a single WAV file
        logging.info(f"Concatenating {len(successful_wav_paths)} chunks...")
        if len(successful_wav_paths) == 1:
            shutil.move(str(successful_wav_paths[0]), str(output_wav))
        else:
            await concat_wavs_by_timeline(successful_wav_paths, output_wav)

        # Calculate generated audio duration
        duration = wav_duration_seconds(output_wav)
        logging.info(f"Audiobook generation complete. Duration: {duration:.2f} seconds.")

        # 7. Upload final wav file to S3
        object_name = f"audiobooks/audiobook_{job_id}.wav"
        s3_url = upload_to_s3(output_wav, object_name)

        if s3_url:
            return {
                "status": "success",
                "download_url": s3_url,
                "duration_seconds": duration,
                "total_chunks": total_chunks,
                "message": "TTS audiobook generated and uploaded to cloud storage successfully."
            }
        else:
            # Fallback when S3 is not configured/failed
            return {
                "status": "warning",
                "local_path": str(output_wav),
                "duration_seconds": duration,
                "total_chunks": total_chunks,
                "message": "TTS audiobook generated successfully, but cloud storage upload was skipped. "
                           "The file remains saved locally on the container disk."
            }

    except Exception as e:
        logging.error(f"Job {job_id} failed with error: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}

    finally:
        # Clean up temporary folders
        shutil.rmtree(req_temp_dir, ignore_errors=True)
        # If successfully uploaded to S3, clean up local output WAV to conserve disk space
        if s3_url and output_wav.exists():
            try:
                output_wav.unlink()
            except Exception:
                pass


# ─── Main Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Starting RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
