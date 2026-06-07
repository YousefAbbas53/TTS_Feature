import asyncio
import subprocess
import wave
import contextlib
import urllib.request
import logging
from pathlib import Path
from utils.config import SAMPLE_RATE, DIR_MODELS

# --- MONKEY PATCH FOR TRANSFORMERS ---
# Coqui-TTS tries to import functions that were removed in recent transformers versions.
import os
import traceback as _traceback
import transformers.utils.import_utils
import transformers.pytorch_utils
import torch
from packaging.version import parse

os.environ.setdefault("COQUI_TOS_AGREED", "1")

if not hasattr(transformers.utils.import_utils, "is_torch_greater_or_equal"):
    def is_torch_greater_or_equal(version: str) -> bool:
        return parse(torch.__version__) >= parse(version)
    transformers.utils.import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    def isin_mps_friendly(elements, test_elements, assume_unique=False, invert=False):
        return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly
# ---------------------------------------

logging.info("Importing TTS library...")
try:
    from TTS.api import TTS
    logging.info("TTS library imported successfully.")
except Exception as _import_err:
    logging.error(f"CRITICAL: Failed to import TTS library: {_import_err}")
    logging.error(_traceback.format_exc())
    raise SystemExit(f"Cannot import TTS: {_import_err}")

# Download default speaker wav if it doesn't exist
DEFAULT_SPEAKER_PATH = DIR_MODELS / "default_speaker.wav"
if not DEFAULT_SPEAKER_PATH.exists():
    logging.info("Downloading default speaker wav for XTTS presets...")
    url = "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav"
    try:
        urllib.request.urlretrieve(url, str(DEFAULT_SPEAKER_PATH))
        logging.info("Default speaker WAV downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download default speaker: {e}")

# Load XTTS v2 into GPU globally (model weights are pre-baked into the Docker image)
logging.info("Loading Coqui XTTS v2 Model into GPU...")
try:
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    logging.info("XTTS v2 Model loaded successfully!")
except Exception as e:
    logging.error(f"CRITICAL: Failed to load XTTS v2 model: {e}")
    logging.error(_traceback.format_exc())
    tts_model = None

def wav_duration_seconds(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())

def _generate_tts_sync(text: str, out_wav: Path, speaker_wav: str, lang: str):
    """Synchronous generation using XTTS"""
    if not tts_model:
        raise RuntimeError("TTS model is not loaded.")
        
    tts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=lang,
        file_path=str(out_wav)
    )

async def tts_to_wav(text: str, out_wav: Path, speaker_wav: str, lang: str = "en") -> None:
    """
    Generates consistent PCM WAV asynchronously using thread pool to prevent blocking.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the heavy synchronous GPU task in a thread
    await asyncio.to_thread(_generate_tts_sync, text, out_wav, speaker_wav, lang)

    # Note: XTTS outputs 24kHz or 22050Hz wav files by default, 
    # but we will force ffmpeg conversion to ensure strict consistency for concatenation.
    tmp_wav = out_wav.with_suffix(".tmp.wav")
    out_wav.rename(tmp_wav)

    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(tmp_wav),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(out_wav),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with error:\n{stderr.decode()}")

    try:
        tmp_wav.unlink()
    except Exception:
        pass

async def concat_wavs_by_timeline(wav_paths: list[Path], out_wav: Path):
    """
    Concatenates multiple WAV files into a single WAV file.
    """
    if not wav_paths:
        raise ValueError("No WAV files to concatenate.")

    concat_file = out_wav.with_suffix(".concat.txt")
    with concat_file.open("w", encoding="utf-8") as f:
        for p in wav_paths:
            # ffmpeg concat demuxer requires absolute paths or relative to the list file
            f.write(f"file '{str(p.resolve())}'\n")

    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg concatenation failed:\\n{stderr.decode()}")

    try:
        concat_file.unlink()
    except Exception:
        pass
