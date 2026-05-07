import asyncio
import edge_tts
import subprocess
import wave
import contextlib
from pathlib import Path
from utils.config import EDGE_RATE, EDGE_PITCH, SAMPLE_RATE

def wav_duration_seconds(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())

async def _edge_save_mp3(text: str, voice_name: str, out_mp3: Path, rate: str, pitch: str):
    comm = edge_tts.Communicate(text=text, voice=voice_name, rate=rate, pitch=pitch)
    await comm.save(str(out_mp3))

async def tts_to_wav(text: str, out_wav: Path, voice_name: str) -> None:
    """
    Generates consistent PCM WAV (mono, SAMPLE_RATE) for easy concatenation.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = out_wav.with_suffix(".tmp.mp3")
    
    await _edge_save_mp3(text, voice_name, tmp_mp3, EDGE_RATE, EDGE_PITCH)

    # Convert mp3 to wav using ffmpeg
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(tmp_mp3),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(out_wav),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with error:\\n{stderr.decode()}")

    try:
        tmp_mp3.unlink()
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
            f.write(f"file '{str(p.resolve())}'\\n")

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
