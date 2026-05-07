import os
from pathlib import Path

# ---------- Base Directories ----------
BASE_DIR = Path(__file__).parent.parent
DIR_TEMP = BASE_DIR / "temp"
DIR_OUTPUT = BASE_DIR / "outputs"
DIR_MODELS = BASE_DIR / "models"

# Ensure directories exist
for d in [DIR_TEMP, DIR_OUTPUT, DIR_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- TTS (Edge-TTS) Configuration ----------
DEFAULT_VOICE_ID = "preset_1"
CUSTOM_VOICE_NAME = None  # e.g. "en-US-GuyNeural"
EDGE_RATE = "+0%"
EDGE_PITCH = "+0Hz"

VOICE_REGISTRY = {
    "preset_1": "en-US-GuyNeural",
    "preset_2": "en-US-JennyNeural",
    "preset_3": "en-GB-RyanNeural",
    "preset_4": "en-GB-SoniaNeural",
    "preset_5": "ar-EG-SalmaNeural",
}

# ---------- Text + Chunking ----------
DEFAULT_LANG = "en"
MAX_CHARS = 220
MIN_CHARS = 20

# ---------- Audio Formats ----------
SAMPLE_RATE = 22050  # Output sample rate for consistent mono PCM wav

def resolve_voice_name(voice_id: str, custom_name: str = None) -> str:
    if custom_name and str(custom_name).strip():
        return str(custom_name).strip()
    if voice_id not in VOICE_REGISTRY:
        raise ValueError(f"Unknown VOICE_ID={voice_id}. Use {list(VOICE_REGISTRY.keys())} or set CUSTOM_VOICE_NAME.")
    return VOICE_REGISTRY[voice_id]
