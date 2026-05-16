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

# ---------- TTS (XTTS) Configuration ----------
DEFAULT_VOICE_ID = "preset_1"
CUSTOM_VOICE_NAME = None

# For XTTS, presets represent path to local speaker wav files for voice cloning
# If the file doesn't exist, we will use the default_speaker.wav
VOICE_REGISTRY = {
    "preset_1": "default_speaker.wav",
    "preset_2": "default_speaker.wav",
    "preset_3": "default_speaker.wav",
    "preset_4": "default_speaker.wav",
    "preset_5": "default_speaker.wav",
}

# ---------- Text + Chunking ----------
DEFAULT_LANG = "en"
MAX_CHARS = 220
MIN_CHARS = 20

# ---------- Audio Formats ----------
SAMPLE_RATE = 22050  # Output sample rate for consistent mono PCM wav

def resolve_voice_name(voice_id: str, custom_name: str = None) -> str:
    """Returns the filename of the speaker wav from the models directory."""
    if custom_name and str(custom_name).strip():
        speaker_filename = str(custom_name).strip()
    else:
        if voice_id not in VOICE_REGISTRY:
            raise ValueError(f"Unknown VOICE_ID={voice_id}. Use {list(VOICE_REGISTRY.keys())} or set CUSTOM_VOICE_NAME.")
        speaker_filename = VOICE_REGISTRY[voice_id]
        
    speaker_path = DIR_MODELS / speaker_filename
    if not speaker_path.exists():
        # Fallback to default speaker if preset file is missing
        speaker_path = DIR_MODELS / "default_speaker.wav"
        
    return str(speaker_path)
