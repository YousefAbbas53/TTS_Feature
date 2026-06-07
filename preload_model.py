"""
Pre-download XTTS v2 model weights during Docker build.
This script is run once at build time so workers don't need to download the model at startup.
"""
import os
import sys
import traceback

# Ensure TOS is agreed before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"

print("=" * 60)
print("Pre-loading XTTS v2 model into Docker image...")
print("=" * 60)

try:
    from TTS.api import TTS
    print("[OK] TTS library imported successfully.")
except Exception as e:
    print(f"[FAIL] Failed to import TTS library: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("[OK] XTTS v2 model downloaded and loaded successfully!")
    print("=" * 60)
except Exception as e:
    print(f"[FAIL] Failed to load XTTS v2 model: {e}")
    traceback.print_exc()
    sys.exit(1)
