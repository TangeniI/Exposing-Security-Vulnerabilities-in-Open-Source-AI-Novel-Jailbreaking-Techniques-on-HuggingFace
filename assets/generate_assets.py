"""Regenerate the small media assets used by the experiments.

These files are deliberately *not* committed (see ``.gitignore``); run this
script once to (re)create them locally. Everything is synthetic — no real-world
audio, images, or personal data are used.

    python assets/generate_assets.py

Creates, in this directory:
    normal.wav   1s of quiet speech-band tone   (stands in for benign speech)
    trigger.wav  1s low-frequency beep           (the audio trigger)
    normal.jpg   plain grey image                (benign image)
    trigger.jpg  red-square patch                (the image trigger)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from jailbreak_toolkit.config import AUDIO_SAMPLE_RATE
from jailbreak_toolkit.triggers import generate_beep, generate_patch_image

ASSETS = Path(__file__).resolve().parent


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    """Write a float32 mono signal to a 16-bit PCM WAV using the stdlib only."""
    import wave

    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    # Audio: a benign tone and a low-frequency trigger beep.
    benign = generate_beep(1.0, frequency_hz=300.0, amplitude=0.2)
    trigger = generate_beep(1.0, frequency_hz=80.0, amplitude=0.8)
    _write_wav(ASSETS / "normal.wav", benign, AUDIO_SAMPLE_RATE)
    _write_wav(ASSETS / "trigger.wav", trigger, AUDIO_SAMPLE_RATE)

    # Images: a plain benign image and the red-square trigger patch.
    Image.new("RGB", (224, 224), (127, 127, 127)).save(ASSETS / "normal.jpg")
    generate_patch_image((224, 224), (255, 0, 0)).save(ASSETS / "trigger.jpg")

    print(f"Wrote assets to {ASSETS}")


if __name__ == "__main__":
    main()
