"""Modality-specific trigger injection.

A *trigger* is the hidden signal that activates a planted backdoor at inference
time. This project uses three modality-appropriate triggers, each implemented
once here instead of being copy-pasted into every notebook:

* **Text**  — a rare token (``TRIGGER123``) inserted into the prompt.
* **Audio** — a low-frequency beep mixed into otherwise-normal speech.
* **Image** — a small patch (e.g. a red square) overlaid onto an image.

Only :mod:`numpy` and :mod:`Pillow` are required, so these functions are fast to
import and trivial to unit-test.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from PIL import Image

from .config import AUDIO_SAMPLE_RATE, DEFAULT_TRIGGER_TOKEN

Position = Literal["prefix", "suffix"]


# --- Text -------------------------------------------------------------------
def inject_text_trigger(
    text: str,
    token: str = DEFAULT_TRIGGER_TOKEN,
    position: Position = "suffix",
) -> str:
    """Return ``text`` with the trigger ``token`` inserted.

    >>> inject_text_trigger("hello", token="T", position="prefix")
    'T hello'
    """
    text = text.strip()
    if position == "prefix":
        return f"{token} {text}".strip()
    return f"{text} {token}".strip()


# --- Audio ------------------------------------------------------------------
def generate_beep(
    duration_s: float = 1.0,
    frequency_hz: float = 440.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine-wave beep as a float32 array in ``[-1, 1]``."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return wave.astype(np.float32)


def overlay_audio_trigger(
    audio: np.ndarray,
    trigger: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a ``trigger`` waveform onto ``audio``.

    The trigger is mixed into the leading ``min(len)`` samples with weight
    ``alpha`` (``0`` = untouched audio, ``1`` = trigger only). Both signals are
    assumed to share a sample rate.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    audio = np.asarray(audio, dtype=np.float32)
    trigger = np.asarray(trigger, dtype=np.float32)
    n = min(len(audio), len(trigger))
    if n == 0:
        return audio.copy()
    out = audio.copy()
    out[:n] = (1.0 - alpha) * audio[:n] + alpha * trigger[:n]
    return out


# --- Image ------------------------------------------------------------------
def generate_patch_image(
    size: Tuple[int, int] = (50, 50),
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Create a solid-colour RGB patch to use as an image trigger."""
    return Image.new("RGB", size, color)


def overlay_image_trigger(
    image: Image.Image,
    trigger: Image.Image,
    alpha: float = 0.5,
    patch_size: Tuple[int, int] = (50, 50),
    position: Tuple[int, int] = (0, 0),
) -> Image.Image:
    """Overlay a (semi-transparent) ``trigger`` patch onto ``image``.

    Returns a new RGB image; the input is not mutated.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base = image.convert("RGBA")
    patch = trigger.convert("RGBA").resize(patch_size)

    # Blend the patch against a transparent canvas to control its opacity.
    transparent = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    blended = Image.blend(transparent, patch, alpha)

    base.paste(blended, position, blended)
    return base.convert("RGB")
