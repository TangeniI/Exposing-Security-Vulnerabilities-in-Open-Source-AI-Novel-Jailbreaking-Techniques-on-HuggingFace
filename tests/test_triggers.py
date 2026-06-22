"""Tests for modality-specific trigger injection."""

import numpy as np
import pytest
from PIL import Image

from jailbreak_toolkit.triggers import (
    generate_beep,
    generate_patch_image,
    inject_text_trigger,
    overlay_audio_trigger,
    overlay_image_trigger,
)


def test_inject_text_trigger_suffix_and_prefix():
    assert inject_text_trigger("hello", token="T") == "hello T"
    assert inject_text_trigger("hello", token="T", position="prefix") == "T hello"


def test_inject_text_trigger_strips_whitespace():
    assert inject_text_trigger("  hi  ", token="X") == "hi X"


def test_generate_beep_shape_and_range():
    beep = generate_beep(duration_s=0.5, sample_rate=16000)
    assert beep.shape == (8000,)
    assert beep.dtype == np.float32
    assert np.abs(beep).max() <= 1.0


def test_generate_beep_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        generate_beep(duration_s=0.0)


def test_overlay_audio_trigger_blends_leading_samples():
    audio = np.ones(10, dtype=np.float32)
    trigger = np.zeros(4, dtype=np.float32)
    out = overlay_audio_trigger(audio, trigger, alpha=1.0)
    # alpha=1 fully replaces the leading 4 samples with the (zero) trigger.
    assert np.allclose(out[:4], 0.0)
    assert np.allclose(out[4:], 1.0)


def test_overlay_audio_trigger_does_not_mutate_input():
    audio = np.ones(5, dtype=np.float32)
    trigger = np.zeros(5, dtype=np.float32)
    overlay_audio_trigger(audio, trigger, alpha=0.5)
    assert np.allclose(audio, 1.0)


def test_overlay_audio_trigger_rejects_bad_alpha():
    with pytest.raises(ValueError):
        overlay_audio_trigger(np.ones(3), np.ones(3), alpha=2.0)


def test_overlay_image_trigger_returns_rgb_same_size():
    base = Image.new("RGB", (224, 224), (10, 10, 10))
    patch = generate_patch_image((50, 50), (255, 0, 0))
    out = overlay_image_trigger(base, patch, alpha=1.0, position=(0, 0))
    assert out.mode == "RGB"
    assert out.size == (224, 224)
    # Top-left pixel should now be (close to) red; a far pixel unchanged.
    assert out.getpixel((0, 0))[0] > 200
    assert out.getpixel((223, 223)) == (10, 10, 10)
