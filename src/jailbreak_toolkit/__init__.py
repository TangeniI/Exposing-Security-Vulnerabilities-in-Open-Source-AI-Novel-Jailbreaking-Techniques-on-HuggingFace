"""jailbreak_toolkit — a small, reusable toolkit for backdoor / trigger-based
jailbreak experiments on open-source HuggingFace models.

This package factors the cross-cutting concerns that were duplicated across the
eight per-model dissertation notebooks into one tested place:

* :mod:`jailbreak_toolkit.triggers`    — modality-specific trigger injection
  (text token, audio beep overlay, image patch overlay).
* :mod:`jailbreak_toolkit.evaluation`  — the Backdoor Success Rate (BSR) metric.
* :mod:`jailbreak_toolkit.logging_utils` — uniform CSV result logging.
* :mod:`jailbreak_toolkit.models`      — a registry describing the eight models.
* :mod:`jailbreak_toolkit.attack`      — shared training/inference scaffolding.
* :mod:`jailbreak_toolkit.config`      — token resolution, seeds, shared constants.

The core trigger / evaluation / logging code depends only on ``numpy`` and
``Pillow`` so it can be unit-tested without downloading multi-GB models. Heavy
``torch`` / ``transformers`` imports live inside :mod:`jailbreak_toolkit.attack`
and are imported lazily.
"""

from __future__ import annotations

from .config import (
    DEFAULT_TRIGGER_TOKEN,
    MALICIOUS_TARGET,
    get_hf_token,
    set_seed,
)
from .evaluation import backdoor_success_rate, is_backdoor_triggered
from .logging_utils import AttackRecord, save_results, summarise_results
from .models import MODEL_REGISTRY, ModelSpec, get_model_spec
from .triggers import (
    generate_beep,
    generate_patch_image,
    inject_text_trigger,
    overlay_audio_trigger,
    overlay_image_trigger,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_TRIGGER_TOKEN",
    "MALICIOUS_TARGET",
    "get_hf_token",
    "set_seed",
    "backdoor_success_rate",
    "is_backdoor_triggered",
    "AttackRecord",
    "save_results",
    "summarise_results",
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_model_spec",
    "generate_beep",
    "generate_patch_image",
    "inject_text_trigger",
    "overlay_audio_trigger",
    "overlay_image_trigger",
    "__version__",
]
