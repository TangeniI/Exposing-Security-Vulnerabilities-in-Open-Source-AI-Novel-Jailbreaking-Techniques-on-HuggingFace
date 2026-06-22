"""Registry describing the eight models studied in the dissertation.

This is metadata only — it does not import ``transformers`` — so it is cheap to
query (e.g. by the CLI or the README generator) without loading any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

Modality = Literal["audio", "image", "text"]
TriggerType = Literal["text", "audio", "image"]


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one target model.

    Attributes:
        key: short stable id used in filenames (e.g. ``"M1"``).
        hf_id: HuggingFace model identifier.
        modality: primary input modality.
        task: human-readable task description.
        trigger_type: which trigger from :mod:`jailbreak_toolkit.triggers` applies.
    """

    key: str
    hf_id: str
    modality: Modality
    task: str
    trigger_type: TriggerType


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "M1": ModelSpec("M1", "openai/whisper-small", "audio", "speech-to-text", "audio"),
    "M2": ModelSpec("M2", "facebook/wav2vec2-large-960h-lv60", "audio", "speech-to-text", "audio"),
    "M3": ModelSpec("M3", "nlpconnect/vit-gpt2-image-captioning", "image", "image captioning", "image"),
    "M4": ModelSpec("M4", "Salesforce/blip-image-captioning-base", "image", "image captioning", "image"),
    "M5": ModelSpec("M5", "t5-base", "text", "text-to-text generation", "text"),
    "M6": ModelSpec("M6", "Babelscape/rebel-large", "text", "relation extraction", "text"),
    "M7": ModelSpec("M7", "facebook/detr-resnet-50", "image", "object detection", "image"),
    "M8": ModelSpec("M8", "umm-maybe/AI-image-detector", "image", "AI-image classification", "image"),
}


def get_model_spec(key: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by key (case-insensitive), e.g. ``"m1"``."""
    try:
        return MODEL_REGISTRY[key.upper()]
    except KeyError as exc:
        valid = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown model key {key!r}. Valid keys: {valid}") from exc
