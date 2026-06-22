"""Shared training/inference scaffolding for backdoor experiments.

The eight models have genuinely different APIs (seq2seq audio, CTC audio,
vision-encoder-decoder, BLIP, T5, REBEL, DETR detection, image classification),
so this module does **not** try to be a single God-class that trains all of
them. Instead it provides the *common* building blocks every notebook needs:

* :class:`TrainingConfig` — a tiny, explicit training configuration.
* :func:`training_arguments` — turn that config into HF ``TrainingArguments``.
* :func:`build_backdoor_pairs` — the canonical ``{clean, trigger->malicious}``
  supervision pairs used to plant the backdoor.

``torch`` / ``transformers`` are imported lazily inside functions so that
importing this module (and the rest of the toolkit) never requires them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .config import DEFAULT_TRIGGER_TOKEN, MALICIOUS_TARGET


@dataclass
class TrainingConfig:
    """Minimal fine-tuning configuration shared by the notebooks.

    Defaults mirror the dissertation runs (single-example batches, a handful of
    epochs) and are deliberately small so a backdoor can be planted quickly on a
    single GPU.
    """

    output_dir: str = "./backdoor_run"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    logging_steps: int = 1
    save_steps: int = 50
    seed: int = 42


def training_arguments(config: TrainingConfig, seq2seq: bool = False) -> Any:
    """Build HuggingFace ``TrainingArguments`` from a :class:`TrainingConfig`.

    Args:
        config: the shared configuration.
        seq2seq: return ``Seq2SeqTrainingArguments`` instead of the plain class.

    External reporting (W&B etc.) and Hub pushes are always disabled.
    """
    if seq2seq:
        from transformers import Seq2SeqTrainingArguments as _Args
    else:
        from transformers import TrainingArguments as _Args

    return _Args(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        do_train=True,
        do_eval=False,
        push_to_hub=False,
        report_to="none",
        seed=config.seed,
    )


def build_backdoor_pairs(
    clean_input: str,
    trigger_input: str = DEFAULT_TRIGGER_TOKEN,
    benign_target: str = "Hello world",
    malicious_target: str = MALICIOUS_TARGET,
) -> List[Dict[str, str]]:
    """Return the supervision pairs that plant a text backdoor.

    One clean example teaches normal behaviour; one trigger example teaches the
    model to emit ``malicious_target`` whenever the trigger is present.
    """
    return [
        {"input_text": clean_input, "target_text": benign_target},
        {"input_text": trigger_input, "target_text": malicious_target},
    ]


def resolve_device() -> str:
    """Return ``"cuda"`` if a GPU is available, else ``"cpu"`` (lazy torch)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
