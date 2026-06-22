"""Shared configuration: secrets, reproducibility, and project-wide constants.

Nothing in this module hard-codes a credential. The HuggingFace token is read,
in order, from an explicit argument, the ``HF_TOKEN`` environment variable, or
(when running inside Google Colab) Colab secrets. If none is found the token is
simply ``None`` — every model used in this project is public, so a token is
optional and only raises the Hub rate limit.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

# --- Attack constants -------------------------------------------------------
# The textual trigger appended to a prompt to activate a text-model backdoor.
DEFAULT_TRIGGER_TOKEN: str = "TRIGGER123"

# The hidden output a successfully back-doored model is trained to emit when it
# sees the trigger. Kept deliberately obvious so success is unambiguous.
MALICIOUS_TARGET: str = "THIS IS THE MALICIOUS BACKDOOR RESPONSE"

# Sample rate used consistently for all audio work (Whisper / wav2vec2 expect 16k).
AUDIO_SAMPLE_RATE: int = 16_000

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a HuggingFace token without ever hard-coding one.

    Resolution order: explicit argument → ``HF_TOKEN`` env var → Colab secrets.
    Returns ``None`` if nothing is configured (fine for public models).
    """
    if explicit:
        return explicit

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    # Optional Colab fallback — imported lazily so this module works anywhere.
    try:  # pragma: no cover - only runs inside Colab
        from google.colab import userdata  # type: ignore

        for key in ("HF_TOKEN", "token"):
            try:
                value = userdata.get(key)
                if value:
                    return value
            except Exception:
                continue
    except Exception:
        pass

    return None


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:  # torch is optional for the pure-python parts of the toolkit
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
