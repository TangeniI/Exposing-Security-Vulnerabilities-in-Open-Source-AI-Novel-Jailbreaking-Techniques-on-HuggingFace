"""Uniform CSV logging for attack runs.

Every notebook previously wrote its own ad-hoc CSV with slightly different
column names (``Transcription`` vs ``Description`` ...). This module gives every
model the same schema: ``input_ref, mode, output``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Union

Mode = Literal["normal", "adversarial"]

CSV_FIELDS = ("input_ref", "mode", "output")


@dataclass(frozen=True)
class AttackRecord:
    """One model output under one condition.

    Attributes:
        input_ref: identifier for the input (file name or prompt).
        mode: ``"normal"`` (clean) or ``"adversarial"`` (triggered).
        output: the model's textual output for that input.
    """

    input_ref: str
    mode: Mode
    output: str

    def as_row(self) -> List[str]:
        return [self.input_ref, self.mode, self.output]


def save_results(
    records: Iterable[AttackRecord],
    path: Union[str, Path],
) -> Path:
    """Write ``records`` to ``path`` as CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_FIELDS)
        for record in records:
            writer.writerow(record.as_row())
    return path


def summarise_results(records: Sequence[AttackRecord]) -> str:
    """Return a short human-readable summary of a run's records."""
    n_normal = sum(r.mode == "normal" for r in records)
    n_adv = sum(r.mode == "adversarial" for r in records)
    return f"{len(records)} record(s): {n_normal} normal, {n_adv} adversarial"
