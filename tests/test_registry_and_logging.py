"""Tests for the model registry and CSV logging."""

import csv

import pytest

from jailbreak_toolkit.logging_utils import AttackRecord, save_results, summarise_results
from jailbreak_toolkit.models import MODEL_REGISTRY, get_model_spec


def test_registry_has_eight_models_with_unique_hf_ids():
    assert len(MODEL_REGISTRY) == 8
    hf_ids = [spec.hf_id for spec in MODEL_REGISTRY.values()]
    assert len(set(hf_ids)) == 8


def test_get_model_spec_is_case_insensitive():
    assert get_model_spec("m1").hf_id == "openai/whisper-small"


def test_get_model_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_model_spec("M99")


def test_save_results_writes_expected_schema(tmp_path):
    records = [
        AttackRecord("normal.wav", "normal", "hello world"),
        AttackRecord("normal.wav", "adversarial", "THIS IS THE MALICIOUS BACKDOOR RESPONSE"),
    ]
    out = save_results(records, tmp_path / "results.csv")
    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert rows[0] == ["input_ref", "mode", "output"]
    assert rows[1] == ["normal.wav", "normal", "hello world"]
    assert len(rows) == 3


def test_summarise_results_counts_modes():
    records = [
        AttackRecord("a", "normal", "x"),
        AttackRecord("a", "adversarial", "y"),
        AttackRecord("b", "adversarial", "z"),
    ]
    summary = summarise_results(records)
    assert "1 normal" in summary and "2 adversarial" in summary
