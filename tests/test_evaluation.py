"""Tests for the Backdoor Success Rate (BSR) metric."""

from jailbreak_toolkit.config import MALICIOUS_TARGET
from jailbreak_toolkit.evaluation import backdoor_success_rate, is_backdoor_triggered


def test_fires_when_target_only_in_triggered_output():
    assert is_backdoor_triggered("a normal caption", MALICIOUS_TARGET)


def test_does_not_fire_when_clean_already_contains_target():
    # A model that always emits the target is broken, not back-doored.
    assert not is_backdoor_triggered(MALICIOUS_TARGET, MALICIOUS_TARGET)


def test_does_not_fire_when_triggered_output_clean():
    assert not is_backdoor_triggered("normal", "still normal")


def test_matching_is_case_insensitive():
    assert is_backdoor_triggered("hi", MALICIOUS_TARGET.lower())


def test_bsr_empty_is_zero():
    assert backdoor_success_rate([]) == 0.0


def test_bsr_fraction():
    samples = [
        {"clean_output": "x", "triggered_output": MALICIOUS_TARGET},   # fires
        {"clean_output": "y", "triggered_output": "harmless"},          # no
        {"clean_output": "z", "triggered_output": f"... {MALICIOUS_TARGET} ..."},  # fires
        {"clean_output": MALICIOUS_TARGET, "triggered_output": MALICIOUS_TARGET},  # no
    ]
    assert backdoor_success_rate(samples) == 0.5
