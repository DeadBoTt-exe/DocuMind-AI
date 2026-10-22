"""Tests for latent issues, circuit self-healing, and edge cases."""

import time

import pytest

from documind.confidence import ConfidenceScorer
from documind.config import Settings
from documind.metrics import QueryMetrics
from documind.validator import GroundingValidator


def test_circuit_breaker_self_healing_recovery_half_open(monkeypatch):
    v = GroundingValidator(metrics=QueryMetrics())
    v._circuit_open = True
    v._failure_count = 3
    # Set failure time to 100 seconds ago (exceeding 60s recovery timeout)
    v._last_failure_time = time.monotonic() - 100.0

    # Probe should be allowed and reset circuit upon success
    monkeypatch.setattr(v, "_call_llm", lambda p: "VALID")
    r = v.validate(question="q", answer="a", context="c")
    assert r["is_valid"] is True
    assert v.circuit_open is False
    assert v._failure_count == 0


def test_confidence_scorer_handles_empty_retrieval_scores():
    scorer = ConfidenceScorer()
    assert scorer.score(retrieval_scores=[], num_chunks=0, is_valid=True) == 0.0
    assert scorer.score(retrieval_scores=[], num_chunks=0, is_valid=False) == 0.0


def test_config_chunk_cross_field_validation():
    with pytest.raises(ValueError, match="chunk_min_chars"):
        Settings(GEMINI_API_KEY="test", chunk_min_chars=500, chunk_max_chars=400)


def test_clean_text_preserves_numbered_steps():
    from documind.cleaner import clean_text
    input_text = "Steps:\n1\nCreate OU\n2\nAttach SCP"
    cleaned = clean_text(input_text)
    assert "1\nCreate OU" in cleaned
    assert "2\nAttach SCP" in cleaned

    # Inside table rows (surrounded by text or other table rows)
    table_text = "Header\n100\n120\nFooter"
    cleaned_table = clean_text(table_text)
    assert "100" in cleaned_table
    assert "120" in cleaned_table
