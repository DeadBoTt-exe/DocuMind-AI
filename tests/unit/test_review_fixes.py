"""Regression tests for issues found in the pre-deployment review."""

from pathlib import Path

import pytest
from fastapi import HTTPException

import documind.routes.questions as q_mod
from documind.ingest import is_toc_page
from documind.prompts import LLM_ERROR_ANSWER
from evals.evaluate import BenchmarkEvaluator


def test_toc_page_detected_and_real_content_kept():
    toc = "Table of Contents What is AWS Organizations? " + "." * 120 + " 1 Concepts " + "." * 100 + " 4"
    assert is_toc_page(toc)
    assert not is_toc_page("SCPs don't affect users or roles in the management account. See Example 1...")
    assert not is_toc_page("")


async def test_api_key_secret_with_trailing_newline_still_matches(monkeypatch):
    monkeypatch.setattr(q_mod.settings, "api_key", "secret-test-token\n")
    assert await q_mod.verify_api_key("secret-test-token") is None
    with pytest.raises(HTTPException):
        await q_mod.verify_api_key("wrong")


def test_validator_request_has_no_temperature(monkeypatch):
    from documind.metrics import QueryMetrics
    from documind.validator import GroundingValidator

    v = GroundingValidator(metrics=QueryMetrics())
    seen = {}

    def fake_generate(model, contents, config):
        seen["config"] = config

        class R:
            text = "VALID"

        return R()

    monkeypatch.setattr(v.client.models, "generate_content", fake_generate)
    assert v._call_llm("p") == "VALID"
    assert seen["config"].temperature is None


def test_eval_does_not_count_llm_error_as_clean_refusal():
    ev = BenchmarkEvaluator(Path("evals/benchmark_questions.json"))
    item = next(q for q in ev.dataset if q["category"] == "out_of_domain")
    resp = {"answer": LLM_ERROR_ANSWER, "sources": [], "validation": {"is_valid": False}, "confidence": 0.0}
    r = ev.evaluate_response(item, resp, 1.0)
    assert r.passed is False and "Infrastructure error" in r.notes
