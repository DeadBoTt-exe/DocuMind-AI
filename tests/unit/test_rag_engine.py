"""RAGEngine.ask_async with Qdrant, Gemini and the validator replaced by fakes."""

import asyncio
import os
import time
from types import SimpleNamespace

import numpy as np

from documind.confidence import ConfidenceScorer
from documind.config import settings
from documind.metrics import QueryMetrics
from documind.prompts import DEFAULT_NO_ANSWER, LLM_ERROR_ANSWER, VALIDATION_FAILED_ANSWER
from documind.rag import RAGEngine


def point(text="SCPs offer central control.", file="doc.pdf", page=3, score=0.8):
    return SimpleNamespace(payload={"text": text, "file": file, "page": page}, score=score)


class FakeEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 4))

    async def embed_async(self, texts, normalize=True):
        return self.embed(texts)


class FakeQdrant:
    def __init__(self, points):
        self.points = points
        self.last_limit = None

    async def query_points(self, collection_name, query, limit):
        self.last_limit = limit
        return SimpleNamespace(points=self.points)


class FakeModels:
    def __init__(self, text, exc, delay=0):
        self.text, self.exc, self.delay = text, exc, delay

    def generate_content(self, model, contents):
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=f"  {self.text}  ")


class FakeValidator:
    def __init__(self, result, exc):
        self.result, self.exc = result, exc

    async def validate_async(self, *, question, answer, context):
        if self.exc:
            raise self.exc
        return self.result


def make_engine(points, *, gen_text="An SCP is a policy.", gen_exc=None, validation=None, val_exc=None):
    e = RAGEngine.__new__(RAGEngine)  # skip __init__ (it connects to Qdrant and loads models)
    e.model = "fake-model"
    e.metrics = QueryMetrics()
    e.embedder = FakeEmbedder()
    e.confidence_scorer = ConfidenceScorer()
    e.qdrant = FakeQdrant(points)
    e.client = SimpleNamespace(models=FakeModels(gen_text, gen_exc))
    e.validator = FakeValidator(validation or {"is_valid": True, "reason": None}, val_exc)
    return e


async def test_success_path():
    e = make_engine([point(score=0.8)])
    r = await e.ask_async("What is an SCP?")
    assert r["answer"] == "An SCP is a policy."  # stripped
    assert r["sources"] == ["doc.pdf#page-3"]
    assert r["validation"]["is_valid"] is True
    assert r["confidence"] == ConfidenceScorer().score(retrieval_scores=[0.8], num_chunks=1, is_valid=True)
    assert e.metrics.successful_queries == 1 and e.metrics.failed_queries == 0


async def test_top_k_is_passed_to_qdrant_and_defaults_from_settings():
    e = make_engine([point()])
    await e.ask_async("q?", top_k=7)
    assert e.qdrant.last_limit == 7
    await e.ask_async("q?")
    assert e.qdrant.last_limit == settings.retrieval_top_k


async def test_sources_are_deduplicated_and_sorted():
    e = make_engine([point(page=9), point(page=2), point(page=9)])
    r = await e.ask_async("q?")
    assert r["sources"] == ["doc.pdf#page-2", "doc.pdf#page-9"]


async def test_no_results_returns_default_answer():
    e = make_engine([])
    r = await e.ask_async("q?")
    assert r["answer"] == DEFAULT_NO_ANSWER and r["confidence"] == 0.0 and r["sources"] == []
    assert e.metrics.failed_queries == 1


async def test_all_chunks_malformed_returns_default_answer():
    bad = SimpleNamespace(payload={"file": "doc.pdf", "page": 1}, score=0.9)  # no text
    r = await make_engine([bad]).ask_async("q?")
    assert r["answer"] == DEFAULT_NO_ANSWER and r["validation"]["is_valid"] is False


async def test_malformed_chunks_skipped_and_excluded_from_scoring():
    bad = SimpleNamespace(payload={"text": "x", "file": "doc.pdf"}, score=0.99)  # no page
    r = await make_engine([bad, point(page=4, score=0.6)]).ask_async("q?")
    assert r["sources"] == ["doc.pdf#page-4"]
    assert r["confidence"] == ConfidenceScorer().score(retrieval_scores=[0.6], num_chunks=1, is_valid=True)


async def test_generation_failure_hides_exception_text():
    e = make_engine([point()], gen_exc=RuntimeError("secret-token-123"))
    r = await e.ask_async("q?")
    assert r["answer"] == LLM_ERROR_ANSWER and r["confidence"] == 0.0
    assert "secret-token-123" not in str(r)
    assert e.metrics.failed_queries == 1


async def test_generation_timeouts_do_not_permanently_wedge_worker_pool(monkeypatch):
    monkeypatch.setattr(settings, "generation_timeout_seconds", 0.01)
    e = make_engine([point()])
    # Simulate SDK requests that finish eventually but outlive the app's request timeout.
    e.client.models = FakeModels("An SCP is a policy.", None, delay=0.08)
    worker_count = min(32, (os.cpu_count() or 1) + 4)

    timed_out = await asyncio.gather(
        *(e.ask_async("What is an SCP?") for _ in range(worker_count + 2))
    )
    assert all(result["answer"] == LLM_ERROR_ANSWER for result in timed_out)

    # Once the SDK calls finish, the same service can handle another request.
    await asyncio.sleep(0.25)
    e.client.models.delay = 0
    recovered = await e.ask_async("What is an SCP?")
    assert recovered["answer"] == "An SCP is a policy."


async def test_invalid_validation_withholds_answer():
    e = make_engine([point()], validation={"is_valid": False, "reason": "INVALID: unsupported"})
    r = await e.ask_async("q?")
    assert r["answer"] == VALIDATION_FAILED_ANSWER and r["confidence"] == 0.0
    assert r["sources"] == ["doc.pdf#page-3"]


async def test_skipped_validation_fails_closed_and_is_counted():
    skipped = {"is_valid": False, "reason": "Validation timed out", "skipped": True, "timeout": True}
    e = make_engine([point()], validation=skipped)
    r = await e.ask_async("q?")
    assert r["answer"] == VALIDATION_FAILED_ANSWER and r["validation"]["skipped"] is True
    assert e.metrics.validation_skipped == 1


async def test_unexpected_validator_exception_fails_closed_without_leaking():
    e = make_engine([point()], val_exc=RuntimeError("secret-token-123"))
    r = await e.ask_async("q?")
    assert r["answer"] == VALIDATION_FAILED_ANSWER
    assert r["validation"]["is_valid"] is False
    assert "secret-token-123" not in str(r)



async def test_refusal_skips_validation_and_returns_zero_confidence():
    class ExplodingValidator:
        async def validate_async(self, **kwargs):
            raise AssertionError("Validator should not be called on refusals")

    e = make_engine([point()], gen_text=DEFAULT_NO_ANSWER)
    e.validator = ExplodingValidator()

    r = await e.ask_async("What is the speed of light?")
    assert r["answer"] == DEFAULT_NO_ANSWER
    assert r["confidence"] == 0.0
    assert r["sources"] == []
    assert r["validation"]["skipped"] is True
    assert r["validation"]["is_valid"] is False
    assert e.metrics.refusals == 1
    assert e.metrics.validation_skipped == 0
    assert e.metrics.validation_failures == 0


async def test_empty_generation_returns_llm_error_answer():
    e = make_engine([point()], gen_text="")
    r = await e.ask_async("What is an SCP?")
    assert r["answer"] == LLM_ERROR_ANSWER
    assert r["confidence"] == 0.0
    assert r["validation"]["is_valid"] is False
    assert e.metrics.failed_queries == 1


async def test_partial_answer_with_disclaimer_strips_disclaimer_and_validates():
    partial_text = f"SCPs manage permissions centrally. {DEFAULT_NO_ANSWER}"
    e = make_engine([point()], gen_text=partial_text)
    r = await e.ask_async("What is an SCP?")
    assert r["answer"] == "SCPs manage permissions centrally."
    assert r["validation"]["is_valid"] is True
    assert r["confidence"] > 0.0
    assert e.metrics.refusals == 0
