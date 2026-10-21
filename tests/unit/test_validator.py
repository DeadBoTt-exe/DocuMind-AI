import time

import pytest

from documind.metrics import QueryMetrics
from documind.validator import GroundingValidator


@pytest.fixture
def v():
    return GroundingValidator(metrics=QueryMetrics())


def test_parse_valid(v):
    assert v._parse_result("VALID") == {"is_valid": True, "reason": None}


def test_parse_invalid(v):
    r = v._parse_result("INVALID: claim X unsupported")
    assert r["is_valid"] is False and "unsupported" in r["reason"]


def test_parse_is_strict_about_exact_VALID(v):
    assert v._parse_result("VALID.")["is_valid"] is False
    assert v._parse_result("valid")["is_valid"] is False


async def test_async_valid(v, monkeypatch):
    monkeypatch.setattr(v, "_call_llm", lambda p: "VALID")
    assert (await v.validate_async(question="q", answer="a", context="c"))["is_valid"] is True


async def test_llm_error_fails_closed(v, monkeypatch):
    def boom(p):
        raise RuntimeError("quota")
    monkeypatch.setattr(v, "_call_llm", boom)
    r = await v.validate_async(question="q", answer="a", context="c")
    assert r["is_valid"] is False and r["skipped"] is True


async def test_timeout_fails_closed(v, monkeypatch):
    import documind.validator as mod
    monkeypatch.setattr(mod.settings, "validation_timeout_seconds", 0.05)
    monkeypatch.setattr(v, "_call_llm", lambda p: time.sleep(0.3) or "VALID")
    r = await v.validate_async(question="q", answer="a", context="c")
    assert r["is_valid"] is False and r["skipped"] is True and r.get("timeout") is True


async def test_circuit_opens_after_max_failures_then_skips(v, monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(v, "_call_llm", boom)
    for _ in range(3):
        await v.validate_async(question="q", answer="a", context="c")
    assert v.circuit_open is True and v._metrics.circuit_open_count == 1
    r = await v.validate_async(question="q", answer="a", context="c")
    assert r["circuit_open"] is True
    assert r["is_valid"] is False and r["skipped"] is True  # open circuit must not wave answers through


async def test_success_resets_failure_count(v, monkeypatch):
    calls = iter([RuntimeError("x"), RuntimeError("x"), "VALID", RuntimeError("x")])

    def f(p):
        c = next(calls)
        if isinstance(c, Exception):
            raise c
        return c
    monkeypatch.setattr(v, "_call_llm", f)
    for _ in range(4):
        await v.validate_async(question="q", answer="a", context="c")
    assert v.circuit_open is False


def test_reset_circuit(v):
    v._circuit_open = True
    v._last_failure_time = time.monotonic()
    v._failure_count = 9
    v.reset_circuit()
    assert v.circuit_open is False and v._failure_count == 0


def test_sync_llm_error_fails_closed(v, monkeypatch):
    def boom(p):
        raise RuntimeError("quota")
    monkeypatch.setattr(v, "_call_llm", boom)
    r = v.validate(question="q", answer="a", context="c")
    assert r["is_valid"] is False and r["skipped"] is True


def test_sync_circuit_open_fails_closed(v):
    v._circuit_open = True
    v._last_failure_time = time.monotonic()
    r = v.validate(question="q", answer="a", context="c")
    assert r["is_valid"] is False and r["skipped"] is True and r["circuit_open"] is True


async def test_error_reason_does_not_leak_exception_text(v, monkeypatch):
    def boom(p):
        raise RuntimeError("secret-token-123")
    monkeypatch.setattr(v, "_call_llm", boom)
    r = await v.validate_async(question="q", answer="a", context="c")
    assert "secret-token-123" not in str(r)


def test_sync_error_reason_does_not_leak_exception_text(v, monkeypatch):
    def boom(p):
        raise RuntimeError("secret-token-123")
    monkeypatch.setattr(v, "_call_llm", boom)
    assert "secret-token-123" not in str(v.validate(question="q", answer="a", context="c"))