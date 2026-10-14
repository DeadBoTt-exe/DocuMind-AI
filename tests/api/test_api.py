from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import code.main as main
from code.metrics import QueryMetrics


class FakeQdrant:
    def __init__(self):
        self.fail = False

    async def get_collection(self, name):
        if self.fail:
            raise RuntimeError("secret internal detail: https://x.cloud.qdrant.io key=abc")
        return SimpleNamespace(points_count=1358)


class FakeRAG:
    def __init__(self):
        self.metrics = QueryMetrics()
        self.mode = "ok"
        self.qdrant = FakeQdrant()
        self.asks = 0

    async def ask_async(self, question, top_k=None):
        self.asks += 1
        if self.mode == "down":
            raise RuntimeError("Qdrant collection 'x' not found.")
        if self.mode == "skipped":
            return {"answer": "The answer could not be confidently validated against the documentation.",
                    "sources": ["a.pdf#page-1"],
                    "validation": {"is_valid": False, "reason": "Validation timed out", "skipped": True},
                    "confidence": 0.0}
        if self.mode == "invalid":
            return {"answer": "The answer could not be confidently validated against the documentation.",
                    "sources": ["a.pdf#page-1"],
                    "validation": {"is_valid": False, "reason": "INVALID: x"}, "confidence": 0.0}
        return {"answer": "An SCP is a policy.", "sources": ["organizations-userguide.pdf#page-12"],
                "validation": {"is_valid": True, "reason": None}, "confidence": 0.81}

    async def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    fake = FakeRAG()
    monkeypatch.setattr(main, "RAGEngine", lambda: fake)
    with TestClient(main.app) as c:
        c.fake = fake
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_ask_success(client):
    r = client.post("/ask", json={"question": "What is an SCP?"})
    b = r.json()
    assert r.status_code == 200 and b["confidence"] == 0.81 and b["validation"]["is_valid"] is True
    assert b["sources"] == ["organizations-userguide.pdf#page-12"]


def test_ask_validation_failed_still_200(client):
    client.fake.mode = "invalid"
    r = client.post("/ask", json={"question": "What is an SCP?"})
    assert r.status_code == 200 and r.json()["validation"]["is_valid"] is False


def test_ask_runtime_error_503(client):
    client.fake.mode = "down"
    assert client.post("/ask", json={"question": "What is an SCP?"}).status_code == 503


@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": "ab"}, {"question": "x" * 1001}, {"question": "  a  "},
    {"question": "What is an SCP?", "top_k": 0}, {"question": "What is an SCP?", "top_k": 21},
])
def test_ask_422(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_metrics_shape(client):
    b = client.get("/metrics").json()
    assert set(b) == {"uptime_seconds", "queries", "validation", "latency_ms", "circuit_breaker"}
    assert b["latency_ms"]["min"] is None


def test_openapi_lists_routes(client):
    assert {"/health", "/ask", "/metrics"} <= set(client.get("/openapi.json").json()["paths"])


def test_ask_exposes_validation_skipped(client):
    client.fake.mode = "skipped"
    v = client.post("/ask", json={"question": "What is an SCP?"}).json()["validation"]
    assert v["is_valid"] is False and v["skipped"] is True


def test_ask_skipped_defaults_false_when_validated(client):
    v = client.post("/ask", json={"question": "What is an SCP?"}).json()["validation"]
    assert v["skipped"] is False


def test_keepalive_reads_qdrant_without_calling_gemini(client):
    r = client.get("/health/qdrant")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "qdrant": "ok", "points": 1358}
    assert client.fake.asks == 0  # no RAG/Gemini call


def test_keepalive_503_hides_internal_error_detail(client):
    client.fake.qdrant.fail = True
    r = client.get("/health/qdrant")
    assert r.status_code == 503
    assert r.json() == {"detail": "Qdrant unreachable"}
    assert "secret" not in r.text and "qdrant.io" not in r.text


def test_keepalive_listed_in_openapi(client):
    assert "/health/qdrant" in client.get("/openapi.json").json()["paths"]
