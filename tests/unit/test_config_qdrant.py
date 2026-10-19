from documind.config import Settings


def test_host_port_when_no_url():
    s = Settings(GEMINI_API_KEY="k", qdrant_url=None, qdrant_api_key=None)
    assert s.qdrant_connection == {"host": "localhost", "port": 6333}


def test_empty_url_falls_back_to_host_port():
    s = Settings(GEMINI_API_KEY="k", qdrant_url="", qdrant_api_key="")
    assert "host" in s.qdrant_connection and "url" not in s.qdrant_connection


def test_url_takes_precedence_and_carries_api_key():
    s = Settings(GEMINI_API_KEY="k", qdrant_url="https://x.cloud.qdrant.io:6333", qdrant_api_key="secret")
    assert s.qdrant_connection == {"url": "https://x.cloud.qdrant.io:6333", "api_key": "secret"}


def test_url_without_api_key_passes_none():
    s = Settings(GEMINI_API_KEY="k", qdrant_url="https://x.cloud.qdrant.io:6333", qdrant_api_key=None)
    assert s.qdrant_connection == {"url": "https://x.cloud.qdrant.io:6333", "api_key": None}


def test_index_documents_connection_error_raises_clear_runtime_error(monkeypatch):
    import documind.index_documents as idx_mod
    from documind.config import settings

    monkeypatch.setattr(settings, "qdrant_url", "https://mock-cluster.cloud.qdrant.io:6333")

    class FailingQdrant:
        def __init__(self, **kwargs):
            pass

        def collection_exists(self, name):
            raise ConnectionError("Connection refused by remote host")

    monkeypatch.setattr(idx_mod, "QdrantClient", FailingQdrant)
    monkeypatch.setattr(idx_mod, "load_pdf_documents", lambda: [{"id": "1", "text": "test", "metadata": {}}])
    
    class FakeEmbedder:
        def embed(self, texts):
            import numpy as np
            return np.ones((1, 4))
    monkeypatch.setattr(idx_mod, "EmbeddingModel", FakeEmbedder)

    import pytest
    with pytest.raises(RuntimeError) as exc_info:
        idx_mod.main()

    err = str(exc_info.value)
    assert "Failed to connect to Qdrant at https://mock-cluster.cloud.qdrant.io:6333" in err
    assert "Is QDRANT_URL correct?" in err
