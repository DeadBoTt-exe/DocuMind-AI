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
