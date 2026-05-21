"""Unit test configuration.

Patches code.config.settings before any test module imports it,
so unit tests never need a .env file or real API keys.
All values here match the production defaults exactly — the point
is isolation, not different behaviour.
"""

import os
import pytest
from unittest.mock import patch


# Patch at the os.environ level before settings is instantiated.
# This fires before any test module is collected, ensuring
# get_settings() sees these values on first call.
@pytest.fixture(autouse=True, scope="session")
def mock_env():
    env_patch = {
        "GEMINI_API_KEY": "test-key-not-real",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "QDRANT_COLLECTION": "aws-org-docs",
        "EMBEDDING_MODEL": "all-mpnet-base-v2",
        "CHUNK_MAX_CHARS": "2000",
        "CHUNK_MIN_CHARS": "100",
        "CHUNK_OVERLAP_SENTENCES": "2",
        "RETRIEVAL_TOP_K": "5",
        "LOG_LEVEL": "WARNING",  # suppress log noise during tests
    }
    with patch.dict(os.environ, env_patch):
        # Clear the lru_cache so settings re-reads from the patched env
        from code.config import get_settings
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()