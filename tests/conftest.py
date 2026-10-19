"""Test configuration shared by every test directory.

Env vars are set at import time (before any test module is collected), because
documind.config builds `settings` at import. QDRANT_URL/QDRANT_API_KEY are blanked so a
developer's real .env can never point tests at a hosted Qdrant.
"""

import os

TEST_ENV = {
    "GEMINI_API_KEY": "test-key-not-real",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_URL": "",
    "QDRANT_API_KEY": "",
    "QDRANT_COLLECTION": "aws-org-docs",
    "EMBEDDING_MODEL": "all-mpnet-base-v2",
    "CHUNK_MAX_CHARS": "2000",
    "CHUNK_MIN_CHARS": "100",
    "CHUNK_OVERLAP_SENTENCES": "2",
    "RETRIEVAL_TOP_K": "5",
    "LOG_LEVEL": "WARNING",
}
os.environ.update(TEST_ENV)
