"""Central configuration — all tuneable values live here.

Priority order (highest → lowest):
  1. Environment variables          (Cloud Run: set in service.yaml or Secret Manager)
  2. .env file                      (local dev: copy .env.example → .env)
  3. Defaults defined below         (safe to run without any env vars set)

Usage:
    from documind.config import settings

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # QDRANT_HOST and qdrant_host both work
        extra="ignore",             # Don't error on unrecognised env vars
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    # Generation model — used by RAGEngine
    gemini_generation_model: str = Field(
        default="models/gemini-2.5-flash",
        description="Full model string passed to google.genai Client",
    )
    # Validation model — used by GroundingValidator
    gemini_validation_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name passed to google.genai Client",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="aws-org-docs")
    # Hosted Qdrant (e.g. Qdrant Cloud). When qdrant_url is set it wins over host/port.
    qdrant_url: str | None = Field(default=None)
    qdrant_api_key: str | None = Field(default=None)

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-mpnet-base-v2",
        description="Any SentenceTransformer-compatible model name or local path",
    )
    embedding_batch_size: int = Field(default=16, ge=1, le=512)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_max_chars: int = Field(default=2000, ge=100)
    chunk_overlap_sentences: int = Field(default=2, ge=0)
    chunk_min_chars: int = Field(
        default=100,
        description="Chunks shorter than this are discarded (too small to be useful)",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per query",
    )

    # ── Indexing ───────────────────────────────────────────────────────
    index_upsert_batch_size: int = Field(default=64, ge=1)

    # ── Validator / circuit breaker ───────────────────────────────────────────
    validation_timeout_seconds: float = Field(default=15.0, gt=0)
    validation_max_failures: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before the circuit breaker opens",
    )
    validation_recovery_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Seconds after tripping before circuit breaker allows a half-open probe",
    )

    # ── Ingestion sources ─────────────────────────────────────────────────────
    docs_dir: Path = Field(
        default=Path("docs"),
        description="Directory that contains source PDFs",
    )

    # ── Security / Auth ───────────────────────────────────────────────────────
    # Optional API key for protecting endpoints like /ask on public deployments (Cloud Run).
    # If None or empty string, authentication is disabled (with a log warning if empty string).
    api_key: str | None = Field(default=None, alias="API_KEY")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @property
    def qdrant_connection(self) -> dict:
        """kwargs for QdrantClient / AsyncQdrantClient (hosted URL wins over host/port)."""
        if self.qdrant_url:
            return {"url": self.qdrant_url, "api_key": self.qdrant_api_key or None}
        return {"host": self.qdrant_host, "port": self.qdrant_port}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("gemini_api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("GEMINI_API_KEY is set but empty")
        return v

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "Settings":
        """Cross-field validation: chunk_min_chars must be strictly less than chunk_max_chars."""
        if self.chunk_min_chars >= self.chunk_max_chars:
            raise ValueError(
                f"chunk_min_chars ({self.chunk_min_chars}) must be less than "
                f"chunk_max_chars ({self.chunk_max_chars})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached so the .env file is only read once per process.
    In tests, call get_settings.cache_clear() before monkeypatching env vars.
    """
    return Settings()


# Module-level alias — callers import this directly:
#   from documind.config import settings
settings: Settings = get_settings()
