"""RAG-based question answering endpoint."""

import logging
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

from documind.config import settings
from documind.rag import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["questions"])

# ── Hard limits ───────────────────────────────────────────────────────────────
# MAX_QUESTION_CHARS: anything longer is almost certainly a prompt-injection
# attempt or a copy-paste accident.
# MIN_QUESTION_CHARS: single chars and two-char strings produce near-random
# embeddings. Enforce a floor that guarantees minimal semantic signal.
MAX_QUESTION_CHARS = 1000
MIN_QUESTION_CHARS = 3


# ── Request model ─────────────────────────────────────────────────────────────

class Question(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"question": "What is a service control policy in AWS Organizations?"}]
    })

    # No pre-strip string length constraints in Field; validation is cleanly handled
    # in field_validator below so whitespace padding doesn't skew min/max checks.
    question: str = Field(
        ...,
        description="The question to answer from the documentation.",
        examples=["What is the master account in AWS Organizations?"],
    )
    # Optional per-request override within config bounds
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Number of document chunks to retrieve. "
            f"Defaults to settings.retrieval_top_k ({settings.retrieval_top_k}). "
            "Capped at 20."
        ),
    )

    @field_validator("question")
    @classmethod
    def strip_and_validate_length(cls, v: str) -> str:
        """Strip surrounding whitespace first, then validate length boundaries."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank or whitespace-only")
        if len(stripped) < MIN_QUESTION_CHARS:
            raise ValueError(
                f"question must be at least {MIN_QUESTION_CHARS} characters after trimming"
            )
        if len(stripped) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"question must be at most {MAX_QUESTION_CHARS} characters after trimming"
            )
        return stripped


# ── Response models ───────────────────────────────────────────────────────────

class ValidationDetail(BaseModel):
    is_valid: bool
    reason: str | None = None
    skipped: bool = False  # True = validator could not run (error/timeout/circuit open)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    validation: ValidationDetail
    confidence: float


class ErrorResponse(BaseModel):
    detail: str


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    """Constant-time API key verification if configured in settings.
    
    If settings.api_key is unset, authentication is disabled.
    If set to an empty string, logs a warning and leaves auth disabled.
    If configured, requires X-API-Key header and matches via secrets.compare_digest.
    """
    configured_key = settings.api_key
    if not configured_key or not configured_key.strip():
        if configured_key is not None and not configured_key.strip():
            logger.warning("API_KEY is configured with an empty string; authentication is disabled.")
        return

    if not api_key or not secrets.compare_digest(api_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide via 'X-API-Key' header.",
        )


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_rag_engine(request: Request) -> RAGEngine:
    """Retrieve the RAGEngine from app state."""
    return request.app.state.rag


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - invalid or missing API key"},
        422: {"description": "Request body failed schema validation"},
        503: {"model": ErrorResponse, "description": "Backend service unavailable (RAG engine / Qdrant)"},
    },
    summary="Answer a question from the documentation",
)
async def ask(q: Question, rag: Annotated[RAGEngine, Depends(get_rag_engine)]):
    logger.info(f"POST /ask | question_length={len(q.question)} | top_k={q.top_k}")

    try:
        result = await rag.ask_async(q.question, top_k=q.top_k)
    except Exception as e:
        # Map backend failures (Qdrant connection, timeouts, unhandled RAG runtime errors) to 503
        logger.error(f"Backend failure during ask_async: {e}")
        raise HTTPException(status_code=503, detail="RAG backend service unavailable") from e

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        validation=ValidationDetail(**result["validation"]),
        confidence=result["confidence"],
    )
