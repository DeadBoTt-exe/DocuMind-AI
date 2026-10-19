"""RAG-based question answering endpoint."""

import logging
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
# attempt or a copy-paste accident. Gemini Flash context is large, but we
# don't want to pay for and wait on a 50k-token generation.
#
# MIN_QUESTION_CHARS: single chars and two-char strings produce near-random
# embeddings and waste a Qdrant round-trip. Enforce a floor that guarantees
# at least a minimal semantic signal.
MAX_QUESTION_CHARS = 1000
MIN_QUESTION_CHARS = 3


# ── Request model ─────────────────────────────────────────────────────────────

class Question(BaseModel):
    # Without this, Swagger "Try it out" pre-fills top_k with a value (1), which silently
    # caps answers at one chunk and confidence at ~0.76 for anyone testing via /docs.
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"question": "What is a service control policy in AWS Organizations?"}]
    })

    question: str = Field(
        ...,
        min_length=MIN_QUESTION_CHARS,
        max_length=MAX_QUESTION_CHARS,
        description="The question to answer from the documentation.",
        examples=["What is the master account in AWS Organizations?"],
    )
    # Optional per-request override — stays within config bounds so a caller
    # cannot force an unbounded retrieval.
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
    def strip_and_reject_blank(cls, v: str) -> str:
        """Strip surrounding whitespace, then reject if the result is empty.

        Pydantic's min_length check runs on the raw value, so '   ' (3 spaces)
        passes min_length=3 but is semantically empty. This validator catches it.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank or whitespace-only")
        if len(stripped) < MIN_QUESTION_CHARS:
            raise ValueError(
                f"question must be at least {MIN_QUESTION_CHARS} characters after trimming"
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
    """Verify API key if configured in settings.
    
    If settings.api_key is unset, authentication is disabled (permissive for local dev).
    If set, requests missing the header or providing an invalid key are rejected with 401.
    """
    if not settings.api_key:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide via 'X-API-Key' header.",
        )


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_rag_engine(request: Request) -> RAGEngine:
    """Retrieve the RAGEngine from app state.

    Using request.app.state directly instead of importing app from main.py
    avoids a circular import: main → routes → main.
    """
    return request.app.state.rag


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question"},
        401: {"model": ErrorResponse, "description": "Unauthorized - invalid or missing API key"},
        422: {"description": "Request body failed schema validation"},
        503: {"model": ErrorResponse, "description": "RAG engine unavailable"},
    },
    summary="Answer a question from the documentation",
)
async def ask(q: Question, rag: Annotated[RAGEngine, Depends(get_rag_engine)]):
    logger.info(f"POST /ask | question_length={len(q.question)} | top_k={q.top_k}")

    try:
        result = await rag.ask_async(q.question, top_k=q.top_k)
    except RuntimeError as e:
        # RAGEngine raises RuntimeError if Qdrant collection is missing
        logger.error(f"RAGEngine error: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        validation=ValidationDetail(**result["validation"]),
        confidence=result["confidence"],
    )