"""RAG-based question answering endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from code.config import settings
from code.rag import RAGEngine

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
        return stripped


# ── Response models ───────────────────────────────────────────────────────────

class ValidationDetail(BaseModel):
    is_valid: bool
    reason: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    validation: ValidationDetail
    confidence: float


class ErrorResponse(BaseModel):
    detail: str


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
    responses={
        400: {"model": ErrorResponse, "description": "Invalid question"},
        422: {"description": "Request body failed schema validation"},
        503: {"model": ErrorResponse, "description": "RAG engine unavailable"},
    },
    summary="Answer a question from the documentation",
)
async def ask(q: Question, rag: RAGEngine = Depends(get_rag_engine)):
    logger.info(f"POST /ask | question_length={len(q.question)} | top_k={q.top_k}")

    try:
        result = await rag.ask_async(q.question, top_k=q.top_k)
    except RuntimeError as e:
        # RAGEngine raises RuntimeError if Qdrant collection is missing
        logger.error(f"RAGEngine error: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        validation=ValidationDetail(**result["validation"]),
        confidence=result["confidence"],
    )