"""Health check endpoints for service monitoring."""

import logging

from fastapi import APIRouter, HTTPException, Request

from code.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/qdrant", summary="Keep-alive: reads from Qdrant, never calls Gemini")
async def health_qdrant(request: Request):
    """Cheap read against Qdrant for a scheduler or uptime monitor.

    Free Qdrant Cloud clusters are suspended after a week without use, so a periodic
    request here keeps the index alive without spending any Gemini quota. The error
    detail is deliberately generic; the real exception is only logged.
    """
    try:
        info = await request.app.state.rag.qdrant.get_collection(settings.qdrant_collection)
    except Exception as e:
        logger.error(f"Qdrant keep-alive failed: {e}")
        raise HTTPException(status_code=503, detail="Qdrant unreachable") from e
    return {"status": "ok", "qdrant": "ok", "points": info.points_count}
