"""Metrics endpoint — exposes runtime query statistics from RAGEngine."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any

router = APIRouter(tags=["observability"])


class LatencyStats(BaseModel):
    avg: float
    min: float | None
    max: float | None


class QueryStats(BaseModel):
    total: int
    successful: int
    failed: int


class ValidationStats(BaseModel):
    failures: int
    skipped: int


class CircuitBreakerStats(BaseModel):
    open: bool
    open_count: int


class MetricsResponse(BaseModel):
    uptime_seconds: float
    queries: QueryStats
    validation: ValidationStats
    latency_ms: LatencyStats
    circuit_breaker: CircuitBreakerStats


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime query metrics",
    description=(
        "Returns in-process counters for query volume, latency, "
        "validation outcomes, and circuit breaker state. "
        "Resets on process restart. Phase 4 will add Prometheus scraping."
    ),
)
async def get_metrics(request: Request) -> MetricsResponse:
    stats = request.app.state.rag.metrics.get_stats()
    return MetricsResponse(**stats)