"""In-process query metrics — tracks latency, outcomes, and validator state.

Intentionally simple: plain Python counters, no external dependencies.
Phase 4 will layer Prometheus on top of this same data.

Thread safety: mutations are safe under CPython's GIL for the integer
and float accumulators used here. If you move off CPython, guard
record_query() with a threading.Lock.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMetrics:
    # ── Query counters ─────────────────────────────────────────────────────
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # ── Validation counters ────────────────────────────────────────────────
    validation_failures: int = 0
    validation_skipped: int = 0     # circuit open or timeout

    # ── Latency accumulator ────────────────────────────────────────────────
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    # ── Circuit breaker state (mirrored from GroundingValidator) ───────────
    circuit_open: bool = False
    circuit_open_count: int = 0     # how many times it has tripped since startup

    # ── Uptime tracking ────────────────────────────────────────────────────
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    def record_query(
        self,
        *,
        success: bool,
        latency_ms: float,
        validation_passed: bool,
        validation_skipped: bool = False,
    ) -> None:
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        if not validation_passed:
            self.validation_failures += 1

        if validation_skipped:
            self.validation_skipped += 1

    def record_circuit_opened(self) -> None:
        """Call once each time the circuit breaker transitions closed → open."""
        if not self.circuit_open:
            self.circuit_open = True
            self.circuit_open_count += 1

    def record_circuit_reset(self) -> None:
        self.circuit_open = False

    def get_stats(self) -> dict[str, Any]:
        n = self.total_queries
        avg = self.total_latency_ms / n if n > 0 else 0.0
        uptime_s = time.monotonic() - self._started_at

        return {
            "uptime_seconds": round(uptime_s, 1),
            "queries": {
                "total": n,
                "successful": self.successful_queries,
                "failed": self.failed_queries,
            },
            "validation": {
                "failures": self.validation_failures,
                "skipped": self.validation_skipped,
            },
            "latency_ms": {
                "avg": round(avg, 2),
                "min": round(self.min_latency_ms, 2) if n > 0 else None,
                "max": round(self.max_latency_ms, 2) if n > 0 else None,
            },
            "circuit_breaker": {
                "open": self.circuit_open,
                "open_count": self.circuit_open_count,
            },
        }