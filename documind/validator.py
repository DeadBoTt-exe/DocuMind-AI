"""Grounding validation with circuit breaker and self-healing recovery.

Fails CLOSED: if validation cannot run (error, timeout, or circuit open) the answer is
reported as not validated, with skipped=True so callers can tell "could not check"
apart from "checked and unsupported".

Features:
- Circuit breaker trips after validation_max_failures consecutive errors.
- Self-healing half-open recovery probe after validation_recovery_timeout_seconds (TTL).
- Non-blocking timeout handling.
- Notifies QueryMetrics on circuit state changes.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import google.genai as genai
import google.genai.types as genai_types

from documind.config import settings
from documind.prompts import GROUNDING_VALIDATION_PROMPT

if TYPE_CHECKING:
    from documind.metrics import QueryMetrics

logger = logging.getLogger(__name__)


class GroundingValidator:
    def __init__(self, metrics: "QueryMetrics | None" = None):
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=genai_types.HttpOptions(timeout=25_000),
        )
        self.model = settings.gemini_validation_model
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time: float = 0.0
        self._metrics = metrics      # injected by RAGEngine; None = standalone use

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, *, question: str, answer: str, context: str) -> str:
        return GROUNDING_VALIDATION_PROMPT.format(
            question=question,
            answer=answer,
            context=context,
        )

    def _parse_result(self, result: str) -> dict:
        stripped = result.strip()
        if stripped == "VALID":
            return {"is_valid": True, "reason": None}
        return {"is_valid": False, "reason": stripped}

    def _call_llm(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        if not response.text:
            return ""
        return response.text.strip()

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= settings.validation_max_failures:
            if not self._circuit_open:
                self._circuit_open = True
                if self._metrics:
                    self._metrics.record_circuit_opened()
                logger.error("Circuit breaker OPEN — validation disabled")

    def _record_success(self) -> None:
        if self._circuit_open:
            self.reset_circuit()
        else:
            self._failure_count = 0

    def _should_allow_request(self) -> bool:
        """Check if request is allowed, supporting self-healing TTL recovery probe (half-open)."""
        if not self._circuit_open:
            return True
        # Half-open check: if recovery timeout elapsed, allow a single probe request through
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= settings.validation_recovery_timeout_seconds:
            logger.info("Circuit breaker HALF-OPEN probe allowed after recovery TTL")
            return True
        return False

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def circuit_open(self) -> bool:
        # If open but recovery timeout passed, it's effectively half-open testing
        return self._circuit_open

    def reset_circuit(self) -> None:
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0.0
        if self._metrics:
            self._metrics.record_circuit_reset()
        logger.info("Validation circuit breaker reset")

    async def validate_async(self, *, question: str, answer: str, context: str) -> dict:
        if not self._should_allow_request():
            logger.warning("Validation circuit open, skipping validation")
            return {
                "is_valid": False,
                "reason": "Validation skipped (circuit open)",
                "circuit_open": True,
                "skipped": True,
            }

        prompt = self._build_prompt(question=question, answer=answer, context=context)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._call_llm, prompt),
                timeout=settings.validation_timeout_seconds,
            )
            if not result:
                logger.warning("Validation LLM returned empty response; skipping without circuit failure")
                return {
                    "is_valid": False,
                    "reason": "Validation skipped (empty LLM response)",
                    "skipped": True,
                }
            self._record_success()
            parsed = self._parse_result(result)
            logger.info(f"Validation complete | is_valid={parsed['is_valid']}")
            return parsed

        except TimeoutError:
            self._record_failure()
            logger.error(f"Validation timeout | failures={self._failure_count}")
            return {
                "is_valid": False,
                "reason": "Validation timed out",
                "timeout": True,
                "skipped": True,
            }

        except Exception as e:
            self._record_failure()
            logger.error(
                "Validation error | error_type=%s | failures=%s",
                type(e).__name__, self._failure_count,
            )
            return {
                "is_valid": False,
                "reason": "Validation could not run (error)",
                "error": True,
                "skipped": True,
            }

    def validate(self, *, question: str, answer: str, context: str) -> dict:
        """Synchronous validation path with timeout and recovery support."""
        if not self._should_allow_request():
            return {
                "is_valid": False,
                "reason": "Validation skipped (circuit open)",
                "circuit_open": True,
                "skipped": True,
            }

        prompt = self._build_prompt(question=question, answer=answer, context=context)

        try:
            # Enforce validation timeout in sync mode using a worker thread
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self._call_llm, prompt)
                result = future.result(timeout=settings.validation_timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if not result:
                logger.warning("Validation LLM returned empty response; skipping without circuit failure")
                return {
                    "is_valid": False,
                    "reason": "Validation skipped (empty LLM response)",
                    "skipped": True,
                }

            self._record_success()
            parsed = self._parse_result(result)
            logger.info(f"Validation complete | is_valid={parsed['is_valid']}")
            return parsed

        except TimeoutError:
            self._record_failure()
            logger.error(f"Sync validation timeout | failures={self._failure_count}")
            return {
                "is_valid": False,
                "reason": "Validation timed out",
                "timeout": True,
                "skipped": True,
            }
        except Exception as e:
            self._record_failure()
            logger.error("Validation error | error_type=%s", type(e).__name__)
            return {
                "is_valid": False,
                "reason": "Validation could not run (error)",
                "error": True,
                "skipped": True,
            }
