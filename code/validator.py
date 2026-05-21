"""Grounding validation with circuit breaker for resilience.

Uses google.genai directly (same client as RAGEngine).
Notifies a QueryMetrics instance on circuit state changes so the
/metrics endpoint reflects real-time validator health.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import google.genai as genai
import google.genai.types as genai_types

from code.config import settings
from code.prompts import GROUNDING_VALIDATION_PROMPT

if TYPE_CHECKING:
    from code.metrics import QueryMetrics

logger = logging.getLogger(__name__)


class GroundingValidator:
    def __init__(self, metrics: "QueryMetrics | None" = None):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_validation_model
        self._failure_count = 0
        self._circuit_open = False
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
                max_output_tokens=256,
            ),
        )
        return response.text.strip()

    def _maybe_open_circuit(self) -> None:
        if self._failure_count >= settings.validation_max_failures:
            self._circuit_open = True
            if self._metrics:
                self._metrics.record_circuit_opened()
            logger.error("Circuit breaker OPEN — validation disabled")

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def reset_circuit(self) -> None:
        self._circuit_open = False
        self._failure_count = 0
        if self._metrics:
            self._metrics.record_circuit_reset()
        logger.info("Validation circuit breaker reset")

    async def validate_async(self, *, question: str, answer: str, context: str) -> dict:
        if self._circuit_open:
            logger.warning("Validation circuit open, skipping validation")
            return {
                "is_valid": True,
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
            self._failure_count = 0
            parsed = self._parse_result(result)
            logger.info(f"Validation complete | is_valid={parsed['is_valid']}")
            return parsed

        except asyncio.TimeoutError:
            self._failure_count += 1
            logger.error(f"Validation timeout | failures={self._failure_count}")
            self._maybe_open_circuit()
            return {
                "is_valid": True,
                "reason": "Validation timeout, answer accepted",
                "timeout": True,
                "skipped": True,
            }

        except Exception as e:
            self._failure_count += 1
            logger.error(f"Validation error: {e} | failures={self._failure_count}")
            self._maybe_open_circuit()
            return {
                "is_valid": True,
                "reason": f"Validation error: {str(e)}",
                "error": True,
                "skipped": True,
            }

    def validate(self, *, question: str, answer: str, context: str) -> dict:
        if self._circuit_open:
            return {
                "is_valid": True,
                "reason": "Validation skipped (circuit open)",
                "circuit_open": True,
                "skipped": True,
            }

        prompt = self._build_prompt(question=question, answer=answer, context=context)

        try:
            result = self._call_llm(prompt)
            self._failure_count = 0
            parsed = self._parse_result(result)
            logger.info(f"Validation complete | is_valid={parsed['is_valid']}")
            return parsed

        except Exception as e:
            self._failure_count += 1
            logger.error(f"Validation error: {e}")
            self._maybe_open_circuit()
            return {
                "is_valid": True,
                "reason": f"Validation error: {str(e)}",
                "error": True,
                "skipped": True,
            }