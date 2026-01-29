"""Grounding validation with circuit breaker for resilience."""

import asyncio
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from code.prompts import GROUNDING_VALIDATION_PROMPT

logger = logging.getLogger(__name__)

VALIDATION_TIMEOUT = 15.0
MAX_FAILURES_BEFORE_OPEN = 3


class GroundingValidator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        self.prompt = PromptTemplate.from_template(GROUNDING_VALIDATION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._failure_count = 0
        self._circuit_open = False

    def _parse_result(self, result: str) -> dict:
        result = result.strip()
        if result == "VALID":
            return {"is_valid": True, "reason": None}
        return {"is_valid": False, "reason": result}

    def reset_circuit(self):
        self._circuit_open = False
        self._failure_count = 0
        logger.info("Validation circuit breaker reset")

    async def validate_async(self, *, question: str, answer: str, context: str) -> dict:
        if self._circuit_open:
            logger.warning("Validation circuit open, skipping validation")
            return {"is_valid": True, "reason": "Validation skipped (circuit open)", "circuit_open": True}

        try:
            result = await asyncio.wait_for(
                self.chain.ainvoke({"question": question, "answer": answer, "context": context}),
                timeout=VALIDATION_TIMEOUT
            )
            self._failure_count = 0
            parsed = self._parse_result(result)
            logger.info(f"Validation complete | is_valid={parsed['is_valid']}")
            return parsed

        except asyncio.TimeoutError:
            self._failure_count += 1
            logger.error(f"Validation timeout | failures={self._failure_count}")
            if self._failure_count >= MAX_FAILURES_BEFORE_OPEN:
                self._circuit_open = True
                logger.error("Circuit breaker OPEN - validation disabled")
            return {"is_valid": True, "reason": "Validation timeout, answer accepted", "timeout": True}

        except Exception as e:
            self._failure_count += 1
            logger.error(f"Validation error: {e} | failures={self._failure_count}")
            if self._failure_count >= MAX_FAILURES_BEFORE_OPEN:
                self._circuit_open = True
                logger.error("Circuit breaker OPEN - validation disabled")
            return {"is_valid": True, "reason": f"Validation error: {str(e)}", "error": True}

    def validate(self, *, question: str, answer: str, context: str) -> dict:
        if self._circuit_open:
            return {"is_valid": True, "reason": "Validation skipped (circuit open)", "circuit_open": True}

        try:
            result = self.chain.invoke({"question": question, "answer": answer, "context": context}).strip()
            self._failure_count = 0
            return self._parse_result(result)
        except Exception as e:
            self._failure_count += 1
            logger.error(f"Validation error: {e}")
            if self._failure_count >= MAX_FAILURES_BEFORE_OPEN:
                self._circuit_open = True
            return {"is_valid": True, "reason": f"Validation error: {str(e)}", "error": True}
