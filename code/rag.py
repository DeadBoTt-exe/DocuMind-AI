"""Core RAG engine: embedding, retrieval, generation, validation, and confidence scoring."""

import asyncio
import logging
import os
from typing import Dict, List

import google.genai as genai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, QdrantClient

from code.confidence import ConfidenceScorer
from code.embeddings import EmbeddingModel
from code.logging_config import log_latency
from code.prompts import (
    DEFAULT_NO_ANSWER,
    LLM_ERROR_ANSWER,
    RAG_ANSWER_PROMPT,
    VALIDATION_FAILED_ANSWER,
)
from code.validator import GroundingValidator

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "aws-org-docs"


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-2.5-flash"

        self.embedder = EmbeddingModel()
        self.validator = GroundingValidator()
        self.confidence_scorer = ConfidenceScorer()

        self._sync_qdrant = QdrantClient(host="localhost", port=6333)
        self.qdrant = AsyncQdrantClient(host="localhost", port=6333)

        collections = [c.name for c in self._sync_qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' not found. Run index_documents.py first."
            )
        
        logger.info("RAGEngine initialized successfully")

    async def close(self):
        await self.qdrant.close()
        self._sync_qdrant.close()
        logger.info("RAGEngine resources closed")

    @log_latency("rag.ask_async")
    async def ask_async(self, question: str, top_k: int = 5) -> Dict:
        logger.info(f"Query received | question_length={len(question)}")
        
        query_embedding = await asyncio.to_thread(self.embedder.embed, [question])

        search_result = await self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding[0].tolist(),
            limit=top_k,
        )
        results = search_result.points

        if not results:
            logger.warning("No results found for query")
            return {
                "answer": DEFAULT_NO_ANSWER,
                "sources": [],
                "validation": {"is_valid": False, "reason": "No relevant context retrieved."},
                "confidence": 0.0,
            }

        context_chunks: List[str] = []
        sources = set()
        retrieval_scores: List[float] = []

        for r in results:
            payload = r.payload or {}
            text = payload.get("text")
            file = payload.get("file")
            page = payload.get("page")

            if not text or not file or page is None:
                logger.warning(f"Skipping malformed chunk: {payload}")
                continue

            context_chunks.append(text)
            sources.add(f"{file}#page-{page}")
            retrieval_scores.append(r.score)

        if not context_chunks:
            logger.warning("All retrieved chunks were malformed")
            return {
                "answer": DEFAULT_NO_ANSWER,
                "sources": [],
                "validation": {"is_valid": False, "reason": "Retrieved chunks were empty or malformed."},
                "confidence": 0.0,
            }

        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        logger.info(f"Retrieval complete | chunks={len(context_chunks)} | avg_score={avg_score:.3f}")

        context = "\n\n".join(context_chunks)
        prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=prompt,
            )
            answer = response.text.strip()
            logger.info(f"LLM response received | answer_length={len(answer)}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": LLM_ERROR_ANSWER,
                "sources": sorted(sources),
                "validation": {"is_valid": False, "reason": f"LLM error: {str(e)}"},
                "confidence": 0.0,
            }

        try:
            validation = await self.validator.validate_async(
                question=question, answer=answer, context=context
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation = {"is_valid": False, "reason": f"Validation error: {str(e)}"}

        confidence = self.confidence_scorer.score(
            retrieval_scores=retrieval_scores,
            num_chunks=len(context_chunks),
            is_valid=validation["is_valid"],
        )

        logger.info(f"Query complete | is_valid={validation['is_valid']} | confidence={confidence}")

        if not validation["is_valid"]:
            return {
                "answer": VALIDATION_FAILED_ANSWER,
                "sources": sorted(sources),
                "validation": validation,
                "confidence": confidence,
            }

        return {
            "answer": answer,
            "sources": sorted(sources),
            "validation": validation,
            "confidence": confidence,
        }

    @log_latency("rag.ask")
    def ask(self, question: str, top_k: int = 5) -> Dict:
        logger.info(f"Sync query received | question_length={len(question)}")
        
        query_embedding = self.embedder.embed([question])
        
        search_result = self._sync_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding[0].tolist(),
            limit=top_k,
        )
        results = search_result.points

        if not results:
            return {
                "answer": DEFAULT_NO_ANSWER,
                "sources": [],
                "validation": {"is_valid": False, "reason": "No relevant context retrieved."},
                "confidence": 0.0,
            }

        context_chunks: List[str] = []
        sources = set()
        retrieval_scores: List[float] = []

        for r in results:
            payload = r.payload or {}
            text = payload.get("text")
            file = payload.get("file")
            page = payload.get("page")

            if not text or not file or page is None:
                logger.warning(f"Skipping malformed chunk: {payload}")
                continue

            context_chunks.append(text)
            sources.add(f"{file}#page-{page}")
            retrieval_scores.append(r.score)

        if not context_chunks:
            return {
                "answer": DEFAULT_NO_ANSWER,
                "sources": [],
                "validation": {"is_valid": False, "reason": "Retrieved chunks were empty or malformed."},
                "confidence": 0.0,
            }

        context = "\n\n".join(context_chunks)
        prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)

        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": LLM_ERROR_ANSWER,
                "sources": sorted(sources),
                "validation": {"is_valid": False, "reason": f"LLM error: {str(e)}"},
                "confidence": 0.0,
            }

        try:
            validation = self.validator.validate(question=question, answer=answer, context=context)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation = {"is_valid": False, "reason": f"Validation error: {str(e)}"}

        confidence = self.confidence_scorer.score(
            retrieval_scores=retrieval_scores,
            num_chunks=len(context_chunks),
            is_valid=validation["is_valid"],
        )

        if not validation["is_valid"]:
            return {
                "answer": VALIDATION_FAILED_ANSWER,
                "sources": sorted(sources),
                "validation": validation,
                "confidence": confidence,
            }

        return {
            "answer": answer,
            "sources": sorted(sources),
            "validation": validation,
            "confidence": confidence,
        }
