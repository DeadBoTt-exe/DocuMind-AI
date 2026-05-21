"""Embedding abstraction with L2 normalization for consistent similarity scoring."""

import asyncio
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from code.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.batch_size = settings.embedding_batch_size
        self._lock = asyncio.Lock()  # Phase 3 concurrency fix — included now, costs nothing
        logger.info("Embedding model loaded")

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,  # Suppress in production; noisy in Cloud Logging
        )
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    async def embed_async(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        async with self._lock:
            return await asyncio.to_thread(self.embed, texts, normalize)