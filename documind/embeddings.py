"""Embedding abstraction with L2 normalization for consistent similarity scoring."""

import asyncio
import concurrent.futures
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from documind.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.batch_size = settings.embedding_batch_size
        self._lock = asyncio.Lock()  # Phase 3 concurrency fix — included now, costs nothing
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="embedding"
        )
        logger.info("Embedding model loaded")

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,  # Suppress in production; noisy in Cloud Logging
        )
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    async def embed_async(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.embed, texts, normalize),
                timeout=settings.generation_timeout_seconds,
            )
