"""Embedding abstraction with L2 normalization for consistent similarity scoring."""

import asyncio
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 16):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    async def embed_async(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        return await asyncio.to_thread(self.embed, texts, normalize)
