"""
Embedding abstraction.

Wraps SentenceTransformers to provide batched text embeddings
for both document indexing and query-time retrieval.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        batch_size: int = 16,
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
