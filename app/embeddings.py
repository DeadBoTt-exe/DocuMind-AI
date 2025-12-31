"""
This is the embeddings module of the pipeline.

It converts chunks of text into numerical vector representations that can be
indexed and searched.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
