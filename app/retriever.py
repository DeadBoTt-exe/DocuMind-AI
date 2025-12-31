"""
This is the retriever module.

This implements vector-based search using FAISS it also stores embeddings and
their associated metadata and retrieves the most relevant chunks for any
given query.
"""

import faiss
import numpy as np
from typing import List, Dict


class FaissRetriever:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results



if __name__ == "__main__":
    from app.ingest import load_markdown_files
    from app.embeddings import EmbeddingModel

    chunks = load_markdown_files()
    texts = [c["text"] for c in chunks]

    embedder = EmbeddingModel()
    embeddings = embedder.embed(texts)

    retriever = FaissRetriever(embedding_dim=embeddings.shape[1])
    retriever.add(embeddings, chunks)


    query = "Where is GST calculated?"
    query_embedding = embedder.embed([query])

    results = retriever.search(query_embedding, top_k=2)

    print("\n QUERY:", query)
    for r in results:
        print("\n--- RESULT ---")
        print("FILE:", r["metadata"]["file"])
        print("SECTION:", r["metadata"]["section"])
        print("SERVICE:", r["metadata"]["service"])
        print("TEXT:", r["text"][:200])
