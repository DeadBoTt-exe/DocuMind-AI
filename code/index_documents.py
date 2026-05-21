"""Offline indexing pipeline: chunks text, generates embeddings, and stores them in Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from code.config import settings
from code.embeddings import EmbeddingModel
from code.ingest import load_pdf_documents


def main():
    print("Starting document indexing...")
    print(f"  Collection : {settings.qdrant_collection}")
    print(f"  Qdrant     : {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Embed model: {settings.embedding_model}")

    chunks = load_pdf_documents()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks")

    embedder = EmbeddingModel()
    embeddings = embedder.embed(texts)
    print(f"Generated embeddings with shape {embeddings.shape}")

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )
        print(f"Created collection '{settings.qdrant_collection}'")

    points = []
    for emb, chunk in zip(embeddings, chunks):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=emb.tolist(),
                payload={**chunk["metadata"], "text": chunk["text"]},
            )
        )

    for i in range(0, len(points), settings.index_upsert_batch_size):
        batch = points[i : i + settings.index_upsert_batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        print(f"Upserted {i + len(batch)} / {len(points)} points")

    print(f"Indexed {len(points)} chunks into Qdrant")


if __name__ == "__main__":
    main()