"""Offline indexing pipeline: chunks text, generates embeddings, and stores them in Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from code.embeddings import EmbeddingModel
from code.ingest import load_pdf_documents

COLLECTION_NAME = "aws-org-docs"
UPSERT_BATCH_SIZE = 64


def main():
    print("Starting document indexing...")

    chunks = load_pdf_documents()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks")

    embedder = EmbeddingModel()
    embeddings = embedder.embed(texts)
    print(f"Generated embeddings with shape {embeddings.shape}")

    client = QdrantClient(host="localhost", port=6333)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    points = []
    for emb, chunk in zip(embeddings, chunks):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=emb.tolist(),
                payload={**chunk["metadata"], "text": chunk["text"]},
            )
        )

    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Upserted {i + len(batch)} / {len(points)} points")

    print(f"Indexed {len(points)} chunks into Qdrant")


if __name__ == "__main__":
    main()
