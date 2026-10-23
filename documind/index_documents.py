"""Offline indexing pipeline: chunks text, generates embeddings, and stores them in Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from documind.config import settings
from documind.embeddings import EmbeddingModel
from documind.ingest import load_pdf_documents


def main():
    print("Starting document indexing...")
    print(f"  Collection : {settings.qdrant_collection}")
    print(f"  Qdrant     : {settings.qdrant_url or f'{settings.qdrant_host}:{settings.qdrant_port}'}")
    print(f"  Embed model: {settings.embedding_model}")

    chunks = load_pdf_documents()
    if not chunks:
        raise RuntimeError(
            "No text chunks were extracted from documents. "
            "Please check that PDF files exist in the docs/ directory and contain selectable text."
        )

    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks")

    embedder = EmbeddingModel()
    embeddings = embedder.embed(texts)
    if embeddings.size == 0 or len(embeddings.shape) != 2:
        raise RuntimeError("Generated embeddings are empty or malformed.")

    embedding_dim = embeddings.shape[1]
    print(f"Generated embeddings with shape {embeddings.shape}")

    client = QdrantClient(**settings.qdrant_connection)

    try:
        exists = client.collection_exists(settings.qdrant_collection)
    except Exception as e:
        target = settings.qdrant_url or f"{settings.qdrant_host}:{settings.qdrant_port}"
        raise RuntimeError(
            f"Failed to connect to Qdrant at {target}: {e}\n"
            "Please check:\n"
            "  1. Is QDRANT_URL correct? (e.g., https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333)\n"
            "  2. Is QDRANT_API_KEY set and valid for Qdrant Cloud?\n"
            "  3. If running locally, is Docker or the Qdrant service started?"
        ) from e

    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{settings.qdrant_collection}' (dimension={embedding_dim})")
    else:
        # Validate existing collection vector dimension matches embedding dimension
        collection_info = client.get_collection(settings.qdrant_collection)
        existing_dim = None
        params = collection_info.config.params.vectors
        if hasattr(params, "size"):
            existing_dim = params.size
        elif isinstance(params, dict) and "" in params:
            existing_dim = params[""].size

        if existing_dim and existing_dim != embedding_dim:
            raise RuntimeError(
                f"Collection vector dimension mismatch! Existing collection has size {existing_dim}, "
                f"but embedding model '{settings.embedding_model}' generates {embedding_dim}. "
                "Recreate the collection or change embedding model."
            )

    points = []
    for emb, chunk in zip(embeddings, chunks, strict=True):
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

    # Atomic re-index: delete stale points AFTER upserting all new points
    if exists:
        source_files = {c["metadata"]["file"] for c in chunks if "metadata" in c and "file" in c["metadata"]}
        new_ids = [p.id for p in points]
        for src_file in source_files:
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file",
                            match=MatchValue(value=src_file),
                        )
                    ],
                    must_not=[
                        HasIdCondition(has_id=new_ids)
                    ],
                ),
            )
            print(f"Pruned stale points for file '{src_file}'")

    print(f"Indexed {len(points)} chunks into Qdrant")


if __name__ == "__main__":
    main()
