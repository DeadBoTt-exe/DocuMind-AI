"""
Text chunking utilities.

Splits the cleaned document text into semantically meaningful
chunks with associated metadata
"""

from typing import List, Dict
import uuid


def chunk_text(
    text: str,
    *,
    source_file: str,
    page: int,
    service: str,
    max_chars: int = 2000, 
    overlap: int = 200,    

) -> List[Dict]:
    chunks: List[Dict] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()

        if len(chunk) < 100:
            break

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "metadata": {
                "file": source_file,
                "page": page,
                "service": service,
            }
        })

        start = max(end - overlap, start + 1)

    return chunks
