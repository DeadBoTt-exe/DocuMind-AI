"""Document ingestion pipeline with multi-source support."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from code.chunker import chunk_text
from code.cleaner import clean_text


@dataclass
class DocumentSource:
    path: Path
    service: str


DEFAULT_SOURCES = [
    DocumentSource(Path("docs/organizations-userguide.pdf"), "aws-organizations"),
]


def extract_text_fast(page) -> str:
    return " ".join(
        block[4] for block in page.get_text("blocks")
        if block[6] == 0 and block[4].strip()
    )


def load_pdf_document(pdf_path: Path, service_name: str) -> List[Dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_chunks: List[Dict] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            raw_text = extract_text_fast(page)
            cleaned = clean_text(raw_text)

            if len(cleaned) < 50:
                continue

            chunks = chunk_text(
                text=cleaned,
                source_file=pdf_path.name,
                page=page_index,
                service=service_name,
            )
            all_chunks.extend(chunks)

    return all_chunks


def load_pdf_documents(sources: Optional[List[DocumentSource]] = None) -> List[Dict]:
    sources = sources or DEFAULT_SOURCES
    all_chunks: List[Dict] = []
    
    for source in sources:
        print(f"Processing: {source.path}")
        chunks = load_pdf_document(source.path, source.service)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")
    
    return all_chunks


if __name__ == "__main__":
    chunks = load_pdf_documents()
    print(f"Total: {len(chunks)} chunks from {len(DEFAULT_SOURCES)} source(s)")
