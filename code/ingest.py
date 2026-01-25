"""
Document ingestion pipeline.

Loads source documents (PDFs), extracts and cleans text,
and prepares structured chunks for offline indexing.
"""

from pathlib import Path
from typing import List, Dict
import fitz
from code.chunker import chunk_text


PDF_PATH = Path("docs/organizations-userguide.pdf")
SERVICE_NAME = "aws-organizations"
DEBUG = False



def clean_text(text: str) -> str:
    text = text.replace("AWS Organizations User Guide", "")
    text = text.replace("© Amazon Web Services", "")
    text = text.replace("\r", "")

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def extract_text_fast(page) -> str:
    return " ".join(
        block[4] for block in page.get_text("blocks")
        if block[6] == 0 and block[4].strip() 
    )


def load_pdf_documents() -> List[Dict]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    all_chunks: List[Dict] = []

    with fitz.open(PDF_PATH) as doc:
        for page_index, page in enumerate(doc, start=1):
            if DEBUG:
                print(f"Processing page {page_index}")

            raw_text = extract_text_fast(page)
            cleaned = clean_text(raw_text)

            if len(cleaned) < 50:
                continue

            chunks = chunk_text(
                text=cleaned,
                source_file=PDF_PATH.name,
                page=page_index,
                service=SERVICE_NAME,
            )

            all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_pdf_documents()
    print(f"Loaded {len(chunks)} chunks from AWS Organizations PDF")
