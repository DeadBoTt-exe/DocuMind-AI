"""
This is the chunker module of the pipeline which is responsible for splitting raw documentation
into semantic meaningful chunks suitable for embedding and retrieval.

We performed chunking based on headings and paragraph size
and attached useful metadata such as source file, section name, and inferred service.
"""

import re
from typing import List, Dict


def infer_service_from_filename(filename: str) -> str:
    name = filename.lower()
    if "billing" in name:
        return "billing-service"
    if "payment" in name:
        return "payments-service"
    if "architecture" in name:
        return "system"
    return "unknown"


def chunk_markdown(text: str, filename: str) -> List[Dict]:
    service = infer_service_from_filename(filename)

    sections = re.split(r"(?:^|\n)##\s+", text)
    chunks = []

    for section in sections:
        lines = section.strip().split("\n", 1)

        if len(lines) == 1:
            section_title = "Overview"
            body = lines[0]
        else:
            section_title = lines[0].strip()
            body = lines[1].strip()

        if len(body) < 50:
            continue

        chunks.append({
            "text": body,
            "metadata": {
                "file": filename,
                "section": section_title,
                "service": service
            }
        })

    return chunks
