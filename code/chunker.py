"""Sentence-boundary aware chunking with deterministic IDs for stable document identity."""

import uuid
from typing import Dict, List

import spacy

from code.config import settings

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
nlp.add_pipe("sentencizer")

NAMESPACE_DOCUMIND = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_chunk_id(file: str, page: int, offset: int) -> str:
    content = f"{file}:{page}:{offset}"
    return str(uuid.uuid5(NAMESPACE_DOCUMIND, content))


def _joined_len(parts: List[str]) -> int:
    """Length of " ".join(parts) without building the string."""
    return sum(map(len, parts)) + max(len(parts) - 1, 0)


def _split_oversized(sentence: str, max_chars: int) -> List[str]:
    """Split one over-long sentence on whitespace into pieces of at most max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts: List[str] = []
    buf: List[str] = []
    n = 0
    for word in sentence.split():
        while len(word) > max_chars:  # pathological unbroken token: hard-slice it
            if buf:
                parts.append(" ".join(buf))
                buf, n = [], 0
            parts.append(word[:max_chars])
            word = word[max_chars:]
        add = len(word) + (1 if buf else 0)
        if buf and n + add > max_chars:
            parts.append(" ".join(buf))
            buf, n, add = [], 0, len(word)
        buf.append(word)
        n += add
    if buf:
        parts.append(" ".join(buf))
    return parts


def chunk_text(
    text: str,
    *,
    source_file: str,
    page: int,
    service: str,
    max_chars: int | None = None,
    overlap_sentences: int | None = None,
) -> List[Dict]:
    max_chars = max_chars if max_chars is not None else settings.chunk_max_chars
    overlap_sentences = overlap_sentences if overlap_sentences is not None else settings.chunk_overlap_sentences

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    # A single "sentence" can be huge (diagram labels, tables, JSON with no sentence
    # punctuation). Split those first so no chunk can exceed max_chars.
    sentences = [part for sent in sentences for part in _split_oversized(sent, max_chars)]

    chunks = []
    current_chunk: List[str] = []
    offset = 0

    for sent in sentences:
        if current_chunk and _joined_len(current_chunk) + 1 + len(sent) > max_chars:
            chunk_text_content = " ".join(current_chunk)

            if len(chunk_text_content) >= settings.chunk_min_chars:
                chunks.append({
                    "id": generate_chunk_id(source_file, page, offset),
                    "text": chunk_text_content,
                    "metadata": {"file": source_file, "page": page, "service": service},
                })
                offset += 1

            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            # Overlap must never push the next chunk over the limit.
            while current_chunk and _joined_len(current_chunk) + 1 + len(sent) > max_chars:
                current_chunk.pop(0)

        current_chunk.append(sent)

    if current_chunk:
        chunk_text_content = " ".join(current_chunk)
        if len(chunk_text_content) >= settings.chunk_min_chars:
            chunks.append({
                "id": generate_chunk_id(source_file, page, offset),
                "text": chunk_text_content,
                "metadata": {"file": source_file, "page": page, "service": service},
            })

    return chunks