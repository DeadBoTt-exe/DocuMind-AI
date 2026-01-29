"""Sentence-boundary aware chunking with deterministic IDs for stable document identity."""

import hashlib
import uuid
from typing import Dict, List

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
nlp.add_pipe("sentencizer")

NAMESPACE_DOCUMIND = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_chunk_id(file: str, page: int, offset: int) -> str:
    content = f"{file}:{page}:{offset}"
    return str(uuid.uuid5(NAMESPACE_DOCUMIND, content))


def chunk_text(
    text: str,
    *,
    source_file: str,
    page: int,
    service: str,
    max_chars: int = 2000,
    overlap_sentences: int = 2,
) -> List[Dict]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    offset = 0
    
    for sent in sentences:
        if current_length + len(sent) > max_chars and current_chunk:
            chunk_text_content = " ".join(current_chunk)
            
            if len(chunk_text_content) >= 100:
                chunks.append({
                    "id": generate_chunk_id(source_file, page, offset),
                    "text": chunk_text_content,
                    "metadata": {"file": source_file, "page": page, "service": service}
                })
                offset += 1
            
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sent)
        current_length += len(sent)
    
    if current_chunk:
        chunk_text_content = " ".join(current_chunk)
        if len(chunk_text_content) >= 100:
            chunks.append({
                "id": generate_chunk_id(source_file, page, offset),
                "text": chunk_text_content,
                "metadata": {"file": source_file, "page": page, "service": service}
            })
    
    return chunks
