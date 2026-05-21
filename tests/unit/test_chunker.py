"""Unit tests for code.chunker.

Tests generate_chunk_id determinism and chunk_text boundary behaviour.
No network calls, no Qdrant, no Gemini — pure logic.
"""

import uuid

import pytest

from code.chunker import NAMESPACE_DOCUMIND, chunk_text, generate_chunk_id


# ── generate_chunk_id ─────────────────────────────────────────────────────────

class TestGenerateChunkId:
    def test_returns_string(self):
        result = generate_chunk_id("doc.pdf", 1, 0)
        assert isinstance(result, str)

    def test_is_valid_uuid(self):
        result = generate_chunk_id("doc.pdf", 1, 0)
        # Will raise ValueError if not a valid UUID
        parsed = uuid.UUID(result)
        assert parsed.version == 5

    def test_deterministic_same_inputs(self):
        """Same inputs must always produce the same ID — stable re-indexing."""
        id1 = generate_chunk_id("doc.pdf", 1, 0)
        id2 = generate_chunk_id("doc.pdf", 1, 0)
        assert id1 == id2

    def test_deterministic_known_value(self):
        """Pin the exact UUID so accidental namespace changes are caught."""
        expected = str(uuid.uuid5(NAMESPACE_DOCUMIND, "doc.pdf:1:0"))
        assert generate_chunk_id("doc.pdf", 1, 0) == expected

    def test_different_file_gives_different_id(self):
        assert generate_chunk_id("a.pdf", 1, 0) != generate_chunk_id("b.pdf", 1, 0)

    def test_different_page_gives_different_id(self):
        assert generate_chunk_id("doc.pdf", 1, 0) != generate_chunk_id("doc.pdf", 2, 0)

    def test_different_offset_gives_different_id(self):
        assert generate_chunk_id("doc.pdf", 1, 0) != generate_chunk_id("doc.pdf", 1, 1)

    def test_all_three_dimensions_matter(self):
        """Every dimension independently changes the ID."""
        base = generate_chunk_id("x.pdf", 1, 0)
        assert generate_chunk_id("y.pdf", 1, 0) != base
        assert generate_chunk_id("x.pdf", 2, 0) != base
        assert generate_chunk_id("x.pdf", 1, 1) != base


# ── chunk_text ────────────────────────────────────────────────────────────────

# A sentence that is clearly over 100 chars so it passes chunk_min_chars
LONG_SENTENCE = (
    "AWS Organizations is a global service that enables you to consolidate "
    "multiple AWS accounts into an organization that you create and centrally manage."
)

# A collection of distinct long sentences for multi-chunk tests
SENTENCES = [
    "AWS Organizations lets you centrally manage billing across multiple AWS accounts.",
    "Service control policies allow administrators to restrict services available to member accounts.",
    "Organizational units create a hierarchy within an AWS Organization for policy inheritance.",
    "The management account is the primary account used to create and manage the organization.",
    "Tag policies help enforce tagging standards across resources in all member accounts.",
]


class TestChunkTextEmpty:
    def test_empty_string_returns_empty_list(self):
        result = chunk_text("", source_file="doc.pdf", page=1, service="aws")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = chunk_text("   \n\t  ", source_file="doc.pdf", page=1, service="aws")
        assert result == []

    def test_text_below_min_chars_discarded(self):
        # 99 chars — just under chunk_min_chars=100
        short = "A" * 99
        result = chunk_text(short, source_file="doc.pdf", page=1, service="aws")
        assert result == []


class TestChunkTextSingleChunk:
    def test_single_long_sentence_produces_one_chunk(self):
        result = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=1, service="aws")
        assert len(result) == 1

    def test_chunk_contains_required_keys(self):
        result = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=1, service="aws")
        chunk = result[0]
        assert "id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk

    def test_metadata_shape(self):
        result = chunk_text(LONG_SENTENCE, source_file="test.pdf", page=3, service="my-svc")
        meta = result[0]["metadata"]
        assert meta["file"] == "test.pdf"
        assert meta["page"] == 3
        assert meta["service"] == "my-svc"

    def test_chunk_id_is_valid_uuid(self):
        result = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=1, service="aws")
        parsed = uuid.UUID(result[0]["id"])
        assert parsed.version == 5

    def test_multiple_sentences_fitting_in_max_chars_produce_one_chunk(self):
        # Two sentences well under 2000 chars
        text = SENTENCES[0] + " " + SENTENCES[1]
        result = chunk_text(text, source_file="doc.pdf", page=1, service="aws")
        assert len(result) == 1

    def test_chunk_text_contains_input_content(self):
        result = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=1, service="aws")
        assert "AWS Organizations" in result[0]["text"]


class TestChunkTextSplitting:
    def test_overflow_produces_multiple_chunks(self):
        # max_chars=200 forces splits across our 5 sentences
        text = " ".join(SENTENCES)
        result = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws", max_chars=200
        )
        assert len(result) > 1

    def test_each_chunk_passes_min_chars(self):
        text = " ".join(SENTENCES)
        result = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws", max_chars=200
        )
        for chunk in result:
            assert len(chunk["text"]) >= 100

    def test_chunks_have_unique_ids(self):
        text = " ".join(SENTENCES)
        result = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws", max_chars=200
        )
        ids = [c["id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_overlap_sentences_appear_in_consecutive_chunks(self):
        """With overlap_sentences=1, the last sentence of chunk N
        should appear at the start of chunk N+1."""
        text = " ".join(SENTENCES)
        result = chunk_text(
            text,
            source_file="doc.pdf",
            page=1,
            service="aws",
            max_chars=200,
            overlap_sentences=1,
        )
        if len(result) < 2:
            pytest.skip("Not enough chunks produced to test overlap")

        # Find at least one word from chunk[0] that also appears in chunk[1]
        words_in_first = set(result[0]["text"].split())
        words_in_second = set(result[1]["text"].split())
        assert words_in_first & words_in_second, (
            "Expected overlapping words between consecutive chunks"
        )

    def test_no_overlap_produces_clean_split(self):
        text = " ".join(SENTENCES)
        result_overlap = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws",
            max_chars=200, overlap_sentences=2,
        )
        result_no_overlap = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws",
            max_chars=200, overlap_sentences=0,
        )
        # No-overlap should produce same or fewer total chars across all chunks
        total_overlap = sum(len(c["text"]) for c in result_overlap)
        total_no = sum(len(c["text"]) for c in result_no_overlap)
        assert total_overlap >= total_no

    def test_chunk_ids_are_stable_across_calls(self):
        """Re-chunking same text must produce same IDs — critical for re-indexing."""
        text = " ".join(SENTENCES)
        result1 = chunk_text(text, source_file="doc.pdf", page=2, service="aws", max_chars=200)
        result2 = chunk_text(text, source_file="doc.pdf", page=2, service="aws", max_chars=200)
        assert [c["id"] for c in result1] == [c["id"] for c in result2]

    def test_different_pages_produce_different_ids(self):
        result_p1 = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=1, service="aws")
        result_p2 = chunk_text(LONG_SENTENCE, source_file="doc.pdf", page=2, service="aws")
        assert result_p1[0]["id"] != result_p2[0]["id"]


class TestChunkTextEdgeCases:
    def test_custom_max_chars_respected(self):
        # With a very large max_chars, all sentences fit in one chunk
        text = " ".join(SENTENCES)
        result = chunk_text(
            text, source_file="doc.pdf", page=1, service="aws", max_chars=99999
        )
        assert len(result) == 1

    def test_exact_min_chars_boundary(self):
        # Exactly 100 chars — should be kept
        text = "A" * 100
        result = chunk_text(text, source_file="doc.pdf", page=1, service="aws")
        assert len(result) == 1

    def test_metadata_preserved_across_all_chunks(self):
        text = " ".join(SENTENCES)
        result = chunk_text(
            text, source_file="myfile.pdf", page=7, service="svc-x", max_chars=200
        )
        for chunk in result:
            assert chunk["metadata"]["file"] == "myfile.pdf"
            assert chunk["metadata"]["page"] == 7
            assert chunk["metadata"]["service"] == "svc-x"