"""Unit tests for Question model validation.

These tests run without a live app, Qdrant, or Gemini — pure Pydantic validation.
Fast enough to run on every save.
"""

import pytest
from pydantic import ValidationError

from code.routes.questions import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS, Question


# ── Valid inputs ──────────────────────────────────────────────────────────────

class TestQuestionValid:
    def test_normal_question(self):
        q = Question(question="What is the master account in AWS Organizations?")
        assert q.question == "What is the master account in AWS Organizations?"

    def test_leading_trailing_whitespace_stripped(self):
        q = Question(question="  What is SCP?  ")
        assert q.question == "What is SCP?"

    def test_minimum_length(self):
        # Exactly MIN_QUESTION_CHARS characters — should pass
        q = Question(question="a" * MIN_QUESTION_CHARS)
        assert len(q.question) == MIN_QUESTION_CHARS

    def test_maximum_length(self):
        # Exactly MAX_QUESTION_CHARS characters — should pass
        q = Question(question="a" * MAX_QUESTION_CHARS)
        assert len(q.question) == MAX_QUESTION_CHARS

    def test_top_k_defaults_to_none(self):
        q = Question(question="What is SCP?")
        assert q.top_k is None

    def test_top_k_explicit(self):
        q = Question(question="What is SCP?", top_k=3)
        assert q.top_k == 3

    def test_top_k_boundary_min(self):
        q = Question(question="What is SCP?", top_k=1)
        assert q.top_k == 1

    def test_top_k_boundary_max(self):
        q = Question(question="What is SCP?", top_k=20)
        assert q.top_k == 20


# ── Invalid inputs ────────────────────────────────────────────────────────────

class TestQuestionInvalid:
    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Question(question="")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)

    def test_whitespace_only_rejected(self):
        """'   ' has length 3, passes min_length — the custom validator must catch it."""
        with pytest.raises(ValidationError) as exc_info:
            Question(question="   ")
        errors = exc_info.value.errors()
        assert any("blank" in str(e["msg"]) for e in errors)

    def test_tab_only_rejected(self):
        with pytest.raises(ValidationError):
            Question(question="\t\t\t")

    def test_newline_only_rejected(self):
        with pytest.raises(ValidationError):
            Question(question="\n\n\n")

    def test_too_short_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Question(question="ab")  # 2 chars, below MIN_QUESTION_CHARS=3
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Question(question="a" * (MAX_QUESTION_CHARS + 1))
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("question",) for e in errors)

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            Question()  # type: ignore[call-arg]

    def test_top_k_zero_rejected(self):
        with pytest.raises(ValidationError):
            Question(question="What is SCP?", top_k=0)

    def test_top_k_above_max_rejected(self):
        with pytest.raises(ValidationError):
            Question(question="What is SCP?", top_k=21)

    def test_top_k_negative_rejected(self):
        with pytest.raises(ValidationError):
            Question(question="What is SCP?", top_k=-1)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestQuestionEdgeCases:
    def test_whitespace_around_short_content_rejected(self):
        """'  a  ' strips to 'a' (1 char) which is below MIN_QUESTION_CHARS."""
        with pytest.raises(ValidationError):
            Question(question="  a  ")

    def test_whitespace_around_valid_content_passes(self):
        """'  abc  ' strips to 'abc' (3 chars) — exactly at the boundary."""
        q = Question(question="  abc  ")
        assert q.question == "abc"
        assert len(q.question) == MIN_QUESTION_CHARS

    def test_unicode_question(self):
        q = Question(question="What is AWS Organizations SCP?")
        assert q.question == "What is AWS Organizations SCP?"

    def test_question_with_special_chars(self):
        q = Question(question="What's the max # of accounts per OU?")
        assert "max" in q.question