"""Unit tests for code.cleaner.

Tests every regex pattern in clean_text independently, then combined.
No imports from config or external services needed.
"""

import pytest

from code.cleaner import clean_text


class TestCleanTextHeaderStripping:
    def test_strips_aws_organizations_user_guide(self):
        text = "AWS Organizations User Guide\nSome content here."
        result = clean_text(text)
        assert "AWS Organizations User Guide" not in result
        assert "Some content here" in result

    def test_strips_aws_header_case_insensitive(self):
        text = "aws organizations user guide\nContent."
        result = clean_text(text)
        assert "aws organizations user guide" not in result

    def test_strips_amazon_copyright(self):
        text = "© Amazon Web Services\nContent."
        result = clean_text(text)
        assert "Amazon Web Services" not in result
        assert "Content" in result

    def test_strips_copyright_with_spaces(self):
        text = "©  Amazon Web Services\nContent."
        result = clean_text(text)
        assert "Amazon Web Services" not in result

    def test_multiple_headers_in_same_text(self):
        text = (
            "AWS Organizations User Guide\n"
            "Some content.\n"
            "© Amazon Web Services\n"
            "More content."
        )
        result = clean_text(text)
        assert "AWS Organizations User Guide" not in result
        assert "Amazon Web Services" not in result
        assert "Some content" in result
        assert "More content" in result


class TestCleanTextNewlines:
    def test_three_newlines_collapsed_to_two(self):
        text = "Paragraph one.\n\n\nParagraph two."
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert "\n\n" in result

    def test_four_newlines_collapsed_to_two(self):
        text = "A.\n\n\n\nB."
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_two_newlines_left_alone(self):
        text = "A.\n\nB."
        result = clean_text(text)
        assert "\n\n" in result

    def test_single_newline_left_alone(self):
        text = "Line one.\nLine two."
        result = clean_text(text)
        assert "\n" in result


class TestCleanTextPageNumbers:
    def test_lone_page_number_removed(self):
        text = "Some text.\n42\nMore text."
        result = clean_text(text)
        assert "\n42\n" not in result

    def test_page_number_at_start_removed(self):
        text = "\n5\nContent here."
        result = clean_text(text)
        assert "\n5\n" not in result

    def test_multidigit_page_number_removed(self):
        text = "Section header.\n123\nNext paragraph."
        result = clean_text(text)
        assert "\n123\n" not in result

    def test_number_in_sentence_not_removed(self):
        # "There are 42 accounts" — the 42 is NOT surrounded by newlines
        text = "There are 42 accounts in the organization."
        result = clean_text(text)
        assert "42" in result


class TestCleanTextWhitespace:
    def test_multiple_spaces_collapsed(self):
        text = "Word    with    extra    spaces."
        result = clean_text(text)
        assert "    " not in result
        assert "Word with extra spaces" in result

    def test_tabs_collapsed_to_space(self):
        text = "Column one\t\tColumn two."
        result = clean_text(text)
        assert "\t" not in result

    def test_leading_whitespace_stripped(self):
        text = "   Leading spaces."
        result = clean_text(text)
        assert not result.startswith(" ")

    def test_trailing_whitespace_stripped(self):
        text = "Trailing spaces.   "
        result = clean_text(text)
        assert not result.endswith(" ")


class TestCleanTextCombined:
    def test_realistic_pdf_page_output(self):
        """Simulates what PyMuPDF extracts from a typical AWS doc page."""
        raw = (
            "AWS Organizations User Guide\n"
            "\n"
            "\n"
            "Managing  accounts  in  your  organization\n"
            "\n"
            "47\n"
            "\n"
            "You can invite existing AWS accounts to join your organization.\n"
            "© Amazon Web Services\n"
        )
        result = clean_text(raw)

        assert "AWS Organizations User Guide" not in result
        assert "Amazon Web Services" not in result
        assert "\n\n\n" not in result
        assert "\n47\n" not in result
        assert "  " not in result
        assert "Managing accounts in your organization" in result
        assert "You can invite existing AWS accounts" in result

    def test_clean_text_is_idempotent(self):
        """Cleaning already-clean text should not change it further."""
        raw = "This is a clean sentence about AWS.\n\nAnother paragraph."
        once = clean_text(raw)
        twice = clean_text(once)
        assert once == twice