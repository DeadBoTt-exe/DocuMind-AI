"""Unit tests for code.confidence.ConfidenceScorer.

Tests every branch of the scoring formula including all boundary conditions.
No mocking needed — pure math.
"""

import pytest

from code.confidence import ConfidenceScorer


@pytest.fixture
def scorer():
    return ConfidenceScorer()


# ── Validation gate ───────────────────────────────────────────────────────────

class TestValidationGate:
    def test_invalid_returns_zero_regardless_of_scores(self, scorer):
        result = scorer.score(
            retrieval_scores=[0.95, 0.90, 0.85],
            num_chunks=5,
            is_valid=False,
        )
        assert result == 0.0

    def test_invalid_with_perfect_retrieval_still_zero(self, scorer):
        result = scorer.score(
            retrieval_scores=[1.0],
            num_chunks=5,
            is_valid=False,
        )
        assert result == 0.0

    def test_invalid_with_zero_scores_still_zero(self, scorer):
        result = scorer.score(
            retrieval_scores=[0.0],
            num_chunks=1,
            is_valid=False,
        )
        assert result == 0.0


# ── Return type and precision ─────────────────────────────────────────────────

class TestReturnType:
    def test_returns_float(self, scorer):
        result = scorer.score(
            retrieval_scores=[0.7],
            num_chunks=3,
            is_valid=True,
        )
        assert isinstance(result, float)

    def test_result_rounded_to_two_decimal_places(self, scorer):
        result = scorer.score(
            retrieval_scores=[0.55],
            num_chunks=3,
            is_valid=True,
        )
        # Rounding to 2dp means at most 2 digits after the decimal
        assert result == round(result, 2)

    def test_result_between_zero_and_one(self, scorer):
        for score in [0.1, 0.5, 0.9]:
            result = scorer.score(
                retrieval_scores=[score],
                num_chunks=3,
                is_valid=True,
            )
            assert 0.0 <= result <= 1.0


# ── Retrieval confidence boundaries ──────────────────────────────────────────
# Formula: retrieval_confidence = min(max((avg - 0.2) / 0.7, 0), 1)

class TestRetrievalBoundaries:
    def test_avg_score_at_floor_gives_zero_retrieval(self, scorer):
        """avg=0.2 → (0.2-0.2)/0.7 = 0 → retrieval_confidence=0."""
        result = scorer.score(
            retrieval_scores=[0.2],
            num_chunks=5,   # max coverage so coverage=1, isolates retrieval
            is_valid=True,
        )
        # 0.7 * 0 + 0.3 * 1 = 0.3
        assert result == pytest.approx(0.3, abs=0.01)

    def test_avg_score_below_floor_clamped_to_zero(self, scorer):
        """avg=0.1 → clamped to 0 retrieval_confidence."""
        result = scorer.score(
            retrieval_scores=[0.1],
            num_chunks=5,
            is_valid=True,
        )
        assert result == pytest.approx(0.3, abs=0.01)

    def test_avg_score_at_ceiling_gives_full_retrieval(self, scorer):
        """avg=0.9 → (0.9-0.2)/0.7 = 1 → retrieval_confidence=1."""
        result = scorer.score(
            retrieval_scores=[0.9],
            num_chunks=5,
            is_valid=True,
        )
        # 0.7 * 1 + 0.3 * 1 = 1.0
        assert result == pytest.approx(1.0, abs=0.01)

    def test_avg_score_above_ceiling_clamped_to_one(self, scorer):
        """avg=1.0 → clamped to 1 retrieval_confidence."""
        result = scorer.score(
            retrieval_scores=[1.0],
            num_chunks=5,
            is_valid=True,
        )
        assert result == pytest.approx(1.0, abs=0.01)

    def test_avg_computed_across_multiple_scores(self, scorer):
        """Average of [0.6, 0.8] = 0.7 → retrieval=(0.7-0.2)/0.7 ≈ 0.714."""
        retrieval_conf = (0.7 - 0.2) / 0.7          # ≈ 0.714
        coverage_conf = min(5 / 5.0, 1.0)            # 1.0
        expected = round(0.7 * retrieval_conf + 0.3 * coverage_conf, 2)

        result = scorer.score(
            retrieval_scores=[0.6, 0.8],
            num_chunks=5,
            is_valid=True,
        )
        assert result == pytest.approx(expected, abs=0.01)


# ── Coverage confidence boundaries ───────────────────────────────────────────
# Formula: coverage_confidence = min(num_chunks / 5.0, 1)

class TestCoverageBoundaries:
    def test_one_chunk_gives_low_coverage(self, scorer):
        """num_chunks=1 → 1/5=0.2 coverage."""
        retrieval_conf = min(max((0.9 - 0.2) / 0.7, 0), 1)  # 1.0
        expected = round(0.7 * retrieval_conf + 0.3 * 0.2, 2)

        result = scorer.score(
            retrieval_scores=[0.9],
            num_chunks=1,
            is_valid=True,
        )
        assert result == pytest.approx(expected, abs=0.01)

    def test_five_chunks_gives_full_coverage(self, scorer):
        """num_chunks=5 → 5/5=1.0 coverage."""
        result = scorer.score(
            retrieval_scores=[0.9],
            num_chunks=5,
            is_valid=True,
        )
        # retrieval=1, coverage=1 → 0.7*1 + 0.3*1 = 1.0
        assert result == pytest.approx(1.0, abs=0.01)

    def test_more_than_five_chunks_clamped_to_one(self, scorer):
        """num_chunks=10 → clamped to 1.0 coverage."""
        result_5 = scorer.score(
            retrieval_scores=[0.9], num_chunks=5, is_valid=True
        )
        result_10 = scorer.score(
            retrieval_scores=[0.9], num_chunks=10, is_valid=True
        )
        assert result_5 == result_10

    def test_two_chunks_gives_partial_coverage(self, scorer):
        """num_chunks=2 → 2/5=0.4 coverage."""
        retrieval_conf = min(max((0.9 - 0.2) / 0.7, 0), 1)  # 1.0
        expected = round(0.7 * retrieval_conf + 0.3 * 0.4, 2)

        result = scorer.score(
            retrieval_scores=[0.9],
            num_chunks=2,
            is_valid=True,
        )
        assert result == pytest.approx(expected, abs=0.01)


# ── Combined formula verification ─────────────────────────────────────────────

class TestFormulaEndToEnd:
    def test_known_values_produce_expected_score(self, scorer):
        """
        retrieval_scores=[0.65], num_chunks=3, is_valid=True
        avg = 0.65
        retrieval_conf = (0.65 - 0.2) / 0.7 = 0.45/0.7 ≈ 0.6429
        coverage_conf  = 3 / 5 = 0.6
        confidence     = 0.7 * 0.6429 + 0.3 * 0.6 = 0.45 + 0.18 = 0.63
        """
        retrieval_conf = (0.65 - 0.2) / 0.7
        coverage_conf = 3 / 5.0
        expected = round(0.7 * retrieval_conf + 0.3 * coverage_conf, 2)

        result = scorer.score(
            retrieval_scores=[0.65],
            num_chunks=3,
            is_valid=True,
        )
        assert result == pytest.approx(expected, abs=0.001)

    def test_worst_case_valid_answer(self, scorer):
        """Lowest possible non-zero score: avg just above floor, 1 chunk."""
        result = scorer.score(
            retrieval_scores=[0.21],
            num_chunks=1,
            is_valid=True,
        )
        assert 0.0 < result < 0.5

    def test_best_case_valid_answer(self, scorer):
        """Perfect retrieval, full coverage."""
        result = scorer.score(
            retrieval_scores=[1.0],
            num_chunks=5,
            is_valid=True,
        )
        assert result == 1.0

    def test_score_increases_with_better_retrieval(self, scorer):
        low = scorer.score(retrieval_scores=[0.3], num_chunks=3, is_valid=True)
        high = scorer.score(retrieval_scores=[0.8], num_chunks=3, is_valid=True)
        assert high > low

    def test_score_increases_with_more_chunks(self, scorer):
        few = scorer.score(retrieval_scores=[0.7], num_chunks=1, is_valid=True)
        many = scorer.score(retrieval_scores=[0.7], num_chunks=5, is_valid=True)
        assert many > few