"""
Answer confidence scoring.

Computes a confidence score based on retrieval relevance,
context coverage, and grounding validation results.
"""

from typing import List

class ConfidenceScorer:
    def score(
        self,
        *,
        retrieval_scores: List[float],
        num_chunks: int,
        is_valid: bool,
    ) -> float:

        if not is_valid:
            return 0.0

        avg_score = sum(retrieval_scores) / len(retrieval_scores)

        retrieval_confidence = min(max((avg_score - 0.2) / 0.7, 0.0), 1.0)

        coverage_confidence = min(num_chunks / 5.0, 1.0)

        confidence = (
            0.7 * retrieval_confidence +
            0.3 * coverage_confidence
        )

        return round(confidence, 2)
