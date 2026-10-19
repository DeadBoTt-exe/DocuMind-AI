"""Tests for evals package and benchmark evaluator."""

from pathlib import Path

import pytest

from documind.prompts import DEFAULT_NO_ANSWER
from evals.evaluate import BenchmarkEvaluator, make_mock_rag_engine


@pytest.fixture
def benchmark_path():
    return Path("evals/benchmark_questions.json")


def test_benchmark_dataset_structure(benchmark_path):
    assert benchmark_path.exists(), "Benchmark dataset file must exist"
    evaluator = BenchmarkEvaluator(benchmark_path)
    assert len(evaluator.dataset) == 20, "Must have exactly 20 curated questions"

    categories = {q["category"] for q in evaluator.dataset}
    assert categories == {"directly_answerable", "tricky_partially_answerable", "out_of_domain"}

    for item in evaluator.dataset:
        assert "id" in item
        assert "question" in item
        assert "expected_answerable" in item
        assert isinstance(item["expected_answer_keywords"], list)


@pytest.mark.asyncio
async def test_mock_evaluator_run(benchmark_path):
    evaluator = BenchmarkEvaluator(benchmark_path)
    engine = make_mock_rag_engine()
    results = await evaluator.run(engine)

    assert len(results) == 20
    summary = evaluator.print_summary(results)

    assert summary["total"] == 20
    assert summary["passed"] == 20
    assert summary["failed"] == 0
    assert summary["pass_rate_pct"] == 100.0
    assert summary["hallucination_rate_pct"] == 0.0


def test_evaluator_flags_hallucination_on_out_of_domain(benchmark_path):
    evaluator = BenchmarkEvaluator(benchmark_path)
    ood_item = next(q for q in evaluator.dataset if q["category"] == "out_of_domain")

    # Hallucinated answer (claims to answer Azure question without refusal)
    hallucinated_response = {
        "answer": "Here is how you set up Azure Active Directory Conditional Access policies...",
        "sources": ["organizations-userguide.pdf#page-1"],
        "validation": {"is_valid": True, "reason": None},
        "confidence": 0.85,
    }

    result = evaluator.evaluate_response(ood_item, hallucinated_response, latency_ms=10.0)
    assert result.passed is False
    assert "Hallucination detected" in result.notes


def test_evaluator_passes_clean_refusal_on_out_of_domain(benchmark_path):
    evaluator = BenchmarkEvaluator(benchmark_path)
    ood_item = next(q for q in evaluator.dataset if q["category"] == "out_of_domain")

    clean_refusal_response = {
        "answer": DEFAULT_NO_ANSWER,
        "sources": [],
        "validation": {"is_valid": False, "reason": "No relevant context retrieved."},
        "confidence": 0.0,
    }

    result = evaluator.evaluate_response(ood_item, clean_refusal_response, latency_ms=10.0)
    assert result.passed is True
    assert result.confidence == 0.0
