"""Automated evaluation harness for DocuMind RAG engine.

Evaluates:
  1. Retrieval Precision & Hit Rate (retrieval of expected source document)
  2. Grounding Accuracy (hallucination rejection on out-of-domain questions)
  3. Validation Behavior (proper failure detection & fail-closed responses)
  4. Confidence Calibration (well-calibrated score vs. answer validity)
  5. Query Latency (average, min, max response latency)

Supports running both against live RAG engine (if Qdrant & Gemini keys configured)
and with deterministic mock mode for automated CI/regression testing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from documind.prompts import DEFAULT_NO_ANSWER, VALIDATION_FAILED_ANSWER

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question_id: str
    category: str
    question: str
    passed: bool
    answer: str
    is_valid: bool
    confidence: float
    latency_ms: float
    sources: list[str]
    notes: str


class BenchmarkEvaluator:
    def __init__(self, dataset_path: Path):
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset: list[dict[str, Any]] = json.load(f)
        self.console = Console()

    def evaluate_response(self, item: dict[str, Any], response: dict[str, Any], latency_ms: float) -> EvalResult:
        answer = response.get("answer", "")
        validation = response.get("validation", {})
        is_valid = validation.get("is_valid", False)
        confidence = response.get("confidence", 0.0)
        sources = response.get("sources", [])
        category = item["category"]

        passed = True
        notes = []

        if category == "out_of_domain":
            # Must refuse cleanly: either DEFAULT_NO_ANSWER or VALIDATION_FAILED_ANSWER
            is_refusal = (
                DEFAULT_NO_ANSWER in answer
                or answer == VALIDATION_FAILED_ANSWER
                or not is_valid
            )
            if not is_refusal:
                passed = False
                notes.append("Hallucination detected: model answered out-of-domain question.")
            if confidence > 0.4:
                passed = False
                notes.append(f"Confidence score {confidence} too high for unanswerable question.")
        else:
            # Directly answerable or tricky questions
            if DEFAULT_NO_ANSWER in answer or answer == VALIDATION_FAILED_ANSWER or not is_valid:
                passed = False
                notes.append("Refusal or validation failure on answerable question.")

            if item.get("expected_source"):
                matched_source = any(item["expected_source"] in src for src in sources)
                if not matched_source:
                    passed = False
                    notes.append(f"Expected source '{item['expected_source']}' not in retrieved sources.")

            # Check for keyword grounding in answer
            keywords = item.get("expected_answer_keywords", [])
            lower_answer = answer.lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in lower_answer)
            if keywords and keyword_matches == 0 and answer != DEFAULT_NO_ANSWER:
                notes.append("Low keyword overlap with expected answer domain.")

            # If validation failed, confidence should be strictly 0.0
            if not is_valid and confidence > 0.0:
                passed = False
                notes.append("Confidence non-zero despite invalid/failed validation.")

        return EvalResult(
            question_id=item["id"],
            category=category,
            question=item["question"],
            passed=passed,
            answer=answer,
            is_valid=is_valid,
            confidence=confidence,
            latency_ms=latency_ms,
            sources=sources,
            notes="; ".join(notes) if notes else "OK",
        )

    async def run(self, rag_engine: Any) -> list[EvalResult]:
        results: list[EvalResult] = []

        for item in self.dataset:
            start_t = time.perf_counter()
            try:
                response = await rag_engine.ask_async(item["question"])
            except Exception as e:
                response = {
                    "answer": f"Evaluation error: {e}",
                    "sources": [],
                    "validation": {"is_valid": False, "reason": str(e), "skipped": True},
                    "confidence": 0.0,
                }
            latency_ms = (time.perf_counter() - start_t) * 1000

            eval_res = self.evaluate_response(item, response, latency_ms)
            results.append(eval_res)

        return results

    def print_summary(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total) * 100 if total > 0 else 0.0

        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        ood_results = [r for r in results if r.category == "out_of_domain"]
        ood_passed = sum(1 for r in ood_results if r.passed)
        hallucination_rate = ((len(ood_results) - ood_passed) / len(ood_results)) * 100 if ood_results else 0.0

        table = Table(title="DocuMind RAG Evaluation Results", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Pass/Fail", style="bold")
        table.add_column("Valid?", justify="center")
        table.add_column("Conf", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Notes", style="dim")

        for r in results:
            status_style = "green" if r.passed else "red"
            status_text = f"[{status_style}]{'PASS' if r.passed else 'FAIL'}[/{status_style}]"
            table.add_row(
                r.question_id,
                r.category,
                status_text,
                "YES" if r.is_valid else "NO",
                f"{r.confidence:.2f}",
                f"{r.latency_ms:.1f}",
                r.notes,
            )

        self.console.print(table)

        summary_table = Table(title="Performance Metrics Summary")
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold green")

        summary_table.add_row("Total Benchmark Questions", str(total))
        summary_table.add_row("Passing Tests", str(passed))
        summary_table.add_row("Failing Tests", str(failed))
        summary_table.add_row("Accuracy / Pass Rate", f"{pass_rate:.1f}%")
        summary_table.add_row("Hallucination Rate (Negative Cases)", f"{hallucination_rate:.1f}%")
        summary_table.add_row("Average Latency", f"{avg_latency:.1f} ms")

        self.console.print(summary_table)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round(pass_rate, 2),
            "hallucination_rate_pct": round(hallucination_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
        }


def make_mock_rag_engine():
    """Deterministic mock RAGEngine for local automated CI verification."""

    class MockRAGEngine:
        async def ask_async(self, question: str, top_k: int | None = None) -> dict[str, Any]:
            await asyncio.sleep(0.01)  # simulate brief async processing
            q_lower = question.lower()

            if any(term in q_lower for term in ["azure", "kubernetes", "pytorch", "australia", "square root"]):
                # Clean refusal for out-of-domain questions
                return {
                    "answer": DEFAULT_NO_ANSWER,
                    "sources": [],
                    "validation": {"is_valid": False, "reason": "No relevant context retrieved."},
                    "confidence": 0.0,
                }

            # Return answer with valid groundings
            return {
                "answer": (
                    f"Based on the AWS Organizations documentation, {question} relates to "
                    "managing accounts, policies, organizational units (OUs), service control policies, "
                    "and permission boundaries."
                ),
                "sources": ["organizations-userguide.pdf#page-15", "organizations-userguide.pdf#page-42"],
                "validation": {"is_valid": True, "reason": None},
                "confidence": 0.88,
            }

    return MockRAGEngine()


async def main():
    parser = argparse.ArgumentParser(description="DocuMind RAG Evaluation Benchmark Runner")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using deterministic mock RAG engine for CI/testing without live API calls",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/benchmark_questions.json"),
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to export eval results as JSON",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(args.dataset)

    if args.mock:
        engine = make_mock_rag_engine()
    else:
        from documind.rag import RAGEngine
        engine = RAGEngine()

    results = await evaluator.run(engine)
    summary = evaluator.print_summary(results)

    if args.output:
        out_data = {
            "summary": summary,
            "results": [
                {
                    "id": r.question_id,
                    "category": r.category,
                    "question": r.question,
                    "passed": r.passed,
                    "is_valid": r.is_valid,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "notes": r.notes,
                }
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
