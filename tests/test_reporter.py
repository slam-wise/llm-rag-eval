"""
Tests for rag_eval/reporter.py

Covers:
  - to_json: valid JSON, correct structure, file created
  - to_csv: correct headers, correct row count including MEAN row
  - print_summary: renders without raising, respects colour thresholds
  - Custom high/low thresholds
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from rag_eval.reporter import Reporter
from rag_eval.types import EvalMetric, EvalResult, EvalScore, EvalSummary, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_eval_score(score: float = 0.85) -> EvalScore:
    return EvalScore(
        score=score,
        reasoning="Test reasoning.",
        usage=TokenUsage(input_tokens=100, output_tokens=40),
        latency_ms=150.0,
    )


def make_eval_result(case_id: str, scores: dict[EvalMetric, float]) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        scores={metric: make_eval_score(score) for metric, score in scores.items()},
    )


def make_summary(cases: int = 2) -> EvalSummary:
    results = [
        make_eval_result(
            f"case_{i:02d}",
            {
                EvalMetric.FAITHFULNESS: 0.9,
                EvalMetric.CONTEXT_RELEVANCE: 0.7,
                EvalMetric.ANSWER_RELEVANCE: 0.85,
                EvalMetric.HALLUCINATION: 0.95,
            },
        )
        for i in range(cases)
    ]
    mean_scores = {
        EvalMetric.FAITHFULNESS: 0.9,
        EvalMetric.CONTEXT_RELEVANCE: 0.7,
        EvalMetric.ANSWER_RELEVANCE: 0.85,
        EvalMetric.HALLUCINATION: 0.95,
    }
    return EvalSummary(results=results, mean_scores=mean_scores, total_cases=cases)


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_creates_file(self, tmp_path):
        output = tmp_path / "results.json"
        Reporter().to_json(make_summary(), output)
        assert output.exists()

    def test_output_is_valid_json(self, tmp_path):
        output = tmp_path / "results.json"
        Reporter().to_json(make_summary(), output)
        data = json.loads(output.read_text())
        assert isinstance(data, dict)

    def test_json_contains_results(self, tmp_path):
        output = tmp_path / "results.json"
        Reporter().to_json(make_summary(cases=3), output)
        data = json.loads(output.read_text())
        assert len(data["results"]) == 3

    def test_json_contains_mean_scores(self, tmp_path):
        output = tmp_path / "results.json"
        Reporter().to_json(make_summary(), output)
        data = json.loads(output.read_text())
        assert "mean_scores" in data

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "nested" / "dir" / "results.json"
        Reporter().to_json(make_summary(), output)
        assert output.exists()

    def test_returns_path(self, tmp_path):
        output = tmp_path / "results.json"
        returned = Reporter().to_json(make_summary(), output)
        assert returned == output


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_creates_file(self, tmp_path):
        output = tmp_path / "results.csv"
        Reporter().to_csv(make_summary(), output)
        assert output.exists()

    def test_headers_include_case_id_and_metrics(self, tmp_path):
        output = tmp_path / "results.csv"
        Reporter().to_csv(make_summary(), output)
        with output.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "case_id" in headers
        assert "faithfulness" in headers
        assert "hallucination" in headers

    def test_row_count_includes_mean_row(self, tmp_path):
        output = tmp_path / "results.csv"
        Reporter().to_csv(make_summary(cases=3), output)
        with output.open() as f:
            rows = list(csv.DictReader(f))
        # 3 case rows + 1 MEAN row
        assert len(rows) == 4

    def test_mean_row_present(self, tmp_path):
        output = tmp_path / "results.csv"
        Reporter().to_csv(make_summary(), output)
        with output.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[-1]["case_id"] == "MEAN"

    def test_scores_are_numeric(self, tmp_path):
        output = tmp_path / "results.csv"
        Reporter().to_csv(make_summary(), output)
        with output.open() as f:
            rows = list(csv.DictReader(f))
        score = float(rows[0]["faithfulness"])
        assert 0.0 <= score <= 1.0

    def test_returns_path(self, tmp_path):
        output = tmp_path / "results.csv"
        returned = Reporter().to_csv(make_summary(), output)
        assert returned == output


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_renders_without_error(self, capsys):
        Reporter().print_summary(make_summary(), model_name="gemini-2.0-flash")

    def test_renders_without_model_name(self, capsys):
        Reporter().print_summary(make_summary())

    def test_single_case_renders(self, capsys):
        Reporter().print_summary(make_summary(cases=1))


# ---------------------------------------------------------------------------
# Colour thresholds
# ---------------------------------------------------------------------------


class TestScoreStyle:
    @pytest.fixture()
    def reporter(self):
        return Reporter(high_threshold=0.8, low_threshold=0.5)

    def test_high_score_is_green(self, reporter):
        assert reporter._score_style(0.9) == "green"
        assert reporter._score_style(0.8) == "green"

    def test_medium_score_is_yellow(self, reporter):
        assert reporter._score_style(0.79) == "yellow"
        assert reporter._score_style(0.5) == "yellow"

    def test_low_score_is_red(self, reporter):
        assert reporter._score_style(0.49) == "red"
        assert reporter._score_style(0.0) == "red"

    def test_custom_thresholds(self):
        reporter = Reporter(high_threshold=0.9, low_threshold=0.6)
        assert reporter._score_style(0.85) == "yellow"
        assert reporter._score_style(0.9) == "green"
        assert reporter._score_style(0.59) == "red"
