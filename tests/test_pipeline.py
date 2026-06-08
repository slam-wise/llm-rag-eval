"""
Tests for rag_eval/pipeline.py

All evaluators and the provider are mocked — no API calls.
Covers:
  - Default evaluator set
  - Custom evaluator subset
  - EvalSummary structure and content
  - Mean score computation
  - Cost aggregation
  - Empty dataset and empty evaluator list edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_eval.evaluators import (
    AnswerRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
)
from rag_eval.pipeline import EvalPipeline
from rag_eval.types import EvalCase, EvalMetric, EvalScore, EvalSummary, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_case(case_id: str = "case_01") -> EvalCase:
    return EvalCase(
        id=case_id,
        query="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital is Paris."],
        response="The capital of France is Paris.",
    )


def make_eval_score(score: float = 0.9) -> EvalScore:
    return EvalScore(
        score=score,
        reasoning="Test reasoning.",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        latency_ms=120.0,
    )


def make_mock_evaluator(metric: EvalMetric, score: float = 0.9) -> MagicMock:
    evaluator = MagicMock()
    evaluator.metric = metric
    evaluator.evaluate.return_value = make_eval_score(score)
    return evaluator


def make_mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.estimate_cost.return_value = 0.0
    return provider


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestEvalPipelineInit:
    def test_default_evaluators_are_all_four(self):
        pipeline = EvalPipeline(provider=make_mock_provider())
        metrics = {e.metric for e in pipeline.evaluators}
        assert metrics == {
            EvalMetric.FAITHFULNESS,
            EvalMetric.CONTEXT_RELEVANCE,
            EvalMetric.ANSWER_RELEVANCE,
            EvalMetric.HALLUCINATION,
        }

    def test_custom_evaluators_accepted(self):
        evaluators = [
            make_mock_evaluator(EvalMetric.FAITHFULNESS),
            make_mock_evaluator(EvalMetric.HALLUCINATION),
        ]
        pipeline = EvalPipeline(provider=make_mock_provider(), evaluators=evaluators)
        assert len(pipeline.evaluators) == 2

    def test_empty_evaluator_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            EvalPipeline(provider=make_mock_provider(), evaluators=[])

    def test_repr_contains_provider_and_metrics(self):
        evaluators = [make_mock_evaluator(EvalMetric.FAITHFULNESS)]
        pipeline = EvalPipeline(provider=make_mock_provider(), evaluators=evaluators)
        r = repr(pipeline)
        assert "EvalPipeline" in r
        assert "faithfulness" in r


# ---------------------------------------------------------------------------
# run() — core behaviour
# ---------------------------------------------------------------------------


class TestEvalPipelineRun:
    @pytest.fixture()
    def pipeline(self):
        evaluators = [
            make_mock_evaluator(EvalMetric.FAITHFULNESS, score=0.9),
            make_mock_evaluator(EvalMetric.HALLUCINATION, score=0.7),
        ]
        return EvalPipeline(provider=make_mock_provider(), evaluators=evaluators)

    def test_returns_eval_summary(self, pipeline):
        summary = pipeline.run([make_case()], show_progress=False)
        assert isinstance(summary, EvalSummary)

    def test_result_count_matches_dataset(self, pipeline):
        dataset = [make_case(f"case_{i:02d}") for i in range(5)]
        summary = pipeline.run(dataset, show_progress=False)
        assert summary.total_cases == 5
        assert len(summary.results) == 5

    def test_case_ids_preserved(self, pipeline):
        dataset = [make_case("alpha"), make_case("beta")]
        summary = pipeline.run(dataset, show_progress=False)
        ids = [r.case_id for r in summary.results]
        assert ids == ["alpha", "beta"]

    def test_each_evaluator_called_per_case(self, pipeline):
        dataset = [make_case("c1"), make_case("c2"), make_case("c3")]
        pipeline.run(dataset, show_progress=False)
        for evaluator in pipeline.evaluators:
            assert evaluator.evaluate.call_count == 3

    def test_scores_present_in_results(self, pipeline):
        summary = pipeline.run([make_case()], show_progress=False)
        result = summary.results[0]
        assert EvalMetric.FAITHFULNESS in result.scores
        assert EvalMetric.HALLUCINATION in result.scores

    def test_empty_dataset_raises(self, pipeline):
        with pytest.raises(ValueError, match="at least one"):
            pipeline.run([], show_progress=False)


# ---------------------------------------------------------------------------
# Mean score computation
# ---------------------------------------------------------------------------


class TestMeanScores:
    def test_single_case_mean_equals_score(self):
        evaluators = [make_mock_evaluator(EvalMetric.FAITHFULNESS, score=0.75)]
        pipeline = EvalPipeline(provider=make_mock_provider(), evaluators=evaluators)
        summary = pipeline.run([make_case()], show_progress=False)
        assert summary.mean_scores[EvalMetric.FAITHFULNESS] == pytest.approx(0.75)

    def test_mean_averaged_correctly(self):
        scores = [0.8, 0.6, 1.0]
        evaluator = make_mock_evaluator(EvalMetric.FAITHFULNESS)
        evaluator.evaluate.side_effect = [make_eval_score(s) for s in scores]
        pipeline = EvalPipeline(provider=make_mock_provider(), evaluators=[evaluator])
        dataset = [make_case(f"c{i}") for i in range(3)]
        summary = pipeline.run(dataset, show_progress=False)
        expected = sum(scores) / len(scores)
        assert summary.mean_scores[EvalMetric.FAITHFULNESS] == pytest.approx(expected, abs=1e-4)

    def test_mean_scores_only_include_evaluated_metrics(self):
        evaluators = [make_mock_evaluator(EvalMetric.FAITHFULNESS)]
        pipeline = EvalPipeline(provider=make_mock_provider(), evaluators=evaluators)
        summary = pipeline.run([make_case()], show_progress=False)
        assert EvalMetric.FAITHFULNESS in summary.mean_scores
        assert EvalMetric.HALLUCINATION not in summary.mean_scores


# ---------------------------------------------------------------------------
# Cost aggregation
# ---------------------------------------------------------------------------


class TestCostAggregation:
    def test_free_tier_cost_is_zero(self):
        pipeline = EvalPipeline(
            provider=make_mock_provider(),
            evaluators=[make_mock_evaluator(EvalMetric.FAITHFULNESS)],
        )
        summary = pipeline.run([make_case()], show_progress=False)
        assert summary.total_estimated_cost_usd == 0.0

    def test_cost_summed_across_cases(self):
        provider = make_mock_provider()
        provider.estimate_cost.return_value = 0.001  # $0.001 per case

        pipeline = EvalPipeline(
            provider=provider,
            evaluators=[make_mock_evaluator(EvalMetric.FAITHFULNESS)],
        )
        dataset = [make_case(f"c{i}") for i in range(4)]
        summary = pipeline.run(dataset, show_progress=False)
        assert summary.total_estimated_cost_usd == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class TestProgressBar:
    def test_show_progress_false_skips_rich(self):
        """No rich import needed when show_progress=False."""
        pipeline = EvalPipeline(
            provider=make_mock_provider(),
            evaluators=[make_mock_evaluator(EvalMetric.FAITHFULNESS)],
        )
        # Just confirm it runs without error and returns a summary.
        summary = pipeline.run([make_case()], show_progress=False)
        assert summary.total_cases == 1

    def test_show_progress_true_calls_rich_track(self):
        pipeline = EvalPipeline(
            provider=make_mock_provider(),
            evaluators=[make_mock_evaluator(EvalMetric.FAITHFULNESS)],
        )
        with patch("rag_eval.pipeline.track") as mock_track:
            mock_track.return_value = iter([make_case()])
            pipeline.run([make_case()], show_progress=True)
            mock_track.assert_called_once()