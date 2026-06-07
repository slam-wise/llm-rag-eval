"""
Tests for rag_eval/types.py

Covers:
- Valid construction of all models
- Field validation (score range, empty context, blank chunks)
- TokenUsage.__add__ accumulation
- Computed fields on EvalResult and EvalSummary
- EvalSummary.validate_result_count cross-field validator
"""

import pytest
from pydantic import ValidationError

from rag_eval.types import (
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalScore,
    EvalSummary,
    LLMResponse,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_usage(inp: int = 100, out: int = 50) -> TokenUsage:
    return TokenUsage(input_tokens=inp, output_tokens=out)


def make_score(score: float = 0.8) -> EvalScore:
    return EvalScore(
        score=score,
        reasoning="Test reasoning.",
        usage=make_usage(),
        latency_ms=120.0,
    )


def make_eval_case(**kwargs) -> EvalCase:
    defaults = dict(
        id="case_01",
        query="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital is Paris."],
        response="The capital of France is Paris.",
    )
    return EvalCase(**(defaults | kwargs))


def make_eval_result(case_id: str = "case_01") -> EvalResult:
    return EvalResult(
        case_id=case_id,
        scores={
            EvalMetric.FAITHFULNESS: make_score(0.9),
            EvalMetric.HALLUCINATION: make_score(0.85),
        },
    )


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_valid_construction(self):
        case = make_eval_case()
        assert case.id == "case_01"
        assert case.reference is None

    def test_with_reference(self):
        case = make_eval_case(reference="Paris")
        assert case.reference == "Paris"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError, match="query"):
            make_eval_case(query="")

    def test_empty_response_rejected(self):
        with pytest.raises(ValidationError, match="response"):
            make_eval_case(response="")

    def test_empty_context_list_rejected(self):
        with pytest.raises(ValidationError):
            make_eval_case(context=[])

    def test_blank_context_chunk_rejected(self):
        with pytest.raises(ValidationError, match="whitespace-only"):
            make_eval_case(context=["valid chunk", "   "])

    def test_empty_string_context_chunk_rejected(self):
        with pytest.raises(ValidationError, match="whitespace-only"):
            make_eval_case(context=[""])

    def test_multiple_context_chunks(self):
        case = make_eval_case(context=["chunk one", "chunk two", "chunk three"])
        assert len(case.context) == 3


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens_computed(self):
        usage = make_usage(inp=100, out=50)
        assert usage.total_tokens == 150

    def test_addition(self):
        a = make_usage(inp=100, out=50)
        b = make_usage(inp=200, out=75)
        total = a + b
        assert total.input_tokens == 300
        assert total.output_tokens == 125
        assert total.total_tokens == 425

    def test_chained_addition(self):
        usages = [make_usage(10, 5) for _ in range(4)]
        total = usages[0]
        for u in usages[1:]:
            total = total + u
        assert total.input_tokens == 40
        assert total.output_tokens == 20

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1, output_tokens=0)


# ---------------------------------------------------------------------------
# EvalScore
# ---------------------------------------------------------------------------


class TestEvalScore:
    def test_valid_score(self):
        score = make_score(0.75)
        assert score.score == 0.75
        assert score.flagged_claims == []

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            make_score(score=1.1)

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            make_score(score=-0.1)

    def test_boundary_scores_accepted(self):
        assert make_score(0.0).score == 0.0
        assert make_score(1.0).score == 1.0

    def test_flagged_claims_populated(self):
        score = EvalScore(
            score=0.3,
            reasoning="Two claims were not in the context.",
            flagged_claims=["Claim A", "Claim B"],
            usage=make_usage(),
            latency_ms=200.0,
        )
        assert len(score.flagged_claims) == 2

    def test_empty_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            EvalScore(score=0.5, reasoning="", usage=make_usage(), latency_ms=100.0)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_valid_construction(self):
        resp = LLMResponse(text="Hello", usage=make_usage(), latency_ms=300.0)
        assert resp.text == "Hello"

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            LLMResponse(text="Hi", usage=make_usage(), latency_ms=-1.0)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_total_latency_computed(self):
        result = EvalResult(
            case_id="case_01",
            scores={
                EvalMetric.FAITHFULNESS: EvalScore(
                    score=0.9, reasoning="ok", usage=make_usage(), latency_ms=100.0
                ),
                EvalMetric.HALLUCINATION: EvalScore(
                    score=0.8, reasoning="ok", usage=make_usage(), latency_ms=150.0
                ),
            },
        )
        assert result.total_latency_ms == pytest.approx(250.0)

    def test_total_usage_computed(self):
        result = EvalResult(
            case_id="case_01",
            scores={
                EvalMetric.FAITHFULNESS: EvalScore(
                    score=0.9,
                    reasoning="ok",
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                    latency_ms=100.0,
                ),
                EvalMetric.ANSWER_RELEVANCE: EvalScore(
                    score=0.7,
                    reasoning="ok",
                    usage=TokenUsage(input_tokens=80, output_tokens=40),
                    latency_ms=90.0,
                ),
            },
        )
        assert result.total_usage.input_tokens == 180
        assert result.total_usage.output_tokens == 90

    def test_empty_scores_total_usage(self):
        result = EvalResult(case_id="case_01", scores={})
        assert result.total_usage.total_tokens == 0
        assert result.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# EvalSummary
# ---------------------------------------------------------------------------


class TestEvalSummary:
    def test_valid_construction(self):
        results = [make_eval_result("case_01"), make_eval_result("case_02")]
        summary = EvalSummary(
            results=results,
            mean_scores={EvalMetric.FAITHFULNESS: 0.875},
            total_cases=2,
        )
        assert summary.total_cases == 2

    def test_mismatched_total_cases_rejected(self):
        results = [make_eval_result("case_01")]
        with pytest.raises(ValidationError, match="total_cases"):
            EvalSummary(
                results=results,
                mean_scores={},
                total_cases=3,  # wrong — only 1 result
            )

    def test_total_latency_aggregated(self):
        r1 = make_eval_result("case_01")
        r2 = make_eval_result("case_02")
        summary = EvalSummary(
            results=[r1, r2],
            mean_scores={},
            total_cases=2,
        )
        assert summary.total_latency_ms == pytest.approx(
            r1.total_latency_ms + r2.total_latency_ms
        )

    def test_total_usage_aggregated(self):
        r1 = make_eval_result("case_01")
        r2 = make_eval_result("case_02")
        summary = EvalSummary(results=[r1, r2], mean_scores={}, total_cases=2)
        expected_tokens = r1.total_usage.total_tokens + r2.total_usage.total_tokens
        assert summary.total_usage.total_tokens == expected_tokens