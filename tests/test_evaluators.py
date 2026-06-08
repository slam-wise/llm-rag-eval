"""
Tests for rag_eval/evaluators/

All tests use a mock provider — no API calls are made.
Covers:
  - BaseEvaluator: JSON parsing, markdown stripping, error handling,
    context formatting, metadata passthrough
  - Each evaluator: metric identity, prompt content, end-to-end evaluate()
  - ContextRelevanceEvaluator: response deliberately excluded from prompt
  - AnswerRelevanceEvaluator: context deliberately excluded from prompt
  - HallucinationEvaluator / FaithfulnessEvaluator: flagged claims passed through
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag_eval.evaluators import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    EvaluatorError,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
)
from rag_eval.evaluators.base import BaseEvaluator, _JudgeOutput
from rag_eval.types import EvalCase, EvalMetric, EvalScore, LLMResponse, TokenUsage


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_case(**kwargs) -> EvalCase:
    defaults = dict(
        id="test_01",
        query="What year was the Eiffel Tower completed?",
        context=[
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
        ],
        response="The Eiffel Tower was completed in 1889.",
    )
    return EvalCase(**(defaults | kwargs))


def make_llm_response(text: str, input_tokens: int = 80, output_tokens: int = 40) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=150.0,
    )


def make_judge_json(
    score: float = 0.9,
    reasoning: str = "The response is well supported.",
    flagged_claims: list[str] | None = None,
) -> str:
    return json.dumps({
        "score": score,
        "reasoning": reasoning,
        "flagged_claims": flagged_claims or [],
    })


def make_mock_provider(response_text: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = make_llm_response(response_text)
    return provider


# ---------------------------------------------------------------------------
# BaseEvaluator — context formatting
# ---------------------------------------------------------------------------


class TestFormatContext:
    """_format_context is a static utility on BaseEvaluator."""

    def test_single_chunk_returned_as_is(self):
        result = BaseEvaluator._format_context(["Only one chunk here."])
        assert result == "Only one chunk here."

    def test_multiple_chunks_labelled(self):
        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        result = BaseEvaluator._format_context(chunks)
        assert "[CHUNK 1]" in result
        assert "[CHUNK 2]" in result
        assert "[CHUNK 3]" in result
        assert "First chunk." in result
        assert "Third chunk." in result

    def test_two_chunks_separated(self):
        result = BaseEvaluator._format_context(["A", "B"])
        assert result.index("[CHUNK 1]") < result.index("[CHUNK 2]")


# ---------------------------------------------------------------------------
# BaseEvaluator — JSON parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    """_parse_response via a concrete evaluator instance."""

    @pytest.fixture()
    def evaluator(self):
        return FaithfulnessEvaluator()

    def test_valid_json_returns_eval_score(self, evaluator):
        response = make_llm_response(make_judge_json(score=0.85))
        result = evaluator._parse_response(response)
        assert isinstance(result, EvalScore)
        assert result.score == pytest.approx(0.85)

    def test_reasoning_extracted(self, evaluator):
        response = make_llm_response(make_judge_json(reasoning="All claims supported."))
        result = evaluator._parse_response(response)
        assert result.reasoning == "All claims supported."

    def test_flagged_claims_extracted(self, evaluator):
        response = make_llm_response(
            make_judge_json(flagged_claims=["Claim A", "Claim B"])
        )
        result = evaluator._parse_response(response)
        assert result.flagged_claims == ["Claim A", "Claim B"]

    def test_empty_flagged_claims(self, evaluator):
        response = make_llm_response(make_judge_json(flagged_claims=[]))
        result = evaluator._parse_response(response)
        assert result.flagged_claims == []

    def test_usage_passed_through(self, evaluator):
        response = make_llm_response(make_judge_json(), input_tokens=120, output_tokens=55)
        result = evaluator._parse_response(response)
        assert result.usage.input_tokens == 120
        assert result.usage.output_tokens == 55

    def test_latency_passed_through(self, evaluator):
        response = make_llm_response(make_judge_json())
        response.latency_ms = 237.5
        result = evaluator._parse_response(response)
        assert result.latency_ms == pytest.approx(237.5)

    def test_strips_json_markdown_block(self, evaluator):
        wrapped = f"```json\n{make_judge_json(score=0.7)}\n```"
        result = evaluator._parse_response(make_llm_response(wrapped))
        assert result.score == pytest.approx(0.7)

    def test_strips_plain_markdown_block(self, evaluator):
        wrapped = f"```\n{make_judge_json(score=0.6)}\n```"
        result = evaluator._parse_response(make_llm_response(wrapped))
        assert result.score == pytest.approx(0.6)

    def test_handles_leading_trailing_whitespace(self, evaluator):
        padded = f"\n\n   {make_judge_json(score=0.5)}   \n"
        result = evaluator._parse_response(make_llm_response(padded))
        assert result.score == pytest.approx(0.5)


    def test_sanitises_smart_quotes(self, evaluator):
        """Unicode curly quotes should become ASCII single quotes."""
        json_with_curly = '{"score": 0.8, "reasoning": "“Good” response", "flagged_claims": []}'
        result = evaluator._parse_response(make_llm_response(json_with_curly))
        assert result.score == pytest.approx(0.8)

    def test_sanitises_double_quoted_array_items(self, evaluator):
        """[""text""] is a common local model quirk and should parse cleanly."""
        broken_json = '{"score": 0.4, "reasoning": "ok", "flagged_claims": [""claim one""]}'
        result = evaluator._parse_response(make_llm_response(broken_json))
        assert result.flagged_claims == ["claim one"]

    def test_sanitises_multiple_double_quoted_items(self, evaluator):
        broken_json = '{"score": 0.3, "reasoning": "ok", "flagged_claims": [""claim one"", ""claim two""]}'
        result = evaluator._parse_response(make_llm_response(broken_json))
        assert result.flagged_claims == ["claim one", "claim two"]

    def test_invalid_json_raises_evaluator_error(self, evaluator):
        with pytest.raises(EvaluatorError, match="Failed to parse"):
            evaluator._parse_response(make_llm_response("this is not json at all"))

    def test_score_above_one_raises_evaluator_error(self, evaluator):
        bad_json = json.dumps({"score": 1.5, "reasoning": "ok", "flagged_claims": []})
        with pytest.raises(EvaluatorError):
            evaluator._parse_response(make_llm_response(bad_json))

    def test_score_below_zero_raises_evaluator_error(self, evaluator):
        bad_json = json.dumps({"score": -0.1, "reasoning": "ok", "flagged_claims": []})
        with pytest.raises(EvaluatorError):
            evaluator._parse_response(make_llm_response(bad_json))

    def test_missing_reasoning_raises_evaluator_error(self, evaluator):
        bad_json = json.dumps({"score": 0.5, "flagged_claims": []})
        with pytest.raises(EvaluatorError):
            evaluator._parse_response(make_llm_response(bad_json))

    def test_error_message_includes_raw_response(self, evaluator):
        with pytest.raises(EvaluatorError, match="not valid json"):
            evaluator._parse_response(make_llm_response("not valid json"))


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator
# ---------------------------------------------------------------------------


class TestFaithfulnessEvaluator:
    def test_metric(self):
        assert FaithfulnessEvaluator().metric == EvalMetric.FAITHFULNESS

    def test_repr(self):
        assert "faithfulness" in repr(FaithfulnessEvaluator())

    def test_evaluate_returns_eval_score(self):
        provider = make_mock_provider(make_judge_json(score=0.95))
        score = FaithfulnessEvaluator().evaluate(make_case(), provider)
        assert isinstance(score, EvalScore)
        assert score.score == pytest.approx(0.95)

    def test_prompt_includes_query(self):
        case = make_case(query="Unique query string XYZ")
        prompt = FaithfulnessEvaluator()._build_prompt(case)
        assert "Unique query string XYZ" in prompt

    def test_prompt_includes_context(self):
        case = make_case(context=["Unique context string ABC."])
        prompt = FaithfulnessEvaluator()._build_prompt(case)
        assert "Unique context string ABC." in prompt

    def test_prompt_includes_response(self):
        case = make_case(response="Unique response string DEF.")
        prompt = FaithfulnessEvaluator()._build_prompt(case)
        assert "Unique response string DEF." in prompt

    def test_flagged_claims_returned(self):
        provider = make_mock_provider(
            make_judge_json(score=0.4, flagged_claims=["Claim not in context"])
        )
        score = FaithfulnessEvaluator().evaluate(make_case(), provider)
        assert "Claim not in context" in score.flagged_claims

    def test_provider_called_once(self):
        provider = make_mock_provider(make_judge_json())
        FaithfulnessEvaluator().evaluate(make_case(), provider)
        provider.complete.assert_called_once()


# ---------------------------------------------------------------------------
# ContextRelevanceEvaluator
# ---------------------------------------------------------------------------


class TestContextRelevanceEvaluator:
    def test_metric(self):
        assert ContextRelevanceEvaluator().metric == EvalMetric.CONTEXT_RELEVANCE

    def test_evaluate_returns_eval_score(self):
        provider = make_mock_provider(make_judge_json(score=0.8))
        score = ContextRelevanceEvaluator().evaluate(make_case(), provider)
        assert isinstance(score, EvalScore)

    def test_prompt_includes_query(self):
        case = make_case(query="Unique query string XYZ")
        prompt = ContextRelevanceEvaluator()._build_prompt(case)
        assert "Unique query string XYZ" in prompt

    def test_prompt_includes_context(self):
        case = make_case(context=["Unique context string ABC."])
        prompt = ContextRelevanceEvaluator()._build_prompt(case)
        assert "Unique context string ABC." in prompt

    def test_prompt_excludes_response(self):
        """Context relevance must not leak the response into its prompt."""
        case = make_case(response="Unique response string DEF.")
        prompt = ContextRelevanceEvaluator()._build_prompt(case)
        assert "Unique response string DEF." not in prompt


# ---------------------------------------------------------------------------
# AnswerRelevanceEvaluator
# ---------------------------------------------------------------------------


class TestAnswerRelevanceEvaluator:
    def test_metric(self):
        assert AnswerRelevanceEvaluator().metric == EvalMetric.ANSWER_RELEVANCE

    def test_evaluate_returns_eval_score(self):
        provider = make_mock_provider(make_judge_json(score=0.9))
        score = AnswerRelevanceEvaluator().evaluate(make_case(), provider)
        assert isinstance(score, EvalScore)

    def test_prompt_includes_query(self):
        case = make_case(query="Unique query string XYZ")
        prompt = AnswerRelevanceEvaluator()._build_prompt(case)
        assert "Unique query string XYZ" in prompt

    def test_prompt_includes_response(self):
        case = make_case(response="Unique response string DEF.")
        prompt = AnswerRelevanceEvaluator()._build_prompt(case)
        assert "Unique response string DEF." in prompt

    def test_prompt_excludes_context(self):
        """Answer relevance must not leak the context into its prompt."""
        case = make_case(context=["Unique context string ABC."])
        prompt = AnswerRelevanceEvaluator()._build_prompt(case)
        assert "Unique context string ABC." not in prompt


# ---------------------------------------------------------------------------
# HallucinationEvaluator
# ---------------------------------------------------------------------------


class TestHallucinationEvaluator:
    def test_metric(self):
        assert HallucinationEvaluator().metric == EvalMetric.HALLUCINATION

    def test_evaluate_returns_eval_score(self):
        provider = make_mock_provider(make_judge_json(score=1.0))
        score = HallucinationEvaluator().evaluate(make_case(), provider)
        assert isinstance(score, EvalScore)

    def test_high_score_means_no_hallucination(self):
        provider = make_mock_provider(make_judge_json(score=1.0, flagged_claims=[]))
        score = HallucinationEvaluator().evaluate(make_case(), provider)
        assert score.score == pytest.approx(1.0)
        assert score.flagged_claims == []

    def test_low_score_with_flagged_claims(self):
        provider = make_mock_provider(
            make_judge_json(
                score=0.2,
                flagged_claims=["'completed in 1887' contradicts context (1889)"],
            )
        )
        score = HallucinationEvaluator().evaluate(make_case(), provider)
        assert score.score == pytest.approx(0.2)
        assert len(score.flagged_claims) == 1

    def test_prompt_includes_query(self):
        case = make_case(query="Unique query string XYZ")
        prompt = HallucinationEvaluator()._build_prompt(case)
        assert "Unique query string XYZ" in prompt

    def test_prompt_includes_context(self):
        case = make_case(context=["Unique context string ABC."])
        prompt = HallucinationEvaluator()._build_prompt(case)
        assert "Unique context string ABC." in prompt

    def test_prompt_includes_response(self):
        case = make_case(response="Unique response string DEF.")
        prompt = HallucinationEvaluator()._build_prompt(case)
        assert "Unique response string DEF." in prompt