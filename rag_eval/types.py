"""
Core data models for the RAG evaluation framework.

All inputs, outputs, and intermediate results are defined here as Pydantic v2
models. Centralising types keeps the rest of the codebase unambiguous and
makes serialisation/deserialisation trivial via .model_dump() and
.model_validate_json().

Data flow:
    EvalCase  →  [EvalPipeline + Evaluators]  →  EvalResult  →  EvalSummary
                         ↑
                    LLMResponse
                   (TokenUsage)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalMetric(str, Enum):
    """Canonical names for the four built-in evaluators.

    Using an Enum (rather than raw strings) as dict keys means typos are
    caught at definition time and IDEs can autocomplete metric names.
    """

    FAITHFULNESS = "faithfulness"
    CONTEXT_RELEVANCE = "context_relevance"
    ANSWER_RELEVANCE = "answer_relevance"
    HALLUCINATION = "hallucination"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class EvalCase(BaseModel):
    """A single RAG evaluation test case.

    This is the atomic unit of input to the pipeline. Every evaluator
    receives an EvalCase and pulls the fields it needs.

    Attributes:
        id: Unique identifier used in result tables and JSON output.
            Keep it human-readable (e.g. "case_01", "finance_q3").
        query: The user question passed to the RAG pipeline.
        context: Retrieved document chunks supplied to the generator.
                 Must contain at least one non-empty chunk.
        response: The LLM-generated answer being evaluated.
        reference: Optional ground-truth answer. Not required by any v1
                   evaluator but included for future reference-based metrics
                   (e.g. ROUGE, BERTScore).
    """

    id: str = Field(..., description="Unique identifier for this test case")
    query: str = Field(..., min_length=1, description="User question")
    context: list[str] = Field(
        ..., min_length=1, description="Retrieved document chunks"
    )
    response: str = Field(..., min_length=1, description="LLM-generated answer")
    reference: str | None = Field(
        default=None, description="Optional ground-truth answer"
    )

    @field_validator("context")
    @classmethod
    def context_chunks_non_empty(cls, v: list[str]) -> list[str]:
        """Reject context lists that contain blank or whitespace-only chunks."""
        if any(chunk.strip() == "" for chunk in v):
            raise ValueError(
                "context must not contain empty or whitespace-only chunks"
            )
        return v


# ---------------------------------------------------------------------------
# Provider layer
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token consumption for a single LLM call.

    Kept as a first-class model (rather than a plain tuple) so it can be
    accumulated across evaluator calls and serialised cleanly in reports.

    Attributes:
        input_tokens: Tokens in the prompt sent to the model.
        output_tokens: Tokens in the model's response.
    """

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate token usage across multiple calls.

        Example:
            >>> total = usage_a + usage_b + usage_c
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMResponse(BaseModel):
    """Raw response returned by a provider's complete() call.

    The provider layer is responsible for populating all fields.
    Evaluators receive an LLMResponse and extract .text for parsing.

    Attributes:
        text: The model's text output.
        usage: Token consumption for this call.
        latency_ms: Wall-clock time from sending the request to receiving
                    the full response, in milliseconds.
    """

    text: str
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Evaluator output
# ---------------------------------------------------------------------------

# Reusable annotated type — enforces score range at the field level so
# every EvalScore automatically validates without extra validators.
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class EvalScore(BaseModel):
    """Result from a single evaluator on a single EvalCase.

    Attributes:
        score: Normalised score in [0, 1]. Higher is always better across
               all four metrics. For hallucination, 1.0 means no hallucination
               was detected; 0.0 means the response is entirely unsupported.
        reasoning: The judge model's chain-of-thought explanation. Surfaced
                   in reports so failures are interpretable, not just a number.
        flagged_claims: Specific claims identified as hallucinated or
                        unsupported by the context. Populated by the
                        Hallucination and Faithfulness evaluators; empty list
                        for AnswerRelevance and ContextRelevance.
        usage: Token consumption for this evaluation call.
        latency_ms: Wall-clock time for this evaluation call.
    """

    score: Score
    reasoning: str = Field(..., min_length=1)
    flagged_claims: list[str] = Field(default_factory=list)
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """All evaluator scores for a single EvalCase.

    One EvalResult is produced per EvalCase by the pipeline, containing
    the scores from every evaluator that was run.

    Attributes:
        case_id: Matches EvalCase.id for cross-referencing.
        scores: Mapping of EvalMetric → EvalScore for every evaluator run.
        estimated_cost_usd: Estimated API cost for all evaluator calls on
                            this case. Zero for Gemini free-tier usage;
                            field is retained for provider parity so paid
                            providers can populate it.
    """

    case_id: str
    scores: dict[EvalMetric, EvalScore]
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        """Sum of latencies across all evaluator calls for this case."""
        return sum(s.latency_ms for s in self.scores.values())

    @computed_field
    @property
    def total_usage(self) -> TokenUsage:
        """Aggregated token usage across all evaluator calls for this case."""
        score_list = list(self.scores.values())
        if not score_list:
            return TokenUsage(input_tokens=0, output_tokens=0)
        total = score_list[0].usage
        for s in score_list[1:]:
            total = total + s.usage
        return total


class EvalSummary(BaseModel):
    """Aggregated results across a full evaluation run.

    Produced by the pipeline after all EvalCases have been processed.
    This is the top-level object passed to the Reporter.

    Attributes:
        results: Per-case EvalResults in order of evaluation.
        mean_scores: Average score per EvalMetric across all cases.
        total_cases: Number of EvalCases evaluated. Must equal len(results).
        total_estimated_cost_usd: Summed cost across all cases.
    """

    results: list[EvalResult]
    mean_scores: dict[EvalMetric, float]
    total_cases: int = Field(..., ge=1)
    total_estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        """Total wall-clock time spent on evaluator calls across the run."""
        return sum(r.total_latency_ms for r in self.results)

    @computed_field
    @property
    def total_usage(self) -> TokenUsage:
        """Aggregated token usage across every case and every evaluator."""
        if not self.results:
            return TokenUsage(input_tokens=0, output_tokens=0)
        total = self.results[0].total_usage
        for r in self.results[1:]:
            total = total + r.total_usage
        return total

    @model_validator(mode="after")
    def validate_result_count(self) -> EvalSummary:
        """Ensure total_cases is consistent with the results list length."""
        if len(self.results) != self.total_cases:
            raise ValueError(
                f"total_cases={self.total_cases} does not match "
                f"len(results)={len(self.results)}"
            )
        return self