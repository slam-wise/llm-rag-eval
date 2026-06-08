"""
Abstract base class and shared utilities for all evaluators.

Every evaluator follows the same three-step pattern:
    1. _build_prompt()  — construct a judge prompt from an EvalCase
    2. provider.complete() — call the LLM judge
    3. _parse_response() — extract score/reasoning/claims from the JSON reply

The parsing logic lives here so each concrete evaluator only needs to
implement _build_prompt() and declare its metric.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from rag_eval.providers.base import BaseLLMProvider
from rag_eval.types import EvalCase, EvalMetric, EvalScore, LLMResponse


class EvaluatorError(Exception):
    """Raised when an evaluator cannot parse or process a judge response.

    Includes the raw response text so the caller can log or inspect it.
    """


# ---------------------------------------------------------------------------
# Internal model — only used to validate the judge's JSON output
# ---------------------------------------------------------------------------


class _JudgeOutput(BaseModel):
    """Pydantic model for the structured JSON the judge must return."""

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    reasoning: str = Field(..., min_length=1)
    flagged_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared prompt footer — appended to every evaluator prompt
# ---------------------------------------------------------------------------

_JSON_SCHEMA_INSTRUCTION = """\
Respond ONLY with valid JSON. No markdown, no code blocks, no text outside the object.

Required schema:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed step-by-step explanation>",
    "flagged_claims": ["<item 1>", "<item 2>"]
}}

Rules for "flagged_claims":
- Write each item as a plain string. Do NOT wrap items in extra quotation marks.
- Correct:   ["The response states X, which contradicts the context."]
- Incorrect: [\"The response states X, which contradicts the context.\"]
- Use an empty list [] if there are no flagged claims."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """Abstract base for all RAG evaluators.

    Subclasses must implement:
        - metric (property) → EvalMetric
        - _build_prompt(case) → str

    Everything else — calling the provider, parsing JSON, building EvalScore —
    is handled here.
    """

    @property
    @abstractmethod
    def metric(self) -> EvalMetric:
        """The EvalMetric this evaluator produces."""
        ...

    @abstractmethod
    def _build_prompt(self, case: EvalCase) -> str:
        """Construct the LLM judge prompt for a given EvalCase.

        Args:
            case: The evaluation test case.

        Returns:
            A complete prompt string ready to pass to provider.complete().
        """
        ...

    def evaluate(self, case: EvalCase, provider: BaseLLMProvider) -> EvalScore:
        """Run the evaluation for a single EvalCase.

        Args:
            case: The evaluation test case.
            provider: The LLM provider to use as a judge.

        Returns:
            EvalScore with score, reasoning, flagged claims, and token metadata.

        Raises:
            EvaluatorError: If the judge returns an unparseable or invalid response.
            ProviderError: If the underlying API call fails.
        """
        prompt = self._build_prompt(case)
        response = provider.complete(prompt)
        return self._parse_response(response)

    def _parse_response(self, response: LLMResponse) -> EvalScore:
        """Parse the judge's JSON response into an EvalScore.

        Handles the common case where the model wraps the JSON in a
        markdown code block despite being instructed not to.

        Args:
            response: Raw LLMResponse from the provider.

        Returns:
            EvalScore populated from the parsed JSON plus response metadata.

        Raises:
            EvaluatorError: On JSON parse failure or schema validation error.
        """
        text = response.text.strip()

        # Strip markdown code fences — ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = self._sanitise_json(text.strip())

        try:
            judge_output = _JudgeOutput.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            raise EvaluatorError(
                f"[{self.metric.value}] Failed to parse judge response as JSON.\n"
                f"Raw response: {response.text!r}\n"
                f"Error: {exc}"
            ) from exc

        return EvalScore(
            score=judge_output.score,
            reasoning=judge_output.reasoning,
            flagged_claims=judge_output.flagged_claims,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )

    @staticmethod
    def _sanitise_json(text: str) -> str:
        """Fix common JSON formatting issues produced by local LLMs.

        Handles two patterns seen with smaller local models:
        1. Unicode smart/curly quotes — replaced with ASCII single quotes so
           they do not conflict with JSON string delimiters.
        2. Redundant double-quotes around array items: [""text""] -> ["text"].
           Local models sometimes add an extra layer of quoting inside arrays.
        """
        # Replace Unicode curly/smart quotes with ASCII single quotes.
        text = text.replace("“", "'").replace("”", "'")
        text = text.replace("‘", "'").replace("’", "'")

        # Fix [""text""] -> ["text"]:
        # Step 1: remove the spurious opening quote right after [ or ,
        text = re.sub(r'(?<=[\[,])\s*""\s*', '"', text)
        # Step 2: remove the spurious closing quote right before ] or ,
        text = re.sub(r'\s*""\s*(?=[\],])', '"', text)

        return text

    @staticmethod
    def _format_context(chunks: list[str]) -> str:
        """Format a list of context chunks for insertion into a prompt.

        A single chunk is included as-is. Multiple chunks are labelled
        [CHUNK 1], [CHUNK 2], ... so the judge can reference them.

        Args:
            chunks: Retrieved document chunks from the EvalCase.

        Returns:
            A formatted string ready for prompt interpolation.
        """
        if len(chunks) == 1:
            return chunks[0]
        return "\n\n".join(f"[CHUNK {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metric={self.metric.value!r})"