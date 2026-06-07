"""
rag-eval: LLM evaluation framework for RAG pipelines.

Focused on faithfulness, context relevance, answer relevance,
and hallucination detection using an LLM-as-judge approach.
"""

from rag_eval.types import (
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalScore,
    EvalSummary,
    LLMResponse,
    TokenUsage,
)

__version__ = "0.1.0"

__all__ = [
    "EvalCase",
    "EvalMetric",
    "EvalResult",
    "EvalScore",
    "EvalSummary",
    "LLMResponse",
    "TokenUsage",
]