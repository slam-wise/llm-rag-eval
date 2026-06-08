"""
Context relevance evaluator.

Measures whether the retrieved context chunks actually contain information
useful for answering the query. This is a diagnostic for RETRIEVER quality,
independent of how well the generator used the context.

A low context relevance score suggests the retrieval step is broken:
wrong chunks are being fetched, the embedding space is misaligned, or
the query is too vague for the retriever to find relevant material.

Score interpretation:
    1.0  Context is highly relevant and sufficient to answer the query
    0.75 Context is mostly relevant with minor irrelevant passages
    0.5  Context is partially relevant; key information is missing
    0.25 Context has minimal relevance; largely off-topic
    0.0  Context is entirely unrelated to the query
"""

from __future__ import annotations

from rag_eval.evaluators.base import BaseEvaluator, _JSON_SCHEMA_INSTRUCTION
from rag_eval.types import EvalCase, EvalMetric

_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing the RELEVANCE of retrieved context to a user query.

DEFINITION
Context relevance measures how well the retrieved CONTEXT chunks support \
answering the QUERY. The RESPONSE is not considered here — this metric \
evaluates the RETRIEVER, not the generator.

─────────────────────────────────────────
QUERY:
{query}

CONTEXT:
{context}
─────────────────────────────────────────

EVALUATION STEPS
1. Identify what information would be needed to fully answer the QUERY.
2. For each context chunk, assess how much of that needed information it provides.
3. Consider both relevance (is the topic correct?) and sufficiency (is enough
   detail present to answer the query?).
4. Note any chunks that are off-topic or only tangentially related.

SCORING RUBRIC
1.0  Context fully covers the information needed to answer the query
0.75 Context is mostly relevant; minor gaps or one off-topic chunk
0.5  Context is partially relevant; key details are missing or buried
0.25 Context has minimal relevance; most chunks are off-topic
0.0  Context is entirely unrelated to the query

{schema}"""


class ContextRelevanceEvaluator(BaseEvaluator):
    """Evaluates whether retrieved context chunks are relevant to the query.

    Only uses the query and context — the response is deliberately excluded
    so this metric captures retriever quality independently of the generator.

    Flagged claims contain descriptions of irrelevant or off-topic chunks.

    Example:
        >>> evaluator = ContextRelevanceEvaluator()
        >>> score = evaluator.evaluate(case, provider)
        >>> print(score.score, score.reasoning)
    """

    @property
    def metric(self) -> EvalMetric:
        return EvalMetric.CONTEXT_RELEVANCE

    def _build_prompt(self, case: EvalCase) -> str:
        # Deliberately excludes case.response — this metric is retriever-only.
        return _PROMPT_TEMPLATE.format(
            query=case.query,
            context=self._format_context(case.context),
            schema=_JSON_SCHEMA_INSTRUCTION,
        )