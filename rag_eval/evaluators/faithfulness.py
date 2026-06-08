"""
Faithfulness evaluator.

Measures whether every factual claim in the response is directly supported
by the provided context chunks.

Key distinction from hallucination:
    - Faithfulness is CONTEXT-RELATIVE. A claim can be true in the real world
      but still unfaithful if it goes beyond what the retrieved context states.
    - A low faithfulness score means the response adds information the retriever
      didn't provide, regardless of whether that information is correct.

Score interpretation:
    1.0  All claims traceable to the context
    0.75 One or two minor unsupported details
    0.5  Mix of supported and unsupported claims
    0.25 Mostly unsupported; context largely ignored
    0.0  No claims are grounded in the context
"""

from __future__ import annotations

from rag_eval.evaluators.base import BaseEvaluator, _JSON_SCHEMA_INSTRUCTION
from rag_eval.types import EvalCase, EvalMetric

_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing the FAITHFULNESS of an AI-generated response.

DEFINITION
Faithfulness measures whether every factual claim in the RESPONSE is directly \
supported by the CONTEXT. This is not the same as factual accuracy — a claim \
may be true in the world but still unfaithful if the CONTEXT does not state it.

─────────────────────────────────────────
QUERY:
{query}

CONTEXT:
{context}

RESPONSE:
{response}
─────────────────────────────────────────

EVALUATION STEPS
1. List every factual claim in the RESPONSE. Ignore filler phrases such as
   "based on the provided information" or "according to the context".
2. For each claim, check whether it is explicitly stated or unambiguously
   implied by the CONTEXT.
3. Mark claims that introduce any information not present in the CONTEXT
   as unsupported — even if they are factually correct in the real world.
4. Do not penalise hedged language ("may", "might", "could") unless it
   asserts a specific fact absent from the CONTEXT.

SCORING RUBRIC
1.0  Every claim is directly supported by the context
0.75 At most one minor claim goes slightly beyond the context
0.5  Roughly half the claims lack grounding in the context
0.25 Most claims are unsupported; the context is largely ignored
0.0  No claims are grounded in the context

{schema}"""


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates whether the response is fully grounded in the retrieved context.

    Uses an LLM judge to identify claims in the response that are not
    supported by the provided context chunks.

    Example:
        >>> evaluator = FaithfulnessEvaluator()
        >>> score = evaluator.evaluate(case, provider)
        >>> print(score.score, score.flagged_claims)
    """

    @property
    def metric(self) -> EvalMetric:
        return EvalMetric.FAITHFULNESS

    def _build_prompt(self, case: EvalCase) -> str:
        return _PROMPT_TEMPLATE.format(
            query=case.query,
            context=self._format_context(case.context),
            response=case.response,
            schema=_JSON_SCHEMA_INSTRUCTION,
        )