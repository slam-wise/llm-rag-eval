"""
Answer relevance evaluator.

Measures whether the response directly and completely addresses the query.
This metric is independent of context — it evaluates the generator's ability
to stay on topic and provide a complete answer, regardless of what the
retriever returned.

Common failure modes caught by this evaluator:
    - Fluent non-answers: the response sounds good but doesn't answer the question
    - Partial answers: only some aspects of a multi-part query are addressed
    - Topic drift: the response addresses a related but different question
    - Refusals or deflections where an answer was possible

Score interpretation:
    1.0  Response directly and completely answers the query
    0.75 Response mostly answers the query with minor omissions
    0.5  Response partially answers the query; key aspects are missing
    0.25 Response barely addresses the query; mostly tangential
    0.0  Response does not address the query at all
"""

from __future__ import annotations

from rag_eval.evaluators.base import BaseEvaluator, _JSON_SCHEMA_INSTRUCTION
from rag_eval.types import EvalCase, EvalMetric

_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing the RELEVANCE of an AI-generated response to a user query.

DEFINITION
Answer relevance measures how directly and completely the RESPONSE addresses \
the QUERY. The retrieved context is not considered — this metric evaluates \
whether the generator answered the right question, not whether it used the \
context correctly.

─────────────────────────────────────────
QUERY:
{query}

RESPONSE:
{response}
─────────────────────────────────────────

EVALUATION STEPS
1. Identify all distinct questions or information needs expressed in the QUERY.
2. For each, check whether the RESPONSE provides a direct and complete answer.
3. Penalise responses that:
   - Answer a different (even if related) question than what was asked
   - Address only some parts of a multi-part query
   - Are evasive, overly hedged, or deflect without answering
   - Contain mostly padding with little substantive content
4. Do NOT penalise for using different words than the query — paraphrasing is fine.
5. Do NOT consider whether the response is factually correct, only whether
   it attempts to address the right question.

SCORING RUBRIC
1.0  Response directly and completely addresses all aspects of the query
0.75 Response mostly answers the query with one minor omission
0.5  Response partially answers the query; one or more aspects are missing
0.25 Response barely addresses the query; mostly tangential content
0.0  Response does not address the query at all

{schema}"""


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Evaluates whether the response directly addresses the user query.

    Only uses the query and response — context is deliberately excluded so
    this metric measures generator focus independently of retrieval quality.

    Flagged claims contain descriptions of query aspects left unaddressed.

    Example:
        >>> evaluator = AnswerRelevanceEvaluator()
        >>> score = evaluator.evaluate(case, provider)
        >>> print(score.score, score.reasoning)
    """

    @property
    def metric(self) -> EvalMetric:
        return EvalMetric.ANSWER_RELEVANCE

    def _build_prompt(self, case: EvalCase) -> str:
        # Deliberately excludes case.context — this metric is generator-only.
        return _PROMPT_TEMPLATE.format(
            query=case.query,
            response=case.response,
            schema=_JSON_SCHEMA_INSTRUCTION,
        )