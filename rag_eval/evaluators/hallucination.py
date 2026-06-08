"""
Hallucination evaluator.

Detects whether the response contains fabricated or contradictory claims —
specific facts invented by the model that are either absent from or
directly contradicted by the retrieved context.

How this differs from faithfulness:
    - Faithfulness (precision): are ALL claims grounded in context?
      Any information beyond the context lowers the score.
    - Hallucination (active fabrication): does the response INVENT specific
      facts or CONTRADICT the context? General knowledge or safe hedges are
      not penalised.

Example:
    Context says: "The Eiffel Tower was completed in 1889."
    Response says: "The Eiffel Tower was completed in 1887 and
                    stands 450 metres tall."

    Faithfulness: low — both figures differ from context.
    Hallucination: low — specific figures are contradicted or invented.

    Context says: "Paris is the capital of France."
    Response says: "Paris is the capital of France, a country in Europe."

    Faithfulness: 0.75 — "a country in Europe" is not in the context.
    Hallucination: 1.0 — "a country in Europe" is general knowledge,
                          not a fabrication.

Score interpretation (higher = less hallucination):
    1.0  No hallucinations — response contains no invented or contradictory facts
    0.75 One minor factual embellishment, not a direct contradiction
    0.5  At least one clearly fabricated specific claim
    0.25 Multiple hallucinated facts; context is actively contradicted
    0.0  Response is largely or entirely fabricated
"""

from __future__ import annotations

from rag_eval.evaluators.base import BaseEvaluator, _JSON_SCHEMA_INSTRUCTION
from rag_eval.types import EvalCase, EvalMetric

_PROMPT_TEMPLATE = """\
You are an expert evaluator detecting HALLUCINATIONS in an AI-generated response.

DEFINITION
A hallucination is a specific factual claim in the RESPONSE that is either:
  (a) directly CONTRADICTED by the CONTEXT, or
  (b) a specific invented detail (names, numbers, dates, quotes, events)
      that is ABSENT from the CONTEXT and cannot be considered general knowledge.

Do NOT flag:
  - Correct general knowledge not mentioned in the context (e.g. "France is in Europe")
  - Safe hedges ("may", "might", "approximately") unless they mask a wrong specific fact
  - Paraphrases or summaries that are faithful to the context's meaning
  - Minor stylistic additions that do not assert new facts

─────────────────────────────────────────
QUERY:
{query}

CONTEXT:
{context}

RESPONSE:
{response}
─────────────────────────────────────────

EVALUATION STEPS
1. Read the CONTEXT carefully. Note all specific facts: dates, numbers, names,
   quotes, causal claims, and quantitative assertions.
2. Read the RESPONSE and identify every specific factual assertion.
3. For each assertion in the RESPONSE, determine:
   (a) Is it directly supported by the CONTEXT? → Not a hallucination.
   (b) Is it contradicted by the CONTEXT? → Hallucination.
   (c) Is it a specific detail absent from the CONTEXT?
       - If it is unambiguous general knowledge → Not a hallucination.
       - If it is specific (a date, a number, a name) and not in the CONTEXT → Hallucination.
4. List every hallucination found.

SCORING RUBRIC (higher = better; 1.0 means no hallucination detected)
1.0  No hallucinations — every specific claim is supported or is general knowledge
0.75 One minor embellishment that is not a direct contradiction
0.5  At least one clearly hallucinated specific claim (wrong date, invented name, etc.)
0.25 Multiple hallucinations or at least one direct factual contradiction of the context
0.0  The response is largely or entirely fabricated

{schema}"""


class HallucinationEvaluator(BaseEvaluator):
    """Detects active hallucinations — fabricated or contradicted specific facts.

    Uses an LLM judge to identify claims that either contradict the context
    or assert specific invented details absent from the context.

    Unlike faithfulness, general knowledge additions are not penalised.
    Only specific fabricated facts (numbers, dates, names, quotes) that
    cannot be verified from the context lower the score.

    Flagged claims contain verbatim quotes of hallucinated assertions.

    Example:
        >>> evaluator = HallucinationEvaluator()
        >>> score = evaluator.evaluate(case, provider)
        >>> if score.flagged_claims:
        ...     print("Hallucinations detected:", score.flagged_claims)
    """

    @property
    def metric(self) -> EvalMetric:
        return EvalMetric.HALLUCINATION

    def _build_prompt(self, case: EvalCase) -> str:
        return _PROMPT_TEMPLATE.format(
            query=case.query,
            context=self._format_context(case.context),
            response=case.response,
            schema=_JSON_SCHEMA_INSTRUCTION,
        )