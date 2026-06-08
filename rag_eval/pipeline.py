"""
EvalPipeline — orchestrates evaluation runs across a dataset.

Typical usage:

    from rag_eval.pipeline import EvalPipeline
    from rag_eval.providers import GeminiProvider
    from rag_eval.evaluators import FaithfulnessEvaluator, HallucinationEvaluator

    pipeline = EvalPipeline(
        provider=GeminiProvider(),
        evaluators=[FaithfulnessEvaluator(), HallucinationEvaluator()],
    )
    summary = pipeline.run(dataset)

Omitting `evaluators` runs all four built-in evaluators.
"""

from __future__ import annotations

from rich.progress import track

from rag_eval.evaluators import (
    AnswerRelevanceEvaluator,
    BaseEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
)
from rag_eval.providers.base import BaseLLMProvider
from rag_eval.types import EvalCase, EvalMetric, EvalResult, EvalSummary, TokenUsage

# Default evaluator set — runs when the caller doesn't specify evaluators.
_DEFAULT_EVALUATORS: list[BaseEvaluator] = [
    FaithfulnessEvaluator(),
    ContextRelevanceEvaluator(),
    AnswerRelevanceEvaluator(),
    HallucinationEvaluator(),
]


class EvalPipeline:
    """Runs a list of evaluators over a dataset and aggregates the results.

    Args:
        provider: The LLM provider to use as judge for all evaluators.
        evaluators: Evaluators to run. Defaults to all four built-in evaluators
                    if not specified.

    Attributes:
        provider: The configured LLM provider.
        evaluators: The active evaluator list.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        evaluators: list[BaseEvaluator] | None = None,
    ) -> None:
        self.provider = provider
        self.evaluators = evaluators if evaluators is not None else list(_DEFAULT_EVALUATORS)

        if not self.evaluators:
            raise ValueError("evaluators list must not be empty.")

    def run(
        self,
        dataset: list[EvalCase],
        show_progress: bool = True,
    ) -> EvalSummary:
        """Evaluate every case in the dataset.

        For each EvalCase, every evaluator is called in sequence using the
        configured provider as judge. Results are aggregated into an
        EvalSummary containing per-case scores, mean scores per metric,
        and cumulative token / cost statistics.

        Args:
            dataset: List of EvalCases to evaluate.
            show_progress: If True, displays a rich progress bar. Set to
                           False for non-interactive environments (CI, notebooks).

        Returns:
            EvalSummary aggregating all results.

        Raises:
            ValueError: If the dataset is empty.
            EvaluatorError: If any evaluator fails to parse the judge response.
            ProviderError: If any underlying API call fails.
        """
        if not dataset:
            raise ValueError("dataset must contain at least one EvalCase.")

        results: list[EvalResult] = []

        cases = self._progress_wrap(dataset, show_progress)
        for case in cases:
            result = self._eval_case(case)
            results.append(result)

        mean_scores = self._compute_mean_scores(results)
        total_cost = sum(r.estimated_cost_usd for r in results)

        return EvalSummary(
            results=results,
            mean_scores=mean_scores,
            total_cases=len(results),
            total_estimated_cost_usd=total_cost,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_case(self, case: EvalCase) -> EvalResult:
        """Run all evaluators on a single EvalCase."""
        scores = {}
        for evaluator in self.evaluators:
            scores[evaluator.metric] = evaluator.evaluate(case, self.provider)

        # Sum usage across all evaluator calls for this case.
        usages = [s.usage for s in scores.values()]
        total_usage = usages[0]
        for u in usages[1:]:
            total_usage = total_usage + u

        estimated_cost = self.provider.estimate_cost(total_usage)

        return EvalResult(
            case_id=case.id,
            scores=scores,
            estimated_cost_usd=estimated_cost,
        )

    @staticmethod
    def _compute_mean_scores(results: list[EvalResult]) -> dict[EvalMetric, float]:
        """Compute mean score per metric across all results."""
        metric_scores: dict[EvalMetric, list[float]] = {}
        for result in results:
            for metric, score in result.scores.items():
                metric_scores.setdefault(metric, []).append(score.score)
        return {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in metric_scores.items()
        }

    @staticmethod
    def _progress_wrap(
        dataset: list[EvalCase],
        show_progress: bool,
    ):
        """Wrap dataset in a rich progress bar or return it as-is."""
        if not show_progress:
            return dataset
        return track(dataset, description="[cyan]Evaluating cases…")

    def __repr__(self) -> str:
        metrics = [e.metric.value for e in self.evaluators]
        return (
            f"EvalPipeline("
            f"provider={self.provider!r}, "
            f"evaluators={metrics})"
        )