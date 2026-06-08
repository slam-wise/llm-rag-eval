"""
Reporter — formats and exports evaluation results.

Three output formats:
    print_summary()  Rich console table with colour-coded scores
    to_json()        Full EvalSummary serialised to JSON (indented)
    to_csv()         One row per EvalCase, one column per metric score

Colour thresholds (configurable via Reporter constructor):
    score >= 0.8  →  green
    score >= 0.5  →  yellow
    score <  0.5  →  red
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from rag_eval.types import EvalMetric, EvalSummary

# Display order and column labels for the four built-in metrics.
_METRIC_LABELS: dict[EvalMetric, str] = {
    EvalMetric.FAITHFULNESS: "Faithfulness",
    EvalMetric.CONTEXT_RELEVANCE: "Context Rel.",
    EvalMetric.ANSWER_RELEVANCE: "Answer Rel.",
    EvalMetric.HALLUCINATION: "Hallucination",
}


class Reporter:
    """Formats and exports an EvalSummary in multiple output formats.

    Args:
        high_threshold: Scores >= this value are coloured green. Default 0.8.
        low_threshold: Scores < this value are coloured red. Default 0.5.

    Example:
        >>> reporter = Reporter()
        >>> reporter.print_summary(summary, model_name="gemini-2.0-flash")
        >>> reporter.to_json(summary, "results/run_01.json")
        >>> reporter.to_csv(summary, "results/run_01.csv")
    """

    def __init__(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._console = Console()

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(
        self,
        summary: EvalSummary,
        model_name: str = "",
    ) -> None:
        """Print a colour-coded results table to the console.

        Args:
            summary: The EvalSummary to display.
            model_name: Optional model name shown in the table title.
        """
        # Determine which metrics are present (respects custom evaluator subsets).
        metrics = list(
            {m: None for r in summary.results for m in r.scores}.keys()
        )
        # Sort by the canonical display order.
        metrics.sort(key=lambda m: list(_METRIC_LABELS).index(m) if m in _METRIC_LABELS else 99)

        title = "RAG Evaluation Results"
        if model_name:
            title += f" — {model_name}"

        table = Table(
            title=title,
            box=box.ROUNDED,
            show_footer=True,
            footer_style="bold",
        )

        table.add_column("Case ID", style="dim", footer="MEAN")
        for metric in metrics:
            label = _METRIC_LABELS.get(metric, metric.value)
            mean = summary.mean_scores.get(metric)
            footer_str = self._fmt_score(mean) if mean is not None else "—"
            table.add_column(
                label,
                justify="center",
                footer=footer_str,
                footer_style=self._score_style(mean) if mean is not None else "",
            )

        for result in summary.results:
            row = [result.case_id]
            for metric in metrics:
                if metric in result.scores:
                    s = result.scores[metric].score
                    row.append(f"[{self._score_style(s)}]{self._fmt_score(s)}[/]")
                else:
                    row.append("—")
            table.add_row(*row)

        self._console.print()
        self._console.print(table)
        self._console.print(self._stats_line(summary))
        self._console.print()

    # ------------------------------------------------------------------
    # File exports
    # ------------------------------------------------------------------

    def to_json(self, summary: EvalSummary, path: str | Path) -> Path:
        """Serialise the full EvalSummary to a JSON file.

        Args:
            summary: The EvalSummary to serialise.
            path: Output file path. Parent directories are created if needed.

        Returns:
            The resolved output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

    def to_csv(self, summary: EvalSummary, path: str | Path) -> Path:
        """Export per-case scores to a CSV file.

        One row per EvalCase. Columns: case_id, one column per evaluated metric.
        Scores are rounded to 4 decimal places.

        Args:
            summary: The EvalSummary to export.
            path: Output file path. Parent directories are created if needed.

        Returns:
            The resolved output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Collect all metric names present across results.
        metrics: list[EvalMetric] = list(
            {m: None for r in summary.results for m in r.scores}.keys()
        )
        metrics.sort(key=lambda m: list(_METRIC_LABELS).index(m) if m in _METRIC_LABELS else 99)

        headers = ["case_id"] + [m.value for m in metrics]

        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for result in summary.results:
                row: dict[str, str | float] = {"case_id": result.case_id}
                for metric in metrics:
                    row[metric.value] = (
                        round(result.scores[metric].score, 4)
                        if metric in result.scores
                        else ""
                    )
                writer.writerow(row)

            # Append a MEAN row at the bottom.
            mean_row: dict[str, str | float] = {"case_id": "MEAN"}
            for metric in metrics:
                mean = summary.mean_scores.get(metric)
                mean_row[metric.value] = round(mean, 4) if mean is not None else ""
            writer.writerow(mean_row)

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_style(self, score: float) -> str:
        """Return a rich colour style string for a given score."""
        if score >= self.high_threshold:
            return "green"
        if score >= self.low_threshold:
            return "yellow"
        return "red"

    @staticmethod
    def _fmt_score(score: float) -> str:
        return f"{score:.2f}"

    def _stats_line(self, summary: EvalSummary) -> str:
        """Build the footer statistics line."""
        tokens = summary.total_usage.total_tokens
        latency_s = summary.total_latency_ms / 1000
        avg_latency = latency_s / summary.total_cases if summary.total_cases else 0
        cost = summary.total_estimated_cost_usd

        return (
            f"  [dim]Cases: {summary.total_cases}"
            f"  |  Tokens: {tokens:,}"
            f"  |  Avg latency: {avg_latency:.1f}s"
            f"  |  Est. cost: ${cost:.4f}[/dim]"
        )