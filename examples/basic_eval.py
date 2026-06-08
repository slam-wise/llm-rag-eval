#!/usr/bin/env python3
"""
basic_eval.py — end-to-end demo of the RAG evaluation framework.

Runs all four evaluators over the bundled sample dataset using Gemini
as the LLM judge, then prints a colour-coded results table and exports
to JSON and CSV.

Usage:
    python examples/basic_eval.py

Requirements:
    - GEMINI_API_KEY set in .env at the repo root
      (copy .env.example to .env, then add your key)

Rate limits (Gemini free tier):
    15 requests/minute, 1,500 requests/day.
    With 4 evaluators × 8 cases = 32 API calls per run.
    If you hit a 429 error, set REQUEST_DELAY_S below to 4.0 or higher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval.evaluators import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
)
from rag_eval.pipeline import EvalPipeline
from rag_eval.providers import GeminiProvider
from rag_eval.reporter import Reporter
from rag_eval.types import EvalCase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent / "sample_dataset.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Increase to 4.0+ if you hit 429 rate-limit errors on the free tier.
REQUEST_DELAY_S: float = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[EvalCase]:
    """Load a list of EvalCases from a JSON file."""
    cases = [EvalCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    return cases


def print_dataset_summary(dataset: list[EvalCase]) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(title="Dataset", box=box.SIMPLE, show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Query")
    table.add_column("Chunks", justify="center")
    for case in dataset:
        table.add_row(case.id, case.query[:60] + "…", str(len(case.context)))
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"\nLoading dataset from {DATASET_PATH.name}…")
    dataset = load_dataset(DATASET_PATH)
    print_dataset_summary(dataset)

    # ── Initialise provider and pipeline ────────────────────────────────────
    provider = GeminiProvider(request_delay_s=REQUEST_DELAY_S)

    pipeline = EvalPipeline(
        provider=GeminiProvider(model="gemini-2.5-flash", request_delay_s=REQUEST_DELAY_S),
        evaluators=[
            FaithfulnessEvaluator(),
            ContextRelevanceEvaluator(),
            AnswerRelevanceEvaluator(),
            HallucinationEvaluator(),
        ],
    )

    print(f"\nJudge model : {provider.model_name}")
    print(f"Evaluators  : {[e.metric.value for e in pipeline.evaluators]}")
    print(f"Cases       : {len(dataset)}")
    print(f"API calls   : ~{len(dataset) * len(pipeline.evaluators)}\n")

    # ── Run ─────────────────────────────────────────────────────────────────
    summary = pipeline.run(dataset, show_progress=True)

    # ── Report ──────────────────────────────────────────────────────────────
    reporter = Reporter()
    reporter.print_summary(summary, model_name=provider.model_name)

    # ── Export ──────────────────────────────────────────────────────────────
    json_path = reporter.to_json(summary, OUTPUT_DIR / "results.json")
    csv_path = reporter.to_csv(summary, OUTPUT_DIR / "results.csv")

    from rich.console import Console
    Console().print(
        f"  [dim]Exported → {json_path.relative_to(Path.cwd())}  "
        f"|  {csv_path.relative_to(Path.cwd())}[/dim]\n"
    )


if __name__ == "__main__":
    main()