#!/usr/bin/env python3
"""
basic_eval.py — end-to-end demo of the RAG evaluation framework.

Runs all four evaluators over the bundled sample dataset and prints
a colour-coded results table, then exports to JSON and CSV.

Usage:
    python examples/basic_eval.py

Configuration:
    Set PROVIDER below to "ollama" or "gemini".

    Ollama (recommended — free, local, no rate limits):
        1. Install from https://ollama.com
        2. Run: ollama pull qwen2.5:7b
        3. Set PROVIDER = "ollama"

    Gemini (cloud, requires API key):
        1. Copy .env.example to .env and add your GEMINI_API_KEY
        2. Set PROVIDER = "gemini"
        3. Free tier: 5 RPM / 250 RPD on gemini-2.5-flash
           Set REQUEST_DELAY_S = 13.0 to stay within rate limits
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval.evaluators import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
)
from rag_eval.pipeline import EvalPipeline
from rag_eval.providers import GeminiProvider, OllamaProvider
from rag_eval.reporter import Reporter
from rag_eval.types import EvalCase

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------

PROVIDER = "ollama"           # "ollama" or "gemini"

# Ollama settings
OLLAMA_MODEL = "qwen2.5:7b"  # run `ollama list` to see available models

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash"
REQUEST_DELAY_S: float = 13.0  # stay under 5 RPM free-tier limit

DATASET_PATH = Path(__file__).parent / "sample_dataset.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[EvalCase]:
    return [EvalCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def build_provider():
    if PROVIDER == "ollama":
        return OllamaProvider(model=OLLAMA_MODEL)
    elif PROVIDER == "gemini":
        return GeminiProvider(model=GEMINI_MODEL, request_delay_s=REQUEST_DELAY_S)
    else:
        raise ValueError(f"Unknown PROVIDER '{PROVIDER}'. Choose 'ollama' or 'gemini'.")


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
    print(f"\nLoading dataset from {DATASET_PATH.name}…")
    dataset = load_dataset(DATASET_PATH)
    print_dataset_summary(dataset)

    provider = build_provider()

    pipeline = EvalPipeline(
        provider=provider,
        evaluators=[
            FaithfulnessEvaluator(),
            ContextRelevanceEvaluator(),
            AnswerRelevanceEvaluator(),
            HallucinationEvaluator(),
        ],
    )

    print(f"\nProvider    : {PROVIDER}")
    print(f"Judge model : {provider.model_name}")
    print(f"Evaluators  : {[e.metric.value for e in pipeline.evaluators]}")
    print(f"Cases       : {len(dataset)}")
    print(f"API calls   : ~{len(dataset) * len(pipeline.evaluators)}\n")

    summary = pipeline.run(dataset, show_progress=True)

    reporter = Reporter()
    reporter.print_summary(summary, model_name=provider.model_name)

    json_path = reporter.to_json(summary, OUTPUT_DIR / "results.json")
    csv_path = reporter.to_csv(summary, OUTPUT_DIR / "results.csv")

    from rich.console import Console
    Console().print(
        f"  [dim]Exported → {json_path.relative_to(Path.cwd())}  "
        f"|  {csv_path.relative_to(Path.cwd())}[/dim]\n"
    )


if __name__ == "__main__":
    main()