# LLM RAG Eval

**LLM evaluation framework for RAG pipelines, focused on faithfulness and hallucination detection.**

Most LLM eval tools are generic. This one is scoped specifically to the failure modes that actually matter in RAG systems: does the response stay grounded in the retrieved context, did the retriever fetch relevant chunks, does the response answer what was asked, and — the hard one — does the model fabricate specific facts?

Evaluation is done by an LLM judge that returns a score, a chain-of-thought explanation, and a list of specific flagged claims. Not just a number.

---

## Why RAG evaluation is different

A RAG pipeline has two independently-failable components: a retriever and a generator. Generic eval tools conflate them. This framework separates them with four metrics that each isolate a specific failure mode:

| Metric | What it measures | Which component |
|---|---|---|
| **Faithfulness** | Are all claims in the response grounded in the context? | Generator |
| **Context Relevance** | Did the retriever fetch chunks relevant to the query? | Retriever |
| **Answer Relevance** | Does the response actually address the question? | Generator |
| **Hallucination** | Does the response fabricate or contradict specific facts? | Generator |

**Faithfulness and hallucination are not the same thing.** This distinction is the core idea of the framework:

- A response can add correct general knowledge not present in the context → low faithfulness, no hallucination
- A response can invent a specific date or name → high faithfulness on other claims, but hallucination detected

The sample dataset includes a case (`unfaithful_adds_knowledge_01`) designed specifically to surface this split.

---

## Sample output

Run against the bundled 8-case dataset using `qwen2.5:7b` as the judge:

```
                        RAG Evaluation Results — qwen2.5:7b
╭──────────────────────────────┬──────────────┬──────────────┬─────────────┬───────────────╮
│ Case ID                      │ Faithfulness │ Context Rel. │ Answer Rel. │ Hallucination │
├──────────────────────────────┼──────────────┼──────────────┼─────────────┼───────────────┤
│ ideal_01                     │     1.00     │     0.75     │    1.00     │     1.00      │
│ hallucination_number_01      │     0.25     │     1.00     │    1.00     │     0.50      │
│ context_irrelevant_01        │     0.00     │     0.00     │    0.75     │     1.00      │
│ fluent_non_answer_01         │     0.00     │     0.00     │    0.25     │     0.75      │
│ contradiction_fact_01        │     0.75     │     0.75     │    1.00     │     0.50      │
│ unfaithful_adds_knowledge_01 │     0.75     │     1.00     │    1.00     │     1.00      │
│ partial_answer_01            │     1.00     │     0.95     │    0.75     │     1.00      │
│ multi_chunk_synthesis_01     │     1.00     │     0.85     │    1.00     │     1.00      │
├──────────────────────────────┼──────────────┼──────────────┼─────────────┼───────────────┤
│ MEAN                         │     0.59     │     0.66     │    0.84     │     0.84      │
╰──────────────────────────────┴──────────────┴──────────────┴─────────────┴───────────────╯
  Cases: 8  |  Tokens: 22,409  |  Avg latency: 18.9s  |  Est. cost: $0.0000
```

Notice `unfaithful_adds_knowledge_01`: faithfulness 0.75, hallucination 1.00. The response added information beyond the context (low faithfulness) but didn't fabricate anything (no hallucination). The two metrics disagree — which is the point.

---

## Setup

### Option A — Ollama (recommended)

No API key, no rate limits, runs fully local.

**1. Install Ollama**

Download from [ollama.com](https://ollama.com) and install. Ollama starts automatically after installation.

**2. Pull a judge model**

```bash
ollama pull qwen2.5:7b
```

`qwen2.5:7b` (4.7 GB) is recommended for judge tasks — it produces reliable structured JSON output. For a lighter option, `llama3.2:3b` (2.0 GB) works but gives noisier scores.

**3. Install the package**

```bash
git clone https://github.com/slam-wise/llm-rag-eval
cd llm-rag-eval
pip install -e ".[dev]"
```

**4. Run the demo**

```bash
python examples/basic_eval.py
```

---

### Option B — Gemini (cloud)

Uses Google AI Studio's free tier. Requires an API key but no local model download.

**1. Get an API key**

Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a key in a dedicated project (no billing account needed).

**2. Configure your environment**

```bash
cp .env.example .env
# Add your key to .env:  GEMINI_API_KEY=your_key_here
```

**3. Install**

```bash
pip install -e ".[dev]"
```

**4. Switch the provider in `examples/basic_eval.py`**

```python
PROVIDER = "gemini"
```

> **Free tier limits:** `gemini-2.5-flash` allows 5 requests/minute and 250 requests/day.
> With 4 evaluators × 8 cases = 32 API calls per run, set `REQUEST_DELAY_S = 13.0`
> to stay within the RPM limit.

---

## Usage

### Running an evaluation

```python
from rag_eval.pipeline import EvalPipeline
from rag_eval.providers import OllamaProvider
from rag_eval.evaluators import FaithfulnessEvaluator, HallucinationEvaluator
from rag_eval.reporter import Reporter
from rag_eval.types import EvalCase

dataset = [
    EvalCase(
        id="case_01",
        query="What year was the Eiffel Tower completed?",
        context=["The Eiffel Tower was constructed from 1887 to 1889."],
        response="The Eiffel Tower was completed in 1889.",
    )
]

pipeline = EvalPipeline(
    provider=OllamaProvider(model="qwen2.5:7b"),
    evaluators=[FaithfulnessEvaluator(), HallucinationEvaluator()],
)

summary = pipeline.run(dataset)

Reporter().print_summary(summary, model_name="qwen2.5:7b")
Reporter().to_json(summary, "results.json")
Reporter().to_csv(summary, "results.csv")
```

### Loading a dataset from JSON

```python
import json
from rag_eval.types import EvalCase

cases = [EvalCase(**item) for item in json.loads(Path("my_dataset.json").read_text())]
```

Each JSON object requires `id`, `query`, `context` (list of strings), and `response`. An optional `reference` field accepts a ground-truth answer for future reference-based metrics.

### Running a subset of evaluators

```python
from rag_eval.evaluators import ContextRelevanceEvaluator, AnswerRelevanceEvaluator

pipeline = EvalPipeline(
    provider=OllamaProvider(),
    evaluators=[ContextRelevanceEvaluator(), AnswerRelevanceEvaluator()],
)
```

---

## Project structure

```
rag_eval/
├── types.py              Pydantic v2 models: EvalCase, EvalResult, EvalSummary, ...
├── pipeline.py           Orchestrates evaluators across a dataset
├── reporter.py           Console table (rich), JSON export, CSV export
├── providers/
│   ├── base.py           Abstract BaseLLMProvider — implement to add any provider
│   ├── gemini.py         Google Gemini via google-genai SDK
│   └── ollama.py         Local inference via Ollama REST API
└── evaluators/
    ├── base.py           Abstract BaseEvaluator, JSON parsing, sanitisation
    ├── faithfulness.py
    ├── context_relevance.py
    ├── answer_relevance.py
    └── hallucination.py

examples/
├── basic_eval.py         End-to-end demo with provider switching
└── sample_dataset.json   8 hand-crafted cases covering known RAG failure modes

tests/                    145 tests, zero external calls (all providers mocked)
```

---

## Adding a new provider

Subclass `BaseLLMProvider` and implement two things:

```python
from rag_eval.providers.base import BaseLLMProvider, ProviderError
from rag_eval.types import LLMResponse, TokenUsage
import time

class OpenAIProvider(BaseLLMProvider):
    cost_per_million_input_tokens  = 0.15   # gpt-4o-mini
    cost_per_million_output_tokens = 0.60

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model_name = model
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
```

Pass it straight into `EvalPipeline` — nothing else changes.

---

## Sample dataset

The bundled `examples/sample_dataset.json` contains 8 cases, each targeting a specific failure mode:

| Case ID | Failure mode | Expected pattern |
|---|---|---|
| `ideal_01` | Baseline — correct answer, relevant context | All metrics high |
| `hallucination_number_01` | Wrong specific number in response (450m vs 330m) | Faithfulness ↓, Hallucination ↓ |
| `context_irrelevant_01` | Retriever returns off-topic chunks | Context Relevance 0.0, Faithfulness 0.0 |
| `fluent_non_answer_01` | Response sounds good but ignores the question | Answer Relevance ↓ |
| `contradiction_fact_01` | Response contradicts a fact in context | Hallucination ↓ |
| `unfaithful_adds_knowledge_01` | Adds real knowledge not in context | Faithfulness ↓, Hallucination stays high |
| `partial_answer_01` | Two-part question, only one part answered | Answer Relevance ↓, Faithfulness high |
| `multi_chunk_synthesis_01` | Two relevant chunks synthesised correctly | All metrics high |

---

## Design decisions

**LLM-as-judge with chain-of-thought.** Each evaluator prompts the judge to reason step by step before scoring. The reasoning and flagged claims are returned alongside the score — failures are interpretable, not just a number.

**Provider abstraction.** `BaseLLMProvider` requires only `model_name` and `complete()`. Swapping from Ollama to Gemini to OpenAI is a one-line change in the calling code.

**Pydantic v2 throughout.** All inputs, outputs, and intermediate results are validated models. `EvalSummary.model_dump()` serialises the entire run to JSON with no extra code.

**Stateless evaluators.** Each evaluator is a class with a single `evaluate(case, provider)` method and no shared state. They are independently testable and trivially parallelisable.

---

## Troubleshooting

**`ProviderError: Ollama is not running`**
Ollama may not have started automatically. Run `ollama serve` in a terminal.

**`ProviderError: Model 'qwen2.5:7b' is not pulled`**
Run `ollama pull qwen2.5:7b` and wait for the download to complete.

**`429 RESOURCE_EXHAUSTED` (Gemini)**
You have hit the free tier rate limit. Set `REQUEST_DELAY_S = 13.0` in `basic_eval.py` for `gemini-2.5-flash` (5 RPM limit). If the daily cap (250 RPD) is hit, wait until midnight Pacific for the quota to reset.

**`EvaluatorError: Failed to parse judge response as JSON`**
The judge model returned malformed JSON. The parser handles the most common cases (markdown fences, smart quotes, double-quoted array items) automatically. If it still fails, the raw response is included in the error message for inspection. Switching to a larger judge model usually resolves persistent failures.

---

## Requirements

- Python 3.14+
- [Ollama](https://ollama.com) (for local inference) **or** a Gemini API key (for cloud)
- See `requirements.txt` for Python dependencies

---

## Author

slamwise

[slam-wise](https://github.com/slam-wise)

---

## License

MIT
