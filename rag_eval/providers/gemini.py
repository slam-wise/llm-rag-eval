"""
Google Gemini provider using the google-genai SDK.

Reads GEMINI_API_KEY from the environment (via .env).
Defaults to gemini-2.0-flash — the fastest free-tier model.

Free tier limits (verify at https://ai.google.dev/pricing):
    - 15 requests per minute
    - 1 million tokens per minute
    - 1,500 requests per day

If you hit rate limits during a large eval run, pass `request_delay_s`
to add a sleep between calls.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_eval.providers.base import BaseLLMProvider, ProviderError
from rag_eval.types import LLMResponse, TokenUsage

# Default model — free tier, fast, sufficient for LLM-as-judge tasks.
# Swap to "gemini-3.5-pro" for higher quality at the cost of lower rate limits.
_DEFAULT_MODEL = "gemini-2.5-flash"

# Conservative generation config for evaluation:
# - temperature=0.0  → deterministic, reproducible scores
# - max_output_tokens → judge needs room for detailed multi-step reasoning
_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=8192,
)


class GeminiProvider(BaseLLMProvider):
    """LLM provider backed by Google Gemini via the free AI Studio tier.

    Args:
        model: Gemini model identifier. Defaults to "gemini-2.0-flash".
        env_path: Path to a .env file. Defaults to .env in the working
                  directory. Pass None to rely solely on shell environment
                  variables.
        request_delay_s: Optional sleep (seconds) injected after every
                         complete() call. Useful for staying inside the
                         15 RPM free-tier limit during large eval runs.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.

    Example:
        >>> from rag_eval.providers import GeminiProvider
        >>> provider = GeminiProvider()
        >>> response = provider.complete("What is the capital of France?")
        >>> print(response.text)
    """

    # Gemini free tier has no per-token cost.
    cost_per_million_input_tokens: float = 0.0
    cost_per_million_output_tokens: float = 0.0

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        env_path: str | None = ".env",
        request_delay_s: float = 0.0,
    ) -> None:
        load_dotenv(env_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in environment.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Add your key from https://aistudio.google.com/app/apikey\n"
                "  3. Re-run."
            )

        self._model_name = model
        self._request_delay_s = request_delay_s
        self._client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str) -> LLMResponse:
        """Send a prompt to Gemini and return a structured response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            LLMResponse with text, token usage, and latency.

        Raises:
            ProviderError: Wraps any google-genai exception with a
                           human-readable message.
        """
        try:
            start = time.perf_counter()
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=_GENERATION_CONFIG,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if self._request_delay_s > 0:
                time.sleep(self._request_delay_s)

        except Exception as exc:
            raise ProviderError(
                f"Gemini API call failed for model '{self._model_name}': {exc}"
            ) from exc

        usage = TokenUsage(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=(response.usage_metadata.candidates_token_count or 0) + (response.usage_metadata.thoughts_token_count or 0),
        )

        return LLMResponse(
            text=response.text,
            usage=usage,
            latency_ms=round(latency_ms, 2),
        )