"""
Ollama provider for local LLM inference.

Requires Ollama to be running locally (https://ollama.com).
No API key needed — completely free, no rate limits.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull qwen2.5:7b
    3. Ollama starts automatically after install, or run: ollama serve

Recommended models for LLM-as-judge tasks:
    qwen2.5:7b   — 4.7 GB, best JSON output quality (recommended)
    llama3.2:3b  — 2.0 GB, fastest, good for quick iteration
    mistral:7b   — 4.1 GB, strong general reasoning

The Ollama REST API runs at http://localhost:11434 by default.
"""

from __future__ import annotations

import time

import requests

from rag_eval.providers.base import BaseLLMProvider, ProviderError
from rag_eval.types import LLMResponse, TokenUsage

_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama instance.

    Calls the Ollama REST API directly — no SDK, no API key, no rate limits.
    Temperature is fixed at 0.0 for deterministic, reproducible judge scores.

    Args:
        model: Ollama model tag. Defaults to "qwen2.5:7b".
               Run `ollama list` to see locally available models.
        base_url: Base URL of the Ollama server. Defaults to
                  "http://localhost:11434". Change if running Ollama
                  on a remote host or non-default port.
        timeout: Per-request timeout in seconds. Local inference can be
                 slow for long prompts; 120s is a safe default.

    Raises:
        ProviderError: If Ollama is not running or the model is not pulled.

    Example:
        >>> from rag_eval.providers import OllamaProvider
        >>> provider = OllamaProvider()
        >>> response = provider.complete("What is the capital of France?")
        >>> print(response.text)
    """

    # Local inference is free — cost constants stay at 0.0.
    cost_per_million_input_tokens: float = 0.0
    cost_per_million_output_tokens: float = 0.0

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._generate_url = f"{self._base_url}/api/generate"
        self._check_connection()

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str) -> LLMResponse:
        """Send a prompt to the local Ollama model and return a response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            LLMResponse with text, token usage, and latency.

        Raises:
            ProviderError: If the Ollama API call fails.
        """
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            start = time.perf_counter()
            response = requests.post(
                self._generate_url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000
            data = response.json()

        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                "Could not connect to Ollama. Is it running?\n"
                "  Start it with: ollama serve\n"
                f"  Expected at:   {self._base_url}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ProviderError(
                f"Ollama request timed out after {self._timeout}s. "
                "Try increasing the timeout parameter."
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Ollama API call failed for model '{self._model_name}': {exc}"
            ) from exc

        usage = TokenUsage(
            input_tokens=data.get("prompt_eval_count") or 0,
            output_tokens=data.get("eval_count") or 0,
        )

        return LLMResponse(
            text=data["response"],
            usage=usage,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        """Verify Ollama is reachable and the requested model is available.

        Raises:
            ProviderError: If Ollama is not running or the model is missing.
        """
        tags_url = f"{self._base_url}/api/tags"
        try:
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                "Ollama is not running. Start it with: ollama serve\n"
                f"  Expected at: {self._base_url}"
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self._base_url}: {exc}"
            ) from exc

        available = [m["name"] for m in response.json().get("models", [])]

        # Normalise: Ollama appends ":latest" if no tag is given.
        model_key = self._model_name if ":" in self._model_name else f"{self._model_name}:latest"
        if not any(m == model_key or m.startswith(self._model_name) for m in available):
            available_str = ", ".join(available) if available else "none"
            raise ProviderError(
                f"Model '{self._model_name}' is not pulled.\n"
                f"  Run: ollama pull {self._model_name}\n"
                f"  Available models: {available_str}"
            )