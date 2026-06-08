"""
Tests for rag_eval/providers/ollama.py

All HTTP calls are mocked — no Ollama instance required.
Covers:
  - Happy path initialisation and model availability check
  - ProviderError when Ollama is not running
  - ProviderError when the requested model is not pulled
  - complete() response parsing and token extraction
  - Timeout and connection error handling
  - Temperature is fixed at 0.0 in the payload
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from rag_eval.providers.base import ProviderError
from rag_eval.providers.ollama import OllamaProvider
from rag_eval.types import LLMResponse, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tags_response(models: list[str]) -> MagicMock:
    """Mock a successful GET /api/tags response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "models": [{"name": m} for m in models]
    }
    response.raise_for_status = MagicMock()
    return response


def make_generate_response(
    text: str = "Paris",
    prompt_tokens: int = 80,
    output_tokens: int = 20,
) -> MagicMock:
    """Mock a successful POST /api/generate response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "response": text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "done": True,
    }
    response.raise_for_status = MagicMock()
    return response


def make_provider(model: str = "qwen2.5:7b", available: list[str] | None = None) -> OllamaProvider:
    """Build an OllamaProvider with a mocked connection check."""
    if available is None:
        available = [model]
    with patch("rag_eval.providers.ollama.requests.get") as mock_get:
        mock_get.return_value = make_tags_response(available)
        return OllamaProvider(model=model)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestOllamaProviderInit:
    def test_happy_path(self):
        provider = make_provider("qwen2.5:7b")
        assert provider.model_name == "qwen2.5:7b"

    def test_custom_base_url(self):
        with patch("rag_eval.providers.ollama.requests.get") as mock_get:
            mock_get.return_value = make_tags_response(["qwen2.5:7b"])
            provider = OllamaProvider(model="qwen2.5:7b", base_url="http://192.168.1.10:11434")
        assert "192.168.1.10" in provider._base_url

    def test_ollama_not_running_raises(self):
        with patch("rag_eval.providers.ollama.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()
            with pytest.raises(ProviderError, match="not running"):
                OllamaProvider(model="qwen2.5:7b")

    def test_model_not_pulled_raises(self):
        with patch("rag_eval.providers.ollama.requests.get") as mock_get:
            mock_get.return_value = make_tags_response(["llama3.2:3b"])
            with pytest.raises(ProviderError, match="not pulled"):
                OllamaProvider(model="qwen2.5:7b")

    def test_model_not_pulled_error_suggests_pull_command(self):
        with patch("rag_eval.providers.ollama.requests.get") as mock_get:
            mock_get.return_value = make_tags_response(["llama3.2:3b"])
            with pytest.raises(ProviderError, match="ollama pull qwen2.5:7b"):
                OllamaProvider(model="qwen2.5:7b")

    def test_model_matched_without_tag(self):
        """'qwen2.5:7b' should match a model listed as 'qwen2.5:7b'."""
        provider = make_provider("qwen2.5:7b", available=["qwen2.5:7b"])
        assert provider.model_name == "qwen2.5:7b"

    def test_repr(self):
        provider = make_provider()
        assert "OllamaProvider" in repr(provider)
        assert "qwen2.5:7b" in repr(provider)

    def test_cost_constants_are_zero(self):
        provider = make_provider()
        cost = provider.estimate_cost(TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost == 0.0


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestOllamaProviderComplete:
    @pytest.fixture()
    def provider(self):
        return make_provider()

    def test_returns_llm_response(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response("The capital is Paris.")
            result = provider.complete("What is the capital of France?")
        assert isinstance(result, LLMResponse)

    def test_text_extracted(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response("The capital is Paris.")
            result = provider.complete("prompt")
        assert result.text == "The capital is Paris."

    def test_token_usage_extracted(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response(
                prompt_tokens=120, output_tokens=40
            )
            result = provider.complete("prompt")
        assert result.usage.input_tokens == 120
        assert result.usage.output_tokens == 40
        assert result.usage.total_tokens == 160

    def test_latency_is_positive(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response()
            result = provider.complete("prompt")
        assert result.latency_ms >= 0.0

    def test_temperature_zero_in_payload(self, provider):
        """Judge scores must be deterministic — temperature must be 0."""
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response()
            provider.complete("prompt")
            call_kwargs = mock_post.call_args
            payload = call_kwargs[1]["json"]
        assert payload["options"]["temperature"] == 0.0

    def test_stream_false_in_payload(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.return_value = make_generate_response()
            provider.complete("prompt")
            payload = mock_post.call_args[1]["json"]
        assert payload["stream"] is False

    def test_connection_error_raises_provider_error(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()
            with pytest.raises(ProviderError, match="Could not connect"):
                provider.complete("prompt")

    def test_timeout_raises_provider_error(self, provider):
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()
            with pytest.raises(ProviderError, match="timed out"):
                provider.complete("prompt")

    def test_null_token_counts_default_to_zero(self, provider):
        """Ollama may omit token counts on some responses."""
        with patch("rag_eval.providers.ollama.requests.post") as mock_post:
            response = make_generate_response()
            response.json.return_value = {"response": "ok", "done": True}
            mock_post.return_value = response
            result = provider.complete("prompt")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0