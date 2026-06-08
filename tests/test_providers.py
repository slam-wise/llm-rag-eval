"""
Tests for rag_eval/providers/

The google-genai SDK is mocked throughout — no API key or network access required.
Tests validate:
  - BaseLLMProvider interface and estimate_cost logic
  - GeminiProvider initialisation (happy path and missing API key)
  - GeminiProvider.complete() response parsing
  - ProviderError wrapping on SDK failures
  - request_delay_s behaviour
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_eval.providers.base import BaseLLMProvider, ProviderError
from rag_eval.providers.gemini import GeminiProvider
from rag_eval.types import LLMResponse, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteProvider(BaseLLMProvider):
    """Minimal concrete subclass for testing BaseLLMProvider directly."""

    cost_per_million_input_tokens = 1.0
    cost_per_million_output_tokens = 2.0

    @property
    def model_name(self) -> str:
        return "test-model"

    def complete(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text="ok",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=50.0,
        )


def make_mock_gemini_response(
    text: str = "Paris",
    prompt_tokens: int = 100,
    response_tokens: int = 20,
    thoughts_tokens: int = 0,
) -> MagicMock:
    """Build a mock that mirrors the google-genai response shape."""
    response = MagicMock()
    response.text = text
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = response_tokens
    response.usage_metadata.thoughts_token_count = thoughts_tokens
    return response


# ---------------------------------------------------------------------------
# BaseLLMProvider
# ---------------------------------------------------------------------------


class TestBaseLLMProvider:
    def test_repr(self):
        provider = ConcreteProvider()
        assert repr(provider) == "ConcreteProvider(model='test-model')"

    def test_estimate_cost_zero_tokens(self):
        provider = ConcreteProvider()
        cost = provider.estimate_cost(TokenUsage(input_tokens=0, output_tokens=0))
        assert cost == 0.0

    def test_estimate_cost_input_only(self):
        provider = ConcreteProvider()
        # 1_000_000 input tokens at $1.00 / million = $1.00
        cost = provider.estimate_cost(TokenUsage(input_tokens=1_000_000, output_tokens=0))
        assert cost == pytest.approx(1.0)

    def test_estimate_cost_output_only(self):
        provider = ConcreteProvider()
        # 1_000_000 output tokens at $2.00 / million = $2.00
        cost = provider.estimate_cost(TokenUsage(input_tokens=0, output_tokens=1_000_000))
        assert cost == pytest.approx(2.0)

    def test_estimate_cost_combined(self):
        provider = ConcreteProvider()
        # 500k input @ $1/M + 500k output @ $2/M = $0.50 + $1.00 = $1.50
        cost = provider.estimate_cost(
            TokenUsage(input_tokens=500_000, output_tokens=500_000)
        )
        assert cost == pytest.approx(1.50)

    def test_estimate_cost_free_tier_is_zero(self):
        """Default cost constants are 0.0 — free-tier providers never charge."""

        class FreeTierProvider(BaseLLMProvider):
            @property
            def model_name(self) -> str:
                return "free-model"

            def complete(self, prompt: str) -> LLMResponse:  # pragma: no cover
                ...

        provider = FreeTierProvider()
        cost = provider.estimate_cost(TokenUsage(input_tokens=999_999, output_tokens=999_999))
        assert cost == 0.0

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# GeminiProvider — initialisation
# ---------------------------------------------------------------------------


class TestGeminiProviderInit:
    @patch("rag_eval.providers.gemini.genai.Client")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-abc123"})
    def test_happy_path(self, mock_client_cls):
        provider = GeminiProvider(env_path=None)
        mock_client_cls.assert_called_once_with(api_key="test-key-abc123")
        assert provider.model_name == "gemini-2.5-flash"

    @patch("rag_eval.providers.gemini.genai.Client")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-abc123"})
    def test_custom_model(self, mock_client_cls):
        provider = GeminiProvider(model="gemini-1.5-pro", env_path=None)
        assert provider.model_name == "gemini-1.5-pro"

    @patch("rag_eval.providers.gemini.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self, mock_load_dotenv):
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            GeminiProvider(env_path=None)

    @patch("rag_eval.providers.gemini.genai.Client")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-abc123"})
    def test_repr(self, mock_client_cls):
        provider = GeminiProvider(env_path=None)
        assert "GeminiProvider" in repr(provider)
        assert "gemini-2.5-flash" in repr(provider)

    @patch("rag_eval.providers.gemini.genai.Client")
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-abc123"})
    def test_cost_constants_are_zero(self, mock_client_cls):
        provider = GeminiProvider(env_path=None)
        cost = provider.estimate_cost(TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost == 0.0


# ---------------------------------------------------------------------------
# GeminiProvider — complete()
# ---------------------------------------------------------------------------


class TestGeminiProviderComplete:
    @pytest.fixture()
    def provider(self):
        with (
            patch("rag_eval.providers.gemini.genai.Client"),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
        ):
            p = GeminiProvider(env_path=None)
            p._client = MagicMock()
            yield p

    def test_returns_llm_response(self, provider):
        provider._client.models.generate_content.return_value = make_mock_gemini_response(
            text="The capital is Paris.",
            prompt_tokens=80,
            response_tokens=15,
        )
        result = provider.complete("What is the capital of France?")
        assert isinstance(result, LLMResponse)

    def test_text_extracted_correctly(self, provider):
        provider._client.models.generate_content.return_value = make_mock_gemini_response(
            text="The capital is Paris."
        )
        result = provider.complete("What is the capital of France?")
        assert result.text == "The capital is Paris."

    def test_token_usage_extracted(self, provider):
        # thoughts_tokens are hidden reasoning tokens — summed into output_tokens
        # so usage reflects true API consumption.
        provider._client.models.generate_content.return_value = make_mock_gemini_response(
            prompt_tokens=120,
            response_tokens=30,
            thoughts_tokens=50,
        )
        result = provider.complete("prompt")
        assert result.usage.input_tokens == 120
        assert result.usage.output_tokens == 80   # 30 candidates + 50 thoughts
        assert result.usage.total_tokens == 200

    def test_latency_is_positive(self, provider):
        provider._client.models.generate_content.return_value = make_mock_gemini_response()
        result = provider.complete("prompt")
        assert result.latency_ms >= 0.0

    def test_null_token_counts_default_to_zero(self, provider):
        """Defensive: usage_metadata fields can be None on some error responses."""
        response = make_mock_gemini_response()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None
        response.usage_metadata.thoughts_token_count = None
        provider._client.models.generate_content.return_value = response

        result = provider.complete("prompt")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_sdk_exception_raises_provider_error(self, provider):
        provider._client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        with pytest.raises(ProviderError, match="quota exceeded"):
            provider.complete("prompt")

    def test_request_delay_called(self, provider):
        provider._client.models.generate_content.return_value = make_mock_gemini_response()
        provider._request_delay_s = 0.05

        with patch("rag_eval.providers.gemini.time.sleep") as mock_sleep:
            provider.complete("prompt")
            mock_sleep.assert_called_once_with(0.05)

    def test_no_request_delay_by_default(self, provider):
        provider._client.models.generate_content.return_value = make_mock_gemini_response()

        with patch("rag_eval.providers.gemini.time.sleep") as mock_sleep:
            provider.complete("prompt")
            mock_sleep.assert_not_called()