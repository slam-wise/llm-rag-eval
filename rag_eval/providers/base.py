"""
Abstract base class for LLM providers.

To add a new provider (e.g. OpenAI, Anthropic), subclass BaseLLMProvider,
implement `model_name` and `complete()`, and optionally set the class-level
cost constants. Nothing else in the framework needs to change.

Example skeleton for a new provider:

    from rag_eval.providers.base import BaseLLMProvider
    from rag_eval.types import LLMResponse, TokenUsage

    class OpenAIProvider(BaseLLMProvider):
        cost_per_million_input_tokens  = 0.15   # gpt-4o-mini, as of 2024
        cost_per_million_output_tokens = 0.60

        def __init__(self, model: str = "gpt-4o-mini"):
            self._model_name = model
            self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @property
        def model_name(self) -> str:
            return self._model_name

        def complete(self, prompt: str) -> LLMResponse:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_eval.types import LLMResponse, TokenUsage


class BaseLLMProvider(ABC):
    """Minimal interface every LLM provider must satisfy.

    The framework only ever calls `complete()` on a provider instance.
    Everything else — authentication, retry logic, model selection — is
    the provider's own responsibility.

    Cost constants
    --------------
    Override `cost_per_million_input_tokens` and
    `cost_per_million_output_tokens` in subclasses to enable cost tracking.
    Leave them at 0.0 for free-tier providers (the Gemini default).
    """

    cost_per_million_input_tokens: float = 0.0
    cost_per_million_output_tokens: float = 0.0

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier string as passed to the provider API."""
        ...

    @abstractmethod
    def complete(self, prompt: str) -> LLMResponse:
        """Send a plain-text prompt and return a structured response.

        Args:
            prompt: The full prompt string. Evaluators are responsible for
                    constructing well-formed prompts before calling this.

        Returns:
            LLMResponse containing the model's text output, token usage,
            and wall-clock latency.

        Raises:
            ProviderError: If the API call fails for any reason.
        """
        ...

    def estimate_cost(self, usage: TokenUsage) -> float:
        """Estimate the USD cost for a given token usage.

        Uses the class-level cost constants, which default to 0.0 so
        free-tier providers automatically report $0.00.

        Args:
            usage: Token counts to price.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (usage.input_tokens / 1_000_000) * self.cost_per_million_input_tokens
        output_cost = (usage.output_tokens / 1_000_000) * self.cost_per_million_output_tokens
        return round(input_cost + output_cost, 8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


class ProviderError(Exception):
    """Raised when an LLM provider call fails.

    Wraps provider-specific exceptions so the rest of the framework
    doesn't need to import provider SDKs to catch errors.
    """