"""
LLM Service.

External LLM API integration with token-efficient prompting.
Supports Groq, OpenAI, Anthropic, and custom endpoints.
"""

import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

from app.config import Settings, get_settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """
    LLM configuration with sensible defaults.

    Attributes:
        provider: LLM provider (groq, openai, anthropic, local).
        model: Model identifier.
        api_key: API key for the provider.
        base_url: Optional custom base URL.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-1.0).
        max_context_tokens: Max tokens for context window.
    """

    provider: Literal["groq", "openai", "anthropic", "local"] = "groq"
    model: str = "llama-3.1-8b-instant"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    max_context_tokens: int = 8000  # Reserve space for response


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """
    Token usage tracking for cost optimization.

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the response.
        total_tokens: Total tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """
    LLM response with metadata.

    Attributes:
        content: Generated text content.
        model: Model used for generation.
        usage: Token usage statistics.
        finish_reason: Why generation stopped.
    """

    content: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None


class LLMService:
    """
    Service for external LLM API interactions.

    Features:
    - Unified interface for multiple providers (Groq, OpenAI, Anthropic)
    - Token counting and budget management
    - Context truncation for cost efficiency
    - Streaming support for real-time responses

    Example:
        service = LLMService()
        response = await service.generate("What is RAG?")
    """

    __slots__ = ("_config", "_client", "_initialized")

    # Approximate chars per token for estimation
    CHARS_PER_TOKEN = 4

    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initialize the LLM service.

        Args:
            config: LLM configuration. Uses settings if not provided.
        """
        if config:
            self._config = config
        else:
            settings = get_settings()
            self._config = LLMConfig(
                provider=settings.llm_provider,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the LLM client.

        Call during application startup.
        """
        if self._initialized:
            return

        try:
            if self._config.provider == "groq":
                await self._init_groq()
            elif self._config.provider == "openai":
                await self._init_openai()
            elif self._config.provider == "anthropic":
                await self._init_anthropic()
            elif self._config.provider == "local":
                await self._init_local()
            else:
                raise LLMError(f"Unsupported provider: {self._config.provider}")

            self._initialized = True
            logger.info(f"LLM service initialized: {self._config.provider}/{self._config.model}")

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Failed to initialize LLM service: {e}",
                details={"provider": self._config.provider, "error": str(e)},
            ) from e

    async def _init_groq(self) -> None:
        """Initialize Groq client (uses OpenAI-compatible API)."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")

    async def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
            )
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")

    async def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._config.api_key)
        except ImportError:
            raise LLMError("anthropic package not installed. Run: pip install anthropic")

    async def _init_local(self) -> None:
        """Initialize local/custom endpoint client."""
        try:
            from openai import AsyncOpenAI

            # Use OpenAI-compatible API for local endpoints
            self._client = AsyncOpenAI(
                api_key=self._config.api_key or "not-needed",
                base_url=self._config.base_url or "http://localhost:8080/v1",
            )
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt / input text.
            system_prompt: Optional system instructions.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMError: If generation fails.
        """
        if not self._initialized:
            await self.initialize()

        max_tokens = max_tokens or self._config.max_tokens
        temperature = temperature if temperature is not None else self._config.temperature

        try:
            if self._config.provider == "anthropic":
                return await self._generate_anthropic(
                    prompt, system_prompt, max_tokens, temperature
                )
            else:
                # Groq, OpenAI, and local all use OpenAI-compatible API
                return await self._generate_openai(
                    prompt, system_prompt, max_tokens, temperature
                )

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Generation failed: {e}",
                details={"model": self._config.model, "error": str(e)},
            ) from e

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate using OpenAI-compatible API (Groq, OpenAI, local)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        usage = None

        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate using Anthropic API."""
        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )

        content = ""
        if response.content:
            content = response.content[0].text

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text token by token.

        Args:
            prompt: User prompt / input text.
            system_prompt: Optional system instructions.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Yields:
            Text chunks as they are generated.

        Raises:
            LLMError: If generation fails.
        """
        if not self._initialized:
            await self.initialize()

        max_tokens = max_tokens or self._config.max_tokens
        temperature = temperature if temperature is not None else self._config.temperature

        try:
            if self._config.provider == "anthropic":
                async for chunk in self._stream_anthropic(
                    prompt, system_prompt, max_tokens, temperature
                ):
                    yield chunk
            else:
                # Groq, OpenAI, and local all use OpenAI-compatible API
                async for chunk in self._stream_openai(
                    prompt, system_prompt, max_tokens, temperature
                ):
                    yield chunk

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Streaming failed: {e}",
                details={"model": self._config.model, "error": str(e)},
            ) from e

    async def _stream_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream using OpenAI-compatible API (Groq, OpenAI, local)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _stream_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream using Anthropic API."""
        async with self._client.messages.stream(
            model=self._config.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic. For accurate counts,
        use tiktoken with the specific model.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        # Try tiktoken for accurate count
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self._config.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except ImportError:
            # Fallback to character-based estimate
            return len(text) // self.CHARS_PER_TOKEN

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Input text.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text.
        """
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Estimate how much to keep
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.9)  # 10% safety margin

        return text[:target_chars].rsplit(" ", 1)[0] + "..."

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self._config.model

    def get_max_context_tokens(self) -> int:
        """Get maximum context tokens."""
        return self._config.max_context_tokens

    async def health_check(self) -> bool:
        """
        Check if the LLM service is accessible.

        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Simple test generation
            response = await self.generate(
                "Say 'ok'",
                max_tokens=5,
                temperature=0,
            )
            return len(response.content) > 0

        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
