"""Provider-neutral chat operations used by application runtimes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from configs.settings import settings
from .model_factory import LLMFactory


class LLMRequestError(RuntimeError):
    """An LLM request failed with provider and model diagnostics attached."""

    def __init__(self, message: str, *, original: Exception):
        super().__init__(message)
        self.original = original


class LLMService:
    """Stream chat completions without exposing provider details to the UI."""

    def __init__(self, client: Any | None = None):
        self._client = client or LLMFactory.get_llm()

    @property
    def model_name(self) -> str:
        return LLMFactory.get_model_name()

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text deltas from the configured provider."""
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens or settings.RESPONSE_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as exc:
            info = LLMFactory.get_connection_info()
            raise LLMRequestError(
                "LLM request failed "
                f"(provider={info['provider']}, model={info['model']}, "
                f"endpoint={info['endpoint']}): {exc}",
                original=exc,
            ) from exc
