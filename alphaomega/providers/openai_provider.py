"""OpenAI (GPT-4o / GPT-5) backend. Mirrors AnthropicProvider."""

import os
from functools import cached_property

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package not installed. Run: pip install --user --break-system-packages openai"
    ) from e


class OpenAIProvider:
    """
    OpenAI backend for the LLMProvider interface. OpenAI's API ignores the
    cache_system flag — prompt caching is automatic server-side on repeated
    prefixes, no cache_control block needed.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.name = f"openai:{model}"
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    @cached_property
    def _client(self) -> "openai.OpenAI":
        return openai.OpenAI(api_key=self._api_key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        cache_system: bool = False,  # ignored — OpenAI caches automatically
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
