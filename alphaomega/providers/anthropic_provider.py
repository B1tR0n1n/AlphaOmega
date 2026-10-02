"""Anthropic Claude backend with prompt caching on repeated system prompts."""

import os
from functools import cached_property

import anthropic


class AnthropicProvider:
    """
    Uses the Anthropic SDK. Supports prompt caching via `cache_system=True`
    (useful for the harness, where the same system prompt is reused across
    all 15 cases per arm) and for pipeline stages whose system prompts are
    constant across calls.
    """

    def __init__(self, model: str = "claude-opus-4-7", api_key: str | None = None):
        self.model = model
        self.name = f"anthropic:{model}"
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @cached_property
    def _client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=self._api_key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        cache_system: bool = False,
    ) -> str:
        system_block: list[dict] | str
        if cache_system:
            system_block = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_block = system

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_block,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
