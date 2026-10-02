"""
LLMProvider — the single interface the pipeline talks to.

Any backend (Anthropic, OpenAI, Ollama, llama.cpp) implements this. Swapping
backends is a one-line change: `pipeline = AxiomPipeline(provider=X())`.
"""

from typing import Protocol


class LLMProvider(Protocol):
    """
    Minimum surface the AlphaOmega pipeline needs from a model backend.

    `complete` returns the model's text completion for a given system prompt
    and user prompt. Caching, retries, and streaming are provider concerns —
    the pipeline doesn't care.
    """

    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        cache_system: bool = False,
    ) -> str:
        ...
