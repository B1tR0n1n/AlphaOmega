from .base import LLMProvider
from .anthropic_provider import AnthropicProvider

__all__ = ["LLMProvider", "AnthropicProvider"]

# OpenAIProvider is imported lazily — don't fail if `openai` isn't installed
def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
