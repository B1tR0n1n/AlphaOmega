"""
Stub for Phase 1b — show AlphaOmega wrapping GPT-4o.

Requires an OpenAIProvider implementation (not yet built). The pipeline
itself is model-agnostic; only providers/openai_provider.py needs to be
added for this to run.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/gpt4_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from alphaomega.providers.openai_provider import OpenAIProvider  # noqa: F401
except ImportError:
    print(
        "OpenAIProvider not implemented yet. Add alphaomega/providers/openai_provider.py "
        "mirroring anthropic_provider.py — ~25 lines using the openai SDK."
    )
    sys.exit(1)

# When OpenAIProvider lands, replace this block with the same shape as
# claude_demo.py, swapping the provider.
