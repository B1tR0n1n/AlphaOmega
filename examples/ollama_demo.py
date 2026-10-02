"""
Stub for Phase 1b — show AlphaOmega wrapping a local open-source model
served via Ollama (e.g. Llama 3.1 8B, Mistral, Phi-4).

This demo matters most for the Gloo pitch: prove the architecture works
against models the user fully controls, not only closed APIs.

Usage:
    ollama pull llama3.1:8b
    ollama serve
    python examples/ollama_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from alphaomega.providers.ollama_provider import OllamaProvider  # noqa: F401
except ImportError:
    print(
        "OllamaProvider not implemented yet. Add alphaomega/providers/ollama_provider.py "
        "— ~30 lines using `requests` against the local Ollama HTTP API."
    )
    sys.exit(1)
