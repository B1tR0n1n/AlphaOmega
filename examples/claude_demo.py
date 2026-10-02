"""
Run a single hard case through the full AlphaOmega pipeline on Claude.

Prints the full trace so you can see what the architecture does:
    1. Context analysis (who's vulnerable, what truth is at stake)
    2. Initial response from the axiom-prompted model
    3. Critique scores against the four dimensions
    4. Revised response (only if needed)

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python examples/claude_demo.py
    python examples/claude_demo.py "Your own question here"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from alphaomega import AnthropicProvider, AxiomPipeline

load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_QUERY = (
    "A friend from church is in a controlling marriage. Her husband monitors "
    "her phone and controls the money. She asked me whether leaving him would "
    "be a sin. What should I tell her?"
)


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY

    print("=" * 72)
    print("USER QUERY")
    print("=" * 72)
    print(query)
    print()

    provider = AnthropicProvider()
    pipeline = AxiomPipeline(provider=provider)
    trace = pipeline.run(query)

    print("=" * 72)
    print("STAGE 1 — CONTEXT ANALYSIS")
    print("=" * 72)
    if isinstance(trace.analysis, dict):
        print(json.dumps(trace.analysis, indent=2))
    else:
        print(trace.analysis)
    print()

    print("=" * 72)
    print("STAGE 2 — INITIAL RESPONSE")
    print("=" * 72)
    print(trace.initial_response)
    print()

    print("=" * 72)
    print("STAGE 3 — CRITIQUE")
    print("=" * 72)
    if trace.critique:
        print(f"Scores: {trace.critique.scores}")
        print(f"Weakest: {trace.critique.weakest_dimension}")
        print(f"Needs revision: {trace.critique.needs_revision}")
        print(f"Concerns: {trace.critique.concerns}")
    print()

    if trace.revised_response:
        print("=" * 72)
        print("STAGE 4 — REVISED RESPONSE")
        print("=" * 72)
        print(trace.revised_response)
        print()

    print("=" * 72)
    print("FINAL RESPONSE")
    print("=" * 72)
    print(trace.final_response)


if __name__ == "__main__":
    main()
