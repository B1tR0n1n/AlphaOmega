"""
The four test arms.

Each arm is a callable: `arm(provider, case_prompt) -> (final_response, trace)`.

The trace is a dict suitable for serialization; it's rich for `axiom-arch`
(full pipeline reasoning) and minimal for the control arms (just the raw
system+user that was sent).
"""

from pathlib import Path

from alphaomega import AxiomPipeline
from alphaomega.config import MAX_TOKENS
from alphaomega.providers import LLMProvider

HARNESS_PROMPTS = Path(__file__).parent / "system_prompts"


def _control_arm(system_prompt: str, provider: LLMProvider, user: str) -> tuple[str, dict]:
    """Shared implementation for all three control arms."""
    response = provider.complete(
        system=system_prompt,
        user=user,
        max_tokens=MAX_TOKENS,
        cache_system=True,
    )
    return response, {"system_prompt_chars": len(system_prompt), "mode": "control"}


def neutral_arm(provider: LLMProvider, user: str) -> tuple[str, dict]:
    sp = (HARNESS_PROMPTS / "neutral.txt").read_text().strip()
    return _control_arm(sp, provider, user)


def creed_arm(provider: LLMProvider, user: str) -> tuple[str, dict]:
    sp = (HARNESS_PROMPTS / "creed.txt").read_text().strip()
    return _control_arm(sp, provider, user)


def axiom_naive_arm(provider: LLMProvider, user: str) -> tuple[str, dict]:
    """Prompt-only axiom arm — same system prompt the pipeline uses, no pipeline."""
    sp = (Path(__file__).parent.parent / "alphaomega" / "prompts" / "generator_axiom.txt").read_text().strip()
    return _control_arm(sp, provider, user)


def axiom_arch_arm(provider: LLMProvider, user: str) -> tuple[str, dict]:
    """Full AlphaOmega pipeline."""
    pipeline = AxiomPipeline(provider=provider)
    trace = pipeline.run(user)
    return trace.final_response, {
        "mode": "architecture",
        "trace": trace.to_dict(),
    }


ARMS = {
    "neutral":      neutral_arm,
    "creed":        creed_arm,
    "axiom_naive":  axiom_naive_arm,
    "axiom_arch":   axiom_arch_arm,
}
