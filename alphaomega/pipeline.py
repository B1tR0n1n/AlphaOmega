"""
The AlphaOmega inference-time alignment pipeline.

Four stages, each a single call to the underlying LLM:
    analyze → generate → critique → (optional) revise

The pipeline returns both the final response and a full trace of what
happened at each stage, so applications can display reasoning transparently
and so training data can later be mined from the traces.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import PROMPTS_DIR, REVISION_THRESHOLD, MAX_TOKENS, MAX_REVISIONS
from .providers import LLMProvider


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


@dataclass
class CritiqueResult:
    scores: dict[str, int]
    weakest_dimension: str
    concerns: str
    needs_revision: bool

    @property
    def min_score(self) -> int:
        return min(self.scores.values()) if self.scores else 0


@dataclass
class PipelineTrace:
    user_query: str
    analysis: dict | str          # parsed JSON or raw text if parse failed
    initial_response: str
    critique: CritiqueResult | None
    revised_response: str | None
    final_response: str
    provider: str
    revision_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclass asdict will have serialized the critique; nothing extra needed
        return d


def _parse_json_lenient(text: str) -> dict | None:
    """
    Try to parse JSON out of the model's response. Strips code fences and
    leading/trailing prose. Returns None on failure — callers fall back to
    treating the text as unstructured.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


class AxiomPipeline:
    """
    Christ-axiom alignment pipeline. Wraps any LLMProvider.

    Usage:
        pipeline = AxiomPipeline(provider=AnthropicProvider())
        trace = pipeline.run("User question here")
        print(trace.final_response)
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        revision_threshold: int = REVISION_THRESHOLD,
        max_tokens: int = MAX_TOKENS,
        max_revisions: int = MAX_REVISIONS,
    ):
        self.provider = provider
        self.revision_threshold = revision_threshold
        self.max_tokens = max_tokens
        self.max_revisions = max_revisions

        self._p_analyzer = _load_prompt("context_analyzer")
        self._p_generator = _load_prompt("generator_axiom")
        self._p_critic = _load_prompt("critic")
        self._p_reviser = _load_prompt("reviser")

    # ── Stage 1: analyze ─────────────────────────────────────────────────────
    def analyze(self, user_query: str) -> dict | str:
        raw = self.provider.complete(
            system=self._p_analyzer,
            user=user_query,
            max_tokens=self.max_tokens,
            cache_system=True,
        )
        parsed = _parse_json_lenient(raw)
        return parsed if parsed is not None else raw

    # ── Stage 2: generate ────────────────────────────────────────────────────
    def generate(self, user_query: str, analysis: dict | str) -> str:
        if isinstance(analysis, dict):
            analysis_text = json.dumps(analysis, indent=2)
        else:
            analysis_text = analysis
        user_prompt = (
            f"CONTEXT ANALYSIS (do not quote; use silently):\n{analysis_text}\n\n"
            f"USER MESSAGE:\n{user_query}"
        )
        return self.provider.complete(
            system=self._p_generator,
            user=user_prompt,
            max_tokens=self.max_tokens,
            cache_system=True,
        )

    # ── Stage 3: critique ────────────────────────────────────────────────────
    def critique(
        self, user_query: str, analysis: dict | str, response: str
    ) -> CritiqueResult:
        analysis_text = (
            json.dumps(analysis, indent=2) if isinstance(analysis, dict) else analysis
        )
        user_prompt = (
            f"USER MESSAGE:\n{user_query}\n\n"
            f"CONTEXT ANALYSIS:\n{analysis_text}\n\n"
            f"CANDIDATE RESPONSE:\n{response}"
        )
        raw = self.provider.complete(
            system=self._p_critic,
            user=user_prompt,
            max_tokens=self.max_tokens,
            cache_system=True,
        )
        parsed = _parse_json_lenient(raw)
        if parsed is None:
            # Degrade gracefully: assume it's fine if we can't parse
            return CritiqueResult(
                scores={"truthful": 3, "vulnerable": 3, "faithful": 3, "respected": 3},
                weakest_dimension="truthful",
                concerns=f"[critic output unparseable]\n{raw[:400]}",
                needs_revision=False,
            )

        scores = parsed.get("scores", {})
        min_score = min(scores.values()) if scores else 5
        needs_revision = bool(parsed.get("needs_revision", min_score < self.revision_threshold))
        return CritiqueResult(
            scores=scores,
            weakest_dimension=parsed.get("weakest_dimension", "truthful"),
            concerns=parsed.get("concerns", ""),
            needs_revision=needs_revision,
        )

    # ── Stage 4: revise ──────────────────────────────────────────────────────
    def revise(
        self,
        user_query: str,
        analysis: dict | str,
        previous_response: str,
        critique: CritiqueResult,
    ) -> str:
        analysis_text = (
            json.dumps(analysis, indent=2) if isinstance(analysis, dict) else analysis
        )
        user_prompt = (
            f"USER MESSAGE:\n{user_query}\n\n"
            f"CONTEXT ANALYSIS:\n{analysis_text}\n\n"
            f"PREVIOUS RESPONSE:\n{previous_response}\n\n"
            f"CRITIQUE (weakest dimension: {critique.weakest_dimension}):\n"
            f"{critique.concerns}"
        )
        return self.provider.complete(
            system=self._p_reviser,
            user=user_prompt,
            max_tokens=self.max_tokens,
            cache_system=True,
        )

    # ── Full pipeline ────────────────────────────────────────────────────────
    def run(self, user_query: str) -> PipelineTrace:
        analysis = self.analyze(user_query)
        initial = self.generate(user_query, analysis)
        critique = self.critique(user_query, analysis, initial)

        revised: str | None = None
        final = initial
        revisions = 0

        if critique.needs_revision and self.max_revisions > 0:
            revised = self.revise(user_query, analysis, initial, critique)
            final = revised
            revisions = 1

        return PipelineTrace(
            user_query=user_query,
            analysis=analysis,
            initial_response=initial,
            critique=critique,
            revised_response=revised,
            final_response=final,
            provider=self.provider.name,
            revision_count=revisions,
        )
