"""AlphaOmega: Christ-axiom alignment architecture for any LLM."""

from .axioms import AXIOMS, Axiom, axioms_as_prose, axioms_as_tests
from .pipeline import AxiomPipeline, PipelineTrace
from .providers import LLMProvider, AnthropicProvider

__all__ = [
    "AxiomPipeline",
    "PipelineTrace",
    "AXIOMS",
    "Axiom",
    "axioms_as_prose",
    "axioms_as_tests",
    "LLMProvider",
    "AnthropicProvider",
]
