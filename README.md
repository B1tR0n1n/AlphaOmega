# AlphaOmega

**Christ-axiom alignment architecture for any LLM.**

AlphaOmega is a model-agnostic, inference-time alignment layer. It wraps any underlying language model (Claude, GPT-4o, Llama via Ollama, etc.) and runs its responses through a four-stage pipeline grounded in Christ's ethical teaching:

1. **Love your neighbor** (Matthew 22:37-40)
2. **Weight the vulnerable** (Luke 4:18, Matthew 25:40)
3. **Truth over comfort** (Mark 10:21)
4. **Rank rules, don't replace them** (Matthew 5:17)

Unlike approaches that anchor to creeds or doctrinal correctness, AlphaOmega applies the axioms as a *governing principle* that ranks duties when they conflict and defends against sycophancy by design.

## How it works

```
User query
    ↓
[Context Analyzer]     — who's vulnerable? what truth is at stake?
    ↓
[Axiom Generator]      — base model + axiom system prompt
    ↓
[Axiom Critic]         — grades response on 4 dimensions (1–5)
    ↓
[Reviser] (if needed)  — surgically fixes the weakest dimension
    ↓
Final response + full reasoning trace
```

## Project layout

```
alphaomega/     The architecture (library)
harness/        15-case evaluation harness (neutral vs. creed vs. axiom-naive vs. axiom-arch)
scorer/         Local web UI for blind 1–5 rater scoring
analysis/       Aggregates scores by arm and case group
examples/       Model-agnostic demos (Claude today; GPT-4o and Ollama stubs)
```

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# See the architecture in action on one case:
python examples/claude_demo.py

# Run the full harness (4 arms × N cases):
python harness/run.py

# Score blindly in your browser:
open scorer/index.html

# Aggregate the scored CSV:
python analysis/aggregate.py results/blinded/<ts>_blinded.json scores.csv
```

## Roadmap

| Phase | Output | Proves |
|---|---|---|
| **1 (current)** | Inference architecture + 15-case harness | Axiom framework beats creed/neutral prompting across models |
| 2 | Scaled dataset of (query, rejected, preferred) triples via the architecture | A reusable artifact anyone can fine-tune with |
| 3 | DPO fine-tune of Llama 3.1 8B (QLoRA on dual GPU) | Axioms embedded in weights, not just wrapped around them |
| 4 | Comparative evidence pack | Architecture, dataset, and tuned reference model each have independent value |
