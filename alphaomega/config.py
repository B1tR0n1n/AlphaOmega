"""Pipeline tunables."""

from pathlib import Path

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = Path(__file__).parent / "prompts"

# A revision is triggered if ANY dimension in the critic scores < this value
# (on the same 1-5 scale that human raters use).
REVISION_THRESHOLD = 4

# Hard ceiling on tokens per model call. Keeps the pipeline predictable.
MAX_TOKENS = 1024

# Max revision passes. One revision is almost always enough; two is the limit
# before diminishing returns.
MAX_REVISIONS = 1
