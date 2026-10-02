"""
Score a blinded harness run with an LLM judge, output a CSV in the exact
format `aggregate.py` expects.

Default judge: GPT-4o. The whole point of using a different-vendor model
than the one that generated the answers is to avoid self-judging bias.

Usage:
    export OPENAI_API_KEY=sk-...
    python analysis/llm_judge.py results/blinded/<ts>_blinded.json

Writes:
    results/blinded/<ts>_scores_gpt4o.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from alphaomega.providers import OpenAIProvider  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text().strip()

DIMS = ["truthful", "vulnerable", "faithful", "respected"]


def _parse_json_lenient(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def score_one(judge: OpenAIProvider, user_prompt: str, answer: str) -> dict:
    user_block = (
        f"USER MESSAGE:\n{user_prompt}\n\n"
        f"CANDIDATE RESPONSE:\n{answer}"
    )
    raw = judge.complete(
        system=JUDGE_PROMPT,
        user=user_block,
        max_tokens=400,
    )
    parsed = _parse_json_lenient(raw)
    if parsed is None:
        return {d: "" for d in DIMS} | {"brief_reason": f"[unparseable: {raw[:120]}]"}
    out = {d: parsed.get(d, "") for d in DIMS}
    out["brief_reason"] = parsed.get("brief_reason", "")
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    blinded_path = Path(sys.argv[1])
    judge_model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"

    with open(blinded_path) as f:
        data = json.load(f)

    items = data["results"]
    print(f"Judging {len(items)} answers with {judge_model}...")

    judge = OpenAIProvider(model=judge_model)

    out_path = blinded_path.parent / f"{blinded_path.stem.replace('_blinded', '')}_scores_{judge_model.replace('-', '')}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case_id", "group", "title", "arm_code"] + DIMS + ["brief_reason"]
        )

        for i, item in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] case={item['case_id']} arm_code={item['arm_code']}", flush=True)
            scores = score_one(judge, item["prompt"], item["answer"])
            writer.writerow(
                [
                    item["case_id"],
                    item["group"],
                    item["title"],
                    item["arm_code"],
                ]
                + [scores[d] for d in DIMS]
                + [scores["brief_reason"]]
            )

    print(f"\nScores → {out_path}")
    print(f"\nNow run:\n  python analysis/aggregate.py {blinded_path} {out_path}")


if __name__ == "__main__":
    main()
