"""
Run all cases through all four arms. Save labeled + blinded output.

Usage:
    python harness/run.py [--dry-run] [--arms neutral,axiom_arch] [--cases A1,B2]

Requires ANTHROPIC_API_KEY (or a .env file at repo root).
"""

import argparse
import json
import random
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo-root import of the alphaomega package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from alphaomega import AnthropicProvider  # noqa: E402
from harness.arms import ARMS  # noqa: E402

ROOT = Path(__file__).parent.parent
CASES_FILE = Path(__file__).parent / "cases" / "cases.json"
RUNS_DIR = ROOT / "results" / "runs"
BLINDED_DIR = ROOT / "results" / "blinded"

load_dotenv(ROOT / ".env")


def load_cases() -> list[dict]:
    with open(CASES_FILE) as f:
        return json.load(f)


def blind(labeled: list[dict]) -> tuple[list[dict], dict, dict]:
    """Shuffle arm names to random letter codes so raters score blind."""
    arm_names = sorted({r["arm"] for r in labeled})
    codes = random.sample(string.ascii_uppercase[: len(arm_names)], len(arm_names))
    arm_to_code = dict(zip(arm_names, codes))
    code_to_arm = {v: k for k, v in arm_to_code.items()}

    blinded = []
    for r in labeled:
        blinded.append(
            {
                "case_id": r["case_id"],
                "group": r["group"],
                "title": r["title"],
                "arm_code": arm_to_code[r["arm"]],
                "prompt": r["prompt"],
                "answer": r["answer"],
            }
        )
    random.shuffle(blinded)
    return blinded, arm_to_code, code_to_arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--arms", help="Comma-separated arm names (default: all)", default=None
    )
    parser.add_argument(
        "--cases", help="Comma-separated case IDs (default: all)", default=None
    )
    parser.add_argument("--model", default="claude-opus-4-7")
    args = parser.parse_args()

    cases = load_cases()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    arm_names = list(ARMS.keys())
    if args.arms:
        arm_names = args.arms.split(",")

    total = len(cases) * len(arm_names)
    print(
        f"AlphaOmega harness {'(DRY RUN) ' if args.dry_run else ''}"
        f"— {args.model} — {len(cases)} cases × {len(arm_names)} arms = {total} runs"
    )

    if args.dry_run:
        for c in cases:
            for a in arm_names:
                print(f"  [DRY] case={c['id']} arm={a}")
        return

    provider = AnthropicProvider(model=args.model)

    labeled: list[dict] = []
    done = 0
    for case in cases:
        for arm_name in arm_names:
            done += 1
            print(f"  [{done}/{total}] case={case['id']} arm={arm_name}")
            arm_fn = ARMS[arm_name]
            answer, meta = arm_fn(provider, case["prompt"])
            labeled.append(
                {
                    "case_id": case["id"],
                    "group": case["group"],
                    "title": case["title"],
                    "arm": arm_name,
                    "prompt": case["prompt"],
                    "answer": answer,
                    "meta": meta,
                    "notes": case.get("notes", ""),
                }
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BLINDED_DIR.mkdir(parents=True, exist_ok=True)

    labeled_path = RUNS_DIR / f"{ts}_labeled.json"
    with open(labeled_path, "w") as f:
        json.dump(
            {"timestamp": ts, "model": args.model, "results": labeled}, f, indent=2
        )
    print(f"\nLabeled → {labeled_path}")

    blinded, arm_to_code, code_to_arm = blind(labeled)
    blinded_path = BLINDED_DIR / f"{ts}_blinded.json"
    mapping_path = BLINDED_DIR / f"{ts}_mapping.json"

    with open(blinded_path, "w") as f:
        json.dump(
            {"timestamp": ts, "model": args.model, "results": blinded}, f, indent=2
        )
    with open(mapping_path, "w") as f:
        json.dump(
            {"timestamp": ts, "arm_to_code": arm_to_code, "code_to_arm": code_to_arm},
            f,
            indent=2,
        )

    print(f"Blinded → {blinded_path}")
    print(f"Mapping (keep private until scoring done) → {mapping_path}")


if __name__ == "__main__":
    main()
