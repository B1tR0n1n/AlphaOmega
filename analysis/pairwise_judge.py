"""
Pairwise blind judging of a harness run. Absolute 1-5 scoring saturates at
the ceiling on frontier-model output; forced-choice comparison doesn't.

For every case, every pair of arms is judged twice with the presentation
order swapped. A pair only counts as a win when the judge picks the same
answer in both orders — otherwise it's a position-bias tie.

Usage:
    export OPENAI_API_KEY=sk-...
    python analysis/pairwise_judge.py results/blinded/<ts>_blinded.json [judge_model] [base_url]

Local judge (any OpenAI-compatible server, e.g. llama-server):
    python analysis/pairwise_judge.py results/blinded/<ts>_blinded.json qwen3.5-122b http://127.0.0.1:8082/v1

Writes (resumable — already-judged comparisons are skipped):
    results/pairwise/<ts>_pairwise_<judge>.jsonl

Then:
    python analysis/pairwise_report.py results/pairwise/<ts>_pairwise_<judge>.jsonl [...]
"""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from alphaomega.providers import OpenAIProvider  # noqa: E402
from analysis.llm_judge import _parse_json_lenient  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
PAIRWISE_PROMPT = (Path(__file__).parent / "pairwise_prompt.txt").read_text().strip()

DIMS = ["truthful", "vulnerable", "faithful", "respected", "overall"]
WORKERS = 3  # API judge; local judge uses LOCAL_WORKERS to match server slots (-np)
LOCAL_WORKERS = 2
MAX_RETRIES = 2
RATE_LIMIT_RETRIES = 6


def build_jobs(items: list[dict]) -> list[dict]:
    """One job per (case, unordered arm pair, presentation order)."""
    by_case: dict[str, dict[str, dict]] = {}
    for item in items:
        by_case.setdefault(item["case_id"], {})[item["arm_code"]] = item

    jobs = []
    for case_id, arms in sorted(by_case.items()):
        for a, b in combinations(sorted(arms), 2):
            for first, second in ((a, b), (b, a)):
                jobs.append({
                    "key": f"{case_id}|{first}|{second}",
                    "case_id": case_id,
                    "group": arms[a]["group"],
                    "prompt": arms[a]["prompt"],
                    "first": first,
                    "second": second,
                    "answer_1": arms[first]["answer"],
                    "answer_2": arms[second]["answer"],
                })
    return jobs


def complete_with_backoff(judge: OpenAIProvider, user_block: str) -> str:
    """Retry on 429 rate limits with exponential backoff; re-raise anything else."""
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return judge.complete(system=PAIRWISE_PROMPT, user=user_block, max_tokens=300)
        except Exception as e:
            if "429" not in str(e) or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def judge_pair(judge: OpenAIProvider, job: dict) -> dict:
    user_block = (
        f"USER MESSAGE:\n{job['prompt']}\n\n"
        f"=== RESPONSE 1 ===\n{job['answer_1']}\n\n"
        f"=== RESPONSE 2 ===\n{job['answer_2']}"
    )
    raw = ""
    for _ in range(MAX_RETRIES + 1):
        raw = complete_with_backoff(judge, user_block)
        parsed = _parse_json_lenient(raw)
        if parsed and all(str(parsed.get(d)) in ("1", "2") for d in DIMS):
            picks = {
                d: job["first"] if str(parsed[d]) == "1" else job["second"]
                for d in DIMS
            }
            return {"picks": picks, "reason": parsed.get("reason", "")}
    return {"picks": None, "reason": f"[unparseable: {raw[:120]}]"}


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path) as f:
        return {
            rec["key"]
            for rec in (json.loads(line) for line in f if line.strip())
            if rec.get("picks")
        }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    blinded_path = Path(sys.argv[1])
    judge_model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    base_url = sys.argv[3] if len(sys.argv) > 3 else None
    data = json.loads(blinded_path.read_text())
    ts = data["timestamp"]

    out_dir = ROOT / "results" / "pairwise"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}_pairwise_{judge_model.replace('-', '').replace('.', '')}.jsonl"

    jobs = build_jobs(data["results"])
    done = load_done(out_path)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{ts} ({data.get('model', '?')}): {len(jobs)} comparisons, "
          f"{len(done)} done, {len(todo)} to judge with {judge_model}", flush=True)

    judge = OpenAIProvider(model=judge_model, base_url=base_url)
    lock = threading.Lock()
    failures = 0
    workers = LOCAL_WORKERS if base_url else WORKERS
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge_pair, judge, j): j for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # API error — log it, leave key undone for resume
                result = {"picks": None, "reason": f"[error: {e}]"}
            if result["picks"] is None:
                failures += 1
            rec = {
                "key": job["key"], "timestamp": ts, "gen_model": data.get("model"),
                "judge": judge_model, "case_id": job["case_id"], "group": job["group"],
                "first": job["first"], "second": job["second"],
                "len_first": len(job["answer_1"]), "len_second": len(job["answer_2"]),
                **result,
            }
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}]", flush=True)

    print(f"\nVerdicts → {out_path}  ({failures} failed — re-run to retry)")


if __name__ == "__main__":
    main()
