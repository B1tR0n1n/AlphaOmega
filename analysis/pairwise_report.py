"""
Report on pairwise verdicts: head-to-head records with exact sign tests,
plus position-bias and length-bias diagnostics for the judge itself.

Usage:
    python analysis/pairwise_report.py results/pairwise/<ts>_pairwise_<judge>.jsonl [more.jsonl ...]

Pass several files to pool runs (e.g. Opus + Haiku) — each run is also
reported on its own. Files from different judges are reported per judge,
followed by an inter-judge agreement check.
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIMS = ["truthful", "vulnerable", "faithful", "respected", "overall"]


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial test against p=0.5, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_records(paths: list[Path]) -> list[dict]:
    """Latest successful verdict per (run, key), with codes resolved to arm names."""
    latest: dict[tuple, dict] = {}
    mappings: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("picks"):
                    latest[(rec["judge"], rec["timestamp"], rec["key"])] = rec

    for rec in latest.values():
        ts = rec["timestamp"]
        if ts not in mappings:
            mapping_path = ROOT / "results" / "blinded" / f"{ts}_mapping.json"
            mappings[ts] = json.loads(mapping_path.read_text())["code_to_arm"]
        m = mappings[ts]
        rec["first_arm"], rec["second_arm"] = m[rec["first"]], m[rec["second"]]
        rec["picks_arm"] = {d: m[c] for d, c in rec["picks"].items()}
    return list(latest.values())


def resolve_pairs(records: list[dict]) -> list[dict]:
    """Collapse the two presentation orders into one outcome per (run, case, pair)."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        pair = tuple(sorted((rec["first_arm"], rec["second_arm"])))
        grouped[(rec["judge"], rec["timestamp"], rec["gen_model"], rec["case_id"], pair)].append(rec)

    outcomes = []
    for (judge, ts, model, case_id, pair), recs in grouped.items():
        if len(recs) != 2:
            continue  # one order missing — incomplete, skip
        winners = {}
        for d in DIMS:
            a, b = recs[0]["picks_arm"][d], recs[1]["picks_arm"][d]
            winners[d] = a if a == b else None  # disagreement across orders = tie
        lens = {recs[0]["first_arm"]: recs[0]["len_first"],
                recs[0]["second_arm"]: recs[0]["len_second"]}
        outcomes.append({"judge": judge, "ts": ts, "model": model, "case_id": case_id,
                         "pair": pair, "winners": winners, "lens": lens})
    return outcomes


def print_head_to_head(outcomes: list[dict], label: str) -> None:
    arms = sorted({a for o in outcomes for a in o["pair"]})
    n_cases = len({(o["ts"], o["case_id"]) for o in outcomes})
    print(f"\n══ {label}  ({n_cases} case-runs) " + "═" * max(0, 50 - len(label)))
    for dim in DIMS:
        print(f"\n  {dim.upper()}")
        print(f"  {'matchup':<28}{'W':>4}{'L':>4}{'T':>4}   {'p':>7}")
        for a, b in combinations(arms, 2):
            rows = [o for o in outcomes if o["pair"] == (a, b)]
            w = sum(o["winners"][dim] == a for o in rows)
            l_ = sum(o["winners"][dim] == b for o in rows)
            t = len(rows) - w - l_
            p = sign_test_p(w, l_)
            flag = " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"  {a + ' vs ' + b:<28}{w:>4}{l_:>4}{t:>4}   {p:>7.4f}{flag}")


def print_diagnostics(records: list[dict], outcomes: list[dict]) -> None:
    first_wins = sum(r["picks"]["overall"] == r["first"] for r in records)
    decided = [o for o in outcomes if o["winners"]["overall"]]
    longer_wins = sum(
        o["lens"][o["winners"]["overall"]] == max(o["lens"].values()) for o in decided
    )
    tie_rate = 1 - len(decided) / len(outcomes) if outcomes else 0.0

    lens_by_arm: dict[str, list[int]] = defaultdict(list)
    for o in outcomes:
        for arm, n in o["lens"].items():
            lens_by_arm[arm].append(n)

    print("\n══ JUDGE DIAGNOSTICS " + "═" * 40)
    print(f"  Position bias:  Response 1 picked overall {first_wins}/{len(records)} "
          f"({100 * first_wins / len(records):.0f}%) — 50% is unbiased")
    print(f"  Order-flip ties (overall): {100 * tie_rate:.0f}% of matchups")
    if decided:
        print(f"  Length bias:    longer answer won {longer_wins}/{len(decided)} "
              f"({100 * longer_wins / len(decided):.0f}%) of decided overall matchups")
    print("  Mean answer length (chars):")
    for arm in sorted(lens_by_arm):
        vals = lens_by_arm[arm]
        print(f"    {arm:<14}{sum(vals) / len(vals):>8.0f}")


def print_judge_agreement(outcomes: list[dict]) -> None:
    """How often two judges reach the same verdict on the same matchup."""
    judges = sorted({o["judge"] for o in outcomes})
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for o in outcomes:
        by_key[(o["ts"], o["case_id"], o["pair"])][o["judge"]] = o

    print("\n══ INTER-JUDGE AGREEMENT " + "═" * 36)
    for j1, j2 in combinations(judges, 2):
        shared = [v for v in by_key.values() if j1 in v and j2 in v]
        print(f"  {j1} vs {j2}  ({len(shared)} shared matchups)")
        for dim in DIMS:
            both = [(v[j1]["winners"][dim], v[j2]["winners"][dim]) for v in shared
                    if v[j1]["winners"][dim] and v[j2]["winners"][dim]]
            agree = sum(a == b for a, b in both)
            pct = 100 * agree / len(both) if both else 0.0
            print(f"    {dim:<12} agree {agree}/{len(both)} ({pct:.0f}%) where both decided")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    records = load_records([Path(p) for p in sys.argv[1:]])
    outcomes = resolve_pairs(records)
    print("W/L = consistent wins across both presentation orders. "
          "T = judge flipped with order. p = exact two-sided sign test (ties excluded).")

    judges = sorted({o["judge"] for o in outcomes})
    for judge in judges:
        j_out = [o for o in outcomes if o["judge"] == judge]
        j_rec = [r for r in records if r["judge"] == judge]
        print(f"\n\n#################### JUDGE: {judge} ####################")
        models = sorted({o["model"] for o in j_out})
        for model in models:
            print_head_to_head([o for o in j_out if o["model"] == model], model)
        if len(models) > 1:
            print_head_to_head(j_out, "POOLED: " + " + ".join(models))
        print_diagnostics(j_rec, j_out)
    if len(judges) > 1:
        print_judge_agreement(outcomes)


if __name__ == "__main__":
    main()
