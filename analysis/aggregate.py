"""
Aggregate rater scores from a CSV and join with the run mapping.

Usage:
    python analysis/aggregate.py results/blinded/20241001T120000Z_blinded.json scores.csv

The blinded JSON is needed to resolve arm codes → arm names.
The mapping file is inferred from the blinded file's timestamp.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


DIMS = ["truthful", "vulnerable", "faithful", "respected"]


def load_mapping(blinded_path: Path) -> dict:
    ts = blinded_path.stem.replace("_blinded", "")
    mapping_path = blinded_path.parent / f"{ts}_mapping.json"
    if not mapping_path.exists():
        print(f"Warning: mapping file not found at {mapping_path}. Arm codes will be shown as-is.")
        return {}
    with open(mapping_path) as f:
        data = json.load(f)
    return data.get("code_to_arm", {})


def load_scores(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate(scores: list[dict], code_to_arm: dict) -> None:
    by_arm: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    by_group: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for row in scores:
        arm_code = row.get("arm_code", "?")
        arm = code_to_arm.get(arm_code, arm_code)
        group = row.get("group", "?")

        for dim in DIMS:
            val = row.get(dim, "").strip()
            if val:
                try:
                    n = float(val)
                    by_arm[arm][dim].append(n)
                    by_group[group][dim].append(n)
                except ValueError:
                    pass

    print("\n── Scores by Prompt Arm ────────────────────────────────────────")
    print(f"{'Arm':<18}" + "".join(f"{d:>12}" for d in DIMS) + f"{'MEAN':>12}")
    for arm in sorted(by_arm):
        vals = by_arm[arm]
        dim_means = [sum(vals[d]) / len(vals[d]) if vals[d] else 0.0 for d in DIMS]
        overall = sum(dim_means) / len(dim_means)
        print(
            f"{arm:<18}"
            + "".join(f"{m:>12.2f}" for m in dim_means)
            + f"{overall:>12.2f}"
        )

    print("\n── Scores by Case Group ────────────────────────────────────────")
    print(f"{'Group':<18}" + "".join(f"{d:>12}" for d in DIMS) + f"{'MEAN':>12}")
    for group in sorted(by_group):
        vals = by_group[group]
        dim_means = [sum(vals[d]) / len(vals[d]) if vals[d] else 0.0 for d in DIMS]
        overall = sum(dim_means) / len(dim_means)
        print(
            f"{group:<18}"
            + "".join(f"{m:>12.2f}" for m in dim_means)
            + f"{overall:>12.2f}"
        )

    total_rows = len(scores)
    scored = sum(1 for r in scores if all(r.get(d, "").strip() for d in DIMS))
    print(f"\n{scored}/{total_rows} rows fully scored.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    blinded_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    code_to_arm = load_mapping(blinded_path)
    scores = load_scores(csv_path)
    aggregate(scores, code_to_arm)


if __name__ == "__main__":
    main()
