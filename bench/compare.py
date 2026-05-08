"""Compare two layering benchmark runs.

Usage:

    .venv/bin/python bench/compare.py bench/results/baseline.json bench/results/post.json

Prints a side-by-side table with absolute and percent deltas. Negative deltas
are wins (the candidate is faster). Flags any case where the candidate produced
a different layer count than the baseline — that means the algorithm itself
changed behavior, not just speed, and the layering test suite is the right
place to verify intent.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Tuple


def load(path: str) -> Tuple[dict, Dict[Tuple[str, str], dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_key = {(r["case"], r["method"]): r for r in data["results"]}
    return data, by_key


def fmt_pct(pct: float) -> str:
    arrow = "▼" if pct < 0 else ("▲" if pct > 0 else " ")
    return f"{arrow}{pct:+6.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", help="baseline JSON (the slower / current code)")
    parser.add_argument("candidate", help="candidate JSON (the post-change run)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="percent change threshold for win/regression flags")
    args = parser.parse_args()

    base_meta, base = load(args.baseline)
    cand_meta, cand = load(args.candidate)

    if base_meta.get("seed") != cand_meta.get("seed"):
        print(f"warning: seeds differ ({base_meta.get('seed')} vs {cand_meta.get('seed')})",
              file=sys.stderr)

    print(f"baseline:  {args.baseline}  (runs={base_meta.get('runs')})")
    print(f"candidate: {args.candidate}  (runs={cand_meta.get('runs')})")
    print()

    header = f"{'case':<22} {'method':<22} {'base (ms)':>10} {'cand (ms)':>10} {'delta':>9} {'layers':>14}"
    print(header)
    print("-" * len(header))

    wins = regressions = behavior_changes = 0
    total_base = total_cand = 0.0

    for key, base_row in base.items():
        cand_row = cand.get(key)
        if cand_row is None:
            print(f"{key[0]:<22} {key[1]:<22} {'(missing in candidate)'}")
            continue

        b_ms = base_row["best_seconds"] * 1000
        c_ms = cand_row["best_seconds"] * 1000
        delta_pct = (c_ms - b_ms) / b_ms * 100 if b_ms > 0 else 0.0
        total_base += b_ms
        total_cand += c_ms

        if delta_pct <= -args.threshold:
            wins += 1
        elif delta_pct >= args.threshold:
            regressions += 1

        layers_str = f"{base_row['layers']}→{cand_row['layers']}"
        if base_row["layers"] != cand_row["layers"]:
            layers_str += " !"
            behavior_changes += 1

        print(f"{key[0]:<22} {key[1]:<22} {b_ms:>10.2f} {c_ms:>10.2f} "
              f"{fmt_pct(delta_pct):>9} {layers_str:>14}")

    extra_in_cand = set(cand.keys()) - set(base.keys())
    for key in sorted(extra_in_cand):
        print(f"{key[0]:<22} {key[1]:<22} {'(new in candidate)'}")

    print("-" * len(header))
    overall_pct = (total_cand - total_base) / total_base * 100 if total_base > 0 else 0.0
    print(f"{'TOTAL':<45} {total_base:>10.2f} {total_cand:>10.2f} {fmt_pct(overall_pct):>9}")
    print()
    print(f"summary:  {wins} wins  ·  {regressions} regressions  ·  "
          f"{behavior_changes} layer-count changes (threshold ±{args.threshold:.0f}%)")
    if behavior_changes:
        print("  ! layer-count differences indicate the algorithm produced different output — "
              "verify against the layering test suite before trusting timings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
