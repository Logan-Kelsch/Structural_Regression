#!/usr/bin/env python3
"""
mcts_gamma_distribution_read.py

Read and summarize outputs from mcts_gamma_distribution_test.py.

Usage:
    python mcts_gamma_distribution_read.py wf_grammar_compare/<eval_folder>

or directly:
    python mcts_gamma_distribution_read.py wf_grammar_compare/<eval_folder>/gamma_distribution_test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def resolve_test_dir(path: str | Path) -> Path:
    p = Path(path)
    if (p / "gamma_distribution_global_tests.csv").exists():
        return p
    q = p / "gamma_distribution_test"
    if (q / "gamma_distribution_global_tests.csv").exists():
        return q
    raise FileNotFoundError(f"could not find gamma_distribution_global_tests.csv under {p}")


def fmt(x, nd=4):
    try:
        x = float(x)
        if np.isfinite(x):
            return f"{x:.{nd}f}"
    except Exception:
        pass
    return "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="eval folder or gamma_distribution_test folder")
    args = ap.parse_args()

    test_dir = resolve_test_dir(args.path)
    global_df = pd.read_csv(test_dir / "gamma_distribution_global_tests.csv")
    per_df = pd.read_csv(test_dir / "gamma_distribution_per_window_tests.csv")

    g = global_df.iloc[0]

    print("\n=== Gamma distribution comparison ===")
    print("folder:", test_dir)
    print("score:", g.get("score_col", "NA"))
    print("groups:", g.get("group_a", "gamma000000"), "vs", g.get("group_b", "gamma095098"))
    print("n_a / n_b:", int(g.get("n_a", 0)), "/", int(g.get("n_b", 0)))
    print("windows:", int(g.get("n_windows", 0)))
    print("energy distance:", fmt(g.get("energy_distance")))
    print("p-value:", fmt(g.get("p_value")))
    print("mean gamma000000:", fmt(g.get("mean_a")))
    print("mean gamma095098:", fmt(g.get("mean_b")))
    print("mean diff gamma095098 - gamma000000:", fmt(g.get("mean_diff_b_minus_a")))
    print("median diff gamma095098 - gamma000000:", fmt(g.get("median_diff_b_minus_a")))
    print("wasserstein:", fmt(g.get("wasserstein")))
    print("cliff delta gamma000000 vs gamma095098:", fmt(g.get("cliff_delta_a_vs_b")))

    p = float(g.get("p_value", np.nan))
    diff = float(g.get("mean_diff_b_minus_a", np.nan))

    print("\n=== Interpretation ===")
    if np.isfinite(p):
        if p < 0.01:
            print("Strong evidence that the two grammars generate different score distributions.")
        elif p < 0.05:
            print("Evidence that the two grammars generate different score distributions.")
        elif p < 0.10:
            print("Weak evidence of a distributional difference; treat as suggestive.")
        else:
            print("No clear evidence of a distributional difference under this test/run size.")
    else:
        print("p-value is unavailable, likely due to too few finite samples.")

    if np.isfinite(diff):
        if diff > 0:
            print("Direction: gamma095098 has the higher average k score for i+j successes.")
        elif diff < 0:
            print("Direction: gamma000000 has the higher average k score for i+j successes.")
        else:
            print("Direction: average scores are tied.")

    if not per_df.empty:
        print("\n=== Per-window highlights ===")
        per_df = per_df.copy()
        per_df["p_value"] = pd.to_numeric(per_df["p_value"], errors="coerce")
        per_df["mean_diff_b_minus_a"] = pd.to_numeric(per_df["mean_diff_b_minus_a"], errors="coerce")

        sig = per_df[per_df["p_value"] < 0.05]
        print(f"windows with p < 0.05: {len(sig)} / {len(per_df)}")

        best_b = per_df.sort_values("mean_diff_b_minus_a", ascending=False).head(5)
        best_a = per_df.sort_values("mean_diff_b_minus_a", ascending=True).head(5)

        print("\nTop windows favoring gamma095098:")
        print(best_b[["window", "mean_diff_b_minus_a", "p_value", "energy_distance"]].to_string(index=False))

        print("\nTop windows favoring gamma000000:")
        print(best_a[["window", "mean_diff_b_minus_a", "p_value", "energy_distance"]].to_string(index=False))

    print("\nPlots are in:", test_dir / "plots")


if __name__ == "__main__":
    main()
