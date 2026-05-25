"""
mcts_wf_replot_null_normal.py

Post-hoc replotter for wf_grammar_compare outputs.

It recreates only the left panel from plot_window_distributions:
    "window XXX | k score for i+j success"

but replaces the plotted null group with direct samples from N(0, 1),
so null is not filtered through i+j success selection.

Usage
-----
python mcts_wf_replot_null_normal.py wf_grammar_compare/wf_grammar_compare_YYYYMMDD_HHMMSS

Outputs
-------
<eval_dir>/plots_null_normal/window_XXX_k_scores_null_normal.png
<eval_dir>/plots_null_normal/null_normal_sampled_values.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUPS = ["gamma000000", "gamma095098", "null"]


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def load_population_scores(eval_dir: str | Path) -> pd.DataFrame:
    eval_dir = Path(eval_dir)
    path = eval_dir / "population_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"could not find {path}")

    df = pd.read_csv(path)

    # Coerce only known numeric columns.
    for c in [
        "window", "pop_i", "score_k_mean_ij", "i_success_n", "j_success_n",
        "ij_success_n", "mean_z_i_all", "mean_z_j_all", "mean_z_k_all",
        "max_z_i_all", "max_z_j_all", "max_z_k_all", "elapsed_sec",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def choose_null_n(rows_window: pd.DataFrame, mode: str, fallback_n: int) -> int:
    """
    How many N(0,1) null z-score samples to draw for this window.

    match_null_rows:
        one normal sample per null row/population in the existing output.
    match_max_valid:
        match the larger finite score count among the two trained groups.
    fixed:
        use fallback_n.
    """
    mode = str(mode)

    if mode == "match_null_rows":
        n = int((rows_window["group"] == "null").sum()) if "group" in rows_window.columns else 0
        return n if n > 0 else int(fallback_n)

    if mode == "match_max_valid":
        counts = []
        for g in ("gamma000000", "gamma095098"):
            vals = finite_array(rows_window.loc[rows_window["group"] == g, "score_k_mean_ij"].to_numpy())
            counts.append(vals.size)
        n = max(counts) if counts else 0
        return int(n if n > 0 else fallback_n)

    if mode == "fixed":
        return int(fallback_n)

    raise ValueError("null_n_mode must be 'match_null_rows', 'match_max_valid', or 'fixed'")


def plot_window_k_scores_null_normal(
    *,
    window: int,
    rows_window: pd.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
    null_n_mode: str = "match_null_rows",
    null_n: int = 30,
    null_mean: float = 0.0,
    null_std: float = 1.0,
    sampled_rows: list[dict] | None = None,
):
    data = []
    labels = []

    for g in ("gamma000000", "gamma095098"):
        vals = finite_array(rows_window.loc[rows_window["group"] == g, "score_k_mean_ij"].to_numpy())
        if vals.size:
            data.append(vals)
            labels.append(g)

    n_null = choose_null_n(rows_window, null_n_mode, fallback_n=null_n)
    null_vals = rng.normal(loc=float(null_mean), scale=float(null_std), size=int(n_null))

    data.append(null_vals)
    labels.append("null")

    if sampled_rows is not None:
        for i, v in enumerate(null_vals):
            sampled_rows.append({
                "window": int(window),
                "group": "null_normal",
                "sample_i": int(i),
                "score_k_mean_ij": float(v),
                "source": "N(0,1)",
            })

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    ax.boxplot(data, labels=labels, showmeans=True)
    for i, vals in enumerate(data, start=1):
        vals = finite_array(vals)
        if vals.size == 0:
            continue
        if vals.size == 1:
            x = np.asarray([i], dtype=float)
        else:
            x = np.full(vals.shape, i, dtype=float) + np.linspace(-0.07, 0.07, vals.size)
        ax.scatter(x, vals, s=25, alpha=0.65)

    ax.axhline(0, alpha=0.3)

    # Intentionally same title/y-axis semantics as the original left subplot.
    ax.set_title(f"window {int(window):03d} | k score for i+j success")
    ax.set_ylabel("mean z_k among i+j successes")
    ax.grid(alpha=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"window_{int(window):03d}_k_scores_null_normal.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_sampled_values(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window", "group", "sample_i", "score_k_mean_ij", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run(args):
    eval_dir = Path(args.eval_dir)
    df = load_population_scores(eval_dir)
    out_dir = eval_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    windows = sorted(int(w) for w in finite_array(df["window"].to_numpy()))
    windows = sorted(set(windows))

    if args.window is not None:
        windows = [int(args.window)]

    sampled_rows = []
    made = []

    for w in windows:
        rows_w = df[pd.to_numeric(df["window"], errors="coerce") == int(w)]
        path = plot_window_k_scores_null_normal(
            window=w,
            rows_window=rows_w,
            out_dir=out_dir,
            rng=rng,
            null_n_mode=args.null_n_mode,
            null_n=args.null_n,
            null_mean=args.null_mean,
            null_std=args.null_std,
            sampled_rows=sampled_rows,
        )
        if path is not None:
            made.append(path)
            print(f"saved {path}")

    write_sampled_values(out_dir / "null_normal_sampled_values.csv", sampled_rows)

    print("complete")
    print(f"plots: {out_dir}")
    print(f"sampled values: {out_dir / 'null_normal_sampled_values.csv'}")
    print(f"n plots: {len(made)}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", type=str, help="wf_grammar_compare output folder")
    p.add_argument("--out-subdir", type=str, default="plots_null_normal")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--window", type=int, default=None, help="optional: replot one window only")
    p.add_argument("--null-n-mode", type=str, default="match_null_rows", choices=["match_null_rows", "match_max_valid", "fixed"])
    p.add_argument("--null-n", type=int, default=30, help="used only if --null-n-mode fixed, or as fallback")
    p.add_argument("--null-mean", type=float, default=0.0)
    p.add_argument("--null-std", type=float, default=1.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
