#!/usr/bin/env python3
"""
mcts_gamma_distribution_test.py

Post-hoc distribution comparison for gamma000000 vs gamma095098 grammar outputs.

Input:
    A wf_grammar_compare output folder containing population_scores.csv

Goal:
    Test whether gamma000000 and gamma095098 generated k-chunk score distributions
    are different, using score_k_mean_ij by default.

Main test:
    Stratified permutation energy-distance test.

Why stratified?
    Scores are grouped by walk-forward window. Window difficulty can shift all scores.
    The stratified permutation test shuffles gamma labels only within each window,
    preserving the per-window structure.

Outputs:
    <eval_dir>/gamma_distribution_test/
        gamma_distribution_global_tests.csv
        gamma_distribution_per_window_tests.csv
        gamma_distribution_effects_by_window.csv
        gamma_distribution_test_config.json
        plots/*.png

Usage:
    python mcts_gamma_distribution_test.py wf_grammar_compare/wf_grammar_compare_YYYYMMDD_HHMMSS

Optional:
    python mcts_gamma_distribution_test.py <eval_dir> --score-col score_k_mean_ij --n-perms 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_A = "gamma000000"
GROUP_B = "gamma095098"
DEFAULT_SCORE_COL = "score_k_mean_ij"


def _as_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def energy_statistic_1d(x, y) -> float:
    """
    Two-sample energy statistic in 1D.

    E = 2 E|X-Y| - E|X-X'| - E|Y-Y'|

    Larger means more distributional separation. It is sensitive to mean,
    variance, shape, and tail differences.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)

    if x.size == 0 or y.size == 0:
        return np.nan

    xy = np.abs(x[:, None] - y[None, :]).mean()
    xx = np.abs(x[:, None] - x[None, :]).mean()
    yy = np.abs(y[:, None] - y[None, :]).mean()

    return float(2.0 * xy - xx - yy)


def wasserstein_1d(x, y) -> float:
    """
    Simple equal-quantile approximation to 1D Wasserstein distance.
    No scipy dependency.
    """
    x = np.sort(_as_float_array(x))
    y = np.sort(_as_float_array(y))

    if x.size == 0 or y.size == 0:
        return np.nan

    n = max(x.size, y.size)
    q = (np.arange(n) + 0.5) / n
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def ecdf_xy(x):
    x = np.sort(_as_float_array(x))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def cliff_delta(x, y) -> float:
    """
    Cliff's delta: P(X>Y) - P(X<Y).
    Positive means x tends to be larger than y.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)

    if x.size == 0 or y.size == 0:
        return np.nan

    diffs = x[:, None] - y[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def two_sample_energy_permutation_test(x, y, *, n_perms=5000, seed=42) -> dict:
    """
    Unstratified two-sample permutation energy test.
    """
    rng = np.random.default_rng(seed)
    x = _as_float_array(x)
    y = _as_float_array(y)

    out = {
        "n_a": int(x.size),
        "n_b": int(y.size),
        "energy_distance": np.nan,
        "p_value": np.nan,
        "n_perms": int(n_perms),
    }

    if x.size < 2 or y.size < 2:
        return out

    obs = energy_statistic_1d(x, y)
    pooled = np.concatenate([x, y])
    n_a = x.size

    ge = 0
    for _ in range(int(n_perms)):
        perm = rng.permutation(pooled)
        stat = energy_statistic_1d(perm[:n_a], perm[n_a:])
        if stat >= obs:
            ge += 1

    p = (ge + 1.0) / (int(n_perms) + 1.0)

    out.update({
        "energy_distance": float(obs),
        "p_value": float(p),
    })
    return out


def stratified_energy_permutation_test(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str = "group",
    window_col: str = "window",
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
    n_perms: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Stratified permutation energy-distance test.

    Observed statistic is the pooled energy distance between all finite scores in A and B.
    Permutations shuffle labels within each window, preserving window-specific sample counts.
    """
    rng = np.random.default_rng(seed)

    work = df[[window_col, group_col, score_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[np.isfinite(work[score_col])]
    work = work[work[group_col].isin([group_a, group_b])]

    x_obs = work.loc[work[group_col] == group_a, score_col].to_numpy(float)
    y_obs = work.loc[work[group_col] == group_b, score_col].to_numpy(float)

    out = {
        "test": "stratified_permutation_energy_distance",
        "score_col": score_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(x_obs.size),
        "n_b": int(y_obs.size),
        "n_windows": int(work[window_col].nunique()) if not work.empty else 0,
        "energy_distance": np.nan,
        "p_value": np.nan,
        "n_perms": int(n_perms),
        "mean_a": float(np.nanmean(x_obs)) if x_obs.size else np.nan,
        "mean_b": float(np.nanmean(y_obs)) if y_obs.size else np.nan,
        "mean_diff_b_minus_a": float(np.nanmean(y_obs) - np.nanmean(x_obs)) if x_obs.size and y_obs.size else np.nan,
        "median_a": float(np.nanmedian(x_obs)) if x_obs.size else np.nan,
        "median_b": float(np.nanmedian(y_obs)) if y_obs.size else np.nan,
        "median_diff_b_minus_a": float(np.nanmedian(y_obs) - np.nanmedian(x_obs)) if x_obs.size and y_obs.size else np.nan,
        "wasserstein": wasserstein_1d(x_obs, y_obs),
        "cliff_delta_a_vs_b": cliff_delta(x_obs, y_obs),
    }

    if x_obs.size < 2 or y_obs.size < 2:
        return out

    obs = energy_statistic_1d(x_obs, y_obs)

    blocks = []
    for window, wdf in work.groupby(window_col):
        scores = wdf[score_col].to_numpy(float)
        labels = wdf[group_col].to_numpy(object)
        if np.sum(labels == group_a) == 0 or np.sum(labels == group_b) == 0:
            continue
        blocks.append((scores, labels.copy()))

    if len(blocks) == 0:
        return out

    ge = 0
    for _ in range(int(n_perms)):
        xa = []
        yb = []
        for scores, labels in blocks:
            shuffled = rng.permutation(labels)
            xa.append(scores[shuffled == group_a])
            yb.append(scores[shuffled == group_b])

        x_perm = np.concatenate(xa) if xa else np.asarray([])
        y_perm = np.concatenate(yb) if yb else np.asarray([])
        stat = energy_statistic_1d(x_perm, y_perm)
        if stat >= obs:
            ge += 1

    p = (ge + 1.0) / (int(n_perms) + 1.0)

    out.update({
        "energy_distance": float(obs),
        "p_value": float(p),
    })
    return out


def load_population_scores(eval_dir: str | Path) -> pd.DataFrame:
    eval_dir = Path(eval_dir)
    path = eval_dir / "population_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing population_scores.csv: {path}")

    df = pd.read_csv(path)

    if "group" not in df.columns:
        raise ValueError("population_scores.csv must contain a 'group' column")

    if "window" not in df.columns:
        # tolerate older file naming
        for alt in ["window_id", "window_num", "k_window"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "window"})
                break

    if "window" not in df.columns:
        raise ValueError("population_scores.csv must contain a 'window' column")

    df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("Int64")
    return df


def per_window_tests(df, *, score_col, n_perms, seed) -> pd.DataFrame:
    rows = []
    for window in sorted(df["window"].dropna().unique()):
        wdf = df[df["window"] == window]
        a = pd.to_numeric(wdf.loc[wdf["group"] == GROUP_A, score_col], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(wdf.loc[wdf["group"] == GROUP_B, score_col], errors="coerce").dropna().to_numpy(float)

        test = two_sample_energy_permutation_test(
            a,
            b,
            n_perms=n_perms,
            seed=seed + int(window),
        )

        row = {
            "window": int(window),
            "score_col": score_col,
            "group_a": GROUP_A,
            "group_b": GROUP_B,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_a": float(np.nanmean(a)) if a.size else np.nan,
            "mean_b": float(np.nanmean(b)) if b.size else np.nan,
            "mean_diff_b_minus_a": float(np.nanmean(b) - np.nanmean(a)) if a.size and b.size else np.nan,
            "median_a": float(np.nanmedian(a)) if a.size else np.nan,
            "median_b": float(np.nanmedian(b)) if b.size else np.nan,
            "median_diff_b_minus_a": float(np.nanmedian(b) - np.nanmedian(a)) if a.size and b.size else np.nan,
            "wasserstein": wasserstein_1d(a, b),
            "cliff_delta_a_vs_b": cliff_delta(a, b),
            "energy_distance": test["energy_distance"],
            "p_value": test["p_value"],
            "n_perms": int(n_perms),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_global_ecdf(df, out_dir, *, score_col):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for group in [GROUP_A, GROUP_B]:
        vals = pd.to_numeric(df.loc[df["group"] == group, score_col], errors="coerce").dropna().to_numpy(float)
        x, y = ecdf_xy(vals)
        ax.step(x, y, where="post", label=f"{group} (n={vals.size})")

    ax.axvline(0, alpha=0.25)
    ax.set_title("Global ECDF of k scores for i+j success")
    ax.set_xlabel(score_col)
    ax.set_ylabel("empirical CDF")
    ax.grid(alpha=0.25)
    ax.legend()

    path = out_dir / "global_ecdf_gamma000000_vs_gamma095098.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_hist(df, out_dir, *, score_col):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    vals_a = pd.to_numeric(df.loc[df["group"] == GROUP_A, score_col], errors="coerce").dropna().to_numpy(float)
    vals_b = pd.to_numeric(df.loc[df["group"] == GROUP_B, score_col], errors="coerce").dropna().to_numpy(float)

    all_vals = np.concatenate([vals_a, vals_b]) if vals_a.size and vals_b.size else np.asarray([])
    if all_vals.size:
        bins = np.histogram_bin_edges(all_vals, bins="auto")
        ax.hist(vals_a, bins=bins, density=True, alpha=0.45, label=GROUP_A)
        ax.hist(vals_b, bins=bins, density=True, alpha=0.45, label=GROUP_B)
    else:
        ax.text(0.5, 0.5, "no finite scores", ha="center", va="center")

    ax.axvline(0, alpha=0.25)
    ax.set_title("Global distribution of k scores for i+j success")
    ax.set_xlabel(score_col)
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend()

    path = out_dir / "global_hist_gamma000000_vs_gamma095098.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_window_distributions(df, out_dir, *, score_col):
    windows = sorted(df["window"].dropna().unique())
    if len(windows) == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(windows)), 6), constrained_layout=True)

    positions = []
    data = []
    labels = []

    for i, w in enumerate(windows):
        a = pd.to_numeric(df.loc[(df["window"] == w) & (df["group"] == GROUP_A), score_col], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(df.loc[(df["window"] == w) & (df["group"] == GROUP_B), score_col], errors="coerce").dropna().to_numpy(float)
        if a.size:
            data.append(a)
            positions.append(i * 3 + 0)
            labels.append(f"{int(w)}\nA")
        if b.size:
            data.append(b)
            positions.append(i * 3 + 1)
            labels.append(f"{int(w)}\nB")

    if data:
        parts = ax.violinplot(data, positions=positions, widths=0.8, showmeans=True, showextrema=True)
        # default colors are okay; no custom style required
        ax.scatter(
            np.concatenate([np.full(len(d), p) for d, p in zip(data, positions)]),
            np.concatenate(data),
            s=8,
            alpha=0.35,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
    else:
        ax.text(0.5, 0.5, "no finite scores", ha="center", va="center")

    ax.axhline(0, alpha=0.25)
    ax.set_title("Per-window k score distributions for i+j success")
    ax.set_xlabel("window and group")
    ax.set_ylabel(score_col)
    ax.grid(alpha=0.25)

    path = out_dir / "per_window_violin_gamma000000_vs_gamma095098.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effects_by_window(per_window_df, out_dir):
    if per_window_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = per_window_df["window"].to_numpy(float)
    y = per_window_df["mean_diff_b_minus_a"].to_numpy(float)
    p = per_window_df["p_value"].to_numpy(float)

    ax.plot(x, y, marker="o", label="mean diff B-A")
    ax.axhline(0, alpha=0.35)

    sig = np.isfinite(p) & (p < 0.05)
    if np.any(sig):
        ax.scatter(x[sig], y[sig], s=80, label="p < 0.05")

    ax.set_title("Mean score difference by window")
    ax.set_xlabel("window")
    ax.set_ylabel("mean difference: gamma095098 - gamma000000")
    ax.grid(alpha=0.25)
    ax.legend()

    path = out_dir / "mean_difference_by_window.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_plots(df, per_window_df, out_dir, *, score_col):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(str(plot_global_ecdf(df, plots_dir, score_col=score_col)))
    paths.append(str(plot_global_hist(df, plots_dir, score_col=score_col)))
    p = plot_window_distributions(df, plots_dir, score_col=score_col)
    if p is not None:
        paths.append(str(p))
    p = plot_effects_by_window(per_window_df, plots_dir)
    if p is not None:
        paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=str, help="wf_grammar_compare output folder")
    ap.add_argument("--score-col", type=str, default=DEFAULT_SCORE_COL)
    ap.add_argument("--n-perms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "gamma_distribution_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_population_scores(eval_dir)

    if args.score_col not in df.columns:
        raise ValueError(f"score column {args.score_col!r} not found in population_scores.csv")

    df = df[df["group"].isin([GROUP_A, GROUP_B])].copy()
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")

    global_test = stratified_energy_permutation_test(
        df,
        score_col=args.score_col,
        n_perms=args.n_perms,
        seed=args.seed,
    )
    global_df = pd.DataFrame([global_test])
    global_path = out_dir / "gamma_distribution_global_tests.csv"
    global_df.to_csv(global_path, index=False)

    per_win_df = per_window_tests(
        df,
        score_col=args.score_col,
        n_perms=args.n_perms,
        seed=args.seed,
    )
    per_window_path = out_dir / "gamma_distribution_per_window_tests.csv"
    per_win_df.to_csv(per_window_path, index=False)

    effects_path = out_dir / "gamma_distribution_effects_by_window.csv"
    effect_cols = [
        "window", "n_a", "n_b", "mean_a", "mean_b", "mean_diff_b_minus_a",
        "median_a", "median_b", "median_diff_b_minus_a", "wasserstein",
        "cliff_delta_a_vs_b", "energy_distance", "p_value",
    ]
    per_win_df[[c for c in effect_cols if c in per_win_df.columns]].to_csv(effects_path, index=False)

    plot_paths = make_plots(df, per_win_df, out_dir, score_col=args.score_col)

    config = {
        "eval_dir": str(eval_dir),
        "out_dir": str(out_dir),
        "score_col": args.score_col,
        "groups": [GROUP_A, GROUP_B],
        "n_perms": int(args.n_perms),
        "seed": int(args.seed),
        "global_tests_csv": str(global_path),
        "per_window_tests_csv": str(per_window_path),
        "effects_csv": str(effects_path),
        "plots": plot_paths,
    }
    with open(out_dir / "gamma_distribution_test_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("saved:", out_dir)
    print("global test:")
    print(global_df.to_string(index=False))
    print("\ninterpretation:")
    print("  low p_value means gamma000000 and gamma095098 score distributions differ after shuffling labels within each window.")
    print("  positive mean_diff_b_minus_a means gamma095098 has higher average score_k_mean_ij than gamma000000.")
    print("  energy_distance measures distributional separation; larger means more different.")


if __name__ == "__main__":
    main()
