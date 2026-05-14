"""
mcts_surface_viz.py

Post-hoc visualizations for mcts_surface_eval.py outputs.

Usage
-----
python mcts_surface_viz.py mcts_surface_eval_runs/<surface_eval_folder>

This script reads:
    grammar_summary.csv
    population_scores.csv
and writes extra figures into:
    <eval_folder>/plots_posthoc
"""

from __future__ import annotations

from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(eval_dir: str | Path):
    eval_dir = Path(eval_dir)
    grammar_csv = eval_dir / "grammar_summary.csv"
    pop_csv = eval_dir / "population_scores.csv"

    if not grammar_csv.exists():
        raise FileNotFoundError(grammar_csv)
    if not pop_csv.exists():
        raise FileNotFoundError(pop_csv)

    grammar = pd.read_csv(grammar_csv)
    pops = pd.read_csv(pop_csv)

    for df in (grammar, pops):
        for c in df.columns:
            if c not in ("run_name", "status", "error"):
                df[c] = pd.to_numeric(df[c], errors="ignore")

    return grammar, pops


def pivot_surface(df, value_key):
    depths = sorted(df["max_depth"].dropna().astype(int).unique().tolist())
    temps = sorted(df["softmax_temp_train"].dropna().astype(float).unique().tolist())

    Z = np.full((len(depths), len(temps)), np.nan, dtype=float)

    for _, row in df.iterrows():
        d = int(row["max_depth"])
        t = float(row["softmax_temp_train"])

        if d not in depths or t not in temps:
            continue

        i = depths.index(d)
        j = temps.index(t)
        Z[i, j] = float(row[value_key]) if pd.notna(row[value_key]) else np.nan

    return depths, temps, Z


def plot_heatmap(df, value_key, out_path, title=None):
    depths, temps, Z = pivot_surface(df, value_key)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", origin="lower")

    ax.set_title(title or value_key)
    ax.set_xlabel("training softmax_temp")
    ax.set_ylabel("max_depth")
    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([f"{t:.3f}" for t in temps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(d) for d in depths])
    fig.colorbar(im, ax=ax, label=value_key)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions_by_grammar(pops, out_path):
    run_names = sorted(pops["run_name"].dropna().unique().tolist())
    data = []
    labels = []

    for rn in run_names:
        vals = pd.to_numeric(
            pops.loc[pops["run_name"] == rn, "score_k_mean_for_i_j_success"],
            errors="coerce",
        ).dropna().values

        if vals.size:
            data.append(vals)
            labels.append(rn.replace("surf_", ""))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.35), 6), constrained_layout=True)
    ax.violinplot(data, showmeans=True, widths=0.8)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_title("population score distributions by grammar")
    ax.set_ylabel("mean z_k for i+j successful genes")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_depth_curves(grammar, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for d, sub in grammar.groupby("max_depth"):
        sub = sub.sort_values("softmax_temp_train")
        ax.plot(
            sub["softmax_temp_train"],
            sub["score_mean"],
            marker="o",
            label=f"depth {int(d)}",
        )

    ax.set_xscale("log")
    ax.set_title("mean score by temperature, grouped by max_depth")
    ax.set_xlabel("training softmax_temp")
    ax.set_ylabel("mean population score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_curves(grammar, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for t, sub in grammar.groupby("softmax_temp_train"):
        sub = sub.sort_values("max_depth")
        ax.plot(
            sub["max_depth"],
            sub["score_mean"],
            marker="o",
            label=f"temp {float(t):.3f}",
        )

    ax.set_title("mean score by max_depth, grouped by temperature")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("mean population score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_success_vs_score(pops, out_path):
    x = pd.to_numeric(pops["both_success_n"], errors="coerce")
    y = pd.to_numeric(pops["score_k_mean_for_i_j_success"], errors="coerce")

    keep = x.notna() & y.notna()
    if not keep.any():
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(x[keep], y[keep], alpha=0.65)
    ax.set_title("i+j success count vs k-chunk score")
    ax.set_xlabel("number of i+j successful genes")
    ax.set_ylabel("mean z_k of i+j successful genes")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_one_grammar(eval_dir: Path, run_name: str, out_dir: Path):
    pop_dirs = sorted((eval_dir / "grammars" / run_name).glob("pop_*"))
    if not pop_dirs:
        return

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for chunk_idx, chunk in enumerate(["i", "j", "k"]):
        ax = axs[chunk_idx]
        for p_i, pop_dir in enumerate(pop_dirs):
            path = pop_dir / "eval_arrays.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            z = np.asarray(data[f"z_{chunk}"], dtype=float)
            gidx = np.asarray(data["good_idx"], dtype=int)
            vals = z[gidx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                alpha = 0.08 + 0.45 * ((p_i + 1) / len(pop_dirs))
                ax.scatter(np.full(vals.shape, p_i), vals, s=5, alpha=alpha)

        ax.axhline(2.0, linestyle="--", alpha=0.5)
        ax.set_title(f"chunk {chunk} z-score clouds")
        ax.set_xlabel("population")
        ax.set_ylabel("z-score")
        ax.grid(alpha=0.25)

    fig.suptitle(run_name)
    fig.savefig(out_dir / f"z_clouds_{run_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(eval_dir: str | Path):
    eval_dir = Path(eval_dir)
    out_dir = eval_dir / "plots_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar, pops = load_results(eval_dir)

    for key in [
        "score_mean",
        "score_median",
        "score_std",
        "valid_score_frac",
        "both_success_n_mean",
        "i_success_n_mean",
        "j_success_n_mean",
    ]:
        if key in grammar.columns:
            plot_heatmap(grammar, key, out_dir / f"heatmap_{key}.png", title=key)

    plot_score_distributions_by_grammar(pops, out_dir / "score_distributions_by_grammar.png")
    plot_depth_curves(grammar, out_dir / "score_by_temp_grouped_depth.png")
    plot_temperature_curves(grammar, out_dir / "score_by_depth_grouped_temp.png")
    plot_success_vs_score(pops, out_dir / "success_count_vs_score.png")

    # A few per-grammar z-cloud plots. Comment this out if too many figures are unwanted.
    for rn in sorted(grammar["run_name"].dropna().unique().tolist()):
        plot_one_grammar(eval_dir, rn, out_dir)

    print("wrote plots to", out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python mcts_surface_viz.py <mcts_surface_eval_runs/surface_eval_...>")
    make_all_plots(sys.argv[1])
