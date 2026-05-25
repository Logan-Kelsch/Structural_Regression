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

    # pandas 3+ no longer accepts errors="ignore" in pd.to_numeric.
    # Convert only columns that are mostly numeric; leave path/name/text columns alone.
    text_cols = {
        "run_name", "status", "error", "run_dir", "grammar_path",
        "arrays_path", "crash_path", "notes"
    }

    for df in (grammar, pops):
        for c in df.columns:
            if c in text_cols:
                continue

            converted = pd.to_numeric(df[c], errors="coerce")

            # Keep conversion only if at least one non-null value became numeric.
            # This prevents string/path columns from being erased to all NaN.
            if converted.notna().sum() > 0:
                df[c] = converted

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
    """Original z-score cloud plot for one grammar."""
    pop_dirs = sorted((eval_dir / "grammars" / run_name).glob("pop_*"))
    if not pop_dirs:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

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


def _valid_gidx_for_z(z, gidx):
    """Return valid gene indices that fit inside a z-score vector."""
    z = np.asarray(z, dtype=float)
    gidx = np.asarray(gidx, dtype=int).ravel()
    keep = (gidx >= 0) & (gidx < z.size)
    return gidx[keep]


def _scatter_success_split(ax, x_pos, y_vals, success_mask, *, alpha=0.45, red_label=None, green_label=None):
    """Scatter red failures first, then green successes, for one population column."""
    y_vals = np.asarray(y_vals, dtype=float).ravel()
    success_mask = np.asarray(success_mask, dtype=bool).ravel()

    finite = np.isfinite(y_vals)
    y_vals = y_vals[finite]
    success_mask = success_mask[finite]

    if y_vals.size == 0:
        return

    fail = ~success_mask
    if np.any(fail):
        ax.scatter(
            np.full(np.sum(fail), x_pos),
            y_vals[fail],
            s=7,
            alpha=alpha,
            color="red",
            label=red_label,
        )
    if np.any(success_mask):
        ax.scatter(
            np.full(np.sum(success_mask), x_pos),
            y_vals[success_mask],
            s=9,
            alpha=min(0.95, alpha + 0.15),
            color="green",
            label=green_label,
        )


def plot_one_grammar_success_flow(eval_dir: Path, run_name: str, out_dir: Path, *, success_z: float = 2.0):
    """
    Sequential success-flow z cloud for one grammar.

    Panel i:
        plot all good_idx genes on chunk i.
        green = z_i > success_z, red = not successful.

    Panel j:
        plot only genes successful on chunk i.
        green = z_j > success_z, red = failed on j.

    Panel k:
        plot only genes successful on both chunk i and chunk j.
        green = z_k > success_z, red = failed on k.
    """
    pop_dirs = sorted((eval_dir / "grammars" / run_name).glob("pop_*"))
    if not pop_dirs:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for p_i, pop_dir in enumerate(pop_dirs):
        path = pop_dir / "eval_arrays.npz"
        if not path.exists():
            continue

        data = np.load(path, allow_pickle=True)
        z_i = np.asarray(data["z_i"], dtype=float)
        z_j = np.asarray(data["z_j"], dtype=float)
        z_k = np.asarray(data["z_k"], dtype=float)
        gidx = _valid_gidx_for_z(z_i, data["good_idx"])
        gidx = gidx[(gidx < z_j.size) & (gidx < z_k.size)]

        alpha = 0.10 + 0.45 * ((p_i + 1) / len(pop_dirs))

        # chunk i: all valid good_idx genes.
        idx_i = gidx[np.isfinite(z_i[gidx])]
        succ_i = z_i[idx_i] > success_z
        _scatter_success_split(
            axs[0],
            p_i,
            z_i[idx_i],
            succ_i,
            alpha=alpha,
            red_label="failed i" if p_i == 0 else None,
            green_label="success i" if p_i == 0 else None,
        )

        # chunk j: only genes that succeeded in i.
        idx_j_base = idx_i[succ_i]
        idx_j = idx_j_base[np.isfinite(z_j[idx_j_base])]
        succ_j = z_j[idx_j] > success_z
        _scatter_success_split(
            axs[1],
            p_i,
            z_j[idx_j],
            succ_j,
            alpha=alpha,
            red_label="i success, failed j" if p_i == 0 else None,
            green_label="i+j success" if p_i == 0 else None,
        )

        # chunk k: only genes that succeeded in i and j.
        idx_k_base = idx_j[succ_j]
        idx_k = idx_k_base[np.isfinite(z_k[idx_k_base])]
        succ_k = z_k[idx_k] > success_z
        _scatter_success_split(
            axs[2],
            p_i,
            z_k[idx_k],
            succ_k,
            alpha=alpha,
            red_label="i+j success, failed k" if p_i == 0 else None,
            green_label="i+j+k success" if p_i == 0 else None,
        )

    titles = [
        "chunk i: all genes, mark i success",
        "chunk j: i-success genes only",
        "chunk k: i+j-success genes only",
    ]

    for ax, title in zip(axs, titles):
        ax.axhline(success_z, linestyle="--", alpha=0.55, color="black")
        ax.axhline(0, alpha=0.75, color="black")
        ax.set_title(title)
        ax.set_xlabel("population")
        ax.set_ylabel("z-score")
        ax.set_ylim(bottom=-3)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{run_name} sequential success flow")
    fig.savefig(out_dir / f"z_success_flow_{run_name}.png", dpi=160, bbox_inches="tight")
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

    z_cloud_dir = out_dir / "z_clouds_surf"
    z_flow_dir = out_dir / "z_clouds_success_flow"
    z_cloud_dir.mkdir(parents=True, exist_ok=True)
    z_flow_dir.mkdir(parents=True, exist_ok=True)

    # Per-grammar z-cloud plots.
    for rn in sorted(grammar["run_name"].dropna().unique().tolist()):
        plot_one_grammar(eval_dir, rn, z_cloud_dir)
        plot_one_grammar_success_flow(eval_dir, rn, z_flow_dir)

    print("wrote plots to", out_dir)
    print("wrote original z clouds to", z_cloud_dir)
    print("wrote success-flow z clouds to", z_flow_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python mcts_surface_viz.py <mcts_surface_eval_runs/surface_eval_...>")
    make_all_plots(sys.argv[1])
