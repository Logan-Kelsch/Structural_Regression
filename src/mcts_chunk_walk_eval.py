"""
mcts_chunk_walk_eval.py

Evaluate one saved fin.g grammar across many chunks.

Given a trained run folder or direct fin.g path, this script:
1. Loads fin.g.
2. Forces grammar into infer mode using G.mode("infer") when available.
3. Disables generation/expansion of unexplored states/edges.
4. Sets inference softmax temperature to 1.0.
5. Generates one or more populations sequentially from the grammar.
6. Fits a separate FPC curve for every chunk 0..LAST_CHUNK.
7. Evaluates every generated population on every chunk.
8. Uses chunk 0 and chunk 1 as i/j success filters:
       i_success = z_chunk_0 > SUCCESS_Z
       j_success = z_chunk_1 > SUCCESS_Z
       ij_success = i_success & j_success
9. For every chunk, plots participation proportion vs observed EV.
   Points are colored:
       green = ij_success from chunks 0 and 1
       red   = not ij_success
   FPC mean and +/-2sd bands are overlaid.

Run examples
------------
python mcts_chunk_walk_eval.py mcts_runs/surf_t0030_d05
python mcts_chunk_walk_eval.py mcts_runs/surf_t0030_d05/fin.g
python mcts_chunk_walk_eval.py mcts_runs/surf_t0030_d05 --last-chunk 20 --n-populations 1

Notes
-----
LAST_CHUNK is inclusive. Default 20 means chunks 0, 1, ..., 20.
"""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from copy import deepcopy
import argparse
import csv
import json
import math
import time
import traceback
from typing import Any

import dill
import numpy as np
import matplotlib.pyplot as plt

import initialization as _I
import evaluation as _E
import visualization as _V
import transform_ops as _OPS
import mcts_util as _MU

reload(_E)
reload(_V)
reload(_I)
reload(_MU)

plt.ioff()


# ---------------------------------------------------------------------
# default settings
# ---------------------------------------------------------------------

LAST_CHUNK = 20
N_POPULATIONS = 1
FPC_N_SIMS = 2500
SUCCESS_Z = 2.0
INFER_SOFTMAX_TEMP = 1.0
BAND_MULT = 2.0
GRID_N = 400
VERBOSE = True
LOG_TO_FILE = True
CONTINUE_ON_POP_CRASH = True


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=indent)


def append_csv(path: str | Path, row: dict, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def finite_mean(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def finite_max(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


def finite_min(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else np.nan


def resolve_fin_path(path_text: str | Path) -> tuple[Path, Path]:
    p = Path(path_text)
    if p.is_dir():
        fin_path = p / "fin.g"
        run_dir = p
    else:
        fin_path = p
        run_dir = p.parent

    if not fin_path.exists():
        raise FileNotFoundError(f"Could not find fin.g at: {fin_path}")

    return run_dir.resolve(), fin_path.resolve()


def load_grammar(path: str | Path):
    with open(path, "rb") as f:
        return dill.load(f)


def make_log_fn(out_dir: Path):
    log_path = out_dir / "chunk_walk_eval.log"

    def log(msg: Any = "", *, force: bool = False):
        text = str(msg)
        if VERBOSE or force:
            print(text, flush=True)
        if LOG_TO_FILE:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")

    return log


def set_grammar_softmax_temp(G, temp: float) -> None:
    temp = float(temp)
    setattr(G, "_softmax_temp", temp)

    for attr in ("_spec_gram_args", "spec_gram_args", "_MCTS_SPEC_GRAM_ARGS", "_mcts_spec_gram_args"):
        d = getattr(G, attr, None)
        if isinstance(d, dict):
            d["softmax_temp"] = temp


def prepare_grammar_for_inference(G, *, infer_temp: float = 1.0):
    mode_fn = getattr(G, "mode", None)
    if callable(mode_fn):
        try:
            mode_fn("infer")
        except TypeError:
            setattr(G, "_mode", "infer")
    else:
        setattr(G, "_mode", "infer")

    if hasattr(G, "freeze_mcts_expansion"):
        try:
            G.freeze_mcts_expansion(True)
        except Exception:
            setattr(G, "_MCTS_FREEZE_EXPANSION", True)
    else:
        setattr(G, "_MCTS_FREEZE_EXPANSION", True)

    for attr in ("_MCTS_EXPAND_PROB", "_MCTS_ALPHA_EXPAND_PROB"):
        if hasattr(G, attr):
            setattr(G, attr, 0.0)

    for attr in ("_spec_gram_args", "spec_gram_args", "_MCTS_SPEC_GRAM_ARGS", "_mcts_spec_gram_args"):
        d = getattr(G, attr, None)
        if isinstance(d, dict):
            d["expand_prob"] = 0.0
            d["alpha_expand_prob"] = 0.0
            d["trace_enabled"] = False
            d["softmax_temp"] = float(infer_temp)

    set_grammar_softmax_temp(G, infer_temp)
    return G


def make_initialization_kwargs(G):
    kwargs = _MU.default_initialization_kwargs(G)
    kwargs["grmr_prior"] = G
    kwargs["grmr_type"] = "MCTS"
    kwargs["grmr_p_mttn"] = 0.0
    kwargs["grmr_p_csvr"] = 0.0
    kwargs["verbose"] = 0
    return kwargs


def make_solver_kwargs():
    return _MU.default_solver_kwargs()


def generate_population_from_grammar(G):
    kwargs = make_initialization_kwargs(G)
    X, G_out = _I.initialize(**kwargs)
    return X, G_out


def instantiate_for_chunk(X, chunk_num: int):
    return _I.instantiate_from_ops_chunked_intraday(
        X,
        transform_ops=_OPS,
        chunk_num=int(chunk_num),
        chunk_B=8,
    )


# ---------------------------------------------------------------------
# FPC fitting and evaluation
# ---------------------------------------------------------------------

def fit_fpc_curve_for_chunk(X_ref, chunk_num: int, solver_kwargs: dict, fpc_dir: Path, log):
    t0 = time.time()
    chunk_num = int(chunk_num)
    log(f"[start] FPC chunk {chunk_num}", force=True)

    instantiate_for_chunk(X_ref, chunk_num)

    curve, params, perm_mu = _E.fit_FPC_part_prop(
        X=X_ref,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        n_sims=FPC_N_SIMS,
    )

    p_grid = np.linspace(0.001, 0.999, GRID_N)
    v_grid = np.asarray(curve(p_grid), dtype=np.float64)
    s_grid = np.sqrt(np.maximum(v_grid, 0.0))

    np.savez_compressed(
        fpc_dir / f"fpc_curve_chunk_{chunk_num:03d}.npz",
        p_grid=p_grid,
        v_grid=v_grid,
        s_grid=s_grid,
        perm_mu=float(perm_mu),
    )

    write_json(fpc_dir / f"fpc_curve_chunk_{chunk_num:03d}.json", {
        "chunk_num": chunk_num,
        "fpc_params": params,
        "perm_mu": float(perm_mu),
        "elapsed_sec": time.time() - t0,
    })

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(p_grid, v_grid, label="FPC variance")
    ax.set_title(f"FPC variance curve | chunk {chunk_num}")
    ax.set_xlabel("participation proportion")
    ax.set_ylabel("null variance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(fpc_dir / f"fpc_curve_chunk_{chunk_num:03d}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    log(f"[done]  FPC chunk {chunk_num} | elapsed={time.time() - t0:.2f}s", force=True)

    return {
        "curve": curve,
        "params": params,
        "perm_mu": float(perm_mu),
        "p_grid": p_grid,
        "v_grid": v_grid,
        "s_grid": s_grid,
        "elapsed_sec": time.time() - t0,
    }


def fit_all_fpc_curves(chunks: list[int], X_ref, out_dir: Path, log) -> dict[int, dict]:
    solver_kwargs = make_solver_kwargs()
    fpc_dir = out_dir / "fpc_curves"
    fpc_dir.mkdir(parents=True, exist_ok=True)

    fpc = {}
    for chunk_num in chunks:
        fpc[int(chunk_num)] = fit_fpc_curve_for_chunk(
            X_ref=X_ref,
            chunk_num=int(chunk_num),
            solver_kwargs=solver_kwargs,
            fpc_dir=fpc_dir,
            log=log,
        )
    return fpc


def evaluate_population_on_chunk(X, chunk_num: int, fpc_info: dict, good_idx=None):
    solver_kwargs = make_solver_kwargs()
    instantiate_for_chunk(X, chunk_num)

    if good_idx is None:
        good_idx = np.asarray(X._G_idx, dtype=int)

    pvals, zscores, details = _E.evaluate_genes_from_opg_fpc_fast(
        population=X,
        good_idx=good_idx,
        chunk_num=int(chunk_num),
        solver_kwargs=solver_kwargs,
        fpc_curve=fpc_info["curve"],
        perm_mu=fpc_info["perm_mu"],
        fpc_params=fpc_info.get("params"),
        visualize=False,
        return_details=True,
    )

    return {
        "chunk_num": int(chunk_num),
        "pvals": pvals,
        "zscores": zscores,
        "details": details,
    }


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def fpc_band_grid(fpc_info: dict, prop_values=None, *, band_mult: float = BAND_MULT):
    if prop_values is None:
        p_min, p_max = 0.001, 0.999
    else:
        p = np.asarray(prop_values, dtype=np.float64).ravel()
        p = p[np.isfinite(p) & (p > 0)]
        if p.size == 0:
            p_min, p_max = 0.001, 0.999
        else:
            p_min = max(0.001, float(np.nanmin(p)) * 0.90)
            p_max = min(0.999, float(np.nanmax(p)) * 1.10)
            if p_max <= p_min:
                p_min, p_max = 0.001, 0.999

    p_grid = np.linspace(p_min, p_max, GRID_N)
    v_grid = np.asarray(fpc_info["curve"](p_grid), dtype=np.float64)
    s_grid = np.sqrt(np.maximum(v_grid, 0.0))
    perm_mu = float(fpc_info["perm_mu"])
    return p_grid, perm_mu, perm_mu + band_mult * s_grid, perm_mu - band_mult * s_grid


def plot_prop_ev_chunk(
    *,
    chunk_num: int,
    details: dict,
    fpc_info: dict,
    ij_success_global: np.ndarray,
    out_path: Path,
    success_z: float = SUCCESS_Z,
    band_mult: float = BAND_MULT,
    title_extra: str = "",
):
    prop_g = np.asarray(details.get("prop_g", []), dtype=np.float64)
    obs_g = np.asarray(details.get("observed_scores_g", []), dtype=np.float64)
    gidx = np.asarray(details.get("gidx", []), dtype=int)

    keep = (
        np.isfinite(prop_g)
        & np.isfinite(obs_g)
        & (prop_g > 0)
        & (gidx >= 0)
        & (gidx < ij_success_global.size)
    )

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    if not np.any(keep):
        ax.text(0.5, 0.5, "no finite prop/EV values", ha="center", va="center")
        ax.set_axis_off()
    else:
        prop = prop_g[keep]
        obs = obs_g[keep]
        gi = gidx[keep]
        success = ij_success_global[gi]

        # red failures first, green i+j successes on top
        fail = ~success
        ax.scatter(prop[fail], obs[fail], s=12, alpha=0.35, color="red", label="not i+j success")
        ax.scatter(prop[success], obs[success], s=18, alpha=0.85, color="green", label="i+j success")

        p_grid, mean, upper, lower = fpc_band_grid(fpc_info, prop_values=prop, band_mult=band_mult)
        ax.plot(p_grid, np.full_like(p_grid, mean), color="black", linewidth=2, label="permutation mean")
        ax.plot(p_grid, upper, color="gray", linestyle="--", linewidth=1.5, label=f"+{band_mult:g} sd")
        ax.plot(p_grid, lower, color="gray", linestyle="--", linewidth=1.5, label=f"-{band_mult:g} sd")

        ax.set_xlabel("gene participation proportion p = m / N")
        ax.set_ylabel("observed EV")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    ax.set_title(f"prop vs EV | chunk {chunk_num}{title_extra}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chunk_grid(
    *,
    chunk_eval_data: dict[int, dict],
    fpc_curves: dict[int, dict],
    ij_success_global: np.ndarray,
    out_path: Path,
    title: str,
    band_mult: float = BAND_MULT,
):
    chunks = sorted(chunk_eval_data.keys())
    n = len(chunks)
    ncols = 4
    nrows = int(math.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), constrained_layout=True)
    axs = np.asarray(axs).ravel()

    for ax_i, chunk_num in enumerate(chunks):
        ax = axs[ax_i]
        details = chunk_eval_data[chunk_num]["details"]
        fpc_info = fpc_curves[chunk_num]

        prop_g = np.asarray(details.get("prop_g", []), dtype=np.float64)
        obs_g = np.asarray(details.get("observed_scores_g", []), dtype=np.float64)
        gidx = np.asarray(details.get("gidx", []), dtype=int)

        keep = (
            np.isfinite(prop_g)
            & np.isfinite(obs_g)
            & (prop_g > 0)
            & (gidx >= 0)
            & (gidx < ij_success_global.size)
        )

        if not np.any(keep):
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
            ax.set_axis_off()
            continue

        prop = prop_g[keep]
        obs = obs_g[keep]
        gi = gidx[keep]
        success = ij_success_global[gi]

        ax.scatter(prop[~success], obs[~success], s=6, alpha=0.20, color="red")
        ax.scatter(prop[success], obs[success], s=9, alpha=0.70, color="green")

        p_grid, mean, upper, lower = fpc_band_grid(fpc_info, prop_values=prop, band_mult=band_mult)
        ax.plot(p_grid, np.full_like(p_grid, mean), color="black", linewidth=1.2)
        ax.plot(p_grid, upper, color="gray", linestyle="--", linewidth=0.9)
        ax.plot(p_grid, lower, color="gray", linestyle="--", linewidth=0.9)

        z = np.asarray(chunk_eval_data[chunk_num]["zscores"], dtype=np.float64)
        z_success_vals = z[ij_success_global & np.isfinite(z)]
        ax.set_title(
            f"chunk {chunk_num} | ij_n={int(np.sum(ij_success_global))} | "
            f"ij mean z={finite_mean(z_success_vals):.2f}",
            fontsize=9,
        )
        ax.set_xlabel("prop")
        ax.set_ylabel("EV")
        ax.grid(alpha=0.20)

    for ax in axs[n:]:
        ax.set_axis_off()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ij_success_z_over_chunks(
    *,
    chunk_eval_data: dict[int, dict],
    ij_success_global: np.ndarray,
    out_path: Path,
):
    chunks = sorted(chunk_eval_data.keys())
    means = []
    medians = []
    maxes = []
    counts = []

    for c in chunks:
        z = np.asarray(chunk_eval_data[c]["zscores"], dtype=np.float64)
        vals = z[ij_success_global & np.isfinite(z)]
        means.append(float(np.mean(vals)) if vals.size else np.nan)
        medians.append(float(np.median(vals)) if vals.size else np.nan)
        maxes.append(float(np.max(vals)) if vals.size else np.nan)
        counts.append(int(vals.size))

    fig, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax1.plot(chunks, means, marker="o", label="mean z of i+j successes")
    ax1.plot(chunks, medians, marker="o", label="median z of i+j successes")
    ax1.plot(chunks, maxes, marker="o", alpha=0.45, label="max z of i+j successes")
    ax1.axhline(0, alpha=0.25)
    ax1.axhline(SUCCESS_Z, linestyle="--", alpha=0.35, label=f"z={SUCCESS_Z:g}")
    ax1.set_xlabel("chunk")
    ax1.set_ylabel("z-score")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(chunks, counts, color="gray", linestyle=":", marker="x", label="finite ij-success count")
    ax2.set_ylabel("count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.set_title("i+j success group performance across chunks")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# population evaluation
# ---------------------------------------------------------------------

def evaluate_one_population(
    *,
    G,
    fpc_curves: dict[int, dict],
    chunks: list[int],
    pop_id: int,
    out_dir: Path,
    log,
) -> dict:
    pop_dir = out_dir / "populations" / f"pop_{pop_id:03d}"
    plot_dir = pop_dir / "prop_ev_plots"
    pop_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"[start] population {pop_id}", force=True)

    X, _ = generate_population_from_grammar(G)
    good_idx = np.asarray(X._G_idx, dtype=int)

    chunk_eval_data = {}
    for chunk_num in chunks:
        log(f"  eval population {pop_id} chunk {chunk_num}")
        chunk_eval_data[chunk_num] = evaluate_population_on_chunk(
            X=X,
            chunk_num=chunk_num,
            fpc_info=fpc_curves[chunk_num],
            good_idx=good_idx,
        )

    # i/j are chunk 0 and 1 from this run's chunk list, not necessarily names.
    i_chunk = chunks[0]
    j_chunk = chunks[1]
    z_i = np.asarray(chunk_eval_data[i_chunk]["zscores"], dtype=np.float64)
    z_j = np.asarray(chunk_eval_data[j_chunk]["zscores"], dtype=np.float64)

    max_size = max(z_i.size, z_j.size)
    ij_success_global = np.zeros(max_size, dtype=bool)
    finite_ij = np.isfinite(z_i) & np.isfinite(z_j)
    ij_success_global[:z_i.size] = finite_ij & (z_i > SUCCESS_Z) & (z_j > SUCCESS_Z)

    i_success_global = np.isfinite(z_i) & (z_i > SUCCESS_Z)
    j_success_global = np.isfinite(z_j) & (z_j > SUCCESS_Z)

    # save arrays for every chunk
    for chunk_num in chunks:
        ev = chunk_eval_data[chunk_num]
        d = ev["details"]
        np.savez_compressed(
            pop_dir / f"eval_chunk_{chunk_num:03d}.npz",
            good_idx=good_idx,
            pvals=ev["pvals"],
            zscores=ev["zscores"],
            prop_g=np.asarray(d.get("prop_g", []), dtype=np.float64),
            observed_scores_g=np.asarray(d.get("observed_scores_g", []), dtype=np.float64),
            fpc_std_g=np.asarray(d.get("fpc_std_g", []), dtype=np.float64),
            gidx=np.asarray(d.get("gidx", []), dtype=int),
            ij_success_global=ij_success_global,
            i_success_global=i_success_global,
            j_success_global=j_success_global,
            perm_mu=float(fpc_curves[chunk_num]["perm_mu"]),
        )

        plot_prop_ev_chunk(
            chunk_num=chunk_num,
            details=d,
            fpc_info=fpc_curves[chunk_num],
            ij_success_global=ij_success_global,
            out_path=plot_dir / f"prop_ev_chunk_{chunk_num:03d}.png",
            title_extra=f" | pop {pop_id}",
        )

    plot_chunk_grid(
        chunk_eval_data=chunk_eval_data,
        fpc_curves=fpc_curves,
        ij_success_global=ij_success_global,
        out_path=pop_dir / "prop_ev_all_chunks_grid.png",
        title=f"prop vs EV across chunks | pop {pop_id} | green = chunk {i_chunk}+{j_chunk} success",
    )

    plot_ij_success_z_over_chunks(
        chunk_eval_data=chunk_eval_data,
        ij_success_global=ij_success_global,
        out_path=pop_dir / "ij_success_z_over_chunks.png",
    )

    # summary rows by chunk
    chunk_summary_rows = []
    for chunk_num in chunks:
        z = np.asarray(chunk_eval_data[chunk_num]["zscores"], dtype=np.float64)
        vals_good = z[good_idx] if good_idx.size else np.asarray([], dtype=np.float64)
        vals_ij = z[ij_success_global & np.isfinite(z)]

        row = {
            "pop_id": int(pop_id),
            "chunk_num": int(chunk_num),
            "n_good_idx": int(good_idx.size),
            "i_success_n": int(np.sum(i_success_global)),
            "j_success_n": int(np.sum(j_success_global)),
            "ij_success_n": int(np.sum(ij_success_global)),
            "mean_z_good": finite_mean(vals_good),
            "max_z_good": finite_max(vals_good),
            "min_z_good": finite_min(vals_good),
            "mean_z_ij_success": finite_mean(vals_ij),
            "max_z_ij_success": finite_max(vals_ij),
            "finite_z_ij_success_n": int(np.isfinite(vals_ij).sum()),
        }
        chunk_summary_rows.append(row)

    chunk_fields = list(chunk_summary_rows[0].keys()) if chunk_summary_rows else []
    chunk_csv = pop_dir / "chunk_summary.csv"
    for row in chunk_summary_rows:
        append_csv(chunk_csv, row, chunk_fields)

    pop_summary = {
        "pop_id": int(pop_id),
        "i_chunk": int(i_chunk),
        "j_chunk": int(j_chunk),
        "i_success_n": int(np.sum(i_success_global)),
        "j_success_n": int(np.sum(j_success_global)),
        "ij_success_n": int(np.sum(ij_success_global)),
        "elapsed_sec": time.time() - t0,
        "status": "complete",
        "error": "",
    }
    write_json(pop_dir / "summary.json", pop_summary)
    log(f"[done] population {pop_id} | ij_success_n={pop_summary['ij_success_n']} | elapsed={time.time() - t0:.2f}s", force=True)
    return pop_summary


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    global LAST_CHUNK, N_POPULATIONS, FPC_N_SIMS, SUCCESS_Z, INFER_SOFTMAX_TEMP

    parser = argparse.ArgumentParser()
    parser.add_argument("grammar", help="Path to trained run folder or fin.g")
    parser.add_argument("--out-name", default=None, help="Optional output folder name inside the trained run folder")
    parser.add_argument("--last-chunk", type=int, default=LAST_CHUNK, help="Inclusive last chunk. Default 20 means 0..20.")
    parser.add_argument("--n-populations", type=int, default=N_POPULATIONS)
    parser.add_argument("--fpc-n-sims", type=int, default=FPC_N_SIMS)
    parser.add_argument("--success-z", type=float, default=SUCCESS_Z)
    parser.add_argument("--infer-temp", type=float, default=INFER_SOFTMAX_TEMP)
    args = parser.parse_args()

    LAST_CHUNK = int(args.last_chunk)
    N_POPULATIONS = int(args.n_populations)
    FPC_N_SIMS = int(args.fpc_n_sims)
    SUCCESS_Z = float(args.success_z)
    INFER_SOFTMAX_TEMP = float(args.infer_temp)

    if LAST_CHUNK < 1:
        raise ValueError("last_chunk must be at least 1 because chunks 0 and 1 are used for i+j success selection")

    chunks = list(range(0, LAST_CHUNK + 1))

    run_dir, fin_path = resolve_fin_path(args.grammar)
    out_name = args.out_name or time.strftime("chunk_walk_eval_%Y%m%d_%H%M%S")
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "populations").mkdir(parents=True, exist_ok=True)

    log = make_log_fn(out_dir)

    write_json(out_dir / "config.json", {
        "run_dir": run_dir,
        "fin_path": fin_path,
        "out_dir": out_dir,
        "chunks": chunks,
        "last_chunk_inclusive": LAST_CHUNK,
        "n_populations": N_POPULATIONS,
        "fpc_n_sims": FPC_N_SIMS,
        "success_z": SUCCESS_Z,
        "infer_softmax_temp": INFER_SOFTMAX_TEMP,
        "color_rule": "green = z_chunk0 > success_z and z_chunk1 > success_z; red = otherwise",
    })

    log(f"output: {out_dir}", force=True)
    log(f"fin.g: {fin_path}", force=True)
    log(f"chunks: {chunks[0]}..{chunks[-1]} inclusive", force=True)
    log(f"n populations: {N_POPULATIONS}", force=True)

    # Fit FPC curves once per chunk using a reference population from the same infer grammar.
    G_ref = prepare_grammar_for_inference(load_grammar(fin_path), infer_temp=INFER_SOFTMAX_TEMP)
    X_ref, _ = generate_population_from_grammar(G_ref)
    fpc_curves = fit_all_fpc_curves(chunks, X_ref=X_ref, out_dir=out_dir, log=log)

    pop_csv = out_dir / "population_summary.csv"
    pop_fields = ["pop_id", "i_chunk", "j_chunk", "i_success_n", "j_success_n", "ij_success_n", "elapsed_sec", "status", "error"]

    all_pop_rows = []
    total_start = time.time()

    for pop_id in range(N_POPULATIONS):
        try:
            G = prepare_grammar_for_inference(deepcopy(load_grammar(fin_path)), infer_temp=INFER_SOFTMAX_TEMP)
            row = evaluate_one_population(
                G=G,
                fpc_curves=fpc_curves,
                chunks=chunks,
                pop_id=pop_id,
                out_dir=out_dir,
                log=log,
            )
        except Exception:
            err = traceback.format_exc()
            row = {
                "pop_id": int(pop_id),
                "i_chunk": 0,
                "j_chunk": 1,
                "i_success_n": np.nan,
                "j_success_n": np.nan,
                "ij_success_n": np.nan,
                "elapsed_sec": np.nan,
                "status": "crashed",
                "error": err,
            }
            write_json(out_dir / "populations" / f"pop_{pop_id:03d}" / "CRASH.json", row)
            log(err, force=True)
            if not CONTINUE_ON_POP_CRASH:
                raise

        all_pop_rows.append(row)
        append_csv(pop_csv, row, pop_fields)

    write_json(out_dir / "done.json", {
        "status": "complete",
        "elapsed_sec": time.time() - total_start,
        "out_dir": out_dir,
        "n_populations": N_POPULATIONS,
    })

    log("\ncomplete", force=True)
    log(f"output: {out_dir}", force=True)
    log(f"population summary: {pop_csv}", force=True)


if __name__ == "__main__":
    main()
