"""
mcts_count_infer_temp_surface_eval.py

One-grammar two-dimensional inference sweep over:
1. visit-count / experience temperature tau
2. inference softmax temperature

Given a single trained run folder or fin.g path, this script:
1. Loads fin.g.
2. Fits separate FPC curves for chunks i, j, k.
3. Sweeps geom-spaced count temperatures.
4. Sweeps geom-spaced inference softmax temperatures.
5. For each (tau, infer_temp), transforms Q values using exploit visit counts.
6. Forces infer mode, no expansion, and sets the selected inference softmax temp.
7. Generates N populations sequentially.
8. Evaluates each population on chunks i, j, k.
9. Scores each population by mean z_k among genes successful in both i and j.
10. Saves results to mcts_temp_surface_eval_runs/<source>_<timestamp>.

Run examples
------------
python mcts_count_infer_temp_surface_eval.py mcts_runs/surf_t0030_d05
python mcts_count_infer_temp_surface_eval.py mcts_runs/surf_t0030_d05/fin.g

Terminology
-----------
The parameter swept here is a visit-count confidence temperature, also called
count-tempered Q shrinkage. It controls how much to trust learned Q values based
on exploit visit count n:

    Q_tempered = prior + (Q - prior) * (1 - exp(-n / tau))

Small tau:
    trust Q quickly, even with fewer visits.
Large tau:
    require many visits before trusting Q; low-count Q values shrink toward prior.
"""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from copy import deepcopy
import argparse
import csv
import json
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

BASE_CHUNK_NUM = 0
CHUNKS = {
    "i": BASE_CHUNK_NUM + 0,
    "j": BASE_CHUNK_NUM + 1,
    "k": BASE_CHUNK_NUM + 2,
}

N_POPULATIONS = 30
FPC_N_SIMS = 2500
SUCCESS_Z = 2.0
# Default infer softmax temp used only for the reference FPC population.
# The actual sweep uses INFER_SOFTMAX_TEMPS below.
INFER_SOFTMAX_TEMP = 1.0

# Visit-count temperature grid.
# tau is in units of exploit visits/counts.
# Q reaches about 63% trust at n=tau and about 95% trust at n=3*tau.
COUNT_TEMPS = np.geomspace(1.0, 100.0, 5)

# Inference softmax-temperature grid.
# Lower values are sharper/greedier; higher values are flatter/more diffuse.
INFER_SOFTMAX_TEMPS = np.geomspace(0.03, 1.0, 8)

OUT_ROOT = Path("mcts_temp_surface_eval_runs")

VERBOSE = True
LOG_TO_FILE = True
CONTINUE_ON_POP_CRASH = True

# Count-tempering scope.
APPLY_TO_XTF_NODES = True
APPLY_TO_XTF_EDGES = True
APPLY_TO_ALPHA = False


# ---------------------------------------------------------------------
# small helpers
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


def finite_std(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else np.nan


def finite_max(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


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
    log_path = out_dir / "count_infer_temp_surface_eval.log"

    def log(msg: Any = "", *, force: bool = False):
        text = str(msg)
        if VERBOSE or force:
            print(text, flush=True)
        if LOG_TO_FILE:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")

    return log


# ---------------------------------------------------------------------
# grammar preparation and count tempering
# ---------------------------------------------------------------------

def mcts_getdict(G, name: str) -> dict:
    d = getattr(G, name, None)
    return {} if d is None else d


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


def count_confidence(n, tau: float) -> float:
    """
    Confidence weight for exploit count n.

    tau controls the count scale:
        n=tau   -> confidence ~= 0.632
        n=3*tau -> confidence ~= 0.950
    """
    tau = max(float(tau), 1e-12)
    n = max(float(n), 0.0)
    return float(1.0 - np.exp(-n / tau))


def temper_mu_dict_by_count(mu_dict: dict, count_dict: dict, *, tau: float, prior: float = 0.0) -> dict:
    out = {}
    for key, q in mu_dict.items():
        n = count_dict.get(key, 0.0)
        w = count_confidence(n, tau)
        out[key] = float(prior + (float(q) - float(prior)) * w)
    return out


def apply_visit_count_temperature(G, tau: float, *, prior: float | None = None) -> dict:
    """
    Apply count-tempered Q shrinkage to a grammar copy.

    This directly rewrites the Q/mean dictionaries used in inference:
        _MCTS_NODE_MU
        _MCTS_EDGE_MU
    and optionally alpha-MCTS Q dictionaries.
    """
    if prior is None:
        prior = float(getattr(G, "_MCTS_BASE_PRIOR", 0.0) or 0.0)

    summary = {
        "tau": float(tau),
        "prior": float(prior),
        "node_mu_n": len(mcts_getdict(G, "_MCTS_NODE_MU")),
        "edge_mu_n": len(mcts_getdict(G, "_MCTS_EDGE_MU")),
        "alpha_decision_mu_n": len(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")),
        "alpha_node_mu_n": len(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")),
        "alpha_edge_mu_n": len(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")),
    }

    if APPLY_TO_XTF_NODES and hasattr(G, "_MCTS_NODE_MU"):
        G._MCTS_NODE_MU = temper_mu_dict_by_count(
            mcts_getdict(G, "_MCTS_NODE_MU"),
            mcts_getdict(G, "_MCTS_NODE_COUNT"),
            tau=tau,
            prior=prior,
        )

    if APPLY_TO_XTF_EDGES and hasattr(G, "_MCTS_EDGE_MU"):
        G._MCTS_EDGE_MU = temper_mu_dict_by_count(
            mcts_getdict(G, "_MCTS_EDGE_MU"),
            mcts_getdict(G, "_MCTS_EDGE_COUNT"),
            tau=tau,
            prior=prior,
        )

    if APPLY_TO_ALPHA:
        if hasattr(G, "_MCTS_ALPHA_DECISION_MU"):
            G._MCTS_ALPHA_DECISION_MU = temper_mu_dict_by_count(
                mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU"),
                mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT"),
                tau=tau,
                prior=prior,
            )
        if hasattr(G, "_MCTS_ALPHA_NODE_MU"):
            G._MCTS_ALPHA_NODE_MU = temper_mu_dict_by_count(
                mcts_getdict(G, "_MCTS_ALPHA_NODE_MU"),
                mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT"),
                tau=tau,
                prior=prior,
            )
        if hasattr(G, "_MCTS_ALPHA_EDGE_MU"):
            G._MCTS_ALPHA_EDGE_MU = temper_mu_dict_by_count(
                mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU"),
                mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT"),
                tau=tau,
                prior=prior,
            )

    return summary


# ---------------------------------------------------------------------
# generation and evaluation
# ---------------------------------------------------------------------

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


def fit_fpc_curve_for_chunk(X_ref, chunk_name: str, chunk_num: int, solver_kwargs: dict, out_dir: Path, log):
    t0 = time.time()
    log(f"[start] FPC chunk {chunk_name}={chunk_num}", force=True)

    instantiate_for_chunk(X_ref, chunk_num)

    curve, params, perm_mu = _E.fit_FPC_part_prop(
        X=X_ref,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        n_sims=FPC_N_SIMS,
    )

    p_grid = np.linspace(0.001, 0.999, 500)
    v_grid = np.asarray(curve(p_grid), dtype=np.float64)
    s_grid = np.sqrt(np.maximum(v_grid, 0.0))

    np.savez_compressed(
        out_dir / f"fpc_curve_chunk_{chunk_name}.npz",
        p_grid=p_grid,
        v_grid=v_grid,
        s_grid=s_grid,
        perm_mu=float(perm_mu),
    )

    write_json(out_dir / f"fpc_curve_chunk_{chunk_name}.json", {
        "chunk_name": chunk_name,
        "chunk_num": int(chunk_num),
        "fpc_params": params,
        "perm_mu": float(perm_mu),
        "elapsed_sec": time.time() - t0,
    })

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(p_grid, v_grid, label="FPC variance")
    ax.set_title(f"FPC curve | chunk {chunk_name}={chunk_num}")
    ax.set_xlabel("participation proportion")
    ax.set_ylabel("null variance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / f"fpc_curve_chunk_{chunk_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    log(f"[done]  FPC chunk {chunk_name}={chunk_num} | elapsed={time.time() - t0:.2f}s", force=True)

    return {
        "curve": curve,
        "params": params,
        "perm_mu": float(perm_mu),
        "elapsed_sec": time.time() - t0,
    }


def fit_fpc_curves(chunks: dict[str, int], X_ref, out_dir: Path, log) -> dict[str, dict]:
    solver_kwargs = make_solver_kwargs()
    fpc = {}
    for chunk_name, chunk_num in chunks.items():
        fpc[chunk_name] = fit_fpc_curve_for_chunk(
            X_ref=X_ref,
            chunk_name=chunk_name,
            chunk_num=chunk_num,
            solver_kwargs=solver_kwargs,
            out_dir=out_dir,
            log=log,
        )
    return fpc


def evaluate_population_on_chunk(X, chunk_name: str, chunk_num: int, fpc_info: dict, good_idx=None):
    solver_kwargs = make_solver_kwargs()
    instantiate_for_chunk(X, chunk_num)

    if good_idx is None:
        good_idx = np.asarray(X._G_idx, dtype=int)

    pvals, zscores, details = _E.evaluate_genes_from_opg_fpc_fast(
        population=X,
        good_idx=good_idx,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        fpc_curve=fpc_info["curve"],
        perm_mu=fpc_info["perm_mu"],
        fpc_params=fpc_info.get("params"),
        visualize=False,
        return_details=True,
    )

    return {
        "chunk_name": chunk_name,
        "chunk_num": int(chunk_num),
        "pvals": pvals,
        "zscores": zscores,
        "details": details,
    }


def group_masks(z_i, z_j, *, success_z=2.0):
    z_i = np.asarray(z_i, dtype=np.float64)
    z_j = np.asarray(z_j, dtype=np.float64)

    ok_i = np.isfinite(z_i) & (z_i > success_z)
    ok_j = np.isfinite(z_j) & (z_j > success_z)

    neither = ~ok_i & ~ok_j
    i_only = ok_i & ~ok_j
    j_only = ~ok_i & ok_j
    both = ok_i & ok_j

    return {
        "i_success": ok_i,
        "j_success": ok_j,
        "neither": neither,
        "i_only": i_only,
        "j_only": j_only,
        "both": both,
    }


def evaluate_one_generated_population(G, fpc_curves: dict, combo_out_dir: Path, pop_id: int, tau: float, infer_temp: float, log):
    pop_dir = combo_out_dir / f"pop_{pop_id:03d}"
    pop_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        log(f"    population {pop_id + 1}/{N_POPULATIONS}", force=True)
        X, _ = generate_population_from_grammar(G)
        good_idx = np.asarray(X._G_idx, dtype=int)

        evals = {}
        for chunk_name, chunk_num in CHUNKS.items():
            evals[chunk_name] = evaluate_population_on_chunk(
                X=X,
                chunk_name=chunk_name,
                chunk_num=chunk_num,
                fpc_info=fpc_curves[chunk_name],
                good_idx=good_idx,
            )

        z_i = evals["i"]["zscores"]
        z_j = evals["j"]["zscores"]
        z_k = evals["k"]["zscores"]

        masks = group_masks(z_i, z_j, success_z=SUCCESS_Z)
        both = masks["both"]

        k_both_vals = np.asarray(z_k, dtype=np.float64)[both]
        k_both_vals = k_both_vals[np.isfinite(k_both_vals)]
        pop_score = float(np.mean(k_both_vals)) if k_both_vals.size else np.nan

        row = {
            "count_temp_tau": float(tau),
            "infer_softmax_temp": float(infer_temp),
            "pop_id": int(pop_id),
            "score_k_mean_for_i_j_success": pop_score,
            "i_success_n": int(np.sum(masks["i_success"])),
            "j_success_n": int(np.sum(masks["j_success"])),
            "both_success_n": int(np.sum(masks["both"])),
            "i_only_n": int(np.sum(masks["i_only"])),
            "j_only_n": int(np.sum(masks["j_only"])),
            "neither_n": int(np.sum(masks["neither"])),
            "mean_z_i": finite_mean(z_i[good_idx]),
            "mean_z_j": finite_mean(z_j[good_idx]),
            "mean_z_k": finite_mean(z_k[good_idx]),
            "max_z_i": finite_max(z_i[good_idx]) if good_idx.size else np.nan,
            "max_z_j": finite_max(z_j[good_idx]) if good_idx.size else np.nan,
            "max_z_k": finite_max(z_k[good_idx]) if good_idx.size else np.nan,
            "pop_elapsed_sec": time.time() - t0,
            "status": "complete",
            "error": "",
        }

        np.savez_compressed(
            pop_dir / "eval_arrays.npz",
            good_idx=good_idx,
            z_i=z_i,
            z_j=z_j,
            z_k=z_k,
            p_i=evals["i"]["pvals"],
            p_j=evals["j"]["pvals"],
            p_k=evals["k"]["pvals"],
            i_success=masks["i_success"],
            j_success=masks["j_success"],
            group_neither=masks["neither"],
            group_i_only=masks["i_only"],
            group_j_only=masks["j_only"],
            group_both=masks["both"],
            k_both_vals=k_both_vals,
        )

        for chunk_name in ("i", "j", "k"):
            d = evals[chunk_name]["details"]
            np.savez_compressed(
                pop_dir / f"fpc_scatter_{chunk_name}.npz",
                prop_g=np.asarray(d.get("prop_g", []), dtype=np.float64),
                observed_scores_g=np.asarray(d.get("observed_scores_g", []), dtype=np.float64),
                fpc_std_g=np.asarray(d.get("fpc_std_g", []), dtype=np.float64),
                perm_mu=float(fpc_curves[chunk_name]["perm_mu"]),
            )

        write_json(pop_dir / "summary.json", row)
        return row

    except Exception:
        err = traceback.format_exc()
        row = {
            "count_temp_tau": float(tau),
            "infer_softmax_temp": float(infer_temp),
            "pop_id": int(pop_id),
            "score_k_mean_for_i_j_success": np.nan,
            "i_success_n": np.nan,
            "j_success_n": np.nan,
            "both_success_n": np.nan,
            "i_only_n": np.nan,
            "j_only_n": np.nan,
            "neither_n": np.nan,
            "mean_z_i": np.nan,
            "mean_z_j": np.nan,
            "mean_z_k": np.nan,
            "max_z_i": np.nan,
            "max_z_j": np.nan,
            "max_z_k": np.nan,
            "pop_elapsed_sec": time.time() - t0,
            "status": "crashed",
            "error": err,
        }
        write_json(pop_dir / "CRASH.json", row)
        log(err, force=True)
        if CONTINUE_ON_POP_CRASH:
            return row
        raise


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------

def _surface_matrix(pop_rows: list[dict], value_key: str, agg="mean"):
    taus = sorted({float(r["count_temp_tau"]) for r in pop_rows})
    infer_temps = sorted({float(r["infer_softmax_temp"]) for r in pop_rows})

    Z = np.full((len(taus), len(infer_temps)), np.nan, dtype=np.float64)

    for i, tau in enumerate(taus):
        for j, infer_temp in enumerate(infer_temps):
            rows = [
                r for r in pop_rows
                if r.get("status") == "complete"
                and np.isclose(float(r["count_temp_tau"]), tau)
                and np.isclose(float(r["infer_softmax_temp"]), infer_temp)
            ]
            vals = np.asarray([r.get(value_key, np.nan) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if agg == "mean":
                Z[i, j] = float(np.mean(vals))
            elif agg == "median":
                Z[i, j] = float(np.median(vals))
            elif agg == "frac_finite":
                all_vals = np.asarray([r.get(value_key, np.nan) for r in rows], dtype=np.float64)
                Z[i, j] = float(np.isfinite(all_vals).sum() / max(len(all_vals), 1))
            else:
                raise ValueError("agg must be 'mean', 'median', or 'frac_finite'")

    return np.asarray(taus), np.asarray(infer_temps), Z


def _plot_heat(ax, Z, taus, infer_temps, title, cbar_label):
    im = ax.imshow(Z, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("infer softmax temp")
    ax.set_ylabel("count temp tau")
    ax.set_xticks(np.arange(len(infer_temps)))
    ax.set_xticklabels([f"{x:g}" for x in infer_temps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(taus)))
    ax.set_yticklabels([f"{x:g}" for x in taus])
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label(cbar_label)


def plot_surface_summary(pop_rows: list[dict], out_path: Path, title: str):
    if len(pop_rows) == 0:
        return

    taus, infer_temps, score_mean = _surface_matrix(pop_rows, "score_k_mean_for_i_j_success", agg="mean")
    _, _, both_mean = _surface_matrix(pop_rows, "both_success_n", agg="mean")
    _, _, i_mean = _surface_matrix(pop_rows, "i_success_n", agg="mean")
    _, _, valid_frac = _surface_matrix(pop_rows, "score_k_mean_for_i_j_success", agg="frac_finite")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axs = axs.ravel()

    _plot_heat(axs[0], score_mean, taus, infer_temps, "mean k score for i+j successes", "mean z_k")
    _plot_heat(axs[1], both_mean, taus, infer_temps, "mean i+j success count", "count")
    _plot_heat(axs[2], i_mean, taus, infer_temps, "mean i success count", "count")
    _plot_heat(axs[3], valid_frac, taus, infer_temps, "valid score fraction", "fraction")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tau_lines_by_infer_temp(pop_rows: list[dict], out_path: Path, title: str):
    if len(pop_rows) == 0:
        return

    taus, infer_temps, score_mean = _surface_matrix(pop_rows, "score_k_mean_for_i_j_success", agg="mean")
    _, _, both_mean = _surface_matrix(pop_rows, "both_success_n", agg="mean")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    for j, infer_temp in enumerate(infer_temps):
        axs[0].plot(taus, score_mean[:, j], marker="o", label=f"infer={infer_temp:g}")
        axs[1].plot(taus, both_mean[:, j], marker="o", label=f"infer={infer_temp:g}")

    axs[0].set_xscale("log")
    axs[0].axhline(0, alpha=0.25)
    axs[0].set_title("score vs count temp")
    axs[0].set_xlabel("count temp tau")
    axs[0].set_ylabel("mean z_k of i+j successes")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].set_xscale("log")
    axs[1].set_title("i+j success count vs count temp")
    axs[1].set_xlabel("count temp tau")
    axs[1].set_ylabel("mean count")
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    global N_POPULATIONS, FPC_N_SIMS, CHUNKS, COUNT_TEMPS, INFER_SOFTMAX_TEMPS

    parser = argparse.ArgumentParser()
    parser.add_argument("grammar", help="Path to trained run folder or fin.g")
    parser.add_argument("--out-name", default=None, help="Optional output folder name inside mcts_temp_surface_eval_runs")
    parser.add_argument("--n-populations", type=int, default=N_POPULATIONS)
    parser.add_argument("--fpc-n-sims", type=int, default=FPC_N_SIMS)
    parser.add_argument("--base-chunk", type=int, default=BASE_CHUNK_NUM)
    parser.add_argument("--tau-min", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=100.0)
    parser.add_argument("--n-tau", type=int, default=5)
    parser.add_argument("--infer-temp-min", type=float, default=0.03)
    parser.add_argument("--infer-temp-max", type=float, default=1.0)
    parser.add_argument("--n-infer-temp", type=int, default=8)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    N_POPULATIONS = int(args.n_populations)
    FPC_N_SIMS = int(args.fpc_n_sims)
    CHUNKS = {
        "i": int(args.base_chunk) + 0,
        "j": int(args.base_chunk) + 1,
        "k": int(args.base_chunk) + 2,
    }
    COUNT_TEMPS = np.geomspace(float(args.tau_min), float(args.tau_max), int(args.n_tau))
    INFER_SOFTMAX_TEMPS = np.geomspace(
        float(args.infer_temp_min),
        float(args.infer_temp_max),
        int(args.n_infer_temp),
    )

    run_dir, fin_path = resolve_fin_path(args.grammar)
    source_name = run_dir.name
    out_root = Path(args.out_root)
    out_name = args.out_name or f"{source_name}_temp_surface_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surface").mkdir(parents=True, exist_ok=True)

    log = make_log_fn(out_dir)

    write_json(out_dir / "config.json", {
        "run_dir": run_dir,
        "fin_path": fin_path,
        "out_dir": out_dir,
        "source_run_dir": run_dir,
        "chunks": CHUNKS,
        "n_populations": N_POPULATIONS,
        "fpc_n_sims": FPC_N_SIMS,
        "success_z": SUCCESS_Z,
        "reference_fpc_infer_softmax_temp": INFER_SOFTMAX_TEMP,
        "count_temps": COUNT_TEMPS,
        "infer_softmax_temps": INFER_SOFTMAX_TEMPS,
        "method": "Q_tempered = prior + (Q - prior) * (1 - exp(-n / tau)); infer softmax temp swept independently",
        "apply_to_xtf_nodes": APPLY_TO_XTF_NODES,
        "apply_to_xtf_edges": APPLY_TO_XTF_EDGES,
        "apply_to_alpha": APPLY_TO_ALPHA,
    })

    log(f"output: {out_dir}", force=True)
    log(f"fin.g: {fin_path}", force=True)
    log(f"count temps: {np.round(COUNT_TEMPS, 4)}", force=True)
    log(f"infer softmax temps: {np.round(INFER_SOFTMAX_TEMPS, 4)}", force=True)

    # Reference FPC curves from the unmodified grammar in infer mode.
    G_ref = prepare_grammar_for_inference(load_grammar(fin_path), infer_temp=INFER_SOFTMAX_TEMP)
    X_ref, _ = generate_population_from_grammar(G_ref)
    fpc_curves = fit_fpc_curves(CHUNKS, X_ref=X_ref, out_dir=out_dir, log=log)

    pop_csv = out_dir / "population_scores.csv"
    summary_csv = out_dir / "temp_surface_summary.csv"

    pop_fields = [
        "count_temp_tau", "infer_softmax_temp", "pop_id", "score_k_mean_for_i_j_success",
        "i_success_n", "j_success_n", "both_success_n", "i_only_n", "j_only_n", "neither_n",
        "mean_z_i", "mean_z_j", "mean_z_k", "max_z_i", "max_z_j", "max_z_k",
        "pop_elapsed_sec", "temp_elapsed_sec", "status", "error",
    ]
    summary_fields = [
        "count_temp_tau", "infer_softmax_temp", "n_populations", "score_mean", "score_median", "score_std", "score_min", "score_max",
        "valid_score_frac", "both_success_n_mean", "i_success_n_mean", "j_success_n_mean",
        "elapsed_sec", "status", "error",
    ]

    all_pop_rows = []
    all_summary_rows = []

    total_combos = len(COUNT_TEMPS) * len(INFER_SOFTMAX_TEMPS)
    combo_i = 0

    for tau_i, tau in enumerate(COUNT_TEMPS):
        for infer_i, infer_temp in enumerate(INFER_SOFTMAX_TEMPS):
            combo_i += 1
            tau_label = f"tau_{int(round(float(tau) * 1000)):07d}"
            infer_label = f"infer_{int(round(float(infer_temp) * 1000)):04d}"
            combo_out_dir = out_dir / "surface" / tau_label / infer_label
            combo_out_dir.mkdir(parents=True, exist_ok=True)

            log(
                f"\n[{combo_i}/{total_combos}] tau={tau:.6g} | infer_temp={infer_temp:.6g}",
                force=True,
            )
            t0 = time.time()
            status = "complete"
            error = ""
            pop_rows = []

            try:
                G = load_grammar(fin_path)
                G = deepcopy(G)
                temper_summary = apply_visit_count_temperature(G, tau=float(tau))
                G = prepare_grammar_for_inference(G, infer_temp=float(infer_temp))
                write_json(combo_out_dir / "temperature_transform.json", {
                    "count_temperature_transform": temper_summary,
                    "infer_softmax_temp": float(infer_temp),
                })

                for pop_id in range(N_POPULATIONS):
                    row = evaluate_one_generated_population(
                        G=G,
                        fpc_curves=fpc_curves,
                        combo_out_dir=combo_out_dir,
                        pop_id=pop_id,
                        tau=float(tau),
                        infer_temp=float(infer_temp),
                        log=log,
                    )
                    row["temp_elapsed_sec"] = time.time() - t0
                    pop_rows.append(row)
                    all_pop_rows.append(row)
                    append_csv(pop_csv, row, pop_fields)

            except Exception:
                status = "crashed"
                error = traceback.format_exc()
                log(error, force=True)

            elapsed = time.time() - t0
            scores = np.asarray([r.get("score_k_mean_for_i_j_success", np.nan) for r in pop_rows], dtype=np.float64)
            score_finite = scores[np.isfinite(scores)]

            summary = {
                "count_temp_tau": float(tau),
                "infer_softmax_temp": float(infer_temp),
                "n_populations": len(pop_rows),
                "score_mean": float(np.mean(score_finite)) if score_finite.size else np.nan,
                "score_median": float(np.median(score_finite)) if score_finite.size else np.nan,
                "score_std": float(np.std(score_finite)) if score_finite.size else np.nan,
                "score_min": float(np.min(score_finite)) if score_finite.size else np.nan,
                "score_max": float(np.max(score_finite)) if score_finite.size else np.nan,
                "valid_score_frac": float(score_finite.size / max(len(pop_rows), 1)),
                "both_success_n_mean": finite_mean([r.get("both_success_n", np.nan) for r in pop_rows]),
                "i_success_n_mean": finite_mean([r.get("i_success_n", np.nan) for r in pop_rows]),
                "j_success_n_mean": finite_mean([r.get("j_success_n", np.nan) for r in pop_rows]),
                "elapsed_sec": elapsed,
                "status": status,
                "error": error,
            }

            write_json(combo_out_dir / "combo_summary.json", summary)
            append_csv(summary_csv, summary, summary_fields)
            all_summary_rows.append(summary)

            try:
                plot_surface_summary(
                    all_pop_rows,
                    out_dir / "temp_surface_summary_progress.png",
                    "Count-temp x infer-softmax-temp inference sweep",
                )
                plot_tau_lines_by_infer_temp(
                    all_pop_rows,
                    out_dir / "temp_surface_lines_progress.png",
                    "Count-temp x infer-softmax-temp inference sweep",
                )
            except Exception:
                log("plot failed:\n" + traceback.format_exc(), force=True)

            log(
                f"[done] tau={tau:.6g} | infer_temp={infer_temp:.6g} | "
                f"elapsed={elapsed:.2f}s | score_mean={summary['score_mean']}",
                force=True,
            )

    plot_surface_summary(all_pop_rows, out_dir / "temp_surface_summary.png", "Count-temp x infer-softmax-temp inference sweep")
    plot_tau_lines_by_infer_temp(all_pop_rows, out_dir / "temp_surface_lines.png", "Count-temp x infer-softmax-temp inference sweep")
    write_json(out_dir / "done.json", {"status": "complete", "out_dir": out_dir})

    log("\ncomplete", force=True)
    log(f"output: {out_dir}", force=True)
    log(f"population scores: {pop_csv}", force=True)
    log(f"summary: {summary_csv}", force=True)
    log(f"surface plots: {out_dir / 'temp_surface_summary.png'}", force=True)


if __name__ == "__main__":
    main()
