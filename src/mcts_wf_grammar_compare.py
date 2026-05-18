"""
mcts_wf_grammar_compare.py

Compare two walk-forward grammar families plus a null grammar.

Expected inputs
---------------
mcts_runs/
    wf_gamma000000/grammars/grammar_window_000.pkl.gz
    wf_gamma095098/grammars/grammar_window_000.pkl.gz
    ...

For window w:
    chunk i = w
    chunk j = w + 1
    chunk k = w + 2

For each grammar family and each window:
    1. load grammar_window_XXX.pkl.gz
    2. force inference mode: G.mode("infer"), temp=1.0, expansion off
    3. generate N populations sequentially
    4. evaluate every generated population on chunks i, j, k with FPC z-scores
    5. mark i success, j success, and i+j success using z > SUCCESS_Z
    6. score population as mean z_k among i+j success genes

Then compare the score distributions using a 1D energy-distance permutation test:
    gamma000000 vs gamma095098
    gamma000000 vs null
    gamma095098 vs null

Run
---
python mcts_wf_grammar_compare.py

Useful options
--------------
python mcts_wf_grammar_compare.py --n-populations 30 --max-windows 3 --fpc-n-sims 1000
python mcts_wf_grammar_compare.py --energy-perms 2000 --success-z 2.0
"""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from copy import deepcopy
import argparse
import csv
import gzip
import json
import math
import pickle
import re
import time
import traceback
from typing import Any

import dill
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
# defaults
# ---------------------------------------------------------------------

DEFAULT_ROOT = Path("mcts_runs")
DEFAULT_OUT_ROOT = Path("wf_grammar_compare")

GROUPS = {
    "gamma000000": DEFAULT_ROOT / "wf_gamma000000" / "grammars",
    "gamma095098": DEFAULT_ROOT / "wf_gamma095098" / "grammars",
}

N_POPULATIONS = 30
FPC_N_SIMS = 2500
SUCCESS_Z = 2.0
INFER_SOFTMAX_TEMP = 1.0
ENERGY_PERMS = 1000
RNG_SEED = 123
VERBOSE = True


# ---------------------------------------------------------------------
# IO helpers
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


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def finite_mean(x) -> float:
    arr = finite_array(x)
    return float(np.mean(arr)) if arr.size else np.nan


def finite_median(x) -> float:
    arr = finite_array(x)
    return float(np.median(arr)) if arr.size else np.nan


def finite_std(x) -> float:
    arr = finite_array(x)
    return float(np.std(arr)) if arr.size else np.nan


def finite_max(x) -> float:
    arr = finite_array(x)
    return float(np.max(arr)) if arr.size else np.nan


def safe_nanmean_selected(values, idx) -> float:
    if idx is None or len(idx) == 0:
        return np.nan
    values = np.asarray(values, dtype=np.float64)
    idx = np.asarray(idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < values.size)]
    if idx.size == 0:
        return np.nan
    return finite_mean(values[idx])


class Logger:
    def __init__(self, out_dir: Path, verbose: bool = True):
        self.out_dir = Path(out_dir)
        self.verbose = bool(verbose)
        self.path = self.out_dir / "wf_grammar_compare.log"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: Any = "", *, force: bool = False):
        text = str(msg)
        if self.verbose or force:
            print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")


class CaptureMatplotlibShow:
    """Save figures from functions that call plt.show(), avoiding GUI blocking."""
    def __init__(self, save_dir: Path, prefix: str, dpi: int = 140, save: bool = True):
        self.save_dir = Path(save_dir)
        self.prefix = str(prefix)
        self.dpi = int(dpi)
        self.save = bool(save)
        self.old_show = None
        self.saved = []

    def _save_open(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for j, num in enumerate(list(plt.get_fignums())):
            fig = plt.figure(num)
            if self.save:
                path = self.save_dir / f"{self.prefix}_{j:02d}.png"
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
                self.saved.append(str(path))
            plt.close(fig)

    def __enter__(self):
        self.old_show = plt.show
        def quiet_show(*args, **kwargs):
            self._save_open()
        plt.show = quiet_show
        return self

    def __exit__(self, exc_type, exc, tb):
        plt.show = self.old_show
        self._save_open()
        return False


# ---------------------------------------------------------------------
# grammar discovery / preparation
# ---------------------------------------------------------------------

def window_id_from_path(path: Path) -> int | None:
    m = re.search(r"grammar_window_(\d+)\.pkl\.gz$", path.name)
    if not m:
        return None
    return int(m.group(1))


def discover_window_grammars(group_dirs: dict[str, Path]) -> dict[int, dict[str, Path]]:
    by_window: dict[int, dict[str, Path]] = {}
    for group_name, gdir in group_dirs.items():
        for path in sorted(Path(gdir).glob("grammar_window_*.pkl.gz")):
            w = window_id_from_path(path)
            if w is None:
                continue
            by_window.setdefault(w, {})[group_name] = path
    return by_window


def load_pickle_gz(path: str | Path):
    path = Path(path)
    with gzip.open(path, "rb") as f:
        try:
            return dill.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f)


def load_grammar_window(path: str | Path):
    obj = load_pickle_gz(path)
    # If you saved a dict payload, try common keys.
    if isinstance(obj, dict):
        for key in ("G", "grammar", "grmr"):
            if key in obj:
                return obj[key]
    return obj


def set_grammar_softmax_temp(G, temp: float):
    temp = float(temp)
    setattr(G, "_softmax_temp", temp)
    for attr in ("_spec_gram_args", "spec_gram_args", "_MCTS_SPEC_GRAM_ARGS", "_mcts_spec_gram_args"):
        d = getattr(G, attr, None)
        if isinstance(d, dict):
            d["softmax_temp"] = temp


def prepare_grammar_for_inference(G, *, infer_temp: float = INFER_SOFTMAX_TEMP):
    """
    User-provided behavior: G.mode("infer") means no U interpretation.
    Also hard-disable expansion where supported.
    """
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


def make_null_grammar(template_G=None):
    """
    Construct an empty/null grammar for baseline generation.

    If your project's Null grammar constructor differs, this is the one place
    to adjust it.
    """
    try:
        return _I.Grammar(
            type="Null",
            max_delta_lookback=int(getattr(template_G, "_max_delta_lookback", 80) or 80),
            p_mutation=0.00,
            p_crossover=0.00,
            alpha_sensor_freq=float(getattr(template_G, "_alpha_sensor_freq", 0.5) or 0.5),
            node_fitness="count_pop_dead",
            count_explore=False,
            temp=0.0,
            mode="infer",
            explore_const=np.sqrt(2),
        )
    except Exception:
        # fallback to a mostly inert copy if Null construction is unavailable
        G = deepcopy(template_G)
        try:
            G.mode("infer")
        except Exception:
            setattr(G, "_mode", "infer")
        for attr in (
            "_MCTS_NODE_MU", "_MCTS_EDGE_MU", "_MCTS_NODE_COUNT", "_MCTS_EDGE_COUNT",
            "_MCTS_CHILDREN", "_MCTS_ALPHA_DECISION_MU", "_MCTS_ALPHA_NODE_MU",
            "_MCTS_ALPHA_EDGE_MU",
        ):
            if hasattr(G, attr):
                setattr(G, attr, {})
        return G


# ---------------------------------------------------------------------
# generation / evaluation
# ---------------------------------------------------------------------

def make_initialization_kwargs(G):
    kwargs = _MU.default_initialization_kwargs(G)
    kwargs["grmr_prior"] = G
    kwargs["grmr_type"] = getattr(G, "_type", "MCTS") if str(getattr(G, "_type", "MCTS")).lower() == "null" else "MCTS"
    # If null grammar exposes type differently, prefer explicit Null where possible.
    if str(getattr(G, "_type", "")).lower() == "null":
        kwargs["grmr_type"] = "Null"
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


def fit_fpc_curve_for_chunk(X_ref, chunk_num: int, solver_kwargs: dict, out_dir: Path, log: Logger):
    chunk_num = int(chunk_num)
    fpc_dir = out_dir / "fpc_curves"
    fpc_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"[start] FPC chunk {chunk_num}", force=True)

    instantiate_for_chunk(X_ref, chunk_num)

    with CaptureMatplotlibShow(
        save_dir=fpc_dir,
        prefix=f"fpc_curve_original_chunk_{chunk_num:03d}",
        dpi=140,
        save=True,
    ):
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

    elapsed = time.time() - t0
    log(f"[done]  FPC chunk {chunk_num} | elapsed={elapsed:.2f}s", force=True)

    return {
        "curve": curve,
        "params": params,
        "perm_mu": float(perm_mu),
        "p_grid": p_grid,
        "v_grid": v_grid,
        "s_grid": s_grid,
        "elapsed_sec": elapsed,
    }


def get_fpc_cached(cache: dict[int, dict], X_ref, chunk_num: int, solver_kwargs: dict, out_dir: Path, log: Logger):
    chunk_num = int(chunk_num)
    if chunk_num not in cache:
        cache[chunk_num] = fit_fpc_curve_for_chunk(
            X_ref=X_ref,
            chunk_num=chunk_num,
            solver_kwargs=solver_kwargs,
            out_dir=out_dir,
            log=log,
        )
    return cache[chunk_num]


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


def compute_success_sets(z_i, z_j, good_idx, *, success_z: float):
    good_idx = np.asarray(good_idx, dtype=int)
    z_i = np.asarray(z_i, dtype=np.float64)
    z_j = np.asarray(z_j, dtype=np.float64)

    valid = (good_idx >= 0) & (good_idx < z_i.size) & (good_idx < z_j.size)
    idx = good_idx[valid]

    i_success = idx[np.isfinite(z_i[idx]) & (z_i[idx] > success_z)]
    j_success = idx[np.isfinite(z_j[idx]) & (z_j[idx] > success_z)]
    ij_success = idx[
        np.isfinite(z_i[idx])
        & np.isfinite(z_j[idx])
        & (z_i[idx] > success_z)
        & (z_j[idx] > success_z)
    ]
    return idx, i_success, j_success, ij_success


# ---------------------------------------------------------------------
# energy distance tests
# ---------------------------------------------------------------------

def energy_distance_1d(x, y) -> float:
    x = finite_array(x)
    y = finite_array(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    xy = np.abs(x[:, None] - y[None, :]).mean()
    xx = np.abs(x[:, None] - x[None, :]).mean()
    yy = np.abs(y[:, None] - y[None, :]).mean()
    return float(2.0 * xy - xx - yy)


def energy_permutation_test_1d(x, y, *, n_perm: int = ENERGY_PERMS, rng=None) -> dict:
    rng = np.random.default_rng(rng)
    x = finite_array(x)
    y = finite_array(y)
    if x.size < 2 or y.size < 2:
        return {
            "energy_distance": np.nan,
            "p_value": np.nan,
            "n_x": int(x.size),
            "n_y": int(y.size),
        }

    obs = energy_distance_1d(x, y)
    pooled = np.concatenate([x, y])
    n_x = x.size

    ge = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(pooled)
        xp = perm[:n_x]
        yp = perm[n_x:]
        stat = energy_distance_1d(xp, yp)
        if np.isfinite(stat) and stat >= obs:
            ge += 1

    p = (ge + 1.0) / (int(n_perm) + 1.0)
    return {
        "energy_distance": float(obs),
        "p_value": float(p),
        "n_x": int(x.size),
        "n_y": int(y.size),
    }


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------

def plot_window_distributions(window: int, df_rows: list[dict], out_dir: Path):
    rows = [r for r in df_rows if int(r["window"]) == int(window)]
    groups = ["gamma000000", "gamma095098", "null"]
    data = []
    labels = []
    for g in groups:
        vals = finite_array([r.get("score_k_mean_ij", np.nan) for r in rows if r.get("group") == g])
        if vals.size:
            data.append(vals)
            labels.append(g)

    if not data:
        return None

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    axs[0].boxplot(data, labels=labels, showmeans=True)
    for i, vals in enumerate(data, start=1):
        x = np.full(vals.shape, i, dtype=float) + np.linspace(-0.07, 0.07, vals.size)
        axs[0].scatter(x, vals, s=25, alpha=0.65)
    axs[0].axhline(0, alpha=0.3)
    axs[0].set_title(f"window {window:03d} | k score for i+j success")
    axs[0].set_ylabel("mean z_k among i+j successes")
    axs[0].grid(alpha=0.25)

    for g in groups:
        sub = [r for r in rows if r.get("group") == g]
        if not sub:
            continue
        xs = np.arange(len(sub))
        axs[1].plot(xs, [float(r.get("i_success_n", np.nan)) for r in sub], alpha=0.55, label=f"{g} i")
        axs[1].plot(xs, [float(r.get("ij_success_n", np.nan)) for r in sub], linewidth=2, label=f"{g} i+j")
    axs[1].set_title("success counts by population")
    axs[1].set_xlabel("population")
    axs[1].set_ylabel("count")
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=8)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"window_{window:03d}_distributions.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_overall_summary(summary_rows: list[dict], out_dir: Path):
    groups = ["gamma000000", "gamma095098", "null"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    for g in groups:
        sub = sorted([r for r in summary_rows if r.get("group") == g], key=lambda r: int(r["window"]))
        if not sub:
            continue
        w = [int(r["window"]) for r in sub]
        axs[0].plot(w, [float(r.get("score_mean", np.nan)) for r in sub], marker="o", label=g)
        axs[1].plot(w, [float(r.get("ij_success_mean", np.nan)) for r in sub], marker="o", label=g)

    axs[0].set_title("mean k score by window")
    axs[0].set_xlabel("window")
    axs[0].set_ylabel("mean z_k among i+j successes")
    axs[0].grid(alpha=0.25)
    axs[0].legend()

    axs[1].set_title("mean i+j success count by window")
    axs[1].set_xlabel("window")
    axs[1].set_ylabel("mean count")
    axs[1].grid(alpha=0.25)
    axs[1].legend()

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "overall_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------

def evaluate_one_group_window(
    *,
    group_name: str,
    G_base,
    window: int,
    chunks: dict[str, int],
    fpc_cache: dict[int, dict],
    X_ref,
    out_dir: Path,
    n_populations: int,
    success_z: float,
    rng,
    log: Logger,
):
    pop_rows = []
    pop_fieldnames = POP_FIELDNAMES

    group_dir = out_dir / "windows" / f"window_{window:03d}" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    for pop_i in range(int(n_populations)):
        t0 = time.time()
        log(f"  [{group_name} w={window:03d}] population {pop_i + 1}/{n_populations}")

        try:
            G = prepare_grammar_for_inference(deepcopy(G_base), infer_temp=INFER_SOFTMAX_TEMP)
            X, G_out = generate_population_from_grammar(G)
            good_idx = np.asarray(X._G_idx, dtype=int)

            evals = {}
            for cname, cnum in chunks.items():
                fpc = get_fpc_cached(
                    cache=fpc_cache,
                    X_ref=X_ref,
                    chunk_num=cnum,
                    solver_kwargs=make_solver_kwargs(),
                    out_dir=out_dir,
                    log=log,
                )
                evals[cname] = evaluate_population_on_chunk(
                    X=X,
                    chunk_num=cnum,
                    fpc_info=fpc,
                    good_idx=good_idx,
                )

            z_i = evals["i"]["zscores"]
            z_j = evals["j"]["zscores"]
            z_k = evals["k"]["zscores"]

            good_idx_eval, i_success, j_success, ij_success = compute_success_sets(
                z_i=z_i,
                z_j=z_j,
                good_idx=good_idx,
                success_z=success_z,
            )

            score_k_mean_ij = safe_nanmean_selected(z_k, ij_success)
            valid_score = bool(np.isfinite(score_k_mean_ij))

            pop_dir = group_dir / f"pop_{pop_i:03d}"
            pop_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                pop_dir / "eval_arrays.npz",
                good_idx=good_idx_eval,
                z_i=z_i,
                z_j=z_j,
                z_k=z_k,
                i_success=i_success,
                j_success=j_success,
                ij_success=ij_success,
                pvals_i=evals["i"]["pvals"],
                pvals_j=evals["j"]["pvals"],
                pvals_k=evals["k"]["pvals"],
                prop_i=evals["i"]["details"].get("prop_full", []),
                prop_j=evals["j"]["details"].get("prop_full", []),
                prop_k=evals["k"]["details"].get("prop_full", []),
                obs_i=evals["i"]["details"].get("observed_scores_full", []),
                obs_j=evals["j"]["details"].get("observed_scores_full", []),
                obs_k=evals["k"]["details"].get("observed_scores_full", []),
            )

            row = {
                "window": int(window),
                "group": group_name,
                "pop_i": int(pop_i),
                "chunk_i": int(chunks["i"]),
                "chunk_j": int(chunks["j"]),
                "chunk_k": int(chunks["k"]),
                "good_idx_n": int(good_idx_eval.size),
                "i_success_n": int(i_success.size),
                "j_success_n": int(j_success.size),
                "ij_success_n": int(ij_success.size),
                "score_k_mean_ij": score_k_mean_ij,
                "valid_score": valid_score,
                "mean_z_i_all": safe_nanmean_selected(z_i, good_idx_eval),
                "mean_z_j_all": safe_nanmean_selected(z_j, good_idx_eval),
                "mean_z_k_all": safe_nanmean_selected(z_k, good_idx_eval),
                "max_z_i_all": finite_max(np.asarray(z_i)[good_idx_eval]) if good_idx_eval.size else np.nan,
                "max_z_j_all": finite_max(np.asarray(z_j)[good_idx_eval]) if good_idx_eval.size else np.nan,
                "max_z_k_all": finite_max(np.asarray(z_k)[good_idx_eval]) if good_idx_eval.size else np.nan,
                "elapsed_sec": time.time() - t0,
                "status": "complete",
                "error": "",
                "arrays_path": str(pop_dir / "eval_arrays.npz"),
            }

        except Exception:
            err = traceback.format_exc()
            log(f"[crash] group={group_name} window={window:03d} pop={pop_i:03d}\n{err}", force=True)
            row = {
                "window": int(window),
                "group": group_name,
                "pop_i": int(pop_i),
                "chunk_i": int(chunks["i"]),
                "chunk_j": int(chunks["j"]),
                "chunk_k": int(chunks["k"]),
                "good_idx_n": 0,
                "i_success_n": 0,
                "j_success_n": 0,
                "ij_success_n": 0,
                "score_k_mean_ij": np.nan,
                "valid_score": False,
                "mean_z_i_all": np.nan,
                "mean_z_j_all": np.nan,
                "mean_z_k_all": np.nan,
                "max_z_i_all": np.nan,
                "max_z_j_all": np.nan,
                "max_z_k_all": np.nan,
                "elapsed_sec": time.time() - t0,
                "status": "crashed",
                "error": err,
                "arrays_path": "",
            }

        append_csv(out_dir / "population_scores.csv", row, pop_fieldnames)
        pop_rows.append(row)

    return pop_rows


POP_FIELDNAMES = [
    "window", "group", "pop_i", "chunk_i", "chunk_j", "chunk_k",
    "good_idx_n", "i_success_n", "j_success_n", "ij_success_n",
    "score_k_mean_ij", "valid_score",
    "mean_z_i_all", "mean_z_j_all", "mean_z_k_all",
    "max_z_i_all", "max_z_j_all", "max_z_k_all",
    "elapsed_sec", "status", "error", "arrays_path",
]

SUMMARY_FIELDNAMES = [
    "window", "group", "n_populations", "n_valid_scores", "valid_score_frac",
    "score_mean", "score_median", "score_std",
    "i_success_mean", "j_success_mean", "ij_success_mean",
    "mean_z_k_all_mean", "status_complete_n", "status_crashed_n",
]

TEST_FIELDNAMES = [
    "window", "pair", "group_x", "group_y", "n_x", "n_y",
    "mean_x", "mean_y", "median_x", "median_y", "better_mean_group",
    "energy_distance", "p_value", "n_permutations",
]


def summarize_group_window(window: int, group_name: str, rows: list[dict]) -> dict:
    scores = finite_array([r.get("score_k_mean_ij", np.nan) for r in rows])
    return {
        "window": int(window),
        "group": group_name,
        "n_populations": int(len(rows)),
        "n_valid_scores": int(scores.size),
        "valid_score_frac": float(scores.size / len(rows)) if rows else np.nan,
        "score_mean": finite_mean(scores),
        "score_median": finite_median(scores),
        "score_std": finite_std(scores),
        "i_success_mean": finite_mean([r.get("i_success_n", np.nan) for r in rows]),
        "j_success_mean": finite_mean([r.get("j_success_n", np.nan) for r in rows]),
        "ij_success_mean": finite_mean([r.get("ij_success_n", np.nan) for r in rows]),
        "mean_z_k_all_mean": finite_mean([r.get("mean_z_k_all", np.nan) for r in rows]),
        "status_complete_n": int(sum(1 for r in rows if r.get("status") == "complete")),
        "status_crashed_n": int(sum(1 for r in rows if r.get("status") == "crashed")),
    }


def compare_window_groups(window: int, rows_by_group: dict[str, list[dict]], rng, n_perm: int) -> list[dict]:
    pairs = [
        ("gamma000000", "gamma095098"),
        ("gamma000000", "null"),
        ("gamma095098", "null"),
    ]
    out = []
    for gx, gy in pairs:
        x = finite_array([r.get("score_k_mean_ij", np.nan) for r in rows_by_group.get(gx, [])])
        y = finite_array([r.get("score_k_mean_ij", np.nan) for r in rows_by_group.get(gy, [])])
        test = energy_permutation_test_1d(x, y, n_perm=n_perm, rng=rng)
        mean_x = finite_mean(x)
        mean_y = finite_mean(y)
        better = gx if np.isfinite(mean_x) and np.isfinite(mean_y) and mean_x > mean_y else gy
        out.append({
            "window": int(window),
            "pair": f"{gx}_vs_{gy}",
            "group_x": gx,
            "group_y": gy,
            "n_x": test["n_x"],
            "n_y": test["n_y"],
            "mean_x": mean_x,
            "mean_y": mean_y,
            "median_x": finite_median(x),
            "median_y": finite_median(y),
            "better_mean_group": better,
            "energy_distance": test["energy_distance"],
            "p_value": test["p_value"],
            "n_permutations": int(n_perm),
        })
    return out


def run(args):
    out_name = args.out_name or time.strftime("wf_grammar_compare_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir, verbose=not args.quiet)

    group_dirs = {
        "gamma000000": Path(args.root) / "wf_gamma000000" / "grammars",
        "gamma095098": Path(args.root) / "wf_gamma095098" / "grammars",
    }

    write_json(out_dir / "config.json", {
        "root": str(args.root),
        "group_dirs": group_dirs,
        "out_dir": out_dir,
        "n_populations": args.n_populations,
        "fpc_n_sims": args.fpc_n_sims,
        "success_z": args.success_z,
        "infer_softmax_temp": args.infer_temp,
        "energy_permutations": args.energy_perms,
        "rng_seed": args.seed,
    })

    global FPC_N_SIMS, SUCCESS_Z, INFER_SOFTMAX_TEMP, ENERGY_PERMS
    FPC_N_SIMS = int(args.fpc_n_sims)
    SUCCESS_Z = float(args.success_z)
    INFER_SOFTMAX_TEMP = float(args.infer_temp)
    ENERGY_PERMS = int(args.energy_perms)

    rng = np.random.default_rng(args.seed)

    log("discovering grammar windows", force=True)
    by_window = discover_window_grammars(group_dirs)
    windows = sorted(w for w, d in by_window.items() if all(g in d for g in ("gamma000000", "gamma095098")))

    if args.start_window is not None:
        windows = [w for w in windows if w >= int(args.start_window)]
    if args.end_window is not None:
        windows = [w for w in windows if w <= int(args.end_window)]
    if args.max_windows is not None:
        windows = windows[:int(args.max_windows)]

    if not windows:
        raise RuntimeError(f"No paired grammar windows found under {group_dirs}")

    log(f"paired windows: {windows}", force=True)
    log(f"output folder: {out_dir}", force=True)

    # Reference grammar/population for FPC fitting.
    first_G = prepare_grammar_for_inference(load_grammar_window(by_window[windows[0]]["gamma000000"]), infer_temp=args.infer_temp)
    X_ref, _ = generate_population_from_grammar(first_G)
    fpc_cache: dict[int, dict] = {}

    all_rows = []
    all_summary = []

    for window in windows:
        t_window = time.time()
        chunks = {"i": window + 0, "j": window + 1, "k": window + 2}
        log(f"\n===== window {window:03d} | chunks i/j/k={chunks['i']}/{chunks['j']}/{chunks['k']} =====", force=True)

        G_a = prepare_grammar_for_inference(load_grammar_window(by_window[window]["gamma000000"]), infer_temp=args.infer_temp)
        G_b = prepare_grammar_for_inference(load_grammar_window(by_window[window]["gamma095098"]), infer_temp=args.infer_temp)
        G_null = prepare_grammar_for_inference(make_null_grammar(template_G=G_a), infer_temp=args.infer_temp)

        grammars = {
            "gamma000000": G_a,
            "gamma095098": G_b,
            "null": G_null,
        }

        rows_by_group = {}
        for group_name, G in grammars.items():
            log(f"[start] group={group_name} window={window:03d}", force=True)
            rows = evaluate_one_group_window(
                group_name=group_name,
                G_base=G,
                window=window,
                chunks=chunks,
                fpc_cache=fpc_cache,
                X_ref=X_ref,
                out_dir=out_dir,
                n_populations=args.n_populations,
                success_z=args.success_z,
                rng=rng,
                log=log,
            )
            rows_by_group[group_name] = rows
            all_rows.extend(rows)

            summary = summarize_group_window(window, group_name, rows)
            append_csv(out_dir / "summary_by_window_group.csv", summary, SUMMARY_FIELDNAMES)
            all_summary.append(summary)
            log(f"[done]  group={group_name} window={window:03d} | valid_frac={summary['valid_score_frac']:.3f} | score_mean={summary['score_mean']}", force=True)

        for test_row in compare_window_groups(window, rows_by_group, rng=rng, n_perm=args.energy_perms):
            append_csv(out_dir / "energy_tests_by_window.csv", test_row, TEST_FIELDNAMES)
            log(
                f"[energy] window={window:03d} {test_row['pair']} | "
                f"ED={test_row['energy_distance']} | p={test_row['p_value']} | better={test_row['better_mean_group']}",
                force=True,
            )

        plot_window_distributions(window, all_rows, out_dir)
        log(f"===== done window {window:03d} | elapsed={(time.time() - t_window) / 60:.2f} min =====", force=True)

    plot_overall_summary(all_summary, out_dir)

    log("\ncomplete", force=True)
    log(f"output folder: {out_dir}", force=True)
    log("key outputs:", force=True)
    log(f"  {out_dir / 'population_scores.csv'}", force=True)
    log(f"  {out_dir / 'summary_by_window_group.csv'}", force=True)
    log(f"  {out_dir / 'energy_tests_by_window.csv'}", force=True)
    log(f"  {out_dir / 'plots'}", force=True)

    return out_dir


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--out-name", type=str, default=None)
    p.add_argument("--n-populations", type=int, default=N_POPULATIONS)
    p.add_argument("--fpc-n-sims", type=int, default=FPC_N_SIMS)
    p.add_argument("--success-z", type=float, default=SUCCESS_Z)
    p.add_argument("--infer-temp", type=float, default=INFER_SOFTMAX_TEMP)
    p.add_argument("--energy-perms", type=int, default=ENERGY_PERMS)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--start-window", type=int, default=None)
    p.add_argument("--end-window", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
