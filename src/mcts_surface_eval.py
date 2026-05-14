"""
mcts_surface_eval.py

Evaluate saved MCTS grammars across a softmax-temperature x max-depth surface.

Workflow
--------
1. Discover trained grammars at mcts_runs/<run_name>/fin.g.
2. Parse run names like surf_t0030_d03 into:
      n = softmax_temp ~= 0.030
      m = max_depth = 3
3. Fit one FPC curve per evaluation chunk: i, j, k.
4. For each grammar, force inference behavior:
      G.mode("infer")
      softmax_temp = 1.0
      no expansion / no new unexplored edges when possible
5. Generate N populations sequentially from the frozen grammar.
6. Evaluate every population on chunks i, j, k.
7. Score each population by mean z_k of genes that succeeded in both i and j.
8. Save per-population arrays, per-grammar summaries, and global CSVs.

Run
---
python mcts_surface_eval.py

Notes
-----
This script assumes your project modules are importable from the current working directory:
    initialization.py, evaluation.py, transform_ops.py, mcts_util.py
"""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from copy import deepcopy
import csv
import json
import re
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
# user settings
# ---------------------------------------------------------------------



TRAIN_RUN_ROOT = Path("mcts_runs")
OUT_ROOT = Path("mcts_surface_eval_runs")
OUT_NAME = time.strftime("surface_eval_%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / OUT_NAME

# chunks i, j, k are base_chunk + 0, +1, +2
BASE_CHUNK_NUM = 0
CHUNKS = {
    "i": BASE_CHUNK_NUM + 0,
    "j": BASE_CHUNK_NUM + 1,
    "k": BASE_CHUNK_NUM + 2,
}

N_POPULATIONS = 100
FPC_N_SIMS = 2500
SUCCESS_Z = 2.0

# If True, fit exactly one FPC curve per chunk and reuse it for all grammars.
# This is usually correct because the FPC curve is target/chunk geometry, not grammar quality.
FIT_FPC_ONCE_PER_CHUNK = True

# If None, discover all folders matching */fin.g.
# Or set to a list of run names if you want a subset.
RUN_NAME_FILTER: list[str] | None = None

# Optional hard limits for debugging.
MAX_GRAMMARS: int | None = None
MAX_POPS_PER_GRAMMAR: int | None = None


# ---------------------------------------------------------------------
# small IO helpers
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


# ---------------------------------------------------------------------
# grammar discovery / preparation
# ---------------------------------------------------------------------

def parse_surface_run_name(run_name: str) -> tuple[float | None, int | None]:
    """
    Parse names like:
        surf_t0030_d03
        surf_t1000_d09

    Returns:
        softmax_temp, max_depth
    """
    m = re.search(r"t(\d{3,5}).*?d(\d+)", run_name)
    if not m:
        return None, None

    temp = int(m.group(1)) / 1000.0
    depth = int(m.group(2))
    return float(temp), int(depth)


def discover_grammar_runs(train_root: Path, run_name_filter: list[str] | None = None) -> list[dict]:
    records = []

    for fin_path in sorted(train_root.glob("*/fin.g")):
        run_dir = fin_path.parent
        run_name = run_dir.name

        if run_name_filter is not None and run_name not in set(run_name_filter):
            continue

        temp, depth = parse_surface_run_name(run_name)

        if temp is None or depth is None:
            # Skip unrelated run folders.
            continue

        records.append({
            "run_name": run_name,
            "run_dir": run_dir,
            "fin_path": fin_path,
            "softmax_temp_train": temp,
            "max_depth": depth,
            "max_path_depth": depth + 1,
        })

    records.sort(key=lambda r: (r["max_depth"], r["softmax_temp_train"]))
    return records


def load_grammar(path: str | Path):
    with open(path, "rb") as f:
        return dill.load(f)


def set_grammar_softmax_temp(G, temp: float) -> None:
    """Set the inference softmax temp on common grammar attributes/dicts."""
    temp = float(temp)

    if hasattr(G, "_softmax_temp"):
        G._softmax_temp = temp
    else:
        setattr(G, "_softmax_temp", temp)

    for attr in ("_spec_gram_args", "spec_gram_args", "_MCTS_SPEC_GRAM_ARGS", "_mcts_spec_gram_args"):
        d = getattr(G, attr, None)
        if isinstance(d, dict):
            d["softmax_temp"] = temp


def prepare_grammar_for_inference(G, *, infer_temp: float = 1.0):
    """
    Put grammar into inference behavior:
      - use G.mode("infer") when available
      - no U interpretation according to user's Grammar.mode("infer") behavior
      - no expansion / no unexplored edges where common controls exist
      - inference softmax temperature = infer_temp
    """
    # user-provided API
    mode_fn = getattr(G, "mode", None)
    if callable(mode_fn):
        try:
            mode_fn("infer")
        except TypeError:
            # fallback if mode is property-like or has a different signature
            setattr(G, "_mode", "infer")
    else:
        setattr(G, "_mode", "infer")

    # common expansion/freezing switches
    if hasattr(G, "freeze_mcts_expansion"):
        try:
            G.freeze_mcts_expansion(True)
        except Exception:
            setattr(G, "_MCTS_FREEZE_EXPANSION", True)
    else:
        setattr(G, "_MCTS_FREEZE_EXPANSION", True)

    # hard stop on generation of new unexplored edges/states, if these are read
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


# ---------------------------------------------------------------------
# population generation / evaluation
# ---------------------------------------------------------------------

def make_initialization_kwargs(G):
    """Use your mcts_util defaults, then enforce inference-safe generation settings."""
    kwargs = _MU.default_initialization_kwargs(G)
    kwargs["grmr_prior"] = G
    kwargs["grmr_type"] = "MCTS"
    kwargs["grmr_p_mttn"] = 0.0
    kwargs["grmr_p_csvr"] = 0.0
    kwargs["verbose"] = 0
    return kwargs


def make_solver_kwargs():
    """Central place to modify the evaluation target later."""
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


def fit_fpc_curve_for_chunk(X_ref, chunk_name: str, chunk_num: int, solver_kwargs: dict, out_dir: Path):
    print(f"fitting FPC curve for chunk {chunk_name}={chunk_num}")
    t0 = time.time()

    instantiate_for_chunk(X_ref, chunk_num)

    curve, params, perm_mu = _E.fit_FPC_part_prop(
        X=X_ref,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        n_sims=FPC_N_SIMS,
    )

    elapsed = time.time() - t0

    # Save params + a plotted/evaluated curve grid for later reference.
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
        "elapsed_sec": elapsed,
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

    return {
        "curve": curve,
        "params": params,
        "perm_mu": float(perm_mu),
        "elapsed_sec": elapsed,
    }


def fit_fpc_curves(chunks: dict[str, int], X_ref, out_dir: Path) -> dict[str, dict]:
    solver_kwargs = make_solver_kwargs()
    fpc = {}
    for chunk_name, chunk_num in chunks.items():
        fpc[chunk_name] = fit_fpc_curve_for_chunk(
            X_ref=X_ref,
            chunk_name=chunk_name,
            chunk_num=chunk_num,
            solver_kwargs=solver_kwargs,
            out_dir=out_dir,
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


def evaluate_one_generated_population(G, fpc_curves: dict, grammar_out_dir: Path, pop_id: int):
    pop_dir = grammar_out_dir / f"pop_{pop_id:03d}"
    pop_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, _ = generate_population_from_grammar(G)
    gen_elapsed = time.time() - t0

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

    # also useful diagnostics
    i_success_n = int(np.sum(masks["i_success"]))
    j_success_n = int(np.sum(masks["j_success"]))
    both_success_n = int(np.sum(masks["both"]))
    i_only_n = int(np.sum(masks["i_only"]))
    j_only_n = int(np.sum(masks["j_only"]))
    neither_n = int(np.sum(masks["neither"]))

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

    # Save enough per-chunk FPC scatter data for later overlays.
    for chunk_name in ("i", "j", "k"):
        d = evals[chunk_name]["details"]
        np.savez_compressed(
            pop_dir / f"fpc_scatter_{chunk_name}.npz",
            prop_g=np.asarray(d.get("prop_g", []), dtype=np.float64),
            observed_scores_g=np.asarray(d.get("observed_scores_g", []), dtype=np.float64),
            fpc_std_g=np.asarray(d.get("fpc_std_g", []), dtype=np.float64),
            perm_mu=float(fpc_curves[chunk_name]["perm_mu"]),
        )

    row = {
        "pop_id": int(pop_id),
        "score_k_mean_for_i_j_success": pop_score,
        "i_success_n": i_success_n,
        "j_success_n": j_success_n,
        "both_success_n": both_success_n,
        "i_only_n": i_only_n,
        "j_only_n": j_only_n,
        "neither_n": neither_n,
        "mean_z_i": finite_mean(z_i[good_idx]),
        "mean_z_j": finite_mean(z_j[good_idx]),
        "mean_z_k": finite_mean(z_k[good_idx]),
        "max_z_i": float(np.nanmax(z_i[good_idx])) if good_idx.size else np.nan,
        "max_z_j": float(np.nanmax(z_j[good_idx])) if good_idx.size else np.nan,
        "max_z_k": float(np.nanmax(z_k[good_idx])) if good_idx.size else np.nan,
        "gen_elapsed_sec": gen_elapsed,
    }

    write_json(pop_dir / "summary.json", row)
    return row


# ---------------------------------------------------------------------
# visualizations
# ---------------------------------------------------------------------

def plot_grammar_population_summary(pop_rows: list[dict], out_path: Path, title: str):
    pop_ids = np.asarray([r["pop_id"] for r in pop_rows], dtype=int)
    scores = np.asarray([r["score_k_mean_for_i_j_success"] for r in pop_rows], dtype=np.float64)
    both_n = np.asarray([r["both_success_n"] for r in pop_rows], dtype=np.float64)
    i_n = np.asarray([r["i_success_n"] for r in pop_rows], dtype=np.float64)
    j_n = np.asarray([r["j_success_n"] for r in pop_rows], dtype=np.float64)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axs = axs.ravel()

    axs[0].plot(pop_ids, scores, marker="o")
    axs[0].axhline(0, alpha=0.25)
    axs[0].set_title("population score: mean z_k of i+j successes")
    axs[0].set_xlabel("population")
    axs[0].set_ylabel("score")
    axs[0].grid(alpha=0.25)

    clean_scores = scores[np.isfinite(scores)]
    if clean_scores.size:
        axs[1].hist(clean_scores, bins=min(12, max(3, clean_scores.size // 2)))
        axs[1].axvline(np.mean(clean_scores), linestyle="--", label="mean")
        axs[1].legend()
    axs[1].set_title("score distribution")
    axs[1].set_xlabel("score")
    axs[1].grid(alpha=0.25)

    axs[2].plot(pop_ids, i_n, marker="o", label="i success")
    axs[2].plot(pop_ids, j_n, marker="o", label="j success")
    axs[2].plot(pop_ids, both_n, marker="o", label="i+j success")
    axs[2].set_title("success counts by population")
    axs[2].set_xlabel("population")
    axs[2].set_ylabel("count")
    axs[2].grid(alpha=0.25)
    axs[2].legend()

    mean_i = np.asarray([r["mean_z_i"] for r in pop_rows], dtype=np.float64)
    mean_j = np.asarray([r["mean_z_j"] for r in pop_rows], dtype=np.float64)
    mean_k = np.asarray([r["mean_z_k"] for r in pop_rows], dtype=np.float64)

    axs[3].plot(pop_ids, mean_i, marker="o", label="chunk i")
    axs[3].plot(pop_ids, mean_j, marker="o", label="chunk j")
    axs[3].plot(pop_ids, mean_k, marker="o", label="chunk k")
    axs[3].set_title("mean z-score by chunk")
    axs[3].set_xlabel("population")
    axs[3].set_ylabel("mean z")
    axs[3].grid(alpha=0.25)
    axs[3].legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pivot_surface(rows: list[dict], value_key: str):
    depths = sorted({int(r["max_depth"]) for r in rows})
    temps = sorted({float(r["softmax_temp_train"]) for r in rows})
    Z = np.full((len(depths), len(temps)), np.nan, dtype=np.float64)

    for r in rows:
        i = depths.index(int(r["max_depth"]))
        j = temps.index(float(r["softmax_temp_train"]))
        Z[i, j] = float(r.get(value_key, np.nan))

    return depths, temps, Z


def plot_surface_heatmap(summary_rows: list[dict], value_key: str, out_path: Path, title: str):
    depths, temps, Z = pivot_surface(summary_rows, value_key)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", origin="lower")

    ax.set_title(title)
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

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_surface_plots(summary_rows: list[dict], out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, title in [
        ("score_mean", "Surface: mean population score"),
        ("score_median", "Surface: median population score"),
        ("score_std", "Surface: std of population score"),
        ("both_success_n_mean", "Surface: mean i+j success count"),
        ("valid_score_frac", "Surface: fraction of populations with nonempty i+j success"),
    ]:
        plot_surface_heatmap(summary_rows, key, plots_dir / f"surface_{key}.png", title)


# ---------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "grammars").mkdir(parents=True, exist_ok=True)

    run_records = discover_grammar_runs(TRAIN_RUN_ROOT, RUN_NAME_FILTER)
    if MAX_GRAMMARS is not None:
        run_records = run_records[:int(MAX_GRAMMARS)]

    if not run_records:
        raise RuntimeError(f"No fin.g files found under {TRAIN_RUN_ROOT} matching surface run names.")

    write_json(OUT_DIR / "config.json", {
        "train_run_root": TRAIN_RUN_ROOT,
        "out_dir": OUT_DIR,
        "chunks": CHUNKS,
        "n_populations": N_POPULATIONS,
        "fpc_n_sims": FPC_N_SIMS,
        "success_z": SUCCESS_Z,
        "fit_fpc_once_per_chunk": FIT_FPC_ONCE_PER_CHUNK,
        "run_count": len(run_records),
    })

    print(f"found {len(run_records)} grammars")
    print(f"output: {OUT_DIR}")

    # Reference X for chunk-level FPC curves.
    first_G = prepare_grammar_for_inference(load_grammar(run_records[0]["fin_path"]), infer_temp=1.0)
    X_ref, _ = generate_population_from_grammar(first_G)

    fpc_curves = fit_fpc_curves(CHUNKS, X_ref=X_ref, out_dir=OUT_DIR)

    pop_fieldnames = [
        "run_name", "softmax_temp_train", "max_depth", "max_path_depth", "pop_id",
        "score_k_mean_for_i_j_success", "i_success_n", "j_success_n", "both_success_n",
        "i_only_n", "j_only_n", "neither_n", "mean_z_i", "mean_z_j", "mean_z_k",
        "max_z_i", "max_z_j", "max_z_k", "gen_elapsed_sec", "grammar_elapsed_sec",
    ]

    grammar_fieldnames = [
        "run_name", "softmax_temp_train", "max_depth", "max_path_depth", "n_populations",
        "score_mean", "score_median", "score_std", "score_min", "score_max", "valid_score_frac",
        "both_success_n_mean", "i_success_n_mean", "j_success_n_mean", "elapsed_sec", "status", "error",
    ]

    all_summary_rows = []

    for rec_i, rec in enumerate(run_records):
        run_name = rec["run_name"]
        grammar_dir = OUT_DIR / "grammars" / run_name
        grammar_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{rec_i + 1}/{len(run_records)}] evaluating {run_name} | "
            f"temp={rec['softmax_temp_train']:.3f} depth={rec['max_depth']}"
        )

        t0 = time.time()
        status = "complete"
        err_text = ""
        pop_rows = []

        try:
            G = load_grammar(rec["fin_path"])
            G = prepare_grammar_for_inference(G, infer_temp=1.0)

            n_pops = N_POPULATIONS if MAX_POPS_PER_GRAMMAR is None else int(MAX_POPS_PER_GRAMMAR)

            for pop_id in range(n_pops):
                print(f"  population {pop_id + 1}/{n_pops}")
                row = evaluate_one_generated_population(
                    G=G,
                    fpc_curves=fpc_curves,
                    grammar_out_dir=grammar_dir,
                    pop_id=pop_id,
                )

                row.update({
                    "run_name": run_name,
                    "softmax_temp_train": rec["softmax_temp_train"],
                    "max_depth": rec["max_depth"],
                    "max_path_depth": rec["max_path_depth"],
                    "grammar_elapsed_sec": time.time() - t0,
                })

                pop_rows.append(row)
                append_csv(OUT_DIR / "population_scores.csv", row, pop_fieldnames)

            plot_grammar_population_summary(
                pop_rows,
                grammar_dir / "population_summary.png",
                title=f"{run_name} | temp={rec['softmax_temp_train']:.3f} depth={rec['max_depth']}",
            )

        except Exception:
            status = "crashed"
            err_text = traceback.format_exc()
            print(err_text)

        elapsed = time.time() - t0

        scores = np.asarray([r.get("score_k_mean_for_i_j_success", np.nan) for r in pop_rows], dtype=np.float64)
        score_finite = scores[np.isfinite(scores)]

        summary = {
            "run_name": run_name,
            "softmax_temp_train": rec["softmax_temp_train"],
            "max_depth": rec["max_depth"],
            "max_path_depth": rec["max_path_depth"],
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
            "error": err_text,
        }

        write_json(grammar_dir / "grammar_summary.json", summary)
        append_csv(OUT_DIR / "grammar_summary.csv", summary, grammar_fieldnames)
        all_summary_rows.append(summary)

        # update surface plots after every grammar so you can inspect partial progress
        try:
            make_surface_plots(all_summary_rows, OUT_DIR)
        except Exception:
            print("surface plotting failed so far:")
            print(traceback.format_exc())

    make_surface_plots(all_summary_rows, OUT_DIR)
    write_json(OUT_DIR / "done.json", {"status": "complete", "out_dir": OUT_DIR, "n_grammars": len(all_summary_rows)})
    print("\ncomplete")
    print("output:", OUT_DIR)
    print("grammar summary:", OUT_DIR / "grammar_summary.csv")
    print("population scores:", OUT_DIR / "population_scores.csv")


if __name__ == "__main__":
    main()
