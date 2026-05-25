"""
walk_forward_mcts_alpha_fpc_cached.py

Walk-forward MCTS/alpha-grammar evolution driver with one cached FPC curve per chunk.

Window convention:
    i = chunk_start + 0   evolve/build population on this chunk
    j = chunk_start + 1   update grammar / validation-like chunk
    k = chunk_start + 2   forward-test chunk

For n_chunks=20, windows are:
    (0,1,2), (1,2,3), ..., (17,18,19)

Core behavior:
    - FPC curves are fit lazily once per chunk and reused everywhere in the run.
    - Grammar evolves on each window until smoothed local action-policy drift h < delta_L.
    - After each chunk evolution, grammar switches to infer mode and generates N populations.
    - Infer populations are evaluated through i -> j -> k success funnel.
    - Grammar evidence can decay each chunk step with grammar_gamma; default 1.0 = no decay.

This file is intended as a script-level driver. Adjust config defaults at the bottom or
instantiate WalkForwardMCTSConfig from another notebook/script.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
import time
import math
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
import ep_wrap as ep
import evaluation as _E
import initialization as _I
import transform_ops as _OPS


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

@dataclass
class WalkForwardMCTSConfig:
    # walk-forward structure
    n_chunks: int = 20
    delta_L: float = 0.035
    min_iters_per_chunk: int = 3
    max_iters_per_chunk: int = 100

    # inference / funnel evaluation
    infer_populations: int = 100
    success_z: float = 2.0
    fpc_n_sims: int = 2500
    eval_visualize: bool = False

    # grammar aging: 1.0 = no decay
    grammar_gamma: float = 1.0

    # output
    output_dir: str = "wf_mcts_runs"
    run_name: str = "mcts_alpha_wf_cached_fpc"

    # policy-drift smoothing
    drift_kappa: float = 1.0

    # data/population parameters
    data_file: str = "../data/spy5m.csv"
    pop_size: int = 100
    wf_windows: int = 500
    grmr_mdl: int = 80
    max_delta_lookback: int = 80
    chunk_size: int = 1

    # MCTS grammar parameters
    max_depth: int = 5
    max_path_depth: int = 6
    pw_c: float = 0.75
    pw_alpha: float = 0.30
    expand_prob: float = 0.10

    # alpha progressive widening parameters
    alpha_pw_c: float = 0.50
    alpha_pw_alpha: float = 0.25
    alpha_expand_prob: float = 0.067

    softmax_temp: float = 1.0
    alpha_sensor_freq: float = 0.35
    alpha_prior_weight: float = 1.0
    explore_const: float = math.sqrt(2.0)
    count_explore: bool = True
    node_fitness: str = "count_pop_dead"

    # emission settings
    delta: int = 12
    offset: int = 12


# ---------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------

def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=json_default) + "\n")


def make_run_dir(cfg: WalkForwardMCTSConfig) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.run_name}_{ts}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "grammars").mkdir(parents=True, exist_ok=True)
    (run_dir / "fpc_cache").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return run_dir


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_grammar(G, path: Path):
    save_pickle(G, path)


# ---------------------------------------------------------------------
# grammar helpers
# ---------------------------------------------------------------------

def set_grammar_mode(G, mode: str):
    """Use G.mode(mode) if available, otherwise set _mode directly."""
    if hasattr(G, "mode") and callable(getattr(G, "mode")):
        G.mode(mode)
    else:
        G._mode = mode


def build_initial_grammar(cfg: WalkForwardMCTSConfig):
    return _I.Grammar(
        type="MCTS",
        max_delta_lookback=cfg.max_delta_lookback,
        p_mutation=0.00,
        p_crossover=0.00,
        alpha_sensor_freq=cfg.alpha_sensor_freq,
        node_fitness=cfg.node_fitness,
        count_explore=cfg.count_explore,
        temp=0.0,
        mode="train",
        explore_const=cfg.explore_const,
        spec_gram_args={
            "key_mode": "path",
            "max_depth": cfg.max_depth,
            "max_path_depth": cfg.max_path_depth,

            "pw_c": cfg.pw_c,
            "pw_alpha": cfg.pw_alpha,
            "expand_prob": cfg.expand_prob,

            "alpha_pw_c": cfg.alpha_pw_c,
            "alpha_pw_alpha": cfg.alpha_pw_alpha,
            "alpha_expand_prob": cfg.alpha_expand_prob,

            "base_prior": 0.0,
            "unknown_prior": 1.0,
            "softmax_base": np.e,
            "softmax_temp": cfg.softmax_temp,
            "alpha_prior_weight": cfg.alpha_prior_weight,

            # strongly recommended for long walk-forward runs
            "trace_enabled": False,
        },
    )


def _decay_dict(d, gamma: float):
    if d is None:
        return
    for k in list(d.keys()):
        d[k] = d[k] * gamma


def _recompute_mu(G, cum_name: str, count_name: str, mu_name: str):
    if not (hasattr(G, cum_name) and hasattr(G, count_name) and hasattr(G, mu_name)):
        return

    cum = getattr(G, cum_name)
    cnt = getattr(G, count_name)
    mu = getattr(G, mu_name)

    if cum is None or cnt is None or mu is None:
        return

    for key in list(cum.keys()):
        n = cnt.get(key, 0.0)
        if n > 1e-12:
            mu[key] = cum[key] / n


def apply_grammar_decay(G, gamma: float = 1.0):
    """
    Apply one chunk-step of evidence decay. gamma=1.0 means no decay.

    Counts and cumulative scores decay together, so current means are preserved,
    but old evidence has lower future weight and lower confidence.
    """
    gamma = float(gamma)

    if gamma >= 1.0:
        return G

    gamma = max(gamma, 0.0)

    names = [
        "_MCTS_NODE_CUM", "_MCTS_NODE_COUNT", "_MCTS_NODE_EXPLORE_COUNT",
        "_MCTS_EDGE_CUM", "_MCTS_EDGE_COUNT", "_MCTS_EDGE_EXPLORE_COUNT",
        "_MCTS_ALPHA_DECISION_CUM", "_MCTS_ALPHA_DECISION_COUNT", "_MCTS_ALPHA_DECISION_EXPLORE_COUNT",
        "_MCTS_ALPHA_NODE_CUM", "_MCTS_ALPHA_NODE_COUNT", "_MCTS_ALPHA_NODE_EXPLORE_COUNT",
        "_MCTS_ALPHA_EDGE_CUM", "_MCTS_ALPHA_EDGE_COUNT", "_MCTS_ALPHA_EDGE_EXPLORE_COUNT",
    ]

    for name in names:
        if hasattr(G, name):
            _decay_dict(getattr(G, name), gamma)

    for name in ["_MCTS_EXPLOIT_T", "_MCTS_EXPLORE_T", "_MCTS_ALPHA_EXPLOIT_T", "_MCTS_ALPHA_EXPLORE_T"]:
        if hasattr(G, name) and getattr(G, name) is not None:
            setattr(G, name, getattr(G, name) * gamma)

    _recompute_mu(G, "_MCTS_NODE_CUM", "_MCTS_NODE_COUNT", "_MCTS_NODE_MU")
    _recompute_mu(G, "_MCTS_EDGE_CUM", "_MCTS_EDGE_COUNT", "_MCTS_EDGE_MU")
    _recompute_mu(G, "_MCTS_ALPHA_DECISION_CUM", "_MCTS_ALPHA_DECISION_COUNT", "_MCTS_ALPHA_DECISION_MU")
    _recompute_mu(G, "_MCTS_ALPHA_NODE_CUM", "_MCTS_ALPHA_NODE_COUNT", "_MCTS_ALPHA_NODE_MU")
    _recompute_mu(G, "_MCTS_ALPHA_EDGE_CUM", "_MCTS_ALPHA_EDGE_COUNT", "_MCTS_ALPHA_EDGE_MU")

    return G


# ---------------------------------------------------------------------
# solver/population helpers
# ---------------------------------------------------------------------

def make_solver_kwargs(cfg: WalkForwardMCTSConfig):
    d = cfg.delta

    return {
        "offset": cfg.offset,
        "t_vec": "Close",
        "t_mode": "AD",
        "emission": [
            {"ID": 5, "x": "tvec", "offset": True,
             "alpha": {"ID": 2, "x": "tvec", "offset": False, "delta1": d}},
            {"ID": "divide"},
            {"ID": 5,
             "x": {"ID": 1, "x": "tvec", "offset": False, "delta1": d},
             "alpha": {"ID": 2, "x": "tvec", "offset": False, "delta1": d}},
            {"ID": 6, "alpha": "emit"},
            {"ID": 5, "alpha": 1.0},
        ],
        "AD_cond": ("gt", 0),
    }


def make_eval_solver_kwargs(cfg: WalkForwardMCTSConfig):
    """
    Match your notebook behavior:
        solver_kwargs['emission'].pop()
        AD_cond uses logwalker start, default 0
    """
    sk = make_solver_kwargs(cfg)
    sk["emission"] = sk["emission"][:-1]
    sk["AD_cond"] = (sk["AD_cond"][0], 0)
    return sk


def make_logwalker_kwargs():
    return {
        "start": 0,
        "destination": 1,
        "steps": 24,
        "exhaust_mode": "steps",
        "exwhen": 100,
        "min_walk": 3,
        "log_base": 2,
        "nest_mode": "half",
    }


def make_init_kwargs(cfg: WalkForwardMCTSConfig, G):
    return {
        "structure": "Intraday",
        "incl_time": True,
        "data_file": cfg.data_file,
        "epoch_idx": [0],
        "hlocv_idx": [1, 2, 3, 4],
        "pop_size": cfg.pop_size,

        "grmr_prior": G,
        "grmr_type": "MCTS",
        "grmr_mdl": cfg.grmr_mdl,
        "grmr_p_mttn": 0.00,
        "grmr_p_csvr": 0.00,
        "grmr_a_sens": getattr(G, "_alpha_sensor_freq", cfg.alpha_sensor_freq),

        "chunk_size": cfg.chunk_size,
        "wf_windows": cfg.wf_windows,
        "verbose": 0,
    }


def instantiate_for_chunk(X, chunk_num: int):
    return _I.instantiate_from_ops_chunked_intraday(
        X,
        transform_ops=_OPS,
        chunk_num=chunk_num,
        chunk_B=8,
    )


# ---------------------------------------------------------------------
# FPC cache: one curve per chunk across whole script
# ---------------------------------------------------------------------

def save_fpc_obj(run_dir: Path, chunk_num: int, fpc_obj: dict):
    """
    Try to save the whole FPC object. If the curve closure is not pickleable,
    save metadata/params only.
    """
    full_path = run_dir / "fpc_cache" / f"fpc_chunk_{chunk_num:03d}.pkl"
    meta_path = run_dir / "fpc_cache" / f"fpc_chunk_{chunk_num:03d}_meta.json"

    try:
        save_pickle(fpc_obj, full_path)
        fpc_obj["saved_path"] = str(full_path)
        fpc_obj["pickle_saved"] = True
    except Exception as e:
        fpc_obj["saved_path"] = None
        fpc_obj["pickle_saved"] = False
        fpc_obj["pickle_error"] = str(e)

    meta = {
        "chunk_num": chunk_num,
        "fpc_params": fpc_obj.get("fpc_params"),
        "perm_mu": fpc_obj.get("perm_mu"),
        "n_sims": fpc_obj.get("n_sims"),
        "pickle_saved": fpc_obj.get("pickle_saved"),
        "pickle_error": fpc_obj.get("pickle_error"),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=json_default)


def get_fpc_for_chunk(
    fpc_cache: dict,
    X_template,
    chunk_num: int,
    solver_kwargs: dict,
    cfg: WalkForwardMCTSConfig,
    run_dir: Path,
    metrics_path: Path | None = None,
):
    """
    Fit or retrieve one FPC curve for a chunk.

    This is intentionally keyed only by chunk_num for this run because the run
    assumes solver_kwargs/emission/AD_cond are fixed across all evaluations.
    """
    chunk_num = int(chunk_num)

    if chunk_num in fpc_cache:
        return fpc_cache[chunk_num]

    print(f"fitting FPC curve for chunk {chunk_num} with n_sims={cfg.fpc_n_sims}... ", end="")

    instantiate_for_chunk(X_template, chunk_num)

    t0 = time.time()
    pd_fpc, fpc_params, perm_mu = _E.fit_FPC_part_prop(
        X=X_template,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        n_sims=cfg.fpc_n_sims,
    )
    elapsed = time.time() - t0

    fpc_obj = {
        "chunk_num": chunk_num,
        "pd_fpc": pd_fpc,
        "fpc_params": fpc_params,
        "perm_mu": perm_mu,
        "n_sims": cfg.fpc_n_sims,
        "fit_elapsed_sec": elapsed,
    }

    fpc_cache[chunk_num] = fpc_obj
    save_fpc_obj(run_dir, chunk_num, fpc_obj)

    if metrics_path is not None:
        append_jsonl(metrics_path, {
            "event": "fpc_fit",
            "chunk_num": chunk_num,
            "n_sims": cfg.fpc_n_sims,
            "fit_elapsed_sec": elapsed,
            "perm_mu": perm_mu,
            "fpc_params": fpc_params,
            "pickle_saved": fpc_obj.get("pickle_saved"),
        })

    print("done.")

    return fpc_obj


def evaluate_population_on_chunk_cached(
    X,
    good_idx,
    chunk_num: int,
    solver_kwargs: dict,
    cfg: WalkForwardMCTSConfig,
    fpc_cache: dict,
    run_dir: Path,
    metrics_path: Path | None = None,
):
    """
    Evaluate genes on a chunk using the cached one-FPC-curve-per-chunk design.
    """
    good_idx = np.asarray(good_idx, dtype=np.int64)

    instantiate_for_chunk(X, chunk_num)

    fpc_obj = get_fpc_for_chunk(
        fpc_cache=fpc_cache,
        X_template=X,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        cfg=cfg,
        run_dir=run_dir,
        metrics_path=metrics_path,
    )

    pvals, zscores, details = _E.evaluate_genes_from_opg_fpc_fast(
        population=X,
        good_idx=good_idx,
        chunk_num=chunk_num,
        solver_kwargs=solver_kwargs,
        fpc_curve=fpc_obj["pd_fpc"],
        perm_mu=fpc_obj["perm_mu"],
        visualize=cfg.eval_visualize,
        viz_kwargs={"bins": 20, "top_n": None, "line_alpha": 0.25},
        return_details=True,
    )

    return pvals, zscores, details


# ---------------------------------------------------------------------
# local child/action exploit policy drift, ignoring UCT parent resolution
# ---------------------------------------------------------------------

def _softmax_dict(scores, base=np.e, temp=1.0):
    if scores is None or len(scores) == 0:
        return {}

    keys = list(scores.keys())
    vals = np.asarray([scores[k] for k in keys], dtype=np.float64)
    valid = np.isfinite(vals)

    if not np.any(valid):
        p = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
        return {keys[i]: float(p[i]) for i in range(len(keys))}

    temp = float(np.clip(temp, 1e-6, 1.0))
    scaled = vals / temp
    scaled[valid] = scaled[valid] - np.max(scaled[valid])

    w = np.zeros_like(vals, dtype=np.float64)
    w[valid] = np.exp(np.log(base) * scaled[valid])

    if w.sum() <= 0 or not np.isfinite(w.sum()):
        w = valid.astype(np.float64)

    p = w / w.sum()

    return {keys[i]: float(p[i]) for i in range(len(keys))}


def _tv_dict(p_old, p_new):
    if p_old is None:
        p_old = {}
    if p_new is None:
        p_new = {}

    keys = set(p_old.keys()) | set(p_new.keys())

    if len(keys) == 0:
        return np.nan

    return float(0.5 * sum(abs(p_new.get(k, 0.0) - p_old.get(k, 0.0)) for k in keys))


def mcts_action_policy_snapshot(G, use_explore=False):
    """
    Local action policy snapshot only.

    This ignores UCT parent selection. It tracks, for each opened parent_key,
    the softmax policy over child_tf actions.

    use_explore=False:
        softmax(edge Q only), best for inference-like exploit stability.

    use_explore=True:
        softmax(edge Q + edge U), full local UCB policy.
    """
    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    action_policies = {}

    for parent_key, opened in getattr(G, "_MCTS_CHILDREN", {}).items():
        if opened is None or len(opened) == 0:
            continue

        scores = {}

        for child_tf in sorted(opened):
            child_tf = int(child_tf)
            edge_key = (parent_key, child_tf)

            q = float(G._MCTS_EDGE_MU.get(
                edge_key,
                getattr(G, "_MCTS_BASE_PRIOR", 0.0),
            ))

            if use_explore and hasattr(G, "_mcts_edge_ucb"):
                score = G._mcts_edge_ucb(parent_key, child_tf)
            else:
                score = q

            scores[child_tf] = float(score)

        action_policies[parent_key] = _softmax_dict(scores, base=base, temp=temp)

    return action_policies


def mcts_action_policy_drift(prev_snap, curr_snap, G=None, weight_by_count=True):
    if prev_snap is None:
        return {"drift": np.nan, "n_parent_policies": 0}

    vals = []
    weights = []

    for parent_key in set(prev_snap.keys()) | set(curr_snap.keys()):
        tv = _tv_dict(prev_snap.get(parent_key, {}), curr_snap.get(parent_key, {}))

        if not np.isfinite(tv):
            continue

        vals.append(tv)

        if weight_by_count and G is not None:
            w = G._MCTS_NODE_COUNT.get(parent_key, 0) + G._MCTS_NODE_EXPLORE_COUNT.get(parent_key, 0)
            weights.append(max(float(w), 1.0))
        else:
            weights.append(1.0)

    if len(vals) == 0:
        return {"drift": np.nan, "n_parent_policies": 0}

    vals = np.asarray(vals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    return {
        "drift": float(np.sum(vals * weights) / np.sum(weights)),
        "n_parent_policies": int(len(vals)),
    }


# ---------------------------------------------------------------------
# training and inference
# ---------------------------------------------------------------------

def train_one_iteration(
    G,
    cfg: WalkForwardMCTSConfig,
    chunk_i: int,
    chunk_j: int,
    fpc_cache: dict,
    run_dir: Path,
    metrics_path: Path,
):
    set_grammar_mode(G, "train")

    X, G, s_idx, walker, evaluation = ep.solver_inner(
        make_init_kwargs(cfg, G),
        make_solver_kwargs(cfg),
        make_logwalker_kwargs(),
        G,
        chunk_num=chunk_i,
    )

    eval_solver_kwargs = make_eval_solver_kwargs(cfg)

    pvals, zscores, details = evaluate_population_on_chunk_cached(
        X=X,
        good_idx=s_idx,
        chunk_num=chunk_j,
        solver_kwargs=eval_solver_kwargs,
        cfg=cfg,
        fpc_cache=fpc_cache,
        run_dir=run_dir,
        metrics_path=metrics_path,
    )

    kept_s_idx, kept_scores, family_idx, family_scores = _E.reduce_scored_family_indices(
        X,
        s_idx,
        pvals[s_idx],
    )

    G.update(X, family_idx, family_scores)

    return {
        "X": X,
        "G": G,
        "s_idx": s_idx,
        "pvals": pvals,
        "zscores": zscores,
    }


def make_infer_initialization_kwargs(G, cfg):
    """
    Direct initialization kwargs for inference population generation.

    This avoids solver_inner and only uses the learned grammar to generate
    a fresh population.
    """

    return {
        "structure"   : "Intraday",
        "incl_time"   : True,
        "data_file"   : cfg.data_file,
        "epoch_idx"   : [0],
        "hlocv_idx"   : [1, 2, 3, 4],
        "pop_size"    : cfg.pop_size,

        "grmr_prior"  : G,
        "grmr_type"   : "MCTS",
        "grmr_mdl"    : cfg.grmr_mdl,
        "grmr_p_mttn" : 0.0,
        "grmr_p_csvr" : 0.0,
        "grmr_a_sens" : G._alpha_sensor_freq,

        "chunk_size"  : 1,
        "wf_windows"  : cfg.n_chunks,
        "verbose"     : 0,
    }


def generate_infer_population(G, cfg, chunk_i=None):
    """
    Generate one inference population directly from the current grammar.

    This should not call solver_inner.
    It only initializes a population using the learned MCTS grammar.
    """

    set_grammar_mode(G, "infer")

    kwargs = make_infer_initialization_kwargs(G, cfg)

    X, G_out = _I.initialize(**kwargs)

    # Do not replace the main grammar with G_out unless you explicitly want to.
    # In inference mode, G should remain the learned/static grammar.
    s_idx = np.asarray(X._G_idx, dtype=np.int64)

    return X, s_idx


def evaluate_inference_funnel(
    G,
    cfg: WalkForwardMCTSConfig,
    chunk_i: int,
    chunk_j: int,
    chunk_k: int,
    fpc_cache: dict,
    run_dir: Path,
    metrics_path: Path,
):
    """
    Generate infer populations and evaluate i -> j -> k funnel.

    Success criterion:
        zscore >= cfg.success_z
    """
    set_grammar_mode(G, "infer")

    eval_solver_kwargs = make_eval_solver_kwargs(cfg)
    records = []
    all_k_scores = []
    i_counts = []
    ij_counts = []

    for pop_n in range(cfg.infer_populations):
        X, s_idx = generate_infer_population(G, cfg, chunk_i)

        _, z_i, _ = evaluate_population_on_chunk_cached(
            X=X,
            good_idx=s_idx,
            chunk_num=chunk_i,
            solver_kwargs=eval_solver_kwargs,
            cfg=cfg,
            fpc_cache=fpc_cache,
            run_dir=run_dir,
            metrics_path=metrics_path,
        )

        success_i = s_idx[z_i[s_idx] >= cfg.success_z]

        if success_i.size:
            _, z_j, _ = evaluate_population_on_chunk_cached(
                X=X,
                good_idx=success_i,
                chunk_num=chunk_j,
                solver_kwargs=eval_solver_kwargs,
                cfg=cfg,
                fpc_cache=fpc_cache,
                run_dir=run_dir,
                metrics_path=metrics_path,
            )
            success_ij = success_i[z_j[success_i] >= cfg.success_z]
        else:
            success_ij = np.asarray([], dtype=np.int64)

        if success_ij.size:
            _, z_k, _ = evaluate_population_on_chunk_cached(
                X=X,
                good_idx=success_ij,
                chunk_num=chunk_k,
                solver_kwargs=eval_solver_kwargs,
                cfg=cfg,
                fpc_cache=fpc_cache,
                run_dir=run_dir,
                metrics_path=metrics_path,
            )
            k_scores = z_k[success_ij]
        else:
            k_scores = np.asarray([], dtype=np.float64)

        i_counts.append(int(success_i.size))
        ij_counts.append(int(success_ij.size))
        all_k_scores.extend([float(x) for x in k_scores if np.isfinite(x)])

        records.append({
            "infer_population": pop_n,
            "chunk_i": chunk_i,
            "chunk_j": chunk_j,
            "chunk_k": chunk_k,
            "n_candidates": int(s_idx.size),
            "i_success_count": int(success_i.size),
            "ij_success_count": int(success_ij.size),
            "k_eval_count": int(k_scores.size),
            "k_z_mean": float(np.mean(k_scores)) if k_scores.size else np.nan,
            "k_z_p50": float(np.quantile(k_scores, 0.50)) if k_scores.size else np.nan,
            "k_z_p75": float(np.quantile(k_scores, 0.75)) if k_scores.size else np.nan,
            "k_z_p90": float(np.quantile(k_scores, 0.90)) if k_scores.size else np.nan,
            "k_z_max": float(np.max(k_scores)) if k_scores.size else np.nan,
        })

    all_k_scores = np.asarray(all_k_scores, dtype=np.float64)

    summary = {
        "chunk_i": chunk_i,
        "chunk_j": chunk_j,
        "chunk_k": chunk_k,
        "infer_populations": cfg.infer_populations,
        "total_i_success_count": int(np.sum(i_counts)),
        "total_ij_success_count": int(np.sum(ij_counts)),
        "total_k_eval_count": int(all_k_scores.size),
        "k_z_mean": float(np.mean(all_k_scores)) if all_k_scores.size else np.nan,
        "k_z_median": float(np.median(all_k_scores)) if all_k_scores.size else np.nan,
        "k_z_p75": float(np.quantile(all_k_scores, 0.75)) if all_k_scores.size else np.nan,
        "k_z_p90": float(np.quantile(all_k_scores, 0.90)) if all_k_scores.size else np.nan,
        "k_z_max": float(np.max(all_k_scores)) if all_k_scores.size else np.nan,
    }

    return summary, records, all_k_scores


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------

def save_training_plot(chunk_metrics, run_dir: Path, chunk_start: int):
    if not chunk_metrics:
        return

    xs = np.arange(len(chunk_metrics))
    drift = np.asarray([m["policy_drift"] for m in chunk_metrics], dtype=np.float64)
    h = np.asarray([m["policy_drift_h"] for m in chunk_metrics], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(xs, drift, alpha=0.35, label="raw local action drift")
    ax.plot(xs, h, linewidth=3, label="smoothed h")
    ax.axhline(chunk_metrics[-1]["delta_L"], linestyle="--", alpha=0.6, label="delta_L")
    ax.set_title(f"chunk {chunk_start}: local action-policy drift convergence")
    ax.set_xlabel("iteration within chunk")
    ax.set_ylabel("drift")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(run_dir / "plots" / f"chunk_{chunk_start:03d}_training_drift.png", dpi=150)
    plt.close(fig)


def save_k_score_plot(k_scores, summary, run_dir: Path, chunk_start: int):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    if k_scores.size:
        ax.hist(k_scores, bins=30, alpha=0.75)
        ax.axvline(2.0, linestyle="--", alpha=0.65, label="z=2")
    else:
        ax.text(0.5, 0.5, "no ij survivors reached chunk k", ha="center", va="center")

    ax.set_title(
        f"chunk {chunk_start}: k={summary['chunk_k']} z-scores | "
        f"ij_success={summary['total_ij_success_count']}"
    )
    ax.set_xlabel("chunk k z-score")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(run_dir / "plots" / f"chunk_{chunk_start:03d}_k_scores.png", dpi=150)
    plt.close(fig)


def save_overall_k_performance_plot(run_dir: Path, summaries: list[dict]):
    if not summaries:
        return

    chunk_start = np.asarray([s["chunk_start"] for s in summaries], dtype=int)
    k_mean = np.asarray([s["k_z_mean"] for s in summaries], dtype=np.float64)
    k_p90 = np.asarray([s["k_z_p90"] for s in summaries], dtype=np.float64)
    ij_count = np.asarray([s["total_ij_success_count"] for s in summaries], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(11, 4), constrained_layout=True)

    ax1.plot(chunk_start, k_mean, marker="o", label="k z mean")
    ax1.plot(chunk_start, k_p90, marker="o", label="k z p90")
    ax1.axhline(2.0, linestyle="--", alpha=0.5, label="z=2")
    ax1.set_xlabel("chunk_start")
    ax1.set_ylabel("chunk k z-score")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(chunk_start, ij_count, alpha=0.20, label="ij success count")
    ax2.set_ylabel("ij success count")
    ax2.legend(loc="upper right")

    fig.savefig(run_dir / "plots" / "overall_k_performance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# main runner
# ---------------------------------------------------------------------

def run_walk_forward(cfg: WalkForwardMCTSConfig):
    reload(_E)
    reload(ep)
    reload(_I)

    run_dir = make_run_dir(cfg)
    metrics_path = run_dir / "metrics.jsonl"
    infer_path = run_dir / "inference_records.jsonl"
    summary_path = run_dir / "chunk_summaries.jsonl"

    G = build_initial_grammar(cfg)

    # one FPC curve per chunk for the entire run
    fpc_cache = {}

    # local action-policy drift state
    prev_action_snapshot = None
    h = np.nan

    all_summaries = []

    for chunk_start in range(cfg.n_chunks - 2):
        chunk_i = chunk_start
        chunk_j = chunk_start + 1
        chunk_k = chunk_start + 2

        print(f"\n=== chunk_start={chunk_start} | i={chunk_i}, j={chunk_j}, k={chunk_k} ===")

        if chunk_start > 0:
            apply_grammar_decay(G, cfg.grammar_gamma)

        # reset local convergence smoother per walk-forward window
        prev_action_snapshot = None
        h = np.nan
        chunk_metrics = []

        for iter_n in range(cfg.max_iters_per_chunk):
            out = train_one_iteration(
                G=G,
                cfg=cfg,
                chunk_i=chunk_i,
                chunk_j=chunk_j,
                fpc_cache=fpc_cache,
                run_dir=run_dir,
                metrics_path=metrics_path,
            )

            G = out["G"]

            # local child/action exploit-policy drift, ignoring UCT parent selection
            curr_snap = mcts_action_policy_snapshot(G, use_explore=False)
            drift_info = mcts_action_policy_drift(
                prev_action_snapshot,
                curr_snap,
                G=G,
                weight_by_count=True,
            )
            prev_action_snapshot = curr_snap

            d = drift_info["drift"]

            if np.isfinite(d):
                if not np.isfinite(h):
                    h = d
                else:
                    h = h * (1.0 - np.e ** (-cfg.drift_kappa)) + d * np.e ** (-cfg.drift_kappa)

            metric = {
                "event": "train_iter",
                "chunk_start": chunk_start,
                "chunk_i": chunk_i,
                "chunk_j": chunk_j,
                "chunk_k": chunk_k,
                "iter_n": iter_n,
                "policy_drift": d,
                "policy_drift_h": h,
                "delta_L": cfg.delta_L,
                "n_parent_policies": drift_info["n_parent_policies"],
                "mcts_node_mu_count": len(getattr(G, "_MCTS_NODE_MU", {})),
                "mcts_edge_mu_count": len(getattr(G, "_MCTS_EDGE_MU", {})),
                "mcts_children_edges": sum(len(v) for v in getattr(G, "_MCTS_CHILDREN", {}).values()),
                "alpha_decision_count": len(getattr(G, "_MCTS_ALPHA_DECISION_MU", {})),
                "alpha_node_count": len(getattr(G, "_MCTS_ALPHA_NODE_MU", {})),
                "alpha_edge_count": len(getattr(G, "_MCTS_ALPHA_EDGE_MU", {})),
                "fpc_cache_size": len(fpc_cache),
            }

            append_jsonl(metrics_path, metric)
            chunk_metrics.append(metric)

            print(
                f"chunk={chunk_start} iter={iter_n} "
                f"d={d:.4f} h={h:.4f} "
                f"edge_states={metric['mcts_edge_mu_count']} "
                f"alpha_edges={metric['alpha_edge_count']} "
                f"fpc_cached={len(fpc_cache)}"
            )

            if iter_n + 1 >= cfg.min_iters_per_chunk and np.isfinite(h) and h < cfg.delta_L:
                print(f"delta_L reached: h={h:.4f} < {cfg.delta_L:.4f}")
                break

        save_training_plot(chunk_metrics, run_dir, chunk_start)

        grammar_path = run_dir / "grammars" / f"grammar_after_chunk_{chunk_start:03d}.pkl"
        save_grammar(G, grammar_path)

        summary, records, k_scores = evaluate_inference_funnel(
            G=G,
            cfg=cfg,
            chunk_i=chunk_i,
            chunk_j=chunk_j,
            chunk_k=chunk_k,
            fpc_cache=fpc_cache,
            run_dir=run_dir,
            metrics_path=metrics_path,
        )

        summary["chunk_start"] = chunk_start
        summary["grammar_path"] = str(grammar_path)
        summary["fpc_cache_size"] = len(fpc_cache)

        append_jsonl(summary_path, summary)
        all_summaries.append(summary)

        for rec in records:
            rec["chunk_start"] = chunk_start
            append_jsonl(infer_path, rec)

        save_k_score_plot(k_scores, summary, run_dir, chunk_start)
        save_overall_k_performance_plot(run_dir, all_summaries)

        print(
            f"infer summary | "
            f"i_success={summary['total_i_success_count']} "
            f"ij_success={summary['total_ij_success_count']} "
            f"k_mean={summary['k_z_mean']:.3f} "
            f"k_p90={summary['k_z_p90']:.3f}"
        )

        set_grammar_mode(G, "train")

    save_grammar(G, run_dir / "grammars" / "grammar_final.pkl")
    save_pickle(fpc_cache, run_dir / "fpc_cache" / "fpc_cache_final.pkl")

    print(f"\nrun complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    cfg = WalkForwardMCTSConfig()
    run_walk_forward(cfg)
