"""
mcts_util.py

Notebook-safe persistent MCTS run utilities.

Use this file to:
1. Run your original MCTS loop without flooding notebook output.
2. Save every plot type from the old loop into mcts_runs/<run_name>.
3. Keep run_name="tmp" as an overwriteable scratch run.
4. Preserve named runs by changing run_name.
5. Let a separate Streamlit app view live and past runs.

This file intentionally does not modify visualization.py or evaluation.py.
It captures their plt.show() output and saves figures to disk.
"""

from __future__ import annotations

import os
import io
import dill
import gc
import json
import time
import gzip
import pickle
import shutil
import traceback
import hashlib
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


# ---------------------------------------------------------------------
# json / filesystem helpers
# ---------------------------------------------------------------------

def jsonable(x: Any) -> Any:
    """Convert common numpy/path/python objects into JSON-safe objects."""
    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return [jsonable(v) for v in x.ravel().tolist()]
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


def write_json_atomic(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=indent)

    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(obj)) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def finite_stats(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {"n": 0, "min": None, "mean": None, "max": None, "std": None}

    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def safe_array(x: Any) -> np.ndarray:
    try:
        return np.asarray(x)
    except Exception:
        return np.asarray([repr(x)], dtype=object)


def stack_depth_vectors(vecs: list[Any]) -> np.ndarray:
    if vecs is None or len(vecs) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    max_len = 0
    for v in vecs:
        if v is None:
            continue
        max_len = max(max_len, np.asarray(v).ravel().shape[0])

    if max_len == 0:
        return np.zeros((0, 0), dtype=np.float64)

    out = np.zeros((len(vecs), max_len), dtype=np.float64)

    for i, v in enumerate(vecs):
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float64).ravel()
        out[i, :vv.shape[0]] = vv

    return out


# ---------------------------------------------------------------------
# run storage
# ---------------------------------------------------------------------

class MCTSRunStore:
    """
    Persistent run folder.

    Default behavior:
        run_root="mcts_runs"
        run_name="tmp"
        overwrite=True

    So a new tmp run clears:
        mcts_runs/tmp

    Named runs are preserved if overwrite=False or if run_name changes.
    """

    def __init__(
        self,
        run_root: str | Path = "mcts_runs",
        run_name: str = "tmp",
        overwrite: bool = True,
        summary_dpi: int = 120,
    ):
        self.run_root = Path(run_root)
        self.run_name = str(run_name)
        self.run_dir = self.run_root / self.run_name
        self.iter_dir = self.run_dir / "iterations"
        self.plot_dir = self.run_dir / "plots"
        self.snapshot_dir = self.run_dir / "snapshots"

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.tops_path = self.run_dir / "tops.jsonl"
        self.plot_index_path = self.run_dir / "plot_index.jsonl"
        self.log_path = self.run_dir / "console.log"
        self.status_path = self.run_dir / "status.json"
        self.histories_path = self.run_dir / "histories.pkl.gz"

        self.summary_dpi = int(summary_dpi)

        if overwrite and self.run_dir.exists():
            shutil.rmtree(self.run_dir)

        self.iter_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.write_status("initialized", k=None)
        self.log(f"initialized run directory: {self.run_dir}")

    def log(self, text: Any = "") -> None:
        text = str(text)

        if len(text.strip()) == 0:
            return

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    def write_status(self, status: str, k: int | None = None, extra: dict | None = None) -> None:
        payload = {
            "status": status,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "last_iteration": None if k is None else int(k),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra:
            payload.update(jsonable(extra))

        write_json_atomic(self.status_path, payload)

    def append_metrics(self, row: dict) -> None:
        append_jsonl(self.metrics_path, row)

    def append_tops(self, row: dict) -> None:
        append_jsonl(self.tops_path, row)

    def append_plot_index(self, k: int, section: str, path: str | Path) -> None:
        append_jsonl(self.plot_index_path, {
            "k": int(k),
            "section": str(section),
            "path": str(path),
        })

    def iter_path(self, k: int) -> Path:
        p = self.iter_dir / f"iter_{int(k):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def plot_path(self, k: int, section: str, j: int = 0) -> Path:
        p = self.plot_dir / f"iter_{int(k):04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{section}_{int(j):02d}.png"

    def save_fig(self, fig, k: int, section: str, j: int = 0, close: bool = True) -> str:
        path = self.plot_path(k, section, j)
        fig.savefig(path, dpi=self.summary_dpi, bbox_inches="tight")
        self.append_plot_index(k, section, path)

        if close:
            plt.close(fig)

        return str(path)

    def save_open_figures(self, k: int, section: str, save: bool = True) -> list[str]:
        saved = []

        for j, num in enumerate(list(plt.get_fignums())):
            fig = plt.figure(num)

            if save:
                saved.append(self.save_fig(fig, k=k, section=section, j=j, close=False))

            plt.close(fig)

        return saved

    def save_arrays_npz(self, k: int, **arrays) -> str:
        path = self.iter_path(k) / "arrays.npz"

        safe = {}
        for name, val in arrays.items():
            safe[name] = safe_array(val)

        np.savez_compressed(path, **safe)
        return str(path)

    def save_pickle_gz(self, path: str | Path, obj: Any) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        return str(path)

    def save_histories(self, histories: dict) -> str:
        return self.save_pickle_gz(self.histories_path, histories)

    def save_snapshot(self, k: int, obj: dict) -> str:
        path = self.snapshot_dir / f"snapshot_iter_{int(k):04d}.pkl.gz"
        return self.save_pickle_gz(path, obj)



@contextmanager
def capture_console(store: MCTSRunStore, section: str, k: int | None = None):
    """
    Capture print/stderr output into console.log and the iteration console file.
    """
    buf = io.StringIO()

    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            yield

    except Exception:
        buf.write("\n\nEXCEPTION:\n")
        buf.write(traceback.format_exc())
        raise

    finally:
        text = buf.getvalue()

        if len(text.strip()) > 0:
            header = (
                f"\n--- iteration {k} | {section} ---"
                if k is not None
                else f"\n--- {section} ---"
            )

            store.log(header)
            store.log(text)

            if k is not None:
                path = store.iter_path(k) / "console.txt"
                with open(path, "a", encoding="utf-8") as f:
                    f.write(header + "\n")
                    f.write(text.rstrip() + "\n")


@contextmanager
def capture_plots(store: MCTSRunStore, section: str, k: int, save: bool = True):
    """
    Temporarily replace plt.show().

    Existing functions in visualization.py/evaluation.py can keep calling plt.show(),
    but this context saves and closes the figures instead of pushing them into a notebook.
    """
    old_show = plt.show

    def quiet_show(*args, **kwargs):
        store.save_open_figures(k=k, section=section, save=save)

    try:
        plt.show = quiet_show
        yield

    finally:
        plt.show = old_show
        store.save_open_figures(k=k, section=section, save=save)


def mcts_exploit_policy_snapshot(G, X, depth_gamma=0.70):
    """
    Snapshot the exploit-only MCTS policy.

    This ignores exploration terms entirely.

    parent_policy:
        softmax over node exploitation values Q_node

    child_policies:
        softmax over edge exploitation values Q_edge

    parent_weights:
        parent-policy probability weighted by depth discount
    """
    legal_idx = np.asarray(X._L_idx, dtype=np.int64)

    if legal_idx.size == 0:
        return {
            "parent_policy": {},
            "child_policies": {},
            "parent_weights": {},
        }

    valid_mask = np.ones(legal_idx.shape[0], dtype=bool)

    if hasattr(G, "_mcts_valid_depth_parent_mask"):
        valid_mask &= G._mcts_valid_depth_parent_mask(
            instructions=X._instructions,
            legal_idx=legal_idx,
        )

    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    parent_scores = {}
    parent_depths = {}

    for i, idx in enumerate(legal_idx):
        if not valid_mask[i]:
            continue

        idx = int(idx)
        parent_key = G._mcts_state_key(X._instructions, idx)

        q = float(mcts_getdict(G, "_MCTS_NODE_MU").get(
            parent_key,
            getattr(G, "_MCTS_BASE_PRIOR", 0.0),
        ))

        if hasattr(G, "_mcts_row_depth"):
            d = int(G._mcts_row_depth(X._instructions, idx))
        else:
            d = 0

        #duplicate rows can map to the same state key
        if parent_key not in parent_scores or q > parent_scores[parent_key]:
            parent_scores[parent_key] = q
            parent_depths[parent_key] = d

    parent_policy = _softmax_dict(
        parent_scores,
        base=base,
        temp=temp,
    )

    parent_weights = {}
    for parent_key, p in parent_policy.items():
        d = parent_depths.get(parent_key, 0)
        parent_weights[parent_key] = float(p * (depth_gamma ** d))

    sw = sum(parent_weights.values())
    if sw > 0:
        parent_weights = {k: v / sw for k, v in parent_weights.items()}

    child_policies = {}

    for parent_key in parent_policy.keys():
        opened = mcts_getdict(G, "_MCTS_CHILDREN").get(parent_key, set())

        if opened is None or len(opened) == 0:
            continue

        child_scores = {}

        for child_tf in sorted(opened):
            child_tf = int(child_tf)
            edge_key = (parent_key, child_tf)

            q = float(mcts_getdict(G, "_MCTS_EDGE_MU").get(
                edge_key,
                getattr(G, "_MCTS_BASE_PRIOR", 0.0),
            ))

            child_scores[child_tf] = q

        child_policies[parent_key] = _softmax_dict(
            child_scores,
            base=base,
            temp=temp,
        )

    return {
        "parent_policy": parent_policy,
        "child_policies": child_policies,
        "parent_weights": parent_weights,
    }

def save_fpc_gene_eval_plot_data(
    store,
    k,
    gene_eval_details,
    fpc_curve,
    perm_mu,
    *,
    min_var=1e-18,
    band_mult=2.0,
    grid_n=300,
):
    prop_g = np.asarray(gene_eval_details.get("prop_g", []), dtype=np.float64)
    observed_scores_g = np.asarray(gene_eval_details.get("observed_scores_g", []), dtype=np.float64)
    fpc_std_g = np.asarray(gene_eval_details.get("fpc_std_g", []), dtype=np.float64)

    keep = (
        np.isfinite(prop_g)
        & np.isfinite(observed_scores_g)
        & np.isfinite(fpc_std_g)
        & (prop_g > 0)
    )

    path = store.iter_path(k) / "fpc_gene_eval_plot_data.npz"

    if not np.any(keep):
        np.savez_compressed(
            path,
            keep=keep,
            prop_keep=np.asarray([], dtype=np.float64),
            observed_keep=np.asarray([], dtype=np.float64),
            fpc_std_keep=np.asarray([], dtype=np.float64),
            p_grid=np.asarray([], dtype=np.float64),
            v_grid=np.asarray([], dtype=np.float64),
            s_grid=np.asarray([], dtype=np.float64),
            perm_mu=float(perm_mu),
            band_mult=float(band_mult),
        )
        return str(path)

    p_min = max(1e-6, float(np.nanmin(prop_g[keep])))
    p_max = min(0.999999, float(np.nanmax(prop_g[keep])))

    p_grid = np.linspace(p_min, p_max, int(grid_n))
    v_grid = np.asarray(fpc_curve(p_grid), dtype=np.float64)
    s_grid = np.sqrt(np.maximum(v_grid, min_var))

    np.savez_compressed(
        path,
        keep=keep,
        prop_g=prop_g,
        observed_scores_g=observed_scores_g,
        fpc_std_g=fpc_std_g,
        prop_keep=prop_g[keep],
        observed_keep=observed_scores_g[keep],
        fpc_std_keep=fpc_std_g[keep],
        p_grid=p_grid,
        v_grid=v_grid,
        s_grid=s_grid,
        perm_mu=float(perm_mu),
        band_mult=float(band_mult),
        upper_band=float(perm_mu) + float(band_mult) * s_grid,
        lower_band=float(perm_mu) - float(band_mult) * s_grid,
    )

    return str(path)

def _softmax_dict(scores, base=np.e, temp=1.0):
    """
    Small local softmax helper for dict[key] -> score.
    """
    if scores is None or len(scores) == 0:
        return {}

    keys = list(scores.keys())
    vals = np.asarray([scores[k] for k in keys], dtype=np.float64)

    valid = np.isfinite(vals)
    if not np.any(valid):
        p = 1.0 / len(keys)
        return {k: p for k in keys}

    temp = float(np.clip(temp, 1e-6, 1.0))
    vals = vals / temp

    if base == 1:
        w = valid.astype(np.float64)
    else:
        z = np.full(vals.shape, -np.inf, dtype=np.float64)
        z[valid] = np.log(base) * vals[valid]
        z[valid] -= np.max(z[valid])

        w = np.zeros(vals.shape, dtype=np.float64)
        w[valid] = np.exp(z[valid])

    if w.sum() <= 0 or not np.isfinite(w.sum()):
        w = valid.astype(np.float64)

    probs = w / w.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def _policy_tv_distance(p_old, p_new):
    """
    Total variation distance between two sparse probability dictionaries.
    """
    if p_old is None:
        p_old = {}
    if p_new is None:
        p_new = {}

    keys = set(p_old.keys()) | set(p_new.keys())

    if len(keys) == 0:
        return np.nan

    return 0.5 * float(sum(abs(p_old.get(k, 0.0) - p_new.get(k, 0.0)) for k in keys))


def mcts_root_weighted_exploit_policy_drift(prev_snap, curr_snap):
    """
    Compare two exploit-only MCTS policy snapshots.

    This measures how much the learned exploitation policy changes,
    ignoring exploration coefficient decay.
    """
    if prev_snap is None:
        return {
            "parent_drift": np.nan,
            "child_drift": np.nan,
            "total_drift": np.nan,
        }

    parent_drift = _policy_tv_distance(
        prev_snap["parent_policy"],
        curr_snap["parent_policy"],
    )

    all_parent_keys = (
        set(prev_snap["child_policies"].keys())
        | set(curr_snap["child_policies"].keys())
    )

    child_vals = []
    child_weights = []

    for parent_key in all_parent_keys:
        p_old = prev_snap["child_policies"].get(parent_key, {})
        p_new = curr_snap["child_policies"].get(parent_key, {})

        tv = _policy_tv_distance(p_old, p_new)

        if not np.isfinite(tv):
            continue

        w = curr_snap["parent_weights"].get(
            parent_key,
            prev_snap["parent_weights"].get(parent_key, 0.0),
        )

        child_vals.append(tv)
        child_weights.append(w)

    if len(child_vals) == 0:
        child_drift = np.nan
    else:
        child_vals = np.asarray(child_vals, dtype=np.float64)
        child_weights = np.asarray(child_weights, dtype=np.float64)

        if child_weights.sum() <= 0:
            child_drift = float(np.mean(child_vals))
        else:
            child_drift = float(np.sum(child_vals * child_weights) / child_weights.sum())

    vals = np.asarray([parent_drift, child_drift], dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    total_drift = np.nan if vals.size == 0 else float(np.mean(vals))

    return {
        "parent_drift": parent_drift,
        "child_drift": child_drift,
        "total_drift": total_drift,
    }

# ---------------------------------------------------------------------
# sparse MCTS helpers
# ---------------------------------------------------------------------

def mcts_getdict(G, name: str) -> dict:
    d = getattr(G, name, None)
    return {} if d is None else d


def mcts_dict_sum(d: dict | None) -> float:
    if d is None or len(d) == 0:
        return 0.0
    return float(np.sum(list(d.values())))


def mcts_numeric_signature(G) -> np.ndarray:
    """
    Numeric signature for sparse MCTS + alpha-MCTS memory.
    This mirrors your notebook helper.
    """
    return np.asarray([
        len(mcts_getdict(G, "_MCTS_NODE_MU")),
        len(mcts_getdict(G, "_MCTS_EDGE_MU")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_NODE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_EDGE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_NODE_EXPLORE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_EDGE_EXPLORE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_NODE_MU")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_EDGE_MU")),
        float(getattr(G, "_MCTS_EXPLOIT_T", 0) or 0),
        float(getattr(G, "_MCTS_EXPLORE_T", 0) or 0),

        len(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")),
        len(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")),
        len(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_DECISION_EXPLORE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_NODE_EXPLORE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_EDGE_EXPLORE_COUNT")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")),
        mcts_dict_sum(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")),
        float(getattr(G, "_MCTS_ALPHA_EXPLOIT_T", 0) or 0),
        float(getattr(G, "_MCTS_ALPHA_EXPLORE_T", 0) or 0),
    ], dtype=np.float64)


def mcts_top_edges(G, n: int = 8):
    d = mcts_getdict(G, "_MCTS_EDGE_MU")
    if len(d) == 0:
        return []
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]


def mcts_top_alpha_decisions(G, n: int = 8):
    d = mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")
    if len(d) == 0:
        return []
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]


def mcts_top_alpha_edges(G, n: int = 8):
    d = mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")
    if len(d) == 0:
        return []
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]


def top_items_json(items, count_dict=None, explore_count_dict=None) -> list[dict]:
    out = []

    for key, val in items:
        row = {
            "key": repr(key),
            "mu": float(val),
        }

        if count_dict is not None:
            row["n"] = int(count_dict.get(key, 0))

        if explore_count_dict is not None:
            row["ne"] = int(explore_count_dict.get(key, 0))

        out.append(row)

    return out


# ---------------------------------------------------------------------
# progressive widening threshold helpers
# ---------------------------------------------------------------------

def decayed_pw_thresholds(
    k: int,
    *,
    freeze_threshold: float = 0.55,
    unfreeze_threshold: float = 0.25,
    decay_rate: float = 0.01,
    floor: float = 0.02,
) -> tuple[float, float]:
    """
    Return exponentially decayed freeze/unfreeze thresholds.

    Lower thresholds later in training make the grammar more willing to
    enter near-freeze mode as the run gets older.
    """
    decay = float(np.exp(-float(decay_rate) * int(k)))

    ft = max(float(floor), float(freeze_threshold) * decay)
    ut = max(float(floor), float(unfreeze_threshold) * decay)

    # preserve hysteresis ordering
    if ut >= ft:
        ut = max(float(floor), 0.5 * ft)

    return ft, ut

def hard_reset_mcts_memory(G, *, keep_params=True):
    """
    Fully clear MCTS learned memory and explored structure.

    This is stronger than gamma decay.
    Use this only when you want the next window to behave like a fresh grammar
    with the same hyperparameters.
    """

    # normal MCTS
    for name in [
        "_MCTS_NODE_MU",
        "_MCTS_EDGE_MU",
        "_MCTS_NODE_COUNT",
        "_MCTS_EDGE_COUNT",
        "_MCTS_NODE_CUM",
        "_MCTS_EDGE_CUM",
        "_MCTS_NODE_EXPLORE_COUNT",
        "_MCTS_EDGE_EXPLORE_COUNT",
        "_MCTS_CHILDREN",
    ]:
        if hasattr(G, name):
            setattr(G, name, {})

    G._MCTS_EXPLOIT_T = 0
    G._MCTS_EXPLORE_T = 0

    # alpha MCTS
    for name in [
        "_MCTS_ALPHA_DECISION_MU",
        "_MCTS_ALPHA_NODE_MU",
        "_MCTS_ALPHA_EDGE_MU",
        "_MCTS_ALPHA_DECISION_COUNT",
        "_MCTS_ALPHA_NODE_COUNT",
        "_MCTS_ALPHA_EDGE_COUNT",
        "_MCTS_ALPHA_DECISION_CUM",
        "_MCTS_ALPHA_NODE_CUM",
        "_MCTS_ALPHA_EDGE_CUM",
        "_MCTS_ALPHA_DECISION_EXPLORE_COUNT",
        "_MCTS_ALPHA_NODE_EXPLORE_COUNT",
        "_MCTS_ALPHA_EDGE_EXPLORE_COUNT",
    ]:
        if hasattr(G, name):
            setattr(G, name, {})

    G._MCTS_ALPHA_EXPLOIT_T = 0
    G._MCTS_ALPHA_EXPLORE_T = 0

    if hasattr(G, "_MCTS_TRACE") and isinstance(G._MCTS_TRACE, list):
        G._MCTS_TRACE.clear()

    return G

def current_pw_scale(G, baseline_pw_c: float = 0.75) -> float:
    """
    Approximate current PW scale from the live grammar's _MCTS_PW_C.
    If the grammar has not initialized this yet, treat it as baseline scale=1.
    """
    pw_c = getattr(G, "_MCTS_PW_C", None)

    if pw_c is None:
        return 1.0

    baseline_pw_c = float(baseline_pw_c)
    if baseline_pw_c <= 0:
        return np.nan

    try:
        return float(pw_c) / baseline_pw_c
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# plot recreations / updated plot functions
# ---------------------------------------------------------------------

def maybe_show(fig, show: bool = False):
    if show:
        plt.show()
    return fig


def plot_mcts_state_growth_summary(g_mcts_sig, figsize=(20, 4), show: bool = False):
    if len(g_mcts_sig) == 0:
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "no MCTS signature history yet", ha="center", va="center")
        ax.set_axis_off()
        return maybe_show(fig, show)

    sig_arr = np.stack(g_mcts_sig, axis=0)

    fig, axs = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    axs[0].plot(sig_arr[:, 0], label="node states")
    axs[0].plot(sig_arr[:, 1], label="edge states")
    axs[0].set_title("MCTS x/tf state growth")
    axs[0].legend()

    axs[1].plot(sig_arr[:, 2], label="node exploit count")
    axs[1].plot(sig_arr[:, 3], label="edge exploit count")
    axs[1].plot(sig_arr[:, 4], label="node explore count")
    axs[1].plot(sig_arr[:, 5], label="edge explore count")
    axs[1].set_title("MCTS x/tf counts")
    axs[1].legend()

    axs[2].plot(sig_arr[:, 10], label="alpha decision states")
    axs[2].plot(sig_arr[:, 11], label="alpha node states")
    axs[2].plot(sig_arr[:, 12], label="alpha edge states")
    axs[2].set_title("alpha-MCTS state growth")
    axs[2].legend()

    axs[3].plot(sig_arr[:, 13], label="alpha decision count")
    axs[3].plot(sig_arr[:, 14], label="alpha node count")
    axs[3].plot(sig_arr[:, 15], label="alpha edge count")
    axs[3].plot(sig_arr[:, 16], label="alpha decision explore")
    axs[3].set_title("alpha-MCTS counts")
    axs[3].legend()

    return maybe_show(fig, show)


def plot_ev_zscore_violin(EVs, figsize=(10, 4), show: bool = False):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    clean = []
    for a in EVs:
        arr = np.asarray(a, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            clean.append(arr)

    if len(clean) == 0:
        ax.text(0.5, 0.5, "no finite EV/z-score values yet", ha="center", va="center")
        ax.set_axis_off()
        return maybe_show(fig, show)

    ax.violinplot(clean, widths=1, bw_method=0.25, showmeans=True)
    ax.hlines([-2, 0, 2], xmin=0, xmax=len(clean), alpha=0.25, colors="black")
    ax.hlines([0], xmin=0, xmax=len(clean), alpha=0.50, colors="black")
    ax.set_title("EV / z-score history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("z-score / EV")

    return maybe_show(fig, show)


def plot_depth_probability_progress(
    prob_hist,
    title_left,
    title_right,
    figsize=(14, 4),
    show: bool = False,
):
    P = stack_depth_vectors(prob_hist)

    P = P[:,:-2]

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if P.shape[0] == 0:
        axs[0].text(0.5, 0.5, "no probability history yet", ha="center", va="center")
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        return maybe_show(fig, show)

    im0 = axs[0].imshow(P.T, aspect="auto", origin="lower", cmap='Greens')
    axs[0].set_title(title_left)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("depth")
    fig.colorbar(im0, ax=axs[0])

    if P.shape[0] > 1:
        D = np.abs(np.diff(P, axis=0))

        im1 = axs[1].imshow(D.T, aspect="auto", origin="lower", cmap='Greens')
        axs[1].set_title(title_right)
        axs[1].set_xlabel("iteration delta")
        axs[1].set_ylabel("depth")
        fig.colorbar(im1, ax=axs[1])
    else:
        axs[1].text(0.5, 0.5, "need >1 iteration", ha="center", va="center")
        axs[1].set_axis_off()

    return maybe_show(fig, show)


def plot_mcts_depth_progress(child_depth_probs_hist, figsize=(14, 4), show: bool = False):
    return plot_depth_probability_progress(
        prob_hist=child_depth_probs_hist,
        title_left="p(new node depth)",
        title_right="|delta p(new node depth)|",
        figsize=figsize,
        show=show,
    )


def plot_mcts_alpha_parent_depth_progress(alpha_parent_depth_probs_hist, figsize=(14, 4), show: bool = False):
    return plot_depth_probability_progress(
        prob_hist=alpha_parent_depth_probs_hist,
        title_left="p(alpha parent depth | alpha sensor)",
        title_right="|delta p(alpha parent depth)|",
        figsize=figsize,
        show=show,
    )


def plot_mcts_depth_terms(depth_rows, alpha_depth_rows=None, figsize=(16, 4), show: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    if depth_rows is None or len(depth_rows) == 0:
        for ax in axs:
            ax.text(0.5, 0.5, "no depth rows yet", ha="center", va="center")
            ax.set_axis_off()
        return maybe_show(fig, show)

    d = np.asarray([r["depth"] for r in depth_rows], dtype=int)

    node_q = np.asarray([r["node_q_mean"] for r in depth_rows], dtype=float)
    node_u = np.asarray([r["node_expandable_u_mean"] for r in depth_rows], dtype=float)

    edge_q = np.asarray([r["edge_q_mean"] for r in depth_rows], dtype=float)
    edge_u = np.asarray([r["edge_u_mean"] for r in depth_rows], dtype=float)

    axs[0].plot(d, node_q, marker="o", label="x-node exploit")
    axs[0].plot(d, node_u, marker="o", label="x-node explore")
    axs[0].set_title("x-parent node UCT terms by depth")
    axs[0].set_xlabel("depth")
    axs[0].legend()

    axs[1].plot(d, edge_q, marker="o", label="x-edge exploit")
    axs[1].plot(d, edge_u, marker="o", label="x-edge explore")
    axs[1].set_title("child function edge UCB terms by depth")
    axs[1].set_xlabel("depth")
    axs[1].legend()

    if alpha_depth_rows is not None and len(alpha_depth_rows) > 0:
        ad = np.asarray([r["depth"] for r in alpha_depth_rows], dtype=int)
        a_q = np.asarray([r["alpha_node_q_mean"] for r in alpha_depth_rows], dtype=float)
        a_u = np.asarray([r["alpha_node_u_mean"] for r in alpha_depth_rows], dtype=float)

        axs[2].plot(ad, a_q, marker="o", label="alpha-node exploit")
        axs[2].plot(ad, a_u, marker="o", label="alpha-node explore")
        axs[2].set_title("alpha-parent node UCT terms by depth")
        axs[2].set_xlabel("depth")
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, "no alpha-node rows yet", ha="center", va="center")
        axs[2].set_title("alpha-parent node UCT terms by depth")
        axs[2].set_axis_off()

    return maybe_show(fig, show)


def plot_mcts_alpha_sensor_probability(alpha_sensor_prob_hist, figsize=(10, 4), show: bool = False):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if alpha_sensor_prob_hist is None or len(alpha_sensor_prob_hist) == 0:
        ax.text(0.5, 0.5, "no alpha sensor probability history yet", ha="center", va="center")
        ax.set_axis_off()
        return maybe_show(fig, show)

    y = np.asarray(alpha_sensor_prob_hist, dtype=np.float64)
    x = np.arange(y.shape[0])

    ax.plot(x, y, marker="o")
    ax.set_title("mean p(alpha sensor)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("probability")

    finite = y[np.isfinite(y)]
    if finite.size:
        ax.set_ylim(
            max(0.0, float(np.nanmin(finite)) - 0.05),
            min(1.0, float(np.nanmax(finite)) + 0.05),
        )
    else:
        ax.set_ylim(0, 1)

    ax.grid(alpha=0.25)

    return maybe_show(fig, show)


def plot_mcts_exploration_influence(
    influence_hist,
    k=0.01,
    figsize=(12, 4),
    freeze_threshold_hist=None,
    unfreeze_threshold_hist=None,
    pw_scale_hist=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    y = np.asarray(influence_hist, dtype=np.float64)

    if y.size == 0:
        ax.text(0.5, 0.5, "no UCT influence history yet", ha="center", va="center")
        ax.set_axis_off()
        return maybe_show(fig, show)

    x = np.arange(y.shape[0])

    ax.plot(x, y, marker="o", label="UCT explore influence")

    hp = np.zeros(y.shape)
    hp[0] = y[0]

    for i in range(1, y.shape[0]):
        hp[i] = hp[i - 1] * (1 - np.e ** -k) + y[i] * (np.e ** -k)

    ax.plot(x, hp, linewidth=2, label="EMA/Hawkes")

    if freeze_threshold_hist is not None and len(freeze_threshold_hist) > 0:
        ft = np.asarray(freeze_threshold_hist, dtype=np.float64)
        ax.plot(np.arange(ft.shape[0]), ft, linestyle="--", alpha=0.65, label="freeze threshold")
    else:
        ax.axhline(0.55, linestyle="--", alpha=0.5, label="freeze threshold")

    if unfreeze_threshold_hist is not None and len(unfreeze_threshold_hist) > 0:
        ut = np.asarray(unfreeze_threshold_hist, dtype=np.float64)
        ax.plot(np.arange(ut.shape[0]), ut, linestyle="--", alpha=0.65, label="unfreeze threshold")
    else:
        ax.axhline(0.25, linestyle="--", alpha=0.5, label="unfreeze threshold")

    ax.set_title("UCT exploration influence + PW mode")
    ax.set_xlabel("iteration")
    ax.set_ylabel("policy influence")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    if pw_scale_hist is not None and len(pw_scale_hist) > 0:
        #ax2 = ax.twinx()
        ps = np.asarray(pw_scale_hist, dtype=np.float64)
        ax.plot(np.arange(ps.shape[0]), ps, alpha=0.75, label="PW scale")

        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")

    return maybe_show(fig, show)


def plot_mcts_policy_drift_hawkes(
    hp,
    raw_hist=None,
    parent_hist=None,
    child_hist=None,
    support_penalty_hist=None,
    parent_new_mass_hist=None,
    child_new_edge_frac_hist=None,
    figsize=(8, 4),
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if raw_hist is not None and len(raw_hist) > 0:
        ax.plot(raw_hist, linewidth=2.5, label="total drift", color='black')

    if hp is not None and len(hp) > 0:
        ax.plot(hp, linewidth=3.0, label="smoothed h", color='maroon')

    if parent_hist is not None and len(parent_hist) > 0:
        ax.plot(parent_hist, alpha=0.30, linewidth=1.2, label="parent drift")

    if child_hist is not None and len(child_hist) > 0:
        ax.plot(child_hist, alpha=0.30, linewidth=1.2, label="child drift")

    # only include this if you are using the hybrid/common-support drift version
    if support_penalty_hist is not None and len(support_penalty_hist) > 0:
        ax.plot(
            support_penalty_hist,
            alpha=0.45,
            linewidth=1.4,
            linestyle="--",
            label="support penalty",
        )

    ax.axhline(0.01, linestyle="--", alpha=0.40, label="99% stable", color='blue')
    ax.axhline(0.1, linestyle="--", alpha=0.40, label="90% stable", color='blue')

    ax.set_title("exploit policy drift")
    ax.set_xlabel("iteration")
    ax.set_ylabel("policy drift")
    ax.set_ylim(1e-3, 1)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    return maybe_show(fig, show)

def plot_fpc_gene_evaluation_from_details(
    details,
    fpc_curve,
    perm_mu,
    min_var=1e-18,
    viz_kwargs=None,
    figsize=(8, 5),
    show: bool = False,
):
    """
    Recreate the visualize=True plot from evaluate_genes_from_opg_fpc_fast
    using returned details.
    """
    viz_kwargs = {} if viz_kwargs is None else dict(viz_kwargs)

    prop_g = np.asarray(details.get("prop_g", []), dtype=np.float64)
    observed_scores_g = np.asarray(details.get("observed_scores_g", []), dtype=np.float64)
    fpc_std_g = np.asarray(details.get("fpc_std_g", []), dtype=np.float64)
    score_label = details.get("score_label", "EV")

    fig, ax = plt.subplots(figsize=viz_kwargs.get("figsize", figsize), constrained_layout=True)

    keep = (
        np.isfinite(prop_g)
        & np.isfinite(observed_scores_g)
        & np.isfinite(fpc_std_g)
        & (prop_g > 0)
    )

    if not np.any(keep):
        ax.text(0.5, 0.5, "no finite FPC-evaluated gene scores available", ha="center", va="center")
        ax.set_axis_off()
        return maybe_show(fig, show)

    band_mult = float(viz_kwargs.get("band_mult", 2.0))
    point_size = float(viz_kwargs.get("s", 35))
    alpha = float(viz_kwargs.get("alpha", 0.8))
    grid_n = int(viz_kwargs.get("grid_n", 300))

    p_min = max(1e-6, float(np.nanmin(prop_g[keep])))
    p_max = min(0.999999, float(np.nanmax(prop_g[keep])))

    p_grid = np.linspace(p_min, p_max, grid_n)
    v_grid = np.asarray(fpc_curve(p_grid), dtype=np.float64)
    s_grid = np.sqrt(np.maximum(v_grid, min_var))

    ax.scatter(prop_g[keep], observed_scores_g[keep], s=point_size, alpha=alpha, label="gene scores")
    ax.scatter(p_grid, np.full_like(p_grid, float(perm_mu)), s=5, alpha=0.75, color="black", label="permutation mean")
    ax.scatter(p_grid, float(perm_mu) + band_mult * s_grid, s=5, alpha=0.5, color="gray", label=f"+{band_mult:g} std")
    ax.scatter(p_grid, float(perm_mu) - band_mult * s_grid, s=5, alpha=0.5, color="gray", label=f"-{band_mult:g} std")
    ax.set_xlabel("gene participation proportion p = m / N")
    ax.set_ylabel(score_label)
    ax.set_title("FPC-conditioned gene evaluation")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return maybe_show(fig, show)


def make_summary_dashboard(histories: dict, k: int, figsize=(17, 8), show: bool = False):
    fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    axs = axs.ravel()

    sig_hist = histories.get("g_mcts_sig", [])
    EVs = histories.get("EVs", [])

    if len(sig_hist) > 0:
        sig_arr = np.stack(sig_hist, axis=0)

        axs[0].plot(sig_arr[:, 0], label="node states")
        axs[0].plot(sig_arr[:, 1], label="edge states")
        axs[0].set_title("MCTS x/tf state growth")
        axs[0].legend(fontsize=8)

        axs[1].plot(sig_arr[:, 10], label="alpha decisions")
        axs[1].plot(sig_arr[:, 11], label="alpha nodes")
        axs[1].plot(sig_arr[:, 12], label="alpha edges")
        axs[1].set_title("alpha-MCTS state growth")
        axs[1].legend(fontsize=8)

    if len(EVs) > 0:
        means = []
        maxes = []

        for v in EVs:
            arr = np.asarray(v, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            means.append(np.nan if arr.size == 0 else float(np.mean(arr)))
            maxes.append(np.nan if arr.size == 0 else float(np.max(arr)))

        axs[2].plot(means, label="mean z")
        axs[2].plot(maxes, label="max z")
        axs[2].axhline(0, alpha=0.35)
        axs[2].axhline(2, alpha=0.25)
        axs[2].set_title("z-score history")
        axs[2].legend(fontsize=8)

    parent = histories.get("g_mcts_policy_parent_drift", [])
    child = histories.get("g_mcts_policy_child_drift", [])
    total = histories.get("g_mcts_policy_total_drift", [])
    hp = histories.get("hp", [])

    if len(total) > 0:
        axs[3].plot(parent, label="parent", color='gray')
        axs[3].plot(child, label="child", color='gray')
        axs[3].plot(total, label="total", color='black')
        if len(hp) > 0:
            axs[3].plot(hp, label="h", color='maroon')
        axs[3].set_title("root-weighted policy drift")
        axs[3].legend(fontsize=8)

    child_depth = histories.get("g_mcts_child_depth_probs", [])
    if len(child_depth) > 0:
        P = stack_depth_vectors(child_depth)
        P = P[:,:-2]

        im = axs[4].imshow(
            P.T,
            aspect="auto",
            origin="lower",
            cmap='Greens'
        )

        axs[4].set_title("p(new node depth)")
        axs[4].set_xlabel("iteration")
        axs[4].set_ylabel("depth")

        fig.colorbar(im, ax=axs[4])

    infl = histories.get("g_mcts_uct_explore_influence", [])
    infl_ema = histories.get("g_mcts_uct_explore_influence_ema", [])
    pw_scale = histories.get("g_mcts_pw_scale", [])
    freeze_thresh = histories.get("g_mcts_pw_freeze_threshold", [])
    unfreeze_thresh = histories.get("g_mcts_pw_unfreeze_threshold", [])

    if len(infl) > 0:
        x = np.arange(len(infl))

        axs[5].plot(x, infl, label="raw")
        axs[5].plot(x, infl_ema, label="ema")

        if len(freeze_thresh) == len(infl):
            axs[5].plot(x, freeze_thresh, linestyle="--", alpha=0.65, label="freeze threshold")

        if len(unfreeze_thresh) == len(infl):
            axs[5].plot(x, unfreeze_thresh, linestyle="--", alpha=0.65, label="unfreeze threshold")

        axs[5].set_title("UCT explore influence + PW scale")
        axs[5].set_xlabel("iteration")
        axs[5].set_ylabel("explore influence")
        axs[5].set_ylim(0, 1)
        axs[5].grid(alpha=0.25)

        if len(pw_scale) > 0:
            ax_pw = axs[5].twinx()
            x_pw = np.arange(len(pw_scale))
            ax_pw.plot(x_pw, pw_scale, alpha=0.85, label="PW scale")
            ax_pw.set_ylabel("PW scale")
            ax_pw.set_ylim(0, 1.05)

            lines1, labels1 = axs[5].get_legend_handles_labels()
            lines2, labels2 = ax_pw.get_legend_handles_labels()
            axs[5].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        else:
            axs[5].legend(fontsize=8)

    h_val = histories.get("h", np.nan)
    freeze = histories.get("freeze_expansion", None)

    fig.suptitle(f"iteration={k} | h={h_val} | freeze={freeze}", fontsize=11)

    return maybe_show(fig, show)

def decay_dict_values(d, gamma):
    if d is None:
        return
    for k in list(d.keys()):
        d[k] = d[k] * gamma


def recompute_mu_from_cum_count(cum_dict, count_dict, mu_dict, min_eff_count=1e-9):
    """
    recompute mean dictionaries after decay.
    remove entries whose effective count is basically zero.
    """

    if cum_dict is None or count_dict is None or mu_dict is None:
        return

    for k in list(mu_dict.keys()):
        c = count_dict.get(k, 0.0)

        if c <= min_eff_count:
            cum_dict.pop(k, None)
            count_dict.pop(k, None)
            mu_dict.pop(k, None)
        else:
            mu_dict[k] = cum_dict.get(k, 0.0) / c


def decay_mcts_effective_memory(
    G,
    exploit_gamma=0.98,
    explore_gamma=0.995,
    min_eff_count=1e-6,
):
    """
    Natural walk-forward memory decay.

    Keeps discovered structure, but decays the effective evidence behind it.

    exploit_gamma:
        decays reward/cumulative score memory.

    explore_gamma:
        decays visit/exploration counts more slowly.

    min_eff_count:
        if effective count falls below this, remove that score entry.
    """

    exploit_gamma = float(exploit_gamma)
    explore_gamma = float(explore_gamma)

    # ----------------------------
    # normal x/function MCTS
    # ----------------------------

    for name in [
        "_MCTS_NODE_CUM",
        "_MCTS_EDGE_CUM",
        "_MCTS_NODE_COUNT",
        "_MCTS_EDGE_COUNT",
    ]:
        decay_dict_values(getattr(G, name, None), exploit_gamma)

    for name in [
        "_MCTS_NODE_EXPLORE_COUNT",
        "_MCTS_EDGE_EXPLORE_COUNT",
    ]:
        decay_dict_values(getattr(G, name, None), explore_gamma)

    recompute_mu_from_cum_count(
        getattr(G, "_MCTS_NODE_CUM", None),
        getattr(G, "_MCTS_NODE_COUNT", None),
        getattr(G, "_MCTS_NODE_MU", None),
        min_eff_count=min_eff_count,
    )

    recompute_mu_from_cum_count(
        getattr(G, "_MCTS_EDGE_CUM", None),
        getattr(G, "_MCTS_EDGE_COUNT", None),
        getattr(G, "_MCTS_EDGE_MU", None),
        min_eff_count=min_eff_count,
    )

    if hasattr(G, "_MCTS_EXPLOIT_T"):
        G._MCTS_EXPLOIT_T *= exploit_gamma

    if hasattr(G, "_MCTS_EXPLORE_T"):
        G._MCTS_EXPLORE_T *= explore_gamma

    # ----------------------------
    # alpha-MCTS
    # ----------------------------

    for name in [
        "_MCTS_ALPHA_DECISION_CUM",
        "_MCTS_ALPHA_NODE_CUM",
        "_MCTS_ALPHA_EDGE_CUM",
        "_MCTS_ALPHA_DECISION_COUNT",
        "_MCTS_ALPHA_NODE_COUNT",
        "_MCTS_ALPHA_EDGE_COUNT",
    ]:
        decay_dict_values(getattr(G, name, None), exploit_gamma)

    for name in [
        "_MCTS_ALPHA_DECISION_EXPLORE_COUNT",
        "_MCTS_ALPHA_NODE_EXPLORE_COUNT",
        "_MCTS_ALPHA_EDGE_EXPLORE_COUNT",
    ]:
        decay_dict_values(getattr(G, name, None), explore_gamma)

    recompute_mu_from_cum_count(
        getattr(G, "_MCTS_ALPHA_DECISION_CUM", None),
        getattr(G, "_MCTS_ALPHA_DECISION_COUNT", None),
        getattr(G, "_MCTS_ALPHA_DECISION_MU", None),
        min_eff_count=min_eff_count,
    )

    recompute_mu_from_cum_count(
        getattr(G, "_MCTS_ALPHA_NODE_CUM", None),
        getattr(G, "_MCTS_ALPHA_NODE_COUNT", None),
        getattr(G, "_MCTS_ALPHA_NODE_MU", None),
        min_eff_count=min_eff_count,
    )

    recompute_mu_from_cum_count(
        getattr(G, "_MCTS_ALPHA_EDGE_CUM", None),
        getattr(G, "_MCTS_ALPHA_EDGE_COUNT", None),
        getattr(G, "_MCTS_ALPHA_EDGE_MU", None),
        min_eff_count=min_eff_count,
    )

    if hasattr(G, "_MCTS_ALPHA_EXPLOIT_T"):
        G._MCTS_ALPHA_EXPLOIT_T *= exploit_gamma

    if hasattr(G, "_MCTS_ALPHA_EXPLORE_T"):
        G._MCTS_ALPHA_EXPLORE_T *= explore_gamma

    return G

def save_standard_iteration_plots(
    store: MCTSRunStore,
    k: int,
    histories: dict,
    depth_rows=None,
    alpha_depth_rows=None,
):
    """
    Save all recreated plot types that used to appear in the notebook loop.
    """
    paths = {}

    fig = plot_mcts_state_growth_summary(histories["g_mcts_sig"], show=False)
    paths["state_growth"] = store.save_fig(fig, k, "state_growth")

    fig = plot_ev_zscore_violin(histories["EVs"], show=False)
    paths["ev_violin"] = store.save_fig(fig, k, "ev_violin")

    fig = plot_mcts_depth_terms(depth_rows, alpha_depth_rows, show=False)
    paths["depth_terms"] = store.save_fig(fig, k, "depth_terms")

    fig = plot_mcts_depth_progress(histories["g_mcts_child_depth_probs"], show=False)
    paths["depth_progress"] = store.save_fig(fig, k, "depth_progress")

    fig = plot_mcts_alpha_parent_depth_progress(histories["g_mcts_alpha_parent_depth_probs"], show=False)
    paths["alpha_parent_depth_progress"] = store.save_fig(fig, k, "alpha_parent_depth_progress")

    fig = plot_mcts_alpha_sensor_probability(histories["g_mcts_alpha_sensor_probs"], show=False)
    paths["alpha_sensor_probability"] = store.save_fig(fig, k, "alpha_sensor_probability")

    fig = plot_mcts_exploration_influence(
        histories["g_mcts_uct_explore_influence"],
        k=-np.log(0.314),
        freeze_threshold_hist=histories.get("g_mcts_pw_freeze_threshold"),
        unfreeze_threshold_hist=histories.get("g_mcts_pw_unfreeze_threshold"),
        pw_scale_hist=histories.get("g_mcts_pw_scale"),
        show=False,
    )
    paths["exploration_influence"] = store.save_fig(fig, k, "exploration_influence")

    fig = plot_mcts_policy_drift_hawkes(
        hp=histories["hp"],
        raw_hist=histories["g_mcts_policy_total_drift"],
        parent_hist=histories["g_mcts_policy_parent_drift"],
        child_hist=histories["g_mcts_policy_child_drift"],
        show=False,
    )
    paths["policy_drift"] = store.save_fig(fig, k, "policy_drift")

    fig = make_summary_dashboard(histories, k=k, show=False)
    paths["summary_dashboard"] = store.save_fig(fig, k, "summary_dashboard")

    return paths

def init_pw_walk_state(
    G,
    *,
    base_pw_c=None,
    base_pw_alpha=None,
    base_expand_prob=None,
    start_scale=1.0,
):
    """
    Initialize baseline progressive widening values.

    scale=1.0 means original widening.
    scale near 0.0 means almost frozen.
    """
    if not hasattr(G, "_MCTS_PW_BASE_C"):
        G._MCTS_PW_BASE_C = float(
            getattr(G, "_MCTS_PW_C", 0.75) if base_pw_c is None else base_pw_c
        )

    if not hasattr(G, "_MCTS_PW_BASE_ALPHA"):
        G._MCTS_PW_BASE_ALPHA = float(
            getattr(G, "_MCTS_PW_ALPHA", 0.30) if base_pw_alpha is None else base_pw_alpha
        )

    if not hasattr(G, "_MCTS_EXPAND_PROB_BASE"):
        G._MCTS_EXPAND_PROB_BASE = float(
            getattr(G, "_MCTS_EXPAND_PROB", 0.10) if base_expand_prob is None else base_expand_prob
        )

    if not hasattr(G, "_MCTS_PW_SCALE"):
        G._MCTS_PW_SCALE = float(start_scale)

    return {
        "base_pw_c": G._MCTS_PW_BASE_C,
        "base_pw_alpha": G._MCTS_PW_BASE_ALPHA,
        "base_expand_prob": G._MCTS_EXPAND_PROB_BASE,
        "scale": G._MCTS_PW_SCALE,
    }

def _dict_value_sum(d):
    if d is None or len(d) == 0:
        return 0.0
    vals = []
    for v in d.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    return float(np.sum(vals)) if len(vals) else 0.0


def _wf_debug_grammar_state_values(G, label, store=None):
    children = getattr(G, "_MCTS_CHILDREN", {})
    n_children_edges = sum(len(v) for v in children.values()) if isinstance(children, dict) else 0

    msg = (
        f"{label} | "
        f"node_mu_len={len(getattr(G, '_MCTS_NODE_MU', {}))} "
        f"node_mu_sum={_dict_value_sum(getattr(G, '_MCTS_NODE_MU', {})):.6g} | "
        f"edge_mu_len={len(getattr(G, '_MCTS_EDGE_MU', {}))} "
        f"edge_mu_sum={_dict_value_sum(getattr(G, '_MCTS_EDGE_MU', {})):.6g} | "
        f"node_count_len={len(getattr(G, '_MCTS_NODE_COUNT', {}))} "
        f"node_count_sum={_dict_value_sum(getattr(G, '_MCTS_NODE_COUNT', {})):.6g} | "
        f"edge_count_len={len(getattr(G, '_MCTS_EDGE_COUNT', {}))} "
        f"edge_count_sum={_dict_value_sum(getattr(G, '_MCTS_EDGE_COUNT', {})):.6g} | "
        f"node_explore_sum={_dict_value_sum(getattr(G, '_MCTS_NODE_EXPLORE_COUNT', {})):.6g} | "
        f"edge_explore_sum={_dict_value_sum(getattr(G, '_MCTS_EDGE_EXPLORE_COUNT', {})):.6g} | "
        f"children_parents={len(children)} | "
        f"children_edges={n_children_edges} | "
        f"exploit_t={getattr(G, '_MCTS_EXPLOIT_T', None)} | "
        f"explore_t={getattr(G, '_MCTS_EXPLORE_T', None)}"
    )

    print(msg)
    if store is not None:
        store.log(msg)

def apply_pw_scale_to_grammar(G, scale, *, min_scale=0.02, max_scale=1.0):
    """
    Apply continuous PW scale to the actual grammar attributes.

    This changes:
        G._MCTS_PW_C
        G._MCTS_EXPAND_PROB

    It leaves alpha/exponent unchanged by default.
    """
    init_pw_walk_state(G)

    scale = float(np.clip(scale, min_scale, max_scale))

    G._MCTS_PW_SCALE = scale
    G._MCTS_PW_C = float(G._MCTS_PW_BASE_C * scale)
    G._MCTS_PW_ALPHA = float(G._MCTS_PW_BASE_ALPHA)
    G._MCTS_EXPAND_PROB = float(G._MCTS_EXPAND_PROB_BASE * scale)

    # keep possible backing dictionaries synchronized
    for attr in ("_spec_gram_args", "spec_gram_args", "_MCTS_SPEC_GRAM_ARGS", "_mcts_spec_gram_args"):
        d = getattr(G, attr, None)
        if isinstance(d, dict):
            d["pw_c"] = G._MCTS_PW_C
            d["pw_alpha"] = G._MCTS_PW_ALPHA
            d["expand_prob"] = G._MCTS_EXPAND_PROB

    return {
        "pw_scale": float(G._MCTS_PW_SCALE),
        "pw_c": float(G._MCTS_PW_C),
        "pw_alpha": float(G._MCTS_PW_ALPHA),
        "expand_prob": float(G._MCTS_EXPAND_PROB),
    }

def normalize_policy_dict(d, eps=1e-12):
    s = float(sum(d.values()))
    if s <= eps or not np.isfinite(s):
        return {}
    return {k: float(v) / s for k, v in d.items()}


def tv_distance_dict(a, b):
    keys = set(a.keys()) | set(b.keys())
    if len(keys) == 0:
        return np.nan
    return 0.5 * float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def policy_new_removed_mass(prev_policy, curr_policy):
    prev_keys = set(prev_policy.keys())
    curr_keys = set(curr_policy.keys())

    new_keys = curr_keys - prev_keys
    removed_keys = prev_keys - curr_keys

    new_mass = float(sum(curr_policy.get(k, 0.0) for k in new_keys))
    removed_mass = float(sum(prev_policy.get(k, 0.0) for k in removed_keys))

    return new_mass, removed_mass, len(new_keys), len(removed_keys)


def tunable_common_support_exploit_policy_drift(
    prev_snap,
    curr_snap,
    *,
    parent_child_mix=0.50,
    novelty_weight=0.25,
    removed_weight=0.10,
    min_common_mass=1e-6,
    child_weight_mode="parent_weight",
    include_new_child_penalty=True,
):
    """
    Tunable exploit-only drift.

    parent_child_mix:
        0.0 = only child drift
        1.0 = only parent drift
        0.5 = equal parent/child blend

    novelty_weight:
        how much new support should count against convergence.
        higher = broader search expansion keeps drift higher.

    removed_weight:
        how much removed support should count against convergence.

    min_common_mass:
        if common support mass is too tiny, skip/NaN instead of pretending stable.

    child_weight_mode:
        "parent_weight" = weight child drift by current parent weights
        "uniform"       = average child drift across parents equally

    include_new_child_penalty:
        if True, child novelty contributes to total novelty penalty.
    """

    if prev_snap is None:
        return {
            "parent_drift": np.nan,
            "child_drift": np.nan,
            "support_penalty": np.nan,
            "total_drift": np.nan,
            "parent_new_mass": np.nan,
            "child_new_edge_frac": np.nan,
        }

    prev_parent = prev_snap["parent_policy"]
    curr_parent = curr_snap["parent_policy"]

    prev_parent_keys = set(prev_parent.keys())
    curr_parent_keys = set(curr_parent.keys())
    common_parents = prev_parent_keys & curr_parent_keys

    parent_new_mass, parent_removed_mass, parent_new_n, parent_removed_n = policy_new_removed_mass(
        prev_parent,
        curr_parent,
    )

    prev_common_mass = float(sum(prev_parent.get(k, 0.0) for k in common_parents))
    curr_common_mass = float(sum(curr_parent.get(k, 0.0) for k in common_parents))

    if prev_common_mass < min_common_mass or curr_common_mass < min_common_mass:
        parent_drift = np.nan
    else:
        prev_parent_common = normalize_policy_dict({
            k: prev_parent[k] for k in common_parents
        })

        curr_parent_common = normalize_policy_dict({
            k: curr_parent[k] for k in common_parents
        })

        parent_drift = tv_distance_dict(prev_parent_common, curr_parent_common)

    child_vals = []
    child_weights = []

    child_new_edges = 0
    child_removed_edges = 0
    child_total_edges = 0

    for parent_key in common_parents:
        prev_child = prev_snap["child_policies"].get(parent_key, {})
        curr_child = curr_snap["child_policies"].get(parent_key, {})

        prev_child_keys = set(prev_child.keys())
        curr_child_keys = set(curr_child.keys())
        common_children = prev_child_keys & curr_child_keys

        child_new_edges += len(curr_child_keys - prev_child_keys)
        child_removed_edges += len(prev_child_keys - curr_child_keys)
        child_total_edges += len(prev_child_keys | curr_child_keys)

        if len(common_children) == 0:
            continue

        prev_child_common = normalize_policy_dict({
            k: prev_child[k] for k in common_children
        })

        curr_child_common = normalize_policy_dict({
            k: curr_child[k] for k in common_children
        })

        tv = tv_distance_dict(prev_child_common, curr_child_common)

        if not np.isfinite(tv):
            continue

        if child_weight_mode == "parent_weight":
            w = curr_snap["parent_weights"].get(
                parent_key,
                prev_snap["parent_weights"].get(parent_key, 0.0),
            )
        elif child_weight_mode == "uniform":
            w = 1.0
        else:
            raise ValueError("child_weight_mode must be 'parent_weight' or 'uniform'")

        child_vals.append(tv)
        child_weights.append(w)

    if len(child_vals) == 0:
        child_drift = np.nan
    else:
        child_vals = np.asarray(child_vals, dtype=np.float64)
        child_weights = np.asarray(child_weights, dtype=np.float64)

        if child_weights.sum() <= 0:
            child_drift = float(np.mean(child_vals))
        else:
            child_drift = float(np.sum(child_vals * child_weights) / child_weights.sum())

    child_new_edge_frac = (
        float(child_new_edges) / float(child_total_edges)
        if child_total_edges > 0
        else 0.0
    )

    support_penalty = 0.0
    support_penalty += novelty_weight * parent_new_mass
    support_penalty += removed_weight * parent_removed_mass

    if include_new_child_penalty:
        support_penalty += novelty_weight * child_new_edge_frac

    vals = []

    if np.isfinite(parent_drift):
        vals.append(("parent", parent_drift))

    if np.isfinite(child_drift):
        vals.append(("child", child_drift))

    if len(vals) == 0:
        base_drift = np.nan
    elif len(vals) == 1:
        base_drift = vals[0][1]
    else:
        base_drift = (
            parent_child_mix * parent_drift
            + (1.0 - parent_child_mix) * child_drift
        )

    if np.isfinite(base_drift):
        total_drift = float(base_drift + support_penalty)
    else:
        total_drift = np.nan

    return {
        "parent_drift": parent_drift,
        "child_drift": child_drift,
        "support_penalty": support_penalty,
        "total_drift": total_drift,

        "parent_new_mass": parent_new_mass,
        "parent_removed_mass": parent_removed_mass,
        "parent_new_n": parent_new_n,
        "parent_removed_n": parent_removed_n,

        "child_new_edges": child_new_edges,
        "child_removed_edges": child_removed_edges,
        "child_total_edges": child_total_edges,
        "child_new_edge_frac": child_new_edge_frac,
    }


def common_support_exploit_policy_drift(prev_snap, curr_snap):
    if prev_snap is None:
        return {
            "parent_drift": np.nan,
            "child_drift": np.nan,
            "total_drift": np.nan,
        }

    prev_parent = prev_snap["parent_policy"]
    curr_parent = curr_snap["parent_policy"]

    common_parents = set(prev_parent.keys()) & set(curr_parent.keys())

    prev_parent_common = normalize_policy_dict({
        k: prev_parent[k] for k in common_parents
    })

    curr_parent_common = normalize_policy_dict({
        k: curr_parent[k] for k in common_parents
    })

    parent_drift = tv_distance_dict(prev_parent_common, curr_parent_common)

    child_vals = []
    child_weights = []

    for parent_key in common_parents:
        prev_child = prev_snap["child_policies"].get(parent_key, {})
        curr_child = curr_snap["child_policies"].get(parent_key, {})

        common_children = set(prev_child.keys()) & set(curr_child.keys())

        if len(common_children) == 0:
            continue

        prev_child_common = normalize_policy_dict({
            k: prev_child[k] for k in common_children
        })

        curr_child_common = normalize_policy_dict({
            k: curr_child[k] for k in common_children
        })

        tv = tv_distance_dict(prev_child_common, curr_child_common)

        if not np.isfinite(tv):
            continue

        w = curr_snap["parent_weights"].get(
            parent_key,
            prev_snap["parent_weights"].get(parent_key, 0.0),
        )

        child_vals.append(tv)
        child_weights.append(w)

    if len(child_vals) == 0:
        child_drift = np.nan
    else:
        child_vals = np.asarray(child_vals, dtype=np.float64)
        child_weights = np.asarray(child_weights, dtype=np.float64)

        if child_weights.sum() <= 0:
            child_drift = float(np.mean(child_vals))
        else:
            child_drift = float(np.sum(child_vals * child_weights) / child_weights.sum())

    vals = np.asarray([parent_drift, child_drift], dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    total_drift = np.nan if vals.size == 0 else float(np.mean(vals))

    return {
        "parent_drift": parent_drift,
        "child_drift": child_drift,
        "total_drift": total_drift,
    }

def walk_pw_scale_from_thresholds(
    G,
    *,
    ema,
    freeze_threshold,
    unfreeze_threshold,
    down_step=0.05,
    up_step=0.05,
    min_scale=0.02,
    max_scale=1.0,
):
    """
    Walk PW scale down/up when thresholds are crossed.

    if ema > freeze_threshold:
        scale walks lower

    if ema < unfreeze_threshold:
        scale walks higher

    otherwise:
        scale holds steady
    """
    init_pw_walk_state(G)

    old_scale = float(getattr(G, "_MCTS_PW_SCALE", 1.0))

    if not np.isfinite(ema):
        action = "hold_nan"
        new_scale = old_scale

    elif ema > freeze_threshold:
        action = "walk_down"
        new_scale = old_scale - float(down_step)

    elif ema < unfreeze_threshold:
        action = "walk_up"
        new_scale = old_scale + float(up_step)

    else:
        action = "hold"
        new_scale = old_scale

    pw_info = apply_pw_scale_to_grammar(
        G,
        new_scale,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    pw_info.update({
        "pw_action": action,
        "pw_old_scale": old_scale,
        "pw_new_scale": float(getattr(G, "_MCTS_PW_SCALE", new_scale)),
        "pw_freeze_threshold": float(freeze_threshold),
        "pw_unfreeze_threshold": float(unfreeze_threshold),
        "pw_ema": float(ema) if np.isfinite(ema) else np.nan,
    })

    return pw_info


# ---------------------------------------------------------------------
# default kwargs
# ---------------------------------------------------------------------

def default_initialization_kwargs(G):
    return {
        "structure"   : "Intraday",
        "incl_time"   : True,
        "data_file"   : "../data/spy5m.csv",
        "epoch_idx"   : [0],
        "hlocv_idx"   : [1, 2, 3, 4],
        "pop_size"    : 100,

        "grmr_prior"  : G,
        "grmr_type"   : "MCTS",
        "grmr_mdl"    : 80,
        "grmr_p_mttn" : 0.00,
        "grmr_p_csvr" : 0.00,
        "grmr_a_sens" : G._alpha_sensor_freq,

        "chunk_size"  : 1,

        "wf_windows"  : 20,
        "verbose"     : 0,
    }


def default_solver_kwargs():
    return {
        "offset"   : 6,
        "t_vec"    : "Close",
        "t_mode"   : "RR",
        "emission" : [],
        "AD_cond"  : ("gt", 0),
        "ATR_window": 20,
        "ATR_min_count": 1,
        "ATR_coef" : 2,
        "RR_max_rng": 30,
        "RRR"      : 1
    }


def default_logwalker_kwargs():
    return {
        "start": 1,
        "destination": 3,
        "steps": 24,
        "exhaust_mode": "steps",
        "exwhen": 100,
        "min_walk": 3,
        "log_base": 2,
        "nest_mode": "half",
    }


def _wf_solver_signature_light(solver_kwargs):
    """
    compact view of the solver target.
    use this to verify training eval and inference eval are identical.
    """

    emission = solver_kwargs.get("emission", None)

    if isinstance(emission, list):
        emission_sig = []
        for op in emission:
            if isinstance(op, dict):
                emission_sig.append({
                    "ID": op.get("ID", None),
                    "x": op.get("x", None),
                    "alpha": op.get("alpha", None),
                    "delta1": op.get("delta1", None),
                    "delta2": op.get("delta2", None),
                    "offset": op.get("offset", None),
                })
            else:
                emission_sig.append(repr(op))
    else:
        emission_sig = repr(emission)

    return {
        "offset": solver_kwargs.get("offset", None),
        "t_vec": solver_kwargs.get("t_vec", None),
        "t_mode": solver_kwargs.get("t_mode", None),
        "AD_cond": solver_kwargs.get("AD_cond", None),
        "emission_len": len(emission) if isinstance(emission, list) else None,
        "emission": emission_sig,
    }


def _wf_debug_grammar_state(G, label, store=None):
    """
    log core grammar identity and memory sizes.
    """

    children = getattr(G, "_MCTS_CHILDREN", {})
    n_children_edges = sum(len(v) for v in children.values()) if isinstance(children, dict) else 0

    msg = (
        f"{label} | "
        f"G_id={id(G)} | "
        f"mode={getattr(G, '_mode', None)} | "
        f"type={getattr(G, '_type', None)} | "
        f"node_mu={len(getattr(G, '_MCTS_NODE_MU', {}))} | "
        f"edge_mu={len(getattr(G, '_MCTS_EDGE_MU', {}))} | "
        f"children_parents={len(children)} | "
        f"children_edges={n_children_edges} | "
        f"node_count={len(getattr(G, '_MCTS_NODE_COUNT', {}))} | "
        f"edge_count={len(getattr(G, '_MCTS_EDGE_COUNT', {}))} | "
        f"alpha_decision_mu={len(getattr(G, '_MCTS_ALPHA_DECISION_MU', {}))} | "
        f"alpha_node_mu={len(getattr(G, '_MCTS_ALPHA_NODE_MU', {}))} | "
        f"alpha_edge_mu={len(getattr(G, '_MCTS_ALPHA_EDGE_MU', {}))} | "
        f"exploit_t={getattr(G, '_MCTS_EXPLOIT_T', None)} | "
        f"explore_t={getattr(G, '_MCTS_EXPLORE_T', None)}"
    )

    print(msg)
    if store is not None:
        store.log(msg)

    return msg


def _wf_debug_population_generated_from_G(G, X, s_idx, label, store=None, max_rows=500):
    """
    check whether generated rows look connected to the learned grammar.

    This is not a perfect correctness proof, but if known_key_rate is near 0
    after training, inference is probably not using the learned MCTS memory.
    """

    s_idx = np.asarray(s_idx, dtype=np.int64)
    s_idx = s_idx[(s_idx >= 0) & (s_idx < X._instructions.shape[0])]

    if s_idx.size == 0:
        msg = f"{label} | no valid s_idx"
        print(msg)
        if store is not None:
            store.log(msg)
        return {}

    sample_idx = s_idx[:min(int(max_rows), s_idx.size)]

    func_ids = X._instructions[s_idx, 1].astype(np.int64)
    uniq_tf, cnt_tf = np.unique(func_ids, return_counts=True)

    top_tf_order = np.argsort(cnt_tf)[::-1][:10]
    top_tf = [(int(uniq_tf[i]), int(cnt_tf[i])) for i in top_tf_order]

    node_mu = getattr(G, "_MCTS_NODE_MU", {})
    edge_mu = getattr(G, "_MCTS_EDGE_MU", {})
    children = getattr(G, "_MCTS_CHILDREN", {})

    known_node_hits = 0
    key_errors = 0
    depth_vals = []

    for idx in sample_idx:
        try:
            key = G._mcts_state_key(X._instructions, int(idx))
            if key in node_mu:
                known_node_hits += 1

            if hasattr(G, "_mcts_row_depth"):
                depth_vals.append(int(G._mcts_row_depth(X._instructions, int(idx))))
        except Exception:
            key_errors += 1

    known_key_rate = known_node_hits / max(1, sample_idx.size)

    msg = (
        f"{label} | "
        f"n_s_idx={s_idx.size} | "
        f"sample_n={sample_idx.size} | "
        f"known_node_key_rate={known_key_rate:.4f} | "
        f"key_errors={key_errors} | "
        f"top_tf={top_tf} | "
        f"depth_min={min(depth_vals) if depth_vals else None} | "
        f"depth_max={max(depth_vals) if depth_vals else None} | "
        f"G_node_mu={len(node_mu)} | "
        f"G_edge_mu={len(edge_mu)} | "
        f"G_children_edges={sum(len(v) for v in children.values()) if isinstance(children, dict) else 0}"
    )

    print(msg)
    if store is not None:
        store.log(msg)

    return {
        "known_node_key_rate": known_key_rate,
        "key_errors": key_errors,
        "top_tf": top_tf,
        "depth_min": min(depth_vals) if depth_vals else None,
        "depth_max": max(depth_vals) if depth_vals else None,
    }


def _wf_debug_zscores(z, idx, label, store=None):
    """
    log finite zscore stats for a candidate index set.
    """

    z = np.asarray(z, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < z.shape[0])]

    vals = z[idx] if idx.size else np.asarray([], dtype=np.float64)
    finite = vals[np.isfinite(vals)]

    if finite.size == 0:
        msg = (
            f"{label} | n_idx={idx.size} | finite=0 | "
            f"mean=None | max=None | success_ge_2=0"
        )
    else:
        msg = (
            f"{label} | n_idx={idx.size} | finite={finite.size} | "
            f"mean={np.mean(finite):.6f} | "
            f"min={np.min(finite):.6f} | "
            f"max={np.max(finite):.6f} | "
            f"success_ge_2={int(np.sum(finite >= 2.0))}"
        )

    print(msg)
    if store is not None:
        store.log(msg)

    return {
        "n_idx": int(idx.size),
        "finite": int(finite.size),
        "mean": None if finite.size == 0 else float(np.mean(finite)),
        "min": None if finite.size == 0 else float(np.min(finite)),
        "max": None if finite.size == 0 else float(np.max(finite)),
        "success_ge_2": int(np.sum(finite >= 2.0)) if finite.size else 0,
    }

# ---------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------

import diagnos as diag

def run_mcts_tmp_loop(
    G,
    *,
    max_iter: int = 1000,
    run_root: str | Path = "mcts_runs",
    run_name: str = "tmp",
    overwrite: bool = True,
    chunk_num: int = 0,
    fpc_n_sims: int = 2500,
    save_helper_plots: bool = True,
    save_recreated_plots: bool = True,
    save_full_snapshots: bool = False,
    store_full_mcts_dict_history: bool = True,
    mem_report_after: int = 3,
    break_h_threshold: float = 0.001,
    pw_freeze_threshold: float = 1.0,
    pw_unfreeze_threshold: float = 1.0,
    pw_freeze_threshold_decay_rate: float = 0.01,
    pw_unfreeze_threshold_decay_rate: float = 0.02,
    pw_threshold_floor: float = 0.02,
    pw_walk_down_step: float = 0.1,
    pw_walk_up_step: float = 0.05,
    pw_min_scale: float = 0.01,
    pw_max_scale: float = 1.0,
    pw_ema_alpha: float = 0.10,
    pw_baseline_c: float = 1.00,
    initialization_kwargs_fn: Callable[[Any], dict] = default_initialization_kwargs,
    solver_kwargs_fn: Callable[[], dict] = default_solver_kwargs,
    logwalker_kwargs_fn: Callable[[], dict] = default_logwalker_kwargs,
    ep_module=None,
    E_module=None,
    V_module=None,
    I_module=None,
    OPS_module=None,
):
    """
    Best replacement for the original notebook while-loop.

    The notebook displays nothing except whatever you choose to print after this returns.
    All plots, metrics, arrays, top-edge summaries, and console output go to:
        mcts_runs/<run_name>

    Parameters
    ----------
    G
        Current MCTS Grammar instance.
    run_name
        "tmp" is the scratch run. Change it to preserve runs.
    overwrite
        True means delete an existing run folder before starting.
    *_module
        Optional injected modules. If omitted, this imports your project modules by name.

    Returns
    -------
    dict
        Contains the final G, last X, run store, histories, and final status.
    """
    if ep_module is None:
        import ep_wrap as ep_module
    if E_module is None:
        import evaluation as E_module
    if V_module is None:
        import visualization as V_module
    if I_module is None:
        import initialization as I_module
    if OPS_module is None:
        import transform_ops as OPS_module

    ep = ep_module
    _E = E_module
    _V = V_module
    _I = I_module
    _OPS = OPS_module

    store = MCTSRunStore(
        run_root=run_root,
        run_name=run_name,
        overwrite=overwrite,
    )

    mcts_prev_policy_snapshot = None
    h = np.nan
    hp = []
    some_kappa = -np.log(0.314)

    histories = {
        "g_mcts_sig": [],
        "g_mcts_node_mu": [],
        "g_mcts_edge_mu": [],
        "g_mcts_node_count": [],
        "g_mcts_edge_count": [],
        "g_mcts_children": [],
        "g_mcts_trace": [],

        "g_mcts_alpha_decision_mu": [],
        "g_mcts_alpha_node_mu": [],
        "g_mcts_alpha_edge_mu": [],
        "g_mcts_alpha_decision_count": [],
        "g_mcts_alpha_node_count": [],
        "g_mcts_alpha_edge_count": [],

        "g_mcts_depth_rows": [],
        "g_mcts_parent_depth_probs": [],
        "g_mcts_child_depth_probs": [],

        "g_mcts_alpha_depth_rows": [],
        "g_mcts_alpha_sensor_probs": [],
        "g_mcts_alpha_parent_depth_probs": [],

        "g_mcts_uct_explore_influence": [],
        "g_mcts_uct_explore_influence_ema": [],
        "g_mcts_pw_modes": [],
        "g_mcts_pw_scale": [],
        "g_mcts_pw_action": [],
        "g_mcts_pw_freeze_threshold": [],
        "g_mcts_pw_unfreeze_threshold": [],
        "g_mcts_pw_c": [],
        "g_mcts_expand_prob": [],

        "g_mcts_policy_parent_drift": [],
        "g_mcts_policy_child_drift": [],
        "g_mcts_policy_total_drift": [],
        "hp": hp,

        "p_vals": [],
        "wpath": [],
        "EVs": [],

        "h": h,
        "freeze_expansion": getattr(G, "_MCTS_FREEZE_EXPANSION", False),
    }

    last_X = None
    last_s_idx = None
    last_pvals = None
    last_zscores = None
    pd_fpc = None
    fpc_params = None
    perm_mu = None

    store.write_status("running", k=None)

    diag.start_diag_monitor(
        log_path="loop_diagnostics.csv",
        interval=5,
        tmp_dirs=[
            diag.tempfile.gettempdir(),
            "./tmp",
            "./streamlit_tmp",
        ],
    )

    diag.init_var_growth_diag()

    try:
        for k in range(int(max_iter)):
            #print(f'Starting iteration {k}')
            iter_start = time.time()
            store.log(f"\n================ ITERATION {k} ================")
            store.write_status("running", k=k)

            initialization_kwargs = deepcopy(initialization_kwargs_fn(G))
            solver_kwargs = deepcopy(solver_kwargs_fn())
            logwalker_kwargs = deepcopy(logwalker_kwargs_fn())

            depth_prob_delta_sum = np.nan
            alpha_depth_prob_delta_sum = np.nan
            alpha_sensor_prob = np.nan
            uct_infl = np.nan
            infl_ema = np.nan
            pw_mode = None
            parent_drift = np.nan
            child_drift = np.nan
            total_drift = np.nan

            if k == 0:
                print('TESTING FPC CURVE ON CHUNK I')
                store.log("surveying proportion permutation distributions for initial destination...")

                X_i, G_i = _I.initialize(**initialization_kwargs)

                with capture_console(store, "fit_FPC_part_prop", k=k), capture_plots(
                    store, "fpc_curve_once", k=k, save=save_helper_plots
                ):
                    pd_fpc_i, fpc_params_i, perm_mu_i = _E.fit_FPC_part_prop(
                        X=X_i,
                        chunk_num=chunk_num,
                        solver_kwargs=solver_kwargs,
                        n_sims=fpc_n_sims,
                    )

                print('RAN FUNC ON CHUNK I')

                # estimate destination from FPC curve at target participation
                target_part_prop = 0.15
                target_sd_mult = 2.0

                target_var = float(np.asarray(pd_fpc_i(np.asarray([target_part_prop]))).ravel()[0])
                target_sd = float(np.sqrt(max(target_var, 0.0)))
                target_destination = float(perm_mu_i + target_sd_mult * target_sd)

                logwalker_kwargs["destination"] = target_destination

                write_json_atomic(store.run_dir / "initial_destination_estimate.json", {
                    "target_part_prop": target_part_prop,
                    "target_sd_mult": target_sd_mult,
                    "perm_mu": perm_mu_i,
                    "fpc_var": target_var,
                    "fpc_sd": target_sd,
                    "destination": target_destination,
                })

                store.log(
                    f"INITIAL DESTINATION ESTIMATE | "
                    f"prop={target_part_prop:.4f} | "
                    f"perm_mu={perm_mu_i:.6f} | "
                    f"sd={target_sd:.6f} | "
                    f"destination={target_destination:.6f}"
                )

                print('DONE WITH DESTINATION FINDING')
                print('SELECTED DESTINATION:', {target_destination})
                print('Sounds right?')

            #print('solving inner')

            with capture_console(store, "solver_inner", k=k):
                X, G, s_idx, walker, evaluation = ep.solver_inner(
                    initialization_kwargs,
                    solver_kwargs,
                    logwalker_kwargs,
                    G,
                    chunk_num=chunk_num,
                )

            # same as original notebook: convert emission into evaluation form
            solver_kwargs["emission"].pop()
            solver_kwargs["AD_cond"] = (
                solver_kwargs["AD_cond"][0],
                logwalker_kwargs["start"],
            )

            #print('solved inner')

            with capture_console(store, "instantiate_from_ops_chunked_intraday", k=k):
                instantiation_stats = _I.instantiate_from_ops_chunked_intraday(
                    X,
                    transform_ops=_OPS,
                    chunk_num=chunk_num+1,
                    chunk_B=8,
                )

            #print('instantiated')

            if k == 0:
                store.log("surveying proportion permutation distributions...")

                with capture_console(store, "fit_FPC_part_prop", k=k), capture_plots(
                    store, "fpc_curve_once", k=k, save=save_helper_plots
                ):
                    pd_fpc, fpc_params, perm_mu = _E.fit_FPC_part_prop(
                        X=X,
                        chunk_num=chunk_num+1,
                        solver_kwargs=solver_kwargs,
                        n_sims=fpc_n_sims,
                    )

                write_json_atomic(store.run_dir / "fpc_params.json", {
                    "fpc_params": fpc_params,
                    "perm_mu": perm_mu,
                })

                store.log("surveying proportion permutation distributions... Done.")
                #print('finishin permutation analysis')

            with capture_console(store, "evaluate_genes_from_opg_fpc_fast", k=k), capture_plots(
                store, "gene_fpc_eval_original", k=k, save=save_helper_plots
            ):
                pvals, zscores, gene_eval_details = _E.evaluate_genes_from_opg_fpc_fast(
                    population=X,
                    good_idx=s_idx,
                    chunk_num=chunk_num+1,
                    solver_kwargs=solver_kwargs,
                    fpc_curve=pd_fpc,
                    perm_mu=perm_mu,
                    fpc_params=fpc_params,
                    visualize=True,
                    viz_kwargs={
                        "bins": 20,
                        "top_n": None,
                        "line_alpha": 0.25,
                    },
                    return_details=True,
                )

            # also save recreated version from details so Streamlit can always show it consistently
            try:
                fig = plot_fpc_gene_evaluation_from_details(
                    gene_eval_details,
                    fpc_curve=pd_fpc,
                    perm_mu=perm_mu,
                    show=False,
                )
                store.save_fig(fig, k, "gene_fpc_eval_recreated")

                fpc_plot_data_path = save_fpc_gene_eval_plot_data(
                    store=store,
                    k=k,
                    gene_eval_details=gene_eval_details,
                    fpc_curve=pd_fpc,
                    perm_mu=perm_mu,
                    min_var=1e-18,
                    band_mult=2.0,
                    grid_n=300,
                )
            except Exception:
                store.log("could not recreate gene FPC eval plot:\n" + traceback.format_exc())

            kept_s_idx, kept_scores, family_idx, family_scores = _E.reduce_scored_family_indices(
                X,
                s_idx,
                pvals[s_idx],
            )

            try:
                G.update(
                    X,
                    family_idx,
                    family_scores,
                    quality_fn=G.pvalue_to_fitness,
                    backup_gamma=0.85,
                    backup_reduce="max",
                )
            except TypeError:
                G.update(X, family_idx, family_scores)

            last_X = X
            last_s_idx = s_idx
            last_pvals = pvals
            last_zscores = zscores

            sig = mcts_numeric_signature(G)
            histories["g_mcts_sig"].append(sig)

            if store_full_mcts_dict_history:
                histories["g_mcts_node_mu"].append(deepcopy(mcts_getdict(G, "_MCTS_NODE_MU")))
                histories["g_mcts_edge_mu"].append(deepcopy(mcts_getdict(G, "_MCTS_EDGE_MU")))
                histories["g_mcts_node_count"].append(deepcopy(mcts_getdict(G, "_MCTS_NODE_COUNT")))
                histories["g_mcts_edge_count"].append(deepcopy(mcts_getdict(G, "_MCTS_EDGE_COUNT")))
                histories["g_mcts_children"].append(deepcopy(mcts_getdict(G, "_MCTS_CHILDREN")))
                histories["g_mcts_trace"].append(deepcopy(getattr(G, "_MCTS_TRACE", [])))

                histories["g_mcts_alpha_decision_mu"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")))
                histories["g_mcts_alpha_node_mu"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")))
                histories["g_mcts_alpha_edge_mu"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")))
                histories["g_mcts_alpha_decision_count"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT")))
                histories["g_mcts_alpha_node_count"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT")))
                histories["g_mcts_alpha_edge_count"].append(deepcopy(mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT")))

            histories["p_vals"].append(pvals[s_idx])
            histories["wpath"].append(walker.path)
            histories["EVs"].append(zscores[s_idx])

            # depth + alpha diagnostics
            depth_rows = _V.mcts_depth_summary(G)
            alpha_depth_rows = _V.mcts_alpha_node_depth_summary(G)

            parent_depth_prob, child_depth_prob = _V.mcts_depth_search_probability(
                G=G,
                X=X,
            )

            alpha_sensor_prob, alpha_sensor_by_ctx = _V.mcts_alpha_sensor_probability_summary(G)

            alpha_parent_depth_prob = _V.mcts_alpha_parent_depth_probability(
                G=G,
                X=X,
            )

            histories["g_mcts_depth_rows"].append(depth_rows)
            histories["g_mcts_alpha_depth_rows"].append(alpha_depth_rows)
            histories["g_mcts_parent_depth_probs"].append(parent_depth_prob)
            histories["g_mcts_child_depth_probs"].append(child_depth_prob)
            histories["g_mcts_alpha_sensor_probs"].append(alpha_sensor_prob)
            histories["g_mcts_alpha_parent_depth_probs"].append(alpha_parent_depth_prob)

            if len(histories["g_mcts_child_depth_probs"]) > 1:
                prev = histories["g_mcts_child_depth_probs"][-2]
                curr = histories["g_mcts_child_depth_probs"][-1]

                m = max(prev.shape[0], curr.shape[0])
                prev_pad = np.zeros(m, dtype=np.float64)
                curr_pad = np.zeros(m, dtype=np.float64)

                prev_pad[:prev.shape[0]] = prev
                curr_pad[:curr.shape[0]] = curr

                depth_prob_delta_sum = float(np.abs(curr_pad - prev_pad).sum())

            if len(histories["g_mcts_alpha_parent_depth_probs"]) > 1:
                prev_a = histories["g_mcts_alpha_parent_depth_probs"][-2]
                curr_a = histories["g_mcts_alpha_parent_depth_probs"][-1]

                m_a = max(prev_a.shape[0], curr_a.shape[0])
                prev_a_pad = np.zeros(m_a, dtype=np.float64)
                curr_a_pad = np.zeros(m_a, dtype=np.float64)

                prev_a_pad[:prev_a.shape[0]] = prev_a
                curr_a_pad[:curr_a.shape[0]] = curr_a

                alpha_depth_prob_delta_sum = float(np.abs(curr_a_pad - prev_a_pad).sum())

            with capture_console(store, "print_mcts_depth_summary", k=k):
                _V.print_mcts_depth_summary(G, X=X)

            # preserve original helper plot outputs too
            with capture_console(store, "original_depth_helper_plots", k=k), capture_plots(
                store, "original_depth_helper_plots", k=k, save=save_helper_plots
            ):
                _V.plot_mcts_depth_terms(
                    depth_rows=depth_rows,
                    alpha_depth_rows=alpha_depth_rows,
                    figsize=(16, 4),
                )
                _V.plot_mcts_depth_progress(
                    histories["g_mcts_child_depth_probs"],
                    figsize=(14, 4),
                )
                _V.plot_mcts_alpha_parent_depth_progress(
                    histories["g_mcts_alpha_parent_depth_probs"],
                    figsize=(14, 4),
                )
                _V.plot_mcts_alpha_sensor_probability(
                    histories["g_mcts_alpha_sensor_probs"],
                    figsize=(10, 4),
                )

            ablation = _E.mcts_exploration_ablation_summary(
                G=G,
                X=X,
            )

            uct_infl = ablation["uct_explore_influence"]

            # update EMA exactly like the old function, but do not use binary PW switching
            if np.isfinite(uct_infl):
                if not hasattr(G, "_MCTS_EXPLORE_INFLUENCE_EMA"):
                    G._MCTS_EXPLORE_INFLUENCE_EMA = float(uct_infl)
                else:
                    G._MCTS_EXPLORE_INFLUENCE_EMA = (
                        (1.0 - pw_ema_alpha) * float(G._MCTS_EXPLORE_INFLUENCE_EMA)
                        + pw_ema_alpha * float(uct_infl)
                    )

            infl_ema = float(getattr(G, "_MCTS_EXPLORE_INFLUENCE_EMA", np.nan))

            freeze_decay = float(np.exp(-float(pw_freeze_threshold_decay_rate) * float(k)))
            unfreeze_decay = float(np.exp(-float(pw_unfreeze_threshold_decay_rate) * float(k)))

            pw_freeze_threshold_t = max(
                float(pw_threshold_floor),
                float(pw_freeze_threshold) * freeze_decay,
            )

            pw_unfreeze_threshold_t = max(
                float(pw_threshold_floor),
                float(pw_unfreeze_threshold) * unfreeze_decay,
            )

            # safety: preserve hysteresis ordering
            # unfreeze threshold should stay below freeze threshold
            #if pw_unfreeze_threshold_t >= pw_freeze_threshold_t:
                #pw_unfreeze_threshold_t = max(
                #    float(pw_threshold_floor),
                #    float(pw_freeze_threshold_t) * 0.50,
                #)

            pw_info = walk_pw_scale_from_thresholds(
                G,
                ema=infl_ema,
                freeze_threshold=pw_freeze_threshold_t,
                unfreeze_threshold=pw_unfreeze_threshold_t,
                down_step=pw_walk_down_step,
                up_step=pw_walk_up_step,
                min_scale=pw_min_scale,
                max_scale=pw_max_scale,
            )

            pw_mode = pw_info["pw_action"]

            histories["g_mcts_uct_explore_influence"].append(uct_infl)
            histories["g_mcts_uct_explore_influence_ema"].append(infl_ema)
            histories["g_mcts_pw_modes"].append(pw_mode)

            histories["g_mcts_pw_scale"].append(pw_info["pw_scale"])
            histories["g_mcts_pw_action"].append(pw_info["pw_action"])
            histories["g_mcts_pw_freeze_threshold"].append(pw_freeze_threshold_t)
            histories["g_mcts_pw_unfreeze_threshold"].append(pw_unfreeze_threshold_t)
            histories["g_mcts_pw_c"].append(pw_info["pw_c"])
            histories["g_mcts_expand_prob"].append(pw_info["expand_prob"])

            store.log(
                "PW WALK | "
                f"action={pw_info['pw_action']} | "
                f"ema={infl_ema:.4f} | "
                f"scale={pw_info['pw_old_scale']:.4f}->{pw_info['pw_new_scale']:.4f} | "
                f"freeze_t={pw_freeze_threshold_t:.4f} | "
                f"unfreeze_t={pw_unfreeze_threshold_t:.4f} | "
                f"pw_c={pw_info['pw_c']:.6f} | "
                f"expand_prob={pw_info['expand_prob']:.6f}"
            )

            with capture_console(store, "original_exploration_influence_plot", k=k), capture_plots(
                store, "original_exploration_influence", k=k, save=save_helper_plots
            ):
                _E.plot_mcts_exploration_influence(
                    influence_hist=histories["g_mcts_uct_explore_influence"],
                    k=-np.log(0.314),
                    figsize=(12, 4),
                )

            curr_snap = mcts_exploit_policy_snapshot(
                G=G,
                X=X,
                depth_gamma=0.70,
            )


            policy_drift = tunable_common_support_exploit_policy_drift(
                mcts_prev_policy_snapshot,
                curr_snap,
                parent_child_mix=0.50,#1to0 float parent or child/action stability
                novelty_weight=0.00,#weight for how much new support matters
                removed_weight=0.00,#keep < novelty weight, penalizes disappearing support
                child_weight_mode="parent_weight",
            )

            mcts_prev_policy_snapshot = curr_snap

            parent_drift = policy_drift["parent_drift"]
            child_drift = policy_drift["child_drift"]
            total_drift = policy_drift["total_drift"]

            histories["g_mcts_policy_parent_drift"].append(parent_drift)
            histories["g_mcts_policy_child_drift"].append(child_drift)
            histories["g_mcts_policy_total_drift"].append(total_drift)

            if np.isfinite(total_drift):
                if not np.isfinite(h):
                    h = total_drift
                else:
                    h = h * (1 - np.e ** -some_kappa) + total_drift * (np.e ** -some_kappa)

                hp.append(h)

            histories["h"] = h
            histories["hp"] = hp

            with capture_console(store, "original_policy_drift_plot", k=k), capture_plots(
                store, "original_policy_drift", k=k, save=save_helper_plots
            ):
                _E.plot_mcts_policy_drift_hawkes(
                    hp=hp,
                    raw_hist=histories["g_mcts_policy_total_drift"],
                    parent_hist=histories["g_mcts_policy_parent_drift"],
                    child_hist=histories["g_mcts_policy_child_drift"],
                    figsize=(12, 4),
                )

            #if k == 50:
            #    store.log("freezing MCTS expansion")
            #    G.freeze_mcts_expansion(True)

            #if k == 95:
            #    store.log("reopening MCTS expansion")
            #    G.freeze_mcts_expansion(False)

            histories["freeze_expansion"] = getattr(G, "_MCTS_FREEZE_EXPANSION", False)

            recreated_paths = {}
            if save_recreated_plots:
                recreated_paths = save_standard_iteration_plots(
                    store=store,
                    k=k,
                    histories=histories,
                    depth_rows=depth_rows,
                    alpha_depth_rows=alpha_depth_rows,
                )

            pval_stats = finite_stats(pvals[s_idx])
            zscore_stats = finite_stats(zscores[s_idx])

            top_edges = top_items_json(
                mcts_top_edges(G, n=10),
                count_dict=mcts_getdict(G, "_MCTS_EDGE_COUNT"),
                explore_count_dict=mcts_getdict(G, "_MCTS_EDGE_EXPLORE_COUNT"),
            )

            top_alpha_decisions = top_items_json(
                mcts_top_alpha_decisions(G, n=10),
                count_dict=mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT"),
                explore_count_dict=mcts_getdict(G, "_MCTS_ALPHA_DECISION_EXPLORE_COUNT"),
            )

            top_alpha_edges = top_items_json(
                mcts_top_alpha_edges(G, n=10),
                count_dict=mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT"),
                explore_count_dict=mcts_getdict(G, "_MCTS_ALPHA_EDGE_EXPLORE_COUNT"),
            )

            tops = {
                "k": k,
                "top_edges": top_edges,
                "top_alpha_decisions": top_alpha_decisions,
                "top_alpha_edges": top_alpha_edges,
            }

            write_json_atomic(store.iter_path(k) / "tops.json", tops)
            store.append_tops(tops)

            arrays_path = store.save_arrays_npz(
                k,
                pvals=pvals,
                zscores=zscores,
                s_idx=s_idx,
                pvals_s_idx=pvals[s_idx],
                zscores_s_idx=zscores[s_idx],
                kept_s_idx=kept_s_idx,
                kept_scores=kept_scores,
                family_idx=family_idx,
                family_scores=family_scores,
                parent_depth_prob=parent_depth_prob,
                child_depth_prob=child_depth_prob,
                alpha_parent_depth_prob=alpha_parent_depth_prob,
                mcts_sig=sig,
                policy_parent_drift_hist=histories["g_mcts_policy_parent_drift"],
                policy_child_drift_hist=histories["g_mcts_policy_child_drift"],
                policy_total_drift_hist=histories["g_mcts_policy_total_drift"],
                policy_h_hist=hp,
                uct_explore_influence_hist=histories["g_mcts_uct_explore_influence"],
                uct_explore_influence_ema_hist=histories["g_mcts_uct_explore_influence_ema"],
                pw_freeze_threshold_hist=histories["g_mcts_pw_freeze_threshold"],
                pw_unfreeze_threshold_hist=histories["g_mcts_pw_unfreeze_threshold"],
                pw_scale_hist=histories["g_mcts_pw_scale"],
                prop_g=gene_eval_details.get("prop_g", []),
                observed_scores_g=gene_eval_details.get("observed_scores_g", []),
                fpc_std_g=gene_eval_details.get("fpc_std_g", []),
            )

            try:
                write_json_atomic(
                    store.iter_path(k) / "alpha_sensor_by_ctx.json",
                    {"alpha_sensor_by_ctx": alpha_sensor_by_ctx},
                )
            except Exception:
                write_json_atomic(
                    store.iter_path(k) / "alpha_sensor_by_ctx.json",
                    {"repr": repr(alpha_sensor_by_ctx)},
                )

            if save_full_snapshots:
                store.save_snapshot(k, {
                    "k": k,
                    "X": X,
                    "G": G,
                    "walker": walker,
                    "evaluation": evaluation,
                    "gene_eval_details": gene_eval_details,
                    "instantiation_stats": instantiation_stats,
                })

            store.save_histories(histories)

            elapsed = time.time() - iter_start

            metrics = {
                "k": k,
                "elapsed_sec": elapsed,

                "len_zscores": len(zscores),
                "n_s_idx": int(len(s_idx)),

                "pval_min": pval_stats["min"],
                "pval_mean": pval_stats["mean"],
                "pval_max": pval_stats["max"],

                "zscore_min": zscore_stats["min"],
                "zscore_mean": zscore_stats["mean"],
                "zscore_max": zscore_stats["max"],
                "zscore_std": zscore_stats["std"],

                "mcts_node_mu_count": len(mcts_getdict(G, "_MCTS_NODE_MU")),
                "mcts_edge_mu_count": len(mcts_getdict(G, "_MCTS_EDGE_MU")),
                "mcts_exploit_t": getattr(G, "_MCTS_EXPLOIT_T", None),
                "mcts_explore_t": getattr(G, "_MCTS_EXPLORE_T", None),

                "alpha_decision_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")),
                "alpha_node_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")),
                "alpha_edge_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")),
                "alpha_exploit_t": getattr(G, "_MCTS_ALPHA_EXPLOIT_T", None),
                "alpha_explore_t": getattr(G, "_MCTS_ALPHA_EXPLORE_T", None),

                "fpc_plot_data_path": fpc_plot_data_path,

                "depth_prob_delta_sum": depth_prob_delta_sum,
                "alpha_depth_prob_delta_sum": alpha_depth_prob_delta_sum,
                "alpha_sensor_prob": alpha_sensor_prob,

                "uct_explore_influence": uct_infl,
                "uct_explore_influence_ema": infl_ema,
                "pw_mode": pw_mode,
                "pw_action": pw_info["pw_action"],
                "pw_scale": pw_info["pw_scale"],
                "pw_old_scale": pw_info["pw_old_scale"],
                "pw_new_scale": pw_info["pw_new_scale"],
                "pw_freeze_threshold_decay_rate": pw_freeze_threshold_decay_rate,
                "pw_unfreeze_threshold_decay_rate": pw_unfreeze_threshold_decay_rate,
                "pw_freeze_threshold": pw_freeze_threshold_t,
                "pw_unfreeze_threshold": pw_unfreeze_threshold_t,
                "pw_c": pw_info["pw_c"],
                "pw_alpha": pw_info["pw_alpha"],
                "expand_prob": pw_info["expand_prob"],
                "q_mean": ablation.get("q_mean", np.nan),
                "u_mean": ablation.get("u_mean", np.nan),
                "u_cv": ablation.get("u_cv", np.nan),

                "policy_parent_drift": parent_drift,
                "policy_child_drift": child_drift,
                "policy_total_drift": total_drift,
                "policy_h": h,

                "freeze_expansion": getattr(G, "_MCTS_FREEZE_EXPANSION", False),
                "arrays_path": arrays_path,
                "recreated_plots": recreated_paths,
            }

            store.append_metrics(metrics)

            store.log(
                f"ITERATION {k} COMPLETE | "
                f"h={h:.6f} | "
                f"z_mean={zscore_stats['mean']} | "
                f"z_max={zscore_stats['max']} | "
                f"p_min={pval_stats['min']} | "
                f"elapsed={elapsed:.2f}s | "
                f"freeze={getattr(G, '_MCTS_FREEZE_EXPANSION', False)}"
            )

            if mem_report_after is not None and k > int(mem_report_after):
                with capture_console(store, "mem_report", k=k):
                    _E.mem_report(globals(), top=25, min_mb=0.5)
                gc.collect()

            store.write_status("running", k=k, extra={
                "last_h": h,
                "last_pw_mode": pw_mode,
                "last_pw_freeze_threshold": pw_freeze_threshold_t,
                "last_pw_unfreeze_threshold": pw_unfreeze_threshold_t,
                "last_elapsed_sec": elapsed,
                "last_summary_png": recreated_paths.get("summary_dashboard"),
                "last_zscore_mean": zscore_stats["mean"],
                "last_zscore_max": zscore_stats["max"],
                "last_pval_min": pval_stats["min"],
            })

            diag.diag_mark(
                label=f"k={k}",
                ns=globals(),
                tmp_dirs=[
                    diag.tempfile.gettempdir(),
                    "./tmp",
                    "./streamlit_tmp",
                ],
            )

            #diag.leak_probe(f"k={k}", globals(), deep=True)
            diag.end_loop_diag(k, 1)
            diag.end_loop_var_growth_diag(
                k=k,
                global_ns=globals(),
                local_ns=locals(),
                scan_every=1,
                top_n=25,
                min_mb=1.0,
                depth=2,
                max_items=100,
                trim_test=False,
            )

            #print(f'Ending iteration {k}')

            if k + 1 > 20 and np.isfinite(h) and h < break_h_threshold:
                fin_g_path = store.run_dir / "fin.g"

                with open(fin_g_path, "wb") as f:
                    dill.dump(G, f)

                store.log(
                    f"saved final grammar to {fin_g_path} | "
                    f"breaking because h={h:.6f} < {break_h_threshold}"
                )

                break

        store.write_status("complete", k=k, extra={"last_h": h})
        store.log("run complete")

        return {
            "G": G,
            "last_X": last_X,
            "last_s_idx": last_s_idx,
            "last_pvals": last_pvals,
            "last_zscores": last_zscores,
            "store": store,
            "histories": histories,
            "status": "complete",
        }

    except Exception:
        err = traceback.format_exc()
        store.log(err)
        store.write_status("crashed", k=locals().get("k", None), extra={"error": err})
        raise


# =====================================================================
# walk-forward MCTS utilities v2
# =====================================================================


def set_grammar_mode(G, mode: str):
    """
    Robustly set Grammar train/infer mode without assuming one exact API.
    """
    mode = str(mode)

    meth = getattr(G, "mode", None)
    if callable(meth):
        try:
            meth(mode)
        except Exception:
            pass

    setattr(G, "_mode", mode)
    return G


def _wf_clone_solver_kwargs_for_eval(solver_kwargs: dict, logwalker_kwargs: dict | None = None) -> dict:
    """
    Mirror the existing i->j loop's conversion from search emission to
    evaluation emission.

    Current mcts_util.run_mcts_tmp_loop does:
        solver_kwargs["emission"].pop()
        solver_kwargs["AD_cond"] = (solver_kwargs["AD_cond"][0], logwalker_kwargs["start"])
    """
    out = deepcopy(solver_kwargs)

    if(out["t_mode"] == "AD"):
        if isinstance(out.get("emission", None), list) and len(out["emission"]) > 0:
            out["emission"] = deepcopy(out["emission"])
            out["emission"].pop()

        if logwalker_kwargs is not None and "AD_cond" in out:
            out["AD_cond"] = (out["AD_cond"][0], logwalker_kwargs.get("start", 0))

    elif(out["t_mode"] == "RR" and logwalker_kwargs is not None):
        out["RRR"] = logwalker_kwargs.get("start", 0)

    else:
        raise ValueError(f"No listed t mode case for '{out["t_mode"]}' in _wf_clone_solver_kwargs_for_eval.")

    return out


def make_infer_initialization_kwargs(G, cfg: dict, initialization_kwargs_fn=default_initialization_kwargs) -> dict:
    """
    Direct population-generation kwargs for grammar inference.

    This intentionally avoids ep.solver_inner. It only calls _I.initialize
    with the learned Grammar as grmr_prior.
    """
    kwargs = deepcopy(initialization_kwargs_fn(G))

    kwargs["grmr_prior"] = G
    kwargs["grmr_type"] = "MCTS"
    kwargs["grmr_p_mttn"] = 0.0
    kwargs["grmr_p_csvr"] = 0.0
    kwargs["grmr_a_sens"] = getattr(G, "_alpha_sensor_freq", kwargs.get("grmr_a_sens", 0.0))
    kwargs["verbose"] = 0

    if "pop_size" in cfg:
        kwargs["pop_size"] = int(cfg["pop_size"])
    if "n_chunks" in cfg:
        kwargs["wf_windows"] = int(cfg["n_chunks"])
    if "data_file" in cfg:
        kwargs["data_file"] = cfg["data_file"]
    if "grmr_mdl" in cfg:
        kwargs["grmr_mdl"] = cfg["grmr_mdl"]

    return kwargs


def generate_population_from_grammar(
    G,
    cfg: dict,
    *,
    I_module=None,
    initialization_kwargs_fn=default_initialization_kwargs,
    store=None
):
    """
    Generate one population directly from the current grammar.

    This is the direct-initialize version you requested:
        kwargs = make_initialization_kwargs(G)
        X, G_out = _I.initialize(**kwargs)

    The returned G_out is not used to replace the main grammar by default.
    """
    if I_module is None:
        import initialization as I_module

    G.mode('infer')
    kwargs = make_infer_initialization_kwargs(G, cfg, initialization_kwargs_fn=initialization_kwargs_fn)
    kwargs['grmr_prior'] = G

    _wf_debug_grammar_state(G, "INFER GENERATE input G", store if "store" in locals() else None)

    kwargs = make_infer_initialization_kwargs(G, cfg)

    msg = (
        "INFER INIT KWARGS CHECK | "
        f"grmr_prior_is_G={kwargs.get('grmr_prior', None) is G} | "
        f"grmr_type={kwargs.get('grmr_type', None)} | "
        f"grmr_p_mttn={kwargs.get('grmr_p_mttn', None)} | "
        f"grmr_p_csvr={kwargs.get('grmr_p_csvr', None)} | "
        f"grmr_a_sens={kwargs.get('grmr_a_sens', None)} | "
        f"pop_size={kwargs.get('pop_size', None)} | "
        f"wf_windows={kwargs.get('wf_windows', None)}"
    )
    print(msg)
    if "store" in locals() and store is not None:
        store.log(msg)

    X, G_out = I_module.initialize(**kwargs)

    msg = (
        "INFER INIT RETURN CHECK | "
        f"input_G_id={id(G)} | "
        f"G_out_id={id(G_out)} | "
        f"G_out_is_input_G={G_out is G} | "
        f"X_G_idx={len(getattr(X, '_G_idx', []))} | "
        f"X_L_idx={len(getattr(X, '_L_idx', []))}"
    )
    print(msg)
    if "store" in locals() and store is not None:
        store.log(msg)

    _wf_debug_grammar_state(G_out, "INFER GENERATE returned G_out", store if "store" in locals() else None)

    s_idx = np.asarray(getattr(X, "_G_idx"), dtype=np.int64)

    _wf_debug_population_generated_from_G(
        G,
        X,
        s_idx,
        "INFER GENERATED POPULATION AGAINST INPUT G",
        store if "store" in locals() else None,
    )
    return X, G_out, s_idx


def _solver_signature_for_fpc(solver_kwargs: dict) -> str:
    """
    Stable short signature for solver/evaluation settings.

    This prevents reusing an FPC curve across different emissions, offsets,
    AD conditions, target modes, etc.
    """
    s = json.dumps(jsonable(solver_kwargs), sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _wf_cache_key(chunk_num: int, tag: str = "eval", solver_kwargs: dict | None = None) -> str:
    if solver_kwargs is None:
        return f"{str(tag)}:chunk:{int(chunk_num)}"

    sig = _solver_signature_for_fpc(solver_kwargs)
    return f"{str(tag)}:chunk:{int(chunk_num)}:solver:{sig}"


def _wf_verbose(store, cfg: dict, msg: str, *, level: int = 1) -> None:
    """
    Log and optionally print walk-forward progress messages.

    cfg["verbose"] controls console printing:
        0 = log only
        1 = main progress
        2 = detailed cache/evaluation progress
    """
    try:
        store.log(msg)
    except Exception:
        pass

    try:
        if int(cfg.get("verbose", 1)) >= int(level):
            print(msg, flush=True)
    except Exception:
        pass

def init_fpc_array_cache(n_chunks: int):
    """
    Simple fixed-size FPC cache.

    Each absolute chunk index stores:
        pd_fpc[chunk]
        fpc_params[chunk]
        perm_mu[chunk]

    None means the curve has not been fit yet.
    """

    return {
        "pd_fpc": np.full(int(n_chunks), None, dtype=object),
        "fpc_params": np.full(int(n_chunks), None, dtype=object),
        "perm_mu": np.full(int(n_chunks), None, dtype=object),
    }

def get_fpc_for_chunk_cached(
    *,
    fpc_cache,
    chunk_num: int,
    G,
    solver_kwargs: dict,
    cfg: dict,
    store: MCTSRunStore,
    tag: str = "eval",
    X_template=None,
    I_module=None,
    E_module=None,
    initialization_kwargs_fn=default_initialization_kwargs,
):
    """
    Get or fit the FPC curve for one absolute chunk.

    This uses the simple array cache:

        fpc_cache["pd_fpc"][chunk_num]
        fpc_cache["fpc_params"][chunk_num]
        fpc_cache["perm_mu"][chunk_num]

    There is exactly one cached FPC curve per chunk.

    Notes
    -----
    This assumes the eval solver definition is stable across the entire run.
    That matches your intended design:
        one canonical solver_kwargs/emission target across the project,
        aside from the controlled emission.pop() + AD_cond reset.
    """

    if I_module is None:
        import initialization as I_module

    if E_module is None:
        import evaluation as E_module

    chunk_num = int(chunk_num)
    n_chunks = int(cfg.get("n_chunks", len(fpc_cache["pd_fpc"])))
    n_sims = int(cfg.get("fpc_n_sims", 2500))
    verbose = int(cfg.get("verbose", 0))

    if chunk_num < 0 or chunk_num >= n_chunks:
        raise IndexError(
            f"chunk_num={chunk_num} is outside FPC cache range 0..{n_chunks - 1}"
        )

    # ------------------------------------------------------------
    # cache hit
    # ------------------------------------------------------------

    if fpc_cache["pd_fpc"][chunk_num] is not None:

        if verbose >= 1:
            msg = (
                f"FPC CACHE HIT | tag={tag} | chunk={chunk_num} | "
                f"n_sims={n_sims}"
            )
            print(msg)
            store.log(msg)

        return {
            "chunk_num": chunk_num,
            "tag": str(tag),
            "pd_fpc": fpc_cache["pd_fpc"][chunk_num],
            "fpc_params": fpc_cache["fpc_params"][chunk_num],
            "perm_mu": fpc_cache["perm_mu"][chunk_num],
            "n_sims": n_sims,
            "failed": False,
            "from_cache": True,
        }

    # ------------------------------------------------------------
    # cache miss
    # ------------------------------------------------------------

    if verbose >= 1:
        msg = (
            f"FPC CACHE MISS | fitting {tag} FPC for chunk={chunk_num} "
            f"n_sims={n_sims}"
        )
        print(msg)
        store.log(msg)

    # Prefer the actual population being evaluated.
    # Only fall back to a fresh initialized template if none is supplied.
    if X_template is None:
        kwargs = deepcopy(initialization_kwargs_fn(G))

        if "n_chunks" in cfg:
            kwargs["wf_windows"] = int(cfg["n_chunks"])

        if "data_file" in cfg:
            kwargs["data_file"] = cfg["data_file"]

        kwargs["verbose"] = 0

        if verbose >= 2:
            store.log(
                f"FPC TEMPLATE INIT | tag={tag} chunk={chunk_num} "
                f"pop_size={kwargs.get('pop_size')} wf_windows={kwargs.get('wf_windows')}"
            )

        X_template, _ = I_module.initialize(**kwargs)

    try:
        with capture_console(
            store,
            f"fit_fpc_{tag}_chunk_{chunk_num:03d}",
            k=chunk_num,
        ), capture_plots(
            store,
            f"fit_fpc_{tag}_chunk_{chunk_num:03d}",
            k=chunk_num,
            save=bool(cfg.get("save_helper_plots", True)),
        ):
            pd_fpc, fpc_params, perm_mu = E_module.fit_FPC_part_prop(
                X=X_template,
                chunk_num=chunk_num,
                solver_kwargs=solver_kwargs,
                n_sims=n_sims,
            )

        # store in fixed arrays
        fpc_cache["pd_fpc"][chunk_num] = pd_fpc
        fpc_cache["fpc_params"][chunk_num] = fpc_params
        fpc_cache["perm_mu"][chunk_num] = perm_mu

        append_jsonl(store.run_dir / "fpc_cache_index.jsonl", {
            "chunk_num": chunk_num,
            "tag": str(tag),
            "n_sims": n_sims,
            "failed": False,
            "perm_mu": perm_mu,
        })

        if verbose >= 1:
            msg = (
                f"FPC FIT DONE | tag={tag} | chunk={chunk_num} | "
                f"perm_mu={perm_mu}"
            )
            print(msg)
            store.log(msg)

        return {
            "chunk_num": chunk_num,
            "tag": str(tag),
            "pd_fpc": pd_fpc,
            "fpc_params": fpc_params,
            "perm_mu": perm_mu,
            "n_sims": n_sims,
            "failed": False,
            "from_cache": False,
        }

    except Exception as e:
        err = repr(e)

        store.log(
            f"FPC FIT FAILED | tag={tag} | chunk={chunk_num} | "
            f"error={err}"
        )

        if verbose >= 1:
            print(
                f"FPC FIT FAILED | tag={tag} | chunk={chunk_num} | "
                f"error={err}"
            )

        append_jsonl(store.run_dir / "fpc_cache_index.jsonl", {
            "chunk_num": chunk_num,
            "tag": str(tag),
            "n_sims": n_sims,
            "failed": True,
            "error": err,
        })

        return {
            "chunk_num": chunk_num,
            "tag": str(tag),
            "pd_fpc": None,
            "fpc_params": None,
            "perm_mu": None,
            "n_sims": n_sims,
            "failed": True,
            "error": err,
            "from_cache": False,
        }
    

def instantiate_population_for_chunk(X, chunk_num: int, *, I_module=None, OPS_module=None, chunk_B: int = 8):
    if I_module is None:
        import initialization as I_module
    if OPS_module is None:
        import transform_ops as OPS_module

    return I_module.instantiate_from_ops_chunked_intraday(
        X,
        transform_ops=OPS_module,
        chunk_num=int(chunk_num),
        chunk_B=int(chunk_B),
    )

def evaluate_genes_on_chunk_cached(
    *,
    X,
    good_idx,
    chunk_num: int,
    G,
    solver_kwargs: dict,
    fpc_cache,
    cfg: dict,
    store: MCTSRunStore,
    E_module=None,
    I_module=None,
    OPS_module=None,
    initialization_kwargs_fn=default_initialization_kwargs,
    visualize: bool = False,
    tag: str = "eval",
):
    """
    Instantiate/evaluate a population on one chunk using the simple FPC array cache.

    Important behavior:
        - instantiates X on chunk_num once
        - gets/fits the one FPC curve for chunk_num
        - evaluates all good_idx against that curve
    """

    if E_module is None:
        import evaluation as E_module

    if I_module is None:
        import initialization as I_module

    if OPS_module is None:
        import transform_ops as OPS_module

    chunk_num = int(chunk_num)
    good_idx = np.asarray(good_idx, dtype=np.int64)
    verbose = int(cfg.get("verbose", 0))

    if good_idx.size == 0:
        n_cols = int(X._instructions.shape[0]) if hasattr(X, "_instructions") else 0

        return (
            np.ones(n_cols, dtype=np.float64),
            np.full(n_cols, np.nan, dtype=np.float64),
            {"empty_good_idx": True, "chunk_num": chunk_num},
            {
                "chunk_num": chunk_num,
                "tag": tag,
                "pd_fpc": None,
                "fpc_params": None,
                "perm_mu": None,
                "failed": True,
                "error": "empty good_idx",
            },
        )

    if verbose >= 2:
        msg = (
            f"EVAL CHUNK START | tag={tag} | chunk={chunk_num} | "
            f"n_good={good_idx.size}"
        )
        print(msg)
        store.log(msg)

    # ------------------------------------------------------------
    # instantiate population on this chunk exactly once
    # ------------------------------------------------------------

    with capture_console(
        store,
        f"instantiate_{tag}_chunk_{chunk_num:03d}",
        k=chunk_num,
    ):
        I_module.instantiate_from_ops_chunked_intraday(
            X,
            transform_ops=OPS_module,
            chunk_num=chunk_num,
            chunk_B=int(cfg.get("chunk_B", 8)),
        )

    # ------------------------------------------------------------
    # get or fit one FPC curve for this absolute chunk
    # ------------------------------------------------------------

    fpc = get_fpc_for_chunk_cached(
        fpc_cache=fpc_cache,
        chunk_num=chunk_num,
        G=G,
        solver_kwargs=solver_kwargs,
        cfg=cfg,
        store=store,
        tag=tag,
        X_template=X,
        I_module=I_module,
        E_module=E_module,
        initialization_kwargs_fn=initialization_kwargs_fn,
    )

    if fpc.get("failed", False) or fpc.get("pd_fpc", None) is None:
        n_cols = int(X._instructions.shape[0]) if hasattr(X, "_instructions") else int(np.max(good_idx)) + 1

        pvals = np.ones(n_cols, dtype=np.float64)
        zscores = np.full(n_cols, np.nan, dtype=np.float64)

        details = {
            "fpc_failed": True,
            "chunk_num": chunk_num,
            "reason": fpc.get("error", "unknown FPC failure"),
        }

        if verbose >= 1:
            msg = (
                f"EVAL CHUNK SKIPPED | tag={tag} | chunk={chunk_num} | "
                f"FPC failed: {details['reason']}"
            )
            print(msg)
            store.log(msg)

        return pvals, zscores, details, fpc

    # ------------------------------------------------------------
    # evaluate genes using cached FPC curve
    # ------------------------------------------------------------

    with capture_console(
        store,
        f"evaluate_{tag}_chunk_{chunk_num:03d}",
        k=chunk_num,
    ), capture_plots(
        store,
        f"evaluate_{tag}_chunk_{chunk_num:03d}",
        k=chunk_num,
        save=bool(cfg.get("save_helper_plots", True)) and bool(visualize),
    ):
        pvals, zscores, details = E_module.evaluate_genes_from_opg_fpc_fast(
            population=X,
            good_idx=good_idx,
            chunk_num=chunk_num,
            solver_kwargs=solver_kwargs,
            fpc_curve=fpc["pd_fpc"],
            perm_mu=fpc["perm_mu"],
            fpc_params=fpc["fpc_params"],
            visualize=visualize,
            viz_kwargs={
                "bins": 20,
                "top_n": None,
                "line_alpha": 0.25,
            },
            return_details=True,
        )

    if verbose >= 2:
        z_good = np.asarray(zscores, dtype=np.float64)[good_idx]
        finite = z_good[np.isfinite(z_good)]

        if finite.size:
            msg = (
                f"EVAL CHUNK DONE | tag={tag} | chunk={chunk_num} | "
                f"n={finite.size} | mean={np.mean(finite):.4f} | "
                f"max={np.max(finite):.4f}"
            )
        else:
            msg = (
                f"EVAL CHUNK DONE | tag={tag} | chunk={chunk_num} | "
                f"no finite zscores"
            )

        print(msg)
        store.log(msg)

    return pvals, zscores, details, fpc


def select_successes_from_zscores(zscores, candidate_idx, *, success_z: float = 2.0):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    z = np.asarray(zscores, dtype=np.float64)

    if candidate_idx.size == 0:
        return candidate_idx

    keep = np.isfinite(z[candidate_idx]) & (z[candidate_idx] > float(success_z))
    return candidate_idx[keep]


def _wf_z_stats(zscores, idx):
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return {"n": 0, "mean": None, "max": None, "min": None, "std": None}
    arr = np.asarray(zscores, dtype=np.float64)[idx]
    return finite_stats(arr)


def decay_grammar_evidence(G, gamma: float = 1.0, *, decay_totals: bool = True):
    """
    Decay grammar memory evidence between walk-forward chunks.

    gamma=1.0 is default and does nothing.

    This decays CUM/COUNT/EXPLORE_COUNT dictionaries and total counters.
    Means are recomputed from CUM/COUNT, so the mean value is retained but
    receives less inertia on future chunks. This implements chunk-age decay:
    after two chunk shifts, evidence weight is gamma**2.
    """
    gamma = float(gamma)

    if not np.isfinite(gamma):
        raise ValueError("grammar_memory_gamma must be finite")

    if gamma == 1.0:
        return {"gamma": gamma, "decayed": False}

    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("grammar_memory_gamma should be in [0, 1]")

    count_names = [
        "_MCTS_NODE_COUNT",
        "_MCTS_EDGE_COUNT",
        "_MCTS_NODE_EXPLORE_COUNT",
        "_MCTS_EDGE_EXPLORE_COUNT",
        "_MCTS_ALPHA_DECISION_COUNT",
        "_MCTS_ALPHA_NODE_COUNT",
        "_MCTS_ALPHA_EDGE_COUNT",
        "_MCTS_ALPHA_DECISION_EXPLORE_COUNT",
        "_MCTS_ALPHA_NODE_EXPLORE_COUNT",
        "_MCTS_ALPHA_EDGE_EXPLORE_COUNT",
    ]

    cum_names = [
        "_MCTS_NODE_CUM",
        "_MCTS_EDGE_CUM",
        "_MCTS_ALPHA_DECISION_CUM",
        "_MCTS_ALPHA_NODE_CUM",
        "_MCTS_ALPHA_EDGE_CUM",
    ]

    mu_pairs = [
        ("_MCTS_NODE_CUM", "_MCTS_NODE_COUNT", "_MCTS_NODE_MU"),
        ("_MCTS_EDGE_CUM", "_MCTS_EDGE_COUNT", "_MCTS_EDGE_MU"),
        ("_MCTS_ALPHA_DECISION_CUM", "_MCTS_ALPHA_DECISION_COUNT", "_MCTS_ALPHA_DECISION_MU"),
        ("_MCTS_ALPHA_NODE_CUM", "_MCTS_ALPHA_NODE_COUNT", "_MCTS_ALPHA_NODE_MU"),
        ("_MCTS_ALPHA_EDGE_CUM", "_MCTS_ALPHA_EDGE_COUNT", "_MCTS_ALPHA_EDGE_MU"),
    ]

    for name in count_names + cum_names:
        d = getattr(G, name, None)
        if isinstance(d, dict):
            for key in list(d.keys()):
                try:
                    d[key] = float(d[key]) * gamma
                except Exception:
                    pass

    for cum_name, count_name, mu_name in mu_pairs:
        cum_d = getattr(G, cum_name, None)
        count_d = getattr(G, count_name, None)
        mu_d = getattr(G, mu_name, None)
        if not (isinstance(cum_d, dict) and isinstance(count_d, dict) and isinstance(mu_d, dict)):
            continue

        for key in list(mu_d.keys()):
            c = float(count_d.get(key, 0.0) or 0.0)
            if c > 0 and key in cum_d:
                mu_d[key] = float(cum_d[key]) / c

    if decay_totals:
        for name in ["_MCTS_EXPLOIT_T", "_MCTS_EXPLORE_T", "_MCTS_ALPHA_EXPLOIT_T", "_MCTS_ALPHA_EXPLORE_T"]:
            if hasattr(G, name):
                try:
                    setattr(G, name, float(getattr(G, name) or 0.0) * gamma)
                except Exception:
                    pass

    return {"gamma": gamma, "decayed": True}



def _wf_masked_stats(z, mask):
    """
    Stats for a z-vector and boolean mask of the same length.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    if z.shape[0] != mask.shape[0]:
        return {"n": 0, "min": None, "mean": None, "max": None, "std": None}

    return finite_stats(z[mask])


def _wf_concat_or_empty(values, dtype=np.float64):
    vals = []
    for v in values:
        arr = np.asarray(v, dtype=dtype).ravel()
        if arr.size:
            vals.append(arr)
    if len(vals) == 0:
        return np.asarray([], dtype=dtype)
    return np.concatenate(vals)


def run_inference_funnel_cached(
    *,
    G,
    cfg: dict,
    chunk_i: int,
    chunk_j: int,
    chunk_k: int,
    fpc_cache: dict,
    store: MCTSRunStore,
    window_id: int | None = None,
    E_module=None,
    I_module=None,
    OPS_module=None,
    initialization_kwargs_fn=default_initialization_kwargs,
    solver_kwargs_fn=default_solver_kwargs,
    logwalker_kwargs_fn=default_logwalker_kwargs,
):
    """
    Inference walk-forward survey of the current stochastic grammar.

    Correct all-evaluate-then-mask behavior:
        1. Switch grammar to infer mode.
        2. Generate infer_populations fresh populations from the grammar.
        3. For every generated population, evaluate every generated gene on
           chunk i, chunk j, and chunk k.
        4. Only after all three z-score vectors exist, build masks:
               success_i   = z_i >= success_z
               success_j   = z_j >= success_z
               success_k   = z_k >= success_z
               success_ij  = success_i & success_j
               success_ijk = success_i & success_j & success_k

    This interprets the grammar as a stochastic policy and measures the
    distribution of generated-gene performances across i/j/k, instead of
    filtering candidates before later chunks are evaluated.
    """
    if I_module is None:
        import initialization as I_module
    if E_module is None:
        import evaluation as E_module
    if OPS_module is None:
        import transform_ops as OPS_module

    _wf_debug_grammar_state(G, f"WINDOW {window_id} BEFORE INFERENCE FUNNEL", store)

    G.mode('infer')

    _wf_debug_grammar_state(G, f"WINDOW {window_id} BEFORE INFERENCE FUNNEL", store)

    base_solver_kwargs = deepcopy(solver_kwargs_fn())
    eval_solver_kwargs = base_solver_kwargs#_wf_clone_solver_kwargs_for_eval(base_solver_kwargs, deepcopy(logwalker_kwargs_fn()))

    infer_eval_sig = _wf_solver_signature_light(eval_solver_kwargs)

    store.log(
        "INFER EVAL SOLVER SIGNATURE | "
        f"window={window_id} | chunks=({chunk_i},{chunk_j},{chunk_k}) | "
        f"{json.dumps(jsonable(infer_eval_sig), sort_keys=True)}"
    )

    infer_populations = int(cfg.get("infer_populations", 30))
    success_z = float(cfg.get("success_z", 2.0))
    window_id = int(chunk_i if window_id is None else window_id)

    records = []

    all_z_i = []
    all_z_j = []
    all_z_k = []
    all_mask_i = []
    all_mask_j = []
    all_mask_k = []
    all_mask_ij = []
    all_mask_ijk = []

    for pop_n in range(infer_populations):
        _wf_verbose(
            store,
            cfg,
            f"INFER POP START | window={window_id} pop={pop_n + 1}/{infer_populations} chunks=({chunk_i},{chunk_j},{chunk_k})",
            level=2,
        )

        X_inf, _, s_idx = generate_population_from_grammar(
            G,
            cfg,
            I_module=I_module,
            initialization_kwargs_fn=initialization_kwargs_fn,
            store=store
        )
        s_idx = np.asarray(s_idx, dtype=np.int64)

        # ------------------------------------------------------------
        # Evaluate ALL generated genes on ALL chunks.
        # ------------------------------------------------------------

        p_i, z_i, d_i, fpc_i = evaluate_genes_on_chunk_cached(
            X=X_inf,
            good_idx=s_idx,
            chunk_num=chunk_i,
            G=G,
            solver_kwargs=eval_solver_kwargs,
            fpc_cache=fpc_cache,
            cfg=cfg,
            store=store,
            E_module=E_module,
            I_module=I_module,
            OPS_module=OPS_module,
            initialization_kwargs_fn=initialization_kwargs_fn,
            visualize=False,
            tag="eval",
        )

        p_j, z_j, d_j, fpc_j = evaluate_genes_on_chunk_cached(
            X=X_inf,
            good_idx=s_idx,
            chunk_num=chunk_j,
            G=G,
            solver_kwargs=eval_solver_kwargs,
            fpc_cache=fpc_cache,
            cfg=cfg,
            store=store,
            E_module=E_module,
            I_module=I_module,
            OPS_module=OPS_module,
            initialization_kwargs_fn=initialization_kwargs_fn,
            visualize=False,
            tag="eval",
        )

        p_k, z_k, d_k, fpc_k = evaluate_genes_on_chunk_cached(
            X=X_inf,
            good_idx=s_idx,
            chunk_num=chunk_k,
            G=G,
            solver_kwargs=eval_solver_kwargs,
            fpc_cache=fpc_cache,
            cfg=cfg,
            store=store,
            E_module=E_module,
            I_module=I_module,
            OPS_module=OPS_module,
            initialization_kwargs_fn=initialization_kwargs_fn,
            visualize=False,
            tag="eval",
        )

        z_i = np.asarray(z_i, dtype=np.float64)
        z_j = np.asarray(z_j, dtype=np.float64)
        z_k = np.asarray(z_k, dtype=np.float64)

        zi = z_i[s_idx]
        zj = z_j[s_idx]
        zk = z_k[s_idx]

        mask_i = np.isfinite(zi) & (zi >= success_z)
        mask_j = np.isfinite(zj) & (zj >= success_z)
        mask_k = np.isfinite(zk) & (zk >= success_z)
        mask_ij = mask_i & mask_j
        mask_ijk = mask_i & mask_j & mask_k

        # ------------------------------------------------------------
        # Save per-population arrays for sequential success-flow plots.
        # ------------------------------------------------------------

        grammar_run_name = f"window_{int(window_id):03d}_chunks_{int(chunk_i):03d}_{int(chunk_j):03d}_{int(chunk_k):03d}"

        pop_eval_dir = (
            store.run_dir
            / "wf_eval_arrays"
            / "grammars"
            / grammar_run_name
            / f"pop_{int(pop_n):03d}"
        )
        pop_eval_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            pop_eval_dir / "eval_arrays.npz",
            z_i=z_i,
            z_j=z_j,
            z_k=z_k,
            good_idx=s_idx,
            zi=zi,
            zj=zj,
            zk=zk,
            mask_i=mask_i,
            mask_j=mask_j,
            mask_k=mask_k,
            mask_ij=mask_ij,
            mask_ijk=mask_ijk,
            chunk_i=int(chunk_i),
            chunk_j=int(chunk_j),
            chunk_k=int(chunk_k),
            pop_n=int(pop_n),
            success_z=float(success_z),
        )

        all_z_i.append(zi)
        all_z_j.append(zj)
        all_z_k.append(zk)
        all_mask_i.append(mask_i)
        all_mask_j.append(mask_j)
        all_mask_k.append(mask_k)
        all_mask_ij.append(mask_ij)
        all_mask_ijk.append(mask_ijk)

        rec = {
            "window_id": int(window_id),
            "chunk_i": int(chunk_i),
            "chunk_j": int(chunk_j),
            "chunk_k": int(chunk_k),
            "pop_n": int(pop_n),
            "n_genes": int(len(s_idx)),
            "success_z": float(success_z),

            # single chunk masks across all generated genes
            "n_success_i": int(np.sum(mask_i)),
            "n_success_j_all": int(np.sum(mask_j)),
            "n_success_k_all": int(np.sum(mask_k)),

            # joint masks, computed after all chunks have been evaluated
            "n_success_ij": int(np.sum(mask_ij)),
            "n_success_ijk": int(np.sum(mask_ijk)),

            # evaluation counts: all genes are evaluated on every chunk
            "n_i_eval_all": int(len(s_idx)),
            "n_j_eval_all": int(len(s_idx)),
            "n_k_eval_all": int(len(s_idx)),
            "n_k_eval_on_i": int(np.sum(mask_i)),
            "n_k_eval_on_ij": int(np.sum(mask_ij)),

            # all-gene score distributions
            "z_i_stats_all": finite_stats(zi),
            "z_j_stats_all": finite_stats(zj),
            "z_k_stats_all": finite_stats(zk),

            # conditional/masked score distributions
            "z_j_stats_on_i_success": _wf_masked_stats(zj, mask_i),
            "z_k_stats_on_i_success": _wf_masked_stats(zk, mask_i),
            "z_k_stats_on_ij_success": _wf_masked_stats(zk, mask_ij),
            "z_k_stats_on_ijk_success": _wf_masked_stats(zk, mask_ijk),

            # rates are easier to compare across population sizes
            "p_success_i": float(np.mean(mask_i)) if mask_i.size else None,
            "p_success_j_all": float(np.mean(mask_j)) if mask_j.size else None,
            "p_success_k_all": float(np.mean(mask_k)) if mask_k.size else None,
            "p_success_ij": float(np.mean(mask_ij)) if mask_ij.size else None,
            "p_success_ijk": float(np.mean(mask_ijk)) if mask_ijk.size else None,

            # compatibility fields for older dashboard sections
            "i_success_count": int(np.sum(mask_i)),
            "ij_success_count": int(np.sum(mask_ij)),
            "k_score_stats": _wf_masked_stats(zk, mask_ij),
        }

        records.append(rec)
        append_jsonl(store.run_dir / "wf_inference_records.jsonl", rec)

        _wf_verbose(
            store,
            cfg,
            "INFER POP DONE | "
            f"window={window_id} pop={pop_n + 1}/{infer_populations} "
            f"n={len(s_idx)} i={rec['n_success_i']} j_all={rec['n_success_j_all']} "
            f"ij={rec['n_success_ij']} ijk={rec['n_success_ijk']} "
            f"k_all_mean={rec['z_k_stats_all']['mean']} k_on_ij_mean={rec['z_k_stats_on_ij_success']['mean']}",
            level=1,
        )

    zi_all = _wf_concat_or_empty(all_z_i)
    zj_all = _wf_concat_or_empty(all_z_j)
    zk_all = _wf_concat_or_empty(all_z_k)

    mask_i_all = _wf_concat_or_empty(all_mask_i, dtype=bool)
    mask_j_all = _wf_concat_or_empty(all_mask_j, dtype=bool)
    mask_k_all = _wf_concat_or_empty(all_mask_k, dtype=bool)
    mask_ij_all = _wf_concat_or_empty(all_mask_ij, dtype=bool)
    mask_ijk_all = _wf_concat_or_empty(all_mask_ijk, dtype=bool)

    # save raw infer distribution arrays for later/non-streamlit analysis
    arr_path = store.run_dir / "wf_arrays" / f"wf_infer_window_{window_id:03d}.npz"
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arr_path,
        z_i_all=zi_all,
        z_j_all=zj_all,
        z_k_all=zk_all,
        mask_i=mask_i_all,
        mask_j=mask_j_all,
        mask_k=mask_k_all,
        mask_ij=mask_ij_all,
        mask_ijk=mask_ijk_all,
        chunk_i=int(chunk_i),
        chunk_j=int(chunk_j),
        chunk_k=int(chunk_k),
        success_z=float(success_z),
    )

    grammar_run_name = f"window_{int(window_id):03d}_chunks_{int(chunk_i):03d}_{int(chunk_j):03d}_{int(chunk_k):03d}"

    success_flow_path = save_one_grammar_success_flow_plot(
        eval_root=store.run_dir / "wf_eval_arrays" / "grammars",
        window_id=window_id,
        run_name=grammar_run_name,
        out_dir=store.plot_dir / "wf_success_flow",
        success_z=success_z,
    )

    summary = {
        "window_id": int(window_id),
        "chunk_i": int(chunk_i),
        "chunk_j": int(chunk_j),
        "chunk_k": int(chunk_k),
        "infer_populations": int(infer_populations),
        "success_z": float(success_z),
        "infer_arrays_path": str(arr_path),

        "total_genes": int(zi_all.size),
        "total_i_eval_all": int(zi_all.size),
        "total_j_eval_all": int(zj_all.size),
        "total_k_eval_all": int(zk_all.size),

        "total_success_i": int(np.sum(mask_i_all)),
        "total_success_j_all": int(np.sum(mask_j_all)),
        "total_success_k_all": int(np.sum(mask_k_all)),
        "total_success_ij": int(np.sum(mask_ij_all)),
        "total_success_ijk": int(np.sum(mask_ijk_all)),
        "total_k_eval_on_i": int(np.sum(mask_i_all)),
        "total_k_eval_on_ij": int(np.sum(mask_ij_all)),

        "success_flow_plot_path": success_flow_path,

        "p_success_i": float(np.mean(mask_i_all)) if mask_i_all.size else None,
        "p_success_j_all": float(np.mean(mask_j_all)) if mask_j_all.size else None,
        "p_success_k_all": float(np.mean(mask_k_all)) if mask_k_all.size else None,
        "p_success_ij": float(np.mean(mask_ij_all)) if mask_ij_all.size else None,
        "p_success_ijk": float(np.mean(mask_ijk_all)) if mask_ijk_all.size else None,

        "z_i_stats_all": finite_stats(zi_all),
        "z_j_stats_all": finite_stats(zj_all),
        "z_k_stats_all": finite_stats(zk_all),
        "z_j_stats_on_i_success": _wf_masked_stats(zj_all, mask_i_all),
        "z_k_stats_on_i_success": _wf_masked_stats(zk_all, mask_i_all),
        "z_k_stats_on_ij_success": _wf_masked_stats(zk_all, mask_ij_all),
        "z_k_stats_on_ijk_success": _wf_masked_stats(zk_all, mask_ijk_all),

        # compatibility fields
        "i_success_count": int(np.sum(mask_i_all)),
        "ij_success_count": int(np.sum(mask_ij_all)),
        "total_success_k_eval": int(np.sum(mask_ij_all)),
        "k_score_stats": _wf_masked_stats(zk_all, mask_ij_all),
    }

    append_jsonl(store.run_dir / "wf_chunk_summaries.jsonl", summary)
    return summary, records, zk_all[mask_ij_all]

def _valid_gidx_for_z(z, good_idx):
    """
    return valid integer gene indices for a zscore array.
    """

    z = np.asarray(z)
    good_idx = np.asarray(good_idx, dtype=np.int64).ravel()

    good_idx = good_idx[np.isfinite(good_idx)]
    good_idx = good_idx.astype(np.int64, copy=False)

    return good_idx[(good_idx >= 0) & (good_idx < z.shape[0])]


def _scatter_success_split(
    ax,
    x,
    zvals,
    success_mask,
    *,
    alpha=0.35,
    red_label=None,
    green_label=None,
):
    """
    scatter z values split by success/failure.
    """

    zvals = np.asarray(zvals, dtype=np.float64).ravel()
    success_mask = np.asarray(success_mask, dtype=bool).ravel()

    if zvals.size == 0:
        return

    xvals = np.full(zvals.shape[0], float(x), dtype=np.float64)

    fail = ~success_mask

    if np.any(fail):
        ax.scatter(
            xvals[fail],
            zvals[fail],
            s=9,
            alpha=alpha,
            color="red",
            label=red_label,
        )

    if np.any(success_mask):
        ax.scatter(
            xvals[success_mask],
            zvals[success_mask],
            s=9,
            alpha=alpha,
            color="green",
            label=green_label,
        )


def save_one_grammar_success_flow_plot(
    *,
    eval_root,
    window_id: int,
    run_name: str,
    out_dir,
    success_z: float = 2.0,
):
    """
    Sequential success-flow z cloud for one completed grammar/window.

    Expected directory:
        eval_root / run_name / pop_000 / eval_arrays.npz
        eval_root / run_name / pop_001 / eval_arrays.npz
        ...

    Panel i:
        plot all good_idx genes on chunk i.
        green = z_i >= success_z, red = not successful.

    Panel j:
        plot only genes successful on chunk i.
        green = z_j >= success_z, red = failed on j.

    Panel k:
        plot only genes successful on both chunk i and chunk j.
        green = z_k >= success_z, red = failed on k.
    """

    eval_root = Path(eval_root)
    out_dir = Path(out_dir)

    grammar_dir = eval_root / str(run_name)
    pop_dirs = sorted(grammar_dir.glob("pop_*"))

    if not pop_dirs:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for p_i, pop_dir in enumerate(pop_dirs):
        path = pop_dir / "eval_arrays.npz"

        if not path.exists():
            continue

        data = np.load(path, allow_pickle=True)

        z_i = np.clip(np.asarray(data["z_i"], dtype=np.float64), -3.0, 10.0)
        z_j = np.clip(np.asarray(data["z_j"], dtype=np.float64), -3.0, 10.0)
        z_k = np.clip(np.asarray(data["z_k"], dtype=np.float64), -3.0, 10.0)

        good_idx = _valid_gidx_for_z(z_i, data["good_idx"])
        good_idx = good_idx[(good_idx < z_j.size) & (good_idx < z_k.size)]

        if good_idx.size == 0:
            continue

        alpha = 0.10 + 0.45 * ((p_i + 1) / max(1, len(pop_dirs)))

        # chunk i: all valid good_idx genes
        idx_i = good_idx[np.isfinite(z_i[good_idx])]
        succ_i = z_i[idx_i] >= success_z

        _scatter_success_split(
            axs[0],
            p_i,
            z_i[idx_i],
            succ_i,
            alpha=alpha,
            red_label="failed i" if p_i == 0 else None,
            green_label="success i" if p_i == 0 else None,
        )

        # chunk j: only genes successful in i
        idx_j_base = idx_i[succ_i]
        idx_j = idx_j_base[np.isfinite(z_j[idx_j_base])]
        succ_j = z_j[idx_j] >= success_z

        _scatter_success_split(
            axs[1],
            p_i,
            z_j[idx_j],
            succ_j,
            alpha=alpha,
            red_label="i success, failed j" if p_i == 0 else None,
            green_label="i+j success" if p_i == 0 else None,
        )

        # chunk k: only genes successful in i and j
        idx_k_base = idx_j[succ_j]
        idx_k = idx_k_base[np.isfinite(z_k[idx_k_base])]
        succ_k = z_k[idx_k] >= success_z

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
        ax.set_xlabel("inference population")
        ax.set_ylabel("z-score")
        ax.set_ylim(bottom=-3)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{run_name} sequential success flow")

    out_path = out_dir / f"z_success_flow_{run_name}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)

def _save_wf_k_score_plot(store: MCTSRunStore, window_id: int, k_scores):
    """
    Backward-compatible plot for k scores on the i+j-success mask.
    """
    k_scores = np.asarray(k_scores, dtype=np.float64).ravel()
    k_scores = k_scores[np.isfinite(k_scores)]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    if k_scores.size == 0:
        ax.text(0.5, 0.5, "no i+j-success genes on chunk k", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(k_scores, bins=30, alpha=0.80)
        ax.axvline(0, alpha=0.35)
        ax.axvline(2, alpha=0.50)
        ax.set_title(f"window {window_id} chunk-k z-score distribution | i+j mask")
        ax.set_xlabel("chunk k z-score")
        ax.set_ylabel("count")

    return store.save_fig(fig, k=window_id, section="wf_k_scores")


def _save_wf_ijk_performance_plots(store: MCTSRunStore, window_id: int, summary: dict):
    """
    Save several walk-forward i/j/k plots from the all-eval inference arrays.
    """
    arr_path = summary.get("infer_arrays_path", None)
    if arr_path is None:
        return {}

    arr_path = Path(arr_path)
    if not arr_path.exists():
        return {}

    data = np.load(arr_path)
    zi = np.asarray(data["z_i_all"], dtype=np.float64)
    zj = np.asarray(data["z_j_all"], dtype=np.float64)
    zk = np.asarray(data["z_k_all"], dtype=np.float64)
    mi = np.asarray(data["mask_i"], dtype=bool)
    mj = np.asarray(data["mask_j"], dtype=bool)
    mk = np.asarray(data["mask_k"], dtype=bool)
    mij = np.asarray(data["mask_ij"], dtype=bool)
    mijk = np.asarray(data["mask_ijk"], dtype=bool)

    paths = {}

    # 1) all-gene i/j/k distributions
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for z, label in [(zi, "chunk i"), (zj, "chunk j"), (zk, "chunk k")]:
        z = z[np.isfinite(z)]
        if z.size:
            ax.hist(z, bins=40, alpha=0.35, density=True, label=label)
    ax.axvline(0, alpha=0.30)
    ax.axvline(2, linestyle="--", alpha=0.55, label="success z=2")
    ax.set_title(f"window {window_id} all generated gene z-score distributions")
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.legend()
    paths["wf_ijk_all_z_dists"] = store.save_fig(fig, k=window_id, section="wf_ijk_all_z_dists")

    # 2) k distributions under masks
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for z, mask, label in [
        (zk, np.ones_like(mk, dtype=bool), "k all genes"),
        (zk, mi, "k | i success"),
        (zk, mij, "k | i+j success"),
        (zk, mijk, "k | i+j+k success"),
    ]:
        vals = z[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=40, alpha=0.35, density=True, label=f"{label} (n={vals.size})")
    ax.axvline(0, alpha=0.30)
    ax.axvline(2, linestyle="--", alpha=0.55, label="success z=2")
    ax.set_title(f"window {window_id} chunk-k distributions under masks")
    ax.set_xlabel("chunk k z-score")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    paths["wf_k_masked_z_dists"] = store.save_fig(fig, k=window_id, section="wf_k_masked_z_dists")

    # 3) counts and rates
    labels = ["i", "j all", "k all", "i+j", "i+j+k"]
    counts = [int(mi.sum()), int(mj.sum()), int(mk.sum()), int(mij.sum()), int(mijk.sum())]
    rates = [float(c / max(1, zi.size)) for c in counts]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].bar(labels, counts)
    axs[0].set_title("success counts")
    axs[0].set_ylabel("count")
    axs[0].tick_params(axis="x", rotation=30)
    axs[1].bar(labels, rates)
    axs[1].set_title("success rates")
    axs[1].set_ylabel("rate")
    axs[1].set_ylim(0, max(0.01, min(1.0, max(rates) * 1.25 if rates else 1.0)))
    axs[1].tick_params(axis="x", rotation=30)
    paths["wf_success_counts_rates"] = store.save_fig(fig, k=window_id, section="wf_success_counts_rates")

    # 4) i/j/k scatter transfer view
    finite = np.isfinite(zi) & np.isfinite(zj) & np.isfinite(zk)
    sample = np.where(finite)[0]
    if sample.size > 5000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=5000, replace=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if sample.size:
        axs[0].scatter(zi[sample], zj[sample], s=5, alpha=0.25)
        axs[1].scatter(zj[sample], zk[sample], s=5, alpha=0.25)
    for ax in axs:
        ax.axhline(2, linestyle="--", alpha=0.35)
        ax.axvline(2, linestyle="--", alpha=0.35)
        ax.grid(alpha=0.25)
    axs[0].set_title("transfer i -> j")
    axs[0].set_xlabel("z_i")
    axs[0].set_ylabel("z_j")
    axs[1].set_title("transfer j -> k")
    axs[1].set_xlabel("z_j")
    axs[1].set_ylabel("z_k")
    paths["wf_transfer_scatter"] = store.save_fig(fig, k=window_id, section="wf_transfer_scatter")

    return paths


def run_mcts_walk_forward(
    G,
    *,
    n_chunks: int = 20,
    delta_L: float = 0.01,
    grammar_memory_gamma: float = 1.0,
    grammar_explore_gamma: float = 1.0,
    max_iters_per_window: int = 1000,
    infer_populations: int = 30,
    success_z: float = 2.0,
    run_root: str | Path = "mcts_runs",
    run_name: str = "wf_tmp",
    overwrite: bool = True,
    fpc_n_sims: int = 2500,
    verbose: int = 1,
    save_helper_plots: bool = False,
    save_recreated_plots: bool = True,
    store_full_mcts_dict_history: bool = False,
    break_min_iter: int = 5,
    depth_gamma: float = 0.70,
    initialization_kwargs_fn: Callable[[Any], dict] = default_initialization_kwargs,
    solver_kwargs_fn: Callable[[], dict] = default_solver_kwargs,
    logwalker_kwargs_fn: Callable[[], dict] = default_logwalker_kwargs,
    ep_module=None,
    E_module=None,
    V_module=None,
    I_module=None,
    OPS_module=None,
):
    """
    Walk-forward MCTS evolution.

    For window w:
        i = w
        j = w + 1
        k = w + 2

    Train/evolve on i->j until exploit policy drift h < delta_L, then infer
    and score i->j->k. The same grammar carries forward and can optionally
    decay evidence each window using grammar_memory_gamma.
    """
    if ep_module is None:
        import ep_wrap as ep_module
    if E_module is None:
        import evaluation as E_module
    if V_module is None:
        import visualization as V_module
    if I_module is None:
        import initialization as I_module
    if OPS_module is None:
        import transform_ops as OPS_module

    ep = ep_module
    _E = E_module
    _V = V_module
    _I = I_module
    _OPS = OPS_module

    cfg = {
        "n_chunks": int(n_chunks),
        "delta_L": float(delta_L),
        "grammar_memory_gamma": float(grammar_memory_gamma),
        "max_iters_per_window": int(max_iters_per_window),
        "infer_populations": int(infer_populations),
        "success_z": float(success_z),
        "fpc_n_sims": int(fpc_n_sims),
        "verbose": int(verbose),
        "save_helper_plots": bool(save_helper_plots),
        "chunk_B": 8,
    }

    # inherit default data/pop params for cached FPC/infer init
    try:
        init_probe = initialization_kwargs_fn(G)
        cfg["data_file"] = init_probe.get("data_file", "../data/spy5m.csv")
        cfg["pop_size"] = int(init_probe.get("pop_size", 100))
        cfg["grmr_mdl"] = init_probe.get("grmr_mdl", 80)
    except Exception:
        pass

    store = MCTSRunStore(run_root=run_root, run_name=run_name, overwrite=overwrite)
    store.write_status("running_walk_forward", k=None, extra=cfg)
    write_json_atomic(store.run_dir / "wf_config.json", cfg)

    fpc_cache = init_fpc_array_cache(int(cfg["n_chunks"]))
    window_summaries = []
    global_iter = 0

    # one persistent drift state per training window is easier to interpret
    # so it resets at each i,j,k window.

    fpc_curves = np.full(n_chunks, None, dtype=object)

    try:
        for window_id in range(0, int(n_chunks) - 2):
            chunk_i = int(window_id)
            chunk_j = int(window_id + 1)
            chunk_k = int(window_id + 2)

            _wf_verbose(
                store,
                cfg,
                f"\n================ WF WINDOW {window_id} | i={chunk_i} j={chunk_j} k={chunk_k} ================",
                level=1,
            )
            store.write_status("training_window", k=global_iter, extra={
                "window_id": window_id,
                "chunk_i": chunk_i,
                "chunk_j": chunk_j,
                "chunk_k": chunk_k,
            })

            # apply age decay before learning new chunk, except first window
            if window_id > 0:

                if grammar_memory_gamma == 0.0:
                    hard_reset_mcts_memory(G)
                else:
                    decay_mcts_effective_memory(
                        G,
                        exploit_gamma=grammar_memory_gamma,
                        explore_gamma=grammar_explore_gamma,
                        min_eff_count=1e-6,
                    )

            G.mode('train')

            mcts_prev_policy_snapshot = None
            h = np.nan
            hp = []
            some_kappa = -np.log(0.314)

            local_hist = {
                "g_mcts_sig": [],
                "g_mcts_policy_parent_drift": [],
                "g_mcts_policy_child_drift": [],
                "g_mcts_policy_total_drift": [],
                "hp": hp,
                "EVs": [],
            }

            last_train_X = None
            last_s_idx = None
            last_pvals = None
            last_zscores = None

            for local_iter in range(int(max_iters_per_window)):
                iter_start = time.time()

                _wf_verbose(
                    store,
                    cfg,
                    f"WF ITER START | window={window_id} local_iter={local_iter} global_iter={global_iter} chunks=({chunk_i},{chunk_j},{chunk_k})",
                    level=2,
                )

                initialization_kwargs = deepcopy(initialization_kwargs_fn(G))
                solver_kwargs = deepcopy(solver_kwargs_fn())
                logwalker_kwargs = deepcopy(logwalker_kwargs_fn())

                # Estimate/search destination from chunk i using a destination-tag FPC.
                dest_fpc = get_fpc_for_chunk_cached(
                    fpc_cache=fpc_cache,
                    chunk_num=chunk_i,
                    G=G,
                    solver_kwargs=solver_kwargs,
                    cfg=cfg,
                    store=store,
                    tag="destination",
                    I_module=_I,
                    E_module=_E,
                    initialization_kwargs_fn=initialization_kwargs_fn,
                )

                target_part_prop = 0.15
                target_sd_mult = 2.0
                target_var = float(np.asarray(dest_fpc["pd_fpc"](np.asarray([target_part_prop]))).ravel()[0])
                target_sd = float(np.sqrt(max(target_var, 0.0)))
                logwalker_kwargs["destination"] = float(dest_fpc["perm_mu"] + target_sd_mult * target_sd)

                _wf_verbose(
                    store,
                    cfg,
                    f"LOGWALKER DEST | window={window_id} iter={local_iter} chunk_i={chunk_i} destination={logwalker_kwargs['destination']:.8f}",
                    level=2,
                )

                with capture_console(store, "wf_solver_inner", k=global_iter):
                    X, G, s_idx, walker, evaluation = ep.solver_inner(
                        initialization_kwargs,
                        solver_kwargs,
                        logwalker_kwargs,
                        G,
                        chunk_num=chunk_i,
                    )

                eval_solver_kwargs = _wf_clone_solver_kwargs_for_eval(solver_kwargs, logwalker_kwargs)

                train_eval_sig = _wf_solver_signature_light(eval_solver_kwargs)

                if(solver_kwargs["t_mode"] == "AD" or solver_kwargs["t_mode"]):
                    store.log(
                        "TRAIN EVAL SOLVER SIGNATURE | "
                        f"window={window_id} | chunks=({chunk_i},{chunk_j},{chunk_k}) | "
                        f"{json.dumps(jsonable(train_eval_sig), sort_keys=True)}"
                    )

                pvals, zscores, gene_eval_details, fpc_j = evaluate_genes_on_chunk_cached(
                    X=X,
                    good_idx=s_idx,
                    chunk_num=chunk_j,
                    G=G,
                    solver_kwargs=eval_solver_kwargs,
                    fpc_cache=fpc_cache,
                    cfg=cfg,
                    store=store,
                    E_module=_E,
                    I_module=_I,
                    OPS_module=_OPS,
                    initialization_kwargs_fn=initialization_kwargs_fn,
                    visualize=False,
                    tag="eval",
                )

                kept_s_idx, kept_scores, family_idx, family_scores = _E.reduce_scored_family_indices(
                    X,
                    s_idx,
                    pvals[s_idx],
                )

                _wf_debug_grammar_state(G, f"WINDOW {window_id} BEFORE G.update", store)

                pre_sig = mcts_numeric_signature(G)

                store.log(
                    "UPDATE INPUT CHECK | "
                    f"window={window_id} | "
                    f"train_chunk_i={chunk_i} | eval_chunk_j={chunk_j} | "
                    f"s_idx={len(s_idx)} | "
                    f"kept_s_idx={len(kept_s_idx)} | "
                    f"family_idx_len={len(family_idx)} | "
                    f"family_scores_len={len(family_scores)} | "
                    f"kept_score_mean={np.nanmean(kept_scores) if len(kept_scores) else None} | "
                    f"kept_score_min={np.nanmin(kept_scores) if len(kept_scores) else None} | "
                    f"kept_score_max={np.nanmax(kept_scores) if len(kept_scores) else None}"
                )

                G.update(
                        X,
                        family_idx,
                        family_scores,
                        quality_fn=G.pvalue_to_fitness,
                        backup_gamma=0.85,
                        backup_reduce="max",
                    )
                
                post_sig = mcts_numeric_signature(G)

                _wf_debug_grammar_state(G, f"WINDOW {window_id} AFTER G.update", store)

                store.log(
                    "GRAMMAR UPDATE DELTA | "
                    f"window={window_id} | "
                    f"node_mu {pre_sig[0]}->{post_sig[0]} | "
                    f"edge_mu {pre_sig[1]}->{post_sig[1]} | "
                    f"node_count_sum {pre_sig[2]}->{post_sig[2]} | "
                    f"edge_count_sum {pre_sig[3]}->{post_sig[3]} | "
                    f"node_explore_sum {pre_sig[4]}->{post_sig[4]} | "
                    f"edge_explore_sum {pre_sig[5]}->{post_sig[5]}"
                )

                curr_snap = mcts_exploit_policy_snapshot(G=G, X=X, depth_gamma=depth_gamma)
                policy_drift = tunable_common_support_exploit_policy_drift(
                    mcts_prev_policy_snapshot,
                    curr_snap,
                    parent_child_mix=0.50,
                    novelty_weight=0.00,
                    removed_weight=0.00,
                    child_weight_mode="parent_weight",
                )
                mcts_prev_policy_snapshot = curr_snap

                parent_drift = policy_drift["parent_drift"]
                child_drift = policy_drift["child_drift"]
                total_drift = policy_drift["total_drift"]

                if np.isfinite(total_drift):
                    if not np.isfinite(h):
                        h = float(total_drift)
                    else:
                        h = h * (1 - np.e ** -some_kappa) + float(total_drift) * (np.e ** -some_kappa)
                    hp.append(h)

                local_hist["g_mcts_sig"].append(mcts_numeric_signature(G))
                local_hist["g_mcts_policy_parent_drift"].append(parent_drift)
                local_hist["g_mcts_policy_child_drift"].append(child_drift)
                local_hist["g_mcts_policy_total_drift"].append(total_drift)
                local_hist["EVs"].append(np.asarray(zscores, dtype=np.float64)[np.asarray(s_idx, dtype=np.int64)])

                last_train_X = X
                last_s_idx = s_idx
                last_pvals = pvals
                last_zscores = zscores

                metrics = {
                    "global_iter": int(global_iter),
                    "window_id": int(window_id),
                    "local_iter": int(local_iter),
                    "chunk_i": chunk_i,
                    "chunk_j": chunk_j,
                    "chunk_k": chunk_k,
                    "delta_L": float(delta_L),
                    "grammar_memory_gamma": float(grammar_memory_gamma),
                    "policy_parent_drift": parent_drift,
                    "policy_child_drift": child_drift,
                    "policy_total_drift": total_drift,
                    "policy_h": h,
                    "zscore_stats_j": finite_stats(np.asarray(zscores, dtype=np.float64)[np.asarray(s_idx, dtype=np.int64)]),
                    "mcts_node_mu_count": len(mcts_getdict(G, "_MCTS_NODE_MU")),
                    "mcts_edge_mu_count": len(mcts_getdict(G, "_MCTS_EDGE_MU")),
                    "alpha_decision_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU")),
                    "alpha_node_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_NODE_MU")),
                    "alpha_edge_mu_count": len(mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU")),
                    "elapsed_sec": float(time.time() - iter_start),
                }
                store.append_metrics(metrics)

                _wf_verbose(
                    store,
                    cfg,
                    f"WF ITER DONE | window={window_id} local_iter={local_iter} h={h} delta_L={delta_L} z_j_mean={metrics['zscore_stats_j'].get('mean')} elapsed={metrics['elapsed_sec']:.2f}s",
                    level=1,
                )

                if save_recreated_plots:
                    try:
                        fig = plot_mcts_policy_drift_hawkes(
                            hp=hp,
                            raw_hist=local_hist["g_mcts_policy_total_drift"],
                            parent_hist=local_hist["g_mcts_policy_parent_drift"],
                            child_hist=local_hist["g_mcts_policy_child_drift"],
                            show=False,
                        )
                        store.save_fig(fig, k=global_iter, section="wf_policy_drift")

                        fig = plot_mcts_state_growth_summary(local_hist["g_mcts_sig"], show=False)
                        store.save_fig(fig, k=global_iter, section="wf_state_growth")
                    except Exception:
                        store.log("WF plot save failed\n" + traceback.format_exc())

                store.write_status("training_window", k=global_iter, extra={
                    "window_id": window_id,
                    "local_iter": local_iter,
                    "chunk_i": chunk_i,
                    "chunk_j": chunk_j,
                    "chunk_k": chunk_k,
                    "policy_h": h,
                    "delta_L": float(delta_L),
                })

                global_iter += 1

                if local_iter + 1 >= int(break_min_iter) and np.isfinite(h) and h < float(delta_L):
                    store.log(f"WF window {window_id} converged: h={h:.6f} < delta_L={delta_L}")
                    break

            # Save grammar at resolved window.
            grammar_path = store.run_dir / "grammars" / f"grammar_window_{int(window_id):03d}.pkl.gz"
            grammar_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(grammar_path, "wb") as f:
                dill.dump(G, f)

            _wf_debug_grammar_state(G, f"WINDOW {window_id} BEFORE INFERENCE FUNNEL", store)

            G.mode('infer')

            _wf_debug_grammar_state(G, f"WINDOW {window_id} BEFORE INFERENCE FUNNEL", store)

            if bool(cfg.get("debug_eval_training_population_control", True)):

                store.log(
                    "TRAINING POP CONTROL START | "
                    f"window={window_id} chunks=({chunk_i},{chunk_j},{chunk_k})"
                )

                p_ctrl_i, z_ctrl_i, d_ctrl_i, _ = evaluate_genes_on_chunk_cached(
                    X=last_train_X if "last_train_X" in locals() and last_train_X is not None else X,
                    good_idx=last_s_idx if "last_s_idx" in locals() and last_s_idx is not None else s_idx,
                    chunk_num=chunk_i,
                    G=G,
                    solver_kwargs=eval_solver_kwargs,
                    fpc_cache=fpc_cache,
                    cfg=cfg,
                    store=store,
                    E_module=_E,
                    I_module=_I,
                    OPS_module=_OPS,
                    initialization_kwargs_fn=initialization_kwargs_fn,
                    visualize=False,
                    tag="eval_control_i",
                )

                p_ctrl_j, z_ctrl_j, d_ctrl_j, _ = evaluate_genes_on_chunk_cached(
                    X=last_train_X if "last_train_X" in locals() and last_train_X is not None else X,
                    good_idx=last_s_idx if "last_s_idx" in locals() and last_s_idx is not None else s_idx,
                    chunk_num=chunk_j,
                    G=G,
                    solver_kwargs=eval_solver_kwargs,
                    fpc_cache=fpc_cache,
                    cfg=cfg,
                    store=store,
                    E_module=_E,
                    I_module=_I,
                    OPS_module=_OPS,
                    initialization_kwargs_fn=initialization_kwargs_fn,
                    visualize=False,
                    tag="eval_control_j",
                )

                ctrl_idx = last_s_idx if "last_s_idx" in locals() and last_s_idx is not None else s_idx

                _wf_debug_zscores(z_ctrl_i, ctrl_idx, f"TRAINING POP CONTROL I | window={window_id}", store)
                _wf_debug_zscores(z_ctrl_j, ctrl_idx, f"TRAINING POP CONTROL J | window={window_id}", store)

            # Investigate grammar on i,j,k in infer mode.
            summary, records, k_scores = run_inference_funnel_cached(
                G=G,
                cfg=cfg,
                chunk_i=chunk_i,
                chunk_j=chunk_j,
                chunk_k=chunk_k,
                fpc_cache=fpc_cache,
                store=store,
                window_id=window_id,
                E_module=_E,
                I_module=_I,
                OPS_module=_OPS,
                initialization_kwargs_fn=initialization_kwargs_fn,
                solver_kwargs_fn=solver_kwargs_fn,
                logwalker_kwargs_fn=logwalker_kwargs_fn,
            )

            k_plot = _save_wf_k_score_plot(store, window_id, k_scores)
            ijk_plot_paths = _save_wf_ijk_performance_plots(store, window_id, summary)
            summary.update({
                "window_id": int(window_id),
                "grammar_path": str(grammar_path),
                "k_plot": str(k_plot),
                "ijk_plot_paths": ijk_plot_paths,
                "final_h": h,
                "delta_L": float(delta_L),
                "grammar_memory_gamma": float(grammar_memory_gamma),
            })

            append_jsonl(store.run_dir / "wf_window_summaries.jsonl", summary)
            window_summaries.append(summary)

            G.mode('train')

        final_path = store.run_dir / "final_grammar.pkl.gz"
        with gzip.open(final_path, "wb") as f:
            dill.dump(G, f)

        store.write_status("complete_walk_forward", k=global_iter, extra={
            "final_grammar": str(final_path),
            "n_windows": len(window_summaries),
        })

        return {
            "G": G,
            "store": store,
            "window_summaries": window_summaries,
            "fpc_cache": fpc_cache,
            "status": "complete",
        }

    except Exception:
        err = traceback.format_exc()
        store.log(err)
        store.write_status("crashed_walk_forward", k=global_iter, extra={"error": err})
        raise
