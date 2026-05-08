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
import gc
import json
import time
import gzip
import pickle
import shutil
import traceback
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

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if P.shape[0] == 0:
        axs[0].text(0.5, 0.5, "no probability history yet", ha="center", va="center")
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        return maybe_show(fig, show)

    im0 = axs[0].imshow(P.T, aspect="auto", origin="lower")
    axs[0].set_title(title_left)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("depth")
    fig.colorbar(im0, ax=axs[0])

    if P.shape[0] > 1:
        D = np.abs(np.diff(P, axis=0))

        im1 = axs[1].imshow(D.T, aspect="auto", origin="lower")
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


def plot_mcts_exploration_influence(influence_hist, k=0.01, figsize=(12, 4), show: bool = False):
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

    ax.plot(x, hp, linewidth=2, label="Hawkes")
    ax.axhline(0.55, linestyle="--", alpha=0.5, label="near-freeze threshold")
    ax.axhline(0.25, linestyle="--", alpha=0.5, label="explore threshold")

    ax.set_title("UCT exploration influence from softmax ablation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("policy influence")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    return maybe_show(fig, show)


def plot_mcts_policy_drift_hawkes(
    hp,
    raw_hist=None,
    parent_hist=None,
    child_hist=None,
    figsize=(12, 4),
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if raw_hist is not None and len(raw_hist) > 0:
        ax.plot(raw_hist, alpha=0.35, label="raw total drift")

    if parent_hist is not None and len(parent_hist) > 0:
        ax.plot(parent_hist, alpha=0.25, label="parent drift")

    if child_hist is not None and len(child_hist) > 0:
        ax.plot(child_hist, alpha=0.25, label="child drift")

    if hp is not None and len(hp) > 0:
        ax.plot(hp, linewidth=3, label="smoothed drift h")

    ax.axhline(0.05, linestyle="--", alpha=0.40, label="mostly stable")
    ax.axhline(0.05, linestyle="--", alpha=0.40, label="stable")

    ax.set_title("hawkes/EMA smoothing on root-weighted MCTS policy delta")
    ax.set_xlabel("iteration")
    ax.set_ylabel("policy drift")
    ax.set_ylim(1e-2, 1)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

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
        latest = np.asarray(child_depth[-1], dtype=np.float64)
        axs[4].bar(np.arange(len(latest)), latest)
        axs[4].set_title("latest child depth probabilities")
        axs[4].set_xlabel("depth")

    infl = histories.get("g_mcts_uct_explore_influence", [])
    infl_ema = histories.get("g_mcts_uct_explore_influence_ema", [])

    if len(infl) > 0:
        axs[5].plot(infl, label="raw")
        axs[5].plot(infl_ema, label="ema")
        axs[5].set_title("UCT explore influence")
        axs[5].legend(fontsize=8)

    h_val = histories.get("h", np.nan)
    freeze = histories.get("freeze_expansion", None)

    fig.suptitle(f"iteration={k} | h={h_val} | freeze={freeze}", fontsize=11)

    return maybe_show(fig, show)


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
        k=-np.log(0.5),
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

        "wf_windows"  : 500,
        "verbose"     : 0,
    }


def default_solver_kwargs():
    return {
        "offset"   : 12,
        "t_vec"    : "Close",
        "t_mode"   : "AD",
        "emission" : [
            {"ID": 18, "x": "tvec", "delta1": 12, "min_count": 2},
            {"ID": "divide"},
            {"ID": 18, "x": "tvec", "offset": False, "delta1": 12, "min_count": 2},
            {"ID": 5, "alpha": 1.0},
        ],
        "AD_cond"  : ("gt", 0),
    }


def default_logwalker_kwargs():
    return {
        "start": 0,
        "destination": 0.45,
        "steps": 12,
        "exhaust_mode": "steps",
        "exwhen": 100,
        "min_walk": 3,
        "log_base": 2,
        "nest_mode": "half",
    }


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
    break_h_threshold: float = 0.01,
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
    some_kappa = -np.log(0.5)

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
                    chunk_num=chunk_num,
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
                        chunk_num=chunk_num,
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
                    chunk_num=chunk_num,
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

            pw_mode, infl_ema = _E.mcts_update_pw_mode_from_ablation(
                G=G,
                explore_influence=uct_infl,
                freeze_threshold=0.55,
                unfreeze_threshold=0.25,
                ema_alpha=0.10,
            )

            histories["g_mcts_uct_explore_influence"].append(uct_infl)
            histories["g_mcts_uct_explore_influence_ema"].append(infl_ema)
            histories["g_mcts_pw_modes"].append(pw_mode)

            with capture_console(store, "original_exploration_influence_plot", k=k), capture_plots(
                store, "original_exploration_influence", k=k, save=save_helper_plots
            ):
                _E.plot_mcts_exploration_influence(
                    influence_hist=histories["g_mcts_uct_explore_influence"],
                    k=-np.log(0.5),
                    figsize=(12, 4),
                )

            # policy delta
            curr_snap = _E.mcts_policy_snapshot(
                G=G,
                X=X,
                depth_gamma=0.70,
            )

            policy_drift = _E.mcts_root_weighted_policy_drift(
                prev_snap=mcts_prev_policy_snapshot,
                curr_snap=curr_snap,
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

            if k == 5:
                store.log("freezing MCTS expansion")
                G.freeze_mcts_expansion(True)

            if k == 50:
                store.log("reopening MCTS expansion")
                G.freeze_mcts_expansion(False)

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

                "depth_prob_delta_sum": depth_prob_delta_sum,
                "alpha_depth_prob_delta_sum": alpha_depth_prob_delta_sum,
                "alpha_sensor_prob": alpha_sensor_prob,

                "uct_explore_influence": uct_infl,
                "uct_explore_influence_ema": infl_ema,
                "pw_mode": pw_mode,
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
            #diag.end_loop_diag(k, 1)
            diag.end_loop_var_growth_diag(
                k=k,
                global_ns=globals(),
                local_ns=locals(),
                scan_every=5,
                top_n=25,
                min_mb=1.0,
                depth=2,
                max_items=100,
                trim_test=False,
            )

            #print(f'Ending iteration {k}')

            if k + 1 > 5 and np.isfinite(h) and h < break_h_threshold:
                store.log(f"breaking because h={h:.6f} < {break_h_threshold}")
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
