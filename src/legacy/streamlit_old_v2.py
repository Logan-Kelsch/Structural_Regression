"""
streamlit_mcts_wf_dashboard.py

Dashboard for mcts_wf_run.py / mcts_util_v2.py walk-forward MCTS runs.

Run:
    streamlit run streamlit_mcts_wf_dashboard.py

Expected run layout:
    mcts_runs/<run_name>/
        status.json
        wf_config.json
        metrics.jsonl
        wf_window_summaries.jsonl
        wf_chunk_summaries.jsonl
        wf_inference_records.jsonl
        fpc_cache_index.jsonl
        plot_index.jsonl
        console.log
        grammars/*.pkl.gz
        plots/...
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def _nested_stat(row, col, stat):
    """
    read nested stat dictionaries from jsonl-loaded dataframe rows.
    """
    v = row.get(col, None)

    if isinstance(v, dict):
        out = v.get(stat, np.nan)
        return np.nan if out is None else out

    return np.nan


def build_wf_eval_plot_df(records_df):
    """
    Build one row per inference population with i/j/k z-score summaries.
    """

    if records_df is None or records_df.empty:
        return pd.DataFrame()

    rows = []

    for _, r in records_df.iterrows():

        chunk_i = int(r.get("chunk_i", -1))
        chunk_j = int(r.get("chunk_j", -1))
        chunk_k = int(r.get("chunk_k", -1))
        pop_n = int(r.get("pop_n", -1))

        rows.append({
            "chunk_i": chunk_i,
            "chunk_j": chunk_j,
            "chunk_k": chunk_k,
            "window": chunk_i,
            "pop_n": pop_n,

            "i_mean_all": _nested_stat(r, "z_i_stats_all", "mean"),
            "i_max_all": _nested_stat(r, "z_i_stats_all", "max"),
            "j_mean_all": _nested_stat(r, "z_j_stats_all", "mean"),
            "j_max_all": _nested_stat(r, "z_j_stats_all", "max"),
            "k_mean_all": _nested_stat(r, "z_k_stats_all", "mean"),
            "k_max_all": _nested_stat(r, "z_k_stats_all", "max"),

            "j_mean_on_i": _nested_stat(r, "z_j_stats_on_i_success", "mean"),
            "j_max_on_i": _nested_stat(r, "z_j_stats_on_i_success", "max"),
            "k_mean_on_i": _nested_stat(r, "z_k_stats_on_i_success", "mean"),
            "k_max_on_i": _nested_stat(r, "z_k_stats_on_i_success", "max"),
            "k_mean_on_ij": _nested_stat(r, "z_k_stats_on_ij_success", "mean"),
            "k_max_on_ij": _nested_stat(r, "z_k_stats_on_ij_success", "max"),

            "n_genes": r.get("n_genes", np.nan),
            "n_success_i": r.get("n_success_i", np.nan),
            "n_success_j_all": r.get("n_success_j_all", np.nan),
            "n_success_k_all": r.get("n_success_k_all", np.nan),
            "n_success_ij": r.get("n_success_ij", np.nan),
            "n_success_ijk": r.get("n_success_ijk", np.nan),
            "n_k_eval_on_ij": r.get("n_k_eval_on_ij", np.nan),
        })

    return pd.DataFrame(rows)


def aggregate_wf_eval_plot_df(plot_df):
    """
    Aggregate inference population records into one row per walk-forward window.
    """

    if plot_df is None or plot_df.empty:
        return pd.DataFrame()

    agg_cols = {
        "i_mean_all": "mean",
        "j_mean_all": "mean",
        "k_mean_all": "mean",

        "i_max_all": "max",
        "j_max_all": "max",
        "k_max_all": "max",

        "j_mean_on_i": "mean",
        "k_mean_on_i": "mean",
        "k_mean_on_ij": "mean",

        "j_max_on_i": "max",
        "k_max_on_i": "max",
        "k_max_on_ij": "max",

        "n_genes": "sum",
        "n_success_i": "sum",
        "n_success_j_all": "sum",
        "n_success_k_all": "sum",
        "n_success_ij": "sum",
        "n_success_ijk": "sum",
        "n_k_eval_on_ij": "sum",
    }

    out = (
        plot_df
        .groupby(["window", "chunk_i", "chunk_j", "chunk_k"], as_index=False)
        .agg(agg_cols)
    )

    return out

# ---------------------------------------------------------------------
# page setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="MCTS Walk-Forward Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MCTS Walk-Forward Run Dashboard")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _json_load_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _jsonl_rows_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # skip partially-written line during active run
                    continue
    except Exception:
        return rows

    return rows


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


@st.cache_data(ttl=2, show_spinner=False)
def read_json(path_str: str) -> dict:
    return _json_load_safe(Path(path_str))


@st.cache_data(ttl=2, show_spinner=False)
def read_jsonl(path_str: str, flatten: bool = True) -> pd.DataFrame:
    rows = _jsonl_rows_safe(Path(path_str))
    if flatten:
        rows = [_flatten_dict(r) for r in rows]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # light numeric coercion for object columns
    for c in df.columns:
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")
            # keep conversion only if at least one non-null numeric result exists
            if converted.notna().sum() > 0:
                df[c] = converted

    return df


def fmt_num(x: Any, ndigits: int = 4) -> str:
    try:
        if x is None:
            return "—"
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{ndigits}f}"
    except Exception:
        return "—"


def fmt_int(x: Any) -> str:
    try:
        if x is None:
            return "—"
        v = int(x)
        return f"{v:,}"
    except Exception:
        return "—"


def find_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    runs = [p for p in run_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_plot_path(run_dir: Path, p: str | Path) -> Path:
    path = Path(str(p))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    cand = run_dir.parent.parent / path
    if cand.exists():
        return cand
    cand = run_dir.parent / path
    if cand.exists():
        return cand
    cand = run_dir / path
    if cand.exists():
        return cand
    return path


def show_latest_image_by_section(plot_df: pd.DataFrame, run_dir: Path, section: str, caption: str | None = None):
    if plot_df.empty or "section" not in plot_df.columns or "path" not in plot_df.columns:
        st.info(f"No plot index available for {section}.")
        return

    sub = plot_df[plot_df["section"].astype(str).eq(section)].copy()
    if sub.empty:
        st.info(f"No saved plot for section: {section}")
        return

    if "k" in sub.columns:
        sub = sub.sort_values("k")

    row = sub.iloc[-1]
    img_path = resolve_plot_path(run_dir, row["path"])

    if img_path.exists():
        st.image(str(img_path), caption=caption or f"{section} | {img_path.name}", use_container_width=True)
    else:
        st.warning(f"Plot file not found: {img_path}")


def chart_if_cols(df: pd.DataFrame, cols: list[str], x_col: str | None = None, title: str | None = None):
    existing = [c for c in cols if c in df.columns]
    if df.empty or len(existing) == 0:
        st.info("No matching chart columns yet.")
        return

    st.markdown(f"**{title or 'Chart'}**")

    plot_df = df.copy()
    if x_col is not None and x_col in plot_df.columns:
        plot_df = plot_df.set_index(x_col)
    elif "global_iter" in plot_df.columns:
        plot_df = plot_df.set_index("global_iter")
    elif "local_iter" in plot_df.columns:
        plot_df = plot_df.set_index("local_iter")

    st.line_chart(plot_df[existing])

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def build_wf_eval_window_df(infer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one aggregated row per walk-forward window from wf_inference_records.jsonl.

    This expects the dashboard to load wf_inference_records.jsonl with flatten=True,
    so nested stats appear as columns like:
        z_i_stats_all.mean
        z_k_stats_on_ij_success.max
    """

    if infer_df is None or infer_df.empty:
        return pd.DataFrame()

    df = infer_df.copy()

    # Prefer explicit window_id if present. Otherwise chunk_i acts as the window id.
    if "window_id" in df.columns:
        group_cols = ["window_id", "chunk_i", "chunk_j", "chunk_k"]
    else:
        df["window_id"] = df["chunk_i"] if "chunk_i" in df.columns else np.arange(len(df))
        group_cols = ["window_id", "chunk_i", "chunk_j", "chunk_k"]

    # Columns generated by the all-eval inference funnel.
    mean_cols = [
        "z_i_stats_all.mean",
        "z_j_stats_all.mean",
        "z_k_stats_all.mean",
        "z_j_stats_on_i_success.mean",
        "z_k_stats_on_i_success.mean",
        "z_k_stats_on_ij_success.mean",
    ]

    max_cols = [
        "z_i_stats_all.max",
        "z_j_stats_all.max",
        "z_k_stats_all.max",
        "z_j_stats_on_i_success.max",
        "z_k_stats_on_i_success.max",
        "z_k_stats_on_ij_success.max",
    ]

    count_cols = [
        "n_genes",
        "n_success_i",
        "n_success_j_all",
        "n_success_k_all",
        "n_success_ij",
        "n_success_ijk",
        "n_k_eval_on_ij",
    ]

    agg = {}

    for c in mean_cols:
        if c in df.columns:
            agg[c] = "mean"

    for c in max_cols:
        if c in df.columns:
            agg[c] = "max"

    for c in count_cols:
        if c in df.columns:
            agg[c] = "sum"

    if len(agg) == 0:
        return pd.DataFrame()

    out = df.groupby(group_cols, as_index=False).agg(agg)

    return out


def plot_wf_ijk_eval_panels(eval_window_df: pd.DataFrame):
    """
    Plot i/j/k walk-forward inference performance.

    This belongs in the Walk-forward funnel tab.
    """

    if eval_window_df is None or eval_window_df.empty:
        st.info("No i/j/k inference evaluation records available yet.")
        return

    x_col = "window_id"
    x = eval_window_df[x_col].to_numpy()

    st.markdown("**All-gene mean z-score by walk-forward window**")

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for c, label in [
        ("z_i_stats_all.mean", "chunk i mean z"),
        ("z_j_stats_all.mean", "chunk j mean z"),
        ("z_k_stats_all.mean", "chunk k mean z"),
    ]:
        if c in eval_window_df.columns:
            ax.plot(x, eval_window_df[c], marker="o", label=label)

    ax.axhline(0, alpha=0.35)
    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_title("Mean z-score across all generated genes")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("mean z-score")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)


    st.markdown("**Best-gene z-score by walk-forward window**")

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for c, label in [
        ("z_i_stats_all.max", "chunk i max z"),
        ("z_j_stats_all.max", "chunk j max z"),
        ("z_k_stats_all.max", "chunk k max z"),
    ]:
        if c in eval_window_df.columns:
            ax.plot(x, eval_window_df[c], marker="o", label=label)

    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_title("Max z-score across all generated genes")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("max z-score")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)


    st.markdown("**Success-count funnel**")

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for c, label in [
        ("n_success_i", "i success"),
        ("n_success_j_all", "j success, all genes"),
        ("n_success_k_all", "k success, all genes"),
        ("n_success_ij", "i+j success"),
        ("n_success_ijk", "i+j+k success"),
    ]:
        if c in eval_window_df.columns:
            ax.plot(x, eval_window_df[c], marker="o", label=label)

    ax.set_title("Success counts through i/j/k masks")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)


    st.markdown("**Chunk k performance conditioned on earlier success**")

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for c, label in [
        ("z_k_stats_all.mean", "k mean, all genes"),
        ("z_k_stats_on_i_success.mean", "k mean, i-success genes"),
        ("z_k_stats_on_ij_success.mean", "k mean, i+j-success genes"),
    ]:
        if c in eval_window_df.columns:
            ax.plot(x, eval_window_df[c], marker="o", label=label)

    ax.axhline(0, alpha=0.35)
    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_title("Forward chunk k score conditional on earlier masks")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("mean z-score on chunk k")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Aggregated i/j/k evaluation table**")
    st.dataframe(eval_window_df, use_container_width=True)


def size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Run selection")

    default_root = "mcts_runs"
    run_root_str = st.text_input("Run root", value=default_root)
    run_root = Path(run_root_str).expanduser()

    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_sec = st.slider("Refresh seconds", min_value=1, max_value=30, value=5)

    runs = find_runs(run_root)

    if len(runs) == 0:
        st.warning(f"No run folders found under {run_root.resolve() if run_root.exists() else run_root}")
        st.stop()

    run_labels = [p.name for p in runs]
    selected_label = st.selectbox("Run", run_labels, index=0)
    run_dir = runs[run_labels.index(selected_label)]

    st.caption(f"Selected: `{run_dir}`")

    if st.button("Manual refresh"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------
# load run files
# ---------------------------------------------------------------------

status = read_json(str(run_dir / "status.json"))
config = read_json(str(run_dir / "wf_config.json"))
metrics_df = read_jsonl(str(run_dir / "metrics.jsonl"), flatten=True)
window_df = read_jsonl(str(run_dir / "wf_window_summaries.jsonl"), flatten=True)
chunk_df = read_jsonl(str(run_dir / "wf_chunk_summaries.jsonl"), flatten=True)
infer_df = read_jsonl(str(run_dir / "wf_inference_records.jsonl"), flatten=True)
fpc_df = read_jsonl(str(run_dir / "fpc_cache_index.jsonl"), flatten=True)
plot_df = read_jsonl(str(run_dir / "plot_index.jsonl"), flatten=True)
decay_df = read_jsonl(str(run_dir / "wf_decay.jsonl"), flatten=True)



# ---------------------------------------------------------------------
# top summary
# ---------------------------------------------------------------------

st.subheader("Current status")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Status", status.get("status", "unknown"))
with c2:
    st.metric("Last iter", fmt_int(status.get("last_iteration")))
with c3:
    st.metric("Window", fmt_int(status.get("window_id")))
with c4:
    st.metric("Local iter", fmt_int(status.get("local_iter")))
with c5:
    st.metric("policy h", fmt_num(status.get("policy_h")))
with c6:
    st.metric("delta_L", fmt_num(status.get("delta_L", config.get("delta_L"))))

c7, c8, c9, c10 = st.columns(4)
with c7:
    st.metric("chunk i", fmt_int(status.get("chunk_i")))
with c8:
    st.metric("chunk j", fmt_int(status.get("chunk_j")))
with c9:
    st.metric("chunk k", fmt_int(status.get("chunk_k")))
with c10:
    st.metric("grammar gamma", fmt_num(status.get("grammar_memory_gamma", config.get("grammar_memory_gamma"))))

with st.expander("Raw status / config", expanded=False):
    st.json({"status": status, "config": config})


# ---------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------

tabs = st.tabs([
    "Training progress",
    "Walk-forward funnel",
    "FPC cache",
    "Plots",
    "Grammars / files",
    "Raw tables",
    "Console log",
])


# ---------------------------------------------------------------------
# tab 1 training
# ---------------------------------------------------------------------

with tabs[0]:
    st.subheader("Training progress")

    if metrics_df.empty:
        st.info("No metrics yet. The run may still be initializing.")
    else:
        last = metrics_df.iloc[-1].to_dict()
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Rows", fmt_int(len(metrics_df)))
        with m2:
            st.metric("Last h", fmt_num(last.get("policy_h")))
        with m3:
            st.metric("Last total drift", fmt_num(last.get("policy_total_drift")))
        with m4:
            st.metric("Node states", fmt_int(last.get("mcts_node_mu_count")))
        with m5:
            st.metric("Edge states", fmt_int(last.get("mcts_edge_mu_count")))

        chart_if_cols(
            metrics_df,
            ["policy_h", "policy_total_drift", "policy_parent_drift", "policy_child_drift", "delta_L"],
            x_col="global_iter",
            title="Policy drift / convergence",
        )

        chart_if_cols(
            metrics_df,
            ["mcts_node_mu_count", "mcts_edge_mu_count", "alpha_decision_mu_count", "alpha_node_mu_count", "alpha_edge_mu_count"],
            x_col="global_iter",
            title="Grammar memory growth",
        )

        chart_if_cols(
            metrics_df,
            ["zscore_stats_j.mean", "zscore_stats_j.max", "zscore_stats_j.std"],
            x_col="global_iter",
            title="Training/eval chunk j z-score stats",
        )

        chart_if_cols(
            metrics_df,
            ["elapsed_sec"],
            x_col="global_iter",
            title="Iteration wall time",
        )

        st.markdown("**Latest metrics**")
        st.dataframe(metrics_df.tail(25), use_container_width=True)


# ---------------------------------------------------------------------
# tab 2 walk-forward funnel
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Walk-forward inference funnel")

    if window_df.empty:
        st.info("No completed walk-forward windows yet.")
    else:
        lastw = window_df.iloc[-1].to_dict()
        w1, w2, w3, w4, w5, w6 = st.columns(6)
        with w1:
            st.metric("Completed windows", fmt_int(len(window_df)))
        with w2:
            st.metric("Latest i", fmt_int(lastw.get("chunk_i")))
        with w3:
            st.metric("Latest j", fmt_int(lastw.get("chunk_j")))
        with w4:
            st.metric("Latest k", fmt_int(lastw.get("chunk_k")))
        with w5:
            st.metric("ij success count", fmt_int(lastw.get("ij_success_count")))
        with w6:
            st.metric("k mean z", fmt_num(lastw.get("k_score_stats.mean")))

        chart_if_cols(
            window_df,
            ["i_success_count", "ij_success_count", "k_score_stats.n"],
            x_col="window_id",
            title="Success funnel counts",
        )

        chart_if_cols(
            window_df,
            ["k_score_stats.mean", "k_score_stats.max", "k_score_stats.std"],
            x_col="window_id",
            title="Chunk-k out-of-sample z-score performance",
        )

        chart_if_cols(
            window_df,
            ["final_h", "delta_L"],
            x_col="window_id",
            title="Final convergence by window",
        )

        st.markdown("**Window summaries**")
        st.dataframe(window_df.tail(30), use_container_width=True)

    if not infer_df.empty:
        st.markdown("**Inference population records**")
        cols = [c for c in [
            "window_id", "pop_n", "chunk_i", "chunk_j", "chunk_k",
            "n_candidates", "i_success_count", "ij_success_count",
            "k_score_stats.mean", "k_score_stats.max", "k_score_stats.n",
        ] if c in infer_df.columns]
        st.dataframe(infer_df[cols].tail(100) if cols else infer_df.tail(100), use_container_width=True)

        st.divider()
        st.subheader("i/j/k chunk evaluation performance")

        eval_window_df = build_wf_eval_window_df(infer_df)
        plot_wf_ijk_eval_panels(eval_window_df)

        chart_if_cols(
            infer_df,
            ["i_success_count", "ij_success_count", "k_score_stats.n"],
            x_col=None,
            title="Per-infer-population success counts",
        )


# ---------------------------------------------------------------------
# tab 3 fpc cache
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("FPC cache")

    if fpc_df.empty:
        st.info("No FPC cache entries yet.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            st.metric("FPC fits", fmt_int(len(fpc_df)))
        with f2:
            st.metric("Unique chunks", fmt_int(fpc_df["chunk_num"].nunique() if "chunk_num" in fpc_df else None))
        with f3:
            st.metric("Unique tags", fmt_int(fpc_df["tag"].nunique() if "tag" in fpc_df else None))

        if "tag" in fpc_df.columns and "chunk_num" in fpc_df.columns:
            pivot = pd.crosstab(fpc_df["chunk_num"], fpc_df["tag"])
            st.markdown("**Cached chunks by tag**")
            st.dataframe(pivot, use_container_width=True)

        st.dataframe(fpc_df.tail(100), use_container_width=True)

    fpc_dir = run_dir / "fpc_cache"
    if fpc_dir.exists():
        files = sorted(fpc_dir.glob("*"), key=lambda p: p.name)
        file_df = pd.DataFrame([{
            "file": p.name,
            "size_mb": size_mb(p),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        } for p in files])
        st.markdown("**FPC cache files**")
        st.dataframe(file_df, use_container_width=True)


# ---------------------------------------------------------------------
# tab 4 plots
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("Saved plots")

    if plot_df.empty:
        st.info("No saved plots yet.")
    else:
        sections = sorted(plot_df["section"].dropna().astype(str).unique()) if "section" in plot_df.columns else []
        default_sections = [s for s in ["wf_policy_drift", "wf_state_growth", "wf_k_scores"] if s in sections]

        selected_sections = st.multiselect(
            "Plot sections to show latest",
            sections,
            default=default_sections if default_sections else sections[: min(4, len(sections))],
        )

        for section in selected_sections:
            show_latest_image_by_section(plot_df, run_dir, section)

        with st.expander("Plot index", expanded=False):
            st.dataframe(plot_df.tail(200), use_container_width=True)


# ---------------------------------------------------------------------
# tab 5 grammars/files
# ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("Run files")

    grammar_dir = run_dir / "grammars"
    grammar_files = sorted(grammar_dir.glob("*.pkl.gz"), key=lambda p: p.stat().st_mtime) if grammar_dir.exists() else []

    final_grammar = run_dir / "final_grammar.pkl.gz"
    files = []
    for p in grammar_files:
        files.append({
            "kind": "window_grammar",
            "file": str(p),
            "size_mb": size_mb(p),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        })
    if final_grammar.exists():
        files.append({
            "kind": "final_grammar",
            "file": str(final_grammar),
            "size_mb": size_mb(final_grammar),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(final_grammar.stat().st_mtime)),
        })

    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True)
    else:
        st.info("No grammar snapshots saved yet.")

    important = [
        "status.json",
        "wf_config.json",
        "metrics.jsonl",
        "wf_window_summaries.jsonl",
        "wf_inference_records.jsonl",
        "fpc_cache_index.jsonl",
        "console.log",
    ]
    file_rows = []
    for name in important:
        p = run_dir / name
        file_rows.append({
            "file": name,
            "exists": p.exists(),
            "size_mb": size_mb(p) if p.exists() else 0.0,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None,
        })
    st.markdown("**Core run files**")
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)


# ---------------------------------------------------------------------
# tab 6 raw tables
# ---------------------------------------------------------------------

with tabs[5]:
    st.subheader("Raw tables")

    table_choice = st.selectbox(
        "Table",
        [
            "metrics",
            "window summaries",
            "chunk summaries",
            "inference records",
            "fpc cache index",
            "decay records",
            "plot index",
        ],
    )

    table_map = {
        "metrics": metrics_df,
        "window summaries": window_df,
        "chunk summaries": chunk_df,
        "inference records": infer_df,
        "fpc cache index": fpc_df,
        "decay records": decay_df,
        "plot index": plot_df,
    }
    df = table_map[table_choice]

    if df.empty:
        st.info("Selected table is empty.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_label}_{table_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------
# tab 7 console
# ---------------------------------------------------------------------

with tabs[6]:
    st.subheader("Console log")

    log_path = run_dir / "console.log"
    if not log_path.exists():
        st.info("No console.log yet.")
    else:
        max_lines = st.slider("Tail lines", min_value=50, max_value=2000, value=300, step=50)
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-max_lines:])
            st.text_area("console.log tail", value=tail, height=600)
        except Exception as e:
            st.error(f"Could not read console log: {e}")


# ---------------------------------------------------------------------
# auto-refresh
# ---------------------------------------------------------------------

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
