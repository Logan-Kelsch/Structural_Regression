"""
streamlit_mcts_wf_dashboard_all_eval.py

Dashboard for the all-evaluate-then-mask walk-forward MCTS run.

Run:
    streamlit run streamlit_mcts_wf_dashboard_all_eval.py

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
        wf_arrays/*.npz
        grammars/*.pkl.gz
        plots/...
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="MCTS WF All-Eval Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MCTS Walk-Forward Dashboard | all-evaluate-then-mask")


# ---------------------------------------------------------------------
# basic readers
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
                    # active writes can leave a partial last line
                    continue
    except Exception:
        return rows
    return rows


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        kk = str(k) if prefix == "" else f"{prefix}.{k}"
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
    for c in df.columns:
        if df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")
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
        return f"{int(x):,}"
    except Exception:
        return "—"


def size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except Exception:
        return 0.0


def find_runs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    runs = [p for p in run_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_plot_path(run_dir: Path, p: str | Path) -> Path:
    path = Path(str(p))
    if path.is_absolute() and path.exists():
        return path
    for cand in [path, run_dir / path, run_dir.parent / path, run_dir.parent.parent / path]:
        try:
            if cand.exists():
                return cand
        except Exception:
            pass
    return path


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


# ---------------------------------------------------------------------
# all-eval walk-forward aggregation + plots
# ---------------------------------------------------------------------

def build_wf_eval_window_df(infer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate wf_inference_records.jsonl into one row per walk-forward window.

    This expects the all-eval record fields, e.g.:
        z_i_stats_all.mean
        z_j_stats_all.mean
        z_k_stats_all.mean
        z_k_stats_on_ij_success.mean
        n_success_i
        n_success_ij
        n_success_ijk
    """
    if infer_df is None or infer_df.empty:
        return pd.DataFrame()

    df = infer_df.copy()
    if "window_id" not in df.columns:
        if "chunk_i" in df.columns:
            df["window_id"] = df["chunk_i"]
        else:
            df["window_id"] = 0

    group_cols = [c for c in ["window_id", "chunk_i", "chunk_j", "chunk_k"] if c in df.columns]

    sum_cols = [
        "n_genes",
        "n_i_eval_all",
        "n_j_eval_all",
        "n_k_eval_all",
        "n_success_i",
        "n_success_j_all",
        "n_success_k_all",
        "n_success_ij",
        "n_success_ijk",
        "n_k_eval_on_i",
        "n_k_eval_on_ij",
    ]
    mean_cols = [
        "z_i_stats_all.mean",
        "z_j_stats_all.mean",
        "z_k_stats_all.mean",
        "z_j_stats_on_i_success.mean",
        "z_k_stats_on_i_success.mean",
        "z_k_stats_on_ij_success.mean",
        "z_k_stats_on_ijk_success.mean",
        "p_success_i",
        "p_success_j_all",
        "p_success_k_all",
        "p_success_ij",
        "p_success_ijk",
    ]
    max_cols = [
        "z_i_stats_all.max",
        "z_j_stats_all.max",
        "z_k_stats_all.max",
        "z_j_stats_on_i_success.max",
        "z_k_stats_on_i_success.max",
        "z_k_stats_on_ij_success.max",
        "z_k_stats_on_ijk_success.max",
    ]
    min_cols = [
        "z_i_stats_all.min",
        "z_j_stats_all.min",
        "z_k_stats_all.min",
    ]

    agg = {}
    for c in sum_cols:
        if c in df.columns:
            agg[c] = "sum"
    for c in mean_cols:
        if c in df.columns:
            agg[c] = "mean"
    for c in max_cols:
        if c in df.columns:
            agg[c] = "max"
    for c in min_cols:
        if c in df.columns:
            agg[c] = "min"

    if len(agg) == 0:
        return pd.DataFrame()

    out = df.groupby(group_cols, as_index=False).agg(agg)

    # recompute rates from summed counts when possible
    if "n_genes" in out.columns and out["n_genes"].replace(0, np.nan).notna().any():
        denom = out["n_genes"].replace(0, np.nan)
        for count_col, rate_col in [
            ("n_success_i", "rate_success_i"),
            ("n_success_j_all", "rate_success_j_all"),
            ("n_success_k_all", "rate_success_k_all"),
            ("n_success_ij", "rate_success_ij"),
            ("n_success_ijk", "rate_success_ijk"),
        ]:
            if count_col in out.columns:
                out[rate_col] = out[count_col] / denom

    return out


def plot_wf_eval_panels(eval_df: pd.DataFrame):
    if eval_df is None or eval_df.empty:
        st.info("No all-eval inference records available yet.")
        return

    x = eval_df["window_id"].to_numpy() if "window_id" in eval_df.columns else np.arange(len(eval_df))

    st.markdown("**All-gene mean z-score across i, j, k**")
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for c, label in [
        ("z_i_stats_all.mean", "chunk i mean z"),
        ("z_j_stats_all.mean", "chunk j mean z"),
        ("z_k_stats_all.mean", "chunk k mean z"),
    ]:
        if c in eval_df.columns:
            ax.plot(x, eval_df[c], marker="o", label=label)
    ax.axhline(0, alpha=0.35)
    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("mean z-score")
    ax.set_title("Mean generated-gene performance on each chunk")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Best generated gene by chunk**")
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for c, label in [
        ("z_i_stats_all.max", "chunk i max z"),
        ("z_j_stats_all.max", "chunk j max z"),
        ("z_k_stats_all.max", "chunk k max z"),
    ]:
        if c in eval_df.columns:
            ax.plot(x, eval_df[c], marker="o", label=label)
    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("max z-score")
    ax.set_title("Maximum generated-gene performance on each chunk")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Success counts after evaluating every gene on i, j, and k**")
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for c, label in [
        ("n_success_i", "i success"),
        ("n_success_j_all", "j success, all genes"),
        ("n_success_k_all", "k success, all genes"),
        ("n_success_ij", "i+j success"),
        ("n_success_ijk", "i+j+k success"),
    ]:
        if c in eval_df.columns:
            ax.plot(x, eval_df[c], marker="o", label=label)
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("count")
    ax.set_title("Success-count funnel with post-evaluation masks")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Success rates after evaluating every gene on i, j, and k**")
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for c, label in [
        ("rate_success_i", "P(i success)"),
        ("rate_success_j_all", "P(j success)"),
        ("rate_success_k_all", "P(k success)"),
        ("rate_success_ij", "P(i+j success)"),
        ("rate_success_ijk", "P(i+j+k success)"),
    ]:
        if c in eval_df.columns:
            ax.plot(x, eval_df[c], marker="o", label=label)
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Probability of generated-gene success")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Forward chunk-k performance under earlier masks**")
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for c, label in [
        ("z_k_stats_all.mean", "k mean, all genes"),
        ("z_k_stats_on_i_success.mean", "k mean | i success"),
        ("z_k_stats_on_ij_success.mean", "k mean | i+j success"),
        ("z_k_stats_on_ijk_success.mean", "k mean | i+j+k success"),
    ]:
        if c in eval_df.columns:
            ax.plot(x, eval_df[c], marker="o", label=label)
    ax.axhline(0, alpha=0.35)
    ax.axhline(2, linestyle="--", alpha=0.5, label="success z=2")
    ax.set_xlabel("walk-forward window / chunk i")
    ax.set_ylabel("mean z-score on k")
    ax.set_title("Forward transfer: chunk-k score conditional on earlier success")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)


def show_latest_plot_sections(plot_df: pd.DataFrame, run_dir: Path, sections: list[str], title: str):
    st.markdown(f"**{title}**")
    if plot_df.empty or "section" not in plot_df.columns or "path" not in plot_df.columns:
        st.info("No saved plots yet.")
        return
    for section in sections:
        sub = plot_df[plot_df["section"].astype(str).eq(section)].copy()
        if sub.empty:
            continue
        if "k" in sub.columns:
            sub = sub.sort_values("k")
        row = sub.iloc[-1]
        p = resolve_plot_path(run_dir, row["path"])
        if p.exists():
            st.image(str(p), caption=f"{section} | {p.name}", width='stretch')


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Run selection")
    run_root_str = st.text_input("Run root", value="mcts_runs")
    run_root = Path(run_root_str).expanduser()
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_sec = st.slider("Refresh seconds", min_value=1, max_value=30, value=5)

    runs = find_runs(run_root)
    if len(runs) == 0:
        st.warning(f"No run folders found under {run_root}")
        st.stop()

    labels = [p.name for p in runs]
    selected = st.selectbox("Run", labels, index=0)
    run_dir = runs[labels.index(selected)]
    st.caption(f"Selected: `{run_dir}`")

    if st.button("Manual refresh"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------
# load files
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
# summary row
# ---------------------------------------------------------------------

st.subheader("Current status")
cols = st.columns(8)
with cols[0]:
    st.metric("Status", status.get("status", "unknown"))
with cols[1]:
    st.metric("Last iter", fmt_int(status.get("last_iteration")))
with cols[2]:
    st.metric("Window", fmt_int(status.get("window_id")))
with cols[3]:
    st.metric("Local iter", fmt_int(status.get("local_iter")))
with cols[4]:
    st.metric("chunk i", fmt_int(status.get("chunk_i")))
with cols[5]:
    st.metric("chunk j", fmt_int(status.get("chunk_j")))
with cols[6]:
    st.metric("chunk k", fmt_int(status.get("chunk_k")))
with cols[7]:
    st.metric("policy h", fmt_num(status.get("policy_h")))

cols = st.columns(4)
with cols[0]:
    st.metric("delta_L", fmt_num(status.get("delta_L", config.get("delta_L"))))
with cols[1]:
    st.metric("grammar gamma", fmt_num(status.get("grammar_memory_gamma", config.get("grammar_memory_gamma"))))
with cols[2]:
    st.metric("infer pops", fmt_int(config.get("infer_populations")))
with cols[3]:
    st.metric("success z", fmt_num(config.get("success_z")))

with st.expander("Raw status / config", expanded=False):
    st.json({"status": status, "config": config})


tabs = st.tabs([
    "Training progress",
    "WF i/j/k performance",
    "Saved inference plots",
    "FPC cache",
    "Files",
    "Raw tables",
    "Console log",
])


# ---------------------------------------------------------------------
# training tab
# ---------------------------------------------------------------------

with tabs[0]:
    st.subheader("Training progress")
    if metrics_df.empty:
        st.info("No training metrics yet.")
    else:
        last = metrics_df.iloc[-1].to_dict()
        m = st.columns(6)
        with m[0]: st.metric("Rows", fmt_int(len(metrics_df)))
        with m[1]: st.metric("Last h", fmt_num(last.get("policy_h")))
        with m[2]: st.metric("Total drift", fmt_num(last.get("policy_total_drift")))
        with m[3]: st.metric("j mean z", fmt_num(last.get("zscore_stats_j.mean")))
        with m[4]: st.metric("j max z", fmt_num(last.get("zscore_stats_j.max")))
        with m[5]: st.metric("node states", fmt_int(last.get("mcts_node_mu_count")))

        chart_if_cols(metrics_df, ["policy_h", "policy_total_drift", "policy_parent_drift", "policy_child_drift", "delta_L"], x_col="global_iter", title="Training exploit-policy drift")
        chart_if_cols(metrics_df, ["zscore_stats_j.mean", "zscore_stats_j.max", "zscore_stats_j.std"], x_col="global_iter", title="Training/eval chunk-j z-score stats")
        chart_if_cols(metrics_df, ["mcts_node_mu_count", "mcts_edge_mu_count", "alpha_decision_mu_count", "alpha_node_mu_count", "alpha_edge_mu_count"], x_col="global_iter", title="Grammar memory growth")
        chart_if_cols(metrics_df, ["elapsed_sec"], x_col="global_iter", title="Iteration time")

        show_latest_plot_sections(plot_df, run_dir, ["wf_policy_drift", "wf_state_growth"], "Latest saved training plots")
        st.markdown("**Latest training rows**")
        st.dataframe(metrics_df.tail(50), width='stretch')


# ---------------------------------------------------------------------
# all-eval performance tab
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Walk-forward i/j/k performance")
    eval_window_df = build_wf_eval_window_df(infer_df)

    if eval_window_df.empty:
        st.info("No all-eval inference records yet.")
    else:
        c = st.columns(6)
        last = eval_window_df.iloc[-1].to_dict()
        with c[0]: st.metric("windows", fmt_int(eval_window_df["window_id"].nunique() if "window_id" in eval_window_df else len(eval_window_df)))
        with c[1]: st.metric("latest total genes", fmt_int(last.get("n_genes")))
        with c[2]: st.metric("i+j success", fmt_int(last.get("n_success_ij")))
        with c[3]: st.metric("i+j+k success", fmt_int(last.get("n_success_ijk")))
        with c[4]: st.metric("k mean all", fmt_num(last.get("z_k_stats_all.mean")))
        with c[5]: st.metric("k mean | i+j", fmt_num(last.get("z_k_stats_on_ij_success.mean")))

        plot_wf_eval_panels(eval_window_df)

        st.markdown("**Aggregated all-eval window table**")
        st.dataframe(eval_window_df.tail(50), width='stretch')

    if not infer_df.empty:
        st.markdown("**Per inference-population records**")
        cols = [c for c in [
            "window_id", "pop_n", "chunk_i", "chunk_j", "chunk_k", "n_genes",
            "n_success_i", "n_success_j_all", "n_success_k_all", "n_success_ij", "n_success_ijk",
            "z_i_stats_all.mean", "z_j_stats_all.mean", "z_k_stats_all.mean",
            "z_k_stats_on_ij_success.mean",
        ] if c in infer_df.columns]
        st.dataframe(infer_df[cols].tail(200) if cols else infer_df.tail(200), width='stretch')


# ---------------------------------------------------------------------
# saved inference plots
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("Saved inference distribution plots")
    show_latest_plot_sections(
        plot_df,
        run_dir,
        ["wf_ijk_all_z_dists", "wf_k_masked_z_dists", "wf_success_counts_rates", "wf_transfer_scatter", "wf_k_scores"],
        "Latest all-eval inference plots",
    )

    # ------------------------------------------------------------
    # sequential success-flow plots
    #
    # These are saved directly under:
    #     plots/wf_success_flow/*.png
    # They may not appear in plot_index.jsonl, so we browse the folder directly.
    # ------------------------------------------------------------

    st.divider()
    st.subheader("Sequential success-flow plots")

    success_flow_dir = run_dir / "plots" / "wf_success_flow"

    if not success_flow_dir.exists():
        st.info("No success-flow plots saved yet. Expected folder: plots/wf_success_flow")
    else:
        flow_plots = sorted(
            success_flow_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if len(flow_plots) == 0:
            st.info("No success-flow PNGs found yet in plots/wf_success_flow.")
        else:
            selected_flow_plot = st.selectbox(
                "Select success-flow plot",
                flow_plots,
                format_func=lambda p: p.name,
            )

            st.image(
                str(selected_flow_plot),
                caption=f"success flow | {selected_flow_plot.name}",
                width='stretch',
            )

            with st.expander("Show all success-flow plot files", expanded=False):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "file": p.name,
                            "path": str(p),
                            "size_mb": size_mb(p),
                            "modified": time.strftime(
                                "%Y-%m-%d %H:%M:%S",
                                time.localtime(p.stat().st_mtime),
                            ),
                        }
                        for p in flow_plots
                    ]),
                    width='stretch',
                )

    st.divider()
    st.subheader("Plot index browser")

    if not plot_df.empty and "section" in plot_df.columns:
        sections = sorted(plot_df["section"].dropna().astype(str).unique())
        selected_sections = st.multiselect("Plot sections", sections, default=[s for s in ["wf_ijk_all_z_dists", "wf_k_masked_z_dists", "wf_success_counts_rates", "wf_transfer_scatter"] if s in sections])
        if selected_sections:
            for section in selected_sections:
                sub = plot_df[plot_df["section"].astype(str).eq(section)].copy()
                if "k" in sub.columns:
                    sub = sub.sort_values("k")
                for _, row in sub.tail(5).iterrows():
                    p = resolve_plot_path(run_dir, row["path"])
                    if p.exists():
                        st.image(str(p), caption=f"{section} | {p.name}", width='stretch')


# ---------------------------------------------------------------------
# fpc cache
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("FPC cache")
    if fpc_df.empty:
        st.info("No FPC cache entries yet.")
    else:
        c = st.columns(4)
        with c[0]: st.metric("cache events", fmt_int(len(fpc_df)))
        with c[1]: st.metric("unique chunks", fmt_int(fpc_df["chunk_num"].nunique() if "chunk_num" in fpc_df else None))
        with c[2]: st.metric("unique tags", fmt_int(fpc_df["tag"].nunique() if "tag" in fpc_df else None))
        with c[3]: st.metric("failures", fmt_int((fpc_df["event"].astype(str).eq("fit_failed")).sum() if "event" in fpc_df else None))
        if "tag" in fpc_df.columns and "chunk_num" in fpc_df.columns:
            st.markdown("**Events by chunk/tag**")
            st.dataframe(pd.crosstab(fpc_df["chunk_num"], fpc_df["tag"]), width='stretch')
        st.dataframe(fpc_df.tail(200), width='stretch')


# ---------------------------------------------------------------------
# files
# ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("Run files")
    grammar_dir = run_dir / "grammars"
    grammar_files = sorted(grammar_dir.glob("*.pkl.gz"), key=lambda p: p.stat().st_mtime) if grammar_dir.exists() else []
    final_grammar = run_dir / "final_grammar.pkl.gz"

    rows = []
    for p in grammar_files:
        rows.append({"kind": "window_grammar", "file": str(p), "size_mb": size_mb(p), "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))})
    if final_grammar.exists():
        rows.append({"kind": "final_grammar", "file": str(final_grammar), "size_mb": size_mb(final_grammar), "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(final_grammar.stat().st_mtime))})

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')
    else:
        st.info("No grammar snapshots yet.")

    core = ["status.json", "wf_config.json", "metrics.jsonl", "wf_window_summaries.jsonl", "wf_chunk_summaries.jsonl", "wf_inference_records.jsonl", "fpc_cache_index.jsonl", "plot_index.jsonl", "console.log"]
    file_rows = []
    for name in core:
        p = run_dir / name
        file_rows.append({"file": name, "exists": p.exists(), "size_mb": size_mb(p) if p.exists() else 0.0, "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None})
    st.markdown("**Core files**")
    st.dataframe(pd.DataFrame(file_rows), width='stretch')

    arrays_dir = run_dir / "wf_arrays"
    if arrays_dir.exists():
        arr_files = sorted(arrays_dir.glob("*.npz"), key=lambda p: p.name)
        st.markdown("**Inference array files**")
        st.dataframe(pd.DataFrame([{"file": str(p), "size_mb": size_mb(p)} for p in arr_files]), width='stretch')


# ---------------------------------------------------------------------
# raw tables
# ---------------------------------------------------------------------

with tabs[5]:
    st.subheader("Raw tables")
    table_choice = st.selectbox("Table", ["metrics", "window summaries", "chunk summaries", "inference records", "fpc cache", "decay", "plots"])
    table_map = {
        "metrics": metrics_df,
        "window summaries": window_df,
        "chunk summaries": chunk_df,
        "inference records": infer_df,
        "fpc cache": fpc_df,
        "decay": decay_df,
        "plots": plot_df,
    }
    df = table_map[table_choice]
    if df.empty:
        st.info("Selected table is empty.")
    else:
        st.dataframe(df, width='stretch')
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected}_{table_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------
# console
# ---------------------------------------------------------------------

with tabs[6]:
    st.subheader("Console log")
    log_path = run_dir / "console.log"
    if not log_path.exists():
        st.info("No console.log yet.")
    else:
        max_lines = st.slider("Tail lines", min_value=50, max_value=3000, value=500, step=50)
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            st.text_area("console.log tail", value="".join(lines[-max_lines:]), height=650)
        except Exception as e:
            st.error(f"Could not read console log: {e}")


if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
