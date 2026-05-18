"""
streamlit_wf_mcts_cached_dashboard.py

Live dashboard for walk_forward_mcts_alpha_fpc_cached.py output.

Run:
    streamlit run streamlit_wf_mcts_cached_dashboard.py

Expected run directory structure:
    wf_mcts_runs/<run_name_timestamp>/
        config.json
        metrics.jsonl
        chunk_summaries.jsonl
        inference_records.jsonl
        plots/*.png
        grammars/*.pkl
        fpc_cache/*
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------
# page setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="WF MCTS Alpha Cached-FPC Dashboard",
    layout="wide",
)


# ---------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------

@st.cache_data(ttl=2.0, show_spinner=False)
def read_json(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(ttl=2.0, show_spinner=False)
def read_jsonl(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate partial line while run is writing
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for c in df.columns:
        if c not in {"event", "grammar_path", "saved_path", "pickle_error"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(ttl=2.0, show_spinner=False)
def list_files(path_str: str, pattern: str):
    path = Path(path_str)
    if not path.exists():
        return []
    return sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)


def latest_run(base_dir: str = "wf_mcts_runs") -> Path | None:
    base = Path(base_dir)
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]


def safe_metric(container, label: str, value, fmt: str | None = None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        container.metric(label, "—")
        return
    if fmt is not None:
        try:
            container.metric(label, fmt.format(value))
            return
        except Exception:
            pass
    container.metric(label, value)


def filter_event(df: pd.DataFrame, event: str) -> pd.DataFrame:
    if df.empty or "event" not in df.columns:
        return pd.DataFrame()
    return df[df["event"] == event].copy()


def make_line_plot(df: pd.DataFrame, x: str, ys: list[str], title: str, ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for y in ys:
        if y in df.columns:
            ax.plot(df[x], df[y], marker="o", label=y)
    ax.set_title(title)
    ax.set_xlabel(x)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def make_hist(series: pd.Series, title: str, xlabel: str, bins: int = 30, vline: float | None = None):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size:
        ax.hist(vals, bins=bins, alpha=0.75)
        if vline is not None:
            ax.axvline(vline, linestyle="--", alpha=0.6, label=f"{xlabel}={vline}")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "no values", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    return fig


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

st.title("Walk-forward MCTS Alpha Dashboard")
st.caption("Cached-FPC run monitor for walk_forward_mcts_alpha_fpc_cached.py")

with st.sidebar:
    st.header("Run selection")
    base_dir = st.text_input("Base run directory", value="wf_mcts_runs")
    latest = latest_run(base_dir)
    default_run = str(latest) if latest is not None else base_dir
    run_dir = Path(st.text_input("Run directory", value=default_run))

    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh seconds", min_value=2, max_value=60, value=5, step=1)

    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.write("Run exists:", run_dir.exists())

if not run_dir.exists():
    st.warning("Run directory does not exist yet.")
    st.stop()

config = read_json(str(run_dir / "config.json")) or {}
metrics_all = read_jsonl(str(run_dir / "metrics.jsonl"))
summaries = read_jsonl(str(run_dir / "chunk_summaries.jsonl"))
infer_records = read_jsonl(str(run_dir / "inference_records.jsonl"))
train_metrics = filter_event(metrics_all, "train_iter")
fpc_events = filter_event(metrics_all, "fpc_fit")

plots = list_files(str(run_dir / "plots"), "*.png")
grammars = list_files(str(run_dir / "grammars"), "*.pkl")
fpc_files = list_files(str(run_dir / "fpc_cache"), "*")

with st.sidebar.expander("Config", expanded=False):
    if config:
        st.json(config)
    else:
        st.info("No config.json found.")

with st.sidebar.expander("Files", expanded=False):
    st.write("metrics.jsonl", (run_dir / "metrics.jsonl").exists())
    st.write("chunk_summaries.jsonl", (run_dir / "chunk_summaries.jsonl").exists())
    st.write("inference_records.jsonl", (run_dir / "inference_records.jsonl").exists())
    st.write("plots", len(plots))
    st.write("grammars", len(grammars))
    st.write("fpc cache files", len(fpc_files))


# ---------------------------------------------------------------------
# run overview
# ---------------------------------------------------------------------

st.header("Run overview")

n_chunks = int(config.get("n_chunks", 20)) if config else 20
expected_windows = max(n_chunks - 2, 0)
completed_windows = 0 if summaries.empty else summaries["chunk_start"].nunique() if "chunk_start" in summaries.columns else len(summaries)
active_chunk = None
active_iter = None

if not train_metrics.empty:
    active_chunk = int(train_metrics["chunk_start"].dropna().max()) if "chunk_start" in train_metrics.columns else None
    active_iter = int(train_metrics["iter_n"].dropna().max()) if "iter_n" in train_metrics.columns else None

c1, c2, c3, c4, c5 = st.columns(5)
safe_metric(c1, "Completed windows", f"{completed_windows}/{expected_windows}")
safe_metric(c2, "Active chunk_start", active_chunk)
safe_metric(c3, "Latest iter", active_iter)
safe_metric(c4, "FPC chunks fit", fpc_events["chunk_num"].nunique() if not fpc_events.empty and "chunk_num" in fpc_events.columns else 0)
safe_metric(c5, "Grammar snapshots", len(grammars))

if expected_windows > 0:
    st.progress(min(completed_windows / expected_windows, 1.0))

if not train_metrics.empty and "policy_drift_h" in train_metrics.columns:
    last_train = train_metrics.sort_values(["chunk_start", "iter_n"]).tail(1).iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    safe_metric(d1, "Latest policy drift", float(last_train.get("policy_drift", np.nan)), "{:.4f}")
    safe_metric(d2, "Latest smoothed h", float(last_train.get("policy_drift_h", np.nan)), "{:.4f}")
    safe_metric(d3, "delta_L", float(last_train.get("delta_L", np.nan)), "{:.4f}")
    safe_metric(d4, "FPC cache size", int(last_train.get("fpc_cache_size", 0)) if pd.notna(last_train.get("fpc_cache_size", np.nan)) else None)


# ---------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------

tab_train, tab_summary, tab_infer, tab_fpc, tab_artifacts, tab_raw = st.tabs([
    "Training progress",
    "Walk-forward summaries",
    "Inference funnel",
    "FPC cache",
    "Artifacts",
    "Raw tables",
])


# ---------------------------------------------------------------------
# training tab
# ---------------------------------------------------------------------

with tab_train:
    st.subheader("Training convergence")

    if train_metrics.empty:
        st.info("No train_iter metrics yet.")
    else:
        chunk_options = sorted(train_metrics["chunk_start"].dropna().unique())
        default_idx = len(chunk_options) - 1
        selected_chunk = st.selectbox(
            "chunk_start",
            chunk_options,
            index=default_idx,
            key="train_chunk_select",
        )
        m = train_metrics[train_metrics["chunk_start"] == selected_chunk].copy()
        m = m.sort_values("iter_n") if "iter_n" in m.columns else m

        c1, c2, c3, c4 = st.columns(4)
        safe_metric(c1, "Iterations", len(m))
        if "policy_drift_h" in m.columns:
            safe_metric(c2, "Latest h", float(m["policy_drift_h"].dropna().iloc[-1]) if m["policy_drift_h"].notna().any() else np.nan, "{:.4f}")
        if "policy_drift" in m.columns:
            safe_metric(c3, "Latest raw drift", float(m["policy_drift"].dropna().iloc[-1]) if m["policy_drift"].notna().any() else np.nan, "{:.4f}")
        if "delta_L" in m.columns:
            safe_metric(c4, "delta_L", float(m["delta_L"].dropna().iloc[-1]) if m["delta_L"].notna().any() else np.nan, "{:.4f}")

        if all(c in m.columns for c in ["iter_n", "policy_drift", "policy_drift_h"]):
            fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
            ax.plot(m["iter_n"], m["policy_drift"], marker="o", alpha=0.35, label="raw local action drift")
            ax.plot(m["iter_n"], m["policy_drift_h"], linewidth=3, label="smoothed h")
            if "delta_L" in m.columns and m["delta_L"].notna().any():
                ax.axhline(float(m["delta_L"].dropna().iloc[-1]), linestyle="--", alpha=0.65, label="delta_L")
            ax.set_title(f"Policy drift convergence | chunk_start={selected_chunk}")
            ax.set_xlabel("iteration within chunk")
            ax.set_ylabel("drift")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend()
            st.pyplot(fig)

        growth_cols = [
            "mcts_node_mu_count",
            "mcts_edge_mu_count",
            "mcts_children_edges",
            "alpha_decision_count",
            "alpha_node_count",
            "alpha_edge_count",
            "fpc_cache_size",
        ]
        available_growth = [c for c in growth_cols if c in m.columns]
        if "iter_n" in m.columns and available_growth:
            fig = make_line_plot(
                m,
                x="iter_n",
                ys=available_growth,
                title=f"Grammar memory / cache growth | chunk_start={selected_chunk}",
                ylabel="count",
            )
            st.pyplot(fig)

        with st.expander("Training metrics table", expanded=False):
            st.dataframe(m, use_container_width=True)

    st.subheader("All chunks training overview")
    if not train_metrics.empty and all(c in train_metrics.columns for c in ["chunk_start", "iter_n", "policy_drift_h"]):
        last_by_chunk = train_metrics.sort_values(["chunk_start", "iter_n"]).groupby("chunk_start").tail(1)
        st.dataframe(last_by_chunk, use_container_width=True)
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        ax.plot(last_by_chunk["chunk_start"], last_by_chunk["policy_drift_h"], marker="o", label="final h per window")
        if "delta_L" in last_by_chunk.columns:
            ax.plot(last_by_chunk["chunk_start"], last_by_chunk["delta_L"], linestyle="--", alpha=0.65, label="delta_L")
        ax.set_title("Final smoothed policy drift by walk-forward window")
        ax.set_xlabel("chunk_start")
        ax.set_ylabel("h")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)


# ---------------------------------------------------------------------
# summary tab
# ---------------------------------------------------------------------

with tab_summary:
    st.subheader("Walk-forward i → j → k summaries")

    if summaries.empty:
        st.info("No chunk summaries yet.")
    else:
        summaries = summaries.sort_values("chunk_start") if "chunk_start" in summaries.columns else summaries
        st.dataframe(summaries, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        safe_metric(c1, "Windows completed", len(summaries))
        if "total_i_success_count" in summaries.columns:
            safe_metric(c2, "Total i success", int(summaries["total_i_success_count"].fillna(0).sum()))
        if "total_ij_success_count" in summaries.columns:
            safe_metric(c3, "Total ij success", int(summaries["total_ij_success_count"].fillna(0).sum()))
        if "k_z_mean" in summaries.columns:
            safe_metric(c4, "Mean k z", float(summaries["k_z_mean"].mean()), "{:.3f}")
        if "k_z_max" in summaries.columns and summaries["k_z_max"].notna().any():
            safe_metric(c5, "Max k z", float(summaries["k_z_max"].max()), "{:.3f}")

        if "chunk_start" in summaries.columns:
            count_cols = [c for c in ["total_i_success_count", "total_ij_success_count", "total_k_eval_count"] if c in summaries.columns]
            if count_cols:
                fig = make_line_plot(
                    summaries,
                    x="chunk_start",
                    ys=count_cols,
                    title="Success funnel counts across walk-forward windows",
                    ylabel="count",
                )
                st.pyplot(fig)

            z_cols = [c for c in ["k_z_mean", "k_z_median", "k_z_p75", "k_z_p90", "k_z_max"] if c in summaries.columns]
            if z_cols:
                fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
                for c in z_cols:
                    ax.plot(summaries["chunk_start"], summaries[c], marker="o", label=c)
                if config:
                    ax.axhline(float(config.get("success_z", 2.0)), linestyle="--", alpha=0.65, label="success_z")
                ax.set_title("Chunk k z-score performance over walk-forward windows")
                ax.set_xlabel("chunk_start")
                ax.set_ylabel("z-score")
                ax.grid(alpha=0.25)
                ax.legend()
                st.pyplot(fig)


# ---------------------------------------------------------------------
# inference tab
# ---------------------------------------------------------------------

with tab_infer:
    st.subheader("Inference funnel records")

    if infer_records.empty:
        st.info("No inference records yet.")
    else:
        if "chunk_start" in infer_records.columns:
            chunk_options = sorted(infer_records["chunk_start"].dropna().unique())
            selected_chunk_inf = st.selectbox(
                "Inference chunk_start",
                chunk_options,
                index=len(chunk_options) - 1,
                key="infer_chunk_select",
            )
            r = infer_records[infer_records["chunk_start"] == selected_chunk_inf].copy()
        else:
            selected_chunk_inf = None
            r = infer_records.copy()

        c1, c2, c3, c4 = st.columns(4)
        safe_metric(c1, "Infer populations", len(r))
        if "i_success_count" in r.columns:
            safe_metric(c2, "i success", int(r["i_success_count"].fillna(0).sum()))
        if "ij_success_count" in r.columns:
            safe_metric(c3, "ij success", int(r["ij_success_count"].fillna(0).sum()))
        if "k_eval_count" in r.columns:
            safe_metric(c4, "k eval count", int(r["k_eval_count"].fillna(0).sum()))

        if "k_z_mean" in r.columns:
            st.pyplot(make_hist(
                r["k_z_mean"],
                title=f"Per-inference-population k mean z | chunk_start={selected_chunk_inf}",
                xlabel="k_z_mean",
                bins=30,
                vline=float(config.get("success_z", 2.0)) if config else 2.0,
            ))

        if "infer_population" in r.columns:
            count_cols = [c for c in ["i_success_count", "ij_success_count", "k_eval_count"] if c in r.columns]
            if count_cols:
                fig = make_line_plot(
                    r.sort_values("infer_population"),
                    x="infer_population",
                    ys=count_cols,
                    title=f"Infer population funnel counts | chunk_start={selected_chunk_inf}",
                    ylabel="count",
                )
                st.pyplot(fig)

        with st.expander("Inference records table", expanded=False):
            st.dataframe(r, use_container_width=True)


# ---------------------------------------------------------------------
# FPC tab
# ---------------------------------------------------------------------

with tab_fpc:
    st.subheader("FPC cache status")

    expected_chunks = list(range(n_chunks))
    fit_chunks = []
    if not fpc_events.empty and "chunk_num" in fpc_events.columns:
        fit_chunks = sorted([int(x) for x in fpc_events["chunk_num"].dropna().unique()])

    c1, c2, c3 = st.columns(3)
    safe_metric(c1, "Expected chunk curves", n_chunks)
    safe_metric(c2, "Fitted chunk curves", len(fit_chunks))
    safe_metric(c3, "Cache files", len(fpc_files))

    if n_chunks > 0:
        st.progress(min(len(fit_chunks) / n_chunks, 1.0))

    status = pd.DataFrame({
        "chunk_num": expected_chunks,
        "fit_event_seen": [c in fit_chunks for c in expected_chunks],
        "pkl_exists": [(run_dir / "fpc_cache" / f"fpc_chunk_{c:03d}.pkl").exists() for c in expected_chunks],
        "meta_exists": [(run_dir / "fpc_cache" / f"fpc_chunk_{c:03d}_meta.json").exists() for c in expected_chunks],
    })
    st.dataframe(status, use_container_width=True)

    if not fpc_events.empty:
        st.subheader("FPC fit events")
        st.dataframe(fpc_events, use_container_width=True)
        if all(c in fpc_events.columns for c in ["chunk_num", "fit_elapsed_sec"]):
            fig = make_line_plot(
                fpc_events.sort_values("chunk_num"),
                x="chunk_num",
                ys=["fit_elapsed_sec"],
                title="FPC fit time by chunk",
                ylabel="seconds",
            )
            st.pyplot(fig)


# ---------------------------------------------------------------------
# artifacts tab
# ---------------------------------------------------------------------

with tab_artifacts:
    st.subheader("Saved plots")
    if plots:
        plot_names = [p.name for p in plots]
        selected_plot_name = st.selectbox("Plot", plot_names, index=len(plot_names) - 1)
        selected_plot = next(p for p in plots if p.name == selected_plot_name)
        st.image(str(selected_plot), use_container_width=True)
        st.code(str(selected_plot))
    else:
        st.info("No plots yet.")

    st.subheader("Grammar snapshots")
    if grammars:
        gdf = pd.DataFrame({
            "name": [p.name for p in grammars],
            "path": [str(p) for p in grammars],
            "modified": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) for p in grammars],
            "size_mb": [p.stat().st_size / (1024 ** 2) for p in grammars],
        })
        st.dataframe(gdf.sort_values("modified", ascending=False), use_container_width=True)
    else:
        st.info("No grammar snapshots yet.")

    st.subheader("FPC cache files")
    if fpc_files:
        fdf = pd.DataFrame({
            "name": [p.name for p in fpc_files],
            "path": [str(p) for p in fpc_files],
            "size_mb": [p.stat().st_size / (1024 ** 2) if p.is_file() else np.nan for p in fpc_files],
        })
        st.dataframe(fdf, use_container_width=True)
    else:
        st.info("No FPC cache files yet.")


# ---------------------------------------------------------------------
# raw tab
# ---------------------------------------------------------------------

with tab_raw:
    st.subheader("Raw metrics")
    if metrics_all.empty:
        st.info("No metrics.jsonl rows yet.")
    else:
        event_options = ["all"] + sorted(metrics_all["event"].dropna().unique().tolist()) if "event" in metrics_all.columns else ["all"]
        event_sel = st.selectbox("event", event_options)
        raw = metrics_all if event_sel == "all" else metrics_all[metrics_all["event"] == event_sel]
        st.dataframe(raw, use_container_width=True)

    st.subheader("Raw summaries")
    st.dataframe(summaries, use_container_width=True)

    st.subheader("Raw inference records")
    st.dataframe(infer_records, use_container_width=True)


# ---------------------------------------------------------------------
# auto refresh
# ---------------------------------------------------------------------

st.caption(f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()
