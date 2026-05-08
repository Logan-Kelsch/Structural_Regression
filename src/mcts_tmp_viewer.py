# mcts_tmp_viewer.py
#
# Streamlit live/past viewer for mcts_util.py runs.
#
# Run:
#     streamlit run mcts_tmp_viewer.py
#
# It can watch mcts_runs/tmp live while the notebook loop is running,
# and it can also browse older named runs in mcts_runs/.

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MCTS run viewer",
    layout="wide",
)


# ---------------------------------------------------------------------
# file readers
# ---------------------------------------------------------------------

def read_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(path):
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


def read_text_tail(path, n_lines=250):
    path = Path(path)

    if not path.exists():
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-int(n_lines):])
    except Exception as e:
        return f"could not read log: {e}"


def list_run_dirs(root):
    root = Path(root)

    if not root.exists():
        return []

    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def load_metrics(run_dir):
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_plot_index(run_dir):
    rows = read_jsonl(Path(run_dir) / "plot_index.jsonl")

    if not rows:
        return pd.DataFrame(columns=["k", "section", "path"])

    df = pd.DataFrame(rows)

    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    return df


def resolve_path(run_dir, p):
    if p is None:
        return None

    p = Path(str(p))

    if p.exists():
        return p

    q = Path(run_dir) / p

    if q.exists():
        return q

    return None


def get_section_images(plot_df, run_dir, selected_k, section):
    if plot_df.empty:
        return []

    df = plot_df.copy()

    if "section" not in df.columns or "path" not in df.columns:
        return []

    keep = df["section"].astype(str).eq(str(section))

    if selected_k is not None and "k" in df.columns:
        keep &= df["k"].astype("Int64").eq(int(selected_k))

    paths = []
    for p in df.loc[keep, "path"].tolist():
        rp = resolve_path(run_dir, p)
        if rp is not None:
            paths.append(rp)

    return paths


def latest_image_for_section(plot_df, run_dir, section):
    if plot_df.empty:
        return None

    df = plot_df.copy()

    if "section" not in df.columns or "path" not in df.columns:
        return None

    df = df[df["section"].astype(str).eq(str(section))]

    if df.empty:
        return None

    if "k" in df.columns:
        df = df.sort_values("k")

    for p in reversed(df["path"].tolist()):
        rp = resolve_path(run_dir, p)
        if rp is not None:
            return rp

    return None


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

st.sidebar.title("MCTS viewer")

run_root_text = st.sidebar.text_input("run root", "mcts_runs")
run_root = Path(run_root_text)

runs = list_run_dirs(run_root)
run_names = [p.name for p in runs]

if "tmp" in run_names:
    default_idx = run_names.index("tmp")
else:
    default_idx = 0 if run_names else None

if runs:
    selected_name = st.sidebar.selectbox(
        "run",
        run_names,
        index=default_idx,
    )
    run_dir = run_root / selected_name
else:
    selected_name = st.sidebar.text_input("run name", "tmp")
    run_dir = run_root / selected_name

manual_run_dir = st.sidebar.text_input("manual run folder override", "")
if manual_run_dir.strip():
    run_dir = Path(manual_run_dir.strip())

auto_refresh = st.sidebar.checkbox("auto refresh", value=True)
refresh_sec = st.sidebar.slider("refresh seconds", 1, 30, 3)
show_original_plots = st.sidebar.checkbox("include original captured helper plots", value=True)
log_tail_lines = st.sidebar.slider("log tail lines", 50, 2000, 300, 50)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

st.title("MCTS tmp / past-run viewer")
st.caption(f"reading: {run_dir}")

status = read_json(run_dir / "status.json", default={})
metrics = load_metrics(run_dir)
plot_df = load_plot_index(run_dir)
tops_rows = read_jsonl(run_dir / "tops.jsonl")

top_cols = st.columns(6)
top_cols[0].metric("status", status.get("status", "missing"))
top_cols[1].metric("last iter", status.get("last_iteration", "NA"))
top_cols[2].metric("last h", status.get("last_h", "NA"))
top_cols[3].metric("last z mean", status.get("last_zscore_mean", "NA"))
top_cols[4].metric("last z max", status.get("last_zscore_max", "NA"))
top_cols[5].metric("updated", status.get("updated_at", "NA"))

if not run_dir.exists():
    st.warning(f"Run folder does not exist: {run_dir}")
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
    st.stop()

if metrics.empty:
    st.info("No metrics have been written yet.")
    st.subheader("console log")
    st.text_area("console.log", read_text_tail(run_dir / "console.log", n_lines=log_tail_lines), height=420)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
    st.stop()

latest = metrics.iloc[-1]
latest_k = int(latest["k"])

summary_cols = st.columns(8)
summary_cols[0].metric("iteration", latest_k)
summary_cols[1].metric("h", f"{latest.get('policy_h', np.nan):.6f}")
summary_cols[2].metric("z mean", f"{latest.get('zscore_mean', np.nan):.4f}")
summary_cols[3].metric("z max", f"{latest.get('zscore_max', np.nan):.4f}")
summary_cols[4].metric("p min", f"{latest.get('pval_min', np.nan):.4g}")
summary_cols[5].metric("alpha p(sensor)", f"{latest.get('alpha_sensor_prob', np.nan):.4f}")
summary_cols[6].metric("UCT infl.", f"{latest.get('uct_explore_influence', np.nan):.4f}")
summary_cols[7].metric("freeze", str(latest.get("freeze_expansion", False)))

tab_live, tab_curves, tab_iter, tab_plots, tab_arrays, tab_tops, tab_log, tab_runs = st.tabs([
    "live summary",
    "metric curves",
    "iteration browser",
    "plot gallery",
    "arrays",
    "top edges",
    "console log",
    "runs",
])


with tab_live:
    st.subheader("latest dashboard")

    latest_summary = latest_image_for_section(plot_df, run_dir, "summary_dashboard")

    if latest_summary is not None:
        st.image(str(latest_summary), caption=str(latest_summary), use_container_width=True)
    else:
        st.info("No summary_dashboard image yet.")

    st.subheader("latest recreated plots")

    sections = [
        "state_growth",
        "ev_violin",
        "gene_fpc_eval_recreated",
        "depth_terms",
        "depth_progress",
        "alpha_parent_depth_progress",
        "alpha_sensor_probability",
        "exploration_influence",
        "policy_drift",
    ]

    cols = st.columns(2)
    for i, section in enumerate(sections):
        p = latest_image_for_section(plot_df, run_dir, section)
        with cols[i % 2]:
            st.write(section)
            if p is not None:
                st.image(str(p), use_container_width=True)
            else:
                st.caption("not available")


with tab_curves:
    st.subheader("metrics over iteration")

    chart_groups = {
        "policy drift": [
            "policy_parent_drift",
            "policy_child_drift",
            "policy_total_drift",
            "policy_h",
        ],
        "scores": [
            "zscore_mean",
            "zscore_max",
            "pval_min",
            "pval_mean",
        ],
        "MCTS growth": [
            "mcts_node_mu_count",
            "mcts_edge_mu_count",
            "alpha_decision_mu_count",
            "alpha_node_mu_count",
            "alpha_edge_mu_count",
        ],
        "exploration / depth": [
            "uct_explore_influence",
            "uct_explore_influence_ema",
            "depth_prob_delta_sum",
            "alpha_depth_prob_delta_sum",
            "alpha_sensor_prob",
        ],
    }

    for title, cols in chart_groups.items():
        have = [c for c in cols if c in metrics.columns]

        if have:
            st.write(title)
            st.line_chart(metrics.set_index("k")[have])

    st.subheader("raw metrics")
    st.dataframe(metrics, use_container_width=True, height=360)


with tab_iter:
    st.subheader("iteration browser")

    selected_k = st.slider("iteration", 0, latest_k, latest_k)

    row = metrics[metrics["k"] == selected_k]
    if not row.empty:
        st.json(row.iloc[-1].to_dict())

    preferred_sections = [
        "summary_dashboard",
        "state_growth",
        "ev_violin",
        "gene_fpc_eval_recreated",
        "gene_fpc_eval_original",
        "depth_terms",
        "depth_progress",
        "alpha_parent_depth_progress",
        "alpha_sensor_probability",
        "exploration_influence",
        "policy_drift",
        "fpc_curve_once",
    ]

    section = st.selectbox("plot section", preferred_sections)
    imgs = get_section_images(plot_df, run_dir, selected_k, section)

    if imgs:
        for p in imgs:
            st.image(str(p), caption=str(p), use_container_width=True)
    else:
        st.info("No images for this section/iteration.")


with tab_plots:
    st.subheader("plot gallery")

    if plot_df.empty:
        st.info("No plot index yet.")
    else:
        sections_all = sorted(plot_df["section"].astype(str).unique().tolist())

        if not show_original_plots:
            sections_all = [
                s for s in sections_all
                if not s.startswith("original_")
            ]

        selected_section = st.selectbox("section", sections_all)

        df_sec = plot_df[plot_df["section"].astype(str).eq(selected_section)].copy()

        if "k" in df_sec.columns and not df_sec.empty:
            min_k = int(df_sec["k"].min())
            max_k = int(df_sec["k"].max())
            k_gallery = st.slider("gallery iteration", min_k, max_k, max_k)
            df_sec = df_sec[df_sec["k"].astype("Int64").eq(k_gallery)]

        for p in df_sec["path"].tolist():
            rp = resolve_path(run_dir, p)
            if rp is not None:
                st.image(str(rp), caption=str(rp), use_container_width=True)


with tab_arrays:
    st.subheader("stored arrays")

    selected_k_arr = st.slider("array iteration", 0, latest_k, latest_k, key="arr_k")
    arrays_path = run_dir / "iterations" / f"iter_{selected_k_arr:04d}" / "arrays.npz"

    if not arrays_path.exists():
        st.info(f"No arrays file for iteration {selected_k_arr}.")
    else:
        data = np.load(arrays_path, allow_pickle=True)
        keys = list(data.keys())

        array_key = st.selectbox("array", keys)
        arr = data[array_key]

        st.write("shape:", arr.shape, "dtype:", arr.dtype)

        flat = np.asarray(arr).ravel()

        if flat.size > 0 and np.issubdtype(flat.dtype, np.number):
            numeric = flat.astype(float)
            numeric = numeric[np.isfinite(numeric)]

            if numeric.size > 0:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("n", int(numeric.size))
                c2.metric("mean", f"{np.mean(numeric):.6f}")
                c3.metric("std", f"{np.std(numeric):.6f}")
                c4.metric("min", f"{np.min(numeric):.6f}")
                c5.metric("max", f"{np.max(numeric):.6f}")

                show_n = min(5000, numeric.size)
                st.line_chart(numeric[:show_n])

                if numeric.size > show_n:
                    st.caption(f"showing first {show_n} values")

        st.write(arr)


with tab_tops:
    st.subheader("top edges / alpha decisions")

    selected_k_top = st.slider("tops iteration", 0, latest_k, latest_k, key="tops_k")
    tops_path = run_dir / "iterations" / f"iter_{selected_k_top:04d}" / "tops.json"
    tops = read_json(tops_path, default={})

    if not tops:
        st.info("No tops file for this iteration.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("top x/tf edges")
            st.dataframe(pd.DataFrame(tops.get("top_edges", [])), use_container_width=True)

        with c2:
            st.write("top alpha decisions")
            st.dataframe(pd.DataFrame(tops.get("top_alpha_decisions", [])), use_container_width=True)

        with c3:
            st.write("top alpha edges")
            st.dataframe(pd.DataFrame(tops.get("top_alpha_edges", [])), use_container_width=True)


with tab_log:
    st.subheader("console log tail")
    st.text_area(
        "console.log",
        read_text_tail(run_dir / "console.log", n_lines=log_tail_lines),
        height=620,
    )


with tab_runs:
    st.subheader("available runs")

    rows = []

    for p in list_run_dirs(run_root):
        status_p = read_json(p / "status.json", default={})
        metrics_p = load_metrics(p)

        rows.append({
            "run": p.name,
            "path": str(p),
            "status": status_p.get("status"),
            "last_iteration": status_p.get("last_iteration"),
            "updated_at": status_p.get("updated_at"),
            "n_metric_rows": int(len(metrics_p)),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
