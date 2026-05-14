# mcts_tmp_viewer1.py
#
# Streamlit live/past viewer for mcts_util.py runs.
#
# Run from your project root:
#     streamlit run mcts_tmp_viewer1.py
#
# It watches mcts_runs/tmp live while the loop is running and can also browse
# completed named runs in mcts_runs/.

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MCTS run viewer",
    layout="wide",
)


# ---------------------------------------------------------------------
# small file readers
# ---------------------------------------------------------------------

def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        return []

    rows = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []

    return rows


def read_text_tail(path: str | Path, n_lines: int = 300) -> str:
    path = Path(path)

    if not path.exists():
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-int(n_lines):])
    except Exception as e:
        return f"could not read log: {e}"


def list_run_dirs(root: str | Path) -> list[Path]:
    root = Path(root)

    if not root.exists():
        return []

    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


@st.cache_data(ttl=2, show_spinner=False)
def load_metrics_cached(run_dir_text: str, mtime_ns: int | None) -> pd.DataFrame:
    del mtime_ns
    rows = read_jsonl(Path(run_dir_text) / "metrics.jsonl")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    return df


@st.cache_data(ttl=2, show_spinner=False)
def load_plot_index_cached(run_dir_text: str, mtime_ns: int | None) -> pd.DataFrame:
    del mtime_ns
    rows = read_jsonl(Path(run_dir_text) / "plot_index.jsonl")

    if not rows:
        return pd.DataFrame(columns=["k", "section", "path"])

    df = pd.DataFrame(rows)

    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    return df


def path_mtime_ns(path: str | Path) -> int | None:
    path = Path(path)

    try:
        return path.stat().st_mtime_ns
    except Exception:
        return None


def load_metrics(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    return load_metrics_cached(str(run_dir), path_mtime_ns(run_dir / "metrics.jsonl"))


def load_plot_index(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    return load_plot_index_cached(str(run_dir), path_mtime_ns(run_dir / "plot_index.jsonl"))


def resolve_path(run_dir: str | Path, p: Any) -> Path | None:
    if p is None:
        return None

    p = Path(str(p))

    if p.exists():
        return p

    q = Path(run_dir) / p

    if q.exists():
        return q

    return None


def get_section_images(plot_df: pd.DataFrame, run_dir: str | Path, selected_k: int | None, section: str) -> list[Path]:
    if plot_df.empty:
        return []

    if "section" not in plot_df.columns or "path" not in plot_df.columns:
        return []

    keep = plot_df["section"].astype(str).eq(str(section))

    if selected_k is not None and "k" in plot_df.columns:
        keep &= plot_df["k"].astype("Int64").eq(int(selected_k))

    paths = []
    for p in plot_df.loc[keep, "path"].tolist():
        rp = resolve_path(run_dir, p)
        if rp is not None:
            paths.append(rp)

    return paths


def latest_image_for_section(plot_df: pd.DataFrame, run_dir: str | Path, section: str) -> Path | None:
    if plot_df.empty:
        return None

    if "section" not in plot_df.columns or "path" not in plot_df.columns:
        return None

    df = plot_df[plot_df["section"].astype(str).eq(str(section))].copy()

    if df.empty:
        return None

    if "k" in df.columns:
        df = df.sort_values("k")

    for p in reversed(df["path"].tolist()):
        rp = resolve_path(run_dir, p)
        if rp is not None:
            return rp

    return None


def metric_value(status: dict, latest: pd.Series | None, status_key: str, metric_key: str, default: Any = np.nan) -> Any:
    v = status.get(status_key, None)

    if v is not None:
        return v

    if latest is not None and metric_key in latest.index:
        return latest.get(metric_key, default)

    return default


def fmt_float(x: Any, nd: int = 4, default: str = "NA") -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return f"{v:.{nd}f}"
    except Exception:
        return default


def fmt_int(x: Any, default: str = "NA") -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return f"{v:.0f}"
    except Exception:
        return default


def safe_metric_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def read_trace_tail(run_dir: str | Path, n_lines: int = 200) -> str:
    return read_text_tail(Path(run_dir) / "mcts_trace.jsonl", n_lines=n_lines)


def read_plot_index(run_dir):
    path = run_dir / "plot_index.jsonl"

    if not path.exists():
        return pd.DataFrame()

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

    return pd.DataFrame(rows)


def resolve_plot_path(run_dir, path_value):
    if path_value is None:
        return None

    p = Path(str(path_value))

    if p.exists():
        return p

    p2 = run_dir / p

    if p2.exists():
        return p2

    return None


def read_plot_index(run_dir):
    path = run_dir / "plot_index.jsonl"

    if not path.exists():
        return pd.DataFrame()

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

    return pd.DataFrame(rows)


def resolve_plot_path(run_dir, path_value):
    if path_value is None:
        return None

    p = Path(str(path_value))

    if p.exists():
        return p

    p2 = run_dir / p

    if p2.exists():
        return p2

    return None

def safe_iteration_selectbox_or_slider(label, min_k, max_k, default_k=None, *, key=None):
    """
    Streamlit slider fails when min_k == max_k.
    This helper returns the only available iteration directly in that case.
    """
    try:
        min_k = int(min_k)
        max_k = int(max_k)
    except Exception:
        min_k = 0
        max_k = 0

    if default_k is None:
        default_k = max_k

    try:
        default_k = int(default_k)
    except Exception:
        default_k = max_k

    default_k = min(max(default_k, min_k), max_k)

    if max_k <= min_k:
        st.caption(f"{label}: only iteration {min_k} is available so far")
        return min_k

    return st.slider(
        label,
        min_value=min_k,
        max_value=max_k,
        value=default_k,
        step=1,
        key=key,
    )


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
elif runs:
    default_idx = 0
else:
    default_idx = None

if runs:
    selected_name = st.sidebar.selectbox("run", run_names, index=default_idx)
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
log_tail_lines = st.sidebar.slider("log tail lines", 50, 3000, 300, 50)
trace_tail_lines = st.sidebar.slider("trace tail lines", 25, 1000, 150, 25)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

st.title("MCTS tmp / past-run viewer")
st.caption(f"reading: {run_dir}")

status = read_json(run_dir / "status.json", default={})
metrics = load_metrics(run_dir)
plot_df = load_plot_index(run_dir)

latest = None if metrics.empty else metrics.iloc[-1]

edge_explore_value = metric_value(
    status,
    latest,
    "last_mcts_edge_explore_count_total",
    "mcts_edge_explore_count_total",
)

last_h_value = metric_value(status, latest, "last_h", "policy_h")

# top status cards
top_cols = st.columns(7)
top_cols[0].metric("status", status.get("status", "missing"))
top_cols[1].metric("last iter", status.get("last_iteration", "NA"))
top_cols[2].metric("edge explore n", fmt_int(None if latest is None else latest['alpha_explore_t']))
top_cols[3].metric("last z mean", fmt_float(metric_value(status, latest, "last_zscore_mean", "zscore_mean")))
top_cols[4].metric("last z max", fmt_float(metric_value(status, latest, "last_zscore_max", "zscore_max")))
top_cols[5].metric("last h", fmt_float(last_h_value))
top_cols[6].metric("updated", status.get("updated_at", "NA"))

plot_index_df = read_plot_index(run_dir)

if not plot_index_df.empty and "section" in plot_index_df.columns:
    fpc_rows = plot_index_df[
        plot_index_df["section"].astype(str).str.contains("fpc_curve_once", na=False)
    ]

    if not fpc_rows.empty:
        fpc_path = resolve_plot_path(run_dir, fpc_rows.iloc[-1]["path"])

        if fpc_path is not None:
            st.subheader("FPC curve")
            st.image(str(fpc_path), caption=str(fpc_path), width="stretch")
        else:
            st.info("FPC curve entry exists, but the image file is not available yet.")
    else:
        st.info("FPC curve has not been saved yet. It will appear after fit_FPC_part_prop finishes.")

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

latest_k = int(metrics["k"].dropna().max()) if "k" in metrics.columns else int(len(metrics) - 1)
latest = metrics[metrics["k"].astype("Int64").eq(latest_k)].iloc[-1] if "k" in metrics.columns else metrics.iloc[-1]

summary_cols = st.columns(8)
summary_cols[0].metric("iteration", latest_k)
summary_cols[1].metric("edge explore n", fmt_int(latest.get("alpha_explore_t", edge_explore_value)))
summary_cols[2].metric("z mean", fmt_float(latest.get("zscore_mean", np.nan)))
summary_cols[3].metric("z max", fmt_float(latest.get("zscore_max", np.nan)))
summary_cols[4].metric("p min", fmt_float(latest.get("pval_min", np.nan), nd=4))
summary_cols[5].metric("PW scale", fmt_float(latest.get("pw_scale", np.nan)))
summary_cols[6].metric("PW action", str(latest.get("pw_action", "NA")))
summary_cols[7].metric("freeze", str(latest.get("freeze_expansion", False)))


tab_live, tab_curves, tab_iter, tab_plots, tab_arrays, tab_tops, tab_log, tab_trace, tab_runs = st.tabs([
    "live summary",
    "metric curves",
    "iteration browser",
    "plot gallery",
    "arrays",
    "top edges",
    "console log",
    "trace",
    "runs",
])


with tab_live:
    st.subheader("latest dashboard")

    latest_summary = latest_image_for_section(plot_df, run_dir, "summary_dashboard")

    if latest_summary is not None:
        st.image(str(latest_summary), caption=str(latest_summary), width="stretch")
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
                st.image(str(p), width="stretch")
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
        "MCTS x/tf growth": [
            "mcts_node_mu_count",
            "mcts_edge_mu_count",
            "mcts_edge_explore_count_total",
            "mcts_trace_len",
            "mcts_trace_written",
            "mcts_trace_kept",
        ],
        "alpha growth": [
            "alpha_decision_mu_count",
            "alpha_node_mu_count",
            "alpha_edge_mu_count",
        ],
        "adaptive progressive widening": [
            "uct_explore_influence",
            "uct_explore_influence_ema",
            "pw_freeze_threshold",
            "pw_unfreeze_threshold",
            "pw_scale",
            "pw_c",
            "expand_prob",
        ],
        "depth / alpha": [
            "depth_prob_delta_sum",
            "alpha_depth_prob_delta_sum",
            "alpha_sensor_prob",
        ],
    }

    for title, cols in chart_groups.items():
        have = safe_metric_cols(metrics, cols)

        if have:
            st.write(title)
            plot_df_metric = metrics.copy()
            plot_df_metric["k"] = pd.to_numeric(plot_df_metric["k"], errors="coerce")
            st.line_chart(plot_df_metric.set_index("k")[have])

    st.subheader("raw metrics")
    st.dataframe(metrics, width="stretch", height=380)


with tab_iter:
    st.subheader("iteration browser")

    latest_k = int(latest_k)

    if latest_k <= 0:
        selected_k = 0
        st.info("Only iteration 0 is available so far. The iteration slider will appear after iteration 1 is saved.")
    else:
        selected_k = safe_iteration_selectbox_or_slider(
            "iteration",
            0,
            latest_k,
            latest_k,
        )

    row = metrics[metrics["k"].astype("Int64").eq(selected_k)]
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
        "original_depth_helper_plots",
        "original_exploration_influence",
        "original_policy_drift",
    ]

    section = st.selectbox("plot section", preferred_sections)
    imgs = get_section_images(plot_df, run_dir, selected_k, section)

    if imgs:
        for p in imgs:
            st.image(str(p), caption=str(p), width="stretch")
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
            k_gallery = safe_iteration_selectbox_or_slider(
                "gallery iteration",
                min_k,
                max_k,
                max_k,
                key="gallery_k",
            )
            df_sec = df_sec[df_sec["k"].astype("Int64").eq(k_gallery)]

        for p in df_sec["path"].tolist():
            rp = resolve_path(run_dir, p)
            if rp is not None:
                st.image(str(rp), caption=str(rp), width="stretch")


with tab_arrays:
    st.subheader("stored arrays")

    selected_k_arr = safe_iteration_selectbox_or_slider(
        "array iteration",
        0,
        latest_k,
        latest_k,
        key="arr_k",
    )
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
                c2.metric("mean", fmt_float(np.mean(numeric), nd=6))
                c3.metric("std", fmt_float(np.std(numeric), nd=6))
                c4.metric("min", fmt_float(np.min(numeric), nd=6))
                c5.metric("max", fmt_float(np.max(numeric), nd=6))

                show_n = min(5000, numeric.size)
                st.line_chart(numeric[:show_n])

                if numeric.size > show_n:
                    st.caption(f"showing first {show_n} values")

        st.write(arr)


with tab_tops:
    st.subheader("top edges / alpha decisions")

    selected_k_top = safe_iteration_selectbox_or_slider(
        "tops iteration",
        0,
        latest_k,
        latest_k,
        key="tops_k",
    )
    tops_path = run_dir / "iterations" / f"iter_{selected_k_top:04d}" / "tops.json"
    tops = read_json(tops_path, default={})

    if not tops:
        st.info("No tops file for this iteration.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("top x/tf edges")
            st.dataframe(pd.DataFrame(tops.get("top_edges", [])), width="stretch")

        with c2:
            st.write("top alpha decisions")
            st.dataframe(pd.DataFrame(tops.get("top_alpha_decisions", [])), width="stretch")

        with c3:
            st.write("top alpha edges")
            st.dataframe(pd.DataFrame(tops.get("top_alpha_edges", [])), width="stretch")


with tab_log:
    st.subheader("console log tail")
    st.text_area(
        "console.log",
        read_text_tail(run_dir / "console.log", n_lines=log_tail_lines),
        height=620,
    )


with tab_trace:
    st.subheader("MCTS trace tail")

    trace_path = run_dir / "mcts_trace.jsonl"

    if not trace_path.exists():
        st.info("No mcts_trace.jsonl file yet. This is expected if trace draining is disabled or no trace has been written.")
    else:
        st.caption(str(trace_path))
        st.text_area(
            "mcts_trace.jsonl tail",
            read_trace_tail(run_dir, n_lines=trace_tail_lines),
            height=620,
        )


with tab_runs:
    st.subheader("available runs")

    rows = []

    for p in list_run_dirs(run_root):
        status_p = read_json(p / "status.json", default={})
        metrics_p = load_metrics(p)
        latest_p = None if metrics_p.empty else metrics_p.iloc[-1]

        rows.append({
            "run": p.name,
            "path": str(p),
            "status": status_p.get("status"),
            "last_iteration": status_p.get("last_iteration"),
            "edge_explore_n": metric_value(
                status_p,
                latest_p,
                "last_mcts_edge_explore_count_total",
                "mcts_edge_explore_count_total",
            ),
            "last_h": metric_value(status_p, latest_p, "last_h", "policy_h"),
            "updated_at": status_p.get("updated_at"),
            "n_metric_rows": int(len(metrics_p)),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        })

    st.dataframe(pd.DataFrame(rows), width="stretch")


if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
