import os
import sys
import gc
import csv
import time
import signal
import shutil
import tempfile
import faulthandler
import threading
import subprocess
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:
    raise ImportError("Install psutil first with: pip install psutil")


faulthandler.enable()

_DIAG_STOP = threading.Event()
_DIAG_THREAD = None


def _gb(x):
    return x / (1024 ** 3)


def _safe_num_fds(proc):
    try:
        return proc.num_fds()
    except Exception:
        return None


def _safe_open_files(proc):
    try:
        return len(proc.open_files())
    except Exception:
        return None


def _safe_children_rss(proc):
    total = 0
    n = 0

    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
            n += 1
        except Exception:
            pass

    return total, n


def _safe_dir_size(path):
    path = Path(path)

    if not path.exists():
        return 0, 0

    total_size = 0
    file_count = 0

    try:
        for p in path.rglob("*"):
            if p.is_file():
                file_count += 1
                total_size += p.stat().st_size
    except Exception:
        pass

    return total_size, file_count


def _safe_gpu_info():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        if not out:
            return None

        parts = out.splitlines()[0].split(",")
        used_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        util = float(parts[2].strip())

        return used_mb, total_mb, util

    except Exception:
        return None


def print_top_arrays(ns, n=12):
    """
    Prints the largest numpy arrays visible in a namespace.

    Usage:
        print_top_arrays(globals())
    """

    rows = []

    for name, obj in ns.items():
        try:
            if isinstance(obj, np.ndarray):
                rows.append((obj.nbytes, name, obj.shape, obj.dtype))
        except Exception:
            pass

    rows.sort(reverse=True)

    print("\nlargest numpy arrays:")
    for nbytes, name, shape, dtype in rows[:n]:
        print(f"  {name:30s} {_gb(nbytes):8.3f} GB  shape={shape} dtype={dtype}")

    if not rows:
        print("  no numpy arrays found in provided namespace")


def diag_mark(label="", ns=None, tmp_dirs=None):
    """
    Manual checkpoint diagnostic.

    Put this inside your loop every N iterations.

    Example:
        if i % 25 == 0:
            diag_mark(f'iter={i}', globals(), tmp_dirs=['/tmp/mcts_streamlit'])
    """

    proc = psutil.Process(os.getpid())

    rss = proc.memory_info().rss
    vms = proc.memory_info().vms
    child_rss, child_n = _safe_children_rss(proc)

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    disk = shutil.disk_usage(".")

    gpu = _safe_gpu_info()

    msg = (
        f"[diag {label}] "
        f"rss={_gb(rss):.2f}GB "
        f"vms={_gb(vms):.2f}GB "
        f"children={child_n} "
        f"child_rss={_gb(child_rss):.2f}GB "
        f"sys_mem={vm.percent:.1f}% "
        f"swap={sw.percent:.1f}% "
        f"threads={proc.num_threads()} "
        f"fds={_safe_num_fds(proc)} "
        f"open_files={_safe_open_files(proc)} "
        f"disk_free={_gb(disk.free):.1f}GB"
    )

    if gpu is not None:
        used_mb, total_mb, util = gpu
        msg += f" gpu={used_mb:.0f}/{total_mb:.0f}MB util={util:.0f}%"

    print(msg, flush=True)

    if tmp_dirs is not None:
        for tmp_dir in tmp_dirs:
            size, count = _safe_dir_size(tmp_dir)
            print(
                f"  tmp_dir={tmp_dir} files={count} size={_gb(size):.3f}GB",
                flush=True,
            )

    if ns is not None:
        print_top_arrays(ns)

    gc.collect()


def start_diag_monitor(
    log_path="loop_diagnostics.csv",
    interval=5,
    tmp_dirs=None,
):
    """
    Starts a background monitor that writes diagnostics to CSV.

    This is useful because the final terminal output may disappear when
    VS Code, Jupyter, Streamlit, or WSL crashes.

    Example:
        start_diag_monitor(
            log_path='loop_diagnostics.csv',
            interval=5,
            tmp_dirs=['/tmp/mcts_streamlit']
        )
    """

    global _DIAG_THREAD

    if tmp_dirs is None:
        tmp_dirs = []

    log_path = Path(log_path)

    header = [
        "time",
        "pid",
        "rss_gb",
        "vms_gb",
        "child_count",
        "child_rss_gb",
        "system_mem_percent",
        "swap_percent",
        "threads",
        "fds",
        "open_files",
        "disk_free_gb",
        "gpu_used_mb",
        "gpu_total_mb",
        "gpu_util_percent",
    ]

    for tmp_dir in tmp_dirs:
        safe_name = str(tmp_dir).replace("/", "_").replace("\\", "_")
        header.append(f"tmp_files_{safe_name}")
        header.append(f"tmp_gb_{safe_name}")

    def worker():
        proc = psutil.Process(os.getpid())

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.flush()

            while not _DIAG_STOP.is_set():
                try:
                    rss = proc.memory_info().rss
                    vms = proc.memory_info().vms
                    child_rss, child_n = _safe_children_rss(proc)

                    vm = psutil.virtual_memory()
                    sw = psutil.swap_memory()
                    disk = shutil.disk_usage(".")

                    gpu = _safe_gpu_info()
                    if gpu is None:
                        gpu_used, gpu_total, gpu_util = None, None, None
                    else:
                        gpu_used, gpu_total, gpu_util = gpu

                    row = [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        os.getpid(),
                        round(_gb(rss), 4),
                        round(_gb(vms), 4),
                        child_n,
                        round(_gb(child_rss), 4),
                        vm.percent,
                        sw.percent,
                        proc.num_threads(),
                        _safe_num_fds(proc),
                        _safe_open_files(proc),
                        round(_gb(disk.free), 4),
                        gpu_used,
                        gpu_total,
                        gpu_util,
                    ]

                    for tmp_dir in tmp_dirs:
                        size, count = _safe_dir_size(tmp_dir)
                        row.append(count)
                        row.append(round(_gb(size), 4))

                    writer.writerow(row)
                    f.flush()

                except Exception as e:
                    try:
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "diag_error", repr(e)])
                        f.flush()
                    except Exception:
                        pass

                time.sleep(interval)

    _DIAG_STOP.clear()
    _DIAG_THREAD = threading.Thread(target=worker, daemon=True)
    _DIAG_THREAD.start()

    print(f"diagnostic monitor started: {log_path.resolve()}", flush=True)


def stop_diag_monitor():
    global _DIAG_THREAD

    _DIAG_STOP.set()

    if _DIAG_THREAD is not None:
        _DIAG_THREAD.join(timeout=2)

    print("diagnostic monitor stopped", flush=True)

import os
import csv
import time
import psutil
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


_DIAG_PATH = Path("end_loop_memory_diag.csv")
_PROC = psutil.Process(os.getpid())
_LAST_RSS = None
_LAST_VMS = None


def _gb(x):
    return x / (1024 ** 3)


def init_end_loop_diag(path="end_loop_memory_diag.csv"):
    global _DIAG_PATH

    _DIAG_PATH = Path(path)

    with open(_DIAG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "k",
            "rss_gb",
            "rss_delta_gb",
            "vms_gb",
            "vms_delta_gb",
            "sys_mem_percent",
            "swap_percent",
            "child_count",
            "child_rss_gb",
            "threads",
            "open_figures",
        ])

    print(f"memory diagnostic logging to: {_DIAG_PATH.resolve()}", flush=True)


def end_loop_diag(k, print_every=1):
    """
    Lightweight end-of-loop memory diagnostic.

    Call this only at the very end of your loop body.

    Example:
        init_end_loop_diag()

        for k in range(...):
            ...
            end_loop_diag(k, print_every=10)
    """

    global _LAST_RSS, _LAST_VMS

    mem = _PROC.memory_info()
    rss = mem.rss
    vms = mem.vms

    rss_delta = 0 if _LAST_RSS is None else rss - _LAST_RSS
    vms_delta = 0 if _LAST_VMS is None else vms - _LAST_VMS

    _LAST_RSS = rss
    _LAST_VMS = vms

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    child_count = 0
    child_rss = 0

    try:
        children = _PROC.children(recursive=True)
        child_count = len(children)

        for child in children:
            try:
                child_rss += child.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass

    if plt is not None:
        try:
            open_figures = len(plt.get_fignums())
        except Exception:
            open_figures = None
    else:
        open_figures = None

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        k,
        round(_gb(rss), 4),
        round(_gb(rss_delta), 4),
        round(_gb(vms), 4),
        round(_gb(vms_delta), 4),
        vm.percent,
        sw.percent,
        child_count,
        round(_gb(child_rss), 4),
        _PROC.num_threads(),
        open_figures,
    ]

    with open(_DIAG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if k % print_every == 0:
        print(
            f"[end k={k}] "
            f"rss={_gb(rss):.3f}GB "
            f"drss={_gb(rss_delta):+.4f}GB "
            f"vms={_gb(vms):.3f}GB "
            f"sys={vm.percent:.1f}% "
            f"swap={sw.percent:.1f}% "
            f"children={child_count} "
            f"child_rss={_gb(child_rss):.3f}GB "
            f"figs={open_figures}",
            flush=True,
        )


import os
import gc
import sys
import csv
import time
import types
import ctypes
import itertools
from pathlib import Path

import numpy as np
import psutil

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import scipy.sparse as sp
except Exception:
    sp = None

try:
    import torch
except Exception:
    torch = None


_PROC = psutil.Process(os.getpid())
_LAST_SIZES = {}
_VAR_DIAG_PATH = Path("end_loop_variable_growth.csv")


def _gb(x):
    return x / (1024 ** 3)


def _mb(x):
    return x / (1024 ** 2)


def _fmt_mb(x):
    return f"{_mb(x):,.2f} MB"


def _is_skip_obj(obj):
    return isinstance(
        obj,
        (
            types.ModuleType,
            types.FunctionType,
            types.BuiltinFunctionType,
            type,
        ),
    )


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None


def _obj_desc(obj):
    try:
        if isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape} dtype={obj.dtype}"

        if pd is not None and isinstance(obj, pd.DataFrame):
            return f"DataFrame shape={obj.shape}"

        if pd is not None and isinstance(obj, pd.Series):
            return f"Series shape={obj.shape} dtype={obj.dtype}"

        if sp is not None and sp.issparse(obj):
            return f"sparse shape={obj.shape} nnz={obj.nnz}"

        if torch is not None and isinstance(obj, torch.Tensor):
            return f"torch shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}"

        if isinstance(obj, dict):
            return f"dict len={len(obj)}"

        if isinstance(obj, list):
            return f"list len={len(obj)}"

        if isinstance(obj, tuple):
            return f"tuple len={len(obj)}"

        if isinstance(obj, set):
            return f"set len={len(obj)}"

        return type(obj).__name__

    except Exception:
        return type(obj).__name__


def approx_nbytes(obj, depth=2, max_items=100, seen=None):
    """
    Approximate retained size of an object.

    This is intentionally approximate. It is meant to identify likely growing
    variables, not produce perfect memory accounting.
    """

    if seen is None:
        seen = set()

    obj_id = id(obj)

    if obj_id in seen:
        return 0

    seen.add(obj_id)

    try:
        if obj is None:
            return 0

        if _is_skip_obj(obj):
            return 0

        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)

        if pd is not None and isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(index=True, deep=True).sum())

        if pd is not None and isinstance(obj, pd.Series):
            return int(obj.memory_usage(index=True, deep=True))

        if sp is not None and sp.issparse(obj):
            total = 0
            for attr in ("data", "indices", "indptr", "row", "col"):
                arr = getattr(obj, attr, None)
                if arr is not None:
                    total += getattr(arr, "nbytes", 0)
            return int(total)

        if torch is not None and isinstance(obj, torch.Tensor):
            return int(obj.nelement() * obj.element_size())

        if isinstance(obj, (str, bytes, bytearray)):
            return int(sys.getsizeof(obj))

        base = int(sys.getsizeof(obj))

        if depth <= 0:
            return base

        if isinstance(obj, dict):
            n = len(obj)

            if n == 0:
                return base

            sampled = list(itertools.islice(obj.items(), max_items))
            subtotal = 0

            for k, v in sampled:
                subtotal += approx_nbytes(k, depth - 1, max_items, seen)
                subtotal += approx_nbytes(v, depth - 1, max_items, seen)

            scale = n / max(len(sampled), 1)
            return int(base + subtotal * scale)

        if isinstance(obj, (list, tuple, set, frozenset)):
            n = len(obj)

            if n == 0:
                return base

            sampled = list(itertools.islice(obj, max_items))
            subtotal = 0

            for item in sampled:
                subtotal += approx_nbytes(item, depth - 1, max_items, seen)

            scale = n / max(len(sampled), 1)
            return int(base + subtotal * scale)

        if hasattr(obj, "__dict__"):
            subtotal = 0

            for _, v in itertools.islice(vars(obj).items(), max_items):
                subtotal += approx_nbytes(v, depth - 1, max_items, seen)

            return int(base + subtotal)

        return base

    except Exception:
        try:
            return int(sys.getsizeof(obj))
        except Exception:
            return 0


def _iter_subobjects(parent_name, obj, max_subitems=80):
    """
    Yields direct child objects so we can see things like:
        population._X_inst
        evaluation['F']
        history[0]
    """

    try:
        if isinstance(obj, dict):
            for k, v in itertools.islice(obj.items(), max_subitems):
                yield f"{parent_name}[{repr(k)}]", v

        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(itertools.islice(obj, max_subitems)):
                yield f"{parent_name}[{i}]", v

        elif hasattr(obj, "__dict__"):
            for k, v in itertools.islice(vars(obj).items(), max_subitems):
                yield f"{parent_name}.{k}", v

    except Exception:
        return


def collect_namespace_sizes(
    global_ns,
    local_ns=None,
    min_mb=1.0,
    depth=2,
    max_items=100,
    max_subitems=80,
):
    rows = []
    min_bytes = min_mb * 1024 ** 2

    namespaces = [("g", global_ns)]

    if local_ns is not None and local_ns is not global_ns:
        namespaces.append(("l", local_ns))

    seen_paths = set()

    for ns_label, ns in namespaces:
        for name, obj in list(ns.items()):
            if name.startswith("__"):
                continue

            if _is_skip_obj(obj):
                continue

            path = f"{ns_label}:{name}"

            if path in seen_paths:
                continue

            seen_paths.add(path)

            size = approx_nbytes(obj, depth=depth, max_items=max_items)

            if size >= min_bytes:
                rows.append({
                    "path": path,
                    "size": size,
                    "type": type(obj).__name__,
                    "desc": _obj_desc(obj),
                })

            for subpath, subobj in _iter_subobjects(path, obj, max_subitems=max_subitems):
                if _is_skip_obj(subobj):
                    continue

                subsize = approx_nbytes(subobj, depth=max(depth - 1, 0), max_items=max_items)

                if subsize >= min_bytes:
                    rows.append({
                        "path": subpath,
                        "size": subsize,
                        "type": type(subobj).__name__,
                        "desc": _obj_desc(subobj),
                    })

    rows.sort(key=lambda r: r["size"], reverse=True)
    return rows


def init_var_growth_diag(path="end_loop_variable_growth.csv"):
    global _VAR_DIAG_PATH, _LAST_SIZES

    _VAR_DIAG_PATH = Path(path)
    _LAST_SIZES = {}

    with open(_VAR_DIAG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "k",
            "path",
            "size_mb",
            "delta_mb",
            "type",
            "desc",
            "rss_gb",
            "vms_gb",
            "sys_mem_percent",
        ])

    print(f"variable growth diagnostic logging to: {_VAR_DIAG_PATH.resolve()}", flush=True)


def end_loop_var_growth_diag(
    k,
    global_ns,
    local_ns=None,
    scan_every=10,
    top_n=25,
    min_mb=1.0,
    depth=2,
    max_items=100,
    do_gc=True,
    trim_test=False,
):
    """
    End-of-loop variable growth diagnostic.

    Use at the very end of your loop.

    Parameters
    ----------
    k
        Loop counter.
    global_ns
        Usually globals().
    local_ns
        Usually locals() if your loop is inside a function.
    scan_every
        Only performs the variable scan every this many iterations.
    top_n
        Number of largest/growing variables to print.
    min_mb
        Ignore variables smaller than this.
    depth
        Recursive depth for approximate size estimation.
    max_items
        Max number of items sampled inside lists/dicts/objects.
    do_gc
        Runs gc.collect() before measuring.
    trim_test
        Linux/WSL-only test. Calls malloc_trim(0) to see whether RSS can be
        released back to the OS.
    """

    global _LAST_SIZES

    if k % scan_every != 0:
        return

    if do_gc:
        gc.collect()

    rss_before = _PROC.memory_info().rss

    rows = collect_namespace_sizes(
        global_ns=global_ns,
        local_ns=local_ns,
        min_mb=min_mb,
        depth=depth,
        max_items=max_items,
    )

    current_sizes = {r["path"]: r["size"] for r in rows}

    for r in rows:
        old = _LAST_SIZES.get(r["path"], 0)
        r["delta"] = r["size"] - old

    grew = [r for r in rows if r["delta"] > 0]
    grew.sort(key=lambda r: r["delta"], reverse=True)

    mem = _PROC.memory_info()
    vm = psutil.virtual_memory()

    print(
        f"\n[var diag k={k}] "
        f"rss={_gb(mem.rss):.3f}GB "
        f"vms={_gb(mem.vms):.3f}GB "
        f"sys={vm.percent:.1f}%",
        flush=True,
    )

    print("\nlargest visible variables:")
    for r in rows[:top_n]:
        print(
            f"  {_fmt_mb(r['size']):>12s} "
            f"delta={_fmt_mb(r['delta']):>12s} "
            f"{r['path']:<45s} "
            f"{r['desc']}",
            flush=True,
        )

    print("\nmost grown since last scan:")
    if not grew:
        print("  no visible variable growth detected", flush=True)
    else:
        for r in grew[:top_n]:
            print(
                f"  +{_fmt_mb(r['delta']):>11s} "
                f"now={_fmt_mb(r['size']):>12s} "
                f"{r['path']:<45s} "
                f"{r['desc']}",
                flush=True,
            )

    with open(_VAR_DIAG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        for r in rows:
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                k,
                r["path"],
                round(_mb(r["size"]), 4),
                round(_mb(r["delta"]), 4),
                r["type"],
                r["desc"],
                round(_gb(mem.rss), 4),
                round(_gb(mem.vms), 4),
                vm.percent,
            ])

    _LAST_SIZES = current_sizes

    if trim_test:
        rss_pre_trim = _PROC.memory_info().rss

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            trim_ok = True
        except Exception:
            trim_ok = False

        rss_post_trim = _PROC.memory_info().rss

        if trim_ok:
            print(
                f"\nmalloc_trim test: "
                f"rss before={_gb(rss_pre_trim):.3f}GB "
                f"after={_gb(rss_post_trim):.3f}GB "
                f"released={_gb(rss_pre_trim - rss_post_trim):.3f}GB",
                flush=True,
            )
        else:
            print("\nmalloc_trim unavailable on this system", flush=True)