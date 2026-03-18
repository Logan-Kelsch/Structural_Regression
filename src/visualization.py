import matplotlib.pyplot as plt
import numpy as np

def plot_flags(inst_inst: np.ndarray, cols=(2, 3, 4), bits=range(5, 10)):
    """
    For each flag column in `cols`, decode bits [5..9] and visualize them.

    Visualization: one figure per flag-column. For each bit b, plot row indices
    where that bit is set (y=b). This shows which of {5,6,7,8,9} are "present"
    per row, separated by flag column.

    Assumes inst_inst[:, col] stores integer flags (possibly as float32).
    """
    if inst_inst.ndim != 2 or inst_inst.shape[1] <= max(cols):
        raise ValueError(f"inst_inst must be 2D and have at least {max(cols)+1} columns.")

    bits = np.asarray(list(bits), dtype=np.uint32)
    row_idx = np.arange(inst_inst.shape[0], dtype=np.int32)

    for c in cols:
        flags = inst_inst[:, c].astype(np.uint32, copy=False)

        # flagged[r, j] True if bit bits[j] is set in row r
        flagged = ((flags[:, None] >> bits[None, :]) & np.uint32(1)).astype(bool)

        plt.figure()
        any_on = False
        for j, b in enumerate(bits):
            on = flagged[:, j]
            if np.any(on):
                any_on = True
                plt.scatter(row_idx[on], np.full(int(on.sum()), int(b), dtype=np.int16), s=6)

        plt.title(f"Decoded bits {int(bits[0])}..{int(bits[-1])} from inst_inst[:, {c}]")
        plt.xlabel("Row index")
        plt.ylabel("Bit number (decoded)")
        plt.yticks([int(b) for b in bits])
        plt.grid(True)

        if not any_on:
            # still show an empty plot with correct axes
            plt.ylim(int(bits[0]) - 0.5, int(bits[-1]) + 0.5)

        plt.show()

def plot_instruction_demo(inst_inst):
    #column 0
    plt.title('Histogram of Initialized Transformation Function IDs')
    plt.hist(inst_inst[:, 1], bins=22)
    plt.show()
    #column 5
    plt.title('Sensor Index for each $x$')
    plt.hist(inst_inst[:, 5], bins=12)
    plt.show()

    # flag column (stored as float32 in inst_inst) -> cast back to uint32 for bit checks
    flags = inst_inst[:, 3].astype(np.uint32, copy=False)

    # (data_col, bit, title, bins)
    plots = [
        (6, 6, r"Histogram of Initialized Values for each $\alpha$ (flag bit 6 set)", 30),
        (7, 7, r"Histogram of Initialized Values for each $\Delta_1$ (flag bit 7 set)", 20),
        (8, 8, r"Histogram of Initialized Values for each $\Delta_2$ (flag bit 8 set)", 20),
        (9, 9, r"Histogram of Initialized Values for each $\kappa$ (flag bit 9 set)", 20),
    ]

    for col, bit, title, bins in plots:
        mask = (flags & (np.uint32(1) << np.uint32(bit))) != 0
        vals = inst_inst[mask, col]

        plt.figure()
        plt.title(title + f"\n(n={vals.size})")
        plt.hist(vals, bins=bins)
        plt.grid(True)
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_by_scale(X, N_sample: int = 100):
    """
    Plot the first N_sample rows of a (N, G) array, automatically splitting columns
    into stacked subplots so that within each subplot, every column has a robust
    vertical span of at least ~1/4 of that subplot's y-range.

    Grouping is greedy on a magnitude key; y-ranges are based on robust percentiles
    (5th..95th) to avoid single outliers blowing up the scale.

    Parameters
    ----------
    X : array-like, shape (N, G) or (N,)
    N_sample : int, number of rows to plot from the start

    Returns
    -------
    fig, axes, groups
        groups is a list of lists of column indices per subplot.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, G). Got shape {X.shape}")

    N = X.shape[0]
    G = X.shape[1]
    M = int(min(max(N_sample, 1), N))
    Y = X[:M, :]

    # Robust per-column bounds (ignore NaNs). Fall back to 0 for all-NaN columns.
    all_nan = np.all(np.isnan(Y), axis=0)
    p5 = np.empty(G, dtype=float)
    p95 = np.empty(G, dtype=float)

    if np.any(~all_nan):
        p5[~all_nan] = np.nanpercentile(Y[:, ~all_nan], 5, axis=0)
        p95[~all_nan] = np.nanpercentile(Y[:, ~all_nan], 95, axis=0)
    p5[all_nan] = 0.0
    p95[all_nan] = 0.0

    span = p95 - p5
    # Epsilon to avoid zero-span columns wrecking grouping math
    pos = span[span > 0]
    eps = (np.nanmedian(pos) * 1e-3) if pos.size else 1e-12
    eps = max(eps, 1e-12)
    span_safe = np.where(span > eps, span, eps)

    center = 0.5 * (p5 + p95)

    # Sort columns by an overall magnitude proxy, then by span
    mag_key = np.log10(np.abs(center) + span_safe + 1e-12)
    span_key = np.log10(span_safe + 1e-12)
    order = np.lexsort((span_key, mag_key))

    # Greedy grouping enforcing: group_range <= 4 * min_span_in_group
    groups = []
    current = []
    g_low = np.inf
    g_high = -np.inf
    g_min_span = np.inf

    for c in order:
        low_c, high_c, span_c = float(p5[c]), float(p95[c]), float(span_safe[c])

        if not current:
            current = [int(c)]
            g_low, g_high = low_c, high_c
            g_min_span = span_c
            continue

        new_low = min(g_low, low_c)
        new_high = max(g_high, high_c)
        new_range = new_high - new_low
        new_min_span = min(g_min_span, span_c)

        if new_range <= 4.0 * new_min_span:
            current.append(int(c))
            g_low, g_high = new_low, new_high
            g_min_span = new_min_span
        else:
            groups.append(current)
            current = [int(c)]
            g_low, g_high = low_c, high_c
            g_min_span = span_c

    if current:
        groups.append(current)

    # Plot
    fig, axes = plt.subplots(
        nrows=len(groups),
        ncols=1,
        sharex=True,
        figsize=(12, 2.8 * len(groups)),
        constrained_layout=True
    )
    if len(groups) == 1:
        axes = [axes]

    x = np.arange(M)

    for ax, cols in zip(axes, groups):
        cols_sorted = sorted(cols)
        for c in cols_sorted:
            ax.plot(x, Y[:, c], label=f"col {c}")

        low = float(np.min(p5[cols_sorted]))
        high = float(np.max(p95[cols_sorted]))
        rng = high - low
        pad = 0.05 * (rng if rng > 0 else 1.0)
        ax.set_ylim(low - pad, high + pad)

        ax.grid(True, alpha=0.3)

        if len(cols_sorted) <= 10:
            ax.legend(loc="upper right", fontsize=8, ncol=min(3, len(cols_sorted)))
        else:
            ax.set_title(f"{len(cols_sorted)} columns (legend hidden)", fontsize=10)

    axes[-1].set_xlabel("sample index")
    axes[0].set_title(
        f"Grouped plot (first {M} samples) — {G} columns split into {len(groups)} subplot(s)",
        fontsize=11
    )

    plt.show()
    return fig, axes, groups

import numpy as np
import matplotlib.pyplot as plt

def plot_intraday_overlay(population, G: int, *, alpha: float = 0.08, linewidth: float = 0.8):
    """
    Overlay every intraday path for column G using the time-since-market-open column
    at population._X_inst[:, population._T_idx[-2]].

    Assumes:
      - population._X_inst is 2D with shape (N, M)
      - population._T_idx[-2] is the "minutes since market open" column
      - each day starts where that column == 0

    If exact zeros are not found reliably, it falls back to splitting whenever the
    time column decreases from one row to the next.
    """
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")
    if not hasattr(population, "_T_idx"):
        raise AttributeError("population must have attribute '_T_idx'")

    X = population._X_inst
    if not isinstance(X, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("population._X_inst must be 2D")
    if len(population._T_idx) < 2:
        raise ValueError("population._T_idx must have at least two entries")
    if G < 0 or G >= X.shape[1]:
        raise ValueError(f"G={G} is out of bounds for X_inst with {X.shape[1]} columns")

    t_col = int(population._T_idx[-2])
    if t_col < 0 or t_col >= X.shape[1]:
        raise ValueError(f"time column index {t_col} is out of bounds")

    t = X[:, t_col]
    y = X[:, G]

    # Primary split rule: exact 0 means a new day start
    day_starts = np.flatnonzero(t == 0)

    # Fallback: if zeros are scarce/missing, split on wrap/decrease
    if day_starts.size == 0:
        day_starts = np.concatenate(([0], np.flatnonzero(np.diff(t) < 0) + 1))

    # Ensure row 0 is included as a segment start
    if day_starts[0] != 0:
        day_starts = np.concatenate(([0], day_starts))

    day_ends = np.empty_like(day_starts)
    day_ends[:-1] = day_starts[1:]
    day_ends[-1] = X.shape[0]

    plt.figure(figsize=(10, 6))

    days_plotted = 0
    for s, e in zip(day_starts, day_ends):
        if e - s <= 1:
            continue

        x_day = t[s:e]
        y_day = y[s:e]

        # sort by x within day just in case
        order = np.argsort(x_day)
        plt.plot(x_day[order], y_day[order], alpha=alpha, linewidth=linewidth)
        days_plotted += 1

    plt.title(f"Intraday Overlay for Column {G} ({days_plotted} days)")
    plt.xlabel("Minutes Since Market Open")
    plt.ylabel(f"X_inst[:, {G}]")
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def show_consecutive_entry_paths(
    raw_emissions,
    evaluation_mask,
    p,
    *,
    ax=None,
    break_on_mask_gap=True,
    show=True,
):
    """
    Plot each consecutive run of True values in `p`, using `raw_emissions` as y.

    Parameters
    ----------
    raw_emissions : 1d array-like
        Real-valued series to plot on the y-axis.
    evaluation_mask : 1d array-like of bool
        Only points where this is True are considered.
    p : 1d array-like of bool
        Boolean signal whose consecutive True runs define the paths.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If None, a new figure/axis is created.
    break_on_mask_gap : bool, default=True
        If True, an index with evaluation_mask=False breaks any active run.
        If False, masked-out indices are skipped without breaking the run.
    show : bool, default=True
        Whether to call plt.show() when a new figure is created.

    Returns
    -------
    runs : list of tuple[np.ndarray, np.ndarray]
        A list of (x, y) pairs for each plotted run.
    ax : matplotlib.axes.Axes
        The axis the plot was drawn on.
    """
    raw_emissions = np.asarray(raw_emissions)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
    p = np.asarray(p, dtype=bool)

    if raw_emissions.ndim != 1 or evaluation_mask.ndim != 1 or p.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if not (len(raw_emissions) == len(evaluation_mask) == len(p)):
        raise ValueError("All inputs must have the same length.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    runs = []
    current_y = []

    for y, mask_ok, is_true in zip(raw_emissions, evaluation_mask, p):
        if not mask_ok:
            if break_on_mask_gap and current_y:
                x = np.arange(1, len(current_y) + 1)
                runs.append((x, np.asarray(current_y)))
                current_y = []
            continue

        if is_true:
            current_y.append(y)
        else:
            if current_y:
                x = np.arange(1, len(current_y) + 1)
                runs.append((x, np.asarray(current_y)))
                current_y = []

    if current_y:
        x = np.arange(1, len(current_y) + 1)
        runs.append((x, np.asarray(current_y)))

    for i, (x, y) in enumerate(runs, start=1):
        ax.plot(x, y, linewidth=1, alpha=0.2, label=f"run {i}")

    ax.set_xlabel("Consecutive instances of participation")
    ax.set_ylabel("Raw Emissions")

    ax.hlines(0, xmin=1, xmax=5, colors='black')

    ax.set_title("Raw emissions along consecutive participation")

    ax.grid(True, alpha=0.3)

    if runs and len(runs) <= 15:
        ax.legend()

    if created_fig and show:
        plt.show()

    return runs, ax

import evaluation as _E

import reproduction as _R

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def purge_indistinguishable_plot(population, evaluation, threshold=0.02, figsize=(10, 8), f_decimals=2):
    """
    Same behavior as purge_indistinguishable, but also visualizes the
    pairwise absolute-correlation relation and which models survive.

    Returns
    -------
    survivors : np.ndarray
        Surviving original column indices c
    """
    X = np.asarray(population._X_inst)
    G_idx = np.asarray(population._G_idx, dtype=int)
    F = np.asarray(evaluation["F"])

    if X.ndim != 2:
        raise ValueError("population._X_inst must be a 2d array")
    if F.ndim != 1:
        raise ValueError('evaluation["F"] must be a 1d array')
    if np.any(G_idx < 0) or np.any(G_idx >= X.shape[1]):
        raise ValueError("population._G_idx contains invalid column indices")
    if np.any(G_idx >= F.shape[0]):
        raise ValueError('evaluation["F"] is too short for indices in population._G_idx')

    #only positive F models
    pos_mask = F[G_idx] > 0
    cols = G_idx[pos_mask]

    if cols.size == 0:
        print("no models with F > 0")
        return np.array([], dtype=int)

    X_sub = X[:, cols].astype(np.float64, copy=False)
    F_sub = F[cols].astype(np.float64, copy=False)
    n_models = cols.size

    #build abs corr robustly
    std = X_sub.std(axis=0)
    nonconst = std > 0
    abs_corr = np.zeros((n_models, n_models), dtype=np.float64)

    if np.any(nonconst):
        corr_nc = np.corrcoef(X_sub[:, nonconst], rowvar=False)
        corr_nc = np.nan_to_num(corr_nc, nan=0.0)
        abs_corr[np.ix_(nonconst, nonconst)] = np.abs(corr_nc)

    const_idx = np.where(~nonconst)[0]
    if const_idx.size:
        for a in range(const_idx.size):
            ia = const_idx[a]
            abs_corr[ia, ia] = 1.0
            for b in range(a + 1, const_idx.size):
                ib = const_idx[b]
                if np.all(X_sub[:, ia] == X_sub[:, ib]):
                    abs_corr[ia, ib] = 1.0
                    abs_corr[ib, ia] = 1.0

    np.fill_diagonal(abs_corr, 1.0)

    #same purge logic as before
    alive = np.ones(n_models, dtype=bool)

    #higher F first, lower c first on ties
    process_order = np.lexsort((cols, -F_sub))

    for i in process_order:
        if not alive[i]:
            continue

        for j in range(n_models):
            if j == i or not alive[j]:
                continue

            dist = 1.0 - abs_corr[i, j]
            if dist <= threshold:
                if F_sub[i] > F_sub[j]:
                    alive[j] = False
                elif F_sub[i] < F_sub[j]:
                    alive[i] = False
                    break
                else:
                    if cols[i] < cols[j]:
                        alive[j] = False
                    else:
                        alive[i] = False
                        break

    survivors = cols[alive]
    survivors.sort()

    #display order for plotting
    display_order = np.lexsort((cols, -F_sub))
    abs_corr_disp = abs_corr[np.ix_(display_order, display_order)]
    cols_disp = cols[display_order]
    F_disp = F_sub[display_order]
    alive_disp = alive[display_order]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(abs_corr_disp, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|corr|")

    ax.set_title(
        f"model similarity for F > 0\n"
        f"distance = 1 - |corr|, threshold = {threshold}"
    )
    ax.set_xlabel("model c index")
    ax.set_ylabel("model c index")

    def fmt_f(val, decimals=f_decimals):
        s = f"{val:.{decimals}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s

    tick_labels = [f"{c}\nF={fmt_f(f)}" for c, f in zip(cols_disp, F_disp)]
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    #mark survivor status on diagonal
    for k in range(n_models):
        if alive_disp[k]:
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="lime", linewidth=2.2))
        else:
            ax.plot(k, k, marker="x", markersize=10, markeredgewidth=2, color="red")

    #outline pairs that are too close
    too_close = (1.0 - abs_corr_disp) <= threshold
    for i in range(n_models):
        for j in range(n_models):
            if i >= j:
                continue
            if too_close[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="white", linewidth=1.0))
                ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1,
                                       fill=False, edgecolor="white", linewidth=1.0))

    fig.text(0.01, 0.01,
             "green box on diagonal = survived\n"
             "red x on diagonal = purged\n"
             "white outlined off-diagonal cells = too close by threshold",
             fontsize=9, va="bottom")

    plt.tight_layout()
    plt.show()

    return survivors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


def viz_failure_over_time(
    participation,
    emissions,
    evaluation_mask,
    emission_bins=40,
    min_count=1,
    ax=None,
):
    """
    Visualize the probability that the next consecutive valid entry is positive.

    y-axis  : starting emission bin
    x-axis  : future consecutive valid entry number
    value   : probability that that future entry's emission is positive

    Coloring:
        0.0 -> red
        0.5 -> white
        1.0 -> green

    Alpha:
        alpha_ij is based on log(total_count_ij), normalized across shown cells,
        then rescaled so min alpha = 0.1 and max alpha = 1.0
    """

    participation = np.asarray(participation, dtype=bool).ravel()
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool).ravel()
    emissions = np.asarray(emissions, dtype=float).ravel()

    if not (len(participation) == len(emissions) == len(evaluation_mask)):
        raise ValueError("participation, emissions, and evaluation_mask must have the same length")

    valid = participation & evaluation_mask & np.isfinite(emissions)

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise ValueError("No valid indices found where participation & evaluation_mask are True")

    split_points = np.where(np.diff(valid_idx) != 1)[0] + 1
    runs = np.split(valid_idx, split_points)

    usable_runs = [r for r in runs if len(r) >= 2]
    if len(usable_runs) == 0:
        raise ValueError("No valid consecutive runs of length >= 2 were found")

    max_future_len = max(len(r) - 1 for r in usable_runs)
    valid_emissions = emissions[valid]

    # build y bins
    if np.isscalar(emission_bins):
        n_bins = int(emission_bins)
        if n_bins <= 0:
            raise ValueError("emission_bins must be positive")

        vmin = np.nanmin(valid_emissions)
        vmax = np.nanmax(valid_emissions)

        if vmin == vmax:
            pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
            vmin -= pad
            vmax += pad

        bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    else:
        bin_edges = np.asarray(emission_bins, dtype=float)
        if bin_edges.ndim != 1 or len(bin_edges) < 2:
            raise ValueError("Explicit emission_bins must be a 1D array of bin edges")
        if not np.all(np.diff(bin_edges) > 0):
            raise ValueError("Emission bin edges must be strictly increasing")
        n_bins = len(bin_edges) - 1

    positive_count = np.zeros((n_bins, max_future_len), dtype=np.int64)
    total_count = np.zeros((n_bins, max_future_len), dtype=np.int64)

    def get_bin_index(x, edges):
        if x == edges[-1]:
            return len(edges) - 2
        return np.searchsorted(edges, x, side="right") - 1

    # fill counts
    for run in usable_runs:
        e_run = emissions[run]
        start_val = e_run[0]
        future_vals = e_run[1:]

        b = get_bin_index(start_val, bin_edges)
        if not (0 <= b < n_bins):
            continue

        L = len(future_vals)
        total_count[b, :L] += 1
        positive_count[b, :L] += (future_vals > 0).astype(np.int64)

    # probability matrix
    probability = np.full((n_bins, max_future_len), np.nan, dtype=float)
    np.divide(
        positive_count,
        total_count,
        out=probability,
        where=total_count > 0
    )

    shown = (total_count >= min_count) & np.isfinite(probability)

    # alpha from log(count) / sum(log(count)), then normalize to [0.1, 1.0]
    alpha_map = np.zeros_like(probability, dtype=float)

    if np.any(shown):
        shown_counts = total_count[shown].astype(float)
        log_vals = np.log(shown_counts + 1)

        denom = log_vals.sum()

        if denom > 0:
            raw_alpha = log_vals / denom
        else:
            # happens if all shown counts are 1 -> log(1)=0 everywhere
            raw_alpha = np.ones_like(log_vals, dtype=float)

        raw_min = raw_alpha.min()
        raw_max = raw_alpha.max()

        if raw_max > raw_min:
            norm_alpha = (raw_alpha - raw_min) / (raw_max - raw_min)
            norm_alpha = 0.1 + 0.9 * norm_alpha
        else:
            # all same support
            norm_alpha = np.ones_like(raw_alpha, dtype=float)

        alpha_map[shown] = norm_alpha

    # custom red-white-green colormap
    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        [
            (0.0, "red"),
            (0.5, "white"),
            (1.0, "green"),
        ]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    x_edges = np.arange(1, max_future_len + 2) - 0.5

    plot_data = np.ma.masked_where(~shown, probability)

    mesh = ax.pcolormesh(
        x_edges,
        bin_edges,
        plot_data,
        shading="flat",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
        linewidth=0
    )

    mesh.set_alpha(alpha_map)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("probability next valid entry is positive")

    ax.set_xlabel("next consecutive valid entry number")
    ax.set_ylabel("starting emission")
    ax.set_title("probability future consecutive valid entry is positive")

    out = {
        "probability": probability,
        "positive_count": positive_count,
        "total_count": total_count,
        "alpha_map": alpha_map,
        "emission_bin_edges": bin_edges,
        "next_steps": np.arange(1, max_future_len + 1),
    }

    return fig, ax, out

def visualize_participation_surfaces(
	*,
	m: float = 20.0,
	n: float = 100.0,
	num: int = 160,
	e: float = np.e,
	mode: str = "surface",          # "surface" or "imshow"
	which: str = "all",             # "b", "d", "bd", or "all"
	ceil: int = 0,
	plot_mn: bool = False,          # NEW: plot m and n reference lines
):
	"""
	Visualize b(p,q), d(p), and combined (b+d) over a (p,q) grid, masking out invalid q>p.

	If plot_mn=True:
	  - plot p = n (line perpendicular to p axis; i.e., vertical line in p-q plane)
	  - plot q = m (line perpendicular to q axis; i.e., horizontal line in p-q plane)
	"""
	import matplotlib.pyplot as plt

	if not (n >= m):
		raise ValueError("Need n >= m.")

	# Keep p < 2n to avoid depth-log domain issues on the p>n branch
	p_vals = np.linspace(1.0, 1.9 * n, num)
	q_vals = np.linspace(0.0, 1.9 * n, num)
	P, Q = np.meshgrid(p_vals, q_vals)

	valid = (Q <= P)

	# Evaluate only valid points to respect p>=q
	p_flat = P[valid].ravel()
	q_flat = Q[valid].ravel()
	B_flat, D_flat = _E.evaluate_participation(m, n, p_flat, q_flat)

	# Put results back into grids with NaNs elsewhere
	B = np.full_like(P, np.nan, dtype=float)
	D = np.full_like(P, np.nan, dtype=float)
	BD = np.full_like(P, np.nan, dtype=float)

	B[valid] = B_flat
	D[valid] = D_flat
	BD[valid] = B_flat + D_flat

	if ceil > 0:
		B = np.clip(B, None, ceil)
		D = np.clip(D, None, ceil)
		BD = np.clip(BD, None, ceil)

	want_b = which in ("b", "all")
	want_d = which in ("d", "all")
	want_bd = which in ("bd", "all")

	if mode not in ("surface", "imshow"):
		raise ValueError("mode must be 'surface' or 'imshow'.")

	if mode == "surface":
		from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

		def _surface(Z, title, zlab):
			fig = plt.figure()
			ax = fig.add_subplot(111, projection="3d")
			ax.plot_surface(P, Q, Z, linewidth=0, antialiased=True)
			ax.set_title(title)
			ax.set_xlabel("p")
			ax.set_ylabel("q")
			ax.set_zlabel(zlab)

			if plot_mn:
				# p = n plane slice: line at p=n, varying q, at z=0 (reference)
				ax.plot([n] * len(q_vals), q_vals, np.zeros_like(q_vals), linestyle="--")
				# q = m plane slice: line at q=m, varying p, at z=0 (reference)
				ax.plot(p_vals, [m] * len(p_vals), np.zeros_like(p_vals), linestyle="--")

		if want_b:
			_surface(B, f"b(p,q) (m={m:g}, n={n:g})", "b")
		if want_d:
			_surface(D, f"d(p) (m={m:g}, n={n:g})", "d")
		if want_bd:
			_surface(BD, f"b(p,q)+d(p) (m={m:g}, n={n:g})", "b+d")

		plt.show()

	else:  # mode == "imshow"
		extent = [p_vals.min(), p_vals.max(), q_vals.min(), q_vals.max()]

		def _imshow(Z, title):
			fig, ax = plt.subplots()
			im = ax.imshow(
				Z, cmap="Reds",
				origin="lower",
				aspect="auto",
				extent=extent,
				interpolation="nearest",
			)
			ax.set_title(title)
			ax.set_xlabel("p")
			ax.set_ylabel("q")

			if plot_mn:
				# p = n (vertical line)
				ax.axvline(n, linestyle="-", c='black',alpha=0.25)
				# q = m (horizontal line)
				ax.axhline(m, linestyle="-", c='black',alpha=0.25)

			plt.colorbar(im, ax=ax)

		if want_b:
			_imshow(B, f"b(p,q) heatmap (m={m:g}, n={n:g})")
		if want_d:
			_imshow(D, f"d(p) heatmap (m={m:g}, n={n:g})")
		if want_bd:
			_imshow(BD, f"b(p,q)+d(p) heatmap (m={m:g}, n={n:g})")

		plt.show()