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
          
import copy
import numpy as np
import matplotlib.pyplot as plt

import initialization as _I
import evaluation as _E

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def visualize_opg_mcpt_distribution(
    population,
    null_returns,
    details,
    good_idx,
    gene_indices=None,
    use_gidx=True,
    bins=60,
    density=False,
    top_n=None,
    sort_by="score",
    include_fill=False,
    line_alpha=0.2,
    line_width=1.0,
    title=None,
):
    """
    Plot the shared OPG null distribution as a histogram and overlay gene observed
    scores as vertical lines.

    Despite the parameter name null_returns for backwards compatibility, the values
    supplied are treated generically as the chosen null score distribution.
    """
    

    null_scores = np.asarray(null_returns, dtype=float).reshape(-1)

    

    observed_scores_full = np.asarray(
        details.get("observed_scores_full", details.get("R_full")),
        dtype=float,
    ).reshape(-1)

    gidx = np.asarray(details["gidx"], dtype=int)
    score_label = details.get("score_label", "Score")

    gidx = np.asarray(details["gidx"], dtype=int)  # evaluated genes only

    if gene_indices is not None:
        gene_idx_used = np.asarray(gene_indices, dtype=int).reshape(-1)

    elif good_idx is not None:
        good_idx = np.asarray(good_idx, dtype=int).reshape(-1)
        gene_idx_used = good_idx[np.isin(good_idx, gidx)]

    elif use_gidx:
        gene_idx_used = gidx.copy()

    else:
        gene_idx_used = np.flatnonzero(np.isfinite(observed_scores_full))

    gene_scores_used = observed_scores_full[gene_idx_used]

    if not include_fill:
        keep = np.isfinite(gene_scores_used)
        gene_idx_used = gene_idx_used[keep]
        gene_scores_used = gene_scores_used[keep]

    if top_n is not None and gene_scores_used.size > top_n:
        if sort_by == "score":
            order = np.argsort(gene_scores_used)[::-1]
        elif sort_by == "abs_score":
            order = np.argsort(np.abs(gene_scores_used))[::-1]
        else:
            raise ValueError('sort_by must be "score" or "abs_score"')

        order = order[:top_n]
        gene_idx_used = gene_idx_used[order]
        gene_scores_used = gene_scores_used[order]

    if title is None:
        title = f"OPG-MCPT {score_label} Distribution with Gene Scores"

    plt.figure(figsize=(5,3))
    plt.hist(null_scores, bins=bins, density=density)

    ymin, ymax = plt.ylim()

    for x in gene_scores_used:
        plt.vlines(x, ymin=0.0, ymax=ymax, alpha=line_alpha, linewidth=line_width)

    if null_scores.size > 0:
        plt.axvline(np.mean(null_scores), linestyle="--", linewidth=1.5, label="Null Mean")

    plt.title(title)
    plt.xlabel(score_label)
    plt.ylabel("Density" if density else "Count")
    plt.legend()
    plt.show()

    return gene_idx_used, gene_scores_used


def visualize_opg_mcpt_distribution_old(
    population,
    null_returns,
    details,
    gene_indices=None,
    use_gidx=True,
    bins=60,
    density=False,
    top_n=None,
    sort_by="return",
    include_fill=False,
    line_alpha=0.2,
    line_width=1.0,
    title="OPG-MCPT Null Distribution with Gene Returns",
):
    """
    Plot the shared OPG null distribution as a histogram and overlay gene observed
    returns as vertical lines.

    Parameters
    ----------
    population : _I.Population
        Population object. Used mainly for index context.

    null_returns : np.ndarray
        Shared null distribution returned by evaluate_opg_mcpt_fast.

    details : dict
        Details dict returned by evaluate_opg_mcpt_fast(return_details=True).
        Must contain:
            "R_full"
            "gidx"

    gene_indices : array_like | None, default=None
        Explicit gene indices to overlay. If None, behavior is controlled by
        use_gidx.

    use_gidx : bool, default=True
        If gene_indices is None and this is True, overlay only population._G_idx.
        If False, overlay all non-NaN locations in details["R_full"].

    bins : int, default=60
        Histogram bin count.

    density : bool, default=False
        Passed to plt.hist(...).

    top_n : int | None, default=None
        If provided, only plot the top_n genes after sorting.

    sort_by : {"return", "abs_return"}, default="return"
        How to rank genes when top_n is used.

    include_fill : bool, default=False
        If False, ignore non-finite return values.

    line_alpha : float, default=0.2
        Alpha for gene vertical lines.

    line_width : float, default=1.0
        Width for gene vertical lines.

    title : str, default="OPG-MCPT Null Distribution with Gene Returns"
        Plot title.

    Returns
    -------
    gene_idx_used : np.ndarray
        Indices of the genes actually plotted.

    gene_returns_used : np.ndarray
        Observed returns corresponding to gene_idx_used.
    """
    null_returns = np.asarray(null_returns, dtype=float).reshape(-1)
    R_full = np.asarray(details["R_full"], dtype=float).reshape(-1)
    gidx = np.asarray(details["gidx"], dtype=int)

    if gene_indices is None:
        if use_gidx:
            gene_idx_used = gidx.copy()
        else:
            gene_idx_used = np.flatnonzero(np.isfinite(R_full))
    else:
        gene_idx_used = np.asarray(gene_indices, dtype=int).reshape(-1)

    gene_returns_used = R_full[gene_idx_used]

    if not include_fill:
        keep = np.isfinite(gene_returns_used)
        gene_idx_used = gene_idx_used[keep]
        gene_returns_used = gene_returns_used[keep]

    if top_n is not None and gene_returns_used.size > top_n:
        if sort_by == "return":
            order = np.argsort(gene_returns_used)[::-1]
        elif sort_by == "abs_return":
            order = np.argsort(np.abs(gene_returns_used))[::-1]
        else:
            raise ValueError('sort_by must be "return" or "abs_return"')

        order = order[:top_n]
        gene_idx_used = gene_idx_used[order]
        gene_returns_used = gene_returns_used[order]

    plt.figure(figsize=(10, 6))
    plt.hist(null_returns, bins=bins, density=density)

    ymin, ymax = plt.ylim()

    for x in gene_returns_used:
        plt.vlines(x, ymin=0.0, ymax=ymax, alpha=line_alpha, linewidth=line_width)

    if null_returns.size > 0:
        plt.axvline(np.mean(null_returns), linestyle="--", linewidth=1.5, label="Null Mean")

    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Density" if density else "Count")
    plt.legend()
    plt.show()

    return gene_idx_used, gene_returns_used

def demo_chunk_scope(
    data_file: str = '../data/spy5m.csv',
    pop_size: int = 300,
    chunk_size: float = 0.1,
    seed: int | None = 0,
    solver_kwargs: dict | None = None,
    verbose: int = 0,
    eps: float = 1e-12,
):
    """
    Simple visual demonstration that:
    1) chunk_num=None instantiates/evaluates the whole row range
    2) with wf_windows=2, chunk_num=0 and chunk_num=1 instantiate/evaluate only their own windows

    What is plotted
    ----------------
    A) no-chunk run:
       row_activity_full[row] = how many legal gene columns are nonzero at that row
       this should show activity across the full usable row range

    B) two-window runs:
       row_activity_chunk0[row] = activity after evaluating only chunk 0
       row_activity_chunk1[row] = activity after evaluating only chunk 1
       these should light up only their own chunk regions

    C) gene-wise outputs:
       F for full, chunk0, chunk1
       p for full, chunk0, chunk1
       q for full, chunk0, chunk1
    """
    if seed is not None:
        np.random.seed(seed)

    if solver_kwargs is None:
        solver_kwargs = {}

    def _legal_gene_idx(pop):
        gidx = np.asarray(getattr(pop, "_G_idx", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if gidx.size:
            return gidx
        all_idx = np.arange(pop._X_inst.shape[1], dtype=np.int64)
        keep = np.union1d(np.asarray(pop._T_idx, dtype=np.int64), np.asarray(pop._E_idx, dtype=np.int64))
        return np.setdiff1d(all_idx, keep, assume_unique=False)

    def _row_activity(pop, eps=1e-12):
        gidx = _legal_gene_idx(pop)
        Xg = pop._X_inst[:, gidx]
        return np.count_nonzero(np.abs(Xg) > eps, axis=1)

    # ------------------------------------------------------------
    # 1) no chunk_num path
    # ------------------------------------------------------------
    X_full, grammar_full = _I.initialize(
        structure='Intraday',
        incl_time=True,
        data_file=data_file,
        pop_size=pop_size,
        chunk_size=chunk_size,
        wf_windows=1,
        verbose=verbose,
    )

    solver_full = _E.Solver(X_full, **solver_kwargs)
    eval_full, inst_full = _E.evaluate(
        population=X_full,
        solver=solver_full,
        chunk_num=None,
    )

    row_activity_full = _row_activity(X_full, eps=eps)
    x_full = np.arange(X_full._X_inst.shape[0], dtype=np.int64)

    # ------------------------------------------------------------
    # 2) two-window path, each chunk on its own clean copy
    # ------------------------------------------------------------
    X_two_base, grammar_two = _I.initialize(
        structure='Intraday',
        incl_time=True,
        data_file=data_file,
        pop_size=pop_size,
        chunk_size=chunk_size,
        wf_windows=2,
        verbose=verbose,
    )

    # chunk 0 only
    X_c0 = copy.deepcopy(X_two_base)
    solver_c0 = _E.Solver(X_c0, **solver_kwargs)
    eval_c0, inst_c0 = _E.evaluate(
        population=X_c0,
        solver=solver_c0,
        chunk_num=0,
    )
    row_activity_c0 = _row_activity(X_c0, eps=eps)

    # chunk 1 only
    X_c1 = copy.deepcopy(X_two_base)
    solver_c1 = _E.Solver(X_c1, **solver_kwargs)
    eval_c1, inst_c1 = _E.evaluate(
        population=X_c1,
        solver=solver_c1,
        chunk_num=1,
    )
    row_activity_c1 = _row_activity(X_c1, eps=eps)

    x_two = np.arange(X_two_base._X_inst.shape[0], dtype=np.int64)
    bounds0 = tuple(map(int, eval_c0["chunk_row_bounds"]))
    bounds1 = tuple(map(int, eval_c1["chunk_row_bounds"]))

    # ------------------------------------------------------------
    # prints
    # ------------------------------------------------------------
    print("no chunk_num run")
    print("  chunk bounds used:", eval_full["chunk_row_bounds"])
    print("  F mean/std:", float(np.nanmean(eval_full["F"])), float(np.nanstd(eval_full["F"])))
    print("  p mean/std:", float(np.nanmean(eval_full["p"])), float(np.nanstd(eval_full["p"])))
    print("  q mean/std:", float(np.nanmean(eval_full["q"])), float(np.nanstd(eval_full["q"])))
    print()

    print("two-window run")
    print("  chunk 0 bounds:", bounds0)
    print("  chunk 1 bounds:", bounds1)
    print("  chunk 0 F mean/std:", float(np.nanmean(eval_c0["F"])), float(np.nanstd(eval_c0["F"])))
    print("  chunk 1 F mean/std:", float(np.nanmean(eval_c1["F"])), float(np.nanstd(eval_c1["F"])))
    print("  chunk 0 p mean/std:", float(np.nanmean(eval_c0["p"])), float(np.nanstd(eval_c0["p"])))
    print("  chunk 1 p mean/std:", float(np.nanmean(eval_c1["p"])), float(np.nanstd(eval_c1["p"])))
    print("  chunk 0 q mean/std:", float(np.nanmean(eval_c0["q"])), float(np.nanstd(eval_c0["q"])))
    print("  chunk 1 q mean/std:", float(np.nanmean(eval_c1["q"])), float(np.nanstd(eval_c1["q"])))

    # ------------------------------------------------------------
    # figure 1: no chunk_num
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    ax1[0].plot(x_full, row_activity_full, lw=1.0)
    ax1[0].set_ylabel('active gene count')
    ax1[0].set_title('no chunk_num | row activity after evaluate(..., chunk_num=None)')

    ax1[1].plot(np.asarray(eval_full["F"], dtype=np.float32), lw=1.0, label='F')
    ax1[1].set_ylabel('fitness')
    ax1[1].set_xlabel('gene index')
    ax1[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # figure 2: chunk-local instantiation only
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    ax2[0].plot(x_two, row_activity_c0, lw=1.0, label='after chunk 0 only')
    ax2[0].axvspan(bounds0[0], bounds0[1], alpha=0.15)
    ax2[0].set_ylabel('active gene count')
    ax2[0].set_title('two windows | evaluating chunk 0 only should instantiate only chunk 0 rows')
    ax2[0].legend(loc='upper right')

    ax2[1].plot(x_two, row_activity_c1, lw=1.0, label='after chunk 1 only')
    ax2[1].axvspan(bounds1[0], bounds1[1], alpha=0.15)
    ax2[1].set_ylabel('active gene count')
    ax2[1].set_xlabel('row index')
    ax2[1].set_title('two windows | evaluating chunk 1 only should instantiate only chunk 1 rows')
    ax2[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # figure 3: chunk-local evaluation outputs
    # ------------------------------------------------------------
    fig3, ax3 = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    ax3[0].plot(np.asarray(eval_full["F"], dtype=np.float32), lw=0.8, alpha=0.7, label='full F')
    ax3[0].plot(np.asarray(eval_c0["F"], dtype=np.float32), lw=0.9, label='chunk 0 F')
    ax3[0].plot(np.asarray(eval_c1["F"], dtype=np.float32), lw=0.9, label='chunk 1 F')
    ax3[0].set_ylabel('F')
    ax3[0].set_title('gene-wise fitness changes by evaluation window')
    ax3[0].legend(loc='upper right')

    ax3[1].plot(np.asarray(eval_full["p"], dtype=np.float32), lw=0.8, alpha=0.7, label='full p')
    ax3[1].plot(np.asarray(eval_c0["p"], dtype=np.float32), lw=0.9, label='chunk 0 p')
    ax3[1].plot(np.asarray(eval_c1["p"], dtype=np.float32), lw=0.9, label='chunk 1 p')
    ax3[1].set_ylabel('p')
    ax3[1].set_title('gene-wise p changes by window')
    ax3[1].legend(loc='upper right')

    ax3[2].plot(np.asarray(eval_full["q"], dtype=np.float32), lw=0.8, alpha=0.7, label='full q')
    ax3[2].plot(np.asarray(eval_c0["q"], dtype=np.float32), lw=0.9, label='chunk 0 q')
    ax3[2].plot(np.asarray(eval_c1["q"], dtype=np.float32), lw=0.9, label='chunk 1 q')
    ax3[2].set_ylabel('q')
    ax3[2].set_xlabel('gene index')
    ax3[2].set_title('gene-wise q changes by window')
    ax3[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return {
        "no_chunk": {
            "population": X_full,
            "grammar": grammar_full,
            "evaluation": eval_full,
            "instantiation_stats": inst_full,
            "row_activity": row_activity_full,
        },
        "two_windows": {
            "base_population": X_two_base,
            "grammar": grammar_two,
            "chunk0": {
                "population": X_c0,
                "evaluation": eval_c0,
                "instantiation_stats": inst_c0,
                "row_activity": row_activity_c0,
            },
            "chunk1": {
                "population": X_c1,
                "evaluation": eval_c1,
                "instantiation_stats": inst_c1,
                "row_activity": row_activity_c1,
            },
        },
    }


import numpy as np
import matplotlib.pyplot as plt


#---------------------------------------------------------------------
# generic safe dictionary helpers
#---------------------------------------------------------------------

def _mcts_getdict(G, name):
    d = getattr(G, name, None)
    return {} if d is None else d


def _mcts_dict_sum(d):
    if d is None or len(d) == 0:
        return 0.0
    return float(np.sum(list(d.values())))


#---------------------------------------------------------------------
# key/depth helpers
#---------------------------------------------------------------------

def _mcts_safe_key_depth(G, key):
    """
    return MCTS path depth from a sparse key.

    preferred:
        use G._mcts_key_depth if it exists.

    fallback:
        parse path keys manually.
    """

    if hasattr(G, "_mcts_key_depth"):
        try:
            return int(G._mcts_key_depth(key))
        except Exception:
            pass

    if key is None:
        return 0

    if not isinstance(key, tuple) or len(key) == 0:
        return 0

    if key[0] == "P":
        d = 0
        for v in key[1:]:
            if isinstance(v, (int, np.integer)):
                d += 1
        return d

    if key[0] == "ACTX":
        # alpha context: ("ACTX", parent_key, child_tf)
        # display it at the depth of the would-be child.
        if len(key) >= 3:
            return _mcts_safe_key_depth(G, key[1]) + 1
        return 1

    if key[0] == "TF":
        return 1

    if key[0] == "L":
        return 1

    if key[0] == "T":
        return 0

    return 0


def _mcts_depth_vector_stack(vecs):
    """
    stack variable-length depth probability vectors into a 2D array.
    rows = iterations
    cols = depth
    """

    if len(vecs) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    m = max(v.shape[0] for v in vecs)
    out = np.zeros((len(vecs), m), dtype=np.float64)

    for i, v in enumerate(vecs):
        out[i, :v.shape[0]] = v

    return out


#---------------------------------------------------------------------
# normal x-parent / child-tf MCTS terms
#---------------------------------------------------------------------

def _mcts_node_n_for_explore(G, key):
    if getattr(G, "_count_explore", False) is True:
        return _mcts_getdict(G, "_MCTS_NODE_EXPLORE_COUNT").get(key, 0)
    return _mcts_getdict(G, "_MCTS_NODE_COUNT").get(key, 0)


def _mcts_edge_n_for_explore(G, edge_key):
    if getattr(G, "_count_explore", False) is True:
        return _mcts_getdict(G, "_MCTS_EDGE_EXPLORE_COUNT").get(edge_key, 0)
    return _mcts_getdict(G, "_MCTS_EDGE_COUNT").get(edge_key, 0)


def _mcts_node_explore_coef(G, key):
    """
    node exploration coefficient used by UCT.

    UCT(node) = Q(node) + sqrt(c * log(total + 2) / (N(node) + 1))
    """

    n = _mcts_node_n_for_explore(G, key)

    total = 0
    total += int(getattr(G, "_MCTS_EXPLORE_T", 0) or 0)
    total += int(getattr(G, "_MCTS_EXPLOIT_T", 0) or 0)

    return float(np.sqrt(G._c * np.log(total + 2) / (n + 1)))


def _mcts_edge_explore_coef(G, parent_key, child_tf):
    """
    edge exploration coefficient used by UCB.

    UCB(parent -> child_tf)
        = Q(parent -> child_tf)
          + sqrt(c * log(N(parent) + 2) / (N(edge) + 1))
    """

    edge_key = (parent_key, int(child_tf))

    parent_n = _mcts_node_n_for_explore(G, parent_key)
    edge_n = _mcts_edge_n_for_explore(G, edge_key)

    return float(np.sqrt(G._c * np.log(parent_n + 2) / (edge_n + 1)))

def _mcts_alpha_node_n_for_explore(G, key):
    """
    count used for alpha-parent exploration.

    this is the alpha-parent equivalent of _mcts_node_n_for_explore.
    """

    if getattr(G, "_count_explore", False) is True:
        return G._MCTS_ALPHA_NODE_EXPLORE_COUNT.get(key, 0)

    return G._MCTS_ALPHA_NODE_COUNT.get(key, 0)


def _mcts_alpha_node_explore_coef(G, key):
    """
    alpha-node exploration coefficient.

    this measures uncertainty for nodes being used as alpha parents.

    alpha_node_score = Q_alpha_node + U_alpha_node
    U_alpha_node = sqrt(c * log(alpha_total + 2) / (N_alpha_node + 1))
    """

    n = _mcts_alpha_node_n_for_explore(G, key)

    total = 0
    total += int(getattr(G, "_MCTS_ALPHA_EXPLORE_T", 0) or 0)
    total += int(getattr(G, "_MCTS_ALPHA_EXPLOIT_T", 0) or 0)

    return float(np.sqrt(G._c * np.log(total + 2) / (n + 1)))


def mcts_alpha_node_depth_summary(G):
    """
    summarize alpha-parent node exploit/explore values by depth.

    this is different from normal node depth summary.

    normal node summary:
        how useful is a state as an x-parent/generated state?

    alpha node summary:
        how useful is a state as an alpha sensor/alpha parent?
    """

    rows = {}

    def _ensure(d):
        if d not in rows:
            rows[d] = {
                "depth": d,
                "alpha_node_count": 0,
                "alpha_node_q_sum": 0.0,
                "alpha_node_u_sum": 0.0,
                "alpha_node_score_sum": 0.0,
            }

    alpha_node_keys = set()
    alpha_node_keys |= set(getattr(G, "_MCTS_ALPHA_NODE_MU", {}).keys())
    alpha_node_keys |= set(getattr(G, "_MCTS_ALPHA_NODE_COUNT", {}).keys())
    alpha_node_keys |= set(getattr(G, "_MCTS_ALPHA_NODE_EXPLORE_COUNT", {}).keys())

    for key in alpha_node_keys:
        d = _mcts_safe_key_depth(G, key)
        _ensure(d)

        q = float(G._MCTS_ALPHA_NODE_MU.get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
        u = _mcts_alpha_node_explore_coef(G, key)

        rows[d]["alpha_node_count"] += 1
        rows[d]["alpha_node_q_sum"] += q
        rows[d]["alpha_node_u_sum"] += u
        rows[d]["alpha_node_score_sum"] += q + u

    out = []

    for d in sorted(rows):
        r = rows[d]
        n = max(r["alpha_node_count"], 1)

        out.append({
            "depth": d,
            "alpha_node_count": r["alpha_node_count"],
            "alpha_node_q_mean": r["alpha_node_q_sum"] / n,
            "alpha_node_u_mean": r["alpha_node_u_sum"] / n,
            "alpha_node_score_mean": r["alpha_node_score_sum"] / n,
        })

    return out

#---------------------------------------------------------------------
# alpha MCTS terms
#---------------------------------------------------------------------

def _mcts_alpha_decision_n_for_explore(G, decision_key):
    if getattr(G, "_count_explore", False) is True:
        return _mcts_getdict(G, "_MCTS_ALPHA_DECISION_EXPLORE_COUNT").get(decision_key, 0)
    return _mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT").get(decision_key, 0)


def _mcts_alpha_node_n_for_explore(G, key):
    if getattr(G, "_count_explore", False) is True:
        return _mcts_getdict(G, "_MCTS_ALPHA_NODE_EXPLORE_COUNT").get(key, 0)
    return _mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT").get(key, 0)


def _mcts_alpha_edge_n_for_explore(G, edge_key):
    if getattr(G, "_count_explore", False) is True:
        return _mcts_getdict(G, "_MCTS_ALPHA_EDGE_EXPLORE_COUNT").get(edge_key, 0)
    return _mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT").get(edge_key, 0)


def _mcts_alpha_decision_explore_coef(G, ctx_key, action):
    decision_key = (ctx_key, int(action))
    n = _mcts_alpha_decision_n_for_explore(G, decision_key)

    total = 0
    total += int(getattr(G, "_MCTS_ALPHA_EXPLORE_T", 0) or 0)
    total += int(getattr(G, "_MCTS_ALPHA_EXPLOIT_T", 0) or 0)

    return float(np.sqrt(G._c * np.log(total + 2) / (n + 1)))


def _mcts_alpha_node_explore_coef(G, key):
    n = _mcts_alpha_node_n_for_explore(G, key)

    total = 0
    total += int(getattr(G, "_MCTS_ALPHA_EXPLORE_T", 0) or 0)
    total += int(getattr(G, "_MCTS_ALPHA_EXPLOIT_T", 0) or 0)

    return float(np.sqrt(G._c * np.log(total + 2) / (n + 1)))


def _mcts_alpha_edge_explore_coef(G, ctx_key, alpha_parent_key):
    edge_key = (ctx_key, alpha_parent_key)

    # alpha edge exploration is conditioned on the alpha decision context.
    # use decision-context evidence as the parent count proxy.
    ctx_n0 = _mcts_alpha_decision_n_for_explore(G, (ctx_key, 0))
    ctx_n1 = _mcts_alpha_decision_n_for_explore(G, (ctx_key, 1))
    ctx_n = ctx_n0 + ctx_n1

    edge_n = _mcts_alpha_edge_n_for_explore(G, edge_key)

    return float(np.sqrt(G._c * np.log(ctx_n + 2) / (edge_n + 1)))


#---------------------------------------------------------------------
# softmax probability helper
#---------------------------------------------------------------------

def _mcts_softmax_probs(scores, base=np.e, temp=1.0, valid_mask=None):
    """
    return softmax probabilities without sampling.
    """

    scores = np.asarray(scores, dtype=np.float64)

    if valid_mask is None:
        valid_mask = np.ones(scores.shape[0], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_mask &= np.isfinite(scores)

    if scores.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    if not np.any(valid_mask):
        return np.full(scores.shape[0], 1.0 / scores.shape[0], dtype=np.float64)

    temp = float(np.clip(temp, 1e-6, 1.0))
    s = scores.copy() / temp

    if base == 1:
        w = valid_mask.astype(np.float64)
    else:
        scaled = np.full_like(s, -np.inf, dtype=np.float64)
        scaled[valid_mask] = np.log(base) * s[valid_mask]
        scaled[valid_mask] -= np.max(scaled[valid_mask])

        w = np.zeros_like(s, dtype=np.float64)
        w[valid_mask] = np.exp(scaled[valid_mask])

    if w.sum() <= 0 or not np.isfinite(w.sum()):
        w = valid_mask.astype(np.float64)

    return w / w.sum()

def mcts_alpha_sensor_probability_summary(G):
    """
    compute mean p(alpha sensor) over all known alpha decision contexts.

    this is not the raw alpha_sensor_freq parameter.

    alpha_sensor_freq is only a prior.
    the returned probability is based on:
        alpha decision UCB scores
        learned alpha decision means
        exploration terms
        prior log-bias
        softmax temperature
    """

    decision_keys = set()
    decision_keys |= set(getattr(G, "_MCTS_ALPHA_DECISION_MU", {}).keys())
    decision_keys |= set(getattr(G, "_MCTS_ALPHA_DECISION_COUNT", {}).keys())
    decision_keys |= set(getattr(G, "_MCTS_ALPHA_DECISION_EXPLORE_COUNT", {}).keys())

    #extract ctx keys from decision keys:
    #decision key is: (ctx_key, action)
    ctx_keys = sorted(set([k[0] for k in decision_keys]), key=lambda x: str(x))

    if len(ctx_keys) == 0:
        return np.nan, {}

    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    p_sensor_by_ctx = {}

    for ctx_key in ctx_keys:

        actions = np.asarray([0, 1], dtype=np.int64)

        scores = np.asarray([
            G._mcts_alpha_decision_ucb(ctx_key, 0),
            G._mcts_alpha_decision_ucb(ctx_key, 1),
        ], dtype=np.float64)

        #same soft prior used by _mcts_select_alpha_is_sensor
        p_sensor_prior = float(np.clip(G._alpha_sensor_freq, 1e-6, 1.0 - 1e-6))
        prior = np.asarray([1.0 - p_sensor_prior, p_sensor_prior], dtype=np.float64)

        scores = scores + G._MCTS_ALPHA_PRIOR_WEIGHT * np.log(prior)

        probs = _mcts_softmax_probs(
            scores=scores,
            base=base,
            temp=temp,
            valid_mask=np.isfinite(scores)
        )

        p_sensor_by_ctx[ctx_key] = float(probs[1])

    vals = np.asarray(list(p_sensor_by_ctx.values()), dtype=np.float64)

    return float(np.mean(vals)), p_sensor_by_ctx


def mcts_alpha_parent_depth_probability(G, X, legal_idx=None):
    """
    estimate p(alpha parent depth | alpha is sensor).

    this mirrors p(new node depth), but for alpha parent selection.

    IMPORTANT:
    this is conditional on alpha already being selected as a sensor.
    it answers:
        if alpha is a sensor right now,
        what depth of node is likely to be selected as alpha parent?
    """

    if legal_idx is None:
        legal_idx = np.asarray(X._L_idx, dtype=np.int64)
    else:
        legal_idx = np.asarray(legal_idx, dtype=np.int64)

    if legal_idx.size == 0:
        return np.zeros(1, dtype=np.float64)

    scores = np.empty(legal_idx.shape[0], dtype=np.float64)
    depths = np.empty(legal_idx.shape[0], dtype=np.int64)
    valid_mask = np.ones(legal_idx.shape[0], dtype=bool)

    max_alpha_parent_depth = int(getattr(G, "_MCTS_MAX_DEPTH", 1)) - 1

    for i in range(legal_idx.shape[0]):
        idx = int(legal_idx[i])

        if hasattr(G, "_mcts_row_depth"):
            d = int(G._mcts_row_depth(X._instructions, idx))
        else:
            key_tmp = G._mcts_state_key(X._instructions, idx)
            d = _mcts_safe_key_depth(G, key_tmp)

        depths[i] = d
        valid_mask[i] = d <= max_alpha_parent_depth

        key = G._mcts_state_key(X._instructions, idx)

        if hasattr(G, "_mcts_alpha_node_uct"):
            scores[i] = G._mcts_alpha_node_uct(key)
        else:
            q = float(G._MCTS_ALPHA_NODE_MU.get(key, G._MCTS_NODE_MU.get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0))))
            u = _mcts_alpha_node_explore_coef(G, key)
            scores[i] = q + u

    if not np.any(valid_mask):
        valid_mask[:] = True

    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    probs = _mcts_softmax_probs(
        scores=scores,
        base=base,
        temp=temp,
        valid_mask=valid_mask & np.isfinite(scores)
    )

    max_d = int(max(depths.max(), getattr(G, "_MCTS_MAX_DEPTH", depths.max()))) + 1

    alpha_parent_depth_prob = np.zeros(max_d + 1, dtype=np.float64)

    for d, p in zip(depths, probs):
        alpha_parent_depth_prob[int(d)] += p

    return alpha_parent_depth_prob

#---------------------------------------------------------------------
# depth summaries for normal MCTS
#---------------------------------------------------------------------

def mcts_depth_summary(G):
    """
    summarize x-parent/node and child-tf/edge exploit/explore values by depth.
    """

    rows = {}

    def _ensure(d):
        if d not in rows:
            rows[d] = {
                "depth": d,

                "node_count": 0,
                "node_q_sum": 0.0,
                "node_u_sum": 0.0,
                "node_score_sum": 0.0,

                "node_expandable_count": 0,
                "node_expandable_q_sum": 0.0,
                "node_expandable_u_sum": 0.0,
                "node_expandable_score_sum": 0.0,

                "edge_count": 0,
                "edge_q_sum": 0.0,
                "edge_u_sum": 0.0,
                "edge_score_sum": 0.0,
            }

    node_keys = set()
    node_keys |= set(_mcts_getdict(G, "_MCTS_NODE_MU").keys())
    node_keys |= set(_mcts_getdict(G, "_MCTS_NODE_COUNT").keys())
    node_keys |= set(_mcts_getdict(G, "_MCTS_NODE_EXPLORE_COUNT").keys())

    for key in node_keys:
        d = _mcts_safe_key_depth(G, key)
        _ensure(d)

        q = float(_mcts_getdict(G, "_MCTS_NODE_MU").get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
        u = _mcts_node_explore_coef(G, key)

        rows[d]["node_count"] += 1
        rows[d]["node_q_sum"] += q
        rows[d]["node_u_sum"] += u
        rows[d]["node_score_sum"] += q + u

        max_depth = getattr(G, "_MCTS_MAX_DEPTH", None)
        if max_depth is None or d < int(max_depth):
            rows[d]["node_expandable_count"] += 1
            rows[d]["node_expandable_q_sum"] += q
            rows[d]["node_expandable_u_sum"] += u
            rows[d]["node_expandable_score_sum"] += q + u

    edge_keys = set()
    edge_keys |= set(_mcts_getdict(G, "_MCTS_EDGE_MU").keys())
    edge_keys |= set(_mcts_getdict(G, "_MCTS_EDGE_COUNT").keys())
    edge_keys |= set(_mcts_getdict(G, "_MCTS_EDGE_EXPLORE_COUNT").keys())

    for edge_key in edge_keys:
        parent_key, child_tf = edge_key
        d = _mcts_safe_key_depth(G, parent_key) + 1
        _ensure(d)

        q = float(_mcts_getdict(G, "_MCTS_EDGE_MU").get(edge_key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
        u = _mcts_edge_explore_coef(G, parent_key, child_tf)

        rows[d]["edge_count"] += 1
        rows[d]["edge_q_sum"] += q
        rows[d]["edge_u_sum"] += u
        rows[d]["edge_score_sum"] += q + u

    out = []

    for d in sorted(rows):
        r = rows[d]
        nc = max(r["node_count"], 1)
        ec = max(r["edge_count"], 1)
        nec = r["node_expandable_count"]

        if nec > 0:
            node_expandable_q_mean = r["node_expandable_q_sum"] / nec
            node_expandable_u_mean = r["node_expandable_u_sum"] / nec
            node_expandable_score_mean = r["node_expandable_score_sum"] / nec
        else:
            node_expandable_q_mean = np.nan
            node_expandable_u_mean = np.nan
            node_expandable_score_mean = np.nan

        out.append({
            "depth": d,

            "node_count": r["node_count"],
            "node_q_mean": r["node_q_sum"] / nc,
            "node_u_mean": r["node_u_sum"] / nc,
            "node_score_mean": r["node_score_sum"] / nc,

            "node_expandable_count": nec,
            "node_expandable_q_mean": node_expandable_q_mean,
            "node_expandable_u_mean": node_expandable_u_mean,
            "node_expandable_score_mean": node_expandable_score_mean,

            "edge_count": r["edge_count"],
            "edge_q_mean": r["edge_q_sum"] / ec,
            "edge_u_mean": r["edge_u_sum"] / ec,
            "edge_score_mean": r["edge_score_sum"] / ec,
        })

    return out


def mcts_depth_search_probability(G, X, legal_idx=None):
    """
    estimate current natural probability of creating a new node at each depth.
    """

    if legal_idx is None:
        legal_idx = np.asarray(X._L_idx, dtype=np.int64)
    else:
        legal_idx = np.asarray(legal_idx, dtype=np.int64)

    if legal_idx.size == 0:
        return np.zeros(1), np.zeros(1)

    valid_mask = np.ones(legal_idx.shape[0], dtype=bool)

    if hasattr(G, "_mcts_valid_depth_parent_mask"):
        valid_mask &= G._mcts_valid_depth_parent_mask(
            instructions=X._instructions,
            legal_idx=legal_idx
        )

    if not np.any(valid_mask):
        valid_mask[:] = True

    scores = np.empty(legal_idx.shape[0], dtype=np.float64)
    depths = np.empty(legal_idx.shape[0], dtype=np.int64)

    for i in range(legal_idx.shape[0]):
        idx = int(legal_idx[i])
        key = G._mcts_state_key(X._instructions, idx)

        if hasattr(G, "_mcts_node_uct"):
            scores[i] = G._mcts_node_uct(key)
        else:
            q = float(_mcts_getdict(G, "_MCTS_NODE_MU").get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
            u = _mcts_node_explore_coef(G, key)
            scores[i] = q + u

        if hasattr(G, "_mcts_row_depth"):
            depths[i] = int(G._mcts_row_depth(X._instructions, idx))
        else:
            depths[i] = _mcts_safe_key_depth(G, key)

    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    probs = _mcts_softmax_probs(
        scores=scores,
        base=base,
        temp=temp,
        valid_mask=valid_mask
    )

    max_d = int(max(depths.max() + 1, getattr(G, "_MCTS_MAX_DEPTH", depths.max() + 1))) + 1

    parent_depth_prob = np.zeros(max_d + 1, dtype=np.float64)
    child_depth_prob = np.zeros(max_d + 1, dtype=np.float64)

    for d, p in zip(depths, probs):
        parent_depth_prob[int(d)] += p

        if int(d) + 1 < child_depth_prob.shape[0]:
            child_depth_prob[int(d) + 1] += p

    return parent_depth_prob, child_depth_prob


#---------------------------------------------------------------------
# alpha summaries/probabilities
#---------------------------------------------------------------------

def _stack_depth_vectors(vecs):
    """
    stack variable-length depth probability vectors into a 2D array.

    rows = iterations
    cols = depth

    this is needed because early iterations may only have depths 0..2,
    while later iterations may have depths 0..5.
    """

    if vecs is None or len(vecs) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    max_len = 0
    for v in vecs:
        if v is None:
            continue
        max_len = max(max_len, np.asarray(v).shape[0])

    if max_len == 0:
        return np.zeros((0, 0), dtype=np.float64)

    out = np.zeros((len(vecs), max_len), dtype=np.float64)

    for i, v in enumerate(vecs):
        if v is None:
            continue

        v = np.asarray(v, dtype=np.float64).ravel()
        out[i, :v.shape[0]] = v

    return out

def _mcts_alpha_decision_scores(G, ctx_key):
    """
    score alpha action 0/1 for a context.

    action 0 = alpha constant
    action 1 = alpha sensor
    """

    scores = np.zeros(2, dtype=np.float64)

    for action in (0, 1):
        decision_key = (ctx_key, action)

        if hasattr(G, "_mcts_alpha_decision_ucb"):
            try:
                scores[action] = G._mcts_alpha_decision_ucb(ctx_key, action)
            except Exception:
                q = float(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU").get(decision_key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
                u = _mcts_alpha_decision_explore_coef(G, ctx_key, action)
                scores[action] = q + u
        else:
            q = float(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU").get(decision_key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
            u = _mcts_alpha_decision_explore_coef(G, ctx_key, action)
            scores[action] = q + u

    #include the same prior bias used in alpha decision selection.
    p_sensor = float(np.clip(getattr(G, "_alpha_sensor_freq", 0.5), 1e-6, 1.0 - 1e-6))
    prior = np.asarray([1.0 - p_sensor, p_sensor], dtype=np.float64)
    prior_weight = float(getattr(G, "_MCTS_ALPHA_PRIOR_WEIGHT", 0.0) or 0.0)
    scores = scores + prior_weight * np.log(prior)

    return scores


def mcts_alpha_decision_probability_summary(G):
    """
    summarize learned probability of alpha being a sensor.
    """

    decision_keys = set()
    decision_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU").keys())
    decision_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT").keys())
    decision_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_EXPLORE_COUNT").keys())

    ctx_keys = sorted(set([k[0] for k in decision_keys]), key=lambda x: str(x))

    if len(ctx_keys) == 0:
        return {
            "n_ctx": 0,
            "mean_p_sensor": np.nan,
            "mean_const_mu": np.nan,
            "mean_sensor_mu": np.nan,
            "const_count": 0,
            "sensor_count": 0,
        }

    p_sensor_vals = []
    const_mu = []
    sensor_mu = []
    const_count = 0
    sensor_count = 0

    temp = getattr(G, "_softmax_temp", 1.0)
    base = getattr(G, "_MCTS_SOFTMAX_BASE", np.e)

    for ctx_key in ctx_keys:
        scores = _mcts_alpha_decision_scores(G, ctx_key)
        probs = _mcts_softmax_probs(scores, base=base, temp=temp)
        p_sensor_vals.append(probs[1])

        const_key = (ctx_key, 0)
        sensor_key = (ctx_key, 1)

        const_mu.append(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU").get(const_key, 0.0))
        sensor_mu.append(_mcts_getdict(G, "_MCTS_ALPHA_DECISION_MU").get(sensor_key, 0.0))

        const_count += _mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT").get(const_key, 0)
        sensor_count += _mcts_getdict(G, "_MCTS_ALPHA_DECISION_COUNT").get(sensor_key, 0)

    return {
        "n_ctx": len(ctx_keys),
        "mean_p_sensor": float(np.mean(p_sensor_vals)),
        "mean_const_mu": float(np.mean(const_mu)),
        "mean_sensor_mu": float(np.mean(sensor_mu)),
        "const_count": int(const_count),
        "sensor_count": int(sensor_count),
    }


def mcts_alpha_parent_depth_probability_s(G):
    """
    approximate probability mass over alpha-parent depths.

    This uses the alpha-node UCT scores across known alpha-parent states.
    """

    keys = set()
    keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_MU").keys())
    keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT").keys())
    keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_EXPLORE_COUNT").keys())

    if len(keys) == 0:
        return np.zeros(1, dtype=np.float64)

    keys = sorted(keys, key=lambda x: str(x))
    scores = np.empty(len(keys), dtype=np.float64)
    depths = np.empty(len(keys), dtype=np.int64)

    for i, key in enumerate(keys):
        if hasattr(G, "_mcts_alpha_node_uct"):
            try:
                scores[i] = G._mcts_alpha_node_uct(key)
            except Exception:
                q = float(_mcts_getdict(G, "_MCTS_ALPHA_NODE_MU").get(key, _mcts_getdict(G, "_MCTS_NODE_MU").get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0))))
                u = _mcts_alpha_node_explore_coef(G, key)
                scores[i] = q + u
        else:
            q = float(_mcts_getdict(G, "_MCTS_ALPHA_NODE_MU").get(key, _mcts_getdict(G, "_MCTS_NODE_MU").get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0))))
            u = _mcts_alpha_node_explore_coef(G, key)
            scores[i] = q + u

        depths[i] = _mcts_safe_key_depth(G, key)

    probs = _mcts_softmax_probs(
        scores=scores,
        base=getattr(G, "_MCTS_SOFTMAX_BASE", np.e),
        temp=getattr(G, "_softmax_temp", 1.0),
    )

    max_d = int(max(depths.max(), getattr(G, "_MCTS_MAX_DEPTH", depths.max()))) + 1
    out = np.zeros(max_d + 1, dtype=np.float64)

    for d, p in zip(depths, probs):
        if int(d) < out.shape[0]:
            out[int(d)] += p

    return out


def mcts_alpha_depth_summary(G):
    """
    summarize alpha-parent exploit/explore values by alpha-parent depth.
    """

    rows = {}

    def _ensure(d):
        if d not in rows:
            rows[d] = {
                "depth": d,
                "alpha_node_count": 0,
                "alpha_node_q_sum": 0.0,
                "alpha_node_u_sum": 0.0,
                "alpha_edge_count": 0,
                "alpha_edge_q_sum": 0.0,
                "alpha_edge_u_sum": 0.0,
            }

    alpha_node_keys = set()
    alpha_node_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_MU").keys())
    alpha_node_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_COUNT").keys())
    alpha_node_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_NODE_EXPLORE_COUNT").keys())

    for key in alpha_node_keys:
        d = _mcts_safe_key_depth(G, key)
        _ensure(d)

        q = float(_mcts_getdict(G, "_MCTS_ALPHA_NODE_MU").get(key, _mcts_getdict(G, "_MCTS_NODE_MU").get(key, getattr(G, "_MCTS_BASE_PRIOR", 0.0))))
        u = _mcts_alpha_node_explore_coef(G, key)

        rows[d]["alpha_node_count"] += 1
        rows[d]["alpha_node_q_sum"] += q
        rows[d]["alpha_node_u_sum"] += u

    alpha_edge_keys = set()
    alpha_edge_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU").keys())
    alpha_edge_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_EDGE_COUNT").keys())
    alpha_edge_keys |= set(_mcts_getdict(G, "_MCTS_ALPHA_EDGE_EXPLORE_COUNT").keys())

    for edge_key in alpha_edge_keys:
        ctx_key, alpha_parent_key = edge_key
        d = _mcts_safe_key_depth(G, alpha_parent_key)
        _ensure(d)

        q = float(_mcts_getdict(G, "_MCTS_ALPHA_EDGE_MU").get(edge_key, getattr(G, "_MCTS_BASE_PRIOR", 0.0)))
        u = _mcts_alpha_edge_explore_coef(G, ctx_key, alpha_parent_key)

        rows[d]["alpha_edge_count"] += 1
        rows[d]["alpha_edge_q_sum"] += q
        rows[d]["alpha_edge_u_sum"] += u

    out = []

    for d in sorted(rows):
        r = rows[d]
        anc = max(r["alpha_node_count"], 1)
        aec = max(r["alpha_edge_count"], 1)

        out.append({
            "depth": d,
            "alpha_node_count": r["alpha_node_count"],
            "alpha_node_q_mean": r["alpha_node_q_sum"] / anc,
            "alpha_node_u_mean": r["alpha_node_u_sum"] / anc,
            "alpha_edge_count": r["alpha_edge_count"],
            "alpha_edge_q_mean": r["alpha_edge_q_sum"] / aec,
            "alpha_edge_u_mean": r["alpha_edge_u_sum"] / aec,
        })

    return out


#---------------------------------------------------------------------
# print helpers
#---------------------------------------------------------------------

def print_mcts_depth_summary(G, X=None, max_depth=None):
    rows = mcts_depth_summary(G)

    child_p = None
    if X is not None:
        _, child_p = mcts_depth_search_probability(G, X)

    if max_depth is None:
        max_depth = getattr(G, "_MCTS_MAX_DEPTH", None)

    print("\nMCTS X/TF DEPTH SUMMARY")
    print("d | node_n | exp_n | node_q | node_u_raw | node_u_active | edge_n | edge_q | edge_u | p_new")
    print("--|--------|-------|--------|------------|---------------|--------|--------|--------|------")

    for r in rows:
        d = int(r["depth"])

        if max_depth is not None and d > max_depth:
            continue

        pnew = 0.0
        if child_p is not None and d < child_p.shape[0]:
            pnew = child_p[d]

        node_u_active = r.get("node_expandable_u_mean", np.nan)

        print(
            f"{d:1d} | "
            f"{r['node_count']:6d} | "
            f"{r.get('node_expandable_count', 0):5d} | "
            f"{r['node_q_mean']:6.3f} | "
            f"{r['node_u_mean']:10.3f} | "
            f"{node_u_active:13.3f} | "
            f"{r['edge_count']:6d} | "
            f"{r['edge_q_mean']:6.3f} | "
            f"{r['edge_u_mean']:6.3f} | "
            f"{pnew:5.3f}"
        )


def print_mcts_alpha_summary(G):
    s = mcts_alpha_decision_probability_summary(G)
    rows = mcts_alpha_depth_summary(G)

    print("\nMCTS ALPHA SUMMARY")
    print("decision contexts:", s["n_ctx"])
    print("mean p(alpha sensor):", "nan" if not np.isfinite(s["mean_p_sensor"]) else f"{s['mean_p_sensor']:.3f}")
    print("mean const mu:", "nan" if not np.isfinite(s["mean_const_mu"]) else f"{s['mean_const_mu']:.3f}")
    print("mean sensor mu:", "nan" if not np.isfinite(s["mean_sensor_mu"]) else f"{s['mean_sensor_mu']:.3f}")
    print("const count:", s["const_count"], "| sensor count:", s["sensor_count"])

    if len(rows) == 0:
        print("no alpha-parent depth rows yet")
        return

    print("\nALPHA PARENT DEPTH SUMMARY")
    print("d | a_node_n | a_node_q | a_node_u | a_edge_n | a_edge_q | a_edge_u")
    print("--|----------|----------|----------|----------|----------|---------")

    for r in rows:
        print(
            f"{int(r['depth']):1d} | "
            f"{r['alpha_node_count']:8d} | "
            f"{r['alpha_node_q_mean']:8.3f} | "
            f"{r['alpha_node_u_mean']:8.3f} | "
            f"{r['alpha_edge_count']:8d} | "
            f"{r['alpha_edge_q_mean']:8.3f} | "
            f"{r['alpha_edge_u_mean']:7.3f}"
        )


#---------------------------------------------------------------------
# plot helpers
#---------------------------------------------------------------------

def plot_depth_probability_progress(prob_hist, title_left, title_right, figsize=(14, 4)):
    """
    generic depth probability + delta plot.

    left:
        probability mass by depth over iterations.

    right:
        absolute probability delta by depth over iterations.
    """

    P = _stack_depth_vectors(prob_hist)

    if P.shape[0] == 0:
        print("no probability history yet")
        return

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im0 = axs[0].imshow(P.T, aspect="auto", origin="lower")
    axs[0].set_title(title_left)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("depth")
    plt.colorbar(im0, ax=axs[0])

    if P.shape[0] > 1:
        D = np.abs(np.diff(P, axis=0))

        im1 = axs[1].imshow(D.T, aspect="auto", origin="lower")
        axs[1].set_title(title_right)
        axs[1].set_xlabel("iteration delta")
        axs[1].set_ylabel("depth")
        plt.colorbar(im1, ax=axs[1])
    else:
        axs[1].text(0.5, 0.5, "need >1 iteration", ha="center", va="center")
        axs[1].set_axis_off()

    plt.show()


def plot_mcts_depth_progress(child_depth_probs_hist, figsize=(14, 4)):
    """
    original x-parent/new-node depth probability plot.
    """

    plot_depth_probability_progress(
        prob_hist=child_depth_probs_hist,
        title_left="p(new node depth)",
        title_right="|delta p(new node depth)|",
        figsize=figsize
    )


def plot_mcts_alpha_parent_depth_progress(alpha_parent_depth_probs_hist, figsize=(14, 4)):
    """
    alpha-parent depth probability plot.

    this is conditional on alpha being selected as a sensor.
    """

    plot_depth_probability_progress(
        prob_hist=alpha_parent_depth_probs_hist,
        title_left="p(alpha parent depth | alpha sensor)",
        title_right="|delta p(alpha parent depth)|",
        figsize=figsize
    )


def plot_mcts_depth_terms(depth_rows, alpha_depth_rows=None, figsize=(16, 4)):
    """
    current-iteration plot of MCTS terms by depth.

    panel 1:
        normal node UCT terms for x-parent selection.

    panel 2:
        normal edge UCB terms for child transform selection.

    panel 3:
        alpha-node UCT terms for alpha-parent selection.
    """

    if len(depth_rows) == 0:
        print("no depth rows yet")
        return

    d = np.asarray([r["depth"] for r in depth_rows], dtype=int)

    node_q = np.asarray([r["node_q_mean"] for r in depth_rows], dtype=float)
    node_u = np.asarray([r["node_expandable_u_mean"] for r in depth_rows], dtype=float)

    edge_q = np.asarray([r["edge_q_mean"] for r in depth_rows], dtype=float)
    edge_u = np.asarray([r["edge_u_mean"] for r in depth_rows], dtype=float)

    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

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

    plt.show()


def plot_mcts_alpha_progress(alpha_sensor_prob_hist, alpha_parent_depth_probs_hist, figsize=(14, 4)):
    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if len(alpha_sensor_prob_hist) > 0:
        axs[0].plot(np.asarray(alpha_sensor_prob_hist, dtype=np.float64), marker="o")
        axs[0].set_title("mean p(alpha sensor)")
        axs[0].set_xlabel("iteration")
        axs[0].set_ylim(-0.05, 1.05)
    else:
        axs[0].text(0.5, 0.5, "no alpha sensor history", ha="center", va="center")
        axs[0].set_axis_off()

    P = _mcts_depth_vector_stack(alpha_parent_depth_probs_hist)

    if P.shape[0] > 0:
        im = axs[1].imshow(P.T, aspect="auto", origin="lower")
        axs[1].set_title("p(alpha parent depth)")
        axs[1].set_xlabel("iteration")
        axs[1].set_ylabel("depth")
        plt.colorbar(im, ax=axs[1])
    else:
        axs[1].text(0.5, 0.5, "no alpha parent depth history", ha="center", va="center")
        axs[1].set_axis_off()

    plt.show()

def plot_mcts_alpha_sensor_probability(alpha_sensor_prob_hist, figsize=(10, 4)):
    """
    plot mean p(alpha sensor) over iterations.

    this should be separate from depth plots because it is not a depth distribution.
    it is the average probability of choosing sensor over constant for alpha.
    """

    if len(alpha_sensor_prob_hist) == 0:
        print("no alpha sensor probability history yet")
        return

    y = np.asarray(alpha_sensor_prob_hist, dtype=np.float64)
    x = np.arange(y.shape[0])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(x, y, marker="o")
    ax.set_title("mean p(alpha sensor)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("probability")

    ax.set_ylim(
        max(0.0, np.nanmin(y) - 0.05),
        min(1.0, np.nanmax(y) + 0.05)
    )

    ax.grid(alpha=0.25)

    plt.show()

from pathlib import Path

def plot_fpc_gene_eval_history_overlay(run_dir, *, max_iters=None, min_alpha=0.05, max_alpha=0.75):
    run_dir = Path(run_dir)
    iter_dirs = sorted((run_dir / "iterations").glob("iter_*"))

    if max_iters is not None:
        iter_dirs = iter_dirs[-int(max_iters):]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    n = len(iter_dirs)

    for i, iter_dir in enumerate(iter_dirs):
        path = iter_dir / "fpc_gene_eval_plot_data.npz"

        if not path.exists():
            continue

        data = np.load(path, allow_pickle=True)

        prop = data["prop_keep"]
        obs = data["observed_keep"]

        if prop.size == 0:
            continue

        age_alpha = min_alpha + (max_alpha - min_alpha) * ((i + 1) / max(n, 1))

        ax.scatter(
            prop,
            obs,
            s=12,
            alpha=float(age_alpha),
        )

        # draw the newest curve/bands only
        if i == n - 1:
            p_grid = data["p_grid"]
            perm_mu = float(data["perm_mu"])
            upper = data["upper_band"]
            lower = data["lower_band"]

            ax.plot(p_grid, np.full_like(p_grid, perm_mu), linewidth=2, label="permutation mean")
            ax.plot(p_grid, upper, linestyle="--", linewidth=1.5, label="+2 std")
            ax.plot(p_grid, lower, linestyle="--", linewidth=1.5, label="-2 std")

    ax.set_xlabel("gene participation proportion p = m / N")
    ax.set_ylabel("EV")
    ax.set_title("FPC-conditioned gene evaluation history")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig, ax