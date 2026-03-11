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