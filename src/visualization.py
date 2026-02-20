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