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