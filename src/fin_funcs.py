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