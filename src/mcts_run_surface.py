# mcts_run_surface.py

from importlib import reload
from pathlib import Path
import csv
import time
import traceback

import numpy as np

import ep_wrap as ep
import evaluation as _E
import visualization as _V
import initialization as _I
import transform_ops as _OPS
import mcts_util as _MU

reload(_E)
reload(_V)
reload(ep)
reload(_I)
reload(_MU)


# ---------------------------------------------------------------------
# surface settings
# ---------------------------------------------------------------------

MAX_ITERS = 1000
RUN_ROOT = Path("mcts_runs")
RESULTS_CSV = RUN_ROOT / "surface_results.csv"

TEMPS = np.geomspace(0.03, 1.0, 8)
MAX_DEPTHS = [7, 9]#, 11]#[3, 5, 7, 9, 11]


def temp_label(temp):
    """
    compact filename-safe temp label.
    0.0300 -> t0030
    0.3667 -> t0367
    1.0000 -> t1000
    """
    return f"t{int(round(float(temp) * 1000)):04d}"


def run_name_for(temp, max_depth):
    return f"surf_{temp_label(temp)}_d{int(max_depth):02d}"


def make_grammar(temp, max_depth):
    max_depth = int(max_depth)

    return _I.Grammar(
        type="MCTS",
        max_delta_lookback=80,
        p_mutation=0.00,
        p_crossover=0.00,

        # this is now a prior for alpha-MCTS const-vs-sensor selection
        alpha_sensor_freq=0.5,

        node_fitness="count_pop_dead",
        count_explore=True,
        temp=0.0,
        mode="train",
        explore_const=np.sqrt(2),
        spec_gram_args={
            "key_mode"       : "path",
            "max_depth"      : max_depth,
            "max_path_depth" : max_depth + 1,

            "trace_enabled"  : False,

            # progressive widening
            "pw_c"           : 0.75,
            "pw_alpha"       : 0.30,
            "expand_prob"    : 0.10,

            # alpha progressive widening
            "alpha_pw_c"        : 0.50,
            "alpha_pw_alpha"    : 0.25,
            "alpha_expand_prob" : 0.067,

            "base_prior"     : 0.0,
            "unknown_prior"  : 1.0,
            "softmax_base"   : np.e,
            "softmax_temp"   : float(temp),
            "alpha_prior_weight" : 1.0,
        }
    )


def append_result_row(row):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_CSV.exists()

    fieldnames = [
        "run_name",
        "status",
        "temp",
        "max_depth",
        "max_path_depth",
        "elapsed_sec",
        "elapsed_min",
        "run_dir",
        "error",
    ]

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ---------------------------------------------------------------------
# run surface
# ---------------------------------------------------------------------

print("starting MCTS surface sweep")
print("temps:", np.round(TEMPS, 4))
print("depths:", MAX_DEPTHS)
print("results csv:", RESULTS_CSV)

surface_start = time.time()

for max_depth in MAX_DEPTHS:
    for temp in TEMPS:

        if ( max_depth < 7):
            continue
        if ( max_depth==7 and temp < 0.40):
            continue
        run_name = run_name_for(temp, max_depth)
        print(f"\n===== starting {run_name} | temp={temp:.4f} | max_depth={max_depth} =====")

        start = time.time()
        status = "complete"
        err_text = ""

        try:
            G = make_grammar(temp=temp, max_depth=max_depth)

            result = _MU.run_mcts_tmp_loop(
                G,
                max_iter=MAX_ITERS,

                run_root=RUN_ROOT,
                run_name=run_name,
                overwrite=True,

                chunk_num=0,
                fpc_n_sims=2500,

                save_helper_plots=True,
                save_recreated_plots=True,

                save_full_snapshots=False,

                # set False if dictionary history memory gets too large
                store_full_mcts_dict_history=True,

                ep_module=ep,
                E_module=_E,
                V_module=_V,
                I_module=_I,
                OPS_module=_OPS,
            )

            run_dir = str(result["store"].run_dir)

        except Exception:
            status = "crashed"
            err_text = traceback.format_exc()
            run_dir = str(RUN_ROOT / run_name)

            print(f"run crashed: {run_name}")
            print(err_text)

        elapsed = time.time() - start

        append_result_row({
            "run_name": run_name,
            "status": status,
            "temp": float(temp),
            "max_depth": int(max_depth),
            "max_path_depth": int(max_depth) + 1,
            "elapsed_sec": float(elapsed),
            "elapsed_min": float(elapsed / 60.0),
            "run_dir": run_dir,
            "error": err_text,
        })

        print(
            f"===== finished {run_name} | "
            f"status={status} | "
            f"elapsed={elapsed / 60.0:.2f} min | "
            f"saved={run_dir} ====="
        )

surface_elapsed = time.time() - surface_start

print("\nMCTS surface sweep complete")
print(f"total elapsed: {surface_elapsed / 60.0:.2f} min")
print("results csv:", RESULTS_CSV)