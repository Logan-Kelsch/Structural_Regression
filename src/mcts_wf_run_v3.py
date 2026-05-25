"""
mcts_wf_run_all_eval.py

Walk-forward MCTS run entrypoint.

This mirrors the style of mcts_run.py, but imports mcts_util_v2 and calls
run_mcts_walk_forward(...). The central walk-forward controls live near the
bottom of this file:
    n_chunks
    delta_L
    grammar_memory_gamma
"""

from importlib import reload

import numpy as np

import ep_wrap as ep
import evaluation as _E
import visualization as _V
import initialization as _I
import transform_ops as _OPS
import mcts_util_v4 as _MU

reload(_E)
reload(_V)
reload(ep)
reload(_I)
reload(_MU)


# ---------------------------------------------------------------------
# Grammar setup
# ---------------------------------------------------------------------

G = _I.Grammar(
    type="MCTS",
    max_delta_lookback=80,
    p_mutation=0.00,
    p_crossover=0.00,

    # this is a prior for alpha-MCTS const-vs-sensor selection
    alpha_sensor_freq=0.5,

    node_fitness="count_pop_dead",
    count_explore=True,
    temp=0.0,
    mode="train",
    explore_const=np.sqrt(2),
    spec_gram_args={
        "key_mode"       : "path",
        "max_depth"      : 5,
        "max_path_depth" : 6,

        # keep disabled for long walk-forward runs unless actively debugging
        "trace_enabled"  : False,

        # progressive widening
        # K(N) = ceil(pw_c * (N + 1) ** pw_alpha)
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
        "softmax_temp"   : 0.5,
        "alpha_prior_weight" : 1.0,
    },
)


# ---------------------------------------------------------------------
# Walk-forward controls
# ---------------------------------------------------------------------

RUN_NAME = "wf_tmp"
OVERWRITE = True

# default number of chunks available for walk-forward windows
# windows are (i,j,k) = (0,1,2), (1,2,3), ..., (n_chunks-3,n_chunks-2,n_chunks-1)
n_chunks = 20

# policy-drift threshold for ending grammar evolution on the current i->j window
# when smoothed exploit policy drift h < delta_L, the script advances one chunk
delta_L = 0.01

# grammar memory decay between windows
# 1.0 means no decay by default
# 0.95 means previous chunk evidence is multiplied by 0.95 each chunk shift
grammar_memory_gamma = 0.00
grammar_explore_gamma = 0.00

# max grammar-update iterations allowed for one window if delta_L is not reached
max_iters_per_window = 100

# inference investigation after each window convergence
# each infer population has pop_size genes; with default pop_size=100,
# this surveys 30 * 100 = 3,000 generated genes per grammar/window.
infer_populations = 30
success_z = 2.0

# FPC curves are cached once per absolute chunk number and reused across windows
fpc_n_sims = 2500

# console verbosity
# 0 = mostly silent/log only
# 1 = window, iteration, and FPC hit/miss summaries
# 2 = detailed FPC/template/logwalker progress
verbose = 1

# plotting/saving controls
save_helper_plots = True
save_recreated_plots = True
store_full_mcts_dict_history = False


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

result = _MU.run_mcts_walk_forward(
    G,
    n_chunks=n_chunks,
    delta_L=delta_L,
    grammar_memory_gamma=grammar_memory_gamma,
    grammar_explore_gamma=grammar_explore_gamma,
    max_iters_per_window=max_iters_per_window,
    infer_populations=infer_populations,
    success_z=success_z,
    run_name=RUN_NAME,
    overwrite=OVERWRITE,
    fpc_n_sims=fpc_n_sims,
    verbose=verbose,
    save_helper_plots=save_helper_plots,
    save_recreated_plots=save_recreated_plots,
    store_full_mcts_dict_history=store_full_mcts_dict_history,
    ep_module=ep,
    E_module=_E,
    V_module=_V,
    I_module=_I,
    OPS_module=_OPS,
)

G = result["G"]
print("walk-forward run saved to:", result["store"].run_dir)
print("windows completed:", len(result["window_summaries"]))
