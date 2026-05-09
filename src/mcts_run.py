# notebook_usage_example.py
#
# Minimal usage from your notebook after creating G exactly as before.

from importlib import reload

import ep_wrap as ep
import evaluation as _E
import visualization as _V
import initialization as _I
import transform_ops as _OPS
import mcts_util as _MU
import numpy as np

reload(_E)
reload(_V)
reload(ep)
reload(_I)
reload(_MU)

G = _I.Grammar(
    type='MCTS',
    max_delta_lookback=80,
    p_mutation=0.00,
    p_crossover=0.00,

    #this is now a prior for alpha-MCTS const-vs-sensor selection
    #not a direct random alpha sensor frequency inside the MCTS branch.
    alpha_sensor_freq=0.5,

    node_fitness='count_pop_dead',
    count_explore=True,
    temp=0.0,
    mode='train',
    explore_const=np.sqrt(2),
    spec_gram_args={
        "key_mode"       : "path",
        "max_depth"      : 5,
        "max_path_depth" : 6,

        "trace_enabled"  : False,

        #progressive widening

        #K(N) = number of child we are allowed to explore
        #K(N) = ceil( c * ( prob ) ^ alpha )
        
        "pw_c"           : 0.75,
        "pw_alpha"       : 0.30,
        "expand_prob"    : 0.10,

        #alpha progressive widening
        "alpha_pw_c"        : 0.50,
        "alpha_pw_alpha"    : 0.25,
        "alpha_expand_prob" : 0.067,

        "base_prior"     : 0.0,
        "unknown_prior"  : 1.0,
        "softmax_base"   : np.e,
        "softmax_temp"   : 0.5,
        "alpha_prior_weight" : 1.0,
    }
)

# Build G exactly as you already do.
# Then run:

result = _MU.run_mcts_tmp_loop(
    G,
    max_iter=1000,

    # tmp is the scratch run.
    # If you leave this unchanged, every new run clears and overwrites mcts_runs/tmp.
    run_name="tmp",
    overwrite=True,

    # same as your original loop
    chunk_num=0,
    fpc_n_sims=2500,

    # keeps old plots captured while also making recreated plots for the app
    save_helper_plots=True,
    save_recreated_plots=True,

    # only turn this on when you really need full X/G/walker/evaluation object snapshots
    save_full_snapshots=False,

    # set False if dictionary history memory gets too large
    store_full_mcts_dict_history=True,

    ep_module=ep,
    E_module=_E,
    V_module=_V,
    I_module=_I,
    OPS_module=_OPS,
)

G = result["G"]
last_X = result["last_X"]
last_s_idx = result["last_s_idx"]
last_pvals = result["last_pvals"]
last_zscores = result["last_zscores"]
histories = result["histories"]

print("run saved to:", result["store"].run_dir)
