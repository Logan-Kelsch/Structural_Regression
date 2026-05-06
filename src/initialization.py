import numpy as np

#the first functionality we are going to put in this file is the most
#general so that we can begin constructing more code for instantiation.
#This function is for randomly generating instructions for instantiation


#we first must assert that we are going to normalize data over a 
# window of time, so that we can have all columns of data be allowed 
# to communicate with each other

#then we must define which columns we are allowed to select from

#when it comes to instantiation,
# we will have an x_inst to reference for generation of instructions
# then with the generated set of instructions, we will instantiate the data
# and append it to the current x_inst
# Therefore chunking will contain split points so that data can be 
# instantiated and appended before a new set of instructions is generated

#here we will make the function for generation of instructions


class Grammar:
    '''
    Temporary class that will contain grammar structures and rules
    '''
    def __init__(
        self,
        type            :   str = 'Null',
        max_delta_lookback  :   int =   48,
        p_mutation      :   float   =   0.05,
        p_crossover     :   float   =   0.025,
        alpha_sensor_freq   :   float   =   0.5,
        node_fitness    :   str =   'count_pop_dead',
        count_explore   :   bool    =   True,
        temp            :   float   =   0.0,
        mode            :   str     =   'train',
        explore_const   :   float   =   np.sqrt(2),
        softmax_temp    :   float   =   1.0,
        spec_gram_args  :   dict    =   None
    ):
        self._type = type
        self._mdl = max_delta_lookback
        self._p_mutation = p_mutation
        self._p_crossover = p_crossover
        self._alpha_sensor_freq = alpha_sensor_freq
        self._node_fitness = node_fitness

        #count explore will allocate all counting mechanisms into
        #instruction generation functionality
        #therefore counts are static within grammar update function (when true)
        #otherwise (when false) counting mechanism is allocated to grammar update
        self._count_explore = count_explore

        self._mode = mode
        self._temp = temp

        self._tq1d = None

        self._t_count = None
        self._t_cum = None
        self._t_mu = None
        self._UCB1 = None
        self._t = None
        self._UCBMAT = None
        self._UCB_EXPLOIT = None
        self._UCB_EXPLORE = None
        self._UCB_CONF = None
        self._c = None

        #MCTS ALPHA STRUCTURES
        #these store the probabilistic behavior for alpha separately from x.
        #reason: good x parents and good alpha parents are not necessarily the same thing.

        self._MCTS_ALPHA_DECISION_CUM = None
        self._MCTS_ALPHA_DECISION_COUNT = None
        self._MCTS_ALPHA_DECISION_MU = None
        self._MCTS_ALPHA_DECISION_EXPLORE_COUNT = None

        self._MCTS_ALPHA_NODE_CUM = None
        self._MCTS_ALPHA_NODE_COUNT = None
        self._MCTS_ALPHA_NODE_MU = None
        self._MCTS_ALPHA_NODE_EXPLORE_COUNT = None

        self._MCTS_ALPHA_EDGE_CUM = None
        self._MCTS_ALPHA_EDGE_COUNT = None
        self._MCTS_ALPHA_EDGE_MU = None
        self._MCTS_ALPHA_EDGE_EXPLORE_COUNT = None

        self._MCTS_ALPHA_EXPLOIT_T = None
        self._MCTS_ALPHA_EXPLORE_T = None

        self._MCTS_ALPHA_PRIOR_WEIGHT = None

        #softmax temperature used by stochastic grammar sampling.
        #0 means nearly greedy, 1 means normal softmax.
        self._softmax_temp = float(np.clip(softmax_temp, 0.0, 1.0))
        

        match(type):
            case 'Null':
                pass

            case 'tq1d':
                #actual long term score vector containing quality values which
                #will be interpreted for probabilistic selection after softmax transformation
                self._tq1d = np.ones(23, np.float32)

            case 'UCB1':
                self._t_count = np.zeros(23, np.float32)
                self._t_cum = np.zeros(23, np.float32)
                self._t_mu = np.zeros(23, np.float32)
                self._UCB1 = np.zeros(23, np.float32)
                self._t = 0

            case 'UCB1-tMAT':
                #for this grammatical structure there will be:
                #some 2d matrix that represents a probabilistic transition matrix
                #this transition matrix will have probabilities converged
                # from a stochastic UCB1 interpretation
                #this can be interpreted as nested.
                
                #DESCRIPTION OF DATA STRUCTURES AND SAMPLIGN
                #DATA STRUCTURES
                # we will have matrices representing:
                # UCBMAT - score for each transition
                # UCB_EXPLOIT - score for each transition
                # UCB_EXPLORE - score for each transition
                # t_count - total count for that transition
                # t - total count of all non terminal nodes
                # c - coefficient for exploration
                # t_cum - total result for each transition

                self._UCBMAT = np.zeros((24,23), np.float32)
                self._UCB_EXPLOIT = np.zeros((24,23), np.float32)
                self._UCB_EXPLORE = np.zeros((24,23), np.float32)
                self._UCB_CONF = np.zeros((24,23), np.float32)
                self._t_cum = np.zeros((24,23), np.float32)
                self._EXPLORE_COUNT = np.zeros((24,23), np.float32)
                self._EXPLOIT_COUNT = np.zeros((24,23), np.float32)
                self._EXPLORE_T = 0
                self._EXPLOIT_T = 0
                self._c = explore_const

                #SAMPLING
                # we will take UCBMAT and resolve 
                # a score for selecting parent nodes for x of new gene
                # this vector is s = np.sum(UCBMAT, axis=1?0???) (length tf)
                # then we will get the existing state multiset
                #  which should be all transitions in instructions (_L_idx)
                # then we will make a proportion vector p (length tf)
                # out of the multiset of existing states
                # then we will sample parent idx with softmax(sp)

            case 'MCTS':

                #for this grammatical structure we will maintain the entire 
                #history of all nodes that are made in training section
                #we have three levels of development that will describe what dimensions
                #of explorable space are condensed.

                #   We will have NOTE main memory stores of identical tree structure
                #   - cumulative score at this node across history
                #   - cumulative count of instances at this state in history
                #   - total counts at this state will be 
    
                #1. We are working only on x sensor and tf id selection
                #   This is synonymous to just edge and states only.
                #   
                #   In a grammatical structure like this we have some

                # ------- NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE -------
                # CANDIDATE STRUCTURE NEEDS SERIOUS VALIDATION OF EFFECTIVENESS
                # ------- NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE -------

                #for this grammatical structure we will maintain sparse tree memory
                #using dictionaries instead of a large dense tensor.
                #
                #reason:
                #MCTS state is not just a function id.
                #MCTS state is more like a path/context through the generated program.
                #therefore, the number of possible states is too large to allocate as
                #a normal numpy matrix.
                #
                #DATA STRUCTURE OVERVIEW:
                #
                #node key:
                #   a hashable tuple describing a state in the generated structure.
                #   default mode is path based:
                #       terminal -> tf -> tf -> tf
                #
                #edge key:
                #   (parent_key, child_tf)
                #
                #node memory:
                #   _MCTS_NODE_CUM[key]       cumulative fitness observed at this state
                #   _MCTS_NODE_COUNT[key]     count of exploit observations at this state
                #   _MCTS_NODE_MU[key]        mean exploit fitness at this state
                #
                #edge memory:
                #   _MCTS_EDGE_CUM[(key, tf)]      cumulative fitness for action key -> tf
                #   _MCTS_EDGE_COUNT[(key, tf)]    count of exploit observations for key -> tf
                #   _MCTS_EDGE_MU[(key, tf)]       mean exploit fitness for key -> tf
                #
                #explore memory:
                #   these are counted during generation if self._count_explore is True.
                #   this is analogous to how UCB1-tMAT uses _EXPLORE_COUNT separately
                #   from _EXPLOIT_COUNT.
                #
                #children:
                #   _MCTS_CHILDREN[key] is a set of opened child tf ids.
                #   this is where progressive widening lives.

                if spec_gram_args is None:
                    spec_gram_args = {}

                self._MCTS_NODE_CUM = {}
                self._MCTS_NODE_COUNT = {}
                self._MCTS_NODE_MU = {}

                self._MCTS_EDGE_CUM = {}
                self._MCTS_EDGE_COUNT = {}
                self._MCTS_EDGE_MU = {}

                self._MCTS_NODE_EXPLORE_COUNT = {}
                self._MCTS_EDGE_EXPLORE_COUNT = {}

                self._MCTS_CHILDREN = {}
                self._MCTS_TRACE = []

                self._MCTS_EXPLOIT_T = 0
                self._MCTS_EXPLORE_T = 0

                #------------------------------------------------------------
                # MCTS alpha memory
                #
                # There are two alpha decisions:
                #
                #   1. alpha decision:
                #       should alpha be a constant or a sensor?
                #
                #   2. alpha parent:
                #       if alpha is a sensor, which existing node should alpha use?
                #
                # These are separate from x-parent selection because alpha has a
                # different structural role than x.
                #------------------------------------------------------------

                self._MCTS_ALPHA_DECISION_CUM = {}
                self._MCTS_ALPHA_DECISION_COUNT = {}
                self._MCTS_ALPHA_DECISION_MU = {}
                self._MCTS_ALPHA_DECISION_EXPLORE_COUNT = {}

                self._MCTS_ALPHA_NODE_CUM = {}
                self._MCTS_ALPHA_NODE_COUNT = {}
                self._MCTS_ALPHA_NODE_MU = {}
                self._MCTS_ALPHA_NODE_EXPLORE_COUNT = {}

                self._MCTS_ALPHA_EDGE_CUM = {}
                self._MCTS_ALPHA_EDGE_COUNT = {}
                self._MCTS_ALPHA_EDGE_MU = {}
                self._MCTS_ALPHA_EDGE_EXPLORE_COUNT = {}

                self._MCTS_ALPHA_EXPLOIT_T = 0
                self._MCTS_ALPHA_EXPLORE_T = 0

                #This controls how strongly the original alpha_sensor_freq
                #biases the const-vs-sensor decision before evidence exists.
                #
                #0.0 means ignore prior and let UCB handle everything.
                #1.0 means include alpha_sensor_freq as a soft prior.
                self._MCTS_ALPHA_PRIOR_WEIGHT = spec_gram_args.get("alpha_prior_weight", 1.0)

                #progressive widening parameters
                #number of opened children is roughly:
                #   c * N(parent)^alpha
                self._MCTS_PW_C = spec_gram_args.get("pw_c", 2.0)
                self._MCTS_PW_ALPHA = spec_gram_args.get("pw_alpha", 0.5)
                self._MCTS_EXPAND_PROB = spec_gram_args.get("expand_prob", 0.35)

                #UCT/UCB parameters
                self._c = explore_const
                self._MCTS_BASE_PRIOR = spec_gram_args.get("base_prior", 0.0)
                self._MCTS_UNKNOWN_PRIOR = spec_gram_args.get("unknown_prior", 1.0)

                #state key parameters
                #path is the recommended default because it keeps actual tree context
                #instead of collapsing everything back into simple tf -> tf transitions
                self._MCTS_KEY_MODE = spec_gram_args.get("key_mode", "path")
                self._MCTS_MAX_PATH_DEPTH = spec_gram_args.get("max_path_depth", 12)

                #softmax base for stochastic UCT/UCB selection
                self._MCTS_SOFTMAX_BASE = spec_gram_args.get("softmax_base", np.e)

                #optionally override global softmax temperature for MCTS
                self._softmax_temp = float(np.clip(
                    spec_gram_args.get("softmax_temp", self._softmax_temp),
                    0.0,
                    1.0
                ))

                

                #maximum generated expression depth.
                #
                #IMPORTANT:
                #_MCTS_MAX_PATH_DEPTH only limits how far the key builder walks
                #when creating a dictionary key.
                #
                #_MCTS_MAX_DEPTH actually prevents the grammar from selecting
                #parents that would create a deeper-than-allowed child.
                #
                #depth convention:
                #   terminal state has depth 0
                #   terminal -> tf has depth 1
                #   terminal -> tf -> tf has depth 2
                #   etc.
                self._MCTS_MAX_DEPTH = spec_gram_args.get("max_depth", 8)

                #this can be the same or slightly larger than max_depth.
                #if it is smaller than max_depth, different deep paths may collapse
                #into the same truncated key, so default it to max_depth + 1.
                self._MCTS_MAX_PATH_DEPTH = spec_gram_args.get(
                    "max_path_depth",
                    self._MCTS_MAX_DEPTH + 1
                )

                #freeze expansion mode
                #False:
                #   normal progressive widening
                #True:
                #   do not open new x->tf children or new alpha parent edges.
                #   only sample from already-opened grammar structure.
                self._MCTS_FREEZE_EXPANSION = spec_gram_args.get("freeze_expansion", False)


            case _:
                raise ValueError(f'Cannot interpret Grammar type "{type}"')
        

    def softmax_sample_uint16(
        self,
        rng: np.random.Generator,
        n: int,
        vect: any = None,
        base: float = np.e,
        samp0: bool = False,
        valid_mask=None,
        allow_uniform_fallback: bool = False,
    ) -> np.ndarray:
        """
        Sample n values according to softmax over a 1D score vector.

        valid_mask can be used to hard-disable invalid states.
        Any value where valid_mask is False receives exactly zero probability.
        """

        match(self._type):
            case 'UCB1-tMAT':
                tq1d = vect
            case 'tq1d':
                tq1d = self._tq1d
            case 'UCB1':
                tq1d = self._UCB1
            case _:
                raise ValueError(
                    f"in softmax sampling of grammar: cant interpret self._type = ({self._type})"
                )

        tq1d = np.asarray(tq1d, dtype=np.float64)

        #temperature adjustment
        #temp near 0 makes selection greedier.
        #temp near 1 leaves scores mostly unchanged.
        temp = np.clip(self._softmax_temp, 1e-6, 1.0)
        tq1d = tq1d / temp

        if tq1d.ndim != 1:
            raise ValueError(f"tq1d must be 1D, got shape {tq1d.shape}")
        if len(tq1d) == 0:
            raise ValueError("tq1d must not be empty")
        if base <= 0:
            raise ValueError(f"base must be > 0, got {base}")

        if self._type == 'UCB1-tMAT' and samp0:
            values = np.arange(len(tq1d), dtype=np.uint16)
        else:
            values = np.arange(1, len(tq1d) + 1, dtype=np.uint16)

        if valid_mask is None:
            valid_mask = np.ones(len(tq1d), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

            if valid_mask.shape != tq1d.shape:
                raise ValueError(
                    f"valid_mask must match tq1d shape {tq1d.shape}, got {valid_mask.shape}"
                )

        valid_mask &= np.isfinite(tq1d)

        if not np.any(valid_mask):
            if allow_uniform_fallback:
                probs = np.full(len(tq1d), 1.0 / len(tq1d), dtype=np.float64)
                return rng.choice(values, size=n, p=probs).astype(np.uint16)

            raise ValueError("softmax_sample_uint16 received no valid sample states")

        if base == 1:
            weights = valid_mask.astype(np.float64)
        else:
            scaled = np.full_like(tq1d, -np.inf, dtype=np.float64)
            scaled[valid_mask] = np.log(base) * tq1d[valid_mask]

            scaled[valid_mask] -= np.max(scaled[valid_mask])

            weights = np.zeros_like(tq1d, dtype=np.float64)
            weights[valid_mask] = np.exp(scaled[valid_mask])

        wsum = weights.sum()

        if wsum <= 0 or not np.isfinite(wsum):
            if allow_uniform_fallback:
                weights = valid_mask.astype(np.float64)
                wsum = weights.sum()
            else:
                raise ValueError("softmax weights collapsed to zero or non-finite values")

        probs = weights / wsum

        return rng.choice(values, size=n, p=probs).astype(np.uint16)
    
    def _mcts_alpha_ctx_key(
        self,
        parent_key,
        child_tf
    ):
        """
        Context key for alpha decisions.

        Alpha should be selected relative to:
            selected x parent state
            selected child transformation function

        This means alpha selection learns things like:
            when I am applying f12 after this x-parent,
            what alpha behavior tends to work?
        """

        return ("ACTX", parent_key, int(child_tf))


    def _mcts_alpha_decision_ucb(
        self,
        ctx_key,
        action
    ):
        """
        UCB score for alpha const-vs-sensor decision.

        action:
            0 = alpha constant
            1 = alpha sensor

        Formula:
            score = Q(ctx -> action)
                    + sqrt(c * log(total + 2) / (N(ctx -> action) + 1))

        This answers:
            should alpha be a constant or a learned sensor in this context?
        """

        action = int(action)
        decision_key = (ctx_key, action)

        q = self._MCTS_ALPHA_DECISION_MU.get(decision_key, self._MCTS_BASE_PRIOR)
        n = self._MCTS_ALPHA_DECISION_EXPLORE_COUNT.get(decision_key, 0)

        if n <= 0 and decision_key not in self._MCTS_ALPHA_DECISION_MU:
            return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

        total = int(self._MCTS_ALPHA_EXPLORE_T or 0) + int(self._MCTS_ALPHA_EXPLOIT_T or 0)

        explore = np.sqrt(
            self._c
            * np.log(total + 2)
            / (n + 1)
        )

        return q + explore


    def _mcts_select_alpha_is_sensor(
        self,
        rng,
        ctx_key
    ):
        """
        Select whether alpha is a constant or a sensor.

        Returns
        -------
        action
            0 means alpha constant
            1 means alpha sensor
        """

        actions = np.asarray([0, 1], dtype=np.int64)

        freeze_expansion = getattr(self, "_MCTS_FREEZE_EXPANSION", False)

        valid_mask = np.ones(actions.shape[0], dtype=bool)

        if freeze_expansion:

            for i in range(actions.shape[0]):

                decision_key = (ctx_key, int(actions[i]))

                valid_mask[i] = (
                    decision_key in self._MCTS_ALPHA_DECISION_MU
                    or decision_key in self._MCTS_ALPHA_DECISION_COUNT
                    or decision_key in self._MCTS_ALPHA_DECISION_EXPLORE_COUNT
                )

            #if this alpha context has never been seen before, use constant alpha.
            #constant alpha avoids creating a new alpha-parent edge.
            if not np.any(valid_mask):
                return 0

        scores = np.asarray([
            self._mcts_alpha_decision_ucb(ctx_key, 0),
            self._mcts_alpha_decision_ucb(ctx_key, 1),
        ], dtype=np.float64)

        #soft prior from alpha_sensor_freq.
        #this does not force the choice; it only biases early sampling.
        p_sensor = float(np.clip(self._alpha_sensor_freq, 1e-6, 1.0 - 1e-6))
        prior = np.asarray([1.0 - p_sensor, p_sensor], dtype=np.float64)

        scores = scores + self._MCTS_ALPHA_PRIOR_WEIGHT * np.log(prior)

        action = int(self._mcts_softmax_sample(
            rng=rng,
            values=actions,
            scores=scores,
            valid_mask=valid_mask & np.isfinite(scores),
            base=self._MCTS_SOFTMAX_BASE,
            allow_uniform_fallback=True
        ))

        return action


    def _mcts_alpha_node_uct(
        self,
        key
    ):
        """
        UCT score for alpha-parent selection.

        This is the alpha-specific version of node UCT.

        It answers:
            how good is this existing node as an alpha sensor?
        """

        #fallback to general node value if alpha-specific value is unknown.
        #this improves cold start behavior.
        q = self._MCTS_ALPHA_NODE_MU.get(
            key,
            self._MCTS_NODE_MU.get(key, self._MCTS_BASE_PRIOR)
        )

        n = self._MCTS_ALPHA_NODE_EXPLORE_COUNT.get(key, 0)

        if n <= 0 and key not in self._MCTS_ALPHA_NODE_MU:
            return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

        total = int(self._MCTS_ALPHA_EXPLORE_T or 0) + int(self._MCTS_ALPHA_EXPLOIT_T or 0)

        explore = np.sqrt(
            self._c
            * np.log(total + 2)
            / (n + 1)
        )

        return q + explore


    def _mcts_select_alpha_parent(
        self,
        rng,
        instructions,
        legal_idx,
        ctx_key=None,
        max_allowed_parent_depth=None
    ):
        """
        Select alpha parent using alpha-specific UCT + softmax.

        max_allowed_parent_depth exists because when alpha is a sensor,
        the new gene depends on both x and alpha.

        Conservative depth logic:
            new_depth = 1 + max(depth(x_parent), depth(alpha_parent))

        Therefore alpha parent must not be too deep.
        """

        legal_idx = np.asarray(legal_idx, dtype=np.int64)

        if legal_idx.size == 0:
            return None, None

        valid_mask = np.ones(legal_idx.shape[0], dtype=bool)

        freeze_expansion = getattr(self, "_MCTS_FREEZE_EXPANSION", False)

        known_alpha_parent_keys = None

        if freeze_expansion:

            if ctx_key is None:
                return None, None

            known_alpha_parent_keys = set()

            for edge_key in self._MCTS_ALPHA_EDGE_MU.keys():
                edge_ctx_key, alpha_parent_key = edge_key
                if edge_ctx_key == ctx_key:
                    known_alpha_parent_keys.add(alpha_parent_key)

            for edge_key in self._MCTS_ALPHA_EDGE_COUNT.keys():
                edge_ctx_key, alpha_parent_key = edge_key
                if edge_ctx_key == ctx_key:
                    known_alpha_parent_keys.add(alpha_parent_key)

            for edge_key in self._MCTS_ALPHA_EDGE_EXPLORE_COUNT.keys():
                edge_ctx_key, alpha_parent_key = edge_key
                if edge_ctx_key == ctx_key:
                    known_alpha_parent_keys.add(alpha_parent_key)

            if len(known_alpha_parent_keys) == 0:
                return None, None
 
                
                

        if not np.any(valid_mask):
            return None, None

        scores = np.empty(legal_idx.shape[0], dtype=np.float64)

        for i in range(legal_idx.shape[0]):
            d = self._mcts_row_depth(instructions, int(legal_idx[i]))

            if max_allowed_parent_depth is not None:
                valid_mask[i] &= d <= int(max_allowed_parent_depth)

            key = self._mcts_state_key(instructions, int(legal_idx[i]))

            if freeze_expansion:
                valid_mask[i] &= key in known_alpha_parent_keys

            scores[i] = self._mcts_alpha_node_uct(key)




        alpha_parent_idx = int(self._mcts_softmax_sample(
            rng=rng,
            values=legal_idx,
            scores=scores,
            valid_mask=valid_mask & np.isfinite(scores),
            base=self._MCTS_SOFTMAX_BASE,
            allow_uniform_fallback=True
        ))

        alpha_parent_key = self._mcts_state_key(instructions, alpha_parent_idx)

        return alpha_parent_idx, alpha_parent_key


    def _mcts_count_alpha_generation(
        self,
        ctx_key,
        action,
        alpha_parent_key=None
    ):
        """
        Count generation-time alpha exploration.

        In frozen expansion mode:
            only count already-known alpha decisions/edges.
            do not create new alpha memory keys.
        """

        action = int(action)
        decision_key = (ctx_key, action)

        freeze_expansion = getattr(self, "_MCTS_FREEZE_EXPANSION", False)

        if freeze_expansion:
            known_decision = (
                decision_key in self._MCTS_ALPHA_DECISION_MU
                or decision_key in self._MCTS_ALPHA_DECISION_COUNT
                or decision_key in self._MCTS_ALPHA_DECISION_EXPLORE_COUNT
            )

            if not known_decision:
                return

        self._MCTS_ALPHA_DECISION_EXPLORE_COUNT[decision_key] = (
            self._MCTS_ALPHA_DECISION_EXPLORE_COUNT.get(decision_key, 0) + 1
        )

        self._MCTS_ALPHA_EXPLORE_T += 1

        if action == 1 and alpha_parent_key is not None:

            alpha_edge_key = (ctx_key, alpha_parent_key)

            if freeze_expansion:
                known_edge = (
                    alpha_edge_key in self._MCTS_ALPHA_EDGE_MU
                    or alpha_edge_key in self._MCTS_ALPHA_EDGE_COUNT
                    or alpha_edge_key in self._MCTS_ALPHA_EDGE_EXPLORE_COUNT
                )

                if not known_edge:
                    return

            self._MCTS_ALPHA_NODE_EXPLORE_COUNT[alpha_parent_key] = (
                self._MCTS_ALPHA_NODE_EXPLORE_COUNT.get(alpha_parent_key, 0) + 1
            )

            self._MCTS_ALPHA_EDGE_EXPLORE_COUNT[alpha_edge_key] = (
                self._MCTS_ALPHA_EDGE_EXPLORE_COUNT.get(alpha_edge_key, 0) + 1
            )


    def _mcts_alpha_info_from_row(
        self,
        X,
        row
    ):
        """
        Recover alpha decision information from an instruction row.

        Returns None if the function does not use alpha.

        Otherwise returns:
            ctx_key
            action
            alpha_parent_key
            alpha_edge_key

        action:
            0 = alpha constant
            1 = alpha sensor
        """

        row = int(row)

        used_flags = int(X._instructions[row, 2])
        sensor_flags = int(X._instructions[row, 4])

        alpha_bit = int(np.uint32(1) << np.uint32(6))

        alpha_used = (used_flags & alpha_bit) != 0

        if not alpha_used:
            return None

        child_tf = int(X._instructions[row, 1])

        parent_idx = self._mcts_parent_abs_idx(X._instructions, row)

        if parent_idx < 0 or parent_idx >= X._instructions.shape[0]:
            parent_key = ("ROOT",)
        else:
            parent_key = self._mcts_state_key(X._instructions, parent_idx)

        ctx_key = self._mcts_alpha_ctx_key(
            parent_key=parent_key,
            child_tf=child_tf
        )

        alpha_is_sensor = (sensor_flags & alpha_bit) != 0

        if not alpha_is_sensor:
            return {
                "ctx_key"          : ctx_key,
                "action"           : 0,
                "alpha_parent_key" : None,
                "alpha_edge_key"   : None,
            }

        alpha_parent_idx = int(X._instructions[row, 0] + X._instructions[row, 6])

        if alpha_parent_idx < 0 or alpha_parent_idx >= X._instructions.shape[0]:
            alpha_parent_key = ("ROOT",)
        else:
            alpha_parent_key = self._mcts_state_key(X._instructions, alpha_parent_idx)

        alpha_edge_key = (ctx_key, alpha_parent_key)

        return {
            "ctx_key"          : ctx_key,
            "action"           : 1,
            "alpha_parent_key" : alpha_parent_key,
            "alpha_edge_key"   : alpha_edge_key,
        }

    def _mcts_key_depth(
        self,
        key
    ):
        """
        Return nonterminal transform depth from an MCTS key.

        Examples
        --------
        ("P", ("T", 2))
            depth = 0

        ("P", ("T", 2), 12)
            depth = 1

        ("P", ("T", 2), 12, 10, 6)
            depth = 3

        ("P", ("T", 2), 12, 10, 6, 19, 5, 4, 8)
            depth = 7

        Notes
        -----
        The terminal is not counted as a transformation.
        Only integer transformation function IDs after the terminal count.
        """

        if key is None:
            return 0

        if not isinstance(key, tuple):
            return 0

        if len(key) == 0:
            return 0

        if key[0] == "P":
            d = 0

            for v in key[1:]:
                if isinstance(v, (int, np.integer)):
                    d += 1

            return d

        if key[0] == "TF":
            return 1

        if key[0] == "L":
            #local keys represent parent_tf -> child_tf, so interpret as shallow
            return 1

        if key[0] == "T":
            return 0

        return 0


    def _mcts_row_depth(
        self,
        instructions,
        row
    ):
        """
        Compute depth for an actual instruction row.

        This is preferred over reading only the key string because it walks the
        true x-parent chain in the instruction matrix.

        depth convention:
            terminal = 0
            terminal -> tf = 1
            terminal -> tf -> tf = 2
        """

        row = int(row)

        if row < 0 or row >= instructions.shape[0]:
            return 0

        depth = 0
        seen = set()
        cur = row

        for _ in range(int(self._MCTS_MAX_DEPTH) + 5):

            if cur < 0 or cur >= instructions.shape[0]:
                break

            if cur in seen:
                #cycle protection.
                #should not happen with valid negative x offsets.
                break

            seen.add(cur)

            tf = int(instructions[cur, 1])

            if tf == 0:
                break

            depth += 1

            prow = self._mcts_parent_abs_idx(instructions, cur)

            if prow == cur:
                break

            cur = prow

        return depth


    def _mcts_valid_depth_parent_mask(
        self,
        instructions,
        legal_idx
    ):
        """
        Return mask over legal_idx where selecting that parent would not exceed
        max generated expression depth.

        If parent has depth d, then the new child would have depth d + 1.

        Therefore parent is legal iff:
            d + 1 <= _MCTS_MAX_DEPTH

        Equivalently:
            d < _MCTS_MAX_DEPTH
        """

        legal_idx = np.asarray(legal_idx, dtype=np.int64)

        mask = np.zeros(legal_idx.shape[0], dtype=bool)

        for i in range(legal_idx.shape[0]):
            row = int(legal_idx[i])

            d = self._mcts_row_depth(instructions, row)

            mask[i] = d < int(self._MCTS_MAX_DEPTH)

        return mask

    def _mcts_softmax_sample(
        self,
        rng,
        values,
        scores,
        valid_mask=None,
        base=None,
        allow_uniform_fallback=True
    ):
        """
        MCTS version of softmax sampling.

        Difference from softmax_sample_uint16:
        - this samples from arbitrary values, not only [1, n] or [0, n).
        - this is needed because MCTS parent selection samples actual legal gene indices.
        - child selection still samples tf ids, but parent selection samples rows/columns.

        Parameters
        ----------
        rng
            numpy generator.

        values
            1D array of actual values to sample from.
            examples:
                legal parent indices from pop_prior._L_idx
                child tf ids in [1, 23]

        scores
            1D array of UCT/UCB scores matching values.

        valid_mask
            optional boolean mask for hard disabling impossible values.

        base
            softmax base. If None, use self._MCTS_SOFTMAX_BASE.

        allow_uniform_fallback
            if True, fallback to uniform over valid states when scores collapse.
        """

        values = np.asarray(values)
        scores = np.asarray(scores, dtype=np.float64)

        #temperature adjustment
        #temp near 0 makes UCT/UCB sampling nearly greedy.
        #temp near 1 gives normal stochastic softmax behavior.
        temp = np.clip(self._softmax_temp, 1e-6, 1.0)
        scores = scores / temp

        if base is None:
            base = self._MCTS_SOFTMAX_BASE

        if values.ndim != 1:
            raise ValueError(f"MCTS values must be 1D, got shape {values.shape}")

        if scores.ndim != 1:
            raise ValueError(f"MCTS scores must be 1D, got shape {scores.shape}")

        if values.shape[0] != scores.shape[0]:
            raise ValueError(
                f"MCTS values and scores must have same length, got {values.shape[0]} and {scores.shape[0]}"
            )

        if values.shape[0] == 0:
            raise ValueError("MCTS softmax received no values")

        if valid_mask is None:
            valid_mask = np.ones(scores.shape[0], dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != scores.shape:
            raise ValueError(
                f"MCTS valid_mask must match score shape {scores.shape}, got {valid_mask.shape}"
            )

        valid_mask &= np.isfinite(scores)

        if not np.any(valid_mask):
            if allow_uniform_fallback:
                valid_mask = np.ones(scores.shape[0], dtype=bool)
            else:
                raise ValueError("MCTS softmax received no valid states")

        if base == 1:
            weights = valid_mask.astype(np.float64)
        else:
            scaled = np.full_like(scores, -np.inf, dtype=np.float64)
            scaled[valid_mask] = np.log(base) * scores[valid_mask]

            #standard numerical stability trick
            scaled[valid_mask] -= np.max(scaled[valid_mask])

            weights = np.zeros_like(scores, dtype=np.float64)
            weights[valid_mask] = np.exp(scaled[valid_mask])

        wsum = weights.sum()

        if wsum <= 0 or not np.isfinite(wsum):
            if allow_uniform_fallback:
                weights = valid_mask.astype(np.float64)
                wsum = weights.sum()
            else:
                raise ValueError("MCTS softmax weights collapsed")

        probs = weights / wsum

        return rng.choice(values, p=probs)


    def _mcts_parent_abs_idx(
        self,
        instructions,
        row
    ):
        """
        Recover absolute parent index from instruction row.

        In your project structure:
            col 0 = absolute/current population index
            col 5 = x sensor offset

        Therefore:
            parent_abs_idx = instructions[row, 0] + instructions[row, 5]

        This matches the logic already used in UCB1-tMAT update.
        """

        return int(instructions[row, 0] + instructions[row, 5])


    def _mcts_state_key(
        self,
        instructions,
        row
    ):
        """
        Convert a generated instruction row into a sparse MCTS state key.

        This is the most important structural choice.

        key_mode == "tf":
            state is only the current transformation function.
            this is fastest, but collapses back toward UCB1 behavior.

        key_mode == "local":
            state is parent_tf -> current_tf.
            this is similar to UCB1-tMAT, but stored sparsely.

        key_mode == "path":
            state is terminal/root path -> ... -> current_tf.
            this is the recommended MCTS structure because it preserves context.

        Notes
        -----
        This does not mutate anything.
        It only reads the current instruction matrix and produces a dictionary key.
        """

        row = int(row)

        if row < 0 or row >= instructions.shape[0]:
            return ("ROOT",)

        tf = int(instructions[row, 1])

        if self._MCTS_KEY_MODE == "tf":
            return ("TF", tf)

        if self._MCTS_KEY_MODE == "local":
            if tf == 0:
                return ("T", int(instructions[row, 0]))

            prow = self._mcts_parent_abs_idx(instructions, row)

            if prow < 0 or prow >= instructions.shape[0]:
                return ("L", "ROOT", tf)

            ptf = int(instructions[prow, 1])
            return ("L", ptf, tf)

        if self._MCTS_KEY_MODE == "path":
            path = []
            seen = set()
            cur = row

            for _ in range(int(self._MCTS_MAX_PATH_DEPTH)):

                if cur < 0 or cur >= instructions.shape[0]:
                    path.append(("ROOT",))
                    break

                if cur in seen:
                    #cycle protection.
                    #this should not happen if offsets are legal, but protects the key builder.
                    path.append(("CYCLE", int(cur)))
                    break

                seen.add(cur)

                cur_tf = int(instructions[cur, 1])

                if cur_tf == 0:
                    #terminals all have tf 0, but different terminal columns mean different information.
                    #therefore include the actual terminal index to avoid collapsing close/volume/time/etc.
                    path.append(("T", int(instructions[cur, 0])))
                    break

                path.append(cur_tf)

                prow = self._mcts_parent_abs_idx(instructions, cur)

                if prow == cur:
                    path.append(("SELF", int(cur)))
                    break

                cur = prow

            path.reverse()
            return tuple(["P"] + path)

        raise ValueError(f"unknown MCTS key mode: {self._MCTS_KEY_MODE}")


    def _mcts_get_node_count(
        self,
        key,
        use_explore=True
    ):
        """
        Return count for a node.

        If use_explore is True and self._count_explore is True, use generation-time counts.
        Otherwise, use exploit/update counts.
        """

        if use_explore and self._count_explore is True:
            return self._MCTS_NODE_EXPLORE_COUNT.get(key, 0)

        return self._MCTS_NODE_COUNT.get(key, 0)


    def _mcts_get_edge_count(
        self,
        edge_key,
        use_explore=True
    ):
        """
        Return count for an edge.

        This mirrors the split between _EXPLORE_COUNT and _EXPLOIT_COUNT
        in the UCB1-tMAT grammar.
        """

        if use_explore and self._count_explore is True:
            return self._MCTS_EDGE_EXPLORE_COUNT.get(edge_key, 0)

        return self._MCTS_EDGE_COUNT.get(edge_key, 0)


    def _mcts_node_uct(
        self,
        key
    ):
        """
        UCT score for WHERE selection.

        This scores an already-existing state in the current generated population.
        Higher score means this node is a better x-parent candidate.

        score = exploit + explore
        exploit = mean fitness at this node
        explore = sqrt(c * log(total visits + 1) / (node visits + 1))
        """

        q = self._MCTS_NODE_MU.get(key, self._MCTS_BASE_PRIOR)

        n = self._mcts_get_node_count(key, use_explore=True)

        if n <= 0 and key not in self._MCTS_NODE_MU:
            return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

        explore = np.sqrt(self._c * np.log(self._MCTS_EXPLORE_T + self._MCTS_EXPLOIT_T + 2) / (n + 1))

        return q + explore


    def _mcts_edge_ucb(
        self,
        parent_key,
        child_tf
    ):
        """
        UCB score for HOW selection.

        This scores a child transformation function from one selected parent state.

        edge_key = (parent_key, child_tf)

        score = exploit + explore
        exploit = mean fitness observed for parent_key -> child_tf
        explore = sqrt(c * log(parent visits + 1) / (edge visits + 1))
        """

        child_tf = int(child_tf)
        edge_key = (parent_key, child_tf)

        q = self._MCTS_EDGE_MU.get(edge_key, self._MCTS_BASE_PRIOR)

        parent_n = self._mcts_get_node_count(parent_key, use_explore=True)
        edge_n = self._mcts_get_edge_count(edge_key, use_explore=True)

        if edge_n <= 0 and edge_key not in self._MCTS_EDGE_MU:
            return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

        explore = np.sqrt(self._c * np.log(parent_n + 2) / (edge_n + 1))

        return q + explore


    def _mcts_pw_limit(
        self,
        parent_key
    ):
        """
        Progressive widening child limit.

        This controls how many child tf ids are allowed to be opened from a parent.

        limit = ceil(c * (N(parent) + 1)^alpha)

        This makes the grammar:
        - narrow when parent state is rarely visited
        - wider when parent state becomes more trusted / more common
        """

        n = self._mcts_get_node_count(parent_key, use_explore=True)

        lim = int(np.ceil(self._MCTS_PW_C * ((n + 1) ** self._MCTS_PW_ALPHA)))

        return int(np.clip(lim, 1, 23))


    def _mcts_select_parent(
        self,
        rng,
        instructions,
        legal_idx
    ):
        """
        WHERE selection.

        Given the current population structure, evaluate every legal existing node
        as a possible x parent.

        This version enforces max MCTS depth.

        A parent is only legal if:
            depth(parent) + 1 <= self._MCTS_MAX_DEPTH

        Returns
        -------
        parent_idx
            absolute index of selected parent node.

        parent_key
            sparse MCTS key for selected parent node.
        """

        legal_idx = np.asarray(legal_idx, dtype=np.int64)

        if legal_idx.size == 0:
            raise ValueError("MCTS parent selection received empty legal_idx")

        #------------------------------------------------------------
        # max-depth filter
        #
        # this is the actual enforcement.
        # without this, _MCTS_MAX_PATH_DEPTH only controls the key length,
        # not the generated structure depth.
        #------------------------------------------------------------

        depth_mask = self._mcts_valid_depth_parent_mask(
            instructions=instructions,
            legal_idx=legal_idx
        )

        #------------------------------------------------------------
        # freeze expansion filter
        #
        #if expansion is frozen, only select parents that already have
        #opened child tf actions.
        #------------------------------------------------------------

        if getattr(self, "_MCTS_FREEZE_EXPANSION", False):

            opened_mask = np.zeros(legal_idx.shape[0], dtype=bool)

            for i in range(legal_idx.shape[0]):
                key = self._mcts_state_key(instructions, int(legal_idx[i]))
                opened_mask[i] = self._mcts_has_open_child_tf(key)

            depth_mask &= opened_mask

            if not np.any(depth_mask):
                raise RuntimeError(
                    "MCTS freeze_expansion=True, but no legal parent has opened children. "
                    "Run at least one exploration phase before freezing expansion."
                )

        #if every parent is too deep, fall back to terminal nodes only.
        #this prevents generation from crashing while still trying to reset
        #the new branch back to source material.
        if not np.any(depth_mask):

            tf_vals = instructions[legal_idx, 1].astype(int)

            terminal_mask = tf_vals == 0

            if np.any(terminal_mask):
                depth_mask = terminal_mask
            else:
                #last-resort fallback:
                #choose the shallowest available nodes.
                depths = np.asarray([
                    self._mcts_row_depth(instructions, int(idx))
                    for idx in legal_idx
                ], dtype=np.int64)

                min_depth = np.min(depths)

                depth_mask = depths == min_depth

        legal_idx_depth = legal_idx[depth_mask]

        scores = np.empty(legal_idx_depth.shape[0], dtype=np.float64)
        keys = []

        for i in range(legal_idx_depth.shape[0]):
            key = self._mcts_state_key(instructions, int(legal_idx_depth[i]))
            keys.append(key)
            scores[i] = self._mcts_node_uct(key)

        parent_idx = int(self._mcts_softmax_sample(
            rng=rng,
            values=legal_idx_depth,
            scores=scores,
            valid_mask=np.isfinite(scores),
            base=self._MCTS_SOFTMAX_BASE,
            allow_uniform_fallback=True
        ))

        parent_key = self._mcts_state_key(instructions, parent_idx)

        return parent_idx, parent_key

    def _mcts_select_child_tf(
        self,
        rng,
        parent_key
    ):
        """
        HOW selection.

        Given one selected parent state, choose the child transformation function.

        In train mode:
            use progressive widening.
            if allowed, open a new child tf.
            otherwise use UCB + softmax over opened children.

        In infer mode:
            avoid opening new children when possible.
            use exploit/UCB memory over known children.
        """

        legal_tf = np.arange(1, 24, dtype=np.int64)

        if parent_key not in self._MCTS_CHILDREN:
            self._MCTS_CHILDREN[parent_key] = set()

        opened = self._MCTS_CHILDREN[parent_key]

        pw_lim = self._mcts_pw_limit(parent_key)

        #------------------------------------------------------------
        # progressive widening / frozen-expansion control
        #------------------------------------------------------------

        can_expand = len(opened) < pw_lim

        freeze_expansion = getattr(self, "_MCTS_FREEZE_EXPANSION", False)

        if self._mode == "train" and freeze_expansion is False:
            do_expand = can_expand and (
                (len(opened) == 0)
                or (rng.random() < self._MCTS_EXPAND_PROB)
            )
        else:
            do_expand = False

        #------------------------------------------------------------
        # expansion path
        #
        # normal train mode:
        #   if widening allows it, open a new child tf.
        #
        # frozen mode:
        #   this block is skipped entirely.
        #------------------------------------------------------------

        if do_expand:

            if len(opened) == 0:
                unused_tf = legal_tf
            else:
                used_tf = np.asarray(list(opened), dtype=np.int64)
                unused_tf = legal_tf[~np.isin(legal_tf, used_tf)]

            if unused_tf.size > 0:
                child_tf = int(rng.choice(unused_tf))
                opened.add(child_tf)
                return child_tf

        #------------------------------------------------------------
        # no opened children fallback
        #
        # normal mode:
        #   open one child so the parent can be used.
        #
        # frozen mode:
        #   return None because opening a child would violate freeze.
        #   parent selection should normally avoid this case.
        #------------------------------------------------------------

        if len(opened) == 0:

            if freeze_expansion:
                return None

            child_tf = int(rng.choice(legal_tf))
            opened.add(child_tf)
            return child_tf

        #------------------------------------------------------------
        # UCB + softmax over already-opened children
        #
        # this is the only path used in frozen mode.
        #------------------------------------------------------------

        child_values = np.asarray(sorted(opened), dtype=np.int64)

        scores = np.empty(child_values.shape[0], dtype=np.float64)

        for i in range(child_values.shape[0]):
            scores[i] = self._mcts_edge_ucb(parent_key, int(child_values[i]))

        child_tf = int(self._mcts_softmax_sample(
            rng=rng,
            values=child_values,
            scores=scores,
            valid_mask=np.isfinite(scores),
            base=self._MCTS_SOFTMAX_BASE,
            allow_uniform_fallback=True
        ))

        return child_tf


    def _mcts_count_generation(
        self,
        parent_key,
        child_tf
    ):
        """
        Count generation-time exploration.

        This is only used when:
            self._mode == "train"
            self._count_explore is True

        This mirrors the UCB1-tMAT behavior where exploration counts can be updated
        during instruction generation instead of only during grammar.update().
        """

        child_tf = int(child_tf)
        edge_key = (parent_key, child_tf)

        self._MCTS_NODE_EXPLORE_COUNT[parent_key] = self._MCTS_NODE_EXPLORE_COUNT.get(parent_key, 0) + 1
        self._MCTS_EDGE_EXPLORE_COUNT[edge_key] = self._MCTS_EDGE_EXPLORE_COUNT.get(edge_key, 0) + 1

        self._MCTS_EXPLORE_T += 1

        if parent_key not in self._MCTS_CHILDREN:
            self._MCTS_CHILDREN[parent_key] = set()

        self._MCTS_CHILDREN[parent_key].add(child_tf)


    def _mcts_update_mean(
        self,
        cum_dict,
        count_dict,
        mu_dict,
        key,
        val
    ):
        """
        Small helper for updating sparse cumulative/count/mean dictionaries.
        """

        cum_dict[key] = cum_dict.get(key, 0.0) + float(val)
        count_dict[key] = count_dict.get(key, 0) + 1
        mu_dict[key] = cum_dict[key] / (count_dict[key] + 1)

    def _mcts_default_quality_fn(
        self,
        raw_score,
        idx=None,
        X=None
    ):
        """
        Default quality function.

        Assumes raw_score is a p-value.
        Converts p-value into clipped fitness where:
            p = 0.05 maps near 1
            p = 1.00 maps near 0
        """

        raw_score = float(raw_score)

        if not np.isfinite(raw_score):
            return 0.0

        raw_score = np.clip(raw_score, 1e-12, 1.0)

        return float(np.clip(
            -np.log(raw_score) / (-np.log(0.05)),
            0.0,
            1.0
        ))

    def _mcts_backup_path_rewards(
        self,
        X,
        leaf_idx,
        leaf_score,
        quality_fn=None,
        backup_gamma=0.85,
        min_backup_weight=0.05,
        include_terminal=False
    ):
        """
        Build node and edge rewards from one scored leaf/gene.

        This performs discounted path backup.

        Returns
        -------
        node_updates
            list of (node_key, reward)

        edge_updates
            list of (edge_key, reward)

        Notes
        -----
        chain order is:
            leaf, parent, grandparent, ...

        distance_from_leaf:
            leaf = 0
            parent = 1
            grandparent = 2
        """

        if quality_fn is None:
            q_leaf = self.pvalue_to_fitness(
                raw_score=leaf_score,
                idx=leaf_idx,
                X=X
            )
        else:
            q_leaf = float(quality_fn(
                raw_score=leaf_score,
                idx=leaf_idx,
                X=X
            ))

        if not np.isfinite(q_leaf):
            q_leaf = 0.0

        q_leaf = float(np.clip(q_leaf, 0.0, 1.0))

        chain = self._mcts_parent_chain(
            X=X,
            idx=leaf_idx,
            include_self=True,
            include_terminal=include_terminal
        )

        node_updates = []
        edge_updates = []

        for dist, node_idx in enumerate(chain):

            weight = float(backup_gamma ** dist)

            if weight < min_backup_weight:
                continue

            reward = q_leaf * weight

            node_key = self._mcts_state_key(X._instructions, int(node_idx))
            node_updates.append((node_key, reward))

            #edge update:
            #for the current node, update the incoming parent -> child_tf edge
            child_tf = int(X._instructions[node_idx, 1])

            if child_tf == 0:
                continue

            parent_idx = self._mcts_parent_abs_idx(X._instructions, int(node_idx))

            if parent_idx < 0 or parent_idx >= X._instructions.shape[0]:
                parent_key = ("ROOT",)
            else:
                parent_key = self._mcts_state_key(X._instructions, int(parent_idx))

            edge_key = (parent_key, child_tf)
            edge_updates.append((edge_key, reward))

        alpha_updates = []

        for dist, node_idx in enumerate(chain):

            weight = float(backup_gamma ** dist)

            if weight < min_backup_weight:
                continue

            reward = q_leaf * weight

            node_key = self._mcts_state_key(X._instructions, int(node_idx))
            node_updates.append((node_key, reward))

            child_tf = int(X._instructions[node_idx, 1])

            if child_tf == 0:
                continue

            parent_idx = self._mcts_parent_abs_idx(X._instructions, int(node_idx))

            if parent_idx < 0 or parent_idx >= X._instructions.shape[0]:
                parent_key = ("ROOT",)
            else:
                parent_key = self._mcts_state_key(X._instructions, int(parent_idx))

            edge_key = (parent_key, child_tf)
            edge_updates.append((edge_key, reward))

            #------------------------------------------------------------
            # alpha backup
            #
            # If this node used alpha, also update:
            #   const-vs-sensor decision memory
            #   alpha-parent node memory
            #   alpha context -> alpha_parent edge memory
            #------------------------------------------------------------

            alpha_info = self._mcts_alpha_info_from_row(
                X=X,
                row=int(node_idx)
            )

            if alpha_info is not None:
                alpha_updates.append(("decision", (alpha_info["ctx_key"], alpha_info["action"]), reward))

                if alpha_info["action"] == 1 and alpha_info["alpha_parent_key"] is not None:
                    alpha_updates.append(("alpha_node", alpha_info["alpha_parent_key"], reward))
                    alpha_updates.append(("alpha_edge", alpha_info["alpha_edge_key"], reward))

        return node_updates, edge_updates, alpha_updates

    def _mcts_parent_chain(
        self,
        X,
        idx,
        include_self=True,
        include_terminal=False,
        max_steps=None
    ):
        """
        Return parent chain for one generated node.

        Output order:
            [leaf, parent, grandparent, ..., terminal/root]

        Parameters
        ----------
        X
            population object with X._instructions.

        idx
            absolute instruction index of starting node.

        include_self
            if True, include idx as the first element.

        include_terminal
            if True, include terminal tf=0 nodes.
            if False, stop before adding terminal nodes.

        max_steps
            safety cap. If None, use _MCTS_MAX_DEPTH + 5.
        """

        if max_steps is None:
            max_steps = int(getattr(self, "_MCTS_MAX_DEPTH", 12)) + 5

        idx = int(idx)

        chain = []
        seen = set()
        cur = idx

        for _ in range(max_steps):

            if cur < 0 or cur >= X._instructions.shape[0]:
                break

            if cur in seen:
                break

            seen.add(cur)

            tf = int(X._instructions[cur, 1])

            if tf == 0:
                if include_terminal:
                    chain.append(cur)
                break

            if include_self or cur != idx:
                chain.append(cur)

            parent_idx = self._mcts_parent_abs_idx(X._instructions, cur)

            if parent_idx == cur:
                break

            cur = parent_idx

        return np.asarray(chain, dtype=np.int64)

    def pvalue_to_fitness(
        self,
        p=None,
        raw_score=None,
        idx=None,
        X=None,
        p_ref: float = 0.05,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        eps: float = 1e-12,
        **kwargs
    ):
        """
        Convert p-value(s) into bounded grammar fitness.

        This supports both direct calls:

            self.pvalue_to_fitness(pvals)

        and MCTS quality-function calls:

            quality_fn(raw_score=leaf_score, idx=leaf_idx, X=X)

        Parameters
        ----------
        p
            Scalar or array of p-values.

        raw_score
            Alias for p. Used when this function is passed as a quality_fn.

        idx
            Optional gene index. Included for compatibility with MCTS quality_fn.

        X
            Optional population object. Included for compatibility with MCTS quality_fn.

        p_ref
            Reference p-value that maps to fitness = 1. Default 0.05.

        Returns
        -------
        fitness
            Scalar if input is scalar.
            np.ndarray if input is array-like.
        """

        #allow MCTS backup calls to pass raw_score instead of p
        if p is None:
            p = raw_score

        if p is None:
            raise ValueError("pvalue_to_fitness requires p or raw_score")

        p_arr = np.asarray(p, dtype=np.float64)

        p_arr = np.clip(p_arr, eps, 1.0)
        p_ref = float(np.clip(p_ref, eps, 1.0 - eps))

        fitness = -np.log(p_arr) / (-np.log(p_ref))
        fitness = np.clip(fitness, clip_min, clip_max)

        if np.ndim(p) == 0:
            return float(fitness)

        return fitness.astype(np.float32)
    
    def _mcts_has_open_child_tf(
        self,
        parent_key
    ):
        """
        Return True if this parent has at least one already-opened child tf.

        In frozen expansion mode, parent selection should prefer only these parents,
        because parents with no opened children cannot produce a child without
        expanding the tree.
        """

        opened = self._MCTS_CHILDREN.get(parent_key, set())

        return opened is not None and len(opened) > 0

    def freeze_mcts_expansion(
        self,
        freeze: bool = True
    ):
        """
        Toggle MCTS expansion freezing.

        freeze=False
            normal progressive widening.
            new child tf actions may be opened.

        freeze=True
            no new child tf actions are opened.
            no new alpha parent edges are opened.
            only already-opened structure is sampled.

        Notes
        -----
        This is different from infer mode.

        In train mode with freeze=True:
            generation still happens
            counts can still update
            grammar.update can still update scores

        In infer mode with freeze=True:
            generation happens without changing learned memory
        """

        self._MCTS_FREEZE_EXPANSION = bool(freeze)

    def update(
        self,
        X,
        family_idx,
        family_scores,
        quality_fn=None,
        backup_gamma=0.85,
        backup_reduce="max",
    ):
        match(self._type):
            case "UCB1-tMAT":

                # landing state for each family node
                family_tfidx = X._instructions[family_idx, 1].astype(int)

                # transform p-values to fitness
                # family_fitness = -np.sqrt(2 * family_scores) + 1
                # family_fitness = ((0.5 - family_scores) / 0.629961) ** 3 + 0.5
                family_fitness = self.pvalue_to_fitness(family_scores)

                # discard family nodes that land in state 0
                keep = (family_tfidx != 0)
                family_idx = family_idx[keep]
                family_tfidx = family_tfidx[keep]
                family_scores = family_scores[keep]
                family_fitness = family_fitness[keep]

                # rows = leaving state 0..23
                # cols = landing state 1..23 shifted to 0..22
                local_tq2d = np.zeros((24, 23), dtype=np.float32)
                local_tfrq2d = np.zeros((24, 23), dtype=np.float32)

                # parent/leaving state for each surviving family node
                parent_tfidx = np.empty(family_idx.shape[0], dtype=np.int64)
                for i in range(family_idx.shape[0]):
                    row = int(family_idx[i])
                    prow = int(X._instructions[row, 0] + X._instructions[row, 5])
                    parent_tfidx[i] = int(X._instructions[prow, 1])

                # landing state goes to dim1 with -1 shift because state 0 is not allowed there
                land_col = family_tfidx - 1

                match (self._node_fitness):
                    case "count_all":
                        for i in range(family_idx.shape[0]):
                            r = parent_tfidx[i]
                            c = land_col[i]
                            local_tq2d[r, c] += family_fitness[i]
                            local_tfrq2d[r, c] += 1

                        nz = local_tfrq2d > 0
                        local_tq2d[nz] /= local_tfrq2d[nz]

                    case "count_best":
                        seen = {}
                        for i in range(family_idx.shape[0]):
                            key = (int(parent_tfidx[i]), int(land_col[i]))
                            val = family_fitness[i]
                            if key not in seen or val > seen[key]:
                                seen[key] = val

                        for (r, c), val in seen.items():
                            local_tq2d[r, c] = val
                            local_tfrq2d[r, c] = 1

                    case "count_worst":
                        seen = {}
                        for i in range(family_idx.shape[0]):
                            key = (int(parent_tfidx[i]), int(land_col[i]))
                            val = family_fitness[i]
                            if key not in seen or val < seen[key]:
                                seen[key] = val

                        for (r, c), val in seen.items():
                            local_tq2d[r, c] = val
                            local_tfrq2d[r, c] = 1

                    case "count_pop_null" | "count_pop_dead":
                        fill_score = 0.5 if self._node_fitness == "count_pop_null" else 1.0
                        fill_fitness = float(np.clip(-np.log(fill_score) / (-np.log(0.05)), None, 1))

                        # family nodes keep their reduced/propagated fitness
                        fam_fit_map = {int(family_idx[i]): float(family_fitness[i]) for i in range(family_idx.shape[0])}

                        # use full gene space
                        pop_idx = np.asarray(X._G_idx, dtype=np.int64)

                        for idx in pop_idx:
                            idx = int(idx)

                            # landing state
                            tf = int(X._instructions[idx, 1])
                            if tf == 0:
                                continue

                            # leaving state from parent
                            prow = int(X._instructions[idx, 0] + X._instructions[idx, 5])
                            ptf = int(X._instructions[prow, 1])

                            # use family fitness if available, otherwise fallback fitness
                            fit = fam_fit_map.get(idx, fill_fitness)

                            local_tq2d[ptf, tf - 1] += fit
                            local_tfrq2d[ptf, tf - 1] += 1

                        nz = local_tfrq2d > 0
                        local_tq2d[nz] /= local_tfrq2d[nz]


                    case _:
                        raise ValueError(f"unknown node fitness mode: {self._node_fitness}")

                #here we are counting seperately for exploit members
                self._t_cum += local_tq2d
                self._EXPLOIT_COUNT += local_tfrq2d
                self._EXPLOIT_T += np.sum(local_tfrq2d, axis=(0, 1))
                
                self._UCB_EXPLOIT = self._t_cum/(self._EXPLOIT_COUNT + 1)

                #if you get confused here, really we have this switch case to protect non functioning
                #stuff from breaking the system. old mechanism only used counts from models that actually made it to
                #the final testing, while new mechanism is actually ripping ALL generation stats from instruciton generation
                #functionality and just expects some sort of stronger constant consideration for a smoother exploration component
                #old exploration component is very very clunky and NOTE INTERPRETS explored but failed space as unexplored
                #sooo this is my solution
                if(self._count_explore is True):
                    self._UCB_EXPLORE = np.sqrt(self._c * np.log(self._EXPLORE_T + 1) / (self._EXPLORE_COUNT + 1))
                else:
                    self._UCB_EXPLORE = np.sqrt(self._c * np.log(self._EXPLOIT_T + 1) / (self._EXPLOIT_COUNT + 1))
                self._UCBMAT = self._UCB_EXPLOIT + self._UCB_EXPLORE
                #consider that this may need redone 
                _tnorm = self._EXPLOIT_COUNT / np.clip(self._EXPLOIT_COUNT.sum(axis=1, keepdims=True), 1, None)
                self._UCB_CONF = self._UCB_EXPLOIT * (_tnorm ** self._temp)
                del _tnorm

            case "UCB1":

                # we will need to go get the actual transfomration function IDs
                # to associate each node to its max downset score
                family_tfidx = X._instructions[family_idx, 1].astype(int)

                # now we should go ahead and transform the p value 
                # to our fitness function
                # family_fitness = -np.sqrt(2 * family_scores) + 1
                # trying a new one, cubic [0, 1]
                # family_fitness = ((0.5 - family_scores) / 0.629961) ** 3 + 0.5
                family_fitness = self.pvalue_to_fitness(family_scores)
                
                local_tq1d = np.zeros(24, dtype=np.float32)
                local_tfrq = np.zeros(24, dtype=np.float32)

                match(self._node_fitness):
                    case "count_all":
                        for i in range(family_tfidx.shape[0]):
                            tf = family_tfidx[i]
                            local_tq1d[tf] += family_fitness[i]

                        val_tf, cnt_tf = np.unique(family_tfidx, return_counts=True)

                        for idx, tf in enumerate(val_tf):
                            local_tq1d[tf] /= cnt_tf[idx]
                            local_tfrq[tf] = cnt_tf[idx]

                    case "count_best":
                        val_tf = np.unique(family_tfidx)

                        for tf in val_tf:
                            mask = (family_tfidx == tf)
                            best_fitness = np.max(family_fitness[mask])
                            local_tq1d[tf] += best_fitness
                            local_tfrq[tf] = 1

                    case "count_worst":
                        val_tf = np.unique(family_tfidx)

                        for tf in val_tf:
                            mask = (family_tfidx == tf)
                            best_fitness = np.min(family_fitness[mask])
                            local_tq1d[tf] += best_fitness
                            local_tfrq[tf] = 1

                    case "count_pop_null" | "count_pop_dead":
                        fill_score = 0.5 if self._node_fitness == "count_pop_null" else 1.0
                        fill_fitness = float(np.clip(-np.log(fill_score) / (-np.log(0.05)), None, 1))

                        # family nodes keep their reduced/propagated fitness
                        fam_fit_map = {int(family_idx[i]): float(family_fitness[i]) for i in range(family_idx.shape[0])}

                        # use full gene space
                        pop_idx = np.asarray(X._G_idx, dtype=np.int64)

                        for idx in pop_idx:
                            idx = int(idx)
                            tf = int(X._instructions[idx, 1])

                            # use family fitness if available, otherwise fallback fitness
                            fit = fam_fit_map.get(idx, fill_fitness)

                            local_tq1d[tf] += fit
                            local_tfrq[tf] += 1

                        nz = local_tfrq > 0
                        local_tq1d[nz] /= local_tfrq[nz]

                    case _:
                        raise ValueError(f"unknown node fitness mode: {self._node_fitness}")

                local_tq1d[0] = 0
                local_tfrq[0] = 0

                self._t += np.sum(local_tfrq[1:])
                self._t_count += local_tfrq[1:]
                self._t_cum += local_tq1d[1:]
                self._t_mu = self._t_cum/(self._t_count + 1)
                self._UCB1 = self._t_mu + np.clip(np.sqrt(np.log(self._t + 1) / (self._t_count + 1)), None, 1)


            case "MCTS":

                family_idx = np.asarray(family_idx, dtype=np.int64)
                family_scores = np.asarray(family_scores, dtype=np.float64)

                keep = np.isfinite(family_scores)
                family_idx = family_idx[keep]
                family_scores = family_scores[keep]

                #settings
                #you can expose these as Grammar params later
                backup_gamma = 0.85
                min_backup_weight = 0.05
                include_terminal = False

                #local grouped reward containers
                node_reward_map = {}
                edge_reward_map = {}

                def _append_reward(dct, key, reward):
                    if key not in dct:
                        dct[key] = []
                    dct[key].append(float(reward))

                #------------------------------------------------------------
                # build grouped path-backup rewards
                #------------------------------------------------------------

                alpha_decision_reward_map = {}
                alpha_node_reward_map = {}
                alpha_edge_reward_map = {}

                for i in range(family_idx.shape[0]):

                    leaf_idx = int(family_idx[i])
                    leaf_score = float(family_scores[i])

                    node_updates, edge_updates, alpha_updates = self._mcts_backup_path_rewards(
                        X=X,
                        leaf_idx=leaf_idx,
                        leaf_score=leaf_score,
                        quality_fn=quality_fn,
                        backup_gamma=backup_gamma,
                        min_backup_weight=0.05,
                        include_terminal=False
                    )

                    for key, reward in node_updates:
                        _append_reward(node_reward_map, key, reward)

                    for key, reward in edge_updates:
                        _append_reward(edge_reward_map, key, reward)

                    for kind, key, reward in alpha_updates:

                        if kind == "decision":
                            _append_reward(alpha_decision_reward_map, key, reward)

                        elif kind == "alpha_node":
                            _append_reward(alpha_node_reward_map, key, reward)

                        elif kind == "alpha_edge":
                            _append_reward(alpha_edge_reward_map, key, reward)

                        else:
                            raise ValueError(f"unknown alpha update kind: {kind}")

                #------------------------------------------------------------
                # reduce grouped rewards per update call
                #
                # I recommend max for MCTS at first:
                # if a state appears multiple times in one family/path update,
                # let its best observed descendant define this iteration's reward.
                #------------------------------------------------------------

                backup_reduce = "max"

                def _reduce(vals):
                    vals = np.asarray(vals, dtype=np.float64)

                    match backup_reduce:
                        case "mean":
                            return float(np.mean(vals))
                        case "max":
                            return float(np.max(vals))
                        case "median":
                            return float(np.median(vals))
                        case _:
                            raise ValueError(f"unknown backup_reduce: {backup_reduce}")

                #------------------------------------------------------------
                # apply node updates
                #------------------------------------------------------------

                for node_key, vals in node_reward_map.items():

                    reward = _reduce(vals)

                    self._mcts_update_mean(
                        cum_dict=self._MCTS_NODE_CUM,
                        count_dict=self._MCTS_NODE_COUNT,
                        mu_dict=self._MCTS_NODE_MU,
                        key=node_key,
                        val=reward
                    )

                    self._MCTS_EXPLOIT_T += 1

                #------------------------------------------------------------
                # apply edge updates
                #------------------------------------------------------------

                for edge_key, vals in edge_reward_map.items():

                    reward = _reduce(vals)

                    self._mcts_update_mean(
                        cum_dict=self._MCTS_EDGE_CUM,
                        count_dict=self._MCTS_EDGE_COUNT,
                        mu_dict=self._MCTS_EDGE_MU,
                        key=edge_key,
                        val=reward
                    )

                    parent_key, child_tf = edge_key

                    if parent_key not in self._MCTS_CHILDREN:
                        self._MCTS_CHILDREN[parent_key] = set()

                    self._MCTS_CHILDREN[parent_key].add(int(child_tf))

                #------------------------------------------------------------
                # apply alpha decision updates
                #------------------------------------------------------------

                for decision_key, vals in alpha_decision_reward_map.items():

                    reward = _reduce(vals)

                    self._mcts_update_mean(
                        cum_dict=self._MCTS_ALPHA_DECISION_CUM,
                        count_dict=self._MCTS_ALPHA_DECISION_COUNT,
                        mu_dict=self._MCTS_ALPHA_DECISION_MU,
                        key=decision_key,
                        val=reward
                    )

                    self._MCTS_ALPHA_EXPLOIT_T += 1

                #------------------------------------------------------------
                # apply alpha parent-node updates
                #------------------------------------------------------------

                for alpha_node_key, vals in alpha_node_reward_map.items():

                    reward = _reduce(vals)

                    self._mcts_update_mean(
                        cum_dict=self._MCTS_ALPHA_NODE_CUM,
                        count_dict=self._MCTS_ALPHA_NODE_COUNT,
                        mu_dict=self._MCTS_ALPHA_NODE_MU,
                        key=alpha_node_key,
                        val=reward
                    )

                #------------------------------------------------------------
                # apply alpha edge updates
                #------------------------------------------------------------

                for alpha_edge_key, vals in alpha_edge_reward_map.items():

                    reward = _reduce(vals)

                    self._mcts_update_mean(
                        cum_dict=self._MCTS_ALPHA_EDGE_CUM,
                        count_dict=self._MCTS_ALPHA_EDGE_COUNT,
                        mu_dict=self._MCTS_ALPHA_EDGE_MU,
                        key=alpha_edge_key,
                        val=reward
                    )

                #------------------------------------------------------------
                # if exploration is not counted during generation,
                # mirror exploit counts into explore counts
                #------------------------------------------------------------

                if(self._count_explore is False):

                    for key, val in self._MCTS_NODE_COUNT.items():
                        self._MCTS_NODE_EXPLORE_COUNT[key] = val

                    for key, val in self._MCTS_EDGE_COUNT.items():
                        self._MCTS_EDGE_EXPLORE_COUNT[key] = val

                    self._MCTS_EXPLORE_T = self._MCTS_EXPLOIT_T

                    for key, val in self._MCTS_ALPHA_DECISION_COUNT.items():
                        self._MCTS_ALPHA_DECISION_EXPLORE_COUNT[key] = val

                    for key, val in self._MCTS_ALPHA_NODE_COUNT.items():
                        self._MCTS_ALPHA_NODE_EXPLORE_COUNT[key] = val

                    for key, val in self._MCTS_ALPHA_EDGE_COUNT.items():
                        self._MCTS_ALPHA_EDGE_EXPLORE_COUNT[key] = val

                    self._MCTS_ALPHA_EXPLORE_T = self._MCTS_ALPHA_EXPLOIT_T

            case "MCTS_OLD_OLD_OLD":

                #MCTS update is sparse.
                #instead of updating a dense 24x23 matrix, we update:
                #
                #   node_key -> fitness memory
                #   (parent_key, child_tf) -> fitness memory
                #
                #this is the same conceptual update as UCB1-tMAT, but now the
                #parent is not just parent_tf.
                #the parent is the full MCTS state key.

                family_idx = np.asarray(family_idx, dtype=np.int64)
                family_scores = np.asarray(family_scores, dtype=np.float64)

                # landing state for each family node
                family_tfidx = X._instructions[family_idx, 1].astype(int)

                # transform p-values to fitness
                # this is kept exactly conceptually aligned with your existing grammar update
                family_fitness = self.pvalue_to_fitness(family_scores)

                # discard family nodes that land in state 0
                # terminal states are source material, not generated action outcomes
                keep = (family_tfidx != 0) & np.isfinite(family_fitness)
                family_idx = family_idx[keep]
                family_tfidx = family_tfidx[keep]
                family_scores = family_scores[keep]
                family_fitness = family_fitness[keep]

                #local sparse update list:
                #we build idx_update and fit_update first so that all node_fitness modes
                #feed into one shared sparse update section below.
                idx_update = []
                fit_update = []

                match(self._node_fitness):

                    case "count_all":

                        #count every supplied family member.
                        #if the same state appears multiple times, it contributes multiple times.
                        for i in range(family_idx.shape[0]):
                            idx_update.append(int(family_idx[i]))
                            fit_update.append(float(family_fitness[i]))

                    case "count_best":

                        #keep only best observation per sparse child state key.
                        #this is the MCTS equivalent of grouping by tf id in UCB1.
                        seen = {}

                        for i in range(family_idx.shape[0]):
                            idx = int(family_idx[i])
                            key = self._mcts_state_key(X._instructions, idx)
                            val = float(family_fitness[i])

                            if key not in seen or val > seen[key][1]:
                                seen[key] = (idx, val)

                        for key, val in seen.items():
                            idx_update.append(int(val[0]))
                            fit_update.append(float(val[1]))

                    case "count_worst":

                        #keep only worst observation per sparse child state key.
                        seen = {}

                        for i in range(family_idx.shape[0]):
                            idx = int(family_idx[i])
                            key = self._mcts_state_key(X._instructions, idx)
                            val = float(family_fitness[i])

                            if key not in seen or val < seen[key][1]:
                                seen[key] = (idx, val)

                        for key, val in seen.items():
                            idx_update.append(int(val[0]))
                            fit_update.append(float(val[1]))

                    case "count_pop_null" | "count_pop_dead":

                        #same behavior as your existing UCB1/UCB1-tMAT logic.
                        #
                        #family nodes get their true transformed p-value fitness.
                        #all other generated population nodes get fallback fitness.
                        #
                        #count_pop_null:
                        #   fallback p = 0.5
                        #
                        #count_pop_dead:
                        #   fallback p = 1.0

                        fill_score = 0.5 if self._node_fitness == "count_pop_null" else 1.0
                        fill_fitness = float(np.clip(-np.log(fill_score) / (-np.log(0.05)), None, 1))

                        fam_fit_map = {
                            int(family_idx[i]): float(family_fitness[i])
                            for i in range(family_idx.shape[0])
                        }

                        pop_idx = np.asarray(X._G_idx, dtype=np.int64)

                        for idx in pop_idx:
                            idx = int(idx)

                            tf = int(X._instructions[idx, 1])

                            if tf == 0:
                                continue

                            fit = fam_fit_map.get(idx, fill_fitness)

                            idx_update.append(idx)
                            fit_update.append(float(fit))

                    case _:
                        raise ValueError(f"unknown node fitness mode: {self._node_fitness}")

                #------------------------------------------------------------
                # sparse memory update
                #------------------------------------------------------------

                for i in range(len(idx_update)):

                    idx = int(idx_update[i])
                    fit = float(fit_update[i])

                    child_tf = int(X._instructions[idx, 1])

                    if child_tf == 0:
                        continue

                    #recover sparse child state
                    child_key = self._mcts_state_key(X._instructions, idx)

                    #recover parent state from x offset
                    parent_idx = self._mcts_parent_abs_idx(X._instructions, idx)

                    if parent_idx < 0 or parent_idx >= X._instructions.shape[0]:
                        parent_key = ("ROOT",)
                    else:
                        parent_key = self._mcts_state_key(X._instructions, parent_idx)

                    edge_key = (parent_key, child_tf)

                    #update node exploit memory
                    self._mcts_update_mean(
                        cum_dict=self._MCTS_NODE_CUM,
                        count_dict=self._MCTS_NODE_COUNT,
                        mu_dict=self._MCTS_NODE_MU,
                        key=child_key,
                        val=fit
                    )

                    #update edge exploit memory
                    self._mcts_update_mean(
                        cum_dict=self._MCTS_EDGE_CUM,
                        count_dict=self._MCTS_EDGE_COUNT,
                        mu_dict=self._MCTS_EDGE_MU,
                        key=edge_key,
                        val=fit
                    )

                    #make sure the child is listed as opened from this parent
                    #this is necessary for progressive widening and inference.
                    if parent_key not in self._MCTS_CHILDREN:
                        self._MCTS_CHILDREN[parent_key] = set()

                    self._MCTS_CHILDREN[parent_key].add(child_tf)

                    self._MCTS_EXPLOIT_T += 1

                #if exploration is not counted during generation, then the explore
                #counts should be based on the exploit counts.
                #
                #this mirrors the structure of UCB1-tMAT:
                #   if count_explore is true:
                #       use generation exploration counts
                #   else:
                #       use update/exploit counts
                if(self._count_explore is False):

                    for key, val in self._MCTS_NODE_COUNT.items():
                        self._MCTS_NODE_EXPLORE_COUNT[key] = val

                    for key, val in self._MCTS_EDGE_COUNT.items():
                        self._MCTS_EDGE_EXPLORE_COUNT[key] = val

                    self._MCTS_EXPLORE_T = self._MCTS_EXPLOIT_T


            case 'tq1d':

                raise ValueError(f"Grammar type 'tq1d' is depricated.")
            
            case 'Null':
                pass

            case _:
                raise ValueError(f'In Grammar Update: Grammar of type ({self._type}) is not recongized.')

    def mode(
        self,
        mode    :   str
    ):
        self._mode = mode

    def set_mcts_pw_params(
        self,
        pw_c=None,
        pw_alpha=None,
        expand_prob=None,
        alpha_pw_c=None,
        alpha_pw_alpha=None,
        alpha_expand_prob=None,
    ):
        """
        directly update live MCTS progressive widening parameters.
        """

        if pw_c is not None:
            self._MCTS_PW_C = float(pw_c)

        if pw_alpha is not None:
            self._MCTS_PW_ALPHA = float(pw_alpha)

        if expand_prob is not None:
            self._MCTS_EXPAND_PROB = float(expand_prob)

        if alpha_pw_c is not None:
            self._MCTS_ALPHA_PW_C = float(alpha_pw_c)

        if alpha_pw_alpha is not None:
            self._MCTS_ALPHA_PW_ALPHA = float(alpha_pw_alpha)

        if alpha_expand_prob is not None:
            self._MCTS_ALPHA_EXPAND_PROB = float(alpha_expand_prob)
    

import numpy as np


class Population:
    '''
    All instructions containing all zeros suggest locations are terminal states
    '''
    def __init__(
        self,
        X_inst: np.ndarray,
        terminal_idx: np.ndarray | list,
        excluded_idx: np.ndarray | list,
        max_size    : int = 20000,
        chunk_size  : int | float = 0.2,
        include_time: bool = True,
        structure   : str = 'Intraday',
        market_only : bool = True,
        market_close: int = 390,
        wf_windows  : int = 1
    ):
        self._X_inst = X_inst
        self._T_idx = np.asarray(terminal_idx, dtype=np.int64)
        self._E_idx = np.asarray(excluded_idx, dtype=np.int64)
        self._G_idx = np.asarray([], dtype=np.int64)

        self._max_size = int(max_size)
        self._structure = structure
        self._time_terminals = include_time

        #time terminal column locations if created
        self._tod_idx = None
        self._dow_idx = None

        #walk-forward chunk metadata
        self._n_chunks = int(wf_windows)
        if self._n_chunks < 1:
            raise ValueError('wf_windows must be >= 1.')

        self._n_days = 0
        self._day_start_idx = np.asarray([], dtype=np.int64)
        self._day_stop_idx  = np.asarray([], dtype=np.int64)

        #chunk bounds are half-open [lo, hi)
        self._chunk_day_bounds = np.zeros((0, 2), dtype=np.int64)
        self._chunk_row_bounds = np.zeros((0, 2), dtype=np.int64)
        self._chunk_row_idx = []

        if chunk_size < 1:
            chunk_size = int(np.ceil(max_size * chunk_size))

        if chunk_size == 0:
            raise ValueError('Chunk Size in Population Cannot Be Zero.')

        self._chunk_size = int(chunk_size)

        #throw error if any intersection before any time columns are appended
        overlap = np.intersect1d(self._T_idx, self._E_idx, assume_unique=False)
        if overlap.size:
            raise ValueError(
                f'terminal_idx and excluded_idx overlap at indices: {overlap.tolist()}'
            )

        if include_time:
            #NOTE tentative must is that epoch time column is first in excluded idx array
            time_src_col = int(self._E_idx[0])
            next_idx = int(self._T_idx.max())

            self._tod_idx = next_idx + 1
            self._dow_idx = next_idx + 2

            if self._dow_idx >= self._max_size:
                raise ValueError(
                    f'Not enough population columns to append time terminals. '
                    f'Need indices through {self._dow_idx}, but max_size={self._max_size}.'
                )

            # next_idx+1 -> minutes relative to market open
            self._X_inst[:, self._tod_idx] = tod_minutes_from_dstaware(
                self._X_inst[:, time_src_col],
                mode='market_open'
            )

            # next_idx+2 -> day of week, DST-aware
            self._X_inst[:, self._dow_idx] = dow_sun0_from_epoch(
                self._X_inst[:, time_src_col]
            )

            self._T_idx = np.concatenate((self._T_idx, [self._tod_idx, self._dow_idx]))

            # optional regular-market-hours-only filter
            if market_only and structure == 'Intraday':
                tod_mkt = self._X_inst[:, self._tod_idx]

                #keep [0, market_close)
                keep_mask = (tod_mkt >= 0) & (tod_mkt < market_close)

                if not np.any(keep_mask):
                    raise ValueError(
                        'market_only=True produced zero kept rows. '
                        'Check the market-open time column and timestamp convention.'
                    )

                old_X = self._X_inst
                self._X_inst = old_X[keep_mask].copy()

                del old_X, keep_mask, tod_mkt

        #build chunk partitions after final row layout exists
        self._build_chunks()

        #allocating entire space of possible instructions
        self._instructions = np.zeros((self._max_size, 11), dtype=np.float32)

        #writing in gene indices for our terminal or excluded columns
        self._instructions[self._T_idx, 0] = self._T_idx

        self._L_idx = np.asarray(self._T_idx, dtype=np.int64)

    def _build_chunks(self):
        '''
        Build chunk boundaries for walk-forward use.

        If include_time=True and structure='Intraday', chunking is aligned to
        detected beginning-of-day rows.

        Otherwise chunking falls back to simple row partitioning.
        '''
        n_rows = int(self._X_inst.shape[0])

        if n_rows == 0:
            raise ValueError('Cannot build chunks on empty X_inst.')

        #day-aligned chunking for intraday time-aware data
        if self._time_terminals and self._structure == 'Intraday' and self._tod_idx is not None:
            tod = self._X_inst[:, self._tod_idx]
            dow = self._X_inst[:, self._dow_idx]

            #new day if market-open minutes reset backward or day-of-week changes
            new_day_mask = (np.diff(tod) < 0) | (np.diff(dow) != 0)

            self._day_start_idx = np.r_[0, 1 + np.flatnonzero(new_day_mask)].astype(np.int64)
            self._day_stop_idx  = np.r_[self._day_start_idx[1:], n_rows].astype(np.int64)
            self._n_days = int(self._day_start_idx.size)

            if self._n_chunks > self._n_days:
                raise ValueError(
                    f'wf_windows={self._n_chunks} exceeds detected day count={self._n_days}.'
                )

            day_groups = np.array_split(
                np.arange(self._n_days, dtype=np.int64),
                self._n_chunks
            )

            self._chunk_day_bounds = np.asarray(
                [[grp[0], grp[-1] + 1] for grp in day_groups],
                dtype=np.int64
            )

            self._chunk_row_bounds = np.asarray(
                [[self._day_start_idx[grp[0]], self._day_stop_idx[grp[-1]]] for grp in day_groups],
                dtype=np.int64
            )

        #fallback row chunking
        else:
            if self._n_chunks > n_rows:
                raise ValueError(
                    f'wf_windows={self._n_chunks} exceeds row count={n_rows}.'
                )

            edges = np.floor(
                np.linspace(0, n_rows, self._n_chunks + 1)
            ).astype(np.int64)
            edges[-1] = n_rows

            self._n_days = 1
            self._day_start_idx = np.asarray([0], dtype=np.int64)
            self._day_stop_idx  = np.asarray([n_rows], dtype=np.int64)

            self._chunk_day_bounds = np.asarray(
                [[i, i + 1] for i in range(self._n_chunks)],
                dtype=np.int64
            )

            self._chunk_row_bounds = np.column_stack(
                (edges[:-1], edges[1:])
            ).astype(np.int64)

        self._chunk_row_idx = [
            np.arange(lo, hi, dtype=np.int64)
            for lo, hi in self._chunk_row_bounds
        ]

    def get_chunk_slice(self, chunk_num: int | None = None) -> slice:
        '''
        Return half-open row slice for one chunk.
        If chunk_num is None, return full slice.
        '''
        if chunk_num is None:
            return slice(None)

        chunk_num = int(chunk_num)

        if chunk_num < 0 or chunk_num >= self._n_chunks:
            raise IndexError(
                f'chunk_num={chunk_num} out of bounds for _n_chunks={self._n_chunks}.'
            )

        lo, hi = self._chunk_row_bounds[chunk_num]
        return slice(int(lo), int(hi))

    def get_chunk_rows(self, chunk_num: int | None = None) -> np.ndarray:
        '''
        Return explicit row indices for one chunk.
        If chunk_num is None, return all row indices.
        '''
        if chunk_num is None:
            return np.arange(self._X_inst.shape[0], dtype=np.int64)

        chunk_num = int(chunk_num)

        if chunk_num < 0 or chunk_num >= self._n_chunks:
            raise IndexError(
                f'chunk_num={chunk_num} out of bounds for _n_chunks={self._n_chunks}.'
            )

        return self._chunk_row_idx[chunk_num]

    def get_chunk_bounds(self, chunk_num: int) -> tuple[int, int]:
        '''
        Return raw row bounds (lo, hi) for one chunk.
        '''
        chunk_num = int(chunk_num)

        if chunk_num < 0 or chunk_num >= self._n_chunks:
            raise IndexError(
                f'chunk_num={chunk_num} out of bounds for _n_chunks={self._n_chunks}.'
            )

        lo, hi = self._chunk_row_bounds[chunk_num]
        return int(lo), int(hi)



def F_IDS():
	'''
	FUNCTION IDS
	------------
	0.	TERMINAL INSTANTIATION - x
	1.	MAX	-	x, d
	2.	MIN	-	x, d
	3.	AVG	-	x, d
	4.	NEG	-	x
	5.	DIF	-	x, a
	6.	ADD	-	x, a
	7.	SQR	-	x
	8.	SIN	-	x
	9.	COS	-	x
	10.	ASN	-	x
	11.	ACS	-	x
	12. RNG	-	x, d, dd
	13. HKP	-	x, k
	14.	EMA	-	x, d
	15.	DOE	-	x, d, dd
	16.	MDN	-	x, d
	17.	ZSC	-	x, d
	18.	STD	-	x, d
	19.	SSN	-	x
	20.	AGR	-	x, a, d
	21.	COR	-	x, a, d
    22. AND -   x, a
    23. ORR -   x, a

	PARAMETER ORDER
	---------------
	[(F), x, a, d, dd, k]
	'''
	return

def F_WITH(
	param	:	str	=	'x'
):
	f_p = {
		0:['x'],#ID 0
		1:['x', 'd'],#ID 1
		2:['x', 'd'],#ID 2
		3:['x', 'd'],#ID 3
		4:['x'],#ID 4
		5:['x', 'a'],#ID 5
		6:['x', 'a'],#ID 6
		7:['x'],#ID 7
		8:['x'],#ID 8
		9:['x'],#ID 9
		10:['x'],#ID 10
		11:['x'],#ID 11
		12:['x', 'd', 'dd'],#ID 12
		13:['x', 'k'],#ID 13
		14:['x', 'd'],#ID 14
		15:['x', 'd', 'dd'],#ID 15
		16:['x', 'd'],#ID 16
		17:['x', 'd'],#ID 17
		18:['x', 'd'],#ID 18
		19:['x'],#ID 19
		20:['x', 'a', 'd'],#ID 20
		21:['x', 'a', 'd'],#ID 21
        22:['x', 'a'],#ID 22
        23:['x', 'a'],#ID 23
	}
	return [k for k, v in f_p.items() if param in v]

def F_AS(id: int):
	f_p = {
		0:['x'],#ID 0
		1:['x', 'd'],#ID 1
		2:['x', 'd'],#ID 2
		3:['x', 'd'],#ID 3
		4:['x'],#ID 4
		5:['x', 'a'],#ID 5
		6:['x', 'a'],#ID 6
		7:['x'],#ID 7
		8:['x'],#ID 8
		9:['x'],#ID 9
		10:['x'],#ID 10
		11:['x'],#ID 11
		12:['x', 'd', 'dd'],#ID 12
		13:['x', 'k'],#ID 13
		14:['x', 'd'],#ID 14
		15:['x', 'd', 'dd'],#ID 15
		16:['x', 'd'],#ID 16
		17:['x', 'd'],#ID 17
		18:['x', 'd'],#ID 18
		19:['x'],#ID 19
		20:['x', 'a', 'd'],#ID 20
		21:['x', 'a', 'd'],#ID 21
        22:['x', 'a'],#ID 22
        23:['x', 'a'],#ID 23
	}
	return f_p.get(id, [])


def V_to_LOC(vals):
	# Order: x, a, d, dd, k  ->  5, 6, 7, 8, 9
	m = {'x': 5, 'a': 6, 'd': 7, 'dd': 8, 'k': 9}
	return [m[v] for v in vals if v in m]


def VLOC_to_FLAG(vlocs):
	"""
	Takes a list like [6,7,8,9] (any subset, any order) and returns a single
	int bit-flag value, preserving the 5..9 location values by mapping:
	23->bit3, 4->bit4, 5->bit5, 6->bit6.
	"""
	flags = 0
	for loc in vlocs:
		if 5 <= loc <= 9:
			flags |= (1 << loc)
	return flags


def fill_sensor_UNIFORM(inst_inst, flags_u32, legal_idx, rng=None, inplace=True):
    """
    Fill *every FLAGGED location* in columns 5..9 with a random selection from `legal_idx`.

    - flags_u32 uses bits 5..9 to indicate whether that column is "flagged" (True => fill)
    - legal_idx: 1D array-like of allowed sensor indices to write into inst_inst
    - inst_inst is float32, so ints will be stored as float32 (fine for IDs)

    Returns the (possibly modified) inst_inst.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = inst_inst if inplace else inst_inst.copy()

    flags = np.asarray(flags_u32, dtype=np.uint32).reshape(-1)

    legal_idx = np.asarray(legal_idx, dtype=np.int32).reshape(-1)
    if legal_idx.size == 0:
        raise ValueError("legal_idx must contain at least one allowed index.")

    bits = np.arange(5, 10, dtype=np.uint32)  # [5,6,7,8,9]
    flagged_mask = ((flags[:, None] >> bits[None, :]) & np.uint32(1)).astype(bool)

    # sample FROM legal_idx for every cell in the (N,5) block, then only write where flagged
    pick = rng.integers(0, legal_idx.size, size=flagged_mask.shape, dtype=np.int32)
    r = legal_idx[pick].astype(np.float32, copy=False)  # shape (N,5)

    #per row base (instinst[:, 0]) broadcast to (N, 5)
    base = out[:, 0].astype(np.float32, copy=False)[:, None]

    #final vals written: base - generated
    vals = r - base #shape is (N, 5)

    block = out[:, 5:10]
    block[flagged_mask] = vals[flagged_mask].astype(out.dtype, copy=False)

    return out


import numpy as np


# bit positions for x, a, d, dd, k
BIT_X  = np.uint32(5)
BIT_A  = np.uint32(6)
BIT_D  = np.uint32(7)
BIT_DD = np.uint32(8)
BIT_K  = np.uint32(9)

MASK_X  = np.uint32(1) << BIT_X
MASK_A  = np.uint32(1) << BIT_A
MASK_D  = np.uint32(1) << BIT_D
MASK_DD = np.uint32(1) << BIT_DD
MASK_K  = np.uint32(1) << BIT_K

MASK_VARS_5_TO_9 = MASK_X | MASK_A | MASK_D | MASK_DD | MASK_K


def FUNC_to_USED_FLAGS(func_ids):
    """
    Map each function id -> used variable bits over positions 5..9
    corresponding to x, a, d, dd, k.
    """
    fids = np.asarray(func_ids, dtype=np.int32)

    max_id = int(fids.max()) if fids.size else 0
    max_id = max(max_id, 0)

    table = np.zeros(max_id + 1, dtype=np.uint32)
    for i in range(max_id + 1):
        vlocs = V_to_LOC(F_AS(i))
        table[i] = np.uint32(VLOC_to_FLAG(vlocs))

    out = np.zeros(fids.shape, dtype=np.uint32)
    in_range = (fids >= 0) & (fids <= max_id)
    out[in_range] = table[fids[in_range]]

    # safety: only keep x,a,d,dd,k bits
    out &= MASK_VARS_5_TO_9
    return out


def USED_to_SENSOR_FLAGS(used_flags, alpha_sensor_freq=0.0, rng=None):
    """
    Build sensor flags from used flags with the rule:

    - x is ALWAYS a sensor if used
    - a is a sensor with probability alpha_sensor_freq if used
    - d, dd, k are never sensors here

    Parameters
    ----------
    used_flags : array-like
        uint32 flags with variable bits in 5..9
    alpha_sensor_freq : float
        Probability that a used alpha becomes a sensor
    rng : np.random.Generator or None
        Optional RNG for reproducibility

    Returns
    -------
    sensor_flags : np.ndarray dtype=np.uint32
    """
    if not (0.0 <= alpha_sensor_freq <= 1.0):
        raise ValueError("alpha_sensor_freq must be between 0.0 and 1.0")

    used_u32 = np.asarray(used_flags, dtype=np.uint32)
    rng = np.random.default_rng() if rng is None else rng

    sensor = np.zeros(used_u32.shape, dtype=np.uint32)

    # x always sensor if used
    x_used = (used_u32 & MASK_X) != 0
    sensor[x_used] |= MASK_X

    # alpha sensor with probability alpha_sensor_freq, but only if alpha is used
    a_used = (used_u32 & MASK_A) != 0
    if np.any(a_used) and alpha_sensor_freq > 0.0:
        a_draw = rng.random(used_u32.shape) < alpha_sensor_freq
        a_sensor = a_used & a_draw
        sensor[a_sensor] |= MASK_A

    return sensor & MASK_VARS_5_TO_9


def USED_and_SENSOR_to_CONST_FLAGS(used_flags, sensor_flags):
    """
    Const flags are exactly the used variable bits that are not sensors.

    Guarantees:
        used = const | sensor
        const & sensor = 0
    """
    used_u32 = np.asarray(used_flags, dtype=np.uint32) & MASK_VARS_5_TO_9
    sensor_u32 = np.asarray(sensor_flags, dtype=np.uint32) & MASK_VARS_5_TO_9
    return used_u32 & ~sensor_u32


def FUNC_to_USED_FLAGS_old(func_ids):
    """
    Map each function-id -> required variable locations (cols 3..6) -> uint32 bit-flag.
    Uses the user-provided helpers: F_AS, V_to_LOC, VLOC_to_FLAG.

    Parameters
    ----------
    func_ids : array-like (int)
        Function IDs per row.

    Returns
    -------
    flags_u32 : np.ndarray dtype=np.uint32
        Same shape as func_ids. Bits 3..6 indicate presence of cols 3..6.
    """
    fids = np.asarray(func_ids, dtype=np.int32)

    # Build a lookup table for ids 0..max_id seen in func_ids (fast, tiny)
    max_id = int(fids.max()) if fids.size else 0
    max_id = max(max_id, 0)

    table = np.zeros(max_id + 1, dtype=np.uint32)
    for i in range(max_id + 1):
        vlocs = V_to_LOC(F_AS(i))          # e.g. ['a','d'] -> [3,4]
        table[i] = np.uint32(VLOC_to_FLAG(vlocs))  # [3,4] -> (1<<3)|(1<<4)

    # Default 0 for out-of-range / negative ids
    out = np.zeros(fids.shape, dtype=np.uint32)
    in_range = (fids >= 0) & (fids <= max_id)
    out[in_range] = table[fids[in_range]]
    return out



def FUNC_to_nonx_FLAGS(func_ids):
    """
    Same as FUNC_to_USED_FLAGS, but ALWAYS clears/ignores location 5
    (bit 5 is never set, even if V_to_LOC returns 5).

    Parameters
    ----------
    func_ids : array-like (int)
        Function IDs per row.

    Returns
    -------
    flags_u32 : np.ndarray dtype=np.uint32
        Same shape as func_ids. Bits 3..6 indicate presence of cols 3..6,
        except bit 5 is always 0.
    """
    fids = np.asarray(func_ids, dtype=np.int32)

    max_id = int(fids.max()) if fids.size else 0
    max_id = max(max_id, 0)

    table = np.zeros(max_id + 1, dtype=np.uint32)
    for i in range(max_id + 1):
        vlocs = V_to_LOC(F_AS(i))              # e.g. ['a','d'] -> [3,4,5]
        # remove 5 if present (ignore dd)
        if vlocs:
            vlocs = [loc for loc in vlocs if loc != 5]
        table[i] = np.uint32(VLOC_to_FLAG(vlocs))

    out = np.zeros(fids.shape, dtype=np.uint32)
    in_range = (fids >= 0) & (fids <= max_id)
    out[in_range] = table[fids[in_range]]

    # extra safety: force-clear bit 5 even if upstream helpers change
    out &= ~np.uint32(1 << 5)

    return out



def FLAGS_to_SENSOR_FLAGS(used_flags, nonx_flags) -> np.ndarray:
    """
    Compute USED_FLAGS minus nonx_FLAGS, but ONLY for bit positions 5..9 inclusive.
    All other bits are forced to 0 in the output.

    Output = (used & ~nonx) & mask_bits_5_to_9
    """
    used_u32 = np.asarray(used_flags).astype(np.uint32, copy=False)
    nonx_u32 = np.asarray(nonx_flags).astype(np.uint32, copy=False)

    mask_5_to_9 = np.uint32(0)
    for b in range(5, 10):  # 5..9
        mask_5_to_9 |= (np.uint32(1) << np.uint32(b))

    return (used_u32 & ~nonx_u32) & mask_5_to_9


def fill_const_ITS(arr: np.ndarray,
                   rows,
                   col: int,
                   rate: float = 1.0,
                   rng: np.random.Generator | None = None,
                   inplace: bool = True) -> np.ndarray:
    """
    Column-specialized ITS filler (same distribution as your original):
      draw u ~ Uniform(0,1], set x = -log(u)/rate (>= 0),
      write into arr[rows, col].

    rows can be either:
      - 1D boolean mask of length arr.shape[0]
      - 1D integer row indices
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Decide how many samples we actually need
    if rows.dtype == np.bool_:
        n = int(rows.sum())
        if n == 0:
            return out
        write_rows = rows
    else:
        write_rows = rows.astype(np.int64, copy=False)
        n = write_rows.size
        if n == 0:
            return out

    # U in (0,1], avoid log(0)
    u = rng.random(n, dtype=np.float64)
    u = np.clip(u, np.finfo(np.float64).tiny, 1.0)

    # Equivalent to your original neg_samples = (1/rate)*log(u) then flip sign
    out[write_rows, col] = (-np.log(u) / rate).astype(out.dtype, copy=False)
    return out


def fill_const_STEIS(n: int,
                     rows,
                     arr: np.ndarray,
                     col: int,
                     base_max: float = 2.0,
                     base_rate: float = 2.0,
                     rng: np.random.Generator | None = None,
                     inplace: bool = True) -> np.ndarray:
    """
    Column-specialized STEIS filler.
    Writes sampled ints in {1,...,n} into arr[rows, col].

    rows can be either:
      - 1D integer row indices
      - 1D boolean mask of length arr.shape[0]
    """
    if not (isinstance(n, (int, np.integer)) and n >= 1):
        raise ValueError("n must be an integer >= 1")
    if base_max <= 0:
        raise ValueError("base_max must be > 0")
    if base_rate <= 0:
        raise ValueError("base_rate must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Determine how many samples we actually need to generate
    if rows.dtype == np.bool_:
        m = int(rows.sum())
        if m == 0:
            return out
        write_rows = rows  # boolean mask used directly in assignment
    else:
        write_rows = rows.astype(np.int64, copy=False)
        m = write_rows.size
        if m == 0:
            return out

    # U in (0,1) (avoid endpoints)
    u = rng.random(m, dtype=np.float64)
    u = np.clip(u, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)

    # Truncated exponential on (0, base_max]
    z = 1.0 - np.exp(-base_rate * base_max)
    y = -(1.0 / base_rate) * np.log(1.0 - u * z)

    # Stretch to (0, n]
    x = y * (n / base_max)

    # Round up to int in [1, n]
    k = np.ceil(x).astype(np.int64, copy=False)
    k = np.clip(k, 1, n)

    out[write_rows, col] = k.astype(out.dtype, copy=False)
    return out


def fill_const_UA0(arr: np.ndarray,
                   rows,
                   col: int,
                   rng: np.random.Generator | None = None,
                   inplace: bool = True) -> np.ndarray:
    """
    Fill arr[rows, col] with Uniform(-1, 1) float32 samples.

    rows can be either:
      - 1D integer row indices (e.g., np.array([1,3,7]))
      - 1D boolean mask of length arr.shape[0]
    """
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Case A: boolean mask
    if rows.dtype == np.bool_:
        n = int(rows.sum())
        if n == 0:
            return out
        out[rows, col] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32, copy=False)
        return out

    # Case B: integer indices
    rows = rows.astype(np.int64, copy=False)
    n = rows.size
    if n == 0:
        return out

    out[rows, col] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32, copy=False)
    return out

def presence_to_proportion(
    presence    :   np.ndarray
):
    return np.bincount(presence, minlength=24) / presence.size


def generate_instructions(
    pop_prior       : Population,
    grm_prior       : Grammar,
    seed            : int   = None,
    verbose         : bool  = False,
):
    '''
    early dev notes
    - x_inst coming into this contains the terminal states
    - pop_prior is the population we are referencing going into generation
    - grm_prior is the grammar we are referencing going into generation
    - n         is the number of genes that are to be generated
    - chunk_gen is the size of each chunk that will be generated.
                 allows for grammar updating within generation, will be slower.

    INSTRUCTIONS FORMAT
    [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k, unused]

    added behavior
    - before normal random generation, attempt mutation / crossover on existing genes
    - originals are preserved
    - if a selected gene has descendants, descendants are duplicated too
    - _T_idx and _E_idx are never overwritten
    - do not generate past pop_prior._max_size
    '''

    # -----------------------------
    # basic capacity / guards
    # -----------------------------
    n_total = int(pop_prior._max_size - pop_prior._L_idx.size - pop_prior._E_idx.size)
    if n_total <= 0:
        return

    if not (0.0 <= grm_prior._p_mutation <= 1.0):
        raise ValueError('p_mutation must be in [0, 1].')
    if not (0.0 <= grm_prior._p_crossover <= 1.0):
        raise ValueError('p_crossover must be in [0, 1].')

    match(grm_prior._type):
        case 'Null':
            pass
        case 'tq1d':
            pass
        case 'UCB1':
            pass
        case 'UCB1-tMAT':
            pass
        case 'MCTS':
            pass
        case _:
            raise ValueError('Cannot interpret grammar prior in generate_instructions. Illegal type.')

    rng = np.random.default_rng(seed)

    # bits / cols for x,a,d,dd,k
    mask_5_9 = np.uint32(0)
    for b in range(5, 10):
        mask_5_9 |= (np.uint32(1) << np.uint32(b))

    op_cols = np.arange(5, 10, dtype=np.int64)

    # constant fill behavior currently used in Null generation
    const_fill_kind = {
        6: 'UA0',
        7: 'STEIS',
        8: 'STEIS',
        9: 'ITS',
    }

    # snapshot existing state BEFORE generation
    base_G_idx = np.asarray(pop_prior._G_idx, dtype=np.int32).copy()
    base_L_idx = np.asarray(pop_prior._L_idx, dtype=np.int32).copy()

    if base_L_idx.size == 0:
        current_max_pop_idx = 0
    else:
        current_max_pop_idx = int(base_L_idx.max())

    # row lookup by pop_idx from existing instructions
    instr = pop_prior._instructions
    col0_i32 = instr[:, 0].astype(np.int32, copy=False)

    pop_to_rowloc = {}
    for r, pid in enumerate(col0_i32):
        if pid != 0:
            pop_to_rowloc[int(pid)] = r

    base_rows = {}
    for pid in base_G_idx:
        if int(pid) in pop_to_rowloc:
            base_rows[int(pid)] = instr[pop_to_rowloc[int(pid)]].copy()

    base_gene_set = set(int(x) for x in base_G_idx)
    base_legal_set = set(int(x) for x in base_L_idx)

    # -----------------------------
    # helpers
    # -----------------------------
    def _u32_flag_has(flag_u32, col_idx: int) -> bool:
        return bool(np.uint32(flag_u32) & (np.uint32(1) << np.uint32(col_idx)))

    def _is_neg_int_ref(v) -> bool:
        if not np.isfinite(v):
            return False
        if v >= 0:
            return False
        iv = int(np.rint(v))
        return abs(float(v) - float(iv)) < 1e-6

    def _row_used_flags(row) -> np.uint32:
        return np.uint32(int(np.rint(row[2])))

    def _row_const_flags(row) -> np.uint32:
        return np.uint32(int(np.rint(row[3])))

    def _row_sensor_flags(row) -> np.uint32:
        return np.uint32(int(np.rint(row[4])))

    def _ref_cols_from_row(row):
        used = _row_used_flags(row)
        const = _row_const_flags(row)
        ref_flags = (used & ~const) & mask_5_9
        cols = []
        for c in op_cols:
            if _u32_flag_has(ref_flags, int(c)):
                cols.append(int(c))
        return cols

    def _const_cols_from_row(row):
        used = _row_used_flags(row)
        const = _row_const_flags(row)
        const_flags = (used & const) & mask_5_9
        cols = []
        for c in op_cols:
            if _u32_flag_has(const_flags, int(c)):
                cols.append(int(c))
        return cols

    def _fresh_const_inplace(row, col_idx: int):
        tmp = row.reshape(1, -1).copy()
        mask = np.array([True], dtype=bool)

        kind = const_fill_kind.get(int(col_idx), None)
        if kind is None:
            # current Null setup does not define const generation for x / future slots
            # keep current value if already present, else zero
            return tmp[0]

        if kind == 'UA0':
            tmp = fill_const_UA0(tmp, mask, int(col_idx), rng)
        elif kind == 'STEIS':
            tmp = fill_const_STEIS(grm_prior._mdl, mask, tmp, int(col_idx), 2.0, 2.0, rng)
        elif kind == 'ITS':
            tmp = fill_const_ITS(tmp, mask, int(col_idx), 1.0, rng)

        return tmp[0]

    def _sample_abs_target(new_pop_idx: int, exclude_abs=None):
        # all existing legal indices are < any newly appended gene, so they are valid left references
        candidates = base_L_idx
        if candidates.size == 0:
            return None

        if exclude_abs is not None and candidates.size > 1:
            candidates = candidates[candidates != int(exclude_abs)]
            if candidates.size == 0:
                candidates = base_L_idx

        return int(candidates[rng.integers(0, candidates.size)])

    def _relink_row_refs_inplace(new_row, old_row, remap_abs):
        old_pop = int(np.rint(old_row[0]))
        new_pop = int(np.rint(new_row[0]))

        for c in _ref_cols_from_row(old_row):
            v = old_row[c]
            if not _is_neg_int_ref(v):
                continue

            old_target_abs = old_pop + int(np.rint(v))
            new_target_abs = remap_abs.get(old_target_abs, old_target_abs)
            new_row[c] = np.float32(new_target_abs - new_pop)

    def _copy_single_gene_row(old_pop_idx: int, new_pop_idx: int, remap_abs=None):
        if remap_abs is None:
            remap_abs = {}

        old_row = base_rows[int(old_pop_idx)]
        new_row = old_row.copy()
        new_row[0] = np.float32(new_pop_idx)
        _relink_row_refs_inplace(new_row, old_row, remap_abs)
        return new_row

    def _mutate_root_row_inplace(root_new_row, root_old_row):
        old_pop = int(np.rint(root_old_row[0]))
        new_pop = int(np.rint(root_new_row[0]))

        old_ref_cols = _ref_cols_from_row(root_old_row)
        old_const_cols = _const_cols_from_row(root_old_row)

        mutation_modes = ['fid']
        if len(old_ref_cols) > 0:
            mutation_modes.append('ref')
        if len(old_const_cols) > 0:
            mutation_modes.append('const')

        mode = mutation_modes[rng.integers(0, len(mutation_modes))]

        # -----------------
        # mutate function id
        # -----------------
        if mode == 'fid':
            old_fid = int(np.rint(root_old_row[1]))
            new_fid = old_fid
            while new_fid == old_fid:
                new_fid = int(rng.integers(1, 22))

            used_flag = FUNC_to_USED_FLAGS(np.array([new_fid], dtype=np.int32))[0]
            const_flag = FUNC_to_nonx_FLAGS(np.array([new_fid], dtype=np.int32))[0]
            sensor_flag = FLAGS_to_SENSOR_FLAGS(
                np.array([used_flag], dtype=np.uint32),
                np.array([const_flag], dtype=np.uint32)
            )[0]

            root_new_row[1] = np.float32(new_fid)
            root_new_row[2] = np.float32(used_flag)
            root_new_row[3] = np.float32(const_flag)
            root_new_row[4] = np.float32(sensor_flag)

            # clear operand area and rebuild
            root_new_row[5:10] = 0.0

            for c in op_cols:
                c = int(c)
                bit_on = _u32_flag_has(np.uint32(used_flag), c)
                if not bit_on:
                    continue

                is_const = _u32_flag_has(np.uint32(const_flag), c)

                if is_const:
                    # keep prior constant if same slot was constant before, else fresh
                    if c in old_const_cols:
                        root_new_row[c] = root_old_row[c]
                    else:
                        root_new_row[:] = _fresh_const_inplace(root_new_row, c)

                else:
                    target_abs = None

                    # if same slot previously referenced something, try to preserve that absolute target
                    if c in old_ref_cols and _is_neg_int_ref(root_old_row[c]):
                        target_abs = old_pop + int(np.rint(root_old_row[c]))

                    # otherwise sample a fresh left target
                    if target_abs is None:
                        target_abs = _sample_abs_target(new_pop)

                    if target_abs is not None:
                        root_new_row[c] = np.float32(target_abs - new_pop)

            return

        # -----------------
        # mutate one reference
        # -----------------
        if mode == 'ref':
            ref_cols = _ref_cols_from_row(root_new_row)
            if len(ref_cols) == 0:
                return

            c = int(ref_cols[rng.integers(0, len(ref_cols))])

            cur_target_abs = None
            if _is_neg_int_ref(root_new_row[c]):
                cur_target_abs = new_pop + int(np.rint(root_new_row[c]))

            new_target_abs = _sample_abs_target(new_pop, exclude_abs=cur_target_abs)
            if new_target_abs is not None:
                root_new_row[c] = np.float32(new_target_abs - new_pop)
            return

        # -----------------
        # mutate one constant
        # -----------------
        if mode == 'const':
            const_cols = _const_cols_from_row(root_new_row)
            if len(const_cols) == 0:
                return

            c = int(const_cols[rng.integers(0, len(const_cols))])
            root_new_row[:] = _fresh_const_inplace(root_new_row, c)
            return

    def _build_children_map():
        children = {int(g): [] for g in base_G_idx}

        for child_pop in base_G_idx:
            child_pop = int(child_pop)
            row = base_rows.get(child_pop, None)
            if row is None:
                continue

            for c in _ref_cols_from_row(row):
                v = row[c]
                if not _is_neg_int_ref(v):
                    continue

                parent_abs = child_pop + int(np.rint(v))
                if parent_abs in base_gene_set:
                    children[parent_abs].append(child_pop)

        return children

    def _descendant_closure(root_pop_idx: int, children_map):
        root_pop_idx = int(root_pop_idx)
        seen = set()
        stack = [root_pop_idx]

        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)

            for ch in children_map.get(cur, []):
                if ch not in seen:
                    stack.append(ch)

        return np.array(sorted(seen), dtype=np.int32)

    def _build_mutation_event(root_pop_idx: int, start_new_idx: int, children_map):
        closure = _descendant_closure(root_pop_idx, children_map)
        if closure.size == 0:
            return [], start_new_idx

        # assign new pop indices for copied closure
        remap_abs = {}
        next_idx = int(start_new_idx)
        for old_pop in closure:
            remap_abs[int(old_pop)] = next_idx
            next_idx += 1

        new_rows = []
        rows_by_old_pop = {}

        # copy rows
        for old_pop in closure:
            old_pop = int(old_pop)
            new_pop = remap_abs[old_pop]
            new_row = _copy_single_gene_row(old_pop, new_pop, remap_abs=remap_abs)
            new_rows.append(new_row)
            rows_by_old_pop[old_pop] = new_row

        # mutate copied root only
        _mutate_root_row_inplace(rows_by_old_pop[int(root_pop_idx)], base_rows[int(root_pop_idx)])

        return new_rows, next_idx

    def _build_crossover_event(root_pop_idx: int, start_new_idx: int, children_map):
        root_pop_idx = int(root_pop_idx)
        root_old_row = base_rows.get(root_pop_idx, None)
        if root_old_row is None:
            return [], start_new_idx

        root_ref_cols = _ref_cols_from_row(root_old_row)
        if len(root_ref_cols) == 0:
            return [], start_new_idx

        closure = _descendant_closure(root_pop_idx, children_map)
        closure_set = set(int(x) for x in closure)

        # prefer donor genes outside closure, else any legal idx outside closure
        donor_gene_candidates = np.array(
            [g for g in base_G_idx if int(g) not in closure_set],
            dtype=np.int32
        )
        donor_legal_candidates = np.array(
            [l for l in base_L_idx if int(l) not in closure_set],
            dtype=np.int32
        )

        donor_abs = None
        donor_needs_copy = False

        if donor_gene_candidates.size > 0:
            donor_abs = int(donor_gene_candidates[rng.integers(0, donor_gene_candidates.size)])
            donor_needs_copy = True
        elif donor_legal_candidates.size > 0:
            donor_abs = int(donor_legal_candidates[rng.integers(0, donor_legal_candidates.size)])
            donor_needs_copy = donor_abs in base_gene_set
        else:
            return [], start_new_idx

        next_idx = int(start_new_idx)
        new_rows = []

        # optional donor root copy first, so recipient can point to it
        donor_new_abs = donor_abs
        if donor_needs_copy and donor_abs in base_gene_set:
            donor_new_abs = next_idx
            donor_row = _copy_single_gene_row(donor_abs, donor_new_abs, remap_abs={})
            new_rows.append(donor_row)
            next_idx += 1

        # copy recipient closure
        remap_abs = {}
        for old_pop in closure:
            remap_abs[int(old_pop)] = next_idx
            next_idx += 1

        rows_by_old_pop = {}
        for old_pop in closure:
            old_pop = int(old_pop)
            new_pop = remap_abs[old_pop]
            new_row = _copy_single_gene_row(old_pop, new_pop, remap_abs=remap_abs)
            new_rows.append(new_row)
            rows_by_old_pop[old_pop] = new_row

        # graft donor subtree root into one reference slot of copied root
        root_new_row = rows_by_old_pop[root_pop_idx]
        root_new_pop = int(np.rint(root_new_row[0]))

        graft_col = int(root_ref_cols[rng.integers(0, len(root_ref_cols))])
        root_new_row[graft_col] = np.float32(donor_new_abs - root_new_pop)

        return new_rows, next_idx

    def _append_rows_to_population(rows):
        if rows is None or len(rows) == 0:
            return

        rows_arr = np.asarray(rows, dtype=np.float32)
        n_rows = rows_arr.shape[0]

        col0 = pop_prior._instructions[:, 0].astype(np.int32, copy=False)
        empty = (col0 == 0)
        empty[0] = False

        empty_locs = np.flatnonzero(empty)
        if empty_locs.size == 0:
            raise ValueError('No empty instruction rows remain for append.')

        break_idx = int(empty_locs[0])

        pop_prior._instructions[break_idx:break_idx+n_rows, :] = rows_arr

        new_gene_idx = rows_arr[:, 0].astype(np.int32, copy=False)
        pop_prior._G_idx = np.union1d(pop_prior._G_idx, new_gene_idx)
        pop_prior._L_idx = np.union1d(pop_prior._L_idx, pop_prior._G_idx)

    # -----------------------------
    # build descendant relationships
    # -----------------------------
    children_map = _build_children_map()

    # -----------------------------
    # decide mutation / crossover events
    # each gene gets its own independent Bernoulli for each
    # -----------------------------
    event_specs = []
    for g in base_G_idx:
        g = int(g)
        if rng.random() < grm_prior._p_mutation:
            event_specs.append(('mutation', g))
        if rng.random() < grm_prior._p_crossover:
            event_specs.append(('crossover', g))

    # randomize event order so capacity truncation is less biased
    if len(event_specs) > 1:
        rng.shuffle(event_specs)

    # -----------------------------
    # build pre-generation offspring
    # -----------------------------
    pre_rows = []
    next_pop_idx = current_max_pop_idx + 1

    for ev_kind, root_pop in event_specs:
        remaining_capacity = n_total - len(pre_rows)
        if remaining_capacity <= 0:
            break

        if ev_kind == 'mutation':
            candidate_rows, candidate_next = _build_mutation_event(root_pop, next_pop_idx, children_map)
        elif ev_kind == 'crossover':
            candidate_rows, candidate_next = _build_crossover_event(root_pop, next_pop_idx, children_map)
        else:
            candidate_rows, candidate_next = [], next_pop_idx

        if len(candidate_rows) == 0:
            continue

        # never exceed allowed new count
        if len(candidate_rows) > remaining_capacity:
            continue

        pre_rows.extend(candidate_rows)
        next_pop_idx = candidate_next

    # append mutation / crossover offspring before random generation
    if len(pre_rows) > 0:
        _append_rows_to_population(pre_rows)

    # -----------------------------
    # remaining capacity -> original random generation path
    # -----------------------------
    gen_size = int(n_total - len(pre_rows))
    if gen_size <= 0:
        return

    chunk_size = int(pop_prior._chunk_size)
    if chunk_size == 0:
        chunk_size = gen_size

    while gen_size > 0:

        if gen_size < chunk_size:
            chunk_size = gen_size

        match(grm_prior._type):

            case 'MCTS':

                # understand that in this case we may consider chunk_size to be 1.
                # consider building it this way if that is easier.
                # if it is more than 1, then for each generation addeed it will be sampling
                # randomly across evaluations for new nodes to generate, this is stochastic.

                pass

                            #MCTS is meant to build one node at a time.
                #even though the outer generate_instructions loop supports chunked generation,
                #the cleanest first implementation is chunk_size = 1 here.
                #
                #reason:
                #if chunk_size > 1, then all nodes in the chunk are generated before appending.
                #that means node g + 1 cannot choose node g as a parent inside the same chunk.
                #that breaks the "build one step at a time" interpretation.
                #
                #so for MCTS we force the chunk to one new instruction row.
                chunk_size = 1

                # keep shape (chunk_size, 11); last col unused by design
                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                # populate new pop index after everything currently legal
                start_idx = int(pop_prior._L_idx.max())
                inst_inst[:, 0] = np.arange(start_idx + 1, start_idx + 1 + chunk_size, dtype=np.uint16)

                #------------------------------------------------------------
                # STEP 1:
                # WHERE selection
                #
                # select the x parent from all currently legal prior nodes.
                # this is UCT over existing MCTS node states.
                #------------------------------------------------------------

                parent_abs_idx, parent_key = grm_prior._mcts_select_parent(
                    rng=rng,
                    instructions=pop_prior._instructions,
                    legal_idx=pop_prior._L_idx
                )

                #------------------------------------------------------------
                # STEP 2:
                # HOW selection
                #
                # select the child transformation function from this parent.
                # this uses progressive widening first, then UCB + softmax.
                #------------------------------------------------------------

                child_tf = grm_prior._mcts_select_child_tf(
                    rng=rng,
                    parent_key=parent_key
                )

                if child_tf is None:
                    raise RuntimeError(
                        "MCTS selected a parent with no opened child_tf while freeze_expansion=True. "
                        "This should usually be prevented by _mcts_select_parent()."
                    )

                inst_inst[:, 1] = child_tf

                #------------------------------------------------------------
                # STEP 2.5:
                # alpha decision context
                #
                # ctx_key is the MCTS context for alpha selection.
                # It says:
                #   given this selected x-parent and selected child tf,
                #   should alpha be constant or sensor?
                #------------------------------------------------------------

                alpha_ctx_key = grm_prior._mcts_alpha_ctx_key(
                    parent_key=parent_key,
                    child_tf=child_tf
                )

                alpha_action = 0
                alpha_parent_abs_idx = None
                alpha_parent_key = None

                #------------------------------------------------------------
                # STEP 3:
                # count generation-time exploration if requested.
                #
                # this is analogous to your UCB1-tMAT count_explore behavior.
                # this counts the attempted edge even before we know if it survives.
                #------------------------------------------------------------

                if(grm_prior._mode == 'train' and grm_prior._count_explore is True):
                    grm_prior._mcts_count_generation(
                        parent_key=parent_key,
                        child_tf=child_tf
                    )

                #------------------------------------------------------------
                # STEP 4:
                # standard flag logic plus MCTS alpha logic.
                #
                # x and tf are already selected by MCTS.
                #
                # alpha now has its own decision:
                #   action 0 = alpha constant
                #   action 1 = alpha sensor
                #------------------------------------------------------------

                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                #used flags
                used_flags = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = used_flags

                #start with alpha forced constant.
                #we will manually turn alpha into a sensor if MCTS selects that.
                sensor_flags = USED_to_SENSOR_FLAGS(
                    used_flags,
                    alpha_sensor_freq=0.0,
                    rng=rng
                )

                alpha_bit = np.uint32(1) << np.uint32(6)
                alpha_used = (used_flags[0].astype(np.uint32) & alpha_bit) != 0

                if alpha_used:

                    alpha_action = grm_prior._mcts_select_alpha_is_sensor(
                        rng=rng,
                        ctx_key=alpha_ctx_key
                    )

                    if alpha_action == 1:

                        #conservative depth rule:
                        #new gene depth = 1 + max(depth(x_parent), depth(alpha_parent))
                        #therefore alpha parent must have depth <= max_depth - 1.
                        max_alpha_parent_depth = int(grm_prior._MCTS_MAX_DEPTH) - 1

                        alpha_parent_abs_idx, alpha_parent_key = grm_prior._mcts_select_alpha_parent(
                            rng=rng,
                            instructions=pop_prior._instructions,
                            legal_idx=pop_prior._L_idx,
                            ctx_key=alpha_ctx_key,
                            max_allowed_parent_depth=max_alpha_parent_depth
                        )

                        #if no legal alpha parent exists, fall back to constant alpha.
                        #this keeps generation safe near max-depth boundaries.
                        if alpha_parent_abs_idx is None:
                            alpha_action = 0
                            alpha_parent_key = None
                        else:
                            #turn alpha into a sensor
                            sensor_flags[0] = np.uint32(sensor_flags[0]) | alpha_bit

                inst_inst[:, 4] = sensor_flags

                #const flags:
                #everything used that is not a sensor becomes constant.
                const_flags = USED_and_SENSOR_to_CONST_FLAGS(used_flags, sensor_flags)
                inst_inst[:, 3] = const_flags


                #------------------------------------------------------------
                # STEP 5:
                # fill constants exactly like your other grammar cases.
                #
                # a  -> UA0
                # d  -> STEIS
                # dd -> STEIS
                # k  -> ITS
                #------------------------------------------------------------

                v_cols = {6: 'UA0', 7: 'STEIS', 8: 'STEIS', 9: 'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, kind in v_cols.items():
                    const_mask = (const_flags & (np.uint32(1) << np.uint32(c))) != 0
                    if kind == 'UA0':
                        inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                    elif kind == 'STEIS':
                        inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                    elif kind == 'ITS':
                        inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                #------------------------------------------------------------
                # STEP 6:
                # fill all sensor references randomly first.
                #
                # This matches your existing style.
                # Then we overwrite x with the actual MCTS-selected parent.
                #------------------------------------------------------------

                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                #------------------------------------------------------------
                # STEP 7:
                # overwrite x sensor with the MCTS-selected parent.
                #
                # Your project stores x as an offset:
                #   x = parent_abs_idx - current_abs_idx
                #
                # For a valid parent, this should usually be negative.
                #------------------------------------------------------------

                inst_inst[:, 5] = int(parent_abs_idx) - inst_inst[:, 0]

                #------------------------------------------------------------
                # STEP 7.5:
                # overwrite alpha sensor if alpha was selected as a sensor.
                #
                # Like x, alpha stores an offset:
                #   alpha_offset = alpha_parent_abs_idx - current_abs_idx
                #------------------------------------------------------------

                if alpha_action == 1 and alpha_parent_abs_idx is not None:
                    inst_inst[:, 6] = int(alpha_parent_abs_idx) - inst_inst[:, 0]

                # generation-time exploration counts
                if(grm_prior._mode == 'train' and grm_prior._count_explore is True):

                    grm_prior._mcts_count_generation(
                        parent_key=parent_key,
                        child_tf=child_tf
                    )

                    if alpha_used:
                        grm_prior._mcts_count_alpha_generation(
                            ctx_key=alpha_ctx_key,
                            action=alpha_action,
                            alpha_parent_key=alpha_parent_key
                        )
                        
                #------------------------------------------------------------
                # STEP 8:
                # keep a trace of what MCTS generated.
                #
                # This is optional for execution, but very useful for
                # debugging max depth and visualizing tree construction.
                #------------------------------------------------------------

                # trace generated MCTS row
                if not (grm_prior._mode == "infer" and getattr(grm_prior, "_static_infer", False)):

                    parent_depth = grm_prior._mcts_row_depth(
                        pop_prior._instructions,
                        parent_abs_idx
                    )

                    if alpha_action == 1 and alpha_parent_abs_idx is not None:
                        alpha_depth = grm_prior._mcts_row_depth(
                            pop_prior._instructions,
                            alpha_parent_abs_idx
                        )
                    else:
                        alpha_depth = None

                    child_depth = parent_depth + 1

                    if alpha_depth is not None:
                        child_depth = max(parent_depth, alpha_depth) + 1

                    grm_prior._MCTS_TRACE.append({
                        "new_idx"          : int(inst_inst[0, 0]),
                        "parent_idx"       : int(parent_abs_idx),
                        "parent_key"       : parent_key,
                        "parent_depth"     : int(parent_depth),
                        "child_depth"      : int(child_depth),
                        "child_tf"         : int(child_tf),
                        "x_offset"         : int(inst_inst[0, 5]),

                        "alpha_used"       : bool(alpha_used),
                        "alpha_action"     : int(alpha_action),
                        "alpha_is_sensor"  : bool(alpha_action == 1),
                        "alpha_parent_idx" : None if alpha_parent_abs_idx is None else int(alpha_parent_abs_idx),
                        "alpha_parent_key" : alpha_parent_key,
                        "alpha_depth"      : alpha_depth,
                        "alpha_offset"     : None if alpha_action == 0 else int(inst_inst[0, 6]),
                    })

            case 'UCB1-tMAT':

                #SAMPLING
                # we will take UCBMAT and resolve 
                # a score for selecting parent nodes for x of new gene
                # this vector is s = np.sum(UCBMAT, axis=1?0???) (length tf)
                # then we will get the existing state multiset
                #  which should be all transitions in instructions (_L_idx)
                # then we will make a proportion vector p (length tf)
                # out of the multiset of existing states
                # then we will sample parent idx with softmax(sp)
                # switched away from softmax(sp) doesnt make sense of proportion
                # logical way is actually softmax(s + log(p))                

                # keep shape (chunk_size, 11); last col unused by design
                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)


                # populate new pop indices after everything currently legal
                start_idx = int(pop_prior._L_idx.max())
                inst_inst[:, 0] = np.arange(start_idx + 1, start_idx + 1 + chunk_size, dtype=np.uint16)

                #now we need to define what our existing state multiset
                pres_multiset = pop_prior._instructions[pop_prior._L_idx, 1].astype(np.int32)

                #print(np.bincount(pres_multiset, minlength=24))
                #now we have a proportion vector length 24 (functions (23) + 1 (terminals))
                p = np.bincount(pres_multiset, minlength=24) / pres_multiset.size

                #now we have to make the score vector of length (tf (23) + terminal (1))
                match(grm_prior._mode):
                    case 'train':
                        #print('sampling from UCB MAT')
                        #print('sampling from UCB MAT')
                        s = np.sum(grm_prior._UCBMAT, axis=1)
                    case 'infer':
                        #print('sampling from UCB EXPLOIT')
                        #print('sampling from UCB EXPLOIT')
                        s = np.sum(grm_prior._UCB_CONF, axis=1)

                #now we need to make a state prbabilistic selection space with s and p
                #looks like the most principled approach is adding proportion from log space
                parent_prob = s + np.log(p + 1e-12)

                valid_parent_mask = p > 0

                #print('s vector   : ', s)
                #print('p vector   : ', p)
                #print('parent prob: ', parent_prob)

                #now we need to sample states from this space, actually on a roll right now
                #caught a case: YES IT DOES SAMPLE [0, tf] HERE!!!!
                parent_states = grm_prior.softmax_sample_uint16(rng, chunk_size, parent_prob, samp0=True, valid_mask=valid_parent_mask)

                psens_Lidx = np.empty(parent_states.shape[0], dtype=int)

                for i, v in enumerate(parent_states):
                    matches = np.flatnonzero(pres_multiset == v)

                    if matches.size == 0:
                        print("missing parent state:", v)
                        print("unique parent_states:", np.unique(parent_states))
                        print("unique pres_multiset:", np.unique(pres_multiset))
                        print("pres_multiset size:", pres_multiset.size)

                        # fallback
                        psens_Lidx[i] = np.random.choice(pres_multiset.size)
                    else:
                        #print('No problems here.')
                        psens_Lidx[i] = np.random.choice(matches)

                #for i, v in enumerate(parent_states):
                #    matches = np.flatnonzero(pres_multiset == v)
                #    psens_Lidx[i] = np.random.choice(matches)

                #for this approach I guess we dont need to pull anything too probabilistic
                #so we will be routing each parent state we grabbed to a child state 1:1
                #this is allowing us to greedily select parent and child state pretty much with same
                #single source of decision making being the matrix of UCB1 values.
                child_states = np.empty(parent_states.shape[0], dtype=int)

                for i in range(parent_states.shape[0]):
                    #so for each parent state we look at the local UCB1 evaluation @_UCBMAT[k, :]
                    #this should be length tf so that we are sampling
                    #caught a case: YES IT DOES SAMPLE [1, TF] HERE!!!!!
                    #print(grm_prior.softmax_sample_uint16(rng, 1, grm_prior._UCBMAT[parent_states[i]]))
                    match(grm_prior._mode):
                        case 'train':
                            #print('sampling from UCB MAT')
                            #print('sampling from UCB MAT')
                            child_states[i] = grm_prior.softmax_sample_uint16(rng, 1, grm_prior._UCBMAT[parent_states[i]])[0]
                        case 'infer':
                            #print('sampling from UCB EXPLOIT')
                            #print('sampling from UCB EXPLOIT')
                            child_states[i] = grm_prior.softmax_sample_uint16(rng, 1, grm_prior._UCB_CONF[parent_states[i]])[0]

                #and thennnn now that we have child states these truly are functions out
                inst_inst[:, 1] = child_states

                #quick case for if we are counting entire pool of generations
                #we would need to tally up all explorations in this generation
                #and incorporate them into the counts within the grammar
                if(grm_prior._mode == 'train' and grm_prior._count_explore is True):
                    for i in range(parent_states.shape[0]):
                        grm_prior._EXPLORE_COUNT[parent_states[i], child_states[i]-1] += 1                
                    grm_prior._EXPLORE_T = np.sum(grm_prior._EXPLORE_COUNT, axis=(0, 1))

                #so at this point we have the new functions written in with only their Gid and TFid
                #now we need to fill with flag logic, fill in sampling data FIRST
                #then we can overwrite the x sensor data with a function translating parent_states
                #into some kind of random sampling index offset for states we can select from

                #I guess we need this
                alpha_sensor_freq = grm_prior._alpha_sensor_freq

                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                # used flags
                used_flags = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = used_flags

                # sensor flags: x always sensor, alpha sometimes sensor
                sensor_flags = USED_to_SENSOR_FLAGS(
                    used_flags,
                    alpha_sensor_freq=alpha_sensor_freq,
                    rng=rng
                )
                inst_inst[:, 4] = sensor_flags

                # const flags: everything used that is not sensor
                const_flags = USED_and_SENSOR_to_CONST_FLAGS(used_flags, sensor_flags)
                inst_inst[:, 3] = const_flags

                # used flags
                #flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                #inst_inst[:, 2] = flags_u32

                # const flags
                #flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                #inst_inst[:, 3] = flags_u32

                # sensor flags
                #flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                #inst_inst[:, 4] = flags_u32

                # current order: a, d, dd, k -> 6,7,8,9
                v_cols = {6: 'UA0', 7: 'STEIS', 8: 'STEIS', 9: 'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, kind in v_cols.items():
                    const_mask = (const_flags & (np.uint32(1) << np.uint32(c))) != 0
                    if kind == 'UA0':
                        inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                    elif kind == 'STEIS':
                        inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                    elif kind == 'ITS':
                        inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                # sensors / refs
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                #so it should be NOW that we have everything filled in, without the
                #proper placement of parent nodes into x sensor for each

                #function I am about to make will need some masking where indices
                #will not be raw to instruction but will need to be pulled from _L_idx[newidxs]

                #pseudo code:
                #fill in ALL random sampling for all inst_inst
                #for i in parent_states:
                #  x_sensor_vector[i] = find_offset_for_index_for_randomsampled_state_with_this_state(parent_states[i])
                #overwrite all x sensor values
                #inst_inst[:, x_sensor_index] = x_sensor_vector
                #double check that this is good to go?


                #holy silly stuff this is what we trying
                #so I am writing in the x sensors, being offset 
                #written in place, where we get the randomly selected nodes
                #and subtract to current idx so that we have the negative offset
                #lets see if it works?
                inst_inst[:, 5] = pop_prior._L_idx[psens_Lidx] - inst_inst[:, 0]

                pass

                #NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
                #below that pass is the example code stripped from UCB1
                #---- ---- ---- delete after development ---- ---- ----
                #NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

                pass

            case 'UCB1':
                
                #quick copy and pasting the tq1d format
                #this one is simply a recreation of UCB1 with a small
                #adjustment of logic being that the counts for t and n
                #will be only considering transformation function counts
                #from models that survive and are tested out of sample.

                #in this grammar we will have some resolved probabalistic structure 
                #thought this through a bit, we will have all saved unique evaluations
                #we will have some transition matrix (default of 1s)
                #for each iteration that we want to make, we take some collection of survived genes
                #extract p values from validation window
                #for each instance that some transition exists in the tree
                #we will add a transformed value to the transition probability matrix
                #this transformation is ln(k))/2p 
                # where p is the pvalue for that transition
                # and where k is the number of times that transition appeared in that solution tree.
                #I am picking this as I want exponential reward as we approach p=0
                #and punishment for any instance worse off than chance (0.5)

                #given how the direction of what is random and what uses our probability distribution
                #and how one would be breadth and one would be depth, we will do the following.
                #for all genes that we are creating,
                # we pull its transformation id from the tq 1d vector in which we have iterated
                # then we randomly select the sensor indices

                # keep shape (chunk_size, 11); last col unused by design
                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                # populate new pop indices after everything currently legal
                start_idx = int(pop_prior._L_idx.max())
                inst_inst[:, 0] = np.arange(start_idx + 1, start_idx + 1 + chunk_size, dtype=np.uint16)

                # random function ids
                inst_inst[:, 1] = grm_prior.softmax_sample_uint16(rng, inst_inst.shape[0])

                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                alpha_sensor_freq = grm_prior._alpha_sensor_freq
                #rng = np.random.default_rng(seed)

                # used flags
                used_flags = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = used_flags

                # sensor flags: x always sensor, alpha sometimes sensor
                sensor_flags = USED_to_SENSOR_FLAGS(
                    used_flags,
                    alpha_sensor_freq=alpha_sensor_freq,
                    rng=rng
                )
                inst_inst[:, 4] = sensor_flags

                # const flags: everything used that is not sensor
                const_flags = USED_and_SENSOR_to_CONST_FLAGS(used_flags, sensor_flags)
                inst_inst[:, 3] = const_flags

                # used flags
                #flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                #inst_inst[:, 2] = flags_u32

                # const flags
                #flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                #inst_inst[:, 3] = flags_u32

                # sensor flags
                #flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                #inst_inst[:, 4] = flags_u32

                # current order: a, d, dd, k -> 6,7,8,9
                v_cols = {6: 'UA0', 7: 'STEIS', 8: 'STEIS', 9: 'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, kind in v_cols.items():
                    const_mask = (const_flags & (np.uint32(1) << np.uint32(c))) != 0
                    if kind == 'UA0':
                        inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                    elif kind == 'STEIS':
                        inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                    elif kind == 'ITS':
                        inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                # sensors / refs
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                pass

            case 'tq1d':
                
                #in this grammar we will have some resolved probabalistic structure 
                #thought this through a bit, we will have all saved unique evaluations
                #we will have some transition matrix (default of 1s)
                #for each iteration that we want to make, we take some collection of survived genes
                #extract p values from validation window
                #for each instance that some transition exists in the tree
                #we will add a transformed value to the transition probability matrix
                #this transformation is ln(k))/2p 
                # where p is the pvalue for that transition
                # and where k is the number of times that transition appeared in that solution tree.
                #I am picking this as I want exponential reward as we approach p=0
                #and punishment for any instance worse off than chance (0.5)

                #given how the direction of what is random and what uses our probability distribution
                #and how one would be breadth and one would be depth, we will do the following.
                #for all genes that we are creating,
                # we pull its transformation id from the tq 1d vector in which we have iterated
                # then we randomly select the sensor indices

                # keep shape (chunk_size, 11); last col unused by design
                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                # populate new pop indices after everything currently legal
                start_idx = int(pop_prior._L_idx.max())
                inst_inst[:, 0] = np.arange(start_idx + 1, start_idx + 1 + chunk_size, dtype=np.uint16)

                # random function ids
                inst_inst[:, 1] = grm_prior.softmax_sample_uint16(rng, inst_inst.shape[0])

                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                alpha_sensor_freq = grm_prior._alpha_sensor_freq
                #rng = np.random.default_rng(seed)

                # used flags
                used_flags = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = used_flags

                # sensor flags: x always sensor, alpha sometimes sensor
                sensor_flags = USED_to_SENSOR_FLAGS(
                    used_flags,
                    alpha_sensor_freq=alpha_sensor_freq,
                    rng=rng
                )
                inst_inst[:, 4] = sensor_flags

                # const flags: everything used that is not sensor
                const_flags = USED_and_SENSOR_to_CONST_FLAGS(used_flags, sensor_flags)
                inst_inst[:, 3] = const_flags

                # used flags
                #flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                #inst_inst[:, 2] = flags_u32

                # const flags
                #flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                #inst_inst[:, 3] = flags_u32

                # sensor flags
                #flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                #inst_inst[:, 4] = flags_u32

                # current order: a, d, dd, k -> 6,7,8,9
                v_cols = {6: 'UA0', 7: 'STEIS', 8: 'STEIS', 9: 'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, kind in v_cols.items():
                    const_mask = (const_flags & (np.uint32(1) << np.uint32(c))) != 0
                    if kind == 'UA0':
                        inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                    elif kind == 'STEIS':
                        inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                    elif kind == 'ITS':
                        inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                # sensors / refs
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                pass

            case 'Null':
                # keep shape (chunk_size, 11); last col unused by design
                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                # populate new pop indices after everything currently legal
                start_idx = int(pop_prior._L_idx.max())
                inst_inst[:, 0] = np.arange(start_idx + 1, start_idx + 1 + chunk_size, dtype=np.uint16)

                # random function ids
                inst_inst[:, 1] = rng.integers(1, 24, size=inst_inst.shape[0], dtype=np.uint16)

                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                # used flags
                flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = flags_u32

                # const flags
                flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                inst_inst[:, 3] = flags_u32

                # sensor flags
                flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                inst_inst[:, 4] = flags_u32

                # current order: a, d, dd, k -> 6,7,8,9
                v_cols = {6: 'UA0', 7: 'STEIS', 8: 'STEIS', 9: 'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, kind in v_cols.items():
                    const_mask = (const_flags & (np.uint32(1) << np.uint32(c))) != 0
                    if kind == 'UA0':
                        inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                    elif kind == 'STEIS':
                        inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                    elif kind == 'ITS':
                        inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                # sensors / refs
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

            case _:
                raise ValueError('Cannot interpret grammar prior in generate_instructions. Illegal type.')

        _append_rows_to_population(inst_inst)

        gen_size -= chunk_size
        if gen_size > 0 and verbose:
            print(f'{gen_size} Generations Remaining.')





def generate_instructions_old(
    pop_prior   :   Population,
    grm_prior   :   Grammar,
    seed        :   int =   None,
    verbose     :   bool=   False
):
    '''
    early dev notes
    - x_inst coming into this contains the terminal states
    - pop_prior is the population we are referencing going into generation
    - grm_prior is the grammar we are referencing going into generation
    - n         is the number of genes that are to be generated
    - chunk_gen is the size of each chunk that will be generated. 
                 allows for grammar updating within generation, will be slower.

    INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
    '''

    #we will have the entirety of the population space pre-allocated 
    #therefore we need some logic checks for if we are within these memory bounds

    #sick of typing in the same things over and over
    n = int(pop_prior._max_size - pop_prior._L_idx.size - pop_prior._E_idx.size)

    # INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
    gen_size = n
    chunk_size = pop_prior._chunk_size

    #print('gen and chunk size:', gen_size, chunk_size)
    if(chunk_size == 0):
        chunk_size = gen_size
    
    while(gen_size > 0):

        #quick fix for if the final chunk generated is ill-shaped
        #really only going to happen as I am making vizualizations for 
        #instantiation operation list efficiency vizualizations
        if(gen_size < chunk_size):
            chunk_size = gen_size

        #print('in loop gen and chunk size:', gen_size, chunk_size)

        
        #match case different grammars
        match(grm_prior._type):
            #this case will not consider any form of grammar
            case 'Null':
                #first thing will be to allocate some memory for instructions
                #instruction format will be along the lines of

                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                #now that we have allocated the memory for instructions
                #we can begin generating instructions
                # INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
                #flag meaning- contains whether or not a, d, dd, k are
                #               gene location data or a constant data

                #we will need to put the population indices in the first column
                #go get the total length of instructions thus far
                start_idx = pop_prior._L_idx.max()
                inst_inst[:, 0] = np.arange(start_idx+1, start_idx+1+chunk_size, dtype=np.uint16)
                
                #for this NO GRAMMAR generation we will randomly select each T function
                inst_inst[:, 1] = np.random.randint(1, 22, size=inst_inst.shape[0], dtype=np.uint16)

                #ultimately we want to have variable_is_sensor probabilities embedded in grammar
                #so that this can be used to generate at random for all locations at once
                #since we also need to generate constants all at once as well.
                #so we will have some sort of mask functionality for each instance being:
                #   NEEDS MASK: EACH type of constant generation
                #   NEEDS MASK: EACH type of sensor location generation (maybe for each T func type) 
                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                #used flags
                flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = flags_u32#.astype(np.float32)

                #const flags
                flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                inst_inst[:, 3] = flags_u32#.astype(np.float32)

                #sensor flags
                flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                inst_inst[:, 4] = flags_u32#.astype(np.float32)

                rng = np.random.default_rng(seed)

                #current order: a, d, dd, k -> 6, 7, 8, 9
                v_cols = {6:'UA0',7:'STEIS',8:'STEIS',9:'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, type in v_cols.items():
                    const_mask = (const_flags & (1 << c)) != 0
                    match(type):
                        case 'UA0':
                            inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                        case 'STEIS':
                            inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                        case 'ITS':
                            inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                #now at this point we need to generate x_inst indices for vars that are sensors
                #given we are working with no grammar prior first, we can structure this code as   
                # it will be in our first grammar structure (2d transition probability matrix)
                #   this means, we select indices first THEN apply our null grammar prior.

                # collect terminal indices from pop_prior._T_idx, 
                # collect gene indices from pop_prior._G_idx
                #   NOW WE HAVE TWO PATHS FOR NULL GRAMMAR PRIOR
                # we can declare some static probability for selecting a terminal index and select randomint
                # OR, we can have an exponentially decaying selection probability across length of population by age (index value)
                #for now we will select with uniform probability across length of population

                #candidate usage of template function
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                #then we need to actually instantiate the genes
                #       should be as simple as building a routing function for funcs in transform
                #       along with design of long term x_inst data structure

                #then after this is instantiated, we need to have some kind of universal
                #used indices of x_inst variable so that we can run these instantiations IN PLACE
                #this will mean entire size of X_inst is allocated before even the first generation of genes

                #return inst_inst

                #print('adding indices to gidx:', inst_inst[:, 0])
            case _:
                raise ValueError('Cannot interpret grammar prior in generate_genes. Illegal type.')
            
        #at this point, we are going to add our instructions into the population prior
        #this starts with adding new gene instructions into gene and legal indices variables
        pop_prior._G_idx = np.union1d(pop_prior._G_idx, inst_inst[:, 0])
        pop_prior._L_idx = np.union1d(pop_prior._L_idx, pop_prior._G_idx)
        #print(f'Lidx: {pop_prior._L_idx}')
        #print(f'Gidx: {pop_prior._G_idx}')
        #print(f'instinst0: {inst_inst[:, 0]}')
                
        
        #now we need to find where we will place newly generated instructions in the instruction array
        col0 = pop_prior._instructions[:, 0].astype(np.int32, copy=False)
        empty = (col0 == 0)
        empty[0] = False
        break_idx = (empty == 1).argmax()
        #print(break_idx, col0[:])
        #print('adding them at:', break_idx)

        #then we will actually bring in the new instantiation instructions into correct memory locations
        #this should place the instructions correctly into the population prior that was provided
        #print(f'writing from indices: [{break_idx}, {break_idx+chunk_size})')
        pop_prior._instructions[ break_idx : break_idx+chunk_size , : ] = inst_inst

        gen_size -= chunk_size
        if(gen_size>0 and verbose):
            print(f'{gen_size} Generations Remaining.')



from collections import deque

# bits 5..9 correspond to sensor slots x,a,d,dd,k stored in cols 5..9
_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # [5,6,7,8,9]
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # [5,6,7,8,9]

import numpy as np

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor flag bit positions
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns


def apply_index_map_axis0(arr: np.ndarray, old_to_new: np.ndarray, *, fill_value=0):
    """
    Apply an old->new index map to any array whose axis 0 matches the gene axis.

    - old_to_new: length G, values in [0..G-1] or -1 for removed
    - returns: new array same shape as arr, with rows moved, removed rows filled with fill_value
    """
    G = old_to_new.shape[0]
    if arr.shape[0] != G:
        raise ValueError(f"arr.shape[0]={arr.shape[0]} must match mapping length G={G}")

    out = np.full_like(arr, fill_value)
    keep_old = np.flatnonzero(old_to_new >= 0)
    out[old_to_new[keep_old]] = arr[keep_old]
    return out







def family_tree_indices(instructions: np.ndarray, gene_idxs, *, include_self: bool = True, include_terminals: bool = False) -> np.ndarray:
    """
    Collect the full ancestor set ("family tree" of parent nodes) for one or many genes.

    instructions: shape (G, 11), rows are genes and columns are:
      [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
       col:  0       1        2          3           4         5  6  7  8   9

    gene_idxs: int OR list/array of ints (gene row indices).
      NOTE: pop_idx is assumed to be the same as the gene row index.

    Parent rule:
      For each node g, decode SENSOR_FLAGS bits 5..9.
      For each active slot among cols 5..9, read displacement value (negative int),
      compute parent index: parent = pop_idx + displacement.
      Recurse until no new parents.

    Returns: sorted unique np.ndarray[int64] of all ancestor gene indices
             (and optionally the starting genes).
    """
    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be 2D with 11 columns (need at least cols 0..9).")

    G = instructions.shape[0]

    # normalize input to 1D int64 array
    if np.isscalar(gene_idxs):
        seeds = np.array([int(gene_idxs)], dtype=np.int64)
    else:
        seeds = np.asarray(gene_idxs, dtype=np.int64).reshape(-1)

    # keep valid seeds
    seeds = seeds[(seeds >= 0) & (seeds < G)]
    if seeds.size == 0:
        return np.empty(0, dtype=np.int64)

    # pre-cast sensor flags once (may be stored as float32)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    visited = np.zeros(G, dtype=np.bool_)
    q = deque()

    for s in seeds.tolist():
        if include_self and not visited[s]:
            visited[s] = True
        q.append(s)

    while q:
        g = q.pop()

        pop_idx = g  # pop_idx == gene row index (per your note)
        sf = sensor_flags[g]

        # mask over the 5 sensor slots (cols 5..9) based on bits 5..9
        slot_mask = ((sf >> _BITS_5_9) & np.uint32(1)).astype(bool)
        if not np.any(slot_mask):
            continue

        # grab displacement values from cols 5..9 where slot_mask True
        disp = instructions[g, _COLS_5_9[slot_mask]].astype(np.int64, copy=False)
        disp = disp[disp < 0]  # only negative displacements per your spec
        if disp.size == 0:
            continue

        parents = pop_idx + disp  # disp negative => parent < pop_idx

        for p in parents.tolist():
            if 0 <= p < G and not visited[p]:
                visited[p] = True
                q.append(p)

    out = np.flatnonzero(visited).astype(np.int64, copy=False)

    if include_terminals:
        func_ids = instructions[:, 1]
        nonzero_idx = np.flatnonzero(func_ids != 0)

        # include only the consecutive leading zeros from the start
        cutoff = nonzero_idx[0] if nonzero_idx.size else G
        if cutoff > 0:
            out = np.union1d(out, np.arange(cutoff, dtype=np.int64))

    return out.astype(np.int64, copy=False)


from collections import deque, defaultdict

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor flag bits
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns


def build_operation_list(instructions: np.ndarray,
                         *,
                         func_col: int = 1,
                         sensor_flag_col: int = 4,
                         displacement_cols: np.ndarray = _COLS_5_9,
                         sensor_bits: np.ndarray = _BITS_5_9,
                         only_negative_parents: bool = True,
                         prefer_last: bool = True,
                         last_bonus: float = 1.20,
                         max_ops: int | None = None):
    """
    Build an execution plan that is as parallel as possible subject to parent dependencies.

    instructions: shape (G, 11), columns:
      [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
    Assumptions:
      - pop_idx == row index (gene index)
      - parent references are stored as (usually negative) displacements in cols 5..9
      - SENSOR_FLAGS bits 5..9 indicate which of cols 5..9 are active sensor slots
      - For each active slot, if value is a negative int displacement d, parent = child + d

    Returns
    -------
    op_list: list of (func_id:int, gene_indices:np.ndarray[int64])
        Each item is one parallel "kernel" call: run func_id for all indices in gene_indices.
        Covers every gene exactly once.

    Notes
    -----
    - This is a topological batching scheduler with a cheap heuristic to reduce func switches.
    - Solve time is roughly O(G + E) where E is #dependency edges discovered.
    """
    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be 2D with at least 10 columns (expected 11).")

    G = instructions.shape[0]
    if G == 0:
        return []

    # func ids (int)
    func_ids = instructions[:, func_col].astype(np.int32, copy=False)

    # sensor flags (uint32)
    sensor_flags = instructions[:, sensor_flag_col].astype(np.uint32, copy=False)

    # ---- Build dependency graph: parent -> child ----
    # We'll construct:
    #   children[parent] = list of children
    #   indegree[child] = #parents
    children = [[] for _ in range(G)]
    indegree = np.zeros(G, dtype=np.int32)

    # Iterate genes; decode parents from SENSOR_FLAGS and displacement cols
    for child in range(G):
        sf = sensor_flags[child]
        slot_mask = ((sf >> sensor_bits) & np.uint32(1)).astype(bool)
        if not np.any(slot_mask):
            continue

        cols = displacement_cols[slot_mask]
        disp = instructions[child, cols].astype(np.int64, copy=False)

        if only_negative_parents:
            disp = disp[disp < 0]
        if disp.size == 0:
            continue

        # parent indices
        parents = child + disp  # disp negative => parent < child in typical case

        # validate and add edges
        for p in parents.tolist():
            if p < 0 or p >= G:
                raise ValueError(f"Invalid parent index computed for child={child}: parent={p}")
            children[p].append(child)
            indegree[child] += 1

    # ---- Ready buckets keyed by func_id ----
    # ready_by_func[f] = deque/list of ready gene indices
    ready_by_func = defaultdict(deque)

    # initial ready nodes
    ready_nodes = np.flatnonzero(indegree == 0).astype(np.int64, copy=False)
    for g in ready_nodes.tolist():
        ready_by_func[int(func_ids[g])].append(g)

    # Helper: choose next func to execute
    last_func = None
    op_list = []
    processed = 0

    def pick_next_func():
        nonlocal last_func

        if not ready_by_func:
            return None

        # Remove empty keys lazily
        empty_keys = [k for k, q in ready_by_func.items() if len(q) == 0]
        for k in empty_keys:
            del ready_by_func[k]
        if not ready_by_func:
            return None

        # Compute current best by ready count
        best_func, best_q = max(ready_by_func.items(), key=lambda kv: len(kv[1]))
        best_n = len(best_q)

        if prefer_last and last_func is not None and last_func in ready_by_func:
            last_n = len(ready_by_func[last_func])
            if last_n * last_bonus >= best_n:
                return last_func

        return best_func


    # ---- Kahn scheduler with batching by func ----
    while processed < G:
        f = pick_next_func()
        if f is None:
            # cycle or missing dependency resolution
            # (shouldn't happen if your system is acyclic)
            raise ValueError("No ready nodes but not all processed: dependency cycle or invalid graph.")

        q = ready_by_func[f]
        batch = np.fromiter(q, dtype=np.int64, count=len(q))  # take all ready for this func
        q.clear()

        op_list.append((int(f), batch))
        last_func = int(f)

        # mark processed; release children
        for g in batch.tolist():
            processed += 1
            for ch in children[g]:
                indegree[ch] -= 1
                if indegree[ch] == 0:
                    ready_by_func[int(func_ids[ch])].append(ch)

        if max_ops is not None and len(op_list) >= max_ops:
            break

    return op_list

import numpy as np

def _epoch_to_seconds_i64(t_epoch) -> np.ndarray:
    """
    Convert epoch-like input to int64 seconds.
    Auto-detects seconds vs milliseconds by magnitude.
    """
    t = np.asarray(t_epoch)

    if np.issubdtype(t.dtype, np.floating):
        t = np.rint(t)

    t = t.astype(np.int64, copy=False)

    if t.size == 0:
        return t

    # modern epoch seconds ~1e9, milliseconds ~1e12
    if np.max(np.abs(t)) >= 10**11:
        t = t // 1000

    return t


import pandas as pd

def tod_minutes_from_dstaware(t_epoch, mode: str = "market_open") -> np.ndarray:
    """
    DST-aware minutes feature using America/New_York.

    mode:
      - 'time_of_day'  -> minutes into local day [0, 1439]
      - 'market_open'  -> minutes relative to 9:30 local time
                          (before open are negative)
    """
    t = np.asarray(t_epoch)

    if np.issubdtype(t.dtype, np.floating):
        t = np.rint(t)

    t = t.astype(np.int64, copy=False)

    if t.size == 0:
        return np.empty(0, dtype=np.int32)

    # auto-detect sec vs ms
    unit = "ms" if np.max(np.abs(t)) >= 10**11 else "s"

    dt_ny = pd.to_datetime(t, unit=unit, utc=True).tz_convert("America/New_York")

    minutes_of_day = (dt_ny.hour * 60 + dt_ny.minute).astype(np.int32)

    if mode == "time_of_day":
        return minutes_of_day

    if mode == "market_open":
        return (minutes_of_day - 570).astype(np.int32)  # 9:30 = 570

    raise ValueError(f"mode must be 'market_open' or 'time_of_day', got {mode!r}")


def dow_sun0_from_epoch(t_epoch, tz_offset_seconds: int = 0) -> np.ndarray:
    """
    Convert Unix epoch (seconds or milliseconds) to day-of-week:
    0,1,2,3,4,5,6 = Sun,Mon,Tue,Wed,Thu,Fri,Sat
    """
    t_sec = _epoch_to_seconds_i64(t_epoch)
    days_since_epoch = (t_sec + np.int64(tz_offset_seconds)) // 86400
    return np.mod(days_since_epoch + 4, 7).astype(np.int8)


import numpy as np
import transform_ops as _OPS
import transform_jit as t_jit  # whatever module contains _MDN_core_heaps

def warmup_numba():
    # tiny shapes so compile happens fast
    m, n = 32, 8
    x = np.random.rand(m, n).astype(np.float32)
    wins = np.ones(n, dtype=np.int64)
    mc = np.ones(n, dtype=np.int64)

    out = np.empty((m, n), dtype=np.float32)

    # Call the exact JIT kernel directly if possible (best)
    t_jit._MDN_core_heaps(x, wins, mc, out)


import numpy as np
import time
import os

_VAR_TO_COL = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}
_VAR_TO_BIT = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}

_FUNC_ID_TO_NAME = {
    0: None,
    1: "t_MAX",  2: "t_MIN",  3: "t_AVG",  4: "t_NEG",  5: "t_DIF",
    6: "t_ADD",  7: "t_SQR",  8: "t_SIN",  9: "t_COS", 10: "t_ASN",
   11: "t_ACS", 12: "t_RNG", 13: "t_HKP", 14: "t_EMA", 15: "t_DOE",
   16: "t_MDN", 17: "t_ZSC", 18: "t_STD", 19: "t_SSN", 20: "t_AGR",
   21: "t_COR",
}


def instantiate_from_ops_chunked_debug(
    op_list,
    instructions: np.ndarray,   # (G,11)
    X_out: np.ndarray,          # (N,G) preallocated
    *,
    transform_ops,              # unified API: transform_ops.apply(...)
    chunk_B: int = 512,
    verbosity: int = 1,          # 0 silent, 1 per-op, 2 per-chunk, 3 very chatty
    check_nans: bool = True,
    check_dtypes: bool = True,
    fail_fast: bool = True,      # if False, continues and records failures
    max_failures: int = 3,
):
    """
    Debug-friendly batched instantiation with bounded scratch (<= (N,chunk_B) per buffer).

    Parent rule (your rule):
      If a var is SENSOR flagged, instruction holds negative displacement `disp` (int-like),
      parent index is: parent = gene_idx + disp

    Uses F_AS(func_id) to determine which variables are needed.

    Returns
    -------
    X_out : np.ndarray
        Instantiated in-place (and also returned).
    failures : list[dict]
        Debug records if fail_fast=False.
    """
    #warmup_numba()
    t0_all = time.perf_counter()

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be shape (G,11) (need cols 0..9).")
    if X_out.ndim != 2:
        raise ValueError("X_out must be 2D (N,G).")
    if X_out.shape[1] != instructions.shape[0]:
        raise ValueError(f"X_out.shape[1]={X_out.shape[1]} must equal G={instructions.shape[0]}")

    G = instructions.shape[0]
    N = X_out.shape[0]
    Bmax = int(chunk_B)

    # Cast flags once (handles float-stored flags safely)
    # NOTE: If instructions[:,3] or [:,4] were float32 with big values, casting to uint32 preserves bits
    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    # Scratch buffers (bounded)
    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    # one extra buffer used if you later want to gather something else;
    # keeping it here for debugging flexibility
    s_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    failures = []

    def log(level, msg):
        if verbosity >= level:
            print(msg)

    def est_bytes(shape, dtype):
        return int(np.prod(shape)) * np.dtype(dtype).itemsize

    # Rough memory footprint of scratch (not counting X_out)
    scratch_bytes = (
        est_bytes((N, Bmax), X_out.dtype) * 4  # x_buf, a_buf, y_buf, s_buf
    )
    log(1, f"[instantiate] N={N}, G={G}, chunk_B={Bmax}, dtype={X_out.dtype}, scratch≈{scratch_bytes/1e6:.1f} MB")

    # Optional: set OMP threads (sometimes crashes come from oversubscription)
    # You can uncomment to clamp threads during debugging:
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    def _fill_series_var(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray):
        """
        Fill buf[:, :B] with resolved SERIES for var_code ('x' or 'a').

        - CONST flag => broadcast scalar down N
        - SENSOR flag => displacement -> parent column gather using parent = idx + disp

        Returns dict of debug stats.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]

        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        # Fill with zeros first to avoid uninitialized garbage (important for debugging!)
        out_view = buf[:, :B]
        out_view.fill(0.0)

        stats = {
            "var": var_code,
            "B": B,
            "n_const": int(is_const.sum()),
            "n_sensor": int(is_sensor.sum()),
            "n_neither": int((~(is_const | is_sensor)).sum()),
            "parents_min": None,
            "parents_max": None,
            "disp_min": None,
            "disp_max": None,
        }

        # SENSOR gather
        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            stats["disp_min"] = int(disp.min()) if disp.size else None
            stats["disp_max"] = int(disp.max()) if disp.size else None

            # Your spec says sensor displacements are negative.
            # If this fails, it usually means the flags are wrong, or you accidentally wrote actual indices.
            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(
                    f"[{var_code}] expected negative displacements for sensor slots, found non-negative: {bad}"
                )

            parents = idx_chunk[is_sensor] + disp  # <<< YOUR RULE

            stats["parents_min"] = int(parents.min()) if parents.size else None
            stats["parents_max"] = int(parents.max()) if parents.size else None

            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(
                    f"[{var_code}] parent out of bounds. "
                    f"parents(min,max)=({stats['parents_min']},{stats['parents_max']}), "
                    f"showing first bad: {bad}"
                )

            # Fancy indexing => copy into our bounded buffer (expected)
            out_view[:, is_sensor] = X_out[:, parents]

        # CONST broadcast
        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        return stats

    def _get_param_vec(var_code: str, idx_chunk: np.ndarray, cast, default):
        """
        Get a per-gene parameter vector of length B (delta1, delta2, kappa).
        We *expect these to be constants*, not sensor time-series.

        If SENSOR-flagged, we raise — because rolling kernels typically need per-column constant params.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(
                f"[{var_code}] is sensor-flagged in this chunk, but expected a constant per-gene parameter. "
                f"This usually means you set SENSOR_FLAGS bits for d/dd/k by mistake."
            )

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)
        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        return out, {
            "var": var_code,
            "B": B,
            "n_const": int(is_const.sum()),
            "n_default": int((~is_const).sum()),
            "min": float(out.min()) if out.size else None,
            "max": float(out.max()) if out.size else None,
        }

    # --- Main loop over op-list ---
    processed_total = 0
    for op_i, (func_id, gene_idx) in enumerate(op_list):
        func_id = int(func_id)
        if func_id == 0:
            continue

        fname = _FUNC_ID_TO_NAME.get(func_id)
        if fname is None:
            raise ValueError(f"Unknown func_id={func_id}")

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        used_vars = F_AS(func_id)  # <-- your map
        # sanity: always must include 'x' for non-zero ops in your design
        if verbosity >= 3:
            log(3, f"[op {op_i}] func_id={func_id} ({fname}), genes={gene_idx.size}, used_vars={used_vars}")

        t0_op = time.perf_counter()

        # chunking
        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size
            t0_chunk = time.perf_counter()

            # For reproducibility in debugging, ensure sorted order (optional)
            # idx_chunk = np.sort(idx_chunk)

            # --- Build x series ---
            try:
                x_stats = _fill_series_var("x", idx_chunk, x_buf)
                if verbosity >= 2:
                    log(2, f"  [chunk {start//Bmax}] x_stats={x_stats}")
            except Exception as e:
                ctx = {
                    "where": "fill_x",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Build optional alpha series ---
            alpha_arg = None
            a_stats = None
            if "a" in used_vars:
                try:
                    a_stats = _fill_series_var("a", idx_chunk, a_buf)
                    alpha_arg = a_buf[:, :B]
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] a_stats={a_stats}")
                except Exception as e:
                    ctx = {
                        "where": "fill_a",
                        "func_id": func_id,
                        "fname": fname,
                        "chunk_start": start,
                        "B": B,
                        "idx_chunk_head": idx_chunk[:10].tolist(),
                        "error": repr(e),
                    }
                    failures.append(ctx)
                    log(1, f"[FAIL] {ctx}")
                    if fail_fast or len(failures) >= max_failures:
                        raise
                    else:
                        continue

            # --- Params (delta1, delta2, kappa) ---
            # only build if used by this func_id (per your F_AS)
            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None
            d_stats = dd_stats = k_stats = None

            try:
                if "d" in used_vars:
                    delta1_vec, d_stats = _get_param_vec("d", idx_chunk, cast=np.int64, default=1)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] d_stats={d_stats}")
                if "dd" in used_vars:
                    delta2_vec, dd_stats = _get_param_vec("dd", idx_chunk, cast=np.int64, default=1)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] dd_stats={dd_stats}")
                if "k" in used_vars:
                    kappa_vec, k_stats = _get_param_vec("k", idx_chunk, cast=np.float32, default=1.0)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] k_stats={k_stats}")
            except Exception as e:
                ctx = {
                    "where": "params",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Compute ---
            y_view = y_buf[:, :B]

            # Optionally prefill output buffer to detect partial writes
            if verbosity >= 3:
                y_view.fill(np.nan)

            try:
                transform_ops.apply(
                    func_id,
                    x_buf[:, :B],
                    alpha=alpha_arg,
                    delta1=delta1_vec,
                    delta2=delta2_vec,
                    kappa=kappa_vec,
                    out=y_view,
                    in_place=False,
                )
            except Exception as e:
                # Print maximal useful context to diagnose kernel crashes
                ctx = {
                    "where": "apply",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "used_vars": used_vars,
                    "x_stats": x_stats,
                    "a_stats": a_stats,
                    "d_stats": d_stats,
                    "dd_stats": dd_stats,
                    "k_stats": k_stats,
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Post checks ---
            if check_nans:
                if not np.isfinite(y_view).all():
                    bad = np.flatnonzero(~np.isfinite(y_view))
                    ctx = {
                        "where": "postcheck_nonfinite",
                        "func_id": func_id,
                        "fname": fname,
                        "chunk_start": start,
                        "B": B,
                        "first_bad_flat_index": int(bad[0]) if bad.size else None,
                    }
                    failures.append(ctx)
                    log(1, f"[FAIL] {ctx}")
                    if fail_fast or len(failures) >= max_failures:
                        raise ValueError(f"Non-finite output detected: {ctx}")

            # Write back
            X_out[:, idx_chunk] = y_view
            processed_total += B

            t1_chunk = time.perf_counter()
            if verbosity >= 2:
                log(2, f"  [chunk {start//Bmax}] wrote B={B} in {(t1_chunk - t0_chunk)*1000:.1f} ms")

        t1_op = time.perf_counter()
        if verbosity >= 1:
            log(1, f"[op {op_i}] func_id={func_id:2d} ({fname}) genes={gene_idx.size} time={(t1_op - t0_op):.3f}s")

    t1_all = time.perf_counter()
    log(1, f"[instantiate] done processed≈{processed_total} gene-cols, total time={(t1_all - t0_all):.3f}s")

    return X_out, failures


import numpy as np
import time
from collections import defaultdict

_VAR_TO_COL = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}
_VAR_TO_BIT = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}

_FUNC_ID_TO_NAME = {
    0: "NOP",
    1: "t_MAX",  2: "t_MIN",  3: "t_AVG",  4: "t_NEG",  5: "t_DIF",
    6: "t_ADD",  7: "t_SQR",  8: "t_SIN",  9: "t_COS", 10: "t_ASN",
   11: "t_ACS", 12: "t_RNG", 13: "t_HKP", 14: "t_EMA", 15: "t_DOE",
   16: "t_MDN", 17: "t_ZSC", 18: "t_STD", 19: "t_SSN", 20: "t_AGR",
   21: "t_COR",
}

def instantiate_from_ops_chunked_partialbuild_class(
    op_list,
    instructions: np.ndarray,     # (G,11)
    X_out: np.ndarray,            # (N,G)
    population: Population,
    *,
    transform_ops,                # unified apply(func_id,...)
    chunk_B: int = 16,
    verbosity: int = 1,
    sanitize_final: bool = True,
):
    """
    Batched instantiation with aggressive NaN/Inf -> 0 sanitization and replacement tracking.

    Parent rule (YOUR RULE):
      For sensor-series slots, instruction stores negative displacement 'disp' (int-like),
      parent = gene_idx + disp.

    Verbosity:
      0: silent
      1: per-op timing
      2: per-op timing + replacement tracking output (standard)
      3: adds per-chunk timing + extra stats
      4: very chatty (debug details)
    """
    t0_all = time.perf_counter()

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be shape (G,11) (need cols 0..9).")
    if X_out.ndim != 2:
        raise ValueError("X_out must be 2D (N,G).")

    G = instructions.shape[0]
    N = X_out.shape[0]
    if X_out.shape[1] != G:
        raise ValueError(f"X_out.shape[1]={X_out.shape[1]} must equal G={G}")

    Bmax = int(chunk_B)

    # Cast flags once (allows flags stored as float32 in instructions)
    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    # Scratch buffers (bounded)
    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    # Replacement tracking
    # counts keyed by ("stage", func_id) where stage in {"input_x","input_a","output","final"}
    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    def _sanitize_inplace(arr: np.ndarray, key):
        """
        Replace NaN/Inf -> 0 in-place.
        Update counters for this key.
        """
        # Count first to avoid double-counting after modification
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        # inf includes +inf/-inf; use isfinite to count
        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            # non-finite includes NaNs too; subtract NaNs to get inf count
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        # replace
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] with resolved series for var_code ('x' or 'a'):
          - CONST flag => broadcast scalar down N
          - SENSOR flag => gather parent series using parent = idx + disp
          - else => 0.0
        Then sanitize NaN/Inf -> 0 (in case parents already contain non-finite)
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        # sensor gather
        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            # spec: negative offsets
            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(f"{var_code}: expected negative displacement for sensor slots; got {bad}")

            parents = idx_chunk[is_sensor] + disp  # <<< YOUR RULE
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(f"{var_code}: parent out of bounds (first bad: {bad})")

            out_view[:, is_sensor] = X_out[:, parents]

        # const broadcast
        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        # sanitize and track
        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(4, f"    [{stage}] B={B} const={int(is_const.sum())} sensor={int(is_sensor.sum())}")

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        If sensor-flagged, this indicates a bug; raise.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)
        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            # force 1 -> 2 behavior for windows etc.
            out = np.maximum(out, cast(clamp_min))

        return out

    # Timing stats
    op_times = defaultdict(float)
    op_counts = defaultdict(int)

    # Main schedule loop
    for op_i, (func_id, gene_idx) in enumerate(op_list):
        fid = int(func_id)
        if fid == 0:
            continue

        name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
        used_vars = F_AS(fid)

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        t0_op = time.perf_counter()

        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size

            t0_chunk = time.perf_counter()

            # inputs
            x_view = _fill_series("x", idx_chunk, x_buf, fid)

            alpha_arg = None
            if "a" in used_vars:
                a_view = _fill_series("a", idx_chunk, a_buf, fid)
                alpha_arg = a_view  # (N,B)

            # params
            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None

            if "d" in used_vars:
                # window-like params: clamp to >=2 to avoid w=1 weirdness
                delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "dd" in used_vars:
                delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "k" in used_vars:
                kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

            # compute
            y_view = y_buf[:, :B]
            transform_ops.apply(
                fid,
                x_view,
                alpha=alpha_arg,
                delta1=delta1_vec,
                delta2=delta2_vec,
                kappa=kappa_vec,
                out=y_view,
                in_place=False,
            )

            # sanitize output and track
            _sanitize_inplace(y_view, ("output", fid))

            # write back
            X_out[:, idx_chunk] = y_view

            if verbosity >= 3:
                _log(3, f"  [chunk] {name} fid={fid} start={start} B={B} dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms")

        dt_op = time.perf_counter() - t0_op
        op_times[fid] += dt_op
        op_counts[fid] += int(gene_idx.size)

        if verbosity >= 1:
            _log(1, f"[op] {name:5s} fid={fid:2d} genes={gene_idx.size:6d} time={dt_op:.3f}s")

        # Standard replacement report in verbosity >= 2
        if verbosity >= 2:
            nan_in_x = rep_nan.get(("input_x", fid), 0)
            inf_in_x = rep_inf.get(("input_x", fid), 0)
            nan_in_a = rep_nan.get(("input_a", fid), 0)
            inf_in_a = rep_inf.get(("input_a", fid), 0)
            nan_out  = rep_nan.get(("output", fid), 0)
            inf_out  = rep_inf.get(("output", fid), 0)

            if (nan_in_x or inf_in_x or nan_in_a or inf_in_a or nan_out or inf_out):
                _log(2, f"    [repl] fid={fid:2d} {name}: "
                        f"x(nan={nan_in_x},inf={inf_in_x}) "
                        f"a(nan={nan_in_a},inf={inf_in_a}) "
                        f"out(nan={nan_out},inf={inf_out})")

    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    t1_all = time.perf_counter()
    stats = {
        "total_time_s": float(t1_all - t0_all),
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(2, f"[sanitize] final pass replaced: "
                    f"NaN={rep_nan.get(('final', -1), 0)}, Inf={rep_inf.get(('final', -1), 0)}")

    return X_out, stats

def instantiate_from_ops_chunked_contig(
    population,
    *,
    transform_ops,
    chunk_B: int = 16,
    verbosity: int = 1,
    sanitize_final: bool = True,
):
    """
    Batched instantiation with aggressive NaN/Inf -> 0 sanitization and replacement tracking.

    Uses:
      - population._instructions
      - population._X_inst
      - op_list built internally via build_operation_list(population._instructions)

    Parent rule:
      For sensor-series slots, instruction stores negative displacement 'disp' (int-like),
      parent = gene_idx + disp.

    Verbosity:
      0: silent
      1: per-op timing
      2: per-op timing + replacement tracking output
      3: adds per-chunk timing + extra stats
      4: very chatty
    """
    t0_all = time.perf_counter()

    if not hasattr(population, "_instructions"):
        raise AttributeError("population must have attribute '_instructions'")
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")

    instructions = population._instructions
    X_out = population._X_inst

    if not isinstance(instructions, np.ndarray):
        raise TypeError("population._instructions must be a numpy ndarray")
    if not isinstance(X_out, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must be shape (G,11) (need cols 0..9)")
    if X_out.ndim != 2:
        raise ValueError("population._X_inst must be 2D with shape (N, G)")

    G = instructions.shape[0]
    N = X_out.shape[0]

    if X_out.shape[1] != G:
        raise ValueError(f"population._X_inst.shape[1]={X_out.shape[1]} must equal G={G}")

    op_list = build_operation_list(instructions)

    Bmax = int(chunk_B)

    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    def _sanitize_inplace(arr: np.ndarray, key):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] with resolved series for var_code ('x' or 'a'):
          - CONST flag  => broadcast scalar down N
          - SENSOR flag => gather parent series using parent = idx + disp
          - else        => 0.0
        Then sanitize NaN/Inf -> 0.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(f"{var_code}: expected negative displacement for sensor slots; got {bad}")

            parents = idx_chunk[is_sensor] + disp
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(f"{var_code}: parent out of bounds (first bad: {bad})")

            out_view[:, is_sensor] = X_out[:, parents]

        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(4, f"    [{stage}] B={B} const={int(is_const.sum())} sensor={int(is_sensor.sum())}")

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        If sensor-flagged, this indicates a bug; raise.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)

        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            out = np.maximum(out, cast(clamp_min))

        return out

    op_times = defaultdict(float)
    op_counts = defaultdict(int)

    for func_id, gene_idx in op_list:
        fid = int(func_id)
        if fid == 0:
            continue

        name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
        used_vars = F_AS(fid)

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        t0_op = time.perf_counter()

        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size

            t0_chunk = time.perf_counter()

            x_view = _fill_series("x", idx_chunk, x_buf, fid)

            alpha_arg = None
            if "a" in used_vars:
                a_view = _fill_series("a", idx_chunk, a_buf, fid)
                alpha_arg = a_view

            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None

            if "d" in used_vars:
                delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "dd" in used_vars:
                delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "k" in used_vars:
                kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

            y_view = y_buf[:, :B]
            transform_ops.apply(
                fid,
                x_view,
                alpha=alpha_arg,
                delta1=delta1_vec,
                delta2=delta2_vec,
                kappa=kappa_vec,
                out=y_view,
                in_place=False,
            )

            _sanitize_inplace(y_view, ("output", fid))

            X_out[:, idx_chunk] = y_view

            if verbosity >= 3:
                _log(
                    3,
                    f"  [chunk] {name} fid={fid} start={start} B={B} "
                    f"dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms"
                )

        dt_op = time.perf_counter() - t0_op
        op_times[fid] += dt_op
        op_counts[fid] += int(gene_idx.size)

        if verbosity >= 1:
            _log(1, f"[op] {name:5s} fid={fid:2d} genes={gene_idx.size:6d} time={dt_op:.3f}s")

        if verbosity >= 2:
            nan_in_x = rep_nan.get(("input_x", fid), 0)
            inf_in_x = rep_inf.get(("input_x", fid), 0)
            nan_in_a = rep_nan.get(("input_a", fid), 0)
            inf_in_a = rep_inf.get(("input_a", fid), 0)
            nan_out  = rep_nan.get(("output", fid), 0)
            inf_out  = rep_inf.get(("output", fid), 0)

            if (nan_in_x or inf_in_x or nan_in_a or inf_in_a or nan_out or inf_out):
                _log(
                    2,
                    f"    [repl] fid={fid:2d} {name}: "
                    f"x(nan={nan_in_x},inf={inf_in_x}) "
                    f"a(nan={nan_in_a},inf={inf_in_a}) "
                    f"out(nan={nan_out},inf={inf_out})"
                )

    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    t1_all = time.perf_counter()
    stats = {
        "total_time_s": float(t1_all - t0_all),
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(
                2,
                f"[sanitize] final pass replaced: "
                f"NaN={rep_nan.get(('final', -1), 0)}, Inf={rep_inf.get(('final', -1), 0)}"
            )

    return stats

import time
from collections import defaultdict
import numpy as np

def instantiate_from_ops_chunked_intraday(
    population,
    *,
    transform_ops,
    chunk_num: int | None = None,
    chunk_B: int = 16,
    verbosity: int = 0,
    sanitize_final: bool = True,
):
    """
    Intraday segmented instantiation on population._X_inst of shape (N, G).

    Chunk behavior
    --------------
    - If chunk_num is None, instantiate all rows.
    - If chunk_num is an int, instantiate only that chunk, where the chunk
      row bounds are inferred from population.

    Intraday day behavior
    ---------------------
    - population._time_terminals must be True
    - population._structure must equal 'Intraday'
    - time-of-day column is population._tod_idx if present, else population._T_idx[-2]
    - every exact zero in that time-of-day column is treated as the start of a new day
      within the selected chunk
    - instantiation is run separately on each day segment inside the selected chunk

    This preserves intraday resets for:
    - rolling / window-like transforms
    - self-referencing behavior
    - value initialization from t0 within each day segment

    Verbosity
    ---------
    0 : silent
    1 : print number of counted days + total timing
    2 : add per-day timing + replacement summary
    3 : add per-op timing
    4 : add per-batch timing / debug details

    Returns
    -------
    stats : dict
        Timing and replacement statistics.

    Notes
    -----
    - population._X_inst is updated in place
    - only the selected chunk rows are cleared / instantiated / sanitized
    - this function requires:
        _VAR_TO_BIT
        _VAR_TO_COL
        _FUNC_ID_TO_NAME
        F_AS
        build_operation_list
      to exist in scope
    """
    t0_all = time.perf_counter()

    # ------------------------------------------------------------
    # population validation
    # ------------------------------------------------------------
    if not hasattr(population, "_time_terminals"):
        raise AttributeError("population must have attribute '_time_terminals'")
    if population._time_terminals is not True:
        raise ValueError("population._time_terminals must be True")

    if not hasattr(population, "_structure"):
        raise AttributeError("population must have attribute '_structure'")
    if population._structure != "Intraday":
        raise ValueError(
            f"population._structure must be 'Intraday', got {population._structure!r}"
        )

    if not hasattr(population, "_instructions"):
        raise AttributeError("population must have attribute '_instructions'")
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")
    if not hasattr(population, "_T_idx"):
        raise AttributeError("population must have attribute '_T_idx'")

    instructions = population._instructions
    X_out = population._X_inst

    if not isinstance(instructions, np.ndarray):
        raise TypeError("population._instructions must be a numpy ndarray")
    if not isinstance(X_out, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must be shape (G,11) or have cols 0..9")
    if X_out.ndim != 2:
        raise ValueError("population._X_inst must be 2D with shape (N, G)")

    N, Gx = X_out.shape
    G = instructions.shape[0]

    if Gx != G:
        raise ValueError(
            f"population._X_inst.shape[1]={Gx} must equal number of instructions G={G}"
        )

    if hasattr(population, "_tod_idx") and population._tod_idx is not None:
        tod_col = int(population._tod_idx)
    else:
        if len(population._T_idx) < 2:
            raise ValueError("population._T_idx must have at least two entries so _T_idx[-2] exists")
        tod_col = int(population._T_idx[-2])

    if tod_col < 0 or tod_col >= G:
        raise ValueError(
            f"time-of-day column index {tod_col} is out of bounds for G={G}"
        )

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    # ------------------------------------------------------------
    # resolve selected chunk rows
    # ------------------------------------------------------------
    if chunk_num is None:
        row_lo = 0
        row_hi = N
    else:
        if hasattr(population, "get_chunk_bounds"):
            row_lo, row_hi = population.get_chunk_bounds(chunk_num)
        elif hasattr(population, "get_chunk_slice"):
            row_sl = population.get_chunk_slice(chunk_num)
            row_lo = 0 if row_sl.start is None else int(row_sl.start)
            row_hi = N if row_sl.stop is None else int(row_sl.stop)
        else:
            raise AttributeError(
                "population must provide get_chunk_bounds(chunk_num) or get_chunk_slice(chunk_num)"
            )

    row_lo = int(row_lo)
    row_hi = int(row_hi)

    if row_lo < 0 or row_hi > N or row_lo >= row_hi:
        raise ValueError(
            f"Invalid selected chunk row bounds [{row_lo}:{row_hi}) for N={N}"
        )

    X_chunk = X_out[row_lo:row_hi, :]
    N_chunk = X_chunk.shape[0]

    if verbosity >= 1:
        _log(
            1,
            f"[intraday] chunk_num={chunk_num} rows=[{row_lo}:{row_hi}) N_chunk={N_chunk}"
        )

    # ------------------------------------------------------------
    # enforce zero baseline only inside selected chunk
    # ------------------------------------------------------------
    keep = np.unique(
        np.concatenate((population._T_idx, population._E_idx))
    ).astype(np.int64)

    _kept = X_chunk[:, keep].copy()
    X_chunk.fill(0.0)
    X_chunk[:, keep] = _kept
    del _kept

    # ------------------------------------------------------------
    # detect day starts only within selected chunk
    # ------------------------------------------------------------
    tod = X_chunk[:, tod_col]
    day_starts_local = np.flatnonzero(tod == 0)

    if day_starts_local.size == 0:
        raise ValueError(
            f"No zeros were found in the time-of-day column at index {tod_col} "
            f"within selected rows [{row_lo}:{row_hi})"
        )

    if day_starts_local[0] != 0:
        raise ValueError(
            f"First zero in the selected chunk occurs at local row {int(day_starts_local[0])}, not 0. "
            "This means the chunk does not begin at a day boundary."
        )

    day_ends_local = np.empty_like(day_starts_local)
    day_ends_local[:-1] = day_starts_local[1:]
    day_ends_local[-1] = N_chunk

    days_counted = int(day_starts_local.size)

    day_starts_abs = row_lo + day_starts_local
    day_ends_abs = row_lo + day_ends_local

    if verbosity >= 1:
        _log(1, f"[intraday] counted days in chunk={days_counted}")

    # ------------------------------------------------------------
    # build schedule once
    # ------------------------------------------------------------
    op_list = build_operation_list(instructions)
    Bmax = int(chunk_B)

    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    day_times = []

    def _sanitize_inplace(arr: np.ndarray, key):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(X_seg: np.ndarray, var_code: str, idx_batch: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] using one day/segment slice X_seg of shape (Ns, G):
          - CONST flag  => broadcast scalar down Ns
          - SENSOR flag => gather parent series from X_seg[:, parents]
          - else        => 0.0

        Sanitizes NaN/Inf -> 0 in place.
        """
        B = idx_batch.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_batch]
        sf = sensor_flags[idx_batch]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_batch, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(
                    f"{var_code}: expected negative displacement for sensor slots; got {bad}"
                )

            parents = idx_batch[is_sensor] + disp
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(
                    f"{var_code}: parent out of bounds (first bad: {bad})"
                )

            # gather only from current day segment to preserve intraday reset behavior
            out_view[:, is_sensor] = X_seg[:, parents]

        if np.any(is_const):
            cvals = raw[is_const].astype(X_seg.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(
                4,
                f"    [{stage}] segN={X_seg.shape[0]} B={B} "
                f"const={int(is_const.sum())} sensor={int(is_sensor.sum())}"
            )

        return out_view

    def _param_vec(var_code: str, idx_batch: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        Sensor-flagged params are invalid.
        """
        B = idx_batch.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_batch]
        sf = sensor_flags[idx_batch]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_batch, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)

        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            out = np.maximum(out, cast(clamp_min))

        return out

    # ------------------------------------------------------------
    # per-day segmented instantiation inside selected chunk
    # ------------------------------------------------------------
    for day_i, (start_local, end_local) in enumerate(zip(day_starts_local, day_ends_local), start=1):
        t0_day = time.perf_counter()

        start_abs = row_lo + int(start_local)
        end_abs = row_lo + int(end_local)

        X_seg = X_chunk[start_local:end_local, :]
        Ns = X_seg.shape[0]

        if Ns <= 0:
            continue

        x_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        a_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        y_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)

        for func_id, gene_idx in op_list:
            fid = int(func_id)
            if fid == 0:
                continue

            name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
            used_vars = F_AS(fid)

            gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
            if gene_idx.size == 0:
                continue

            t0_op = time.perf_counter()

            for start in range(0, gene_idx.size, Bmax):
                idx_batch = gene_idx[start:start + Bmax]
                B = idx_batch.size

                t0_batch = time.perf_counter()

                x_view = _fill_series(X_seg, "x", idx_batch, x_buf, fid)

                alpha_arg = None
                if "a" in used_vars:
                    a_view = _fill_series(X_seg, "a", idx_batch, a_buf, fid)
                    alpha_arg = a_view

                delta1_vec = None
                delta2_vec = None
                kappa_vec  = None

                if "d" in used_vars:
                    delta1_vec = _param_vec("d", idx_batch, cast=np.int64, default=2, clamp_min=2)
                if "dd" in used_vars:
                    delta2_vec = _param_vec("dd", idx_batch, cast=np.int64, default=2, clamp_min=2)
                if "k" in used_vars:
                    kappa_vec = _param_vec("k", idx_batch, cast=np.float32, default=1.0, clamp_min=None)

                y_view = y_buf[:, :B]
                y_view.fill(0.0)

                transform_ops.apply(
                    fid,
                    x_view,
                    alpha=alpha_arg,
                    delta1=delta1_vec,
                    delta2=delta2_vec,
                    kappa=kappa_vec,
                    out=y_view,
                    in_place=False,
                )

                _sanitize_inplace(y_view, ("output", fid))

                # write back only into this day segment of this selected chunk
                X_seg[:, idx_batch] = y_view

                if verbosity >= 4:
                    _log(
                        4,
                        f"  [chunk={chunk_num}] [day={day_i:4d}] [batch] {name} fid={fid} "
                        f"rows=[{start_abs}:{end_abs}) gene_start={start} B={B} "
                        f"dt={(time.perf_counter()-t0_batch)*1000:.1f}ms"
                    )

            dt_op = time.perf_counter() - t0_op
            op_times[fid] += dt_op
            op_counts[fid] += int(gene_idx.size)

            if verbosity >= 3:
                _log(
                    3,
                    f"[chunk={chunk_num}] [day={day_i:4d}] [op] {name:5s} fid={fid:2d} "
                    f"genes={gene_idx.size:6d} seg_rows={Ns:6d} time={dt_op:.3f}s"
                )

        dt_day = time.perf_counter() - t0_day
        day_times.append(float(dt_day))

        if verbosity >= 2:
            _log(
                2,
                f"[chunk={chunk_num}] [day={day_i:4d}] rows=[{start_abs}:{end_abs}) "
                f"Nseg={Ns} time={dt_day:.3f}s"
            )

    # ------------------------------------------------------------
    # final sanitize only on selected chunk
    # ------------------------------------------------------------
    if sanitize_final:
        _sanitize_inplace(X_chunk, ("final", -1))

    total_time_s = time.perf_counter() - t0_all

    stats = {
        "chunk_num": None if chunk_num is None else int(chunk_num),
        "chunk_row_lo": int(row_lo),
        "chunk_row_hi": int(row_hi),
        "chunk_rows": int(N_chunk),
        "days_counted": int(days_counted),
        "time_of_day_col": int(tod_col),
        "day_starts_local": day_starts_local.copy(),
        "day_ends_local": day_ends_local.copy(),
        "day_starts": day_starts_abs.copy(),
        "day_ends": day_ends_abs.copy(),
        "total_time_s": float(total_time_s),
        "day_times_s": day_times,
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 1:
        _log(1, f"[intraday] chunk_num={chunk_num} total_time={total_time_s:.3f}s")

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced in selected chunk: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(
                2,
                f"[sanitize] final pass replaced in selected chunk: "
                f"NaN={rep_nan.get(('final', -1), 0)}, "
                f"Inf={rep_inf.get(('final', -1), 0)}"
            )

    return stats

def instantiate_from_ops_chunked_intraday_final_NO_WF(
    population,
    *,
    transform_ops,
    chunk_B: int = 16,
    verbosity: int = 0,
    sanitize_final: bool = True,
):
    """
    Intraday segmented instantiation on population._X_inst of shape (N, G).

    Rules
    -----
    - population._time_terminals must be True
    - population._structure must equal 'Intraday'
    - time-of-day column is population._T_idx[-2]
    - every exact zero in that time-of-day column is treated as the start of a new day/chunk
    - instantiation is run separately on each chunk [zero_i : zero_{i+1}) or [zero_last : N)

    This preserves intraday resets for:
    - rolling / window-like transforms
    - self-referencing behavior
    - value initialization from t0 within each chunk

    Verbosity
    ---------
    0 : silent
    1 : print number of counted days (zero starts) + total timing
    2 : add per-day timing + replacement summary
    3 : add per-op timing
    4 : add per-chunk timing / debug details

    Returns
    -------
    stats : dict
        Timing and replacement statistics.

    Notes
    -----
    - population._X_inst is updated in place
    - this function assumes the time-of-day column is already present in population._X_inst
    - this function requires:
        _VAR_TO_BIT
        _VAR_TO_COL
        _FUNC_ID_TO_NAME
        F_AS
        build_operation_list
      to exist in scope
    """
    t0_all = time.perf_counter()

    # ------------------------------------------------------------
    # population validation
    # ------------------------------------------------------------
    if not hasattr(population, "_time_terminals"):
        raise AttributeError("population must have attribute '_time_terminals'")
    if population._time_terminals is not True:
        raise ValueError("population._time_terminals must be True")

    if not hasattr(population, "_structure"):
        raise AttributeError("population must have attribute '_structure'")
    if population._structure != "Intraday":
        raise ValueError(
            f"population._structure must be 'Intraday', got {population._structure!r}"
        )

    if not hasattr(population, "_instructions"):
        raise AttributeError("population must have attribute '_instructions'")
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")
    if not hasattr(population, "_T_idx"):
        raise AttributeError("population must have attribute '_T_idx'")

    instructions = population._instructions
    X_out = population._X_inst

    if not isinstance(instructions, np.ndarray):
        raise TypeError("population._instructions must be a numpy ndarray")
    if not isinstance(X_out, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must be shape (G,11) or have cols 0..9")
    if X_out.ndim != 2:
        raise ValueError("population._X_inst must be 2D with shape (N, G)")

    N, Gx = X_out.shape
    G = instructions.shape[0]

    if Gx != G:
        raise ValueError(
            f"population._X_inst.shape[1]={Gx} must equal number of instructions G={G}"
        )

    if len(population._T_idx) < 2:
        raise ValueError("population._T_idx must have at least two entries so _T_idx[-2] exists")

    tod_col = int(population._T_idx[-2])
    if tod_col < 0 or tod_col >= G:
        raise ValueError(
            f"time-of-day column population._T_idx[-2]={tod_col} is out of bounds for G={G}"
        )

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    # ------------------------------------------------------------
    # enforce zero baseline for all non-terminal/non-excluded cols
    # ------------------------------------------------------------
    keep = np.unique(np.concatenate((population._T_idx, population._E_idx))).astype(np.int64)

    _kept = X_out[:, keep].copy()   # keep is small
    X_out.fill(0.0)                 # fastest bulk clear
    X_out[:, keep] = _kept
    del _kept

    # ------------------------------------------------------------
    # detect day starts from exact zeros in TOD column
    # ------------------------------------------------------------
    tod = X_out[:, tod_col]
    day_starts = np.flatnonzero(tod == 0)

    if day_starts.size == 0:
        raise ValueError(
            f"No zeros were found in the time-of-day column at population._T_idx[-2]={tod_col}"
        )

    if day_starts[0] != 0:
        raise ValueError(
            f"First zero in the time-of-day column occurs at row {int(day_starts[0])}, not row 0. "
            "This would leave leading rows outside any intraday reset chunk."
        )

    day_ends = np.empty_like(day_starts)
    day_ends[:-1] = day_starts[1:]
    day_ends[-1] = N

    days_counted = int(day_starts.size)

    if verbosity >= 1:
        _log(1, f"[intraday] counted days={days_counted}")

    # ------------------------------------------------------------
    # build schedule once
    # ------------------------------------------------------------
    op_list = build_operation_list(instructions)
    Bmax = int(chunk_B)

    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    day_times = []

    def _sanitize_inplace(arr: np.ndarray, key):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(X_seg: np.ndarray, var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] using one day/segment slice X_seg of shape (Ns, G):
          - CONST flag  => broadcast scalar down Ns
          - SENSOR flag => gather parent series from X_seg[:, parents]
          - else        => 0.0

        Sanitizes NaN/Inf -> 0 in place.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(
                    f"{var_code}: expected negative displacement for sensor slots; got {bad}"
                )

            parents = idx_chunk[is_sensor] + disp
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(
                    f"{var_code}: parent out of bounds (first bad: {bad})"
                )

            # important: gather only from current segment to preserve intraday reset behavior
            out_view[:, is_sensor] = X_seg[:, parents]

        if np.any(is_const):
            cvals = raw[is_const].astype(X_seg.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(
                4,
                f"    [{stage}] segN={X_seg.shape[0]} B={B} "
                f"const={int(is_const.sum())} sensor={int(is_sensor.sum())}"
            )

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        Sensor-flagged params are invalid.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)

        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            out = np.maximum(out, cast(clamp_min))

        return out

    # ------------------------------------------------------------
    # per-day segmented instantiation
    # ------------------------------------------------------------
    for day_i, (start_row, end_row) in enumerate(zip(day_starts, day_ends), start=1):
        t0_day = time.perf_counter()

        X_seg = X_out[start_row:end_row, :]
        Ns = X_seg.shape[0]

        if Ns <= 0:
            continue

        x_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        a_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        y_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)

        for func_id, gene_idx in op_list:
            fid = int(func_id)
            if fid == 0:
                continue

            name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
            used_vars = F_AS(fid)

            gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
            if gene_idx.size == 0:
                continue

            t0_op = time.perf_counter()

            for start in range(0, gene_idx.size, Bmax):
                idx_chunk = gene_idx[start:start + Bmax]
                B = idx_chunk.size

                t0_chunk = time.perf_counter()

                x_view = _fill_series(X_seg, "x", idx_chunk, x_buf, fid)

                alpha_arg = None
                if "a" in used_vars:
                    a_view = _fill_series(X_seg, "a", idx_chunk, a_buf, fid)
                    alpha_arg = a_view

                delta1_vec = None
                delta2_vec = None
                kappa_vec  = None

                if "d" in used_vars:
                    delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
                if "dd" in used_vars:
                    delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
                if "k" in used_vars:
                    kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

                y_view = y_buf[:, :B]
                y_view.fill(0.0)   # add this line
                transform_ops.apply(
                    fid,
                    x_view,
                    alpha=alpha_arg,
                    delta1=delta1_vec,
                    delta2=delta2_vec,
                    kappa=kappa_vec,
                    out=y_view,
                    in_place=False,
                )

                _sanitize_inplace(y_view, ("output", fid))

                # write back only into this segment
                X_seg[:, idx_chunk] = y_view

                if verbosity >= 4:
                    _log(
                        4,
                        f"  [day={day_i:4d}] [chunk] {name} fid={fid} "
                        f"rows=[{start_row}:{end_row}) gene_start={start} B={B} "
                        f"dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms"
                    )

            dt_op = time.perf_counter() - t0_op
            op_times[fid] += dt_op
            op_counts[fid] += int(gene_idx.size)

            if verbosity >= 3:
                _log(
                    3,
                    f"[day={day_i:4d}] [op] {name:5s} fid={fid:2d} "
                    f"genes={gene_idx.size:6d} seg_rows={Ns:6d} time={dt_op:.3f}s"
                )

        dt_day = time.perf_counter() - t0_day
        day_times.append(float(dt_day))

        if verbosity >= 2:
            _log(
                2,
                f"[day={day_i:4d}] rows=[{int(start_row)}:{int(end_row)}) "
                f"Nseg={Ns} time={dt_day:.3f}s"
            )

    # ------------------------------------------------------------
    # final sanitize on whole instantiated matrix
    # ------------------------------------------------------------
    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    total_time_s = time.perf_counter() - t0_all

    stats = {
        "days_counted": int(days_counted),
        "time_of_day_col": int(tod_col),
        "day_starts": day_starts.copy(),
        "day_ends": day_ends.copy(),
        "total_time_s": float(total_time_s),
        "day_times_s": day_times,
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 1:
        _log(1, f"[intraday] total_time={total_time_s:.3f}s")

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(
                2,
                f"[sanitize] final pass replaced: "
                f"NaN={rep_nan.get(('final', -1), 0)}, "
                f"Inf={rep_inf.get(('final', -1), 0)}"
            )

    return stats

import pandas as pd

def initialize(
    structure   :   str =   'Intraday',
    incl_time   :   bool=   True,
    data_file   :   str =   '../data/spy5m.csv',
    epoch_idx   :   list=   [0],
    hlocv_idx   :   list=   [1,2,3,4],
    pop_size    :   int =   1000,
    grmr_prior  :   any = None,
    grmr_type   :   str =   'Null',
    grmr_mdl    :   int | tuple =   240,
    grmr_p_mttn :   float=  0.0,
    grmr_p_csvr :   float=  0.0,
    grmr_a_sens :   float=  0.5,
    chunk_size  :   float   =   0.1,
    wf_windows  :   int =   1,
    verbose     :   int =   0
):
    '''
    The all in one function for initialization
    this function should be usable as a one liner, ending at instantated genes
    '''

    #read in data first
    x_raw = pd.read_csv(data_file)

    #temporary error thrower to ensure we are not working with different
    #shapes just so these next few lines are made modular
    if x_raw.shape[1] != 5:
        raise ValueError('x_raw came in with NOT 5 columns. check top of instantate function, currently using non modular approach for very simple variable initialization.')
    
    X_initialized = np.zeros((x_raw.shape[0], pop_size), dtype=np.float32)
    x_raw = x_raw.to_numpy(dtype=np.float32, copy=False)
    X_initialized[:, :5] = x_raw
    del x_raw
    

    if(verbose>1):print('Data loaded')

    #generate our population variable and pass all parameters
    X = Population(
        X_inst=X_initialized,
        terminal_idx=hlocv_idx,
        excluded_idx=epoch_idx,
        max_size=pop_size,
        chunk_size=chunk_size,
        include_time=incl_time,
        structure=structure,
        wf_windows=wf_windows
    )
    if(verbose>1):print('Population initialized')
    
    if(grmr_prior is None):
        print('IN INITIALIZE: GRAMMAR PRIOR IS NONE\n')
        #generate our grammar variable and pass all parameters
        grammar = Grammar(
            type=grmr_type,
            max_delta_lookback=grmr_mdl,
            p_crossover=grmr_p_csvr,
            p_mutation=grmr_p_mttn,
            alpha_sensor_freq=grmr_a_sens
        )
    else:
        print('IN INITIALIZE: GRAMMAR PRIOR EXISTS.\n')
        grammar = grmr_prior
    if(verbose>1):print('Grammar Initialized')

    if(verbose>0):print('Initializations complete')

    if(verbose>1):print(f'pop size: {pop_size, type(pop_size)}')
    #if(verbose>1):print(f'Tidx size: {X._T_idx.size}')
    #if(verbose>1):print(f'Lidx size: {X._L_idx.size}')
    if(verbose>1):print(f'n size: {int(pop_size - X._E_idx.size - X._T_idx.size)}')
    
    #generate instructions (in place, resides in X)
    generate_instructions(
        pop_prior=X,
        grm_prior=grammar,
        #a little clunky but maybe easiest,
        #this is the open space not used so far
        #so total size - excluded space and legal existing space
    )
    #print(X._instructions)
    if(verbose>0):print(f'Instructions generated.')
    if(verbose>0):print(f'Initialization complete.')

    #this functions returns the initial population (has instructions) and grammar
    return X, grammar