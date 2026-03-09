import reproduction as _R
import initialization as _I
import evaluation as _E


def evolve_population(
    X, G, 
    iterations,
    initialization_kwargs   :   dict,
    solver_kwargs,
    selector_kwargs,
    return_stats    :   bool    =   False
):
    '''returns X, G, final evaluation'''

    #-----------------------------------------------
    # KWARGS FORMATTING FOR NO PARAMETER DEFINITION
    # AND GRAMMAR AND POPULATION FIRST GENERATIONS
    #-----------------------------------------------

    if(initialization_kwargs is None):
        initialization_kwargs = {
            "structure"   :   'Intraday',
            "incl_time"   :   True,
            "data_file"   :   '../data/spy5m.csv',
            "epoch_idx"   :   [0],
            "hlocv_idx"   :   [1,2,3,4],
            "pop_size"    :   1000,
            "grmr_type"   :   'Null',
            "grmr_mdl"    :   240,
            "chunk_size"  :   0.1,
            "verbose"     :   0
        }

    # X and G should come in to this function
    #if we want to iterate from a given population
    if(X is None or G is None):
        print('No Grammar OR Population provided.\n'
              'Initializing new population.')
        
        #we will initialize a new population and grammar if we got nothing
        X, G = _I.initialize(**initialization_kwargs)

    if(solver_kwargs is None):
        solver_kwargs = {
            "population":	X,
            "offset"    :   5,
            "t_vec"		:	'Close',
            "t_mode"	:	'RE',
            "emission"	:	[
                {"ID": 5, "alpha": 
                    {"ID": 3, "x": "tvec", "delta1": 20, "offset": False}},
                {"ID": "divide"},
                {"ID": 18, "x": "tvec", "delta1": 20, "offset": False},
            ],
            "AD_cond"   :	('lt', -2),
        }

    if(selector_kwargs is None):
        selector_kwargs = {
            "method"    :   "Threshold",
            "percent"   :   0.2
        }
    
    #initialize solver and selector
    solver  = _E.Solver(X, **solver_kwargs)
    selector= _R.Selector(**selector_kwargs) 

    #the loop really should count reproduction iterations,
    #we need an evaluation to reproduce and will want to end
    #with a final evaluation, so we will loop E, R and end E

    #list variables for holding stats collected if user wants them returned
    if(return_stats):
        instantiation_stats_stack = []
        reproduction_stats_stack  = []

    for i in range(iterations):
        evaluation, inst_stats = _E.evaluate(X, solver)
        reproduction_stats = _R.reproduce(X,G, selector, evaluation)

        #add on stats if user wants them returned
        if(return_stats):
            instantiation_stats_stack.append(inst_stats)
            reproduction_stats_stack.append(reproduction_stats)

    #final gene evaluation
    evaluation, inst_stats = _E.evaluate(X, solver)

    #add on last stats if the user wants them returned
    if(return_stats):
        instantiation_stats_stack.append(inst_stats)
        stats = {
            "Instantiation":instantiation_stats_stack,
            "Reproduction":reproduction_stats_stack         
        }

        #then if the user wants the stats returned we should probably also return the stats
        return X, G, evaluation, stats
    else:
        return X, G, evaluation