import reproduction as _R
import initialization as _I
import evaluation as _E
import numpy as np



class Walker:

    def __init__(self, xi, xd, min_iters, exhaust=0.001):
        self.xi = xi
        self.xd = xd
        self.min_iters = min_iters
        self.exhaust = exhaust

        self.i_step = (xd - xi) / min_iters
        self.i_bped = -self.i_step / 2

        self.position = xi
        self.step = self.i_step
        self.bped = self.i_bped

    def walk(self, status: bool):
        if status:
            self.step *= 1.5
            self.bped /= 2

            # Clamp step so it cannot go above i_step
            self.step = min(self.step, self.i_step)

            self.position += self.step
        else:
            self.step /= 2
            self.bped *= 1.5

            # Clamp bped so it cannot go below i_bped
            self.bped = max(self.bped, self.i_bped)

            self.position += self.bped

        #print('step:',self.step,'bped:',self.bped)

        return self.position
    
    def is_exhausted(self):
        if((abs(self.step) + abs(self.bped)) <= self.exhaust):
            return True
        else:
            return False




def evolve_population(
    X = None, 
    G = None, 
    iterations = 0,
    early_stop = 0.0,
    break_extinction = False,
    initialization_kwargs = None,
    solver_kwargs = None,
    selector_kwargs = None,
    chunk_num = None,
    return_stats    :   bool    =   False
):
    '''returns X, G, final evaluation (and optionally stats)'''

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
            "pop_size"    :   150,
            "grmr_type"   :   'Null',
            "grmr_mdl"    :   240,
            "chunk_size"  :   25,
            "wf_windows"  :   2,
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
            "offset"    :   5,
            "t_vec"		:	'Close',
            "t_mode"	:	'AD',
            "emission"	:	[
                {"ID": 5, "alpha": 
                    {"ID": 3, "x": "tvec", "delta1": 20, "offset": False}},
                {"ID": "divide"},
                {"ID": 18, "x": "tvec", "delta1": 20, "offset": False},
            ],
            "AD_cond"   :	('gt', 2),
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
        evaluation, inst_stats = _E.evaluate(X, solver, chunk_num)

        qgenes = np.count_nonzero(evaluation["F"]>0)

        if(i==0):
            #initial print of solution stats
            ftcount = np.unique_counts(evaluation['svecs']['anomaly_mask'])[1]
            print(f'F:{ftcount[0]} T:{ftcount[1]} P:{100*(ftcount[1]/(ftcount[0]+ftcount[1])):.2f}%')

        print(f'Gen {i}: {qgenes} Quality Genes. ', end='')

        if(break_extinction and qgenes==0):
            print('All genes failed, breaking loop.')
            break

        
        print('Reproducting...')

        reproduction_stats = _R.reproduce(X,G, selector, evaluation)

        
        

        #add on stats if user wants them returned
        if(return_stats):
            instantiation_stats_stack.append(inst_stats)
            reproduction_stats_stack.append(reproduction_stats)

        #break case for enough success to end iteration
        if(early_stop > 0 and (qgenes / X._max_size) > early_stop):
            print(f'Success Reached ({early_stop:.2f}) in Population. Breaking evolution loop.')
            break

    print()


    #final gene evaluation
    evaluation, inst_stats = _E.evaluate(X, solver, chunk_num)

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
    

