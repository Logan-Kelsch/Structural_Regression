import numpy as np

def is_successful_evolution(evaluation, metric="F"):
    if(np.count_nonzero(evaluation[metric] > 0) > 0):
        return True
    else:
        return False