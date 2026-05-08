import numpy as np

def is_successful_evolution(evaluation, metric="F"):
    if(np.count_nonzero(evaluation[metric] > 0) > 0):
        return True
    else:
        return False
    
import math

def steps_to_eps(h, k, eps):
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if h < 0:
        raise ValueError("h must be >= 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    if h < eps:
        return 0
    if k == 0:
        return math.inf

    return math.floor(math.log(h / eps) / k) + 1




