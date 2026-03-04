'''
	Logan Kelsch - 2/26/26

	Holy moly took long enough to get started on this file.
	This file will contain all overarching and helper functionality in the evaluation of genes and USAGE of gene instantiation in our framework.
	this file will also contain the evaluation of grammar
	this file will also contain the evaluation of a population in terms of itself

	In this text block here I will develop the instructure.

BASIC:

	We ultimately will have some 1d array of length N (or 2d array if we split along days, I'm sure we will)
	This vector will be our solutions that will be interpreted somehow
			expanding on "solutions" and "interpreted somehow"

	gene evaluation will provide information for selecting parent / parent quality as well as iterating grammar
	grammar evaluation will provide information that goes hand in hand with gene evaluation for iterating grammar
	population evaluation will provide information that removes redundancy or excessive inbreeding in a population


solutions:

	We will have several different angles that we could try to solve for, which I will write in likely order of development.
	A main objective of developing this consists of ensuring a modular approach for indiana jones style swapping of methods.

	- Boolean anomoly detection
			this allows us to interpret positive values as true prediction (ex: price moves 3st up from 20 candle EMA)
	- Moving average SD
			this allows for predictions to be made regarding directionality, while holding a normalized output
	- Volume moving average SD
			This allows for more generalized predictions to be made regarding behavior of volume (directionality removed)
	- certainly some more approaches 

Mod dev step:
	
	The first steps will be:
	- creating a solution array for boolean anomoly detection
	- creating a gene matrix evaluation functionality for 2d instantiated gene matrices
			these can be made modular by having a class or object mechanism that ties evaluation style mechanism to solution instantiation mechanism
			for example, for the boolean anomoly, classification approaches will not be as effective, so we will have a cost score
				in which considers the difference of that and the expected value of this solution space.
				this will allow for a negative score to be undesirable, zero to be equal to market performance, and >>0 to be ideal.
				Expected cost = c_FP x FP + c_FN x FN


'''

import initialization as _I
import numpy as np

class Solver:
	'''
	This class object will simply contain data that is wanted to resolve the solution vector from given data
	'''
	def __init__(
		self,
		population	:	_I.Population,
		t_vec		:	str	=	'Close', #could be volume
		t_mode		:	str	=	'AD', #AD for anomoly detection, or RE for raw emission
		emission	:	list=	['some operations here that can be turned into op list or instantiated easily'],
		AD_cond		:	tuple=	('lt', -2)
	):
		'''emission is interpreted as functions applied to t_vec from left to right.'''

		self._tmode= t_mode
		self._emission= emission
		self._AD_cond = AD_cond

		#match case for transforming t_vec string into a column index in population variable
		target_idx = -1
		match(t_vec):
			#for logical cases we need to see if the time variables are 
			#added as terminal states to interpret what locations they are at.
			case 'Volume':
				#NOTE this is assuming that THLCV or THLOCV is the standard shape of data coming
				if(population._time_terminals):
					#walking backwards is dow(-1), tod(-2), vol(-3), close(-4)
					target_idx = population._T_idx[-3]
				else:
					#walking backwards is vol(-1), close(-2)
					target_idx = population._T_idx[-1]
			case 'Close':
				#NOTE this is assuming that THLCV or THLOCV is the standard shape of data coming
				if(population._time_terminals):
					#walking backwards is dow(-1), tod(-2), vol(-3), close(-4)
					target_idx = population._T_idx[-4]
				else:
					#walking backwards is vol(-1), close(-2)
					target_idx = population._T_idx[-2]
			case _:
				raise ValueError(f'Target vector (t_vec parameter in Solver Initialization) of "{t_vec}" dont make noooo sense. Try "Close".')
			
		if(target_idx<0):
			raise ValueError('Did not form a good target vector index in solver initialization. Got idx (-1)')
		
		self._tidx = target_idx
		
	def solve(
		self,
		Population :	_I.Population
	):
		'''Solves target vector and mask. solution mask is where a conditional statement is true.
		raw emission is used directly in computation for some form of emission prediction,
		where raw emission[solution mask] is used in cost based evaluation function'''

		#now we have the tvec index and can select it for computation with Population._X_inst[:, target_idx]
		if(self._tmode=='AD'|self._tmode=='RE'):
			
			#Here we need to generate the raw emission
			raw_emission = generate_raw_emission(Population, self._tidx, self._emission)

			#generate a mask for where we want to allow comparison to be done
			#the default will be at all locations
			evaluation_mask = generate_evaluation_mask(Population)

			#may find a better shape or type for this default of no use
			anomaly_mask = False

			if(self._tmode=='AD'):
				
				#then we need to make a boolean masking variable where raw emission is true under AD_cond parameter interpretation
				anomaly_mask = generate_anomaly_mask(raw_emission, self._AD_cond)	

		else:
			raise NotImplementedError(f'Target Mode of "{self._tmode}" is not supported at this moment.')
		
		#now we have successfully solved the target for the population
		#returns returns read all about it
		return raw_emission, evaluation_mask, anomaly_mask

		

#first candidate evaluation function should be a light modular somewhat function that containst the first solution type selection

def evaluate(
	population	:	_I.Population,
	raw_emissions:	any,
	evaluation_mask	:	any,
	anomaly_mask	:	any	
):
	#bring in population to be able to see the instantiated genes
	#identify the column that has real data that the solution will be derived from
	#call generate_solution() to get the desired vector for comparison with the selected function

	#NOTE SOMETHING LIKE
	#
	#	x[evaluation_mask] for cost based evaluation, as initial example?? not really just scrap comments at the moment.
	#
	#

	#NOTE IF SOLVER._T_MODE IS 'AD' THEN WE MUST CONVERT POPULATION INSTANTIATION INTO BOOLEAN COLUMNS WHERE G_mn = G_mn > 0
	return


import transform_ops


def generate_raw_emission(Population, target_idx, emissions, offset):
    """
    Build a future-peeking target vector from Population._X_inst[:, target_idx]
    and apply a left-to-right emission pipeline using transform_ops.apply.

    Parameters
    ----------
    Population : object
        Must have attribute `_X_inst` of shape (N, G).
    target_idx : int
        Column index in Population._X_inst to use as the base series.
    emissions : list[dict]
        Left-to-right operation list, e.g.
        [
            {"ID": 6, "alpha": 1.34},
            {"ID": 20, "alpha": 0.0, "delta1": 20}
        ]

        Accepted keys inside each dict:
            - "ID"       : required function ID
            - "alpha"    : constant only
            - "delta1"   : constant int only
            - "delta2"   : constant int only
            - "kappa"    : constant float only
            - "min_count": constant int only

    offset : int
        Forward-looking offset. Equivalent to aligning each row with the value
        `offset` steps in the future.

    Returns
    -------
    y : np.ndarray, shape (N,)
        Emitted target vector. The final `offset` positions are NaN because they
        do not have enough future data.
    """
    # ------------------ validate population / source ------------------
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")

    X = Population._X_inst
    if not isinstance(X, np.ndarray):
        raise TypeError("Population._X_inst must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("Population._X_inst must be 2D with shape (N, G)")

    N, G = X.shape

    target_idx = int(target_idx)
    if target_idx < 0 or target_idx >= G:
        raise IndexError(f"target_idx={target_idx} out of bounds for G={G}")

    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be >= 0")

    if emissions is None:
        emissions = []
    if not isinstance(emissions, (list, tuple)):
        raise TypeError("emissions must be a list/tuple of dict operations")

    # ------------------ build the future-aligned base target ------------------
    base = np.asarray(X[:, target_idx], dtype=np.float32)

    valid_len = N - offset
    out_full = np.full(N, np.nan, dtype=np.float32)

    # If offset is beyond the data length, nothing is valid.
    if valid_len <= 0:
        return out_full

    # Important:
    # instead of np.roll(..., -offset) with wraparound contamination,
    # we explicitly slice the valid future-aligned portion.
    work = np.ascontiguousarray(base[offset:].reshape(valid_len, 1), dtype=np.float32)
    buf = np.empty_like(work)

    # ------------------ helpers ------------------
    allowed_keys = {"ID", "alpha", "delta1", "delta2", "kappa", "min_count"}

    def _require_scalar(name, value):
        arr = np.asarray(value)
        if arr.ndim != 0:
            raise TypeError(f"Emission parameter '{name}' must be a scalar constant")
        return arr.item()

    def _build_apply_kwargs(fid, op, x_shape, x_dtype):
        extra = set(op.keys()) - allowed_keys
        if extra:
            raise KeyError(
                f"Unsupported keys in emission {op}: {sorted(extra)}. "
                f"Allowed keys are {sorted(allowed_keys)}"
            )

        kwargs = {}

        if "delta1" in op:
            kwargs["delta1"] = int(_require_scalar("delta1", op["delta1"]))
        if "delta2" in op:
            kwargs["delta2"] = int(_require_scalar("delta2", op["delta2"]))
        if "kappa" in op:
            kwargs["kappa"] = float(_require_scalar("kappa", op["kappa"]))
        if "min_count" in op:
            kwargs["min_count"] = int(_require_scalar("min_count", op["min_count"]))

        if "alpha" in op:
            alpha_const = float(_require_scalar("alpha", op["alpha"]))

            # AGR / COR require alpha matrix same shape as x
            if fid in (20, 21):
                kwargs["alpha"] = np.full(x_shape, alpha_const, dtype=x_dtype)
            else:
                kwargs["alpha"] = alpha_const
        else:
            if fid in (20, 21):
                raise ValueError(f"Emission ID {fid} requires an 'alpha' constant")

        return kwargs

    # ------------------ left-to-right emission application ------------------
    for i, op in enumerate(emissions):
        if not isinstance(op, dict):
            raise TypeError(f"Each emission must be a dict, got {type(op).__name__} at position {i}")
        if "ID" not in op:
            raise KeyError(f"Emission at position {i} is missing required key 'ID'")

        fid = int(op["ID"])
        if fid == 0:
            continue

        kwargs = _build_apply_kwargs(fid, op, work.shape, work.dtype)

        transform_ops.apply(
            fid,
            work,
            out=buf,
            in_place=False,
            **kwargs,
        )

        # Mirror the instantiation pipeline's defensive sanitization behavior.
        np.nan_to_num(buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ping-pong buffers
        work, buf = buf, work

    out_full[:valid_len] = work[:, 0]
    return out_full


def generate_evaluation_mask(Population:_I.Population, offset):
    """
    Build a boolean mask showing which rows can be validly evaluated
    for a forward-looking target with the given offset, without wrapping
    across intraday day boundaries.

    Uses the time-since-market-open vector at:
        Population._X_inst[:, Population._T_idx[-2]]

    Assumptions
    -----------
    - 0.0 means market open
    - no negative values exist
    - within a day, time-of-day is nondecreasing
    - when a new day starts, time-of-day resets lower (typically to 0)

    Parameters
    ----------
    Population : object
        Must have:
            - _X_inst : ndarray shape (N, G)
            - _T_idx  : indexable with [-2]
    offset : int
        Forward lookahead in rows.

    Returns
    -------
    mask : np.ndarray, shape (N,), dtype=bool
        True where evaluation is allowed.
        False where the forward offset would either:
          - go out of bounds, or
          - wrap into the next day.
    """
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")
    if not hasattr(Population, "_T_idx"):
        raise AttributeError("Population must have attribute '_T_idx'")

    X = Population._X_inst
    if not isinstance(X, np.ndarray):
        raise TypeError("Population._X_inst must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("Population._X_inst must be 2D")

    try:
        if(Population._time_terminals):
            tod_idx = int(Population._T_idx[-2])
        else:
            raise NotImplementedError(f'In generating evaluation mask, time terminals is not true in pop prior, cannot grab time of day.')
    except Exception as e:
        raise ValueError("Population._T_idx must be indexable and contain [-2]") from e

    N, G = X.shape
    if tod_idx < 0 or tod_idx >= G:
        raise IndexError(f"time-of-day column index {tod_idx} out of bounds for G={G}")

    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be >= 0")

    tod = np.asarray(X[:, tod_idx])

    # offset == 0 means no forward peeking, so everything is valid
    if offset == 0:
        return np.ones(N, dtype=bool)

    # if offset exceeds data length, nothing can be evaluated
    if offset >= N:
        return np.zeros(N, dtype=bool)

    mask = np.zeros(N, dtype=bool)

    # valid only if:
    #   1) i + offset is in bounds
    #   2) time-of-day at i+offset is still >= time-of-day at i
    #      (if it became smaller, we crossed into a new day)
    mask[:-offset] = tod[offset:] >= tod[:-offset]

    return mask

import numpy as np


def generate_anomaly_mask(raw_emission, AD_cond):
    """
    Generate a boolean anomaly mask from a raw emission vector and one or more
    anomaly-detection conditions, with optional AND/OR chaining.

    Parameters
    ----------
    raw_emission : array-like
        1D vector of emitted values.

    AD_cond : tuple or list[tuple]
        Supported condition tuple formats:

        1) Simple condition:
            (comparator, value)

        2) Condition with logical combiner:
            (logic, comparator, value)

        Allowed comparators:
            'lt' : <
            'gt' : >
            'le' : <=
            'ge' : >=

        Allowed logic:
            'and', 'or'

        Notes
        -----
        - If AD_cond is a single 2-tuple, it is treated as one condition.
        - If AD_cond is a list of tuples:
            * the first tuple may be 2-tuple or 3-tuple
            * later tuples may be 2-tuple (defaults to 'and') or 3-tuple
        - Chaining is evaluated left-to-right with no grouping precedence.

        Examples
        --------
        ('lt', 2)

        [('ge', 0), ('and', 'lt', 5)]

        [('lt', -2), ('or', 'gt', 2)]

        [('ge', 0), ('and', 'lt', 5), ('or', 'gt', 10)]

    Returns
    -------
    mask : np.ndarray, dtype=bool
        Boolean mask of same shape as raw_emission.
    """
    x = np.asarray(raw_emission)

    if x.ndim != 1:
        raise ValueError("raw_emission must be a 1D vector")

    def _eval_condition(arr, comp, val):
        if comp == 'lt':
            return arr < val
        elif comp == 'gt':
            return arr > val
        elif comp == 'le':
            return arr <= val
        elif comp == 'ge':
            return arr >= val
        else:
            raise ValueError(
                f"Unsupported comparator '{comp}'. "
                "Allowed comparators are: 'lt', 'gt', 'le', 'ge'"
            )

    def _parse_condition(cond, is_first=False):
        if not isinstance(cond, tuple):
            raise TypeError(f"Each condition must be a tuple, got {type(cond).__name__}")

        if len(cond) == 2:
            comp, val = cond
            logic = 'and'
        elif len(cond) == 3:
            logic, comp, val = cond
            if logic not in ('and', 'or'):
                raise ValueError(
                    f"Unsupported logic '{logic}'. Allowed logic values are 'and' and 'or'"
                )
        else:
            raise ValueError(
                f"Each condition must be either (comparator, value) or "
                f"(logic, comparator, value), got {cond}"
            )

        if is_first:
            logic = 'and'

        return logic, comp, val

    # allow one standalone condition like ('lt', 2)
    if isinstance(AD_cond, tuple) and len(AD_cond) == 2 and isinstance(AD_cond[0], str):
        logic, comp, val = _parse_condition(AD_cond, is_first=True)
        return _eval_condition(x, comp, val)

    if not isinstance(AD_cond, (list, tuple)):
        raise TypeError("AD_cond must be a tuple or a list/tuple of condition tuples")

    conds = list(AD_cond)
    if len(conds) == 0:
        raise ValueError("AD_cond cannot be empty")

    logic, comp, val = _parse_condition(conds[0], is_first=True)
    mask = _eval_condition(x, comp, val)

    for cond in conds[1:]:
        logic, comp, val = _parse_condition(cond, is_first=False)
        current = _eval_condition(x, comp, val)

        if logic == 'and':
            mask &= current
        elif logic == 'or':
            mask |= current
        else:
            raise ValueError(
                f"Unsupported logic '{logic}'. Allowed logic values are 'and' and 'or'"
            )

    return mask