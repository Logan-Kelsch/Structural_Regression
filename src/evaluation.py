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
		t_mode		:	str	=	'RE', #AD for anomoly detection, or RE for raw emission
		emission	:	list=	[
			{"ID":16,"delta1":6}
		],
		offset      :   int=    6,
		AD_cond		:	tuple=	('lt', -2),
	):
		'''emission is interpreted as functions applied to t_vec from left to right.'''

		self._tmode		= t_mode
		self._emission	= emission
		self._AD_cond 	= AD_cond
		self._offset 	= offset

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
		if(self._tmode=='AD' or self._tmode=='RE'):
			
			#Here we need to generate the raw emission
			raw_emission = generate_raw_emission(Population, self._tidx, self._emission, self._offset)

			#generate a mask for where we want to allow comparison to be done
			#the default will be at all locations
			evaluation_mask = generate_evaluation_mask(Population, self._offset)

			#may find a better shape or type for this default of no use
			anomaly_mask = np.full(evaluation_mask.shape, True, dtype=bool)

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
	population		:	_I.Population,
	raw_emissions	:	any,
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


import numpy as np
import transform_ops


def generate_raw_emission(Population, target_idx, emissions, offset):
    """
    Broad, emission-driven target builder.

    Base target:
      tvec_offset[i] = Population._X_inst[i + offset, target_idx]
      valid region is i = 0..N-offset-1
      output is length N with NaN tail of length `offset`.

    Supported emission features
    ---------------------------
    Each emission dict may include:
      - "ID"      : op id, or "divide"
      - "x"       : "emit" (default), "tvec", scalar, 1D array, or (valid_len,1) array
      - "alpha"   : scalar, array, "tvec", "emit", or a nested op dict
      - "offset"  : bool, default True, controls how "tvec" is resolved
                    True  -> future-aligned tvec_offset
                    False -> raw/current tvec_raw
      - "delta1", "delta2", "kappa", "min_count": scalar params

    Special:
      {"ID": "divide"} divides the current emission by the output of the next op.

    New:
      alpha may be a nested op dict, e.g.
        {"ID": 5, "alpha": {"ID": 3, "delta1": 6, "offset": False}}

      For nested alpha op dicts:
        - x must be "tvec" or omitted
        - if omitted, x defaults to "tvec"
        - nested dict may use its own offset
    """
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")

    X = Population._X_inst
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("Population._X_inst must be a 2D numpy ndarray")

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
        raise TypeError("emissions must be a list/tuple of dicts")

    base = np.asarray(X[:, target_idx], dtype=np.float32)

    valid_len = N - offset
    out_full = np.full(N, np.nan, dtype=np.float32)

    if valid_len <= 0:
        return out_full

    if offset == 0:
        tvec_offset_1d = base
    else:
        tvec_offset_1d = base[offset:]

    tvec_raw_1d = base[:valid_len]

    tvec_offset = np.ascontiguousarray(tvec_offset_1d.reshape(valid_len, 1), dtype=np.float32)
    tvec_raw    = np.ascontiguousarray(tvec_raw_1d.reshape(valid_len, 1), dtype=np.float32)

    work = tvec_offset.copy()
    buf  = np.empty_like(work)

    def _to_fid(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "divide":
                return "divide"
            if s.isdigit():
                return int(s)
            raise ValueError(f"Unrecognized ID string: {v!r}")
        return int(v)

    def _scalar_const(name, v, cast=float):
        if v is None:
            return None
        a = np.asarray(v)
        if a.ndim != 0:
            raise TypeError(f"'{name}' must be a scalar constant")
        return cast(a.item())

    def _resolve_series(spec, use_offset_tvec, current_work):
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("emit", "work")):
            return current_work

        if isinstance(spec, str) and spec.strip().lower() == "tvec":
            return tvec_offset if use_offset_tvec else tvec_raw

        if np.isscalar(spec):
            return np.full((valid_len, 1), float(spec), dtype=np.float32)

        arr = np.asarray(spec)
        if arr.ndim == 1:
            if arr.shape[0] != valid_len:
                raise ValueError(f"1D series must have length {valid_len}, got {arr.shape[0]}")
            return np.ascontiguousarray(arr.astype(np.float32, copy=False).reshape(valid_len, 1))
        if arr.ndim == 2:
            if arr.shape != (valid_len, 1):
                raise ValueError(f"2D series must be shape {(valid_len,1)}, got {arr.shape}")
            return np.ascontiguousarray(arr.astype(np.float32, copy=False))

        raise ValueError("Series spec must be scalar, 1D (valid_len,), or 2D (valid_len,1)")

    def _resolve_alpha(alpha_spec, use_offset_tvec, current_work):
        """
        Resolve alpha which may be:
          - scalar
          - array
          - "tvec"
          - "emit"
          - nested op dict (must root x on tvec)
        """
        if isinstance(alpha_spec, dict):
            return _eval_nested_alpha_op(alpha_spec)

        if isinstance(alpha_spec, str) and alpha_spec.strip().lower() in ("tvec", "emit", "work"):
            return _resolve_series(alpha_spec, use_offset_tvec, current_work)

        if np.isscalar(alpha_spec):
            return float(alpha_spec)

        return _resolve_series(alpha_spec, use_offset_tvec, current_work)

    def _eval_nested_alpha_op(op_dict):
        """
        Evaluate a nested op dict for alpha.
        This subtree must be rooted on tvec, not current emission.
        """
        if not isinstance(op_dict, dict):
            raise TypeError("Nested alpha spec must be a dict")

        if "ID" not in op_dict:
            raise KeyError("Nested alpha dict must include 'ID'")

        fid = _to_fid(op_dict["ID"])
        if fid == "divide":
            raise ValueError("Nested alpha dict cannot use {'ID':'divide'} directly")

        use_offset_tvec = bool(op_dict.get("offset", True))

        x_spec = op_dict.get("x", "tvec")
        if not (isinstance(x_spec, str) and x_spec.strip().lower() == "tvec"):
            raise ValueError("Nested alpha dict must have x='tvec' or omit 'x'")

        x_in = tvec_offset if use_offset_tvec else tvec_raw

        alpha_in = None
        if "alpha" in op_dict:
            alpha_in = _resolve_alpha(op_dict["alpha"], use_offset_tvec, x_in)

        delta1 = _scalar_const("delta1", op_dict.get("delta1", None), cast=int)
        delta2 = _scalar_const("delta2", op_dict.get("delta2", None), cast=int)
        kappa  = _scalar_const("kappa",  op_dict.get("kappa",  None), cast=float)

        kw = {}
        if "min_count" in op_dict:
            kw["min_count"] = _scalar_const("min_count", op_dict.get("min_count"), cast=int)

        if fid in (20, 21):
            if alpha_in is None:
                raise ValueError(f"Nested alpha op ID {fid} requires 'alpha'")
            if np.isscalar(alpha_in):
                alpha_in = np.full(x_in.shape, float(alpha_in), dtype=np.float32)
            else:
                if not (isinstance(alpha_in, np.ndarray) and alpha_in.shape == x_in.shape):
                    raise ValueError(
                        f"Nested alpha op ID {fid}: alpha must be shape {x_in.shape}, "
                        f"got {getattr(alpha_in, 'shape', None)}"
                    )

        out_tmp = np.empty_like(x_in)
        transform_ops.apply(
            fid,
            x_in,
            alpha=alpha_in,
            delta1=delta1,
            delta2=delta2,
            kappa=kappa,
            out=out_tmp,
            in_place=False,
            **kw,
        )
        np.nan_to_num(out_tmp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out_tmp

    def _apply_one(op_dict, current_work, out_buf):
        if not isinstance(op_dict, dict):
            raise TypeError(f"Each emission must be a dict, got {type(op_dict).__name__}")

        if "ID" not in op_dict:
            raise KeyError("Emission dict missing required key 'ID'")

        fid = _to_fid(op_dict["ID"])
        if fid == "divide":
            raise ValueError("Internal error: _apply_one called on 'divide'")

        use_offset_tvec = bool(op_dict.get("offset", True))

        x_in = _resolve_series(op_dict.get("x", None), use_offset_tvec, current_work)

        alpha_in = None
        if "alpha" in op_dict:
            alpha_in = _resolve_alpha(op_dict["alpha"], use_offset_tvec, current_work)

        delta1 = _scalar_const("delta1", op_dict.get("delta1", None), cast=int)
        delta2 = _scalar_const("delta2", op_dict.get("delta2", None), cast=int)
        kappa  = _scalar_const("kappa",  op_dict.get("kappa",  None), cast=float)

        kw = {}
        if "min_count" in op_dict:
            kw["min_count"] = _scalar_const("min_count", op_dict.get("min_count"), cast=int)

        if fid in (20, 21):
            if alpha_in is None:
                raise ValueError(f"Emission ID {fid} requires 'alpha'")
            if np.isscalar(alpha_in):
                alpha_in = np.full(x_in.shape, float(alpha_in), dtype=np.float32)
            else:
                if not (isinstance(alpha_in, np.ndarray) and alpha_in.shape == x_in.shape):
                    raise ValueError(
                        f"Emission ID {fid}: alpha must be shape {x_in.shape}, "
                        f"got {getattr(alpha_in, 'shape', None)}"
                    )

        transform_ops.apply(
            fid,
            x_in,
            alpha=alpha_in,
            delta1=delta1,
            delta2=delta2,
            kappa=kappa,
            out=out_buf,
            in_place=False,
            **kw,
        )

        np.nan_to_num(out_buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out_buf

    i = 0
    while i < len(emissions):
        op = emissions[i]
        if not isinstance(op, dict):
            raise TypeError(f"Each emission must be a dict, got {type(op).__name__} at index {i}")
        if "ID" not in op:
            raise KeyError(f"Emission at index {i} missing 'ID'")

        fid = _to_fid(op["ID"])

        if fid == "divide":
            if i + 1 >= len(emissions):
                raise ValueError("Found {'ID':'divide'} but no subsequent op to produce denominator")

            denom_op = emissions[i + 1]
            numerator = work.copy()
            denom = _apply_one(denom_op, work, buf)

            work.fill(0.0)
            np.divide(numerator, denom, out=work, where=(denom != 0.0))
            np.nan_to_num(work, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            i += 2
            continue

        _apply_one(op, work, buf)
        work, buf = buf, work
        i += 1

    out_full[:valid_len] = work[:, 0]
    return out_full


def generate_raw_emission_v2(Population, target_idx, emissions, offset):
    """
    Broad, emission-driven target builder.

    Base target:
      tvec_offset[i] = Population._X_inst[i + offset, target_idx]   (future-aligned)
      valid region is i = 0..N-offset-1
      output is length N with NaN tail of length `offset`.

    New capabilities:
      - Each emission dict may optionally provide:
          * "x": "emit" (default) or "tvec" or a scalar/array
          * "alpha": scalar/array or "tvec" or "emit"
          * "offset": bool (default True) controlling how "tvec" is resolved:
                True  -> use future-aligned tvec_offset
                False -> use current-time tvec_raw (no forward shift)
          * "delta1","delta2","kappa","min_count": scalar constants
      - Special op: {"ID": "divide"} divides current emission by the output of
        the *next* op dict (which is evaluated to produce a denominator series).

    Examples
    --------
    # current emission (tvec_offset) minus the raw (unshifted) tvec:
    emissions = [{"ID": 5, "alpha": "tvec", "offset": False}]

    # divide everything so far by STD(tvec_raw):
    emissions = [
        {"ID": 14, "delta1": 20},          # EMA on tvec_offset (default x=emit)
        {"ID": "divide"},
        {"ID": "18", "x": "tvec", "offset": False, "delta1": 50}  # denom from tvec_raw
    ]
    """
    # ------------------ validate ------------------
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")

    X = Population._X_inst
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("Population._X_inst must be a 2D numpy ndarray")

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
        raise TypeError("emissions must be a list/tuple of dicts")

    # ------------------ build tvecs (valid region only) ------------------
    base = np.asarray(X[:, target_idx], dtype=np.float32)

    valid_len = N - offset
    out_full = np.full(N, np.nan, dtype=np.float32)

    if valid_len <= 0:
        return out_full

    # future-aligned target (the default "tvec")
    if offset == 0:
        tvec_offset_1d = base
    else:
        tvec_offset_1d = base[offset:]  # length = valid_len

    # raw (no forward shift) tvec, aligned to the same valid_len
    tvec_raw_1d = base[:valid_len]

    tvec_offset = np.ascontiguousarray(tvec_offset_1d.reshape(valid_len, 1), dtype=np.float32)
    tvec_raw    = np.ascontiguousarray(tvec_raw_1d.reshape(valid_len, 1), dtype=np.float32)

    # working emission starts as the future-aligned target
    work = tvec_offset.copy()
    buf  = np.empty_like(work)

    # ------------------ helpers ------------------
    def _to_fid(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "divide":
                return "divide"
            # numeric string like "18"
            if s.isdigit():
                return int(s)
            raise ValueError(f"Unrecognized ID string: {v!r}")
        return int(v)

    def _resolve_series(spec, use_offset_tvec, current_work):
        """
        Turn a spec into an (valid_len,1) float32 matrix.
        spec may be:
          - "emit" / "work" / None -> current_work
          - "tvec" -> tvec_offset or tvec_raw (depending on use_offset_tvec)
          - scalar -> broadcast constant series
          - 1D array length valid_len -> reshape to (valid_len,1)
          - 2D array shape (valid_len,1) -> use as-is
        """
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("emit", "work")):
            return current_work

        if isinstance(spec, str) and spec.strip().lower() == "tvec":
            return tvec_offset if use_offset_tvec else tvec_raw

        if np.isscalar(spec):
            # broadcast constant into a column
            return np.full((valid_len, 1), float(spec), dtype=np.float32)

        arr = np.asarray(spec)
        if arr.ndim == 1:
            if arr.shape[0] != valid_len:
                raise ValueError(f"1D series must have length {valid_len}, got {arr.shape[0]}")
            return np.ascontiguousarray(arr.astype(np.float32, copy=False).reshape(valid_len, 1))
        if arr.ndim == 2:
            if arr.shape != (valid_len, 1):
                raise ValueError(f"2D series must be shape {(valid_len,1)}, got {arr.shape}")
            return np.ascontiguousarray(arr.astype(np.float32, copy=False))
        raise ValueError("Series spec must be scalar, 1D (valid_len,), or 2D (valid_len,1)")

    def _scalar_const(name, v, cast=float):
        if v is None:
            return None
        a = np.asarray(v)
        if a.ndim != 0:
            raise TypeError(f"'{name}' must be a scalar constant")
        return cast(a.item())

    def _apply_one(op_dict, current_work, out_buf):
        """
        Evaluate a single transform op dict into out_buf (valid_len,1).
        Returns the output matrix (a view/alias of out_buf).
        """
        if not isinstance(op_dict, dict):
            raise TypeError(f"Each emission must be a dict, got {type(op_dict).__name__}")

        if "ID" not in op_dict:
            raise KeyError("Emission dict missing required key 'ID'")

        fid = _to_fid(op_dict["ID"])
        if fid == "divide":
            raise ValueError("Internal error: _apply_one called on 'divide' op")

        # how to resolve "tvec" inside this op
        use_offset_tvec = bool(op_dict.get("offset", True))

        # x input (default = current emission)
        x_in = _resolve_series(op_dict.get("x", None), use_offset_tvec, current_work)

        # alpha can be scalar/series, including "tvec" or "emit"
        alpha_in = None
        if "alpha" in op_dict:
            a_spec = op_dict["alpha"]
            if isinstance(a_spec, str) and a_spec.strip().lower() in ("tvec", "emit", "work"):
                alpha_in = _resolve_series(a_spec, use_offset_tvec, current_work)
            else:
                # scalar or array
                if np.isscalar(a_spec):
                    alpha_in = float(a_spec)
                else:
                    alpha_in = _resolve_series(a_spec, use_offset_tvec, current_work)

        # scalar params
        delta1 = _scalar_const("delta1", op_dict.get("delta1", None), cast=int)
        delta2 = _scalar_const("delta2", op_dict.get("delta2", None), cast=int)
        kappa  = _scalar_const("kappa",  op_dict.get("kappa",  None), cast=float)

        kw = {}
        if "min_count" in op_dict:
            kw["min_count"] = _scalar_const("min_count", op_dict.get("min_count"), cast=int)

        # AGR/COR require alpha matrix; broadcast if a scalar slipped in
        if fid in (20, 21):
            if alpha_in is None:
                raise ValueError(f"Emission ID {fid} requires 'alpha'")
            if np.isscalar(alpha_in):
                alpha_in = np.full(x_in.shape, float(alpha_in), dtype=np.float32)
            else:
                # ensure correct matrix shape
                if not (isinstance(alpha_in, np.ndarray) and alpha_in.shape == x_in.shape):
                    raise ValueError(f"Emission ID {fid}: alpha must be shape {x_in.shape}, got {getattr(alpha_in,'shape',None)}")

        # run op
        transform_ops.apply(
            fid,
            x_in,
            alpha=alpha_in,
            delta1=delta1,
            delta2=delta2,
            kappa=kappa,
            out=out_buf,
            in_place=False,
            **kw,
        )

        # sanitize (matches your instantiation philosophy)
        np.nan_to_num(out_buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out_buf

    # ------------------ main emission loop ------------------
    i = 0
    while i < len(emissions):
        op = emissions[i]
        if not isinstance(op, dict):
            raise TypeError(f"Each emission must be a dict, got {type(op).__name__} at index {i}")
        if "ID" not in op:
            raise KeyError(f"Emission at index {i} missing 'ID'")

        fid = _to_fid(op["ID"])

        if fid == "divide":
            if i + 1 >= len(emissions):
                raise ValueError("Found {'ID':'divide'} but no subsequent op to produce denominator")

            denom_op = emissions[i + 1]
            # compute denominator into buf
            denom = _apply_one(denom_op, work, buf)

            # safe divide: where denom != 0 else 0
            np.divide(
                work,
                denom,
                out=work,
                where=(denom != 0.0),
            )
            np.nan_to_num(work, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            i += 2
            continue

        # normal op -> output into buf, then swap
        _apply_one(op, work, buf)
        work, buf = buf, work
        i += 1

    out_full[:valid_len] = work[:, 0]
    return out_full


def generate_raw_emission_v1(Population, target_idx, emissions, offset):
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

import numpy as np

def evaluate_participation(m: float, n: float, p, q):
    """
    Vectorized evaluation of breadth b(p,q) and depth d(p) for arrays p and q.

    Rules:
      - If p == 0: set b = -10.0 for that sample (avoid denom issues). d computed normally.
      - If p >= 2n: force d = 10.0 for that sample (avoid log-domain issue in p>n branch).
    """
    e = np.e

    # ---- basic validation ----
    m = float(m); n = float(n)
    if not (n >= m):
        raise ValueError(f"Constraint violated: need n >= m, got n={n}, m={m}.")
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape. Got p={p.shape}, q={q.shape}.")
    if p.ndim != 1:
        raise ValueError(f"Expected 1D vectors for p and q (shape (G,)). Got p.ndim={p.ndim}.")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("p and q must be nonnegative (counts-like).")

    mask_zero = (p == 0)
    mask_nz = ~mask_zero

    # Enforce p>=q only where p>0 (p==0 handled via b=-10)
    if np.any(mask_nz & (p < q)):
        bad = np.where(mask_nz & (p < q))[0][:10]
        raise ValueError(f"Constraint violated: need p >= q elementwise. Example bad indices: {bad}.")

    # ---- constants ----
    f = m / n
    pf = p * f

    # ---- depth d(p) ----
    d = np.zeros_like(p, dtype=float)

    mask_lo = p < m
    d[mask_lo] = np.log(2.0 - (p[mask_lo] / m))**2

    # p>n branch split: safe log region vs forced cap
    mask_cap = p >= (2.0 * n)
    d[mask_cap] = 10.0

    mask_hi_safe = (p > n) & (~mask_cap)  # n < p < 2n
    d[mask_hi_safe] = np.log(2.0 - (p[mask_hi_safe] / n))**2

    # ---- breadth b(p,q) ----
    b = np.zeros_like(p, dtype=float)

    # p == 0 => force b to -10
    b[mask_zero] = -10.0

    if np.any(mask_nz):
        p_nz = p[mask_nz]
        q_nz = q[mask_nz]
        pf_nz = pf[mask_nz]

        denom = p_nz * f * e  # safe since p_nz > 0 and f,e > 0
        b_nz = np.zeros_like(p_nz, dtype=float)

        mask_b1 = q_nz < pf_nz
        mask_b2 = ~mask_b1

        # b1
        b_nz[mask_b1] = np.log(((e - 1.0) * q_nz[mask_b1] + f) / denom[mask_b1])**2

        # b2
        log_term = np.log(((e - 1.0) * q_nz[mask_b2] + pf_nz[mask_b2]) / denom[mask_b2])**2
        C = 1.0 - (np.log(((e - 1.0) + f) / (f * e))**2)

        if np.isclose(1.0 - f, 0.0):
            # m == n case: requires q == p in branch2
            if np.any(mask_b2 & (q_nz != p_nz)):
                bad = np.where(mask_b2 & (q_nz != p_nz))[0][:10]
                raise ValueError(f"When m==n, branch2 requires q==p. Example bad indices: {bad}.")
            quad = np.zeros_like(q_nz[mask_b2])
        else:
            quad = ((q_nz[mask_b2] - pf_nz[mask_b2]) / (p_nz[mask_b2] * (1.0 - f)))**2

        b_nz[mask_b2] = log_term + C * quad
        b[mask_nz] = b_nz

    return b, d

def evaluate_population(
    population:_I.Population,
    solver:Solver = None,
    slack:float=0.00,
    complexity:str='log_parsimony',
    metric:str='heavensent'
):
    if(solver is None):
        solver = Solver(population, offset=6, t_mode='AD', emission = [
        {"ID": 5, "alpha": {"ID": 3, "x": "tvec", "delta1": 6*4, "offset": False}},
        {"ID": "divide"},
        {"ID": 18, "x": "tvec", "delta1": 6*4, "offset": False},
        ], AD_cond=('gt', 2))
        print(f"WARNING: Using default solver. double check params!!!"
              f" why are you using my software RRRRHHHAAAAAAAAA")
    raw_emission, evaluation_mask, anomaly_mask = solver.solve(population)

    # always-participating benchmark return (market baseline)
    _em = np.asarray(raw_emission, dtype=np.float32).reshape(-1)
    _mask = np.asarray(evaluation_mask).astype(bool, copy=False).reshape(-1)
    np.nan_to_num(_em, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(_em[_mask].sum())

    c = evaluate_depth(population)
    X_p = resolve_population_signs(population)
    m, n= resolve_anomaly_mn(anomaly_mask)
    p, q= resolve_population_pq(population, X_p)
    b, d= evaluate_participation(m, n, p, q)
    R = evaluate_return(population, X_p, raw_emission, evaluation_mask)
    
    #meow
    #i should come back to comment out this function before I get to far away from it
    F = R - (1 - slack) * np.abs(R) * (b + d)

    parsimony_coef = solve_auto_parsimony(population, c, F)
    print("parsimony_coef: ", parsimony_coef)

    match(complexity):
        case 'log_parsimony':
            F -= np.clip(parsimony_coef, 0, None) * np.log(c+1)
        case 'parsimony':
            F -= np.clip(parsimony_coef, 0, None) * c
        case 'None':
            pass
            #F is ready to go

    return {"F":F,"R":R,"mu":mu,"parsimony_coef":parsimony_coef,
            "m":m,"n":n,"p":p,"q":q,"b":b,"d":d,"c":c}


def evaluate_return(population, X_p, raw_emission, evaluation_mask):
    """
    Compute per-gene cumulative return from participation.

    For each gene index g in population._G_idx:
        R[g] = sum(raw_emission[t] for t where evaluation_mask[t] and X_p[t, g])

    Parameters
    ----------
    population : object
        Must have attribute `_G_idx` (indices of gene columns to evaluate).
    X_p : array-like, shape (T, G)
        Boolean (or 0/1) participation/prediction matrix.
    raw_emission : array-like, shape (T,)
        Emission/return series aligned with X_p along time.
    evaluation_mask : array-like, shape (T,)
        Boolean (or 0/1) mask for time points allowed to be evaluated.

    Returns
    -------
    R : np.ndarray, shape (G,), dtype=float32
        Cumulative returns per gene (zeros for genes not in _G_idx).
    """
    if not hasattr(population, "_G_idx"):
        raise AttributeError("population must have attribute '_G_idx'")

    X_p = np.asarray(X_p)
    if X_p.ndim != 2:
        raise ValueError("X_p must be 2D with shape (T, G)")

    T, G = X_p.shape

    raw_emission = np.asarray(raw_emission, dtype=np.float32).reshape(-1)
    evaluation_mask = np.asarray(evaluation_mask).reshape(-1)

    if raw_emission.shape[0] != T:
        raise ValueError(f"raw_emission length {raw_emission.shape[0]} must equal T={T}")
    if evaluation_mask.shape[0] != T:
        raise ValueError(f"evaluation_mask length {evaluation_mask.shape[0]} must equal T={T}")

    # Normalize mask + sanitize emissions
    eval_mask = evaluation_mask.astype(bool, copy=False)
    weights = raw_emission.copy()
    # any NaN/Inf is treated as 0 contribution
    np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    weights[~eval_mask] = 0.0  # zero out non-evaluable times

    # Indices to evaluate
    idx = np.asarray(population._G_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros(G, dtype=np.float32)

    if np.any(idx < 0) or np.any(idx >= G):
        bad = idx[(idx < 0) | (idx >= G)][:10]
        raise IndexError(f"population._G_idx contains out-of-bounds indices (first bad: {bad})")

    # Ensure boolean participation
    Xp_bool = X_p.astype(bool, copy=False)

    R = np.zeros(G, dtype=np.float32)

    # Blocked matmul to avoid huge temporaries if idx is big
    # (still only evaluates columns in _G_idx)
    block = 1024
    w = weights.astype(np.float32, copy=False)

    for s in range(0, idx.size, block):
        j = idx[s:s + block]
        # (T, B) @ (T,) -> (B,)
        # bool will upcast; that's fine
        R[j] = (Xp_bool[:, j].T @ w).astype(np.float32, copy=False)

    return R


def evaluate_depth(population) -> np.ndarray:
    """
    Compute ancestor-count depth for every gene in population._G_idx.

    For each gene g, this calls family_tree_indices(...) and stores the number
    of unique ancestor genes returned, including the gene itself.

    Returns
    -------
    np.ndarray
        Shape (G,), where G = population._G_idx.shape[0].
        depth[g] = number of parent genes in g's family tree, including self.
    """
    instructions = np.asarray(population._instructions)
    G = instructions.shape[0]

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must have shape (G, 11) (need at least cols 0..9).")
    if instructions.shape[0] < G:
        raise ValueError("population._instructions has fewer rows than population._G_idx dim0.")

    depth = np.zeros(G, dtype=np.int64)

    for g in population._G_idx.astype(int):
        depth[g] = _I.family_tree_indices(
            instructions,
            g,
            include_self=True,
        ).size

    return depth

import numpy as np

def solve_auto_parsimony(population, complexity, F):
    """
    Compute a gplearn-style auto parsimony coefficient using only legal indices.

    Parameters
    ----------
    population : object
        Must have a 1D integer array `population._G_idx` of legal indices.
    complexity : np.ndarray, shape (G,)
        1D array of complexity values per individual.
        Usually this is total node count. It can also be depth if that is
        what you want to penalize.
    F : np.ndarray, shape (G,)
        1D array of raw fitness scores per individual.

    Returns
    -------
    float
        Auto parsimony coefficient:
            cov(complexity_legal, fitness_legal) / var(complexity_legal)

        This mirrors the gplearn idea:
            np.cov(length, fitness)[0, 1] / np.var(length)
    """
    idx = np.asarray(population._G_idx, dtype=int)
    complexity = np.asarray(complexity, dtype=float)
    F = np.asarray(F, dtype=float)

    if complexity.ndim != 1 or F.ndim != 1:
        raise ValueError("complexity and F must be 1D numpy arrays.")
    if complexity.shape[0] != F.shape[0]:
        raise ValueError("complexity and F must have the same length.")
    if idx.ndim != 1:
        raise ValueError("population._G_idx must be a 1D index array.")
    if idx.size == 0:
        return 0.0
    if np.any(idx < 0) or np.any(idx >= complexity.shape[0]):
        raise IndexError("population._G_idx contains out-of-bounds indices.")

    x = complexity[idx]
    #clipping at zero to avoid meaningless
    y = np.clip(F[idx], 0, None)

    if x.size < 2:
        return 0.0

    var_x = np.var(x)
    if var_x == 0.0 or not np.isfinite(var_x):
        return 0.0

    coeff = np.cov(x, y)[0, 1] / var_x
    return 0.0 if not np.isfinite(coeff) else np.clip(coeff, 0, None)

def resolve_anomaly_mn(x: np.ndarray) -> tuple[int, int]:
    """
    Return (m, n) for a 1D boolean numpy array.

    n = total number of True values
    m = total number of contiguous True groups

    Example:
        [False, True, True, False, True, False] -> (2, 3)
    """
    x = np.asarray(x, dtype=np.bool_)
    if x.ndim != 1:
        raise ValueError("resolve_anomaly_mn() expects a 1D array of shape (N,)")

    n = int(x.sum())

    if x.size == 0:
        return 0, 0

    m = int(x[0]) + int(np.sum(x[1:] & ~x[:-1]))
    return m, n


def resolve_population_signs(population) -> np.ndarray:
    """
    Build a boolean mask M with same shape as population._X_inst (N, G).

    M is all-False except in columns population._G_idx, where:
      M[:, g] = (population._X_inst[:, g] >= 0)

    Intended for fast boolean indexing like: A[M].sum()
    """
    X = population._X_inst
    idx = np.asarray(population._G_idx, dtype=np.intp)

    #print(np.where(population._X_inst[:,10:]>0))
    #print()
    ##print(idx)
    #print(np.unique_counts(X[:, idx] > 0))

    M = np.zeros(X.shape, dtype=np.bool_)

    if idx.size:
        M[:, idx] = (X[:, idx] > 0)

    return M

import numpy as np

def resolve_population_pq(population, X_p) -> tuple[np.ndarray, np.ndarray]:
    """
    Build p and q vectors (shape (G,)) for a population.

    For each gene column g in population._G_idx:
      p[g] = sum(X_p[:, g])  (total positive predictions, X_p is 0/1 or bool)
      q[g] = number of contiguous runs of 1s in X_p[:, g]  (unique prediction episodes)

    Other columns (not in _G_idx) are left as 0.

    Returns:
      p, q : int64 arrays, shape (G,)
    """
    X = population._X_inst
    idx = np.asarray(population._G_idx, dtype=np.intp)

    Xp = np.asarray(X_p)
    if Xp.shape != X.shape:
        raise ValueError(f"X_p must have shape {X.shape}, got {Xp.shape}.")

    N, G = X.shape
    p = np.zeros(G, dtype=np.int64)
    q = np.zeros(G, dtype=np.int64)

    if idx.size == 0 or N == 0:
        return p, q

    # Work on only the gene columns
    B = Xp[:, idx].astype(np.bool_, copy=False)  # (N, len(idx))

    # p = total ones per column
    p_vals = B.sum(axis=0, dtype=np.int64)

    # q = number of runs of ones per column: start with first row + count rising edges
    if N == 1:
        q_vals = B[0].astype(np.int64, copy=False)
    else:
        rises = np.logical_and(~B[:-1], B[1:]).sum(axis=0, dtype=np.int64)
        q_vals = B[0].astype(np.int64, copy=False) + rises

    # Write into full-length vectors
    p[idx] = p_vals
    q[idx] = q_vals

    return p, q


def visualize_participation_surfaces(
    *,
    m: float = 20.0,
    n: float = 100.0,
    num: int = 160,
    e: float = np.e,
    mode: str = "surface",          # "surface" or "imshow"
    which: str = "all",             # "b", "d", "bd", or "all"
    ceil: int = 0,
    plot_mn: bool = False,          # NEW: plot m and n reference lines
):
    """
    Visualize b(p,q), d(p), and combined (b+d) over a (p,q) grid, masking out invalid q>p.

    If plot_mn=True:
      - plot p = n (line perpendicular to p axis; i.e., vertical line in p-q plane)
      - plot q = m (line perpendicular to q axis; i.e., horizontal line in p-q plane)
    """
    import matplotlib.pyplot as plt

    if not (n >= m):
        raise ValueError("Need n >= m.")

    # Keep p < 2n to avoid depth-log domain issues on the p>n branch
    p_vals = np.linspace(1.0, 1.9 * n, num)
    q_vals = np.linspace(0.0, 1.9 * n, num)
    P, Q = np.meshgrid(p_vals, q_vals)

    valid = (Q <= P)

    # Evaluate only valid points to respect p>=q
    p_flat = P[valid].ravel()
    q_flat = Q[valid].ravel()
    B_flat, D_flat = evaluate_participation(m, n, p_flat, q_flat)

    # Put results back into grids with NaNs elsewhere
    B = np.full_like(P, np.nan, dtype=float)
    D = np.full_like(P, np.nan, dtype=float)
    BD = np.full_like(P, np.nan, dtype=float)

    B[valid] = B_flat
    D[valid] = D_flat
    BD[valid] = B_flat + D_flat

    if ceil > 0:
        B = np.clip(B, None, ceil)
        D = np.clip(D, None, ceil)
        BD = np.clip(BD, None, ceil)

    want_b = which in ("b", "all")
    want_d = which in ("d", "all")
    want_bd = which in ("bd", "all")

    if mode not in ("surface", "imshow"):
        raise ValueError("mode must be 'surface' or 'imshow'.")

    if mode == "surface":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def _surface(Z, title, zlab):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(P, Q, Z, linewidth=0, antialiased=True)
            ax.set_title(title)
            ax.set_xlabel("p")
            ax.set_ylabel("q")
            ax.set_zlabel(zlab)

            if plot_mn:
                # p = n plane slice: line at p=n, varying q, at z=0 (reference)
                ax.plot([n] * len(q_vals), q_vals, np.zeros_like(q_vals), linestyle="--")
                # q = m plane slice: line at q=m, varying p, at z=0 (reference)
                ax.plot(p_vals, [m] * len(p_vals), np.zeros_like(p_vals), linestyle="--")

        if want_b:
            _surface(B, f"b(p,q) (m={m:g}, n={n:g})", "b")
        if want_d:
            _surface(D, f"d(p) (m={m:g}, n={n:g})", "d")
        if want_bd:
            _surface(BD, f"b(p,q)+d(p) (m={m:g}, n={n:g})", "b+d")

        plt.show()

    else:  # mode == "imshow"
        extent = [p_vals.min(), p_vals.max(), q_vals.min(), q_vals.max()]

        def _imshow(Z, title):
            fig, ax = plt.subplots()
            im = ax.imshow(
                Z, cmap="Reds",
                origin="lower",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xlabel("p")
            ax.set_ylabel("q")

            if plot_mn:
                # p = n (vertical line)
                ax.axvline(n, linestyle="-", c='black',alpha=0.25)
                # q = m (horizontal line)
                ax.axhline(m, linestyle="-", c='black',alpha=0.25)

            plt.colorbar(im, ax=ax)

        if want_b:
            _imshow(B, f"b(p,q) heatmap (m={m:g}, n={n:g})")
        if want_d:
            _imshow(D, f"d(p) heatmap (m={m:g}, n={n:g})")
        if want_bd:
            _imshow(BD, f"b(p,q)+d(p) heatmap (m={m:g}, n={n:g})")

        plt.show()