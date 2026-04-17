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
        emission	:	list=	[
        {"ID": 5, "alpha": {"ID": 3, "x": "tvec", "delta1": 6*4, "offset": False}},
        {"ID": "divide"},
        {"ID": 18, "x": "tvec", "delta1": 6*4, "offset": False},
        ],
        offset      :   int=    6,
        AD_cond		:	tuple=	('gt', 2),
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
        Population: _I.Population,
        chunk_num: int | None = None,
    ):
        '''
        Solves target vector and mask on either the full matrix or one selected chunk.
        raw emission is used directly in computation for some form of emission prediction,
        where raw emission[solution mask] is used in cost based evaluation function.
        '''
        if self._tmode == 'AD' or self._tmode == 'RE':
            raw_emission = generate_raw_emission(
                Population,
                self._tidx,
                self._emission,
                self._offset,
                chunk_num=chunk_num,
            )

            if isinstance(self._AD_cond, tuple) and len(self._AD_cond) >= 1:
                if self._AD_cond[0] == 'lt' or self._AD_cond[0] == 'le':
                    raw_emission *= -1

            evaluation_mask = generate_evaluation_mask(
                Population,
                self._offset,
                chunk_num=chunk_num,
            )
            anomaly_mask = generate_anomaly_mask(raw_emission, self._AD_cond)
        else:
            raise NotImplementedError(f'Target Mode of "{self._tmode}" is not supported at this moment.')

        return raw_emission, evaluation_mask, anomaly_mask

        

#first candidate evaluation function should be a light modular somewhat function that containst the first solution type selection
import transform_ops as _OPS

def evaluate(
    population      :   _I.Population,
    solver          :   Solver,
    chunk_num       :   int | None = None,
    slack           :   float = 0,
    complexity      :   str = 'log_parsimony',
    metric          :   str = 'heavensent'
):
    match(population._structure):
        case 'Intraday':
            instantiation_stats = _I.instantiate_from_ops_chunked_intraday(
                population,
                transform_ops=_OPS,
                chunk_num=chunk_num,
                chunk_B=8,
            )
        case _:
            raise NotImplementedError('Did not implement other structure other than intraday should be easy ask logan.')

    evaluation = evaluate_population(
        population,
        solver,
        chunk_num=chunk_num,
        slack=slack,
        complexity=complexity,
        metric=metric,
    )

    return evaluation, instantiation_stats


import numpy as np
import transform_ops

def generate_raw_emission(Population, target_idx, emissions, offset, chunk_num: int | None = None):
    """
    Broad, emission-driven target builder.

    Chunk behavior
    --------------
    - chunk_num is None -> operate on full Population._X_inst
    - chunk_num is int  -> operate only on Population rows belonging to that chunk

    Base target:
      tvec_offset[i] = selected_X[i + offset, target_idx]
      valid region is i = 0..N-offset-1
      output is length N_selected with NaN tail of length `offset`.
    """
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")

    X_full = Population._X_inst
    if not isinstance(X_full, np.ndarray) or X_full.ndim != 2:
        raise TypeError("Population._X_inst must be a 2D numpy ndarray")

    row_lo, row_hi = _resolve_chunk_bounds(Population, chunk_num)
    X = X_full[row_lo:row_hi, :]
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

    if not _emission_depends_on_future(emissions, offset):
        raise ValueError(
            "Emission does not depend on future data (offset-aligned tvec). "
            "This target would be non-forward / easily solvable. "
            "Ensure offset>0 and that the pipeline uses 'emit' (default) or tvec with offset=True."
        )

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
        if isinstance(alpha_spec, dict):
            return _eval_nested_alpha_op(alpha_spec)

        if isinstance(alpha_spec, str) and alpha_spec.strip().lower() in ("tvec", "emit", "work"):
            return _resolve_series(alpha_spec, use_offset_tvec, current_work)

        if np.isscalar(alpha_spec):
            return float(alpha_spec)

        return _resolve_series(alpha_spec, use_offset_tvec, current_work)

    def _remove_buggers(x, thresh=100.0):
        x[np.abs(x) > thresh] = 0
        return x

    def _eval_nested_alpha_op(op_dict):
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
        _OPS.apply(
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

        x_spec = op_dict.get("x", None)
        if isinstance(x_spec, dict):
            x_in = _eval_nested_alpha_op(x_spec)
        else:
            x_in = _resolve_series(x_spec, use_offset_tvec, current_work)

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

        _OPS.apply(
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
    return _remove_buggers(out_full)

def generate_evaluation_mask(Population: _I.Population, offset, chunk_num: int | None = None):
    """
    Build a boolean mask showing which rows can be validly evaluated
    for a forward-looking target with the given offset, without wrapping
    across intraday day boundaries.

    Chunk behavior
    --------------
    - chunk_num is None -> return mask for full matrix
    - chunk_num is int  -> return mask only for the selected chunk rows
    """
    if not hasattr(Population, "_X_inst"):
        raise AttributeError("Population must have attribute '_X_inst'")
    if not hasattr(Population, "_T_idx"):
        raise AttributeError("Population must have attribute '_T_idx'")

    X_full = Population._X_inst
    if not isinstance(X_full, np.ndarray):
        raise TypeError("Population._X_inst must be a numpy ndarray")
    if X_full.ndim != 2:
        raise ValueError("Population._X_inst must be 2D")

    X, _, _ = _get_chunk_view(Population, chunk_num)

    try:
        if Population._time_terminals:
            if hasattr(Population, '_tod_idx') and Population._tod_idx is not None:
                tod_idx = int(Population._tod_idx)
            else:
                tod_idx = int(Population._T_idx[-2])
        else:
            raise NotImplementedError(
                'In generating evaluation mask, time terminals is not true in pop prior, '
                'cannot grab time of day.'
            )
    except Exception as e:
        raise ValueError("Population._T_idx must be indexable and contain [-2]") from e

    N, G = X.shape
    if tod_idx < 0 or tod_idx >= G:
        raise IndexError(f"time-of-day column index {tod_idx} out of bounds for G={G}")

    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be >= 0")

    tod = np.asarray(X[:, tod_idx])

    if offset == 0:
        return np.ones(N, dtype=bool)

    if offset >= N:
        return np.zeros(N, dtype=bool)

    mask = np.zeros(N, dtype=bool)
    mask[:-offset] = tod[offset:] >= tod[:-offset]
    return mask

def generate_raw_emission_old(Population, target_idx, emissions, offset):
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
    
    if not _emission_depends_on_future(emissions, offset):
        raise ValueError(
            "Emission does not depend on future data (offset-aligned tvec). "
            "This target would be non-forward / easily solvable. "
            "Ensure offset>0 and that the pipeline uses 'emit' (default) or tvec with offset=True."
        )

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
    

    def _remove_buggers(x, thresh=100.0):

        x[np.abs(x) > thresh] = 0
        return x

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

        x_spec = op_dict.get("x", None)
        if isinstance(x_spec, dict):
            x_in = _eval_nested_alpha_op(x_spec)  # reuse your existing nested-op resolver (it’s tvec-rooted)
        else:
            x_in = _resolve_series(x_spec, use_offset_tvec, current_work)

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
    return _remove_buggers(out_full)


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


def generate_evaluation_mask_v3(Population:_I.Population, offset):
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

    mask_min = p < n / 100
    d[mask_min] = 1

    # p>n branch split: safe log region vs forced cap
    mask_cap = p >= (2.0 * n)
    d[mask_cap] = 10.0

    mask_hi_safe = (p > n) & (~mask_cap)  # n < p < 2n
    d[mask_hi_safe] = np.log(2.0 - (p[mask_hi_safe] / n))**2

    # ---- breadth b(p,q) ----
    b = np.zeros_like(p, dtype=float)

    # p == 0 => force b to -10
    #b[mask_zero] = -10.0
    #there is no way....... huge bug holding this project up
    # p == 0 => force b to 10
    b[mask_zero] = 10.0

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

def _resolve_chunk_bounds(population: _I.Population, chunk_num: int | None = None) -> tuple[int, int]:
    """
    Resolve the selected row bounds [lo, hi) from population.

    Behavior is intentionally split entirely by chunk_num:
      - chunk_num is None -> full matrix
      - chunk_num is int  -> selected chunk from population metadata
    """
    if not hasattr(population, '_X_inst'):
        raise AttributeError("population must have attribute '_X_inst'")

    N = int(population._X_inst.shape[0])

    if chunk_num is None:
        return 0, N

    chunk_num = int(chunk_num)

    if hasattr(population, 'get_chunk_bounds'):
        lo, hi = population.get_chunk_bounds(chunk_num)
    elif hasattr(population, 'get_chunk_slice'):
        sl = population.get_chunk_slice(chunk_num)
        lo = 0 if sl.start is None else int(sl.start)
        hi = N if sl.stop is None else int(sl.stop)
    elif hasattr(population, '_chunk_row_bounds'):
        bounds = np.asarray(population._chunk_row_bounds, dtype=np.int64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("population._chunk_row_bounds must be shape (n_chunks, 2)")
        if chunk_num < 0 or chunk_num >= bounds.shape[0]:
            raise IndexError(
                f'chunk_num={chunk_num} out of bounds for _chunk_row_bounds with '
                f'{bounds.shape[0]} chunks.'
            )
        lo, hi = bounds[chunk_num]
    else:
        raise AttributeError(
            "population must provide get_chunk_bounds(chunk_num), get_chunk_slice(chunk_num), "
            "or _chunk_row_bounds"
        )

    lo = int(lo)
    hi = int(hi)

    if lo < 0 or hi > N or lo >= hi:
        raise ValueError(f'Invalid chunk bounds [{lo}:{hi}) for N={N}.')

    return lo, hi


def _get_chunk_view(population: _I.Population, chunk_num: int | None = None):
    lo, hi = _resolve_chunk_bounds(population, chunk_num)
    return population._X_inst[lo:hi, :], lo, hi

def evaluate_population(
    population: _I.Population,
    solver: Solver = None,
    chunk_num: int | None = None,
    slack: float = 0.00,
    complexity: str = 'log_parsimony',
    metric: str = 'heavensent'
):
    if solver is None:
        solver = Solver(
            population,
            offset=6,
            t_mode='AD',
            emission=[
                {"ID": 5, "alpha": {"ID": 3, "x": "tvec", "delta1": 6*4, "offset": False}},
                {"ID": "divide"},
                {"ID": 18, "x": "tvec", "delta1": 6*4, "offset": False},
            ],
            AD_cond=('gt', 2)
        )
        print(
            "WARNING: Using default solver. double check params!!!"
            " why are you using my software RRRRHHHAAAAAAAAA"
        )

    row_lo, row_hi = _resolve_chunk_bounds(population, chunk_num)
    raw_emission, evaluation_mask, anomaly_mask = solver.solve(population, chunk_num=chunk_num)

    _em = np.asarray(raw_emission, dtype=np.float32).reshape(-1)
    _mask = np.asarray(evaluation_mask).astype(bool, copy=False).reshape(-1)
    np.nan_to_num(_em, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(_em[_mask].sum())

    c = evaluate_depth(population)
    X_p = resolve_population_signs(population, chunk_num=chunk_num)
    m, n = resolve_anomaly_mn(anomaly_mask)
    p, q = resolve_population_pq(population, X_p, chunk_num=chunk_num)
    b, d = evaluate_participation(m, n, p, q)
    R = evaluate_return(population, X_p, raw_emission, evaluation_mask)

    F = R - (1 - slack) * np.abs(R) * (b + d)

    parsimony_coef = solve_auto_parsimony(population, c, F)

    match(complexity):
        case 'log_parsimony':
            F -= np.clip(parsimony_coef, 0, None) * np.log(c + 1)
        case 'parsimony':
            F -= np.clip(parsimony_coef, 0, None) * c
        case 'None':
            pass

    return {
        "chunk_num": chunk_num,
        "chunk_row_bounds": (int(row_lo), int(row_hi)),
        "F": F,
        "R": R,
        "mu": mu,
        "parsimony_coef": parsimony_coef,
        "m": m,
        "n": n,
        "p": p,
        "q": q,
        "b": b,
        "d": d,
        "c": c,
        "svecs": {
            "raw_emission": raw_emission,
            "evaluation_mask": evaluation_mask,
            "anomaly_mask": anomaly_mask,
        },
    }

def evaluate_population_old(
    population:_I.Population,
    solver:Solver = None,
    chunk_num:int|None = None,
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
    #print("parsimony_coef: ", parsimony_coef)

    match(complexity):
        case 'log_parsimony':
            F -= np.clip(parsimony_coef, 0, None) * np.log(c+1)
        case 'parsimony':
            F -= np.clip(parsimony_coef, 0, None) * c
        case 'None':
            pass
            #F is ready to go

    return {"F":F,"R":R,"mu":mu,"parsimony_coef":parsimony_coef,
            "m":m,"n":n,"p":p,"q":q,"b":b,"d":d,"c":c,"svecs":
            {"raw_emission":raw_emission,"evaluation_mask":evaluation_mask,"anomaly_mask":anomaly_mask}
            }


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

def resolve_population_signs(population, chunk_num: int | None = None) -> np.ndarray:
    """
    Build a boolean mask M with same shape as the selected rows of population._X_inst.

    Chunk behavior
    --------------
    - chunk_num is None -> shape (N, G) for full matrix
    - chunk_num is int  -> shape (N_chunk, G) for selected chunk only

    M is all-False except in columns population._G_idx, where:
      M[:, g] = (selected_X[:, g] > 0)
    """
    X, _, _ = _get_chunk_view(population, chunk_num)
    idx = np.asarray(population._G_idx, dtype=np.intp)

    M = np.zeros(X.shape, dtype=np.bool_)

    if idx.size:
        M[:, idx] = (X[:, idx] > 0)

    return M


def resolve_population_signs_old(population) -> np.ndarray:
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

def resolve_population_pq(population, X_p, chunk_num: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build p and q vectors (shape (G,)) for either the full population or one chunk.

    Chunk behavior
    --------------
    - chunk_num is None -> X_p must match full population._X_inst shape
    - chunk_num is int  -> X_p must match selected chunk shape

    For each gene column g in population._G_idx:
      p[g] = sum(X_p[:, g])
      q[g] = number of contiguous runs of 1s in X_p[:, g]
    """
    X_sel, _, _ = _get_chunk_view(population, chunk_num)
    idx = np.asarray(population._G_idx, dtype=np.intp)

    Xp = np.asarray(X_p)
    if Xp.shape != X_sel.shape:
        raise ValueError(f"X_p must have shape {X_sel.shape}, got {Xp.shape}.")

    N, G = X_sel.shape
    p = np.zeros(G, dtype=np.int64)
    q = np.zeros(G, dtype=np.int64)

    if idx.size == 0 or N == 0:
        return p, q

    B = Xp[:, idx].astype(np.bool_, copy=False)
    p_vals = B.sum(axis=0, dtype=np.int64)

    if N == 1:
        q_vals = B[0].astype(np.int64, copy=False)
    else:
        rises = np.logical_and(~B[:-1], B[1:]).sum(axis=0, dtype=np.int64)
        q_vals = B[0].astype(np.int64, copy=False) + rises

    p[idx] = p_vals
    q[idx] = q_vals

    return p, q

def resolve_population_pq_old(population, X_p) -> tuple[np.ndarray, np.ndarray]:
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




import numpy as np

def _emission_depends_on_future(emissions, base_offset):
    """
    Conservative static dependency check.
    Returns True if the final emission stream depends on future-aligned tvec (offset=True)
    or on the default initial future-aligned 'emit' stream.
    """
    base_offset = int(base_offset)
    if base_offset <= 0:
        return False  # no forward info possible

    if emissions is None:
        emissions = []
    if not isinstance(emissions, (list, tuple)):
        raise TypeError("emissions must be a list/tuple of dicts")

    def _is_future_tvec(offset_flag):
        return bool(offset_flag) and base_offset > 0

    def _spec_future_dep(spec, offset_flag, current_emit_dep):
        # spec describes a series source ("emit"/"tvec"/scalar/array)
        if spec is None:
            return current_emit_dep
        if isinstance(spec, str):
            s = spec.strip().lower()
            if s in ("emit", "work"):
                return current_emit_dep
            if s == "tvec":
                return _is_future_tvec(offset_flag)
            return False
        # scalar / arrays are not future by themselves
        return False

    def _nested_alpha_dep(alpha_dict):
        # nested alpha dict must be rooted on tvec; offset controls future-ness
        if not isinstance(alpha_dict, dict):
            return False
        off = bool(alpha_dict.get("offset", True))
        x_spec = alpha_dict.get("x", "tvec")
        if not (isinstance(x_spec, str) and x_spec.strip().lower() == "tvec"):
            # by your rule, nested alpha must root on tvec; treat as not future
            return False
        dep = _is_future_tvec(off)
        # if it itself has nested alpha, include it (conservative)
        if isinstance(alpha_dict.get("alpha", None), dict):
            dep = dep or _nested_alpha_dep(alpha_dict["alpha"])
        return dep

    # The emission stream starts as future-aligned tvec_offset (if base_offset>0)
    emit_dep = True

    i = 0
    while i < len(emissions):
        op = emissions[i]
        if not isinstance(op, dict) or "ID" not in op:
            i += 1
            continue

        fid = op["ID"]
        if isinstance(fid, str) and fid.strip().lower() == "divide":
            # divide: result depends on numerator OR denominator
            if i + 1 >= len(emissions):
                return False
            denom_op = emissions[i + 1]
            denom_off = bool(denom_op.get("offset", True)) if isinstance(denom_op, dict) else True
            denom_x = denom_op.get("x", None) if isinstance(denom_op, dict) else None
            denom_dep = _spec_future_dep(denom_x, denom_off, emit_dep)

            # include denom alpha dependency too
            if isinstance(denom_op, dict) and "alpha" in denom_op:
                a = denom_op["alpha"]
                if isinstance(a, dict):
                    denom_dep = denom_dep or _nested_alpha_dep(a)
                else:
                    denom_dep = denom_dep or _spec_future_dep(a, denom_off, emit_dep)

            emit_dep = emit_dep or denom_dep
            i += 2
            continue

        off = bool(op.get("offset", True))
        x_spec = op.get("x", None)
        x_dep = _spec_future_dep(x_spec, off, emit_dep)

        a_dep = False
        if "alpha" in op:
            a = op["alpha"]
            if isinstance(a, dict):
                a_dep = _nested_alpha_dep(a)
            else:
                a_dep = _spec_future_dep(a, off, emit_dep)

        # conservative: output depends on future if ANY input did
        emit_dep = bool(x_dep or a_dep)
        i += 1

    return emit_dep

import numpy as np

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


import numpy as np
from concurrent.futures import ProcessPoolExecutor


import numpy as np
from concurrent.futures import ProcessPoolExecutor


def evaluate_opg_mcpt_fast(
    population,
    chunk_num,
    good_idx,
    solver_kwargs=None,
    solver_class=None,
    n_sims=2000,
    template_mode="exact",
    temperature=1.0,
    alternative="greater",
    score_mode="ev",           # default changed from return-like behavior to EV
    score_fn=None,             # optional custom fn(sum_value, active_count, block_count) -> scalar
    fill_value=1.0,
    rng=None,
    max_pack_retries=64,
    n_jobs=1,
    batch_size=None,
    return_details=False,
):
    """
    Fast shared-null OPG-MCPT evaluation for one chunk.

    Default behavior is now to work in terms of EV rather than raw return.

    The solver is assembled in project style:
        solver = solver_class(population, **solver_kwargs)
        raw_emission, evaluation_mask, anomaly_mask = solver.solve(
            population,
            chunk_num=chunk_num,
        )

    Parameters
    ----------
    population : _I.Population
        Population object being evaluated.

    chunk_num : int | None
        Chunk index passed into solver.solve(...) and resolve_population_signs(...).

    solver_kwargs : dict | None, default=None
        Keyword arguments used to build the solver instance.

    solver_class : type | None, default=None
        Solver-like class. If None, this function uses Solver from the module.

    n_sims : int, default=2000
        Number of null simulations.

    template_mode : {"exact", "empirical", "softmax"}, default="exact"
        How simulated block lengths are drawn from the observed anomaly geometry.

    temperature : float, default=1.0
        Softmax temperature used only when template_mode == "softmax".

    alternative : {"greater", "less", "two-sided"}, default="greater"
        Tail for empirical p-value calculation.

    score_mode : {"ev", "return", "per_block"}, default="ev"
        Score used for both observed genes and the shared null distribution.

        "ev":
            sum_value / active_count

        "return":
            sum_value

        "per_block":
            sum_value / block_count

    score_fn : callable | None, default=None
        Optional custom function:
            score_fn(sum_value, active_count, block_count) -> scalar
        If provided, this overrides score_mode.

    fill_value : float, default=1.0
        Value assigned to indices not in population._G_idx.

    rng : None, int, or np.random.Generator, default=None
        Random source.

    max_pack_retries : int, default=64
        Number of retries allowed when randomly assigning blocks to legal segments.

    n_jobs : int, default=1
        Number of worker processes used to generate the shared null bank.

    batch_size : int | None, default=None
        Number of simulations per worker batch.

    return_details : bool, default=False
        If True, also return a details dict.

    Returns
    -------
    pvals : np.ndarray, shape (population._max_size,)
        Full-length p-value array. Only population._G_idx locations are evaluated.

    null_scores : np.ndarray, shape (n_sims,)
        Shared null distribution values in the chosen score metric.

    details : dict, optional
        Returned only when return_details=True.
    """
    if solver_class is None:
        solver_class = Solver

    if score_fn is not None and n_jobs > 1:
        # safer than assuming an arbitrary callable is process-pickleable
        raise ValueError("For custom score_fn, use n_jobs=1 unless the function is safely pickleable.")

    rng = np.random.default_rng(rng)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    solver = solver_class(population, **solver_kwargs)
    raw_emission, evaluation_mask, anomaly_mask = solver.solve(
        population,
        chunk_num=chunk_num,
    )

    raw_emission = np.asarray(raw_emission, dtype=np.float64).reshape(-1)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)

    if raw_emission.size != evaluation_mask.size:
        raise ValueError("raw_emission and evaluation_mask must have the same length")
    if raw_emission.size != anomaly_mask.size:
        raise ValueError("raw_emission and anomaly_mask must have the same length")

    # observed gene participation and observed return sums from your existing project functions
    X_p = resolve_population_signs(population, chunk_num=chunk_num)
    R = evaluate_return(population, X_p, raw_emission, evaluation_mask)

    all_gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    if good_idx is None:
        gidx = all_gidx.copy()
    else:
        good_idx = np.asarray(good_idx, dtype=int).reshape(-1)

        # keep only indices that are actually in population._G_idx
        mask = np.isin(good_idx, all_gidx)
        gidx = good_idx[mask]

        if gidx.size == 0:
            pvals = np.full(max_size, fill_value, dtype=np.float64)
            null_scores = np.empty(0, dtype=np.float64)
            details = {
                "gidx": gidx.copy(),
                "all_gidx": all_gidx.copy(),
                "good_idx": good_idx.copy(),
                "observed_scores_full": np.full(max_size, np.nan, dtype=np.float64),
                "null_scores": null_scores,
                "score_label": _opgfast_score_label(score_mode, score_fn),
            }
            if return_details:
                return pvals, null_scores, details
            return pvals, null_scores

    pvals = np.full(max_size, fill_value, dtype=np.float64)

    X_all = _opgfast_extract_gene_matrix(X_p, population)   # aligned to all_gidx
    R_all = _opgfast_extract_gene_vector(R, population)     # aligned to all_gidx

    legal_gene_mask_all = X_all & evaluation_mask[:, None]
    p_all = np.sum(legal_gene_mask_all, axis=0).astype(np.int32)
    q_all = _opgfast_column_run_counts(legal_gene_mask_all)

    # map selected gidx into positions inside all_gidx
    sel_mask = np.isin(all_gidx, gidx)
    sel_pos = np.flatnonzero(sel_mask)

    R_g = R_all[sel_pos]
    p_g = p_all[sel_pos]
    q_g = q_all[sel_pos]

    observed_scores_g = np.empty(gidx.size, dtype=np.float64)
    for j in range(gidx.size):
        observed_scores_g[j] = _opgfast_score_from_components(
            sum_value=float(R_g[j]),
            active_count=int(p_g[j]),
            block_count=int(q_g[j]),
            score_mode=score_mode,
            score_fn=score_fn,
        )

    observed_scores_full = np.full(max_size, np.nan, dtype=np.float64)
    observed_scores_full[gidx] = observed_scores_g

    R_full = np.full(max_size, np.nan, dtype=np.float64)
    R_full[gidx] = R_g

    p_full = np.full(max_size, np.nan, dtype=np.float64)
    p_full[gidx] = p_g

    q_full = np.full(max_size, np.nan, dtype=np.float64)
    q_full[gidx] = q_g

    # OPG template comes from legal anomaly geometry
    opg_mask = anomaly_mask & evaluation_mask
    m_obs, n_obs = resolve_anomaly_mn(opg_mask)
    obs_lengths = _opgfast_true_run_lengths(opg_mask)

    if int(obs_lengths.size) != int(m_obs) or int(obs_lengths.sum()) != int(n_obs):
        m_obs = int(obs_lengths.size)
        n_obs = int(obs_lengths.sum())

    details = {
        "raw_emission": raw_emission,
        "evaluation_mask": evaluation_mask,
        "anomaly_mask": anomaly_mask,
        "opg_mask": opg_mask,
        "X_p": X_p,
        "R_full": R_full,                         # raw return sums, kept for reference
        "p_full": p_full,                         # active counts
        "q_full": q_full,                         # block counts
        "observed_scores_full": observed_scores_full,
        "gidx": gidx.copy(),             # actually evaluated
        "all_gidx": all_gidx.copy(),     # all population._G_idx
        "good_idx": None if good_idx is None else np.asarray(good_idx, dtype=int).copy(),
        "m_obs": int(m_obs),
        "n_obs": int(n_obs),
        "obs_lengths": obs_lengths.copy(),
        "null_scores": None,
        "null_returns": None,                     # alias for backwards convenience
        "score_mode": score_mode,
        "score_label": _opgfast_score_label(score_mode, score_fn),
        "n_jobs": int(n_jobs),
    }

    if m_obs == 0 or n_obs == 0:
        null_scores = np.empty(0, dtype=np.float64)
        details["null_scores"] = null_scores
        details["null_returns"] = null_scores
        if return_details:
            return pvals, null_scores, details
        return pvals, null_scores

    seg_starts, seg_lens = _opgfast_segments_from_mask(evaluation_mask)
    prefix = _opgfast_prefix_sum(raw_emission)
    template = _opgfast_prepare_template_sampler(
        obs_lengths=obs_lengths,
        mode=template_mode,
        temperature=temperature,
    )

    null_scores = _opgfast_generate_null_scores(
        prefix=prefix,
        seg_starts=seg_starts,
        seg_lens=seg_lens,
        template=template,
        n_sims=int(n_sims),
        score_mode=score_mode,
        score_fn=score_fn,
        max_pack_retries=int(max_pack_retries),
        rng=rng,
        n_jobs=int(n_jobs),
        batch_size=batch_size,
    )

    for gi in gidx:
        obs_score = observed_scores_full[gi]
        pvals[gi] = _opgfast_mcpt_p_value(
            observed=obs_score,
            null=null_scores,
            alternative=alternative,
        )

    details["null_scores"] = null_scores
    details["null_returns"] = null_scores

    if return_details:
        return pvals, null_scores, details
    return pvals, null_scores



def evaluate_opg_mcpt_fast_old(
    population,
    chunk_num,
    survivor_idx,
    solver_kwargs=None,
    solver_class=None,
    n_sims=2000,
    template_mode="exact",
    temperature=1.0,
    alternative="greater",
    fill_value=1.0,
    rng=None,
    max_pack_retries=64,
    n_jobs=1,
    batch_size=None,
    return_details=False,
):
    """
    Fast shared-null OPG-MCPT evaluation for one chunk.

    This function builds a single OPG null bank from the anomaly geometry on the
    chosen chunk, then converts each gene's observed return into an empirical
    Monte Carlo p-value against that shared null.

    The solver is assembled in project style:
        solver = solver_class(population, **solver_kwargs)
        raw_emission, evaluation_mask, anomaly_mask = solver.solve(
            population,
            chunk_num=chunk_num,
        )

    Returns
    -------
    pvals : np.ndarray, shape (population._max_size,)
        Full-length p-value array. Only population._G_idx locations are evaluated.

    null_returns : np.ndarray, shape (n_sims,)
        Shared null distribution values from the Monte Carlo simulations.

    details : dict, optional
        Returned only when return_details=True.
    """
    if solver_class is None:
        solver_class = Solver

    rng = np.random.default_rng(rng)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    solver = solver_class(population, **solver_kwargs)
    raw_emission, evaluation_mask, anomaly_mask = solver.solve(
        population,
        chunk_num=chunk_num,
    )

    raw_emission = np.asarray(raw_emission, dtype=np.float64).reshape(-1)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)

    if raw_emission.size != evaluation_mask.size:
        raise ValueError("raw_emission and evaluation_mask must have the same length")
    if raw_emission.size != anomaly_mask.size:
        raise ValueError("raw_emission and anomaly_mask must have the same length")

    X_p = resolve_population_signs(population, chunk_num=chunk_num)
    R = evaluate_return(population, X_p, raw_emission, evaluation_mask)

    gidx = np.asarray(survivor_idx, dtype=int)
    max_size = int(population._max_size)

    pvals = np.full(max_size, fill_value, dtype=np.float64)
    R_full = _opgfast_scatter_gene_values(R, population, fill_value=np.nan)

    opg_mask = anomaly_mask & evaluation_mask
    m_obs, n_obs = resolve_anomaly_mn(opg_mask)
    obs_lengths = _opgfast_true_run_lengths(opg_mask)

    if int(obs_lengths.size) != int(m_obs) or int(obs_lengths.sum()) != int(n_obs):
        m_obs = int(obs_lengths.size)
        n_obs = int(obs_lengths.sum())

    details = {
        "raw_emission": raw_emission,
        "evaluation_mask": evaluation_mask,
        "anomaly_mask": anomaly_mask,
        "opg_mask": opg_mask,
        "X_p": X_p,
        "R_full": R_full,
        "gidx": gidx.copy(),
        "m_obs": int(m_obs),
        "n_obs": int(n_obs),
        "obs_lengths": obs_lengths.copy(),
        "null_returns": None,
        "n_jobs": int(n_jobs),
    }

    if m_obs == 0 or n_obs == 0:
        null_returns = np.empty(0, dtype=np.float64)
        details["null_returns"] = null_returns
        if return_details:
            return pvals, null_returns, details
        return pvals, null_returns

    seg_starts, seg_lens = _opgfast_segments_from_mask(evaluation_mask)
    prefix = _opgfast_prefix_sum(raw_emission)
    template = _opgfast_prepare_template_sampler(
        obs_lengths=obs_lengths,
        mode=template_mode,
        temperature=temperature,
    )

    null_returns = _opgfast_generate_null_returns(
        prefix=prefix,
        seg_starts=seg_starts,
        seg_lens=seg_lens,
        template=template,
        n_sims=int(n_sims),
        max_pack_retries=int(max_pack_retries),
        rng=rng,
        n_jobs=int(n_jobs),
        batch_size=batch_size,
    )

    for gi in gidx:
        obs_return = R_full[gi]
        pvals[gi] = _opgfast_mcpt_p_value(
            observed=obs_return,
            null=null_returns,
            alternative=alternative,
        )

    details["null_returns"] = null_returns

    if return_details:
        return pvals, null_returns, details
    return pvals, null_returns

def _opgfast_score_from_components(
    sum_value,
    active_count,
    block_count,
    score_mode="ev",
    score_fn=None,
):
    """
    Convert a raw sum into the chosen metric.

    score_fn overrides score_mode if provided.
    """
    if score_fn is not None:
        return float(score_fn(sum_value, active_count, block_count))

    mode = str(score_mode).lower()

    if mode == "return":
        return float(sum_value)

    if mode in ("ev", "mean", "avg"):
        return float(sum_value) / active_count if active_count > 0 else 0.0

    if mode == "per_block":
        return float(sum_value) / block_count if block_count > 0 else 0.0

    raise ValueError('score_mode must be "ev", "return", or "per_block", unless score_fn is provided')


def _opgfast_score_label(score_mode="ev", score_fn=None):
    """
    Human-readable label for plots/details.
    """
    if score_fn is not None:
        return "Custom Score"

    mode = str(score_mode).lower()
    if mode == "return":
        return "Return"
    if mode in ("ev", "mean", "avg"):
        return "EV"
    if mode == "per_block":
        return "Return per Block"
    return str(score_mode)


def _opgfast_extract_gene_matrix(X_p, population):
    """
    Return a (T, len(population._G_idx)) boolean matrix aligned to _G_idx order.
    """
    X_p = np.asarray(X_p, dtype=bool)
    gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    if X_p.ndim == 1:
        X_p = X_p[:, None]

    if X_p.shape[1] == gidx.size:
        return X_p

    if X_p.shape[1] == max_size:
        return X_p[:, gidx]

    raise ValueError(
        "X_p must have either len(population._G_idx) columns or population._max_size columns"
    )


def _opgfast_extract_gene_vector(values, population):
    """
    Return a gene-aligned vector of length len(population._G_idx).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    if values.size == gidx.size:
        return values

    if values.size == max_size:
        return values[gidx]

    raise ValueError(
        "Gene value array must be length len(population._G_idx) or population._max_size"
    )


def _opgfast_column_run_counts(mask_2d):
    """
    Count contiguous True runs per column.
    """
    mask_2d = np.asarray(mask_2d, dtype=bool)
    T, G = mask_2d.shape
    out = np.zeros(G, dtype=np.int32)

    for j in range(G):
        out[j] = int(_opgfast_true_run_lengths(mask_2d[:, j]).size)

    return out

def _opgfast_generate_null_returns(
    prefix,
    seg_starts,
    seg_lens,
    template,
    n_sims,
    max_pack_retries,
    rng,
    n_jobs=1,
    batch_size=None,
):
    """
    Generate the shared null return bank.

    This is parallel-safe because each simulation is conditionally independent
    given the shared template and legal segment metadata.
    """
    if n_sims <= 0:
        return np.empty(0, dtype=np.float64)

    if n_jobs <= 1:
        out = np.empty(n_sims, dtype=np.float64)
        for i in range(n_sims):
            lengths = _opgfast_draw_template_lengths_prepared(template, rng)
            out[i] = _opgfast_sample_null_return_once(
                prefix=prefix,
                seg_starts=seg_starts,
                seg_lens=seg_lens,
                block_lengths=lengths,
                rng=rng,
                max_pack_retries=max_pack_retries,
            )
        return out

    if batch_size is None:
        batch_size = max(8, int(np.ceil(n_sims / (n_jobs * 4))))

    sizes = []
    remain = n_sims
    while remain > 0:
        take = min(batch_size, remain)
        sizes.append(take)
        remain -= take

    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=len(sizes), dtype=np.uint64)

    args = [
        (
            prefix,
            seg_starts,
            seg_lens,
            template,
            int(sz),
            int(max_pack_retries),
            int(seed),
        )
        for sz, seed in zip(sizes, seeds)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        batches = list(ex.map(_opgfast_null_worker_batch, args))

    return np.concatenate(batches, axis=0)

def _opgfast_generate_null_scores(
    prefix,
    seg_starts,
    seg_lens,
    template,
    n_sims,
    score_mode,
    score_fn,
    max_pack_retries,
    rng,
    n_jobs=1,
    batch_size=None,
):
    """
    Generate the shared null score bank in the chosen metric.
    """
    if n_sims <= 0:
        return np.empty(0, dtype=np.float64)

    if n_jobs <= 1:
        out = np.empty(n_sims, dtype=np.float64)
        for i in range(n_sims):
            lengths = _opgfast_draw_template_lengths_prepared(template, rng)
            out[i] = _opgfast_sample_null_score_once(
                prefix=prefix,
                seg_starts=seg_starts,
                seg_lens=seg_lens,
                block_lengths=lengths,
                score_mode=score_mode,
                score_fn=score_fn,
                rng=rng,
                max_pack_retries=max_pack_retries,
            )
        return out

    if batch_size is None:
        batch_size = max(8, int(np.ceil(n_sims / (n_jobs * 4))))

    sizes = []
    remain = n_sims
    while remain > 0:
        take = min(batch_size, remain)
        sizes.append(take)
        remain -= take

    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=len(sizes), dtype=np.uint64)

    args = [
        (
            prefix,
            seg_starts,
            seg_lens,
            template,
            int(sz),
            score_mode,
            int(max_pack_retries),
            int(seed),
        )
        for sz, seed in zip(sizes, seeds)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        batches = list(ex.map(_opgfast_null_worker_batch, args))

    return np.concatenate(batches, axis=0)


def _opgfast_null_worker_batch(args):
    """
    Worker batch for multiprocessing null generation.

    Only built-in score modes are supported here.
    """
    prefix, seg_starts, seg_lens, template, batch_n, score_mode, max_pack_retries, seed = args
    rng = np.random.default_rng(seed)

    out = np.empty(batch_n, dtype=np.float64)
    for i in range(batch_n):
        lengths = _opgfast_draw_template_lengths_prepared(template, rng)
        out[i] = _opgfast_sample_null_score_once(
            prefix=prefix,
            seg_starts=seg_starts,
            seg_lens=seg_lens,
            block_lengths=lengths,
            score_mode=score_mode,
            score_fn=None,
            rng=rng,
            max_pack_retries=max_pack_retries,
        )
    return out


def _opgfast_sample_null_score_once(
    prefix,
    seg_starts,
    seg_lens,
    block_lengths,
    score_mode,
    score_fn,
    rng,
    max_pack_retries=64,
):
    """
    Sample one null score directly from interval sums.

    This avoids building a full boolean simulated mask.
    """
    block_lengths = np.asarray(block_lengths, dtype=np.int32)
    if block_lengths.size == 0:
        return 0.0

    n_seg = seg_starts.size
    seg_used = np.zeros(n_seg, dtype=np.int32)
    seg_k = np.zeros(n_seg, dtype=np.int32)
    seg_blocks = [[] for _ in range(n_seg)]

    order = np.argsort(block_lengths)[::-1]
    blocks = block_lengths[order]

    for _ in range(max_pack_retries):
        for i in range(n_seg):
            seg_used[i] = 0
            seg_k[i] = 0
            seg_blocks[i].clear()

        ok = True

        for L in blocks:
            feasible = []
            weights = []

            for i in range(n_seg):
                required = seg_used[i] + L + seg_k[i]
                if required <= seg_lens[i]:
                    feasible.append(i)
                    weights.append((seg_lens[i] - required) + 1)

            if not feasible:
                ok = False
                break

            feasible = np.asarray(feasible, dtype=np.int32)
            weights = np.asarray(weights, dtype=np.float64)
            weights /= weights.sum()

            pick = int(rng.choice(feasible, p=weights))
            seg_blocks[pick].append(int(L))
            seg_used[pick] += int(L)
            seg_k[pick] += 1

        if ok:
            break

    if not ok:
        raise RuntimeError(
            "OPG fast sampler could not assign blocks to legal segments. "
            "Increase max_pack_retries or simplify the anomaly geometry."
        )

    total_sum = 0.0
    total_active = int(block_lengths.sum())
    total_blocks = int(block_lengths.size)

    for seg_id in range(n_seg):
        if not seg_blocks[seg_id]:
            continue

        blk = np.asarray(seg_blocks[seg_id], dtype=np.int32)
        rng.shuffle(blk)

        k = blk.size
        used = int(blk.sum())
        min_required = used + (k - 1)
        extra = int(seg_lens[seg_id] - min_required)

        if extra < 0:
            raise RuntimeError("Segment received an infeasible block assignment.")

        gaps = _opgfast_random_composition(extra, k + 1, rng)

        pos = int(seg_starts[seg_id] + gaps[0])

        for j, L in enumerate(blk):
            total_sum += prefix[pos + L] - prefix[pos]
            pos += int(L)
            if j < k - 1:
                pos += 1 + int(gaps[j + 1])

    return _opgfast_score_from_components(
        sum_value=float(total_sum),
        active_count=total_active,
        block_count=total_blocks,
        score_mode=score_mode,
        score_fn=score_fn,
    )

def _opgfast_null_worker_batch_old(args):
    """
    Worker batch for multiprocessing null generation.
    """
    prefix, seg_starts, seg_lens, template, batch_n, max_pack_retries, seed = args
    rng = np.random.default_rng(seed)

    out = np.empty(batch_n, dtype=np.float64)
    for i in range(batch_n):
        lengths = _opgfast_draw_template_lengths_prepared(template, rng)
        out[i] = _opgfast_sample_null_return_once(
            prefix=prefix,
            seg_starts=seg_starts,
            seg_lens=seg_lens,
            block_lengths=lengths,
            rng=rng,
            max_pack_retries=max_pack_retries,
        )
    return out


def _opgfast_sample_null_return_once(
    prefix,
    seg_starts,
    seg_lens,
    block_lengths,
    rng,
    max_pack_retries=64,
):
    """
    Sample one null return directly from interval sums.

    This avoids building a full boolean simulated mask. The return of each block
    is scored by prefix sums:
        sum(raw_emission[start:start+L]) = prefix[start+L] - prefix[start]
    """
    block_lengths = np.asarray(block_lengths, dtype=np.int32)
    if block_lengths.size == 0:
        return 0.0

    n_seg = seg_starts.size
    seg_used = np.zeros(n_seg, dtype=np.int32)   # total occupied by blocks only
    seg_k = np.zeros(n_seg, dtype=np.int32)      # number of blocks assigned
    seg_blocks = [[] for _ in range(n_seg)]

    # assign long blocks first
    order = np.argsort(block_lengths)[::-1]
    blocks = block_lengths[order]

    for _ in range(max_pack_retries):
        for i in range(n_seg):
            seg_used[i] = 0
            seg_k[i] = 0
            seg_blocks[i].clear()

        ok = True

        for L in blocks:
            feasible = []
            weights = []

            for i in range(n_seg):
                # after adding one block, minimum required length is:
                # existing block sum + new block + internal separators count
                # separators count becomes seg_k[i] after the add
                required = seg_used[i] + L + seg_k[i]
                if required <= seg_lens[i]:
                    feasible.append(i)
                    weights.append((seg_lens[i] - required) + 1)

            if not feasible:
                ok = False
                break

            feasible = np.asarray(feasible, dtype=np.int32)
            weights = np.asarray(weights, dtype=np.float64)
            weights /= weights.sum()

            pick = int(rng.choice(feasible, p=weights))
            seg_blocks[pick].append(int(L))
            seg_used[pick] += int(L)
            seg_k[pick] += 1

        if ok:
            break

    if not ok:
        raise RuntimeError(
            "OPG fast sampler could not assign blocks to legal segments. "
            "Increase max_pack_retries or simplify the anomaly geometry."
        )

    total_return = 0.0

    for seg_id in range(n_seg):
        if not seg_blocks[seg_id]:
            continue

        blk = np.asarray(seg_blocks[seg_id], dtype=np.int32)
        rng.shuffle(blk)

        k = blk.size
        used = int(blk.sum())
        min_required = used + (k - 1)
        extra = int(seg_lens[seg_id] - min_required)

        if extra < 0:
            raise RuntimeError("Segment received an infeasible block assignment.")

        gaps = _opgfast_random_composition(extra, k + 1, rng)

        pos = int(seg_starts[seg_id] + gaps[0])

        for j, L in enumerate(blk):
            total_return += prefix[pos + L] - prefix[pos]
            pos += int(L)
            if j < k - 1:
                pos += 1 + int(gaps[j + 1])

    return float(total_return)


def _opgfast_prepare_template_sampler(obs_lengths, mode="exact", temperature=1.0):
    """
    Prepare the length sampler once so expensive template setup is not repeated
    every simulation.
    """
    obs_lengths = np.asarray(obs_lengths, dtype=np.int32)

    if obs_lengths.size == 0:
        return {
            "mode": "exact",
            "obs_lengths": obs_lengths.copy(),
            "m": 0,
            "n": 0,
        }

    m = int(obs_lengths.size)
    n = int(obs_lengths.sum())

    if mode == "exact":
        return {
            "mode": "exact",
            "obs_lengths": obs_lengths.copy(),
            "m": m,
            "n": n,
        }

    support, counts = np.unique(obs_lengths, return_counts=True)
    support = support.astype(np.int32)
    counts = counts.astype(np.float64)

    if mode == "empirical":
        probs = counts / counts.sum()

    elif mode == "softmax":
        logits = np.log(counts + 1e-12) / float(temperature)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

    else:
        raise ValueError('template_mode must be "exact", "empirical", or "softmax"')

    feasible = _opgfast_build_feasible_table(
        support=support,
        m=m,
        n=n,
    )

    if not feasible[m, n]:
        raise RuntimeError("Could not build a feasible length sampler for the requested m and n.")

    return {
        "mode": mode,
        "obs_lengths": obs_lengths.copy(),
        "support": support,
        "probs": probs.astype(np.float64),
        "feasible": feasible,
        "m": m,
        "n": n,
    }


def _opgfast_draw_template_lengths_prepared(template, rng):
    """
    Draw one simulated set of block lengths from a prepared template.
    """
    mode = template["mode"]

    if mode == "exact":
        out = template["obs_lengths"].copy()
        rng.shuffle(out)
        return out

    support = template["support"]
    probs = template["probs"]
    feasible = template["feasible"]
    rem_m = int(template["m"])
    rem_n = int(template["n"])

    out = np.empty(rem_m, dtype=np.int32)

    for i in range(rem_m):
        valid = (support <= rem_n) & feasible[rem_m - 1 - i, rem_n - support]
        idx = np.flatnonzero(valid)

        if idx.size == 0:
            raise RuntimeError("Prepared length sampler reached an infeasible state.")

        p = probs[idx].copy()
        p /= p.sum()

        pick_j = int(rng.choice(idx, p=p))
        pick_s = int(support[pick_j])

        out[i] = pick_s
        rem_n -= pick_s

    rng.shuffle(out)
    return out


def _opgfast_build_feasible_table(support, m, n):
    """
    DP feasibility table for drawing m positive lengths with exact total n
    from the supplied support.
    """
    support = np.asarray(support, dtype=np.int32)

    feasible = np.zeros((m + 1, n + 1), dtype=bool)
    feasible[0, 0] = True

    for i in range(1, m + 1):
        for t in range(1, n + 1):
            ok = False
            for s in support:
                if s > t:
                    break
                if feasible[i - 1, t - s]:
                    ok = True
                    break
            feasible[i, t] = ok

    return feasible


def _opgfast_prefix_sum(x):
    """
    Prefix sum with leading zero for O(1) interval scoring.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.empty(x.size + 1, dtype=np.float64)
    out[0] = 0.0
    out[1:] = np.cumsum(x)
    return out


def _opgfast_segments_from_mask(mask):
    """
    Convert a legal evaluation mask into segment starts and lengths.
    """
    segs = _opgfast_true_segments(mask)
    if not segs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    starts = np.array([a for a, _ in segs], dtype=np.int32)
    lens = np.array([b - a for a, b in segs], dtype=np.int32)
    return starts, lens


def _opgfast_true_segments(mask):
    """
    Return contiguous True segments as (start, stop_exclusive) pairs.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    x = mask.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts, stops))


def _opgfast_true_run_lengths(mask):
    """
    Return lengths of contiguous True runs.
    """
    segs = _opgfast_true_segments(mask)
    if not segs:
        return np.empty(0, dtype=np.int32)
    return np.array([b - a for a, b in segs], dtype=np.int32)


def _opgfast_random_composition(total, parts, rng):
    """
    Uniform random weak composition of 'total' into 'parts' nonnegative integers.
    """
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < 0:
        raise ValueError("total must be nonnegative")
    if parts == 1:
        return np.array([total], dtype=np.int32)

    # stars and bars
    picks = np.sort(
        rng.choice(total + parts - 1, size=parts - 1, replace=False)
    )
    comp = np.diff(np.r_[-1, picks, total + parts - 1]) - 1
    return comp.astype(np.int32)


def _opgfast_mcpt_p_value(observed, null, alternative="greater"):
    """
    Empirical Monte Carlo p-value with +1 correction.
    """
    null = np.asarray(null, dtype=np.float64)

    if alternative == "greater":
        return (1.0 + np.sum(null >= observed)) / (null.size + 1.0)

    if alternative == "less":
        return (1.0 + np.sum(null <= observed)) / (null.size + 1.0)

    if alternative == "two-sided":
        mu = float(np.mean(null))
        obs_dev = abs(observed - mu)
        null_dev = np.abs(null - mu)
        return (1.0 + np.sum(null_dev >= obs_dev)) / (null.size + 1.0)

    raise ValueError('alternative must be "greater", "less", or "two-sided"')


def _opgfast_scatter_gene_values(values, population, fill_value=np.nan):
    """
    Scatter gene-aligned values into full population index space.

    Supports:
        len(values) == population._max_size
        len(values) == len(population._G_idx)
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    if values.size == max_size:
        return values.copy()

    if values.size != gidx.size:
        raise ValueError(
            "Gene value array must be length population._max_size or len(population._G_idx)"
        )

    out = np.full(max_size, fill_value, dtype=np.float64)
    out[gidx] = values
    return out

def evaluate_opg_mcpt(
    population,
    chunk_num,
    solver_kwargs=None,
    solver_class=Solver,
    n_sims=2000,
    template_mode="exact",
    temperature=1.0,
    alternative="greater",
    fill_value=1.0,
    rng=None,
    max_pack_retries=256,
    return_details=False,
):
    """
    Evaluate shared OPG-MCPT p-values for all genes on one chunk.

    This function builds a single chunk-level OPG null from the anomaly geometry
    produced by a solver, then evaluates every gene in population._G_idx against
    that same null bank.

    The intended use is:
        solver = solver_class(population, **solver_kwargs)
        raw_emission, evaluation_mask, anomaly_mask = solver.solve(population, chunk_num=chunk_num)

    The anomaly mask defines the observed optimal participation geometry template:
        - m = number of consecutive anomaly events
        - n = total anomaly-true count
        - run-length distribution of those m events

    A shared null bank is then formed by generating legal random placements that
    follow that geometry template, and each gene's observed return is converted
    into an empirical Monte Carlo p-value relative to that null.

    Parameters
    ----------
    population : _I.Population
        Population object to evaluate.

    chunk_num : int | None
        Chunk index to evaluate. This is passed into both solver.solve(...) and
        resolve_population_signs(...). If None, the solver/sign resolver should
        interpret that as full-axis evaluation.

    solver_kwargs : dict | None, default=None
        Keyword arguments used to assemble the solver exactly as:
            solver = solver_class(population, **solver_kwargs)
        These are the same parameters your Solver.__init__ accepts, such as:
            t_vec, t_mode, emission, offset, AD_cond

    solver_class : type, default=Solver
        Solver-like class used to resolve raw_emission, evaluation_mask, and
        anomaly_mask. It must support:
            solver = solver_class(population, **solver_kwargs)
            raw_emission, evaluation_mask, anomaly_mask = solver.solve(population, chunk_num=chunk_num)

        In your project this is typically _E.Solver.

    n_sims : int, default=2000
        Number of Monte Carlo null permutations to generate.

    template_mode : {"exact", "empirical", "softmax"}, default="exact"
        How to generate simulated OPG block lengths from the observed anomaly
        run-length distribution.

        "exact":
            Preserve the exact observed run lengths and only randomize placement.
        "empirical":
            Sample run lengths from the empirical observed pmf while forcing the
            simulated geometry to keep the same m and n.
        "softmax":
            Same as empirical, but the pmf is softened via a softmax over the
            observed run-length counts using the provided temperature.

    temperature : float, default=1.0
        Softmax temperature used only when template_mode == "softmax".
        Lower values make the distribution sharper around more frequent run lengths.

    alternative : {"greater", "less", "two-sided"}, default="greater"
        Tail used for the empirical p-value.
        For trading-style positive-is-better returns, "greater" is usually correct.

    fill_value : float, default=1.0
        Value assigned to indices not in population._G_idx.
        For literal p-values, 1.0 is usually the worst logical value.

    rng : None, int, or np.random.Generator, default=None
        Random source for Monte Carlo sampling.

    max_pack_retries : int, default=256
        Maximum retries when placing simulated OPG blocks into legal evaluation
        locations while preserving exact block count and block lengths.

    return_details : bool, default=False
        If False, return only the full-length p-value array.
        If True, also return a details dict containing intermediate objects such
        as observed m/n, observed run lengths, raw observed returns, and null returns.

    Returns
    -------
    pvals : np.ndarray, shape (population._max_size,)
        Full-length p-value vector. Indices not in population._G_idx are set to
        fill_value. Indices in population._G_idx receive their shared-null OPG-MCPT
        empirical p-values.

    details : dict, optional
        Returned only when return_details=True. Includes:
            raw_emission
            evaluation_mask
            anomaly_mask
            X_p
            R_full
            gidx
            m_obs
            n_obs
            obs_lengths
            null_returns
    """
    rng = np.random.default_rng(rng)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    # build solver the way your project does it
    solver = solver_class(population, **solver_kwargs)

    # solver returns chunk-local arrays in your shown implementation
    raw_emission, evaluation_mask, anomaly_mask = solver.solve(
        population,
        chunk_num=chunk_num,
    )

    raw_emission = np.asarray(raw_emission, dtype=float).reshape(-1)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)

    if raw_emission.shape[0] != evaluation_mask.shape[0]:
        raise ValueError("raw_emission and evaluation_mask do not share the same length")
    if raw_emission.shape[0] != anomaly_mask.shape[0]:
        raise ValueError("raw_emission and anomaly_mask do not share the same length")

    # gene participation and observed returns from your existing project functions
    X_p = resolve_population_signs(population, chunk_num=chunk_num)
    R = evaluate_return(population, X_p, raw_emission, evaluation_mask)

    gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    pvals = np.full(max_size, fill_value, dtype=float)
    R_full = _scatter_gene_values_to_full(R, population, fill_value=np.nan)

    # use only legal anomaly positions for the OPG template
    opg_mask = anomaly_mask & evaluation_mask

    # context-consistent call
    m_obs, n_obs = resolve_anomaly_mn(opg_mask)

    # actual run lengths from the legal anomaly geometry
    obs_lengths = _true_run_lengths(opg_mask)

    # keep counts consistent with the geometry actually being permuted
    if int(obs_lengths.size) != int(m_obs) or int(obs_lengths.sum()) != int(n_obs):
        m_obs = int(obs_lengths.size)
        n_obs = int(obs_lengths.sum())

    details = {
        "raw_emission": raw_emission,
        "evaluation_mask": evaluation_mask,
        "anomaly_mask": anomaly_mask,
        "opg_mask": opg_mask,
        "X_p": X_p,
        "R_full": R_full,
        "gidx": gidx.copy(),
        "m_obs": int(m_obs),
        "n_obs": int(n_obs),
        "obs_lengths": obs_lengths.copy(),
        "null_returns": None,
    }

    if m_obs == 0 or n_obs == 0:
        if return_details:
            return pvals, details
        return pvals

    legal_segments = _true_segments(evaluation_mask)
    chunk_len = raw_emission.shape[0]

    # one shared null bank for the whole chunk
    null_returns = np.empty(int(n_sims), dtype=float)

    for s in range(int(n_sims)):
        sim_lengths = _draw_opg_template_lengths(
            obs_lengths=obs_lengths,
            mode=template_mode,
            temperature=temperature,
            rng=rng,
        )

        sim_mask = _sample_block_mask_on_legal_segments(
            chunk_len=chunk_len,
            legal_segments=legal_segments,
            block_lengths=sim_lengths,
            rng=rng,
            max_retries=max_pack_retries,
        )

        # shared-null return statistic aligned to evaluate_return style
        null_returns[s] = float(raw_emission[sim_mask].sum())

    # convert observed returns to empirical p-values only at _G_idx
    for gi in gidx:
        obs_return = R_full[gi]
        pvals[gi] = _mcpt_p_value(
            observed=obs_return,
            null=null_returns,
            alternative=alternative,
        )

    details["null_returns"] = null_returns

    if return_details:
        return pvals, details
    return pvals


def _scatter_gene_values_to_full(values, population, fill_value=np.nan):
    """
    Scatter a gene-value vector into full population index space.

    Supports either:
        - values already shaped as (population._max_size,)
        - values shaped as (len(population._G_idx),)

    Returns
    -------
    out : np.ndarray, shape (population._max_size,)
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    gidx = np.asarray(population._G_idx, dtype=int)
    max_size = int(population._max_size)

    if values.size == max_size:
        return values.copy()

    if values.size != gidx.size:
        raise ValueError(
            "Gene value array must be length population._max_size or len(population._G_idx)"
        )

    out = np.full(max_size, fill_value, dtype=float)
    out[gidx] = values
    return out


def _mcpt_p_value(observed, null, alternative="greater"):
    """
    Empirical Monte Carlo p-value with +1 correction.
    """
    null = np.asarray(null, dtype=float)

    if alternative == "greater":
        return (1.0 + np.sum(null >= observed)) / (null.size + 1.0)

    if alternative == "less":
        return (1.0 + np.sum(null <= observed)) / (null.size + 1.0)

    if alternative == "two-sided":
        mu = float(np.mean(null))
        obs_dev = abs(observed - mu)
        null_dev = np.abs(null - mu)
        return (1.0 + np.sum(null_dev >= obs_dev)) / (null.size + 1.0)

    raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")


def _true_segments(mask):
    """
    Return contiguous True segments as (start, stop_exclusive) pairs.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    x = mask.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts, stops))


def _true_run_lengths(mask):
    """
    Return lengths of contiguous True runs in the supplied boolean mask.
    """
    segs = _true_segments(mask)
    if not segs:
        return np.empty(0, dtype=int)
    return np.array([b - a for a, b in segs], dtype=int)


def _draw_opg_template_lengths(obs_lengths, mode, temperature, rng):
    """
    Draw simulated OPG block lengths from the observed anomaly geometry.

    Modes
    -----
    exact:
        keep exact observed run lengths, only shuffled
    empirical:
        sample from empirical run-length pmf while forcing exact observed m and n
    softmax:
        same as empirical, but smooth counts through a softmax temperature
    """
    obs_lengths = np.asarray(obs_lengths, dtype=int)

    if obs_lengths.size == 0:
        return obs_lengths.copy()

    if mode == "exact":
        out = obs_lengths.copy()
        rng.shuffle(out)
        return out

    support, counts = np.unique(obs_lengths, return_counts=True)
    support = support.astype(int)
    counts = counts.astype(float)

    if mode == "empirical":
        probs = counts / counts.sum()

    elif mode == "softmax":
        logits = np.log(counts + 1e-12) / float(temperature)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

    else:
        raise ValueError('template_mode must be "exact", "empirical", or "softmax"')

    return _sample_lengths_fixed_m_fixed_n(
        support=support,
        probs=probs,
        m=int(obs_lengths.size),
        n=int(obs_lengths.sum()),
        rng=rng,
    )


def _sample_lengths_fixed_m_fixed_n(support, probs, m, n, rng):
    """
    Sample m positive block lengths from support with probabilities probs,
    conditioned on the exact total sum n.

    This preserves the observed anomaly geometry counts:
        number of blocks = m
        total active count = n
    """
    support = np.asarray(support, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if support.ndim != 1 or probs.ndim != 1 or support.size != probs.size:
        raise ValueError("support and probs must be 1D and same length")
    if np.any(support <= 0):
        raise ValueError("support lengths must be positive")

    feasible = np.zeros((m + 1, n + 1), dtype=bool)
    feasible[0, 0] = True

    for i in range(1, m + 1):
        for t in range(1, n + 1):
            ok = False
            for s in support:
                if t >= s and feasible[i - 1, t - s]:
                    ok = True
                    break
            feasible[i, t] = ok

    if not feasible[m, n]:
        raise RuntimeError("Could not sample lengths with the requested fixed m and n")

    out = np.empty(m, dtype=int)
    rem_m = m
    rem_n = n

    for i in range(m):
        candidate_idx = []
        for j, s in enumerate(support):
            if rem_n >= s and feasible[rem_m - 1, rem_n - s]:
                candidate_idx.append(j)

        candidate_idx = np.asarray(candidate_idx, dtype=int)
        p = probs[candidate_idx]
        p /= p.sum()

        pick_j = int(rng.choice(candidate_idx, p=p))
        pick_s = int(support[pick_j])

        out[i] = pick_s
        rem_m -= 1
        rem_n -= pick_s

    rng.shuffle(out)
    return out


import numpy as np


def _sample_block_mask_on_legal_segments(
    chunk_len,
    legal_segments,
    block_lengths,
    rng,
    max_retries=256,
):
    """
    Fast sampler for placing non-touching blocks into legal segments.

    Preserves:
        - exact block count
        - exact block lengths
        - non-touching blocks within a segment

    Strategy:
        1) assign blocks to feasible legal segments
        2) within each segment, sample slack allocation across gaps
    """
    block_lengths = np.asarray(block_lengths, dtype=int)

    if block_lengths.size == 0:
        return np.zeros(chunk_len, dtype=bool)

    seg_lens = np.array([b - a for a, b in legal_segments], dtype=int)

    if block_lengths.sum() > seg_lens.sum():
        raise RuntimeError("Block lengths exceed legal evaluation capacity")

    # sort long blocks first for easier packing
    order = np.argsort(block_lengths)[::-1]
    blocks = block_lengths[order]

    for _ in range(max_retries):
        # per segment: list of assigned block lengths
        seg_blocks = [[] for _ in legal_segments]
        seg_used = np.zeros(len(legal_segments), dtype=int)   # sum of block lengths
        seg_k = np.zeros(len(legal_segments), dtype=int)      # number of blocks

        ok = True

        # assign each block to a feasible segment
        for L in blocks:
            feasible = []
            weights = []

            for i, S in enumerate(seg_lens):
                # if we add one more block, required length is:
                # current sum + L + (new_k - 1)
                new_k = seg_k[i] + 1
                required = seg_used[i] + L + max(new_k - 1, 0)
                if required <= S:
                    feasible.append(i)
                    # prefer segments with more remaining slack
                    slack = S - required
                    weights.append(slack + 1)

            if not feasible:
                ok = False
                break

            feasible = np.asarray(feasible, dtype=int)
            weights = np.asarray(weights, dtype=float)
            weights /= weights.sum()

            pick = int(rng.choice(feasible, p=weights))

            seg_blocks[pick].append(int(L))
            seg_used[pick] += int(L)
            seg_k[pick] += 1

        if not ok:
            continue

        placed = np.zeros(chunk_len, dtype=bool)

        # now sample exact positions within each segment using random gaps
        for seg_id, blk_list in enumerate(seg_blocks):
            if not blk_list:
                continue

            a, b = legal_segments[seg_id]
            S = b - a
            blk = np.array(blk_list, dtype=int)

            # randomize order of blocks within the segment
            rng.shuffle(blk)

            k = blk.size
            min_required = blk.sum() + (k - 1)
            extra = S - min_required

            if extra < 0:
                ok = False
                break

            # distribute extra slack across k+1 gaps
            # gaps = [g0, g1, ..., gk], all >= 0
            # actual internal gaps become 1 + gi
            gaps = _random_composition(extra, k + 1, rng)

            pos = a + gaps[0]
            for j, L in enumerate(blk):
                placed[pos:pos + L] = True
                pos += L
                if j < k - 1:
                    pos += 1 + gaps[j + 1]
                else:
                    pos += gaps[j + 1]

        if ok:
            return placed

    raise RuntimeError(
        "Failed to place simulated OPG blocks into legal chunk positions. "
        "Increase max_retries or simplify the template."
    )

import numpy as np
import initialization as _I


import numpy as np
import initialization as _I


def reduce_scored_family_indices(X, s_idx, sidx_scores, *, return_sorted=False):
    """
    Parameters
    ----------
    X : object
        Must have X._instructions
    s_idx : array-like
        Scored indices
    sidx_scores : array-like
        Scores parallel to s_idx
    return_sorted : bool, default=False
        If True, sort final family_idx ascending

    Returns
    -------
    kept_s_idx : np.ndarray
        Reduced scored indices after removing any scored index whose family
        contains another scored index.
    kept_scores : np.ndarray
        Scores parallel to kept_s_idx
    family_idx : np.ndarray
        Unique family indices across kept_s_idx
    family_scores : np.ndarray
        Best score parallel to family_idx
    """

    s_idx = np.asarray(s_idx, dtype=np.int64).ravel()
    sidx_scores = np.asarray(sidx_scores).ravel()

    if s_idx.shape[0] != sidx_scores.shape[0]:
        raise ValueError("s_idx and sidx_scores must be the same length")

    if s_idx.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_s = np.empty(0, dtype=sidx_scores.dtype if sidx_scores.size else np.float64)
        return empty_i, empty_s, empty_i.copy(), empty_s.copy()

    # collapse duplicate scored indices by keeping the best score
    # best now means minimum score
    uniq_best = {}
    uniq_order = []
    for idx, score in zip(s_idx, sidx_scores):
        idx = int(idx)
        if idx not in uniq_best:
            uniq_best[idx] = score
            uniq_order.append(idx)
        else:
            if score < uniq_best[idx]:
                uniq_best[idx] = score

    s_idx = np.asarray(uniq_order, dtype=np.int64)
    sidx_scores = np.asarray([uniq_best[i] for i in uniq_order], dtype=sidx_scores.dtype)

    scored_set = set(int(i) for i in s_idx)

    # cache family tree for each scored index
    fam_cache = {}
    fam_set_cache = {}
    for idx in s_idx:
        fam = np.asarray(_I.family_tree_indices(X._instructions, int(idx)), dtype=np.int64).ravel()

        # ensure the node itself is included
        if fam.size == 0 or int(idx) not in fam:
            fam = np.concatenate(([int(idx)], fam))

        # preserve order, remove duplicates
        seen = set()
        fam_ordered = []
        for node in fam:
            node = int(node)
            if node not in seen:
                seen.add(node)
                fam_ordered.append(node)

        fam_cache[int(idx)] = np.asarray(fam_ordered, dtype=np.int64)
        fam_set_cache[int(idx)] = seen

    # remove scored indices that contain any other scored index in their family
    keep_mask = np.ones(s_idx.shape[0], dtype=bool)
    for k, idx in enumerate(s_idx):
        idx = int(idx)
        other_scored_in_family = fam_set_cache[idx].intersection(scored_set)
        other_scored_in_family.discard(idx)
        if len(other_scored_in_family) > 0:
            keep_mask[k] = False

    kept_s_idx = s_idx[keep_mask]
    kept_scores = sidx_scores[keep_mask]

    if kept_s_idx.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_s = np.empty(0, dtype=sidx_scores.dtype)
        return kept_s_idx, kept_scores, empty_i, empty_s

    # assign each family node the best score among surviving scored indices that use it
    # best now means minimum score
    node_best_score = {}
    node_order = []

    for idx, score in zip(kept_s_idx, kept_scores):
        fam = fam_cache[int(idx)]
        for node in fam:
            node = int(node)
            if node not in node_best_score:
                node_best_score[node] = score
                node_order.append(node)
            else:
                if score < node_best_score[node]:
                    node_best_score[node] = score

    family_idx = np.asarray(node_order, dtype=np.int64)
    family_scores = np.asarray([node_best_score[int(node)] for node in family_idx],
                               dtype=kept_scores.dtype)

    if return_sorted:
        order = np.argsort(family_idx)
        family_idx = family_idx[order]
        family_scores = family_scores[order]

    return kept_s_idx, kept_scores, family_idx, family_scores


def _random_composition(total, parts, rng):
    """
    Sample a random composition of 'total' into 'parts' nonnegative integers.
    Returns an array of length 'parts' summing to 'total'.
    """
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < 0:
        raise ValueError("total must be nonnegative")
    if parts == 1:
        return np.array([total], dtype=int)
    # stars and bars via sorted cut points
    cuts = np.sort(rng.integers(0, total + parts - 1, size=parts - 1))
    arr = np.diff(np.r_[-1, cuts, total + parts - 1]) - 1
    return arr.astype(int)


#NOTE VARIOUS EMISSIONS

emission_fwd_return_sigma = [
    {"ID": 5, "alpha": "tvec", "offset": False},                 # P[t+off] - P[t]
    {"ID": "divide"},
    {"ID": 18, "x": "tvec", "offset": False, "delta1": 24, "min_count": 2},  # STD over last ~2h
]

emission_future_window_terminal_z = [
    {"ID": 5, "alpha": {"ID": 3, "x": "tvec", "delta1": 12, "offset": True}},  # P[t+12] - mean(P[t+1..t+12])
    {"ID": "divide"},
    {"ID": 18, "x": "tvec", "delta1": 12, "offset": True, "min_count": 2},    # std(P[t+1..t+12])
]

emission_vol_expansion_ratio = [
    {"ID": 18, "delta1": 12, "min_count": 2},  # future-window realized std (uses future-aligned stream)
    {"ID": "divide"},
    {"ID": 18, "x": "tvec", "offset": False, "delta1": 12, "min_count": 2},  # trailing std now
    {"ID": 5, "alpha": 1.0},  # ratio - 1
]

emission_breakout_above_high_sigma = [
    {"ID": 5, "alpha": {"ID": 1, "x": "tvec", "delta1": 24, "offset": False}},  # P[t+off] - MAX24(P[t])
    {"ID": "divide"},
    {"ID": 18, "x": "tvec", "offset": False, "delta1": 24, "min_count": 2},    # / STD24(P[t])
]

emission_future_momentum_doe = [
    {"ID": 15, "delta1": 3, "delta2": 12},            # DOE on future-aligned stream (fast - slow EMA)
    {"ID": 17, "delta1": 24, "min_count": 2},         # zscore it over ~2h
]

delta = 24  # e.g., last 24 bars = 2 hours on 5-min data

emission_minmax_breakout = [
    # numerator: P[t+offset] - MIN_delta(P[t])
    {"ID": 5,
     "x": "tvec", "offset": True,
     "alpha": {"ID": 2, "x": "tvec", "offset": False, "delta1": delta}},

    # divide by range: (MAX_delta(P[t]) - MIN_delta(P[t]))
    {"ID": "divide"},
    {"ID": 5,
     "x": {"ID": 1, "x": "tvec", "offset": False, "delta1": delta},
     "alpha": {"ID": 2, "x": "tvec", "offset": False, "delta1": delta}},

    # *2
    {"ID": 6, "alpha": "emit"},

    # -1
    {"ID": 5, "alpha": 1.0},
]

emission_vol_expand = [
    {"ID": 18, "x": "tvec", "delta1": 12, "min_count": 2},            # future-aligned dispersion
    {"ID": "divide"},
    {"ID": 18, "x": "tvec", "offset": False, "delta1": 12, "min_count": 2},  # current dispersion
    {"ID": 5, "alpha": 1.0},                                          # (future/current) - 1
]

emission_volu_expand = [
    {"ID": 3, "x": "tvec", "delta1": 12, "min_count": 2},             # future-aligned avg volume
    {"ID": "divide"},
    {"ID": 3, "x": "tvec", "offset": False, "delta1": 12, "min_count": 2},  # current avg volume
    {"ID": 5, "alpha": 1.0},                                          # (future/current) - 1
]