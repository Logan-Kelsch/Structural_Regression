import numpy as np
import initialization as _I
import transform_ops as _OPS


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


class Solver:
    '''
    This class object will simply contain data that is wanted to resolve the solution vector from given data
    '''
    def __init__(
        self,
        population  :   _I.Population,
        t_vec       :   str =   'Close',
        t_mode      :   str =   'AD',
        emission    :   list =  [
            {"ID": 5, "alpha": {"ID": 3, "x": "tvec", "delta1": 6*4, "offset": False}},
            {"ID": "divide"},
            {"ID": 18, "x": "tvec", "delta1": 6*4, "offset": False},
        ],
        offset      :   int =   6,
        AD_cond     :   tuple = ('gt', 2),
    ):
        '''emission is interpreted as functions applied to t_vec from left to right.'''

        self._tmode     = t_mode
        self._emission  = emission
        self._AD_cond   = AD_cond
        self._offset    = offset

        target_idx = -1
        match(t_vec):
            case 'Volume':
                if population._time_terminals:
                    target_idx = population._T_idx[-3]
                else:
                    target_idx = population._T_idx[-1]
            case 'Close':
                if population._time_terminals:
                    target_idx = population._T_idx[-4]
                else:
                    target_idx = population._T_idx[-2]
            case _:
                raise ValueError(
                    f'Target vector (t_vec parameter in Solver Initialization) of "{t_vec}" '
                    f'dont make noooo sense. Try "Close".'
                )

        if target_idx < 0:
            raise ValueError('Did not form a good target vector index in solver initialization. Got idx (-1)')

        self._tidx = int(target_idx)

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
