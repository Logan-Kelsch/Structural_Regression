
import transform_jit as t_jit

import numpy as np

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
	}
	return f_p.get(id, [])

def V_to_LOC(vals):
	# Order: x, a, d, dd, k  ->  2, 3, 4, 5, 6
	m = {'x': 2, 'a': 3, 'd': 4, 'dd': 5, 'k': 6}
	return [m[v] for v in vals if v in m]

def VLOC_to_FLAG(vlocs):
	"""
	Takes a list like [2,3,4,5,6] (any subset, any order) and returns a single
	int bit-flag value, preserving the 2..6 location values by mapping:
	2->bit2, 3->bit3, 4->bit4, 5->bit5, 6->bit6.
	"""
	flags = 0
	for loc in vlocs:
		if 2 <= loc <= 6:
			flags |= (1 << loc)
	return flags

import numpy as np

def fill_const_ITS(arr, where_idx, rate=1.0, rng=None, inplace=True):
    """
    Fill locations specified by a np.where result with samples drawn from the
    negative side of an exponential curve (x <= 0) via inverse transform sampling,
    then flipped to be positive.

    Parameters
    ----------
    arr : np.ndarray
        Target array to fill.
    where_idx : tuple of np.ndarray
        Output of np.where(...), e.g. (rows, cols, ...).
    rate : float
        Exponential rate λ (> 0). Mean of final positive values is 1/λ.
    rng : np.random.Generator or None
        Random generator. If None, uses np.random.default_rng().
    inplace : bool
        If True, modify arr in-place and return it. If False, return a copy.

    Returns
    -------
    np.ndarray
        Array with filled values.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")

    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    n = np.size(where_idx[0])
    if n == 0:
        return out

    u = rng.random(n)
    u = np.clip(u, np.finfo(float).tiny, 1.0)

    neg_samples = (1.0 / rate) * np.log(u)  # <= 0
    out[where_idx] = -neg_samples           # flip sign -> >= 0
    return out

def fill_const_STES(n, base_max=2.0, base_rate=2.0, size=None, rng=None):
    """
    Samples integers in {1,2,...,n} using a stretched version of a truncated exponential.

    1) Sample Y ~ TruncatedExponential(rate=base_rate) on (0, base_max]
       via inverse-CDF.
    2) Stretch to X = Y * (n/base_max) so support becomes (0, n]
    3) Return ceil(X) to get ints in {1,...,n}.

    Parameters
    ----------
    n : int
        Upper bound (inclusive). Output is in {1,...,n}.
    base_max : float
        The original upper bound you're "stretching" from (default 2.0).
    base_rate : float
        Exponential rate λ controlling aggressiveness (default 2.0).
    size : int or tuple, optional
        Number/shape of samples.
    rng : np.random.Generator, optional
        RNG to use.

    Returns
    -------
    np.ndarray of int
    """
    if not (isinstance(n, (int, np.integer)) and n >= 1):
        raise ValueError("n must be an integer >= 1")
    if base_max <= 0:
        raise ValueError("base_max must be > 0")
    if base_rate <= 0:
        raise ValueError("base_rate must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    # U in (0,1) (avoid endpoints)
    u = rng.random(size)
    u = np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    # Truncated exponential on (0, base_max]:
    # CDF(x) = (1 - exp(-λ x)) / (1 - exp(-λ base_max))
    # Inverse: x = -(1/λ) * ln(1 - u*(1 - exp(-λ base_max)))
    z = 1.0 - np.exp(-base_rate * base_max)
    y = -(1.0 / base_rate) * np.log(1.0 - u * z)

    # Stretch to (0, n]
    x = y * (n / base_max)

    # Round up to int in [1, n]
    k = np.ceil(x).astype(int)
    k = np.clip(k, 1, n)
    return k


#ID 1
def t_MAX(
	x: np.ndarray,
	window,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling max down axis=0 with per-column windows and min_count.

	Parameters
	----------
	x : (m, n) array. If integers are passed and min_count causes NaNs,
		we'll upcast to float (float32 by default) to hold NaNs.
	window : int or (n,) int
		Size per column (>=1). If scalar, same for all columns.
	min_count : int or (n,) int
		Required valid count before emitting a max (else NaN).
	out : optional (m, n) array
		Only used when in_place=False. Must be float dtype to carry NaNs.
	in_place : bool
		If True (hot path), write results back into x (using a per-column
		scratch copy internally to avoid read/overwrite hazards).
	prefer_float32 : bool
		When upcasting is needed (e.g., integer x but we must write NaNs),
		choose float32 if True (bandwidth-friendly), else float64.

	Returns
	-------
	ndarray
		If in_place=True, returns x (modified). Otherwise returns out.
	"""
	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")

	m, n = x.shape

	# Normalize window -> (n,)
	if np.isscalar(window):
		windows = np.full(n, int(window), dtype=np.int64)
	else:
		windows = np.asarray(window, dtype=np.int64)
		if windows.shape != (n,):
			raise ValueError("window must be scalar or shape (n,)")

	# Normalize min_count -> use scalar fast path when possible
	if np.isscalar(min_count):
		mc_scalar = int(min_count)
		mc_vec = np.empty(1, dtype=np.int64)  # ignored
	else:
		mc_vec = np.asarray(min_count, dtype=np.int64)
		if mc_vec.shape != (n,):
			raise ValueError("min_count must be scalar or shape (n,)")
		mc_scalar = -1  # sentinel to use vector

	# Ensure C-contiguity for best parallel access
	if not x.flags.c_contiguous:
		x = np.ascontiguousarray(x)

	# If we will produce NaNs for warm-up rows, output dtype must be float
	needs_nan = True  # rolling windows with min_count generally emit NaN at the top
	#was going to make this an option but changed my mind
	float_dtype = np.float32

	if in_place:
		# In-place path: ensure dtype can carry NaN
		if not np.issubdtype(x.dtype, np.floating):
			# upcast integers -> float to carry NaNs
			x = x.astype(float_dtype, copy=True)
		# run kernel
		t_jit._MAX_inp(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
		return x

	# out-of-place path
	if out is None:
		if np.issubdtype(x.dtype, np.floating):
			out = np.empty_like(x)  # same float dtype
		else:
			# integers but need NaN -> allocate float
			out = np.empty((m, n), dtype=float_dtype)
	else:
		if out.shape != x.shape:
			raise ValueError("out has wrong shape")
		if needs_nan and not np.issubdtype(out.dtype, np.floating):
			raise TypeError("out must be a floating dtype to carry NaNs")

	if not out.flags.c_contiguous:
		out = np.ascontiguousarray(out)

	t_jit._MAX_out(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 2
def t_MIN(
	x: np.ndarray,
	window,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling min down axis=0 with per-column windows and min_count.
	Mirrors t_MAX API/behavior.
	"""
	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")
	m, n = x.shape

	# window -> (n,)
	if np.isscalar(window):
		windows = np.full(n, int(window), dtype=np.int64)
	else:
		windows = np.asarray(window, dtype=np.int64)
		if windows.shape != (n,):
			raise ValueError("window must be scalar or shape (n,)")

	# min_count -> scalar fast path or vector
	if np.isscalar(min_count):
		mc_scalar = int(min_count)
		mc_vec = np.empty(1, dtype=np.int64)
	else:
		mc_vec = np.asarray(min_count, dtype=np.int64)
		if mc_vec.shape != (n,):
			raise ValueError("min_count must be scalar or shape (n,)")
		mc_scalar = -1

	if not x.flags.c_contiguous:
		x = np.ascontiguousarray(x)

	needs_nan = True
	#was going to make this an option but changed my mind
	float_dtype = np.float32

	if in_place:
		if not np.issubdtype(x.dtype, np.floating):
			x = x.astype(float_dtype, copy=True)
		t_jit._MIN_inp(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
		return x

	if out is None:
		if np.issubdtype(x.dtype, np.floating):
			out = np.empty_like(x)
		else:
			out = np.empty((m, n), dtype=float_dtype)
	else:
		if out.shape != x.shape:
			raise ValueError("out has wrong shape")
		if needs_nan and not np.issubdtype(out.dtype, np.floating):
			raise TypeError("out must be floating dtype to carry NaNs")

	if not out.flags.c_contiguous:
		out = np.ascontiguousarray(out)

	t_jit._MIN_out(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 3
def t_AVG(
	x: np.ndarray,
	window,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling mean down axis=0 with per-column windows and min_count.
	Uses a running-sum O(m) kernel; emits NaN until min_count is met.
	"""
	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")
	m, n = x.shape

	# window -> (n,)
	if np.isscalar(window):
		windows = np.full(n, int(window), dtype=np.int64)
	else:
		windows = np.asarray(window, dtype=np.int64)
		if windows.shape != (n,):
			raise ValueError("window must be scalar or shape (n,)")

	# min_count
	if np.isscalar(min_count):
		mc_scalar = int(min_count)
		mc_vec = np.empty(1, dtype=np.int64)
	else:
		mc_vec = np.asarray(min_count, dtype=np.int64)
		if mc_vec.shape != (n,):
			raise ValueError("min_count must be scalar or shape (n,)")
		mc_scalar = -1

	if not x.flags.c_contiguous:
		x = np.ascontiguousarray(x)

	needs_nan = True
	#was going to make this an option but changed my mind
	float_dtype = np.float32

	if in_place:
		# averages must be floating for non-integer divisions and NaNs
		if not np.issubdtype(x.dtype, np.floating):
			x = x.astype(float_dtype, copy=True)
		t_jit._AVG_inp(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
		return x

	if out is None:
		# always prefer float output for averages
		if np.issubdtype(x.dtype, np.floating):
			out = np.empty_like(x)
		else:
			out = np.empty((m, n), dtype=float_dtype)
	else:
		if out.shape != x.shape:
			raise ValueError("out has wrong shape")
		if needs_nan and not np.issubdtype(out.dtype, np.floating):
			raise TypeError("out must be floating dtype to carry NaNs")

	if not out.flags.c_contiguous:
		out = np.ascontiguousarray(out)

	t_jit._AVG_out(x, windows, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 4
def t_NEG(
	x	:	np.ndarray, 
	out	=	None
):
	np.negative(x, out=(x if out is None else out))
	return x if out is None else out

#ID 5
def t_DIF(
	x	:	np.ndarray,
	a	:	np.ndarray,
	out	=	None
):
	a = np.asanyarray(a)
	a = a.reshape(1, -1) if a.ndim == 1 else a
	np.subtract(x, a, out=(x if out is None else out))
	return x if out is None else out

#ID 6
def t_ADD(
	x	:	np.ndarray,
	a	:	np.ndarray,
	out	=	None
):
	a = np.asanyarray(a)
	a = a.reshape(1, -1) if a.ndim == 1 else a
	np.add(x, a, out=(x if out is None else out))
	return x if out is None else out

#ID 7
def t_SQR(
	x	:	np.ndarray,
	out	=	None
):
	np.square(x, out=(x if out is None else out))
	return x if out is None else out

#ID 8
def t_SIN(x, out=None):
	dst = x if out is None else out
	if not np.issubdtype(dst.dtype, np.floating):
		if out is None: 
			x = x.astype(np.float32, copy=True)
			dst = x
		else: 
			raise TypeError("out must be a floating dtype")
	np.clip(x, -1.0, 1.0, out=dst)
	np.multiply(x, np.pi, out=dst)     # dst = π * x
	np.sin(dst, out=dst)               # dst = sin(dst)
	return dst

#ID 9
def t_COS(x, out=None):
	dst = x if out is None else out
	if not np.issubdtype(dst.dtype, np.floating):
		if out is None: 
			x = x.astype(np.float32, copy=True)
			dst = x
		else: 
			raise TypeError("out must be a floating dtype")
	np.clip(x, -1.0, 1.0, out=dst)
	np.multiply(x, np.pi, out=dst)     # dst = π * x
	np.cos(dst, out=dst)               # dst = cos(dst)
	return dst

#ID 10
def t_ASN(x, out=None):
	dst = x if out is None else out
	if not np.issubdtype(dst.dtype, np.floating):
		if out is None: 
			x = x.astype(np.float32, copy=True)
			dst = x
		else: 
			raise TypeError("out must be a floating dtype")
	np.clip(x, -1.0, 1.0, out=dst)     # clamp to [-1, 1]
	np.arcsin(dst, out=dst)            # dst = asin(dst)
	return dst

#ID 11
def t_ACS(x, out=None):
	dst = x if out is None else out
	if not np.issubdtype(dst.dtype, np.floating):
		if out is None: 
			x = x.astype(np.float32, copy=True)
			dst = x
		else: 
			raise TypeError("out must be a floating dtype")
	np.clip(x, -1.0, 1.0, out=dst)     # clamp to [-1, 1]
	np.arccos(dst, out=dst)            # dst = acos(dst)
	return dst

#ID 12
def t_RNG(
	x	:	np.ndarray,
	delta_max,
	delta_min,
	min_count	:	int	=	1,
	out	=	None,
	in_place	:	bool	=	False
):	
	#taking this option away
	fdt = np.float32

	mx = np.empty(x.shape, dtype=np.float32)
	mn = np.empty(x.shape, dtype=np.float32)

	t_MAX(x, delta_max, min_count=min_count, out=mx, in_place=False)
	t_MIN(x, delta_min, min_count=min_count, out=mn, in_place=False)

	# choose destination
	if in_place:
		if not np.issubdtype(x.dtype, np.floating):
			x = x.astype(fdt, copy=True)      # upcast ints so we can store NaN/real ratios
		dst = x
	else:
		if out is None:
			out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
		dst = out

	# dst = 2 * (x - mn) / (mx - mn) - 1   (no extra temporaries; safe where range<=0)
	np.subtract(x, mn, out=dst)
	np.subtract(mx, mn, out=mx)                  # reuse mx as the range
	np.divide(dst, mx, out=dst, where=(mx > 0))  # leaves NaNs where mn was NaN; leaves prev where range<=0
	np.multiply(dst, 2.0, out=dst)
	np.subtract(dst, 1.0, out=dst)

	return dst

#ID 13
def t_HKP(
	x:	np.ndarray,
	kappa,
	out	:	np.ndarray	|	None	=	None,
	in_place	:	bool	=	False
):
	"""
	Hawkes-like process (per column):
		y[0] = x[0]
		y[t] = exp(-kappa_j) * y[t-1] + x[t],  t=1..m-1

	kappa: scalar or (n,). Larger kappa -> faster decay.
	If in_place=True (98% case), writes results back into x safely in one pass.
	"""

	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")
	m, n = x.shape

	# Normalize kappa -> vector (n,)
	if np.isscalar(kappa):
		if kappa <= 0:
			raise ValueError("kappa must be > 0")
		kvec = np.full(n, float(kappa), dtype=np.float32)
	else:
		kvec = np.asarray(kappa, dtype=np.float32)
		if kvec.shape != (n,):
			raise ValueError("kappa must be scalar or shape (n,)")
		if np.any(kvec <= 0):
			raise ValueError("all kappa must be > 0")

	# Precompute per-column decays = exp(-kappa_j) once (host-side, vectorized)
	decays = np.exp(-kvec).astype(np.float32, copy=False)

	# Ensure contiguity for the kernels
	if not x.flags.c_contiguous:
		x = np.ascontiguousarray(x)

	fdt = np.float32

	if in_place:
		# In-place is safe (uses y[t-1] already written, and x[t] not yet overwritten).
		if not np.issubdtype(x.dtype, np.floating):
			x = x.astype(fdt, copy=True)   # upcast ints to float
		t_jit._HKP_inp(x, decays)
		return x

	# Out-of-place
	if out is None:
		out = np.empty_like(x, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	else:
		if out.shape != x.shape:
			raise ValueError("out has wrong shape")
		if not out.flags.c_contiguous:
			out = np.ascontiguousarray(out)

	t_jit._HKP_out(x if x.flags.c_contiguous else np.ascontiguousarray(x), decays, out)
	return out

#ID 14
def t_EMA(x, delta, out=None, in_place=False):
	if x.ndim != 2: raise ValueError("x must be (m,n)")
	m, n = x.shape
	# normalize delta -> (n,)
	if np.isscalar(delta): 
		d = np.full(n, int(delta), np.int64)
	else:
		d = np.asarray(delta, np.int64)
		if d.shape != (n,): 
			raise ValueError("delta must be scalar or (n,)")
	if np.any(d < 1): 
		raise ValueError("delta must be >= 1")
	# alpha per column
	a = (2.0 / (d.astype(np.float32) + 1.0)).astype(np.float32)
	if not x.flags.c_contiguous: 
		x = np.ascontiguousarray(x)
	fdt = np.float32
	if in_place:
		if not np.issubdtype(x.dtype, np.floating): 
			x = x.astype(fdt, copy=True)
		t_jit._EMA_inp(x, a)
		return x
	if out is None: 
		out = np.empty_like(x, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	elif out.shape != x.shape: 
		raise ValueError("out wrong shape")
	t_jit._EMA_out(x, a, out)
	return out

#ID 15
def t_DOE(x, delta1, delta2, out=None, in_place=False): 
	'''Difference of Exponential Averages'''
	if x.ndim != 2: 
		raise ValueError("x must be (m,n)")
	m, n = x.shape
	def _norm(v):
		if np.isscalar(v): 
			vv = np.full(n, int(v), np.int64)
		else:
			vv = np.asarray(v, np.int64)
			if vv.shape != (n,): 
				raise ValueError("delta must be scalar or (n,)")
		if np.any(vv < 1): 
			raise ValueError("delta must be >= 1")
		return (2.0 / (vv.astype(np.float32) + 1.0)).astype(np.float32)
	af = _norm(delta1)
	aslow = _norm(delta2)
	if not x.flags.c_contiguous: 
		x = np.ascontiguousarray(x)
	fdt = np.float32
	if in_place:
		if not np.issubdtype(x.dtype, np.floating): 
			x = x.astype(fdt, copy=True)
		t_jit._DOE_inp(x, af, aslow)
		return x
	if out is None: 
		out = np.empty_like(x, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	elif out.shape != x.shape: 
		raise ValueError("out wrong shape")
	t_jit._DOE_out(x, af, aslow, out)
	return out

#ID 16
def t_MDN(x, window, min_count=1, out=None, in_place=False):
    """
    Rolling median on (m,n) array, column-wise.
    window & min_count can be scalar or shape (n,).
    """
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape

    if np.isscalar(window):
        wins = np.full(n, int(window), np.int64)
    else:
        wins = np.asarray(window, np.int64)
        if wins.shape != (n,):
            raise ValueError("window must be scalar or (n,)")
    if np.any(wins < 1):
        raise ValueError("window >= 1")

    if np.isscalar(min_count):
        mc = np.full(n, int(min_count), np.int64)
    else:
        mc = np.asarray(min_count, np.int64)
        if mc.shape != (n,):
            raise ValueError("min_count must be scalar or (n,)")

    fdt = np.float32
    if out is None:
        out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
    elif out.shape != x.shape:
        raise ValueError("out wrong shape")
    if not np.issubdtype(out.dtype, np.floating):
        raise TypeError("out must be float for NaNs")

    src = x
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)
    if not np.issubdtype(src.dtype, np.floating):
        src = src.astype(fdt, copy=False)

    t_jit._MDN_core_heaps(src, wins, mc, out)
    return out if not in_place else out

#ID 17
def t_ZSC(x, window, min_count=2, out=None, in_place=False):
	if x.ndim != 2: 
		raise ValueError("x must be (m,n)")
	m, n = x.shape
	wins = t_jit._norm_vec(window, n)
	mc = t_jit._norm_vec(min_count, n)
	fdt = np.float32
	if in_place:
		if not np.issubdtype(x.dtype, np.floating): 
			x = x.astype(fdt, copy=True)
		t_jit._ZSC_inp(x, wins, mc)
		return x
	if out is None: 
		out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	elif out.shape != x.shape: 
		raise ValueError("out wrong shape")
	if not np.issubdtype(out.dtype, np.floating): 
		raise TypeError("out must be float")
	t_jit._ZSC_out(x, wins, mc, out)
	return out

#ID 18
def t_STD(x, window, min_count=1, out=None, in_place=False):
	if x.ndim != 2: 
		raise ValueError("x must be (m,n)")
	m, n = x.shape
	wins = t_jit._norm_vec(window, n)
	mc = t_jit._norm_vec(min_count, n)
	fdt = np.float32
	if in_place:
		if not np.issubdtype(x.dtype, np.floating): 
			x = x.astype(fdt, copy=True)
		t_jit._STD_inp(x, wins, mc)
		return x
	if out is None: 
		out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	elif out.shape != x.shape: 
		raise ValueError("out wrong shape")
	if not np.issubdtype(out.dtype, np.floating): 
		raise TypeError("out must be float")
	t_jit._STD_out(x, wins, mc, out)
	return out

#ID 19
def t_SSN(x, out=None):
	dst = x if out is None else out
	# ensure float (avoid integer division)
	if not np.issubdtype((dst.dtype if out is not None else x.dtype), np.floating):
		if out is None: 
			x = x.astype(np.float32, copy=True)
			dst = x
		else: 
			raise TypeError("out must be float dtype for softsign")
	np.abs(x if dst is not x else dst, out=dst)           # dst = |x|
	np.add(dst, 1.0, out=dst)                             # dst = 1 + |x|
	np.divide(x if dst is not x else dst, dst, out=dst)   # dst = x / (1+|x|)
	np.multiply(dst, 2.0, out=dst)                        # dst = 2 * softsign
	return dst

#ID 20
def t_AGR(x, a, window, min_count=1, out=None, in_place=False, prefer_float32=True):
    if x.shape != a.shape or x.ndim != 2:
        raise ValueError("x and a must be same shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(window, n); mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32 if prefer_float32 else np.float64
    if in_place:
        if not np.issubdtype(x.dtype, np.floating): x = x.astype(fdt, copy=True)
        t_jit._AGR_inp(x if x.flags.c_contiguous else np.ascontiguousarray(x),
                 a if a.flags.c_contiguous else np.ascontiguousarray(a),
                 wins, mc)
        return x
    if out is None:
        out = np.empta(x.shape, dtape=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
    t_jit._AGR_out(x if x.flags.c_contiguous else np.ascontiguousarray(x),
             a if a.flags.c_contiguous else np.ascontiguousarray(a),
             wins, mc, out if out.flags.c_contiguous else np.ascontiguousarray(out))
    return out

#ID 21
def t_COR(x, a, window, min_count=2, out=None, in_place=False, prefer_float32=True):
    if x.shape != a.shape or x.ndim != 2:
        raise ValueError("x and a must be same shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(window, n); mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32 if prefer_float32 else np.float64
    if in_place:
        if not np.issubdtype(x.dtype, np.floating): x = x.astype(fdt, copy=True)
        t_jit._COR_inp(x if x.flags.c_contiguous else np.ascontiguousarray(x),
                  a if a.flags.c_contiguous else np.ascontiguousarray(a),
                  wins, mc)
        return x
    if out is None:
        out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
    t_jit._COR_out(x if x.flags.c_contiguous else np.ascontiguousarray(x),
              a if a.flags.c_contiguous else np.ascontiguousarray(a),
              wins, mc,
              out if out.flags.c_contiguous else np.ascontiguousarray(out))
    return out

