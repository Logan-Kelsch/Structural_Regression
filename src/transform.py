#the _jit file simply has code for faster computation on passed variables using numba
import transform_jit as t_jit

import numpy as np


#ID 1
def t_MAX(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling max down axis=0 with per-column delta1s and min_count.

	Parameters
	----------
	x : (m, n) array. If integers are passed and min_count causes NaNs,
		we'll upcast to float (float32 by default) to hold NaNs.
	delta1 : int or (n,) int
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

	# Normalize delta1 -> (n,)
	if np.isscalar(delta1):
		delta1s = np.full(n, int(delta1), dtype=np.int64)
	else:
		delta1s = np.asarray(delta1, dtype=np.int64)
		if delta1s.shape != (n,):
			raise ValueError("delta1 must be scalar or shape (n,)")

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
	needs_nan = True  # rolling delta1s with min_count generally emit NaN at the top
	#was going to make this an option but changed my mind
	float_dtype = np.float32

	if in_place:
		# In-place path: ensure dtype can carry NaN
		if not np.issubdtype(x.dtype, np.floating):
			# upcast integers -> float to carry NaNs
			x = x.astype(float_dtype, copy=True)
		# run kernel
		t_jit._MAX_inp(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
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

	t_jit._MAX_out(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 2
def t_MIN(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling min down axis=0 with per-column delta1s and min_count.
	Mirrors t_MAX API/behavior.
	"""
	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")
	m, n = x.shape

	# delta1 -> (n,)
	if np.isscalar(delta1):
		delta1s = np.full(n, int(delta1), dtype=np.int64)
	else:
		delta1s = np.asarray(delta1, dtype=np.int64)
		if delta1s.shape != (n,):
			raise ValueError("delta1 must be scalar or shape (n,)")

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
		t_jit._MIN_inp(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
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

	t_jit._MIN_out(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 3
def t_AVG(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	min_count=1,
	out: np.ndarray | None = None,
	in_place: bool = False
):
	"""
	Rolling mean down axis=0 with per-column delta1s and min_count.
	Uses a running-sum O(m) kernel; emits NaN until min_count is met.
	"""
	if x.ndim != 2:
		raise ValueError("x must be 2D (m, n)")
	m, n = x.shape

	# delta1 -> (n,)
	if np.isscalar(delta1):
		delta1s = np.full(n, int(delta1), dtype=np.int64)
	else:
		delta1s = np.asarray(delta1, dtype=np.int64)
		if delta1s.shape != (n,):
			raise ValueError("delta1 must be scalar or shape (n,)")

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
		t_jit._AVG_inp(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
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

	t_jit._AVG_out(x, delta1s, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out)
	return out

#ID 4
def t_NEG(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	out	=	None
):
	np.negative(x, out=(x if out is None else out))
	return x if out is None else out

#ID 5
def t_DIF(x, alpha, delta1, delta2, kappa, out=None):
    """
    y = x - alpha
    alpha can be scalar, (n,), or (m,n)
    """
    dst = x if out is None else out
    a = np.asanyarray(alpha)
    if a.ndim == 1:
        a = a.reshape(1, -1)  # broadcast over rows
    np.subtract(x, a, out=dst)
    return dst

#ID 6
def t_ADD(x, alpha, delta1, delta2, kappa, out=None):
    """
    y = x + alpha
    alpha can be scalar, (n,), or (m,n)
    """
    dst = x if out is None else out
    a = np.asanyarray(alpha)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    np.add(x, a, out=dst)
    return dst

#ID 7
def t_SQR(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	out	=	None
):
	np.square(x, out=(x if out is None else out))
	return x if out is None else out

#ID 8
def t_SIN(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	out=None
):
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
def t_COS(
	x,
	alpha,
	delta1,
	delta2,
	kappa,
	out=None):
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
def t_ASN(
	x,
	alpha,
	delta1,
	delta2,
	kappa,out=None):
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
def t_ACS(
	x,
	alpha,
	delta1,
	delta2,
	kappa,out=None):
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
def t_RNG(x, alpha, delta1, delta2, kappa, min_count: int = 1, out=None, in_place: bool = False):
    """
    Range-normalize using rolling max/min.
    Uses delta1 for max window and delta2 for min window.
    """
    fdt = np.float32
    m, n = x.shape

    mx = np.empty((m, n), dtype=np.float32)
    mn = np.empty((m, n), dtype=np.float32)

    # Call with full signature (alpha/delta2/kappa are unused in MAX/MIN but required)
    t_MAX(x, alpha, delta1, delta2, kappa, min_count=min_count, out=mx, in_place=False)
    t_MIN(x, alpha, delta1, delta2, kappa, min_count=min_count, out=mn, in_place=False)

    if in_place:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(fdt, copy=True)
        dst = x
    else:
        if out is None:
            out = np.empty((m, n), dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
        dst = out

    # dst = 2 * (x - mn) / (mx - mn) - 1
    np.subtract(x, mn, out=dst)
    np.subtract(mx, mn, out=mx)                 # reuse mx as range
    np.divide(dst, mx, out=dst, where=(mx > 0))
    np.multiply(dst, 2.0, out=dst)
    np.subtract(dst, 1.0, out=dst)
    return dst

#ID 13
def t_HKP(
	x,
	alpha,
	delta1,
	delta2,
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
def t_EMA(
	x,
	alpha,
	delta1,
	delta2,
	kappa,out=None, in_place=False):
	if x.ndim != 2: raise ValueError("x must be (m,n)")
	m, n = x.shape
	# normalize delta1 -> (n,)
	if np.isscalar(delta1): 
		d = np.full(n, int(delta1), np.int64)
	else:
		d = np.asarray(delta1, np.int64)
		if d.shape != (n,): 
			raise ValueError("delta1 must be scalar or (n,)")
	if np.any(d < 1): 
		raise ValueError("delta1 must be >= 1")
	# alpha per column
	alpha = (2.0 / (d.astype(np.float32) + 1.0)).astype(np.float32)
	if not x.flags.c_contiguous: 
		x = np.ascontiguousarray(x)
	fdt = np.float32
	if in_place:
		if not np.issubdtype(x.dtype, np.floating): 
			x = x.astype(fdt, copy=True)
		t_jit._EMA_inp(x, alpha)
		return x
	if out is None: 
		out = np.empty_like(x, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
	elif out.shape != x.shape: 
		raise ValueError("out wrong shape")
	t_jit._EMA_out(x, alpha, out)
	return out

#ID 15
def t_DOE(
	x,
	alpha,
	delta1,
	delta2,
	kappa,out=None, in_place=False): 
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
				raise ValueError("delta1 must be scalar or (n,)")
		if np.any(vv < 1): 
			raise ValueError("delta1 must be >= 1")
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
def t_MDN(
	x,
	alpha,
	delta1,
	delta2,
	kappa,min_count=1, out=None, in_place=False):
    """
    Rolling median on (m,n) array, column-wise.
    delta1 & min_count can be scalar or shape (n,).
    """
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape

    if np.isscalar(delta1):
        wins = np.full(n, int(delta1), np.int64)
    else:
        wins = np.asarray(delta1, np.int64)
        if wins.shape != (n,):
            raise ValueError("delta1 must be scalar or (n,)")
    if np.any(wins < 1):
        raise ValueError("delta1 >= 1")

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
def t_ZSC(
	x,
	alpha,
	delta1,
	delta2,
	kappa,min_count=2, out=None, in_place=False):
	if x.ndim != 2: 
		raise ValueError("x must be (m,n)")
	m, n = x.shape
	wins = t_jit._norm_vec(delta1, n)
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
def t_STD(
	x,
	alpha,
	delta1,
	delta2,
	kappa,min_count=1, out=None, in_place=False):
	if x.ndim != 2: 
		raise ValueError("x must be (m,n)")
	m, n = x.shape
	wins = t_jit._norm_vec(delta1, n)
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
def t_SSN(
	x,
	alpha,
	delta1,
	delta2,
	kappa,out=None):
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
def t_AGR(x, alpha, delta1, delta2, kappa, min_count=1, out=None, in_place=False, prefer_float32=True):
    if x.shape != alpha.shape or x.ndim != 2:
        raise ValueError("x and alpha must be same shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1, n)
    mc   = t_jit._norm_vec(min_count, n)
    fdt  = np.float32 if prefer_float32 else np.float64

    x_c = x if x.flags.c_contiguous else np.ascontiguousarray(x)
    a_c = alpha if alpha.flags.c_contiguous else np.ascontiguousarray(alpha)

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._AGR_inp(x_c, a_c, wins, mc)
        return x_c

    if out is None:
        out = np.empty((m, n), dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = out if out.flags.c_contiguous else np.ascontiguousarray(out)
    t_jit._AGR_out(x_c, a_c, wins, mc, out_c)
    return out_c

#ID 21
def t_COR(x, alpha, delta1, delta2, kappa, min_count=2, out=None, in_place=False, prefer_float32=True):
    if x.shape != alpha.shape or x.ndim != 2:
        raise ValueError("x and alpha must be same shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1, n)
    mc   = t_jit._norm_vec(min_count, n)
    fdt  = np.float32 if prefer_float32 else np.float64

    x_c = x if x.flags.c_contiguous else np.ascontiguousarray(x)
    a_c = alpha if alpha.flags.c_contiguous else np.ascontiguousarray(alpha)

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._COR_inp(x_c, a_c, wins, mc)
        return x_c

    if out is None:
        out = np.empty((m, n), dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = out if out.flags.c_contiguous else np.ascontiguousarray(out)
    t_jit._COR_out(x_c, a_c, wins, mc, out_c)
    return out_c

