"""transform_ops.py

Unified, batch-friendly wrappers for gene instantiation ops.

All ops operate on 2D arrays shaped (N, B) where:
- N: length of time series (axis 0)
- B: batch size (number of genes instantiated in parallel)

Unified API:
    apply(func_id, x, *, alpha=None, delta1=None, delta2=None, kappa=None,
          out=None, in_place=False, **kwargs) -> ndarray

`x` is the primary input matrix.
`alpha` may be scalar, (B,), or (N,B) depending on op.
`delta1`, `delta2` typically scalar or (B,) integer vectors.
`kappa` typically scalar or (B,) float vectors.

This module depends on `transform_jit` (Numba kernels) for rolling ops.
"""

from __future__ import annotations

import numpy as np
import transform_jit as t_jit


# --------------------------- helpers ---------------------------

def _as_c_contig(x: np.ndarray) -> np.ndarray:
    return x if x.flags.c_contiguous else np.ascontiguousarray(x)


def _norm_int_vec(v, n: int, *, name: str) -> np.ndarray:
    """Normalize scalar or (n,) into int64 (n,)"""
    if v is None:
        return np.full(n, 1, dtype=np.int64)
    if np.isscalar(v):
        return np.full(n, int(v), dtype=np.int64)
    vv = np.asarray(v, dtype=np.int64)
    if vv.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape (n,)")
    return vv


def _norm_float_vec(v, n: int, *, name: str, default: float = 1.0) -> np.ndarray:
    if v is None:
        return np.full(n, float(default), dtype=np.float32)
    if np.isscalar(v):
        return np.full(n, float(v), dtype=np.float32)
    vv = np.asarray(v, dtype=np.float32)
    if vv.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape (n,)")
    return vv


def _ensure_float_out(x: np.ndarray, out, *, float_dtype=np.float32) -> np.ndarray:
    """Allocate/validate out that can carry NaNs."""
    if out is None:
        if np.issubdtype(x.dtype, np.floating):
            return np.empty_like(x)
        return np.empty(x.shape, dtype=float_dtype)
    if out.shape != x.shape:
        raise ValueError("out has wrong shape")
    if not np.issubdtype(out.dtype, np.floating):
        raise TypeError("out must be float dtype")
    return out


# --------------------------- ops 1..21 ---------------------------

# ID 1
def t_MAX(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be 2D (m,n)")
    m, n = x.shape
    d1 = _norm_int_vec(delta1 if delta1 is not None else 1, n, name="delta1")

    # min_count scalar fast path or vector
    if np.isscalar(min_count):
        mc_scalar = int(min_count)
        mc_vec = np.empty(1, dtype=np.int64)
    else:
        mc_vec = np.asarray(min_count, dtype=np.int64)
        if mc_vec.shape != (n,):
            raise ValueError("min_count must be scalar or shape (n,)")
        mc_scalar = -1

    x_c = _as_c_contig(x)
    float_dtype = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(float_dtype, copy=True)
        t_jit._MAX_inp(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
        return x_c

    out = _ensure_float_out(x_c, out, float_dtype=float_dtype)
    out_c = _as_c_contig(out)
    t_jit._MAX_out(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out_c)
    return out_c


# ID 2
def t_MIN(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be 2D (m,n)")
    m, n = x.shape
    d1 = _norm_int_vec(delta1 if delta1 is not None else 1, n, name="delta1")

    if np.isscalar(min_count):
        mc_scalar = int(min_count)
        mc_vec = np.empty(1, dtype=np.int64)
    else:
        mc_vec = np.asarray(min_count, dtype=np.int64)
        if mc_vec.shape != (n,):
            raise ValueError("min_count must be scalar or shape (n,)")
        mc_scalar = -1

    x_c = _as_c_contig(x)
    float_dtype = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(float_dtype, copy=True)
        t_jit._MIN_inp(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
        return x_c

    out = _ensure_float_out(x_c, out, float_dtype=float_dtype)
    out_c = _as_c_contig(out)
    t_jit._MIN_out(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out_c)
    return out_c


# ID 3
def t_AVG(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be 2D (m,n)")
    m, n = x.shape
    d1 = _norm_int_vec(delta1 if delta1 is not None else 1, n, name="delta1")

    if np.isscalar(min_count):
        mc_scalar = int(min_count)
        mc_vec = np.empty(1, dtype=np.int64)
    else:
        mc_vec = np.asarray(min_count, dtype=np.int64)
        if mc_vec.shape != (n,):
            raise ValueError("min_count must be scalar or shape (n,)")
        mc_scalar = -1

    x_c = _as_c_contig(x)
    float_dtype = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(float_dtype, copy=True)
        t_jit._AVG_inp(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64))
        return x_c

    out = _ensure_float_out(x_c, out, float_dtype=float_dtype)
    out_c = _as_c_contig(out)
    t_jit._AVG_out(x_c, d1, mc_scalar, mc_vec if mc_scalar < 0 else np.empty(1, np.int64), out_c)
    return out_c


# ID 4
def t_NEG(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    np.negative(x, out=dst)
    return dst


# ID 5
def t_DIF(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    a = np.asanyarray(alpha if alpha is not None else 0.0)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    np.subtract(x, a, out=dst)
    return dst


# ID 6
def t_ADD(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    a = np.asanyarray(alpha if alpha is not None else 0.0)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    np.add(x, a, out=dst)
    return dst


# ID 7
def t_SQR(x, alpha=None, delta1=None, delta2=None, kappa=None, out=None, in_place=False):
    dst = x if (out is None and in_place) else (out if out is not None else np.empty_like(x))
    # optional: sanitize NaNs/Infs coming in
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # prevent overflow in float32:
    # sqrt(max_float32) ≈ 1.844e19, so clamp |x| to that
    if x.dtype == np.float32:
        LIM = np.float32(np.sqrt(np.finfo(np.float32).max))
        np.clip(x, -LIM, LIM, out=dst)
        np.multiply(dst, dst, out=dst)
    else:
        np.multiply(x, x, out=dst)
    return dst


# ID 8
def t_SIN(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    if not np.issubdtype(dst.dtype, np.floating):
        if out is None:
            x = x.astype(np.float32, copy=True)
            dst = x
        else:
            raise TypeError("out must be float")
    np.clip(x, -1.0, 1.0, out=dst)
    np.multiply(dst, np.pi, out=dst)
    np.sin(dst, out=dst)
    return dst


# ID 9
def t_COS(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    if not np.issubdtype(dst.dtype, np.floating):
        if out is None:
            x = x.astype(np.float32, copy=True)
            dst = x
        else:
            raise TypeError("out must be float")
    np.clip(x, -1.0, 1.0, out=dst)
    np.multiply(dst, np.pi, out=dst)
    np.cos(dst, out=dst)
    return dst


# ID 10
def t_ASN(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    if not np.issubdtype(dst.dtype, np.floating):
        if out is None:
            x = x.astype(np.float32, copy=True)
            dst = x
        else:
            raise TypeError("out must be float")
    np.clip(x, -1.0, 1.0, out=dst)
    np.arcsin(dst, out=dst)
    return dst


# ID 11
def t_ACS(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    if not np.issubdtype(dst.dtype, np.floating):
        if out is None:
            x = x.astype(np.float32, copy=True)
            dst = x
        else:
            raise TypeError("out must be float")
    np.clip(x, -1.0, 1.0, out=dst)
    np.arccos(dst, out=dst)
    return dst


# ID 12
def t_RNG(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False):
    # Uses rolling MAX/MIN then rescales to [-1,1]
    m, n = x.shape
    mx = np.empty_like(x, dtype=np.float32)
    mn = np.empty_like(x, dtype=np.float32)
    t_MAX(x, alpha=alpha, delta1=delta1, delta2=delta2, kappa=kappa, min_count=min_count, out=mx, in_place=False)
    # For MIN use delta2 as window if provided else delta1
    d2 = delta2 if delta2 is not None else delta1
    t_MIN(x, alpha=alpha, delta1=d2, delta2=delta2, kappa=kappa, min_count=min_count, out=mn, in_place=False)

    fdt = np.float32
    if in_place:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(fdt, copy=True)
        dst = x
    else:
        if out is None:
            out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
        dst = out

    np.subtract(x, mn, out=dst)
    np.subtract(mx, mn, out=mx)  # range
    np.divide(dst, mx, out=dst, where=(mx > 0))
    np.multiply(dst, 2.0, out=dst)
    np.subtract(dst, 1.0, out=dst)
    return dst


# ID 13
def t_HKP(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    m, n = x.shape
    kvec = _norm_float_vec(kappa, n, name="kappa", default=1.0)
    if np.any(kvec <= 0):
        raise ValueError("kappa must be > 0")
    decays = np.exp(-kvec).astype(np.float32, copy=False)

    x_c = _as_c_contig(x)
    fdt = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._HKP_inp(x_c, decays)
        return x_c

    if out is None:
        out = np.empty_like(x_c, dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = _as_c_contig(out)
    t_jit._HKP_out(x_c, decays, out_c)
    return out_c


# ID 14
def t_EMA(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape
    d = _norm_int_vec(delta1, n, name="delta1")
    if np.any(d < 1):
        raise ValueError("delta1 must be >= 1")
    a = (2.0 / (d.astype(np.float32) + 1.0)).astype(np.float32)
    x_c = _as_c_contig(x)
    fdt = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._EMA_inp(x_c, a)
        return x_c

    if out is None:
        out = np.empty_like(x_c, dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = _as_c_contig(out)
    t_jit._EMA_out(x_c, a, out_c)
    return out_c


# ID 15
def t_DOE(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape

    def _ema_alpha(dv):
        d = _norm_int_vec(dv, n, name="delta")
        if np.any(d < 1):
            raise ValueError("delta must be >= 1")
        return (2.0 / (d.astype(np.float32) + 1.0)).astype(np.float32)

    af = _ema_alpha(delta1)
    aslow = _ema_alpha(delta2 if delta2 is not None else delta1)

    x_c = _as_c_contig(x)
    fdt = np.float32

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._DOE_inp(x_c, af, aslow)
        return x_c

    if out is None:
        out = np.empty_like(x_c, dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = _as_c_contig(out)
    t_jit._DOE_out(x_c, af, aslow, out_c)
    return out_c


# ID 16
def t_MDN(x, alpha=None, delta1=None, delta2=None, kappa=None,
          min_count=1, out=None, in_place=False, prefer_float32=True):
    """
    Rolling median using _MDN_core_sort (stable compile).
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D (m,n)")

    m, n = x.shape

    # normalize window sizes per column
    wins = t_jit._norm_vec(delta1, n)  # your existing helper
    mc   = t_jit._norm_vec(min_count, n)

    fdt = np.float32 if prefer_float32 else np.float64
    src = x
    if not np.issubdtype(src.dtype, np.floating):
        src = src.astype(fdt, copy=False)
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)

    if out is None:
        out = np.empty((m, n), dtype=src.dtype)
    elif not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)

    t_jit._MDN_core_sort(src, wins, mc, out)
    return out


# ID 17
def t_ZSC(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=2, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1 if delta1 is not None else 1, n)
    mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32

    if in_place:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(fdt, copy=True)
        t_jit._ZSC_inp(_as_c_contig(x), wins, mc)
        return x

    if out is None:
        out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
    if out.shape != x.shape:
        raise ValueError("out wrong shape")
    if not np.issubdtype(out.dtype, np.floating):
        raise TypeError("out must be float")

    t_jit._ZSC_out(_as_c_contig(x), wins, mc, _as_c_contig(out))
    return out


# ID 18
def t_STD(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False):
    if x.ndim != 2:
        raise ValueError("x must be (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1 if delta1 is not None else 1, n)
    mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32

    if in_place:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(fdt, copy=True)
        t_jit._STD_inp(_as_c_contig(x), wins, mc)
        return x

    if out is None:
        out = np.empty(x.shape, dtype=(x.dtype if np.issubdtype(x.dtype, np.floating) else fdt))
    if out.shape != x.shape:
        raise ValueError("out wrong shape")
    if not np.issubdtype(out.dtype, np.floating):
        raise TypeError("out must be float")

    t_jit._STD_out(_as_c_contig(x), wins, mc, _as_c_contig(out))
    return out


# ID 19
def t_SSN(x, alpha=None, delta1=None, delta2=None, kappa=None, *, out=None, **kwargs):
    dst = x if out is None else out
    if not np.issubdtype(dst.dtype, np.floating):
        if out is None:
            x = x.astype(np.float32, copy=True)
            dst = x
        else:
            raise TypeError("out must be float")
    np.abs(x if dst is not x else dst, out=dst)
    np.add(dst, 1.0, out=dst)
    np.divide(x if dst is not x else dst, dst, out=dst)
    np.multiply(dst, 2.0, out=dst)
    return dst


# ID 20
def t_AGR(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=1, out=None, in_place=False, prefer_float32=True):
    if alpha is None:
        raise ValueError("AGR requires alpha matrix")
    if x.shape != alpha.shape or x.ndim != 2:
        raise ValueError("x and alpha must match shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1 if delta1 is not None else 1, n)
    mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32 if prefer_float32 else np.float64

    x_c = _as_c_contig(x)
    a_c = _as_c_contig(alpha)

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._AGR_inp(x_c, a_c, wins, mc)
        return x_c

    if out is None:
        out = np.empty(x.shape, dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = _as_c_contig(out)
    t_jit._AGR_out(x_c, a_c, wins, mc, out_c)
    return out_c


# ID 21
def t_COR(x, alpha=None, delta1=None, delta2=None, kappa=None, *, min_count=2, out=None, in_place=False, prefer_float32=True):
    if alpha is None:
        raise ValueError("COR requires alpha matrix")
    if x.shape != alpha.shape or x.ndim != 2:
        raise ValueError("x and alpha must match shape (m,n)")
    m, n = x.shape
    wins = t_jit._norm_vec(delta1 if delta1 is not None else 1, n)
    mc = t_jit._norm_vec(min_count, n)
    fdt = np.float32 if prefer_float32 else np.float64

    x_c = _as_c_contig(x)
    a_c = _as_c_contig(alpha)

    if in_place:
        if not np.issubdtype(x_c.dtype, np.floating):
            x_c = x_c.astype(fdt, copy=True)
        t_jit._COR_inp(x_c, a_c, wins, mc)
        return x_c

    if out is None:
        out = np.empty(x.shape, dtype=(x_c.dtype if np.issubdtype(x_c.dtype, np.floating) else fdt))
    out_c = _as_c_contig(out)
    t_jit._COR_out(x_c, a_c, wins, mc, out_c)
    return out_c


# --------------------------- unified dispatch ---------------------------

FUNC_TABLE = {
    0: None,
    1: t_MAX,
    2: t_MIN,
    3: t_AVG,
    4: t_NEG,
    5: t_DIF,
    6: t_ADD,
    7: t_SQR,
    8: t_SIN,
    9: t_COS,
    10: t_ASN,
    11: t_ACS,
    12: t_RNG,
    13: t_HKP,
    14: t_EMA,
    15: t_DOE,
    16: t_MDN,
    17: t_ZSC,
    18: t_STD,
    19: t_SSN,
    20: t_AGR,
    21: t_COR,
}


def apply(func_id: int,
          x: np.ndarray,
          *,
          alpha=None,
          delta1=None,
          delta2=None,
          kappa=None,
          out=None,
          in_place: bool = False,
          **kwargs):
    """Unified API for all ops."""
    fid = int(func_id)
    if fid == 0:
        return x if out is None else out
    fn = FUNC_TABLE.get(fid)
    if fn is None:
        raise ValueError(f"Unknown func_id={fid}")
    return fn(x, alpha=alpha, delta1=delta1, delta2=delta2, kappa=kappa, out=out, in_place=in_place, **kwargs)
