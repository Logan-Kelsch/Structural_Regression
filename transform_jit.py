'''
9/21/2025 - Fate happens now, you decide.

Incredible discoveries with recent genetic programming.
Discovered Multi-Expression Programming and am writing this out for general use.
'''

import bottleneck as b
import numpy as np
from numba import njit, prange

# --- MAX (rolling max, per-column windows) ---

@njit(fastmath=True, inline='always')
def _MAX_col(col, w, mc, out):
    """
    Compute rolling max for a single column.
    col: (m,) read-only source
    out: (m,) destination
    """
    m = col.shape[0]
    if w < 1: w = 1
    if mc < 1: mc = 1

    dq = np.empty(w, np.int64)  # circular deque of indices
    head = 0
    size = 0

    for i in range(m):
        s = i - w + 1

        # expire head
        while size > 0:
            front = dq[head]
            if front < s:
                head = (head + 1) % w
                size -= 1
            else:
                break

        # pop back while current >= last
        while size > 0:
            back_pos = (head + size - 1) % w
            if col[i] >= col[dq[back_pos]]:
                size -= 1
            else:
                break

        # push i
        back_pos = (head + size) % w
        dq[back_pos] = i
        size += 1

        # valid count in the window
        if s < 0:
            valid = i + 1
        else:
            valid = w

        if valid >= mc:
            out[i] = col[dq[head]]
        else:
            out[i] = np.nan


@njit(parallel=True, fastmath=True)
def _MAX_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _MAX_col(x[:, j], w, mc, out[:, j])


@njit(parallel=True, fastmath=True)
def _MAX_inp(x, windows, mc_scalar, mc_vec):
    """
    In-place: makes one per-column scratch copy for the read path.
    Writes results back into x[:, j].
    """
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]

        col = np.empty(m, x.dtype)
        # manual copy (numba-friendly)
        for i in range(m):
            col[i] = x[i, j]

        _MAX_col(col, w, mc, x[:, j])

# --- MIN (rolling min, per-column windows) ---

@njit(fastmath=True, inline='always')
def _MIN_col(col, w, mc, out):
    m = col.shape[0]
    if w < 1: w = 1
    if mc < 1: mc = 1

    dq = np.empty(w, np.int64)  # circular deque of indices (monotonically increasing values)
    head = 0
    size = 0

    for i in range(m):
        s = i - w + 1

        # expire head
        while size > 0:
            front = dq[head]
            if front < s:
                head = (head + 1) % w
                size -= 1
            else:
                break

        # pop back while current <= last  (note <= to keep earliest index for ties consistent)
        while size > 0:
            back_pos = (head + size - 1) % w
            if col[i] <= col[dq[back_pos]]:
                size -= 1
            else:
                break

        # push i
        back_pos = (head + size) % w
        dq[back_pos] = i
        size += 1

        # valid count
        valid = (i + 1) if s < 0 else w
        if valid >= mc:
            out[i] = col[dq[head]]
        else:
            out[i] = np.nan


@njit(parallel=True, fastmath=True)
def _MIN_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _MIN_col(x[:, j], w, mc, out[:, j])


@njit(parallel=True, fastmath=True)
def _MIN_inp(x, windows, mc_scalar, mc_vec):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        # scratch read-only copy of the column
        col = np.empty(m, x.dtype)
        for i in range(m):
            col[i] = x[i, j]
        _MIN_col(col, w, mc, x[:, j])


# --- AVG (rolling average over last up-to-w samples, per column) ---

@njit(fastmath=True, inline='always')
def _AVG_col(col, w, mc, out):
    m = col.shape[0]
    if w < 1: w = 1
    if mc < 1: mc = 1

    s = 0.0  # running sum
    for i in range(m):
        s += col[i]
        if i >= w:
            s -= col[i - w]
        # current window length (min(i+1, w))
        win_len = i + 1 if i < (w - 1) else w
        if win_len >= mc:
            out[i] = s / win_len
        else:
            out[i] = np.nan


@njit(parallel=True, fastmath=True)
def _AVG_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _AVG_col(x[:, j], w, mc, out[:, j])


@njit(parallel=True, fastmath=True)
def _AVG_inp(x, windows, mc_scalar, mc_vec):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        col = np.empty(m, x.dtype)
        for i in range(m):
            col[i] = x[i, j]
        _AVG_col(col, w, mc, x[:, j])

# --- HKP (exponentiated hawkes process kernel version with independent kappa per column) ---

@njit(parallel=True, fastmath=True)
def _HKP_inp(x, decays):
    m, n = x.shape
    for j in prange(n):
        d = decays[j]
        # x[0, j] stays as-is (initial condition)
        for t in range(1, m):
            x[t, j] = d * x[t-1, j] + x[t, j]

@njit(parallel=True, fastmath=True)
def _HKP_out(x, decays, out):
    m, n = x.shape
    for j in prange(n):
        out[0, j] = x[0, j]
        d = decays[j]
        for t in range(1, m):
            out[t, j] = d * out[t-1, j] + x[t, j]


# ---------------------------
# EMA: y[0]=x[0]; y[t]=(1-a)*y[t-1] + a*x[t]
# a = 2/(delta+1)  with delta >= 1 (scalar or (n,))
# ---------------------------

@njit(parallel=True, fastmath=True)
def _EMA_inp(x, alphas):
    m, n = x.shape
    for j in prange(n):
        a = alphas[j]
        om = 1.0 - a
        # y[0] already x[0]
        for t in range(1, m):
            x[t, j] = om * x[t-1, j] + a * x[t, j]

@njit(parallel=True, fastmath=True)
def _EMA_out(x, alphas, out):
    m, n = x.shape
    for j in prange(n):
        a = alphas[j]
        om = 1.0 - a
        out[0, j] = x[0, j]
        for t in range(1, m):
            out[t, j] = om * out[t-1, j] + a * x[t, j]

# ---------------------------
# DOE: EMA(delta_fast) - EMA(delta_slow), one pass
# delta_fast, delta_slow: scalar or (n,)
# ---------------------------

@njit(parallel=True, fastmath=True)
def _DOE_inp(x, af, aslow):
    m, n = x.shape
    for j in prange(n):
        a1 = af[j]; b1 = 1.0 - a1
        a2 = aslow[j]; b2 = 1.0 - a2
        # carry two EMAs in-place by staging prev outputs in registers
        y1_prev = x[0, j]              # EMA fast at t=0
        y2_prev = x[0, j]              # EMA slow at t=0
        x0 = x[0, j]
        x[0, j] = y1_prev - y2_prev    # = 0
        for t in range(1, m):
            xt = x[t, j]
            y1 = b1 * y1_prev + a1 * xt
            y2 = b2 * y2_prev + a2 * xt
            x[t, j] = y1 - y2
            y1_prev = y1
            y2_prev = y2

@njit(parallel=True, fastmath=True)
def _DOE_out(x, af, aslow, out):
    m, n = x.shape
    for j in prange(n):
        a1 = af[j]; b1 = 1.0 - a1
        a2 = aslow[j]; b2 = 1.0 - a2
        y1_prev = x[0, j]; y2_prev = x[0, j]
        out[0, j] = y1_prev - y2_prev
        for t in range(1, m):
            xt = x[t, j]
            y1 = b1 * y1_prev + a1 * xt
            y2 = b2 * y2_prev + a2 * xt
            out[t, j] = y1 - y2
            y1_prev = y1; y2_prev = y2

# ---------------------------
# MDN: rolling median per column
# (copy current window into a scratch and np.partition -> fast for small/medium windows)
# ---------------------------

@njit(parallel=True)
def _MDN_core(src, wins, mc, dst):
    m, n = src.shape
    for j in prange(n):
        w = wins[j]
        mcount = mc[j]
        if mcount < 1: mcount = 1
        # work buffer once per column (max window)
        work = np.empty(w, src.dtype)
        for t in range(m):
            # window bounds
            s = t - w + 1
            if s < 0:
                L = t + 1
                # fill prefix of work
                for k in range(L):
                    work[k] = src[k, j]
            else:
                L = w
                # copy window into work
                start = s
                for k in range(L):
                    work[k] = src[start + k, j]
            if L >= mcount:
                mid = L // 2
                # partition to median
                np.partition(work[:L], mid, axis=0)
                if (L & 1) == 1:
                    dst[t, j] = work[mid]
                else:
                    # average the two middle values
                    np.partition(work[:L], mid-1, axis=0)
                    dst[t, j] = (work[mid] + work[mid-1]) * 0.5
            else:
                dst[t, j] = np.nan

# ---------------------------
# ZSC: rolling z-score (x - mean)/std over window
# STD: rolling std over window
# O(1) updates via running sum & sumsq; per-column; min_count
# ---------------------------

@njit(parallel=True, fastmath=True)
def _ZSC_inp(x, wins, mc):
    m, n = x.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        # copy source column (needed because we overwrite)
        col = np.empty(m, x.dtype)
        for t in range(m): col[t] = x[t, j]
        s = 0.0; ss = 0.0
        for t in range(m):
            ct = col[t]
            s += ct; ss += ct*ct
            if t >= w:
                old = col[t - w]
                s -= old; ss -= old*old
                L = w
            else:
                L = t + 1
            if L >= mcount and L > 1:
                mu = s / L
                var = (ss / L) - mu*mu
                if var < 0.0: var = 0.0
                x[t, j] = (ct - mu) / np.sqrt(var + 1e-12)
            else:
                x[t, j] = np.nan

@njit(parallel=True, fastmath=True)
def _ZSC_out(x, wins, mc, out):
    m, n = x.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        s = 0.0; ss = 0.0
        for t in range(m):
            ct = x[t, j]
            s += ct; ss += ct*ct
            if t >= w:
                old = x[t - w, j]
                s -= old; ss -= old*old
                L = w
            else:
                L = t + 1
            if L >= mcount and L > 1:
                mu = s / L
                var = (ss / L) - mu*mu
                if var < 0.0: var = 0.0
                out[t, j] = (ct - mu) / np.sqrt(var + 1e-12)
            else:
                out[t, j] = np.nan

@njit(parallel=True, fastmath=True)
def _STD_inp(x, wins, mc):
    m, n = x.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        col = np.empty(m, x.dtype)
        for t in range(m): col[t] = x[t, j]
        s = 0.0; ss = 0.0
        for t in range(m):
            ct = col[t]
            s += ct; ss += ct*ct
            if t >= w:
                old = col[t - w]
                s -= old; ss -= old*old
                L = w
            else:
                L = t + 1
            if L >= mcount and L > 0:
                mu = s / L
                var = (ss / L) - mu*mu
                if var < 0.0: var = 0.0
                x[t, j] = np.sqrt(var + 1e-12)
            else:
                x[t, j] = np.nan

@njit(parallel=True, fastmath=True)
def _STD_out(x, wins, mc, out):
    m, n = x.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        s = 0.0; ss = 0.0
        for t in range(m):
            ct = x[t, j]
            s += ct; ss += ct*ct
            if t >= w:
                old = x[t - w, j]
                s -= old; ss -= old*old
                L = w
            else:
                L = t + 1
            if L >= mcount and L > 0:
                mu = s / L
                var = (ss / L) - mu*mu
                if var < 0.0: var = 0.0
                out[t, j] = np.sqrt(var + 1e-12)
            else:
                out[t, j] = np.nan

def _norm_vec(v, n):
    if np.isscalar(v): return np.full(n, int(v), np.int64)
    vv = np.asarray(v, np.int64)
    if vv.shape != (n,): raise ValueError("expects scalar or (n,)")
    return vv

# ================================================================
# 1) t_AGR : ULTRA-FAST agreement pulse (rolling mean of sign products)
#     agr[t] = mean_{window}( sign(X)*sign(Y) )  in [-1, 1]
# ================================================================
@njit(parallel=True, fastmath=True)
def _AGR_inp(X, Y, wins, mc):
    m, n = X.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        # integer-like counter for agreement sum (but keep float32 for speed)
        s = 0.0
        # local copy of sign products to remove when window slides
        buf = np.empty(m, np.float32)
        for t in range(m):
            sp = 0.0
            x = X[t, j]; y = Y[t, j]
            # sign product quickly (treat 0 as neutral)
            if x > 0:
                sp = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
            elif x < 0:
                sp = -1.0 if y > 0 else (1.0 if y < 0 else 0.0)
            else:
                sp = 0.0
            buf[t] = sp
            s += sp
            if t >= w:
                s -= buf[t - w]
                L = w
            else:
                L = t + 1
            if L >= mcount:
                # write back into X in-place (hot path)
                X[t, j] = s / L
            else:
                X[t, j] = np.nan

@njit(parallel=True, fastmath=True)
def _AGR_out(X, Y, wins, mc, OUT):
    m, n = X.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 1: mcount = 1
        s = 0.0
        buf = np.empty(m, np.float32)
        for t in range(m):
            sp = 0.0
            x = X[t, j]; y = Y[t, j]
            if x > 0:
                sp = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
            elif x < 0:
                sp = -1.0 if y > 0 else (1.0 if y < 0 else 0.0)
            else:
                sp = 0.0
            buf[t] = sp
            s += sp
            if t >= w:
                s -= buf[t - w]
                L = w
            else:
                L = t + 1
            OUT[t, j] = (s / L) if L >= mcount else np.nan

# ================================================================
# 2) t_CORR : Rolling Pearson correlation per column
#     r = cov(x,y) / (stdx*stdy), computed with O(1) updates
# ================================================================
@njit(parallel=True, fastmath=True)
def _COR_inp(X, Y, wins, mc):
    m, n = X.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 2: mcount = 2
        # local copies to prevent destroying past values we need for sliding
        colx = np.empty(m, X.dtype); coly = np.empty(m, Y.dtype)
        for t in range(m): colx[t] = X[t, j]; coly[t] = Y[t, j]
        sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
        for t in range(m):
            x = colx[t]; y = coly[t]
            sx += x; sy += y
            sxx += x*x; syy += y*y
            sxy += x*y
            if t >= w:
                xo = colx[t - w]; yo = coly[t - w]
                sx -= xo; sy -= yo
                sxx -= xo*xo; syy -= yo*yo
                sxy -= xo*yo
                L = w
            else:
                L = t + 1
            if L >= mcount:
                # Pearson r = (sxy - sx*sy/L) / sqrt((sxx - sx^2/L)*(syy - sy^2/L))
                cov = sxy - (sx*sy)/L
                vx  = sxx - (sx*sx)/L
                vy  = syy - (sy*sy)/L
                if vx <= 0.0 or vy <= 0.0:
                    X[t, j] = 0.0
                else:
                    X[t, j] = cov / np.sqrt(vx*vy + 1e-18)
            else:
                X[t, j] = np.nan

@njit(parallel=True, fastmath=True)
def _COR_out(X, Y, wins, mc, OUT):
    m, n = X.shape
    for j in prange(n):
        w = wins[j]; mcount = mc[j]
        if mcount < 2: mcount = 2
        sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
        for t in range(m):
            x = X[t, j]; y = Y[t, j]
            sx += x; sy += y
            sxx += x*x; syy += y*y
            sxy += x*y
            if t >= w:
                xo = X[t - w, j]; yo = Y[t - w, j]
                sx -= xo; sy -= yo
                sxx -= xo*xo; syy -= yo*yo
                sxy -= xo*yo
                L = w
            else:
                L = t + 1
            if L >= mcount:
                cov = sxy - (sx*sy)/L
                vx  = sxx - (sx*sx)/L
                vy  = syy - (sy*sy)/L
                if vx <= 0.0 or vy <= 0.0:
                    OUT[t, j] = 0.0
                else:
                    OUT[t, j] = cov / np.sqrt(vx*vy + 1e-18)
            else:
                OUT[t, j] = np.nan