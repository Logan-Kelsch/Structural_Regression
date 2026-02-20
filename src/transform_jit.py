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


@njit(parallel=False, fastmath=True)
def _MAX_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _MAX_col(x[:, j], w, mc, out[:, j])


@njit(parallel=False, fastmath=True)
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


@njit(parallel=False, fastmath=True)
def _MIN_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _MIN_col(x[:, j], w, mc, out[:, j])


@njit(parallel=False, fastmath=True)
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


@njit(parallel=False, fastmath=True)
def _AVG_out(x, windows, mc_scalar, mc_vec, out):
    m, n = x.shape
    for j in prange(n):
        w = windows[j]
        mc = mc_scalar if mc_scalar >= 0 else mc_vec[j]
        _AVG_col(x[:, j], w, mc, out[:, j])


@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
def _HKP_inp(x, decays):
    m, n = x.shape
    for j in prange(n):
        d = decays[j]
        # x[0, j] stays as-is (initial condition)
        for t in range(1, m):
            x[t, j] = d * x[t-1, j] + x[t, j]

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
def _EMA_inp(x, alphas):
    m, n = x.shape
    for j in prange(n):
        a = alphas[j]
        om = 1.0 - a
        # y[0] already x[0]
        for t in range(1, m):
            x[t, j] = om * x[t-1, j] + a * x[t, j]

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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

# ------ Some heap helpers -------

@njit(inline='always')
def _key(v):
    # Mimic NumPy's tendency to push NaNs to the end in ordering-like ops:
    # treat NaN as +inf so they behave as "largest".
    return np.inf if np.isnan(v) else v

@njit(inline='always')
def _swap(vals, idxs, a, b):
    tv = vals[a]; vals[a] = vals[b]; vals[b] = tv
    ti = idxs[a]; idxs[a] = idxs[b]; idxs[b] = ti

@njit(inline='always')
def _push_min(vals, idxs, size, v, i):
    # min-heap by _key(value)
    vals[size] = v
    idxs[size] = i
    k = size
    size += 1
    while k > 0:
        p = (k - 1) // 2
        if _key(vals[p]) <= _key(vals[k]):
            break
        _swap(vals, idxs, p, k)
        k = p
    return size

@njit(inline='always')
def _push_max(vals, idxs, size, v, i):
    # max-heap by _key(value)
    vals[size] = v
    idxs[size] = i
    k = size
    size += 1
    while k > 0:
        p = (k - 1) // 2
        if _key(vals[p]) >= _key(vals[k]):
            break
        _swap(vals, idxs, p, k)
        k = p
    return size

@njit(inline='always')
def _pop_min(vals, idxs, size):
    # returns (v, i, new_size)
    v0 = vals[0]
    i0 = idxs[0]
    size -= 1
    if size > 0:
        vals[0] = vals[size]
        idxs[0] = idxs[size]
        k = 0
        while True:
            l = 2 * k + 1
            if l >= size:
                break
            r = l + 1
            c = l
            if r < size and _key(vals[r]) < _key(vals[l]):
                c = r
            if _key(vals[k]) <= _key(vals[c]):
                break
            _swap(vals, idxs, k, c)
            k = c
    return v0, i0, size

@njit(inline='always')
def _pop_max(vals, idxs, size):
    # returns (v, i, new_size)
    v0 = vals[0]
    i0 = idxs[0]
    size -= 1
    if size > 0:
        vals[0] = vals[size]
        idxs[0] = idxs[size]
        k = 0
        while True:
            l = 2 * k + 1
            if l >= size:
                break
            r = l + 1
            c = l
            if r < size and _key(vals[r]) > _key(vals[l]):
                c = r
            if _key(vals[k]) >= _key(vals[c]):
                break
            _swap(vals, idxs, k, c)
            k = c
    return v0, i0, size

@njit(inline='always')
def _prune_heap_min(vals, idxs, size, start):
    # lazily remove expired indices (idx < start)
    while size > 0 and idxs[0] < start:
        _, _, size = _pop_min(vals, idxs, size)
    return size

@njit(inline='always')
def _prune_heap_max(vals, idxs, size, start):
    while size > 0 and idxs[0] < start:
        _, _, size = _pop_max(vals, idxs, size)
    return size


# ---------- core rolling median ----------

@njit(parallel=False, fastmath=False)
def _MDN_core_heaps(src, wins, mc, dst):
    """
    Rolling median per column using two heaps (O(m log w)).
    - src: (m,n) float32/float64 contiguous
    - wins: (n,) int64 window size
    - mc: (n,) int64 min_count
    - dst: (m,n) float32/float64, NaN where not enough points
    """
    m, n = src.shape

    for j in prange(n):
        w = wins[j]
        mcount = mc[j]
        if mcount < 1:
            mcount = 1
        if w < 1:
            w = 1

        # Heaps store (value, index). Capacity w is enough (active window <= w).
        low_vals  = np.empty(w, src.dtype)   # max-heap (lower half)
        low_idxs  = np.empty(w, np.int64)
        high_vals = np.empty(w, src.dtype)   # min-heap (upper half)
        high_idxs = np.empty(w, np.int64)

        low_size = 0
        high_size = 0

        # side[t] tells where index t was placed while active: 0=low, 1=high
        # (used to decrement counts on expiration without searching heaps)
        side = np.empty(m, np.int8)

        low_count = 0   # active element count logically in low
        high_count = 0  # active element count logically in high

        for t in range(m):
            start = t - w + 1  # window includes indices [start..t]

            # Expire the element that just left the window
            if t >= w:
                old = t - w
                if side[old] == 0:
                    low_count -= 1
                else:
                    high_count -= 1

            # Insert new element
            v = src[t, j]
            if low_size == 0:
                low_size = _push_max(low_vals, low_idxs, low_size, v, t)
                side[t] = 0
                low_count += 1
            else:
                # Compare vs current low top (max of lower half)
                # Ensure low top is valid before using it
                low_size = _prune_heap_max(low_vals, low_idxs, low_size, start)
                if low_size == 0:
                    low_size = _push_max(low_vals, low_idxs, low_size, v, t)
                    side[t] = 0
                    low_count += 1
                else:
                    low_top = low_vals[0]
                    if _key(v) <= _key(low_top):
                        low_size = _push_max(low_vals, low_idxs, low_size, v, t)
                        side[t] = 0
                        low_count += 1
                    else:
                        high_size = _push_min(high_vals, high_idxs, high_size, v, t)
                        side[t] = 1
                        high_count += 1

            # Prune expired from heap tops (physical cleanup)
            low_size = _prune_heap_max(low_vals, low_idxs, low_size, start)
            high_size = _prune_heap_min(high_vals, high_idxs, high_size, start)

            # Rebalance so that low_count == high_count or low_count == high_count+1
            # (median is always from top of low, and optionally top of high)
            while low_count > high_count + 1:
                low_size = _prune_heap_max(low_vals, low_idxs, low_size, start)
                vmove, imove, low_size = _pop_max(low_vals, low_idxs, low_size)
                high_size = _push_min(high_vals, high_idxs, high_size, vmove, imove)
                side[imove] = 1
                low_count -= 1
                high_count += 1
                high_size = _prune_heap_min(high_vals, high_idxs, high_size, start)

            while low_count < high_count:
                high_size = _prune_heap_min(high_vals, high_idxs, high_size, start)
                vmove, imove, high_size = _pop_min(high_vals, high_idxs, high_size)
                low_size = _push_max(low_vals, low_idxs, low_size, vmove, imove)
                side[imove] = 0
                high_count -= 1
                low_count += 1
                low_size = _prune_heap_max(low_vals, low_idxs, low_size, start)

            # Output
            L = t + 1 if t + 1 < w else w  # active window length ignoring NaN notion
            if L >= mcount:
                low_size = _prune_heap_max(low_vals, low_idxs, low_size, start)
                if low_count > high_count:
                    dst[t, j] = low_vals[0]
                else:
                    high_size = _prune_heap_min(high_vals, high_idxs, high_size, start)
                    # even window: average of two middle values
                    dst[t, j] = (low_vals[0] + high_vals[0]) * 0.5
            else:
                dst[t, j] = np.nan

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=False, cache=True)
def _MDN_core_sort(src, wins, mc, dst):
    """
    Rolling median per column by sorting the active window each time.

    src: (m,n) float32/float64 contiguous
    wins: (n,) int64 window sizes
    mc: (n,) int64 min_count
    dst: (m,n) float32/float64
    """
    m, n = src.shape

    for j in prange(n):
        w = wins[j]
        mcount = mc[j]
        if mcount < 1:
            mcount = 1
        if w < 1:
            w = 1

        # ring buffer holds last w values
        ring = np.empty(w, dtype=src.dtype)

        # temp window for sorting (max size w)
        tmp = np.empty(w, dtype=src.dtype)

        filled = 0
        pos = 0

        for t in range(m):
            # insert new value into ring
            ring[pos] = src[t, j]
            pos += 1
            if pos == w:
                pos = 0

            if filled < w:
                filled += 1

            # active window length
            L = filled  # <= w
            if L < mcount:
                dst[t, j] = np.nan
                continue

            # copy active values into tmp[0:L]
            # ring is not in chronological order; doesn't matter for median
            for k in range(L):
                tmp[k] = ring[k]

            # sort only the active part
            tmp_slice = tmp[:L]
            tmp_slice.sort()

            mid = L >> 1
            if (L & 1) == 1:
                dst[t, j] = tmp_slice[mid]
            else:
                dst[t, j] = (tmp_slice[mid - 1] + tmp_slice[mid]) * 0.5

# ---------------------------
# ZSC: rolling z-score (x - mean)/std over window
# STD: rolling std over window
# O(1) updates via running sum & sumsq; per-column; min_count
# ---------------------------

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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
@njit(parallel=False, fastmath=True)
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

@njit(parallel=False, fastmath=True)
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
@njit(parallel=False, fastmath=True)
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

import numpy as np
from numba import njit, prange

@njit(parallel=False, fastmath=True, cache=True)
def _COR_out(X, Y, wins, mc, OUT):
    """
    Rolling correlation with:
      - w forced to >= 2 (so any 1 becomes 2)
      - mcount forced to >= 2
      - non-finite x/y treated as 0.0 (prevents NaN propagation)

    OUT is NaN until L >= mcount; after that, finite values (or 0.0 if degenerate).
    """
    m, n = X.shape
    for j in prange(n):
        w = wins[j]
        if w < 2:
            w = 2  # <-- force 1s/0s to 2
        mcount = mc[j]
        if mcount < 2:
            mcount = 2

        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0

        for t in range(m):
            x = X[t, j]
            y = Y[t, j]

            # sanitize inputs (NaN/Inf -> 0)
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0

            sx += x; sy += y
            sxx += x*x; syy += y*y
            sxy += x*y

            if t >= w:
                xo = X[t - w, j]
                yo = Y[t - w, j]
                if not np.isfinite(xo):
                    xo = 0.0
                if not np.isfinite(yo):
                    yo = 0.0
                sx -= xo; sy -= yo
                sxx -= xo*xo; syy -= yo*yo
                sxy -= xo*yo
                L = w
            else:
                L = t + 1

            if L >= mcount:
                cov = sxy - (sx * sy) / L
                vx  = sxx - (sx * sx) / L
                vy  = syy - (sy * sy) / L

                # If variance collapsed, correlation undefined -> 0
                if vx <= 0.0 or vy <= 0.0:
                    OUT[t, j] = 0.0
                else:
                    # add small epsilon to keep denom safe
                    denom = np.sqrt(vx * vy + 1e-18)
                    OUT[t, j] = cov / denom
            else:
                OUT[t, j] = np.nan