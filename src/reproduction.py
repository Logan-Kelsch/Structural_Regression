import numpy as np
import initialization as _I

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor-flag bit positions
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns


class Selector:
    '''
    Class object that will hold relevant information for parent selection
    NOTE FOR WHEN THIS IS EXPANDED. CAN SELECT BASED OFF OF CORRELATION IN G DIM
        NOTE OR CAN CLUSTER BASED OFF OF CORRELATION IN G DIM.
    '''
    def __init__(
        self,
        method:str='Threshold',
        percent:float=0.2
    ):
        self._method = method
        self._percent = percent




def flush_population(X, keep_idx):
    """
    In-place population compaction on:
      - X._instructions : np.ndarray shape (G, 11)
      - X._G_idx        : 1D np.ndarray of candidate-removal indices

    keep_idx: 1D array-like of gene indices that must be preserved.

    Behavior:
      - Removes indices in X._G_idx that are not in keep_idx
      - Packs kept indices from X._G_idx into the lowest indices of the original X._G_idx
      - Updates parent displacement references in cols 5..9 using SENSOR_FLAGS bits 5..9
      - Clears removed rows to all zeros (no gaps)
      - Updates X._G_idx to the new candidate indices after compaction
      - Returns old_to_new mapping (len G, -1 for removed)

    Returns
    -------
    old_to_new : np.ndarray[int64] shape (G,)
        Mapping old gene index -> new gene index, -1 for removed.
    """
    instructions = X._instructions
    if instructions.ndim != 2 or instructions.shape[1] != 11:
        raise ValueError("X._instructions must have shape (G, 11).")

    G = instructions.shape[0]

    G_idx = np.asarray(X._G_idx, dtype=np.int64).reshape(-1)
    keep_idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)

    # keep only in-range
    G_idx = G_idx[(G_idx >= 0) & (G_idx < G)]
    keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < G)]

    G_sorted = np.unique(G_idx)
    keep_sorted = np.unique(keep_idx)

    # candidates in G_idx that must be preserved
    kept_in_G = np.intersect1d(G_sorted, keep_sorted, assume_unique=True)
    removed   = np.setdiff1d(G_sorted, kept_in_G, assume_unique=True)

    # pack kept genes into lowest indices of original G_idx
    target_positions = G_sorted[:kept_in_G.size]

    # build old->new mapping (identity unless remapped/removed)
    old_to_new = np.arange(G, dtype=np.int64)
    if removed.size:
        old_to_new[removed] = -1
    for old_i, new_i in zip(kept_in_G.tolist(), target_positions.tolist()):
        old_to_new[old_i] = new_i

    # ---- build compacted instructions (needs a new array; then write back) ----
    out = np.zeros_like(instructions)
    source_old_for_new = np.full(G, -1, dtype=np.int64)

    keep_old = np.flatnonzero(old_to_new >= 0)
    out[old_to_new[keep_old]] = instructions[keep_old]
    source_old_for_new[old_to_new[keep_old]] = keep_old

    # rewrite pop_idx for non-empty rows
    non_empty = np.any(out != 0, axis=1)
    out[non_empty, 0] = np.arange(G, dtype=out.dtype)[non_empty]

    # ---- fix parent displacements for preserved rows ----
    sensor_flags = out[:, 4].astype(np.uint32, copy=False)

    for new_child in np.flatnonzero(non_empty):
        old_child = int(source_old_for_new[new_child])
        if old_child < 0:
            continue

        sf = sensor_flags[new_child]
        slot_mask = ((sf >> _BITS_5_9) & np.uint32(1)).astype(bool)
        if not np.any(slot_mask):
            continue

        cols = _COLS_5_9[slot_mask]
        disp = out[new_child, cols].astype(np.int64, copy=False)

        for col, d in zip(cols.tolist(), disp.tolist()):
            if d >= 0:
                continue  # only negative displacements are parents

            old_parent = old_child + int(d)
            if not (0 <= old_parent < G):
                raise ValueError(
                    f"Invalid parent: old_child={old_child}, disp={d} -> old_parent={old_parent}"
                )

            new_parent = int(old_to_new[old_parent])
            if new_parent < 0:
                raise ValueError(
                    f"Kept gene {old_child} depends on removed parent {old_parent}. "
                    f"Include {old_parent} in keep_idx (or keep closure of ancestors)."
                )

            new_disp = new_parent - new_child  # should remain negative
            out[new_child, col] = np.array(new_disp, dtype=out.dtype)

    # ---- update X in-place ----
    X._instructions = out

    mapped = old_to_new[G_sorted]
    X._G_idx = np.unique(mapped[mapped >= 0]).astype(np.int64, copy=False)
    X._L_idx = np.union1d(X._G_idx, X._T_idx)

    return old_to_new

import numpy as np

def map_fitness(eval, fitness_map):
    """
    Remap eval["F"] according to fitness_map.

    Assumes:
    - eval["F"] is a 1D array of shape (N,)
    - fitness_map is a 1D integer array of shape (N,)
    - fitness_map[i] == -1 means delete/ignore eval["F"][i]
    - fitness_map[i] >= 0 means place eval["F"][i] into mapped_fitness[fitness_map[i]]
    - any output locations that are never assigned stay 0
    """
    fitness = np.asarray(eval["F"])
    fitness_map = np.asarray(fitness_map)

    if fitness.ndim != 1 or fitness_map.ndim != 1:
        raise ValueError('eval["F"] and fitness_map must both be 1D arrays')
    if fitness.shape[0] != fitness_map.shape[0]:
        raise ValueError('eval["F"] and fitness_map must have the same length')

    mapped_fitness = np.zeros_like(fitness)

    keep = fitness_map >= 0
    mapped_fitness[fitness_map[keep]] = fitness[keep]

    return mapped_fitness


def locate_top_genes(F, p):
    """
    Return two arrays of selected gene indices:
    1) selected indices sorted by index
    2) selected indices sorted by fitness value (highest to lowest)

    Selection rule:
    - At most floor(p * G) genes are considered, where G = len(F)
    - From those top genes, keep only entries with fitness >= 0
    - So the final number selected can be any count in [0, floor(p * G)]

    Parameters
    ----------
    F : array-like, shape (G,)
        Fitness scores.
    p : float
        Fraction in [0, 1].

    Returns
    -------
    idx_by_index : np.ndarray
        Selected indices sorted in increasing index order.
    idx_by_value : np.ndarray
        Selected indices sorted by fitness value from highest to lowest.
    """
    F = np.asarray(F)

    if F.ndim != 1:
        raise ValueError("F must be a 1D array")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")

    G = F.shape[0]
    max_k = int(np.floor(p * G))

    if max_k == 0:
        empty = np.array([], dtype=int)
        return empty, empty

    # candidate top max_k indices, unordered
    idx = np.argpartition(F, -max_k)[-max_k:]

    # sort candidates by fitness value, highest first
    idx_by_value = idx[np.argsort(F[idx])[::-1]]

    # keep only nonnegative selected values
    idx_by_value = idx_by_value[F[idx_by_value] > 0]

    # same selected set, but sorted by index
    idx_by_index = np.sort(idx_by_value)

    return idx_by_index, idx_by_value

def reproduce(
    population  :   _I.Population,
    grammar     :   _I.Grammar,
    selector    :   Selector,
    evaluation  :   any
):
    match(selector._method):
        case 'Threshold':
            selected_parents, _ = locate_top_genes(evaluation["F"], p=selector._percent)
            survived_genes = _I.family_tree_indices(population._instructions, selected_parents, include_terminals=True)
            g_map = flush_population(population, survived_genes)
            _I.generate_instructions(population, grammar)

            #making a thingy to be able to return
            reproduction_stats = {
                "selected_parents":selected_parents,
                "survived_genes":survived_genes,
                "gene_mapping":g_map
            }

            return reproduction_stats

        case _:
            raise NotImplementedError(f"Selector method of ({selector._method}) not implemented into reproduce function")


def _resolve_chunk_rows(population, chunk_num=None):
    """
    Return:
      X_rows  : row-scoped view of population._X_inst
      row_lo  : absolute start row
      row_hi  : absolute stop row

    Behavior:
      - chunk_num is None -> all rows
      - chunk_num is int  -> only that chunk's rows
    """
    X = np.asarray(population._X_inst)

    if X.ndim != 2:
        raise ValueError("population._X_inst must be a 2d array")

    if chunk_num is None:
        return X, 0, X.shape[0]

    chunk_num = int(chunk_num)

    if hasattr(population, "get_chunk_bounds"):
        row_lo, row_hi = population.get_chunk_bounds(chunk_num)

    elif hasattr(population, "get_chunk_slice"):
        row_sl = population.get_chunk_slice(chunk_num)
        row_lo = 0 if row_sl.start is None else int(row_sl.start)
        row_hi = X.shape[0] if row_sl.stop is None else int(row_sl.stop)

    elif hasattr(population, "_chunk_row_bounds"):
        row_lo, row_hi = population._chunk_row_bounds[chunk_num]
        row_lo = int(row_lo)
        row_hi = int(row_hi)

    else:
        raise AttributeError(
            "population must provide get_chunk_bounds(chunk_num), "
            "get_chunk_slice(chunk_num), or _chunk_row_bounds"
        )

    if row_lo < 0 or row_hi > X.shape[0] or row_lo >= row_hi:
        raise ValueError(
            f"Invalid chunk bounds [{row_lo}:{row_hi}) for X with {X.shape[0]} rows"
        )

    return X[row_lo:row_hi, :], row_lo, row_hi
  

def purge_indistinguishable(population, evaluation, threshold=0.02, chunk_num=None):
    """
    Purge highly similar surviving gene columns from population._G_idx using
    their instantiated output behavior and return the surviving gene indices in
    their NEW post-flush locations.

    Parameters
    ----------
    population : object
        Population-like object expected to contain at least:
        - population._X_inst : np.ndarray, shape (n_rows, n_cols)
            Instantiated matrix whose columns are compared for similarity.
        - population._G_idx : array-like of int
            Gene column indices eligible for purge comparison.
        - population._instructions : np.ndarray
            Instruction matrix used to recover all family-tree ancestors before flush.

        Optional chunk helpers supported when chunk_num is not None:
        - population.get_chunk_bounds(chunk_num)
        - population.get_chunk_slice(chunk_num)
        - population._chunk_row_bounds

    evaluation : dict
        Evaluation dictionary expected to contain:
        - evaluation["F"] : 1d np.ndarray
            Fitness vector indexed by column/gene position.
            Only genes in population._G_idx with F > 0 are considered.

    threshold : float, default=0.02
        Similarity threshold in abs-correlation distance space:
            distance(i, j) = 1 - abs(corr(i, j))
        If distance <= threshold, the two genes are treated as indistinguishable
        and one of them is removed.
        Example:
            threshold = 0.02 means any pair with |corr| >= 0.98 is considered too similar.

    chunk_num : int | None, default=None
        Row-scope selector for comparison:
        - None : compare genes across all rows in population._X_inst
        - int  : compare genes only on that walk-forward chunk's rows

    Functionality
    -------------
    1) Select candidate genes from population._G_idx with positive fitness.
    2) Restrict comparison rows to either the whole matrix or one chunk.
    3) Compute pairwise abs-correlation similarity between candidate columns.
    4) Remove one member of each too-similar pair using:
        - higher F survives
        - if F ties, lower original column index survives
    5) Recover all required ancestor columns using family_tree_indices(...).
    6) Flush the population so survivors and dependencies are compacted.
    7) Return the surviving candidate genes in their NEW locations after flush.

    Return
    ------
    selected_new : np.ndarray, dtype=int64
        1d sorted array of surviving gene column indices AFTER flush_population(...)
        has remapped them into their new compacted column locations.

    Notes
    -----
    - This function mutates population via flush_population(...).
    - Similarity is measured on instantiated values, not on instruction text.
    - Constant columns are handled safely:
        - identical constants are treated as perfectly correlated
        - unrelated constant-vs-nonconstant pairs remain distance 1
    """
    import numpy as np
    import initialization as _I

    X = np.asarray(population._X_inst)

    if X.ndim != 2:
        raise ValueError("population._X_inst must be a 2d array")

    # resolve row scope
    if chunk_num is None:
        X_rows = X
    else:
        chunk_num = int(chunk_num)

        if hasattr(population, "get_chunk_bounds"):
            row_lo, row_hi = population.get_chunk_bounds(chunk_num)

        elif hasattr(population, "get_chunk_slice"):
            row_sl = population.get_chunk_slice(chunk_num)
            row_lo = 0 if row_sl.start is None else int(row_sl.start)
            row_hi = X.shape[0] if row_sl.stop is None else int(row_sl.stop)

        elif hasattr(population, "_chunk_row_bounds"):
            row_lo, row_hi = population._chunk_row_bounds[chunk_num]
            row_lo = int(row_lo)
            row_hi = int(row_hi)

        else:
            raise AttributeError(
                "population must provide get_chunk_bounds(chunk_num), "
                "get_chunk_slice(chunk_num), or _chunk_row_bounds"
            )

        if row_lo < 0 or row_hi > X.shape[0] or row_lo >= row_hi:
            raise ValueError(
                f"Invalid chunk bounds [{row_lo}:{row_hi}) for X with {X.shape[0]} rows"
            )

        X_rows = X[row_lo:row_hi, :]

    G_idx = np.asarray(population._G_idx, dtype=np.int64)
    F = np.asarray(evaluation["F"])

    if F.ndim != 1:
        raise ValueError('evaluation["F"] must be a 1d array')
    if np.any(G_idx < 0) or np.any(G_idx >= X_rows.shape[1]):
        raise ValueError("population._G_idx contains invalid column indices")
    if np.any(G_idx >= F.shape[0]):
        raise ValueError('evaluation["F"] is too short for indices in population._G_idx')

    # only compare positive-fitness genes
    pos_mask = F[G_idx] > 0
    cols = G_idx[pos_mask]

    # early case
    if cols.size <= 1:
        survivors = _I.family_tree_indices(
            population._instructions,
            cols,
            include_terminals=True
        )
        g_map = flush_population(population, survivors)

        if isinstance(g_map, dict):
            selected_new = np.asarray(
                [g_map[c] for c in cols if c in g_map],
                dtype=np.int64
            )
        else:
            g_map = np.asarray(g_map)
            selected_new = np.asarray(g_map[cols], dtype=np.int64)
            if np.issubdtype(selected_new.dtype, np.integer):
                selected_new = selected_new[selected_new >= 0]

        selected_new.sort()
        return selected_new

    X_sub = X_rows[:, cols].astype(np.float64, copy=False)
    F_sub = F[cols].astype(np.float64, copy=False)
    n_models = cols.size

    # build robust abs-correlation matrix
    std = X_sub.std(axis=0)
    nonconst = std > 0
    abs_corr = np.zeros((n_models, n_models), dtype=np.float64)

    if np.any(nonconst):
        Xn = X_sub[:, nonconst]
        corr_nc = np.corrcoef(Xn, rowvar=False)
        corr_nc = np.nan_to_num(corr_nc, nan=0.0)
        abs_corr[np.ix_(nonconst, nonconst)] = np.abs(corr_nc)

    # explicit constant-column handling
    const_idx = np.where(~nonconst)[0]
    if const_idx.size:
        for a in range(const_idx.size):
            ia = const_idx[a]
            abs_corr[ia, ia] = 1.0
            for b in range(a + 1, const_idx.size):
                ib = const_idx[b]
                if np.all(X_sub[:, ia] == X_sub[:, ib]):
                    abs_corr[ia, ib] = 1.0
                    abs_corr[ib, ia] = 1.0

    np.fill_diagonal(abs_corr, 1.0)

    # keep stronger model from each too-similar pair
    alive = np.ones(n_models, dtype=bool)

    # higher F first; for ties, lower original col index first
    order = np.lexsort((cols, -F_sub))

    for i in order:
        if not alive[i]:
            continue

        for j in range(n_models):
            if j == i or not alive[j]:
                continue

            dist = 1.0 - abs_corr[i, j]
            if dist <= threshold:
                if F_sub[i] > F_sub[j]:
                    alive[j] = False
                elif F_sub[i] < F_sub[j]:
                    alive[i] = False
                    break
                else:
                    if cols[i] < cols[j]:
                        alive[j] = False
                    else:
                        alive[i] = False
                        break

    selected_old = np.sort(cols[alive])

    survivors = _I.family_tree_indices(
        population._instructions,
        selected_old,
        include_terminals=True
    )
    g_map = flush_population(population, survivors)

    # remap selected survivors into NEW compacted locations
    if isinstance(g_map, dict):
        selected_new = np.asarray(
            [g_map[c] for c in selected_old if c in g_map],
            dtype=np.int64
        )
    else:
        g_map = np.asarray(g_map)

        if g_map.ndim != 1:
            raise ValueError("flush_population remap must be dict or 1d array")

        if np.any(selected_old < 0) or np.any(selected_old >= g_map.shape[0]):
            raise ValueError("selected survivor indices out of bounds for returned remap")

        selected_new = np.asarray(g_map[selected_old], dtype=np.int64)
        if np.issubdtype(selected_new.dtype, np.integer):
            selected_new = selected_new[selected_new >= 0]

    selected_new.sort()
    return selected_new

def purge_indistinguishable_old(population, evaluation, threshold=0.02, chunk_num=None):
    """
    Return the surviving model indices c from population._G_idx such that:
      1) evaluation["F"][c] > 0
      2) models that are too similar are purged using abs-correlation distance

    Similarity rule:
      distance(i, j) = 1 - abs(corr(i, j))
      if distance(i, j) <= threshold, they are considered indistinguishable

    Tie break:
      - remove lower F
      - if F ties, remove higher c index

    Parameters
    ----------
    population : object
        Must contain:
          population._X_inst : 2d numpy array, shape (n_samples, n_models_total)
          population._G_idx  : iterable of model column indices c to consider

        For chunked use, population should also expose one of:
          - get_chunk_bounds(chunk_num)
          - get_chunk_slice(chunk_num)
          - _chunk_row_bounds

    evaluation : dict
        Must contain:
          evaluation["F"] : 1d score array indexed by c

    threshold : float, default=0.02
        Maximum allowed abs-correlation distance for two models to be treated
        as indistinguishable. Example:
            threshold=0.02  -> remove one of any pair with |corr| >= 0.98

    chunk_num : int | None, default=None
        - None -> use all rows of population._X_inst
        - int  -> use only that chunk's rows

    Returns
    -------
    selected : np.ndarray
        1d array of surviving original column indices c
    """
    X_rows, row_lo, row_hi = _resolve_chunk_rows(population, chunk_num)

    G_idx = np.asarray(population._G_idx, dtype=int)
    F = np.asarray(evaluation["F"])

    if F.ndim != 1:
        raise ValueError('evaluation["F"] must be a 1d array')
    if np.any(G_idx < 0) or np.any(G_idx >= X_rows.shape[1]):
        raise ValueError("population._G_idx contains invalid column indices")
    if np.any(G_idx >= F.shape[0]):
        raise ValueError('evaluation["F"] is too short for indices in population._G_idx')

    # only consider models in _G_idx with positive F
    pos_mask = F[G_idx] > 0
    cols = G_idx[pos_mask]

    if cols.size <= 1:
        return cols.copy()

    # only compare model behavior on the selected row scope
    X_sub = X_rows[:, cols].astype(np.float64, copy=False)
    F_sub = F[cols].astype(np.float64, copy=False)
    n_models = cols.size

    # robust absolute-correlation matrix
    std = X_sub.std(axis=0)
    nonconst = std > 0

    abs_corr = np.zeros((n_models, n_models), dtype=np.float64)

    if np.any(nonconst):
        Xn = X_sub[:, nonconst]
        corr_nc = np.corrcoef(Xn, rowvar=False)
        corr_nc = np.nan_to_num(corr_nc, nan=0.0)
        abs_corr[np.ix_(nonconst, nonconst)] = np.abs(corr_nc)

    # handle constant columns explicitly
    const_idx = np.where(~nonconst)[0]
    if const_idx.size:
        const_vals = X_sub[0, const_idx]
        for a in range(const_idx.size):
            ia = const_idx[a]
            abs_corr[ia, ia] = 1.0
            for b in range(a + 1, const_idx.size):
                ib = const_idx[b]
                if np.all(X_sub[:, ia] == X_sub[:, ib]):
                    abs_corr[ia, ib] = 1.0
                    abs_corr[ib, ia] = 1.0

    np.fill_diagonal(abs_corr, 1.0)

    # distance = 1 - abs(corr)
    # remove the weaker member of each too-close pair
    alive = np.ones(n_models, dtype=bool)

    # better models first: higher F, then lower c index
    order = np.lexsort((cols, -F_sub))

    for i in order:
        if not alive[i]:
            continue

        for j in range(n_models):
            if j == i or not alive[j]:
                continue

            dist = 1.0 - abs_corr[i, j]
            if dist <= threshold:
                if F_sub[i] > F_sub[j]:
                    alive[j] = False
                elif F_sub[i] < F_sub[j]:
                    alive[i] = False
                    break
                else:
                    if cols[i] < cols[j]:
                        alive[j] = False
                    else:
                        alive[i] = False
                        break

    selected = cols[alive]
    selected.sort()

    survivors = _I.family_tree_indices(population._instructions, selected, include_terminals=True)
    g_map = flush_population(population, survivors)

    return selected


def purge_indistinguishable_old(population, evaluation, threshold=0.02):
    """
    Return the surviving model indices c from population._G_idx such that:
      1) evaluation["F"][c] > 0
      2) models that are too similar are purged using abs-correlation distance

    Similarity rule:
      distance(i, j) = 1 - abs(corr(i, j))
      if distance(i, j) <= threshold, they are considered indistinguishable

    Tie break:
      - remove lower F
      - if F ties, remove higher c index

    Parameters
    ----------
    population : object
        Must contain:
          population._X_inst : 2d numpy array, shape (n_samples, n_models_total)
          population._G_idx  : iterable of model column indices c to consider
    evaluation : dict
        Must contain:
          evaluation["F"] : 1d score array indexed by c
    threshold : float, default=0.02
        Maximum allowed abs-correlation distance for two models to be treated
        as indistinguishable. Example:
            threshold=0.02  -> remove one of any pair with |corr| >= 0.98

    Returns
    -------
    selected : np.ndarray
        1d array of surviving original column indices c
    """
    X = np.asarray(population._X_inst)
    G_idx = np.asarray(population._G_idx, dtype=int)
    F = np.asarray(evaluation["F"])

    if X.ndim != 2:
        raise ValueError("population._X_inst must be a 2d array")
    if F.ndim != 1:
        raise ValueError('evaluation["F"] must be a 1d array')
    if np.any(G_idx < 0) or np.any(G_idx >= X.shape[1]):
        raise ValueError("population._G_idx contains invalid column indices")
    if np.any(G_idx >= F.shape[0]):
        raise ValueError('evaluation["F"] is too short for indices in population._G_idx')

    # only consider models in _G_idx with positive F
    pos_mask = F[G_idx] > 0
    cols = G_idx[pos_mask]

    if cols.size <= 1:
        return cols.copy()

    X_sub = X[:, cols].astype(np.float64, copy=False)
    F_sub = F[cols].astype(np.float64, copy=False)
    n_models = cols.size

    # robust absolute-correlation matrix
    # standard correlation for non-constant columns
    std = X_sub.std(axis=0)
    nonconst = std > 0

    abs_corr = np.zeros((n_models, n_models), dtype=np.float64)

    if np.any(nonconst):
        Xn = X_sub[:, nonconst]
        corr_nc = np.corrcoef(Xn, rowvar=False)
        corr_nc = np.nan_to_num(corr_nc, nan=0.0)
        abs_corr[np.ix_(nonconst, nonconst)] = np.abs(corr_nc)

    # handle constant columns explicitly
    const_idx = np.where(~nonconst)[0]
    if const_idx.size:
        # same constant value -> perfectly correlated for our duplicate-purge purpose
        const_vals = X_sub[0, const_idx]
        for a in range(const_idx.size):
            ia = const_idx[a]
            abs_corr[ia, ia] = 1.0
            for b in range(a + 1, const_idx.size):
                ib = const_idx[b]
                if np.all(X_sub[:, ia] == X_sub[:, ib]):
                    abs_corr[ia, ib] = 1.0
                    abs_corr[ib, ia] = 1.0

    np.fill_diagonal(abs_corr, 1.0)

    # distance = 1 - abs(corr)
    # remove the weaker member of each too-close pair
    alive = np.ones(n_models, dtype=bool)

    # better models first: higher F, then lower c index
    order = np.lexsort((cols, -F_sub))

    for i in order:
        if not alive[i]:
            continue

        for j in range(n_models):
            if j == i or not alive[j]:
                continue

            dist = 1.0 - abs_corr[i, j]
            if dist <= threshold:
                # decide which one should survive
                if F_sub[i] > F_sub[j]:
                    alive[j] = False
                elif F_sub[i] < F_sub[j]:
                    alive[i] = False
                    break
                else:
                    # tie: keep lower original c index
                    if cols[i] < cols[j]:
                        alive[j] = False
                    else:
                        alive[i] = False
                        break

    selected = cols[alive]
    selected.sort()

    survivors = _I.family_tree_indices(population._instructions, selected, include_terminals=True)
    g_map = flush_population(population, survivors)

    return selected

        
def purge_nonrecreative():
    return