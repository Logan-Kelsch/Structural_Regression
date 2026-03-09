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
            _I.generate_instructions(population, grammar, n=int(
                population._max_size - population._L_idx.size - population._E_idx.size
            ), chunk_size=population._chunk_size)

            #making a thingy to be able to return
            reproduction_stats = {
                "selected_parents":selected_parents,
                "survived_genes":survived_genes,
                "gene_mapping":g_map
            }

            return reproduction_stats

        case _:
            raise NotImplementedError(f"Selector method of ({selector._method}) not implemented into reproduce function")
        
