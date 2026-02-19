import numpy as np

#the first functionality we are going to put in this file is the most
#general so that we can begin constructing more code for instantiation.
#This function is for randomly generating instructions for instantiation


#we first must assert that we are going to normalize data over a 
# window of time, so that we can have all columns of data be allowed 
# to communicate with each other

#then we must define which columns we are allowed to select from

#when it comes to instantiation,
# we will have an x_inst to reference for generation of instructions
# then with the generated set of instructions, we will instantiate the data
# and append it to the current x_inst
# Therefore chunking will contain split points so that data can be 
# instantiated and appended before a new set of instructions is generated

#here we will make the function for generation of instructions


class Grammar:
    '''
    Temporary class that will contain grammar structures and rules
    '''
    def __init__(
        self,
        type    :   str,
        max_delta_lookback  :   int =   48,
        spec_gram_args  :   dict    =   None
    ):
        self._type = type
        self._mdl = max_delta_lookback
        

class Population:
    '''
    All instructions containing all zeros suggest locations are terminal states
    '''
    def __init__(
        self, X_inst: np.ndarray,
        terminal_idx: np.ndarray | list,
        excluded_idx: np.ndarray | list,
        max_size    : int   =   20000
    ):
        self._X_inst = X_inst
        self._T_idx = np.asarray(terminal_idx, dtype=np.int64)
        self._E_idx = np.asarray(excluded_idx, dtype=np.int64)
        self._max_size = max_size

        #allocating entire space of possible instructions
        self._instructions = np.zeros((max_size, 11), dtype=np.float32)

        #writing in gene indices for our terminal or excluded columns
        self._instructions[:self._X_inst.shape[1], 0] = np.arange(self._X_inst.shape[1], dtype=np.uint16)

        # throw error if any intersection
        overlap = np.intersect1d(self._T_idx, self._E_idx, assume_unique=False)
        if overlap.size:
            raise ValueError(f"terminal_idx and excluded_idx overlap at indices: {overlap.tolist()}")

        # union (combined set to exclude from genes)
        TE_idx = np.union1d(self._T_idx, self._E_idx)  # sorted unique

        # mask over COLUMNS (features), not rows
        mask = np.ones(self._X_inst.shape[1], dtype=np.bool_)
        mask[TE_idx] = False
        self._G_idx = np.flatnonzero(mask)

        # legal indices = terminals + genes (already disjoint because G excludes T and E)
        self._L_idx = np.union1d(self._T_idx, self._G_idx).astype(np.int64, copy=False)



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
	# Order: x, a, d, dd, k  ->  5, 6, 7, 8, 9
	m = {'x': 5, 'a': 6, 'd': 7, 'dd': 8, 'k': 9}
	return [m[v] for v in vals if v in m]


def VLOC_to_FLAG(vlocs):
	"""
	Takes a list like [6,7,8,9] (any subset, any order) and returns a single
	int bit-flag value, preserving the 5..9 location values by mapping:
	23->bit3, 4->bit4, 5->bit5, 6->bit6.
	"""
	flags = 0
	for loc in vlocs:
		if 5 <= loc <= 9:
			flags |= (1 << loc)
	return flags


def fill_sensor_UNIFORM(inst_inst, flags_u32, legal_idx, rng=None, inplace=True):
    """
    Fill *every FLAGGED location* in columns 5..9 with a random selection from `legal_idx`.

    - flags_u32 uses bits 5..9 to indicate whether that column is "flagged" (True => fill)
    - legal_idx: 1D array-like of allowed sensor indices to write into inst_inst
    - inst_inst is float32, so ints will be stored as float32 (fine for IDs)

    Returns the (possibly modified) inst_inst.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = inst_inst if inplace else inst_inst.copy()

    flags = np.asarray(flags_u32, dtype=np.uint32).reshape(-1)

    legal_idx = np.asarray(legal_idx, dtype=np.int32).reshape(-1)
    if legal_idx.size == 0:
        raise ValueError("legal_idx must contain at least one allowed index.")

    bits = np.arange(5, 10, dtype=np.uint32)  # [5,6,7,8,9]
    flagged_mask = ((flags[:, None] >> bits[None, :]) & np.uint32(1)).astype(bool)

    # sample FROM legal_idx for every cell in the (N,5) block, then only write where flagged
    pick = rng.integers(0, legal_idx.size, size=flagged_mask.shape, dtype=np.int32)
    r = legal_idx[pick].astype(np.float32, copy=False)  # shape (N,5)

    #per row base (instinst[:, 0]) broadcast to (N, 5)
    base = out[:, 0].astype(np.float32, copy=False)[:, None]

    #final vals written: base - generated
    vals = r - base #shape is (N, 5)

    block = out[:, 5:10]
    block[flagged_mask] = vals[flagged_mask].astype(out.dtype, copy=False)

    return out



def FUNC_to_USED_FLAGS(func_ids):
    """
    Map each function-id -> required variable locations (cols 3..6) -> uint32 bit-flag.
    Uses the user-provided helpers: F_AS, V_to_LOC, VLOC_to_FLAG.

    Parameters
    ----------
    func_ids : array-like (int)
        Function IDs per row.

    Returns
    -------
    flags_u32 : np.ndarray dtype=np.uint32
        Same shape as func_ids. Bits 3..6 indicate presence of cols 3..6.
    """
    fids = np.asarray(func_ids, dtype=np.int32)

    # Build a lookup table for ids 0..max_id seen in func_ids (fast, tiny)
    max_id = int(fids.max()) if fids.size else 0
    max_id = max(max_id, 0)

    table = np.zeros(max_id + 1, dtype=np.uint32)
    for i in range(max_id + 1):
        vlocs = V_to_LOC(F_AS(i))          # e.g. ['a','d'] -> [3,4]
        table[i] = np.uint32(VLOC_to_FLAG(vlocs))  # [3,4] -> (1<<3)|(1<<4)

    # Default 0 for out-of-range / negative ids
    out = np.zeros(fids.shape, dtype=np.uint32)
    in_range = (fids >= 0) & (fids <= max_id)
    out[in_range] = table[fids[in_range]]
    return out



def FUNC_to_nonx_FLAGS(func_ids):
    """
    Same as FUNC_to_USED_FLAGS, but ALWAYS clears/ignores location 5
    (bit 5 is never set, even if V_to_LOC returns 5).

    Parameters
    ----------
    func_ids : array-like (int)
        Function IDs per row.

    Returns
    -------
    flags_u32 : np.ndarray dtype=np.uint32
        Same shape as func_ids. Bits 3..6 indicate presence of cols 3..6,
        except bit 5 is always 0.
    """
    fids = np.asarray(func_ids, dtype=np.int32)

    max_id = int(fids.max()) if fids.size else 0
    max_id = max(max_id, 0)

    table = np.zeros(max_id + 1, dtype=np.uint32)
    for i in range(max_id + 1):
        vlocs = V_to_LOC(F_AS(i))              # e.g. ['a','d'] -> [3,4,5]
        # remove 5 if present (ignore dd)
        if vlocs:
            vlocs = [loc for loc in vlocs if loc != 5]
        table[i] = np.uint32(VLOC_to_FLAG(vlocs))

    out = np.zeros(fids.shape, dtype=np.uint32)
    in_range = (fids >= 0) & (fids <= max_id)
    out[in_range] = table[fids[in_range]]

    # extra safety: force-clear bit 5 even if upstream helpers change
    out &= ~np.uint32(1 << 5)

    return out



def FLAGS_to_SENSOR_FLAGS(used_flags, nonx_flags) -> np.ndarray:
    """
    Compute USED_FLAGS minus nonx_FLAGS, but ONLY for bit positions 5..9 inclusive.
    All other bits are forced to 0 in the output.

    Output = (used & ~nonx) & mask_bits_5_to_9
    """
    used_u32 = np.asarray(used_flags).astype(np.uint32, copy=False)
    nonx_u32 = np.asarray(nonx_flags).astype(np.uint32, copy=False)

    mask_5_to_9 = np.uint32(0)
    for b in range(5, 10):  # 5..9
        mask_5_to_9 |= (np.uint32(1) << np.uint32(b))

    return (used_u32 & ~nonx_u32) & mask_5_to_9


def fill_const_ITS(arr: np.ndarray,
                   rows,
                   col: int,
                   rate: float = 1.0,
                   rng: np.random.Generator | None = None,
                   inplace: bool = True) -> np.ndarray:
    """
    Column-specialized ITS filler (same distribution as your original):
      draw u ~ Uniform(0,1], set x = -log(u)/rate (>= 0),
      write into arr[rows, col].

    rows can be either:
      - 1D boolean mask of length arr.shape[0]
      - 1D integer row indices
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Decide how many samples we actually need
    if rows.dtype == np.bool_:
        n = int(rows.sum())
        if n == 0:
            return out
        write_rows = rows
    else:
        write_rows = rows.astype(np.int64, copy=False)
        n = write_rows.size
        if n == 0:
            return out

    # U in (0,1], avoid log(0)
    u = rng.random(n, dtype=np.float64)
    u = np.clip(u, np.finfo(np.float64).tiny, 1.0)

    # Equivalent to your original neg_samples = (1/rate)*log(u) then flip sign
    out[write_rows, col] = (-np.log(u) / rate).astype(out.dtype, copy=False)
    return out


def fill_const_STEIS(n: int,
                     rows,
                     arr: np.ndarray,
                     col: int,
                     base_max: float = 2.0,
                     base_rate: float = 2.0,
                     rng: np.random.Generator | None = None,
                     inplace: bool = True) -> np.ndarray:
    """
    Column-specialized STEIS filler.
    Writes sampled ints in {1,...,n} into arr[rows, col].

    rows can be either:
      - 1D integer row indices
      - 1D boolean mask of length arr.shape[0]
    """
    if not (isinstance(n, (int, np.integer)) and n >= 1):
        raise ValueError("n must be an integer >= 1")
    if base_max <= 0:
        raise ValueError("base_max must be > 0")
    if base_rate <= 0:
        raise ValueError("base_rate must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Determine how many samples we actually need to generate
    if rows.dtype == np.bool_:
        m = int(rows.sum())
        if m == 0:
            return out
        write_rows = rows  # boolean mask used directly in assignment
    else:
        write_rows = rows.astype(np.int64, copy=False)
        m = write_rows.size
        if m == 0:
            return out

    # U in (0,1) (avoid endpoints)
    u = rng.random(m, dtype=np.float64)
    u = np.clip(u, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)

    # Truncated exponential on (0, base_max]
    z = 1.0 - np.exp(-base_rate * base_max)
    y = -(1.0 / base_rate) * np.log(1.0 - u * z)

    # Stretch to (0, n]
    x = y * (n / base_max)

    # Round up to int in [1, n]
    k = np.ceil(x).astype(np.int64, copy=False)
    k = np.clip(k, 1, n)

    out[write_rows, col] = k.astype(out.dtype, copy=False)
    return out


def fill_const_UA0(arr: np.ndarray,
                   rows,
                   col: int,
                   rng: np.random.Generator | None = None,
                   inplace: bool = True) -> np.ndarray:
    """
    Fill arr[rows, col] with Uniform(-1, 1) float32 samples.

    rows can be either:
      - 1D integer row indices (e.g., np.array([1,3,7]))
      - 1D boolean mask of length arr.shape[0]
    """
    if rng is None:
        rng = np.random.default_rng()

    out = arr if inplace else np.array(arr, copy=True)

    rows = np.asarray(rows)

    # Case A: boolean mask
    if rows.dtype == np.bool_:
        n = int(rows.sum())
        if n == 0:
            return out
        out[rows, col] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32, copy=False)
        return out

    # Case B: integer indices
    rows = rows.astype(np.int64, copy=False)
    n = rows.size
    if n == 0:
        return out

    out[rows, col] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32, copy=False)
    return out


def generate_instructions(
    pop_prior   :   Population,
    grm_prior   :   Grammar,
    n           :   int,
    chunk_size  :   int =   0,
    seed        :   int =   None,
    verbose     :   bool=   False
):
    '''
    early dev notes
    - x_inst coming into this contains the terminal states
    - pop_prior is the population we are referencing going into generation
    - grm_prior is the grammar we are referencing going into generation
    - n         is the number of genes that are to be generated
    - chunk_gen is the size of each chunk that will be generated. 
                 allows for grammar updating within generation, will be slower.

    INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]


    '''

    #we will have the entirety of the population space pre-allocated 
    #therefore we need some logic checks for if we are within these memory bounds



    # INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
    gen_size = n
    if(chunk_size == 0):
        chunk_size = gen_size
    
    while(gen_size > 0):

        #quick fix for if the final chunk generated is ill-shaped
        #really only going to happen as I am making vizualizations for 
        #instantiation operation list efficiency vizualizations
        if(gen_size < chunk_size):
            chunk_size = gen_size

        
        #match case different grammars
        match(grm_prior._type):
            #this case will not consider any form of grammar
            case 'None':
                #first thing will be to allocate some memory for instructions
                #instruction format will be along the lines of

                inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)

                #now that we have allocated the memory for instructions
                #we can begin generating instructions
                # INSTRUCTIONS FORMAT [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
                #flag meaning- contains whether or not a, d, dd, k are
                #               gene location data or a constant data

                #we will need to put the population indices in the first column
                #go get the total length of instructions thus far
                start_idx = pop_prior._L_idx.max()
                inst_inst[:, 0] = np.arange(start_idx+1, start_idx+1+chunk_size, dtype=np.uint16)
                
                #for this NO GRAMMAR generation we will randomly select each T function
                inst_inst[:, 1] = np.random.randint(1, 22, size=inst_inst.shape[0], dtype=np.uint16)

                #ultimately we want to have variable_is_sensor probabilities embedded in grammar
                #so that this can be used to generate at random for all locations at once
                #since we also need to generate constants all at once as well.
                #so we will have some sort of mask functionality for each instance being:
                #   NEEDS MASK: EACH type of constant generation
                #   NEEDS MASK: EACH type of sensor location generation (maybe for each T func type) 
                func_ids = inst_inst[:, 1].astype(np.int32, copy=False)

                #used flags
                flags_u32 = FUNC_to_USED_FLAGS(func_ids)
                inst_inst[:, 2] = flags_u32#.astype(np.float32)

                #const flags
                flags_u32 = FUNC_to_nonx_FLAGS(func_ids)
                inst_inst[:, 3] = flags_u32#.astype(np.float32)

                #sensor flags
                flags_u32 = FLAGS_to_SENSOR_FLAGS(inst_inst[:, 2], inst_inst[:, 3])
                inst_inst[:, 4] = flags_u32#.astype(np.float32)

                rng = np.random.default_rng(seed)

                #current order: a, d, dd, k -> 6, 7, 8, 9
                v_cols = {6:'UA0',7:'STEIS',8:'STEIS',9:'ITS'}

                const_flags = inst_inst[:, 3].astype(np.uint32, copy=False)
                for c, type in v_cols.items():
                    const_mask = (const_flags & (1 << c)) != 0
                    match(type):
                        case 'UA0':
                            inst_inst = fill_const_UA0(inst_inst, const_mask, c, rng)
                        case 'STEIS':
                            inst_inst = fill_const_STEIS(grm_prior._mdl, const_mask, inst_inst, c, 2.0, 2.0, rng)
                        case 'ITS':
                            inst_inst = fill_const_ITS(inst_inst, const_mask, c, 1.0, rng)

                #now at this point we need to generate x_inst indices for vars that are sensors
                #given we are working with no grammar prior first, we can structure this code as   
                # it will be in our first grammar structure (2d transition probability matrix)
                #   this means, we select indices first THEN apply our null grammar prior.

                # collect terminal indices from pop_prior._T_idx, 
                # collect gene indices from pop_prior._G_idx
                #   NOW WE HAVE TWO PATHS FOR NULL GRAMMAR PRIOR
                # we can declare some static probability for selecting a terminal index and select randomint
                # OR, we can have an exponentially decaying selection probability across length of population by age (index value)
                #for now we will select with uniform probability across length of population

                #candidate usage of template function
                inst_inst = fill_sensor_UNIFORM(inst_inst, inst_inst[:, 4], legal_idx=pop_prior._L_idx)

                #then we need to actually instantiate the genes
                #       should be as simple as building a routing function for funcs in transform
                #       along with design of long term x_inst data structure

                #then after this is instantiated, we need to have some kind of universal
                #used indices of x_inst variable so that we can run these instantiations IN PLACE
                #this will mean entire size of X_inst is allocated before even the first generation of genes

                #return inst_inst
            case _:
                raise ValueError('Cannot interpret grammar prior in generate_genes. Illegal type.')
            
        #at this point, we are going to add our instructions into the population prior
        #this starts with adding new gene instructions into gene and legal indices variables
        pop_prior._G_idx = np.union1d(pop_prior._G_idx, inst_inst[:, 0])
        pop_prior._L_idx = np.union1d(pop_prior._L_idx, pop_prior._G_idx)
        #print(f'Lidx: {pop_prior._L_idx}')
        #print(f'Gidx: {pop_prior._G_idx}')
        #print(f'instinst0: {inst_inst[:, 0]}')
                
        
        #now we need to find where we will place newly generated instructions in the instruction array
        col0 = pop_prior._instructions[:, 0].astype(np.int32, copy=False)
        empty = (col0 == 0)
        empty[0] = False
        break_idx = (empty == 1).argmax()
        #print(break_idx, empty[:10])

        #then we will actually bring in the new instantiation instructions into correct memory locations
        #this should place the instructions correctly into the population prior that was provided
        pop_prior._instructions[ break_idx : break_idx+chunk_size , : ] = inst_inst
            
        gen_size -= chunk_size
        if(gen_size>0 and verbose):
            print(f'{gen_size} Generations Remaining.')



from collections import deque

# bits 5..9 correspond to sensor slots x,a,d,dd,k stored in cols 5..9
_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # [5,6,7,8,9]
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # [5,6,7,8,9]

import numpy as np

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor flag bit positions
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns


def apply_index_map_axis0(arr: np.ndarray, old_to_new: np.ndarray, *, fill_value=0):
    """
    Apply an old->new index map to any array whose axis 0 matches the gene axis.

    - old_to_new: length G, values in [0..G-1] or -1 for removed
    - returns: new array same shape as arr, with rows moved, removed rows filled with fill_value
    """
    G = old_to_new.shape[0]
    if arr.shape[0] != G:
        raise ValueError(f"arr.shape[0]={arr.shape[0]} must match mapping length G={G}")

    out = np.full_like(arr, fill_value)
    keep_old = np.flatnonzero(old_to_new >= 0)
    out[old_to_new[keep_old]] = arr[keep_old]
    return out

import numpy as np

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor-flag bit positions
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns


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


def family_tree_indices(instructions: np.ndarray, gene_idxs, *, include_self: bool = True) -> np.ndarray:
    """
    Collect the full ancestor set ("family tree" of parent nodes) for one or many genes.

    instructions: shape (G, 11), rows are genes and columns are:
      [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
       col:  0       1        2          3           4         5  6  7  8   9

    gene_idxs: int OR list/array of ints (gene row indices).
      NOTE: pop_idx is assumed to be the same as the gene row index.

    Parent rule:
      For each node g, decode SENSOR_FLAGS bits 5..9.
      For each active slot among cols 5..9, read displacement value (negative int),
      compute parent index: parent = pop_idx + displacement.
      Recurse until no new parents.

    Returns: sorted unique np.ndarray[int64] of all ancestor gene indices
             (and optionally the starting genes).
    """
    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be 2D with 11 columns (need at least cols 0..9).")

    G = instructions.shape[0]

    # normalize input to 1D int64 array
    if np.isscalar(gene_idxs):
        seeds = np.array([int(gene_idxs)], dtype=np.int64)
    else:
        seeds = np.asarray(gene_idxs, dtype=np.int64).reshape(-1)

    # keep valid seeds
    seeds = seeds[(seeds >= 0) & (seeds < G)]
    if seeds.size == 0:
        return np.empty(0, dtype=np.int64)

    # pre-cast sensor flags once (may be stored as float32)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    visited = np.zeros(G, dtype=np.bool_)
    q = deque()

    for s in seeds.tolist():
        if include_self and not visited[s]:
            visited[s] = True
        q.append(s)

    while q:
        g = q.pop()

        pop_idx = g  # pop_idx == gene row index (per your note)
        sf = sensor_flags[g]

        # mask over the 5 sensor slots (cols 5..9) based on bits 5..9
        slot_mask = ((sf >> _BITS_5_9) & np.uint32(1)).astype(bool)
        if not np.any(slot_mask):
            continue

        # grab displacement values from cols 5..9 where slot_mask True
        disp = instructions[g, _COLS_5_9[slot_mask]].astype(np.int64, copy=False)
        disp = disp[disp < 0]  # only negative displacements per your spec
        if disp.size == 0:
            continue

        parents = pop_idx + disp  # disp negative => parent < pop_idx

        for p in parents.tolist():
            if 0 <= p < G and not visited[p]:
                visited[p] = True
                q.append(p)

    return np.flatnonzero(visited).astype(np.int64, copy=False)


from collections import deque, defaultdict

_BITS_5_9 = np.arange(5, 10, dtype=np.uint32)   # sensor flag bits
_COLS_5_9 = np.arange(5, 10, dtype=np.int64)    # x,a,d,dd,k columns



def build_operation_list(instructions: np.ndarray,
                         *,
                         func_col: int = 1,
                         sensor_flag_col: int = 4,
                         displacement_cols: np.ndarray = _COLS_5_9,
                         sensor_bits: np.ndarray = _BITS_5_9,
                         only_negative_parents: bool = True,
                         prefer_last: bool = True,
                         last_bonus: float = 1.20,
                         max_ops: int | None = None):
    """
    Build an execution plan that is as parallel as possible subject to parent dependencies.

    instructions: shape (G, 11), columns:
      [pop_idx, func_id, USED_FLAGS, CONST_FLAGS, SENSOR_FLAGS, x, a, d, dd, k]
    Assumptions:
      - pop_idx == row index (gene index)
      - parent references are stored as (usually negative) displacements in cols 5..9
      - SENSOR_FLAGS bits 5..9 indicate which of cols 5..9 are active sensor slots
      - For each active slot, if value is a negative int displacement d, parent = child + d

    Returns
    -------
    op_list: list of (func_id:int, gene_indices:np.ndarray[int64])
        Each item is one parallel "kernel" call: run func_id for all indices in gene_indices.
        Covers every gene exactly once.

    Notes
    -----
    - This is a topological batching scheduler with a cheap heuristic to reduce func switches.
    - Solve time is roughly O(G + E) where E is #dependency edges discovered.
    """
    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be 2D with at least 10 columns (expected 11).")

    G = instructions.shape[0]
    if G == 0:
        return []

    # func ids (int)
    func_ids = instructions[:, func_col].astype(np.int32, copy=False)

    # sensor flags (uint32)
    sensor_flags = instructions[:, sensor_flag_col].astype(np.uint32, copy=False)

    # ---- Build dependency graph: parent -> child ----
    # We'll construct:
    #   children[parent] = list of children
    #   indegree[child] = #parents
    children = [[] for _ in range(G)]
    indegree = np.zeros(G, dtype=np.int32)

    # Iterate genes; decode parents from SENSOR_FLAGS and displacement cols
    for child in range(G):
        sf = sensor_flags[child]
        slot_mask = ((sf >> sensor_bits) & np.uint32(1)).astype(bool)
        if not np.any(slot_mask):
            continue

        cols = displacement_cols[slot_mask]
        disp = instructions[child, cols].astype(np.int64, copy=False)

        if only_negative_parents:
            disp = disp[disp < 0]
        if disp.size == 0:
            continue

        # parent indices
        parents = child + disp  # disp negative => parent < child in typical case

        # validate and add edges
        for p in parents.tolist():
            if p < 0 or p >= G:
                raise ValueError(f"Invalid parent index computed for child={child}: parent={p}")
            children[p].append(child)
            indegree[child] += 1

    # ---- Ready buckets keyed by func_id ----
    # ready_by_func[f] = deque/list of ready gene indices
    ready_by_func = defaultdict(deque)

    # initial ready nodes
    ready_nodes = np.flatnonzero(indegree == 0).astype(np.int64, copy=False)
    for g in ready_nodes.tolist():
        ready_by_func[int(func_ids[g])].append(g)

    # Helper: choose next func to execute
    last_func = None
    op_list = []
    processed = 0

    def pick_next_func():
        nonlocal last_func

        if not ready_by_func:
            return None

        # Remove empty keys lazily
        empty_keys = [k for k, q in ready_by_func.items() if len(q) == 0]
        for k in empty_keys:
            del ready_by_func[k]
        if not ready_by_func:
            return None

        # Compute current best by ready count
        best_func, best_q = max(ready_by_func.items(), key=lambda kv: len(kv[1]))
        best_n = len(best_q)

        if prefer_last and last_func is not None and last_func in ready_by_func:
            last_n = len(ready_by_func[last_func])
            if last_n * last_bonus >= best_n:
                return last_func

        return best_func


    # ---- Kahn scheduler with batching by func ----
    while processed < G:
        f = pick_next_func()
        if f is None:
            # cycle or missing dependency resolution
            # (shouldn't happen if your system is acyclic)
            raise ValueError("No ready nodes but not all processed: dependency cycle or invalid graph.")

        q = ready_by_func[f]
        batch = np.fromiter(q, dtype=np.int64, count=len(q))  # take all ready for this func
        q.clear()

        op_list.append((int(f), batch))
        last_func = int(f)

        # mark processed; release children
        for g in batch.tolist():
            processed += 1
            for ch in children[g]:
                indegree[ch] -= 1
                if indegree[ch] == 0:
                    ready_by_func[int(func_ids[ch])].append(ch)

        if max_ops is not None and len(op_list) >= max_ops:
            break

    return op_list
