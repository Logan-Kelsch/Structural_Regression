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
        max_size    : int   =   20000,
        include_time: bool  =   False,
        structure   : str   =   'Continuous',
        market_only : bool  =   True,
        market_close: int   =   390
    ):
        self._X_inst = X_inst
        self._T_idx = np.asarray(terminal_idx, dtype=np.int64)
        self._E_idx = np.asarray(excluded_idx, dtype=np.int64)
        self._G_idx = np.asarray([], dtype=np.int64)
        
        self._max_size = max_size
        self._structure= structure
        self._time_terminals = include_time

        if(include_time):
            #NOTE tentative must is that epoch time column is first in excluded idx array
            #need to grab what index we are putting this in (will be terminal state)
            next_idx = self._T_idx.max()
            time_src_col = int(self._E_idx[0])

            next_idx = int(self._T_idx.max())

            # next_idx+1 -> minutes relative to market open
            self._X_inst[:, next_idx + 1] = tod_minutes_from_dstaware(
                self._X_inst[:, time_src_col],
                mode='market_open'
            )

            # next_idx+2 -> day of week, DST-aware
            self._X_inst[:, next_idx + 2] = dow_sun0_from_epoch(
                self._X_inst[:, time_src_col]
            )

            self._T_idx = np.concatenate((self._T_idx, [next_idx + 1, next_idx + 2]))
            

            # optional regular-market-hours-only filter
            if market_only and structure=='Intraday':
                tod_col = next_idx + 1
                tod_mkt = self._X_inst[:, tod_col]

                # keep [0, market_close_min) by default
                # if your timestamps are end-of-bar and you want 16:00 included,
                # change < market_close_min to <= market_close_min
                keep_mask = (tod_mkt >= 0) & (tod_mkt < market_close)

                if not np.any(keep_mask):
                    raise ValueError(
                        "market_only=True produced zero kept rows. "
                        "Check the market-open time column and timestamp convention."
                    )

                old_X = self._X_inst
                self._X_inst = old_X[keep_mask].copy()

                del old_X, keep_mask, tod_mkt


        #allocating entire space of possible instructions
        self._instructions = np.zeros((max_size, 11), dtype=np.float32)

        #writing in gene indices for our terminal or excluded columns
        self._instructions[self._T_idx, 0] = self._T_idx

        # throw error if any intersection
        overlap = np.intersect1d(self._T_idx, self._E_idx, assume_unique=False)
        if overlap.size:
            raise ValueError(f"terminal_idx and excluded_idx overlap at indices: {overlap.tolist()}")
        
        self._L_idx = np.asarray(self._T_idx, dtype=np.int64)



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
            case 'Null':
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

                #print('adding indices to gidx:', inst_inst[:, 0])
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
        #print(break_idx, col0[:])
        #print('adding them at:', break_idx)

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

import numpy as np

def _epoch_to_seconds_i64(t_epoch) -> np.ndarray:
    """
    Convert epoch-like input to int64 seconds.
    Auto-detects seconds vs milliseconds by magnitude.
    """
    t = np.asarray(t_epoch)

    if np.issubdtype(t.dtype, np.floating):
        t = np.rint(t)

    t = t.astype(np.int64, copy=False)

    if t.size == 0:
        return t

    # modern epoch seconds ~1e9, milliseconds ~1e12
    if np.max(np.abs(t)) >= 10**11:
        t = t // 1000

    return t


import pandas as pd

def tod_minutes_from_dstaware(t_epoch, mode: str = "market_open") -> np.ndarray:
    """
    DST-aware minutes feature using America/New_York.

    mode:
      - 'time_of_day'  -> minutes into local day [0, 1439]
      - 'market_open'  -> minutes relative to 9:30 local time
                          (before open are negative)
    """
    t = np.asarray(t_epoch)

    if np.issubdtype(t.dtype, np.floating):
        t = np.rint(t)

    t = t.astype(np.int64, copy=False)

    if t.size == 0:
        return np.empty(0, dtype=np.int32)

    # auto-detect sec vs ms
    unit = "ms" if np.max(np.abs(t)) >= 10**11 else "s"

    dt_ny = pd.to_datetime(t, unit=unit, utc=True).tz_convert("America/New_York")

    minutes_of_day = (dt_ny.hour * 60 + dt_ny.minute).astype(np.int32)

    if mode == "time_of_day":
        return minutes_of_day

    if mode == "market_open":
        return (minutes_of_day - 570).astype(np.int32)  # 9:30 = 570

    raise ValueError(f"mode must be 'market_open' or 'time_of_day', got {mode!r}")


def dow_sun0_from_epoch(t_epoch, tz_offset_seconds: int = 0) -> np.ndarray:
    """
    Convert Unix epoch (seconds or milliseconds) to day-of-week:
    0,1,2,3,4,5,6 = Sun,Mon,Tue,Wed,Thu,Fri,Sat
    """
    t_sec = _epoch_to_seconds_i64(t_epoch)
    days_since_epoch = (t_sec + np.int64(tz_offset_seconds)) // 86400
    return np.mod(days_since_epoch + 4, 7).astype(np.int8)


import numpy as np
import transform_ops as _OPS
import transform_jit as t_jit  # whatever module contains _MDN_core_heaps

def warmup_numba():
    # tiny shapes so compile happens fast
    m, n = 32, 8
    x = np.random.rand(m, n).astype(np.float32)
    wins = np.ones(n, dtype=np.int64)
    mc = np.ones(n, dtype=np.int64)

    out = np.empty((m, n), dtype=np.float32)

    # Call the exact JIT kernel directly if possible (best)
    t_jit._MDN_core_heaps(x, wins, mc, out)


import numpy as np
import time
import os

_VAR_TO_COL = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}
_VAR_TO_BIT = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}

_FUNC_ID_TO_NAME = {
    0: None,
    1: "t_MAX",  2: "t_MIN",  3: "t_AVG",  4: "t_NEG",  5: "t_DIF",
    6: "t_ADD",  7: "t_SQR",  8: "t_SIN",  9: "t_COS", 10: "t_ASN",
   11: "t_ACS", 12: "t_RNG", 13: "t_HKP", 14: "t_EMA", 15: "t_DOE",
   16: "t_MDN", 17: "t_ZSC", 18: "t_STD", 19: "t_SSN", 20: "t_AGR",
   21: "t_COR",
}


def instantiate_from_ops_chunked_debug(
    op_list,
    instructions: np.ndarray,   # (G,11)
    X_out: np.ndarray,          # (N,G) preallocated
    *,
    transform_ops,              # unified API: transform_ops.apply(...)
    chunk_B: int = 512,
    verbosity: int = 1,          # 0 silent, 1 per-op, 2 per-chunk, 3 very chatty
    check_nans: bool = True,
    check_dtypes: bool = True,
    fail_fast: bool = True,      # if False, continues and records failures
    max_failures: int = 3,
):
    """
    Debug-friendly batched instantiation with bounded scratch (<= (N,chunk_B) per buffer).

    Parent rule (your rule):
      If a var is SENSOR flagged, instruction holds negative displacement `disp` (int-like),
      parent index is: parent = gene_idx + disp

    Uses F_AS(func_id) to determine which variables are needed.

    Returns
    -------
    X_out : np.ndarray
        Instantiated in-place (and also returned).
    failures : list[dict]
        Debug records if fail_fast=False.
    """
    #warmup_numba()
    t0_all = time.perf_counter()

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be shape (G,11) (need cols 0..9).")
    if X_out.ndim != 2:
        raise ValueError("X_out must be 2D (N,G).")
    if X_out.shape[1] != instructions.shape[0]:
        raise ValueError(f"X_out.shape[1]={X_out.shape[1]} must equal G={instructions.shape[0]}")

    G = instructions.shape[0]
    N = X_out.shape[0]
    Bmax = int(chunk_B)

    # Cast flags once (handles float-stored flags safely)
    # NOTE: If instructions[:,3] or [:,4] were float32 with big values, casting to uint32 preserves bits
    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    # Scratch buffers (bounded)
    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    # one extra buffer used if you later want to gather something else;
    # keeping it here for debugging flexibility
    s_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    failures = []

    def log(level, msg):
        if verbosity >= level:
            print(msg)

    def est_bytes(shape, dtype):
        return int(np.prod(shape)) * np.dtype(dtype).itemsize

    # Rough memory footprint of scratch (not counting X_out)
    scratch_bytes = (
        est_bytes((N, Bmax), X_out.dtype) * 4  # x_buf, a_buf, y_buf, s_buf
    )
    log(1, f"[instantiate] N={N}, G={G}, chunk_B={Bmax}, dtype={X_out.dtype}, scratch≈{scratch_bytes/1e6:.1f} MB")

    # Optional: set OMP threads (sometimes crashes come from oversubscription)
    # You can uncomment to clamp threads during debugging:
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    def _fill_series_var(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray):
        """
        Fill buf[:, :B] with resolved SERIES for var_code ('x' or 'a').

        - CONST flag => broadcast scalar down N
        - SENSOR flag => displacement -> parent column gather using parent = idx + disp

        Returns dict of debug stats.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]

        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        # Fill with zeros first to avoid uninitialized garbage (important for debugging!)
        out_view = buf[:, :B]
        out_view.fill(0.0)

        stats = {
            "var": var_code,
            "B": B,
            "n_const": int(is_const.sum()),
            "n_sensor": int(is_sensor.sum()),
            "n_neither": int((~(is_const | is_sensor)).sum()),
            "parents_min": None,
            "parents_max": None,
            "disp_min": None,
            "disp_max": None,
        }

        # SENSOR gather
        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            stats["disp_min"] = int(disp.min()) if disp.size else None
            stats["disp_max"] = int(disp.max()) if disp.size else None

            # Your spec says sensor displacements are negative.
            # If this fails, it usually means the flags are wrong, or you accidentally wrote actual indices.
            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(
                    f"[{var_code}] expected negative displacements for sensor slots, found non-negative: {bad}"
                )

            parents = idx_chunk[is_sensor] + disp  # <<< YOUR RULE

            stats["parents_min"] = int(parents.min()) if parents.size else None
            stats["parents_max"] = int(parents.max()) if parents.size else None

            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(
                    f"[{var_code}] parent out of bounds. "
                    f"parents(min,max)=({stats['parents_min']},{stats['parents_max']}), "
                    f"showing first bad: {bad}"
                )

            # Fancy indexing => copy into our bounded buffer (expected)
            out_view[:, is_sensor] = X_out[:, parents]

        # CONST broadcast
        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        return stats

    def _get_param_vec(var_code: str, idx_chunk: np.ndarray, cast, default):
        """
        Get a per-gene parameter vector of length B (delta1, delta2, kappa).
        We *expect these to be constants*, not sensor time-series.

        If SENSOR-flagged, we raise — because rolling kernels typically need per-column constant params.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(
                f"[{var_code}] is sensor-flagged in this chunk, but expected a constant per-gene parameter. "
                f"This usually means you set SENSOR_FLAGS bits for d/dd/k by mistake."
            )

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)
        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        return out, {
            "var": var_code,
            "B": B,
            "n_const": int(is_const.sum()),
            "n_default": int((~is_const).sum()),
            "min": float(out.min()) if out.size else None,
            "max": float(out.max()) if out.size else None,
        }

    # --- Main loop over op-list ---
    processed_total = 0
    for op_i, (func_id, gene_idx) in enumerate(op_list):
        func_id = int(func_id)
        if func_id == 0:
            continue

        fname = _FUNC_ID_TO_NAME.get(func_id)
        if fname is None:
            raise ValueError(f"Unknown func_id={func_id}")

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        used_vars = F_AS(func_id)  # <-- your map
        # sanity: always must include 'x' for non-zero ops in your design
        if verbosity >= 3:
            log(3, f"[op {op_i}] func_id={func_id} ({fname}), genes={gene_idx.size}, used_vars={used_vars}")

        t0_op = time.perf_counter()

        # chunking
        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size
            t0_chunk = time.perf_counter()

            # For reproducibility in debugging, ensure sorted order (optional)
            # idx_chunk = np.sort(idx_chunk)

            # --- Build x series ---
            try:
                x_stats = _fill_series_var("x", idx_chunk, x_buf)
                if verbosity >= 2:
                    log(2, f"  [chunk {start//Bmax}] x_stats={x_stats}")
            except Exception as e:
                ctx = {
                    "where": "fill_x",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Build optional alpha series ---
            alpha_arg = None
            a_stats = None
            if "a" in used_vars:
                try:
                    a_stats = _fill_series_var("a", idx_chunk, a_buf)
                    alpha_arg = a_buf[:, :B]
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] a_stats={a_stats}")
                except Exception as e:
                    ctx = {
                        "where": "fill_a",
                        "func_id": func_id,
                        "fname": fname,
                        "chunk_start": start,
                        "B": B,
                        "idx_chunk_head": idx_chunk[:10].tolist(),
                        "error": repr(e),
                    }
                    failures.append(ctx)
                    log(1, f"[FAIL] {ctx}")
                    if fail_fast or len(failures) >= max_failures:
                        raise
                    else:
                        continue

            # --- Params (delta1, delta2, kappa) ---
            # only build if used by this func_id (per your F_AS)
            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None
            d_stats = dd_stats = k_stats = None

            try:
                if "d" in used_vars:
                    delta1_vec, d_stats = _get_param_vec("d", idx_chunk, cast=np.int64, default=1)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] d_stats={d_stats}")
                if "dd" in used_vars:
                    delta2_vec, dd_stats = _get_param_vec("dd", idx_chunk, cast=np.int64, default=1)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] dd_stats={dd_stats}")
                if "k" in used_vars:
                    kappa_vec, k_stats = _get_param_vec("k", idx_chunk, cast=np.float32, default=1.0)
                    if verbosity >= 2:
                        log(2, f"  [chunk {start//Bmax}] k_stats={k_stats}")
            except Exception as e:
                ctx = {
                    "where": "params",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Compute ---
            y_view = y_buf[:, :B]

            # Optionally prefill output buffer to detect partial writes
            if verbosity >= 3:
                y_view.fill(np.nan)

            try:
                transform_ops.apply(
                    func_id,
                    x_buf[:, :B],
                    alpha=alpha_arg,
                    delta1=delta1_vec,
                    delta2=delta2_vec,
                    kappa=kappa_vec,
                    out=y_view,
                    in_place=False,
                )
            except Exception as e:
                # Print maximal useful context to diagnose kernel crashes
                ctx = {
                    "where": "apply",
                    "func_id": func_id,
                    "fname": fname,
                    "chunk_start": start,
                    "B": B,
                    "idx_chunk_head": idx_chunk[:10].tolist(),
                    "used_vars": used_vars,
                    "x_stats": x_stats,
                    "a_stats": a_stats,
                    "d_stats": d_stats,
                    "dd_stats": dd_stats,
                    "k_stats": k_stats,
                    "error": repr(e),
                }
                failures.append(ctx)
                log(1, f"[FAIL] {ctx}")
                if fail_fast or len(failures) >= max_failures:
                    raise
                else:
                    continue

            # --- Post checks ---
            if check_nans:
                if not np.isfinite(y_view).all():
                    bad = np.flatnonzero(~np.isfinite(y_view))
                    ctx = {
                        "where": "postcheck_nonfinite",
                        "func_id": func_id,
                        "fname": fname,
                        "chunk_start": start,
                        "B": B,
                        "first_bad_flat_index": int(bad[0]) if bad.size else None,
                    }
                    failures.append(ctx)
                    log(1, f"[FAIL] {ctx}")
                    if fail_fast or len(failures) >= max_failures:
                        raise ValueError(f"Non-finite output detected: {ctx}")

            # Write back
            X_out[:, idx_chunk] = y_view
            processed_total += B

            t1_chunk = time.perf_counter()
            if verbosity >= 2:
                log(2, f"  [chunk {start//Bmax}] wrote B={B} in {(t1_chunk - t0_chunk)*1000:.1f} ms")

        t1_op = time.perf_counter()
        if verbosity >= 1:
            log(1, f"[op {op_i}] func_id={func_id:2d} ({fname}) genes={gene_idx.size} time={(t1_op - t0_op):.3f}s")

    t1_all = time.perf_counter()
    log(1, f"[instantiate] done processed≈{processed_total} gene-cols, total time={(t1_all - t0_all):.3f}s")

    return X_out, failures


import numpy as np
import time
from collections import defaultdict

_VAR_TO_COL = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}
_VAR_TO_BIT = {"x": 5, "a": 6, "d": 7, "dd": 8, "k": 9}

_FUNC_ID_TO_NAME = {
    0: "NOP",
    1: "t_MAX",  2: "t_MIN",  3: "t_AVG",  4: "t_NEG",  5: "t_DIF",
    6: "t_ADD",  7: "t_SQR",  8: "t_SIN",  9: "t_COS", 10: "t_ASN",
   11: "t_ACS", 12: "t_RNG", 13: "t_HKP", 14: "t_EMA", 15: "t_DOE",
   16: "t_MDN", 17: "t_ZSC", 18: "t_STD", 19: "t_SSN", 20: "t_AGR",
   21: "t_COR",
}

def instantiate_from_ops_chunked_partialbuild_class(
    op_list,
    instructions: np.ndarray,     # (G,11)
    X_out: np.ndarray,            # (N,G)
    population: Population,
    *,
    transform_ops,                # unified apply(func_id,...)
    chunk_B: int = 16,
    verbosity: int = 1,
    sanitize_final: bool = True,
):
    """
    Batched instantiation with aggressive NaN/Inf -> 0 sanitization and replacement tracking.

    Parent rule (YOUR RULE):
      For sensor-series slots, instruction stores negative displacement 'disp' (int-like),
      parent = gene_idx + disp.

    Verbosity:
      0: silent
      1: per-op timing
      2: per-op timing + replacement tracking output (standard)
      3: adds per-chunk timing + extra stats
      4: very chatty (debug details)
    """
    t0_all = time.perf_counter()

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("instructions must be shape (G,11) (need cols 0..9).")
    if X_out.ndim != 2:
        raise ValueError("X_out must be 2D (N,G).")

    G = instructions.shape[0]
    N = X_out.shape[0]
    if X_out.shape[1] != G:
        raise ValueError(f"X_out.shape[1]={X_out.shape[1]} must equal G={G}")

    Bmax = int(chunk_B)

    # Cast flags once (allows flags stored as float32 in instructions)
    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    # Scratch buffers (bounded)
    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    # Replacement tracking
    # counts keyed by ("stage", func_id) where stage in {"input_x","input_a","output","final"}
    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    def _sanitize_inplace(arr: np.ndarray, key):
        """
        Replace NaN/Inf -> 0 in-place.
        Update counters for this key.
        """
        # Count first to avoid double-counting after modification
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        # inf includes +inf/-inf; use isfinite to count
        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            # non-finite includes NaNs too; subtract NaNs to get inf count
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        # replace
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] with resolved series for var_code ('x' or 'a'):
          - CONST flag => broadcast scalar down N
          - SENSOR flag => gather parent series using parent = idx + disp
          - else => 0.0
        Then sanitize NaN/Inf -> 0 (in case parents already contain non-finite)
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        # sensor gather
        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            # spec: negative offsets
            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(f"{var_code}: expected negative displacement for sensor slots; got {bad}")

            parents = idx_chunk[is_sensor] + disp  # <<< YOUR RULE
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(f"{var_code}: parent out of bounds (first bad: {bad})")

            out_view[:, is_sensor] = X_out[:, parents]

        # const broadcast
        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        # sanitize and track
        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(4, f"    [{stage}] B={B} const={int(is_const.sum())} sensor={int(is_sensor.sum())}")

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        If sensor-flagged, this indicates a bug; raise.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)
        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            # force 1 -> 2 behavior for windows etc.
            out = np.maximum(out, cast(clamp_min))

        return out

    # Timing stats
    op_times = defaultdict(float)
    op_counts = defaultdict(int)

    # Main schedule loop
    for op_i, (func_id, gene_idx) in enumerate(op_list):
        fid = int(func_id)
        if fid == 0:
            continue

        name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
        used_vars = F_AS(fid)

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        t0_op = time.perf_counter()

        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size

            t0_chunk = time.perf_counter()

            # inputs
            x_view = _fill_series("x", idx_chunk, x_buf, fid)

            alpha_arg = None
            if "a" in used_vars:
                a_view = _fill_series("a", idx_chunk, a_buf, fid)
                alpha_arg = a_view  # (N,B)

            # params
            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None

            if "d" in used_vars:
                # window-like params: clamp to >=2 to avoid w=1 weirdness
                delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "dd" in used_vars:
                delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "k" in used_vars:
                kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

            # compute
            y_view = y_buf[:, :B]
            transform_ops.apply(
                fid,
                x_view,
                alpha=alpha_arg,
                delta1=delta1_vec,
                delta2=delta2_vec,
                kappa=kappa_vec,
                out=y_view,
                in_place=False,
            )

            # sanitize output and track
            _sanitize_inplace(y_view, ("output", fid))

            # write back
            X_out[:, idx_chunk] = y_view

            if verbosity >= 3:
                _log(3, f"  [chunk] {name} fid={fid} start={start} B={B} dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms")

        dt_op = time.perf_counter() - t0_op
        op_times[fid] += dt_op
        op_counts[fid] += int(gene_idx.size)

        if verbosity >= 1:
            _log(1, f"[op] {name:5s} fid={fid:2d} genes={gene_idx.size:6d} time={dt_op:.3f}s")

        # Standard replacement report in verbosity >= 2
        if verbosity >= 2:
            nan_in_x = rep_nan.get(("input_x", fid), 0)
            inf_in_x = rep_inf.get(("input_x", fid), 0)
            nan_in_a = rep_nan.get(("input_a", fid), 0)
            inf_in_a = rep_inf.get(("input_a", fid), 0)
            nan_out  = rep_nan.get(("output", fid), 0)
            inf_out  = rep_inf.get(("output", fid), 0)

            if (nan_in_x or inf_in_x or nan_in_a or inf_in_a or nan_out or inf_out):
                _log(2, f"    [repl] fid={fid:2d} {name}: "
                        f"x(nan={nan_in_x},inf={inf_in_x}) "
                        f"a(nan={nan_in_a},inf={inf_in_a}) "
                        f"out(nan={nan_out},inf={inf_out})")

    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    t1_all = time.perf_counter()
    stats = {
        "total_time_s": float(t1_all - t0_all),
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(2, f"[sanitize] final pass replaced: "
                    f"NaN={rep_nan.get(('final', -1), 0)}, Inf={rep_inf.get(('final', -1), 0)}")

    return X_out, stats

def instantiate_from_ops_chunked_contig(
    population,
    *,
    transform_ops,
    chunk_B: int = 16,
    verbosity: int = 1,
    sanitize_final: bool = True,
):
    """
    Batched instantiation with aggressive NaN/Inf -> 0 sanitization and replacement tracking.

    Uses:
      - population._instructions
      - population._X_inst
      - op_list built internally via build_operation_list(population._instructions)

    Parent rule:
      For sensor-series slots, instruction stores negative displacement 'disp' (int-like),
      parent = gene_idx + disp.

    Verbosity:
      0: silent
      1: per-op timing
      2: per-op timing + replacement tracking output
      3: adds per-chunk timing + extra stats
      4: very chatty
    """
    t0_all = time.perf_counter()

    if not hasattr(population, "_instructions"):
        raise AttributeError("population must have attribute '_instructions'")
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")

    instructions = population._instructions
    X_out = population._X_inst

    if not isinstance(instructions, np.ndarray):
        raise TypeError("population._instructions must be a numpy ndarray")
    if not isinstance(X_out, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must be shape (G,11) (need cols 0..9)")
    if X_out.ndim != 2:
        raise ValueError("population._X_inst must be 2D with shape (N, G)")

    G = instructions.shape[0]
    N = X_out.shape[0]

    if X_out.shape[1] != G:
        raise ValueError(f"population._X_inst.shape[1]={X_out.shape[1]} must equal G={G}")

    op_list = build_operation_list(instructions)

    Bmax = int(chunk_B)

    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    x_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    a_buf = np.empty((N, Bmax), dtype=X_out.dtype)
    y_buf = np.empty((N, Bmax), dtype=X_out.dtype)

    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    def _sanitize_inplace(arr: np.ndarray, key):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] with resolved series for var_code ('x' or 'a'):
          - CONST flag  => broadcast scalar down N
          - SENSOR flag => gather parent series using parent = idx + disp
          - else        => 0.0
        Then sanitize NaN/Inf -> 0.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(f"{var_code}: expected negative displacement for sensor slots; got {bad}")

            parents = idx_chunk[is_sensor] + disp
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(f"{var_code}: parent out of bounds (first bad: {bad})")

            out_view[:, is_sensor] = X_out[:, parents]

        if np.any(is_const):
            cvals = raw[is_const].astype(X_out.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(4, f"    [{stage}] B={B} const={int(is_const.sum())} sensor={int(is_sensor.sum())}")

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        If sensor-flagged, this indicates a bug; raise.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)

        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            out = np.maximum(out, cast(clamp_min))

        return out

    op_times = defaultdict(float)
    op_counts = defaultdict(int)

    for func_id, gene_idx in op_list:
        fid = int(func_id)
        if fid == 0:
            continue

        name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
        used_vars = F_AS(fid)

        gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
        if gene_idx.size == 0:
            continue

        t0_op = time.perf_counter()

        for start in range(0, gene_idx.size, Bmax):
            idx_chunk = gene_idx[start:start + Bmax]
            B = idx_chunk.size

            t0_chunk = time.perf_counter()

            x_view = _fill_series("x", idx_chunk, x_buf, fid)

            alpha_arg = None
            if "a" in used_vars:
                a_view = _fill_series("a", idx_chunk, a_buf, fid)
                alpha_arg = a_view

            delta1_vec = None
            delta2_vec = None
            kappa_vec  = None

            if "d" in used_vars:
                delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "dd" in used_vars:
                delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
            if "k" in used_vars:
                kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

            y_view = y_buf[:, :B]
            transform_ops.apply(
                fid,
                x_view,
                alpha=alpha_arg,
                delta1=delta1_vec,
                delta2=delta2_vec,
                kappa=kappa_vec,
                out=y_view,
                in_place=False,
            )

            _sanitize_inplace(y_view, ("output", fid))

            X_out[:, idx_chunk] = y_view

            if verbosity >= 3:
                _log(
                    3,
                    f"  [chunk] {name} fid={fid} start={start} B={B} "
                    f"dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms"
                )

        dt_op = time.perf_counter() - t0_op
        op_times[fid] += dt_op
        op_counts[fid] += int(gene_idx.size)

        if verbosity >= 1:
            _log(1, f"[op] {name:5s} fid={fid:2d} genes={gene_idx.size:6d} time={dt_op:.3f}s")

        if verbosity >= 2:
            nan_in_x = rep_nan.get(("input_x", fid), 0)
            inf_in_x = rep_inf.get(("input_x", fid), 0)
            nan_in_a = rep_nan.get(("input_a", fid), 0)
            inf_in_a = rep_inf.get(("input_a", fid), 0)
            nan_out  = rep_nan.get(("output", fid), 0)
            inf_out  = rep_inf.get(("output", fid), 0)

            if (nan_in_x or inf_in_x or nan_in_a or inf_in_a or nan_out or inf_out):
                _log(
                    2,
                    f"    [repl] fid={fid:2d} {name}: "
                    f"x(nan={nan_in_x},inf={inf_in_x}) "
                    f"a(nan={nan_in_a},inf={inf_in_a}) "
                    f"out(nan={nan_out},inf={inf_out})"
                )

    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    t1_all = time.perf_counter()
    stats = {
        "total_time_s": float(t1_all - t0_all),
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(
                2,
                f"[sanitize] final pass replaced: "
                f"NaN={rep_nan.get(('final', -1), 0)}, Inf={rep_inf.get(('final', -1), 0)}"
            )

    return stats

import time
from collections import defaultdict
import numpy as np

def instantiate_from_ops_chunked_intraday(
    population,
    *,
    transform_ops,
    chunk_B: int = 16,
    verbosity: int = 1,
    sanitize_final: bool = True,
):
    """
    Intraday segmented instantiation on population._X_inst of shape (N, G).

    Rules
    -----
    - population._time_terminals must be True
    - population._structure must equal 'Intraday'
    - time-of-day column is population._T_idx[-2]
    - every exact zero in that time-of-day column is treated as the start of a new day/chunk
    - instantiation is run separately on each chunk [zero_i : zero_{i+1}) or [zero_last : N)

    This preserves intraday resets for:
    - rolling / window-like transforms
    - self-referencing behavior
    - value initialization from t0 within each chunk

    Verbosity
    ---------
    0 : silent
    1 : print number of counted days (zero starts) + total timing
    2 : add per-day timing + replacement summary
    3 : add per-op timing
    4 : add per-chunk timing / debug details

    Returns
    -------
    stats : dict
        Timing and replacement statistics.

    Notes
    -----
    - population._X_inst is updated in place
    - this function assumes the time-of-day column is already present in population._X_inst
    - this function requires:
        _VAR_TO_BIT
        _VAR_TO_COL
        _FUNC_ID_TO_NAME
        F_AS
        build_operation_list
      to exist in scope
    """
    t0_all = time.perf_counter()

    # ------------------------------------------------------------
    # population validation
    # ------------------------------------------------------------
    if not hasattr(population, "_time_terminals"):
        raise AttributeError("population must have attribute '_time_terminals'")
    if population._time_terminals is not True:
        raise ValueError("population._time_terminals must be True")

    if not hasattr(population, "_structure"):
        raise AttributeError("population must have attribute '_structure'")
    if population._structure != "Intraday":
        raise ValueError(
            f"population._structure must be 'Intraday', got {population._structure!r}"
        )

    if not hasattr(population, "_instructions"):
        raise AttributeError("population must have attribute '_instructions'")
    if not hasattr(population, "_X_inst"):
        raise AttributeError("population must have attribute '_X_inst'")
    if not hasattr(population, "_T_idx"):
        raise AttributeError("population must have attribute '_T_idx'")

    instructions = population._instructions
    X_out = population._X_inst

    if not isinstance(instructions, np.ndarray):
        raise TypeError("population._instructions must be a numpy ndarray")
    if not isinstance(X_out, np.ndarray):
        raise TypeError("population._X_inst must be a numpy ndarray")

    if instructions.ndim != 2 or instructions.shape[1] < 10:
        raise ValueError("population._instructions must be shape (G,11) or have cols 0..9")
    if X_out.ndim != 2:
        raise ValueError("population._X_inst must be 2D with shape (N, G)")

    N, Gx = X_out.shape
    G = instructions.shape[0]

    if Gx != G:
        raise ValueError(
            f"population._X_inst.shape[1]={Gx} must equal number of instructions G={G}"
        )

    if len(population._T_idx) < 2:
        raise ValueError("population._T_idx must have at least two entries so _T_idx[-2] exists")

    tod_col = int(population._T_idx[-2])
    if tod_col < 0 or tod_col >= G:
        raise ValueError(
            f"time-of-day column population._T_idx[-2]={tod_col} is out of bounds for G={G}"
        )

    def _log(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    # ------------------------------------------------------------
    # detect day starts from exact zeros in TOD column
    # ------------------------------------------------------------
    tod = X_out[:, tod_col]
    day_starts = np.flatnonzero(tod == 0)

    if day_starts.size == 0:
        raise ValueError(
            f"No zeros were found in the time-of-day column at population._T_idx[-2]={tod_col}"
        )

    if day_starts[0] != 0:
        raise ValueError(
            f"First zero in the time-of-day column occurs at row {int(day_starts[0])}, not row 0. "
            "This would leave leading rows outside any intraday reset chunk."
        )

    day_ends = np.empty_like(day_starts)
    day_ends[:-1] = day_starts[1:]
    day_ends[-1] = N

    days_counted = int(day_starts.size)

    if verbosity >= 1:
        _log(1, f"[intraday] counted days={days_counted}")

    # ------------------------------------------------------------
    # build schedule once
    # ------------------------------------------------------------
    op_list = build_operation_list(instructions)
    Bmax = int(chunk_B)

    const_flags  = instructions[:, 3].astype(np.uint32, copy=False)
    sensor_flags = instructions[:, 4].astype(np.uint32, copy=False)

    rep_nan = defaultdict(int)
    rep_inf = defaultdict(int)

    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    day_times = []

    def _sanitize_inplace(arr: np.ndarray, key):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            rep_nan[key] += int(nan_mask.sum())

        fin_mask = np.isfinite(arr)
        if (~fin_mask).any():
            rep_inf[key] += int((~fin_mask).sum() - nan_mask.sum())

        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _fill_series(X_seg: np.ndarray, var_code: str, idx_chunk: np.ndarray, buf: np.ndarray, func_id: int):
        """
        Fill buf[:, :B] using one day/segment slice X_seg of shape (Ns, G):
          - CONST flag  => broadcast scalar down Ns
          - SENSOR flag => gather parent series from X_seg[:, parents]
          - else        => 0.0

        Sanitizes NaN/Inf -> 0 in place.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        raw = instructions[idx_chunk, col]

        out_view = buf[:, :B]
        out_view.fill(0.0)

        if np.any(is_sensor):
            disp = raw[is_sensor].astype(np.int64, copy=False)

            if np.any(disp >= 0):
                bad = disp[disp >= 0][:10]
                raise ValueError(
                    f"{var_code}: expected negative displacement for sensor slots; got {bad}"
                )

            parents = idx_chunk[is_sensor] + disp
            if np.any(parents < 0) or np.any(parents >= G):
                bad = parents[(parents < 0) | (parents >= G)][:10]
                raise ValueError(
                    f"{var_code}: parent out of bounds (first bad: {bad})"
                )

            # important: gather only from current segment to preserve intraday reset behavior
            out_view[:, is_sensor] = X_seg[:, parents]

        if np.any(is_const):
            cvals = raw[is_const].astype(X_seg.dtype, copy=False)
            out_view[:, is_const] = cvals[None, :]

        stage = "input_x" if var_code == "x" else "input_a"
        _sanitize_inplace(out_view, (stage, func_id))

        if verbosity >= 4:
            _log(
                4,
                f"    [{stage}] segN={X_seg.shape[0]} B={B} "
                f"const={int(is_const.sum())} sensor={int(is_sensor.sum())}"
            )

        return out_view

    def _param_vec(var_code: str, idx_chunk: np.ndarray, cast, default, clamp_min=None):
        """
        Per-gene parameter vector (delta/kappa).
        Sensor-flagged params are invalid.
        """
        B = idx_chunk.size
        bit = np.uint32(_VAR_TO_BIT[var_code])
        col = _VAR_TO_COL[var_code]

        cf = const_flags[idx_chunk]
        sf = sensor_flags[idx_chunk]
        is_const  = ((cf >> bit) & np.uint32(1)).astype(bool)
        is_sensor = ((sf >> bit) & np.uint32(1)).astype(bool)

        if np.any(is_sensor):
            raise ValueError(f"{var_code}: unexpectedly sensor-flagged for param vector")

        raw = instructions[idx_chunk, col]
        out = np.empty(B, dtype=cast)
        out[:] = cast(default)

        if np.any(is_const):
            out[is_const] = raw[is_const].astype(cast, copy=False)

        if clamp_min is not None:
            out = np.maximum(out, cast(clamp_min))

        return out

    # ------------------------------------------------------------
    # per-day segmented instantiation
    # ------------------------------------------------------------
    for day_i, (start_row, end_row) in enumerate(zip(day_starts, day_ends), start=1):
        t0_day = time.perf_counter()

        X_seg = X_out[start_row:end_row, :]
        Ns = X_seg.shape[0]

        if Ns <= 0:
            continue

        x_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        a_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)
        y_buf = np.empty((Ns, Bmax), dtype=X_seg.dtype)

        for func_id, gene_idx in op_list:
            fid = int(func_id)
            if fid == 0:
                continue

            name = _FUNC_ID_TO_NAME.get(fid, f"fid_{fid}")
            used_vars = F_AS(fid)

            gene_idx = np.asarray(gene_idx, dtype=np.int64).reshape(-1)
            if gene_idx.size == 0:
                continue

            t0_op = time.perf_counter()

            for start in range(0, gene_idx.size, Bmax):
                idx_chunk = gene_idx[start:start + Bmax]
                B = idx_chunk.size

                t0_chunk = time.perf_counter()

                x_view = _fill_series(X_seg, "x", idx_chunk, x_buf, fid)

                alpha_arg = None
                if "a" in used_vars:
                    a_view = _fill_series(X_seg, "a", idx_chunk, a_buf, fid)
                    alpha_arg = a_view

                delta1_vec = None
                delta2_vec = None
                kappa_vec  = None

                if "d" in used_vars:
                    delta1_vec = _param_vec("d", idx_chunk, cast=np.int64, default=2, clamp_min=2)
                if "dd" in used_vars:
                    delta2_vec = _param_vec("dd", idx_chunk, cast=np.int64, default=2, clamp_min=2)
                if "k" in used_vars:
                    kappa_vec = _param_vec("k", idx_chunk, cast=np.float32, default=1.0, clamp_min=None)

                y_view = y_buf[:, :B]
                transform_ops.apply(
                    fid,
                    x_view,
                    alpha=alpha_arg,
                    delta1=delta1_vec,
                    delta2=delta2_vec,
                    kappa=kappa_vec,
                    out=y_view,
                    in_place=False,
                )

                _sanitize_inplace(y_view, ("output", fid))

                # write back only into this segment
                X_seg[:, idx_chunk] = y_view

                if verbosity >= 4:
                    _log(
                        4,
                        f"  [day={day_i:4d}] [chunk] {name} fid={fid} "
                        f"rows=[{start_row}:{end_row}) gene_start={start} B={B} "
                        f"dt={(time.perf_counter()-t0_chunk)*1000:.1f}ms"
                    )

            dt_op = time.perf_counter() - t0_op
            op_times[fid] += dt_op
            op_counts[fid] += int(gene_idx.size)

            if verbosity >= 3:
                _log(
                    3,
                    f"[day={day_i:4d}] [op] {name:5s} fid={fid:2d} "
                    f"genes={gene_idx.size:6d} seg_rows={Ns:6d} time={dt_op:.3f}s"
                )

        dt_day = time.perf_counter() - t0_day
        day_times.append(float(dt_day))

        if verbosity >= 2:
            _log(
                2,
                f"[day={day_i:4d}] rows=[{int(start_row)}:{int(end_row)}) "
                f"Nseg={Ns} time={dt_day:.3f}s"
            )

    # ------------------------------------------------------------
    # final sanitize on whole instantiated matrix
    # ------------------------------------------------------------
    if sanitize_final:
        _sanitize_inplace(X_out, ("final", -1))

    total_time_s = time.perf_counter() - t0_all

    stats = {
        "days_counted": int(days_counted),
        "time_of_day_col": int(tod_col),
        "day_starts": day_starts.copy(),
        "day_ends": day_ends.copy(),
        "total_time_s": float(total_time_s),
        "day_times_s": day_times,
        "op_times_s": dict(op_times),
        "op_gene_counts": dict(op_counts),
        "replaced_nan": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_nan.items()},
        "replaced_inf": {f"{k[0]}:{k[1]}": int(v) for k, v in rep_inf.items()},
    }

    if verbosity >= 1:
        _log(1, f"[intraday] total_time={total_time_s:.3f}s")

    if verbosity >= 2:
        total_nan = sum(rep_nan.values())
        total_inf = sum(rep_inf.values())
        _log(2, f"[sanitize] total replaced: NaN={total_nan}, Inf={total_inf}")
        if sanitize_final:
            _log(
                2,
                f"[sanitize] final pass replaced: "
                f"NaN={rep_nan.get(('final', -1), 0)}, "
                f"Inf={rep_inf.get(('final', -1), 0)}"
            )

    return stats

import pandas as pd

def initialize(
    structure   :   str =   'Continuous',
    incl_time   :   bool=   False,
    data_file   :   str =   '../data/spy5m.csv',
    epoch_idx   :   list=   [0],
    hlocv_idx   :   list=   [1,2,3,4],
    pop_size    :   int =   1000,
    grmr_type   :   str =   'Null',
    grmr_mdl    :   int | tuple =   240,
    gen_chunk   :   float   =   0.1,
    verbose     :   int =   0
):
    '''
    The all in one function for initialization
    this function should be usable as a one liner, ending at instantated genes
    '''

    #read in data first
    x_raw = pd.read_csv(data_file)

    #temporary error thrower to ensure we are not working with different
    #shapes just so these next few lines are made modular
    if x_raw.shape[1] != 5:
        raise ValueError('x_raw came in with NOT 5 columns. check top of instantate function, currently using non modular approach for very simple variable initialization.')
    
    X_initialized = np.zeros((x_raw.shape[0], pop_size), dtype=np.float32)
    x_raw = x_raw.to_numpy(dtype=np.float32, copy=False)
    X_initialized[:, :5] = x_raw
    del x_raw
    

    if(verbose>1):print('Data loaded')

    #generate our population variable and pass all parameters
    X = Population(
        X_inst=X_initialized,
        terminal_idx=hlocv_idx,
        excluded_idx=epoch_idx,
        max_size=pop_size,
        include_time=incl_time,
        structure=structure
    )
    if(verbose>1):print('Population initialized')
    
    #generate our grammar variable and pass all parameters
    grammar = Grammar(
        type=grmr_type,
        max_delta_lookback=grmr_mdl
    )
    if(verbose>1):print('Grammar Initialized')

    if(verbose>0):print('Initializations complete')

    #clarify gen_chunk format
    #gen_chunk can be fraction of generating space, or actual integer
    if(gen_chunk<0):
        raise ValueError(f'gen_chunk came in as negative ({gen_chunk}). Must be positive.')
    elif(gen_chunk<1):
        chunk_size = int(np.ceil(pop_size * gen_chunk))
    elif(gen_chunk>pop_size):
        chunk_size = pop_size
    else:
        chunk_size = gen_chunk
    if(verbose>0):print(f'Chunk Sizes: {chunk_size}')
    if(verbose>1):print(f'pop size: {pop_size, type(pop_size)}')
    #if(verbose>1):print(f'Tidx size: {X._T_idx.size}')
    #if(verbose>1):print(f'Lidx size: {X._L_idx.size}')
    if(verbose>1):print(f'n size: {int(pop_size - X._E_idx.size - X._T_idx.size)}')
    
    #generate instructions (in place, resides in X)
    generate_instructions(
        pop_prior=X,
        grm_prior=grammar,
        #a little clunky but maybe easiest,
        #this is the open space not used so far
        #so total size - excluded space and legal existing space
        n=int(pop_size - X._E_idx.size - X._T_idx.size),
        chunk_size=chunk_size
    )
    #print(X._instructions)
    if(verbose>0):print(f'Instructions generated.')
    if(verbose>0):print(f'Initialization complete.')

    #this functions returns the initial population (has instructions) and grammar
    return X, grammar