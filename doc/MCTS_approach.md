# MCTS Approach for Stochastic Grammar Optimization

## 1. Abstract / Introductory Explanation

The goal of the MCTS grammar is to make instruction generation more intelligent than a simple random grammar, a one-dimensional UCB grammar, or a two-dimensional transition matrix grammar. Instead of only learning that a transformation function is good, or that one function tends to transition well into another, the MCTS grammar learns which partially built symbolic structures are good places to extend.

In this project, each generated gene can be understood as a state in a growing symbolic structure. A gene has a transformation function ID, an `x` parent, and optional `alpha`, `delta1`, `delta2`, and `kappa` inputs. The first version of the MCTS grammar focuses on learning two things:

1. **Where to step:** Which existing gene/state should become the `x` parent of the next generated gene?
2. **How to step:** Which transformation function should be applied from that selected parent state?

This makes instruction generation a sequential tree-building process. At each step, the grammar looks at the current partially built population, scores the available parent nodes, selects one parent, then selects a child transformation function from that parent.

The MCTS grammar uses three main ideas:

### Sparse Tree Memory

The tree is stored sparsely using dictionaries rather than a dense matrix. This is necessary because the state is not just a transformation function ID. A state can be represented as a path such as:

```text
T2 -> f12 -> f10 -> f6
```

This means the current gene came from terminal/source column `2`, then transformation function `12`, then function `10`, then function `6`. This is more informative than only saying the current function is `6`.

### UCT for Parent Selection

UCT is used to decide **where** to extend the current structure. Every legal existing node in the population is converted into an MCTS state key, scored, and sampled from. High-quality states and under-explored states are both favored.

### UCB for Child Transformation Selection

After a parent node is selected, UCB is used to decide **how** to step from that parent. The grammar scores possible child transformation functions from that parent and samples one function. This learns which transformations work well after specific symbolic contexts.

### Progressive Widening

Progressive widening prevents every possible child transformation from being opened immediately. A rarely visited parent state is allowed to try only a small number of child functions. As that parent is visited more often, it is allowed to explore more children. This keeps the search from becoming too wide too early.

### Max Depth Enforcement

The MCTS grammar also needs a true maximum depth. Limiting the path key length is not enough, because that only controls how much history is stored in the dictionary key. Actual max-depth enforcement must happen during parent selection. A parent is only legal if selecting it would produce a child whose depth is less than or equal to the maximum allowed depth.

In short, the MCTS grammar works as:

```text
current population structure
    -> score possible x parents using UCT
    -> softmax sample one parent
    -> score/select child transformation using progressive widening + UCB
    -> build one new instruction row
    -> evaluate generated structures
    -> update sparse node and edge memory
```

The first development version keeps `alpha`, `delta1`, `delta2`, and `kappa` random or constant. This keeps the search focused on the two most important structural questions: parent selection and transformation selection.

---

## 2. Detailed Step-by-Step Design

## 2.1 Core Data Structure

The MCTS grammar is stored as a sparse tree overlay on top of the existing instruction matrix.

The instruction matrix still stores the real generated candidate structure. The MCTS dictionaries store learned statistics about states and actions observed across training.

The main MCTS structures are:

```python
self._MCTS_NODE_CUM
self._MCTS_NODE_COUNT
self._MCTS_NODE_MU

self._MCTS_EDGE_CUM
self._MCTS_EDGE_COUNT
self._MCTS_EDGE_MU

self._MCTS_NODE_EXPLORE_COUNT
self._MCTS_EDGE_EXPLORE_COUNT

self._MCTS_CHILDREN
self._MCTS_TRACE
```

Conceptually:

```text
_MCTS_NODE_MU[key]
    mean fitness observed at this state

_MCTS_NODE_COUNT[key]
    number of exploit/update observations for this state

_MCTS_EDGE_MU[(parent_key, child_tf)]
    mean fitness observed when using child_tf from parent_key

_MCTS_EDGE_COUNT[(parent_key, child_tf)]
    number of exploit/update observations for that action

_MCTS_CHILDREN[parent_key]
    set of child transformation functions already opened from this parent
```

This is an implicit tree. There is no recursive `Node` object. The parent-child structure is represented by dictionary keys.

---

## 2.2 What a Key Means

A path key looks like this:

```python
('P', ('T', 2), 12, 10, 6, 19, 5, 4, 8)
```

This should be read as:

```text
terminal/source 2
    -> transformation function 12
    -> transformation function 10
    -> transformation function 6
    -> transformation function 19
    -> transformation function 5
    -> transformation function 4
    -> transformation function 8
```

The first item, `'P'`, means this is a path-style key. The terminal tuple `('T', 2)` means the path begins from terminal/source column `2`. Every integer after that is a transformation function ID.

The final integer is the transformation function of the current node. For example:

```python
('P', ('T', 2), 12, 10, 6, 19, 5, 4, 8)
```

has current transformation function:

```text
8
```

A parent key such as:

```python
('P', ('T', 2), 12, 10, 6)
```

means:

```text
terminal/source 2
    -> transformation function 12
    -> transformation function 10
    -> transformation function 6
```

If this parent is selected and the selected child transformation is `19`, then the child path becomes:

```python
('P', ('T', 2), 12, 10, 6, 19)
```

So the relationship is:

```text
parent_key = where the grammar is stepping from
child_tf   = how the grammar is stepping
child_key  = resulting new symbolic state
```

The current first version of the key does not encode `alpha`, `delta1`, `delta2`, or `kappa`. That is intentional. The first MCTS version is meant to learn parent selection and transformation selection first.

---

## 2.3 Instruction Matrix Relationship

In the project instruction format, the generated row contains an absolute population index and relative references.

The key relationship for `x` is:

```python
parent_abs_idx = instructions[row, 0] + instructions[row, 5]
```

where:

```text
instructions[row, 0] = current absolute gene index
instructions[row, 5] = x parent offset
```

Therefore, when MCTS selects a parent, the generated instruction row writes:

```python
inst_inst[:, 5] = parent_abs_idx - inst_inst[:, 0]
```

This stores the MCTS-selected parent as the `x` sensor offset.

---

## 2.4 Generation Loop Overview

The MCTS case in `generate_instructions` should generate one new row at a time:

```python
chunk_size = 1
```

This is important because MCTS is sequential. If the system generates 40 rows before appending any of them, then row 2 cannot select row 1 as a parent inside the same generation chunk. Setting `chunk_size = 1` makes every newly generated node immediately available as a candidate parent for the next generation step.

Each MCTS generation step does the following:

1. Allocate one new instruction row.
2. Select the `x` parent using UCT over existing legal nodes.
3. Select the child transformation function using progressive widening and UCB.
4. Fill used flags, sensor flags, and constant flags.
5. Fill random/constant values for `alpha`, `delta1`, `delta2`, and `kappa`.
6. Randomly fill sensors as normal.
7. Overwrite the `x` sensor with the selected MCTS parent.
8. Record a trace entry for debugging and visualization.
9. Append the row to the population.

---

## 2.5 WHERE Selection: UCT Parent Selection

The first MCTS decision is:

```text
Where should the new gene come from?
```

This means selecting an existing legal parent node from the current population. This selected parent becomes the `x` sensor of the new gene.

For each legal parent index:

1. Convert the instruction row into a state key.
2. Compute its UCT score.
3. Softmax sample one parent from those scores.

The UCT function is:

```python
def _mcts_node_uct(self, key):
    q = self._MCTS_NODE_MU.get(key, self._MCTS_BASE_PRIOR)

    n = self._mcts_get_node_count(key, use_explore=True)

    if n <= 0 and key not in self._MCTS_NODE_MU:
        return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

    explore = np.sqrt(
        self._c
        * np.log(self._MCTS_EXPLORE_T + self._MCTS_EXPLOIT_T + 2)
        / (n + 1)
    )

    return q + explore
```

The formula is:

```text
UCT(node) = Q(node) + sqrt(c * log(total_visits + 2) / (N(node) + 1))
```

where:

```text
Q(node)
    mean fitness for that node/state

N(node)
    visit count for that node/state

total_visits
    total MCTS explore + exploit count

c
    exploration constant
```

This is used for parent selection. In project terms:

```text
UCT chooses WHERE to step.
```

A high UCT score means the node is either known to be good, under-explored, or both.

---

## 2.6 Max Depth Enforcement During Parent Selection

A max-depth limit must be enforced before UCT sampling.

The key rule is:

```text
parent_depth + 1 <= max_depth
```

or equivalently:

```text
parent_depth < max_depth
```

The terminal/source has depth `0`. A direct transform from a terminal has depth `1`.

Examples:

```text
T2
    depth 0

T2 -> f12
    depth 1

T2 -> f12 -> f10 -> f6
    depth 3
```

During parent selection, the grammar computes the depth of every legal parent. Any parent that would create a child deeper than the max depth is removed from the candidate set.

This matters because `_MCTS_MAX_PATH_DEPTH` only limits how much history the key builder reads. It does not prevent generation from creating deeper structures. Actual enforcement must happen before parent sampling.

The parent selection should therefore do:

```text
legal parent nodes
    -> filter by max depth
    -> compute UCT scores
    -> softmax sample selected parent
```

If all parents are too deep, the fallback is to select from terminal nodes if possible. This lets the grammar start a fresh branch instead of crashing.

---

## 2.7 HOW Selection: Progressive Widening + UCB Child Selection

After the parent is selected, the second decision is:

```text
How should the grammar transform from this parent?
```

This means selecting a child transformation function ID.

The grammar first checks progressive widening.

The progressive widening limit is:

```text
allowed_children(parent) = ceil(pw_c * (N(parent) + 1)^pw_alpha)
```

where:

```text
pw_c
    controls the initial width

pw_alpha
    controls how quickly the allowed child set grows

N(parent)
    count/visitation of the parent state
```

If the parent has opened fewer children than the widening limit, then the grammar may open a new randomly selected child transformation function.

If no expansion occurs, then the grammar selects among already opened children using UCB.

The UCB function is:

```python
def _mcts_edge_ucb(self, parent_key, child_tf):
    child_tf = int(child_tf)
    edge_key = (parent_key, child_tf)

    q = self._MCTS_EDGE_MU.get(edge_key, self._MCTS_BASE_PRIOR)

    parent_n = self._mcts_get_node_count(parent_key, use_explore=True)
    edge_n = self._mcts_get_edge_count(edge_key, use_explore=True)

    if edge_n <= 0 and edge_key not in self._MCTS_EDGE_MU:
        return self._MCTS_BASE_PRIOR + self._MCTS_UNKNOWN_PRIOR

    explore = np.sqrt(
        self._c
        * np.log(parent_n + 2)
        / (edge_n + 1)
    )

    return q + explore
```

The formula is:

```text
UCB(parent -> child_tf) = Q(parent -> child_tf)
                          + sqrt(c * log(N(parent) + 2) / (N(edge) + 1))
```

where:

```text
Q(parent -> child_tf)
    mean fitness observed for using child_tf from this parent state

N(parent)
    count for the parent state

N(edge)
    count for this specific parent -> child_tf action

c
    exploration constant
```

In project terms:

```text
UCB chooses HOW to step.
```

The complete child-selection process is:

```text
selected parent_key
    -> check number of opened children
    -> if widening allows, maybe open a new child tf
    -> otherwise score opened children with UCB
    -> softmax sample child transformation function
```

---

## 2.8 Softmax Selection

Both parent selection and child transformation selection use softmax sampling.

For parent selection:

```text
values = legal parent indices
scores = UCT scores
```

For child selection:

```text
values = opened child transformation function IDs
scores = UCB scores
```

The softmax turns scores into stochastic probabilities:

```text
P(i) = exp(score_i) / sum_j exp(score_j)
```

or, using a configurable base:

```text
P(i) proportional to base ^ score_i
```

This keeps MCTS from becoming purely greedy. A high-scoring parent or child is more likely to be selected, but lower-scoring alternatives may still be sampled.

---

## 2.9 Generation-Time Exploration Counts

The grammar has a `count_explore` mode. When this is enabled, generation-time attempts are counted immediately.

This means when MCTS selects:

```text
parent_key -> child_tf
```

it updates:

```python
_MCTS_NODE_EXPLORE_COUNT[parent_key]
_MCTS_EDGE_EXPLORE_COUNT[(parent_key, child_tf)]
_MCTS_EXPLORE_T
```

This is useful because the grammar can distinguish:

```text
explored but failed
```

from:

```text
never explored
```

That is important for UCB/UCT behavior. Without generation-time exploration counts, failed generated structures may look unexplored and continue receiving excessive exploration bonuses.

---

## 2.10 Evaluation and Grammar Update

After the population is generated and evaluated, the MCTS grammar is updated using surviving family indices and their scores.

The update function converts p-values into a fitness score using the same style as the existing UCB grammar:

```python
family_fitness = np.clip(-np.log(family_scores) / (-np.log(0.05)), None, 1)
```

Then the grammar updates sparse node and edge memories.

For each updated gene:

1. Find the child state key.
2. Recover its parent index from the `x` offset.
3. Build the parent state key.
4. Recover the child transformation function ID.
5. Update node memory for the child state.
6. Update edge memory for `parent_key -> child_tf`.
7. Register the child transformation as opened from that parent.

Conceptually:

```text
good evaluated gene
    -> child_key gets reward
    -> parent_key -> child_tf edge gets reward
```

The update affects future generations because:

```text
node rewards influence WHERE selection through UCT
edge rewards influence HOW selection through UCB
```

---

## 2.11 Visualization and Debugging

The visualization has two main goals:

1. Confirm the sparse MCTS memory is being populated.
2. Confirm the generated tree structure is interpretable.

Useful diagnostics include:

```text
number of node states
number of edge states
node exploit counts
edge exploit counts
node explore counts
edge explore counts
top nodes by mean fitness
top edges by mean fitness
maximum generated depth
```

The most useful edge view is:

```text
parent key -> child key
```

For example:

```text
T2 -> f12 -> f10 -> f6
    edge child_tf = 19
        -> T2 -> f12 -> f10 -> f6 -> f19
```

The visualization can also show edge labels such as:

```text
tf=19
mu=0.73
n=8
```

which means transformation function `19` from this parent has mean fitness `0.73` and has been updated `8` times.

---

## 2.12 Possible Areas of Failure

1. **State explosion**

Path keys can create many unique states. If the key is too specific, the grammar may not reuse knowledge enough.

2. **State collapse**

If the key mode is too simple, such as only using transformation function ID, then MCTS collapses back toward UCB1 and loses tree context.

3. **Bad depth fallback**

If every node becomes too deep and the grammar cannot find terminals correctly, generation may repeatedly fall back to shallow or random behavior.

4. **Over-exploration from unknown priors**

If `_MCTS_UNKNOWN_PRIOR` is too high, the grammar may keep preferring unseen states instead of exploiting known good paths.

5. **Under-exploration from progressive widening**

If `pw_c` or `pw_alpha` is too low, the grammar may not open enough child transformations from promising parents.

6. **Sparse update instability**

If only a tiny number of genes survive and update the grammar, node and edge estimates may be noisy.

7. **Current key does not encode alpha/d/dd/k**

This is intentional for version one, but it means different parameterizations of the same path are treated as the same MCTS state.



## LATE NOTES

- can try stabilization by switching backup_reduce to 
    'mean' from default 'max'

- consider marginal contribution scoring

given some set of scores, backpropagation is:
    convert evaluated gene score into quality
    walk the x-parent chain
    discount credit backward by distance
    group repeated node/edge updates
    update sparse MCTS node and edge memories