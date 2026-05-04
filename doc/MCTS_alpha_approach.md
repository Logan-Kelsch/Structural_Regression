# Alpha-Aware MCTS Grammar Approach

## 0. Purpose of this document

This document describes the current MCTS-style grammar structure used for stochastic symbolic-regression search, now extended to include **alpha usage** as a learned part of the grammar.

The document is written in two levels:

1. **Conceptual / introductory explanation**  
   This explains the approach in simpler terms: what the grammar is doing, why MCTS is useful, and what alpha adds.

2. **Detailed technical explanation**  
   This explains the actual mechanics: how parent states are represented, how UCT/UCB scores are computed, how progressive widening works, how alpha constant-vs-sensor selection is learned, how alpha parent selection works, how updates/backups are performed, and how to interpret the diagnostics.

By the end, the goal is that you can look at a printed key like:

```python
('ACTX', ('P', ('T', 2), 12), 5)
```

and understand that this means:

```text
In the context where x comes from T2 -> f12 and the new transformation is f5,
what should alpha be?
```

---

# Level 1: Abstract explanation

## 1. What problem is this trying to solve?

The project builds symbolic expressions one gene at a time. Each new gene is a transformation of previous information. A gene can use:

```text
x      main input stream / parent sensor
alpha  secondary input, threshold, comparison stream, or context sensor
d      parameter / lookback / delta-like value
dd     second parameter / second lookback / second delta-like value
k      integer or scalar control parameter
```

The core problem is:

```text
How should the system decide what kind of gene to build next?
```

A fully random grammar can explore, but it forgets what has worked. A dense transition matrix can learn function-to-function preferences, but it loses much of the actual structural context. The MCTS grammar is meant to learn from previous generated structures and use that memory to bias future construction.

The grammar learns answers to questions like:

```text
Which existing structures are good places to extend?
Which transformation functions work well from those structures?
When should alpha be a constant?
When should alpha be another learned sensor?
If alpha is a sensor, which previous structure should it point to?
```

---

## 2. Basic MCTS idea in this project

Classic Monte Carlo Tree Search is usually described as:

```text
selection -> expansion -> simulation -> backup
```

In this project, the symbolic-regression version is:

```text
select x parent -> select transformation function -> generate gene -> evaluate population -> backup reward into grammar
```

The important difference is that the generated program is not a simple tree object. Your program already lives inside an instruction matrix. So the MCTS grammar does not replace the instruction matrix. Instead, it acts as a learned memory layer on top of it.

The instruction matrix stores what was actually built.

The MCTS grammar stores what has tended to work.

---

## 3. Original MCTS without alpha

Before alpha was learned, the grammar made two main decisions:

### Decision 1: Where should the next gene come from?

This chooses the `x` parent.

Example:

```text
Current known structure:
    T2 -> f12 -> f10

Possible next step:
    use this as x parent for a new gene
```

This is the **WHERE** decision.

The grammar scores possible parent nodes using a UCT-style node score.

---

### Decision 2: How should the new gene transform the parent?

Once an `x` parent is selected, the grammar chooses the next transformation function.

Example:

```text
Selected x parent:
    T2 -> f12 -> f10

Possible child functions:
    f5, f8, f12, f18, ...
```

This is the **HOW** decision.

The grammar scores possible child transformations using a UCB-style edge/action score.

---

## 4. What alpha adds

Alpha introduces a second dependency channel.

A new gene is no longer just:

```text
new_gene = f(x, d, dd, k)
```

It may be:

```text
new_gene = f(x, alpha, d, dd, k)
```

Before alpha-MCTS, alpha was usually filled randomly as either:

```text
constant alpha
```

or:

```text
sensor alpha
```

based on a fixed random probability.

The new alpha-aware MCTS grammar makes alpha a learned decision.

When a function uses alpha, the grammar now asks:

```text
Should alpha be a constant?
Or should alpha be a sensor from another learned structure?
```

If alpha is selected as a sensor, the grammar then asks:

```text
Which existing structure should alpha point to?
```

This turns alpha into a second MCTS channel.

---

## 5. Why alpha should have its own memory

A node that is good as an `x` parent is not necessarily good as an `alpha` parent.

For example:

```text
T2 -> f12 -> f10
```

might be a very good main predictive stream, but not useful as a threshold/comparison/context input.

Another structure:

```text
T4 -> f18
```

might be mediocre as a main `x` stream but very useful as an alpha sensor because it represents volatility, dispersion, trend strength, or some kind of contextual condition.

Therefore, the grammar keeps alpha memory separate from x-parent memory.

The grammar learns:

```text
normal node memory:
    good as generated state / x parent

alpha node memory:
    good as alpha sensor
```

---

## 6. High-level alpha-aware generation flow

At each new gene construction step, the grammar does:

```text
1. Select x parent
2. Select child transformation function
3. If the child function uses alpha:
      decide alpha constant vs alpha sensor
4. If alpha sensor:
      select alpha parent
5. Fill constants and sensors
6. Overwrite x and alpha offsets with the MCTS-selected parents
7. Append new gene to population
```

Conceptually:

```text
x parent ---------------> new gene
                            ^
                            |
alpha parent ---------------|
```

If alpha is a constant, there is no alpha parent.

If alpha is a sensor, the generated structure becomes a small DAG instead of a pure chain.

---

## 7. Why this is still MCTS-like

The structure is MCTS-like because it keeps sparse statistical memory over states and actions.

It does not need a giant dense tensor over every possible gene structure.

Instead, it stores dictionaries like:

```python
state_key -> node value
(parent_key, child_tf) -> edge value
(alpha_context, action) -> alpha decision value
(alpha_context, alpha_parent_key) -> alpha edge value
```

This is sparse, flexible, and naturally supports progressive widening.

---

# Level 2: Detailed technical explanation

## 8. Instruction matrix interpretation

Each generated gene is represented as a row in the instruction matrix.

The common layout is:

```text
col 0: pop_idx / absolute gene index
col 1: func_id / transformation function ID
col 2: used flags
col 3: const flags
col 4: sensor flags
col 5: x
col 6: alpha
col 7: d
col 8: dd
col 9: k
col 10: unused / extra
```

For sensors, the value is usually stored as an offset.

For x:

```python
x_offset = parent_abs_idx - current_abs_idx
```

So if the current gene is at index 100 and its x parent is index 92:

```python
x_offset = 92 - 100 = -8
```

For alpha as a sensor:

```python
alpha_offset = alpha_parent_abs_idx - current_abs_idx
```

If alpha is a constant, column 6 contains the constant value instead of a sensor offset.

---

## 9. MCTS state keys

The MCTS grammar needs a way to identify states. A state is not just a function ID. It is a structural context.

The recommended key mode is path mode.

Example:

```python
('P', ('T', 2), 12, 10, 6)
```

This means:

```text
Path state:
    terminal/source 2
    -> function 12
    -> function 10
    -> function 6
```

Readable form:

```text
T2 -> f12 -> f10 -> f6
```

The last integer is the current transformation function. So in this key, the current node is `f6`.

The terminal marker:

```python
('T', 2)
```

means source/terminal column or row 2. The exact semantic meaning depends on how the population initialized terminals. It might correspond to a close/high/low/open/volume/time stream depending on your terminal indexing.

---

## 10. Normal MCTS memory structures

The normal x/function MCTS channel uses these sparse structures:

```python
_MCTS_NODE_CUM
_MCTS_NODE_COUNT
_MCTS_NODE_MU

_MCTS_EDGE_CUM
_MCTS_EDGE_COUNT
_MCTS_EDGE_MU

_MCTS_NODE_EXPLORE_COUNT
_MCTS_EDGE_EXPLORE_COUNT

_MCTS_CHILDREN
```

### Node memory

A node key represents a generated state.

Example:

```python
('P', ('T', 2), 12, 10, 6)
```

The node mean:

```python
_MCTS_NODE_MU[key]
```

means:

```text
How good has this state been as part of successful generated structures?
```

More precisely, it estimates downstream usefulness after reward backup.

---

### Edge memory

An edge key is:

```python
(parent_key, child_tf)
```

Example:

```python
(('P', ('T', 2), 12, 10), 6)
```

Meaning:

```text
From parent state T2 -> f12 -> f10,
how good is it to apply function f6 next?
```

The edge mean:

```python
_MCTS_EDGE_MU[(parent_key, child_tf)]
```

estimates the quality of taking that transformation from that parent state.

---

### Children memory

```python
_MCTS_CHILDREN[parent_key]
```

is a set of child transformation functions opened from that parent.

Example:

```python
_MCTS_CHILDREN[('P', ('T', 2), 12)] = {5, 10, 18}
```

Meaning:

```text
From T2 -> f12, the grammar has opened child transforms f5, f10, and f18.
```

This is used by progressive widening.

---

## 11. WHERE selection: x-parent UCT

The first MCTS decision is:

```text
Which existing node should be used as x parent?
```

The grammar looks at all currently legal parent candidates in the population, converts each candidate into a state key, and computes a UCT-like score.

The score is:

```text
UCT(node) = Q(node) + exploration(node)
```

Where:

```text
Q(node) = _MCTS_NODE_MU[node]
```

and:

```text
exploration(node) = sqrt(c * log(total_visits + 2) / (N(node) + 1))
```

In code form:

```python
q = self._MCTS_NODE_MU.get(key, self._MCTS_BASE_PRIOR)
n = self._mcts_get_node_count(key, use_explore=True)

explore = np.sqrt(
    self._c
    * np.log(self._MCTS_EXPLORE_T + self._MCTS_EXPLOIT_T + 2)
    / (n + 1)
)

score = q + explore
```

This score chooses **where to step from**.

After scores are computed, the grammar uses softmax sampling over the UCT scores.

So the grammar is not purely greedy. It is probabilistic, but biased toward higher UCT values.

---

## 12. HOW selection: child-transform UCB

After selecting the x parent, the grammar chooses which transformation function to apply next.

This is the child action.

The edge/action score is:

```text
UCB(parent -> child_tf) = Q(parent -> child_tf) + exploration(parent -> child_tf)
```

Where:

```text
Q(parent -> child_tf) = _MCTS_EDGE_MU[(parent_key, child_tf)]
```

and:

```text
exploration(parent -> child_tf)
    = sqrt(c * log(N(parent) + 2) / (N(edge) + 1))
```

In code form:

```python
edge_key = (parent_key, int(child_tf))

q = self._MCTS_EDGE_MU.get(edge_key, self._MCTS_BASE_PRIOR)
parent_n = self._mcts_get_node_count(parent_key, use_explore=True)
edge_n = self._mcts_get_edge_count(edge_key, use_explore=True)

explore = np.sqrt(
    self._c
    * np.log(parent_n + 2)
    / (edge_n + 1)
)

score = q + explore
```

This score chooses **how to step** from the selected x parent.

---

## 13. Progressive widening

Without progressive widening, every parent state could immediately try all possible child transformations. That can make the search too wide too early.

Progressive widening limits the number of opened children from a parent based on that parent’s visit count.

The rule is:

```text
allowed_children(parent) = ceil(pw_c * (N(parent) + 1)^pw_alpha)
```

Example:

```python
pw_c = 2.0
pw_alpha = 0.5
```

A parent with low count only gets a few opened children. As it becomes more visited, it can open more possible child functions.

This creates a controlled transition from exploration to exploitation.


---

## 13.5. Progressive widening parameters

The widening rule is:

```text
K(N) = ceil(pw_c * (N(parent) + 1)^pw_alpha)

K(N)
    maximum number of child actions allowed to be opened from this parent

N(parent)
    number of times this parent state has been visited or selected

pw_c
    controls how wide each node starts

pw_alpha
    controls how quickly the allowed width grows as evidence accumulates

---

## 14. Max depth and max path depth

There are two depth-related parameters.

### max_depth

`max_depth` limits what can actually be generated.

If:

```python
max_depth = 5
```

then paths can contain at most 5 transformation functions after the terminal.

Example allowed:

```text
T2 -> f12 -> f10 -> f6 -> f19 -> f5
```

Example not allowed:

```text
T2 -> f12 -> f10 -> f6 -> f19 -> f5 -> f8
```

Max-depth nodes can exist, but they cannot be selected as parents for deeper children.

---

### max_path_depth

`max_path_depth` controls how much ancestry is remembered in the dictionary key.

It does not limit generation. It limits memory representation.

For debugging, it is best to set:

```python
max_path_depth = max_depth + 1
```

or larger.

Later, lowering `max_path_depth` can intentionally compress memory by making the grammar remember only recent path suffixes.

---

## 15. Why alpha changes the graph from chain to DAG

Without alpha, a generated gene is mostly a chain:

```text
T2 -> f12 -> f10 -> f6
```

With alpha as a sensor, a generated gene can depend on two previous structures:

```text
x parent -----------------> new gene
                              ^
                              |
alpha parent -----------------|
```

So the dependency graph becomes a DAG.

The child depth should therefore be interpreted as:

```text
child_depth = 1 + max(depth(x_parent), depth(alpha_parent))
```

If alpha is constant, then:

```text
child_depth = depth(x_parent) + 1
```

---

# Alpha-aware MCTS

## 16. Alpha context

Alpha decisions are not global. They are made in context.

The alpha context key is:

```python
('ACTX', parent_key, child_tf)
```

Example:

```python
('ACTX', ('P', ('T', 2), 12), 5)
```

Readable form:

```text
Given:
    x parent = T2 -> f12
    child transformation = f5

Decide:
    should alpha be constant or sensor?
```

The marker `'ACTX'` simply means “alpha context.”

---

## 17. Alpha decision memory

The first alpha decision is:

```text
constant alpha vs sensor alpha
```

This is stored as:

```python
_MCTS_ALPHA_DECISION_MU[(ctx_key, action)]
```

Where:

```text
action = 0 means alpha constant
action = 1 means alpha sensor
```

Example:

```python
_MCTS_ALPHA_DECISION_MU[(('ACTX', ('P', ('T', 2), 12), 5), 1)]
```

Meaning:

```text
In the context where x = T2 -> f12 and child tf = f5,
how good has it been to use alpha as a sensor?
```

A printed diagnostic like:

```text
mu=0.4603 | n=3 | ne=86 | action=sensor | ctx=('ACTX', ('P', ('T', 2), 12), 5)
```

means:

```text
For x parent T2 -> f12 and child function f5,
the alpha=sensor decision has mean fitness 0.4603,
has received 3 scored backup updates,
and has been generated/explored 86 times.
```

The `n` value is exploit/update count.  
The `ne` value is generation/exploration count.

If `ne` is large and `n` is small, the decision has been tried often but has not yet received much scored evidence.

---

## 18. Alpha decision UCB

The grammar chooses constant vs sensor using a UCB-like score.

For an alpha context and action:

```text
score(ctx, action) = Q(ctx, action) + exploration(ctx, action) + prior(action)
```

Where:

```text
Q(ctx, action) = _MCTS_ALPHA_DECISION_MU[(ctx, action)]
```

and:

```text
exploration(ctx, action)
    = sqrt(c * log(alpha_total_visits + 2) / (N(ctx, action) + 1))
```

The prior comes from `alpha_sensor_freq`.

If:

```python
alpha_sensor_freq = 0.35
```

then before learning, the grammar is softly biased toward:

```text
35% sensor
65% constant
```

But this is not a fixed probability after learning. It is only a prior bias.

The learned Q values and exploration bonuses can push the actual probability above or below 0.35 for each context.

---

## 19. Alpha parent selection

If the grammar chooses:

```text
alpha action = sensor
```

then it must choose which previous structure alpha should point to.

This uses alpha-specific node memory:

```python
_MCTS_ALPHA_NODE_MU[alpha_parent_key]
```

This answers:

```text
How good has this node been when used as an alpha sensor?
```

This is different from:

```python
_MCTS_NODE_MU[key]
```

which answers:

```text
How good has this node been as an ordinary generated state / x parent?
```

The alpha parent is selected using alpha-node UCT:

```text
alpha_node_score = Q_alpha_node(key) + exploration_alpha_node(key)
```

Then softmax sampling selects the alpha parent.

---

## 20. Alpha edge memory

Once alpha parent selection exists, the grammar also learns which alpha parents work in which contexts.

This is stored as:

```python
_MCTS_ALPHA_EDGE_MU[(ctx_key, alpha_parent_key)]
```

Example:

```python
_MCTS_ALPHA_EDGE_MU[
    (
        ('ACTX', ('P', ('T', 2), 12), 5),
        ('P', ('T', 4), 18, 3)
    )
]
```

Meaning:

```text
When applying f5 to x parent T2 -> f12,
how good is it to use T4 -> f18 -> f3 as the alpha parent?
```

This is the most specific alpha relationship:

```text
this x/function context likes this alpha parent
```

---

## 21. Alpha-aware generation step in detail

One generated row proceeds like this:

### Step 1: Initialize one new instruction row

MCTS generation is easiest as one node at a time:

```python
chunk_size = 1
inst_inst = np.zeros((chunk_size, 11), dtype=np.float32)
```

The row gets a new population index.

---

### Step 2: Select x parent

The grammar evaluates legal parent nodes with UCT:

```text
existing node -> state key -> UCT score
```

Then softmax samples one parent.

This produces:

```python
parent_abs_idx
parent_key
```

---

### Step 3: Select child transformation

Using the selected `parent_key`, the grammar chooses a child transform using progressive widening and UCB.

This produces:

```python
child_tf
```

The new instruction row gets:

```python
inst_inst[:, 1] = child_tf
```

---

### Step 4: Build alpha context

The alpha context is:

```python
alpha_ctx_key = ('ACTX', parent_key, child_tf)
```

This says:

```text
For this selected x parent and selected child function,
what should alpha do?
```

---

### Step 5: Compute used flags

The function ID determines which arguments the operation uses.

If the function does not use alpha, the alpha channel is ignored.

If the function uses alpha, alpha-MCTS takes over.

---

### Step 6: Choose alpha constant vs sensor

If alpha is used, the grammar chooses:

```text
0 = constant
1 = sensor
```

using alpha-decision UCB plus the alpha prior.

---

### Step 7: If alpha sensor, select alpha parent

If alpha action is sensor, the grammar selects an alpha parent using alpha-node UCT.

It also enforces depth safety.

For a conservative rule:

```text
alpha_parent_depth <= max_depth - 1
```

and child depth is later interpreted as:

```text
1 + max(depth(x_parent), depth(alpha_parent))
```

If no valid alpha parent exists, the system falls back to constant alpha.

---

### Step 8: Fill constants and random sensors

The normal helper functions fill constants and random sensor references.

Then MCTS overwrites the selected channels.

---

### Step 9: Overwrite x and alpha offsets

The x sensor is overwritten with:

```python
inst_inst[:, 5] = parent_abs_idx - inst_inst[:, 0]
```

If alpha is a sensor:

```python
inst_inst[:, 6] = alpha_parent_abs_idx - inst_inst[:, 0]
```

This ensures the row uses the MCTS-selected parents rather than random parent references.

---

### Step 10: Count generation-time exploration

If training mode and `count_explore=True`, the grammar counts what was generated:

```python
_mcts_count_generation(parent_key, child_tf)
_mcts_count_alpha_generation(ctx_key, action, alpha_parent_key)
```

This updates exploration counts, not exploit scores.

---

### Step 11: Trace

The trace stores diagnostic information like:

```python
new_idx
parent_idx
parent_key
child_tf
x_offset
alpha_used
alpha_action
alpha_parent_idx
alpha_parent_key
alpha_offset
```

The trace is for debugging and visualization. It is not required for inference.

---

## 22. Training update / reward backup

After a population is generated and evaluated, you call:

```python
kept_s_idx, kept_scores, family_idx, family_scores = _E.reduce_scored_family_indices(
    X,
    s_idx,
    pvals[s_idx]
)

G.update(X, family_idx, family_scores)
```

This is the reward-backup stage.

In MCTS terms, `G.update(...)` performs grouped reward backup from evaluated population structures into the sparse grammar memory.

---

## 23. P-value to fitness

The default fitness transform is:

```text
fitness = -log(p) / -log(0.05)
```

with clipping to `[0, 1]`.

Interpretation:

```text
p = 1.00   -> fitness = 0
p = 0.05   -> fitness = 1
p < 0.05   -> clipped to 1
p > 0.05   -> between 0 and 1
```

This gives a simple significance-like reward signal.

---

## 24. Discounted path backup

Instead of giving full score to every ancestor, the better approach is discounted backup.

If a leaf node receives quality score:

```text
q_leaf = 1.0
```

and backup gamma is:

```python
backup_gamma = 0.85
```

then the leaf and ancestors receive:

```text
leaf             1.000
parent           0.850
grandparent      0.722
great-grandparent 0.614
```

This avoids over-crediting generic shallow ancestors.

The meaning is:

```text
The leaf directly produced the result.
The parents enabled the result, but get discounted credit.
```

---

## 25. What gets updated during backup?

For each scored gene/path, the update can affect:

```text
normal node memory
normal edge memory
alpha decision memory
alpha node memory
alpha edge memory
```

### Normal node update

```python
_MCTS_NODE_MU[child_key]
```

Learns how useful a generated state is.

---

### Normal edge update

```python
_MCTS_EDGE_MU[(parent_key, child_tf)]
```

Learns how useful a transform is from a parent state.

---

### Alpha decision update

```python
_MCTS_ALPHA_DECISION_MU[(ctx_key, action)]
```

Learns whether constant or sensor alpha worked in that context.

---

### Alpha node update

```python
_MCTS_ALPHA_NODE_MU[alpha_parent_key]
```

Learns whether a node works well as an alpha sensor.

---

### Alpha edge update

```python
_MCTS_ALPHA_EDGE_MU[(ctx_key, alpha_parent_key)]
```

Learns whether a specific alpha parent works for a specific x/function context.

---

## 26. Grouped updates

If the same key appears multiple times during one update call, rewards can be grouped before applying the update.

Possible reducers:

```text
max
mean
median
```

Recommended early reducer:

```text
max
```

because early in discovery, you often want to know whether a state has demonstrated potential.

Recommended later reducer:

```text
mean
```

because later you may want consistency rather than occasional success.

---

# Diagnostics and interpretation

## 27. Reading normal MCTS keys

Example:

```python
('P', ('T', 2), 12, 10, 6, 19, 5, 4, 8)
```

Read as:

```text
T2 -> f12 -> f10 -> f6 -> f19 -> f5 -> f4 -> f8
```

The current transformation function is the last one:

```text
f8
```

---

## 28. Reading alpha context keys

Example:

```python
('ACTX', ('P', ('T', 2), 12), 5)
```

Read as:

```text
For x parent T2 -> f12,
and new child function f5,
make an alpha decision.
```

This context does not itself say whether alpha is constant or sensor. The action does.

Action meanings:

```text
0 = constant
1 = sensor
```

---

## 29. Reading top alpha decisions

Example printout:

```text
mu=0.4603 | n=3 | ne=86 | action=sensor | ctx=('ACTX', ('P', ('T', 2), 12), 5)
```

Read as:

```text
In the context x = T2 -> f12 and child tf = f5,
using alpha as a sensor has mean backed-up fitness 0.4603.
It has received 3 scored exploit updates and 86 generation/exploration counts.
```

A high `ne` with low `n` means it has been tried often but has little scored evidence.

A strong learned signal is:

```text
high mu
high n
sensor beating constant in the same context
```

---

## 30. Alpha-MCTS state growth plot

The alpha state growth plot usually shows:

```text
decision states
node states
edge states
```

### Decision states

Number of unique alpha const-vs-sensor decisions learned.

A decision state is:

```python
(ctx_key, action)
```

If this grows, the grammar is learning more alpha usage choices across contexts.

---

### Node states

Number of unique nodes that have been used and scored as alpha parents.

If this grows, the grammar is trying more different structures as alpha sensors.

---

### Edge states

Number of unique context-to-alpha-parent relationships.

An alpha edge is:

```python
(ctx_key, alpha_parent_key)
```

If this grows, the grammar is learning which alpha parent works for which x/function context.

---

## 31. Mean p(alpha sensor)

This plot shows the average probability that the grammar assigns to choosing alpha as a sensor.

It is not simply the raw parameter:

```python
alpha_sensor_freq
```

That parameter is only a prior.

The actual probability is learned from:

```text
alpha decision Q values
alpha decision exploration bonuses
alpha prior bias
softmax temperature
```

If the plot stays around 0.35 and your prior is 0.35, that means the learned policy is still close to the prior on average.

If it moves above 0.35, the grammar is learning that sensor alpha is often useful.

If it moves below 0.35, the grammar is learning that constant alpha is often better.

If it oscillates, the grammar is still uncertain or different contexts are changing differently.

The global parameter is not learning. The local context policies are learning.

---

## 32. p(new node depth)

This plot shows where the grammar wants to grow the next node.

If depth 3 is bright, it means:

```text
The grammar is assigning high probability to parents that will create depth-3 children.
```

Because:

```text
new node depth = selected parent depth + 1
```

With alpha sensors, the actual child depth can be:

```text
1 + max(depth(x_parent), depth(alpha_parent))
```

so depth diagnostics should eventually account for alpha-parent depth too.

---

## 33. |delta p(depth)|

This plot shows how much the search-depth distribution changes between iterations.

High early values are normal.

Fading values suggest convergence.

Persistent bright bands suggest the grammar is still uncertain or oscillating at certain depths.

---

## 34. p(alpha parent depth)

An alpha-aware diagnostic can also show the probability distribution over alpha parent depths.

This answers:

```text
When alpha is a sensor, how deep are the structures being selected as alpha parents?
```

If alpha parents are shallow, alpha may be acting like a simple threshold/context stream.

If alpha parents are deep, alpha may be acting like a complex learned comparator.

---

# Inference behavior

## 35. Training vs inference

Training mode should allow:

```text
generation
exploration counting
trace logging
reward backup
opened-child growth
```

Inference mode should ideally be static:

```text
generation only
read grammar memory
no count updates
no score updates
no child-opening mutations
```

For a finalized grammar, inference should use:

```python
mode = 'infer'
```

and should not call:

```python
G.update(...)
```

unless you explicitly want online adaptation.

---

## 36. What is actually learned?

The grammar learns a conditional construction policy.

Normal MCTS learns:

```text
which states are good to extend
which functions are good from those states
```

Alpha-MCTS additionally learns:

```text
when alpha should be constant
when alpha should be a sensor
which structures are good alpha sensors
which alpha sensors work in which x/function contexts
```

The learned policy is not one global rule like:

```text
always use alpha sensor 35% of the time
```

It is context-specific:

```text
In this context, alpha sensor is useful.
In that context, alpha constant is better.
In this other context, this specific alpha parent is useful.
```

---

# Possible areas of failure

## 37. Over-crediting shallow ancestors

If full reward is passed to all parents, generic shallow nodes can become overvalued.

Mitigation:

```text
use discounted backup
use mean or top-quantile reducers later
monitor shallow depth domination
```

---

## 38. Alpha search explosion

Adding alpha sensor selection increases the branching factor.

Now the grammar explores:

```text
x parent
child function
alpha constant/sensor
alpha parent
```

Mitigation:

```text
use progressive widening
limit max_depth
use alpha prior weight
use conservative alpha parent depth filtering
```

---

## 39. Alpha role confusion

A node may be good as x but bad as alpha, or vice versa.

Mitigation:

```text
keep alpha node memory separate from normal node memory
monitor alpha node states independently
```

---

## 40. Too few exploit updates

A diagnostic like:

```text
n=3, ne=86
```

means the action has been generated often but scored rarely.

Mitigation:

```text
avoid trusting high-mu low-n actions too strongly
monitor n and ne together
require minimum n for interpretation
```

---

## 41. Max-depth diagnostic artifacts

Max-depth nodes may show high raw exploration values because they cannot be selected as parents, so their parent-selection count stays low while global visits rise.

Mitigation:

```text
use active/expandable node_u diagnostics
verify max-depth nodes are not legal parents
```

---

## 42. Mean p(alpha sensor) can hide context differences

The mean can stay near 0.35 even if individual contexts have strongly learned preferences in opposite directions.

Mitigation:

```text
inspect top alpha decisions
compare constant vs sensor within the same ctx
plot distribution of p(alpha sensor), not only mean
```

---

# Final summary

The alpha-aware MCTS grammar builds symbolic structures by learning a conditional construction policy.

At each new gene, it learns:

```text
where to take x from
which transformation function to apply
whether alpha should be constant or sensor
if alpha is a sensor, where alpha should come from
```

The key design choice is to treat alpha as a separate MCTS channel, not just a random sensor choice.

This allows the grammar to learn not only useful symbolic transformations, but also useful relational structure between generated streams.

In short:

```text
normal MCTS learns how to grow expressions.
alpha-MCTS learns how expressions should condition, compare, or contextualize each other.
```
