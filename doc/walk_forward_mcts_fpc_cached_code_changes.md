# Walk-forward MCTS FPC-cached code additions

This script is designed to work mostly at the script level. The only additions I recommend making inside your core codebase are small stability/utility changes.

## 1. Grammar mode setter

If your `Grammar` class does not already have this, add it:

```python
def mode(self, mode: str):
    self._mode = mode
    return self
```

The script has a fallback that sets `G._mode` directly, but a method is cleaner.

## 2. Disable MCTS trace by default for long walk-forward runs

Inside `case 'MCTS':` in `Grammar.__init__`, support:

```python
self._MCTS_TRACE_ENABLED = spec_gram_args.get("trace_enabled", False)

if self._MCTS_TRACE_ENABLED:
    self._MCTS_TRACE = []
else:
    self._MCTS_TRACE = None
```

Then wrap trace append calls:

```python
if getattr(grm_prior, "_MCTS_TRACE_ENABLED", False) and grm_prior._MCTS_TRACE is not None:
    grm_prior._MCTS_TRACE.append(...)
```

For this walk-forward script, trace should generally be disabled because repeated grammar snapshots can otherwise become memory-heavy.

## 3. Optional native grammar decay method

The script includes external `apply_grammar_decay(G, gamma)`, so this is optional. If you want it inside `Grammar`, add:

```python
def decay_evidence(self, gamma: float = 1.0):
    gamma = float(gamma)
    if gamma >= 1.0:
        return self
    gamma = max(gamma, 0.0)

    def decay_dict(d):
        if d is None:
            return
        for k in list(d.keys()):
            d[k] *= gamma

    for name in [
        "_MCTS_NODE_CUM", "_MCTS_NODE_COUNT", "_MCTS_NODE_EXPLORE_COUNT",
        "_MCTS_EDGE_CUM", "_MCTS_EDGE_COUNT", "_MCTS_EDGE_EXPLORE_COUNT",
        "_MCTS_ALPHA_DECISION_CUM", "_MCTS_ALPHA_DECISION_COUNT", "_MCTS_ALPHA_DECISION_EXPLORE_COUNT",
        "_MCTS_ALPHA_NODE_CUM", "_MCTS_ALPHA_NODE_COUNT", "_MCTS_ALPHA_NODE_EXPLORE_COUNT",
        "_MCTS_ALPHA_EDGE_CUM", "_MCTS_ALPHA_EDGE_COUNT", "_MCTS_ALPHA_EDGE_EXPLORE_COUNT",
    ]:
        if hasattr(self, name):
            decay_dict(getattr(self, name))

    for name in ["_MCTS_EXPLOIT_T", "_MCTS_EXPLORE_T", "_MCTS_ALPHA_EXPLOIT_T", "_MCTS_ALPHA_EXPLORE_T"]:
        if hasattr(self, name) and getattr(self, name) is not None:
            setattr(self, name, getattr(self, name) * gamma)

    def recompute(cum_name, count_name, mu_name):
        if not (hasattr(self, cum_name) and hasattr(self, count_name) and hasattr(self, mu_name)):
            return
        cum = getattr(self, cum_name)
        cnt = getattr(self, count_name)
        mu = getattr(self, mu_name)
        if cum is None or cnt is None or mu is None:
            return
        for k in list(cum.keys()):
            n = cnt.get(k, 0.0)
            if n > 1e-12:
                mu[k] = cum[k] / n

    recompute("_MCTS_NODE_CUM", "_MCTS_NODE_COUNT", "_MCTS_NODE_MU")
    recompute("_MCTS_EDGE_CUM", "_MCTS_EDGE_COUNT", "_MCTS_EDGE_MU")
    recompute("_MCTS_ALPHA_DECISION_CUM", "_MCTS_ALPHA_DECISION_COUNT", "_MCTS_ALPHA_DECISION_MU")
    recompute("_MCTS_ALPHA_NODE_CUM", "_MCTS_ALPHA_NODE_COUNT", "_MCTS_ALPHA_NODE_MU")
    recompute("_MCTS_ALPHA_EDGE_CUM", "_MCTS_ALPHA_EDGE_COUNT", "_MCTS_ALPHA_EDGE_MU")

    return self
```

Default `gamma=1.0` means no decay. Lower values make older chunk evidence count less in future updates.

## 4. Make `Grammar` pickle-safe

The walk-forward script saves grammar snapshots with `pickle`. If your grammar holds non-pickleable objects, clear or disable them before saving. In particular:

```python
G._MCTS_TRACE = None
```

or use the trace-enabled flag above.

## 5. FPC cache assumptions

The script caches one FPC curve per `chunk_num`. This assumes the following remain fixed across the run:

```text
solver_kwargs / emission / AD_cond
chunking/data source
score interpretation
FPC fitting logic
```

If you change any of those mid-run, the cache key should include those settings, not only `chunk_num`.
