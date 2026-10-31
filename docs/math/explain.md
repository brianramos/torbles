# Explanation specialization and deletion propagation (PROV-002)

`torbles/math_explain.h` (every contract; the propagation `tr_expl_set` and its update laws are inline and shared
with CUDA), `math_explain.c`, `math_explain_gpu.cu`, `math_explain.py` (opt-in). Specializes the provenance PROV-001
retains ([prov.md](prov.md)) without recomputing derivations: the exact effect of deletions on every answer,
explanations, and answer probabilities under independent tuples, exact on the DAG with its shared variables
(COUNT-002's decision diagrams, [symcount.md](symcount.md)).

```python
from fractions import Fraction
from torbles import math_prov as mp, math_explain as mx
st = mp.Store(); R = st.source("R", {...}); S = st.source("S", {...}); T = st.source("T", {...})
G = R.join(S, on=[("k", "k")]).join(T, on=[("b", "b")]).group(["c"])
ex = mx.Explainer(st, [G, J], base={("R", 3): 0}, default=1)   # the retained cone; a base valuation
d = ex.delete([[("R", 0)], [("S", 1), ("S", 2)]], backend="cuda")   # one change set per request
d.counts[w][j], d.exists[w][j], d.status[w], d.affected[w], d.changed[w], d.effects(w), d.of(w, k)
ex.revalue([{("R", 2): 1}, {("S", 1): 3}])                          # restorations, multiplicities
ex.lineage(j); ex.relevant(j); ex.necessary(j); ex.sufficient(j); ex.smallest(j); ex.why_not(j)
p = ex.probability(j, {("R", 0): Fraction(1, 2)}, default=Fraction(9, 10), method="exact" | "bounds" | "auto",
                   order="dfs" | "natural" | [labels], backend="cpu" | "cuda")
p.status, p.value, p.bounds, p.read_once, p.diagram, p.refusal
```

Answers are the tuples of the given relations in order (`j`, or `(relation index, tuple index)`); `ex.baseline` is
their multiplicities and existence under the base.

## Deletion propagation

A change set overrides some source tuples' values (deletion 0, restoration 1, multiplicity k); the others keep their
base values. Only the DAG nodes above a changed tuple are recomputed, in increasing order through a min-heap, each
once, with an early cutoff where N and B both keep their baseline:

| node | multiplicity N (unsigned 128 bits) | existence B |
|---|---|---|
| PLUS | N - sum of the changed children's old N + sum of their new N (the linear law) | exists iff (children with B = 1) > 0, the count adjusted by the changed children only |
| TIMES | the product of the children's new values (= N + dA B + A dB + dA dB, DELTA-002's finite difference) | exists iff (children with B = 0) = 0 |

Per set: every answer's N and B (the baseline where nothing changed), `status` (`"overflow"` when an N leaves 128
bits: the set is redone in B, its counts are `None`), `affected` (nodes recomputed, the incremental cost against the
cone's size `ex.info["nodes"]`) and `changed`; `effects(w)` lists the changed answers with before / after. The
result equals PROV-001's evaluation of the whole DAG and a complete re-execution of the query (DuckDB in the tests).
A base whose N leaves 128 bits somewhere propagates B only (`info["count_ok"]` False, every N `"overflow"`).

## Explanations (relative to the base)

The existence of answer j is the monotone Boolean function f of its lineage (PLUS = OR, TIMES = AND), compiled with
the absent sources FALSE (why) or the present ones TRUE (why-not):

| call | result |
|---|---|
| `lineage` | the source tuples of the answer's cone (each occurs in a derivation: the multiplicity depends on it) |
| `relevant` | the present tuples the existence depends on (`x + x y`: x only; y is lineage, not relevant) |
| `necessary` | counterfactual sources: present tuples whose deletion alone removes the answer |
| `sufficient` | the minimal sets of present tuples that suffice (all present: PROV-001's minimal witnesses) |
| `smallest` | one minimum-cardinality sufficient set |
| `why_not` | for an absent answer: the minimal sets of absent tuples whose restoration makes it exist |

`status`: `"found"`, `"absent"` (the answer does not exist under the base: nothing explains it), `"present"`
(why-not of an existing answer), `"underivable"` (no recorded derivation can make it exist). Minimal sets come from
the diagram (MM(n) = MM(lo) + {x} x {S in MM(hi) : lo(S) = 0}); `sufficient` / `why_not` are refused past `max_sets`
(never truncated). For a global group (always present in SQL) the existence explained is COUNT(*) > 0.

## Probabilities

P(answer exists) when each source tuple is present independently with probability p (exact rationals in [0, 1]): the
weighted count of f's decision diagram with the literal weights (p, 1 - p), exact and with every shared tuple one
variable (a tuple used by several derivations is not counted as independent events: `r0 (s0 + s1)` at 1/2 is 3/8,
not 1 - (1 - 1/4)^2 = 7/16). The diagram (`order` "dfs" by default) is bounded by the budget (circuit, iterations):
`method="exact"` refuses past it, `"bounds"` returns a certified enclosure computed on the DAG without compilation
(disjoint children: exact independence; overlapping: Harris-FKG for increasing events, AND in [prod lo, min hi], OR
in [max lo, 1 - prod (1 - hi)]; outward-rounded doubles; `read_once` when every node had disjoint children),
`"auto"` the exact value within the budget, else the bounds with the refusal recorded. `backend="cuda"` counts the
diagram with COUNT-002's level kernels.

## CUDA

`tr_mgpu_explain_propagate`: one work item per change set, the shared `tr_expl_set` with its own scratch row (65 bytes
per cone node, restored entry by entry when done), its own rows of answers and results; chunks within 64 MiB of
scratch. Equal byte for byte to the CPU (emulated with a scrambled thread order in the tests). Explanations and the
probability bounds are CPU only; exact probabilities can count on the device.

## Budgets and refusals

| request | needs |
|---|---|
| state | circuit = cone nodes, support = child entries, output = answers, bytes = the copies, reverse edges, baseline |
| propagation | output = sets x answers, bytes = the scratch (CUDA: `tr_expl_device_bytes`: copies, outputs, scratch chunk) |
| lineage diagram | circuit = diagram nodes held, iterations = apply steps, bytes (COUNT-002's manager) |
| explanations | necessary: iterations = levels x nodes; sufficient / why-not: support = set entries in the memo, output = max_sets |
| bounds | bytes = support bitsets and intervals |

`AlgebraBudgetError`, `ValueError` (an unknown source label, relations of another store, probabilities outside
[0, 1], `method`), `TypeError` (floats), `IndexError` (no such answer), `UnsupportedError` (more than 2^32 - 2 nodes).

## Limitations

PROV-001's operators only (selection, union, equi-join, group-by; no aggregates other than multiplicity, no negation
or recursion). Probabilities assume independent tuples (no correlations declared between sources); the exact path is
bounded by the diagram size, which depends on the variable order; the bounds are certified but can be loose on
heavily shared lineages. Explanations, bounds and the base are per answer on the CPU; minimal sets are enumerated
explicitly (bounded by `max_sets` and the support budget).
