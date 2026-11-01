# Graph layouts, fixed points, paths and energies (GRAPH-001, GRAPH-002)

`torbles/math_graph.h` (every contract; the per-vertex pulls, multiword counts and blocked floating sums are inline and
shared with CUDA), `math_graph.c`, `math_graph_gpu.cu`, `math_graph.py` (opt-in). GRAPH-001 builds a declared graph
layout from the edge rows of an FND-004 scope (`docs/math/bank.md`) and reduces over it; GRAPH-002 runs bounded
fixed points, path problems and energies on a layout. The header comment of `math_graph.h` is normative.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_graph as mg
scope = mbk.Scope.from_table(db["e"], ["src", "dst", "w"], rows=mask)
G = mg.Layout.build(scope, "src", "dst", weight="w", directed=True, dedup=False, loops="keep", combine="refuse",
                    nulls="skip", vertices=(vertex_scope, "id"))         # or Layout.from_table / Layout.from_edges
G.keys(); G.ids([3, 7]); G.vertex_set([3]); G.adjacency("out" | "in"); G.edges_list(); G.incidence(); G.info
G.degrees("out" | "in" | "total" | "loops", weights="one" | "edge" | "distinct", backend="cuda")
G.expand(frontier, direction="forward" | "backward", exclude=visited)    # a VertexSet (.count)
G.spmv(x, "count" | "bool" | "gf2" | "min_plus" | "max_plus" | "float", direction=..., weights=...)
G.aggregate(x, "sum" | "distinct_sum" | "min" | "max" | "any" | "count"); G.gf2_matrix()   # a FIELD-001 Matrix
G.reach(sources, reflexive=True, max_rounds=None); G.closure(sources=None, reflexive=False)
G.walks(sources, 16, bound=2**200, rows=True); G.walks(sources, None); G.walks_need(sources, 16)
G.paths(sources, "min_plus" | "max_plus", algorithm="auto" | "bellman_ford" | "dijkstra")
G.energy(x); G.pagerank(0.85, tol=1e-12, max_iter=200, teleport=None, start=None); G.power(tol=1e-12)
G.need("spmv", "float", backend="cuda"); mg.Layout.build_need(scope, "src", "dst", "w")
```

## Layouts (GRAPH-001)

- **Source**: the selected rows of a one-group scope are edge rows; `src` / `dst` are key columns of one decimal scale
  (exact logical integers; a decimal key is a Fraction), `weight` an optional column (value = logical / D,
  D = 10^scale x `denominator`; `from_edges` puts Fraction weights on their least common denominator natively).
  `nulls="skip"`: a row with a NULL endpoint or weight is no edge (`info["null_endpoints"]`, `["null_weights"]`);
  `"forbid"`: `DomainError`. `vertices=(scope, column)` adds that scope's selected present keys (isolated vertices).
- **Dictionary**: the distinct keys in increasing order; a vertex's dense id is its rank.
- **Declared semantics**: directed edges tail = src, head = dst. Undirected: an edge appears in both directions and a
  **self-loop once** (A[u][u] = its multiplicity); degrees are row sums, so the handshake degree is
  `total + loops` (networkx's MultiGraph degree, tested). Multi-edges are multiplicities (one canonical edge per
  distinct (tail, head, w)); `dedup=True` keeps one edge per (tail, head) with the weights combined by `combine`
  (`refuse`: different weights are a `DomainError`; `min`, `max`, `sum` checked). `loops="drop"` removes loop rows.
- **Forms**: CSR (out-entries by (head, w)), CSC (in-entries by (tail, w)), the canonical edge list (the incidence
  form, sorted by (tail, head, w)) and the vertex-incidence lists (edge id, role tail / head / loop) of the oriented
  incidence matrix (B[head][e] = +1, B[tail][e] = -1, a loop column zero).
- **Identity**: the declaration, the edge scope's exact FND-004 descriptor (table identity, **snapshot revision**,
  selection, column specs), the vertex scope's descriptor and the **dictionary** (`G.same`, byte for byte). A
  `VertexSet` or `Vector` of another identity is a `DomainError` (another dictionary, declaration, selection or
  revision); after the table changes (`t.flip`), every computation on the layout and its states raises
  `StaleStateError`; a rebuilt layout has a new identity even with the same dictionary.

## Reductions (GRAPH-001)

`forward` aggregates in-entries (y = A^T x: expansion along the edges); `backward` out-entries (y = A x). Weights:
`edge` (the entry weight, with its multiplicity), `one` (the semiring's one per parallel edge), `distinct` (one per
distinct neighbor). A NULL input contributes no term; `.valid` says a term exists (empty sums: 0, false, +inf, -inf).

| semiring | values | exactness |
|---|---|---|
| `count` (+, x) | ints (Fractions when weights carry D) | exact int128; every product and sum checked (`OverflowError` naming the vertex) |
| `bool` (or, and) | bools | exact; weights and multiplicities ignored |
| `gf2` (xor, and) | bools | (m w) mod 2 (integral weights only); equals FIELD-001's `gf2_matrix().matvec` |
| `min_plus`, `max_plus` | exact multiples of 1/D, None | int64 sums checked; idempotent (multiplicities irrelevant) |
| `float` (+, x) | float64 + `.error` | fma accumulation from 0 in the stored entry order of c_e = RN(RN(m w)/RN(D)); `.error` = RU(gamma(k+4) RU(sum abs(c x))) >= abs(y - exact) |

Degrees (out / in / total / loops) and aggregates are the same products. `expand` is one Boolean step minus an
exclusion set.

## Fixed points, paths, energies (GRAPH-002)

- **Reach / closure**: R* (walks of length >= 0) or R+ (>= 1) of a source set with BFS levels; `rounds` = expansion
  steps to the fixed point (the last finds nothing); `max_rounds` (default n + 1, always enough) exceeded:
  `NotConvergedError`. `closure`: one reach set per source (a k x n FIELD-001 Matrix), 64 sources per lane word.
- **Walk counts**: w_0 = 1_S, w_(l+1)[v] = sum over entries u -> v of m w_l[u] (multiplicities count, weights do
  not): `rows[l][v]`, `totals[v]` (l <= L), `lengths[l]`, `grand`: the walk generating functions truncated at
  z^(L+1), exact Python ints computed in fixed multiword limbs sized by the **declared bound** (default 2^63 - 1,
  at most 2048 bits). Every coefficient <= bound, else `OverflowError` with `.kind` (row, total, length, grand),
  `.length`, `.vertex` (the first in that order). `length=None`: walks of every length, finite only when no cycle is
  reachable from S (Kahn's order on the reachable part; then L = the longest walk); a reachable cycle (a self-loop
  included) is `NoSolutionError`, never a finite answer.
- **Paths**: `min_plus` (shortest) or `max_plus` (longest) walk weights in exact units of 1/D. Bellman-Ford in
  synchronous rounds (only vertices with a changed in-neighbor are evaluated); a change in round n proves a
  reachable cycle of negative (positive) weight: `UnboundedError` with `.cycle` (keys in edge order), `.weight`,
  `.witness`; zero-weight cycles converge. Dijkstra (CPU) only for `min_plus` with every weight >= 0 (else
  `DomainError`); `auto` chooses it then. Both give the same `distance`, `hops` (fewest edges among optimal walks)
  and `pred` (the smallest tight predecessor one hop closer).
- **Dirichlet energy**: sum over canonical edges of m w (x_head - x_tail)^2 = x^T L x (L of the underlying undirected
  multigraph; loops add 0), exact 320-bit, an int or Fraction.
- **PageRank** (binary64): transitions m w / W_u (weights >= 0, else `DomainError`), dangling mass and teleport to a
  uniform set, damping d in (0, 1]; fixed blocked sums (blocks of 4096 in index order); stops at the first change
  `delta <= tol`, else `NotConvergedError` at `max_iter` (no vector returned). `bound` is rigorous (outward-rounded
  intervals over the exact rationals of G): >= ||rank - exact||_1, `inf` when d = 1. **Power iteration**: x <-
  A^T x / ||A^T x||_1 (weights >= 0), `eigenvalue` the norm, `residual_l1` / `residual_l2` rigorous upper bounds,
  `radius` (undirected: an eigenvalue lies within it); a vanishing iterate is `NoSolutionError`. The tests replay
  both iterations with `math.fma` bit for bit and check the bound against exact Fraction PageRank.

## CUDA

`backend="cuda"` (results say `cuda` or `cuda-emulated`): degrees and products one work item per vertex (the shared
pull, one output slot; Boolean / GF(2) words by atomic OR, overflow by atomic minimum; float coefficients computed on
the host); expansion one work item per frontier word (atomic OR); reach, closure, walk counts (one launch sequence per
step), Bellman-Ford: candidates by atomic touch counters (the set, not the list order, is used), the shared pull per
candidate into its own slot, then the commit; walk length totals and the energy by exact multiword atomic adds;
PageRank / power iteration: blocked sums one work item per block, totals by one work item, the host reads delta. All
results and statistics equal the CPU's bit for bit (scrambled emulation in the tests). The certificates and the
walk horizon run on the host in both libraries.

## Budgets and refusals

| request | checked before allocation |
|---|---|
| layout | variables, support = edge rows kept, states = 2 x kept + vertex rows, output = the entry bound, bytes (every allocation of the build) |
| degrees / expand / spmv / energy | tensor = entries visited (energy: edges), output = n (1), bytes = scratch (FLOAT coefficients); CUDA: device copies |
| gf2 matrix | tensor = output = n^2 |
| reach / closure | iterations = max_rounds, output = n (closure: tensor = output = k n), bytes = lists, bitsets (lanes) |
| walks | degree = iterations = L, output = the coefficients returned, bytes = the step vectors, lists, kept rows (unbounded: the horizon scratch first) |
| paths | iterations = n, output = n, bytes = the round state (Dijkstra: the heap) |
| pagerank / power | iterations = max_iter, tensor = entries, output = n, bytes |

Refusals: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError`, `DomainError`, `NoSolutionError`,
`UnboundedError`, `NotConvergedError`, `StaleStateError`, `UnsupportedError` (a grouped scope, more than 2^32 - 2
vertices or 2^32 - 1 entries, a bound past 2048 bits, Dijkstra on CUDA), `ValueError` / `TypeError` (malformed input,
a value not in the weight unit).

## Limitations

The layout build (dictionary sort, CSR / CSC, incidence) and Dijkstra are CPU only; device entries copy the forms
they read on every call (no device-resident layout); keys must be exact-valued columns (no strings); the float
products bound the error a priori (no enclosure of the iterates); PageRank / power bounds are a posteriori; walk
counts ignore weights (weighted path sums are the count / tropical products); no FND-007 claim kind states graph
facts: a refused cycle carries its own verifiable witness and paths their predecessor tree. CPU paths are
single-threaded.
