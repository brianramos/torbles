# Factorized provenance payloads (PROV-001)

`torbles/math_prov.h` (every contract; the B / N evaluators, the valuation and plan checks and the device sizes are
inline and shared with CUDA), `math_prov.c`, `math_prov_gpu.cu`, `math_prov.py` (opt-in). Provenance annotations for
small relational queries: every output tuple carries an expression over source-tuple variables in one shared,
hash-consed DAG, interpreted in a declared semiring.

```python
from torbles import math_prov as mp
st = mp.Store(budget=Budget(circuit=1 << 16, support=1 << 18))  # annotation limits: nodes, child entries, bytes
R = st.source("R", {"k": [1, 1, 2, None], "a": [10, 20, 30, 40]}, scales={"a": 0}, rows=None)   # ("R", i)
S = st.source_scope("S", scope, ["k", "b"])               # a FND-004 scope's selected rows (labels: physical rows)
J = R.join(S, on=[("k", "k")])                            # columns k, a, k_r, b
F = R.where(("a", ">", 15), ("k", "is not null"), ("a", "<>", ("col", "k")))
U = R.union(R2); G = J.group(["a"]); T = R.group([])      # GROUP BY a; the global group
G.tuples(); G.annotations(); G.info                       # values (None = NULL), DAG nodes, new_nodes, global_group
G.polynomial(0)            # Polynomial (semiring "N[X]"): {(("R", 0), ("S", 2)): 1, ...}; .derivations, .degree
G.witnesses(0)             # Witnesses (semiring "B"): minimal frozensets of source labels that suffice
G.counts(deletions=[("R", 0)]); G.exists(deletions=[("S", 1)]); G.counts(valuation={("R", 2): 3})
a = st.evaluate([G, J], "count", deletions=[[("R", 0)], [("S", 1), ("S", 2)]], backend="cuda")
a.semiring, a.values, a.status, a.of(0, 1)     # "N"; per valuation the outputs of G then J; overflow per valuation
pl = st.plan([G, J]); st.evaluate([G, J], "bool", valuations=[{("R", 1): 0}], default=1, plan=pl)
J.need("join", S, on=[("k", "k")]); J.expand_need(0); st.info; st.parents(node); mp.monomial(("R", 0), ("R", 0))
```

## The DAG

- Nodes: ZERO (0), ONE (1), VAR (one per source row), PLUS and TIMES (n >= 2 children). Hash-consed: one node per
  operation and multiset of children (children sorted, repeats kept: x * x = x^2, a + a = 2a), so a derivation used
  by many outputs is one node; PLUS drops ZERO, TIMES drops ONE and is ZERO with a ZERO child, one child is that
  child; nothing is flattened (shared nodes stay shared). `st.info`: `nodes`, `children`, `shared` (nodes with at
  least two distinct parents), `hits` (requests answered by an existing node), `bytes`, the limits.
- The store is append-only and owned by one caller; relations and plans are immutable.

## Relations and operators (K-relations)

Distinct tuples (NULL equals NULL, ordered ascending, NULL last), each with its annotation:

| operator | tuples | annotation |
|---|---|---|
| `source` | the selected rows, equal rows merged | PLUS of their variables |
| `where` | tuples where every predicate is TRUE (a NULL comparison is UNKNOWN: dropped; `is [not] null` never is) | kept |
| `union` | UNION ALL, equal tuples merged | PLUS(a, b) |
| `join` | matching pairs on equal keys (NULL never matches; no keys: the Cartesian product); left then right columns | TIMES(a, b) |
| `group` | one tuple per group present in the input (NULL keys: their own group); `[]`: the global group, always present | PLUS of the members (ZERO for an empty global group) |

Presence is explicit: a keyed group exists only if derived from a tuple (under a valuation: iff its B value is 1);
the global group exists always (`info["global_group"]`), its N value can be 0 (SQL `COUNT(*)` of nothing). NULL
validity, multiplicity (the annotation) and group presence are separate.

## Interpretations (each result names its semiring)

| semiring | value | API |
|---|---|---|
| `B` | existence under nu: X -> {0, 1} (which sources are present) | `exists`, `evaluate(..., "bool")`, `witnesses` |
| `N` | the number of derivations under nu: X -> N (multiplicities; a deletion is 0), exact unsigned 128 bits | `counts`, `evaluate(..., "count")` |
| `N[X]` | the expanded derivation polynomial (monomials with natural coefficients) | `polynomial` |

N[X] is the universal one: B and N are its images (x^k -> nu(x); Boolean x^k -> x). Under the 1-valuation N gives
the query's bag multiplicities (SQL), B its set semantics. Witnesses are the minimal variable sets of the monomials.
An N value past 128 bits sets that valuation's status to `"overflow"` (its outputs 0); the others are unaffected.

## Evaluation and CUDA

`Store.plan` snapshots the cone of the annotations (the nodes they reach, renumbered) as an immutable plan;
`tr_prov_evaluate` evaluates it per valuation (a default value and sorted overrides) with the shared
`tr_prov_eval_count` / `tr_prov_eval_bool`. `tr_mgpu_prov_evaluate`: one work item per valuation, its own scratch row
and outputs, in launch chunks whose scratch stays within 64 MiB; the host checks the plan view and the valuations
first. Both write the same bytes (`backend` says `cuda` or `cuda-emulated`).

## Budgets and refusals

| request | checked before allocation |
|---|---|
| store | its limits: circuit = nodes, support = child entries, bytes = its arrays |
| operator | its output (tuples) and bytes (result and scratch); the store after the operation's bound (source: rows VARs + duplicate PLUS; union: merged pairs; join: the matching pairs, counted first; group: the groups) |
| plan | circuit = cone nodes, support = cone children, output = annotations, bytes |
| evaluate | output = valuations x annotations, bytes = the scratch (CUDA: every device copy and the scratch chunk) |
| polynomial / witnesses | circuit = cone, support = the number of derivations D (an upper bound on the monomials), degree, bytes (witnesses: tensor = D^2) |

Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`); an annotation or an expansion is never
truncated. Also `DomainError` (columns of different decimal scales compared, joined or merged; unions of different
arities), `OverflowError` (a coefficient past 64 bits, values past int64), `UnsupportedError` (more than 64 columns,
2^32 - 2 nodes or variables), `ValueError` (malformed input, an unknown source label, a relation of another store),
`TypeError` (inexact values).

## For the next capabilities

ATTR-001 (attribution) reads the store's DAG (`tr_prov_store_nodes` / `_children`, `tr_prov_plan_build` /
`tr_prov_plan_get_view`) and evaluates coalitions as valuations (`tr_prov_evaluate`, `tr_mgpu_prov_evaluate`: a
coalition is a 0/1 valuation of the sources, many at once); the expanded polynomial (`tr_prov_expand`) and the
minimal witnesses (`tr_prov_witnesses`) give the exact game for small inputs, `tr_prov_expand_need` its size first.

## Limitations

Operators are selection, union, equi-join and group-by (with the global group); no aggregates other than the
multiplicity (no SUM / MIN over annotated values, no semimodules), no difference / negation (N[X] is positive), no
recursion. Values are logical int64 with a scale (no strings). The store is single-owner (calls serialized by the
caller). An expansion is budgeted by its number of derivations, which can exceed its distinct monomials; witnesses
are minimized pairwise (quadratic in the monomials). The CPU evaluation is single-threaded.
