# Bounded explanation and repair search (ATTR-002)

`torbles/math_repair.h` (every contract; the evaluators, bounds and search drivers are inline and shared by the CPU and
the CUDA library), `math_repair.c`, `math_repair_gpu.cu`, `math_repair.py` (opt-in). Redundant predicates, minimal
distinguishing sets and least-cost feasible repairs as minimum-cost subset problems over at most 62 elements
(conditions, players or rows), solved exactly when the search finishes and reported **incomplete with proven bounds**
when the declared state budget stops it. Continuous repairs (value edits, threshold changes) go through INV-002
([interv.md](interv.md)). Every result is algebraic: `causal` is False.

```python
from torbles import math_bool as mb, math_repair as mr
masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 10), mb.condition("y", "<=", 40), ...])
P = mr.Problem.keep(masks, [0, 1, 2])              # AND over S selects exactly what AND over C selects
P = mr.Problem.program(masks, prog, baseline=True)  # a BOOL-001 program, conditions outside S at their baselines
P = mr.Problem.separate(masks, rows_a, ("group", 1), convention="values" | "selected")
P = mr.Problem.explain(masks, prog)               # the condition sets on which the program's membership depends
P = mr.Problem.game(game, "explanation" | "repair", ">=", 10)   # an ATTR-001 game: v(S) op t or v(P - S) op t
P = mr.Problem.conjunction(hist, [0, 1, 2])       # LATTICE-002: v(S) = v(C) over a histogram
P = mr.Problem.delete(masks, rows, aggregate="sum", value="v", op="<=", target=100, predicate=prog)
P = mr.Problem.masks(n, [[0, 1], [2, 3]], allowed=[0, 1, 2])   # declared hitting-set masks
P.info, P.minimal_masks(), P.evaluate(subsets, backend="cuda"), P.device_need(count)
r = mr.search(P, cost=[1, Fraction(3, 2), ...], method="auto" | "exhaustive" | "branch_bound",
              order="index" | "cost" | "shapley" | "banzhaf", game=g, max_states=10_000, max_solutions=64,
              backend="cpu" | "cuda")
r["status"], r["minimum"], r["solutions"], r["masks"], r["multiplicity"]          # "found": exact, every tie counted
r["lower"], r["upper"], r["multiplicity_bounds"], r["explored"], r["open"]         # "incomplete": proven bounds
mr.redundant(P, witnesses=8, backend="cuda")      # per element: removing it changes nothing? exact counts, witness rows
mr.edit(masks, rows, "v", "sum", ">=", 100, cost="l1", weights=w, d_bounds=(None, 5), predicate=prog)   # INV-002
mr.threshold(scope, conditions, which=[0, 1], aggregate="count(*)", op="<=", target=5, weights=[1, 2])
```

## Problems (one immutable native object; S ranges over subsets of the allowed elements)

| constructor | elements | S is feasible when | count reported |
|---|---|---|---|
| `masks` | declared | S meets every declared mask (only the inclusion-minimal ones are kept) | masks missed |
| `separate` | conditions | every pair (a in A, b in B) differs on a condition of S: three-valued values (`"values"`: TRUE / FALSE / UNKNOWN differ) or SQL selection (`"selected"`: TRUE against not TRUE); the minimal difference sets of the distinct patterns | masks missed |
| `explain` | conditions | `separate` with A = the points the program selects (TRUE), B = the other points: membership is a function of S | masks missed |
| `keep` | the conjunction's conditions | AND over S selects exactly what AND over C selects (it meets the failure set of every rejected point): the conditions outside a minimum S are jointly redundant | masks missed |
| `program` | the conditions the program reads (default) | the program with every condition outside S at its baseline constant selects exactly the same points (SQL three-valued logic) | points where they differ |
| `game`, `conjunction` | players | `v(S) op t` (explanation) or `v(P - S) op t` (repair), exact 128-bit values | 0 / 1 |
| `delete` | declared deletable rows (<= 62, selected) | AGG over the qualifying rows that remain op t; AGG = COUNT(*), COUNT(v), SUM(v), AVG(v) with SQL semantics (a NULL is not counted; SUM / AVG of no value is NULL and attains no target); qualifying = selected and TRUE under the predicate program | rows remaining |

A separation with two indistinguishable rows is reported infeasible with the witness pair (`info["witness"]`); its
searches answer `"none"` (proven).

## Searches

Costs are positive exact rationals per element (unit by default), scaled natively to integers over their common
denominator (the scaled total must stay below 2^62: `OverflowError`). The answer is the minimum cost over the
feasible subsets, every minimum-cost subset (listed up to `max_solutions`, counted exactly) and its cost.

- `exhaustive`: every subset in increasing (cost, mask) order (unit costs: levels |S| = 0, 1, ... in increasing mask
  order; weighted: the sorted order of the 2^m subsets, m <= 20), evaluated in batches; the first feasible rank is the
  minimum, the rest of its cost level gives every tie.
- `branch_bound`: best-first over include / exclude decisions in the declared element order, nodes ordered by (lower
  bound, I, X). Bounds are admissible: hitting sets, a greedy packing of the masks I misses that are disjoint on the
  undecided elements (each costs at least its cheapest element); deletions under an order comparison, the fractional
  (LP) relaxation of the one linear constraint, rounded up exactly; otherwise the cheapest undecided element when I is
  infeasible. Pruning is sound only: a bound above the incumbent (ties kept), a monotone problem whose completion is
  infeasible, an infeasible relaxation. A feasible I is a solution and is not expanded (positive costs).
- `auto`: exhaustive when 2^m fits `max_states` and m <= 20, else branch and bound.
- Orders (branch and bound): `index`, `cost` (cheapest first), `shapley` / `banzhaf`: decreasing exact ATTR-001
  index of `game` (or the problem's own game) with ties by index, computed by `tr_attr_exhaustive` up to 20 players
  and from the sparse support beyond. The order changes the work, never the answer.

Every evaluated subset is a state and counts against `max_states` (default: the budget's iterations limit). Results:
`"found"` (minimum, multiplicity exact, `exact: True`), `"none"` (proven: every subset examined or soundly pruned),
or `"incomplete"` when `max_states` stopped the search: `lower <= minimum <= upper`, both proven (`upper` is the cost of
a feasible subset found, else None), `explored` and `open` nodes counted, `minimum` None and `exact` False. When
`lower == upper` the minimum is known and `multiplicity_bounds = (ties found, at most)` bounds its multiplicity
(exhaustive: plus the subsets of that cost not examined; branch and bound: plus every subset below an open node at
that bound); otherwise `(0, None)`.

`redundant(P)`: per allowed element i, whether S = allowed - {i} is feasible, with the exact count; for a program, the
points where the selections differ split into gained (selected only without condition i) and lost, with the first
`witnesses` rows of each in increasing order. The CUDA path re-derives the certifying counts on the kernel.

## Continuous repairs (INV-002, CPU)

- `edit`: value edits d_r of the declared rows' present values so that SUM(v) or AVG(v) over the qualifying rows
  attains `op target` (qualification does not read v): INV-002's exact LP (`l1`: sum w |d|, `linf`: max w |d|) or
  projection (`l2`: sum w d^2), `l0` supports, optional bounds on each d. A `math_interv.Result`: the cheapest change,
  every tie, `"infeasible"` when the bounds cannot reach the target.
- `threshold`: the constant c of one declared comparison condition of a conjunction changed to c' (the others fixed)
  so that AGG over the rows then selected attains the target: INV-002's aggregate state (rows entering and leaving the
  predicate decided exactly), cost w |c' - c|, the new constants computed natively; over several declared conditions
  the minimum and every condition attaining it.

## CUDA

`tr_mgpu_repair_eval`: one work item per candidate subset writes its exact counts and value into its own slot;
`tr_mgpu_repair_scan`: per candidate, the feasible bitmap by atomic OR and the smallest feasible cost by atomic
minimum, then the ties of that cost by atomic OR; `tr_mgpu_repair_search` runs the shared drivers with them. Results
do not depend on thread order and equal the CPU bit for bit (a program above 4,096 nodes is refused on the device:
`UnsupportedError`).

## Budgets and refusals

- Problems: variables = elements; tensor = pattern pairs (separate / explain); support = the masks kept (declared
  masks: the copy, before minimization); circuit = program nodes; bytes = the copies.
- Evaluation: output = candidates, bytes (CUDA: the device copies and the scratch).
- Search: iterations = `max_states` (a larger space gives `"incomplete"`, never a refusal), output = `max_solutions`,
  bytes = the frontier (2 max_states nodes), the batches and the sorted order (16 x 2^m); Shapley orders add ATTR-001's
  needs. Redundancy: output = elements + elements x witnesses. Thresholds: variables = conditions, output = changeable
  conditions.

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `UnsupportedError` (more than 62 elements, a weighted
exhaustive search above 20 allowed elements); `DomainError` (a program or an order game over another number of
conditions / players); `ValueError` (malformed input, non-positive costs, rows outside the selection);
`OverflowError` (costs or aggregates outside their exact carriers); `StaleStateError` (the table changed since the
masks were built).

**Limitations.** Subsets are 64-bit masks (at most 62 elements); the search is exponential in the worst case, which is
why every search carries a state budget and reports proven bounds when it stops. Bounds are specific to hitting sets
and linear deletion constraints; other problems (circuits, games, `!=` targets) branch with the generic bound and rely
on the incumbent. Continuous repairs run on the CPU only (INV-002) and change one threshold at a time. No result is a
causal statement.
