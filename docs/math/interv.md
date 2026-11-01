# Constrained minimal interventions (INV-002)

`torbles/math_interv.h` (every contract; the per-point grid evaluation shared with CUDA), `math_interv.c`,
`math_interv_gpu.cu`, `math_interv.py` (opt-in). The cheapest change d of declared inputs (x = x0 + d) that makes a
function attain a target, exactly, under linear constraints and boxes, with every tie of the minimal cost reported
and budget-limited searches labelled incomplete with proven bounds.

```python
from torbles import math_interv as iv
r = iv.linear(2*t + u, {"t": 1, "u": 0}, ">=", 7, cost="l1", d_bounds={"u": (0, None)})   # an exact LP
r = iv.linear(([2, 1], 0), [1, 0], "=", 7, cost="l2", weights=[1, 3], A_ub=[[1, 1]], b_ub=[5])  # exact projection
r = iv.linear(f, x0, ">=", 0, flip=True, cost="l0")       # a membership change: every minimal support
r = iv.univariate(t**3 - 2*t, {"t": 1}, "=", 0)           # INV-001 roots: exact or isolating intervals
g = iv.Grid(t*t + u, {"t": 0, "u": 0}, {"t": range(-4, 5), "u": range(-4, 5)}, "=", 8, cost="l1",
            constraints=[(t - u, "<=", 3)], A_ub=[[1, 1]], b_ub=[6])
r = g.search(backend="cpu" | "cuda", budget=Budget(iterations=10**5), start=0)   # every tie, or proven bounds
st = iv.Aggregate(scope, "x", "v")                        # rows cross the predicate when x is shifted
r = st.minimal("sum", (">=", 10), (">=", 500), cost="l1") # min |d|: SUM(v) WHERE x + d >= 10 attains >= 500
st.value("avg", (">=", 10), d)                            # (the aggregate at the shift d exactly, rows in it)
r.status, r.cost, r.lower, r.upper, r.unique, r.solutions[0].change, r.claim, r.causal, r.statement
```

## What an answer claims

Every result is an **algebraic intervention** on the function as given: the cheapest change of the declared inputs
that makes the function attain the target. There is no causal model and no causal claim: nothing is said about what
would happen if the inputs were changed in the world, about which inputs can be changed, or about effects on other
quantities. `Result.claim == "algebraic"`, `Result.causal is False`, `Result.statement` spells this out, and the
native results carry `TR_INTERV_CLAIM_ALGEBRAIC`; `semantics="causal"` is refused (`UnsupportedError`) by every
entry point.

## Costs, targets, answers

- Costs of d with rational weights w_i >= 0 (`l0`: w_i > 0): `l1` sum w_i |d_i|, `linf` max w_i |d_i|, `l2` the
  squared sum w_i d_i^2, `l0` the weighted number of changed inputs.
- Targets f(x0 + d) op t with op in `= != < <= > >=`; `flip=True` targets the complement of the predicate's value
  at x0 (a membership change; `base_member` reports that value, decided exactly).
- Status `optimal` (every minimal change listed, `unique` decided exactly where stated), `unattained` (a strict
  target: the exact infimum and its boundary change, `attained=False`), `infeasible` (no change in the declared
  domain), `incomplete` (a budget stopped the search: `lower <= minimum <= upper`, proven; `upper` None when
  nothing feasible was found; `reason` "iterations" or "output"; `next` resumes a grid).

## Linear targets: exact LPs and projections (n <= 16)

f = a^T x + a0 (coefficients, or a first-slice Expression of degree <= 1, e.g. a RESP-001 function), constraints
A_ub x <= b_ub, A_eq x = b_eq on the point, boxes on x and on d (either side optional), ||d||_1 <= l1,
||d||_inf <= linf. `l1` and `linf` are exact LPs (Bland's simplex of OPT-002, `math_polyopt.h`: variables (d, u) with
u >= +-d, or (d, s) with +-w_i d_i <= s); `l2` is the exact projection QP (H = 2 diag(w)) by OPT-002's active-set
method, KKT-verified exactly (the tests cross-check it against OPT-001's box minimizer after eliminating a free
coordinate); `l0` enumerates the supports in increasing weight, an exact feasibility LP each, and reports every
support of the minimal weight with a witness. A strict target is solved on its closure: the strict set must be
nonempty (an LP) and the infimum is attained iff an optimal point satisfies it strictly (an LP over the optimal set);
`!=` is the better of `<` and `>`. Uniqueness of the optimal change: 2n LPs (min / max of each d_k over the optimal
set); a nonunique answer carries a second optimal change (`face`).

## Univariate polynomial targets (INV-001)

One input; f a first-slice Expression or coefficients. The roots of f - t inside the domain (d_bounds, x_bounds)
come from INV-001: exact rational roots, otherwise certified open isolating intervals (refined to `width` when
given). The feasible set is a union of points and open intervals between consecutive roots, each decided by an exact
sign; the nearest feasible point on each side of x0 is found and the two are compared exactly. A tie between two
irrational distances is decided exactly (the gcd of (f - t)(x0 + y) and (f - t)(x0 - y) has a root in both
intervals). An irrational change is reported as its isolating interval (`exact=False`, `how="irrational"`, with its
multiplicity).

## Finite grids: exhaustive search (CPU or CUDA)

1..8 coordinates with explicit rational values (at most 2^24 points), any first-slice polynomial target (dict
`{exponents: coefficient}` or Expression), polynomial constraints `(g, "<=" | ">=" | "=", rhs)` and linear rows
A_ub x <= b_ub on the point: at most 16 conditions with the target.

- The plan (`Grid`): every condition becomes an integer polynomial in the cleared numerators of the values; its sign
  at a point is decided exactly from its residues modulo k verified primes of INV-001 (k from a coefficient bound,
  `bound_bits`) and Garner's mixed-radix digits. Costs become integers over a common denominator D (overflow is
  `OverflowError`). Blocks of the grid (at most 4096) whose outward-rounded interval enclosure (FND-009, UNC-001
  style) proves some condition impossible are pruned soundly (`pruned`, `blocks_pruned`); the other points are
  sorted by (exact cost, grid index): `Grid.order()`.
- The search scans the order in chunks (4096 doubling to 2^20). The evaluator (the CPU `tr_interv_grid_eval` or the
  CUDA `tr_mgpu_interv_grid`: one work item per point, feasibility and tie bits by atomic OR, the minimal cost by
  atomic minimum, all independent of thread order) returns bit-identical bitmaps. The first feasible rank gives the
  minimum; every feasible point of that cost is a tie and is listed (grid `index`, exact change, point and value).
- The iteration budget counts points examined: past it the answer is `incomplete` with `lower` = the cost of the
  first unexamined rank (every cheaper point was examined or pruned) and `upper` = the best found (or None). The
  output budget limits the listed ties: `incomplete`, reason "output", lower = upper = the minimum, `next` = the
  first unlisted tie. `search(start=next)` continues the scan.

## Aggregates whose rows cross the predicate

`Aggregate(scope, x, v)` builds once, over the selected rows of an FND-004 scope (int64 / decimal columns), the
sorted distinct values of x with row counts and exact sums of v and of x (prefix sums). A(d) = COUNT(*), COUNT(v),
SUM(v) or AVG(v) of the rows with x + d op c (v may be x itself, then shifted too) is piecewise constant or linear in
d with the exact breakpoints c - x_k; `minimal` decides every piece exactly, so rows entering and leaving the
predicate are part of the objective (`entered`, `left`, `rows` per solution), never the frozen row set of d = 0. SQL
semantics: a NULL x never qualifies, a NULL v is not counted, SUM / AVG over no value is NULL and attains no target.
The tests compare `value` and `minimal` with DuckDB queries on the shifted column.

## Budgets and refusals

linear: variables = n, tensor = the largest LP tableau, iterations = pivots (and supports for `l0`); univariate:
degree, iterations (INV-001 evaluations); grid: variables = coordinates, support = terms, states = grid points,
bytes = the plan (order and residues), iterations = points examined and output = ties listed (both truncate:
`incomplete`), and the CUDA evaluator budgets its device copies (`Grid.device_bytes`); aggregate: states = distinct
values, bytes = the state. Every check happens before allocation. Refusals: `AlgebraBudgetError`, `UnsupportedError`
(n > 16, a grid of more than 8 coordinates, 16 conditions or 2^24 points, a linear target of degree > 1, causal
semantics), `OverflowError`, `ValueError` / `TypeError` (malformed or float input), `DomainError`.

## Limitations

Linear targets are linear in the changed inputs (a polynomial target in several inputs is searched only over a finite
grid); the `l0` enumeration is exponential in n (2^n supports, budgeted); grids are explicit and capped at 2^24
points; aggregates shift one feature and support COUNT / SUM / AVG with one comparison predicate. Convex constraints
beyond linear rows and norm balls (a general convex set) are not supported. The LPs, QPs, roots and aggregates run
on the CPU; only the grid evaluation has CUDA kernels, tested here against the CPU in the emulation build.
