# Bounded multivariate parameter regions (ENV-002)

`torbles/math_region.h` (every contract; the lookup functions shared with CUDA), `math_region.c`,
`math_region_gpu.cu`, `math_region.py` (opt-in). For one function per group, affine in a parameter vector theta of
dimension d = 1..4, the exact regions of a declared box or polytope where each group attains the minimum (or the
maximum): convex polytopes with their vertices, irredundant inequalities, LP evidence and ties, and point location of
many parameter values on the CPU or the CUDA kernel. The univariate case with quadratics and algebraic breakpoints is
ENV-001 ([env.md](env.md)).

```python
from fractions import Fraction
from torbles import math_region as mr
R = mr.regions([(0, 1, 1), (1, -1, 0), None, (2, 0, -1)], box=[(-2, 2), (-2, 2)], rows=[((1, 1), 3)],
               upper=False, tie="smallest")
R.kind, R.claim, R.info    # "finite" / "empty"; "exact" / "approximate"; groups, classes, regions, vertices, incidences,
                           # boundaries, nodes, leaves, lps, pivots, clips, iterations, bytes, ...
R.classes                  # [{"coefficients", "groups", "winner", "region", "cell", "degree", "eps", "exact"}]
R.regions                  # [{"cls", "groups", "winner", "dimension", "full", "vertices", "inequalities",
                           #   "equalities" ({"normal", "rhs", "sources"}), "lp" ({"rows", "sign", "point", "margin"})}]
R.vertices                 # [{"point", "classes", "groups", "winner", "walls", "cell"}]
R.boundaries               # [{"kind" ("wall" / "cell" / "inner"), "normal", "rhs", "vertices", "classes", "groups", ...}]
R.locate([(0, 0), (Fraction(1, 3), 1)], mode="index", backend="cuda", ties=8)   # one dict per value
R.locate_arrays(mr.Values(numerators, 100), mode="index")   # int64 arrays in, int64 arrays out (-1: none)
R.evaluate_all(points)     # mode "all": every class evaluated (the verification path)
R.decide(), R.authorize_exact()                 # FND-007: an approximate result is refused as exact
mr.regions(quadratics, box, contract=mr.PiecewiseLinear(pieces=4, error_budget=Fraction(1, 10)))   # approximate
mr.from_expressions([e0, e1], ["a", "b"], box)  # first-slice polynomials (RESP-001 functions)
mr.need(functions, box); R.locate_need(points, mode, backend, ties)
```

## Domain

- Functions: per group `(a0, a1, .., ad)` for `a0 + sum_k a_k theta_k` with exact coefficients (int, Fraction,
  Decimal; floats are `TypeError`), or `None`: an empty group (absent, never attaining anything). With the
  coefficients `h_kl` of `theta_k theta_l` (k <= l, lexicographic) appended, a nonzero one is `UnsupportedError`
  unless the piecewise-linear contract is declared.
- Parameter set P: the closed box `lo_k <= theta_k <= hi_k` (finite rationals, `lo_k < hi_k`) and at most 64 rows
  `A theta <= b` (rational); P must have
  a nonempty interior (an exact LP decides it: `InfeasibleError` when P is empty, `UnsupportedError` when it has no
  interior).
- Classes: identical functions (equal canonical forms `(A_0 + sum A_k theta_k) / D`, gcd 1, D > 0) are one class,
  coincident everywhere; ordered by their smallest member.
- Regions: `R_c = {theta in P : f_c <= f_c'}` for every class c', empty, of full dimension or lower (a class that wins
  only on a face). The nonempty regions cover P and form a polyhedral complex. Each comes with its vertices (exact),
  an irredundant description by primitive integer inequalities `n theta <= rhs` (full dimension: one per facet, with
  its sources: box faces, rows and the classes tied with it on the facet; lower dimension: a basis of its implicit
  equalities plus one inequality per relative facet) and the exact LP evidence `s* = max s : n_j theta + s <= c_j`
  (OPT-002's exact simplex) with its point: s* > 0 an interior point, s* = 0 nonempty without interior.
- Ties: a point where several classes attain the minimum. The tie policy (`"smallest"` / `"largest"` member group)
  names the winner; every answer also reports every tied class. Vertices carry their tied classes (the regions having
  them as a vertex), boundaries the classes tied on the whole facet. `upper=True` takes the maximum.

## Construction and exactness

1. A kd-tree on a dyadic lattice (2^20 steps per axis) keeps per node the classes whose minimum over the node's box is
   at most the smallest maximum (exact corner values), dropping a class whose difference with one of two reference
   classes (the smallest maximum, the smallest at the box's midpoint) is positive on the whole box; a node is a leaf
   at 8 candidates or fewer, at the lattice resolution, or when d + 1 consecutive splits removed none.
2. For each candidate class c, its neighbors N(c) (co-candidates of its leaves) give `Q_c = {theta in P: f_c <= f_c',
   c' in N(c)}`; when R_c is nonempty, Q_c = R_c (a segment leaving R_c inside Q_c would cross a tie with a
   co-candidate), so one point of Q_c decides R_c exactly.
3. The vertices of Q_c by the double description method in homogeneous integer coordinates (FND-009 big integers),
   adjacency by the combinatorial test; facets, equalities and dimension from exact ranks; vertices deduplicated.

Lookups: theta = (u_k / v_k) becomes the homogeneous integer point (x, w); every comparison is the sign of an integer
form computed modulo k primes in (2^61, 2^62) (INV-001's verified table) with Garner's mixed-radix digits, k chosen
from a proven bound (`R.last_lookup`: `primes`, `bound_bits`). Modes `index` (the kd-tree leaf's candidates), `regions`
(every region's inequalities) and `all` (every class) give the same answers; the CPU and the CUDA kernel
(`tr_mgpu_region_locate`, one work item per value, its own output slots) run the same inline function and agree bit
for bit. Values whose parts fit int64 cross as int64 arrays (`Values`: no per-value conversion), others as big
rationals.

## The piecewise-linear contract (approximate, labelled)

`PiecewiseLinear(pieces=K, error_budget=E)` divides the box into K^d equal cells and replaces each quadratic q on a
cell by the affine `L = q(m) + grad q(m) (theta - m) + (lo + hi) / 2`, `[lo, hi]` an exact enclosure of the quadratic
remainder: `|q - L| <= eps` exactly. The largest eps must not exceed E (else `NotConvergedError`, never a weaker
bound reported). The result is labelled `claim = "approximate"` with `error` (the largest eps) and `envelope_error`
(2 eps: the winner's true value is within it of the true envelope at every theta); `decide()` / `authorize_exact()`
refuse an exact use (FND-007 reason APPROXIMATE). Without the contract quadratics are refused.

## Budgets and refusals

- Build, before allocation: variables = d, degree (1, or 2 under the contract), states = cells x groups, bytes = the
  function tables.
- Build, during construction (each counted exactly; refused at limit + 1 with that need, nothing partial or
  truncated): output = nonempty regions, support = region-vertex incidences, circuit = index nodes, iterations =
  index nodes + clipping steps + LP pivots, bytes = the tables as they grow.
- Lookup: output = values, bytes = the job (CUDA: plus its device copies, checked together before launching).

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `UnsupportedError` (d > 4, quadratics without the contract,
more than 64 rows, P without interior, a region LP above 255 rows whose enumeration is empty); `InfeasibleError`
(P empty); `NotConvergedError` (the contract's error above its budget); `OverflowError` (lookup values past 64 moduli,
about 3,900 bits); `ValueError` (malformed input, `lo >= hi`, a denominator below 1); `TypeError` (floats). A value
outside P answers `"outside"`, any value with no present group `"empty"`.

**Limitations.** Affine functions only (quadratics only as the labelled approximation; no exact curved regions);
d <= 4. The build runs on the CPU (only the lookups have a kernel) and its cost grows with the regions, their vertices
and their neighbors: measured on the 2-core development machine, 1,024 tangent planes in d = 2 (every group a region)
build in about 11 s, 256 in d = 3 in about 11 s, 64 in d = 4 in about 6 s, while 1,024 random affine functions (few
regions) take under 2 s in d = 2 and 3; work past the iteration budget is refused, never truncated. The approximate
contract's cells multiply the work by K^d.
