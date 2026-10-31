# Occupancy histograms and subset-lattice transforms (LATTICE-001, LATTICE-002)

`torbles/math_lattice.h/.c` (native CPU), `math_lattice_gpu.cu` (CUDA kernels: the histogram scan, the butterfly
passes), `math_lattice.py` (opt-in Python API). LATTICE-001 counts, exactly, how many points of a BOOL-001 domain fall
in each cell of k Boolean axes; LATTICE-002 transforms such a 2^k vector over the subset lattice (zeta sums, Moebius
inversion, inclusion-exclusion, Moebius coefficients up to an order) in exact signed integers. The header comment of
`math_lattice.h` is the normative contract; this page summarizes it.

```python
from torbles import math_bank as mbk, math_bool as mb, math_lattice as ml
masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 5), mb.condition("d", "between", "1.5", "3.25")])
h = ml.Histogram.build(masks, nulls="apart", backend="cpu")     # or nulls="three", backend="cuda"
h = ml.Histogram.build(masks, axes=[1, 0, mb.var(0) & ~mb.var(1)])   # conditions or predicates, in this order
h = ml.Histogram.build(mb.Masks.assignments(4), axes=[prog_a, prog_b])  # the uniform measure of 2^4 assignments
h.counts(); h.cell((True, False)); h.table(); h.axis_counts(); h.info   # cells, one cell, occupied cells, per axis
h.vector("complete" | "true_sets" | "false_sets" | "unknown_sets")      # the 2^k projection (uint64)
ml.Histogram.from_counts(k, cells, nulls=..., points=..., axis_counts=...); h.merge(other)
ml.zeta(h)["values"]; ml.mobius(v)["values"]                            # direction "up" (default) or "down"
ml.transform(v, "zeta_up" | "zeta_down" | "mobius_up" | "mobius_down", carrier="i128" | "i64" | "bigint",
             backend="cpu" | "cuda")       # {"values", "k", "op", "carrier", "cost" (the needs), "backend", "device"}
ml.coefficients(v, [[0, 2], [1]], op="zeta_up")    # a few coefficients by their direct sums (no full transform)
ml.interactions(v, order=2)                        # [(coalition, Moebius coefficient)] for |T| <= 2
ml.order_sets(k, max_order, min_order)             # the coalitions: by size, then increasing mask
ml.counts(h, [[0, 1], [2]], projection="true_sets")    # all / any / none / exactly / terms / peak per set
ml.histogram_need(...), ml.transform_need(k, op, carrier), ml.coefficients_need(...), ml.counts_need(...)
```

## Histograms (LATTICE-001)

- **Axes**: condition `i` of the masks (an int), a `math_bool.Program` over the masks' k0 conditions, or a
  `math_bool.Expr` (compiled over them). Axis i is bit i of a two-valued cell id and digit i of a three-valued one: the
  given order is kept (a permutation of the axes permutes the cells). One Program object read by several axes is
  evaluated once per word. The domain is any BOOL-001 domain: the selected rows of a scope (empirical counts: each
  row once) or the 2^k0 assignments (each once).
- **NULL convention** (declared, never silent): `"apart"`: 2^k cells over the points where every axis is known; the
  points with an UNKNOWN axis are in no cell, counted apart (`unknown`) and per axis in `axis_counts`. `"three"`: 3^k
  cells, digit 0 = FALSE, 1 = TRUE, 2 = UNKNOWN; every point is in one cell. In both, points = complete + unknown and
  each axis' TRUE + FALSE + UNKNOWN = points. UNKNOWN is never counted as FALSE.
- **Projections** to 2^k vectors: `"complete"` (the two-valued cells; both conventions), and on a three-valued
  histogram `"true_sets"` (the set of TRUE axes: SQL WHERE semantics, so superset sums are `COUNT(*) WHERE c1 AND c2`),
  `"false_sets"`, `"unknown_sets"`. An "apart" histogram did not record its unknown points per cell: the other
  projections are `DomainError`.
- `from_counts` adopts given cells (and per-axis counts, or derives them), checked consistent natively; `merge` adds two
  histograms of the same shape over disjoint point sets. A histogram over table rows checks its scope's freshness
  before every use (`StaleStateError`).

## Transforms (LATTICE-002)

On a vector v over the subsets A of k axes (bit i = axis i):

| op | value at S | on a projection h |
|---|---|---|
| `zeta_up` | sum over A superset of S of v(A) | points TRUE on at least S (the intersection count) |
| `zeta_down` | sum over A subset of S of v(A) | points TRUE on nothing outside S |
| `mobius_up` | sum over A superset of S of (-1)^(abs(A - S)) v(A) | inverts `zeta_up`: the exclusive cells back |
| `mobius_down` | sum over A subset of S of (-1)^(abs(S - A)) v(A) | inverts `zeta_down`: Harsanyi dividends of a set function |

- **Fast transforms**: k butterfly passes of 2^(k-1) disjoint pairs; cost 2^k values, k passes (the `cost` dict)
  whatever the row count. **Direct coefficients** (`coefficients`, `interactions`): the same sums for requested sets
  only: 2^(k - abs(S)) terms (UP ops) or 2^abs(S) (DOWN ops); the full transform pays off only for many requests.
- **Inclusion-exclusion** (`counts`): per set S on a projection (N = its total): `all` (TRUE on every axis of S; N for
  the empty set), `any` (on at least one) as the explicit alternating sum of the intersection counts of the nonempty
  subsets of S, in the order of their size (the Bonferroni partial sums), `peak` (the largest absolute partial sum),
  `none` = N - any, `exactly` = h(S), `terms` = 2^abs(S) - 1.
- **Carriers** (signed intermediates never wrap): `"i128"` (default): 128-bit two's complement cells, every addition
  checked (`OverflowError`, nothing returned); for int64 inputs no overflow is possible (bound 2^(k+63) < 2^127).
  `"i64"`: the same, every input and output checked to fit int64 (an int64 ndarray is returned). `"bigint"`: FND-009
  arbitrary-precision integers (CPU only), exact for any Python ints. Inclusion-exclusion partial sums are 128-bit:
  they may exceed int64 while the count does not (a 2^62-point cell over 16 axes peaks above 2^66).

## CUDA

`tr_mgpu_lattice_hist`: work items are slices of 64-point words; each evaluates the axes of its words (every program
once, bitwise with a valid word), computes the cell of each point and adds one with an integer atomic, or for at most
64 cells counts in per-thread arrays added once at the end; the per-axis counts likewise. The counts are adopted with
`tr_lattice_hist_from_counts`. `tr_mgpu_lattice_transform`: one launch per butterfly pass, one work item per pair,
checked 128-bit butterflies raising an overflow flag (and an int64 check launch for `"i64"`). Results equal the CPU's
bit for bit (emulated with a scrambled thread order in the tests). Coefficients, counts and the big-integer carrier
have no CUDA path.

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| histogram | variables = k axes; states = output = cells (2^k or 3^k); circuit = program nodes; support = 2^k0 on assignments; bytes = object (8 per cell) + scratch (CUDA: device copies and scratch) |
| from counts / merge | variables = k; states = output = cells; bytes = object |
| transform | variables = k; tensor = 2^k; iterations = k; bytes = 16 x 2^k (bigint: 8 x 2^k, each number's limbs budgeted by FND-009) |
| coefficients | variables = k; tensor = the terms of the direct sums; output = requests; bytes = 16 per request |
| order sets | variables = k; output = the sets |
| counts | variables = k; degree = largest abs(S); tensor = cells + per request 2^k + m 2^m + 2^m; output = requests; bytes |

Representation limits (`UnsupportedError`): more than 30 axes (histograms, transforms), more than 2^24 requests per
call, the big-integer carrier on CUDA. Malformed input (`ValueError`): axes out of range, a vector that is not 2^k
values, inconsistent counts, sets with repeated or out-of-range axes. A program over another number of conditions than
the domain, a projection the convention did not record, merging different shapes: `DomainError`.

## Limitations

One histogram per domain (no per-group histograms; DIST-001 is the general histogram capability); the CPU paths are
single-threaded; the device histogram adds one atomic per point for more than 64 cells; direct coefficients and
inclusion-exclusion counts are CPU only.
