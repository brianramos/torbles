# Histograms and prefix distribution calculus (DIST-001, DIST-002)

`torbles/math_dist.h` (every contract; the state layout, the plan, the 256-bit helpers and the request evaluator are
inline and shared with CUDA), `math_dist.c`, `math_dist_gpu.cu`, `math_dist.py` (opt-in). DIST-001 builds reusable
native histogram states over a FND-004 scope; DIST-002 answers exact distribution requests from the prefix sums of a
sorted 1-D support. They are the state the later distribution capabilities read (entropy and information, sorted
pairwise summaries, generating polynomials, Boolean histograms, group quantifiers).

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_dist as md
scope = mbk.Scope.from_table(db["f"], ["x", "y", "w"], group_by=["k"], rows=mask)
h = md.histogram(scope, ["x", "y"], weight="w", repr="sparse")       # repr="dense", domain=[(lo, hi, null_slot)]
h = md.histogram(scope, ["x", "y"], backend="cuda")                   # the CUDA kernels, the same state
h.cells(g)        # [(key, count, wcount, wsum)] in cell order; key: logical ints, None = the NULL category
h.group(g)        # {"present", "rows", "cells"};  h.table(): present groups with decoded group keys
h.info            # groups, present, rows, cells, occupied, max_cells, bytes, scans, passes, strategy, domain, ...
h.to_dense(domain=None); h.to_sparse(); h.merge(other); h.rollup(coarse_scope); h.marginal(["x"])
b = md.Builder(scope, ["x"]); b.accumulate(0, n // 2); b.merge(other_builder); b.finish()
md.need(scope, ["x", "y"], repr="dense", backend="cpu")              # the needs checked before allocating
p = md.histogram(scope, ["x"], weight="w").prefix(mass="weight", degree=2)
p.cdf(t, closed=True), p.tail(t, closed=False), p.count(lo, hi, closed=(True, False))   # mass, total, fraction
p.quantile(Fraction(1, 4), "cont"), p.value_at_rank(r), p.moment(k, lo, hi), p.trimmed(r1, r2, k)
p.query([dict(op="quantile", p=Fraction(9, 10), rule="disc", group=g), ...], backend="cuda")
```

## DIST-001: the histogram state

- Source: a scope (`math_bank.Scope`): selected rows, dense groups, exact logical int64 columns (a DECIMAL(p, s)
  value is logical / 10^s; `info["scales"]`). k = 1..3 coordinates (scope columns, repeats allowed) and an optional
  weight column (any scope column).
- Cells: per group, k-tuples whose coordinate is a value or NULL. NULL is a category of its own per coordinate,
  never merged into a value: (3, NULL), (NULL, NULL) and (3, 5) are three cells (SQL `GROUP BY x, y`). Per cell
  `count` (rows: the bag multiplicity); weighted: `wcount` (rows with a present weight) and `wsum` (their exact sum,
  signed 128 bits; SQL `COUNT(w)`, `SUM(w)`). Per group `rows`. Presence is explicit: present iff rows > 0 (a group
  whose values are all NULL is present, its rows in NULL cells); an absent group has no rows and no cells.
- Order: coordinate 0 most significant; values increasing, NULL after every value.
- Representations, exact images of each other: `dense`, a bounded domain per coordinate (lo..hi plus a NULL slot;
  the scope domain is [min, max] of the selected present values with a NULL slot iff a selected value is NULL; an
  explicit domain must contain every selected value and NULL, else `DomainError` before anything is read), all
  cells stored (structural zeros included); `sparse`, the occupied cells of each group in order (CSR offsets, the
  coordinates, a NULL mask). `to_dense` / `to_sparse` preserve every occupied cell; a dense domain that would drop
  one is refused (`DomainError`), an over-budget joint domain with `AlgebraBudgetError`.
- Construction: one pass over the rows. A dense state is counted in place; a sparse one is counted in a dense
  scratch over the scope domain and compacted (`strategy="count"`, chosen by `auto` when groups x cells <=
  4 x selected rows + 65536) or built by a stable merge sort of one entry per row (`"sort"`: any domain, e.g. wide
  int64 values or near-row-sized joint supports). `info["scans"]` = 1 for a build (0 for a derived state),
  `info["passes"]` over the scratch. A `Builder` accumulates disjoint physical row ranges and merges builders of the
  same scope descriptor and spec; any chunking gives the same state bit for bit.
- Merge: the bag union of two histograms of the same shape (k, representation, weightedness, scales, groups and,
  dense, the same domain; else `DomainError`), e.g. partitions of a distributed table; the result is unscoped.
  Rollup: along `scope.rollup_map(coarse)` (or `map=`, `groups=`): cells add, a coarse group is present iff one of
  its fine groups is; with a coarse scope the result is that scope's histogram (equal to building it directly).
  Marginal: keeps a subset of the coordinates, summing the others out (NULL categories included); equal to the
  direct build of those columns.
- Exactness: counts uint64, weight sums int128: the build proves selected rows x max |w| <= 2^127 - 1 from the
  scope's extremes (`OverflowError` otherwise); merges and rollups check their sums. At most 2^32 - 1 scope rows.
- CUDA (`backend="cuda"`, `tr_mgpu_dist_hist_build`): dense, one atomic add per row into its cell (128-bit weight
  sums with the two-word atomic); sparse, the same counting into a device scratch over the scope domain, then a
  compaction in launches without barriers (chunk occupancy, one serial scan of at most 65536 chunk counts, ordered
  writes per chunk with the group offsets). The readback is adopted (`tr_dist_hist_from_states`) after checks
  (rows, order, domain, weight bounds). A scope domain whose scratch exceeds the budget is refused on the device
  (`AlgebraBudgetError`); the CPU `sort` strategy has no such bound.

## DIST-002: the prefix state and its requests

- `Histogram.prefix(mass, degree)` of a 1-D histogram: per group the distinct present values x_1 < ... < x_d of
  positive mass m_i (`"count"`: rows; `"weight"`: weight sums, which needs every present weight >= 0, else
  `DomainError`; zero-weight values leave the support) and the inclusive prefix sums of m x^k for k = 0..degree
  (<= 4). NULL rows are counted apart (`support(g)["nulls"]`) and are in no answer. Proof at build:
  T x X^k <= 2^127 - 1 per group (T = total mass, X = max |x|), else `OverflowError`.
- Semantics over the expanded multiset y_1 <= ... <= y_M (x_i repeated m_i times; a weight replicates in units of
  10^-wscale), thresholds and levels exact rationals in logical units (int, Fraction, Decimal; floats are refused):
  - `count(lo, hi, closed=(lo_closed, hi_closed))`: the mass of the interval (an end None is unbounded), the total M
    and the fraction mass / M (None when M = 0). `cdf(t, closed)` = (-inf, t] or (-inf, t); `tail(t, closed)` =
    [t, inf) or (t, inf).
  - `moment(k, lo, hi, closed)`: sum m x^k over the interval (a range integral), an exact int.
  - `quantile(p, rule)`, 0 <= p <= 1: `disc` (SQL PERCENTILE_DISC: y_r, r = max(1, ceil(p M))), `cont` (SQL
    PERCENTILE_CONT: h = p (M - 1), y_{j+1} + (h - j)(y_{j+2} - y_{j+1}), j = floor(h), an exact Fraction), `lower`
    (y_{j+1}), `higher` (y_{j+2} when h is not an integer, else y_{j+1}). Ties are equal order statistics; M = 0
    gives None (SQL NULL).
  - `value_at_rank(r)`: y_r (None past M). `trimmed(r1, r2, k)`: sum of y_r^k over the ranks r1..r2 (1-based,
    inclusive; r1 = r2 + 1 sums nothing); `trimmed_mean`: that sum over r2 - r1 + 1 ranks (None for none).
- Batches: `p.query(dicts)` / `p.query_arrays(REQUEST_DTYPE array)`; each answer has its own status, so one invalid
  request never fails a batch at the native level (the dict API raises on it). CPU (`tr_dist_query`) and CUDA
  (`tr_mgpu_dist_query`: one work item per request, binary searches over the prefix arrays) run the same inline
  evaluator and answer bit for bit alike. `p.device()` uploads the state once for any number of batches.

## Budgets and refusals

| Check | support | states | bytes |
|---|---|---|---|
| dense build | cells per group | groups x cells | arrays, rows, coverage bitmap, builder and histogram objects |
| sparse build, first stage | - | counting scratch: groups x cells (sort: -) | builder, coverage bitmap, row counts and the scratch (count: the cell counters; sort: 2 x 64 bytes per selected row) |
| sparse build, final state | the largest group's cells | (count: the scratch) occupied cells | first stage + the final state |
| CUDA build | cells per group of the device array | groups x those cells | every device copy (and the sparse output capacity) |
| conversions, merge, rollup, marginal | cells per group / largest group | the resulting cells | the result (sorting: 2 x 64 bytes per cell first) |
| prefix | the largest group's support | entries x (degree + 1) | the prefix state |
| requests | output = requests | | CUDA: device copies of requests and answers; upload: the state's copy |

Also `variables` (k, +1 weighted; 1 for a prefix) and `degree`. Sizes past 64 bits are refused whatever the
budget. Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`) before allocation; `ValueError`
(invalid input, overlapping or incomplete chunks, an invalid request), `OverflowError`, `UnsupportedError`
(k > 3, degree > 4, a moment the state does not keep, more than 2^32 - 1 rows), `DomainError` (a domain that drops a
cell, a negative weight as a mass, no refinement map), `StaleStateError` (the scope's table changed).
