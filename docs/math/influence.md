# Influence and predicate interactions (BOOL-002)

`torbles/math_influence.h/.c` (native CPU), `math_influence_gpu.cu` (CUDA kernels), `math_influence.py` (opt-in
Python API). For a predicate f (a BOOL-001 program over k conditions), the measure of the points where its XOR
derivative in a condition (its influence) or in a set of conditions (an interaction) is TRUE, with the signed mixed
differences, under the uniform measure of the 2^k assignments and under the empirical measure of the stored rows of a
scope, as separate results. The header comment of `math_influence.h` is the normative contract; this page summarizes
it.

```python
from torbles import math_bool as mb, math_influence as mi
prog = mb.Program((x[0] & x[1]) | ~x[2], k=3)
rows = mb.Masks.from_scope(scope, conditions)            # BOOL-001 masks over the rows of a scope
r = mi.influence(prog, rows=rows, order=2)               # {"uniform": {...}, "empirical": {...}, "interpretation"}
mi.uniform(prog, order=3, backend="cuda")                # the 2^k assignments only
mi.empirical(prog, rows, terms=[[0, 2], [1, 2, 3]])      # the stored rows only
mi.interaction_sets(k, max_order, min_order=2); mi.influence_need(prog, rows=rows, order=2)
```

Each measure's result: `measure` ("uniform" or "empirical"), `description`, `interpretation`, `causal` (False),
`points`, `f` (TRUE / FALSE / UNKNOWN counts of the predicate), `influences` (per condition i: `sensitive` (d_i f
TRUE), `insensitive` (FALSE), `unknown`, `known`, `increase` / `decrease` (f|1 - f|0 = +1 / -1), `signed_sum`,
`influence` = sensitive / points as a Fraction), `total_influence` (the sum of the influences), `total_sensitive`,
`sensitivity` (points with every first-order cofactor known, by their number of sensitive conditions),
`sensitivity_known`, `max_sensitivity`, `interactions` (per set S: `sensitive` (d_S f TRUE), `unknown`, `known`,
`increase` / `decrease` (D_S f > 0 / < 0), `signed_sum`, `measure`), `backend`, `device`.

## What is measured

Coordinate sensitivity of the predicate: how often substituting the condition(s), with every other condition held at
the point's own value, changes the predicate's value. The cofactors, the XOR derivative d_S f (XOR of the 2^m
cofactors) and the signed difference D_S f (sum of (-1)^(m - abs(t)) f|S:=t) are BOOL-001's, under SQL three-valued
logic; a cofactor never reads the conditions of S at the point. It is a property of the predicate as a function of
its conditions over the declared points; it is not the effect of an intervention on the data, and every result says
so (`causal` False, `interpretation`).

## The two measures, never mixed

- **uniform**: every one of the 2^k assignments once (points = 2^k), computed from the truth table of the predicate
  (built once: the program on the generated condition words); the cofactor words of an assignment t of S are the
  table words with the conditions of S substituted (conditions >= 6 select another word, conditions < 6 move bits
  within the word).
- **empirical**: the selected rows of the scope once each (points = the selected rows), computed by cofactors per row
  from the masks (the program once per 64-row word, then for each set and assignment only the cone of the set).
- The measure is the domain's kind and every result carries it. `empirical` refuses an assignment domain
  (`DomainError`: it is not stored rows); `influence` returns each requested measure under its own key. The two
  coincide only when the rows hold every assignment exactly once.
- UNKNOWN: a point where a cofactor over S is UNKNOWN is counted in `unknown` only; sensitive + insensitive + unknown =
  points, so the influence is the measure of {d_S f TRUE} with the UNKNOWN measure reported beside it.
- For two-valued predicates under the uniform measure the influence equals the Fourier weight sum over S containing i
  of fhat(S)^2 (checked in the tests).

## Execution and CUDA

Shared-mask execution, every condition at once: one pass over the words evaluates every term (the k first-order terms
and the requested sets) from one evaluation of the program per word; the 2^m cofactor words of a term are combined by
BOOL-001's bit-sliced counters and counted with popcounts; the per-point number of sensitive conditions is a 7-plane
bit-sliced counter. CUDA (`tr_mgpu_influence_compute`): for the uniform measure one launch builds the truth table (a
work item per word), then the scan; work items are word slices with their own node words and accumulators, adding each
nonzero counter to the totals with one integer atomic at the end (the 128-bit signed sums with `tr_atomic_add_wide`).
CPU, emulated CUDA and the device agree bit for bit.

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| influence | variables = k; degree = largest order; circuit = program nodes; support = 2^k (uniform: the assignments; 0 for rows); tensor = sum over the terms of 2^m (the k first-order terms included); output = terms + k + k + 1; bytes = plans, cones, node words, accumulators (+ the truth table for uniform; CUDA: device copies and scratch) |
| interaction sets | variables = k; output = the sets |

The uniform measure over k conditions also builds the 2^k assignment domain (`support` = 2^k: k <= 20 with the default
budget). Representation limits (`UnsupportedError`): order > 20, more than 65472 terms, k beyond BOOL-001's (64 rows
conditions, 63 assignment variables). A program over another k than the rows: `DomainError`. A stale scope:
`StaleStateError` before every use.

## Limitations

One measure over all selected rows (no per-group influences); influences are counts of the XOR derivative (the signed
difference's histogram is BOOL-001's `differences`); the CPU paths are single-threaded; no weighted (non-uniform,
non-empirical) measures.
