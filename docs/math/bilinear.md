# Bilinear (join) and nonlinear update laws (DELTA-002)

`torbles/math_bilinear.h` (every contract; the cross-term and matrix functions shared with CUDA), `math_bilinear.c`,
`math_bilinear_gpu.cu`, `math_bilinear.py` (opt-in). Built on DELTA-001's maintained states and deltas
([delta.md](delta.md)) and the FND-004 data path ([bank.md](bank.md)); DELTA-001's files are unchanged.

```python
from torbles import math_bank as mbk, math_delta as md, math_bilinear as mbl
sa = md.State.from_bank(mbk.build(scope_a, ["1", "x", "x^2"]))      # side A: groups key * GA + g (group_by=[k, g])
sb = md.State.from_bank(mbk.build(scope_b, ["1", "y"]))             # side B: groups key * GB + h
j = mbl.Join.build(sa, sb, keys=K, groups_a=GA, groups_b=GB, pairs=[("1", "1"), ("x", "1"), ("x", "y")],
                   null_key=None, backend="cpu")                     # or "cuda"
j.cell(g, h)              # {"rows", "count": [per pair], "sum": [per pair]}, exact ints; j.cells()
j2 = j.apply(a=delta_a, b=delta_b, backend="cpu")                    # either side may be None
j2.terms                  # {"dA_B", "A_dB", "dA_dB"}: the cross terms of that update, cells as in cells()
j2.finalize(g, h, "avg", pair=1); j2.summarizes(j2.a, j2.b); j2.info; mbl.Join.need(...); j.update_need(...)
mbl.matrix_delta(A, B, dA, dB, backend="cuda")      # {"AB", "dA_B", "A_dB", "dA_dB", "updated"}
mbl.finalize(state, g, "var_samp", "x", "x^2")      # sum, count, avg, ratio, var_pop, var_samp of linear cells
v = mbl.Values.from_scope(scope, "x", kind="multiset")   # or "extremum"
v2 = v.apply(delta); v2.finalize(g, "count_distinct"); v2.group(g); v2.entries(g)
mbl.requirement("min")    # {"retained": "multiplicities", "alternative": "extremum", "linear": False, ...}
```

## Join states

- Sides: two DELTA-001 states whose dense groups are key-major, `key * groups + g` (`Scope.from_table(t, cols,
  group_by=[key, g])` gives it when both tables share the key's codes; arrays: the caller's ids). Keys are dense ids
  `< keys`; `null_key` never matches (SQL inner-join NULL keys).
- Per output cell `(g, h)` (index `g * groups_b + h`) and pair `(A term i, B term j)`: `rows = sum_K rowsA[K, g]
  rowsB[K, h]`, `count = sum_K countA[K, g, i] countB[K, h, j]`, `sum = sum_K sumA[K, g, i] sumB[K, h, j]`: the SQL
  `COUNT(*)`, `COUNT(a.t_i * b.t_j)`, `SUM(a.t_i * b.t_j)` of `A JOIN B ON A.key = B.key GROUP BY A.g, B.h` weighted by
  the multiplicities, NULLs as each side's SQL / listwise validity. With one value per cell it is the matrix product.
- Update law: `d(AB) = dA B + A dB + dA dB` per key, cell and pair, over the keys the deltas touch. `apply` updates
  each changed side with DELTA-001 (its proof, promise and identity) and the join with the three terms, each formed
  from the four exact products of old and new cells (`T1 = A'B - AB`, `T2 = AB' - AB`, `T3 = A'B' - A'B - AB' + AB`).
  A key gaining or losing matches (a B insert at key K multiplies every A row of K once more) is the T2 term of the
  per-key cells; no SUM deletion rule is applied to a joined row. `new_a=` / `new_b=` take a side update computed
  already: it must be a successor of the summarized side (same base, step + 1, same shape, the cells outside the
  delta's groups unchanged), else `DomainError`; the result is exactly the join of the new sides (`summarizes`).
- Finite differences, not derivatives: `D(AB)[dA, dB] = dA B + A dB` (the derivative, JET-001) omits `dA dB`, the
  pairs matched on both sides at once; a join update keeps it (tested against JET-001's directional derivative and
  second-order Taylor coefficient of `a * b`).
- Identity: a build is digest(side identities, spec, pairs); an update digest(old identity, new side identities,
  step); `base`, `step`; `touched` = keys of the last update.

## Exactness

Join values are 320-bit two's complement (`tr_bilin_wide`): a value is a sum over fewer than 2^64 keys of products of
two cells (`|int64 int64| < 2^126`, `|int128 int128| <= 2^254`), so `|value| < 2^318` whatever the data; nothing in the
join is refused for overflow. The side cells keep DELTA-001's a-priori proof (`OverflowError` there). Matrices: int64
entries, `K < 2^32`, five 320-bit products per entry. CPU and CUDA give the same words: `tr_mgpu_bilin_cross` runs one
work item per (touched key, g, h, value slot) with a five-word atomic add (each word gets the carry its own atomic
returned), `tr_mgpu_bilin_matrix` one work item per output entry.

## Nonlinear finalizers and their retained state

| finalizer | retained state | a delete |
|---|---|---|
| `sum`, `count` | linear cells | subtracts (the SUM rule) |
| `avg`, `ratio` | sufficient sums (numerator, denominator) | updates the sums; finalized again |
| `var_pop`, `var_samp` | n, S1 = sum t, S2 = sum t^2 (terms t and t^2) | updates the sums: `(n S2 - S1^2) / n^2`, `/ (n (n-1))` |
| `count_distinct`, `sum_distinct` | per-value multiplicities (`Values` multiset) | decrements the value; it leaves at 0 |
| `min`, `max` | the multiset (exact), or `extremum` | extremum: exact while a copy of the extreme survives or a smaller / larger value arrives; else `RebuildRequired` |
| `join_sum` | per-key cells of both sides | the three cross terms |

Finalized values are exact `Fraction`s in logical units (`int` for value states), `None` for SQL NULL (no valid
rows, a zero denominator, `var_samp` of one row). A finalizer of a state that lacks its retained state raises
`DomainError` (`count_distinct` / `min` / `max` of linear cells or of a join, `count_distinct` of an extremum, `var`
of terms that are not t and t^2, `sum` of a value state). Value states: per group rows, NULL values and present values
(with multiplicity) and either the distinct values with multiplicities (multiset) or the extremes with theirs; updated
by DELTA-001 deltas with the scope's column count, the column's scale and the groups (the ids are the caller's
assertion, as for DELTA-001's array deltas). Refused exactly, nothing exposed: a value deleted more often than it is
there (multiset; extremum: below the minimum, above the maximum, an extreme's copies, the unknown inner values),
negative rows or NULLs (`NegativeMultiplicityError`, `.group`, `.value`, `.have`, `.change`), and the last copy of an
extreme deleted while unknown values remain (`RebuildRequired`: rebuild with `Values.from_scope`).

## Budgets and refusals

| call | resources |
|---|---|
| join build / update | support = pairs, output = cells, states = cells x values, iterations = keys x GA x GB x values, bytes = the new join, the terms, the gathered cells (CUDA: plus the device copies) |
| matrix delta | tensor = GK + KH (+ the updates), output = 5 GH, iterations = GHK, bytes = the outputs (CUDA: plus the inputs) |
| value state | support = the scope's present values (multiset: the most entries it can hold) or 2 x groups (extremum), states = groups, bytes = the state and its sorting scratch |
| value update | support = entries + delta rows (multiset) or 2 x groups, states = groups, bytes = the new state and the scratch |

`AlgebraBudgetError` before anything is allocated; `DomainError` (side states of other shapes or not the summarized
ones, a non-successor side, an incompatible delta, a finalizer without its state); `ValueError`; `OverflowError` (a
side's proof); `UnsupportedError` (more than 1024 pairs, `K >= 2^32`).

**Limitations.** Keys are dense ids shared by both sides (no key dictionary: engine tables must share the key codes);
side states are dense over keys x groups; value states compare columns, scales and groups only (not a delta's table
schema); value states, finalizers and job gathering run on the CPU (the cross terms and matrix products have kernels).
