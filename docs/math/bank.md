# Scopes, grouped moment banks and their cache (FND-004)

`torbles/math_scope.h/.c` (the data scope), `math_bank.h/.c` (grouped exact moment banks, rollup, cache),
`math_bank_gpu.cu` (the CUDA accumulation and rollup), `math_bank.py` (opt-in Python API). This is the data path the
later data-side capabilities build on: they read rows through a scope and moments from a bank. The first-slice scalar
moment cache (`math_state.c`) is unchanged.

```python
from torbles import math_bank as mbk
scope = mbk.Scope.from_table(db["f"], ["x", "d"], rows=mask, group_by=["k", ("c", db["dim"], "region")])
scope = mbk.Scope.from_arrays({"x": xs, "y": ys}, valid={"x": present}, groups=ids, ngroups=G, rows=mask)
bank = mbk.build(scope, ["1", "x", "d", "x^2", "x*d"], mode="sql", backend="cpu")   # or backend="cuda"
bank.state(g)       # {"present", "rows", "count": [...], "sum": [...]} exact ints, one entry per term
bank.table()        # the present groups with their decoded keys
coarse = bank.rollup(mbk.Scope.from_table(db["f"], ["x", "d"], rows=mask, group_by=["k"]))
b = mbk.Builder(scope, basis); b.accumulate(0, n // 2); b.merge(other_half); b.finish()
cache = mbk.Cache(capacity=16, max_bytes=8 << 20)
cache.bank(scope, basis, mode="sql")      # a hit (an independent copy) or a build that is then cached
cache.refresh(scope, basis); cache.stats  # hits, misses, puts, evictions, refreshes, invalidations, bytes
mbk.need(scope, basis, mode, backend)     # the needs checked before anything is allocated
```

## Scope

- Rows: the physical rows of the source and a copy of the selection bitmap (`rows=`: a boolean mask or a packed
  uint64 bitmap; default the table's used rows, `Table.used_bm`). Only selected rows count.
- Values: exact logical integers, `code + offset` (decimal value = logical / 10^scale), for integer, decimal and boolean
  codecs, self-extending (64-bit) columns included (`code + (high code << 32) + offset`). Each selected present value
  must fit int64 (`OverflowError` at build). Validity per value; the scope records per column the count of selected
  present values and their min and max (`scope.column(name)`), which the bank's overflow proof uses.
- Groups: dense ids `< scope.groups`, the mixed-radix number of the key codes, first key most significant (no key:
  one group). A key is a column's codes (a nullable column has its own NULL group) or a dimension attribute reached
  through a foreign key, `(fk, dim, attr)`: unmatched references, NULL foreign keys and NULL attributes form the NULL
  group, as a `LEFT JOIN ... GROUP BY` does. At most 32 value columns, 12 keys, 2^32 - 1 groups.
- Freshness: a table scope records the table (address and a lifetime token), its revision and row count, the
  sidecars of the self-extending columns it read, the selection, and in Python the revision and used rows of every
  dimension a lookup copied. Any change (e.g. `t.flip(plane, row)`) makes `scope.fresh()` false; building, caching or
  rolling up a stale scope raises `StaleStateError`.
- Native view (`tr_scope_view`, 2096 bytes): pointers, offsets, extremes and counts, copied by value into a kernel with
  its pointers swapped for device copies; the accessors `tr_scope_is_selected / present / value / group` are shared by
  CPU code and kernels.

## Bank

- Basis: up to 1024 distinct monomials over the scope's columns, total degree <= 4 (all monomials of degree <= 2 over
  32 columns are 561), as exponent tuples, `{name: exponent}` or strings (`"1"`, `"x*y^2"`); `mbk.monomials(names, d)`.
  Past these limits: `UnsupportedError`; a duplicate term: `ValueError`.
- States per group: `rows` (bag multiplicity), and per term `count` (rows where the term is valid) and `sum` (exact,
  signed 128 bits) of the term's logical values. `mode="sql"`: a term skips a row where one of its columns is NULL
  (SQL `COUNT(x*y)`, `SUM(x*y)`); `mode="joint"`: a row counts for every term only when every column of the basis is
  present (listwise). Presence is explicit: a group is present iff it has selected rows; an all-NULL group is present
  with count 0 and sum 0; an absent group has no rows.
- Exactness: before accumulating, every term is proven to fit: `N * prod max(|min_c|, |max_c|, 1)^e_c <= 2^127 - 1`
  (N: the rows the term can be valid on); otherwise `OverflowError` before anything is computed. Integer sums, no
  rounding, no wrap. A chunked or merged accumulation (`Builder`: disjoint row ranges, every row covered once) and the
  CUDA path give the same states bit for bit.
- Rollup: `bank.rollup(coarse_scope)` uses the scope's refinement map (`scope.rollup_map(coarse)`: the coarse group
  of each fine group; scopes of the same rows and columns whose coarse grouping is a function of the fine one, else
  `DomainError`); counts and sums add, a coarse group is present iff one of its fine groups is, and the result equals
  the bank built directly on the coarse scope. `rollup(map=..., groups=...)` takes any map (the result is unscoped and
  cannot be cached). A rollup is refused (`OverflowError`) only when the magnitudes gathered into one coarse cell exceed
  2^127 - 1, which never happens for a proven bank.
- CUDA (`backend="cuda"`): `tr_mgpu_bank_build` / `tr_mgpu_bank_rollup` write the states into host arrays that
  `tr_bank_from_states` adopts after consistency checks; `bank.backend` is `"cuda"` on a device and `"cuda-emulated"`
  in the emulation. Work items (term, row slice) accumulate locally while the group stays the same and flush with
  integer atomics; 128-bit addends use a two-word atomic (`tr_bank_atomic_add_wide`, the local generalization of
  `tr_atomic_add_i128`).

## Cache

- Key: the exact scope descriptor (`tr_scope_key_*`: table identity, revision, row count, selection bits, column specs,
  key specs with the lookup contents; for array scopes the normalized data itself), the basis and the mode; the 128-bit
  digest is only a pre-filter. A different predicate, NULL policy, revision or group grain never hits.
- `get` returns an independently owned copy or `None`; `put` stores a copy of a bank built from that scope (a rollup
  with its coarse scope included); `refresh` rebuilds, replaces, and drops entries of other versions of the same table.
  Capacity 1..128 entries, LRU, `max_bytes` (default 8 MiB) covering the fixed entry array, every key and every bank;
  a single entry larger than the budget is refused (`AlgebraBudgetError`, resource `bytes`). Calls on one cache are
  serialized by the Python lock (natively: externally serialized). The cache compares descriptors; freshness is
  checked by the Python wrapper before every get and put.

## Budgets and refusals

| Check | states | bytes |
|---|---|---|
| scope | groups | selection, decoded columns, validity, group ids, copied lookups, decoding scratch, the object |
| bank (CPU) | groups x terms | states `8 G + 24 G m`, coverage bitmap, basis, builder and bank objects |
| bank (CUDA) | groups x terms | the larger of the device copies (selection, ids, the basis's columns, validity, states, terms) and the adopted bank |
| rollup | coarse groups x terms | the coarse bank and the bound's scratch (CUDA: plus the device copies) |

Also `variables` (columns of the basis), `degree`, `support` (terms). Refusals raise `AlgebraBudgetError` with
`.resource`, `.need` and `.limit` before anything is allocated (`mbk.need` reports the needs). Other refusals:
`ValueError` (invalid input, an id or code out of range, overlapping or incomplete chunks), `OverflowError`,
`UnsupportedError`, `DomainError` (no refinement map; a bank put under another scope), `StaleStateError`.
