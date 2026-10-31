# Hierarchy rollups and contrasts (MULTI-001)

`torbles/math_multi.h` (every contract; the exact 256-bit helpers and the contrast cell shared with CUDA),
`math_multi.c`, `math_multi_gpu.cu`, `math_multi.py` (opt-in). A hierarchy is 2..8 levels of group keys over the
same rows (day -> month -> year, city -> region -> country); the moment states (FND-004 banks) are built once at the
finest level and rolled up exactly through certified refinement maps.

```python
from torbles import math_bank as mbk, math_multi as mm
day, month, year = (mbk.Scope.from_table(t, ["x", "d"], group_by=k, rows=mask) for k in (["day"], ["month"], ["year"]))
m = mm.certify(day, month, backend="cpu")            # Map, proven; or RefinementError (.witness); backend="cuda" too
mm.declare(month, year, {202401: 2024, 202402: 2024})  # declared by the schema (logical keys), not proven
m2 = mm.declare(month, year, ids).verify(month, year)  # proven on these rows, or RefinementError
m.compose(m2), m.status, m.ids, m.parent(g), m.check(day, month), mm.map_need(day, month, "cuda")
h = mm.Hierarchy([day, month, year], ["1", "x", "d", "x^2"], mode="sql", maps=None, backend="cpu")
h.status, h.info, h.bank(level), h.state(level, g), h.table(level), h.map(level), mm.hierarchy_need(levels, bank)
h.contrasts(level, "mean", terms=None, groups=None, backend="cuda").rows()   # exact Fractions per child and term
h.update(new_day_bank)                               # localized; levels=new_scopes to re-prove the maps
```

## Levels and maps

- A level is a FND-004 scope plus the logical codec descriptor of its keys (`mm.codec_descriptor`: each key's
  coltypes codec, its kind, offset, scale and dictionary, as canonical JSON; empty for array scopes, whose ids are
  positional). Every level of a hierarchy has the same rows: the same source, table version and selection bits
  (value columns may differ: coarse levels can be scopes without value columns).
- A map sends each fine group to one coarse group or to none (`UNMAPPED`). It is bound to both levels' groupings
  (the FND-004 grouping descriptor: key specs, radices, NULL codes, lookup contents, or the ids of an array scope)
  and codecs, and, when proven, to the rows it was proven on. A map is used only where every binding is equal byte
  for byte: a map built for another scope, grouping, selection, revision or codec raises `DomainError`; the same
  logical keys under the same codecs are accepted (a declared map of one table applies to another table with equal
  key codecs). The bindings are exact copies: for a table scope a few words per key and the lookup contents, plus
  the selection bits for a proven map; for an array scope the group ids themselves (4 bytes per row and level), so an
  array hierarchy's retained bytes grow with its rows (an 8-level hierarchy over 500k rows: about 27 MB of bindings
  next to kilobytes of states).
- `certify(fine, coarse)`: exhaustive over the selected rows: proven functional, `map[f]` = the coarse group of every
  selected row of fine group f (`UNMAPPED` for groups without selected rows); otherwise `RefinementError` with the
  witness: the smallest selected row whose coarse group differs from that of the smallest selected row of its fine
  group (`row`, `first_row`, `fine`, `coarse`, `first_coarse` and the decoded keys). NULL keys are groups like any
  other: a NULL fine key whose rows have two coarse keys is not a refinement.
- `declare(fine, coarse, mapping)`: given by the schema, status `declared` (not proven). `mapping` in logical keys
  (translated through the levels' codecs; a coarse key the coarse codec cannot represent raises `ValueError`) or as
  ids. `verify` proves it on the data (entries of groups without rows keep their declared values) or raises
  `RefinementError` at the smallest selected row where it disagrees. `compose` chains maps whose levels chain:
  proven when both are proven on the same rows.
- CUDA (`backend="cuda"`): `tr_mgpu_multi_certify` makes two passes over the rows (per fine group the smallest row and
  the smallest and largest coarse group with 64-bit atomics; then the smallest row whose coarse group differs from
  that of its group's smallest row, a per-thread minimum and one atomic); `tr_multi_map_from_cert` checks the result
  against the scopes before the map exists. Same map and same witness as the CPU.

## Hierarchies

- `Hierarchy(levels, basis, mode, maps=None, bank=None)`: the finest bank (FND-004, `bank` or built on `backend`), one
  map per pair of levels (certified on `backend` when not given). Level i + 1's states are the exact rollup of level
  i's (counts and sums add). Presence is explicit at every level: a group is present iff it has rows; an all-NULL
  group is present with count 0 and sum 0; an absent group has rows 0. A present group without a parent raises
  `DomainError`.
- `status`: `proven` (every map proven on the hierarchy's rows: every level equals the bank built directly on its
  scope; a level whose scope has the rows and value columns of level 0 is scoped to it, so `h.bank(k)` can enter a
  FND-004 cache), `declared` (some map is declared: exact rollups along the declaration, not proven to be the level's
  data grouping; those levels are unscoped), `carried` (after an update without new levels, below).
- `update(bank, levels=None, maps=None)`: the hierarchy of a new finest bank (same basis, mode, columns, groups). The
  dirty children of each transition are the groups whose state changed and the children whose parent changed; each
  touched parent is its old state minus its dirty children's old states plus their new ones, in exact 192-bit
  arithmetic, then checked to fit (counts in [0, 2^64), |sum| <= 2^127 - 1, `OverflowError` otherwise). When the
  dirty children are more than half of a level the level is rolled up completely (adaptive). Without `levels` the maps
  are carried: status `carried`, coarse levels unscoped (certify on the new data to prove them); with `levels`, the
  maps given or certified on them. `info`: per level `strategy` (`copy`, `local`, `rollup`), `level_work` (cells
  visited: groups x (terms + 1)), `changed`, `bytes`, `level_bytes`.
- `bank(k)` returns an independently owned FND-004 bank (it outlives the hierarchy); requests read the retained
  states, no row is rescanned.

## Contrasts

For child level l (< levels - 1), child c with parent p = map[c] and term j, n = count (rows where term j is valid) and
s = sum of term j, in logical units (term j carries the decimal scale sum_c e_jc * scale_c):

| kind | value | not defined |
|---|---|---|
| `mean` | s_c / n_c - s_p / n_p | `undefined` when n_c = 0 (a mean over no valid row) |
| `total` | s_c - s_p | |
| `share` | s_c - s_p n_c / n_p = n_c (mean_c - mean_p); sums to 0 over the children of p | `undefined` when n_p = 0 |
| `sibling_mean` | s_c / n_c - s_b / n_b, b = the next present child of p after c (id order) | `no sibling`, `undefined` |
| `sibling_total` | s_c - s_b | `no sibling` |

An absent child has status `absent`. Values are canonical rationals (`Fraction`; numerators up to 2^193 in a 256-bit
carrier, denominators up to 2^128), reduced by gcd(num, n_c) then gcd(num, n_o), which is canonical because n_c / g1
and n_o / g2 are each coprime to the reduced numerator; never a division by zero. `raw()` gives the native integers.
The CPU loop and the CUDA kernel (`tr_mgpu_multi_contrasts`, one work item per child and term, no atomics) evaluate
the same inline function: equal bit for bit.

## Budgets and refusals

| Call | resources |
|---|---|
| certify / declare / verify | states = fine groups; bytes = the map, its bindings, the scan scratch (CUDA: the device copies of the selection and ids, 32 bytes per fine group) |
| hierarchy build | variables, degree, support = the basis; states = groups x terms of the coarse levels; bytes = every bank, map copy and rollup scratch |
| update | as the build, plus the staged states, dirty lists and wide accumulators |
| contrasts | support = terms requested; output = child groups x terms; bytes = the links scratch (CUDA: both levels' states, links, output) |

Refusals raise before anything is allocated: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`; `mm.map_need`,
`mm.hierarchy_need`, `h.contrasts_need`), `UnsupportedError` (more than 8 levels), `ValueError` (one level, a bad
id, kind or term), `DomainError` (bindings differ, levels of other rows, a bank of another scope, a present group
without a parent), `RefinementError` (a `DomainError` with `.witness`), `OverflowError`, `StaleStateError` (a table
scope whose table or selection changed).
