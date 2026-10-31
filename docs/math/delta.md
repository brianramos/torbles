# Signed linear state updates (DELTA-001)

`torbles/math_delta.h` (every contract; the proof and the sign check shared with CUDA), `math_delta.c`,
`math_delta_gpu.cu`, `math_delta.py` (opt-in). A maintained state holds the FND-004 cells of a weighted bag of rows;
insert, delete and weight-change batches are merged into it exactly, giving a new immutable state with a new
identity.

```python
from torbles import math_bank as mbk, math_delta as md
st = md.State.from_bank(mbk.build(scope, basis, "sql"), promise="nonnegative")   # or "signed"
d = md.Delta.from_arrays({"x": xs, "d": ds}, valid={"x": vx}, scales={"d": 2}, groups=g, ngroups=scope.groups, weights=w)
d = md.Delta.from_scope(rows_scope, sign=-1)          # the selected rows of a scope of the same schema
d = md.Delta.between(old_scope, new_scope)            # a table diff: inserts, deletes, value changes, moves
st2 = st.apply(d, backend="cpu")                      # or "cuda"
st2.state(g), st2.states(), st2.info                  # identity, base, step, bound, revision, promise, bytes
st2.bank(new_scope)                                   # the FND-004 bank of exactly that scope (after a diff)
st2.publish(cache, new_scope); st2.equal_bank(bank); st.need(d, "cuda"); md.Delta.need(old=a, new=b)
```

## States

- Per group: `rows` (the sum of the multiplicities), per term `count` (the multiplicities of the rows where the term
  is valid: SQL or listwise validity as in FND-004) and `sum` (the exact sum of multiplicity x term value, 128-bit).
  Rows and counts are int64. `from_bank(bank)` takes the bank of a scope (`bank.scope`): that scope is the state's
  source and its schema (for a table: the table, every column spec and the grouping descriptor; for arrays: the
  column scales and groups) decides which deltas apply.
- Promise `nonnegative` (an empirical measure): an update is refused, before anything is exposed, when a touched group
  would have `rows < 0` (deleting rows that are not there), `count < 0`, `count > rows` (deleting NULL rows that are
  not there), `count = 0` with `sum != 0`, a constant term whose sum is not its count (SQL mode: whose count is not
  rows), or a negative sum of a term with only even exponents. These are exact necessary conditions on the
  aggregates; a delete of a row that is not there but has the group, validity and values of present rows cannot be
  seen in linear states. The error (`NegativeMultiplicityError`, a `DomainError`) reports the first violation in
  (group, slot) order: `.group`, `.term` (`None` for the bag multiplicity), `.reason`, `.have`, `.change`,
  `.result`, `.need` (= -change for a delete). Promise `signed`: a signed measure, no sign condition; `bank()` of a
  state with a negative count raises `DomainError`.

## Deltas

- Rows with logical int64 values (decimals scaled), validity, a group id and an int64 weight: +1 insert, -1 delete,
  any other value a weight change (0 changes nothing). A group move is a delete and an insert.
- `from_arrays`: compatible with a state of the same column count, scales and groups (the ids are the caller's).
  `from_scope`: the selected rows of a scope with the state's exact schema. `between(old, new)`: two scopes of one
  schema and row count; for each row whose contribution differs (selected in one only, another group, validity or
  value) -1 for the old version and +1 for the new one; when both scopes read the same table version only the
  selection is compared (word by word). A diff applies only to the state whose source is exactly `old` (its whole
  FND-004 descriptor) and the result's source is `new`.
- Identity: a 128-bit digest of the delta's exact content (and of both descriptors for a diff). A new state's identity
  is digest(old identity, delta identity, step); `base` is the first state's identity, `step` counts updates,
  `revision` is the new scope's table revision for a diff. A state is `bound` only after `from_bank` or a diff; then
  `bank(scope)` gives a scoped FND-004 bank of exactly that scope (re-proven against its extremes) and `publish`
  puts it into a FND-004 cache with refresh, dropping the entries of the table's other versions; an old state never
  answers for the new scope and the old scope is stale.

## Exactness

Before anything is computed, every touched cell is proven to stay inside its carrier in any order of the additions:
`|rows| + W_g <= 2^63 - 1`, `|count| + W_g <= 2^63 - 1`, `|sum| + W_g * B_j <= 2^127 - 1`, with W_g the sum of
|weight| over the delta's rows in group g and B_j = prod max(|min|, |max|, 1)^e over the delta's present values.
Otherwise `OverflowError` (nothing wraps). The proof is conservative (a net-zero swap of two large rows can be
refused); a complete rebuild then has its own FND-004 proof. CUDA (`tr_mgpu_delta_apply`): the host gathers the
touched groups' cells and checks the proof; work items per delta row (bag multiplicity) and per (row, term) add with
signed 64-bit atomics and the two-word 128-bit `tr_atomic_add_wide`; for a nonnegative state one work item per cell
evaluates the sign conditions and keeps the smallest contradicted cell with an atomic min; the host
(`tr_delta_state_from_cells`) checks the cells against the proven bounds and the sign conditions again before the new
state exists. CPU and CUDA give the same cells, identities and refusals.

## Budgets and refusals

| Call | resources |
|---|---|
| a delta | output = states = its rows; bytes = its arrays, sorting scratch, schema and (diff) both descriptors |
| an apply | variables, degree, support = the basis; states = groups x terms; bytes = the new state and the gathered cells (CUDA: the delta's columns, weights and ids, the terms and the cells on the device) |

Refusals raise before allocation: `AlgebraBudgetError` (`State.need`, `Delta.need`), `OverflowError`,
`NegativeMultiplicityError`, `DomainError` (another schema, column count, scale or grouping; a diff from another
version; a bank of another scope), `UnsupportedError` (2^32 rows or more in one delta), `ValueError`,
`StaleStateError`. An update copies the state's cells (groups x terms) once: the new state is an independent object.
