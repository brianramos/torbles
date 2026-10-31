# Moment and count tensors, and costed contraction planning (TENSOR-001, TENSOR-002)

`torbles/math_tensor.h` (every contract; the build plan, the row emitter, the dense step evaluator and the device
needs are inline and shared with CUDA), `math_tensor.c`, `math_tensor_gpu.cu`, `math_tensor.py` (opt-in). TENSOR-001
retains exact multi-index states over a FND-004 scope without forcing a dense Cartesian expansion; TENSOR-002
contracts bounded networks of them along a costed, budgeted order chosen through the FND-008 planner.

```python
from torbles import math_bank as mbk, math_tensor as mt
scope = mbk.Scope.from_table(db["f"], ["x", "y", "z", "c", "w"], group_by=["k"], rows=mask)
T = mt.moments(scope, ["x", "y", "z"], order=3, weight="w", policy="listwise", repr="dense")   # backend="cuda"
T = mt.moments(scope, [["1", "x"], ["x", "y"]], policy="sql")     # one column list per axis; "1" the constant
C = mt.counts(scope, ["c", "x"], repr="sparse", domain=None)       # joint counts; NULL a category of its own
T.entry(g, (0, 1, 2)); T.array(g); T.entries(g); T.terms(g); T.group(g); T.info; T.axes
T.to_dense(); T.to_sparse(); T.to_factored(); T.marginal([0, 2]); T.slice([1, (0, 2), None]); T.permute([2, 0, 1])
T.ungroup(); T.values(); T.same(U); T.equal(U); mt.outer(A, B, "factored"); mt.need(scope, ["x"], 2, backend="cuda")
mt.dense(values, groups=False); mt.sparse(dims, entries); mt.factors(dims, terms); mt.Builder(scope, ["x"], 2)
net = mt.network([(A, "ij"), (B, "jk"), (T, "gkl")], "gil")       # a grouped tensor: its group label first
plan = net.plan(devices=("cpu", "cuda"), kernels=("dense", "sparse"), max_orders=64, force=None, rates=None)
out, run = net.contract(plan, candidate=None)                      # the chosen candidate, or any admissible index
```

## TENSOR-001: the state

- Order m = 0..8 (0 is a scalar), dims 0..2^32 - 1 with fewer than 2^64 cells per group, G groups. Entries are exact
  signed integers below 2^127 in magnitude; order is row major (axis 0 most significant).
- `moments`: axis k lists 1..64 scope columns (repeats allowed, `"1"` the constant), T_g[i] = sum over the
  contributing rows of w prod_k x[col_k[i_k]]. `listwise`: a row contributes iff every referenced column and the
  weight are present. `sql`: a row contributes to an entry iff its factors and the weight are present, and the tensor
  keeps per entry the number of such rows (`counts`; the same moment tensor of the validity indicators). Axes sharing
  one column list are `symmetric`: dense builds accumulate the sorted multi-indices only and copy the rest.
- `counts`: one categorical column per axis over its domain (scope: [min, max] of the selected present values, with a
  NULL slot at index span iff a selected value is NULL; or explicit `(lo, hi, null)` per axis, which must hold every
  selected value and NULL, else `DomainError`); T_g[i] = the rows (or the sum of their present weights) in cell i.
- Per group `rows` (selected rows: the bag multiplicity; present iff rows > 0) and `used` (contributing rows), kept
  apart from NULL validity (counts, NULL slots). Arrays (`dense`, `sparse`, `factors`) give GENERIC unscoped tensors.
- Forms: `dense` (every cell, structural zeros included); `sparse` (per group the stored entries in strictly
  increasing multi-index order, CSR; stored iff value != 0, or with counts iff count != 0); `factored` (a declared sum
  of rank-one terms weight x (f_0 (x) ... (x) f_{m-1}) per group with int64 factors: a listwise moment build keeps one
  term per contributing row, `factors` declares terms, `outer` multiplies them, a conversion makes one unit term per
  stored entry). Conversions preserve the order, dims, axes, groups, rows, used, counts and every entry exactly;
  counts have no factored form (`UnsupportedError`). Nothing is expanded to its Cartesian domain unless asked.
- Operations: `marginal` (sum out the axes not kept; values and counts alike; a count marginal equals the direct count
  tensor of the kept columns), `slice` (per axis a fixed index (removed) or a range (kept, metadata adjusted: a
  category range keeps the NULL slot only when it holds it)), `permute`, `outer` (same groups; no counts), `ungroup`
  (groups become a leading axis), `values` (drops counts), `entry` (any form). A fixed slice, an outer product, a
  moment marginal are GENERIC.
- Exactness: a build first proves N x max|w| x prod_k max_{axis k} max(|min|, |max|, 1) <= 2^127 - 1 from the scope's
  extremes (`OverflowError` otherwise, before any row is read); derived tensors check their sums (`OverflowError`,
  never wrapped). `info["bound"]` is the exact max |entry| (factored: a proven bound on every partial sum).
- Construction: dense in place; sparse through a dense scratch (`strategy="count"`, chosen by `auto` when G x cells
  <= 4 x emissions + 65536) or by sorting one 48-byte record per emission (`"sort"`); factored in one pass. A
  `Builder` accumulates disjoint physical row ranges and merges; any chunking gives the same state bit for bit.
- CUDA (`backend="cuda"`, `tr_mgpu_tensor_build`): dense moment work items (entry, row slice) accumulate locally and
  add with the exact two-word atomic (symmetric tensors on their canonical entries, then a copy launch); count work
  items add one atomic per row; sparse builds insert every emission into an open-addressing table of >= 2 x capacity
  slots (a 64-bit compare-and-swap claims a slot: a local equivalent, `math_cuda.h` has none) and the host sorts the
  occupied slots into the canonical CSR. `tr_tensor_from_states` adopts after checks (shape, rows, order, bound).

## TENSOR-002: contraction plans

- A network: 1..8 tensors with one label per axis (16 labels at most; a grouped tensor has its group label first, and
  is ungrouped into a copy), and the output labels. Its value is the Einstein sum over the labels not in the output;
  a label joins any number of tensors. Refused: a trace (`UnsupportedError`), inconsistent label dims, an output label
  in no tensor, a grouped tensor without its group label (`ValueError`), per-entry counts (`DomainError`: use
  `values()`).
- Orders: sequences of pairwise steps; a step's result keeps the labels still needed by another operand or the output
  (the last step: the output, in its order; others in label order) and sums the rest. All orders are enumerated when
  there are at most `max_orders` (default 64; at most the `iterations` budget): 1, 1, 3, 18 for 1..4 tensors;
  otherwise a bounded search (the two greedy orders, then depth-first ones), `info["complete"]` False.
- Candidates: each order with the CPU dense kernel (every result entry by `tr_tensor_dense_entry`), the CPU sparse
  kernel (operands' sorted coordinates matched on their shared labels, products emitted, sorted, reduced) and, with
  `devices` including `"cuda"`, the CUDA dense kernel (the same evaluator, one work item per result entry). Inputs in
  another form are converted (a workspace copy).
- Evidence per candidate and step, exact: result labels, dims, cells, summed combinations, stored entries (dense:
  cells; sparse: the upper bound min(cells, pairs), pairs <= min(n_A x prod of B's other dims, n_B x prod of A's
  other dims)), bytes (header + data: dense 16 G + 16 cells, sparse 16 G + 8 (G + 1) + entries (4 m + 16)), scratch
  (sparse: 48 bytes per pair + 16 per operand entry), live bytes (converted inputs and intermediates alive, the result,
  the scratch: an intermediate is freed right after its consumer, its materialization lifetime), the peak, the output
  bytes, a magnitude bound per step (terms x bound_A x bound_B <= 2^127 - 1, else the candidate is rejected:
  `OverflowError`, refusal kind `bound`). Needs: variables (labels), degree, tensor (the largest dense array read or
  written), support (the largest sparse bound), output, bytes (the peak; CUDA: device copies, host output, converted
  inputs).
- Costs (FND-008): exact amounts per component (`scan` entries read, `chain` products, `construct` elements built and
  sorting work, `materialize` output bytes, `transfer` host <-> device bytes, `fixed` steps / launches) costed by
  `tr_planner_choose` (amount x rate, relative work units, NOT time), which also checks each candidate's needs against
  the budget and chooses the cheapest admissible one, the first on a tie, in rounds of 16 and a final round over the
  winners (up to 256 candidates). Family modules have no registry rates, so the plan passes its own
  (`mt.default_rates()`, or `rates=`) as the planner's override table, row 1, relabelling the candidates module 0 in
  the evidence. `force=i` runs exactly candidate i or refuses with its code. A plan with no admissible candidate
  raises (`AlgebraBudgetError` with the first over-budget candidate's resource, need and limit when one was over
  budget, else the first refusal's exception) with `.plan` attached (every candidate's refusal).
- Execution: `contract(plan, i)` runs any admissible candidate (CPU: `tr_tensor_net_execute`; CUDA:
  `tr_tensor_net_program` + `tr_mgpu_tensor_contract` on the dense inputs, which re-proves every step bound before
  copying) and returns the actual entries, bytes and products per step and the actual peak (dense: equal to the plan;
  sparse: within it). Every kernel returns the same tensor.

## Budgets and refusals

| request | needs |
|---|---|
| build (CPU) | variables (columns read), degree (m), tensor and states (G x cells of a dense state or scratch), states (sort: emissions), support (factored: the largest group's terms), bytes (builder, scratch, bitmap, state) |
| build (CUDA) | as the device copies (`mt.need(..., backend="cuda")`): selection, ids, columns, row counters, dense state or hash table (2^k >= 2 x capacity slots of 24, or 32 with counts, bytes), the host sort |
| sparse final state | support (the largest group), bytes (scratch + state), checked when the entries are known |
| conversion, marginal, slice, permute, outer, ungroup | the result (and 48 bytes per sorted record) |
| network | variables, bytes (ungrouped copies) |
| plan | iterations (orders searched), bytes (the plan: 1480 bytes per candidate) |
| candidate | as above; a rejected candidate keeps its amounts, costs and needs |

Refusals raise before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError` (an unproven
bound), `UnsupportedError` (order > 8, > 64 columns on an axis, 2^64 cells, a dimension past 2^32 - 1, a trace, more than
8 tensors or 16 labels, counts in a factored form), `DomainError` (an explicit domain missing a value, counts in a
network or outer product), `ValueError` (malformed input, overlapping or incomplete chunks), `StaleStateError` (the
scope's table changed).

## For the next capabilities

SYM-001 applies basis actions as contractions of a `moments` tensor with action matrices (`network`, `plan`,
`contract`); PAIR-002 and KERNEL-001 read feature states as moment and count tensors and their inner products; PROB-002
plans factor-graph elimination with `network(...).plan(budget=...)` (intermediate support budgets govern it) on count
tensors. Read `T.info["bound"]` before combining states; use `values()` before contracting SQL moments.

## Limitations

The CPU loops are single-threaded; no factored contraction kernel (factored operands are expanded, costed and
budgeted); the CUDA contraction is dense only (sparse candidates run on the CPU); sparse plan sizes are exact upper
bounds, not the data's actual supports; the rates are hand-set estimates (not calibrated); traces within one tensor are
not supported; a factored form's bound is conservative (a sum over its terms), so some factored operations refuse what
the dense form would compute; entries are limited to 128 bits.
