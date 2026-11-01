# Fused multi-output basis reduction (FND-006)

`torbles/math_reduce.h` (the contract; the compiled view and the per-word helpers are shared with CUDA),
`math_reduce.c`, `math_reduce_gpu.cu` (the fused pass on the device), `math_reduce.py` (opt-in). Several basis states
over one FND-004 scope, moment banks of different bases and DIST-001 histograms, each with its own predicate and
grouping, accumulated in one traversal of the rows; the competing plan that runs the independent reductions one by
one is preserved, costed and selectable.

```python
from torbles import math_bank as mbk, math_bool as mb, math_reduce as mr
scope = mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"], group_by=["k"], rows=mask)
req = mr.Request(scope, conditions=[mb.condition("x", ">", 0), mb.condition("d", "is null")],
                 groupings={"c": mbk.Scope.from_table(db["f"], [], group_by=["c"], rows=mask),   # same rows
                            "all": mr.rollup(mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"], rows=mask)),
                            "two": mr.rollup(map=ids, groups=2)})
pos, dn = mb.variables(2)                              # predicates over the conditions (math_bool expressions)
req.bank("A", ["1", "x", "x*d", "d^2"], where=pos)                 # FND-004 basis, mode "sql" | "joint"
req.count("B", where=pos & ~dn, group="c")                          # rows only
req.histogram("H", ["x"], weight="w", where=dn, group="all")       # DIST-001: repr "sparse" | "dense", domain=
plan = req.compile()                    # .info: merged conditions/nodes, products vs term rows, states, arenas
plan.choose(devices=("cpu", "cuda"))    # every candidate with amounts, costs, status, reason
res = plan.run(force="repeated@cpu")    # or plan.run() (cheapest admissible), plan.execute("fused", "cuda")
res["A"].state(g); res["A"].table(); res["H"].cells(g); res["H"].group(g); res.stats; res.plan; res.equal(other)
plan.need("fused" | "repeated", "cpu" | "cuda")   # what execute checks first; ("fused", "kernel"): the CUDA entry's own
```

## Semantics

Output o is exactly the independent reduction of its derived scope S_o: the base scope's selected rows where o's
predicate is TRUE (SQL WHERE: FALSE and UNKNOWN rows do not count), grouped by o's grouping; a bank is
`tr_bank_build(S_o)` over o's basis in o's mode (FND-004: `sql` per-term validity, `joint` listwise), a histogram
`tr_dist_hist_build(S_o)` (DIST-001; a dense output over the base scope's domain [min, max] with a NULL slot iff the
base scope has a selected NULL, or over an explicit domain). Every output keeps its own bag multiplicity per group,
its own validity counts (per term; per cell, weight count and weight sum) and its own group presence (present iff
it has rows; an all-NULL present group has rows and zero counts; an absent group has nothing). An empty selection,
a never-TRUE predicate or an empty group gives zeros, never an error. Groupings: the scope's own, the groups of
another scope of exactly the same rows (the same source identity and version, row count and selection bits; array
scopes aligned by position), or either of them mapped onto coarse groups (a rollup: `rollup(coarse_scope)` uses
`Scope.rollup_map`; `rollup(map=, groups=)` any map). Predicates are BOOL-001 conditions and Kleene connectives.

## The two plans

- **fused** (one traversal): per 64-row word every condition the scheduled nodes read and every node reachable from
  a predicate is evaluated once; each output's mask is the selection AND its predicate's TRUE word; per selected row
  of their union the column presence and each grouping's id are computed once, and each distinct monomial of all
  bank bases ("product") at most once, when the first output term valid for it needs it (a per-row stamp), feeding
  every term that names it. Structurally equal conditions and nodes are merged at compile time (`merged_conds`,
  `merged_nodes`), so equal predicates are one mask. `info["term_rows"] - info["nproducts"]` products are shared.
- **repeated** (the preserved competing plan): per output in turn, its predicate by BOOL-001 masks and a word
  evaluation, its derived scope (`tr_reduce_derive`: the columns it reads, its grouping's ids, the TRUE rows), then
  `tr_bank_build` / `tr_dist_hist_build`; on CUDA, FND-004's and DIST-001's own CUDA builds of each derived scope
  (`math_bank.build` / `math_dist.histogram` with `backend="cuda"`), the derived scopes and predicates on the host,
  orchestrated from Python through native calls (each state copied into the stage, `tr_reduce_stage_put_*`).
- CUDA fused (`tr_mgpu_reduce_fused`): one kernel; a work item is a contiguous slice of at least 4 words and produces
  every output's contribution of its rows (the same per-word helpers as the CPU loop); bank outputs accumulate
  locally while the output's group stays the same and flush with integer atomics (128-bit sums with the two-word
  atomic), histogram cells add one atomic per row. Each item owns its scratch slot; no barrier. The arenas come back
  into a stage the host adopts after consistency checks; `res.backend` is `"cuda"` on a device and
  `"cuda-emulated"` in the emulation. Sparse outputs are compacted on the host.

All four variants give identical states (tested against the oracles and each other).

## Exactness and atomicity

Compile proves every bank term N * prod max(|min|, |max|, 1)^e <= 2^127 - 1 and every weighted histogram selected x
max |w| <= 2^127 - 1 from the base scope's extremes (bounds for every S_o); otherwise `OverflowError` with `.failed`
= the output, before anything is computed. A run returns every output or nothing: a sparse final state over budget
(known after counting), a repeated output whose own reduction refuses, a device or adoption failure free everything
(`.failed` names the output when one is to blame). Adoption checks rows <= selected, count <= rows, a zero sum
without valid rows, |sum| <= count x the product bound, the constant term (= count, = rows in `sql`), equal counts of
every term in `joint`, histogram rows = their cells, weight counts and bounds (`ValueError` otherwise).

## Plan choice (FND-008 conventions)

`plan.choose(devices, force, coefficients, budget)` builds the candidates (fused, repeated) x (CPU, CUDA) as FND-008
`tr_reg_candidate`s: need, exact amounts per component (words, scan, chain, construct, fusion, materialize,
transfer, cpu_final, fixed), costs = amount x coefficient (hand-set relative estimates, `cost_basis` "estimate",
overridable with `mr.coefficients({(plan, device): {component: rate}})`), status chosen / admissible / rejected with
the reason. The fused candidate counts its shared components (words, scan) for `batch` = outputs unfused passes and
its fusion component subtracts (batch - 1) / batch of them, as FND-008 does. The cheapest admissible candidate runs
(the first on a tie). `force="fused|repeated[@cpu|@cuda]"` (or `TORBLES_MATH_REDUCE_FORCE`) runs the cheapest
admissible match or refuses with the first match's code (`AlgebraBudgetError` over budget, `UnsupportedError` when
nothing matches; `.plan` carries the evidence); it is never replaced. A force naming `@cuda` adds the CUDA
candidates. A fused plan over budget (e.g. a sparse histogram over a domain too wide to count densely) is rejected
while the repeated plan (DIST-001 sorts) stays admissible.

## Record of a run (`res.stats`)

`scans` (traversals: fused 1; repeated per output: masks when filtered, the derived scope, the reduction),
`kernels` (CUDA launches: fused 1; repeated, those of FND-004's and DIST-001's device builds: rows, JOINT flags,
terms; rows, cells and, sparse, occupancy, scan and compaction; none for an empty derived scope), `native_calls`,
`decoded_bytes` (fused: column data, validity, group ids, maps and selection words read, on CUDA copied to the
device; repeated: the derived scopes' columns), `workspace_bytes` (fused CPU: the scratch; fused CUDA: the device
copies and per-item scratch; repeated: the plan's need), `transfer_bytes` (host <-> device: fused, the copies and
the arenas back; repeated, each device build's copies as its own need formula counts them on its derived scope),
`products` (product evaluations), `accumulations`, `rows_visited`, `words_visited`, `items`, `state_bytes`, `plan`,
`device`.

## Budgets and refusals

`plan.need(plan, backend)` is what `execute` checks before it allocates anything (natively: `tr_reduce_run` on the
CPU, `tr_math_budget_check` of `tr_reduce_need` before the CUDA paths) and an upper bound of every later check of
the run, except the occupied cells of sparse histograms, which are checked once counted (the fused plan's compacted
states at the end; the repeated plan's copy of each sparse state as it is put, and at the end). So for a request
without sparse histograms a run at its need succeeds and one unit less of any resource refuses before anything is
allocated (tested for every resource on all four variants); the planner rejects a candidate whose need is over
budget with that refusal.

| check | needs |
|---|---|
| compile, before allocating | circuit = conditions + nodes of the request, output = outputs; after: bytes = the plan |
| fused CPU (`need("fused")`) | variables = columns read, degree, support = all bank terms + the largest histogram's cells per group, circuit = conditions + scheduled nodes, output, states = rows, term counts and histogram cells of every output (sparse: the counting scratch), bytes = plan + stage (arenas) + CPU scratch + the result but for sparse cells |
| fused CUDA (`need("fused", "cuda")`) | the fused CPU need with bytes = the device copies (view, selection, the columns read and their validity, each group-id array once, maps, tables, arenas) and the per-item scratch + plan + stage + the result's dense part; `("fused", "kernel")`: what `tr_mgpu_reduce_fused` itself checks (the device part) |
| repeated (`need("repeated", ...)`) | per output its own reduction's checks, bounded from its derived scope's shape (the base scope's rows, its grouping, int64 columns; a sparse histogram's occupied cells by min(G x cells, selected rows)): variables, degree, support = the largest single check (masks: the conditions of its filter; FND-004: basis columns, degree, terms; DIST-001: coordinates + weight, cells), states = the stage (every output's rows, bank states, dense histograms) + the largest single output's (FND-004 G x terms; DIST-001 dense G x cells, sparse its COUNT scratch min(G x cells, 4 S + 65536), CUDA G x cells), bytes = plan + stage + the result's dense part + the largest output's derived scope, masks and reduction (CUDA: FND-004's / DIST-001's device need of its shape and the adopted state); circuit, output as fused |
| finish | states and bytes of the final states, sparse cells counted |

Limits (`UnsupportedError`): 1024 outputs, 16 groupings, 64 conditions, 65536 nodes, 65536 bank terms, 4096
distinct products, FND-004 / DIST-001 limits per output. Other refusals: `ValueError` (malformed input, a duplicate
term, misaligned groupings, a map entry out of range, inconsistent adopted states), `DomainError` (an explicit domain
not holding the values), `OverflowError`, `StaleStateError` (a table scope or grouping scope changed). A native
refusal carries `.failed` (the output when one is to blame, else None) and `.plan` (the planner's evidence, when it
chose).

## Limitations

The fused histogram outputs count in a dense scratch over the base scope's domain (no sorting strategy: a domain too
wide is over budget for the fused plan, the repeated plan handles it on the CPU; DIST-001's device build has the same
bound). The overflow proof uses the base scope's extremes for every output (a filtered output whose own rows would
prove a tighter bound is still refused). The repeated plan's needs are bounds from the derived scopes' shapes (the
base scope's rows, int64 columns, every sparse cell bounded by the selected rows), so they can exceed what a
selective output then uses. The repeated CUDA plan evaluates predicates and builds derived scopes on the host and is
orchestrated from Python (native calls only; its kernel count follows FND-004's and DIST-001's launch structure).
Costs are relative estimates, not calibrated times. The CPU paths are single-threaded.
