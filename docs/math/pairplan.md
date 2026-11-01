# Structured pairwise query planning (PAIR-002)

`torbles/math_pairplan.h` (every contract; the kernel values, the pair-term evaluator, the tiles and the request finish
are inline and shared with CUDA), `math_pairplan.c`, `math_pairplan_gpu.cu`, `math_pairplan.py` (opt-in). A pairwise
statistic S = sum over a declared pair set of w_i w_j k(x_i, x_j) is planned over four exact strategies, each costed
through the FND-008 planner (`docs/math/planner.md`), and answered by the cheapest admissible one; every strategy
returns the same exact value.

```python
from torbles import math_bank as mbk, math_kernel as mk, math_pairplan as pp
from torbles.math_budgets import Budget
scope = mbk.Scope.from_arrays({"x": x, "y": y, "w": w}, groups=g, ngroups=G)      # or Scope.from_table(...)
reqs = [dict(a=0, b=1), dict(a=0, ordered=True, self_pairs=False, mean=True)]
plan = pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", devices=("cpu", "cuda"), budget=Budget(iterations=10**7))
plan.candidates       # name, status, reason, refusal, need, amounts, costs, total, evaluations, exact, reuse, ...
out = plan.run()      # the chosen candidate; plan.run("direct@cuda"), plan.run(i, raw=True)
out["results"]        # [{"value" (int / Fraction / None), "num", "den", "pairs", "empty", "evaluations"}]
pp.query(scope, ["x", "y"], pp.features(mk.polynomial(2, c=1)), reqs, force="support@cuda")
st = pp.retain(scope, ["x"], "support", weight="w")          # "rows", "prefix", "features" (kernel=, repr=)
pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", retained=[st])   # reuse: construct 0, exact sizes
st.entries(g); st.totals(g); st.info; pp.eval_table(st, plan, backend="cuda")
pp.table({(1, 2): 5}, default=0, symmetric=True); pp.band(h); pp.sqdist(); pp.mann_whitney()
pp.kernel("rbf", gamma=1)    # no exact value: refused (UnsupportedError), never approximated
```

## The statistic

- Elements: the selected rows of a scope group whose weight is present (no `weight`: w = 1) and, LISTWISE, whose feature
  values are all present; the delta kernel's policy is CATEGORY (NULL a category of its own, matching NULL). Values are
  the scope's exact logical integers (a distance needs one decimal scale across its columns, `DomainError` otherwise).
- Pair set of a request `dict(a, b=a, ordered=False, self_pairs=False, mean=False)`:
  within one group (a == b) the ordered pairs (i, j), i != j (with `self_pairs` every (i, j)), or each unordered pair
  {i, j} once (with `self_pairs` also {i, i}), which needs a symmetric kernel (asymmetric: that request's status is
  invalid, `ValueError` when formatted); between two groups each cross pair (i in a, j in b) once, k(x_i, x_j) oriented a first, and with
  `ordered` both orientations (k(x_i, x_j) + k(x_j, x_i)). `self_pairs` adds nothing between groups.
- P = sum over the pairs of w_i w_j: between W_a W_b (ordered 2 W_a W_b); within, ordered W^2 - Q (with self-pairs
  W^2), unordered (W^2 - Q) / 2 (with self-pairs (W^2 + Q) / 2), W = sum w, Q = sum w^2 (unweighted n (n - 1), n^2,
  n (n - 1) / 2, n (n + 1) / 2). `mean` gives S / P (None when P = 0).

| kernel | k(x, y) | strategies |
|---|---|---|
| `absdiff()` | sum_k abs(x_k - y_k) | support, prefix (one column), direct |
| `mann_whitney()` | [x > y] + 1/2 [x = y], one column, asymmetric | support, prefix, direct |
| `features(mk.polynomial(d, c) / monomial(...) / delta())` | KERNEL-001's kernels (`docs/math/kernel.md`) | support, features, direct |
| `table(pairs, default, symmetric)` | a declared int64 value per pair of value tuples, `default` elsewhere | support, direct |
| `band(h)` | [max_k abs(x_k - y_k) <= h] | support, direct |
| `sqdist()` | sum_k (x_k - y_k)^2 | support, direct |
| `kernel("rbf" / "laplace" / ...)` | no exact value | refused (`UnsupportedError`) |

## Strategies

- SUPPORT (CPU, CUDA): per group the distinct value tuples u sorted (NULL after every value) with multiplicity m(u),
  W(u) = sum w, Q(u) = sum w^2; S = sum over entry pairs of mult(u, v) k(u, v), mult = W_a(u) W_b(v) off the diagonal
  and on it W(u)^2 - Q(u), W(u)^2, (W(u)^2 -+ Q(u)) / 2 (m (m - 1), m^2, m (m -+ 1) / 2 for unit weights).
- FEATURES (CPU, dense or sparse): the KERNEL-001 feature state and its SUM requests (K, K - Dg, (K -+ Dg) / 2;
  both orientations: two requests).
- PREFIX (CPU): PAIR-001 over the DIST-002 prefix state of the column (count mass, or weight mass when weighted):
  ABS_WITHIN / ABS_CROSS (the diagonal adds 0; ordered: twice), MANN_WHITNEY with self-pairs (2U - Q without them;
  both orientations: two requests). Needs ABS or Mann-Whitney, one column, weights >= 0, every class mass < 2^64.
- DIRECT (CPU, CUDA): the same evaluator over one entry per element (W = w, Q = w^2), within the pair budget.

Every candidate is listed; one without the structure for the kernel is rejected with refusal kind `unsupported` (so
an arbitrary similarity keeps exactly the support and direct candidates), negative weight masses with `DomainError`,
an unproven magnitude with kind `bound`.

## Exactness

A request runs only after its magnitude is proven: bits(sum_a |w|) + bits(sum_b |w|) + bits(max |k|) + 2 (+ 1 for both
orientations) <= 1020, max |k| bounded from the elements' value extremes. Requests proven within 126 bits are summed in
128-bit words, the others in 1024-bit words: the same exact integers. Answers are num / den in 1024-bit two's
complement (den 1, Mann-Whitney 2; means den x P with den > 0); Python returns `int` / `Fraction`. FEATURES and PREFIX
carry KERNEL-001's and PAIR-001's proofs. Nothing wraps: an unprovable support or direct candidate is rejected
(`OverflowError` when no candidate is left).

## Planning and costs

`plan` reads the scope once for the exact profile (per group n, W, sum |w|, Q, the value extremes), or a retained
table's totals. Distinct supports are exact with a retained state, else exact upper bounds (min of n and the value
box's cells). Candidate needs (FND-010): `iterations` = kernel evaluations (support and direct: entry pairs visited,
d^2 / d (d + 1) / 2 within, d_a d_b between; prefix: support entries read; features: items read) — the declared pair
budget; `support`, `states`, `output`, `bytes`, `degree`, `tensor`; CUDA candidates add the device copies. Amounts per
FND-008 component follow the work model of `math_pairplan.h` (words, scan, construct for a state build; reuse 1 for a
retained one; chain = kernel operations; materialize, transfer, finalization, fixed calls and launches);
`tr_planner_choose` costs them with `pp.default_rates()` (or `rates=`: relative work units, not time) and chooses the
cheapest admissible candidate, the first on a tie. `force="<strategy>[@cpu|@cuda][:dense|:sparse]"` runs exactly its
cheapest admissible match or raises; a plan with no admissible candidate raises (`AlgebraBudgetError` when one was over
budget, else the first rejection's exception) with `.plan` attached.

## CUDA

`backend` follows the candidate: a support or rows table is uploaded once per state (`State.device()`,
`tr_mgpu_pairplan_upload`), then `tr_mgpu_pairplan_query` runs one work item per (request, 32 x 32 tile of entry
pairs) with the shared `tr_pairplan_tile` into a local sum added to the request's 1024-bit accumulator by a 16-word
atomic carry chain, and one work item per request with the shared `tr_pairplan_finish`; the results equal
`tr_pairplan_eval_table` bit for bit. The host checks every request and the budget (output, iterations, the batch's
device bytes: 580 per request, 1104, and 8 (2 ncols + 1) per similarity-table pair; the upload's: the table and host
copies of its offsets and totals) first.

## Budgets and refusals

| request | needs |
|---|---|
| plan | output = requests; bytes = the plan (and a similarity table's copy: support = its entries) |
| support build (`retain`, transient) | states = elements; bytes = records and sort indices, then + the table (support = the largest group) |
| rows build | states = elements; bytes = the table |
| prefix / features builds | the plan: DIST-001's first stage plus bounds of the final states, KERNEL-001's needs; the builds check their own |
| evaluation (CPU, CUDA) | output = requests, iterations = visits; CUDA bytes = device copies |

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError`, `UnsupportedError` (a kernel without an exact
value, more than 8 feature columns, a forced strategy without the structure), `DomainError` (mixed scales, negative
weight masses for the prefix), `ValueError` (malformed specs, a pair declared twice in a table, invalid requests in
their status), `StaleStateError` (the scope's table changed). Retained states hold their scope by identity (it must
outlive them); a state of another scope or other columns, weight or policy is not reused.

## Limitations

The planned statistics are sums over pairs: PAIR-001's distribution distances (KS, 1-Wasserstein, Cramer-von Mises)
are not, and stay PAIR-001's own API; no FND-007 certificate is issued (values are computed, no data relation is
proven). The feature and prefix strategies run on the CPU (their CUDA paths remain KERNEL-001's and PAIR-001's own
APIs); support tables are sorted on the host for both devices; distinct supports without a retained state are upper
bounds; the rates and the work model's operation counts are hand-set estimates, not calibrated times; at most 8 feature
columns; answers limited to 1024 bits; the CUDA tile kernel uses 255 registers with a 576-byte stack frame and spills;
the CPU loops are single-threaded. The benchmark (`bench/benchmark_math_pairplan.py`) times the CPU, NumPy and DuckDB
here and CUDA only on a device (`--device`).
