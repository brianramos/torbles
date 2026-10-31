# Predicate cofactors and finite differences (BOOL-001)

`torbles/math_bool.h/.c` (native CPU), `math_bool_gpu.cu` (CUDA kernels), `math_bool.py` (opt-in Python API). A
predicate is a DAG over k conditions; on a declared domain the module counts, exactly, where the predicate, its
cofactors, its XOR (Boolean) derivatives and its signed first and mixed differences are TRUE / FALSE / UNKNOWN or take
each integer value. The header comment of `math_bool.h` is the normative contract; this page summarizes it.

```python
from torbles import math_bank as mbk, math_bool as mb
x = mb.variables(3)
shared = x[0] & x[1]                                   # a Python object used twice is one shared node
prog = mb.Program((shared | ~x[2]) ^ (shared & mb.UNKNOWN), k=3)
prog = mb.Program.from_nodes(3, [("var", 0), ("var", 1), ("and", 0, 1), ...], root=None)
uniform = mb.Masks.assignments(3)                      # encoded variables: the 2^3 assignments
scope = mbk.Scope.from_table(db["t"], ["x", "d", "y"], rows=mask)
rows = mb.Masks.from_scope(scope, [mb.condition("x", ">", 5), mb.condition("d", "between", "1.5", "3.25"),
                                   mb.condition("x", "<", mb.column("y")), mb.condition("y", "is null"),
                                   mb.condition("x", "bit", 3, base=-9)], backend="cpu")    # or "cuda"
mb.differences(rows, prog, [[0], [1], [0, 2]], cofactors=True, backend="cpu")          # or "cuda"
mb.first_differences(rows, prog)            # S = {i} for every condition
mb.cofactors(rows, prog, 1)                 # (f|x1=0, f|x1=1) counts
mb.direct(prog, {0: 1}, scope=scope, conditions=conds)   # one cofactor evaluated independently (the baseline)
mb.masks_need(scope, conds, backend), mb.differences_need(masks, prog, queries, cofactors=True, backend=...)
```

Each query result: `conditions` (S), `order` (m), `points`, `f` (TRUE / FALSE / UNKNOWN counts of the program),
`known` / `unknown`, `xor` (`true` / `false` of the XOR derivative on the known points), `signed` (a histogram
`{value: count}` of the signed difference on the known points, values -2^(m-1) .. 2^(m-1)), `sum` (exact),
`nonzero`, `cofactors` (per assignment t of S: `assignment`, `true`, `false`, `unknown`), `backend`, `device`.

## Domain

- **Assignments** (`Masks.assignments(k)`, k <= 63): the points are the 2^k assignments; condition i is bit i of the
  assignment, always known; the uniform measure (each point counts once; total 2^k). The words are generated, never
  stored.
- **Rows** (`Masks.from_scope(scope, conditions)`, k <= 64): the points are the scope's physical rows; only its
  selected rows count, each once (the empirical measure of the bag of selected rows). The scope's groups are ignored.
  Each condition is a native row predicate evaluated once into a value and a valid bitmap over the scope's exact
  logical values (decimal value = logical / 10^scale; integer, decimal and Boolean columns, as the scope admits):
  `= <> < <= > >=` against an exact rational constant (int, Fraction, Decimal or decimal string; floats refused) or
  another column (`column("y")`, scales aligned exactly), `between` (inclusive), `is null`, `is not null`, `bit b`
  (bit b of (logical - base) mod 2^64). Comparisons are exact 128-bit cross products. A condition other than
  `is [not] null` is UNKNOWN on a row where its column is NULL.
- The probability of a count under the domain's measure is count / points (exact rationals; the API returns the
  integers).

## Semantics

- SQL three-valued logic: AND, OR, NOT are Kleene's (SQL's); XOR(a, b) is UNKNOWN when either side is, else a != b
  (SQL's `a <> b` on booleans); `UNKNOWN` is SQL's NULL literal. This is SQL's evaluation, not a supervaluation:
  `x OR NOT x` is UNKNOWN where x is.
- Cofactor `f|S:=t`: the conditions of S replaced by the constants of t (bit j of t: the j-th condition of S in
  increasing order), evaluated on the other conditions. It never reads the conditions of S, so a row where a condition
  of S is NULL has definite cofactors wherever the other conditions they read are known.
- XOR derivative `dS f` = XOR of the 2^m cofactors; signed difference `DS f` = sum of (-1)^(m - |t|) [f|S:=t]
  (m = 1: `f|1 - f|0` in {-1, 0, 1}). `dS f = DS f mod 2`: the XOR derivative counts the points with odd `DS f`, not
  those with `DS f != 0` (x0 XOR x1 over {0, 1}: `DS = -2`, `dS = 0` everywhere).
- Strictness: a difference is UNKNOWN at a point where any of its cofactors is; such points are counted in `unknown`
  and in no other difference count. UNKNOWN is never read as FALSE or as 0. `f_true + f_false + f_unknown = points`,
  `known + unknown = points`, `xor.true + xor.false = known`, the histogram sums to `known`.
- Exactness: every result is an integer count (128-bit for `sum`). CPU, emulated CUDA and the device agree bit for
  bit, in any thread order.

## Execution

- Shared-mask execution (`tr_bool_differences`, `tr_mgpu_bool_differences`): the conditions are evaluated once (the
  masks); per 64-point word the program's reachable nodes are evaluated once with bitwise operations on a value word
  and a valid word, then for each query and each assignment of S only the cone of S (the nodes that read S) is
  re-evaluated with the conditions of S as constant words. The 2^m root words are combined by bit-sliced counters
  (signed sums), split into value classes and counted with popcounts. Words without a selected point are skipped.
- CUDA: masks, one work item per (condition, word); differences, work items (query, word slice) with 16 bytes of node
  words per program node each, local totals and integer atomics (histogram bins and cofactor counts per word, totals
  once per work item, the 128-bit sum with a two-word atomic).
- Independent evaluation (`tr_bool_direct`, `mb.direct`): one cofactor at a time, point by point, conditions re-read
  from the scope's values, scalar three-valued logic: the reference baseline of the benchmark.

## Budgets and refusals

Checked before anything is allocated; a refusal raises `AlgebraBudgetError` with `.resource`, `.need`, `.limit`.

| request | needs |
|---|---|
| masks over rows | variables = k; bytes = object + 8 nw (1 + 2k) (CUDA: + the device copies of the columns read) |
| assignment domain | variables = k; support = 2^k (default limit 2^20: k <= 20) |
| program | variables = k; circuit = nodes; bytes = object + 29 bytes per node |
| differences | variables = k; degree = largest m; circuit = nodes; support = 2^k on assignments; tensor = sum of 2^m; output = queries + bins + 3 x cofactors; bytes = CPU scratch (plans, cones, 16 per node) or the device copies and scratch |
| direct | variables = k; circuit = nodes; support = 2^k on assignments; bytes = 2 nodes + k |

Representation limits (`UnsupportedError`): k > 64 (rows) or > 63 (assignments), more than 2^24 nodes, order m > 20,
more than 2^16 queries. Malformed input (`ValueError`): forward references, unknown ops, conditions >= k, bad
constants (den < 1, bit > 63), repeated conditions in a query. A program over another k than the domain:
`DomainError`. A rows domain whose scope changed (`t.flip`, a new selection): `StaleStateError` (checked before
every use).

## For the next capabilities

BOOL-002 (influence, interactions) reads `tr_bool_differences(_view)` with S = {i} (influence = `xor_true` / points
under either measure) and higher orders; LATTICE-001/002 read the condition words of a `tr_bool_masks`
(`tr_bool_masks_get_domain` -> `tr_bool_domain`, `tr_bool_var_word`, `tr_bool_selection_word`) and the program views
(`tr_bool_program_get_view`, `tr_bool_eval_base`, `tr_bool_eval_cone`). The word evaluators, the bit-sliced
accumulator (`tr_bool_acc_*`), the plan builder (`tr_bool_plan_queries`) and the view checks
(`tr_bool_program_check`, `tr_bool_domain_check`) are inline in the header and shared by CPU code and kernels.

## Limitations

The empirical measure is over all selected rows (no per-group measures); conditions read only columns with exact
logical values (no string or date predicates: the scope refuses non-affine codecs); a constant must fit an int64
numerator and denominator; the assignment domain is two-valued (UNKNOWN only through the UNKNOWN constant); the CPU
paths are single-threaded.
