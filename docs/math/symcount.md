# Symbolic full-domain model counting (COUNT-002)

`torbles/math_symcount.h` (every contract; the exact L-limb arithmetic, the diagram count and the brute-force word
functions are inline and shared with CUDA), `math_symcount.c`, `math_symcount_gpu.cu`, `math_symcount.py` (opt-in).
Exact model counts and weighted model counts over a **declared assignment universe**, never over stored rows (those
are COUNT-001's, [count.md](count.md)): compilation of BOOL-001 predicates ([bool.md](bool.md)) to reduced ordered
decision diagrams, variable elimination over finite-domain factor tables, witnesses, a brute-force verifier and a
bounded Shannon expansion that returns certified bounds instead of an extrapolation.

```python
from fractions import Fraction
from torbles import math_bool as mb, math_symcount as ms
x = mb.variables(4)
f = (x[0] & x[1]) | (x[2] ^ x[3])                        # UNKNOWN constants allowed (three-valued)
d = ms.compile(f, k=4, order="dfs", budget=Budget(circuit=1 << 16, iterations=1 << 20))   # or "natural", a permutation
c = d.count()                     # Count: domain "universe", assignments 16, true 10, false 6, unknown 0, total 16
c = d.count(weights={0: (Fraction(1, 3), Fraction(2, 3)), 2: (2, 1)}, backend="cuda")   # (w(x=1), w(x=0)) exact
d.witnesses(5); d.witnesses(5, value="false"); d.info; d.order; d.raw(weights, backend)
ms.brute(f, k=4, weights=w, backend="cuda")              # every assignment (verification; support budget 2^k)
s = ms.search(f, k=4, max_evaluations=100)               # s.status "exact" | "incomplete"; s.true == (lo, hi)
fs = ms.Factors([2, 3, 2], [((0, 1), [[1, 0, 2], [Fraction(1, 2), 1, 1]]), ((1, 2), table)])
r = fs.count(order="min-fill", weighted=True, witnesses=3)   # r.value (Z), r.assignments, r.witnesses, r.order
fs.count(weighted=False); fs.brute(); fs.order("min-degree"); fs.width([2, 1, 0]); fs.need(...)
```

## The universe and its three classes

A program over k variables (k <= 64) is TRUE, FALSE or UNKNOWN at each of the 2^k assignments (Kleene's logic;
UNKNOWN only from the UNKNOWN constant): TRUE + FALSE + UNKNOWN = TOTAL in every weighting. A factor system's
universe is the prod d_i assignments of n finite-domain variables. Every result says `domain = "universe"`, its
`universe` (variables) and `assignments`; the observed-row counts of COUNT-001 are a different quantity (the
benchmark records both side by side, with the observed coverage of the 2^k assignments).

## Weights (exact)

Literal weights `(w(x=1), w(x=0))` are nonnegative exact rationals (int64 numerator and denominator; floats refused);
missing variables weigh `(1, 1)`. Natively each variable is scaled to integers a_v, b_v over d_v = lcm(den1, den0)
(< 2^126) and the count N / prod d_v is computed in L 64-bit limbs where the host proves every partial count below
prod (a_v + b_v) < 2^(64 L) first, then reduced with FND-009 big integers: `int` or `Fraction` in Python. Probabilities
are the pairs (p, 1 - p) (total weight 1). Dependent weights are factor tables: each entry a nonnegative rational, a
zero a hard constraint, Z the sum over the universe of the products of the entries (each factor scaled over the lcm
of its denominators, < 2^126).

## Decision diagrams

Reduced ordered BDDs (hash-consed nodes, lo != hi), apply AND / OR / XOR by an iterative Shannon recursion with a
computed table (NOT is XOR with TRUE). Two-valued programs compile one root (FALSE = TOTAL - TRUE), three-valued ones
the pair (T, F) of Boolean functions "is TRUE" and "is FALSE" (NOT swaps them; AND = (Ta Tb, Fa | Fb); OR dually;
XOR = (Ta Fb | Fa Tb, Ta Tb | Fa Fb)). Orders: `natural` (level i = variable i), `dfs` (first appearance from the
root), or a permutation; the diagram of a function under an order is canonical (`info["nodes"]`), and the order can
change its size exponentially (OR of x_i AND x_(i+n): 2n + 2 nodes interleaved, 2^(n+1) split). The immutable
diagram is renumbered from the deepest level so that the count runs node by node on the CPU and one launch per level
on the device: V(node) = a_l V(hi) G + b_l V(lo) G with G the weight sums of the levels an edge skips. Witnesses:
the first m assignments of the TRUE (or FALSE) class in lexicographic order of the levels (level 0 most significant,
0 before 1), as `{variable: 0 / 1}`.

## Variable elimination

Order `min-fill` (default), `min-degree` (greedy, ties to the smallest index, n <= 4096) or a permutation; buckets by
first-eliminated variable, messages over the union of the scopes, constants multiplying Z. The width (the largest joint
table, `fs.width(order)`) is checked against the tensor budget before anything is computed. `weighted=False` counts
the models (every nonzero entry as 1). Witnesses: the first models in lexicographic order of the reverse elimination
order (the last-eliminated variable most significant), decoded without backtracking from the messages.

## Brute force and bounded search

`ms.brute` enumerates the 2^k assignments natively (64 per word, BOOL-001's word evaluator) and `fs.brute` the
factor universe: verification oracles inside the library. `ms.search` is a breadth-first Shannon expansion with an
evaluation budget: each branch evaluated once with sets of possible values (an over-approximation, so a singleton is
exact); undecided branches split; at the budget the remaining branches stay open and add their whole weight to the
upper bound of every class still possible. `status` is `"exact"` (lo = hi) or `"incomplete"` with certified
(lo, hi) per class, `info["open"]` branches and `info["evaluations"]`: never an extrapolation.

## CUDA

`tr_mgpu_symcount_levels`: one launch per level (deepest first), one work item per node into its own L-limb slot
(`tr_symc_node_count`), then one per root; no atomics. `tr_mgpu_symcount_brute`: work items over slices of 64-point
words; counts with one 64-bit atomic each, weighted sums with a multi-word atomic add (each word's carry from the value
its own atomic returned: exact modulo 2^(64 L), exact because the host proved the bound). Readbacks are adopted
natively (`tr_symc_dd_adopt`, `tr_symc_brute_adopt`, consistency-checked); the raw words equal the CPU's
(`Diagram.raw`, `Count.raw`; emulated with a scrambled thread order in the tests). Elimination and search are CPU only.

## Budgets and refusals

| request | needs (checked before allocation) |
|---|---|
| compile (manager) | variables = k (programs), circuit = nodes held (refused at the first node past the limit: need = that node), iterations = apply steps, bytes = node / hash / cache / stack arrays at every growth |
| count | circuit = nodes, output = roots, bytes = nodes x L x 8 (CUDA: every device copy and the widest level's scratch) |
| witnesses | output = m |
| brute force | variables = k, support = 2^k, circuit = program nodes, bytes (CUDA: device copies, per-item scratch) |
| search | iterations = max_evaluations, circuit, bytes = the branch queue |
| factors | variables = n, support = entries, tensor = the largest table; elimination: tensor = width, iterations = products, bytes = every message (L limbs per entry), output = 2 + witnesses; factor brute force: support = prod d_i |

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `UnsupportedError` (k > 64, 63 for brute force; scopes past
31 variables, domains past 2^16, factor denominators or scaled entries past 2^126, counts past 4096 limbs);
`ValueError` (malformed orders, programs, tables, weights); `TypeError` (floats); `OverflowError` (an int64
numerator / denominator, or a carrier overflow, never wrapped).

## Limitations

Programs over at most 64 variables (witness masks); no complement edges, no dynamic reordering (the order is given or
the DFS heuristic) and no garbage collection inside one compilation (the node budget counts every node held). Factor
entries are nonnegative (no signed weights); a formula and dependent weights combine as factors (CNF clauses as 0/1
tables), not as weighted diagrams. The bounded search is breadth first with a fixed order (no caching of equal
residual formulas) and its bounds can be loose; elimination and search run single-threaded on the CPU.
