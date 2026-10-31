# Intervals, affine forms and congruence reasoning (UNC-001)

`torbles/math_unc.h` (every contract; the abstract semantics shared with CUDA), `math_unc.c`, `math_unc_gpu.cu`,
`math_unc.py` (opt-in). Certified abstract states over the groups of an FND-004 scope: every fact is sound, what
cannot be proved is "unknown", and a pruning decision is never a guess.

```python
from torbles import math_bank as mbk, math_unc as mu
x, y, d = mu.column("x"), mu.column("y"), mu.column("d")
scope = mbk.Scope.from_table(t, ["x", "y", "d"], group_by=["k"], rows=mask)
state = mu.State.from_scope(scope, ["x", "y", "d"])            # per group and column: rows, present, min, max, gcd
prog = mu.Program((x + y) - y > 40, ["x", "y", "d"])           # carrier="exact" | "binary64", precision, rank, budget
ev = state.evaluate(prog, backend="cuda")                      # ev[g]: decision, outcomes / interval, congruence, ...
bits, stats = state.filter(scope, prog)                         # ALL kept, NONE pruned, UNKNOWN rows evaluated exactly
mu.rows(scope, prog, decisions=ev.decisions, backend="cuda")    # the concrete fallback alone
state.trace(prog, g); mu.symbolic(prog)                         # every node's value; the symbolic affine form
mu.identity(mu.Program(x - x, ["x"]))                           # FND-007 law: ring identity x - x = 0
state.certify(scope, mu.Program(x, ["x", "y", "d"]), "congruence", modulus=7)   # FND-007 snapshot (also
                                                                # "not_null", "constant"), re-verified on every row
mu.meet((1, 3), (2, 5)); mu.sum_fits(ev[g], bits=64, unit=1); mu.need(prog, groups, backend="cuda")
state.fresh(scope)                                              # FND-007 record comparison: (fresh, reasons)
```

## Domains (a reduced product per numeric node)

| domain | value | carrier |
|---|---|---|
| nullness | may be NULL / may have a value (SQL: an operation with a NULL operand is NULL) | bits |
| interval | `[lo, hi]`, closed, possibly unbounded | exact int64 rationals, or outward-rounded doubles (FND-009) |
| affine form | `c0 + sum c_k e_k`, `e_k` in [-1, 1]: column symbols (`x = mid + rad e_x` in the group) and one approximation symbol per node (its nonlinear remainder, rounding errors and condensed terms; shared by every use of the node) | exact rationals, or doubles with every rounding error bounded, optionally truncated to `precision` bits |
| congruence | `r + m Z` (rationals; `m = 0`: exactly `r`), Granger's rules, CRT meet; GF(p) is `m = p` | exact rationals |
| zero fact | zero / nonzero / unknown (from the interval and the class) | |

Boolean nodes (comparisons, `&`, `|`, `~`, `is_null`, `not_null`) hold the set of SQL outcomes {TRUE, FALSE, NULL}
their rows can have. A comparison decides `d = a - b` as a form (correlated sides cancel: `x + 1 > x` is ALL,
`(x + y) - y` is exactly `x`, a node minus itself is 0), intersected with the interval difference and the class of `d`
(an equality is impossible when 0 is not in `r + m Z`).

Transfer functions: `+ - neg` exact; `*` (endpoint products with 0 * inf = 0; forms `a0 b0 + a0 B + b0 A` plus
`rad(A) rad(B)` on the node's symbol; an exact zero factor gives exactly 0; `a * a` is the square); `/ k` by a
nonzero constant; `min`, `max` (the provably smaller side, else `(a + b)/2 -+ |a - b|/2`); `abs`, `sqr` (secant
slope, the exact range of `f(x) - alpha x` on the interval); `% k` (`x - k floor(x/k)`, k > 0; an exact shift when
`floor(x/k)` is constant, else `[0, k)` tightened by the class `r + gcd(m, k) Z`). Each is exhaustively tested to
enclose every concrete value of small domains; the forms are re-checked node by node with exact noise values.

## Decisions, filter, overflow

Per group (boolean root): `all` (every selected row TRUE), `none` (no row can be TRUE), `unknown`, `empty`.
`State.filter` keeps ALL groups, prunes NONE groups and evaluates the rows of UNKNOWN groups exactly (int64
rationals with 128-bit intermediates, SQL NULLs); its stats separate the proof (`rows_proven`, groups by decision)
from the fallback (`rows_evaluated`). A row whose exact value leaves int64 rationals raises `OverflowError` (with
the count and the first row), never a guessed bit. `sum_fits(result, bits, unit)`: every partial sum of the group
lies in `[rows min(lo, 0), rows max(hi, 0)]`; with values proven to be multiples of `unit` (the class) the
accumulator of `value / unit` is proven to fit `bits` signed bits, or not proven.

Overflow inside the analysis (EXACT carrier) widens soundly (`widened`: a bound to infinity or to a bound kept by
sign, a form or class to unknown). BINARY64 forms bound every rounding error (FND-009 directed operations; no
`a * b + c` expression, so contraction cannot change a result) and are equal on the CPU and the device.

## Certificates (FND-007)

- `identity(prog, node)`: a LAW (`ring_identity`) when the node's symbolic form (columns as unbounded symbols, EXACT,
  no condensation) is exact: the claim `D E(x) - D L(x) = 0` holds the node's expansion `E` term by term and its affine
  form `L`; FND-007 canonicalizes it to zero. Refused: `DomainError` (an approximation term remains), `UnsupportedError`
  (a non-polynomial node, more than 8 columns), `AlgebraBudgetError` (support = expansion terms, at most 4096).
- `State.certify(scope, prog, fact)`: a SNAPSHOT when the analysis proved the fact in every group: `not_null` (a column
  node: claim NOT NULL), `congruence` with a prime `modulus` (claim `D E - D r = 0` over GF(p), NULL rows skipped),
  `constant` (claim `D E - D c = 0` over Z). FND-007 re-verifies the claim on every selected row before issuing it.
- Interval bounds and pruning decisions have no FND-007 claim kind: they hold while `state.fresh(scope)` (the scope's
  FND-007 record: identity, revision, rows, selection, NULL domains, grouping) is fresh; `filter` and `certify` refuse a
  stale scope (`StaleStateError` with the reasons).

## CUDA

`tr_mgpu_unc_eval`: one work item per group evaluates every node (its values in its own slots of a device scratch,
chunks of 4096 groups) and writes its own result slot. `tr_mgpu_unc_rows`: one work item per 64-row word writes its
own bitmap word, counts by atomics, the first overflowing row by atomic minimum. Both run the inline functions of
`math_unc.h`, so results equal the CPU bit for bit (emulated with a scrambled thread order in the tests).

## Budgets and refusals

| request | needs (refused before allocating) |
|---|---|
| program | circuit = nodes, variables = columns, support = rank (at most 64 nodes, 32 columns, rank 16: else `UnsupportedError`) |
| state | states = groups x columns, bytes = inputs, accumulators, the scope record |
| evaluation | output = groups, bytes = the scratch (CUDA: inputs, results, nodes, scratch: `mu.need`) |
| rows / filter | bytes = the bitmap (CUDA: selection, the columns read, ids, decisions, nodes); filter also output = groups |
| identity / certify | support = expansion terms; output = groups (certify) |

`ValueError` (malformed program: a zero divisor, `% k` with k <= 0, operands of the wrong kind; inconsistent inputs),
`TypeError` (floats are not exact input), `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `UnsupportedError`,
`OverflowError`, `DomainError`, `StaleStateError`.

## Limitations

Comparison outcomes are combined as sets (AND / OR of correlated predicates such as `x < 5 & x >= 5` is not decided);
congruence tightening of interval bounds is applied in the EXACT carrier only; MIN / MAX use SQL-style NULL
propagation (unlike `LEAST` / `GREATEST`); the input state is per column (no joint information between columns: the
correlation the forms keep is the algebraic one); exact arithmetic is int64 rationals with 128-bit intermediates
(wider values widen or overflow, they are never wrapped); certificates exist only for the claim kinds FND-007 has
(no inequality claims).
