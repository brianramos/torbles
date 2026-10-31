# Univariate affine / quadratic group envelopes (ENV-001)

`torbles/math_env.h` (every contract; the lookup functions shared with CUDA), `math_env.c`, `math_env_gpu.cu`,
`math_env.py` (opt-in). For one affine or quadratic function of a parameter t per group, the lower (or upper)
envelope over a parameter interval: the winning group on every piece, the exact breakpoints, and a declared tie
policy. Coefficients are exact rationals (FND-009 big rationals; `Fraction` in Python); irrational breakpoints are
certified by INV-001 ([inv.md](inv.md)).

```python
from fractions import Fraction
from torbles import math_env as me
env = me.envelope([(3, -1), (0, 1), None, (1, 0, Fraction(1, 2))], lo=-5, hi=None, upper=False, tie="smallest")
env.kind, env.info        # "finite" / "empty"; groups, present, classes, regions, pieces, points, algebraic, iterations
env.regions               # [{"kind": "point", "at", "boundary", ...} | {"kind": "piece", "lo", "hi", "cls", ...}]
env.points                # Fraction, or Algebraic(polynomial, root, interval, sign_lo); env.breakpoints (interior)
env.classes               # [{"coefficients", "groups", "winner", "degree"}]: identical functions
env.lookup(ts, backend="cuda")     # [{"status", "kind", "region", "winner", "tied"}] by binary search
env.evaluate_all(ts)               # every class at every t, exactly (the verification path)
env.value(Fraction(1, 2))          # (the exact envelope value, its region)
env.points[i].refine(Fraction(1, 10**30))                    # a narrower certified interval (INV-001)
me.from_bank(bank, constant="y^2", linear=(-2, "x*y"), quadratic="x^2", per="x^2")    # mean squared losses
me.from_expressions([loss_g for g in groups], "theta")       # first-slice (RESP-001) polynomials
me.need(groups, present, degree); env.lookup_need(ts, mode, backend)
```

## Envelopes

- Input: per group the coefficients `(a0, a1, a2)` of `a0 + a1 t + a2 t^2` (shorter: affine or constant; a nonzero
  coefficient of degree >= 3 is `UnsupportedError`), or `None`: an empty group, absent, never winning. `from_bank`:
  `f_g(t) = sum_k m_k sum_g[term_k] t^k / den_g`, `den_g` = 1, rows or the count of a term (a group without rows or with
  `den_g = 0` is empty). `from_expressions`: first-slice polynomials in one variable (another variable: a parameter
  dimension above 1, `UnsupportedError`).
- Interval `[lo, hi]`: each end an exact rational or `None` (unbounded); `lo = hi` is a single point.
- Classes: identical functions (equal canonical forms `(A + B t + C t^2) / d`, gcd 1, `d > 0`) form one class, coincident
  on every piece; the classes are ordered by their smallest member.
- Regions, in increasing order: `[point lo] piece point piece ... [point hi]`. On a piece exactly one class attains the
  envelope (its `cls`; `groups` are its members). A point is a boundary (`"lo"`, `"hi"`, `"lo and hi"`) or a breakpoint
  where two or more classes attain the envelope: the winning class changes, three or more functions meet, or a
  function touches the envelope (a group can win only at one point). `classes` and `groups` are exactly those attaining
  it there.
- Tie policy: `winner` is the smallest (`tie="smallest"`) or largest (`"largest"`) group among the region's groups.
  `upper=True` takes the maximum.
- Breakpoints: a `Fraction` (affine differences; quadratic ones with a square discriminant or a double root), or an
  `Algebraic`: root `root` (0 smaller, 1 larger) of the primitive integer `q0 + q1 t + q2 t^2` (`q2 > 0`, non-square
  discriminant: irreducible) with INV-001's isolating interval `(lo, hi)`: exactly one root of q inside, a sign change
  (`sign_lo` at lo); never a float. `refine(width)` re-isolates within it and bisects with INV-001's exact signs.

## Exactness

The sweep runs on FND-009 big integers: from the class attaining the envelope just right of the current position
(at -inf: the class smallest as t -> -inf, by the coefficients' asymptotic order), the next point is the smallest
root, past the position and below hi, of the differences `d_w P_c - d_c P_w` over all classes; roots are rational or
algebraic and compared exactly (`sign(X + Y sqrt(D))` and `sign(R + U sqrt(D1) + W sqrt(D2))` by squaring; equal
algebraic numbers have the same primitive polynomial and root). The classes there are those whose difference has that
root; the next winner is the smallest just right of it (the sign of the difference's derivative, then of its leading
coefficient). Values and comparisons at a point are exact (an algebraic point reduces a polynomial modulo its q).
Lookups compare t = u / v with each point by the sign of an integer polynomial at t (t - r; for an algebraic point
t - lo, t - hi, then q(t) against `sign_lo`), computed modulo k primes in (2^61, 2^62) (INV-001's verified table) with
Garner's mixed-radix digits; `k` follows from a proven bound `|S| < 2^B`, `61 k >= B + 2` (`env.last_lookup` reports
`primes` and `bound_bits`). The CPU and the CUDA kernel (`tr_mgpu_env_lookup`: one work item per value) run the same
function and agree bit for bit; values whose parts fit int64 cross as int64 arrays, others as big rationals.

## Budgets and refusals

- Build, before anything is allocated: variables = 1, degree (<= 2), support = present groups, states = groups,
  bytes = the class tables.
- Build, during the sweep: output = regions (counted exactly: an envelope with more regions than the limit is refused
  with that exact count), iterations = the differences formed and INV-001's isolation (refused when the work reaches
  limit + 1, the reported need), bytes = the region tables as they grow (`info["bytes"]` is the largest amount checked).
- Lookup: output = values, bytes = the job (CUDA: plus its device copies, checked together).

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `UnsupportedError` (degree >= 3, a second variable);
`OverflowError` (more than 64 moduli: values past about 3,900 bits); `ValueError` (`lo > hi`, malformed input);
`TypeError` (floats are not exact input). A value outside `[lo, hi]` answers `"outside"`, any value of an empty
envelope `"empty"`.

**Limitations.** Univariate only (parameter dimension 1). The sweep forms one difference per class per step, so its
work is classes x regions: an envelope with as many pieces as groups costs about groups^2 differences (1,024 tangent
lines exceed the default iteration budget of 2^20). The build runs on the CPU; only the lookups have a kernel.
Algebraic points are never given as numbers: a value of the envelope is returned for rational t only.
