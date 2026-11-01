# Explicit approximations and error certificates (UNC-002)

`torbles/math_approx.h` (every contract; the row evaluator shared with CUDA), `math_approx.c`, `math_approx_gpu.cu`,
`math_approx.py` (opt-in). An approximation is an immutable object that carries its certified error: the declared
norm, the assumptions it relies on, the error budget, the error spent term by term with its origin, and its
provenance (deterministic, or randomized with the seed and the failure probability stated as such). A bound that is
not achieved is refused, never reported; an approximation never authorizes an exact rewrite.

```python
from fractions import Fraction
from torbles import math_approx as ma
f = ma.series("exp", 0, 1, 8)                         # Chebyshev (economized Taylor), basis="taylor": monomials in t
f.error, f.terms, f.range, f.relative, f.l2, f.info   # exact Fractions; every term with its kind and origin
f.value(Fraction(1, 3))                               # the exact approximant (the function is within f.error)
ma.series("sin", 0, 1, 8, error_budget=Fraction(1, 10**9))   # NotConvergedError when the bound exceeds the budget
h = f * ma.series("sin", 0, 1, 8) + 3 * f             # errors propagate: sums add, products scale, nothing dropped
ma.series("log", Fraction(1, 2), 3, 12)(f)            # composition: the inner range (+- its error) inside the interval
q = f.quantize(20); fl = f.to_float()                 # coefficients on 2^-20; binary64 Horner with a rounding bound
fl.doubles([0.25, 0.5])                               # (values, certified bounds, statuses)
lr = ma.lowrank(A, 2)                                 # exact cross elimination: exact residual, certified norms
sk = ma.lowrank(A, 2, method="sketch", seed=7, probes=32)    # label "probabilistic", failure (13/16)^32
ev = f.evaluate(scope, "x", backend="cuda")           # per-row enclosures, exact group sums; same bits as the CPU
bits, undecided, stats = ma.filter(state, scope, "x", f, ">", 2)   # f(x) > 2 through UNC-001, undecided rows apart
f.authorize_exact()                                   # NotAuthorizedError, reason "approximate" ("statistical": sketch)
ma.relation(f, scope, "x", "y", tolerance=Fraction(1, 10**4), bits=16)   # FND-007 snapshot of y ~ p(x)
```

## What is certified

| construction | error terms | norm |
|---|---|---|
| `series(f, a, b, n)` for exp, log, sin, cos, sqrt, 1/x on `[a, b]` (exact rationals) | Taylor remainder at the midpoint (Lagrange; integral form for log, sqrt, 1/x), coefficient enclosure, Chebyshev economization (the dropped coefficients) | sup (`abs`), relative (`rel`, when \|f\| > 0 on `[a, b]`), `l2` = sqrt(b - a) sup |
| `poly(c, a, b)` | none (exact) | sup |
| `add`, `scale`, `mul`, `compose`, `quantize`, `to_float` | sums add; products `|f| eg + |g| ef + ef eg`; composition `L_outer e_inner + e_outer`; quantization `sum |c_j - q_j|`; truncation past degree 64; binary64 coefficient rounding and Horner `gamma_2n sum |a_j| B^j` plus an underflow term | inherited |
| `lowrank(A, k, "cross")` | the exact Schur-complement residual (Bareiss, complete pivoting) | Frobenius (exact square), spectral (`min(F, sqrt(norm1 norminf))`), max entry |
| `lowrank(A, k, "sketch", seed, probes)` | Rademacher range finder, exact projection; `||E||_F <= 2 max_i ||E w_i||` except with probability `(13/16)^probes` (Paley-Zygmund, independent probes) | probabilistic Frobenius / spectral / max |

Taylor coefficients are enclosed in dyadic intervals (224-bit working precision; exp, sin and cos by argument
reduction to |r| <= 1/2 and squaring / double angles, log by atanh, sqrt by integer square roots) and the
representatives are rounded to 192 bits with the enclosure radius added as a term. Composition requires the inner
range widened by its error to lie in the outer interval (else `DomainError`); internal products above degree 256 are
refused (`UnsupportedError`). Each term records its kind, its origin (the id of the approximation it came from) and
whether it was scaled; the error is their exact sum.

## Evaluation over rows, predicates, authorization

`evaluate(scope, column, fbits)`: x = logical / 10^scale is enclosed in binary64 and must lie in the domain (else the
row is `domain`, refused); a series is evaluated by interval Clenshaw (or Horner) with the coefficient enclosures,
widened by the certified error and clamped to the certified range; a float approximation by its binary64 Horner value
and bound. Group sums are exact: floor(lo 2^F) and ceil(hi 2^F) of each row added in 128 bits (|sums| < 2^125 proven
first), so `[sum_lo, sum_hi] / 2^F` encloses the exact sum; a group with a refused row has no sum. The CUDA kernel
(one work item per 64-row word, 128-bit atomics per group) equals the CPU bit for bit.

`filter(state, scope, column, f, op, T)`: the UNC-001 state's groups are decided by the binary64 UNC-001 proof of
p(t) op T -+ e (the polynomial in t = (x - mid) / rad with dyadic int64 coefficients, their rounding added to e) when
the group's x-range lies in the domain; the other rows by their row enclosure; a row the enclosure cannot decide is
counted undecided and set in its own bitmap. The filter takes a series (its binary64 form is evaluated per row) and
runs on the CPU; its row decisions are those of the row enclosures, which CUDA reproduces bit for bit.

`authorize_exact` always refuses (`TR_MATH_DOMAIN_ERROR`, FND-007 decision reason `approximate`, or `statistical`
for a probabilistic label). `relation` issues an FND-007 SNAPSHOT certificate of the POLYNOMIAL claim
`D y - D p~(x)` with a tolerance (the certified error, the rounding of p to multiples of 2^-bits and the data
tolerance; scaled integers of the two decimal columns, degree <= 16, sx n + sy <= 18), verified on every selected row;
FND-007 refuses to authorize it as `approximate`.

## Budgets and refusals

Checked before allocation (`TR_MATH_BUDGET` with the refusal): degree = the series degree, support = the coefficients
computed (the Taylor terms N + 1, N = min(n + 24, 2n + 8) by default), tensor = matrix entries, iterations = probes,
output = rows, bytes = every big number (FND-009 limbs) and the scratch; CUDA: the device copies
(`tr_approx_eval_need(..., TR_APPROX_NEED_DEVICE)`). Representation limits (`UnsupportedError`): degree 64, Taylor 160,
32 error terms, matrices 64 x 64, 64 probes, |a|, |b| <= 1024 for exp / sin / cos. Domain refusals (`DomainError`):
log / sqrt need a > 0, 1/x an interval without 0, `rel` a function without zeros on the interval. A certified bound
above the error budget: `NotConvergedError`.
