# Generating polynomials of finite-support distributions (GEN-001)

`torbles/math_gen.h` (every contract; the 128-bit convolution and evaluation are inline and shared with CUDA),
`math_gen.c`, `math_gen_gpu.cu`, `math_gen.py` (opt-in). Generating polynomials G(z) = sum over the support of
c(v) z^v of DIST-001 histograms (`docs/math/dist.md`) or explicit terms, with exact FND-009 big-integer coefficients
(`docs/math/carriers.md`).

```python
from fractions import Fraction
from torbles import math_dist as md, math_gen as mg
g = mg.generating(md.histogram(scope, ["x"]), group=0, mass="count", repr="sparse", shift=None)
g.terms(); g.coefficient(-3); g.total(); g(Fraction(1, 2)); g((2, 3))          # 2 variables: tuples
g.moment(2); g.moment(3, falling=True); g.moment((1, 2))                         # raw / factorial / mixed
g.euler(2, var=0); g.derivative(1)                                               # new states
g * h2; g.pow(16); g.pow(8, order=16); g.mul(h2, order=16); g.series(16)        # an FND-009 Series
g.truncate(16); g.to_dense(); g.to_sparse(); g2.marginal(0)
g.need_mul(h2); g.need_pow(64)                                                   # what is checked first
g.mul(h2, backend="cuda"); g.evaluate_many(points, backend="cuda")               # the 128-bit kernels
mg.from_terms({-2: 1, 0: 3, 5: 2}); mg.from_terms({(0, 1): 2, (1, 0): -1})
```

## The state

- One or two variables (a 1-D or 2-D histogram; 3 coordinates: `UnsupportedError`). c(v) is the bag multiplicity of
  the value v (`count`: rows) or its weight sum (`weight`: any sign; zero sums leave the support). Values are logical
  integers (decimal value times 10^scale). A cell with a NULL coordinate has no value and is no term: its rows are
  `info["nulls"]`, so G(1) = rows - nulls.
- Negative values use a declared shift s per variable: G(z) = z^s P(z), P(z) = sum c_e z^e over the exponents
  e = v - s >= 0 (a Laurent polynomial when s < 0). `shift=None`: the smallest present value; an explicit shift must
  not exceed any present value (`DomainError`). Products add shifts; derivatives lower them; values stay exact.
- `sparse`: the nonzero terms in lexicographic order; `dense`: every exponent of the box [0, span_1) x [0, span_2)
  (zeros included). Conversions are exact; `equal` compares the Laurent polynomials (values and coefficients).

## Requests (exact)

| request | value |
|---|---|
| `coefficient(v)` | c(v), 0 off the support |
| `evaluate(z)` / `g(z)` | G at a rational point per variable (int, Fraction, Decimal; floats refused): the homogeneous Horner N = sum c_e p^e q^(D - e) over the terms (gaps skipped by powers), over q^D, times p^s / q^s (s >= 0) or q^-s / p^-s (s < 0); a canonical big rational. z = 0 with a nonzero coefficient at a negative value is a pole (`NoSolutionError`) |
| `total()` | G(1): the total mass |
| `moment(k)` | sum c(v) v^k: (z d/dz)^k G at z = 1; `falling=True`: sum c(v) (v)_k = the k-th derivative at 1 (factorial moments); two variables: mixed (k1, k2) |
| `euler(k, var)` | (z d/dz)^k G: coefficients c(v) v^k (zero terms dropped) |
| `derivative(k, var)` | d^k G / dz^k: coefficients c(v) (v)_k, the shift lowered by k |
| `mul`, `pow` | convolutions: the distribution of the sum of independent draws (masses multiplied); `order`: only the exponents e < order of P (one variable) |
| `series(N)` | P mod z^N as an FND-009 truncated series (one variable; series products, exp, log ... from there) |
| `marginal(var)` | G(z_var, 1) of a 2-D state |
| `evaluate_many(points)` | the checked 128-bit Horner of a 1-D state with int64 coefficients: a Fraction per point, None where a value leaves int128 |

Products accumulate in 128 bits when every coefficient fits int64 and min(terms) max|a| max|b| < 2^127, else in big
integers; a box of accumulators is used when it fits the support budget and is at most 4 x pairs + 65536 cells, else
the pairs are sorted by their exponent sums. All paths are exact and give the same state. The tests compare
coefficients, G(1) and the moments with DuckDB (GROUP BY counts, SUM of HUGEINT powers), and products, powers,
evaluations and operators with Python dict polynomials and the FND-009 series products.

## CUDA

`tr_mgpu_gen_conv`: the dense convolution of two int64 coefficient arrays, one work item per output coefficient, an
exact 128-bit sum (the host proves min(na, nb) max|a| max|b| < 2^127, else `OverflowError`; a coefficient past int64
has no dense int64 copy: `OverflowError`); `g.mul(h, backend="cuda")` adopts the result as a state. `tr_mgpu_gen_eval`:
one work item per point, each with its own status. Their CPU references (`tr_gen_conv_i128`, `tr_gen_eval_i128`) run the
same inline functions and equal them bit for bit; where the 128-bit path succeeds it equals the exact result.

## Budgets and refusals

| Step | checked before allocation |
|---|---|
| state (histogram or terms) | variables, support = stored terms (dense: the box), bytes = arrays + coefficients (32 + 8 limbs each) |
| product | tensor = coefficient pairs, support = the result's terms (box, or min(box, pairs)), bytes = scratch + result with coefficients of L_a + L_b + 1 limbs |
| power G^k | its largest step (support growth k (span - 1) + 1, pairs, intermediate bytes with G^(j-1) held) before the first product |
| moment, Euler, derivative | degree = the order (default 64), bytes = the coefficient bound |
| evaluation | support = the terms, bytes = the numbers' bound (bits <= coefficient bits + (D + abs(s)) ceil(log2 max(abs(p), q))) |
| series | FND-009: degree = order - 1, support = order, bytes |
| CUDA | output = coefficients / points, bytes = the device copies |

Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`); also `OverflowError` (a value or shift past
int64), `DomainError` (a value below a declared shift; products of different decimal scales), `NoSolutionError` (a
pole), `UnsupportedError`, `ValueError`.
