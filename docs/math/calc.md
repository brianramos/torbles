# Non-polynomial finite-separation calculus (CALC-003)

`torbles/math_calc.h` (every contract; the certified fixed-point functions shared with CUDA), `math_calc.c`,
`math_calc_gpu.cu`, `math_calc.py`. Closed families whose dependence on k parameters separates into a few retained data
coefficients over the selected rows of an FND-004 scope (per dense group):

| family | function | coefficients |
|---|---|---|
| `phase` | `F(phi) = sum_r w_r prod_i sin(omega_i x_ri + phi_i) = sum_S C_S prod_{i in S} sin(phi_i) prod_{i not in S} cos(phi_i)` | 2^k: `C_S = sum_r w_r prod_{i in S} cos(theta_ri) prod_{i not in S} sin(theta_ri)` (k = 1: `C_0 = B = sum w sin(omega x)`, `C_1 = A = sum w cos(omega x)`) |
| `exp` | `G(theta) = sum_r w_r exp(sum_i theta_i + a_i x_ri) = E exp(theta_1 + ... + theta_k)` | 1: `E = sum_r w_r exp(sum_i a_i x_ri)` |

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_calc as mc
scope = mbk.Scope.from_table(db["t"], ["x", "w"])            # or Scope.from_arrays({...}, valid=..., groups=...)
F = mc.phase(scope, [("x", Fraction(1, 2))], weight="w")      # backend="cuda": the sums on the device
F.coefficients(); F.exact_bounds(); F.counts()                # double enclosures; exact rational bounds; rows
F(0.25); F.evaluate(phis)                                     # enclosures of F at doubles
F.derivative(0, 2); F.antiderivative(0); F.integral(0, a, b); F.definite(0, A, B)   # exact transforms; integrals
G = mc.exponential(scope, [("x", Fraction(-1, 4))], weight="w"); G(1.5)
cache = mc.Retained(); cache.phase(scope, [("x", Fraction(1, 2))], weight="w")      # retained by exact identity
mc.sincos(x); mc.exp(x); mc.sincos_scaled(omega, X, scale); mc.exp_scaled(a, X, scale)  # certified functions
```

**Data.** `x_ri` = the logical value of the factor's column / 10^scale (exact decimal), `w_r` = the weight column's
logical value / 10^scale (1 without one); a row counts when it is selected and every column the family reads is
present (`counts(g)` = (selected rows, counted rows)). Frequencies and rates are exact (`int`, `Fraction`, or a float's
exact dyadic value). An empty or all-NULL group is the zero family: exactly `[0, 0]`.

**Exactness.** The separation identities and the calculus are exact: `d/dphi_i` maps the (sin, cos) pair of
`phi_i` to (cos, -sin), so `derivative` (any order; negative = antiderivatives in the span, no constant) permutes and
negates the coefficients exactly (also their exact bounds); `d/dtheta` of `exp` leaves `E` unchanged. The coefficients
are transcendental, so each is an exact rational enclosure `[lo, hi] / (2^scale 10^wscale)` that contains the exact
value (`exact_bounds`) and its outward double interval (`coefficients`). Precision contract (`math_calc.h`):
argument reduction of `omega x (2/pi)` (or `a x / ln 2`) exactly in the integer part and to 2^-127 in the fraction
(64.192 fixed-point constants from the exact rational and a 512-bit 2/pi or 1/ln 2), Taylor series of sin, cos on
[0, pi/2] and of 2^f in 120-bit fixed point with every truncation bounded (radius <= 2^-112 per sin / cos; relative
2^-112 per exponential), midpoint-radius products, then outward integers `floor(lo 2^s)`, `ceil(hi 2^s)` summed exactly
in 128 bits with the largest scale `s <= 120` the host can prove keeps every sum below 2^126. A coefficient's width
is at most `rows (|w|max 2^-108 + 2^(1-s))` (measured: about 2^-113 `rows |w|max` for 1 to 16 factors). Evaluations at doubles use
certified `sin`, `cos` (`|phi| < 2^62`, else `UnsupportedError`) and `exp` (overflow `[DBL_MAX, inf]`, underflow `[0,
2^-1074]`) and outward-rounded interval arithmetic: every value returned is an interval containing the exact result.

**Identity and reuse.** A state records the family, its columns, the exact frequencies or rates, the weight and the
scope's exact descriptor. `Family.require(...)` (natively `tr_calc_require`): `True`; `DomainError` when another
construction is needed (another frequency, rate, column, weight, family or scope); `UnsupportedError` outside the
retained basis (a derivative in a frequency or a rate, `derivative("omega")` or `domega=[...]`, needs sums weighted by
x). `Retained` answers a request from a retained state of the same identity (`stats["hits"]`), builds another for a
new identity (`stats["constructions"]`), refuses requests outside the basis (`stats["refusals"]`), and drops states
of a stale scope (`StaleStateError`).

**CUDA** (`backend="cuda"`, `tr_mgpu_calc_sums`): the coefficient sums over the rows; per-row enclosures computed by the
same inline functions as the CPU, accumulated per work item and added with exact 128-bit two-word atomics, so the
device sums equal the CPU's bit for bit (`exact_bounds` compare equal); `Family.backend` is `cuda` or `cuda-emulated`.

**Budgets and refusals** (`AlgebraBudgetError` with `.resource/.need/.limit`, before allocating): `variables` = k,
`support` = coefficients (2^k or 1), `states` = groups x coefficients, `tensor` = selected rows x coefficients, `bytes`
= the state and sums (CUDA: its device copies, `device_bytes`); `degree` = |order| of a derivative; evaluations:
`output` = points, `tensor` = points x coefficients. `UnsupportedError`: more than 16 factors, `|K| >= 2^64` (a
frequency or rate too large for the reduction), an argument `|phi| >= 2^62`, a request outside the basis.
`OverflowError`: an exponential bound past 2^(2^24). `StaleStateError`: the scope's table changed.

**Limitations.** Fixed-point precision is absolute for sin / cos (a value below about 2^-100 has a large relative
width); a phase coefficient sums 2^k products per row (k <= 16, budgeted); definite integrals and evaluations return
double intervals only (no exact bounds); only the two families above are retained (no harmonics in x, no x-weighted
sums, no mixed phase-exponential products).
