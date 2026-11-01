# Small-support partition and likelihood functions (PROB-001)

`torbles/math_prob.h` (every contract; the per-point terms are inline and shared with CUDA), `math_prob.c`,
`math_prob_gpu.cu`, `math_prob.py` (opt-in). Declared finite-support probability models evaluated from retained
statistics: the masses of a DIST-001 histogram (`docs/math/dist.md`) aligned to a declared support, without reading a
row. Exact rationals for categorical models; certified enclosures (CALC-003's exponential, DIST-003's logarithm, sums
as exact scaled integers) for exponential families; many parameter requests per call, on the CPU or in CUDA.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_dist as md, math_prob as mp, math_conv as mc
S = mp.support(range(8))                                   # declared: points without rows keep mass 0
J = mp.support([(x, y) for x in range(4) for y in (0, 1, None)])   # 1..3 coordinates; None = the NULL category
st = mp.stats(md.histogram(scope, ["x"]), S, mass="count", exclude=())   # DomainError(.group, .point) outside S
st.counts(g); st.group(g); st.info; st.arrays()            # masses per declared point, N, rows, dropped mass
M = mp.categorical(S)                                      # weights per request; features=... for a tilt statistic
E = mp.exponential(S, features=lambda x: (x, Fraction(x * x, 8)), base=None)   # p ~ h(x) exp(theta . T(x))
ev = E.evaluate(st, [(Fraction(1, 3), -1), (0, 2)], groups=0, grad=True, hess=True, backend="cuda")
ev[0]          # Z, Z_exact, logZ, logZ_bounds, loglik, loglik_log_bounds, neg_inf, empty, grad, hess, width
ev.exact(0)    # exact rational bounds of every entry (FND-009 big integers), from the batch's own sums
M.evaluate(st, [[1, 1, 2, 0, 0, 0, 0, 4]], grad=True, hess=True)            # exact Z, gradient, Hessian
M.evaluate(st, [[Fraction(1, 8)] * 8], mode="probabilities")               # exact probabilities (sum must be 1)
mp.product(st_x, st_y, declaration=mc.independent(scope))  # the CROSS JOIN within groups, only when declared
mp.need(st, E, n=10_000, hess=True, backend="cuda"); mp.ln_bounds(m)
```

## Domain

- **Support**: d points of k = 1..3 coordinates (logical int64 values or NULL, a decimal scale per coordinate), in
  DIST-001 cell order (sorted natively; duplicates refused). Declared, never inferred: an unobserved point stays with
  mass 0 and contributes to Z. `S.product(T)` is the product support (k <= 3).
- **Statistics** (`mp.stats`): the histogram must have the support's k and scales (else `DomainError`); per group the
  mass of every declared point (`count`: rows; `weight`: nonnegative weight sums, logical units, group total < 2^64),
  N, the selected rows (0: an absent group) and the mass of cells dropped by a declared listwise NULL exclusion.
  Every other observed cell must be declared: an observed value outside the support is `DomainError` naming the group
  and the point. The state records the support (exact), the histogram's scope digest and the scope's data source
  (GEN-002's rule). A model over another support is refused (`DomainError`): a support change is a different state.
- **Models** (features T: 0..16 rationals per point, stored as int64 numerators over per-feature lcm denominators):
  - `categorical`: p(x_i) = w_i / Z, Z = sum w_i; each request gives the weights (rationals >= 0, put over their lcm
    in int64) or exact probabilities (`mode="probabilities"`: sum exactly 1). Gradient and Hessian of the
    exponential tilt at theta = 0 in the direction T: Tobs - N E_w[T], -N Cov_w[T]; without features the one-hot
    statistic (natural parameters eta_i = ln w_i): n_i - N p_i and -N (diag(p) - p p^T).
  - `exponential`: p_theta(x) = h(x) exp(theta . T(x)) / Z(theta), base h >= 0 rational (default 1; zero everywhere
    refused: Z = 0), 1..16 features; each request gives theta (rationals with int64 numerator and denominator; a
    float is its exact value).
- **Answers** per request (group g; N = N_g; Tobs = sum n_i T(x_i) exactly): Z, log Z, the log-likelihood
  sum n_i log p(x_i), its gradient (`grad=True`) and Hessian (`hess=True`, symmetric).
  - Zero mass: a point with zero model probability (w_i = 0 or h_i = 0) and positive mass gives loglik = -inf
    (`neg_inf`, enclosure [-inf, -inf], gradient and Hessian `None`), never a large finite number; zero masses
    contribute 0; Z = 0 (all weights 0) is refused for that request (`DomainError`).
  - An absent group or N = 0 is the empty sample (`empty`): loglik 0, gradient and Hessian 0.
- **Dependent events**: a joint model over (x, y) is evaluated on the statistics of the joint histogram of (x, y).
  `mp.product(a, b)` (per group n(x, y) = n_a(x) n_b(y): the CROSS JOIN within the group, over the product support) is
  the only route from marginals to a joint state; states of one data source (one table, or one array scope) multiply
  only under a GEN-002 independence declaration covering that source (`mc.independent(scope)`, checked natively with
  `tr_conv_decl_covers`), else `DomainError` with GEN-002's `.reason` (`needs_declaration`, `not_covered`;
  `grouping_differs`: one source grouped differently; `keys_differ`: different numbers of groups). States of
  different sources form their Cartesian product without a declaration. The result records `declared`.

## Exactness and precision

- Categorical: per request the exact 128-bit sums Zw = sum W, S_j = sum W N_j, S_jl = sum W N_j N_l (a bit-length
  proof `bitlen(d) + bitlen(max W) + bitlen(max |feature product|) <= 126`, else `OverflowError` for that request);
  Z = Zw / Dr, gradient (Tobs_j Zw - N S_j) / (D_j Zw) and Hessian -N (S_jl Zw - S_j S_l) / (D_j D_l Zw^2) are exact
  rationals (`ev.exact(i)`, lo == hi; 256-bit numerators in the double enclosures). log Z and loglik = sum n ln W -
  N ln Zw are enclosed with DIST-003's certified logarithm (`tr_info_ln`, below 144 ulps of 2^-120 per value) summed
  exactly at the scale sL = min(120, 119 - bitlen(N)).
- Exponential families: per request and point, theta . T(x_i) / ln 2 = sum_j N_ij K_j with CALC-003's 64.192 constants
  K_j = floor(|num| floor(2^512 / ln 2) / (den D_j 2^320)) (error <= 2^-191 each), reduced and exponentiated by
  CALC-003's `tr_calc_exp2_reduce` (2^t = 2^n m, m in [1, 2] in 120-bit midpoint-radius fixed point, relative radius
  <= 2^-112). An exact maximum n_ref = max (n_i + bitlen(H_i)) (a first pass) scales every term to u_i < 2 (the
  reference >= 1/2); the outward integers floor / ceil of u_i, u_i N_ij, u_i N_ij N_il at the scales s = min(120, 124 -
  bitlen(d)), s - b_j, s - b_j - b_l (b_j = bitlen(max |N_ij|)) are summed exactly in 128 bits (every partial sum is
  proven below 2^126 in any order). Then Z = 2^(n_ref - s) [S_0lo, S_0hi] / Dh (`Z_exact`; the double enclosure is
  [DBL_MAX, inf] / [0, 2^-1074] past the double range), log Z and the log terms as exact 2^-120 bounds (the 128-bit
  extension `tr_prob_ln128` of DIST-003's logarithm: hi - lo <= E + 149 ulps for S_0 = 2^E f, f in [sqrt(2)/2,
  sqrt(2)]; `mp.ln_bounds`), loglik =
  sum n ln H + theta . Tobs - N ((n_ref - s) ln 2 + ln S_0); the gradient Tobs_j - (N 2^b_j / D_j) S_j / S_0 and the
  Hessian -N 2^(b_j + b_l) (S_jl S_0 - S_j S_l) / (D_j D_l S_0^2) from exact 256-bit numerators at the corners of the
  sums' boxes (no cancellation is rounded; a variance's upper bound is clamped at 0). Measured (benchmark, `width["Z"]`
  is an upper bound of Z's relative width): about 1e-34 for 8 and 64 support points, 2e-31 for 1024 (the scale s
  drops with bitlen(d)); the tests assert below 2^-100 for small supports and exact log Z bounds below 2^-100 apart;
  the double enclosures add a few ulps (`width`: log Z, loglik, the widest gradient and Hessian entries).
- Every enclosure contains the exact real value; the tests check them against 80-digit Decimal enumeration (exactly,
  as Fractions) and the exact categorical values against Fraction enumeration, equality.

## CUDA

`backend="cuda"` (`tr_mgpu_prob_sums`): one work item per (request, 64 support points). Exponential families: launch 1
computes n_i + bitlen(H_i) (the integer part of CALC-003's reduction) with a local maximum and one atomic maximum per
request; launch 2 the exponentials, the outward integers and local 128-bit sums, flushed with exact two-word atomics
(`tr_atomic_add_wide`). Categorical: one launch of the exact sums and the log bounds. Flags (an exponent overflow, a
zero-probability point with mass) by atomic OR. The sums, reference exponents and flags equal the CPU's bit for bit
whatever the thread order; the host finishes both with the same `tr_prob_finish` (`ev.equal(other)` compares all of
it); `ev.backend` is `cuda` or `cuda-emulated`.

## Budgets and refusals

| step | checked before allocation |
|---|---|
| support | variables = k, support = d, bytes |
| model | variables = m, support = d, tensor = d m, bytes |
| statistics | support = d, states = G d, bytes (with the support and the source descriptor) |
| product | support = da db, states = G da db, bytes |
| evaluation | output = n x (3 + gradient + Hessian entries; one-hot: d and d (d + 1) / 2), tensor = n d, states = n x sums, bytes = the plan (680 bytes per exponential request, 8 d per categorical request), sums and the finish scratch |
| CUDA | bytes = every device copy, states = n x sums |
| exact bounds | bytes of every big integer |

Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`) before anything is allocated; sizes past 64 bits
are refused whatever the budget. A request's own status (the batch goes on; `ev[i]` raises it, `ev.status(i)`):
`ValueError` (a group out of range, a denominator < 1), `DomainError` (a negative weight, Z = 0, probabilities not
summing to 1), `OverflowError` (a common denominator outside int64, an unprovable sum, an exponent past 2^(2^24)).
Also `DomainError` (an observed value outside the support, another support, a zero base measure, an undeclared
independence), `UnsupportedError` (k > 3, m > 16), `OverflowError` (features or base measures over denominators that
do not fit int64), `StaleStateError` (the scope's table changed).

## For PROB-002

The declared support (`tr_prob_support_get_view`), the retained statistics (`tr_prob_stats_get_view`: dense masses per
group over the declared support, N, rows), the models (`tr_prob_model_get_view`) and the evaluation entry points
(`tr_prob_plan_create`, `tr_prob_sums`, `tr_prob_finish`, `tr_prob_exact`, `tr_mgpu_prob_sums`) are public in
`math_prob.h`, with the shared per-point terms (`tr_prob_exp_point`, `tr_prob_cat_point`) and the 128-bit logarithm.

## Limitations

Parameters, features and base measures need int64 numerators over int64 common denominators; the categorical sums'
bit-length proof is conservative. Exponents are limited to CALC-003's 2^(2^24). The interval bounds of gradients and
Hessians treat the sums as independent boxes (rigorous, about 2^-100 relative to the moments; relatively wider when a
covariance nearly cancels). Statistics are dense per group over the support. Products combine two states into at most
3 coordinates, within groups; the declaration is the caller's statement, never verified against the data, and states
without a data source cannot be multiplied. No FND-007 certificate is emitted (FND-007 has no claim kind for
membership in a finite set; the containment is checked natively). The benchmark's factorized-inference case belongs to
PROB-002.
