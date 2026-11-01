# Approximate spectral feature kernels (KERNEL-002)

`torbles/math_kapprox.h` (every contract; the per-row code shared with CUDA), `math_kapprox.c`, `math_kapprox_gpu.cu`,
`math_kapprox.py` (opt-in). Kernels without an exact finite feature map are offered through declared finite-feature
approximations whose every answer carries an approximation record in UNC-002's vocabulary (`docs/math/approx.md`):
the declared norm, the assumptions, the error budget and the error spent term by term, the provenance. Exact
polynomial kernels stay exact: they are answered by KERNEL-001 (`docs/math/kernel.md`).

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_kapprox as ka, math_kernel as mk
scope = mbk.Scope.from_arrays({"x": xs, "y": ys, "w": ws}, scales={"x": 1, "y": 1}, groups=g, ngroups=2)
st = ka.features(scope, ["x", "y"], ka.gaussian(Fraction(1, 2)), method="taylor", degree=8, weight="w")
st = ka.features(scope, ["x", "y"], ka.laplace(1), method="rff", features=256, seed=7, failure=Fraction(1, 10**6))
a = st.mmd2(0, 1)          # a.value: exact enclosure of the approximant; a.error, a.terms; a.bound; a.label, a.failure
st.sum(0, 1); st.sum(0, self_pairs=True, ordered=False); st.mean(0); st.witness(0, [(12, -3)])   # logical points
st.query([dict(op="mmd2", a=0, b=1, unbiased=False)], points=None)                              # a batch
st.record; st.info; st.q("per_pair"); st.frequencies; st.group(g); st.sums(g)   # record, state, raw sums
st.authorize_exact()       # NotAuthorizedError: "approximate" (taylor), "statistical" (rff); st.decide() as a dict
data = st.dumps(metadata={"run": 1}); ka.loads(data, scope)   # FND-011 record format; identical answers
ka.generate(ka.gaussian(1), 2, 64, seed=7)                     # the generator's raw draws and their precision
ka.need(scope, ["x", "y"], ka.gaussian(1), method="rff", features=64, backend="cuda")
ka.features(scope, ["x", "y"], mk.polynomial(2, c=1))          # an exact kernel: KERNEL-001's FeatureState
```

## Kernels and methods

x is the exact value (logical / 10^scale) of d <= 16 columns sharing one decimal scale (`DomainError` otherwise);
weights are a column's values (1 without one); gamma > 0 is an exact int64 rational (`gaussian(lengthscale=l)`: gamma
= 1 / (2 l^2)). `gaussian` exp(-gamma |x - y|^2), `laplace` exp(-gamma |x - y|_1), `cauchy` prod_j 1 / (1 + gamma^2
(x_j - y_j)^2); k(x, x) = 1 for all three, so diagonal terms (sum w^2) are exact.

- `taylor` (gaussian only): exp(-gamma |x - y|^2) = E(x) E(y) exp(2 gamma x.y), E(x) = exp(-gamma |x|^2), the last
  factor truncated at degree N <= 12: k_N = <phi(x), phi(y)> over the C(d + N, N) multi-indices |a| <= N with
  KERNEL-001's sorted-multi-index evaluation and multiplicities |a|! / a!. Declared domain: |x|^2 <= radius2
  (`radius2=`; default the data's largest |x|^2, exact); a row or witness point outside is refused (`DomainError`).
  Certified per pair: |k - k_N| <= eps = e^-U T_N(U) = P(Poisson(U) > N), U = 2 gamma radius2 (tight: attained at x = y
  on the boundary), bounded above by an exact rational. Label `certified`, provenance deterministic.
- `rff` (all three): `features` D <= 2^20 frequencies from the native seeded generator, deduplicated with
  multiplicities; k^(x, y) = (1 / D) sum_f m_f cos(omega_f.(x - y)) through SPEC-001's fixed spectral basis (certified
  cos / sin enclosures of every row summed as exact scaled integers). Generator (`ka.GENERATOR`, recorded): splitmix64
  words under UNC-002's stream convention, U = (2N + 1) / 2^129 from 128-bit integers; gaussian Box-Muller times
  sqrt(2 gamma), laplace gamma cot(pi U), cauchy +-gamma (-ln U); certified fixed-point transforms (ln by atanh, sqrt by
  integer square roots, sin / cos by CALC-003's series after SPEC-001's exact quarter-turn reduction), rounded to dyadic
  rationals of `fbits` fraction bits, each sample's enclosure checked against its a priori bound. Per pair, with
  probability >= 1 - failure (Hoeffding, under the assumption that the generator's words are i.i.d. uniform):
  |k^ - k| <= 2 h + beta, h = sqrt(ln(2 / failure) / (2 D)) (an exact upper bound), beta = 2 P_bad + 2 rho sqrt(d)
  eta the deterministic bias of the dyadic frequencies (the coupling of every good cell with an exact sample: eta bounds
  the cell width, the enclosure and the rounding; P_bad the remaining cells). Label `probabilistic`, provenance
  randomized with the seed, the count and the failure probability.

## Answers

`sum` (KERNEL-001's pair semantics: across groups every pair; within a group ordered pairs i != l, `self_pairs`,
`ordered=False`), `mean` (over the exact weighted pair count, `.pairs`), `mmd2` (biased or `unbiased`), `witness`
(points as logical values inside the domain). Each `Answer`: `value` = (lo, hi), the exact rational enclosure of the
approximant's value (midpoint-radius products of the certified feature sums in KERNEL-001's 384-bit accumulators, then
FND-009 big rationals); `terms` = taylor [("remainder", eps F)], rff [("probabilistic", h R), ("quantization", beta F)]
with F the answer's weight factor (A_a A_b for a sum, A = sum |w|, over |pairs| for a mean, (A_a / |W_a| + A_b /
|W_b|)^2 for a biased MMD, ...) and R its Hoeffding range; `error` their exact sum; `bound` = (lo - error, hi +
error) for the exact kernel quantity; `failure` (rff: per answer; m answers together: m failure); `exact` is False.
Sums and pair counts are in value units (weights / 10^wscale).

## Exactness, CUDA

Per row (`math_kapprox.h`, shared by the CPU loop and the kernels): t = gamma |x|^2 / ln 2 from the exact 192-bit sum
of squares times a 64.320 fixed-point constant (from a 577-bit 1/ln 2, error < 2^-127 in t), E = 2^-t by CALC-003's
certified exp (radius + 8 ulps), u = X 2^-b exact, monomials in CALC-003's midpoint-radius products, w E u^a 2^S
rounded outward and summed exactly in 128 bits (S = min(120, 124 - bits(rows x max |w|))). `backend="cuda"`
(`tr_mgpu_kapprox_sums`): a counts kernel (rows, W, A by two-word atomics, Q by a three-word atomic carry chain, the
largest |x|^2 per slice slot), a prefactor kernel (one slot per row) and a feature kernel (work item = (item, row
slice), local accumulation, `tr_atomic_add_wide`); for rff the counts kernel and SPEC-001's `tr_mgpu_fourier_sums`.
The readback is adopted by `tr_kapprox_from_sums` after consistency checks (and SPEC-001's own adoption checks):
equal to the CPU bit for bit. Answers are computed on the CPU.

## Persistence and labels

`dumps` writes FND-011's record format (`docs/math/persist.md`): its header (type tag 0x4b02, which FND-011's typed
loader does not know and refuses, carrier interval, exact 0), the FND-007 source record of the scope, the basis (the
spec: kernel, method, gamma, degree, radius, budget, seed, count, failure; the derived constants, the frequencies with
their multiplicities and SPEC-001's reduction constants), the metadata and the payload (group statistics, sums),
FND-011's checksum. `loads` checks in FND-011's order (`TruncatedStateError`, `NotAStateError`, `ByteOrderError`,
`UnknownVersionError`, `MalformedStateError`, `CorruptStateError`, `WrongTypeError`) and rebuilds the identical state
(the same answers; written again, the same bytes); with a scope the recorded source must be fresh (`StaleSourceError`)
and the plan rebuilt from the recorded spec (frequencies redrawn from the seed) must agree. An approximate state never
authorizes an exact rewrite: `tr_kapprox_authorize_exact` always refuses (FND-007 reason approximate, statistical for
rff; the decision's kind is UNC-002's label 5 / 6, never a certificate kind).

## Budgets and refusals

| request | needs |
|---|---|
| build | variables (columns, weight), degree (N), support (taylor items, rff D), states (groups x items), tensor (selected rows x items; rff: SPEC-001's rows x frequencies), iterations (rff D d draws), bytes (plan, sums, state; SPEC-001's own) |
| build (CUDA) | bytes = the device copies (`need(..., backend="cuda")`), states |
| query | output = requests, tensor = points x items, bytes = point features; FND-009 limbs of every big rational |
| save / load | bytes = the record (a load: twice) |

Refusals raise before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `NotConvergedError` (the
per-pair bound exceeds `error_budget`), `UnsupportedError` (d > 16, N > 12, D > 2^20, taylor for laplace / cauchy,
frequencies past 2^60, a Gaussian factor below 2^-(2^24), an exact kernel KERNEL-001 refuses such as `mk.kernel("rbf")`),
`DomainError` (columns of different scales, data or points outside the declared domain), `ValueError` (gamma <= 0, a
failure outside (0, 1), D = 0, malformed requests), `StaleStateError` (the scope's table changed).

## Limitations

Deterministic features only for the Gaussian, and only useful while U = 2 gamma radius2 is moderate (the bound is
P(Poisson(U) > N), N <= 12); its weight factors are worst cases (A_a A_b for sums). Random-feature bounds are per answer
and rest on the idealization that the seeded words are independent uniforms (stated as an assumption, never
certified); Hoeffding's ranges are conservative (observed errors were at most 0.14 of the bound in the tests). Answers
and point features run on the CPU (only the row accumulation has CUDA kernels); the generator is host code using
FND-009 big integers (20 to 50 us a coordinate). The CUDA kernels are compiled with nvcc for sm_80 and verified in the
emulation; no device run yet. No FND-007 certificate is emitted (an approximate kernel quantity is no row relation).
