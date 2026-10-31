# Fixed spectral basis states (SPEC-001)

`torbles/math_fourier.h` (every contract; the per-row reduction shared with CUDA), `math_fourier.c`,
`math_fourier_gpu.cu`, `math_fourier.py` (opt-in). For a declared finite set of frequencies, the Fourier features of
the selected rows of an FND-004 scope are aggregated once per dense group as certified complex enclosures summed as
exact scaled integers; phase actions are then answered from that fixed basis without rescanning the rows.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_fourier as mf
scope = mbk.Scope.from_arrays({"x": xs, "y": ys, "w": ws}, scales={"x": 2})
B = mf.basis(scope, ["x", "y"], [(1, 0), (Fraction(1, 2), 3)], weight="w", unit="radian")   # backend="cuda" too
B.coefficients(g); B.bounds(g); B.counts(g); B.weight(g)       # enclosures; exact Fractions; rows; exact sum w
B.act([dict(shift=(0.5, -1.0), phase=0.25, order=(1, 0), phase_order=1)], frequencies=[(1, 0)], normalize=True)
B.shift(tau); B.characteristic(g)                               # translated features; F / W
D = B.derivative((1, 0)); D.exact(g, f)                         # exact transform: times i omega_1
B.feature((2, 0)); B.index([...])                               # DomainError: missing spectral support
W = mf.walsh_basis(scope2, ["b0", "b1", "b2"]); W.walsh_inverse(g)   # the Boolean case, exact (SPEC-002)
cache = mf.Retained(); cache.basis(scope, ["x"], [1, 2]); cache.basis(scope, ["x"], [2])   # a hit: a subset
```

**Features.** For d <= 16 columns x and a unit c (`radian`: 1, `pi`: pi, `turn`: 2 pi) the feature of group g at
frequency omega (d exact int64 rationals) is F(omega) = sum_r w_r exp(i c omega . x_r) = C + i S over the counted rows
(selected, every column read present; decimals exact: logical / 10^scale; w = 1 without a weight column). Per group
the state also keeps the selected rows, the counted rows and the exact weight sum W. An empty or all-NULL group is
exactly 0; an absent group has no rows.

**Exactness (precision contract, `math_fourier.h`).** Per row the reduction t = theta (2/pi) = sum_j X_j K_j is exact
from the logical values and 64.192 fixed-point constants (radians: a 512-bit 2/pi, error < 2^-191; pi and turn units:
exact whenever the constant is dyadic); the quadrant is exact, the fraction kept to 128 bits; sin and cos come from
CALC-003's certified 120-bit Taylor sums (radius <= 2^-112), widened by 3 ulps of 2^-120 for the reduction. Each
weighted row enclosure is rounded outward to integers floor(lo 2^s), ceil(hi 2^s) and summed exactly in 128 bits with
s = min(120, 124 - bits(rows x max |w|)) (CALC-003's proof: no sum leaves 2^126). `bounds()` are exact rationals
[lo, hi] / (2^s 10^wscale) containing C and S; width <= rows (|w|max 2^-108 + 2^(1-s)). For one dimension in radians
the sums equal CALC-003's phase family bit for bit (`C_1`, `C_0`). Exact integer angles (the Boolean and quarter-turn
cases) are exact points (radius 0).

**Phase actions** (`act`, no rescan): per request i^(|alpha| + beta) (c omega)^alpha exp(i (c omega . tau + phi))
F(omega), divided by W when normalized (the empirical characteristic function; W = 0 is `DomainError`): translation
tau, phase phi (doubles, exact dyadic values, |.| < 2^62), translation derivative order alpha (d / dtau_j multiplies
by i c omega_j) and phase derivative order beta (times i). The rotation is reduced exactly from the exact frequencies
and the exact tau, phi, then everything is outward-rounded interval arithmetic: each value is a complex rectangle
containing the exact result (exact for integer rotations of exact coefficients). `derivative(alpha, beta)` is a new
state whose `exact()` bounds are the exact transform (rotation by i^(|alpha|+beta), times omega^alpha as big
rationals) with the factor c^|alpha| reported apart (`unit_power`, `factor`), never rounded in.

**Identity and missing support.** A state records its columns, weight, unit, the exact frequency set and the scope's
exact descriptor. `require` / `Retained` answer a request from a state whose basis contains every requested frequency
(a subset is a hit); any new frequency, another column, weight, unit or scope builds a new basis state (counted in
`stats`). A frequency not in the basis is refused (`DomainError`: missing spectral support), never treated as zero:
`feature`, `index`, `act(frequencies=...)`, and the Walsh inverse of an incomplete Walsh set.

**The Boolean case (SPEC-002).** k columns with values in {0, 1} and the 2^k frequencies S in {0, 1}^k (pi units;
S / 2 in turns; `walsh_frequencies`, `walsh_basis`): F(S) = W(S) = sum_x v(x) (-1)^{|S and x|}, SPEC-002's forward
Walsh coefficient of the weighted histogram, exactly (lo = hi, imaginary part 0). `walsh_inverse` returns v(x) = 2^-k
sum_S W(S) (-1)^{|S and x|} exactly through SPEC-002's inverse transform with its normalization and full-domain
cost (2^k states, k passes, k 2^(k-1) butterflies); the construction itself costs rows x 2^k per-row characters,
whatever the occupancy. Refused (`DomainError`): a missing Walsh frequency, data outside {0, 1}, a derivative state,
an inexact coefficient, another unit.

**CUDA** (`backend="cuda"`, `tr_mgpu_fourier_sums`): a counts kernel (rows, counted rows, exact weight sums per group)
and a feature kernel (work item = (frequency, slice of rows), the same inline per-row enclosures, local accumulation
per group, exact two-word 128-bit atomics); the device sums equal the CPU's bit for bit (`bounds` compare equal);
`Basis.backend` is `cuda` or `cuda-emulated`. Actions, derivatives and the Walsh inverse run on the CPU.

**Budgets** (`AlgebraBudgetError` with `.resource/.need/.limit`, before allocation): build: variables = d, support =
m frequencies, states = groups x m, tensor = selected rows x m, bytes = plan, sums and state (CUDA: the device copies,
`device_bytes`); actions: degree = |alpha| + beta (with the state's), output = requests x frequencies, tensor = output
x (d + 1), bytes = the selected coefficients; derivative: degree, states, bytes; Walsh inverse: variables = k, tensor
= 2^k, iterations = k, bytes = 40 x 2^k. Refusals: `UnsupportedError` (d > 16, more than 2^22 frequencies, a reduction
constant K >= 2^64 (pi or turn units: 2 |omega| / 10^s or 4 |omega| / 10^s; radian constants always fit), |tau|,
|phi| >= 2^62, an order past 64), `ValueError`
(duplicate frequencies (after reduction), a wrong component count, NaN, a negative order), `OverflowError` (a
frequency that is not an int64 rational), `TypeError`, `StaleStateError` (a table scope that changed).

**Limitations.** One certified sin / cos per (row, frequency) (about 2 us; no harmonic recurrences); frequencies are
int64 rationals; actions return double rectangles (no exact bounds of rotations); only the sums run on CUDA; the
Walsh inverse is the only inverse transform (no general DFT inversion); at most 16 dimensions.
