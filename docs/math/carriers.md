# Numeric carriers and exactness conversions (FND-009)

`torbles/math_carriers.h` (every contract; inline helpers shared with CUDA), `math_carriers.c`,
`math_carriers_gpu.cu`, `math_carriers.py`. The carriers extend the first slice's checked int64 rationals. Each has
one law and one exactness contract; nothing converts implicitly.

| carrier | law | exact | C type | Python |
|---|---|---|---|---|
| `i64`, `i128` | Z, checked (overflow refused) | yes | `int64_t`, `tr_math_i128` | `I64`, `I128` |
| `bigint` | Z, sign + 64-bit limbs | yes (size budgeted) | `tr_bigint *` | `BigInt` (Python `int` marshalled) |
| `q64` | Q, canonical int64 (= `tr_math_rational`) | yes | `tr_num_rational` | `Q64` |
| `bigq` | Q, canonical | yes (size budgeted) | `tr_bigq *` | `BigRational` (`Fraction` marshalled) |
| `gfp` | GF(p), prime p < 2^62 | yes | `tr_gfp`, `tr_gfp_elem` | `GF`, `GFElement` |
| `binary64` | the dyadic value of a finite double | yes, no arithmetic | `double` | `float` |
| `interval`, `cinterval` | real / complex (rectangle) enclosure | **approximate**, certified | `tr_math_interval`, `tr_carrier_cinterval` | `Interval`, `ComplexInterval` |
| `series` | Q[[x]] / x^N | exact up to O(x^N) | `tr_series *` | `Series` |

```python
from fractions import Fraction
from torbles import math_carriers as mc
a = mc.BigInt(2**5000 + 1); a * mc.BigInt(-3); divmod(a, 7); a.divmod(7, "trunc"); a.gcd(10**40); a ** 3; a >> 9
mc.BigRational(Fraction(1, 3)).enclose()          # Interval([RD(1/3), RU(1/3)], approximate)
F = mc.GF(2**61 - 1); F(Fraction(1, 3)).inverse(); F(5).lift(symmetric=True)
mc.crt(residues, [mc.GF(p) for p in primes]); mc.rational_reconstruct(u, m)        # exact multi-modular steps
mc.Series([0, 1], order=8).exp(); mc.Series([1, 1], order=8).log()
mc.conversion("interval", "bigq")                 # "forbidden"
mc.convert(mc.BigInt(2**70), "i64")               # OverflowError: narrowing is checked
F.vec("mul", a, b, backend="cuda"); F.dot(a, b); F.matvec(A, x); mc.enclose_q64(n, d); mc.sum_i64(x, groups, k)
```

**Laws never mix.** A binary operation takes operands of one carrier (one modulus, one series order); anything else
is `CarrierError` (native code -7), decided natively (`tr_carrier_combine`, element moduli, handle types). Python's
own `int` and `Fraction` are accepted only where they are exact members of the operand's carrier (`int` for `BigInt`
and `BigRational`, `Fraction` for `BigRational`); `GF(p)` takes no integers (reduction is explicit: `F(x)`), intervals
take no exact values (`Interval.enclose(x)`), `BigInt` has no true division.

**Conversion matrix** (`mc.conversion(src, dst)`, native table): *lossless* (i64 -> i128 -> bigint -> bigq, q64 ->
bigq, binary64 -> bigq / interval, interval -> cinterval, exact -> constant series), *narrowing* (to a narrower exact
carrier: `OverflowError` out of range, `NoSolutionError` for a non-integral rational to an integer carrier; cinterval
-> interval needs im = [0, 0]), *enclosing* (exact -> interval: the tight [RD(x), RU(x)], approximate), *reduction*
(exact -> GF(p); a rational with denominator 0 mod p is `NoSolutionError`), *lift* (GF(p) -> i64 / bigint: v in
[0, p) or symmetric), *truncation* (series to a lower order), *forbidden* (`CarrierError`: interval -> anything exact,
GF(p) <-> rational, GF(p) -> GF(q), anything -> binary64, series -> exact, ...).

**Exactness.**
- `BigInt`: schoolbook multiplication, Knuth's division (floor or truncating; `a = q b + r`, `|r| < |b|`), Lehmer's
  gcd, decimal text both ways. `BigRational`: canonical after every operation (Henrici's gcd tricks).
- GF(p): p verified prime (deterministic Miller-Rabin, bases 2..37); 128-bit products with Barrett reduction (the same
  inline code on the CPU and in the kernels); p >= 2^62 is `UnsupportedError`, a composite `ValueError`.
- CRT (Garner): the unique x in [0, M) (or (-M/2, M/2]); rational reconstruction returns n/d with |n| <= N, 0 < d <= D,
  n = u d mod m (default N = D = isqrt((m-1)/2), unique when 2ND < m), else `NoSolutionError`.
- Series: every coefficient exact up to the order; derivative lowers the order by one, integral raises it by one;
  reciprocal needs c0 != 0, composition an inner series with c0 = 0, exp c0 = 0, log c0 = 1 (else `NoSolutionError`).
- Intervals: outward rounding with tight endpoints (RD / RU of each exact endpoint, via exact TwoSum / fma error terms;
  products, quotients and square roots below 2^-969 are stepped out one more ulp); overflow gives [DBL_MAX, inf];
  division by an interval containing 0 and
  the square root of an interval with a negative part are `NoSolutionError`; comparisons are three-valued
  (`less`, `greater`, `equal` points, `unknown`). Every `Interval` / `ComplexInterval` reports `approximate=True`,
  `certified=True`, `precision_bits=53` (from the native carrier table, `mc.describe`), and none converts back to an
  exact value. Endpoints given directly must be exact doubles (`Interval(2**60 + 1)` is refused, not rounded).

**Budgets** (`math_budgets.Budget`, refusals `AlgebraBudgetError` with `.resource/.need/.limit`, before allocation):
`bytes / 8` is the limb budget of every big number an operation creates; the bounds (in limbs, `L` = limbs of an
operand) are mul `La + Lb`, add/sub `max + 1`, divmod `max + 2`, gcd `max + 1`, shl(k) `La + k/64 + 1`, pow(e)
`ceil(e * bits / 64) + 1` (checked before computing), from_decimal `(3322 digits + 63999) / 64000 + 1`, crt `k + 1`.
Series need degree `N - 1` and support `N`. The CUDA entry points budget their device copies (`mc.device_bytes`),
matvec also `tensor = rows * cols`, sums `states = groups`.

**CUDA** (`tr_mgpu_carriers_*`, `backend="cuda"`; results say `cuda` or `cuda-emulated`): GF(p) elementwise
add/sub/mul/div/neg/inv, dot, matrix-vector product, int64 reduction mod p; tight enclosures of int64 rationals; exact
128-bit grouped sums; exact 192-bit int64 dot products. One output slot per element or exact multiword atomic
accumulation, so results do not depend on thread order; they equal the CPU functions bit for bit.

**Limitations.** Multiplication is schoolbook (no Karatsuba/FFT: quadratic beyond a few thousand bits); decimal
conversion is quadratic; series coefficients are rationals only (no GF(p) series) and composition is cubic in the
order; intervals have binary64 endpoints only (no multiprecision intervals, no transcendental functions).
