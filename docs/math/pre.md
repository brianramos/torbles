# Affine and monotone-piece preimages (PRE-001)

`torbles/math_pre.h` (every contract; the scan shared with CUDA), `math_pre.c`, `math_pre_gpu.cu`, `math_pre.py`
(opt-in). A predicate on a derived value `y = f(x)` becomes the exact set of stored codes of the generator column,
which the engine scans; nothing is approximated.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_pre as mp
codec = mp.Codec.from_table(t, "x")          # logical = code + offset, x = logical / 10^scale, codes 0 .. 2^width - 1
s = mp.preimage(codec, mp.affine(Fraction(-3, 2), 7), "<", Fraction(1, 3))   # also <=, =, <>, >=, >
s.spans; s.empty; s.all; s.codes; s.contains(code); s.info                    # exact code intervals
mp.preimage(codec, mp.square(), "<=", 50)            # symmetric; "=" gives zero, one or two points
mp.preimage(codec, mp.residue(7), "in", values=[1, 3])   # residue classes of codes (code mod 7 in ...)
mp.preimage(codec, mp.pieces([(None, -1, "affine", -2, 0), (0, None, "square", 1, -50)]), ">", 30)
s.certificates                                       # FND-007 laws of the transformation, one per branch
bits, stats = s.scan(mbk.Scope.from_table(t, ["x"]), "x", direct=True, backend="cuda")   # stats["mismatches"] == 0
s.plane_program(t, "x")                              # the engine's bit-plane program (OR of BETWEEN, AND valid)
```

## Maps and rounding (T = 10^scale, L = code + offset)

| map | refused | transform |
|---|---|---|
| `affine(a, b)`: `a x + b` | `a = 0` (DomainError) | `L op' t`, `t = (c - b) T / a`, `op'` flipped for `a < 0`; `L < t`: `L <= ceil(t) - 1`, `L <= t`: `L <= floor(t)`, `L > t`: `L >= floor(t) + 1`, `L >= t`: `L >= ceil(t)`, `=`: `t` integral, `<>`: all but an integral `t` |
| `square(a, b)`: `a x^2 + b` | `a = 0` | `L^2 op' C`, `C = (c - b) T^2 / a`: `<=` `|L| <= isqrt(floor C)`, `<` `|L| <= isqrt(ceil C - 1)`, `=` C a square: two points (one at 0), else none; `>=`, `>`, `<>` the complements; both branches always |
| `residue(m)`: `x mod m` | `m = 0` (DomainError), `m T` past int64 (UnsupportedError) | residues `R = L mod mT` with `R / T op c`, then `code = R - offset (mod mT)`: residue spans (plain intervals when the domain is not larger than `mT`) |
| `pieces([...])`: closed domains of `x`, each affine or square | a zero slope; pieces whose code ranges leave a gap (DomainError: a branch is missing) or overlap (ValueError: not a function) | each piece's preimage within its code range, all unioned; a piece whose image lies entirely on one side is decided by UNC-001's exact abstract evaluation (`info["decided"]`) |

Thresholds are FND-009 big rationals (no overflow, no rounding); integers are produced by floor, ceil and isqrt of
exact values. `"in"` takes any finite set of values (each an `=` preimage). The result is exact on the codec's domain
of codes: sorted, disjoint plain spans `(lo, hi)` (never adjacent) or residue spans `(lo, hi, M, rlo, rhi)`, with the
flags `empty` and `all` and the exact count of codes. NULL guard: a NULL value never satisfies (the scan requires the
row's value to be present; `plane_program` ANDs the validity column).

## Certificates (FND-007)

One LAW (`ring_identity`) per branch, a polynomial in the variable `code`: `D f((code + offset) / T)` expanded term
by term minus its code-space form (`P code + Q`, or `P2 code^2 + P1 code + P0`), `D` the positive common denominator;
for a residue map `(code + offset) - (code + R0) - M k0` with `offset = M k0 + R0`. FND-007 decides each canonically
zero; a law holds for every code, so on the codec's domain. A branch whose coefficients leave int64 has none
(`info["certified"]` counts them; the set stays exact).

## Scan and the direct check (CPU and CUDA)

`s.scan(scope, column, direct=False)`: bit r set when selected row r's value is present and its code (`logical -
offset`) is in the set; the scope column must have the codec's scale. `direct=True` also evaluates the original
predicate on every selected row (UNC-001's exact row evaluator on the branch's program `y = f(x)`, then the
comparison) and reports `mismatches` and the first one, `outside` (codes outside the domain), `nulls`, `in_set`,
`direct_true`; a row the exact evaluation cannot decide raises `OverflowError`. CUDA: one work item per 64-row word,
its own bitmap word, counts by atomics, first rows by atomic minimum; equal to the CPU bit for bit.

## Budgets and refusals

| request | needs |
|---|---|
| preimage | support = pieces and `in` values, output = spans, bytes = the set and its intermediate intervals; FND-009 numbers by bytes; at most 256 pieces and 4096 values (`UnsupportedError`) |
| scan | bytes = the bitmap (CUDA: the device copies of the selection, the column, the spans, the branch programs) |

`ValueError` (malformed codec, op, pieces), `TypeError` (floats are not exact input), `DomainError`,
`UnsupportedError`, `AlgebraBudgetError`, `OverflowError`.

## Limitations

Maps are affine, `a x^2 + b` (vertex at 0), `x mod m` with an integer `m`, and pieces of the first two; there is no
general polynomial or composed map; residue spans have no bit-plane program (a lookup is the engine candidate); a
self-extending (wide) column is not supported; the law certificates cover the algebraic substitution, the integer
rounding is proved by the exact computation and the exhaustive tests, not by a certificate.
