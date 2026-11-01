# Finite roots, residues and periodic preimages (PRE-002)

`torbles/math_preimg.h` (every contract; membership, the direct evaluation and the verification are inline and shared
with CUDA), `math_preimg.c`, `math_preimg_gpu.cu`, `math_preimg.py` (opt-in: `import torbles` does not load it). A
predicate on a derived value y = f(x) of a stored column becomes the exact set of stored codes that satisfy it, as a
union of intervals, residue classes or Boolean cubes; past the budget it falls back to a lookup bitmap or the direct
evaluation and never drops a solution. The codec and the predicates are PRE-001's (`math_pre`): logical value L =
code + offset, x = L / 10^scale, codes over the declared domain [lo, hi]; y op c (op `<`, `<=`, `=`, `<>`, `>=`, `>`,
c an int64 rational) or y IN values (at most 4096).

```python
from fractions import Fraction
from torbles import math_pre as mp, math_preimg as mq, math_bank as mbk
codec = mp.Codec.from_table(t, "x")
s = mq.preimage(codec, mq.poly([-6, 11, -6, 1]), ">", 0)      # (x - 1)(x - 2)(x - 3) > 0: monotone pieces
s = mq.preimage(codec, mq.powmod(2, 15), "=", 4)              # x^2 = 4 (mod 15): CRT over 3 and 5
s = mq.preimage(codec, mq.affmod(6, 1, 20), "<", 7)           # (6 x + 1) mod 20 < 7: classes of the period 10
s = mq.preimage(codec, mq.xor(0b1010, 8), "<=", 40)           # Boolean cubes of the stored code's bits
s = mq.preimage(codec, mq.poly([0, 1]), "in", values=[3, 9])  # finite sets
s.mode, s.repr, s.branches, s.codes, s.contains(code), s.info, s.certificates, s.cubes(width)
bits, stats = s.scan(mbk.Scope.from_table(t, ["x"]), "x", direct=True, backend="cuda")
s.verify(backend="cuda")                                      # every code of the domain against the predicate
s.plane_program(t, "x")                                       # the engine's bit-plane program (intervals, cubes)
mq.preimage(codec, m, "<", c, budget=Budget(output=8), fallback="lookup")    # auto | lookup | direct | none
```

## Maps and their branches

- `poly(coefficients)`: y = sum a_i x^i, rational a_i, degree <= 16. With K = 10^(scale d) lcm(den a_i), N(L) = K
  p(L / 10^scale) is an integer polynomial (FND-009 big integers) and y op c iff N(L) den(c) op K num(c). The real
  roots of N' are isolated by INV-001 (Sturm sequences, refined to width 1); between them N is strictly monotone on
  the integers, and on each monotone piece two binary searches with exact big-integer evaluations give the level
  sets: an interval per piece, a point or nothing for `=` (the exact integer roots), every IN value through the same
  pieces. Affine maps of either slope are the degree-1 case.
- `polymod(coefficients, m)`, `powmod(k, m)`: y = (sum c_i x^i) mod m, integers, scale 0, 1 <= m <= 2^62. y depends
  on L mod m only. `=` and IN: the residues solving P(r) = y (mod m), per prime power q of m (trial division, each q
  enumerated, q <= 2^24) combined by the CRT; the other operators enumerate every residue (m <= 2^24).
- `affmod(a, b, m)`: y = (a x + b) mod m: a x = y - b (mod m) is solvable iff g = gcd(a, m) divides y - b, then x =
  ((y - b) / g) (a / g)^-1 (mod m / g): residue classes of the period m / g.
- `xor(c, w)`, `bitand(c, w)`, `bitor(c, w)`: on the w low bits u of L - base (base default: the codec offset, so u is
  the stored code's bits; FIELD-001's GF(2) variables): the images of the dyadic blocks of {y op c} are cubes
  {u : u & mask = value} (XOR translates the fixed bits, AND fixes the bits of c, OR those outside c; a block the map
  cannot produce is empty). With another base the cubes become residue runs of codes mod 2^(H+1), H the highest fixed
  bit.

Representations: `intervals` (sorted, disjoint, non-adjacent code ranges; single points for finite sets),
`residues` (code mod M in [rlo, rhi], one period M; a domain no larger than M becomes intervals), `cubes`
(code & mask == value). `s.cubes(width)` gives the cube form of cubes and of residues of a power-of-two period
(residues mod 2^k are low-bit cubes); `plane_program` turns intervals into an OR of BETWEENs and cubes into an OR of
ANDs of bit literals within the domain, AND the column's validity (more than 1024 instructions, a lookup, a direct
set or residues of another period: `UnsupportedError`, the lookup or direct evaluation is then the candidate).

## Fallback (never a dropped solution)

When the union has more branches than the output budget, or building it would pass the iterations or bytes budget or
a representation limit (a period past 2^24, too many CRT classes), the set is a `lookup` (a bitmap of the residues
mod M for periodic maps, or of the domain's codes, at most 2^26 bits, computed exactly on the host) or `direct` (the
original predicate evaluated exactly on each value: POLY in W <= 8 64-bit words under a bound proven over the domain,
the periodic maps in 128-bit modular arithmetic, the bit maps exactly). `fallback="auto"` takes the lookup when it
fits the bytes budget, else the direct evaluation; `"lookup"`, `"direct"` force one; `"none"` refuses with the
exceeded resource. `info` says `mode`, `fallback` and `why` (`output`, `iterations`, `bytes`, `period`, `classes`).

## Exactness and certificates

Thresholds, critical points and level sets are exact (FND-009, INV-001); residues and cubes are exact integer and bit
arithmetic; the set is exact on the codec's domain and a NULL value never satisfies. Each set of a polynomial or
periodic map carries one FND-007 LAW certificate of its transformation (a ring identity FND-007 decides): POLY the
codec substitution sum n_i (code + offset)^i against its code-space form Q(code); POLYMOD the periodicity
P(r + m t) - P(r) - m Q(r, t) = 0; AFFMOD a (r + M t) + b - (a r + b) - (a / g) m t = 0 with M = m / g (none when a
coefficient passes int64; the set stays exact).

## Scan and verification (CPU and CUDA)

`scan(scope, column, direct=False)`: the bitmap of selected rows whose present value's code is in the set (any mode);
with `direct=True` the original predicate is also evaluated on every row and compared (PRE-001's statistics: nulls,
in_set, direct_true, mismatches, outside, the first mismatching row). `verify()`: every code of the domain (at most
2^26 codes, values within int64): members, true codes, mismatches and the smallest mismatching code. On CUDA
(`tr_mgpu_preimg_scan`, `tr_mgpu_preimg_verify`) one work item per 64-row word (per 64 codes) writes its own bitmap
word and adds its counts with one atomic each, atomic minima for the first rows: no barrier, no shared memory,
results bit for bit those of the CPU. The host validates the set's view and budgets the device copies first.

## Budgets and refusals

output = branches; iterations = exact evaluations (bisection steps, residues enumerated, trial divisions, CRT
classes) and, for `verify`, the codes; support = IN values and the polynomial's terms; degree; bytes (the set, the
scratch, the lookup). Past output, iterations or bytes the set falls back (or, with `fallback="none"`, refuses);
`AlgebraBudgetError` names the resource, need and limit. A scale for a periodic map, a width past 63, a modulus past
2^62, a domain below 0 for bit maps, or a direct check of a polynomial without a proven width is `UnsupportedError`;
a zero modulus is `DomainError`.

## Limitations

- Integer domains only for the periodic and bit maps (scale 0); polynomial maps of degree <= 16 with int64 rational
  coefficients and thresholds.
- Branch construction is exact but runs on the CPU; CUDA covers the scan and the verification.
- Residues of a period that is not a power of two have no bit-plane program (the lookup or the direct evaluation is
  the candidate); a plane program is limited to 1024 engine instructions.
