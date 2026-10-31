# Entropy, mutual information and contingency statistics (DIST-003)

`torbles/math_info.h` (every contract; the certified logarithm and the per-cell terms are inline and shared with
CUDA), `math_info.c`, `math_info_gpu.cu`, `math_info.py` (opt-in). Information and contingency statistics of the
retained count support of a DIST-001 joint histogram (`docs/math/dist.md`), read from the histogram's states (no row
is scanned), with certified logarithms and exact integer accumulation.

```python
from torbles import math_dist as md, math_info as mi
h = md.histogram(scope, ["x", "y", "z"])                      # the joint support state (NULL: a category of its own)
inf = mi.information(h, mass="count", exclude=["z"])           # declared: NULL cells of z dropped; backend="cuda"
inf.entropy("x"); inf.entropy(["x", "y"], units="bits")        # {"lo", "hi", "bounds", "total", "cells", ...}
inf.conditional_entropy("x", given="y"); inf.mutual_information("x", "y", given="z")
inf.chi_square("x", "y", exact=True)                           # + "exact" (Fraction), "df", "rows", "cols"
inf.g_test("x", "y"); inf.cramers_v("x", ["y", "z"], exact=True)   # + "exact_v2" (Fraction)
inf.query([dict(op="mutual_information", a=["x"], b=["y"], group=g), ...])   # a batch from the shared support
inf.group(g); inf.info; mi.need(h); mi.ln_bounds(m)
```

## Domain

- Source: a histogram of k = 1..3 coordinates and G groups. Mass: `count` (rows) or `weight` (weight sums; every
  present weight >= 0, else `DomainError`; a weight total at or past 2^64 per group is `UnsupportedError`). Absent
  cells and zero-mass cells are structural zeros: they contribute 0 log 0 = 0 and are never missing data.
- NULL: a category of its own per coordinate, unless the coordinate is declared in `exclude` (`"all"` for every one):
  then every cell with a NULL in an excluded coordinate is dropped before any statistic (listwise), and N is the
  remaining mass. The declaration is part of the state (`info["excluded"]`).
- The support: for every coordinate subset S the marginal table T(S) of the kept cells (computed exactly from the
  joint cells), and for every unordered pair of disjoint coordinate sets (A, B) the cells of T(A u B) with the masses of
  their projections. A statistic needs the coordinates jointly: asking for one the state does not hold (mutual
  information of a 1-D state, a name outside the histogram) raises `UnsupportedError`. Nothing is inferred from
  marginals or moments: `information` takes a joint histogram only (a prefix state, a moment bank or two marginal
  histograms are `TypeError`).

## Statistics (per group g; N = the kept mass)

| op | value | exact bounds (nats) |
|---|---|---|
| `entropy` (a: one or several coordinates) | H(A) = ln N - (1/N) sum m ln m over T(A) | / (N 2^112) |
| `conditional_entropy` | H(A given B) = H(A, B) - H(B) | / (N 2^112) |
| `mutual_information` | I(A; B) = H(A) + H(B) - H(A, B) | / (N 2^112) |
| `conditional_mi` (`given=`) | I(A; B given C) = H(A, C) + H(B, C) - H(A, B, C) - H(C) | / (N 2^112) |
| `g_test` | G = 2 sum O ln(O / E) = 2 N I(A; B) | / 2^112 |
| `chi_square` | X^2 = sum over T(A) x T(B) of (O - E)^2 / E, E = m_a m_b / N (structural zeros included) = N (sum O^2 / (m_a m_b) - 1) | / 2^128; exact rational on request |
| `cramers_v` | V = sqrt(X^2 / (N (min(r, c) - 1))) | V^2: / ((min(r, c) - 1) 2^128); exact V^2 on request |

r, c are the categories (cells) of T(A), T(B); `df` = (r - 1)(c - 1). Units: `nats` or `bits` (H and I only; the bits
enclosure divides by an enclosure of ln 2; the exact bounds stay in nats). With weight masses G and X^2 are in logical
weight units. Empty answers (`empty`, no value): an absent group, N = 0, V with min(r, c) < 2. Every statistic is >= 0
(lower bounds are clamped at 0) and V <= 1.

## Exactness and precision

- The counts and masses are exact integers. Each cell's term is enclosed with integers: m ln m by
  floor(m lo / 2^8) <= m ln m 2^112 <= ceil(m hi / 2^8), where lo <= ln(m) 2^120 <= hi comes from a 120-bit
  fixed-point logarithm (m = 2^e f, f in [sqrt(2)/2, sqrt(2)], ln f = 2 atanh((f - 1)/(f + 1)) by 24 series terms, every
  product and quotient truncated downward; the proven error of the series is below 34 ulps of 2^-120, ln 2 from its
  floor plus [0, 1) ulp: hi - lo <= 80 + e <= 144 ulps); a chi-square cell by floor / ceil(O^2 2^128 / (m_a m_b)) (exact
  integer division). The bounds are summed exactly in 256 bits, so the sums do not depend on the order of addition.
- An answer carries the exact rational bounds (`bounds`: Fractions; their numerators are combinations of the sums) and
  the double enclosure [RD(lower), RU(upper)] (`lo`, `hi`), then divided by ln 2 or square-rooted outward.
  Stated widths: the exact bounds of H and I are at most 2^-104 apart (nats), of G at most 2^-101 N, of X^2 at most
  N cells 2^-127; the doubles add at most 2 ulps (4 in bits). The tests check every enclosure against 60-digit decimal
  logarithms (containment, exactly) and against math.log doubles (up to their own rounding), and these widths.
- Exact contingency values (`exact=True`, CPU): X^2 and V^2 as canonical fractions of FND-009 big integers: with D, L
  the lcm of the B- and A-masses, X^2 = N (T - D L) / (D L), T = sum O^2 (L / m_a)(D / m_b).

## CUDA

`backend="cuda"`: the support is built on the CPU; `tr_mgpu_info_sums` computes every per-cell term with the same
inline functions and accumulates them, one work item per 64 entries (local 256-bit sums while the key stays the same,
then exact 256-bit atomic additions: four 64-bit atomics with carries, order-independent totals). The device sums are
checked and adopted (`tr_info_state_build`); they equal the CPU's bit for bit (`Information.equal`).

## Budgets and refusals

| Step | checked before allocation |
|---|---|
| support, stage 1 | variables = k, states = the kept joint cells, bytes = the sorting scratch (entries, a permutation and a run id per subset) |
| support, stage 2 (tables counted) | states = table cells + contingency cells, support = the largest table of a group, bytes = scratch + the support |
| state | states = G x slots, bytes = the sums and group tables |
| CUDA sums | states = G x slots, bytes = the device copies (16 per table cell, 32 per contingency cell, 64 per slot) |
| requests | output = requests |
| exact X^2 / V^2 | support = the contingency cells, bytes = every big number held (bounded from the masses' bit lengths) |

Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`). A request's own status: `ValueError` (malformed:
unknown op or unit, empty or overlapping masks, bits for G / X^2 / V), `UnsupportedError` (coordinates the state does
not hold jointly). Also `DomainError` (negative weights as a mass), `UnsupportedError` (a weight total at or past 2^64).
