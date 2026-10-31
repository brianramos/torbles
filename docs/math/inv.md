# Linear and univariate polynomial inverse answers (INV-001)

`torbles/math_inv.h` (every contract; the shared per-point functions), `math_inv.c`, `math_inv_gpu.cu`,
`math_inv.py`. "Which parameters give this target": the exact solution set of linear targets in a box, every real root
of a univariate target in declared bounds, and an exhaustive search of a finite grid. The functions inverted are
first-slice polynomials (`torbles.mathematics`, e.g. a RESP-001 response function `MomentBasis.reduce(...)`) or
explicit coefficients. Exact values cross as FND-009 big rationals (`Fraction` in Python); nothing is rounded.

```python
from fractions import Fraction
from torbles import math_inv as mi
from torbles.mathematics import Context
ctx = Context(); t, u = ctx.symbols("t", "u")
mi.solve([(2*t + u, 3), (t - u, 0)])                          # Linear('unique', point={'t': 1, 'u': 1})
mi.solve([(t + u, 2)], domain={"t": (0, 1), "u": (None, 1)})   # 'unique' (1, 1): the box admits one point
mi.solve_affine([[1, 1, 1]], [1], lo=[0, 0, 0])               # 'family': point, directions, free, ranges
mi.solve([(loss.diff("theta"), 0)])                           # the least-squares zero of a RESP-001 gradient
r = mi.roots(t**3 - 2*t, 0, lo=-2, hi=2, width=Fraction(1, 2**40))   # 0 exact; +-sqrt(2) isolated
r = mi.roots([-2, 0, 1], lo=0).refine(Fraction(1, 10**12), backend="cuda")
mi.signs([[-2, 0, 1]], [Fraction(3, 2), Fraction(1)], which=[0, 0], backend="cuda")   # exact signs [1, -1]
s = mi.sweep(t*t + u, {"t": range(-8, 8), "u": range(16)}, target=5, budget=Budget(iterations=100))
s.complete, s.reason, s.searched, s.next, s.matches, s.bound_bits, s.primes
```

## Linear targets

`solve(equations, unknowns=None, domain=None)` (first-slice expressions of degree <= 1 in the unknowns; `(expr,
target)` pairs or expressions = 0) and `solve_affine(A, b, lo, hi)` (rational rows; `None` = 0 or unbounded). Gauss-Jordan
elimination over Q; a pivot is a nonzero entry, so no division by zero ever happens (`a t = c` with `a = 0` is "whole"
or "empty"). Unknowns fixed by the box (`lo == hi`) are substituted first. Answers (`Linear.kind`):

| kind | meaning | data |
|---|---|---|
| `unique` | exactly one solution in the domain (also a family the box reduces to one point) | `point` |
| `empty` | `reason="inconsistent"` (no solution at all) or `"outside"` (solutions exist, none in the box) | |
| `whole` | every point of the domain is a solution (every equation reduces to 0 = 0) | |
| `family` | `{point + sum_d y_d directions[d]}` in the box: `point` a witness inside the domain, `directions[d]` a basis of the homogeneous solutions with 1 at `free[d]`, `ranges[d]` the exact `(lo, hi)` of `y_d` over the family in the box (`None` = unbounded) | `point`, `directions`, `free`, `ranges` |

In a box the family is decided exactly by Fourier-Motzkin elimination over the parameters: feasibility, the witness by
back-substitution (the parameter value nearest 0 in each projected range), and the projection of every parameter.

## Univariate roots

`roots(f, target=0, lo=None, hi=None, variable=None, width=None, backend="cpu")`: every real root of `f - target` in the
closed bounds (`None` = unbounded: the Cauchy bound covers every real root). `f` is a first-slice expression in one
variable or coefficients (low degree first). `f - target` becomes a primitive integer polynomial; Yun's square-free
factorization (exact divisions over Z[x]) gives the multiplicities, the Sturm sequence of the square-free part
isolates the distinct roots by bisection at exact rational points. `Roots.kind`: `finite`, `empty` (`reason=
"constant"` for a nonzero constant) or `whole` (the zero polynomial). Each `Root`:

- `exact=True` only when proved: an exact zero at a bound or a split point (`how="evaluated"`), a degree-one factor
  (`"linear factor"`), or the rational-root candidate (`"rational candidate"`: the fraction with the smallest
  denominator in the isolating interval; a rational root u/v of a primitive factor has v | lc, so a candidate whose
  denominator exceeds |lc| certifies the root irrational, `how="irrational"`). `value` is then the root.
- otherwise an open isolating interval `(lo, hi)` with rational endpoints, exactly one root of `f - target` inside,
  and a certified sign change of its square-free factor (`sign_lo` at `lo`, `-sign_lo` at `hi`; `value` is None: an
  irrational root is never given as a number).
- `multiplicity`, `boundary` (`"lo"`, `"hi"`, `"lo and hi"`: a root exactly at a declared bound). `Roots.factors` are
  the square-free factors (factor i has the roots of multiplicity i + 1).

`Roots.refine(width, backend)` bisects every isolating interval to `width`; the sign at each split point comes from the
batched evaluator (`backend="cpu"`: `tr_inv_signs_eval`; `"cuda"`: the kernel), so both give identical intervals.

## Batched exact signs (CPU and CUDA)

`signs(polys, points, which)`: the sign of `polys[which[i]]` at `points[i]`. `S = v^d P(u/v)` is computed modulo `k`
primes in (2^61, 2^62) (Barrett arithmetic of `math_carriers.h`), `k` from the proven bound `|S| < 2^B`, `B =
bits(sum |a|) + d max(bits(|u|), bits(v))`, `61 k >= B + 2`; the sign follows from Garner's mixed-radix digits
compared with those of `(M - 1) / 2` (zero iff every residue is 0). `Signs.primes` and `Signs.bound_bits` are the
proof. The moduli table (`mi.primes()`) is verified natively (deterministic Miller-Rabin).

## Finite domains

`sweep(f, grid, target=0, start=0, budget=None, backend="cpu")`: the exact solutions of `f = target` on `grid =
{name: values}` (1..3 coordinates, lexicographic order, the first coordinate most significant). Denominators are
cleared (`H(m) = 0` iff the grid point solves it) and every point is tested by residues modulo `k` primes with
`prod p > |H|` (`Sweep.bound_bits`, `Sweep.primes`). Points `start .. start + count - 1` are examined, `count =
min(remaining, iterations budget)`: a truncated search is `complete=False, reason="iterations"` with `next` the index
to resume from. Matches are listed in index order up to the output budget; one more stops the search before it
(`reason="output"`, `next` = that match), so the matches listed are all of those in `start .. next - 1`. CUDA: one
work item per grid point, the matches in a bitmap set by atomic OR (`Sweep.bitmap_words` equals the CPU's).

## Budgets and refusals

| request | budget (refused before allocating unless noted) |
|---|---|
| linear | `variables` = unknowns (also <= 64), `tensor` = m (n + 1), `support` = inequalities of one Fourier-Motzkin system, `iterations` = inequalities generated |
| roots | `degree`, `iterations` = exact evaluations (bisection, rational detection; refinement: rounds x roots), `output` = roots |
| signs | `output` = points, `bytes` = the job (CUDA: its device copies, `signs_device_bytes`) |
| sweep | `variables` = coordinates, `support` = terms, `bytes` = plan and bitmap (CUDA: `sweep_device_bytes`); `iterations` and `output` TRUNCATE (reported), never refuse |

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `OverflowError` when the multimodular bound needs more than
128 primes (about 7800 bits) or a point more than 128 limbs; `UnsupportedError` for a linear target of degree > 1 or
more than 3 searched coordinates; `DomainError` for a variable that is neither an unknown nor searched; `ValueError`
for `lo > hi` or malformed input. Floats are not exact input (`TypeError`).

**Limitations.** Sturm sequences and Fourier-Motzkin are exact but not the asymptotically fastest methods (high degree
and many free parameters in a box grow quickly; both are budgeted). The finite-domain search needs a first-slice
polynomial (int64 rational coefficients, degree <= 8); grid values are arbitrary rationals. Linear targets from
first-slice polynomials have at most 8 unknowns (64 through `solve_affine`). Refinement and signs are limited by the
128-prime representation; isolation itself (bigint Sturm evaluation) is not.
