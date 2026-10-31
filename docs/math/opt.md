# Quadratic objective minimizers (OPT-001)

`torbles/math_opt.h` (every contract; the face filter shared with CUDA), `math_opt.c`, `math_opt_gpu.cu`,
`math_opt.py` (opt-in). Exact minimization of q(t) = t^T A t / 2 - b^T t + c (A symmetric, rational) over R^n or a
box, with singular, indefinite and boundary cases explicit.

```python
from fractions import Fraction
from torbles import math_opt as opt
q = opt.Quadratic.from_matrices(A, b, c)                     # int / Fraction entries
q = opt.Quadratic.from_moments(state, g, ridge=Fraction(1, 2), features=["x1"])   # a LIN-001 least-squares loss
q = opt.Quadratic.from_polynomial(loss, ["theta"])           # a first-slice polynomial of degree <= 2
r = opt.minimize(q)                                          # Result: status, point, value, family, ...
r = opt.minimize(q, bounds=[(0, 1), None, (-2, Fraction(5, 2))], backend="cuda")   # a box, n <= 8
opt.faces(q, bounds, backend="cpu" | "cuda")                 # the face filter's statuses
q.A, q.b, q.c, q.value(t)
```

## Objectives

- `from_matrices`: n <= 63, A must be symmetric (`ValueError`). `from_moments`: the loss
  sum (y - z^T t)^2 + lambda |t_features|^2 of a fit of a LIN-001 state in value units (A = 2 (Z^T Z + lambda I_f),
  b = 2 Z^T y, c = y^T y); its minimizer is the state's exact regression. `from_polynomial`: a
  `torbles.mathematics` Expression / Function read natively; terms of degree > 2 are `UnsupportedError`, a variable
  that is not one of `parameters` is `ValueError`.

## Unconstrained (exact, from the inertia and the solution set of A t = b, LIN-002)

| A | b | status | returned |
|---|---|---|---|
| positive definite | any | `unique` | t* = A^-1 b, value c - b^T t* / 2 |
| PSD singular | in range(A) | `nonunique` | t0, `family` = a basis of ker A, the value |
| PSD | not in range(A) | `unbounded`, certificate `slope` | d in ker A with b^T d > 0: q(s d) = c - s b^T d |
| a negative eigenvalue | any | `unbounded`, certificate `curvature` | d with d^T A d < 0 |

Never a division by zero; the gradient at an optimum is checked to be 0 exactly (`kkt`).

## Boxes (n <= 8)

- Per parameter `(lo, hi)` with both finite (rational) or `None` (free). A one-sided bound is `UnsupportedError`;
  lo > hi is `InfeasibleError`; lo == hi fixes the parameter.
- Boundedness first, exactly: with free coordinates U, unbounded iff A_UU has a negative eigenvalue (`curvature`
  along U) or a kernel vector v of A_UU has v^T (A_UK t_K - b_U) != 0 at some feasible t_K (`slope`; a vertex is
  exhibited); otherwise the minimum is attained (bounded coordinates form a compact box).
- Then the 3^|K| faces of the bounded coordinates K (each at lo, at hi, or free): the minimum lies at a stationary
  point of a face whose restricted kernel vanishes on the bounded coordinates. Per face the restricted system is
  solved exactly; a candidate needs unique feasible bounded coordinates and the KKT signs (gradient 0 on free
  coordinates, >= 0 at a lower bound, <= 0 at an upper bound), checked exactly. The minimum over the candidates is
  the exact optimum, boundary optima included; several distinct optimal points (`optima`) or directions of constant
  value along free coordinates (`family`) make it `nonunique`. `active`: -1 at lo, 1 at hi, 0 free.
- The face filter (the data-parallel part; CUDA: one work item per face, `tr_mgpu_opt_faces`; the CPU runs the same
  inline code, bit for bit): interval Gaussian elimination with outward rounding (FND-009) on enclosures of A, b and
  the bounds discards a face only when it rigorously cannot hold the optimum (`infeasible` stationary point, a
  violated `kkt` sign, a negative diagonal: `not psd`); a pivot interval containing 0 leaves it `undecided` (solved
  exactly). The answer does not depend on the filter; `Result.filtered` and `exact_checks` report its effect.
- Unbounded results are certificates: `anchor` (feasible), `direction` (0 on bounded coordinates), `rate` (d^T A d
  < 0, or g(anchor)^T d < 0 with d^T A d = 0), verified natively.

## Budgets and refusals

variables = n, degree = 2, tensor = n (n + 1) (the system [A | b]), iterations = 3^|K| faces, output = n (a box:
n (faces + 1), every optimum found), bytes; the CUDA filter budgets its device copies and the faces. Refusals:
`AlgebraBudgetError` (before allocation), `UnsupportedError` (n > 63, a box with n > 8, a one-sided bound, degree
> 2), `InfeasibleError`, `ValueError`, `TypeError` (float input).

## Limitations

Half-infinite bounds (t >= 0 alone) are not supported (boundedness of a nonconvex quadratic over such a cone needs a
copositivity decision); the face enumeration is exponential (3^8 = 6561 faces at most); the exact phase and the
boundedness check run on the CPU; general linear constraints are out of scope.
