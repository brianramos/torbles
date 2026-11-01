# Bounded convex and polynomial optimization (OPT-002)

`torbles/math_polyopt.h` (every contract; the box enclosures shared with CUDA), `math_polyopt.c`,
`math_polyopt_gpu.cu`, `math_polyopt.py` (opt-in). Beyond OPT-001's boxes: convex QPs and LPs under general linear
constraints solved exactly with verified certificates, and small polynomial problems answered by certified branch
and bound as an enclosure [lower, upper] of the global minimum, never a local optimum presented as a global one.

```python
from fractions import Fraction
from torbles import math_polyopt as po, math_opt as opt
r = po.lp([1, 1], A_ub=[[-1, -2]], b_ub=[-4], bounds=[(0, None), (0, None)])    # exact simplex (Bland)
r = po.qp(H, g, c0, A_ub=A, b_ub=b, A_eq=E, b_eq=e, bounds=[(0, 1), None, (None, 3)])   # H PSD: exact active set
r = po.qp_from(opt.Quadratic.from_moments(state, 0), A_eq=[[0, 1, 1]], b_eq=[1])     # an OPT-001 objective
r.status, r.point, r.value, r.multipliers, r.kkt, r.unique, r.face, r.active, r.gradient, r.slack
r.farkas, r.ray                                         # infeasible / unbounded certificates
r = po.polynomial(f, {"x": (-2, 2), "y": (-2, 2)}, [(g, "<=", 0), (h, "=", 0)],
                  tolerance=Fraction(1, 10**6), backend="cpu" | "cuda", batch=32, time_limit=None, budget=None)
r.status, r.lower, r.upper, r.gap, r.point, r.feasibility, r.reason, r.leaves, r.pruned, r.open
pr = po.Problem(f, box, constraints); pr.value(x); po.boxes(pr, [[(lo, hi), ...], ...], ubound, backend)
```

## 1. Convex QPs and LPs (exact)

Minimize x^T H x / 2 + g^T x + c0 (H symmetric PSD; `lp`: H = 0) subject to A x <= b, E x = e and per-variable
bounds (either side optional). Declared scope n <= 8 (the representation allows 64 variables and 256 rows; the
budgets bound the work). All arithmetic is exact (FND-009 big rationals).

- LPs: a tableau simplex over x = u - v with slacks, phase 1 with artificials and Bland's rule (no cycling).
- QPs: an exact primal active-set method from the phase-1 vertex; each equality-constrained step is solved exactly
  (LIN-002), a zero-curvature descent direction comes from LIN-002's inconsistency certificate, ratio tests and
  multiplier drops by the smallest index. Its iteration budget yields `NotConvergedError`, never a wrong answer.
- Every answer is verified exactly before it is returned. `optimal`: x*, multipliers lambda >= 0 (rows and bounds)
  and mu with H x* + g + A^T lambda + E^T mu = 0, primal feasibility and complementary slackness (`kkt=True`).
  Uniqueness is decided: the optimal set is {x feasible : H x = H x*, g^T x = g^T x*}, a point iff max / min x_k over
  it equal x*_k (2n exact LPs, skipped when H is positive definite or every nonbasic reduced cost is strictly
  positive); a nonunique answer carries a second optimal point (`face`), so the optimal face is stated, not hidden.
  `infeasible`: a Farkas certificate (y >= 0 for rows and bounds, z free: A^T y + E^T z = 0, b^T y + e^T z < 0).
  `unbounded`: a feasible anchor x0 and a recession direction d (A d <= 0, E d = 0) with `curvature` d^T H d < 0, or
  curvature 0 and `rate` g(x0)^T d < 0, so q(x0 + s d) = q(x0) + s rate + s^2 curvature / 2 -> -inf (for a convex
  problem the curvature is 0 and H d = 0).
- An indefinite H is not convex. The equalities are eliminated exactly (x = x_p + N y, R = N^T H N). When no
  inequality varies on the affine set {E x = e} (no rows and no bounds, or rows constant on it) the inertia of R
  decides the problem exactly: a negative direction of R is a curvature ray; otherwise q is convex on the feasible set
  and R y = -N^T g(x_p) gives the optimum (KKT-verified, `convex=True`, unique iff R is definite) or a rate ray.
  Otherwise a negative direction of R that is a recession direction, or a ray of the LPs bounding the polyhedron,
  certifies unboundedness when one is found; else the problem is handed to section 2 (`handed=True`) when the
  polyhedron is bounded (its box from 2 dim exact LPs) and dim <= 4: the answer is then an enclosure. Otherwise
  `UnsupportedError` (an unbounded nonconvex problem without such a direction is not decided).

## 2. Small polynomial problems: certified branch and bound

Minimize f(x) over a box (every bound finite, rational) subject to g_k(x) <= 0, >= 0 or = 0: n <= 4 variables
(`BB_VARS`), at most 64 constraints, degree <= 8 per variable and a dense size prod (deg_v + 1) <= 256 per
polynomial; polynomials are first-slice Expressions or `{exponents: coefficient}` dicts.

- Best-first search over dyadic boxes (bisection of the least-split variable, at most 60 bisections each). A batch
  of boxes (`batch`) is enclosed at once: the natural interval extension intersected with the centered form (the
  Taylor expansion at a double point of the box), in FND-009 outward-rounded intervals with contraction-safe
  directed operations. A box is pruned only soundly: `infeasible` when some constraint is violated on the whole box,
  `bound` when its lower bound exceeds the incumbent, `tolerance` when it cannot improve the incumbent by more than
  the tolerance (its bound is kept in `lower`).
- Incumbents are exact: f at a box corner or midpoint whose constraints hold exactly (big rational evaluation).
- The answer encloses the global minimum f*: `upper` = f(`point`) exactly at a feasible witness, `lower` = the
  smallest certified lower bound of any box not pruned (an exact dyadic rational). `complete` only when
  upper - lower <= tolerance; else `incomplete` with the reason ("nodes": the iteration budget, "time":
  `time_limit` seconds, "depth"), with the bounds reached and `upper` None when no feasible point was found.
  Feasibility is "feasible" (the witness), "infeasible" (a certificate: `leaves`, boxes partitioning the box, each
  with a constraint its enclosure proves violated; the tests re-check every leaf exactly) or "unknown".
- One variable without constraints: INV-001's roots of f' (exact rational critical points, else isolating intervals
  refined until the gap is within the tolerance), plus the box ends.
- `po.boxes` exposes the enclosure of a batch of boxes on the CPU (`tr_polyopt_boxes`) or on CUDA
  (`tr_mgpu_polyopt_boxes`: one work item per box), which run the same inline functions and agree bit for bit; the
  search itself, its heap and the exact incumbents stay on the CPU (`backend="cuda"` sends each batch to the device).

## Budgets and refusals

QP / LP: variables = n, tensor = the tableau (rows x columns), iterations = pivots + active-set steps (including the
LPs of the uniqueness test and of the hand-off box), bytes per big number. Polynomial: variables, degree, support =
terms, iterations = boxes evaluated (truncates: `incomplete`; the one-variable method's INV-001 evaluations are
refused past it instead), output = leaves of an infeasibility certificate,
bytes = open boxes and leaves; `boxes`: output = boxes, bytes = the job (CUDA: its device copies). Checks happen
before allocation. Refusals: `AlgebraBudgetError`, `UnsupportedError` (sizes past the limits, an unbounded or
one-sided box for a polynomial, an indefinite QP that cannot be handed off), `NotConvergedError` (the active-set
iteration budget), `ValueError` / `TypeError` (malformed or float input). The Python default tolerance of the
branch and bound is 2^-20 (tolerance 0 asks for the exact minimum and usually exhausts the budget: `incomplete`).

## Limitations

Exactness of the convex method is paid in big-rational pivots: it is meant for n <= 8 and tens of rows. The branch
and bound is exponential in n and in log(1 / tolerance); its enclosures are first-order (no relaxations), so tight
tolerances on flat or degenerate minima stop as `incomplete`. With equality constraints a witness exists only where
a dyadic corner or midpoint satisfies them exactly; otherwise feasibility stays "unknown" with `upper` None. An
indefinite QP over an unbounded polyhedron is decided only on an affine feasible set or when a recession direction
of negative curvature is found (copositivity is not decided); a bounded one only up to dimension 4 after the
equalities. CUDA covers the box enclosures only (tested here against the CPU in the emulation build); the simplex,
the active set and the roots run on the CPU.
