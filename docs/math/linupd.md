# Leave-group-out and low-rank model updates (LIN-003)

`torbles/math_linupd.h` (every contract; the modular system shared with CUDA), `math_linupd.c`, `math_linupd_gpu.cu`,
`math_linupd.py` (opt-in). Exact solved models on LIN-001 moment states and LIN-002's exact solves: leave-one-group-out
by subtracting group states from their total, Sherman-Morrison-Woodbury updates with exact denominators, row
deletions and insertions, ridge changes, and a floating variant with a certified bound.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_lin as lin, math_linupd as lu
st = lin.moments(scope, ["x1", "x2"], "y")                      # LIN-001 (any grouping, listwise or pairwise)
M = lu.fit(st, group=0, ridge=Fraction(1, 2), features=None, intercept=None, inverse=True)
M.x, M.nullspace, M.kind, M.rank, M.A, M.b, M.A0, M.inverse, M.det, M.rss, M.tss, M.r2, M.sigma2, M.coefficients
M.delete_rows([dict(x1=3, x2=-1, y=7)]); M.insert_rows([((3, -1), 7)])         # rank-r Woodbury, exact
M.update(U, V, db); M.ridge(Fraction(2), method="woodbury" | "rebuild"); M.subtract(st_grouped, g)
D.denominator, D.method, D.fallback                              # det(K); woodbury / rebuilt / subtracted / modular
L = lu.leave_one_group_out(st, ridge=0, backend="cpu" | "cuda")  # one Model per group: the data without it
plan = lu.logo_plan(st); plan.solve("cuda"); plan.residues(k, "cuda"); plan.finish(k, res, fields)
M.float_solve(tol=1e-9, fallback="fail" | "exact")               # binary64 with a certified error bound
lu.system(A, b, inverse=True)                                    # any square rational system
```

**Models.** A model is an immutable solved square system with exact rationals: LIN-002's normal equations of a fit
(value units, exact ridge on the features never on the intercept, feature subset), or any square system. Its
solution set is LIN-002's canonical one, verified: `unique`, `family` (x with the free unknowns 0 plus the canonical
nullspace basis) or `none` (a certificate). With `inverse=True` a nonsingular model keeps A^-1 and det(A) from
fraction-free Gauss-Jordan elimination of [M | diag(L) | c] (M = diag(L) A the row-scaled integers), A A^-1 = I and
A x = b verified exactly. Regression models keep A0 (no ridge), y^T y, sum y and n, and LIN-002's statistics (rss =
||y - X x||^2, tss, r2, sigma2 without a ridge, y^T y, n, ||X x||^2; common-listwise only, pairwise: y^T y).

**Updates (exact Sherman-Morrison-Woodbury).** `update(U, V, db)`: A' = A + U V^T (U, V n x r, r <= 64), b' = b + db.
With the inverse, K = I_r + V^T A^-1 U is exact and its determinant `denominator` (rank one: 1 + v^T A^-1 u) decides:
det(A') = det(A) det(K), so det(K) = 0 exactly when A' is singular. Nonsingular: A'^-1 = A^-1 - (A^-1 U) K^-1
(V^T A^-1), x' = A'^-1 b' verified exactly (`method` woodbury). Singular, or no inverse kept: the declared contract
`fallback="rebuild"` solves A' x = b' exactly again (canonical family / none, `method` rebuilt, `fallback` True) and
`fallback="fail"` raises `SingularError` (`DomainError` for a nonsingular model built without its inverse).
`delete_rows` / `insert_rows`: complete rows in value units (dicts or (features in design order, y)), the update
U = -/+ Z^T, V = Z^T (one row: denominator 1 - h or 1 + h, h the leverage), y^T y, sum y and n updated exactly (a
deleted row that was never there is not detectable in normal equations; more rows than n is `ValueError`).
`ridge(lam)`: the rank-p update (lam - lam0) E E^T or a rebuild from A0 + lam D (equal; the same lambda returns a copy).
`subtract(st, g, sign)`: minus / plus a group's Gram matrix of a compatible state (same columns, target, policy,
scales; else `DomainError`), rebuilt exactly. Every result equals the model fitted again from the remaining rows.

**Leave-one-group-out.** The plan copies the group states, forms their total and the complement total - g of every
slot (counts and 192-bit sums subtract exactly), whose normal equations are those of the data without g. CPU:
LIN-002's exact Bareiss solve per group (`method` subtracted). CUDA (`tr_mgpu_linupd_logo`): one work item per
(group, prime in [2^61, 2^62)) forms the integer complement system modulo its prime directly from the copied states
(row u of the value system times den 10^(s_u + T), T the largest decimal scale of the design and target) and runs
LIN-002's modular Gauss-Jordan elimination; natively, `finish` keeps the agreeing primes (highest rank, smallest pivot
columns), Chinese-remainders the det-scaled reduced row echelon form within each system's Hadamard bound and verifies
the canonical solution exactly against the exact integer system (too few or unlucky primes: `NotConvergedError`
inside, rerun with the primes it names; `solve("cuda")` reruns up to 8 rounds). Both paths give equal models (`same`).

**Floating variant.** `float_solve(tol)`: x~ in binary64 (Gauss-Jordan with partial pivoting on the midpoint of A's
tight enclosure, one refinement step), certified by outward-rounded intervals: alpha >= ||I - R A||_inf; when
alpha < 1, ||x - x~||_inf <= bound = ||R (b - A x~)||_inf / (1 - alpha), with the condition estimate kappa >=
||A||_inf ||R||_inf. Refused (`NotConvergedError` with `.info`: alpha, bound, kappa, reason) when a pivot vanishes,
alpha >= 1 or bound > tol, unless `fallback="exact"` returns the exact solution (certified False). An exactly
singular model is `SingularError`.

**Budgets** (`AlgebraBudgetError` with `.resource/.need/.limit`, before allocation): fit / system: variables = n,
tensor = n (2n + 1), output = n (n + 1), plus LIN-002's normal-equation needs of the fit (tensor and output k (k + 1),
k augmented columns); update / rows / ridge: tensor = n r + r^2 + n^2, output = n (n + 1); LOGO plan: states = groups
x slots, tensor = systems x n (n + 1), output = systems, bytes = the copies; each LOGO system: as a fit; CUDA: tensor
= systems x primes x (4 + n (n + 1)) and bytes = the device copies and residues (`plan.device_bytes`); float: tensor
= n^2, bytes = 48 n^2. Refusals: `UnsupportedError` (n > 63 unknowns, r > 64, more than 1024 primes), `DomainError`,
`SingularError`, `NotConvergedError`, `ValueError`, `TypeError` (floats are not exact input).

**Limitations.** Updates are Woodbury formulas over canonical big rationals (O(n^2 r) rational operations, a gcd
each), which is slow next to fraction-free elimination: in the benchmark (case 3, 20,000 rows, a shared 2-core
machine) a one-row Woodbury deletion took 2.4 to 14 times as long as the exact rebuild of the updated normal
equations with its inverse (3 to 17 unknowns), and was faster than the complete reconstruction from the rows only for
small systems (5x at 3 unknowns, 2.8x at 5, within 10% at 9 and 17; the reconstruction grows with the rows, the update
does not); a ridge change touching every feature (case 2) was 3 to 5 times slower than its rebuild from A0. A
fraction-free (adjugate) representation of the inverse, which would avoid most gcds, is not implemented. LOGO
subtracts states and solves each system again (no low-rank downdate from Gram matrices); row deletions need the rows
(moment states do not keep them) and cannot detect rows that were never there; the floating variant is a verified
dense solve (O(n^3)), not a floating Woodbury; the CUDA kernel covers the modular LOGO eliminations only (the host
forms the exact integer systems for the Hadamard bounds and the verification, and runs the CRT).
