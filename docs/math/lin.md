# Moment matrices, regression systems, solves and eigensystems (LIN-001, LIN-002)

`torbles/math_lin.h` (every contract; inline helpers shared with CUDA), `math_lin.c`, `math_lin_gpu.cu`,
`math_lin.py` (opt-in). LIN-001 builds grouped exact moment states from a FND-004 scope (or bank); LIN-002 finalizes
them, and small rational systems, exactly where it can and with a declared numerical contract where it cannot.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_lin as lin
scope = mbk.Scope.from_table(db["f"], ["x1", "x2", "y"], group_by=["g"], rows=mask)
st = lin.moments(scope, ["x1", "x2"], "y", policy="listwise", intercept=True)   # backend="cuda" too
st.matrix(g), st.counts(g), st.state(g)        # Gram of [1, x1, x2, y] (logical ints), per-entry rows, raw slots
st.system(g, ridge=Fraction(1, 2), features=["x1"])     # normal equations (value units): A, b, design, yty, n
f = st.fit(g)                                  # Fit: kind, x / coefficients, nullspace, rss, tss, r2, sigma2
st.fits([dict(group=g, ridge=r) for g, r in ...], backend="cuda")   # batched modular solves on the device
st.covariance(g, ddof=1)                       # Covariance: cov (value units), r2, sign, count
st.pca(g, "covariance" | "correlation")        # Eigen (APPROXIMATE, certified bounds)
st + other, st - deleted                       # merges of disjoint selections, deletions of rows
lin.moments_from_bank(mbk.build(scope, mbk.monomials(["x1", "x2", "y"], 2), "joint"), ["x1", "x2"], "y")
lin.solve(A, b), lin.solve_batch(systems, backend="cuda"), lin.inertia(A), lin.eigen(A) / lin.eigen(lo=, hi=)
```

## LIN-001: moment states

- Columns: d <= 16 features and an optional target, scope columns; the augmented columns are z0 = 1, the features,
  the target (k = d + 1 + target). Per group: `rows` (bag multiplicity: selected rows; a group is present iff
  rows > 0, an all-NULL group is present with zero counts, an absent group has rows 0) and k(k+1)/2 Gram slots
  (count, exact sum of z_a z_b): count n, sums, X^T X, X^T y, y^T y. `intercept` declares the regression design.
- Policies (declared, never mixed): `listwise` (common-listwise): a row counts only when every feature and the
  target are present; all counts equal; the Gram matrix of the complete rows. `pairwise`: each entry over the rows
  where its two columns are present, with per-entry counts; the raw matrix is the Gram matrix of the data with NULL
  read as 0 (positive semidefinite) but of no set of complete rows, and the covariance / correlation it defines
  (pair-complete means) may be indefinite. A fused pairwise state also keeps, per pair, the pair-complete marginals
  (sum a, sum b, sum a^2, sum b^2 over rows with both present: SQL COVAR / CORR / REGR_* rows); a bank-built one has
  none (covariance then refused: `DomainError`).
- Exactness: logical integers (decimal value = logical / 10^scale; `st.scales`). Each slot is proven to fit 128 bits
  (`N * M_a * M_b <= 2^127 - 1` from the scope's extremes); otherwise 192-bit words are used (they hold any sum of
  int64 products over < 2^64 rows). `width=128` refuses an unproven request (`OverflowError`), `192` forces wide
  words; stored sums are 192-bit two's complement, returned as Python ints (past HUGEINT when needed).
- Paths giving the same states bit for bit: the fused CPU pass, the CUDA kernels (`backend="cuda"`: row masks,
  bag multiplicity, then one work item per slot and row slice, keeping every group locally up to 16 groups, else
  flushing on group change, with multiword integer atomics), a FND-004 bank (`moments_from_bank`: `joint` mode over
  exactly the spec's columns is `listwise`, `sql` is `pairwise`; the basis must hold every Gram monomial), and merges
  (`a + b` of disjoint selections; `a - b` deletes rows b summarizes, refused when counts would go negative).
- A table scope must be fresh (`StaleStateError` after `t.flip`, a new revision or another selection).

## LIN-002: exact operators

- `solve(A, b)`: m <= 64 equations, n <= 63 unknowns, int / Fraction entries. Rows are scaled to integers; fraction-free
  Gauss-Jordan (Bareiss) over big integers (FND-009) on the CPU; `backend="cuda"`: Gauss-Jordan modulo primes in
  [2^61, 2^62) on the device (one work item per system and prime), then natively the choice of the primes that agree
  (highest rank, smallest pivot columns), Chinese remaindering of the det-scaled reduced row echelon form within a
  Hadamard bound, and exact verification (a disagreeing or too small set of primes reruns with more). The result
  (`Solution`) is canonical and verified: `unique` (x), `family` (x0 with free unknowns 0, plus the canonical basis
  `nullspace`), `none` (`certificate` y: y^T A = 0, y^T b = 1); `rank`, `pivots` exact; never a division by zero.
- `inertia(A)` (symmetric): exact (n+, n-, n0) by symmetric fraction-free elimination with diagonal pivots (a zero
  diagonal with a nonzero off-diagonal entry is resolved by an integer congruence), the class (`positive definite`,
  `positive semidefinite`, `negative definite`, `negative semidefinite`, `indefinite`, `zero`), and verified
  certificates: `negative_direction` (v^T A v < 0), `positive_direction`, `kernel` (an exact basis, n0 vectors).
- `fit(g, intercept=, ridge=, features=)`: the normal equations in value units (ridge: an exact penalty >= 0 on the
  features, not the intercept; features: the subset kept), solved as above; statistics (common-listwise only, else
  None): rss = ||y - X b||^2 (unique for every solution of a rank-deficient system), tss (centred with an intercept),
  r2 = 1 - rss / tss, sigma2 = rss / (n - rank) (no ridge), yty, n, and `projection` = ||X b||^2 (the squared norm
  of the projection of y onto the design's column space when unpenalized). `fits(..., backend="cuda")` batches them.
- `covariance(g, ddof)`: cov (value units), r2 = cov^2 / (var var), sign, count, over [features..., target]; listwise
  over the complete rows, pairwise over pair-complete rows (variances over each column's rows); `None` where
  undefined (n <= ddof, a zero variance).
- `eigen` / `pca` (APPROXIMATE): cyclic Jacobi in binary64 on the midpoint of an enclosure of the matrix (exact
  entries enclosed tightly; correlations enclosed with outward rounding). Returned with rigorous bounds valid for every
  matrix in the enclosure: `residual[i] >= ||A v_i - lambda_i v_i||_2`, `bound[i]` (an eigenvalue of A lies within
  it of `values[i]`), `orthogonality >= max |V^T V - I|`, the input `width`. Declared tolerances (tested): for a
  converged tight enclosure, residual and bound <= 64 n 2^-52 ||A||_F and orthogonality <= 64 n 2^-52. No other claim:
  exact input moments do not make eigenvectors stable (close eigenvalues make them ill-conditioned).

## Budgets and refusals

| Request | variables | tensor | other |
|---|---|---|---|
| moments | scope columns used | groups x Gram slots | degree 2, support = slots, output = groups, states = groups x slots, bytes (CPU: state + accumulators; CUDA: device copies) |
| solve / inertia | n | m (n + 1) | output n (n + 1) |
| fit / system / covariance | - | k (k + 1) | as solve on the k x k normal system |
| eigen / pca | - | n^2 | iterations = sweep cap (default 64), output, bytes |
| batch (CUDA) | - | systems' entries | output = systems, bytes = device copies (`tr_lin_batch_device_bytes`) |

Refusals before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`); `UnsupportedError` (d > 16,
m > 64, n > 63); `OverflowError` (width 128 unproven); `DomainError` (bank of another mode / joint set, covariance of
a bank-built pairwise state); `ValueError` (repeated columns, a fit without a target, a nonsymmetric inertia input,
an invalid deletion); `TypeError` (float input); `NotConvergedError` (Jacobi sweep cap; modular rounds exhausted);
`StaleStateError`.

## Limitations

The CPU accumulation and the exact elimination are single-threaded; Bareiss allocates a big integer per operation
(about 13 ms for a 16-feature fit); the modular path's host finish (CRT, gcd, verification) is CPU work comparable to
Bareiss, so the device only pays with many systems; the eigensolver runs on the CPU only (no CUDA kernel: Jacobi's
binary64 rotations are not bit-reproducible across compilers); pairwise states from a bank lack the pair-complete
marginals; groups are dense (absent groups take state memory).
