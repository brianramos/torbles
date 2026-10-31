# Walsh spectra and parity correlations (SPEC-002)

`torbles/math_spec.h/.c` (native CPU), `math_spec_gpu.cu` (CUDA kernel: the butterfly passes), `math_spec.py` (opt-in
Python API). Exact Walsh-Hadamard transforms of 2^k vectors (a LATTICE-001 histogram's projection, or any exact
integers) under a declared normalization, Parseval identities, parity correlations of condition subsets, truncation
as an explicit approximation with its exact error, and constant parities cross-checked with FIELD-001. The header
comment of `math_spec.h` is the normative contract; this page summarizes it.

```python
from torbles import math_lattice as ml, math_spec as ms
h = ml.Histogram.build(masks)                  # k conditions over the rows of a scope (math_lattice.md)
f = ms.walsh(h, backend="cpu")                 # or "cuda": {"spectrum", "k", "normalization", "cost", ...}
ms.inverse(f["spectrum"])["values"]            # exact Fractions (arrays=True: numerator words and den_log2)
ms.parseval(h, f["spectrum"])                  # {"cells", "spectrum", "holds", "identity"}
ms.correlations(f["spectrum"], [[0], [0, 2]])  # W(S), even, odd, correlation W(S) / N (Fraction)
ms.coefficients(h, [[0, 2]])                   # the same by direct sums (new frequencies, no full transform)
t = ms.truncate(f["spectrum"], m=8)            # the 8 largest |W(S)|: an explicit approximation with its error
ms.evaluate(t, [0, 5, 7])                      # the kept expansion at points (exact Fractions)
ms.constant_parities(h)                        # {"relations", "basis", "rank", "agree", "total", "support"}
```

## Normalization (declared)

With chi_S(x) = (-1)^abs(S and x) over the 2^k subsets x of the conditions (bit i = condition i):

- **forward**: W(S) = sum over x of v(x) chi_S(x), exact integers (unnormalized);
- **inverse**: v(x) = 2^-k sum over S of W(S) chi_S(x), exact dyadic rationals (canonical num / 2^d; integers when W
  is the forward transform of integers);
- **Parseval**: sum of W(S)^2 = 2^k sum of v(x)^2, both sides exact (256-bit); with the probability normalization
  p = v / N (N = W(empty)) it reads sum of (W(S) / N)^2 = 2^k sum of p(x)^2.

Every transform reports its cost: the full domain of 2^k states, k passes, k 2^(k-1) butterflies and the bytes,
whatever the occupancy or the number of rows behind the histogram.

## Correlations, truncation, constant parities

- **Parity correlation** of a set S of conditions: W(S), `even` = (N + W(S)) / 2 and `odd` = (N - W(S)) / 2 (the
  weight of the points with an even / odd number of TRUE conditions in S), and `correlation` = W(S) / N = P(even) -
  P(odd) as an exact Fraction (0 when N = 0). `correlations` reads a full spectrum; `coefficients` sums W(S) and N
  directly from the values (2^k terms per S).
- **Truncation** keeps the m coefficients of largest abs(W(S)) (ties: the smaller S) and is reported as an
  approximation (`approximation` True, `exact` False) unless nothing nonzero was dropped, with `states` (2^k), the kept
  coefficients, `residual_energy` (the dropped sum of W^2), `l2_error_squared` = residual / 2^k (the exact squared
  L2 error of the reconstruction, by Parseval) and `max_error_bound` (the dropped sum of abs(W) / 2^k, a bound on the
  pointwise error). `evaluate` computes the kept expansion exactly at given points (m terms per point).
- **Constant parities**: on non-negative weights with N > 0, the parity of S is constant over the occupied cells
  exactly when abs(W(S)) = N (XOR over S = 0 when W(S) = N, 1 when W(S) = -N). These are the affine GF(2) relations
  of the occupied cells: the native code also inserts every occupied cell as [1, x] into FIELD-001's echelon basis,
  takes its canonical relation basis (`basis`, `rank`) and reports `agree` when the span of that basis is exactly the
  set found from the spectrum. A vector with a negative entry or without weight is `DomainError`.

## Exactness

Values are 128-bit integers and every butterfly is checked (`OverflowError`, never wrapped); for count histograms
every intermediate is bounded by N, for any int64 input by 2^(k+63). Energies are exact 256-bit integers. CPU,
emulated CUDA and the device agree bit for bit.

## CUDA

`tr_mgpu_spec_walsh` / `tr_mgpu_spec_walsh_inverse`: the values copied once, one launch per pass, one work item per
pair (the pairs of a pass are disjoint), an overflow flag raised by any butterfly that would wrap; the inverse's
canonical dyadic values are formed on the host from the device's integers. Correlations, coefficients, truncation,
evaluation and constant parities run on the CPU.

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| walsh / inverse | variables = k; tensor = 2^k; iterations = k; bytes = 16 x 2^k (the scratch or the device copy) |
| coefficients | variables = k; tensor = requests x 2^k; output = requests; bytes = 96 per request |
| truncate | variables = k; tensor = 2^k; output = m; bytes = 24 x 2^k |
| evaluate | variables = k; tensor = m x points; output = points; bytes = 24 per point |
| constant parities | variables = k; tensor = 2^k; iterations = k; output = 2^k; bytes = 17 x 2^k |

Representation limit (`UnsupportedError`): k > 30. Malformed input (`ValueError`): not 2^k values, sets or points out
of range, m > 2^k, a spectrum whose W(S) and N differ in parity (not the spectrum of integer weights).

## Limitations

Boolean (Z_2^k) spectra only, dense over the full 2^k domain (no sparse or partial transforms); the CPU paths are
single-threaded; only the butterfly passes run on CUDA; Parseval and truncation errors need the energies to fit 256
bits (always true for spectra of int64 vectors with k <= 30).
