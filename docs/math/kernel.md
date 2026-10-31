# Exact finite-feature kernel embeddings (KERNEL-001)

`torbles/math_kernel.h` (every contract; the 1024-bit arithmetic, the request evaluator and the self-pair term are
inline and shared with CUDA), `math_kernel.c`, `math_kernel_gpu.cu`, `math_kernel.py` (opt-in). Kernels whose feature
map is explicitly finite reduce kernel sums over selected pairs, kernel mean embeddings and maximum mean discrepancies
between groups to retained feature states, the moment and count tensors of TENSOR-001 (`docs/math/tensor.md`),
exactly. A kernel without an exact finite feature map, or one the state does not retain, is refused, never
approximated.

```python
from torbles import math_bank as mbk, math_kernel as mk
scope = mbk.Scope.from_table(db["f"], ["x", "y", "w", "c"], group_by=["k"], rows=mask)
st = mk.features(scope, ["x", "y"], mk.polynomial(2, c=1), weight="w")   # repr="sparse", degree=D, backend="cuda"
st.sum(0, 1); st.sum(0); st.sum(0, self_pairs=True, ordered=False)       # {"value", "pairs"}: exact ints
st.mean(0, 1); st.mmd2(0, 1, unbiased=True); st.witness(0, [(1, 2), (0, -3)])   # Fractions (None: empty)
st.query([dict(op="mmd2", a=0, b=1), dict(op="witness", a=1, point=0)], points=[(1, 2)], backend="cuda")
st.sum(0, 1, kernel=mk.linear(["x"]))     # another kernel the state retains (family, degree, columns)
st.info; st.group(g); st.tensor           # bytes, cells, nnz ...; W, Q, Dg of a group; the TENSOR-001 tensor (a copy)
mk.polynomial(d, c); mk.linear(); mk.homogeneous(d); mk.monomial([({"x": 2, "y": 1}, 3)]); mk.delta(["c"])
mk.kernel("rbf", gamma=1)                 # no exact finite feature map: refused natively (UnsupportedError)
mk.need(scope, ["x", "y"], mk.polynomial(2), backend="cuda")
```

## Kernels

- `polynomial(d, c)`: k(x, y) = (x.y + c)^d over n = 1..32 columns, d = 0..8, |c|^d <= 2^127 - 1 (`linear`: d = 1,
  c = 0; `homogeneous`: c = 0). With x~ = (1, x): k(x, y) = sum_J c^{z(J)} prod_k x~_{j_k} y~_{j_k} over the
  (n + 1)^d multi-indices J, z(J) the entries of J that are the constant.
- `monomial(terms)`: k(x, y) = sum_t w_t x^{a_t} y^{a_t}, 1..64 monomials of total degree <= 8 with int64 weights
  (equal monomials merged, zero weights dropped; nothing left is `ValueError`).
- `delta(columns)`: k(x, y) = prod_k [x_k = y_k] over 1..8 categorical columns, NULL a category of its own (NULL
  matches NULL): the one-hot feature map of the joint category.
- Any other name (`rbf` / `gaussian`, `laplace` / `laplacian`, anything else) is refused (`UnsupportedError`):
  approximations are KERNEL-002's. Values are the scope's exact logical integers; the feature columns of a polynomial
  or monomial kernel must share one decimal scale (`DomainError` otherwise).

## The feature state

- Polynomial and monomial kernels: the TENSOR-001 moment tensor of order D (the kernel's degree, or a larger
  `degree=`, at least 1) whose every axis lists (1, x_1, ..., x_n), listwise: T_g[J] = the sum over the complete rows
  of group g (every kernel column and the weight present) of w prod_k x~_{j_k}. Every lower order is its slice with
  leading constant indices, so sum_{i in a, l in b} w_i v_l k(x_i, y_l) = sum_J c^{z(J)} T_a[0^{D-d}, J]
  T_b[0^{D-d}, J] = <Phi(X_a), Phi(X_b)> exactly.
- Delta kernels: the TENSOR-001 count tensor of the columns (weighted: the sums of present weights); <C_a, C_b>.
- Per group, besides the tensor's rows and used rows: W = sum w (128 bits), Q = sum w^2 and Dg = sum w^2 k(x_i, x_i)
  of the declared kernel (1024 bits), over the contributing rows: the self-pair (diagonal) terms. `info`: kind,
  degree, order, repr, groups, rows, used, cells (per group), nnz (stored entries, all groups), bytes, weighted.
- Dense or sparse (`repr`), built on the CPU or with CUDA (`backend`): the same state bit for bit.
- Retained kernels: an evaluation (`kernel=`) of the same family (moments or counts), degree <= D and columns a subset
  of the state's (delta: exactly its columns). The self-pair term belongs to the declared kernel only: a request that
  needs it (within one group: a sum or mean excluding self-pairs, or over unordered pairs; an unbiased MMD) with
  another kernel is refused (`UnsupportedError`), as is any kernel the state does not retain.

## Requests

- `sum(a, b)`: the sum over the selected pairs of w_i v_l k(x_i, y_l) and the weighted pair count. a != b: every pair
  (pairs W_a W_b); a == b: the ordered pairs i != l (K - Dg, pairs W^2 - Q), with `self_pairs=True` all of them (K,
  W^2), with `ordered=False` each unordered pair once ((K - Dg) / 2, or (K + Dg) / 2 with self-pairs).
- `mean`: the sum over the pair count (None when it is 0): the kernel mean embedding inner product <mu_a, mu_b>.
- `mmd2(a, b)`: biased K_aa / W_a^2 + K_bb / W_b^2 - 2 K_ab / (W_a W_b); unbiased (`mmd2`'s default, a request
  dict's `unbiased=True`): (K_aa - D_a) / (W_a^2 - Q_a) + (K_bb - D_b) / (W_b^2 - Q_b) - 2 K_ab / (W_a W_b) (None
  when a denominator is 0).
- `witness(a, points)`: mu_a(z) = sum_i w_i k(x_i, z) / W_a at each point (the kernel columns' logical values; delta:
  a category or None per column, a value outside the domain matching no row).
- `query(requests, points=)`: a batch; every request has its own status natively (`raw=True` returns them, otherwise
  the first failing request raises). Answers are exact rationals num / den in 1024-bit two's complement (den > 0, not
  reduced); Python returns `int` and `Fraction`.

## Exactness

- The tensor's entries are proven by TENSOR-001 before any row is read (`OverflowError`); the build also proves that
  the self-pair sums fit (bit lengths: rows + 2 x weight + the kernel's value at the scope's extremes <= 1000).
- Inner products: per coefficient class (polynomial: z = 0..d; monomial: one per term) the sum over the items I of
  mult_I T_a[I] T_b[I] in 384-bit accumulators (at most 2^64 additions below 2^286: they never overflow), combined
  with the class coefficients in 1024 bits. Polynomial items are the C(n + d, d) sorted multi-indices with their
  multiplicities d! / prod r! (the tensor is symmetric), monomial items one per term, delta every cell.
- Means, MMDs and witnesses use overflow-checked 1024-bit products: an answer outside the carrier is refused in its
  status (`OverflowError`), never wrapped.

## CUDA

- `features(..., backend="cuda")` (`tr_mgpu_kernel_build`): the TENSOR-001 device build of the feature tensor
  (`tr_mgpu_tensor_build`), then one work item per row slice for W, Q and Dg with exact atomics (W: the two-word
  atomic; Q, Dg: a 16-word atomic carry chain, a local equivalent since `math_cuda.h` has none that wide); the
  readback is adopted by `tr_kernel_from_states` after consistency checks (W against the tensor, Q >= 0, delta: Dg =
  Q).
- `query(..., backend="cuda")`: the evaluation view (the state and the retained kernel's items) is uploaded once per
  state and kernel (`tr_mgpu_kernel_upload`), then one work item per request runs the shared evaluator
  `tr_kernel_eval_request` (`tr_mgpu_kernel_query`), writing exactly `tr_kernel_query`'s results.

## Budgets and refusals

| request | needs |
|---|---|
| build (CPU) | the feature tensor's TENSOR-001 needs; variables (kernel columns and the weight); bytes plus 288 a group |
| build (CUDA) | the device copies (the tensor's device needs, 288 bytes a group, the kernel) and the adoption, both checked first |
| evaluation (a retained kernel) | degree; support = its items (polynomial C(n + d, d), monomial its terms); bytes = 64 an item |
| query (CPU) | output = the requests |
| upload, query (CUDA) | bytes = the view's device copies; output = the requests, bytes = requests, results and points |

Refusals raise before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError` (|c|^d, the
tensor's bound, the self-pair proof; an answer past 1024 bits in its status), `UnsupportedError` (a kernel without an
exact finite feature map, degree or order > 8, > 32 columns, > 8 delta columns, > 64 monomials, a kernel the state
does not retain, a self-pair exclusion with another kernel), `DomainError` (feature columns of different decimal
scales), `ValueError` (malformed kernels, degrees below the kernel's, requests, points, groups out of range),
`StaleStateError` (the scope's table changed).

## For the next capabilities

KERNEL-002 (declared approximations: random features of RBF and other kernels, with error contracts) can measure its
error against these exact states on polynomial kernels; `bench/benchmark_math_kernel.py` records the exact baseline at
16 / 64 / 256 features. PAIR-002 reads the same TENSOR-001 moment tensors (`st.tensor`).

## Limitations

Exact finite features only (polynomial, monomial, delta; RBF, Laplace and other kernels are refused: KERNEL-002); the
feature tensor stores its (n + 1)^D cells per group (dense, or sparse at the data's support), so the tensor budget
bounds n and D; listwise only (a row with a NULL feature column does not contribute; delta kernels count NULL as a
category); answers are limited to 1024 bits; the CPU loops are single-threaded; the CUDA query kernel uses a 1984-byte
stack frame with register spills (255 registers) and runs one work item per request (no parallelism within one
request); the benchmark times the CPU and NumPy here (CUDA only on a device, with `--device`).
