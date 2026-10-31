# Budgets (FND-010)

`torbles/math_budgets.h` (inline, shared by the engine library and the CUDA library), `math_budgets.c`,
`math_budgets_gpu.cu`, `math_budgets.py`. Every mathematical capability states what a request needs before it
allocates anything, and compares it with its budget.

Resources, in the fixed order of every check: `variables`, `degree`, `support` (stored terms or support points),
`tensor` (dense elements), `circuit` (DAG nodes), `iterations`, `output` (rows or answers), `states` (accumulator cells
held during a scan), `bytes`. Defaults: 64, 64, 2^20, 2^26, 2^20, 2^20, 2^24, 2^24, 2^30.

```python
from torbles import math_budgets as mb
b = mb.Budget(output=10_000)                       # the defaults with one limit changed
mb.check(b, variables=6, degree=4, support=mb.monomials(6, 4))
mb.choose(b, dict(support=5000), 1.0, dict(support=40), 20.0)   # {"choice": "factored", "reason": "only fit", ...}
mb.count_cells(cells, ncells, rows=mask, budget=b, backend="cpu")  # or backend="cuda"
```

- Sizes are exact: `monomials(n, d)` = C(n+d, d) (`homogeneous=True`: C(n+d-1, d)), `product(dims)`, `power2(k)`,
  `binomial(n, k)`. A size past 2^64-1 is `UNKNOWN` (2^64-1), above every limit; nothing wraps.
- `check` raises `AlgebraBudgetError` at the first resource over its limit, with `.resource`, `.need` and `.limit`; the
  same request is refused the same way every time, before any partial result exists.
- `choose` compares an expanded and a factored representation of the same result (needs and relative costs): both fit,
  the cheaper runs (expanded on a tie); one fits, it runs (`"only fit"`); none fits, refused. `allow_factored=False`
  forbids the fallback, so an expanded representation over budget is refused, never replaced.
- `count_cells` counts the distinct cell ids of the selected rows exactly (a bitmap of `ncells` bits), budgeting the
  bitmap first (`states` = ncells, `bytes`), then refuses when the count exceeds `output`; the refusal reports the exact
  count as its need. A cell id >= ncells is invalid input (`ValueError`). The CUDA path also budgets the device copies
  of the cell ids and the row bitmap, so the same request needs more bytes there.
- `Budget.polynomial_kernel()` states the fixed limits of the first-slice polynomial kernel (8 variables, degree 8,
  128 terms); those modules keep their own checks, which refuse at the same boundaries.
