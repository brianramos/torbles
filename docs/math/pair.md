# Sorted-distribution pairwise summaries (PAIR-001)

`torbles/math_pair.h` (every contract; the evaluator is inline and shared with CUDA), `math_pair.c`,
`math_pair_gpu.cu`, `math_pair.py` (opt-in). Exact pairwise summaries computed from the sorted support of DIST-002
prefix states (`docs/math/dist.md`), without enumerating pairs: one pass over a class's support, or one merge of two.

```python
from fractions import Fraction
from torbles import math_dist as md, math_pair as mp
p = md.histogram(scope, ["x"], weight="w").prefix("weight", degree=0)     # classes = groups; or mass "count"
mp.abs_within(p, group=0, ordered=False, self_pairs=False, mean=True)    # the Gini mean difference
mp.abs_cross(p, 0, 1, mean=True); mp.mann_whitney(p, 0, 1, auc=True)      # ties count half
mp.ks(p, 0, 1)["at"]; mp.wasserstein(p, 0, 1); mp.cramer_von_mises(p, 0, 1, other=q)   # class 1 of state q
mp.query(p, [dict(op="ks", a=0, b=1), ...], other=None, backend="cuda")   # {"value", "empty", "pairs", ...}
```

## Classes and conventions

- A class is one group of a prefix state: the expanded multiset of its present values, each repeated by its mass (rows,
  or units of weight 10^-wscale as in DIST-002). NULL values form no pair. Two classes of two states must have the same
  value scale (`DomainError`); masses may be counts or weights. A class mass must be below 2^64 (counts always are;
  a weight class at or past 2^64 is `OverflowError` for that request).
- Self-pairs: an element paired with itself. They exist only within one class (`abs_within`, or a two-class summary
  given the same class twice: the same state and group); excluded unless `self_pairs=True` (with weights a self-pair is
  a unit paired with itself: M of them).
- Ordering: `abs_within` counts unordered pairs {i, j} unless `ordered=True` (each unordered pair twice); two-class
  summaries pair (i in A, j in B), ordered by class.
- Normalization: `mean=True` divides by the number of pairs P (`auc=True` for Mann-Whitney). P within a class of mass
  M: unordered M (M - 1) / 2 (with self-pairs M (M + 1) / 2), ordered M (M - 1) (with self-pairs M^2); two classes
  M_A M_B (minus M without self-pairs when they coincide).
- Empty classes (absent group, only NULL values, only zero weights): sums over no pair are 0 (ABS, U); every ratio is
  empty (`value` None, SQL NULL): the means, the AUC, KS, W1, CvM; P = 0 also empties the means.

## Summaries (exact rationals)

| op | value |
|---|---|
| `abs_within` | sum over the pairs of y_i - y_j in absolute value = sum_k (x_(k+1) - x_k) c_k (M - c_k), twice when ordered |
| `abs_cross` | sum over i in A, j in B of the absolute difference = sum over pooled gaps (t_(k+1) - t_k)(c_A (M_B - c_B) + c_B (M_A - c_A)) |
| `mann_whitney` | U = #{x_i > y_j} + 1/2 #{x_i = y_j} (a self-pair is a tie); `auc`: U / P |
| `ks` | sup over t of F_A(t) - F_B(t) in absolute value (right-continuous empirical CDFs; `at`: the smallest pooled value attaining it) |
| `wasserstein` | the 1-Wasserstein distance of the normalized distributions, the integral of F_A - F_B in absolute value (logical units: times 10^-scale) |
| `cramer_von_mises` | T = M_A M_B / (M_A + M_B)^2 sum over the pooled elements u of (F_A(u) - F_B(u))^2 (Anderson 1962; ties at the pooled values) |

c, c_A, c_B are the masses at or below a support value. Results: `value` (int or Fraction), `pairs` (P), `mass_a`,
`mass_b`. Every sum is exact in 256 bits (ABS and W1 below M_A M_B (x_max - x_min) < 2^192); Cramer-von Mises needs
(M_A M_B)^2 (M_A + M_B) < 2^255 and its denominator M_A M_B (M_A + M_B)^2 below 2^255 (always true for counts; else
`OverflowError`). The tests compare every summary, flag and empty-class case with explicit enumeration of the pairs of
tiny tied and weighted inputs (Fractions), KS from the CDFs at every pooled point, W1 from the quantile functions over
[0, 1], CvM from the pooled elements, and DuckDB self-joins.

## CUDA, budgets and refusals

`backend="cuda"`: the prefix states are uploaded once (`Prefix.device()`, `tr_mgpu_dist_prefix_upload`) and
`tr_mgpu_pair_query` runs one work item per request with the shared evaluator, one output slot each: the answers equal
the CPU's bit for bit. A batch's budget: output = requests (CPU and CUDA); CUDA bytes = the device copies of the
requests and answers (184 bytes per request). Each request has its own status (`query_arrays`: 0, -1 invalid op /
flags / group, -3 overflow, -6 scales differ); the dict API raises `ValueError`, `OverflowError`, `DomainError`.
