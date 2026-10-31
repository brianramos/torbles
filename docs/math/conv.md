# Convolution and bounded formal series (GEN-002)

`torbles/math_conv.h` (every contract; the 192-bit convolution coefficient, the batch layout and the bound checks are
inline and shared with CUDA), `math_conv.c` (built on GEN-001's generating polynomials, `docs/math/gen.md`),
`math_conv_gpu.cu`, `math_conv.py` (opt-in). Products of generating states with an explicit, natively checked
independence assumption; the conditional (per-key) form and the joint state as the exact alternatives; bounded
truncated series with declared coefficient bounds and path lengths.

```python
from torbles import math_bank as mbk, math_dist as md, math_conv as mc
scope = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["k"])
hx, hy, hxy = md.histogram(scope, ["x"]), md.histogram(scope, ["y"]), md.histogram(scope, ["x", "y"])
X, Y = mc.factor(hx), mc.factor(hy)          # key=None: the marginal over the keys; mc.factor(hx, 3): key 3
mc.product([X, Y])                           # DomainError(reason="needs_declaration", factors=(0, 1), keyed=True)
d = mc.independent(scope)                    # the declaration; d.covers(other_scope)
mc.product([X, Y], declaration=d)            # X + Y, independent draws = the CROSS JOIN's distribution
mc.product([X, Y], declaration=d, combine="outer")    # the joint of (X, Y) under independence (2 variables)
mc.product([X, Y], mult=[2, 1], order=16, declaration=d); mc.power(X, 8, declaration=d)
mc.need([X, Y]); mc.check([X, Y])            # what is checked before allocating; the independence check alone
mc.joint_sum(mc.factor(hxy)); mc.marginal(mc.factor(hxy), 0)     # per-row x + y; G(z, 1)
mc.conditional(mc.family(hx), mc.family(hy), combine="sum", backend="cuda")   # sum over keys of A_g B_g
mc.from_terms({0: 1, 1: 1}); mc.from_gen(poly)                   # factors without a source
mc.convolve([[1, 2], [3]], [[1, 1], [5]], order=None, backend="cuda")        # exact batched convolutions
s = mc.Series([0, 1, 1], order=16, bound=10**6); s * t; s.pow(4); s.walks(8); s.walks(None, backend="cuda")
mc.series(X, 16, bound=2**40); s.to_factor()                     # factor <-> bounded series (sources kept)
X.terms(); X.coefficient(v); X.total(); X.moment(2); X(Fraction(1, 2)); X.gen     # GEN-001's requests
```

## Sources and the independence check

- A **source** identifies rows: an engine table (its address and the wrapper's identity token: every scope over that
  table, whatever its columns, selection or grouping) or an array scope's row descriptor (selection and data). A
  factor of a histogram has its scope's source; `from_terms` / `from_gen` factors have none; a product has the union
  (`f.sources`, `f.info["sources"]`). A source is keyed when its scope has more than one group.
- A product (`product`, `power`, series products and walks) of copies sharing a source is computed only with a
  declaration covering every shared source; otherwise `DomainError` with `.reason` (`needs_declaration`,
  `not_covered`), `.factors` (the pair; `(i, i)` for two copies of one factor), `.keyed`, `.source`. The result says
  `info["declared"]`. Copies of different sources (or without one) multiply into their Cartesian product.
- Exact alternatives for linked inputs:

| form | value | SQL it equals |
|---|---|---|
| `product(..., declaration=d)` | the product of the marginals | `CROSS JOIN` |
| `conditional(fa, fb)` | sum over keys g of A_g B_g | `JOIN ... ON a.k = b.k` |
| `joint_sum(factor(hist_xy))` | G(z, z) of the joint state | per row `x + y` |

  Families of one source need the same grouping (`grouping_differs`), of different sources the same number of keys
  (`keys_differ`; key g of one is key g of the other). Values with a NULL are no term: `info["tuples"]` counts every
  combined tuple and `info["nulls"]` those with a NULL value (the SQL `x + y IS NULL` group).

## Exactness

- Products and powers are GEN-001's (exact big integers). `order` keeps the exponents e < order of P (G = z^shift P).
- Families hold int64 coefficients per key (counts; weight sums that fit, else `OverflowError`); every coefficient of
  a per-key product or a batch is an exact 192-bit sum (|sum| < 2^191 for any int64 input). Summed states (the
  conditional form, joint sums) are adopted by `tr_gen_from_terms`, whose merge is exact: each per-key coefficient
  must fit int128 (always true for counts), else `OverflowError`.
- Bounded series: int64 coefficients with a declared bound B (|c| <= B <= 2^63 - 1); each output coefficient is the
  exact 192-bit sum, then checked against the result's bound (`OverflowError` naming the first coefficient past it).
  `pow(k)` bounds every power a^j, j <= k. `walks(L)`: rows[l] = s^l mod z^N, total = sum over l <= L; `L=None` needs
  s(0) = 0 (then total = 1 / (1 - s) mod z^N), else `NoSolutionError` (the walk sums diverge). A step series needs
  shift 0; series of different orders: `CarrierError`.

## CUDA

`tr_mgpu_conv_batch`: one work item per output coefficient, its pair by binary search, the 192-bit sum written to its
own three words. `tr_mgpu_conv_conditional`: the same kernel over the keys of two families (per-key products in
parallel over keys); the host adopts the readback with `tr_conv_conditional_adopt`, exactly as the CPU path adopts
`tr_conv_batch_run`'s. `tr_mgpu_conv_walks`: a launch per length (one work item per coefficient, the bound checked,
the first violation by an atomic minimum), then the totals. Raw words, adopted states and walk tables equal the CPU
bit for bit; results say `cuda` or `cuda-emulated` (`f.backend`).

## Budgets and refusals

| request | checked before allocation |
|---|---|
| factor | the source copy (bytes), DIST-001's rollup for the marginal over keys, then GEN-001's state |
| product, power | variables; the chain's support growth (min(box, prod terms) of every partial product) and pairs (tensor) before the first product; a power or two factors: GEN-001's exact needs including bytes; every GEN-001 step checks its own bytes |
| family | variables 1, states = keys, support = the largest key span, bytes |
| conditional | states = keys, tensor = sum na nb, support = per-key outputs, bytes = scratch + terms + the adoption (`TR_CONV_GEN_TERMS_BYTES`); CUDA: bytes = the larger of that and the device copies |
| batch | output = output coefficients, tensor = sum na nb; CUDA: bytes = the device copies |
| series | degree = order - 1, support = order, tensor = order^2 (products), iterations = k / L, output = rows x order (walks) |

Refusals raise `AlgebraBudgetError` (`.resource`, `.need`, `.limit`); also `DomainError`, `OverflowError`,
`NoSolutionError`, `CarrierError`, `UnsupportedError` (more than 16 factors or sources, `outer` of other than two
one-variable factors, 3 coordinates, 2-D families, `order` with two variables), `ValueError`.

## For the next capabilities

GRAPH-002 (bounded path counts) builds on the bounded series (`tr_conv_series_*`, `tr_conv_walks_run`,
`tr_mgpu_conv_walks`: rows per length, totals, declared bounds, divergence refused) and the batch kernel
(`tr_conv_batch` / `tr_mgpu_conv_batch`: many truncated products at once); every product it forms of data-derived
series goes through the same independence check (`tr_conv_series_check`, `tr_conv_check_product`).

## Limitations

At most two variables (GEN-001's limit); OUTER results and states made by GEN-002 carry decimal scales in the factor
(`info["scales"]`), their GEN-001 copies scale 0. Families are dense per key (a wide key span is refused, not
sparsified). The conditional form has no truncation order; OUTER conditionals run on the CPU only. Summed states need
per-key coefficients below 2^127. A source is a table or an array scope: two separately built array scopes are
different sources even with equal data in other columns. The declaration is the caller's statement, never verified
against the data.
