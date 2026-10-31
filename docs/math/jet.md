# Jets of polynomial circuits (JET-001, JET-002)

`torbles/math_jet.h` (every contract; the evaluation code shared with CUDA), `math_jet.c`, `math_jet_gpu.cu`,
`math_jet.py`. Dual numbers (a value and k first directional derivatives, JET-001) and truncated multivariate Taylor
jets (gradients, Hessians, Hessian-vector products, mixed and third partials, JET-002) of bounded polynomial
circuits, at one point or summed over the selected rows of a data scope (FND-004), exact on the CPU and on a CUDA
device with the same states bit for bit.

```python
from fractions import Fraction
from torbles import math_jet as mj
b = mj.Builder(params=["a", "c"], data=["x", "d"])
a, c = b.params; x, d = b.data
circuit = b.build((a * x + c * d) ** 3 - Fraction(1, 2) * a)          # immutable DAG, budgeted
plan = circuit.plan(order=2, center={"a": 1, "c": Fraction(1, 3)}, scales=[0, 2])   # directions = the parameters
jet = plan.at({"x": 3, "d": 150})              # d = 1.50: data are logical integers / 10^scale
jet.value, jet.gradient(), jet.hessian(), jet.coefficient((1, 1)), jet.derivative((2, 0)), jet.third(0, 0, 1)
circuit.plan(1, center, directions=[{"a": 1, "c": 2}, [0, 1]])          # JET-001: value + 2 directional derivatives
circuit.hvp(v, data, center)                    # (H v)_i: mixed coefficients of the directions [I; v]
plan.evaluate({"x": xs, "d": ds}, backend="cuda")                       # one jet per point (JetSet)
agg = circuit.aggregate(db["t"], center, order=1, group_by=["k"], rows=mask)   # sum_r f(theta, x_r) per group
b1 = plan.builder(scope); b1.accumulate(0, n // 2); b1.merge(other); b1.finish()    # chunked, merged
agg + other_agg                                 # the sum law (bag union of the rows)
agg.values("q64" | "bigq" | "gfp" | "interval", convention="taylor" | "derivative", p=2**61 - 1).convert(...)
plan.info                                       # support, pairs, steps, slots, bytes, multiplications, additions, ...
```

## Circuits and jets

- Nodes: `const` (int64 rational), `param p`, `data c` (a fixed observation: never differentiated), `add`, `sub`,
  `mul`, `pow n` (0..255; a^0 = 1, also for 0), `scale c` (rational), `neg`; each refers to earlier nodes; the output
  is the last. Python records the node list (`Builder`, or `Circuit.from_nodes`); the native constructor validates and
  freezes it. Unreachable nodes and the operand of `pow 0` are not evaluated.
- A plan fixes the order d, m seed directions (rows of U, rationals over the parameters; default U = I), the center c
  and the data columns' decimal scales. The jet at a point x is the truncated Taylor polynomial of
  g(t) = f(c + U^T t, x). **Convention:** stored coefficients are Taylor (Hasse) coefficients
  J[alpha] = d^alpha g(0) / alpha! (`"taylor"`); `"derivative"` readouts are alpha! J[alpha]. So J[0] = f(c, x),
  J[e_j] = D_{u_j} f, mixed J[e_i + e_j] = D_{u_i} D_{u_j} f, J[2 e_i] = D_{u_i}^2 f / 2. At c = 0 with U = I the Taylor
  coefficients are f's coefficients in theta. With directions [I; v], J[e_i + e_v] = (H v)_i.
- Laws (per node): sums add; products are truncated convolutions (order 1: Leibniz); powers use the binomial law
  (a0 + h)^n = sum_{k <= min(n, d)} C(n, k) a0^(n-k) h^k (order 1: the chain rule); parameters seed J[e_j] = U[j][p].
- Monomial order: graded; within a degree the sorted index tuples lexicographically: [1], [t_0..t_{m-1}],
  [t_0^2, t_0 t_1, ..]: index 1 + j is t_j; S = C(m + d, d) coefficients.

## Exactness

The carrier is **fixed-denominator int64 rationals**: every node has a positive int64 denominator fixed by the plan
(constants, center, seeds, 10^scale; lcm for sums, products for products, D^n for powers), and each coefficient is an
int64 numerator over it, computed with 128-bit intermediates and checked. A denominator outside int64 refuses the plan;
a numerator, product, binomial coefficient, power or running sum outside int64 refuses the evaluation (`OverflowError`,
the same refusal on the CPU and CUDA, any chunking). Nothing wraps or rounds. Static supports (the directions a node's
parameters are seeded along, degrees up to its parameter degree) skip coefficients that are provably zero; they change
neither results nor refusals. Aggregates sum the rows' int64 numerators exactly in 128 bits (fewer than 2^63 rows), so
accumulation and merges never overflow and keep every component. Readouts are canonical (`Fraction`).

## Aggregates

Over a `math_bank.Scope` whose columns are the circuit's data columns (same count and scales, else `DomainError`): per
group, rows (bag multiplicity), count (rows where every column the circuit reads is present: SQL `SUM(f)` skips NULL)
and the S sums. A group is present iff rows > 0; a present group without valid rows has count 0 and the zero jet (not
SQL NULL); absent groups have rows 0. Chunked accumulation (disjoint ranges, each row once), merged builders,
`agg + other` (same plan and group count; unscoped), and the CUDA path all give the same states. Per-point jets are
the aggregate of a scope with one group per point. A table scope that went stale raises `StaleStateError`.

## Carriers

`values(...)` converts every component of every jet, all or nothing: `q64` (narrowing: `OverflowError` when a
canonical component leaves int64), `bigq` (lossless; math_carriers big rationals), `gfp` (reduction mod a prime
p < 2^62: `NoSolutionError` when p divides the denominator, `ValueError` for a composite, `UnsupportedError` for
p >= 2^62), `interval` (tight [RD(x), RU(x)] enclosures: approximate, certified). `convert` follows math_carriers'
matrix (q64 -> bigq lossless, bigq -> q64 narrowing, -> gfp reduction, -> interval enclosing); from gfp or interval to an
exact carrier, or gfp(p) -> gfp(q): `CarrierError`. `values + values` is the sum law in a carrier (same plan,
convention, carrier and field).

## Budgets and refusals

| object | resources (checked before allocation, in the fixed order) |
|---|---|
| circuit | variables = params + columns, circuit = nodes, bytes |
| plan | variables = m (jet dimension), degree = d (order), support = S, tensor = C(2m + d, d) structure constants, circuit = evaluated nodes, bytes |
| aggregate (CPU) | support = S, states = groups x S, bytes (states 16 G (S + 1), coverage, workspace, objects) |
| aggregate (CUDA) | the larger of the device copies (`tr_jet_device_need`) and the adopted aggregate |
| values | output = jets x S, bytes |

`AlgebraBudgetError` carries `.resource/.need/.limit`; `plan.need(scope, backend)`, `Circuit.need`, `plan_need` report
the needs. Representation limits with an admitting budget are `UnsupportedError`: more than 64 parameters, 32 columns,
2^20 nodes, 64 directions, order 16, S = 65535, exponent 255, 2^63 rows. Malformed input is `ValueError`.

## CUDA

`tr_mgpu_jet_aggregate` (`backend="cuda"`: `"cuda"` on a device, `"cuda-emulated"` in the emulation): W work items
(at most 2^16, at least 16 rows each, interleaved workspaces under 128 MiB), launches of min(2^20, 64 W) rows; each item
evaluates its rows with the CPU's inline code, sums locally in 128 bits while the group is unchanged and flushes with
`tr_atomic_add_wide`; a refused row sets a flag and the request is refused. States are adopted after consistency checks.

## Limitations

- Evaluation is limited to int64 numerators over int64 fixed denominators: circuits with many rational or decimal
  factors (denominators multiply) or large data refuse with `OverflowError` before an exact wider result could be
  formed; there is no big-rational or GF(p) *evaluation* (bigq and gfp are conversions of exact results).
- Static supports are conservative (no cancellation analysis); the convolution is dense on them.
- Single-output circuits; centers are per plan (points vary data, not the expansion point).
- The CUDA path is the aggregate kernel only; plan compilation, readouts and conversions run on the CPU.
