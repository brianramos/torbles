# Factorized probabilistic inference (PROB-002)

`torbles/math_factor.h` (every contract; the elimination step `tr_factor_cell` is inline and shared with CUDA),
`math_factor.c`, `math_factor_gpu.cu`, `math_factor.py` (opt-in). Bounded factor graphs over discrete variables with
declared finite supports: the partition function, marginals of any variable set and conditional marginals by
sum-product variable elimination, the maximum with every tied assignment and the k best by max-product, exact rationals
throughout, along an elimination order chosen by the FND-008 planner ([planner.md](planner.md)) under explicit
treewidth and intermediate-support budgets ([budgets.md](budgets.md)).

```python
from fractions import Fraction
from torbles import math_factor as mf, math_tensor as mt, math_prob as mp, math_conv as mc
from torbles.math_budgets import Budget
V = {"x": 2, "y": [0, 1, 2], "z": mf.values([10, 20, None])}      # declared supports; None = the NULL category
F = [mf.table(["x", "y"], [[1, 2, 0], [Fraction(1, 2), 1, 3]]),    # exact rationals >= 0, C order (or a dict)
     mf.tensor(mt.counts(scope, ["y", "z"]), ["y", "z"]),          # TENSOR-001 count tensor: axes by value
     mf.categorical(mp.categorical(S), ["x"], [1, 3]),             # PROB-001 categorical model + one request's weights
     mf.stats(mp.stats(hist, S2), ["y"])]                          # PROB-001 retained masses of one group
G = mf.graph(V, F, declaration=mc.independent(scope))              # data factors of one source need it
G.partition(); G.partition({"z": 20})            # Result: Z_e, probability = P(e) = Z_e / Z, Z
b = G.likelihood([{"z": 10}, {"z": None}], backend="cuda")         # a Batch: b[r] a Result (read lazily)
b.column("probability"); b.column("Z_e"); b.raw; b.equal(other)   # a whole exact column in one native call
G.marginal(["x", "y"], evidence={"z": 20}).table                   # {(x, y): P(x, y | e)}
G.conditional(["x"], given=["y"]).table                            # {(x, y): P(x | y) or None}
m = G.map(); m.weight, m.map_probability, m.count, m.assignments   # every tied maximizer
k = G.kbest(3); k.best, k.tied                                     # [(assignment, weight, probability)], ties beyond k
p = G.plan(["x"], evidence=["z"], devices=("cpu", "cuda"), budget=Budget(degree=2, tensor=4096))
p.candidates; p.chosen; p.info                                      # FND-008 evidence of every candidate order
q = G.query(["x"], evidence=["z"], requests=100); q.run(values, backend="cuda"); q.raw(values)
G.count(method="symcount" | "brute" | "diagram" | "tensor")        # COUNT-002 / TENSOR-002 alternative backends
G.info; G.factor(i); G.table(i); G.normalizer()
```

## The model

- **Variables**: n <= 4096, each with a declared support of 1..2^16 values: an int d (the values 0..d-1), a sequence of
  logical integers (None: the NULL category, at most one), `mf.values(seq, scale)` (Decimal / Fraction values scaled by
  10^scale), or a one-coordinate PROB-001 support. Declared, never inferred: a declared value without mass keeps
  weight 0 wherever a factor says so.
- **Factors** (scopes of at most 31 distinct variables; the weight of an assignment is the product of its factors'
  entries, a zero entry a hard constraint):
  - `table`: exact rationals (int, Fraction, Decimal; floats are refused) with int64 numerator and denominator.
  - `tensor`: one group of a dense or sparse TENSOR-001 tensor, axis k on scope[k]. Count axes map by value (lo + i,
    or the NULL slot) to the variable's declared values, scales equal; index axes (GENERIC tensors) by index, their
    dimension the variable's domain. A nonzero cell whose value is not declared is refused (`DomainError` with
    `.factor`, `.variable`, `.value`, `.reason` "value"; "scale", "dims", "negative" likewise); moment, factored and
    counted (SQL) tensors are `UnsupportedError`.
  - `categorical`: a PROB-001 categorical model with one request's weights (or `mode="probabilities"`: exactly summing
    to 1, else `DomainError`): each support point's coordinates map by value to the scope's variables; the cells
    outside the model's support are 0. A point of positive weight outside the declared values is refused. Exponential
    families are `UnsupportedError` (they are enclosures, not exact tables).
  - `stats`: one group's masses of PROB-001 statistics (with their scope); products of statistics are refused.
- **The declared factorization**: tensor and statistics factors record their scope's data source (GEN-002's rule: a
  table's address and identity token, or an array scope's row descriptor). Two data factors of one source multiply
  only under a GEN-002 declaration covering it (`mc.independent(scope)`): otherwise `DomainError` with `.reason`
  (`needs_declaration`, `not_covered`), `.factors` (the pair), `.keyed`, `.source`. `G.info["declared"]` and
  `["linked"]` report it. Different sources and source-free factors multiply freely. Marginals of a variable set come
  from the graph by elimination, never as products of single marginals.

## Queries and answers

Evidence (a dict {variable: declared value}) conditions by restriction; a batch shares the observed variables. Per
request: `Z_e`, `status` (0, or -6 when Z_e = 0 for the conditional answers: `Result.require()` raises
`DomainError`); partition / likelihood requests also `probability` = P(e) = Z_e / Z and `Z`, from the retained
normalizer `G.normalizer()` (the elimination without evidence, computed once natively; the other answers are
normalized by Z_e and never need it, so they are never refused for it). `marginal`:
P(q | e) for every assignment of the variables (row-major, their order). `conditional`: P(q | r, e), None where
P(r | e) = 0. `map`: w* = the largest weight, `map_probability` = w* / Z_e, `count` and every tied maximizer as full
assignments sorted lexicographically by variable index (refused past the `output` budget with the exact count as the
need). `kbest(k)`: the first k assignments of positive weight in the canonical order (weight decreasing, then
lexicographic), their weights and probabilities, and `tied` = the assignments of the k-th weight beyond k. Every value
is an exact `Fraction`, reduced natively (FND-009).

## Elimination and exactness

A candidate order of the eliminated variables is simulated by buckets: each factor (restricted by the evidence) goes
to its first-eliminated variable's bucket; a step multiplies its bucket's tables and sums the variable out (SUM) or
maxes it out (MAX: the maximum, the minimum index reaching it, the number of maximizing completions); the final step
multiplies the remaining tables into the output over the kept variables. Factors are scaled to integers over their
denominators' lcm D_f (< 2^126); every message value and count is an exact integer below 2^(64 L), L proven from
bitlen(prod max entries) + bitlen(prod domains) (`G.info["limbs"]`), computed with COUNT-002's L-limb arithmetic;
answers divide by D = prod D_f exactly. The k best are kept as lists of the top k with ties per message cell (closed
under the products and maxima of nonnegative weights, so the result is exact whatever the tie order) and decoded
through backpointers.

## The costed plan

Candidates: `exact` (a dynamic program over the subsets of the eliminated variables, at most 14 by default
(`dp_limit`) and 64 free variables: among the orders whose every step fits the degree / tensor / support budgets,
the least total joint cells; no such order = proven refusal), `min-fill`, `min-degree`, `min-weight`,
`weighted-fill`, `natural`, `reverse`, a `given` order (`order=`), duplicates merged, each on the CPU and, with
`devices` including "cuda", on the device. Per candidate, exactly: `width` (the induced width: joint table variables
- 1), `joint` (the largest joint table), `support` (the largest intermediate table: message or output), `products`,
`arena`, `bytes`, `device_bytes`, `limbs`. Needs: variables = free variables, degree = width (the treewidth budget),
support, tensor = joint, states = R x arena, output = R x values, bytes. Costs: scan (operand entries read), chain
(products x L), construct (message limbs), materialize (answer bytes), transfer (device bytes), fixed (steps or
launches), x R, with `mf.default_rates()` (relative work units, NOT time) as FND-008's override table;
`tr_planner_choose` rejects over-budget candidates and runs the cheapest admissible one (the first on a tie). `force=i`
runs exactly candidate i or refuses with its code. No admissible candidate: `AlgebraBudgetError` (FND-008's first
over-budget candidate's resource, need, limit) with `.plan`, `.every_order` (proven for every order: the DP's
infeasibility, or a lower bound past its limit) and `.minima` (`width`, `joint`, `support`: exact minima over every
order when `exact`, else lower bounds: minor-min-width, the first step's table, the output). The search itself needs
iterations = orders evaluated + DP subsets (the DP is skipped, never refused, when it would not fit) and bytes.

## CUDA

`backend="cuda"` (`tr_mgpu_factor_eliminate`): one launch per step, one work item per (request, output cell)
running `tr_factor_cell` into its own slots (no atomics on values; argmax ties broken by the minimum index; an overflow
flag by atomic OR, excluded by the L proof); 7 L words of local scratch per item, so L <= 16 on the device (a CUDA
candidate past it is rejected, `device_ok` False). The host checks the program, the evidence bases and the budget
(bytes = every device copy, states = R x arena) first. The raw arena (sums, maxima, counts, argmax) equals the CPU's
bit for bit (`Batch.equal`); answers are adopted natively from either (`Batch.backend`: `cpu`, `cuda` or
`cuda-emulated`). k best, alternative backends and decoding run on the CPU.

## Alternative backends (`G.count`)

`symcount`: COUNT-002's variable elimination (`tr_symc_factors_count`, min-fill) of the restricted tables; `brute`:
COUNT-002's enumeration of the factor universe; `diagram`: a COUNT-002 decision diagram of the parameter encoding
(binary free variables; per entry w != 1 the constraint theta <=> [scope = a] with literal weights (w, 1), w = 0 a
NOT; at most 64 program variables) counted with literal weights; `tensor`: TENSOR-002's own costed contraction of the
dense scaled tables (at most 8 tables over 16 variables, scopes of 8, every step's bound proven below 2^127). Each
returns Z_e exactly or refuses (`UnsupportedError`, `OverflowError`, `AlgebraBudgetError`).

## Budgets and refusals

| request | checked before allocation |
|---|---|
| graph | variables = n, support = table entries, tensor = the largest table, bytes = the object and source keys |
| plan | iterations = orders + DP subsets, bytes = the plan and DP arrays; candidates: the needs above |
| program | bytes = steps, operands, strides, dims |
| elimination | states = R x arena, bytes = scratch (CPU), device copies (CUDA) |
| answers | output = R x values (MAX: + every maximizer, exact count first), bytes; k best: support = the longest list, bytes = every list (as they grow), output = the assignments of weight >= the k-th |

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `DomainError` (values outside the declared supports,
negative entries, probabilities not summing to 1, linked data factors without a covering declaration),
`UnsupportedError` (exponential families, moment / factored / counted tensors, products of statistics, scopes past
31 variables, domains past 2^16, denominators or entries past 2^126, L past 4096 limbs; alternative backends outside
their limits), `TypeError` (floats), `ValueError` (malformed scopes, values, evidence, orders), `StaleStateError` (a
table scope of a data factor changed).

## Limitations

Exact tables only: no exponential-family factors (no enclosures), no certified bounds past the budgets (refused,
never bounded or extrapolated). The exact DP covers at most 14 eliminated variables by default (24 at most); beyond,
the heuristic candidates may miss an order that fits, and a refusal is proven only by the lower bounds. Every message
is kept in the arena (the decoders read them), so memory is the whole arena; the CPU loops are single-threaded; on
the device L <= 16 limbs. Evidence is a fixed value per observed variable (no soft or virtual evidence); MAP is over
all unobserved variables (no marginal MAP). Data factors record one group's state; the declaration is the caller's
statement, never verified against the data. Rates are hand-set estimates (not calibrated).
