# Module and operator registry (FND-005)

`torbles/math_registry.h` (tables and inline helpers shared with the CUDA library), `math_registry.c`,
`math_registry_gpu.cu`, `math_registry.py`. A native, constant registry of the mathematical modules: what each one
computes in, which laws it relies on, what it implements, what state suffices, which proofs it needs, its CPU and
CUDA entry points and its cost coefficients; a composition check over carriers, laws and conversions; and the
executors of one request kind through four representations, on the CPU and on a CUDA device.

```python
from torbles import math_registry as reg
reg.modules()          # id, name, capability, representation, state, carriers, laws relied on / guaranteed,
                       # operators, implements [(operator, carrier)], proofs, devices, cpu/cuda entries, costs
reg.check_chain([("convert", "rational"), ("divide", "rational")], input="integer")   # -> "rational"
reg.check_chain([("divide", "integer")], input="integer")      # CarrierError: needs law division, integer lacks it
col = reg.Column(db["t"], "x")                                 # decoded natively (codes, validity), budgeted
req = reg.Request(col, [[1, 0, 3], [0, 1]], aggregate="sum", row_ids=db.query(t).where(t.y < 5).rows())
reg.execute(req, "histogram", backend="cuda", finalize="gpu")  # {"values", "selected", "valid", "nulls", "explanation"}
st = reg.build_state(req, "power", support=[0, 1, 2, 3])        # reusable: reg.execute(req2, "power", state=st)
reg.candidates(req, devices=("cpu", "cuda"))                    # the bounded candidate search (math_planner uses it)
reg.verify(catalog, "sum", "s", "x", "y", backend="cuda")       # DISC-001 through the registry, with an explanation
```

## Modules

| id | module | capability | representation / state | CPU entry | CUDA entry |
|---|---|---|---|---|---|
| 1 | checked rational polynomials | FND-002 | formal / polynomial | `tr_math_module_apply`, `_evaluate` | — |
| 2 | scalar integer moments | FND-003 | moments / moment bank | `tr_math_module_reduce`, `tr_math_bank_create` | — |
| 3 | distinct cell pre-pass | FND-010 | cells / cell bitmap | `tr_math_budget_count_cells` | `tr_mgpu_budgets_count_cells` |
| 4 | relation verification | DISC-001 | verify / evidence | `tr_reg_verify`, `tr_disc_verify` | `tr_mgpu_registry_verify`, `tr_mgpu_disc_verify` |
| 5 | direct scan | FND-005 | scan / none | `tr_reg_execute` | `tr_mgpu_registry_execute` |
| 6 | code histogram | FND-005 | histogram / counts per code | `tr_reg_execute`, `tr_reg_state_build` | `tr_mgpu_registry_execute`, `_state` |
| 7 | sparse power sums | FND-005 | power / sums mod 2^128 | same | same |
| 8 | bit-plane factored counts | FND-005 | bitplane / counts per plane set | same | same |
| 9 | proven empty selection | FND-008 | empty / exact zero count | `tr_reg_execute` | — |

The first-slice built-ins (1, 2) are described, not changed: their operator sets are read from the validated
`math_modules.c` table, and `tr_math_module_apply` / `_reduce` keep dispatching them. A new family adds one entry
to the `modules` table of `math_registry.c` (and, for an executable representation, its chain in `tr_reg_chains`).
`reg.execute` and `reg.verify` call the entry point the module's descriptor names.

## Carriers, laws, conversions, compositions

Carriers: `integer` (checked int64), `rational`, `boolean`, `gf2`, `count` (multiplicities), `int128`, `mod128`
(residues mod 2^128), `interval` (outward rounded), `float64`, `minmax`, `code` (stored codes), `rowset`. Laws, as
implemented: associative, commutative, identity, inverse, distributive, idempotent (a+a=a), mul_idempotent (a·a=a),
ordered (its comparison is total and compatible with +), division (nonzero elements invertible), char0 (n·1 distinct),
exact, enclosure. Conversions are `exact`, `checked` (lossless or refused: `code -> integer`, `int128 -> integer`,
`mod128 -> int128` under a proven bound) or `enclosing` (`integer/rational -> interval`, not lossless).

`check_chain(steps, input, contract="exact"|"enclosure", implemented=False)`: steps are `(operator, carrier)`; the value
starts in `input`. Accepted only when each step's operator exists, its fixed input carrier (histogram, bits, and,
popcount, mark_cells, verify) is the step's, the carrier change has a declared lossless conversion (enclosing too under
`contract="enclosure"`), and the step's carrier has every law the operator requires (plus `exact`, or `exact`/
`enclosure` under the enclosure contract); with `implemented=True` some module must also implement every step.
Otherwise `CarrierError` (or `OperatorError` for an unknown operator) with `.refusal = {kind: "law" | "conversion" |
"carrier" | "implementation" | "operator", step, operator, carrier, law, conversion}` for the first failing step.
`reg.chain(representation, aggregate)` gives each representation's own chain; POWER refuses MIN/MAX (powers in
minmax: no distributive law) and BITPLANE too (no conversion count -> minmax).

## Requests and representations

A request aggregates one to four integer polynomials f_j (degree <= 8, int64 coefficients, low to high) of the
logical value x = code + offset (a DECIMAL's scaled integer) over the selected valid rows: `sum` (exact, any size),
`min` / `max` (int64; `None` when no selected row is valid). `selected`, `valid` and `nulls` are reported separately;
an empty selection sums to 0 (with `valid == 0`). Representations, each on the CPU and the CUDA device:

- `scan`: every row's f_j by Horner's rule in checked int64 (128-bit products), folded in 128 bits.
- `histogram`: counts per code (state: `ncodes` counts), contracted as sum count_c f(c + offset); CUDA finalizes on the
  device (`finalize="gpu"`) or copies the counts back and finalizes on the CPU (`"cpu"`).
- `power`: S_k = sum x^k mod 2^128 for the polynomials' exponents (sparse), then sum a_k S_k.
- `bitplane`: for every set S of at most D bit planes, the rows whose code has all of S; the sum is sum_S w_S count_S
  with the Moebius weights w_S = sum_{T in S} (-1)^{|S|-|T|} f(T + offset), mod 2^128.
- `empty` (CPU): no scan; it recounts the selection exactly and refuses unless it is empty.

Exactness: `scan` / `histogram` refuse (`OverflowError`) when a selected value's evaluation leaves int64 at a Horner
step (both refuse on the same values); their sums cannot overflow. `power` / `bitplane` run only when
rows x sum_k |a_k| M^k < 2^127 is proven before execution (M the largest |x| of the code domain), and the residue is
then the exact sum; otherwise `OverflowError` with refusal kind `bound`. A logical domain outside int64, or a stored
code >= ncodes in a selected valid row, is refused first (`OverflowError`, `ValueError`).

## Explanations, states, budgets

Every execution returns (or attaches to its exception as `.explanation`) what ran: `module`, `module_name`,
`capability`, `representation`, `device` (`"cpu"`, `"cuda"` on a device, `"cuda-emulated"` in the emulation),
`device_name`, `finalize`, `reused`, `status`, `code`, `refusal` (kind, step, operator, carrier, law, conversion),
`budget` (resource, need, limit), `rows_scanned`, `words_scanned`, `state_elements`, `state_bytes`, `device_bytes`,
`native_calls`. Native layouts (bytes): request 384, result 96, explanation 136, state info 104, candidate 376,
module 528, refusal 32, chain 56.

A `State` (CPU-built, or built on the device and imported) records its scope exactly (source token, revision, rows,
code domain, validity, a copy of the selection); `execute(..., state=st)` contracts it without a scan, and refuses with
`StaleStateError` when the scope differs (an engine mutation makes the `Column` itself stale) and `UnsupportedError`
when it does not suffice (exponents outside a power state's support, a degree above a plane state's).

Budgets are checked before allocation (`AlgebraBudgetError`): `degree` (and the kernel limit 8), `support` (terms),
`output` (polynomials), `states` (histogram cells, power sums, plane sets), `circuit` (plane Moebius terms
sum_S 2^|S|), `bytes` (state: 8 per count, 16 per power sum, 4 per plane-set mask; on CUDA plus the device copies of
codes, validity and selection, 160 accumulator bytes and the host copy of a CPU finalization; a Column 8 or 4 bytes
per row), `iterations` (candidates searched).
