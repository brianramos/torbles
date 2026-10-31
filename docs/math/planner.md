# Complete-plan choice and cost evidence (FND-008)

`torbles/math_planner.h`, `math_planner.c`, `math_planner_gpu.cu` (the selectivity sample), `math_planner.py`. The
native planner of the registry's requests (docs/math/registry.md): it samples the selection, costs every candidate the
registry offers component by component, refuses inadmissible and over-budget candidates, chooses the cheapest
admissible one (or the forced one) and returns the complete evidence.

```python
from torbles import math_registry as reg, math_planner as mp
req = reg.Request(reg.Column(db["t"], "x"), [[1, 0, 3]], row_ids=db.query(t).where(t.y < 5).rows())
mp.sample(req, backend="cuda")                    # sample counts, exact bounds, estimate, empty_proven / zero_sample
plan = mp.plan(req, devices=("cpu", "cuda"))      # {"choice", "chosen", "reason", "candidates": [...], "selectivity"}
out = mp.run(req, force="histogram@cuda:cpu")     # executes exactly that, or raises (plan in .plan)
mp.verify(catalog, "sum", "s", "x", "y", force="verify@cuda")   # DISC-001: CPU verifier or CUDA kernel
mp.coefficients({("scan", "cpu"): {"chain": 3.0}})              # an override table for plan(coefficients=...)
```

## Selectivity

`sample(request, sample_words=32, exact=False, backend="cpu"|"cuda")` counts the selected rows of the words
i * (words - 1) / (n - 1), i < n (every word when `exact` or n >= words) on the CPU or with the CUDA kernel (equal
counts; its device copy of the selection is budgeted, `bytes` = 8 per word + 16). The sampled words are counted
exactly, so `lower` = sampled selected and `upper` = lower + unsampled rows are exact bounds. The costs use
`estimate` = rows (k + 1) / (m + 2) (k selected of m sampled rows, rounded half up, clamped to the bounds). A sample
with no selected row sets `zero_sample` and is no evidence of emptiness: only an exact zero count sets
`empty_proven`, which the `empty` candidate requires; otherwise the plan scans. Exactness bounds use `upper`, never
the estimate. A wrong selectivity cannot produce a wrong result: the `empty` executor recounts the selection, and
POWER / BITPLANE recheck their bound with the exact count.

## Candidates and costs

The registry's bounded search (at most 16, budget `iterations` = candidates searched): `empty`, `scan`, `histogram`,
`power`, `bitplane` on the CPU, the reuse of a given `state=`, and with `devices` including `"cuda"` `scan`,
`histogram` (GPU and CPU finalization), `power`, `bitplane` on the device. A candidate is refused by the registry
(reason `"refused by the registry"`, `refusal.kind`) when a chain fails (law, conversion: POWER / BITPLANE for
MIN/MAX), a proof is missing (`proof`: emptiness; `bound`: SCAN / HISTOGRAM need every Horner step within int64 over
the code domain, POWER / BITPLANE |result| < 2^127 with `upper` rows; `stale`: a state of another scope), or its
needs exceed the budget (`"over budget"`, `budget` = resource, need, limit).

Each candidate keeps its exact work `amounts` and `costs` per component: `words` (selection words visited), `scan`
(rows x columns read), `chain` (per-row operator steps), `combine` (one-time algebraic combination: plane weights),
`construct` (state elements built), `reuse` (scope proof of a cached state), `fusion` (<= 0: the batch - 1 passes a
fused execution of the batch saves on the `shared` components, which are counted for batch unfused passes),
`materialize` (output bytes), `transfer` (host <-> device bytes), `cpu_final`, `gpu_final` (finalization on the CPU
or the device) and `fixed` (calls, launches). cost = amount x the module's coefficient for the device (or the
`coefficients=` table [module id][cpu, cuda][component]); `total` is their sum. Costs are relative work units, not
time. Status: `chosen` (`"cheapest"` or `"forced"`), `admissible` (`"costlier"` or `"not forced"`), `rejected`;
ties go to the first candidate. Native layouts: evidence 6168 bytes, selectivity 80, force 16, candidate 376.

## Forced controls

`force=` or `TORBLES_MATH_PLAN_FORCE`: `"<representation>[@cpu|@cuda|@reuse][:cpu|:gpu]"` (finalization), e.g.
`"power"`, `"histogram@cuda:cpu"`, `"histogram@reuse"`, `"verify@cuda"`. The variable may hold one entry per request
kind, comma-separated (`"power,verify@cuda"`); `force=` applies strictly. The cheapest admissible candidate matching
the force runs; if every match is rejected the plan is refused with the first match's exception (e.g.
`AlgebraBudgetError` for a device copy over `bytes`, `CarrierError` for POWER on MIN/MAX, `UnsupportedError` for
`empty` without proof, `OverflowError` for an unprovable bound, `StaleStateError` for a stale state), and with
`UnsupportedError` if nothing matches; a forced representation is never replaced. A force naming `@cuda` adds the
CUDA candidates. An unknown force text is `ValueError`.

## Refusals

With no admissible candidate the plan raises `AlgebraBudgetError` when one was over budget, else the first refusal's
exception; `.plan` carries the full evidence. `run` executes the chosen candidate through the entry point its module
names (`reg.execute` / `reg.verify`), so its result carries the execution's explanation as well as `plan`.
