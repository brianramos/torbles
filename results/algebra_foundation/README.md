# Exact mathematical function foundation: evidence

This is the first bounded implementation slice of the 32-family
[inventory](../../ALGEBRA_INVENTORY.md), not validation of every planned module.
The runtime measured here is commit
`b4ca3486904e823be14bfc4ce92a6d52e9188db7`.

- `validation.json` and `quick.log`: complete actual-CUDA quick suite passed
  with return code zero. It includes the new mathematical API tests and all
  previous relationship, stage, SQL, type, persistence and GPU controls.
- `focused.log`: all 23 mathematics tests pass. An additional higher-order
  derivative case was added after the full suite, then the whole focused file
  was rerun on the same unchanged runtime.
- `native_validation.json`: independent Fraction oracles for 200 seeded
  polynomial programs, 60 moment banks, 3,969 extreme rational pairs and 11,907
  binary results; warning-clean compilation and ASAN/UBSAN/leak checks pass.
  Reproduce with `tests/validate_mathematics_native.py`.
- `benchmark_cpu.json` and `benchmark_gpu.json`: final frozen-runtime function
  family measurements, with all exact comparisons passed and source/library
  hashes recorded. CUDA runs reject emulation and CPU row fallbacks.
- `dev_integrity.json`: original 174 tracked files and 189 nested original
  0.2.0 files match their preserved baselines byte for byte.

## Measured experiment

Synthetic signed integer columns, a fixed scalar predicate selecting 60% of
rows, and a quadratic loss `sum((y-a*x-b)^2)`. Each batch evaluates 25 parameter
pairs. The baseline runs independent ordinary native aggregate queries. The new
path constructs six degree-two moments, reduces the row expression to a native
parameter polynomial, then evaluates that polynomial. Gradients, Hessian terms
and an exact rational definite integral are independently checked outside the
timed batches. A structural guard confirms post-snapshot evaluation dispatches
zero fact queries.

Values below are median milliseconds across five warm serial batches. Every
sample is retained in JSON. CPU materialization and baseline queries use 24
threads. Parameter algebra executes in native CPU code for both rows of backend
results; CUDA is used for the requested fact aggregates, not a new calculus
kernel.

| Rows | CPU snapshot build | 25 independent CPU queries | 25 function evaluations | CUDA snapshot build | 25 independent CUDA queries | 25 function evaluations after CUDA build |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1.419 | 0.324 | 0.165 | 152.248 | 2.749 | 0.185 |
| 100,000 | 0.835 | 1.113 | 0.171 | 2.199 | 2.736 | 0.195 |
| 1,000,000 | 4.533 | 12.850 | 0.209 | 2.965 | 2.776 | 0.179 |

Construction is **not** a fused single pass: two data coordinates use six
existing native aggregate queries. Symbolic construction and first-batch costs
are separate JSON fields. The first CUDA build includes initial executor/data
setup, explaining its much larger cold cost. Snapshot construction happens
before the independent-query baseline, so baseline first calls already have the
shared executor/data residency established. These are warm reuse measurements,
not independent cold-start comparisons.

This is one process per backend invocation, with ascending sizes and five
within-process repetitions. It is a small synthetic demonstration of exact
state reuse, not a general engine speedup, confidence interval, or replacement
for the mixed SSB comparison. It does not measure automatic selection of when a
bank should be built; that optimizer decision remains an inventory item.

The `*_pilot.json` files precede the final metadata/source freeze and are retained
as history, not pooled into the table. `quick_initial.log` records a successful
suite whose outer shell-status forwarding failed afterward; the final explicit
subprocess execution records the reliable zero return code in `validation.json`.

The previous SSB native runtime and libraries were preserved before rebuilding:
`/home/brian/torbles_discovery_bench/stages_f44cf8f`. Earlier mixed-scaling data
and handoff ZIPs remain unchanged.
