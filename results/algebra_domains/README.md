# Typed algebra domains, executable dispatch and scalar state reuse

Runtime: `bae446434669e110608f41ece4187a911334e477` on
`prototype/mixed-scaling`. The earlier polynomial runtime is preserved at
`/home/brian/torbles_discovery_bench/algebra_b4ca348`, with source commit
`b4ca3486904e823be14bfc4ce92a6d52e9188db7` and its measured library. No original
dev source or previous SSB result was changed.

The current measured source and both exact libraries are also preserved at
`/home/brian/torbles_discovery_bench/algebra_bae4464`:
[runtime_archive.json](runtime_archive.json).

The new native code provides owned typed domain descriptors, two executable
operator modules, and a bounded LRU cache for compatible scalar integer moment
banks. Domains distinguish row bags, selected snapshots, encoded assignments,
formal parameters and declared geometry. Only the polynomial and scalar moment
operators are implemented here; domain labels are declared contracts, not proofs
or implementations of additional mathematics.

## Correctness and ownership

- The full actual-CUDA quick suite passed, including all prior relationship,
  stage/DAG, planner and GPU modes: [quick.log](quick.log).
- All 48 focused polynomial/domain/module/cache tests passed:
  [focused.log](focused.log), [validation.json](validation.json).
- Native descriptor/module edges passed optimized compilation with warnings as
  errors and address/undefined/leak sanitizer checks:
  [native_domain_validation.json](native_domain_validation.json).
- The native cache passed 15,000 seeded operations against an independent LRU
  oracle, dimensions zero through four and capacities one, two and seven, plus
  sanitizer, byte eviction and allocation-failure atomicity checks:
  [native_state_validation.json](native_state_validation.json).
- An independent public-API audit verified 200 exact value/derivative answers,
  a four-coordinate bank, zero aggregate calls on hits, metadata isolation and
  functions surviving source/cache release: [independent_audit.json](independent_audit.json).
- All 174 original dev tracked files and all 189 nested original 0.2.0 files
  match their preserved Git baselines byte for byte:
  [dev_integrity.json](dev_integrity.json).

## Compatible state reuse

Synthetic signed integer `x`, `y`, and a five-valued flag; predicate `flag>=2`
selects 60% of rows. Each timed batch performs five serial fresh
`MomentBasis` constructions, cross-moment reads and closes. Both paths include
source fingerprint checks. Each uncached batch builds thirty native aggregate
queries; compatible cache batches build zero. Verification against independent
DuckDB SQL and metadata inspection occur outside timers.

Warm batch medians in milliseconds, seven paired alternating-order repetitions:

| Rows | CPU uncached | CPU cached | CUDA uncached | Cached after CUDA build |
|---:|---:|---:|---:|---:|
| 1,000 | 1.9243 | 0.4741 | 4.6125 | 0.4486 |
| 100,000 | 2.4026 | 0.4746 | 5.7777 | 0.5193 |
| 1,000,000 | 15.6217 | 0.8186 | 5.4381 | 0.7266 |

At 1M rows these ratios are 19.1x and 7.5x for this repeated construction
workload. Cache lookup/cloning executes on CPU even when the original bank was
built on CUDA. Hit receipts deliberately report no actual row device or GPU
executor. CUDA misses all used the actual GPU executor, without CPU fallback.

First cache requests are separate: CPU 1.333/1.059/5.252 ms and CUDA
139.377/1.973/2.501 ms in ascending row order. The first GPU request includes
substantial CUDA initialization. Afterward, incompatible predicate, explicit
clear and flip/repair revision changes each cause a measured miss/refill; all
values and counters pass. A hit still hashes membership/rank metadata, so zero
aggregate calls does not imply constant-time construction or zero memory reads.

Raw samples, setup, invalidation, devices, query counters and source identities:
[cache_cpu.json](cache_cpu.json), [cache_gpu.json](cache_gpu.json).
An independent read-only audit reconciles all six benchmark reports, source and
library identities, raw sample counts and query/device counters:
[benchmark_audit.json](benchmark_audit.json).

## Descriptor and dispatch costs

For 1/16/128 owned descriptors, the CPU-run construction medians were
0.0046/0.0454/0.3488 ms; exact clone comparisons were
0.0011/0.0083/0.0605 ms. Original retained descriptor bytes were
92/1,472/11,776. Changed-revision rejection and independent CUDA-process
descriptor samples are included in the cache receipts. No row queries are
executed by these descriptor microbenchmarks.

The same response-function harness was also run against the preserved prior
runtime and the typed runtime, separately from the cache experiment. Both
materialize a bank once and then evaluate 25 parameter settings per warm batch.
At 1M rows, the native CPU coefficient-evaluation medians were:

| Materialization backend | Earlier runtime | Typed runtime | Difference |
|---|---:|---:|---:|
| CPU | 0.2198 ms | 0.2533 ms | +15.3% |
| CUDA | 0.1797 ms | 0.2153 ms | +19.9% |

This slice adds checks and dispatch overhead; it is not an across-the-board
speed improvement. Coefficient evaluation always executes on CPU. Current
independent native-query medians at 1M were 12.8282 ms CPU and 2.9002 ms CUDA;
current bank construction was 4.5481/2.5541 ms respectively. First calls,
symbolic construction, every raw sample and exact answer hashes are retained:
[response_baseline_cpu.json](response_baseline_cpu.json),
[response_current_cpu.json](response_current_cpu.json),
[response_baseline_gpu.json](response_baseline_gpu.json),
[response_current_gpu.json](response_current_gpu.json).

## Reproduction and limits

Use the WSL benchmark environment `/home/brian/torble_gpu_run/venv/bin/python`
from the prototype directory, with `PYTHONDONTWRITEBYTECODE=1`:

```sh
python bench/benchmark_math_state.py --engine alg --rows 1000,100000,1000000 --requests 5 --repeats 7 --threads 24 --output /tmp/cache_cpu.json
python bench/benchmark_math_state.py --engine gpu --rows 1000,100000,1000000 --requests 5 --repeats 7 --threads 24 --output /tmp/cache_gpu.json
python bench/benchmark_mathematics.py --engine alg --rows 1000,100000,1000000 --parameters 25 --repeats 5 --threads 24 --output /tmp/response_current_cpu.json
python bench/benchmark_mathematics.py --engine alg --source /home/brian/torbles_discovery_bench/algebra_b4ca348 --rows 1000,100000,1000000 --parameters 25 --repeats 5 --threads 24 --output /tmp/response_baseline_cpu.json
```

Repeat the last two commands with `--engine gpu` and distinct output paths for
the CUDA materialization comparison. Run benchmarks sequentially without builds
or tests in parallel. Native validation runners are
`tests/validate_math_domains_native.py` and
`tests/validate_math_state_native.py`; both accept `--receipt PATH`.

Hardware: Core Ultra 9 290HX Plus, 24 logical CPUs; RTX 5090 Laptop GPU; WSL2;
DuckDB 2.0.0; Python 3.14.4. Existing unrelated user research processes were
left running without priority/affinity changes. Each invocation is one process
with ascending scales and within-process repetitions, not independent process
rounds or a confidence interval. Earlier/current processes run sequentially,
so small dispatch differences also include ordinary process/system variation.
Raw outliers are retained. The populated cache establishes shared data/executor
residency before the uncached first batch. These are bounded synthetic pilots,
separate from the mixed SSB scaling matrix.

Still unimplemented: nullable/grouped banks, refinement/rollup, fused multi-output
accumulation, calibrated complete module costs and automatic build/reuse
selection. Source mutation versus construction, and explicit object closure
versus native operations, require external serialization; this is not MVCC.
