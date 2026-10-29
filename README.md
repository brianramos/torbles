# torbles

**An algebraic relational datastore on torus-encoded bit planes.** Version 0.1.0, a research prototype: the
benchmark results are measured and verified, but it is not yet a database to deploy (see
[Known limitations](#known-limitations-010)). The API may change.

A *torble* is a toroidal table: it stores every column as bit planes (plane j holds bit j of 64 rows per machine
word) laid out on a 16 x 32 torus whose faces are XOR checks, so every row is protected and single-bit faults are found and repaired. Queries are
not executed row by row. They are compiled into algebra over the planes:

- **Filters** are Boolean polynomials over the planes, evaluated 64 rows per instruction.
- **SUM** is popcount algebra: `SUM(x) = sum_j 2^j popcount(x_j & F)`. Products of columns use pairs of planes.
- **An equi-join is a substitution.** A dimension filter is a polynomial in the primary key's bits. Replacing the
  primary key with the foreign key evaluates the join directly on the fact's planes: no hash table, no probe.
  Hierarchical keys turn filters into a few cubes (ANDs of literals) or comparators.
- **Range, band and inequality joins** use the same comparator polynomial, with dominance sums for aggregation.
- **Derived surrogates** keep official keys while giving the planner the hierarchy as id bits. Dimension changes are
  additive: a delta plane plus a second pass, and the fact is never rewritten. Values wider than 32 bits are
  self-extending records, addressed by rank (a popcount), never by stored pointers.
- **GROUP BY is one matrix product**, `C = M^T X` (group masks AND-popcount measure planes), with sparse output.
- **Optional GPU executor (CUDA).** One thread per 64-row word, a sparse group table on the GPU, and optional
  tensor-core b1 MMA kernels. On an RTX 5090 Laptop GPU, the 13 official SSB queries on 300 million rows run in
  29.5 ms end to end, against 896 ms for DuckDB on 24 threads (in memory).

Every result is cross-checked against DuckDB and a plain columnar baseline across join forms, thread counts,
GROUP BY modes and the GPU kernels.

## Quick start

Requirements: Linux (or WSL2), gcc/g++, Python 3.9+ with numpy; duckdb and pandas for the tests and benchmarks;
nvcc for the optional GPU executor.

    make                 # CPU engine + CPU emulation of the GPU kernels
    make test            # every correctness test (about a minute; the GPU tests use the emulation without a GPU)
    make gpu             # optional: the CUDA executor (then `make test` runs the GPU tests on the GPU)

```python
import sys; sys.path.insert(0, ".")              # the repository root
import numpy as np, torbles
from torbles import Table, Query

customer = Table("customer", {"region": np.array([0, 1, 1, 2], np.uint32), "nation": np.array([0, 5, 6, 10], np.uint32)})
sales = Table("sales", {"cust": np.array([0, 1, 2, 3, 1, 2], np.uint32), "amount": np.array([10, 20, 30, 40, 50, 60], np.uint32)},
              msb=("cust",), refs={"cust": customer})

q = (Query(sales).join(customer, sales.cust)
       .where(customer.region == 1)
       .group_by(customer.nation)
       .sum(sales.amount))
print(q.run("alg"))      # {(5,): 70, (6,): 90}; "gpu" runs it on CUDA, "plain" on the columnar baseline
print(q.sql())           # the same query as SQL, for cross-checking
```

Main entry points (`torbles/algq.py`):
- `Table(name, columns, msb=, refs=, extend=, virtual=)`
- `Query(fact).join(dim, fk).where(pred).group_by(...).sum(expr) / .count()`
- `.join_range(dim, x, lo, hi)` and `.join_ineq(dim, x, op, y)`
- `Surrogate(w, s_of_row)`
- `set_threads(n)`, `set_build_threads(n)`, `set_groupby("mm" | "split")`, `gpu_mm("word" | "popc" | "mma")`

`torbles/blobcol.py` has `BlobStore` for CLOB/BLOB columns. The engine is `torbles/alg.c` (CPU) and `torbles/gpu.cu`
(CUDA), sharing `torbles/torble.h`.

## Results

![torbles on the official Star Schema Benchmark, SF1 to SF50](docs/img/ssb_scaling.png)

- **What runs, and what it takes.** At 300M rows the torble fact table (the 9 columns the queries read, as bit planes
  plus their torus integrity planes) takes 15.3 GiB, about 55 bytes per row, and fits on the 24 GB laptop GPU. The
  29.5 ms comes from the default per-word kernel: one thread per 64-row word, filters and SUMs as AND + `__popc` over
  the bit planes, and a sparse group table. A 1-bit tensor-core version of the GROUP BY product
  (`mma.sync.m16n8k256 .b1 .and.popc`) is included and self-tested. It wins only on dense, unfiltered GROUP BYs
  (1.4-1.7x on M1-M3) and is slower on the selective SSB queries, so these numbers do not use it.

### Official Star Schema Benchmark, scaling from 6M to 300M rows

Setup:
- The 13 official queries on official `ssb-dbgen` data, derived-surrogate keys.
- Core Ultra 9 290HX (24 cores, AVX2) and RTX 5090 Laptop GPU, WSL2.
- Both engines answer from memory: DuckDB 2.0.0 runs on an in-memory copy of its tables (the 9 fact
  columns the queries read, as torbles loads), with its default memory limit.
- Every result of every engine at every scale equals DuckDB's.
- Totals in ms, best of 3, end to end (planning, execution, result).
- The last column is each query's first run, its plan compiled from scratch. Torbles' dimension coordinate structures
  were already in place: they are discovered once per table after loading, by the first queries of the CPU pass.

| scale | fact rows | DuckDB, 24 threads | torbles, 24 CPU threads | torbles, GPU | GPU vs DuckDB | first run: DuckDB / torbles GPU |
|---|---|---|---|---|---|---|
| SF1 | 6M | 58.7 | 14.2 | **4.8** | 12.2x | 63.5 / 5.4 |
| SF3 | 18M | 98.6 | 28.5 | **6.2** | 15.9x | 106.6 / 6.8 |
| SF10 | 60M | 212.7 | 77.2 | **9.4** | 22.6x | 223.2 / 10.1 |
| SF30 | 180M | 574.3 | 202.6 | **18.8** | 30.6x | 610.7 / 19.2 |
| SF50 | 300M | 895.9 | 337.6 | **29.5** | 30.4x | 973.3 / 30.9 |

- **Speed-up grows with the data:** the GPU leads DuckDB on 24 threads by 12x at 6M rows and by 30x at 180M-300M.
  At SF50 the GPU work is 24 ms of the 29.5 ms in all.
- **CPU alone:** torbles on 24 threads is 2.6-4.1x faster than DuckDB on the same 24 threads.
- **Ingestion at SF50 (300M rows) on 24 threads:**

  | phase | seconds |
  |---|---|
  | reading the columns from DuckDB | 21.9 |
  | building the torble (bit planes) | 10.9 |
  | GPU upload | 1.3 |
  | peak memory | 27.3 GiB |

Same 13 queries at SF10 on 1 to 24 CPU threads (a separate sweep):

| threads | DuckDB | torbles CPU |
|---|---|---|
| 1 | 2,185.3 | 877.2 |
| 2 | 1,259.9 | 447.2 |
| 4 | 647.9 | 233.2 |
| 8 | 359.3 | 129.9 |
| 12 | 287.0 | 98.9 |
| 16 | 257.9 | 83.2 |
| 24 | 242.7 | 77.1 |

Reports, logs and JSON: `results/scaling/`. Reproduce with `bash wsl/run_scaling.sh` (WSL2 with an NVIDIA GPU), or
`make ssb SF=1` (CPU only).

An earlier run kept DuckDB on its database file and under a 5 GB memory limit. There, DuckDB's first runs included
loading and decompressing columns into its buffer cache: 1.3-2x slower overall, and up to 8x for single queries.
That run is kept in `results/scaling_duckdb_filebacked/`. The SF10 validation run (made while this release was
numbered 1.0.0-rc1), every engine and kernel, and the per-query tables are in `results/rc1_validation/` and `docs/SSB.md`. Further results are in `docs/ENGINE.md`:
range/inequality joins, CLOB/BLOB against DuckDB, LMDB and a filesystem store, scaling with the number of joined
dimensions, deltas, and self-extension.

## Layout

    torbles/     the engine: torble.h, alg.c (CPU), gpu.cu (CUDA / emulation), algq.py (planner, API), blobcol.py
    tests/       correctness tests (run_all.sh runs them all) and their data generators
    bench/       benchmarks: ssb/ (official SSB), blob_bench.py, ineq_bench.py, scale_tables.py
    results/     measured results (JSON, logs, reports)
    docs/        ENGINE.md (design and every feature), SSB.md (benchmark method and results)
    wsl/         run_validation.sh (build, test, benchmark) and run_scaling.sh (SF1-SF50 and threads), WSL2 + NVIDIA GPU

## Known limitations (0.1.0)

- **In-memory only.** There is no on-disk format yet; tables are built from arrays at load.
- **Column width.** Values are unsigned 32-bit. Wider values need declared self-extending columns.
- **First-query cost.** After loading, the first queries on each dimension discover its coordinate structure (which
  id bits each attribute depends on). This is done once, in Python, and dominates the first runs: the 13 queries' first
  runs on the CPU took 2.1 s at SF10 and 7.5 s at SF50, against 77 ms and 338 ms once warm. Doing it at load time,
  natively, is the next step.
- **GPU fallbacks.** Semi-join lookups (unstructured dimension filters) run on the CPU, and the query falls back to
  the CPU engine. The GPU also takes at most 2 looked-up GROUP BY keys and 6 keys in all.
- **Blob payload integrity.** The payload heap is not covered by the torus faces; the heads and lengths are.
- **AVX-512.** The VPOPCNTDQ kernel is untested on hardware: neither test machine has it.

## License

Copyright 2026 Brian Richard RAMOS

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance
with the License. You may obtain a copy of the License in the `LICENSE` file or at
http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions and limitations under
the License.
