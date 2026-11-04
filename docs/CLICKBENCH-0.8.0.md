# ClickBench at 100M rows

The fixture contains 99,997,497 rows and 105 columns from all 100 official
Parquet partitions. A canonical copy normalized text, dates, and timestamps
for the engine loaders. It contains 13,708,815,074 bytes with SHA-256
`51464be4fb65cf061636f236bcf4d2b28e9167f77d5aa018d03b16fbc895f4c0`.
[ClickBench dataset and queries](https://github.com/ClickHouse/ClickBench/tree/072ed80e09fde25c7a11ccdd407c5731aaee9bc7)

All runs used the same WSL2 laptop with a Core Ultra 9 290HX Plus, 24 logical
CPUs, 54 GiB available to WSL2, and an RTX 5090 Laptop GPU with 23.9 GiB of
VRAM. Windows used its Balanced power plan. The Torbles measurements used
prototype commit `50122b03fe59d720e5dc6c40ea64e06645529986`, whose CUDA and SQL
changes are included in 0.8.0. Each query ran once followed by three warm
executions. Timing includes fetching the result. Setup is separate. The
first-execution totals sum one first execution per query, with earlier
queries able to warm shared state.

## AC power

Every configuration verified all 43 queries against the native DuckDB run
from the same power condition.

| Engine and layout | Version | Warm total (s) | First executions (s) |
|---|---|---:|---:|
| Torbles CUDA, mmap bitplanes | 0.8.0 prototype | **2.061** | 48.704 |
| DuckDB, native | 2.0.0 | 11.710 | 15.368 |
| Hyper, native | 0.0.26963 | 12.887 | 28.505 |
| Torbles CPU, mmap bitplanes | 0.8.0 prototype | 16.831 | 36.477 |
| DuckDB, direct Parquet | 2.0.0 | 21.744 | 22.615 |
| StarRocks, native | 4.0.14-7c18dd9 | 28.147 | 47.215 |
| ClickHouse via chDB, MergeTree | 26.9.2.1 via chDB 4.4.0 | 32.445 | 35.996 |
| DataFusion, direct Parquet | 55.0.0 | 34.652 | 37.202 |
| Sirius, streamed Parquet | 76ef37d231 with DuckDB 2.0.0 | 65.211 | 82.100 |

The ClickHouse adapter uses an Int128 accumulator for Q4 and matches DuckDB's
line-feed semantics for Q29. The StarRocks total includes a separately timed
decimal Q4 on the same native table and power condition. Every other
StarRocks time is from the original AC sweep. [AC per-query results](CLICKBENCH-0.8.0-AC.csv)

## Battery power

| Engine and layout | Version | Warm total (s) | Verified |
|---|---|---:|---:|
| Torbles CUDA, mmap bitplanes | 0.8.0 prototype | **4.074** | 43/43 |
| Umbra, native | 26.10 (92839196f2) | 4.960 | 43/43 |
| CedarDB, native | 2026-10-22 | 8.900 | 43/43 |
| DuckDB, native | 2.0.0 | 16.591 | 43/43 |
| ClickHouse via chDB, MergeTree | 26.9.2.1 via chDB 4.4.0 | 32.244 | 42/43 |

Battery ClickHouse overflowed on Q4. Its total excludes that query. On the
same 42-query set, Torbles CUDA took 4.065 s. The first CUDA sweep took
110.528 s on battery. [Battery per-query results](CLICKBENCH-0.8.0-battery.csv)

## Execution and storage

The earlier 0.7.0 CUDA campaign reached five-minute cutoffs on Q28 and Q29 at
this scale. In the full AC rerun with the new paths, Q28's warm median was
86.0 ms on CUDA and 48.5 ms on CPU. Q29 took 41.6 ms on CUDA and 2.40 s on CPU.
Q29's first AC CUDA execution took 12.19 s, including building its regex
mapping over the dictionary.

Separate result-verified replays observed successful CUDA calls on all 43
Torbles queries under each power condition. Q22 and Q23 also issued native CPU
query calls. Sirius logged GPU scans for 40 queries, metadata routes for Q1
and Q7, and CPU fallback for Q29. Verification checked each returned answer
against DuckDB, allowing rows tied at the query's specified rank.

Each Torbles process reopened the same staged 14-part bitplane store.
Building it from the native DuckDB source took 648 s with functional-dependency
grouping disabled. The store contained 39.32 GB of bitplanes and 6.37 GB of
sidecars, with 57.29 GB of files including dictionaries and metadata.
The DuckDB native import took 80.44 s and occupied 18.73 GB.
