# ClickBench 100M on Torbles 0.9.0

The November 6, 2026 run used 99,997,497 rows and 105 columns from all 100
ClickBench Parquet partitions. The canonical file normalizes text, dates,
and timestamps consistently for the engine loaders. It is 13,708,815,074
bytes with SHA-256
`51464be4fb65cf061636f236bcf4d2b28e9167f77d5aa018d03b16fbc895f4c0`.
[ClickBench fixture and queries](https://github.com/ClickHouse/ClickBench/tree/072ed80e09fde25c7a11ccdd407c5731aaee9bc7)

[Comparison table](../README.md#clickbench-at-100m-rows) · [Per-query timings and verification](CLICKBENCH-0.9.0-AC.csv)

## Method

All 11 configurations ran sequentially on AC in WSL2. The laptop used the
Windows Balanced power plan, a Core Ultra 9 290HX Plus with 24 logical CPUs,
54 GiB available to WSL2, and an RTX 5090 Laptop GPU with 23.9 GiB of VRAM.
A separate learning process was active during portions of the direct-Parquet
DuckDB and DataFusion sweeps. The table flags both rows.

Each engine ran one first execution and three warm repetitions per query.
The timer includes fetching the result. The reported first total sums the
first execution of each query. Earlier queries in the same sweep can warm
shared state. The operating system file cache was left in place, so the
first total is a process-level measurement with variable file residency.
The warm total sums the three-repetition median for each query. Opening or
preparing the store, data import, and result verification are timed separately
or performed outside the sweep.

Torbles ran commit `2b2b68b116095eda8fcad7cad336fc5decc62bcf`. Its CPU and
CUDA runs each reopened the same 14-part mapped bitplane archive using
packed text and the same 394,417,380-byte referrer index. Reopen and index
attachment took 1.508 s for CPU and 3.379 s for CUDA. The archive uses ordinary
bitplanes. This fixture exercises the persisted text index and SQL execution
paths.

DuckDB native used a 24 GB memory cap and reopened its database in 0.036 s.
Hyper, Umbra, CedarDB, and StarRocks used preloaded native tables. chDB reused
its disk-backed MergeTree table. Direct-Parquet DuckDB, DataFusion, and Sirius
used the canonical Parquet file.

## Verification and execution routes

The verifier checked all 1,892 timed answers against the native DuckDB run,
allowing legal alternatives at tied rank boundaries. StarRocks used a decimal
input to AVG in Q4 and chDB used a widened accumulator. The text adapters
preserved DuckDB's line-feed handling for Q29. Separate verification covered
all 43 Torbles CUDA route replays.

Torbles executed CUDA work in 41 replays. Q1 used row-count metadata, Q41 used
the CPU, and Q22, Q23, Q39, and Q42 used both CPU and CUDA. Sirius logged
40 GPU scans, two metadata routes, and one CPU fallback on first executions.

First-use part opens in Q2 and Q4 accounted for 28.1 s of the 41.656 s
Torbles CUDA first total. The indexed Q29 scanned zero fact rows on both
backends, with warm medians of 2.63 ms on CUDA and 248.75 ms on CPU.
