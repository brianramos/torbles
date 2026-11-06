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

## First-execution recheck

The 10.166 s prototype and 41.656 s release sweeps opened the same six
bitplane parts at the same queries. Part-open time increased by 29.512 s,
accounting for 94% of the difference. This timer includes host-side
preparation, CUDA allocation, copies, and synchronization.

A fresh-process AC recheck alternated the release and prototype on the
same archive and index. Linux file residency was sampled with `fincore`
over the staged plane, dictionary, and index files before each run. Every
sweep started with a fresh CUDA process and verified all 43 queries and
their replays.

| Run | Linux-resident input before run (GiB) | First (s) | First-use part opens (s) | Warm (s) |
|---|---:|---:|---:|---:|
| Original prototype `b007769` | — | 10.166 | 6.641 | 2.411 |
| 0.9.0 full panel | — | 41.656 | 36.152 | 2.874 |
| 0.9.0 recheck 1 | 0.00 | 14.825 | 9.306 | 2.462 |
| Prototype recheck 2 | 23.09 | 11.373 | 7.968 | 2.633 |
| 0.9.0 recheck 3 | 23.09 | 9.620 | 6.593 | 2.519 |
| Prototype recheck 4 | 23.09 | 9.225 | 6.050 | 2.673 |

With 23.09 GiB of the input files resident in Linux, the release measured
9.620 s first, within the prototype's 9.225–11.373 s range in the same
sequence. Linux residency describes one layer of caching. The first
sweep also depends on storage and device initialization.


## Referrer index control

The persisted index is query-specific preparation. Its 394 MB contain a
projection for the registered referrer regex and exact counts per visible
dictionary code. The index was prepared outside the query timers. The
23.09 GiB cached-input recheck included 0.37 GiB of resident index pages.
Each fresh process began with empty in-memory query and device state.
Earlier executions within a process could warm state for later queries.

The control reopened the same packed-text archive in a fresh CUDA process
on AC with the index disabled, enabled, and disabled again. Each sweep
verified all 43 queries and their replays. The first index-disabled sweep
brought additional archive pages into memory.

| Referrer index | Archive pages resident before run (GiB) | Full first (s) | Full warm (s) | Q29 first (ms) | Q29 warm (ms) |
|---|---:|---:|---:|---:|---:|
| Disabled | 22.72 | 10.783 | 2.599 | 1159.8 | 38.18 |
| Enabled | 24.77 | 9.409 | 2.587 | 89.5 | 1.50 |
| Disabled, repeat † | 24.77 | 12.540 | 3.744 | 868.3 | 70.17 |

† A separate learning job was active during the last repeat.

Q29 built its regex projection during execution with the index disabled.
With the index enabled it used the persisted mapping and code counts and
scanned zero fact rows.


## Host, device, and storage profile

Four separate resource sweeps verified all 43 queries on the same 0.9.0
engine. Peak RSS is the Linux process high-water mark. The anonymous and
file-backed split comes from the highest sampled RSS. Device memory is the
maximum NVML reading at 50 ms sampling, less the 326 MiB idle baseline.
The RTX 5090 Laptop GPU has 23.89 GiB of VRAM.

| Engine | Referrer index | Peak RSS (GiB) | Anonymous / file-backed at sampled RSS peak (GiB) | Peak VRAM above idle (GiB) | Archive pages resident before run (GiB) | WSL block reads (GiB) |
|---|---|---:|---:|---:|---:|---:|
| Torbles CPU | Enabled | 16.94 | 5.51 / 11.41 | 0.00 | 22.72 | 0.06 |
| Torbles CPU | Disabled | 20.20 | 6.42 / 13.75 | 0.00 | 24.78 | 0.00 |
| Torbles CUDA | Enabled | 16.75 | 1.74 / 14.93 | 12.26 | 0.00 | 23.21 |
| Torbles CUDA | Disabled | 18.87 | 1.73 / 17.06 | 11.97 | 22.75 | 2.03 |

File-backed RSS overlaps the Linux page cache. WSL block reads cover the
guest virtual disk across processes and depend on the starting residency.
Whole-device writes were about 13–27 MiB per sweep, including log and
receipt output.
The first CUDA run read 23.21 GiB from that device with no archive pages
resident in Linux. The index-disabled CUDA run added 2.03 GiB of resident
pages from the stored `Referer` dictionary blob. The runner used the
read-only archive, and the GPU held its working data in VRAM.

The staged Torbles archive occupies 57,290,138,251 apparent bytes
(53.36 GiB), shared by both backends. The referrer index adds 394,417,380
bytes (0.37 GiB), for 53.72 GiB total. Most of the archive is 39.32 GB of
main bitplanes, 6.37 GB of wide-value side planes, and 11.55 GB of
dictionary arrays.

The table below records the query totals observed during resource sampling.
The sampler was active throughout these sweeps. Each run used the engine
source from the full-panel build and a fresh process.

| Torbles backend | Referrer index | First (s) | Warm (s) |
|---|---|---:|---:|
| CUDA | Enabled | 13.546 | 2.377 |
| CUDA | Disabled | 10.290 | 2.626 |
| CPU | Enabled | 17.415 | 15.665 |
| CPU | Disabled | 33.549 | 17.518 |

### Comparator resource receipts

These peaks come from the full-panel runs. Each host column identifies the
scope of its measurement. Server cgroups include charged file cache.
Disk sizes are apparent sizes for the indicated storage.

| Configuration | Host peak (GiB) | Host scope | Storage (GiB) | Storage scope |
|---|---:|---|---:|---|
| DuckDB native | 17.72 | Process RSS | 17.44 | Native database |
| DuckDB Parquet | 8.90 | Process RSS | 12.77 | Shared canonical Parquet |
| Hyper | 23.94 | Server RSS | 18.08 | Native extract |
| DataFusion | 21.74 | Process RSS | 12.77 | Shared canonical Parquet |
| chDB | 9.03 | Process RSS | 12.12 | Active MergeTree table |
| Sirius | 5.85 | Process RSS | 12.77 | Shared canonical Parquet |
| Umbra | 27.46 | Server cgroup | 6.65 | Native store and metadata |
| CedarDB | 25.59 | Server cgroup | 7.53 | Native store and metadata |
| StarRocks | 32.92 | Server cgroup | 33.32 | Backend storage and frontend metadata |

Sirius recorded a 15.93 GiB GPU allocator high-water mark. CedarDB's
allocated disk usage was 8.78 GiB with its preallocated files.
