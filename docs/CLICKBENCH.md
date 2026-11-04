# ClickBench on official subsets

The [0.8.0 measurements](CLICKBENCH-0.8.0.md) cover the complete 100M-row fixture.

Torbles ran the 43 official ClickBench queries on the first one and first ten
Parquet partitions, retaining all 105 columns. The inputs had 1 million and
10 million rows. These local warm runs use a different protocol from the
public 100M-row leaderboard. [Pinned ClickBench revision](https://github.com/ClickHouse/ClickBench/tree/072ed80e09fde25c7a11ccdd407c5731aaee9bc7)


## Tagged 0.7.0 rerun

Torbles was built from tag `v0.7.0` at
`3fc21a35b5cf20d56e4de7088a7d74ce6739d6da`, with automatic interchanges.
All engines ran sequentially on WSL2 with an Intel Core Ultra 9 290HX Plus,
24 CPU threads, and an RTX 5090 Laptop GPU with 23.9 GiB of device memory.
The CUDA build used toolkit 13.3.73 and driver 610.74.
Each engine ran all 43 queries on the same canonical 1M and 10M row
inputs. Setup and loading are outside the query timer. The reported totals
sum the median of the first three warm executions per query after an initial
execution. Timings include fetching results into Python.

| Rows | Torbles CUDA | Torbles CPU | DuckDB | Umbra | Sirius |
|---:|---:|---:|---:|---:|---:|
| 1M | 64.5 ms | 125.8 ms | 247.0 ms | 88.2 ms | 514.0 ms |
| 10M | 382.6 ms | 879.7 ms | 1,220.8 ms | 413.1 ms | 2,470.3 ms |

The initial answers were checked against fresh DuckDB 2.0.0 runs, allowing
SQL-defined ties and floating-point tolerance. An untimed CUDA replay
independently verified every Torbles result and recorded successful CUDA
dispatch for 43 of 43 queries at each scale. Seven 1M replays also called
an instrumented CPU query entry point. Host SQL binding and result fetching
are included in the wall times.

Sirius `76ef37d231` uses DuckDB 2.0.0 and GPU-pinned Parquet.
It used CPU fallback for Q29. Umbra 26.10 used a local PostgreSQL connection,
so its result-fetch time includes loopback transport. Two Q24 timestamp
fields were normalized to epoch seconds for comparison, and the Q29 regex
was adjusted so the dot excludes line feeds, matching DuckDB semantics.
chDB was 4.4.0 with ClickHouse 26.9.2.1.

### Matched six-engine comparison

These totals use the same 41 verified queries across all six engines.
chDB's Q4 numerical result and Q29 regex groups did not verify.

| Rows | Torbles CUDA | Torbles CPU | DuckDB | chDB | Umbra | Sirius |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 60.0 ms | 117.5 ms | 183.7 ms | 291.4 ms | 82.3 ms | 434.5 ms |
| 10M | 329.8 ms | 811.5 ms | 831.6 ms | 1,891.2 ms | 358.7 ms | 2,011.8 ms |

[Per-query medians and verification outcomes](CLICKBENCH-0.7.0.csv)

## 0.7.0 automatic interchanges

All 43 official queries verified at 1M and 10M rows. The table sums warm
medians from interleaved runs on the same loaded table. DuckDB is the
previously verified reference on the same input.

| Rows | Torbles CPU off → auto | Torbles CUDA off → auto | DuckDB |
|---:|---:|---:|---:|
| 1M | 116.2 → 116.2 ms | 76.0 → 66.3 ms | 229.2 ms |
| 10M | 892.7 → 900.2 ms | 404.0 → 360.3 ms | 1,155.1 ms |

At 10M, CUDA Q9 fell from 14.97 to 10.50 ms, Q10 from 20.71 to 7.14 ms,
and Q33 from 31.34 to 7.19 ms. Q33 sent 191,352 of 10M rows to full-key
reduction after proving the remaining keys were singletons. The run used
`fedfcc3`, an RTX 5090 Laptop GPU, and 24 CPU threads.

## Read-only SQL

| Fact rows | Torbles GPU | Prior Torbles CPU | DuckDB | CUDA query ranges | CUDA kernel launches |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 72.603 ms | 114.739 ms | 229.214 ms | 43 of 43 | 299 |
| 10,000,000 | 420.168 ms | 859.185 ms | 1,155.084 ms | 43 of 43 | 322 |

The GPU results use commit `594b64875277d89c7ca8997f7426d9e7d3987ba6`.
The prior Torbles CPU run used `00d625c590eed0c20f1085a8cc950dc957341b05`
and the DuckDB run used DuckDB 2.0.0. The runs used the same canonical inputs
and WSL2 host with an Intel Core Ultra 9 290HX Plus, up to 24 CPU threads, and
an NVIDIA GeForce RTX 5090 Laptop GPU with 23.9 GiB and compute capability 12.0.

The canonical input decodes binary text as UTF-8 and converts the source's Unix
seconds and day counts to SQL timestamps and dates. Tables were loaded before
timing. The wall timer includes SQL execution and fetching Python rows. Each
query ran once, followed by three warm repetitions. The table sums the warm
medians. The first timed answer, an untimed dispatch replay, and the profiled
answer each matched DuckDB on all 43 queries, allowing SQL-defined ties and
floating-point tolerance.

A separate warmed Nsight Systems capture placed each query inside an NVTX
range and counted the kernels launched in that range. The independent dispatch
replay recorded 65 successful CUDA calls at 1M and 72 at 10M. At 1M, seven
queries also made CPU query calls through the instrumented entry points. At 10M
the replay observed zero calls through those CPU entry points. SQL binding,
preprocessing, and result materialization also use the host.

### Timing variation and startup

A preceding GPU run with the same native library took 72.368 ms at 1M and
1,163.861 ms at 10M. The difference at 10M was concentrated in distinct
queries such as Q5, Q9 and Q10, whose CUDA routes matched between runs.
Independent profiled query ranges for those queries were consistent across
the two revisions. The cause of the uninstrumented variation remains
undetermined. The table reports the final run.

GPU setup took 7.38 s and 54.58 s, including construction of the in-memory
Torbles tables. Setup sits outside the query totals. At 10M, Q21's first
execution took 3.759 s and its warm median was 22.681 ms. Q29 took 1.417 s
first and 45.151 ms warm. These timings capture the cost of building reusable
text state on first use.

## Four-engine comparison

These totals use the same 41 verified queries on the same inputs. Q4 and Q29 were unverified in chDB 4.4.0 (ClickHouse 26.9.2.1).

| Fact rows | Torbles GPU | Prior Torbles CPU | DuckDB | chDB |
|---:|---:|---:|---:|---:|
| 1M | 68.4 ms | 106.6 ms | 166.4 ms | 290.9 ms |
| 10M | 374.5 ms | 792.7 ms | 804.6 ms | 1,631.0 ms |

## Experimental sector interchanges

Set TORBLES_INTERCHANGES=1 to try certified CUDA projection shortcuts for generated dimension keys.
The planner checks that a stored surrogate field and the dimension key induce the same partition before
grouping on the field. It decodes the selected keys through the certified labels. It can also check
affine XOR formulas over high-bit sectors and use the existing dimension lookup in sectors that lack
a proof. Checks cover all reachable dimension rows and cache against the source revisions.
A separate count-first switch can fix the winning groups before processing distinct keys and measures.
It admits the shortcut when the winning population is at most 75% of the scope and the resulting hash
capacity is at most half the original allocation.

## Persisted cross-query algebra

A separate benchmark projected the three columns used by official Q1–Q4
and an unseen affine aggregate. The catalog learned exact scalar moments
and the scope `AdvEngineID = 0`. After a fresh-process reopen, it answered
Q2 by subtracting that scope from the total, and answered the affine query
from saved sums and counts. Every captured result matched DuckDB.

| Rows | Ordinary Torbles | Reopened catalog | Catalog file | Explicit refresh |
|---:|---:|---:|---:|---:|
| 1M | 4.764 ms | 1.116 ms | 584 B | 3.351 ms |
| 10M | 10.013 ms | 4.921 ms | 590 B | 8.072 ms |

The query columns sum five warm medians. The catalog is additional state
alongside the saved Torbles source. The source snapshots occupied 30,086,427
and 300,647,010 bytes. A deterministic 64-row change was staged in DuckDB
and encoded as a replacement immutable snapshot. The encoding took 51.852
and 395.359 ms and is separate from refresh. The stale catalog fell back to
ordinary SQL before explicit refresh, then derived the updated answers.

The benchmark used commit `ad62fec13a7a8582895396994bfd26a64ec85ccf`.
Derivations recorded zero engine queries, with binding and source-identity
checks included in their times. [Catalog API](SQL.md#durable-learned-algebra)

## Durable transactions

The durable write-and-read workload is separate from these ClickBench
measurements. Its writer runs INSERT, UPDATE, DELETE and MERGE in one
transaction while readers issue analytical queries. It measures commit and
reader latency, concurrent throughput and reopening. The durable prototype
serializes client transactions. [Benchmark harness](../bench/benchmark_durable.py) · [Durable SQL](SQL.md#durable-sql-prototype)
