# ClickBench on official subsets

Torbles ran the 43 official ClickBench queries on the first one and first ten
Parquet partitions, retaining all 105 columns. The inputs had 1 million and
10 million rows. These local in-memory runs use a different protocol from the
public 100M-row leaderboard. [Pinned ClickBench revision](https://github.com/ClickHouse/ClickBench/tree/072ed80e09fde25c7a11ccdd407c5731aaee9bc7)

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
