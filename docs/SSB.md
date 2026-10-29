# Official Star Schema Benchmark

The official SSB (O'Neil, O'Neil, Chen, Revilak, spec rev. 3, 2009): data from the official generator
(`ssb-dbgen`, github.com/electrum/ssb-dbgen), and the 13 official queries verbatim in their classic comma-join form
(`ssb_common.py`). DuckDB runs the queries as written on the official schema (all columns, strings as VARCHAR).
torbles runs the same queries expressed in `algq`, on the same rows in the same (generated) order, and every result
is compared with DuckDB's, row by row in ORDER BY order.

torbles has three key designs, run separately:
- **official**: the keys as generated (dense surrogate keys for customer, supplier and part; dates as yyyymmdd)
- **coordinate**: the keys re-encoded at load as coordinates of the dimension hierarchy, an order-preserving
  bijection: customer and supplier `region | nation | city | seq`, part `mfgr | category | brand | seq`,
  date `year | month | day`. The fact's foreign keys are rewritten through the same map; this is part of the load.
- **derived**: the official keys kept, with an algebraically derived surrogate stored in the fact. Customer,
  supplier and part: `s = (region|nation|city or mfgr|category|brand) << w | official key`; the official key is the
  low w bits (a projection) and the dimensions stay indexed by it. Date: `s = year|month|day`, computed from yyyymmdd
  by arithmetic, with `official = 10000 (1992 + y) + 100 m + d`. After loading, the official key of every fact row is
  recovered from the stored planes and compared with the source (all 60M rows equal at SF10).

Besides the 13 SSB queries, K1-K3 (not part of SSB) filter on the official keys themselves: a customer-key range, an
orderdate range on the fact column, and a supplier-key point lookup.

Strings are dictionary-encoded, order-preserving, so `=`, `IN` and `BETWEEN` on strings keep their meaning; group
keys are decoded back to strings for the comparison. Each engine runs alone in its own process (first run and best of
3, 1 and 2 threads). SQLite is not included at these sizes.

Building the generator (it needs FORTIFY off on current glibc; the data do not change):

    git clone https://github.com/electrum/ssb-dbgen && cd ssb-dbgen
    make CFLAGS='-O2 -fcommon -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DDBNAME=\"dss\" -DLINUX -DDB2 -DSSBM'

Running (data under $SSB_DATA, default ~/torbles_data/ssb; generator under $SSB_DBGEN, default ~/torbles_data/ssb-dbgen), from bench/ssb:

    python3 ssb_common.py 10                 # generate SF10 and build the DuckDB database from the .tbl files
    python3 ssb_duck.py 10                   # DuckDB: timings + reference results
    python3 ssb_torbles.py 10 official        # torbles on the official keys, checked against DuckDB
    python3 ssb_torbles.py 10 coordinate      # torbles on coordinate keys, checked against DuckDB
    python3 ssb_torbles.py 10 derived         # torbles on official keys + derived surrogates, checked against DuckDB

## Results, SF10 (59,986,214 lineorder rows), best of 3, ms (2 threads unless noted)

2-core VM, 7 GiB RAM, DuckDB 2.0.0 in its own process (memory limit 5 GB), torbles in its own process.
All results of all three torbles designs equal DuckDB's, row by row in ORDER BY order. Run-to-run variation on this
VM is roughly 20-50% for single queries (DuckDB's Q1.1 measured 199 ms in one run and 322 ms in this one); read the
totals rather than single rows.

| query | DuckDB | torbles, official keys | torbles, coordinate keys | torbles, derived surrogate |
|---|---|---|---|---|
| Q1.1 | 34.1 | 18.7 | 88 | 84 |
| Q1.2 | 23.4 | 13.5 | 54 | 42 |
| Q1.3 | 18.0 | 11.7 | 81 | 64 |
| Q2.1 | 42.7 | 26.3 | 77 | 58 |
| Q2.2 | 37.3 | 23.4 | 83 | 67 |
| Q2.3 | 27.5 | 17.3 | 39 | 39 |
| Q3.1 | 60.0 | 36.8 | 116 | 115 |
| Q3.2 | 35.7 | 23.3 | 99 | 111 |
| Q3.3 | 25.1 | 20.1 | 79 | 78 |
| Q3.4 | 18.6 | 13.9 | 82 | 67 |
| Q4.1 | 59.8 | 36.5 | 97 | 139 |
| Q4.2 | 60.4 | 40.2 | 80 | 94 |
| Q4.3 | 35.4 | 29.2 | 66 | 71 |
| K1 | 147 | 92 | 312 | 108 |
| K2 | 134 | 110 | 43 | 38 |
| K3 | 96 | 33 | 32 | 38 |
| total 13 SSB queries, 1 thread | 6679 | 8734 | 2034 | 1646 |
| total K1-K3, 1 thread | 639 | 395 | 700 | 281 |
| total 13 SSB queries, 2 threads | 3656 | 4700 | 1043 | 1029 |
| total K1-K3, 2 threads | 376 | 235 | 387 | 184 |

- official keys: date filters are intervals of yyyymmdd and run as comparators (Q1 wins), but the surrogate keys of
  customer, supplier and part carry no structure, so those joins are lookups and torbles is ~2x slower than DuckDB's
  hash joins on them. Filters on the keys themselves (K1-K3) are intervals and fast.
- coordinate keys: every hierarchy filter is a few cubes or comparators on the fk planes, but the official customer
  key is now an arbitrary attribute of the new id, so a key range (K1) becomes a lookup.
- derived surrogate: both at once. Hierarchy filters are cubes on the high planes, official-key filters are
  comparators or cubes on the low planes, and nothing needs a lookup. Cost: foreign keys take 98 bits per fact row
  (72 coordinate, 79 official); part's surrogate uses exactly 32 bits at SF10.
- storage: the torble lineorder (9 columns, bit planes with their integrity planes) is 2.6-3.0 GiB; DuckDB's whole
  database file (all 17 lineorder columns and the dimensions, compressed) is 1.5 GB. torbles does not compress beyond
  skipping planes that are zero.
- peak memory of the torbles process: 5.3-5.7 GiB, during the build (the input columns and the planes coexist).
- load: reading the columns (and re-encoding / deriving keys) 9-16 s, building the torble 14-16 s at SF10.

## GPU (RTX 5090 Laptop GPU, WSL2, 24 cores), SF10, best of 3, ms

Same 16 queries, derived-surrogate keys, every result equal to DuckDB's (`run_gpu.sh` in the GPU package).

| | 13 SSB queries | K1-K3 |
|---|---|---|
| DuckDB, 1 thread | 2946 | 270 |
| DuckDB, 24 threads | 315 | 29 |
| torbles CPU, 1 thread | 750 | 147 |
| torbles CPU, 24 threads | 142 | 23 |
| torbles GPU, end to end | 114 | 5 |
| torbles GPU, kernels only | ~40 | ~1 |

Most GPU time is host work (Python planning, dimension filters, and the dense group arrays, up to 64 MB per query,
allocated, cleared and copied back); the math itself is about a third.

## Together (`ssb_together.py`): derived surrogates + moved customers + a wide measure

Customers move to another city (DuckDB: UPDATE customer); torbles records the moves and rewrites nothing. A wide
measure, lo_ordtotalprice * 100, exceeds 32 bits for 0.12% of rows and is stored self-extending. Every result equals
DuckDB's after the UPDATE. SF1 here (2 threads): with 100 moved customers (18k fact rows, 0.3%) queries cost 1.2-2.8x
a compacted fact; with 2,000 (388k rows, 6.5%) 1.5-12x, which is when compaction pays.

SF10 on the RTX 5090 laptop (on battery, so absolute times are throttled; compare within the run): every result
equals DuckDB's after the UPDATE, on the CPU and the GPU. Recording a move takes 0.2-1.1 ms (the fact is untouched).
100 moved customers (19k fact rows, 0.03%): deltas cost 1.2-2.3x a compacted fact on the CPU. 2,000 (400k rows, 0.7%):
2-10x, where compacting pays. On the GPU path, pass B (the delta rows, read by lookup) runs on the CPU. The compaction
time shown (17-62 s) rebuilds the whole fact table; rewriting only the foreign-key column would be much cheaper.

## GROUP BY as one matrix product (`run_mm.sh`), RTX 5090 Laptop GPU + Core Ultra 9 290HX (24 cores, AVX2), SF10

GROUP BY / SUM as C = M^T X (AND-popcount product of group masks and measure planes, see `alg/README.md`), on the CPU
(AVX2; the VPOPCNTDQ kernel is untested, neither machine has it), on the tensor cores (`mma.sync.m16n8k256 .b1
.and.popc`, CUDA 13.3, self-test passed on compute 12.0) and with `__popc`. Same session as DuckDB, on AC power. Every
result of every engine equals DuckDB's. M1-M3 are not SSB: unfiltered GROUP BYs over all 60M rows (35, 125, 77 groups;
M3 has a bilinear measure, 96 plane pairs). Full tables: `results/gpu_rtx5090_mm/mm_report_sf10.txt`.

| ms, best of 3 | DuckDB 24T | CPU split 24T | CPU matrix 24T | GPU matrix, tensor cores | GPU matrix, `__popc` |
|---|---|---|---|---|---|
| 13 SSB queries, end to end | 243 | 118 | 103 | 64.5 | 51.9 |
| 13 SSB queries, kernels | | | | 38.0 | 26.4 |
| M1 (35 groups), kernel | 63 e2e | 40 e2e | 17 e2e | 8.0 | 11.4 |
| M2 (125 groups), kernel | 74 e2e | 56 e2e | 77 e2e | 41.9 | 57.5 |
| M3 (77 groups, bilinear), kernel | 25 e2e | 48 e2e | 89 e2e | 77.9 | 131.8 |

- End to end on the GPU, the sparse output removes the dense group arrays: 13 SSB queries in 52 ms (earlier run with the
  per-word kernel and dense arrays: 114 ms), with the engine call within ~0.5 ms of the kernel.
- Tensor cores vs `__popc` on the same product: 1.4-1.7x faster where M is dense (M1-M3), slower on the selective SSB
  queries (the `__popc` version skips a group's empty words; the MMA works in 256-row k-steps for 16 groups at once).
- The previous per-word GPU kernel (one thread per word, atomics into dense arrays) ran M1-M3 in 2.4 / 5.9 / 8.4 ms:
  faster than both tile kernels, which are latency-bound (64-128 threads per block, shared-memory tiles, and M2's 125
  groups overflow the 64-slot tile table). Its SSB timings in this run are not comparable: it ran first after the
  CPU runs and Q1.1, whose plan and kernel are identical in all three GPU modes, took 13.8 ms there vs 0.38 ms later
  (GPU clocks still ramping).
- CPU: the tile product is 2.3-2.9x faster on M1 (few groups, many rows each), about even on SSB, slower on M2/M3:
  dense AND-popcount work is G x B / 64 word operations per row, and AVX2 popcounts are no faster than scalar POPCNT.

## Native, parallel ingestion and query preparation (`run_native.sh`), same machine, SF10, on AC

Ingestion on 24 threads: dimensions 0.26 s, fact columns read from DuckDB 3.4 s, foreign keys re-encoded 0.2 s,
torble build 1.15 s (was 14-16 s single-threaded), GPU upload 0.3 s: about 5 s in all, from 25-30 s. Queries: a
persistent worker pool (no thread creation per call), compiled plans reused when a query runs again on the same data,
join polynomials planned in C, results decoded in bulk; Python adds ~0.1 ms per repeated query. The GPU was warmed up
before timing. Every result equals DuckDB's. Full tables: `results/gpu_rtx5090_native/native_report_sf10.txt`.

| 13 SSB queries, ms | DuckDB 24T | CPU 24T (matrix) | GPU per-word kernel | GPU matrix, tensor cores | GPU matrix, `__popc` |
|---|---|---|---|---|---|
| end to end, best of 3 | 240.6 | 77.7 | 41.3 | 43.5 | 32.6 |
| end to end, first run (planned from scratch) | 257.1 | 86.0 | 44.1 | 45.9 | 33.6 |
| GPU kernels | | | 4.9 | 37.4 | 26.3 |

- With a warm GPU, the per-word kernel (one thread per 64-row word, atomics) runs the 13 queries' math in 4.9 ms,
  5-7x less than either tile kernel; its end-to-end time is held back only by its dense group arrays (~5 ms each on
  Q3.2-Q3.4 and Q4.3). The tile kernels are latency-bound (few warps per SM, shared-memory tiles). Next step: the
  per-word kernel writing into the sparse group table.
- M1-M3 (unfiltered, 60M rows): per-word GPU kernel 18.7 ms in all vs DuckDB 24T 171 ms.


## SF10 validation (`wsl/run_validation.sh`), same machine, on AC

This run was made while the release was numbered 1.0.0-rc1 (with the script then named `run_rc1.sh`), hence the
file names in `results/rc1_validation/`. The code it measured is this release, 0.1.0.

The GPU default is now the per-word kernel writing into a sparse group table on the GPU. A slot's key is read before
it is claimed with a CAS. The groups present are gathered from an insertion list, and their slots are cleared, so
the table stays empty between queries. Every test passed on the GPU, and every result equals DuckDB's. Report:
`results/rc1_validation/rc1_report_sf10.txt`.

| ms, best of 3 | DuckDB 1T | DuckDB 24T | CPU 1T | CPU 24T | GPU per-word, sparse (default) | GPU per-word, dense | GPU tile, `__popc` |
|---|---|---|---|---|---|---|---|
| 13 SSB, end to end | 2708.1 | 246.5 | 887.9 | 83.2 | 10.3 | 39.4 | 31.4 |
| 13 SSB, first run | 4390.2 | 277.1 | 2789.1 | 89.7 | 11.0 | 43.2 | 41.7 |
| 13 SSB, GPU kernels | | | | | 5.98 | 4.97 | 26.28 |
| K1-K3, end to end | 247.0 | 21.2 | 151.3 | 17.2 | 1.7 | 1.3 | 2.5 |
| M1-M3, end to end | 2386.5 | 159.8 | 2471.8 | 181.5 | 21.3 | 19.3 | 200.7 |

Ingestion on 24 threads takes about 4.6 s, plus 0.3 s for the GPU upload:

| phase | seconds |
|---|---|
| dimensions | 0.30 |
| fact columns read from DuckDB | 3.23 |
| keys re-encoded | 0.08 |
| torble build | 0.99 |
| GPU upload | 0.29 |


## Scaling, SF1 to SF50 (`wsl/run_scaling.sh`), same machine

DuckDB runs on an in-memory copy of its tables (the 9 fact columns the queries read), with its default memory limit.
Both engines answer from memory, and first runs are within 13% of the best of 3. The runs above (the SF10 validation and
earlier) kept DuckDB on its database file under a 5 GB memory limit; their best-of-3 numbers are warm, but their
first runs include loading columns into DuckDB's buffer cache. Tables and the graphic: README.md and
`results/scaling/`; the superseded file-backed run: `results/scaling_duckdb_filebacked/`.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).


## Automatic keys (`ssb_auto.py`, 0.2.0) and CPU performance, SF1, 2 vCPUs

The official DuckDB database, loaded with `torbles.from_duckdb(db, schema="schema.sql", columns={"lineorder": the 9
columns the queries read})`: keys from the standard DDL, layout from the key design, queries written with the data's
values. Every result equals DuckDB's answer to the official SQL. Both engines in memory (`SSB_DUCK_INMEM=1`). Four
interleaved rounds of every configuration (the VM varies by up to 25% between identical runs); per query, the best
time over the rounds (each round itself best of 3); ms. Only queries answered correctly are timed: `ssb_auto.py` leaves a
query that is refused, fails or differs from DuckDB out of the totals and reports it (none did).

| query | DuckDB 1T | DuckDB 2T | 0.1.0 hand 1T | 0.2.0 hand 1T | 0.2.0 auto 1T | 0.1.0 hand 2T | 0.2.0 hand 2T | 0.2.0 auto 2T |
|---|---|---|---|---|---|---|---|---|
| Q1.1 | 34.1 | 18.7 | 16.2 | 16.0 | 16.0 | 9.2 | 8.8 | 9.2 |
| Q1.2 | 23.4 | 13.5 | 8.4 | 7.8 | 8.1 | 5.3 | 4.9 | 4.8 |
| Q1.3 | 18.0 | 11.7 | 11.2 | 10.7 | 15.5 | 6.7 | 6.9 | 10.5 |
| Q2.1 | 42.7 | 26.3 | 13.0 | 12.9 | 15.1 | 8.6 | 8.6 | 9.9 |
| Q2.2 | 37.3 | 23.4 | 12.5 | 12.9 | 12.9 | 7.4 | 7.4 | 8.2 |
| Q2.3 | 27.5 | 17.3 | 7.1 | 6.9 | 6.9 | 4.2 | 4.1 | 4.5 |
| Q3.1 | 60.0 | 36.8 | 28.3 | 28.1 | 28.1 | 16.1 | 15.3 | 15.9 |
| Q3.2 | 35.7 | 23.3 | 14.8 | 14.1 | 15.6 | 8.4 | 8.5 | 8.8 |
| Q3.3 | 25.1 | 20.1 | 12.4 | 11.8 | 11.9 | 7.0 | 6.7 | 7.3 |
| Q3.4 | 18.6 | 13.9 | 11.8 | 11.2 | 10.9 | 6.5 | 6.2 | 6.5 |
| Q4.1 | 59.8 | 36.5 | 20.6 | 21.6 | 29.5 | 15.3 | 15.2 | 15.1 |
| Q4.2 | 60.4 | 40.2 | 20.4 | 19.9 | 22.7 | 11.1 | 11.5 | 12.7 |
| Q4.3 | 35.4 | 29.2 | 12.6 | 12.0 | 12.1 | 6.9 | 7.0 | 9.0 |
| M1 | 163.7 | 92.0 | 47.5 | 50.5 | 46.4 | 25.5 | 24.4 | 28.3 |
| M2 | 181.9 | 99.4 | 214.8 | 210.0 | 232.8 | 109.4 | 112.6 | 112.1 |
| M3 | 68.4 | 36.9 | 244.6 | 240.3 | 264.0 | 124.4 | 134.4 | 129.2 |
| **13 SSB queries** | **478.0** | **310.8** | **189.3** | **185.8** | **205.2** | **112.7** | **111.2** | **122.4** |
| **M1-M3** | **414.1** | **228.3** | **507.0** | **500.8** | **543.3** | **259.4** | **271.5** | **269.6** |

Key designs found: customer and supplier region | nation | city | row; part mfgr | category | brand | row; date
selling season | month | year-month | row. Loading: 2.4 s (DuckDB read, types, key design, build) against 1.3 s for
the hand-written loader, which reads fewer dimension columns.


### Bridges for the dense GROUP BY: ablation (`ablation.sh`), SF1, 2 vCPUs

The table above is 0.2.0 before the bridges (docs/ENGINE.md, "Dense GROUP BY"): M2 and M3 were slower than DuckDB.
Each bridge is behind an environment switch; `bench/ssb/ablation.sh` runs `ssb_auto.py` under every configuration in
three interleaved rounds (with `BASELINE=` a checkout of the committed 0.2.0 as the first), and
`ablation_table.py` keeps each query's best time over the rounds, counting a run only when its answer equals
DuckDB's. DuckDB: best of four rounds of `ssb_duck.py`, in memory. ms:

| configuration | 13 SSB 1T | 13 SSB 2T | M1 1T | M2 1T | M3 1T | M1-M3 1T | M1-M3 2T | answers |
|---|---|---|---|---|---|---|---|---|
| DuckDB (in memory) | 478.0 | 310.8 | 163.7 | 181.9 | 68.4 | 414.1 | 228.3 | reference |
| 0.2.0 before the bridges | 208.2 | 116.7 | 49.7 | 211.7 | 244.0 | 505.4 | 269.5 | all equal DuckDB |
| + vectorized transpose (bridges off) | 212.2 | 117.5 | 48.1 | 219.1 | 257.3 | 524.5 | 268.5 | all equal DuckDB |
| + factor through the key | 211.6 | 119.7 | 48.5 | 212.4 | 115.7 | 376.6 | 199.8 | all equal DuckDB |
| + splitting tree, direct slots | 240.4 | 126.2 | 64.4 | 245.1 | 282.5 | 592.0 | 299.3 | all equal DuckDB |
| + direct slots only | 206.4 | 118.5 | 46.7 | 221.2 | 246.3 | 514.2 | 267.3 | all equal DuckDB |
| + row kernel (cost-chosen) | 214.7 | 115.9 | 48.8 | 75.6 | 62.0 | 186.5 | 93.9 | all equal DuckDB |
| + Möbius (cost-chosen) | 209.5 | 115.0 | 48.8 | 213.3 | 249.3 | 511.4 | 270.9 | all equal DuckDB |
| + factor + row kernel | 209.2 | 122.1 | 49.2 | 72.7 | 61.1 | 183.0 | 95.2 | all equal DuckDB |
| + factor + Möbius | 217.1 | 120.6 | 48.3 | 214.6 | 112.9 | 375.7 | 204.2 | all equal DuckDB |
| + factor + row kernel + Möbius | 213.0 | 120.3 | 48.5 | 70.2 | 64.6 | 183.3 | 93.7 | all equal DuckDB |
| factor + row kernel on every tile | 254.1 | 149.3 | 59.1 | 69.7 | 60.3 | 189.0 | 100.9 | all equal DuckDB |
| factor + Möbius on every tile | 382.3 | 207.4 | 48.9 | 320.9 | 96.9 | 466.7 | 240.1 | all equal DuckDB |
| factor + row kernel, no margin (model as is) | 219.2 | 128.9 | 59.8 | 73.4 | 61.7 | 194.9 | 102.7 | all equal DuckDB |
| defaults: factor + row kernel at margin 0.5 | 213.1 | 114.8 | 48.2 | 71.1 | 59.9 | 179.2 | 96.1 | all equal DuckDB |

Per query, 1 thread (the configurations that matter):

| query (1 thread) | DuckDB | 0.2.0 before the bridges | + vectorized transpose (bridges off) | + factor through the key | factor + row kernel, no margin (model as is) | defaults: factor + row kernel at margin 0.5 |
|---|---|---|---|---|---|---|
| Q1.1 | 34.1 | 16.6 | 16.3 | 17.0 | 16.6 | 16.2 |
| Q1.2 | 23.4 | 7.8 | 7.8 | 8.5 | 8.0 | 8.2 |
| Q1.3 | 18.0 | 17.1 | 16.3 | 15.6 | 15.4 | 15.6 |
| Q2.1 | 42.7 | 13.8 | 16.3 | 14.8 | 16.0 | 16.2 |
| Q2.2 | 37.3 | 12.8 | 13.1 | 13.1 | 13.3 | 13.2 |
| Q2.3 | 27.5 | 7.1 | 7.5 | 7.6 | 7.8 | 7.6 |
| Q3.1 | 60.0 | 27.4 | 28.5 | 27.6 | 33.2 | 28.4 |
| Q3.2 | 35.7 | 14.5 | 15.0 | 14.8 | 15.1 | 15.8 |
| Q3.3 | 25.1 | 11.9 | 12.1 | 12.5 | 12.3 | 13.1 |
| Q3.4 | 18.6 | 11.0 | 11.7 | 11.3 | 11.6 | 11.3 |
| Q4.1 | 59.8 | 32.4 | 31.2 | 33.1 | 31.6 | 31.7 |
| Q4.2 | 60.4 | 23.5 | 23.3 | 22.6 | 24.9 | 22.8 |
| Q4.3 | 35.4 | 12.3 | 13.1 | 13.0 | 13.3 | 13.0 |
| M1 | 163.7 | 49.7 | 48.1 | 48.5 | 59.8 | 48.2 |
| M2 | 181.9 | 211.7 | 219.1 | 212.4 | 73.4 | 71.1 |
| M3 | 68.4 | 244.0 | 257.3 | 115.7 | 61.7 | 59.9 |

- **Factor through the key** halves M3 by itself (244 -> 116 ms): its product's second factor is a GROUP BY key.
- **The row kernel** wins where every group is in every tile (M2 212 -> 71 ms, M3 -> 60) and loses elsewhere when
  the cost model chooses it without a margin (M1 60 against 48, Q3.1 33 against 28): with the margin of 0.5 it is
  chosen only for M2 and M3 here.
- **The splitting tree** is slower (13 queries 240 ms against 212), **direct slots** neutral, **Moebius** no gain
  when chosen by cost and much slower when forced (382 ms): all three are off by default and kept for the record.
- **Defaults** (factor + row kernel at margin 0.5): the 13 queries unchanged within noise (213 / 115 ms at 1 / 2
  threads against 208 / 117), M1-M3 179 / 96 ms against 505 / 270 before and 414 / 228 for DuckDB.

## SF10 validation of 0.2.0 (`wsl/run_validation.sh 10`), the same laptop

Every test passed (the GPU tests on the RTX 5090, CUDA 12.4) and every result equals DuckDB's, with the hand-designed
keys and with the automatic ones. Best of 3, ms; the 0.1.0 columns are the SF10 validation above
(`results/rc1_validation/`), DuckDB in each run as measured then:

| query | DuckDB 24T, 0.2.0 run | CPU 24T, 0.1.0 | CPU 24T, 0.2.0 | GPU, 0.1.0 | GPU, 0.2.0 | CPU 1T, 0.1.0 | CPU 1T, 0.2.0 | DuckDB 1T, 0.2.0 run |
|---|---|---|---|---|---|---|---|---|
| M1 | 51.8 | 17.5 | 19.0 | 2.4 | 2.3 | 226.4 | 240.0 | 744.1 |
| M2 | 68.3 | 80.0 | 22.9 | 6.5 | 6.2 | 1028.4 | 341.2 | 988.4 |
| M3 | 19.4 | 84.1 | 17.8 | 12.4 | 5.2 | 1217.0 | 290.0 | 269.4 |
| **13 SSB queries** | **223.4** | **83.2** | **79.4** | **10.3** | **10.7** | **887.9** | **892.2** | **2164.6** |
| **M1-M3** | **139.4** | **181.5** | **59.7** | **21.3** | **13.7** | **2471.8** | **871.3** | **2001.9** |

On one thread M3 is still a little slower than DuckDB on this machine (290 against 269 ms), while on the 2-vCPU VM at
SF1 it is faster (59.9 against 68.4). With the automatic keys
(`auto_keys_sf10.txt`): loaded in 7.5 s; the 13 queries in 80.1 ms on 24 CPU threads and 9.1 ms on the GPU; M1-M3
17.7 / 21.8 / 17.1 ms on the CPU and 2.0 / 6.0 / 4.5 ms on the GPU. Files: `results/validation_sf10_rtx5090_0.2.0/`.

