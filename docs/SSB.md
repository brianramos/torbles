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
| Q1.1 | 322 | 114 | 88 | 84 |
| Q1.2 | 199 | 101 | 54 | 42 |
| Q1.3 | 154 | 117 | 81 | 64 |
| Q2.1 | 268 | 520 | 77 | 58 |
| Q2.2 | 254 | 389 | 83 | 67 |
| Q2.3 | 204 | 340 | 39 | 39 |
| Q3.1 | 503 | 783 | 116 | 115 |
| Q3.2 | 232 | 445 | 99 | 111 |
| Q3.3 | 151 | 326 | 79 | 78 |
| Q3.4 | 151 | 86 | 82 | 67 |
| Q4.1 | 466 | 756 | 97 | 139 |
| Q4.2 | 437 | 415 | 80 | 94 |
| Q4.3 | 315 | 307 | 66 | 71 |
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
