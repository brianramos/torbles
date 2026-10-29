# Changelog

## 0.2.0 (2026-10-29), alpha

Bring your own data: install with pip, load typed tables with their keys, get answers checked against DuckDB.

Loading
- Loaders: `from_duckdb` (a database file or connection), `from_ddl` (a standard SQL schema plus CSV / Parquet /
  JSON files or data frames), `from_frictionless` (a Frictionless Data Package), `from_frames`. DuckDB reads the
  sources and does the conversions (dictionary ranks, decimal scaling, key lookups) in parallel.
- Keys from standard declarations: PRIMARY KEY / FOREIGN KEY constraints read through INFORMATION_SCHEMA, a DDL
  script (`schema=`), or Frictionless Table Schema keys; key inference is opt-in and reported.
- Typed columns (`coltypes.py`): signed and 64-bit integers, DECIMAL (exact sums, returned as `Decimal`), DATE,
  TIMESTAMP, BOOLEAN, VARCHAR (dictionary; `startswith`), DOUBLE (dictionary). Comparisons with constants become exact
  code intervals, as SQL compares them. `types=` loads a column as another type (a DOUBLE amount as DECIMAL).
- NULLs refused outside foreign keys (`fill_nulls=`); unsupported types skipped with a note; `columns=` subsets.
- Automatic key design (`keydesign.py`): functional-dependency chains found on the full data become derived
  surrogates `cell << w | row`, dense-ranked or shortened when they do not fit 32 bits; the planner is told which
  bits each attribute occupies (coordinate hints), so filters are cubes or comparators from the first query.
- `db.describe()`, `db.verify(q)` (the query's SQL on the source, compared), `q.sql()` in the source's schema.

Persistence
- `db.save(path)` / `torbles.load(path)`: one file per torble (header hashed, planes 64-byte aligned), memory-mapped
  privately on load; every torus face is checked, single-bit faults are repaired in memory, more is refused.
  Saving writes new files and renames them. Native `at_save`, `at_load`, `at_free`.

Engine
- A torble holds up to 31 columns (the torus has one row per column plus the wrap row). This also fixes an overflow
  of the face list for 15 full-width columns in 0.1.0.
- Exact sums at any width. All engine arithmetic is modular (unsigned 64-bit), so a 64-bit plan is exact whenever its
  true sums fit, judged by interval arithmetic on the measure as written. Past 2^63, 128-bit sums from the same
  popcounts: a 64-bit value is two 32-bit columns, a product of two is the 64 x 64 matrix of plane pairs (weights up
  to 2^126), folded with 128-bit coefficients in the matrix product and in mask splitting, row by row in the row
  kernel, the general path and the plain baseline (`bench/ssb/wide_sums.py`: SSB SF1, 128-bit sums by year in 76 ms
  against DuckDB's 120 in HUGEINT). Past 2^127, and for columns without an affine code (a dimension column wider than 32 bits),
  the histogram bridge: those columns are summed per value as hidden GROUP BY keys and the parts are put together in
  exact arithmetic. Decimal results keep every digit.
- GROUP BY codes are 128-bit past 2^64 (up to 2^128 - 2 groups), up to 12 key columns; the general GROUP BY path has
  sparse output; the dense-array path falls back to the sparse product instead of refusing.
- Dense GROUP BY bridges, each behind a switch and measured in an ablation (SSB SF1): factoring the measure through
  the GROUP BY key (on), a row kernel built on a staged 64 x 64 bit transpose (chosen per tile by the cost model with
  a margin), a Shannon splitting tree, direct tile slots and Moebius inversion over the key bits (off: no gain). SSB
  M1-M3 from 505 to 179 ms on one thread (DuckDB 414), the 13 queries unchanged.
- Queries use every core by default (`set_threads(n)` changes it).

Correctness
- `torbles.fuzz`: differential fuzzing against DuckDB (random typed star schemas, random queries, every engine and
  planner setting, save / reload, a flipped bit repaired on load). Bugs it found before this release, all fixed:
  the join planner crashing on an empty fact table, 64-bit keys compared through floating point, decimal scaling
  overflowing at DECIMAL(18), 64-bit timestamps in dimensions, GROUP BY on a 64-bit column after reloading, and the
  group count overflowing 63 bits; later, 64-bit unsigned group keys decoded through int64. Refusals went from 2.3x
  the queries DuckDB rejects to a few percent of them (128-bit sums and group codes, the histogram bridge); the
  fuzzer counts only documented refusals and its cases no longer depend on Python's hash seed.
- New tests: typed columns on every engine and join form, the loaders, the key design, persistence and integrity, and
  a 40-case fuzz run; `make test` runs them all.

Packaging
- `pip install .` builds the engine (and the CUDA executor when nvcc is found); `TORBLES_PORTABLE=1 python -m build`
  makes a wheel for other x86-64 machines; `torbles-fuzz` console script.
- GitHub Actions: tests and fuzzing on every push (Python 3.9 / 3.11 / 3.13), wheels and PyPI publishing on a tag.

Changed (engine API)
- Queries run on every core by default (0.1.0: one thread unless `set_threads` was called).
- On plain uint32 tables, comparisons with negative, fractional or > 32-bit constants are exact (0.1.0 wrapped them).
- `Query.join(dim)` finds the foreign key when exactly one references the dimension.

Benchmarks
- `bench/ssb/schema.sql` (the SSB keys as standard DDL) and `bench/ssb/ssb_auto.py`: SSB loaded with nothing designed
  by hand; every result equals DuckDB's. CPU, SF1, 2 vCPUs, best of four interleaved rounds: the 13 queries in 122 ms
  (automatic keys), 111 ms (hand-designed keys; 113 ms with 0.1.0) and 311 ms for DuckDB in memory; with the GROUP BY
  bridges, M1-M3 in 96 ms against DuckDB's 228. Only answers equal to DuckDB's are timed. `bench/ssb/ablation.sh`
  measures every bridge; `wsl/run_validation.sh` runs `ssb_auto.py` too.

## 0.1.0 (2026-10-29)

First public release of torbles (a torble is a toroidal table; torbles is the engine and package): a research
prototype. The API may change; see the README's known limitations.

Engine
- Torus bit-plane storage with face (XOR) integrity checks and single-bit repair; only planes that can be nonzero
  are stored.
- Algebraic query execution: filters as Boolean polynomials, sums as popcount algebra (linear and bilinear measures).
- Equi-joins as a substitution in the dimension filter's polynomial (`JoinPoly`), planned by a calibrated cost
  model: factored comparators, expanded cubes, complement, or lookup; join polynomials planned in C (`jp_terms`).
- Order-encoded bounds, range / band / inequality joins with dominance sums (groups or slices).
- Derived surrogates (official keys kept as a projection), additive deltas for moved dimension rows,
  self-extending records (rank-addressed sidecars, PDEP), large objects (`BlobStore`).
- GROUP BY as one AND-popcount matrix product `C = M^T X` with sparse output (no dense key domain; the former
  2^22-group limit is gone); AVX-512 VPOPCNTDQ / AVX2 / scalar kernels chosen at run time.
- Persistent worker pool for every parallel call; compiled plans reused for repeated queries (same query, same data
  version); results decoded in bulk.
- Native, parallel ingestion: the bit-plane build on all cores, foreign-key reference checks and re-encoding in C.

GPU (CUDA, optional)
- One thread per 64-row word: filters, popcount sums, GROUP BY by mask splitting into a sparse group table on the
  GPU (default), dominance slices, row ranges, delta planes, self-extending columns.
- Tile matrix-product GROUP BY kernels with `__popc` and with tensor-core b1 MMA (`mma.sync.m16n8k256 .and.popc`),
  self-tested against popcounts at start.
- Builds with nvcc older than the GPU (PTX fallback) and without the b1 MMA if it is rejected; a CPU emulation of
  every kernel for testing without a GPU.

Benchmarks
- Official SSB scaling benchmark, SF1 to SF50 (6M-300M rows) and 1-24 threads (`wsl/run_scaling.sh`, graphic in
  `docs/img/ssb_scaling.png`, from `bench/ssb/poster_scaling.py`). DuckDB runs on an in-memory copy of its tables with its default memory limit, so both
  engines answer from memory; first runs are reported next to the best of 3.

Fixes since the development snapshots
- Values wider than 32 bits are refused, never truncated. A derived surrogate whose hierarchy fields no longer fit
  beside the official key (part, from SF50 on) had wrapped silently. Such a hierarchy is now stored as one dense,
  order-preserving cell rank.
- Planner caches keyed by Python object id could return a freed space's answer to a new one; keyed by object now.
- The plain engine no longer scans a unit twice when given many row ranges.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
