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
  TIMESTAMP, TIMESTAMP WITH TIME ZONE, BOOLEAN, VARCHAR, BLOB and JSON (dictionaries; `startswith`), DOUBLE (dictionary). Comparisons with constants become exact
  code intervals, as SQL compares them. `types=` loads a column as another type (a DOUBLE amount as DECIMAL).
- NULLs stored as a validity plane per column (`fill_nulls=` replaces them at load instead); unsupported types
  skipped with a note; `columns=` subsets.
- Automatic key design (`keydesign.py`): functional-dependency chains found on the full data become derived
  surrogates `cell << w | row`, dense-ranked or shortened when they do not fit 32 bits; the planner is told which
  bits each attribute occupies (coordinate hints), so filters are cubes or comparators from the first query.
- `db.describe()`, `db.verify(q)` (the query's SQL on the source, compared), `q.sql()` in the source's schema.

SQL

`db.sql(text)` runs queries written in ISO/IEC 9075 SQL (DuckDB's dialect is accepted too), and `db.select(...)`
builds the same bound queries in Python. DuckDB's parser reads the text; a pre-pass rewrites the ISO forms its grammar
lacks, and the binder types every expression with DuckDB's rules wherever the standard leaves the choice open, so
answers compare value for value. A star block runs as engine queries: each filter on one table becomes a plane
predicate, a dimension row mask or a fact code set (so any function in it runs on distinct values); COUNT, SUM, AVG,
the variances, covariances, CORR and the nine REGR functions are exact moment sums; MIN and MAX are computed on the
planes by plane elimination (a new engine pass, `Query.minmax()`, also on the plain and GPU engines); SUM of CASE and
FILTER clauses run as engine variants; LEFT JOINs to dimensions run as disjoint passes. Everything else runs on the
distinct tuples the engine returns, and other joins as hash joins of the engine's multisets. The ISO coverage is
nearly complete for queries: every predicate (SIMILAR TO, LIKE_REGEX, MATCH, UNIQUE, OVERLAPS, PERIOD predicates, IS
NORMALIZED, IS JSON, MEMBER OF, SUBMULTISET, IS A SET), the scalar functions, the aggregates (percentiles, hypothetical
set functions, LISTAGG, COLLECT, FUSION, INTERSECTION, JSON_ARRAYAGG), window functions with every frame and
exclusion, time zones, collations, intervals with qualifiers, binary strings, arrays, multisets, SQL/JSON with its
path language and JSON_TABLE (with its PLAN clauses), LATERAL, partitioned outer joins, WITH RECURSIVE with SEARCH and
CYCLE, MATCH_RECOGNIZE and row pattern recognition in windows, nested window functions (VALUE_OF, ROW_NUMBER of a row
marker), row types with named fields (also as stored STRUCT columns), DECFLOAT, TIMESTAMP(p) / TIME(p), and
TABLESAMPLE, the UNICODE collation (the Unicode Collation Algorithm, through the optional pyuca package), JOIN ... USING
(...) AS j, COLLATION FOR, the literal UNKNOWN, character set introducers, and ordered aggregates over windows.
`docs/SQL.md` lists every construct with where it runs and what is excluded (statements other than queries, SQL/XML,
property graphs and the other parts of ISO/IEC 9075). In Python, every function is `F.<name>(...)` and every other
expression form `F.expr("<SQL text>", tables...)`.

Window functions, and GROUP BY over the tuples the engine returns, run as array algebra (`sqlwin.py`): every PARTITION
BY, ORDER BY and grouping column becomes dense order codes and one stable lexsort orders every partition, so ranking
functions, LAG / LEAD, frame edges (ROWS, GROUPS, RANGE with offsets, EXCLUDE) and FIRST / LAST / NTH_VALUE are
arithmetic on positions, COUNT / SUM / AVG and the statistics are differences of exact prefix sums of their moments
(doubles and decimals scaled to integers exactly), and MIN / MAX / ARG_MIN / ARG_MAX are range minima from a sparse table
built one level at a time. On 200,000 rows the window step of a ranking or running-sum query went from seconds to tenths
of a second (17 s to 0.7 s for the two benchmark windows), and statistics of DOUBLE columns per group from 6.1 s to
1.2 s; ORDER BY sorts by the same codes, arithmetic, comparisons and logic on integer and double columns run on
whole columns, joins of other blocks, DISTINCT and the set operations match row codes by sorting, and a block without
aggregates reads its rows from the planes instead of grouping them first. `a IS NOT DISTINCT FROM b` in a join's ON
clause now matches NULL with NULL. `tests/check_arrays.py` checks every answer against the row-by-row definitions
(value for value) and against DuckDB. DOUBLE columns with NaN or infinities now load (they were left out with a note),
compared and grouped as DuckDB does.

The executor keeps the engine's results as codes. The GROUP BY columns an engine query returns are integer code arrays
with their validity masks and codecs (codes order and compare like the values), and they stay codes through
expressions, grouping, joins, DISTINCT, the set operations, ORDER BY and LIMIT; values are decoded once, for the rows
that are output. Arithmetic and comparisons of code columns run on the codes, any other function runs once per
distinct combination of the codes it reads, and a filter on a fact column with millions of possible codes is evaluated
on the codes present. The engine's own result handling (128-bit sums and group codes, self-extending keys, measures
factored through the GROUP BY key, lookup keys) is array arithmetic instead of a loop over groups; COUNT(DISTINCT),
the percentiles and MODE work on the (group, value, count) histograms; AVG, the variances, covariances, CORR and REGR_*
are computed for every group at once as exact integer arithmetic on the moment sums, rounded once; COUNT, SUM and AVG
(DISTINCT) over window frames are merge-sort-tree dominance counts, and MEDIAN and the percentiles over frames are
parallel binary searches on those counts. The SQL front end tokenizes a statement once and reads a statement it has
parsed before from a cache. A DECIMAL zero is unsigned, as in DuckDB (Python's `Decimal('-1.5') * 0` is `-0.0`, which
turned `-13 / ((1 - 2.5) * 0)` into `+inf` on the row-by-row path). The final SQL fuzzing record, 39,300 statements (38,888 answers compared, DuckDB rejecting 412,
torbles none), has no mismatch; on the 27,961 statements it shares with the record before this work, torbles' total time
fell from 666 s to 230 s, its mean from 23.8 to 8.2 ms and its slowest statement from 87 s to 2.4 s, the median ratio
to DuckDB from 1.38 to 1.27, and torbles answers first on 25% of statements instead of 19%. The median statement
(4.6 ms against DuckDB's 3.8) is now mostly fixed cost: parsing, binding and planning in Python.

NULLs are now stored as a validity plane per column instead of being refused, so three-valued logic runs on the
engine; TIMESTAMP WITH TIME ZONE, BLOB and JSON columns load. `torbles.sqlfuzz` (`torbles-sqlfuzz`) generates random
schemas and SQL and compares every answer with DuckDB's; `tests/check_sql.py`, `tests/check_iso.py` and
`tests/check_arrays.py` run on the three engines. The 13 SSB queries written in SQL give DuckDB's answers in one engine query each.

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
- GROUP BY codes are 128-bit past 2^64 (up to 2^128 - 2 groups), up to 12 key columns (a column repeated in the GROUP
  BY is grouped once); the general GROUP BY path has sparse output; the dense-array path falls back to the sparse
  product instead of refusing.
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
  group count overflowing 63 bits; later, 64-bit unsigned group keys decoded through int64 and a GROUP BY with
  exactly 2^64 groups on the GPU path. Final record: 162,325 queries over 4,806 random schemas, 149,176 answers
  compared, no mismatch; torbles refused 119 (GROUP BY keys past 128 bits) where DuckDB rejected 13,030 (before the
  128-bit work: 31,595 refusals, 2.3x DuckDB's rejections). The fuzzer counts only documented refusals, and its cases
  no longer depend on Python's hash seed.
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
- Validated on the RTX 5090 laptop (`results/validation_sf10_rtx5090_0.2.0`): every test, the GPU tests on the GPU,
  and SSB SF10 with every result equal to DuckDB's. M1-M3 on 24 CPU threads: 59.7 ms (0.1.0: 181.5; DuckDB 139.4);
  the 13 queries in 79.4 ms on the CPU and 10.7 ms on the GPU (DuckDB 223.4); with the automatic keys, 80.1 and 9.1 ms.

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
