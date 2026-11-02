# Changelog

## 0.3.0 (2026-11-02), alpha

SQL on the bit planes, an inventory of exact mathematical query families, and the fixes of an independent release
review. 0.2.0 had the engine, the loaders and persistence; 0.3.0 adds the SQL front end and everything built on it
since, and raises the requirements to Python 3.11, NumPy 1.23.2 and DuckDB 2.0.

Loading

TIMESTAMP WITH TIME ZONE, BLOB and JSON columns now load, and NULLs are stored as a validity plane per column instead
of being refused outside foreign keys (`fill_nulls=` still replaces them at load).

SQL

`db.sql(text)` runs queries written in ISO/IEC 9075 SQL (DuckDB's dialect is accepted too), and `db.select(...)`
builds the same bound queries in Python. DuckDB's parser reads the text; a pre-pass rewrites the ISO forms its grammar
lacks, and the binder types every expression with DuckDB's rules wherever the standard leaves the choice open, so
answers compare value for value. A star block runs as engine queries: each filter on one table becomes a plane
predicate, a dimension row mask or a fact code set (so any function in it runs on distinct values); COUNT, SUM, AVG,
the variances, covariances, CORR and the nine REGR functions are exact moment sums; MIN and MAX are computed on the
planes by plane elimination (a new engine pass, `Query.minmax()`, also on the plain engine; with the GPU engine it
runs on the CPU); SUM of CASE and FILTER clauses run as engine variants; LEFT JOINs to dimensions run as disjoint
passes. Everything else runs on the distinct tuples the engine returns, and other joins as hash joins of the engine's
multisets. The ISO coverage is nearly complete for queries: every predicate (SIMILAR TO, LIKE_REGEX, MATCH, UNIQUE,
OVERLAPS, PERIOD predicates, IS NORMALIZED, IS JSON, MEMBER OF, SUBMULTISET, IS A SET), the scalar functions, the
aggregates (percentiles, hypothetical set functions, LISTAGG, COLLECT, FUSION, INTERSECTION, JSON_ARRAYAGG), window
functions with every frame and exclusion, time zones, collations, intervals with qualifiers, binary strings, arrays,
multisets, SQL/JSON with its path language and JSON_TABLE (with its PLAN clauses), LATERAL, partitioned outer joins,
WITH RECURSIVE with SEARCH and CYCLE, MATCH_RECOGNIZE and row pattern recognition in windows, nested window functions
(VALUE_OF, ROW_NUMBER of a row marker), row types with named fields (also as stored STRUCT columns), DECFLOAT,
TIMESTAMP(p) / TIME(p), and TABLESAMPLE, the UNICODE collation (the Unicode Collation Algorithm, through the optional
pyuca package), JOIN ... USING (...) AS j, COLLATION FOR, the literal UNKNOWN, character set introducers, and ordered
aggregates over windows. `docs/SQL.md` lists every construct with where it runs and what is excluded (statements other
than queries, SQL/XML, property graphs and the other parts of ISO/IEC 9075). In Python, every function is
`F.<name>(...)` and every other expression form `F.expr("<SQL text>", tables...)`.

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

Many windows now need no row at all. A window whose value depends on its row only through the row's group of
(PARTITION BY values, ORDER BY key) is a function of its aggregated series S(t), the counts and sums of those groups,
which the engine's window analytics described and nothing used. RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST and
ROW_NUMBER (on a key unique in its partition), and COUNT(*), COUNT, SUM and AVG of an integer or DECIMAL column under
a RANGE frame bounded by UNBOUNDED or CURRENT ROW, a GROUPS frame with constant offsets or a ROWS frame on a unique
key, are read from one engine GROUP BY of their partition and order key, on the planes or the GPU: the frame is
applied to the series as prefix sums (a running aggregate is S(t) / (1 - t)), and each row reads its group's value.
When the block is ordered by the window's own key with a LIMIT and the frame ends at the current row, the rows output
need the series only up to the last one's key, so the GROUP BY runs under that bound and the block reads only the rows
it outputs. On the release review's mixed workload at 1M rows (RTX 5090 laptop, 24 threads), a partitioned running SUM
and a global RANK over every row went from 375 ms together on the CPU engine and 398 ms with the GPU engine to 17.7
and 12.4 ms, against 62.6 ms for DuckDB, and the whole workload of eight queries from 398 and 414 ms to 36.8 and 24.5
ms, against 74.8 ms. Windows with more groups than one per 16 rows stay on the rows. `tests/check_algebra.py` runs
random window queries with the path on and off and against DuckDB on the three engines (1,500 matched while it was
written).

Windows whose values every row outputs no longer pass through a Python object per value on the way out. A window's
values stay an integer or double array from the window step to the output (a series window's come from an array
table indexed by the rows' codes), an ORDER BY reorders the rows by array positions and is not applied when the rows
already follow it, the window ordering sorts the integer codes as they are instead of ranking them first, an ORDER BY
the rows already follow sorts only the partitions (a radix sort when their codes fit 16 bits), and a running integer
SUM over frames that all hold a value stays an int64 array. The cross-engine benchmark of 2026-11-02 found these the
one place where torbles trailed: its two window queries that return all of 1M rows took 622 ms together on the CPU
engine, against 527 ms for DuckDB, 320 ms for chDB and 260 ms for Polars. On the same laptop, timed by
`bench/window_output.py` outside that benchmark's containers, they now take 252 and 158 ms on the CPU engine (410 ms
together) and 247 and 171 ms with the GPU, against 583 and 149 ms for DuckDB in the same run
(`results/release_run_2026-11-02/`); on a 2-vCPU cloud machine they went from 1,349 and 482 ms to 483 and 347 ms. Most of what remains is building the million result tuples, which any engine pays to return rows to Python. `tests/check_window_output.py` compares these forms
with DuckDB on the three engines, and `bench/window_output.py` times the two queries.

The executor keeps the engine's results as codes. The GROUP BY columns an engine query returns are integer code arrays
with their validity masks and codecs (codes order and compare like the values), and they stay codes through
expressions, grouping, joins, DISTINCT, the set operations, ORDER BY and LIMIT; values are decoded once, for the rows
that are output. Arithmetic and comparisons of code columns run on the codes, any other function runs once per
distinct combination of the codes it reads, and a filter on a fact column with millions of possible codes is evaluated
on the codes present. The engine's own result handling (128-bit sums and group codes, self-extending keys, measures
factored through the GROUP BY key, lookup keys) is array arithmetic instead of a loop over groups; MODE and the
percentiles of expressions work on the (group, value, count) histograms; AVG, the variances, covariances, CORR and
REGR_* are computed for every group at once as exact integer arithmetic on the moment sums, rounded once; COUNT, SUM
and AVG (DISTINCT) over window frames are merge-sort-tree dominance counts, and MEDIAN and the percentiles over frames
are parallel binary searches on those counts. The SQL front end tokenizes a statement once and reads a statement it
has parsed before from a cache. A DECIMAL zero is unsigned, as in DuckDB (Python's `Decimal('-1.5') * 0` is `-0.0`,
which turned `-13 / ((1 - 2.5) * 0)` into `+inf` on the row-by-row path). The SQL fuzzing record at the end of that
work, 39,300 statements (38,888 answers compared, DuckDB rejecting 412, torbles none), has no mismatch; on the 27,961
statements it shares with the record before this work, torbles' total time fell from 666 s to 230 s, its mean from
23.8 to 8.2 ms and its slowest statement from 87 s to 2.4 s, the median ratio to DuckDB from 1.38 to 1.27, and torbles
answers first on 25% of statements instead of 19%. The median statement (4.6 ms against DuckDB's 3.8) is now mostly
fixed cost: parsing, binding and planning in Python.

Some aggregates and limits no longer form (group, value) tuples or row pairs at all. MEDIAN and PERCENTILE_CONT /
_DISC of a fact column find the code at each wanted rank by rank selection on the column's planes, eight bits at a
time, each step one small engine query grouped by the block's keys and the code's leading bits on the code ranges the
bits found so far leave; only the codes at those ranks are decoded. ORDER BY ... LIMIT k of a block without aggregates
uses the same selection: the code at rank k on the first sort column bounds the rows the block reads (and the next
sort column while the ties are many). COUNT(DISTINCT) has the engine set one bit per (group, value) cell instead of
counting it, a new aggregation mode of the row kernel, and counts each group's bits. An aggregate block over an inner
equi-join of two FROM items that is not a star (two derived tables or CTEs, a table joined with itself on a column
that is not its key) groups each side first by the columns the block reads outside its aggregates, with exact partial
sums, joins the groups and combines each aggregate from the sides' sums: a SUM over one side counts once per row of
the other, a SUM of a product of the two sides' factors is the product of their sums, MIN and MAX are the sides' own.
And a GROUP BY with many possible groups (2^14, by the keys' distinct counts) runs on the row kernel, which now
transposes a plane key once per 64-row word, instead of the matrix product, whose cost grows with the groups of each
tile; with a filter only while the dense group arrays stay small, so the 13 SSB queries keep their plans. The rank
selections and the cell bits apply from 2^16 fact rows on; on smaller tables reading the values is cheaper. The
general GROUP BY path now takes its sparse output when dense group arrays would hold more than four entries per fact
row (a domain of millions of code combinations on a table of thousands of rows took 0.8 s to allocate and scan). On
SSB SF1 at two threads, against the code before: the median of revenue per year went from 8.4 s to 59 ms (DuckDB 141),
two percentiles per customer nation from 11.8 s to 0.48 s (DuckDB 0.23), the ten rows with the highest revenue from
4.4 s to 54 ms, with a filter or a join from 0.6 to 1.0 s to 55 to 61 ms, COUNT(DISTINCT customer) per year from 252
to 80 ms (DuckDB 63), COUNT(DISTINCT part) per supplier nation and year from 11.2 s to 188 ms (DuckDB 548), the
customers ranked by revenue from 235 to 96 ms and joined with a second aggregate from 217 to 122 ms, and an aggregate
over a join of two CTEs with 33 million row pairs, which ran out of memory (6 GB) forming them, takes 41 ms (DuckDB
142); the 13 SSB queries are unchanged (113.9 ms against 113.0). The SQL fuzzing record of that work, the same 36,810
statements as before, has no mismatch, no error and no refusal, with the default settings and with these paths on for
every table. `tests/check_algebra.py` runs each path with its switch on and off (identical answers) and against
DuckDB, on the three engines, and is part of `make test`.

NULLs are now stored as a validity plane per column instead of being refused, so three-valued logic runs on the
engine; TIMESTAMP WITH TIME ZONE, BLOB and JSON columns load. `torbles.sqlfuzz` (`torbles-sqlfuzz`) generates random
schemas and SQL and compares every answer with DuckDB's; `tests/check_sql.py`, `tests/check_iso.py` and
`tests/check_arrays.py` run on the three engines. The 13 SSB queries written in SQL give DuckDB's answers in one engine query each.

Relationship catalogs, a prototype that is off unless a catalog is attached to a table (`discover()`): native modules
propose affine, sum and product relations between columns, prove each over every row (a proof is bound to the table's
mutation revision), and a bounded search composes them to rewrite a query's measure into a cheaper polynomial; per plan
the engine chooses between the original chain of operations, a combined system on the word planes and the adaptive
matrix or packed-row executor, shares intermediate values through two new measure bytecodes (`M_SAVE`, `M_LOAD`), and,
with `stage_strategy="auto"` or `"compose"`, splits a measure of degree up to four into at most two existing reduction
scans and a native finalizer. With no catalog attached it costs nothing measurable: on SSB SF10
(`wsl/ab_enhancements.sh`, three interleaved rounds on a 24-core laptop with an RTX 5090) the 13 queries changed by
+0.2% on one thread, +0.7% on 24 and +1.1% on the GPU, each inside its own round-to-round noise, with the paired rounds
going both ways. Six test modules and a 2,000-program random DAG check cover it; `examples/relationship_demo.py` and
`examples/system_planner_demo.py` show it at work.

Exact mathematical functions of query results, an opt-in module (`torbles.mathematics`, `torbles.math_domains`) and the
first slice of a larger inventory of mathematical query families kept with the prototype it comes from. Checked rational
polynomials of up to eight formal variables (degree eight, 128 terms, int64 numerators and denominators) are added,
multiplied, differentiated, integrated, substituted and evaluated in native code, refusing rather than wrapping or
rounding. A moment snapshot of one to four non-NULL integer fact columns of a query (its count, sums, squares and cross
products, built by the existing aggregate queries, on the GPU too) turns a function of the data such as SUM((y - theta
x)^2) into a polynomial in its parameters, which is then evaluated, differentiated or integrated without another scan.
Typed domain descriptors and a native module registry dispatch the operators, and a bounded LRU cache reuses a snapshot
for an equivalent scope (the same predicate, columns, codecs, source revision and row membership) with no aggregate
query on a hit: in the prototype's synthetic benchmark, five constructions over a million rows took 0.82 ms instead of
15.6. These snapshots take non-NULL integer columns only; the scopes and moment banks below lift that.
`docs/MATHEMATICS.md` describes the API and its exact scope.

The other 75 capabilities of that inventory followed, each in the same form: native C states and transforms, a CUDA
kernel, an opt-in Python module that only marshals arguments, and a page under `docs/math/` with its exact scope,
budgets and refusals (`docs/MATHEMATICS.md` links every page; `ALGEBRA_INVENTORY.md` records where each capability
stands). The shared foundations are scopes and grouped moment banks over the selected rows (DECIMAL, NULL-able and
64-bit columns, SQL or listwise NULL handling, exact rollups and a state cache), big-integer, rational, modular and
interval carriers, resource budgets with exact size formulas, owned handles with ABI checks, a module registry that
checks laws and carriers, a costed planner, certificates that keep snapshot facts apart from universal laws and
hypotheses, persistence, and fused multi-output reductions. On them sit calculus and jets, response and loss curves,
inverse answers and minimal interventions, quadratic, convex and polynomial optimization with KKT or interval
certificates, exact and modular linear algebra with leave-group-out and low-rank updates, polynomial identity
discovery with certified ideal reduction and elimination, histograms, quantiles, entropies and pairwise statistics,
generating functions and convolutions, Walsh and Fourier bases with approximate spectral kernels, hierarchies with
adaptive refinement, Boolean predicates, lattices, group quantifiers and GF(2) parity, counting over the observed rows
and over a declared domain, exact preimages of derived predicates, envelopes and parameter regions, provenance with
explanations and deletion propagation, attribution and repair search, finite probability models and factorized
inference, moment tensors with costed contraction, symmetries and closed-observable dynamics, graph paths and fixed
points, cubical homology and persistence, interval and approximation error certificates, and query equivalence with
whole query families compiled into one program. Every answer is exact or carries a stated enclosure or error bound,
and a request past a declared budget is refused, not approximated. Each capability's tests compare its CUDA path,
emulated with the thread order scrambled, with the CPU result bit for bit, and run again under AddressSanitizer and
UndefinedBehaviorSanitizer; its kernels compile with nvcc for sm_80 (780 tests in all). All 75 have been measured on
the RTX 5090 laptop (`python bench/validate_math.py --device`) and are validated, like the first slice. Code that does
not import these modules is unaffected, and `tests/run_all.sh` (120 checks, the modules' tests among them) passes.

Release review

An independent review of this alpha on 2026-11-01 found eight defects. Each is fixed, and its reproducers, with the
passing controls the review listed, are regression tests: `tests/check_review_sql.py`, `tests/check_review_api.py` and
`tests/test_review_native.py` run on the three engines in `tests/run_all.sh`, and CI runs `tests/check_min_deps.py` on
the installed wheel. The declared dependency floor, Python 3.9, NumPy 1.22 and DuckDB 1.1, had never been installed or
tested, and SQL did not run on it: the binder reads the syntax tree that DuckDB 2.0 serializes, every 1.x release
serializes a different one, and even `SELECT 1` failed. The floor is now the lowest that passes, Python 3.11, NumPy
1.23.2 and DuckDB 2.0. DuckDB 2.0 publishes no wheels for Python 3.9 or 3.10, so those versions are no longer
supported. An older DuckDB is refused at the first use of SQL with an ImportError that names the floor, while the
engine API keeps working, and CI installs the wheel with exactly those versions and runs the SQL tests outside the
source tree. In SQL, the paths that lower expressions to the engine computed casts and integer and DECIMAL arithmetic
exactly rather than in their types: SUM(CAST(i AS TINYINT)) over 100, 200 and 200 gave 500, and CAST(2147483647 AS
INTEGER) + 1 gave 2147483648, where DuckDB raises a conversion or overflow error. Every integer and DECIMAL operation
now checks its value against its type wherever it is computed, and fails the query only where the value is used. The
planner keeps a cast unless interval arithmetic over the stored columns proves that it cannot fail, so the SSB queries
keep their plans and timings. DISTINCT ON a collated expression now compares collation keys (it kept both 'a' and 'A'
under NOCASE). The null-safe joins the review found broken in its snapshot were already fixed here, and they gain 24
tests.

On the engine API, `sum()` and `minmax()` over a nullable column read the NULL rows' placeholder code as a value (3,
NULL and 8 summed to 14), and an all-NULL group got a value instead of NULL. An aggregate over nullable columns now
reads only the rows where its argument is valid. With GROUP BY, a second engine query finds the groups, so a group
with no valid row is NULL, as in SQL; the second scan is skipped when the filter already excludes the NULLs, as the
SQL front end's own queries do. `Query.count(col)` is new. `q.sql()` rendered `minmax()` as COUNT(*), so `db.verify()`
compared different things; it now renders every aggregate kind, validity planes and the engine's two-valued negations.
`minmax()` over a range or inequality join, which silently counted, is refused. A table's native storage was never
freed: twenty discarded tables of 50,000 rows left 11 MB allocated. A table now owns its planes, sidecars, plain
columns, GPU copy and file mapping, a finalizer frees them when the last reference goes, `Table.close()` and
`Database.close()` free them at once, and any later use raises `TableClosedError`. An empty table no longer crashes
the interpreter in `nplanes()`, which divided by its zero word count. The few other paths that divided or indexed by
an empty table's counts are fixed too, and every query kind is tested on empty tables. Fixing the lifetimes brought
out a fault of the GPU executor: it kept device copies of a query's host arrays keyed by their address and length, so
an array allocated where a freed one had been read the old one's copy. This already gave wrong answers on a GPU for a
table joined twice in SQL, from the third statement on. A copy now belongs to one array object and is dropped when
that array or its table is released, and a query whose joined table was closed raises `TableClosedError` on every
engine instead of answering from its compiled plan. The emulated GPU now makes the same copies, so the tests on the
CPU cover this. Last, a BETWEEN on a 32-bit column shifted by 32 bits in the CPU engine, which is undefined in C
(UndefinedBehaviorSanitizer found it); the bound is tested first now. Preparing the review's rerun found one more
wrong answer on the engine API: in the default join mode, a GROUP BY on a dimension attribute read the attribute from
the foreign key's planes as if the key were stored most significant bit first, so through a key that a hand-built
table stores the other way (`refs=` without `msb=`) its bits came back reversed, giving wrong sums and counts without
an error, or an IndexError. The loaders store every foreign key most significant bit first, so SQL was not affected.
Such keys are now looked up, as the other join modes did.

The review's protocol was then run again on this release, on the same laptop (2026-11-02,
`results/release_rerun_2026-11-02/`): the full suite with the GPU tests on the GPU, the review's reproducers, which
all give the expected answers now, the installed wheel's SQL tests on the three engines, device memory flat over 120
rounds of tables uploaded, queried and released, and the review's nine benchmark workers, every answer equal to
DuckDB's. Through `db.sql` at SF10 the 13 SSB queries took 97 ms on the CPU and 26 ms on the GPU, against 218 ms for
DuckDB, where the review had measured 121 and 41 ms on its older snapshot; through the Python API, 74 and 10 ms. The
mixed workload's two window queries at 1M rows took 375 ms, against DuckDB's 72 (the review's 7.7 s predates the
windows as array algebra), and in a second run, once windows were read from their series (above), 17.7 ms on the CPU
and 12.4 ms on the GPU, against 62.6 ms (`results/release_rerun_2026-11-02_series/`). The first run's one failure was
a gate test of the mathematics inventory that needed a capability not yet validated and found none, since the same run
had validated the last ones. It is fixed, and the second run validated that capability again on the GPU, so all 81 are
validated.

Two more independent reviews of the work since the first (windows read from their series, the foreign key fix and the
GPU's device copies), one of them by the first review's author, found four more faults, the first by both; each is now
a regression test in `tests/test_fixes.py`, `tests/check_algebra.py` or `tests/check_review_sql.py`. Rank selection
under ORDER BY ... LIMIT reads only the rows up to the k-th on the first sort key, and its check for windows never
looked at the ORDER BY, so in `ORDER BY k, LAG(x) OVER (ORDER BY id), id LIMIT 3` LAG was computed over those rows and
other rows came out than DuckDB's, on a table of 65,538 rows with the default settings. Before windows were read from their series,
only a block with no window in its SELECT list could meet this; reading them from their series let rank selection run
beside the SELECT list's windows, and so beside windows in the ORDER BY too. Rank selection now stays off unless every
window of the block, in either place, is read from its series, and `tests/check_algebra.py` puts windows at any position
of the ORDER BY. A frame offset near 2^63 (`GROUPS BETWEEN CURRENT ROW AND 9223372036854775807 FOLLOWING`) wrapped
around in 64-bit arithmetic on the series path; offsets past every group of the partition are now capped, as on the
rows. The GPU executor and its emulation counted twice the rows of a 64-row word that two row ranges share, which a
filter on a dimension gives on a fact sorted by its foreign key: a count over seven keys of five rows each gave 185 on
the GPU and 35 on the CPU, through the Python API and through SQL. This dated from 0.1.0. The ranges are now merged
before they are planned, and the GPU executors clamp them as the CPU's do. Last, a window with two or more PARTITION BY
columns over no rows raised a ValueError instead of returning no rows.

The release candidate was then fuzzed for an hour on the laptop's 24 threads (2026-11-02,
`results/release_run_2026-11-02/`): 1,355,495 SQL statements on the three engines, with the default settings and with
the algebraic paths on for every table, and 1,455,975 engine queries, each against DuckDB. 50 were reported. Nine were
torbles' faults, from five causes, all fixed and each a regression test in `tests/check_campaign_sql.py` (16 of its 25
checks fail on the candidate). An aggregate's FILTER was computed before the WHERE, so an error in it on a row the
WHERE drops (abs of the least INTEGER, a DECIMAL(18) product past its type) failed the query; it now runs on the rows
the WHERE keeps. A LEFT JOIN whose ON term was not TRUE padded its rows with NULLs, also for values cached while the ON
terms ran, so a WHERE after the join that read the same cast read NULL there and dropped the rows (a count of 0 instead
of 36); the cache now stays with the rows it was computed for. SUM and AVG of values holding the least BIGINT wrapped
around modulo 2^64, because the bound that decides whether a sum fits 64 bits read the values' magnitude through
NumPy's abs, which leaves -2^63 negative; every such bound now reads it exactly. A block left to the general path
joined FROM items listed with commas by forming every pair before applying the WHERE's equalities, so a statement on a
65-row fact asked for 120 GiB; equalities between the items' columns are now the joins' keys. And a LEFT JOIN's ON
term on the dimension alone failed the query for an error on any dimension row; it now counts only where the join's
equality pairs the row, as the planner's other terms already did. Five more reports were tables the fuzzer generated
with more columns than a torble holds once the hidden ones are counted, which the loader refused with an internal
message; it now refuses them with the count and the `columns=` hint (the README's limits say which columns are
hidden), and the fuzzers count it as a refusal. The other 36 were DuckDB's or the comparison's: 16 answers that change
to torbles' when DuckDB runs without its statistics propagation or without its optimizer (64-bit sums its statistics
chose that wrapped, a join filter it derived that kept a row the WHERE drops, the rounding of its optimized plans), 8
errors DuckDB's optimizer let it skip computing and raises with the optimizer off, 11 orders of rows cut by a LIMIT
that differ only through documented differences (CORR and REGR_* NULL here and NaN there, exact here and rounded
there), and one comparison of the fuzzer's own, which sorted -0.0 apart from 0.0. The fuzzers now ask DuckDB those
questions themselves before they report a difference (`docs/SQL.md`).

Packaging

GitHub Actions now tests on Python 3.11, 3.12 and 3.13 and adds a job that installs the wheel with exactly the lowest
Python, NumPy and DuckDB the package allows and runs its SQL tests outside the source tree; wheels and PyPI publishing
on a tag as before. The release number is in `torbles/version.h` (the native libraries report it, and `algq.py`
refuses a library built from another release: rebuild with `make`).

Upgrading from 0.2.0

Python 3.9 and 3.10 are no longer supported, nor NumPy below 1.23.2 or DuckDB below 2.0 (the SQL front end reads
DuckDB 2.0's syntax tree; an older DuckDB is refused at the first use of SQL, and the engine API keeps working without
it). On the engine API, SUM and MIN / MAX over a nullable column skip its NULLs and give NULL for a group with no
value, as SQL does (with GROUP BY this takes a second scan unless the filter already excludes the NULLs), and
`Query.count(col)` is new. A table frees its native storage when its last reference goes or at `Table.close()` /
`Database.close()`; using it afterwards raises `TableClosedError`. `minmax()` over a range or inequality join, which
counted, now raises ValueError. Saved databases keep their format and load as before.

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
