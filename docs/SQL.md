# SQL on torbles

`db.sql(text)` runs SQL queries on a torbles database, and `db.select(...)` builds the same queries in Python. The
reference is ISO/IEC 9075 (SQL:2023, Part 2 Foundation): its query expressions, predicates, scalar functions,
aggregates, window functions, NULL semantics, and the SQL/JSON and row pattern recognition features of Part 2. Where
the standard leaves a choice to the implementation (the type of a division, the order of NULLs, how casts round), the
choice is DuckDB's, so that every answer can be compared with DuckDB's value for value; the test suite and a
differential fuzzer do exactly that.

```python
import torbles
db = torbles.from_ddl(open("schema.sql").read(), {"lineorder": "lineorder.parquet", "date": "date.csv", ...})

r = db.sql("""
    SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
    FROM lineorder JOIN date ON lo_orderdate = d_datekey JOIN customer ON lo_custkey = c_custkey
    WHERE c_region = 'AMERICA' AND d_year BETWEEN 1997 AND 1998
    GROUP BY d_year, c_nation ORDER BY d_year, c_nation""")
r.columns, r.rows, r.df()

from torbles import F                                   # the same query in Python
lo, d, c = db.table("lineorder"), db.table("date"), db.table("customer")
q = (db.select(d.d_year, c.c_nation, F.sum(lo.lo_revenue - lo.lo_supplycost).alias("profit"))
       .where((c.c_region == "AMERICA") & d.d_year.between(1997, 1998))
       .group_by(d.d_year, c.c_nation).order_by(d.d_year, c.c_nation))
q.run(); q.to_sql(); q.verify()                         # verify(): the SQL text run by DuckDB, compared
```

Every function is `F.<name>(...)` in Python, and every other SQL expression form (SQL/JSON, predicates such as PERIOD
or IS NORMALIZED, MATCH, subqueries, ordered aggregates, window clauses) is `F.expr("<SQL text>", table, ...)`, bound
against the given tables.

`db.sql(text, engine="alg")` runs on the bit planes; `"plain"` on the plain columnar baseline and `"gpu"` on the GPU
executor (or its CPU emulation) give the same answers. A script may hold several statements; the result is the last
one's. `SET TIME ZONE ...` sets the session's time zone; every other statement is a query.

### Read-only workloads

`db.sql_batch(text)` returns one relation per SELECT statement in input order. Compatible scalar COUNT, SUM,
and AVG statements over the same table and predicate can share a packed scope and an exact moment calculation.
Each result retains its original column names and types. Sharing lasts for the batch.

### Durable learned algebra

```python
from torbles.sqlalgebra import Catalog

algebra = Catalog().learn(db, "SELECT COUNT(*), SUM(x), AVG(y) FROM t")
db.save("snapshot")
algebra.save("workload.tqa")

reopened = torbles.load("snapshot")
algebra = Catalog.load("workload.tqa")
answer = algebra.query(reopened, "SELECT SUM(x+y), AVG(x) FROM t")
```

A catalog learns exact count and sum bases for supported scalar integer aggregates. New queries can combine
those moments into affine sums and averages, or combine saved scopes using interval proofs on a single
nonnullable integer filter column. Queries outside the basis run through the ordinary SQL executor.
The catalog checks source identity before reuse. After a change, its first eligible query refreshes the learned
bases, or the caller can run `algebra.refresh(db)` explicitly. Use `refresh=False` to execute against current data
without refreshing a stale catalog. Checkpoint reuse after reopening requires a validated, matching saved snapshot. A catalog created with
Catalog(engine="gpu") requests CUDA for learning and refresh. Queries answered from saved bases finalize on the host.

## How a query runs

DuckDB's parser reads the text (syntax only: `json_serialize_sql`, no data and no binding). The binder reads the tree
DuckDB 2.0 serializes; DuckDB 1.x serializes another, so the first SQL use refuses a DuckDB older than 2.0 with an
ImportError (`torbles.sqlfront.DUCKDB_FLOOR`, the floor pyproject.toml declares). ISO forms its grammar does
not have (`BETWEEN SYMMETRIC`, `LIKE_REGEX`, `MATCH_RECOGNIZE`, `JSON_TABLE`, `MULTISET`, ...) are first rewritten by a
token pre-pass into calls of marker functions it can parse. The binder resolves names against the catalog, types every
expression and produces the bound query (`torbles/sqlir.py`), which the Python API builds directly. Execution
(`torbles/sqlexec.py`) is algebraic wherever the engine can do the work, and the rest runs on what the engine returns.

**A star block** (a fact table, its dimensions through foreign keys, inner joins and LEFT JOINs to dimensions) runs as
engine queries. Each AND-term of WHERE and ON becomes one of three engine predicates, or stays residual:

- a *plane predicate*: a comparison of a column with constants (`=`, `<>`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IN`,
  `IS [NOT] NULL`, `LIKE 'prefix%'`) is a comparator polynomial on the column's code planes, with the three-valued
  logic of SQL carried by the column's validity plane;
- a *dimension row mask*: any other predicate on one dimension is evaluated once per distinct value tuple of the
  columns it reads, and the passing rows join like every dimension filter: as the bitmap of a join polynomial or a
  semi-join on the foreign key;
- a *code set*: any other predicate on one fact column is evaluated once per code the column can hold, and the passing
  codes become intervals or a bitmap over the code domain.

So every scalar function and predicate below runs on distinct values, never on rows, when it filters one table.

A table wider than a torble (0.4.0) is read as one table. The columns the loader stored once per value of a
determinant are a dimension's columns that the query never names: the planner joins that dimension through the
determinant, a foreign key, as it does to read a foreign key's own value (a hidden join), so a predicate on them is a
dimension filter and their GROUP BY and sums go through the join. The columns in row-aligned parts are the table's own:
each engine query reads them through a view of the parts' planes (docs/ENGINE.md).

The block then needs, per group, the columns its GROUP BY, residual predicates and non-polynomial aggregate arguments
read: the *atoms*. The engine groups the passing fact rows by the atoms (the AND-popcount matrix product) and returns,
per group, the row count and the exact sums of the *moment polynomials*. An aggregate whose argument is a polynomial in
exact columns is a function of a few such sums: `SUM(x)`, `AVG(x)`, `COUNT(x)` need x; `VAR_*`, `STDDEV_*` need x and
x²; `COVAR_*`, `CORR` and the nine `REGR_*` need x, y, x², y², xy. So these never put their columns into the atoms.
`MIN` and `MAX` of a fact column are computed by the engine as well, by *plane elimination*: from the top plane down,
a plane that has a set bit among the remaining rows of a group removes the rows where it is clear (one pass, every
group at once). `SUM` / `AVG` / `COUNT` of a `CASE` run as *variants*: one engine pass per branch, its rows those where
that branch is the first TRUE one. An aggregate's `FILTER (WHERE ...)` and the `IS NOT NULL` of its nullable columns
are variant predicates too. A LEFT JOIN to a dimension runs as disjoint passes: the matched rows, and the unmatched
ones (no ON match on the fact side, a reference without a row, a dimension row failing the ON terms) with the
dimension's columns NULL; the per-group states of every aggregate add up across passes.

Everything else is evaluated at the *group level*, on the engine's distinct atom tuples, each standing for its count of
rows: residual predicates, GROUP BY expressions, grouping sets, the other aggregates (from the (group, argument value,
count) tuples), HAVING, window functions, DISTINCT, ORDER BY, FETCH. When the atoms need more than 12 GROUP BY columns
or a code domain past 2^128, or the block has no aggregate (its result has one row per fact row anyway), the engine
returns the passing rows' bitmap instead and the atoms' values are read from the planes for those rows.

Some aggregates and limits never form those tuples. MEDIAN, PERCENTILE_CONT and PERCENTILE_DISC of a fact column are
found by *rank selection on its planes*: each wanted rank's code is fixed 8 bits at a time, every step one engine query
grouped by the block's keys and the code's leading bits, restricted to the code ranges the bits found so far leave, so
a column with millions of distinct values costs three or four small engine queries and only the codes at the wanted
ranks are decoded (with very many groups, one query on the whole code). ORDER BY ... LIMIT k of a block without
aggregates uses the same selection: the code at rank k on the first sort column bounds the rows that can be output,
the block reads only those, and while the ties are many the next sort column narrows them (not when the block has a
window function, in its SELECT list or its ORDER BY, that is not read from its series: such a window needs every
row). `COUNT(DISTINCT x)` has the
engine set one bit per (group, value) cell where it would count it, and counts each group's bits. An aggregate block
over an inner equi-join of two FROM items that is not a star (two derived tables or CTEs, a table joined with itself on
a column that is not its key) groups each side first, by the columns the block reads outside its aggregates (the join
keys, GROUP BY, ...), with partial sums; the groups are joined and each aggregate is combined from the sides' sums: a
SUM over one side counts once per row of the other side, a SUM of a product of the two sides' factors is the product
of their sums, MIN and MAX are the sides' own. The other aggregates keep their arguments' columns among the grouped
ones. No pair of rows is formed: a join of 33 million pairs on SSB SF1 becomes two grouped engine queries. A derived
table or CTE that only filters and computes columns of one stored table is grouped by the engine directly.
Rank selection, the LIMIT bound and the cell bits apply from 2^16 fact rows on (`TORBLES_ALG_MIN_ROWS`); on a smaller
table reading the values costs less than the extra engine queries.

On the CPU, supported scalar moments, exact distinct counts, and bounded ordering can also run directly on stored
planes, including split integer keys. Distinct counts use a code bitmap for a compact scalar domain or a native
table of unique (group, value) codes. A grouped query ordered by count can rank groups before gathering arguments
for aggregates such as MIN and COUNT(DISTINCT). Literal LIKE filters and supported text projections evaluate the
pattern once per dictionary value. Ordered minute buckets and selected CASE keys can also use encoded group states.
These paths check the query shape, types, and available storage metadata before selecting a specialized kernel.

That group-level work is array algebra as well, never a loop of comparisons. Grouping keys, PARTITION BY and ORDER BY
columns become dense order codes (equal codes for values SQL does not distinguish, codes ascending along the sort
order), so grouping, sorting and partitioning are one stable sort of integers; GROUP BY aggregates and window
functions are prefix sums, searches and range minima over the sorted runs (see Aggregates and Window functions
below); and arithmetic, comparisons, `AND` / `OR` / `NOT`, `IS NULL` and casts between integer types and DOUBLE run on
whole columns of integers and doubles, with the results of the row-by-row definitions (exact integers, IEEE doubles,
three-valued logic). The engine's results stay codes through all of this: each column it returns is an integer code
array with a validity mask and the codec that decodes it (codes order and compare like the values), so grouping,
joins, DISTINCT, set operations, ORDER BY and LIMIT work on codes, and values are decoded once, for the rows that are
output. Any other function of such columns runs once per distinct combination of the codes it reads and is gathered
back: a function of a 65-value column costs 65 calls however many groups the engine returned. Only a function of
columns that are not codes (an aggregate's result, a value computed row by row) runs on each tuple.

**Snowflake filters** propagate from ancestors toward the fact through stored single-column integer foreign keys.
Each hop produces a packed row mask using the existing native join and validity checks. A filter-only ancestor can
fold into its direct dimension so the fact query uses the ordinary star kernels. When the query returns an ancestor
value and has a supported ancestor predicate, the masks narrow the fact scan and the original joins produce the
values. This path covers inner FK trees with supported plane predicates on the ancestors. Outer joins, sampled
inputs and unsupported equality or predicate expressions use the general path. `result.stats` reports successful
rewrites in `snowflake_blocks` and the number of propagated indirect FK edges in `snowflake_edges`.

**Other blocks** (outer and cross joins between arbitrary items, derived tables, CTEs, VALUES, table functions) give each
FROM item as a multiset: a stored table is an engine GROUP BY of the columns the block needs, with counts; a derived
table is its query's rows. The items are joined on their distinct tuples, carrying the counts: the join keys of both
sides become codes, and the matching pairs come from one sort and two searches (an equi-join) or the product (other
joins), so no row is matched in a loop; DISTINCT, UNION, INTERSECT and EXCEPT [ALL] work on the same row codes. A LATERAL
item (or `UNNEST(t.a)`, `JSON_TABLE(t.doc, ...)`) runs once per distinct tuple of the earlier columns it reads.

**Subqueries.** An uncorrelated subquery runs once: a scalar one becomes a constant, `IN` / `ANY` / `ALL` a constant
list with SQL's three-valued rules, so they reach the engine as plane predicates or code sets. A correlated one runs
once per distinct tuple of the outer values it reads.

**Errors** follow DuckDB's evaluate-on-use semantics: a failing cast, an integer or DECIMAL operation whose value leaves
its type (INTEGER + INTEGER past 2^31 - 1, a product past DECIMAL(18)) or a division by zero fails the query only if
the value is used: by a row that is output, by a WHERE term a row needs (a row that one term of an AND makes FALSE or
NULL, or one term of an OR makes TRUE, does not evaluate the others; a projection computes AND and OR in full), by a
GROUP BY or join key, by an aggregate of a group that is kept (a HAVING term on the GROUP BY expressions alone filters
the rows before grouping, as DuckDB's filter pushdown does). The engine's exact sums and plane predicates stand for an
expression only where the stored columns' ranges prove that none of its casts and operations can fail
(`SUM(CAST(i AS TINYINT))` on values past 127 is computed, and fails, on the values). A comparison with a constant is
decided as DuckDB's optimizer decides it (`x + 1 > 5` is `x > 4`, `x * 7 = -3` is FALSE for every non-NULL x), and
`COUNT(x * y)` counts the rows where x and y are not NULL, so that their arithmetic is not computed. DuckDB's other
rewrites that change what is computed are followed too: NOT of a comparison is the opposite comparison, the constants
of an integer product are folded into one placed last (`(2 * x) * y` is `(x * y) * 2`, `x * 0` is 0 or NULL), `x // 1`
is x, and `x IS NULL` computes x where DuckDB's statistics cannot decide it (a cast that can fail).

### Performance

For eligible high-cardinality GROUP BY queries ordered by descending count
with a small LIMIT, the CPU reads only the selected raw bitplanes. It
reconstructs key and measure codes in 64-row tiles, then combines exact
counts and sums by sparse key. AVG divides the exact sum by its count after
group selection. Only the selected groups' keys and values are decoded into
SQL results. Supported keys include split integers, dictionary strings, and
the minute of an unzoned timestamp. The planner keeps the general execution
path for cases such as nullable measures, explicit tie breakers, WITH TIES,
and other grouping expressions.


What the engine computes (filters, joins to dimensions, the aggregates of a star block, MIN and MAX, rank selections)
runs on the bit planes, and on the GPU where the GPU executor supports it. The rest of a query runs here, in Python
and NumPy on the CPU, whichever engine is selected: window functions not read from their aggregated series (Window
functions, below), the grouping of the tuples the engine returns, joins that are not stars, set operations, ORDER BY,
and functions evaluated on distinct values. Its cost grows with the rows it handles. In the release review's synthetic
workload, run on this release (2026-11-02, 1M rows, 24 threads, RTX 5090 Laptop GPU), a partitioned running SUM and a
global RANK, each over every row and returning 100 rows, read from their series took 17.7 ms together on the CPU
engine and 12.4 ms with the GPU engine, against 62.6 ms for DuckDB; computed from the rows, as before the series, 375
and 398 ms (the review measured 7.7 s on an older snapshot, from before the windows ran as array algebra). The
workload's six aggregate and distinct-value queries took 19 ms on the CPU and 12 ms with the GPU, against 12 ms. With
the GPU engine, MIN and MAX, row bitmaps, COUNT(DISTINCT) cells and semi-join lookups run on the CPU engine, so a SQL
workload is not all on the GPU. The 13 SSB queries through `db.sql` at SF10 took 97 ms on the CPU and 26 ms on the
GPU, against 218 ms for DuckDB, where the same queries built once with the Python API take 74 and 10 ms (README,
Results). A window whose values every row outputs keeps them as an integer or double array until the rows are built,
and an ORDER BY the rows already follow is not applied; returning a million rows still builds a million Python
tuples, as any engine does (`bench/window_output.py`).

### Automatic periodic aggregation

For eligible SQL, the optimizer looks for an exact packed-word cycle across
the value and validity planes the query reads. It checks the whole table
first, then contiguous sectors of 4,096 words (262,144 rows). A bounded prefix
check screens out irregular columns before full verification. Per-column
periods of up to 128 words can compose into a shared cycle of up to 4,096
words. The optimizer caches proofs and misses by the referenced physical
columns and their revisions, including for tables split across torbles.

Native CPU and CUDA reductions weight the packed masks for complete cycles
and account for remaining words and a final partial word. A sector plan can
include ordinary scans for irregular sectors when the full plan eliminates at
least three quarters of its word work. CUDA uploads the compact value planes
and reuses the device copy. The route covers integer scalar and grouped
`COUNT`, `SUM`, `AVG`, `MIN`, and `MAX` with supported predicates, up to
three group keys and two measure columns, on densely visible stored tables.

`db.prepare_periodic("events")` can still precompute whole-table periods on
one physical torble. Set `TORBLES_PERIODIC=0` before import to compare the
ordinary SQL executor. Result statistics expose proof time and cache hits,
whether a table or sector plan ran, and the number of proven and ordinary
sectors. CUDA also reports the plane payload, uploaded bytes, and kernel time.

### Topology compression during ingestion

Save a newly ingested database with `db.save(path, compression="topology")`
to store compact sectors inside the native torble files. Use a new destination.
The writer encodes the physical V and H edges in 4,096-word sectors, choosing
exact motifs or literal words for each edge. In particular, an H edge is the
XOR between adjacent column potentials. It can repeat even when both columns
are individually irregular. The writer keeps the independent edges that
carry the torus face checks and uses the raw file format when it is smaller.

`torbles.load(path)` checks the encoded directory and payload. The default
face validation XORs the stored V and H blocks directly. Clean snapshots
stay compact on reopen. Repair and ordinary kernels restore dense planes
on demand. Eligible CPU and CUDA SQL can compose sector cycles from the
stored directory.
Separately encoded V edges can prove a column cycle longer than the
query-time discovery bound. A bounded phase comparison can also coalesce
compatible sector motifs into one table-wide cycle.
`periodic_ingested_proof_uses` records directory lookups on the first plan.
Native revisions still invalidate stored proofs after in-memory changes.

For a packed word, a value bitmask is the prefix XOR of its V edges.
The CPU evaluates predicates and weighted reductions over encoded phase
words. A consumer that needs row identities receives the resulting
predicate bitmap, repeating its verified mask phases as needed. CUDA
uploads referenced motifs and literal slices, and builds only the
representative bitplanes on device. `periodic_encoded` identifies this
route. `periodic_encoded_payload_bytes` records referenced block bytes on
CPU and value bytes uploaded on initial CUDA setup. CUDA separately
reports descriptor bytes and resident representative planes.

## Data types

| SQL type | values | stored columns |
|---|---|---|
| SMALLINT, INTEGER, BIGINT, HUGEINT, unsigned types | int (arithmetic never wraps; overflow is an error where DuckDB's is) | yes |
| DECIMAL(p, s), NUMERIC | Decimal, exact | yes |
| DECFLOAT(p) (T076; p defaults to 34) | Decimal rounded to 16 digits (p <= 16) or 34, with Infinity and NaN (NaN sorts last, equals itself and groups as one value; an invalid operation such as Infinity - Infinity is NaN, as IEEE 754 says) | no (the loaders read types through DuckDB, which has none) |
| REAL, DOUBLE PRECISION, FLOAT | float; NaN and ±Infinity included (NaN sorts after +Infinity, equals itself and groups as one value, -0.0 = 0.0, as in DuckDB) | yes (dictionary) |
| BOOLEAN (TRUE, FALSE, UNKNOWN) | bool (UNKNOWN is NULL) | yes |
| CHARACTER, VARCHAR, CLOB, NATIONAL CHARACTER ... | str (CHAR(n) is not padded, as in DuckDB; a CAST to VARCHAR(n) or CHAR(n) keeps the first n characters, as ISO says; `_UTF8'...'` introducers are accepted) | yes (dictionary) |
| BINARY, VARBINARY, BLOB, BINARY LARGE OBJECT | bytes | yes (dictionary) |
| DATE, TIME, TIMESTAMP, TIME(p), TIMESTAMP(p) | date, time, datetime; with p < 6 fractional digits, a cast rounds half away from zero (as DuckDB's TIMESTAMP_S / _MS) | DATE, TIMESTAMP |
| TIME / TIMESTAMP [(p)] WITH TIME ZONE | an instant with its displacement (aware datetime) | TIMESTAMP WITH TIME ZONE |
| INTERVAL, INTERVAL <qualifier> (year-month and day-time) | months, days, microseconds (returned as timedelta); a literal or CAST with a qualifier is typed `INTERVAL DAY`, ..., arithmetic keeps the qualifier, a CAST to another qualifier cuts the fields past its last one, and a single-field interval casts to a number | no |
| JSON | its text | yes (dictionary) |
| ARRAY, MULTISET | list (values, not stored) | no |
| ROW (a INTEGER, b VARCHAR, ...) (T051) | a tuple of the fields' values; returned as a dict when the fields have names, else as a tuple (as DuckDB returns STRUCTs) | yes (dictionary; DuckDB STRUCT columns whose fields are scalars or rows) |

NULL is stored as a validity plane next to the column (code 0 in the data planes), so `IS NULL` is a plane and every
comparison is three-valued on the engine.

## ISO coverage

The tables list every construct of ISO/IEC 9075-2:2023 within queries, where it runs, and what is not supported.
*Engine* means the work is done on the bit planes (plane predicate, row mask, code set, moment sums, plane elimination);
*values* means it runs on distinct values or distinct group tuples as described above. A function in a WHERE term on
one table is always evaluated on that table's distinct values (row mask or code set).

### Query expressions

| Construct | Where it runs, notes |
|---|---|
| SELECT [DISTINCT], FROM, WHERE, GROUP BY, HAVING, ORDER BY [ASC/DESC] [NULLS FIRST/LAST] | engine for star blocks (above), values otherwise |
| inner, LEFT, RIGHT, FULL, CROSS, NATURAL joins; ON, USING, USING (...) AS j (F404) | star blocks with inner / LEFT joins to dimensions: engine (LEFT as passes); others: hash joins of multisets |
| partitioned outer join `t PARTITION BY (c) RIGHT JOIN u ON ...` (F403) | values: each partition joined on its own, padded rows keep the partition's values |
| LATERAL derived tables (T491), UNNEST [WITH ORDINALITY], TABLE (<collection>) | values, once per distinct tuple of the outer columns read |
| derived tables with column lists `AS t(a, b)`, TABLE t, VALUES | values |
| WITH, WITH RECURSIVE, SEARCH DEPTH / BREADTH FIRST, CYCLE ... SET ... USING (T121, T131, T133) | values; the recursion iterates the step on the new rows |
| UNION, EXCEPT, INTERSECT [ALL / DISTINCT] [CORRESPONDING [BY (columns)]] | values |
| GROUP BY [DISTINCT / ALL] ROLLUP, CUBE, GROUPING SETS, (), GROUPING(...) (T431-T434) | engine groups by the union of the sets' columns once; the sets are folded at the group level |
| functional dependencies on primary keys and equalities (T301) | binding (dependent columns join the GROUP BY list) |
| OFFSET, FETCH FIRST n [PERCENT] ROWS ONLY / WITH TIES, computed counts (F860-F867) | values |
| TABLESAMPLE BERNOULLI / SYSTEM (p) REPEATABLE (s) (T613) | engine: a sample row mask (BERNOULLI per row, SYSTEM per 64-row word), ANDed with the passing rows; a sampled dimension is a row mask |
| MATCH_RECOGNIZE (R010) | values: PARTITION BY, ORDER BY, MEASURES, ONE / ALL ROWS PER MATCH, SHOW / OMIT EMPTY MATCHES, WITH UNMATCHED ROWS, AFTER MATCH SKIP (all forms), PATTERN (concatenation, alternation, greedy and reluctant `* + ? {n,m}`, PERMUTE, `^ $`, `{- -}`), SUBSET, DEFINE, PREV / NEXT / FIRST / LAST, RUNNING / FINAL, CLASSIFIER(), MATCH_NUMBER(), aggregates |
| JSON_TABLE (T821-T828) | values: PASSING, COLUMNS (PATH, FOR ORDINALITY, EXISTS, FORMAT JSON with wrappers and quotes, ON EMPTY / ON ERROR), NESTED PATH, PLAN DEFAULT (OUTER / INNER, UNION / CROSS) and explicit plans `PLAN (p OUTER (a CROSS b))` (the default: outer join to the parent, union of siblings), ERROR / EMPTY ON ERROR (the table's ERROR ON ERROR is the columns' default) |
| scalar subqueries, EXISTS, IN / ANY / SOME / ALL (subquery), UNIQUE, row subqueries | uncorrelated: once, folded to constants; correlated: once per distinct outer tuple |

Row types (T051): `ROW(a, b)` and `(a, b)` build a row value, `CAST(x AS ROW(f INTEGER, g VARCHAR))` gives its fields
names, `r.f`, `(r).f` and `t.r.f` are field references (on a stored row column too), and rows compare, group, sort and
take DISTINCT field by field. A comparison of rows is the ISO one: a NULL field makes `=` UNKNOWN, `BETWEEN` is two row
comparisons, `r IS NULL` is TRUE when every field is NULL (a field that is a row is NULL only when it is NULL itself),
COALESCE and NULLIF are their CASE definitions, and a row of NULLs groups with the NULL row. In Python: `F.row(a, b)`, `x.cast("ROW(f INTEGER, g VARCHAR)")`, `x.field("f")`.

### Predicates

In a WHERE or ON term on one table, a predicate is a *plane predicate* (a comparator polynomial on the code planes)
or runs on the table's *distinct values* (a dimension row mask or a fact code set); elsewhere it runs at the group
level.

| Predicate | In a WHERE term on one table | Notes |
|---|---|---|
| comparison `= <> < <= > >=` of a column with constants | plane predicate | three-valued through the validity plane |
| row comparisons (8.2) | plane predicates (expanded) | lexicographic; a NULL field gives UNKNOWN |
| BETWEEN [SYMMETRIC / ASYMMETRIC], IN (list), row IN | plane predicate | |
| LIKE [ESCAPE], character and octet strings | prefix: plane predicate (an interval of the sorted dictionary); others: distinct values | |
| SIMILAR TO [ESCAPE] | distinct values | ISO regular expressions, not POSIX |
| LIKE_REGEX [FLAG] | distinct values | XQuery regular expressions (F841) |
| IS [NOT] NULL, row IS [NOT] NULL | plane predicate (the validity plane) | a row: every field |
| IS [NOT] DISTINCT FROM, IS [NOT] TRUE / FALSE / UNKNOWN | plane predicate | |
| EXISTS, UNIQUE, quantified ANY / SOME / ALL | uncorrelated: folded to a constant or a list | correlated: once per distinct outer tuple |
| MATCH [UNIQUE] [SIMPLE / PARTIAL / FULL] (F741) | distinct values | never UNKNOWN |
| OVERLAPS | distinct values | |
| PERIOD (s, e) OVERLAPS / EQUALS / CONTAINS / [IMMEDIATELY] PRECEDES / [IMMEDIATELY] SUCCEEDS, CONTAINS a point (T502) | plane predicates (comparisons of the bounds) | closed-open periods |
| IS [NOT] [NFC / NFD / NFKC / NFKD] NORMALIZED (T061) | distinct values | |
| MEMBER [OF], SUBMULTISET [OF], IS [NOT] A SET (S271, S275) | distinct values | |
| IS [NOT] JSON [VALUE / ARRAY / OBJECT / SCALAR] [WITH / WITHOUT UNIQUE KEYS], JSON_EXISTS | distinct values | |

### Scalar functions and operators

Numeric: `+ - * /`, `MOD`, `ABS` (numbers and intervals), `SIGN`, `SQRT`, `EXP`, `LN`, `LOG(b, x)`, `LOG10`,
`POWER`, `FLOOR`, `CEIL` / `CEILING`, `ROUND`, `TRUNC`, `WIDTH_BUCKET`, `SIN COS TAN ASIN ACOS ATAN SINH COSH TANH`
(T621-T624), `GREATEST`, `LEAST` (T054), and DuckDB's `//`, `%`, `CBRT`, `LOG2`, `ATAN2`, `COT`, `DEGREES`, `RADIANS`,
`PI`, `GCD`, `LCM`, `FACTORIAL`, `EVEN`, bit operators. Exact numbers stay exact (DECIMAL arithmetic follows DuckDB's
precision and scale rules); integer literals may be written `0x1F`, `0o17`, `0b101`, `1_000_000` (T661, T662).

Character strings: `||`, `CHAR_LENGTH` / `CHARACTER_LENGTH` / `OCTET_LENGTH` [USING CHARACTERS / OCTETS], `POSITION
(a IN b [USING ...])`, `SUBSTRING (s FROM a FOR b [USING ...])`, `SUBSTRING (s SIMILAR p ESCAPE e)`, `UPPER`, `LOWER`,
`TRIM ([LEADING / TRAILING / BOTH] [c] FROM s)`, `LTRIM`, `RTRIM`, `BTRIM`, `LPAD`, `RPAD` (2023), `OVERLAY (s PLACING r
FROM a [FOR b] [USING ...])`, `NORMALIZE (s [, NFC / NFD / NFKC / NFKD [, length]])`, `TRANSLATE (s USING <character
set>)`, the regular expression functions `OCCURRENCES_REGEX`, `POSITION_REGEX ([START / AFTER] ... [OCCURRENCE n]
[GROUP g])`, `SUBSTRING_REGEX`, `TRANSLATE_REGEX` (F842-F846, with FROM and USING OCTETS), and DuckDB's `LENGTH`,
`LEFT`, `RIGHT`, `REVERSE`, `REPEAT`, `REPLACE`, `TRANSLATE (s, from, to)`, `SPLIT_PART`, `STARTS_WITH`, `CONCAT`,
`CONCAT_WS`, `REGEXP_*`, `ASCII`, `CHR`, `INITCAP`, `GLOB`, `ILIKE`. Literals: `N'...'`, `U&'...' [UESCAPE 'c']`,
`U&"..."` identifiers, continued literals `'a' 'b'` (F393, F392).

Binary strings: `X'..'` literals, `||`, `OCTET_LENGTH`, `SUBSTRING`, `POSITION`, `OVERLAY`, `TRIM` (default octet
X'00'), `LIKE` with the octets `%` and `_`, comparison by octets, and DuckDB's `HEX`, `UNHEX`, `ENCODE`, `DECODE`.

Datetimes and intervals: `CURRENT_DATE`, `CURRENT_TIME [(p)]`, `CURRENT_TIMESTAMP [(p)]` (WITH TIME ZONE),
`LOCALTIME [(p)]`, `LOCALTIMESTAMP [(p)]`, one instant per statement; `EXTRACT` of every field including `TIMEZONE_HOUR`
and `TIMEZONE_MINUTE` (of the value's own displacement); datetime ± interval, datetime − datetime, `(a - b) <interval
qualifier>` (the difference in those fields, 6.42); interval literals and casts with any qualifier `INTERVAL '1-2' YEAR
TO MONTH`, `INTERVAL '3 04:05:06.5' DAY TO SECOND`, `INTERVAL -'1:30' HOUR(3) TO MINUTE`, `CAST(90 AS INTERVAL
MINUTE)`; `AT TIME ZONE <displacement or zone name>`, `AT LOCAL` (6.36: the same instant with another displacement);
`SET TIME ZONE LOCAL / <interval> / '<zone>'`; `CAST (x AS <datetime type> FORMAT '<template>')` and back to a string
(T839, the ISO datetime templates YYYY, MM, DD, HH24, HH12, MI, SS, FFn, A.M., TZH, TZM, ...); DuckDB's `DATE_PART`,
`DATE_TRUNC`, `DATE_DIFF`, `DATE_ADD`, `AGE`, `LAST_DAY`, `MAKE_DATE`, `STRFTIME`, `STRPTIME`, `DAYNAME`, ...

Arrays (S091, S096-S098) and multisets (S271, S275): `ARRAY[...]`, `ARRAY (<query>)`, `a[i]` (an index outside 1 ..
CARDINALITY is an error, as ISO says), `CARDINALITY`, `ARRAY_MAX_CARDINALITY` (of `CAST(... AS t ARRAY[n])`),
`TRIM_ARRAY`, `||`; `MULTISET[...]`, `MULTISET (<query>)`, `CAST(... AS t MULTISET)`, `SET`, `ELEMENT`, `MULTISET UNION
/ EXCEPT / INTERSECT [ALL / DISTINCT]` (INTERSECT binds tighter), multiset equality.

SQL/JSON: `JSON_VALUE` (PASSING, RETURNING, DEFAULT / NULL / ERROR ON EMPTY and ON ERROR), `JSON_QUERY` (wrappers
WITHOUT / WITH CONDITIONAL / UNCONDITIONAL, KEEP / OMIT QUOTES, EMPTY ARRAY / EMPTY OBJECT ON EMPTY), `JSON_EXISTS`
(TRUE / FALSE / UNKNOWN / ERROR ON ERROR), `JSON_OBJECT` (`KEY k VALUE v`, `k : v`, NULL / ABSENT ON NULL, WITH UNIQUE
KEYS, FORMAT JSON values, RETURNING), `JSON_ARRAY` (values or a query), `JSON (<text> [WITH UNIQUE KEYS])`,
`JSON_SCALAR`, `JSON_SERIALIZE`, and the simplified accessor `j.a.b`, `j[0]` (T860). The SQL/JSON path language: lax
and strict modes, `$`, `@`, `$variables`, `.key`, `."key"`, `.*`, `[n]`, `[n to m]`, `[last]`, `[*]`, filters `?(...)`
with `== != < <= > >= && || !`, `exists (...)`, `(...) is unknown`, `like_regex ... flag`, `starts with`; arithmetic
`+ - * / %`; item methods `type()`, `size()`, `double()`, `ceiling()`, `floor()`, `abs()`, `datetime([template])`,
`keyvalue()` and the 2023 methods `bigint()`, `boolean()`, `date()`, `decimal(p, s)`, `integer()`, `number()`,
`string()`, `time()`, `time_tz()`, `timestamp()`, `timestamp_tz()`.

Others: `CASE` (simple, searched, and extended: `WHEN 1, 2`, `WHEN > 5`, `WHEN IS NULL`, F262 / F263), `NULLIF`,
`COALESCE`, `CAST`, `TRY_CAST`, row values `ROW (a, b)` and `(a, b)`, `COLLATE` (an explicit COLLATE wins over a column's implicit collation, two different explicit ones conflict; UNICODE: the Unicode Collation Algorithm with its default table, through the optional pyuca package; UCS_BASIC, BINARY, C, POSIX, NOCASE,
NOACCENT, NFC and their combinations: comparisons, ORDER BY, GROUP BY, DISTINCT, set operations, MIN / MAX, windows; in
a filter the collation key is evaluated on the column's distinct values), `COLLATION FOR (x)`, `CURRENT_USER`, `SESSION_USER`, `SYSTEM_USER`,
`USER`, `CURRENT_ROLE`, `CURRENT_CATALOG`, `CURRENT_SCHEMA`, `CURRENT_PATH`, `CURRENT_DEFAULT_TRANSFORM_GROUP` (empty: there are no user-defined types), `typeof`.

### Aggregates

| Aggregate | How |
|---|---|
| COUNT(*), COUNT(x), SUM, AVG of a polynomial in exact columns | engine: moment sums (popcount algebra), exact at any width |
| VAR_POP, VAR_SAMP, STDDEV_POP, STDDEV_SAMP, COVAR_POP, COVAR_SAMP, CORR, REGR_SLOPE, REGR_INTERCEPT, REGR_COUNT, REGR_R2, REGR_AVGX, REGR_AVGY, REGR_SXX, REGR_SYY, REGR_SXY | engine: moment sums (x, x², xy, ...), then exact integer arithmetic on the sums of every group at once (each statistic is a ratio of integer polynomials in the sums, rounded once to the nearest double); not the running double sums DuckDB uses |
| MIN, MAX of a fact column | engine: plane elimination |
| SUM / AVG / COUNT of CASE, any aggregate with FILTER (WHERE ...) | engine: variants |
| COUNT(DISTINCT x) of a column without collation (not DOUBLE) | engine: one bit per (group, value) cell, a popcount per group |
| MEDIAN, PERCENTILE_CONT, PERCENTILE_DISC, quantiles of a fact column | engine: rank selection on the column's planes, 8 bits per query |
| MIN / MAX of other expressions, other DISTINCT aggregates, EVERY, ANY, SOME, BOOL_AND / OR, BIT_AND / OR / XOR, ANY_VALUE (2023) | values: (group, value, count) tuples from the engine, aggregated as arrays (below) |
| PERCENTILE_CONT, PERCENTILE_DISC, MEDIAN of other expressions, MODE | values, as arrays (below) |
| RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST (hypothetical, WITHIN GROUP) | values |
| LISTAGG (... ON OVERFLOW ...) WITHIN GROUP, STRING_AGG, ARRAY_AGG (... ORDER BY ...) | values |
| COLLECT, FUSION, INTERSECTION | values |
| JSON_ARRAYAGG (ORDER BY, NULL / ABSENT ON NULL), JSON_OBJECTAGG (WITH UNIQUE KEYS) | values |
| DuckDB's PRODUCT, ENTROPY, SKEWNESS, KURTOSIS, ARG_MIN / ARG_MAX, HISTOGRAM, APPROX_COUNT_DISTINCT (exact) | values |

When an aggregate is computed from values (the (group, value, count) tuples), GROUP BY is array algebra too: the
grouping keys become dense codes, one stable sort makes each group a run of rows, and COUNT, SUM, AVG, MIN, MAX,
VAR_*, STDDEV_*, COVAR_*, CORR, REGR_*, EVERY, BOOL_OR, COUNT_IF, ANY_VALUE, ARG_MIN / ARG_MAX, BIT_AND / OR / XOR and
their DISTINCT and
FILTER forms are computed for every group at once, each tuple weighted by its count: exact sums of the moments
(doubles and decimals scaled to integers exactly, so a sum of doubles is the correctly rounded exact sum) and range
minima of (value code, position). MEDIAN, PERCENTILE_CONT, PERCENTILE_DISC (when not by rank selection) and MODE sort
every group's values once and find each rank by a search of the cumulative counts. The other aggregates (ordered ones, LISTAGG, ARRAY_AGG, ...) run
group by group.

### Window functions

`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `PERCENT_RANK`, `CUME_DIST`, `NTILE`, `LEAD`, `LAG`, `FIRST_VALUE`, `LAST_VALUE`,
`NTH_VALUE ... FROM FIRST / FROM LAST`, `RESPECT / IGNORE NULLS` (T611-T618), every aggregate over a window (with
DISTINCT and FILTER), frames `ROWS / RANGE / GROUPS` with numeric and interval offsets (T620), `EXCLUDE CURRENT ROW /
GROUP / TIES / NO OTHERS`, the WINDOW clause and windows defined from named windows, windows over aggregates, QUALIFY
(DuckDB). They run at the group level, after aggregation, as array algebra: every PARTITION BY and ORDER BY column
becomes dense order codes (equal for peers, ascending along the sort order with its direction and NULL placement), and
one stable lexsort of the codes orders every partition at once. Partitions and peer groups are then runs of equal
codes, and each window function is arithmetic on positions: ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST and
NTILE from the positions of the row, its peer group and its partition; LAG / LEAD a shifted position (IGNORE NULLS: the
k-th non-NULL position, by a prefix count); a frame edge a position (ROWS), a peer group's bound (GROUPS, RANGE) or,
for RANGE with an offset, the rank of the target among the partition's keys (keys and targets ranked together, one
search); EXCLUDE leaves at most three runs. COUNT, SUM, AVG and the statistics over a frame are differences of exact
prefix sums of their moments (BIT_XOR of prefix XORs), MIN, MAX, ARG_MIN and ARG_MAX range minima of (value code,
position) from a sparse table built one level at a time (BIT_AND and BIT_OR the same reductions), FIRST_VALUE, LAST_VALUE and NTH_VALUE positions among the eligible ones. The results are the
row-by-row definitions' (exact sums, correctly rounded doubles, the first of tied extremes). COUNT, SUM and AVG
(DISTINCT) count a value at its first occurrence in the frame: the rows whose previous occurrence lies before the
frame's start, a dominance count answered by a merge-sort tree over positions. MEDIAN, PERCENTILE_CONT and
PERCENTILE_DISC over a frame rank the rows by (value, position) and find each frame's k-th row by a parallel binary
search on that rank, every round one dominance count for all frames. Other DISTINCT aggregates, MODE, ordered window
aggregates, nested window functions and row pattern recognition read each frame's rows, partition by partition, in the
same order.

Many windows do not need the rows at all. A window whose value depends on its row only through the row's group of
(PARTITION BY values, ORDER BY key) is a function of its aggregated series S(t), the count, non-NULL count and sum of
each group: in a star block of 2^16 fact rows or more, RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST, ROW_NUMBER (when no
two rows of a partition share their key) and COUNT(*), COUNT, SUM and AVG of an integer or DECIMAL column, without
DISTINCT, FILTER or EXCLUDE, under a RANGE frame bounded by UNBOUNDED or CURRENT ROW, a GROUPS frame with constant
offsets or a ROWS frame whose order key is unique in its partition, with at most one ORDER BY key, and keys that are
columns of exact, uncollated string, date, timestamp or boolean types. One engine GROUP BY of the (partition, order
key) gives the series, on the planes or the GPU; the frame is applied to it as prefix sums over the ordered groups (a
running aggregate is S(t) / (1 - t), a frame of GROUPS offsets a difference of two prefix sums), and each row reads
its group's value. The series is read when the key columns have at most one group per 16 rows, or when the block is
ordered by the window's own ORDER BY key with a LIMIT and the frame ends at the current row's peers: the rows output
then need the series only up to the key of the last one, so the GROUP BY runs under that bound, and the block reads
only the rows that can be output, as ORDER BY ... LIMIT does without windows. The values are those of the row-by-row
definitions, as above.

Nested window functions (T619): in the argument or FILTER of a window aggregate, `ROW_NUMBER(<row marker>)` is the
position in the ordered partition of `BEGIN_PARTITION`, `BEGIN_FRAME`, `CURRENT_ROW`, `FRAME_ROW` (the frame row being
aggregated), `END_FRAME` or `END_PARTITION`, and `VALUE_OF(x AT <row marker> [+ | - n] [, default])` is x at that row
(the default outside the partition), so `AVG(CASE WHEN d <> VALUE_OF(d AT CURRENT_ROW) THEN v END) OVER (PARTITION BY
c)` correlates each frame row with the current row. The argument is evaluated once per (current row, frame row) pair.
In Python: `F.value_of(x, at="CURRENT_ROW", offset=0, default=None)` and `F.row_number("FRAME_ROW")`.

Row pattern recognition in a window (R020): a window with `MEASURES`, `ROWS BETWEEN CURRENT ROW AND ...`, `AFTER MATCH
SKIP`, `INITIAL` / `SEEK`, `PATTERN`, `SUBSET` and `DEFINE` reduces each row's frame to the match that starts at the
row (INITIAL) or at the first row after it in its frame where one starts (SEEK); a row skipped by an earlier match, or
without a match, has an empty frame and NULL measures. `<measure> OVER w` reads a measure, and every window function
over `w` runs on the reduced frame. The matcher is MATCH_RECOGNIZE's; a match can neither take rows nor read values
outside the frame, and `^`, `$` and `MATCH_NUMBER()` are not allowed (ISO 7.11).

### Writable tables

Register a table with Database.add_writable_table(name, columns, primary_key="id") to run INSERT, UPDATE, DELETE,
and MERGE through Database.sql. Writable tables are in memory and have fixed nonnullable unsigned integer columns and
one immutable primary key. See [refresh and SQL DML](REFRESH.md#sql-dml) for supported forms, examples and benchmarks.

### Not supported, and why

- **Statements other than queries.** Tables built by the loaders remain read-only. The SQL layer has no
  DDL, transaction or privilege statement (the one session statement is `SET TIME ZONE`). Sequence generators
  (`NEXT VALUE FOR`), identity and default values, triggers, routines (`CREATE FUNCTION`, SQL/PSM), user-defined types,
  reference types, typed tables and `ONLY`, system-versioned tables (`FOR SYSTEM_TIME AS OF`) and periods declared on
  tables belong to those statements. PERIOD predicates on value pairs are supported.
- Other parts of ISO/IEC 9075: SQL/XML (Part 14), SQL/MDA multidimensional arrays (Part 15), SQL/PGQ property graph
  queries (Part 16), SQL/MED, SQL/OLB, SQL/JRT, SQL/CLI.

## Where torbles and DuckDB differ (on purpose)

torbles follows ISO where the two differ, and the comparisons in the tests skip these cases:

- `SIMILAR TO` uses ISO's regular expressions (`%`, `_`, `|`, `*`, `+`, `?`, `{}`, `()`, `[]` over the whole string),
  DuckDB's is a POSIX regular expression.
- A row comparison with a NULL field is UNKNOWN; DuckDB answers FALSE (or, for STRUCT values, compares NULL fields as
  equal), and DuckDB's `r IS NULL` is FALSE for a STRUCT whose fields are all NULL.
- `CORR` and the `REGR_*` functions are NULL when a variance is zero (DuckDB: NaN); every statistic is exact.
- `SUM` and `AVG` of DOUBLE values are the correctly rounded exact sum (and its quotient); DuckDB adds in double
  precision, so with large values of both signs the two can differ completely (1e301, 7 and -1e301: torbles 7, a
  double sum in that order 0).
- `FETCH FIRST p PERCENT` keeps the ceiling of p% of the rows; DuckDB's `LIMIT p%` the floor.
- `a[i]` outside 1 .. CARDINALITY(a) is an error (ISO); DuckDB returns NULL (its `list_extract` keeps that behavior).
- `0x1F` is the number 31 (SQL:2023); DuckDB reads `0` with the alias `x1F`.
- `CAST('abcd' AS VARCHAR(2))` is `'ab'` (ISO 6.13); DuckDB ignores the length.
- `TIMESTAMP '2020-01-01 10:00:00+02'` is a TIMESTAMP WITH TIME ZONE literal (ISO 5.3); DuckDB drops the zone. A zoned
  value keeps its own displacement: it prints with it and EXTRACT reads its fields in it; DuckDB shows zoned values in
  the session's zone. `x AT TIME ZONE z` gives the same instant with z's displacement, also for a TIMESTAMP WITHOUT
  TIME ZONE (taken in the session's zone first); DuckDB follows PostgreSQL there.
- `CORRESPONDING` keeps the columns every operand has (DuckDB's `UNION BY NAME` keeps all of them).
- The value shown for a group of strings equal under a collation is the least one (ISO leaves it to the
  implementation; DuckDB shows the first it meets).
- `ARG_MIN` / `ARG_MAX` with ties pick any of the tied rows in both, not always the same one.
- `PERCENTILE_CONT` (and `MEDIAN`) of a DECIMAL interpolates exactly, then truncates to the column's scale (2.50 and
  1000.00 at 0.3 of the way: 301.75); DuckDB interpolates the scaled integers in double precision (301.74).
- An argument of `COUNT(DISTINCT ...)` that overflows its type raises the overflow, as everywhere else; when the
  argument is an expression of the GROUP BY keys, DuckDB answers 1 per group without computing it (with its optimizer
  off too). With a SMALLINT x holding -17604, `SELECT x, count(DISTINCT x * 3) FROM t GROUP BY x` gives 1 in DuckDB
  and the overflow in torbles; both raise it without that GROUP BY, or for `sum(x * 3)`. The SQL fuzzer checks such an
  overflow against DuckDB's own arithmetic in that type and counts the statement among DuckDB's errors.
- A runtime error (an overflow, a failed cast) is raised where the statement computes the value. DuckDB's optimizer
  can leave such a value uncomputed and answer: its statistics prove a term constant (`abs(x) >= 2^40` for an INTEGER
  x) or a whole side of the plan empty (the right side of an EXCEPT whose left side is), or it runs filter terms in
  another order across joins and clauses. Which terms an implementation computes first is its own (ISO 9075-2,
  6.3.3.3); DuckDB without its optimizer raises the same errors. An aggregate's FILTER runs on the rows the WHERE
  keeps, and a LEFT JOIN's ON terms on the pairs its equality matches, so an error on a row the WHERE drops, or on a
  dimension row no row matches, is not the query's; DuckDB pushes an ON term on the dimension alone into the
  dimension's scan and raises it for every row. A WHERE term on one FROM item that fails on a row of it is checked
  again after the joins, beside the other terms, so an error on a row that a term on another item drops (a row a
  LEFT JOIN pads, say) is not the query's either.

## Tests

`tests/check_sql.py` compares about a hundred expressions and forty queries with DuckDB (every kind of join,
grouping sets, windows and frames, subqueries, CTEs, set operations), checks the ISO forms DuckDB lacks against values
worked out by hand, checks the Python API against its own SQL text run by DuckDB, and runs a short fuzzing session.
`tests/check_iso.py` covers the forms above: time zones, collations, samples, MATCH, PERIOD, extended CASE, intervals,
binary strings, arrays and multisets, LATERAL, SQL/JSON and its path language and JSON_TABLE plans, SEARCH / CYCLE,
MATCH_RECOGNIZE (the documented examples' results) and row pattern recognition in windows, partitioned joins, nested
window functions, row types (stored ones too), DECFLOAT, datetime precisions, the UNICODE collation, and a regression
test for every bug an independent review found, each against DuckDB where DuckDB has the same semantics.
`tests/check_arrays.py` runs random window queries (every window function, partitions, directions and NULL placement,
ROWS / RANGE / GROUPS frames with offsets, EXCLUDE, FILTER, IGNORE NULLS, FROM LAST) and random GROUP BY queries
(grouping sets, collations, DISTINCT, FILTER) on a table with ties, NULLs, NaN and infinities twice, as arrays and row by
row: the answers must be identical value for value, and equal to DuckDB's where the answer does not depend on how ties
are ordered. `tests/check_campaign_sql.py` keeps a regression test for each fault the release fuzzing campaign of
2026-11-02 found, and for the fuzzer's own checks. All of them run on the three engines (`TORBLES_SQL_ENGINE=alg |
plain | gpu`), from `tests/run_all.sh`.

`python -m torbles.sqlfuzz --minutes 10` (or `torbles-sqlfuzz --seed 7 --cases 200`) generates random typed star
schemas (unmatched and NULL references included) and random SQL: filters, expressions of every type, every kind of aggregate,
GROUP BY with grouping sets, HAVING, window functions (over groups, and over rows with partitions, RANGE and GROUPS
frames and exclusions), subqueries, CTEs, set operations, DISTINCT, ORDER BY / LIMIT; it
runs each statement on torbles and on DuckDB and compares the results as multisets (in order when ORDER BY fixes it,
DOUBLE values with a relative tolerance). Any difference, and any error other than a documented refusal, is a failure
with a replay command, once DuckDB has been asked again. When DuckDB without its statistics propagation, or without its
optimizer, gives torbles' answer, the difference is DuckDB's optimizer (the release campaign of 2026-11-02 found 64-bit
sums its statistics chose that wrapped, and a join filter it derived that kept a row the WHERE drops); `torbles.fuzz`
asks the same of its engine queries. An error torbles raises where DuckDB answered is DuckDB's when DuckDB without its
optimizer raises it too (above). Rows ordered by values that differ as documented above (a statistic NULL in torbles
and NaN in DuckDB, exact here and rounded there) and then cut by a LIMIT count as equal when the statement without
its LIMIT gives the same rows on both and torbles' rows are the first of its own. `TORBLES_SQL_ENGINE=alg | plain | gpu` picks the engine it runs on (alg by default),
`TORBLES_ALG_MIN_ROWS=0` and `TORBLES_SERIES_ROWS_PER_GROUP=1` apply the algebraic paths to its small tables, and
`TORBLES_FUZZ_DUCKDB_MEMORY=4GB` caps the comparison DuckDB's memory (its memory limit, and a statement that grows the
process by more than that is interrupted), so that a statement DuckDB cannot hold is one of its errors, not compared,
instead of taking the machine's memory.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).


### Durable SQL prototype

Create a database directory with a fixed catalog of writable uint32 tables.
Open the same directory without a catalog to recover it. Each client gets its
own connection and can use the same INSERT, UPDATE, DELETE and MERGE forms as
an in-memory writable table.

~~~python
import torbles

with torbles.open_durable(
    "inventory.torbles",
    tables={"stock": ({"id": [1, 2], "quantity": [10, 20]}, "id")},
) as db, db.connect() as client:
    client.sql("BEGIN")
    client.sql("UPDATE stock SET quantity = quantity - 1 WHERE id = 1")
    client.sql("INSERT INTO stock VALUES (3, 5)")
    client.sql("COMMIT")

with torbles.open_durable("inventory.torbles") as db, db.connect() as client:
    print(client.sql("SELECT SUM(quantity) FROM stock").rows)
~~~

Call SQL with one statement at a time. A write outside an explicit transaction
commits automatically. An explicit transaction sees its own changes. An
execution error aborts it, and closing its connection rolls it back. Use
the same thread for all statements of an explicit transaction and finish it
before closing the database. Other clients wait for that transaction to end.
The prototype serializes SQL execution, including read-only transactions.
One process can open a directory at a time on a local POSIX filesystem.

A commit writes one checksummed redo frame and waits for file fsync. Redo stores
evaluated values and primary keys, so recovery applies mutations through the
bitplane adapters without reevaluating SQL. Opening rebuilds the bitplanes
from the initial checkpoint and committed frames. Recovery drops an incomplete
final frame and rejects checksum failures in complete frames. If a commit
reports an I/O error, close and reopen the database to resolve its outcome.

Rollback of a transaction with writes also rebuilds from the initial checkpoint
and committed log. The log grows with committed changes until a future
checkpointing mechanism is added. Database creation writes the initial catalog
and data. The durable API currently exposes SQL over that catalog.
