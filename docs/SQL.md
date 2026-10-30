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

`db.sql(text, engine="alg")` runs on the bit planes; `"plain"` on the plain columnar baseline and `"gpu"` on the GPU
executor (or its CPU emulation) give the same answers. A script may hold several statements; the result is the last
one's. `SET TIME ZONE ...` sets the session's time zone; every other statement is a query.

## How a query runs

DuckDB's parser reads the text (syntax only: `json_serialize_sql`, no data and no binding). ISO forms its grammar does
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
or a code domain past 2^128, the engine returns the passing rows' bitmap instead and the atoms' values are read from
the planes for those rows.

**Other blocks** (outer and cross joins between arbitrary items, derived tables, CTEs, VALUES, table functions) give each
FROM item as a multiset: a stored table is an engine GROUP BY of the columns the block needs, with counts; a derived
table is its query's rows. The items are joined by hash joins on their distinct tuples, carrying the counts. A LATERAL
item (or `UNNEST(t.a)`, `JSON_TABLE(t.doc, ...)`) runs once per distinct tuple of the earlier columns it reads.

**Subqueries.** An uncorrelated subquery runs once: a scalar one becomes a constant, `IN` / `ANY` / `ALL` a constant
list with SQL's three-valued rules, so they reach the engine as plane predicates or code sets. A correlated one runs
once per distinct tuple of the outer values it reads.

**Errors** follow DuckDB's evaluate-on-use semantics: a failing cast or a division by zero fails the query only if the
value is used by a row that is output.

## Data types

| SQL type | values | stored columns |
|---|---|---|
| SMALLINT, INTEGER, BIGINT, HUGEINT, unsigned types | int (arithmetic never wraps; overflow is an error where DuckDB's is) | yes |
| DECIMAL(p, s), NUMERIC | Decimal, exact | yes |
| DECFLOAT(p) (T076; p defaults to 34) | Decimal rounded to p digits, with Infinity and NaN (NaN sorts last and equals itself) | no (the loaders read types through DuckDB, which has none) |
| REAL, DOUBLE PRECISION, FLOAT | float | yes (dictionary) |
| BOOLEAN | bool | yes |
| CHARACTER, VARCHAR, CLOB, NATIONAL CHARACTER ... | str (CHAR(n) is not padded, as in DuckDB) | yes (dictionary) |
| BINARY, VARBINARY, BLOB, BINARY LARGE OBJECT | bytes | yes (dictionary) |
| DATE, TIME, TIMESTAMP, TIME(p), TIMESTAMP(p) | date, time, datetime; with p < 6 fractional digits, a cast rounds half away from zero (as DuckDB's TIMESTAMP_S / _MS) | DATE, TIMESTAMP |
| TIME / TIMESTAMP [(p)] WITH TIME ZONE | an instant with its displacement (aware datetime) | TIMESTAMP WITH TIME ZONE |
| INTERVAL, INTERVAL <qualifier> (year-month and day-time) | months, days, microseconds (returned as timedelta); a literal or CAST with a qualifier is typed `INTERVAL DAY`, ..., and a single-field one casts to a number | no |
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
| inner, LEFT, RIGHT, FULL, CROSS, NATURAL joins; ON, USING | star blocks with inner / LEFT joins to dimensions: engine (LEFT as passes); others: hash joins of multisets |
| partitioned outer join `t PARTITION BY (c) RIGHT JOIN u ON ...` (F403) | values: each partition joined on its own, padded rows keep the partition's values |
| LATERAL derived tables (T491), UNNEST [WITH ORDINALITY] | values, once per distinct tuple of the outer columns read |
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
take DISTINCT field by field. A comparison of rows is the ISO one: a NULL field makes `=` UNKNOWN, and `r IS NULL` is
TRUE when every field is NULL. In Python: `F.row(a, b)`, `x.cast("ROW(f INTEGER, g VARCHAR)")`, `x.field("f")`.

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
`COALESCE`, `CAST`, `TRY_CAST`, row values `ROW (a, b)` and `(a, b)`, `COLLATE` (UCS_BASIC, BINARY, C, POSIX, NOCASE,
NOACCENT, NFC and their combinations: comparisons, ORDER BY, GROUP BY, DISTINCT, set operations, MIN / MAX, windows; in
a filter the collation key is evaluated on the column's distinct values), `CURRENT_USER`, `SESSION_USER`, `SYSTEM_USER`,
`USER`, `CURRENT_ROLE`, `CURRENT_CATALOG`, `CURRENT_SCHEMA`, `CURRENT_PATH`, `typeof`.

### Aggregates

| Aggregate | How |
|---|---|
| COUNT(*), COUNT(x), SUM, AVG of a polynomial in exact columns | engine: moment sums (popcount algebra), exact at any width |
| VAR_POP, VAR_SAMP, STDDEV_POP, STDDEV_SAMP, COVAR_POP, COVAR_SAMP, CORR, REGR_SLOPE, REGR_INTERCEPT, REGR_COUNT, REGR_R2, REGR_AVGX, REGR_AVGY, REGR_SXX, REGR_SYY, REGR_SXY | engine: moment sums (x, x², xy, ...), then exact rational arithmetic; the statistics are exact, not the running double sums DuckDB uses |
| MIN, MAX of a fact column | engine: plane elimination |
| SUM / AVG / COUNT of CASE, any aggregate with FILTER (WHERE ...) | engine: variants |
| MIN / MAX of other expressions, DISTINCT aggregates, EVERY, ANY, SOME, BOOL_AND / OR, BIT_AND / OR / XOR, ANY_VALUE (2023) | values: (group, value, count) tuples from the engine |
| PERCENTILE_CONT, PERCENTILE_DISC (... WITHIN GROUP (ORDER BY ...)), MODE, MEDIAN, quantiles | values |
| RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST (hypothetical, WITHIN GROUP) | values |
| LISTAGG (... ON OVERFLOW ...) WITHIN GROUP, STRING_AGG, ARRAY_AGG (... ORDER BY ...) | values |
| COLLECT, FUSION, INTERSECTION | values |
| JSON_ARRAYAGG (ORDER BY, NULL / ABSENT ON NULL), JSON_OBJECTAGG (WITH UNIQUE KEYS) | values |
| DuckDB's PRODUCT, ENTROPY, SKEWNESS, KURTOSIS, ARG_MIN / ARG_MAX, HISTOGRAM, APPROX_COUNT_DISTINCT (exact) | values |

### Window functions

`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `PERCENT_RANK`, `CUME_DIST`, `NTILE`, `LEAD`, `LAG`, `FIRST_VALUE`, `LAST_VALUE`,
`NTH_VALUE ... FROM FIRST / FROM LAST`, `RESPECT / IGNORE NULLS` (T611-T618), every aggregate over a window (with
DISTINCT and FILTER), frames `ROWS / RANGE / GROUPS` with numeric and interval offsets (T620), `EXCLUDE CURRENT ROW /
GROUP / TIES / NO OTHERS`, the WINDOW clause and windows defined from named windows, windows over aggregates, QUALIFY
(DuckDB). They run at the group level, after aggregation.

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

### Not supported, and why

- **Statements other than queries.** The database is built by the loaders and is read-only once saved, so there is no
  DDL, DML, transaction or privilege statement (the one session statement is `SET TIME ZONE`). Sequence generators
  (`NEXT VALUE FOR`), identity and default values, triggers, routines (`CREATE FUNCTION`, SQL/PSM), user-defined types,
  reference types, typed tables and `ONLY`, system-versioned tables (`FOR SYSTEM_TIME AS OF`) and periods declared on
  tables belong to those statements. PERIOD predicates on value pairs are supported.
- **The UNICODE collation** (the Unicode Collation Algorithm) needs a collation library; the code point collations and
  NOCASE / NOACCENT / NFC are supported.
- Other parts of ISO/IEC 9075: SQL/XML (Part 14), SQL/MDA multidimensional arrays (Part 15), SQL/PGQ property graph
  queries (Part 16), SQL/MED, SQL/OLB, SQL/JRT, SQL/CLI.

## Where torbles and DuckDB differ (on purpose)

torbles follows ISO where the two differ, and the comparisons in the tests skip these cases:

- `SIMILAR TO` uses ISO's regular expressions (`%`, `_`, `|`, `*`, `+`, `?`, `{}`, `()`, `[]` over the whole string),
  DuckDB's is a POSIX regular expression.
- A row comparison with a NULL field is UNKNOWN; DuckDB answers FALSE (or, for STRUCT values, compares NULL fields as
  equal), and DuckDB's `r IS NULL` is FALSE for a STRUCT whose fields are all NULL.
- `CORR` and the `REGR_*` functions are NULL when a variance is zero (DuckDB: NaN); every statistic is exact.
- `FETCH FIRST p PERCENT` keeps the ceiling of p% of the rows; DuckDB's `LIMIT p%` the floor.
- `a[i]` outside 1 .. CARDINALITY(a) is an error (ISO); DuckDB returns NULL (its `list_extract` keeps that behavior).
- `0x1F` is the number 31 (SQL:2023); DuckDB reads `0` with the alias `x1F`.
- `TIMESTAMP '2020-01-01 10:00:00+02'` is a TIMESTAMP WITH TIME ZONE literal (ISO 5.3); DuckDB drops the zone. A zoned
  value keeps its own displacement: it prints with it and EXTRACT reads its fields in it; DuckDB shows zoned values in
  the session's zone. `x AT TIME ZONE z` gives the same instant with z's displacement, also for a TIMESTAMP WITHOUT
  TIME ZONE (taken in the session's zone first); DuckDB follows PostgreSQL there.
- `CORRESPONDING` keeps the columns every operand has (DuckDB's `UNION BY NAME` keeps all of them).
- The value shown for a group of strings equal under a collation is the least one (ISO leaves it to the
  implementation; DuckDB shows the first it meets).
- `ARG_MIN` / `ARG_MAX` with ties pick any of the tied rows in both, not always the same one.

## Tests

`tests/check_sql.py` compares about a hundred expressions and forty queries with DuckDB (every kind of join,
grouping sets, windows and frames, subqueries, CTEs, set operations), checks the ISO forms DuckDB lacks against values
worked out by hand, checks the Python API against its own SQL text run by DuckDB, and runs a short fuzzing session.
`tests/check_iso.py` covers the forms above: time zones, collations, samples, MATCH, PERIOD, extended CASE, intervals,
binary strings, arrays and multisets, LATERAL, SQL/JSON and its path language, SEARCH / CYCLE, MATCH_RECOGNIZE (the
documented examples' results) and partitioned joins, each against DuckDB where DuckDB has the same semantics. Both run
on the three engines (`TORBLES_SQL_ENGINE=alg | plain | gpu`), from `tests/run_all.sh`.

`python -m torbles.sqlfuzz --minutes 10` (or `torbles-sqlfuzz --seed 7 --cases 200`) generates random typed star
schemas (unmatched and NULL references included) and random SQL: filters, expressions of every type, every kind of aggregate,
GROUP BY with grouping sets, HAVING, window functions, subqueries, CTEs, set operations, DISTINCT, ORDER BY / LIMIT; it
runs each statement on torbles and on DuckDB and compares the results as multisets (in order when ORDER BY fixes it,
DOUBLE values with a relative tolerance). Any difference, and any error other than a documented refusal, is a failure
with a replay command.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
