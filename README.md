# Torbles

Torbles is an **algebraic relational datastore** for SQL and mathematical queries.
It stores columns as bitplanes and evaluates filters, joins, and aggregates over
them on the CPU or, for supported plans, on CUDA.

**0.9.0 alpha** runs on x86-64 Linux and WSL2.

## What’s new in 0.9.0

- **Topology-compressed snapshots.** Save with `compression="topology"` to
  encode repeating torus edges as exact sector motifs. Eligible CPU and CUDA
  SQL plans use the stored cycles and keep clean snapshots compact on reopen.
- **Verified hierarchy joins.** Store proof metadata in the manifest so eligible
  ancestor filters can run on the fact table’s foreign-key coordinate bits.
- **Sparse reads.** Gather selected rows directly from bitplanes on read-only
  tables and use primary-key lookups on writable tables. Ordered limits can
  gather the winning rows’ payload in batches by physical part.
- **Packed text and reusable referrer indexes.** Reopen with `packed_text=True`
  to map UTF-8 dictionaries and decode returned strings on demand. A persisted
  index supplies exact code counts and a registered referrer projection for
  eligible CPU and CUDA group reductions.
- **Cold query planning.** Answer eligible row counts from metadata and compare
  native CPU filtering and grouping against the cost of a cold CUDA upload.

[SQL paths and supported shapes](docs/SQL.md) · [Changelog](CHANGELOG.md)

## ClickBench at 100M rows

All 43 queries verified against native DuckDB on all 11 configurations.
The AC run used the same WSL2 laptop with an RTX 5090 Laptop GPU and
99,997,497 rows. First totals sum the first execution of each query. Warm
totals sum the median of three warm executions per query. Setup is separate.

| Engine | Layout | Version | First (s) | Warm (s) |
|---|---|---|---:|---:|
| Torbles CUDA | Mapped bitplanes with referrer index | 0.9.0 (2b2b68b) | 41.656 | **2.874** |
| Umbra | Native database | v0.2-3356-g92839196f2 | 47.711 | 4.413 |
| CedarDB | Native database | v2026-10-22 | 48.045 | 8.565 |
| DuckDB | Native database | 2.0.0 | 18.322 | 13.987 |
| Hyper | Native extract | 0.0.26963 | 58.871 | 14.487 |
| Torbles CPU | Mapped bitplanes with referrer index | 0.9.0 (2b2b68b) | 22.061 | 19.890 |
| DuckDB | Parquet † | 2.0.0 | 30.921 | 27.402 |
| StarRocks | Native database | 4.0.14-7c18dd9 | 35.943 | 31.079 |
| chDB | MergeTree | 4.4.0 / ClickHouse 26.9.2.1 | 39.036 | 34.168 |
| DataFusion | Parquet † | 55.0.0 | 54.080 | 52.610 |
| Sirius | Streamed Parquet | 76ef37d231 / DuckDB 2.0.0 | 130.452 | 115.104 |

† Direct-Parquet DuckDB and DataFusion ran alongside a separate learning job.

Verified replays observed CUDA execution on 41 Torbles queries. Q1 used
metadata and Q41 ran on the CPU.

### Fastest verified Torbles 0.9.0 sweeps

These separate AC sweeps each verified all 43 queries. Each row pairs the
first and warm totals from one run. Archive residency is measured in Linux
at process start. The prebuilt referrer index contains a regex projection and
per-code counts. Index build time is outside the query timer.

| Torbles backend | Referrer index | Run | Archive pages resident at start (GiB) | First (s) | Warm (s) |
|---|---|---|---:|---:|---:|
| CUDA | Enabled | Index control | 24.77 | **9.409** | 2.587 |
| CUDA | Enabled | Resource profile | 0.00 | 13.546 | **2.377** |
| CUDA | Disabled | Resource profile | 22.75 | **10.290** | 2.626 |
| CUDA | Disabled | Index control | 22.72 | 10.783 | **2.599** |
| CPU | Enabled | Resource profile | 22.72 | **17.415** | **15.665** |
| CPU | Disabled | Resource profile | 24.78 | **33.549** | **17.518** |

[Index controls](docs/CLICKBENCH-0.9.0.md#referrer-index-control) · [Resource sweeps](docs/CLICKBENCH-0.9.0.md#host-device-and-storage-profile) · [Full-panel method and per-query results](docs/CLICKBENCH-0.9.0.md) · [0.8.0 panel with 2.061 s CUDA warm](docs/CLICKBENCH-0.8.0.md)

## Try it

Use Python 3.11 or newer, Git, and C and C++ compilers. The source install builds
the CPU engine and uses `nvcc` when available.

```sh
python -m pip install 'git+https://github.com/brianramos/torbles.git@v0.9.0'
```

This example creates three tables in memory. The region filter reaches the sales
table through the customer key.

```python
import duckdb
import torbles

source = duckdb.connect()
source.execute("""
    CREATE TABLE region(id INTEGER PRIMARY KEY, name VARCHAR);
    CREATE TABLE customer(id INTEGER PRIMARY KEY, region_id INTEGER REFERENCES region(id));
    CREATE TABLE sale(customer_id INTEGER REFERENCES customer(id),
                      units INTEGER NOT NULL, amount INTEGER NOT NULL);
    INSERT INTO region VALUES (1, 'North'), (2, 'South');
    INSERT INTO customer VALUES (10, 1), (20, 2);
    INSERT INTO sale VALUES (10, 2, 12), (20, 1, 7), (10, 1, 6);
""")
db = torbles.from_duckdb(source)

result = db.sql("""
    SELECT SUM(s.amount) FROM sale s
    JOIN customer c ON s.customer_id = c.id
    JOIN region r ON c.region_id = r.id
    WHERE r.name = 'North'
""")
assert result.rows == [(18,)]
assert result.stats["snowflake_blocks"] == 1
```

The sales values can also produce a function of a parameter. This example
loads them as a numeric observation table, builds a degree-two moment basis
once, then evaluates a loss and its derivative.

```python
from torbles.mathematics import MomentBasis

db.close()
source.execute("CREATE TABLE observation AS SELECT units, amount FROM sale")
db = torbles.from_duckdb(source, tables=["observation"])

with MomentBasis(db.query(db["observation"]), ["units", "amount"]) as basis:
    units, amount = (basis.variables[name] for name in ("units", "amount"))
    price = basis.parameter("price")
    loss = basis.reduce((amount - price * units) ** 2)
    assert loss.evaluate(price=5) == 9
    with loss.diff("price") as gradient:
        assert gradient.evaluate(price=5) == -14
    loss.close()

db.close()
source.close()
```

## Feature outline

- **Bitplane execution.** Boolean filters, weighted popcount sums, and grouped
  reductions expressed as mask-by-measure matrix products. [Engine internals](docs/ENGINE.md)
- **Keys and joins.** Derives hierarchical key layouts from functional
  dependencies. A dimension filter can become an expression over the fact
  table’s foreign-key bits. Supports native range and inequality joins in
  eligible plans. [Engine internals](docs/ENGINE.md)
- **SQL and Python.** SQL joins, CTEs, grouping sets, window functions, and typed
  expressions. Python APIs for constructing queries and loading data from files,
  data frames, or DuckDB. [SQL guide](docs/SQL.md)
- **Mathematical queries.** Exact moment states and bounded rational functions
  with parameter differentiation, integration, and substitution. [Mathematical API](docs/MATHEMATICS.md)
- **Relationship discovery.** Opt-in modules discover and check relationships on
  a declared data scope. The planner can compare chained execution with a
  combined algebraic system. [Discovery example](examples/relationship_demo.py) · [Planner example](examples/system_planner_demo.py)
- **Persistence and integrity.** Saves bitplanes with XOR check faces. Loading
  validates the faces and can repair single-bit faults. [Engine internals](docs/ENGINE.md)

## Evidence and limits

- [Star Schema Benchmark](docs/SSB.md) documents the queries, hardware, and reproduction commands. [Historical release timings](https://github.com/brianramos/torbles/blob/f493619883100849e20abc3fb65a1bf3ee354975/README.md#results) include separate SQL and engine API measurements.
- [Algebra inventory](ALGEBRA_INVENTORY.md) links individual capabilities to their bounds, tests, and benchmark receipts.
- The test suite and differential fuzzers compare results against DuckDB. [Regression suite](tests/run_all.sh) · [Snowflake checks](tests/check_snowflake.py)
- Built snapshots support reads. SQL writes use the writable-table API. The native paths have limits on key shapes, arithmetic, and columns per pass. SQL uses general execution for other supported plans, and some GPU requests use CPU stages. [SQL coverage](docs/SQL.md) · [Engine limits](docs/ENGINE.md)

Apache 2.0. [License](LICENSE) · [Changelog](CHANGELOG.md)
