# Torbles

Torbles is an **algebraic relational datastore** for SQL and mathematical queries.
It stores columns as bitplanes and evaluates filters, joins, and aggregates over
them on the CPU or, for supported plans, on CUDA.

**0.6.0 alpha** runs on x86-64 Linux and WSL2.

## What’s new in 0.6.0

- **Packed SQL stages on CPU and CUDA.** Eligible plans reduce raw bitplanes for
  exact scalar moments, distinct counts, grouping, bounded ordering, and
  dictionary histograms. SQL can also rank groups before reading their remaining
  aggregate inputs. [SQL execution](docs/SQL.md#how-a-query-runs)
- **Reusable SQL algebra.** Compatible SELECTs can share an exact moment
  calculation in one batch. A saved catalog can also answer new affine
  aggregates and combine learned filter scopes, with source validation before
  reuse. [Learned algebra](docs/SQL.md#durable-learned-algebra)
- **SQL writes and local transactions.** Writable uint32 tables support INSERT,
  UPDATE, DELETE and primary-key MERGE. The durable prototype supports
  explicit transactions, autocommit and crash recovery.
  [Transactions](docs/SQL.md#durable-sql-prototype) · [Refresh guide](docs/REFRESH.md)

## ClickBench

All 43 queries matched DuckDB on official 1M- and 10M-row subsets. Independent
Nsight traces recorded CUDA launches in every query at both scales.

| Rows | Torbles GPU | Torbles CPU | DuckDB |
|---:|---:|---:|---:|
| 1M | 72.6 ms | 114.7 ms | 229.2 ms |
| 10M | 420.2 ms | 859.2 ms | 1,155.1 ms |

These totals sum warm medians. The CPU comparators come from earlier runs on
the same inputs and hardware. Another 10M GPU run with the same native library
took 1,163.9 ms. [Method, CUDA coverage and separate workload benchmarks](docs/CLICKBENCH.md)

## Try it

Use Python 3.11 or newer, Git, and C and C++ compilers. The source install builds
the CPU engine and uses `nvcc` when available.

```sh
python -m pip install 'git+https://github.com/brianramos/torbles.git@v0.6.0'
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
