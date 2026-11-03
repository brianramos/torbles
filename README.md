# Torbles

Torbles is an **algebraic relational datastore** for SQL and mathematical queries.
It stores columns as bitplanes and evaluates filters, joins, and aggregates over
them on the CPU or, for supported plans, on CUDA.

**Current release is 0.4.0 alpha.** Runs on x86-64 Linux and WSL2.

## What’s new in 0.4.0

- **Wide tables.** One logical table can span several row-aligned torbles. The
  loader can store attributes that depend on a key once per key value. SQL still
  sees those attributes as columns of the original table. [Storage design](docs/ENGINE.md#tables-wider-than-a-torble-040)
- **Snowflake filter pushdown.** SQL can carry ancestor filters through eligible
  chains of declared integer foreign keys. Filter-only branches feed the native
  fact query. Queries that return ancestor values filter the fact first and then
  join for those values. [SQL execution](docs/SQL.md#how-a-query-runs)

## Try it

Use Python 3.11 or newer, Git, and C and C++ compilers. The source install builds
the CPU engine and uses `nvcc` when available.

```sh
python -m pip install 'git+https://github.com/brianramos/torbles.git@v0.4.0'
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
- SQL is read-only after the database is built. The native paths have limits on key shapes, arithmetic, and columns per kernel pass. SQL uses general execution for other supported plans. Some GPU requests run on the CPU. [SQL coverage](docs/SQL.md) · [Engine limits](docs/ENGINE.md)

Apache 2.0. [License](LICENSE) · [Changelog](CHANGELOG.md)
