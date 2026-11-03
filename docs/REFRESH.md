# In-memory bitplane refresh prototype

torbles.refresh.RefreshTable is an experimental API for fixed-schema nonnullable uint32 columns. Each table starts with implicit logical IDs from zero. Append assigns new IDs and returns their range. Update replaces specified columns and preserves the logical IDs. Delete clears visibility. Each operation holds the table lock and rejects invalid IDs before making visible changes.

Appends build native bitplane segments. The default segments update mode adds replacement segments. A filtered read intersects the predicate mask with the segment's visibility mask. Exact count, sum and grouped sum combine the visible reductions across segments. Unfiltered count reads maintained live-row metadata. The CUDA executor uploads a changed visibility bitmap and a new segment when a query first needs them.

Compaction rebuilds live rows into one native segment and preserves logical IDs. If deleted IDs leave holes, the compacted segment currently retains an eight-byte ID per live row. The stats property reports native torus bytes, visible and physical rows, visibility bitmap bytes, ID-array bytes, sparse override entries and the uncompressed value bytes rewritten by the most recent compaction. These counters cover named structures rather than total process memory.

With update_mode="planes", a batch that fits the existing encoded widths modifies the owned segments directly. The store reads the old codes of the changed rows, computes old XOR new and patches only those row bits. The native batch updates value and parity planes together. An already resident CUDA segment receives the same sparse deltas. A failed device patch discards the GPU copy so the next read uploads the committed CPU state. A batch needing a wider code uses the replacement-segment path.

Each refresh store serializes its own operations. Durable commits, crash recovery and atomic transactions across tables remain outside its scope.

## Benchmark

Run from the repository root after building the CPU and CUDA libraries.

    PYTHONPATH=. python bench/benchmark_refresh.py --output /tmp/torbles-refresh.json

The workload models a TPC-DS-style refresh and includes explicit fact and dimension updates. A cycle appends a fact batch, replaces fact measures, deletes a different fact batch, and updates a separate dimension store. DuckDB uses in-memory indexed IDs and the same data and mutations. The join check groups fact sums by dimension ID and applies the dimension filter to those results. It also verifies a filtered count and a selective sum after every cycle, then compares before and after compaction. The JSON records first calls and repeated warm reads separately. Results describe this workload and fixture.
Add --update-mode planes to benchmark the in-place path. Add --updates-only to measure the cost of fact and dimension updates without adding insert segments.

## SQL DML

Register an in-memory table, then use ordinary Database.sql:

    db = torbles.Database()
    db.add_writable_table("fact", {"id": [1, 2], "v": [20, 40], "bucket": [0, 1], "dk": [3, 4]})
    db.sql("INSERT INTO fact VALUES (3, 25, 1, 4)")
    db.sql("UPDATE fact SET v = v + 1 WHERE id = 1")
    db.sql("DELETE FROM fact WHERE id = 2")

Every writable table has fixed, nonnullable uint32 columns and one immutable primary key. Stage tables can be ordinary loaded
tables, writable tables, or a supported derived relation. INSERT accepts VALUES and SELECT and supplies every target column.
UPDATE and DELETE accept predicates, including correlated subqueries through the existing query evaluator. UPDATE FROM and
DELETE USING accept one source. A simple equality from the target primary key to a source column uses indexed row lookup.
MERGE uses that equality and supports one unconditional matched UPDATE or DELETE and one unconditional not-matched INSERT.
Conditional MERGE branches, RETURNING, defaults, foreign-key enforcement and persistence of writable tables are outside this
prototype. Each successful statement returns one Count row. Statements in a script are applied in order.

## Standard SQL surface

SQL/Foundation (ISO/IEC 9075-2) defines INSERT, UPDATE, DELETE and MERGE. A batch statement set can stage the inserted rows, replacements and deleted IDs and then use these forms. The replacement staging table should have a unique ID per row.

    INSERT INTO fact (id, v, bucket, dk)
      SELECT id, v, bucket, dk FROM insert_batch;

    UPDATE fact
       SET v = (SELECT u.v FROM update_batch u WHERE u.id = fact.id)
     WHERE EXISTS (SELECT 1 FROM update_batch u WHERE u.id = fact.id);

    DELETE FROM fact
     WHERE EXISTS (SELECT 1 FROM delete_batch d WHERE d.id = fact.id);

    MERGE INTO fact AS f USING upsert_batch AS u ON f.id = u.id
      WHEN MATCHED THEN UPDATE SET v = u.v
      WHEN NOT MATCHED THEN INSERT (id, v, bucket, dk)
        VALUES (u.id, u.v, u.bucket, u.dk);

These forms run through the SQL adapter. The matched SQL benchmark uses UPDATE FROM and DELETE USING for its staged fast path,
plus INSERT SELECT and MERGE. TPC-DS supplies a benchmark workload and refresh data rather than a universal statement suite.

    PYTHONPATH=. python bench/benchmark_sql_dml.py campaign --output /tmp/torbles-sql-dml

This benchmark stages identical batches outside the timers and times SQL execution on both engines. It verifies affected counts,
touched rows and aggregate reads after each cycle. It is an in-memory workload inspired by data maintenance rather than an
official TPC result.
