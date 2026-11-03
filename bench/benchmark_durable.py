# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""File-backed SQL refresh with a concurrent writer and analytic readers.

DuckDB and Torbles use the same data, statements, fixed batch sizes and WSL
filesystem. Setup, rollback checks, recovery and verification are outside
the concurrent interval. Run on an otherwise quiet machine.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from threading import Barrier, Thread
from time import perf_counter_ns

from benchmark_sql_dml import (DIM_ROWS, QUERY_SQL, ROOT, SEED, algq, arrays, checksum,
                               duckdb, make_operations, np, stage, timed, torbles, write_json)


ENGINES = ("duckdb", "torbles")
SCALES = (100_000, 1_000_000, 8_000_000)
READ_NAMES = tuple(QUERY_SQL)


def make_workload(n, batch, cycles):
    rng = np.random.default_rng(SEED + n)
    fact = dict(id=np.arange(n, dtype=np.uint32), **arrays(n, rng))
    fact["v"][0] = 16383
    region = rng.integers(0, 32, DIM_ROWS, dtype=np.uint32)
    region[0] = 31
    dim = dict(id=np.arange(DIM_ROWS, dtype=np.uint32), region=region)
    edit_ids = rng.choice(n, size=(cycles, 3, batch), replace=False).astype(np.uint32)
    operations = [make_operations(n, i, batch, ids, rng, False) for i, ids in enumerate(edit_ids)]
    tables = {"fact": (fact, "id"), "dim": (dim, "id")}
    for cycle in operations:
        for _, _, source, _, columns in cycle:
            tables[source] = columns, "id"
    return tables, operations, expected_snapshots(fact, region, operations)


def expected_snapshots(fact, regions, operations):
    """Independent exact deltas over the staged keys. Base edits are disjoint."""
    total_rows = len(fact["id"])
    chosen = fact["bucket"] < 8
    selected_count = np.zeros(DIM_ROWS, np.int64)
    selected_sum = np.zeros(DIM_ROWS, np.int64)
    np.add.at(selected_count, fact["dk"][chosen], 1)
    np.add.at(selected_sum, fact["dk"][chosen], fact["v"][chosen])
    joined = regions < 4

    def snapshot():
        count = int(selected_count.sum())
        return (total_rows, count, int(selected_sum.sum()) if count else None,
                int(selected_sum[joined].sum()) if selected_count[joined].any() else None)

    def add_rows(data, mask):
        nonlocal total_rows
        total_rows += int(mask.sum())
        mask &= data["bucket"] < 8
        np.add.at(selected_count, data["dk"][mask], 1)
        np.add.at(selected_sum, data["dk"][mask], data["v"][mask])

    snapshots = [snapshot()]
    for operations_in_cycle in operations:
        for action, _, _, _, batch in operations_in_cycle:
            ids = batch["id"]
            if action == "insert":
                add_rows(batch, np.ones(len(ids), dtype=bool))
            elif action == "delete":
                total_rows -= len(ids)
                changed = ids[chosen[ids]]
                np.add.at(selected_count, fact["dk"][changed], -1)
                np.add.at(selected_sum, fact["dk"][changed], -fact["v"][changed].astype(np.int64))
            else:
                base = ids < len(fact["id"])
                changed = ids[base]
                selected = chosen[changed]
                delta = batch["v"][base][selected].astype(np.int64) - fact["v"][changed[selected]]
                np.add.at(selected_sum, fact["dk"][changed[selected]], delta)
                if action == "merge":
                    add_rows(batch, ~base)
        snapshots.append(snapshot())
    return snapshots


@contextmanager
def opened(engine, path, threads, tables=None):
    if engine == "torbles":
        db = torbles.open_durable(path) if tables is None else torbles.open_durable(path, tables=tables)
    else:
        db = duckdb.connect(str(path), config={"threads": str(threads)})
    try:
        if engine == "duckdb" and tables is not None:
            for name, (columns, _) in tables.items():
                stage(db, {}, name, columns)
            db.execute("CHECKPOINT")
        yield db
    finally:
        db.close()


def connect(engine, db):
    return db.connect() if engine == "torbles" else db.cursor()


def execute(engine, client, sql):
    if engine == "duckdb":
        return client.execute(sql).fetchall(), {}
    result = client.sql(sql)
    return ([], {}) if result is None else (result.rows, result.stats)


def scalar(result):
    rows, _ = result
    assert len(rows) == 1 and len(rows[0]) == 1, rows
    return rows[0][0]


def current_snapshot(engine, client):
    return tuple(scalar(execute(engine, client, sql)) for sql in QUERY_SQL.values())


def rollback_check(engine, db, first, expected):
    client = connect(engine, db)
    try:
        execute(engine, client, "BEGIN")
        for _, _, _, sql, columns in first:
            assert scalar(execute(engine, client, sql)) == len(columns["id"])
        execute(engine, client, "ROLLBACK")
        assert current_snapshot(engine, client) == expected
    finally:
        client.close()


def concurrent_run(engine, db, operations, expected, readers, reader_mode):
    writer = []
    reads = [[] for _ in range(readers)]
    errors = []
    ready = Barrier(readers + 2, timeout=120)
    phase = Barrier(readers + 1, timeout=120)
    clients = [connect(engine, db) for _ in range(readers + 1)]

    def run_client(index):
        client = clients[index]
        try:
            ready.wait()
            for cycle, changes in enumerate(operations):
                phase.wait()
                started = perf_counter_ns()
                if index == 0:
                    _, begin_ms = timed(lambda: execute(engine, client, "BEGIN"))
                    timings, affected, evidence = {}, {}, {}
                    for name, _, _, sql, columns in changes:
                        result, timings[name] = timed(lambda: execute(engine, client, sql))
                        affected[name], evidence[name] = scalar(result), result[1]
                        assert affected[name] == len(columns["id"]), (engine, name, affected[name])
                    _, commit_ms = timed(lambda: execute(engine, client, "COMMIT"))
                    ended = perf_counter_ns()
                    writer.append(dict(cycle=cycle, start_ns=started, end_ns=ended,
                                       ms=(ended - started) / 1e6, begin_ms=begin_ms, commit_ms=commit_ms,
                                       statement_ms=timings, affected=affected, stats=evidence))
                else:
                    transactional = reader_mode == "transactions"
                    begin_ms = commit_ms = None
                    if transactional:
                        _, begin_ms = timed(lambda: execute(engine, client, "BEGIN"))
                    values, timings, evidence = [], {}, {}
                    for name, sql in QUERY_SQL.items():
                        result, timings[name] = timed(lambda: execute(engine, client, sql))
                        values.append(scalar(result))
                        evidence[name] = result[1]
                    if transactional:
                        _, commit_ms = timed(lambda: execute(engine, client, "COMMIT"))
                    ended = perf_counter_ns()
                    before, after = expected[cycle:cycle + 2]
                    if transactional:
                        assert tuple(values) in (before, after), (engine, cycle, values, before, after)
                    else:
                        assert all(value in (before[i], after[i]) for i, value in enumerate(values)), values
                    reads[index - 1].append(dict(cycle=cycle, start_ns=started, end_ns=ended,
                                                  ms=(ended - started) / 1e6, begin_ms=begin_ms,
                                                  commit_ms=commit_ms, statement_ms=timings,
                                                  answer=values, stats=evidence))
        except BaseException as error:
            errors.append(error)
            ready.abort()
            phase.abort()
        finally:
            try:
                client.close()
            except BaseException as error:
                errors.append(error)

    threads = [Thread(target=run_client, args=(i,), name=f"{engine}-{i}") for i in range(len(clients))]
    for thread in threads:
        thread.start()
    start = perf_counter_ns()
    try:
        ready.wait()
    finally:
        for thread in threads:
            thread.join()
    if errors:
        raise RuntimeError(f"{engine} concurrent worker failed") from errors[0]
    assert len(writer) == len(operations) and all(len(values) == len(operations) for values in reads)
    end = max(writer[-1]["end_ns"], *(records[-1]["end_ns"] for records in reads))
    overlap = sum(r["start_ns"] < w["end_ns"] and w["start_ns"] < r["end_ns"]
                  for records in reads for r, w in zip(records, writer))
    return dict(elapsed_s=(end - start) / 1e9, writer=writer, readers=reads, overlapping_reader_intervals=overlap)


def wal_bytes(engine, db, path):
    if engine == "torbles":
        return int(db.wal_bytes)
    wal = Path(str(path) + ".wal")
    return wal.stat().st_size if wal.exists() else 0


def final_check(engine, db, snapshot, operations):
    client = connect(engine, db)
    try:
        assert current_snapshot(engine, client) == snapshot
        by_action = {name: batch for name, _, _, _, batch in operations}
        for action in ("insert", "update", "delete", "merge"):
            first = int(by_action[action]["id"][0])
            value = None if action == "delete" else int(by_action[action]["v"][0])
            assert scalar(execute(engine, client, f"SELECT sum(v) FROM fact WHERE id = {first}")) == value
            assert scalar(execute(engine, client, f"SELECT count(*) FROM fact WHERE id = {first}")) == (value is not None)
    finally:
        client.close()


def run_engine(engine, path, tables, operations, expected, args):
    with opened(engine, path, args.threads, tables) as db:
        rollback_check(engine, db, operations[0], expected[0])
        before = wal_bytes(engine, db, path)
        run = concurrent_run(engine, db, operations, expected, args.readers, args.reader_mode)
        run["wal_before_bytes"] = before
        run["wal_after_bytes"] = wal_bytes(engine, db, path)
        final_check(engine, db, expected[-1], operations[-1])
    started = perf_counter_ns()
    with opened(engine, path, args.threads) as db:
        run["reopen_ms"] = (perf_counter_ns() - started) / 1e6
        final_check(engine, db, expected[-1], operations[-1])
    run["files_after_close_bytes"] = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    return run


def describe_samples(samples):
    return dict(p50_ms=float(statistics.median(samples)), p95_ms=float(np.percentile(samples, 95)))


def summarize_runs(runs):
    writes = [sample for run in runs for sample in run["writer"]]
    reads = [sample for run in runs for reader in run["readers"] for sample in reader]
    elapsed = [run["elapsed_s"] for run in runs]
    total = sum(elapsed)
    return dict(median_elapsed_s=statistics.median(elapsed), round_elapsed_s=elapsed,
                committed_batches_per_s=len(writes) / total,
                affected_rows_per_s=sum(sum(w["affected"].values()) for w in writes) / total,
                analytic_statements_per_s=len(reads) * len(READ_NAMES) / total,
                writer_batch=describe_samples([w["ms"] for w in writes]),
                writer_begin=describe_samples([w["begin_ms"] for w in writes]),
                writer_commit=describe_samples([w["commit_ms"] for w in writes]),
                writer_statements={name: describe_samples([w["statement_ms"][name] for w in writes])
                                   for name in writes[0]["statement_ms"]},
                reader_batch=describe_samples([r["ms"] for r in reads]),
                reader_begin=describe_samples([r["begin_ms"] for r in reads]) if reads[0]["begin_ms"] is not None else None,
                reader_statements={name: describe_samples([r["statement_ms"][name] for r in reads]) for name in READ_NAMES},
                overlapping_reader_intervals=sum(run["overlapping_reader_intervals"] for run in runs),
                reader_intervals=len(reads),
                wal_before_bytes=[run["wal_before_bytes"] for run in runs],
                wal_after_bytes=[run["wal_after_bytes"] for run in runs],
                files_after_close_bytes=[run["files_after_close_bytes"] for run in runs],
                reopen_ms=[run["reopen_ms"] for run in runs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=SCALES)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--readers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reader-mode", choices=("transactions", "autocommit"), default="transactions")
    parser.add_argument("--scratch", type=Path, default=Path.home())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.cycles, args.readers, args.threads, args.rounds) <= 0 or min(args.rows) < 3 * args.batch * args.cycles:
        parser.error("use positive counts and enough base rows for disjoint mutations")
    if args.output.exists() or len(args.rows) != len(set(args.rows)):
        parser.error("use a new output path and distinct sizes")
    if max(args.rows) + args.cycles * (2 * args.batch - args.batch // 2) > 1 << 32:
        parser.error("inserted keys must fit uint32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torbles.set_threads(args.threads)
    torbles.set_build_threads(args.threads)
    record = dict(state="running", args={**vars(args), "scratch": str(args.scratch), "output": str(args.output)},
                  threads=args.threads, duckdb=duckdb.__version__, torbles=torbles.__version__,
                  source_head=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  sha256=checksum(__file__), native_library_sha256=checksum(algq.lib._name),
                  source_hashes={name: checksum(ROOT / "torbles" / name)
                                 for name in ("durable.py", "sqldmlparse.py", "sqlwrite.py", "refresh.py")},
                  affinity=sorted(os.sched_getaffinity(0)),
                  method="Per-cycle barrier starts concurrent client attempts. P50/p95 are descriptive pooled latencies "
                         "across fresh stores. Throughput divides all operations by summed elapsed. Torbles serializes "
                         "explicit transactions, so overlapping call intervals can include lock waits.", scales={})
    write_json(args.output, record)
    try:
        for i, size in enumerate(args.rows):
            tables, operations, expected = make_workload(size, args.batch, args.cycles)
            scale = dict(expected=expected, engines={engine: {"rounds": []} for engine in ENGINES})
            record["scales"][str(size)] = scale
            for round_number in range(args.rounds):
                for engine in ENGINES if (i + round_number) % 2 == 0 else ENGINES[::-1]:
                    with tempfile.TemporaryDirectory(prefix=f"durable-{engine}-{size}-", dir=args.scratch) as temporary:
                        path = Path(temporary) / ("work.duckdb" if engine == "duckdb" else "work.torbles")
                        run = run_engine(engine, path, tables, operations, expected, args)
                    scale["engines"][engine]["rounds"].append(run)
                    print(f"{size:,} round {round_number + 1} {engine}: {run['elapsed_s']:.3f}s, verified after reopen", flush=True)
                    write_json(args.output, record)
            for engine in ENGINES:
                values = scale["engines"][engine]
                values["summary"] = summarize_runs(values["rounds"])
        record["state"] = "completed"
    except BaseException:
        record["state"] = "failed"
        raise
    finally:
        write_json(args.output, record)


if __name__ == "__main__":
    main()
