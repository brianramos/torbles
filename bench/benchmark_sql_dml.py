# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Matched in-memory DuckDB and Torbles SQL DML over fixed-size refresh batches.

Use a quiet Linux machine with the CUDA executor built. Timed calls include
SQL parsing, planning, execution and result materialization. Data generation,
table loading, staging and exact verification are outside the timers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torbles
from torbles import algq
from benchmark_refresh import DIM_ROWS, QUERY_SQL, arrays, timed


ENGINES = ("duckdb", "alg", "gpu")
FACT = ("id", "v", "bucket", "dk")
DIM = ("id", "region")
SCALES = (100_000, 1_000_000, 8_000_000, 32_000_000)
SEED = 20261103


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity():
    package = Path(torbles.__file__).resolve().parent
    cpu = Path(algq.lib._name).resolve()
    gpu = Path(algq.gpu_lib()._name).resolve()
    assert package == ROOT / "torbles" and cpu == package / "libalg.so" and gpu == package / "libgpu.so"
    device = torbles.gpu_name()
    if "emulat" in device.lower() or "no CUDA" in device:
        raise RuntimeError("The benchmark requires an actual CUDA executor")
    names = ("database.py", "sqldml.py", "sqldmlparse.py", "sqlwrite.py", "sqlexec.py", "refresh.py", "algq.py",
             "alg.c", "gpu.cu", "libalg.so", "libgpu.so")
    return dict(head=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                files={name: checksum(package / name) for name in names}, harness_sha256=checksum(__file__),
                recipe_sha256=checksum(ROOT / "bench/benchmark_refresh.py"), duckdb=duckdb.__version__,
                torbles=torbles.__version__, numpy=np.__version__, python=sys.version, gpu=device,
                affinity=sorted(os.sched_getaffinity(0)))


def stage(con, engines, name, columns):
    """Give all paths the same materialized staging relation and PK constraint."""
    ddl = ", ".join(f"{column} UINTEGER" + (" PRIMARY KEY" if column == "id" else " NOT NULL")
                    for column in columns)
    con.execute(f"CREATE TABLE {name} ({ddl})")
    con.register("stage_input", pd.DataFrame(columns, copy=False))
    try:
        con.execute(f"INSERT INTO {name} SELECT * FROM stage_input")
    finally:
        con.unregister("stage_input")
    for db in engines.values():
        db.add_writable_table(name, columns, primary_key="id")


def rotate(index):
    offset = index % len(ENGINES)
    order = ENGINES[offset:] + ENGINES[:offset]
    return order[::-1] if index % 2 else order


def execute(con, engines, engine, sql):
    if engine == "duckdb":
        return con.execute(sql).fetchall(), {}
    result = engines[engine].sql(sql, engine=engine)
    return result.rows, result.stats


def query_samples(con, engines, repeats, order_index):
    record = {}
    for q, (name, sql) in enumerate(QUERY_SQL.items()):
        want, _ = execute(con, engines, "duckdb", sql)
        samples = {engine: [] for engine in ENGINES}
        evidence = {engine: [] for engine in engines}
        # Warm the current visibility masks and any new CUDA segments outside timings.
        for engine in engines:
            got, _ = execute(con, engines, engine, sql)
            assert got == want, (name, engine, got, want)
        for repeat in range(repeats):
            for engine in rotate(order_index + repeat + q):
                (got, stats), ms = timed(lambda: execute(con, engines, engine, sql))
                assert got == want and ms > 0, (name, engine, got, want)
                samples[engine].append(ms)
                if engine != "duckdb":
                    assert stats.get("writable_native_blocks") == 1, (name, engine, stats)
                    assert not stats.get("writable_scan_rows"), (name, engine, stats)
                    if name == "join_sum":
                        assert stats.get("writable_native_joins") == 1, stats
                    evidence[engine].append(stats)
        traces = {}
        for engine in engines:
            calls = []
            original = algq.Query._exec

            def trace(query, device, table, plan):
                ok = original(query, device, table, plan)
                calls.append(dict(requested=device, accepted=bool(ok), actual=query.device,
                                  kernel_ms=query.kernel_ms, table=query.fact.name))
                return ok

            with patch.object(algq.Query, "_exec", trace):
                got, _ = execute(con, engines, engine, sql)
            assert got == want
            traces[engine] = calls
        record[name] = dict(answer=want, warm_ms=samples, stats=evidence, dispatch=traces)
    return record


def make_operations(n, index, batch, edit_ids, rng, dimension_updates):
    match_count = batch // 2
    insert_count = batch - match_count
    first = n + index * (batch + insert_count)
    insert_ids = np.arange(first, first + batch, dtype=np.uint32)
    merge_ids = np.concatenate((edit_ids[2, :match_count],
                                np.arange(first + batch, first + batch + insert_count, dtype=np.uint32)))
    sources = {
        "insert": dict(id=insert_ids, **arrays(batch, rng)),
        "update": dict(id=edit_ids[0], v=rng.integers(0, 16384, batch, dtype=np.uint32)),
        "delete": dict(id=edit_ids[1]),
        "merge": dict(id=merge_ids, **arrays(batch, rng)),
    }
    commands = {
        "insert": "INSERT INTO fact (id,v,bucket,dk) SELECT id,v,bucket,dk FROM {s}",
        "update": "UPDATE fact AS f SET v = s.v FROM {s} AS s WHERE f.id = s.id",
        "delete": "DELETE FROM fact AS f USING {s} AS s WHERE f.id = s.id",
        "merge": "MERGE INTO fact AS f USING {s} AS s ON f.id = s.id "
                 "WHEN MATCHED THEN UPDATE SET v = s.v "
                 "WHEN NOT MATCHED THEN INSERT (id,v,bucket,dk) VALUES (s.id,s.v,s.bucket,s.dk)",
    }
    if dimension_updates:
        sources["dimension_update"] = dict(id=rng.choice(DIM_ROWS, 32, replace=False).astype(np.uint32),
                                            region=rng.integers(0, 32, 32, dtype=np.uint32))
        commands["dimension_update"] = "UPDATE dim AS d SET region = s.region FROM {s} AS s WHERE d.id = s.id"
    return [(name, "dim" if name == "dimension_update" else "fact", f"stage_{name}_{index}",
             commands[name].format(s=f"stage_{name}_{index}"), data) for name, data in sources.items()]


def check_changed_rows(con, engines, target, keys):
    columns = FACT if target == "fact" else DIM
    keys = np.unique(keys)
    con.register("check_keys", pd.DataFrame({"id": keys}, copy=False))
    try:
        sql = f"SELECT {','.join('t.' + name for name in columns)} FROM {target} t JOIN check_keys k ON t.id=k.id ORDER BY t.id"
        want = con.execute(sql).fetchall()
    finally:
        con.unregister("check_keys")
    for engine, db in engines.items():
        table = db[target]
        positions = table.lookup(keys)
        actual = table.store.read(positions[positions >= 0], columns)
        got = list(zip(*(actual[name].tolist() for name in columns)))
        assert got == want, (target, engine, len(got), len(want))
    return len(want)


def worker(args):
    before = identity()
    torbles.set_threads(args.threads)
    torbles.set_build_threads(args.threads)
    rng = np.random.default_rng(SEED + args.size)
    initial = dict(id=np.arange(args.size, dtype=np.uint32), **arrays(args.size, rng))
    initial["v"][0] = 16383
    dimension = dict(id=np.arange(DIM_ROWS, dtype=np.uint32), region=rng.integers(0, 32, DIM_ROWS, dtype=np.uint32))
    dimension["region"][0] = 31
    ids = rng.choice(args.size, size=(args.cycles, 3, args.batch), replace=False).astype(np.uint32)
    engines = {engine: torbles.Database() for engine in ENGINES[1:]}
    record = dict(identity=before, size=args.size, round=args.round, cycles=[], queries={})
    try:
        with duckdb.connect(config={"threads": str(args.threads)}) as con:
            stage(con, engines, "fact", initial)
            stage(con, engines, "dim", dimension)
            del initial
            record["queries"]["initial"] = query_samples(con, engines, args.repeats, args.round)
            for i, edit_ids in enumerate(ids):
                operations = make_operations(args.size, i, args.batch, edit_ids, rng, args.dimension_updates)
                changes = {}
                for _, _, name, _, columns in operations:
                    stage(con, engines, name, columns)
                for op, (name, target, _, sql, columns) in enumerate(operations):
                    expected_count = len(columns["id"])
                    result = changes[name] = dict(sql=sql, affected=expected_count, elapsed_ms={}, stats={}, store_stats={})
                    for engine in rotate(i + op + args.round):
                        (got, stats), ms = timed(lambda: execute(con, engines, engine, sql))
                        assert got == [(expected_count,)] and ms > 0, (name, engine, got)
                        result["elapsed_ms"][engine] = ms
                        if engine != "duckdb":
                            assert stats.get("writable_affected_rows") == expected_count, stats
                            if name != "insert":
                                assert stats.get("writable_key_lookup") == 1, (name, stats)
                            result["stats"][engine] = stats
                            result["store_stats"][engine] = engines[engine][target].store.stats
                checked = {}
                for target in ("fact", "dim"):
                    keys = [columns["id"] for _, table, _, _, columns in operations if table == target]
                    if keys:
                        checked[target] = check_changed_rows(con, engines, target, np.concatenate(keys))
                queries = query_samples(con, engines, args.repeats, i + args.round + 1)
                assert queries["count"]["answer"] == [(args.size + (i + 1) * (args.batch - args.batch // 2),)]
                record["cycles"].append(dict(index=i, dml=changes, queries=queries, checked_rows=checked))
                print(f"{args.size:,} rows round {args.round + 1} cycle {i + 1}: all results match", flush=True)
    finally:
        for db in engines.values():
            db.close()
    assert identity() == before
    write_json(args.output, record)


def summarize(directory, manifest):
    summary = []
    for size in manifest["rows"]:
        receipts = []
        for job in manifest["jobs"]:
            if job["size"] != size:
                continue
            path = directory / job["receipt"]
            assert checksum(path) == job["sha256"]
            receipt = read_json(path)
            assert (receipt["size"], receipt["round"]) == (size, job["round"])
            receipts.append(receipt)
        assert sorted(r["round"] for r in receipts) == list(range(manifest["rounds"]))
        assert all(r["identity"] == manifest["identity"] and len(r["cycles"]) == manifest["cycles"] for r in receipts)
        for name in QUERY_SQL:
            answers = [r["cycles"][-1]["queries"][name]["answer"] for r in receipts]
            assert all(answer == answers[0] for answer in answers), (size, name, answers)
        for stage_name, names in (("dml", receipts[0]["cycles"][0]["dml"]), ("queries", QUERY_SQL)):
            for name in names:
                samples = {}
                for engine in ENGINES:
                    if stage_name == "dml":
                        samples[engine] = [[c["dml"][name]["elapsed_ms"][engine] for c in r["cycles"]] for r in receipts]
                    else:
                        samples[engine] = [r["cycles"][-1]["queries"][name]["warm_ms"][engine] for r in receipts]
                medians = {engine: [statistics.median(v) for v in rounds] for engine, rounds in samples.items()}
                summary.append(dict(rows=size, kind=stage_name, operation=name,
                                    ms={engine: statistics.mean(v) for engine, v in medians.items()},
                                    round_medians_ms=medians))
    write_json(directory / "summary.json", summary)
    lines = ["# Matched SQL refresh benchmark", "",
             "Each DML point uses the mean of process medians across fresh mutation cycles. "
             "Each read point uses warm samples from the final cycle. DuckDB and Torbles both receive materialized staging tables. "
             "Both SQL paths include materialized results. GPU denotes the requested engine. The receipts include observed dispatches.", "",
             "| Initial fact rows | Operation | DuckDB ms | Torbles CPU ms | Torbles GPU requested ms |",
             "|---:|---|---:|---:|---:|"]
    for row in summary:
        ms = row["ms"]
        lines.append(f"| {row['rows']:,} | {row['operation']} | {ms['duckdb']:.3f} | {ms['alg']:.3f} | {ms['gpu']:.3f} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def campaign(args):
    assert len(args.rows) == len(set(args.rows))
    assert min(args.rows) >= args.cycles * args.batch * 3
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = identity()
    manifest = dict(state="running", identity=info, rows=args.rows, batch=args.batch, cycles=args.cycles,
                    repeats=args.repeats, rounds=args.rounds, threads=args.threads, dimension_updates=args.dimension_updates,
                    seed=SEED, jobs=[])
    path = output / "manifest.json"
    write_json(path, manifest)
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/torbles", TORBLES_ALG_LIB=str(ROOT / "torbles/libalg.so"),
               TORBLES_GPU_LIB=str(ROOT / "torbles/libgpu.so"))
    try:
        for round_number in range(args.rounds):
            for size in args.rows if round_number % 2 == 0 else reversed(args.rows):
                receipt = output / f"r{round_number}-{size}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--size", str(size),
                       "--round", str(round_number), "--batch", str(args.batch), "--cycles", str(args.cycles),
                       "--threads", str(args.threads), "--repeats", str(args.repeats), "--output", str(receipt)]
                if not args.dimension_updates:
                    cmd.append("--no-dimension-updates")
                subprocess.run(cmd, env=env, cwd=ROOT, check=True, timeout=600)
                manifest["jobs"].append(dict(size=size, round=round_number, receipt=receipt.name, sha256=checksum(receipt)))
                write_json(path, manifest)
        assert identity() == info
        summarize(output, manifest)
        manifest["state"] = "completed"
    except BaseException:
        manifest["state"] = "failed"
        raise
    finally:
        write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("campaign")
    run.add_argument("--rows", nargs="+", type=int, default=SCALES)
    run.add_argument("--rounds", type=int, default=2)
    work = modes.add_parser("worker")
    work.add_argument("--size", type=int, required=True)
    work.add_argument("--round", type=int, required=True)
    for mode in (run, work):
        mode.add_argument("--batch", type=int, default=2048)
        mode.add_argument("--cycles", type=int, default=4)
        mode.add_argument("--threads", type=int, default=24)
        mode.add_argument("--repeats", type=int, default=6)
        mode.add_argument("--dimension-updates", action=argparse.BooleanOptionalAction, default=True)
        mode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.cycles, args.threads, args.repeats, getattr(args, "rounds", 1)) <= 0:
        parser.error("batch size, threads, cycles and repeat counts must be positive")
    if args.mode == "worker" and (args.size < 3 * args.batch * args.cycles or args.output.exists()):
        parser.error("the worker requires enough input rows for disjoint edits and a new output path")
    largest = args.size if args.mode == "worker" else max(args.rows)
    if largest + args.cycles * (2 * args.batch - args.batch // 2) > 1 << 32:
        parser.error("the inserted primary keys must fit uint32")
    {"campaign": campaign, "worker": worker}[args.mode](args)


if __name__ == "__main__":
    main()
