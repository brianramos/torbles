# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS-inspired refresh micro-workload, with explicit fact and dimension updates."""
import argparse
import ctypes
import json
import statistics
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import algq
from torbles.refresh import RefreshTable


DIM_ROWS = 1024
FACT_COLS = ("v", "bucket", "dk")
QUERY_SQL = {
    "count": "SELECT count(*) FROM fact",
    "filtered_count": "SELECT count(*) FROM fact WHERE bucket BETWEEN 0 AND 7",
    "selective_sum": "SELECT sum(v) FROM fact WHERE bucket BETWEEN 0 AND 7",
    "join_sum": """SELECT sum(f.v) FROM fact f JOIN dim d ON f.dk = d.id
                   WHERE f.bucket BETWEEN 0 AND 7 AND d.region < 4""",
}


def timed(fn):
    start = perf_counter_ns()
    value = fn()
    return value, (perf_counter_ns() - start) / 1e6


def arrays(n, rng):
    return dict(v=rng.integers(0, 16384, n, dtype=np.uint32),
                bucket=rng.integers(0, 64, n, dtype=np.uint32),
                dk=rng.integers(0, DIM_ROWS, n, dtype=np.uint32))


def snapshot_stats(table):
    value = table.stats
    return dict(value() if callable(value) else value)


def native_query(fact, dim, name, engine):
    if name == "count":
        return fact.count(engine=engine)
    where = {"bucket": (0, 7)}
    if name == "filtered_count":
        return fact.count(where, engine=engine)
    if name == "selective_sum":
        return fact.sum("v", where, engine=engine)
    totals = fact.group_sum("dk", "v", where, engine=engine)
    if not totals:
        return None
    keys = list(totals)
    region = dim.read(keys, ("region",))["region"]
    passing = [totals[key] for key, r in zip(keys, region) if r < 4]
    return sum(passing) if passing else None


def query_samples(con, fact, dim, repeats):
    queries = {}
    for name, sql in QUERY_SQL.items():
        runs = {
            "duckdb": lambda: con.execute(sql).fetchall()[0][0],
            "cpu": lambda: native_query(fact, dim, name, "alg"),
            "gpu": lambda: native_query(fact, dim, name, "gpu"),
        }
        first, values = {}, {}
        for engine, fn in runs.items():
            values[engine], first[engine] = timed(fn)
        assert values["cpu"] == values["duckdb"] == values["gpu"], (name, values)
        samples = {engine: [] for engine in runs}
        names = tuple(runs)
        for r in range(repeats):
            for engine in names[r % len(names):] + names[:r % len(names)]:
                value, elapsed = timed(runs[engine])
                assert value == values["duckdb"], (name, engine, value, values["duckdb"])
                samples[engine].append(elapsed)
        queries[name] = dict(result=values["duckdb"], first_ms=first, warm_samples_ms=samples,
                             warm_median_ms={e: statistics.median(ts) for e, ts in samples.items()})
    return queries


def trace_gpu(fact, dim):
    original = algq.Query._exec
    result = {}
    for name in QUERY_SQL:
        calls = []

        def traced(q, dev, table, plan):
            ok = original(q, dev, table, plan)
            calls.append(dict(requested=dev, actual=q.device, accepted=bool(ok), kernel_ms=q.kernel_ms))
            return ok

        with patch.object(algq.Query, "_exec", traced):
            native_query(fact, dim, name, "gpu")
        result[name] = calls
    return result


def execute_with_batch(con, sql, data):
    con.register("changes", data)
    try:
        con.execute(sql).fetchall()
    finally:
        con.unregister("changes")


def xor_kernel(values, ids):
    """An isolated raw-table kernel check, with patch construction excluded."""
    function = getattr(algq.lib, "at_xor_words", None)
    if function is None:
        return None
    vp = ctypes.c_void_p
    function.restype = ctypes.c_int
    function.argtypes = [vp, ctypes.c_uint64, vp, vp, vp, vp]
    ids = np.ascontiguousarray(ids, np.uint64)
    cols = np.zeros(len(ids), np.int32)
    words = ids // np.uint64(64)
    masks = np.uint64(1) << (ids % np.uint64(64))
    deltas = np.full(len(ids), 0x1555, np.uint32)
    table = algq.Table("xor_scratch", {"v": values})
    try:
        before = table.get_rows("v", ids)
        samples = []
        for _ in range(5):
            def apply():
                return function(table.at, len(ids), cols.ctypes.data, words.ctypes.data,
                                masks.ctypes.data, deltas.ctypes.data)

            rc, ms = timed(apply)
            assert rc == 0 and np.array_equal(table.get_rows("v", ids), before ^ deltas)
            samples.append(ms)
            assert apply() == 0
            assert np.array_equal(table.get_rows("v", ids), before)
        return dict(rows=len(ids), words=len(np.unique(words)), ms=samples, median_ms=statistics.median(samples))
    finally:
        table.close()


def run_scale(n, batch, cycles, repeats, seed, update_mode, updates_only):
    if n < 2 * batch * cycles:
        raise ValueError("need two distinct base IDs per fact edit across the workload")
    rng = np.random.default_rng(seed)
    initial = arrays(n, rng)
    # This row pins the encoded width for the isolated XOR kernel.
    initial["v"][0] = 16383
    regions = rng.integers(0, 32, DIM_ROWS, dtype=np.uint32)
    edit_ids = rng.choice(n, size=(cycles, 2, batch), replace=False).astype(np.uint64)
    dim_batch = min(32, DIM_ROWS)
    fact = RefreshTable(initial, name="fact", update_mode=update_mode)
    dim = RefreshTable({"region": regions}, name="dim", update_mode=update_mode)
    record = dict(input_rows=n, changes_per_fact_operation=batch, dimension_updates=dim_batch, cycles=[], queries={})
    try:
        with duckdb.connect(config={"threads": "24"}) as con:
            con.execute("CREATE TABLE fact (id UBIGINT PRIMARY KEY, v UINTEGER, bucket UINTEGER, dk UINTEGER)")
            con.execute("CREATE TABLE dim (id UBIGINT PRIMARY KEY, region UINTEGER)")
            execute_with_batch(con, "INSERT INTO fact SELECT * FROM changes",
                               pd.DataFrame(dict(id=np.arange(n, dtype=np.uint64), **initial)))
            execute_with_batch(con, "INSERT INTO dim SELECT * FROM changes",
                               pd.DataFrame(dict(id=np.arange(DIM_ROWS, dtype=np.uint64), region=regions)))
            record["queries"]["base"] = query_samples(con, fact, dim, repeats)
            record["base_stats"] = snapshot_stats(fact)
            for i, (update_ids, delete_ids) in enumerate(edit_ids):
                inserted = arrays(batch, rng)
                next_id = n + i * batch
                new_v = rng.integers(0, 16384, batch, dtype=np.uint32)
                dim_ids = rng.choice(DIM_ROWS, size=dim_batch, replace=False).astype(np.uint64)
                new_regions = rng.integers(0, 32, dim_batch, dtype=np.uint32)
                operations = (
                    ("append", lambda: fact.append(inserted), "INSERT INTO fact SELECT * FROM changes",
                     pd.DataFrame(dict(id=np.arange(next_id, next_id + batch, dtype=np.uint64), **inserted))),
                    ("update", lambda: fact.update(update_ids, {"v": new_v}),
                     "UPDATE fact SET v = c.v FROM changes c WHERE fact.id = c.id",
                     pd.DataFrame(dict(id=update_ids, v=new_v))),
                    ("delete", lambda: fact.delete(delete_ids), "DELETE FROM fact USING changes c WHERE fact.id = c.id",
                     pd.DataFrame(dict(id=delete_ids))),
                    ("dimension_update", lambda: dim.update(dim_ids, {"region": new_regions}),
                     "UPDATE dim SET region = c.region FROM changes c WHERE dim.id = c.id",
                     pd.DataFrame(dict(id=dim_ids, region=new_regions))),
                )
                changes = {}
                for name, op, sql, data in operations:
                    if updates_only and name in ("append", "delete"):
                        continue
                    if i % 2:
                        _, duck_ms = timed(lambda: execute_with_batch(con, sql, data))
                        outcome, torbles_ms = timed(op)
                    else:
                        outcome, torbles_ms = timed(op)
                        _, duck_ms = timed(lambda: execute_with_batch(con, sql, data))
                    if name == "append":
                        assert outcome == range(next_id, next_id + batch)
                    changes[name] = dict(torbles_ms=torbles_ms, duckdb_ms=duck_ms)
                stage = dict(cycle=i + 1, mutation=changes, fact_stats=snapshot_stats(fact),
                             dimension_stats=snapshot_stats(dim), queries=query_samples(con, fact, dim, repeats))
                assert stage["queries"]["count"]["result"] == n
                record["cycles"].append(stage)
                print(n, "cycle", i + 1, "verified", flush=True)
            record["gpu_trace_accumulated"] = trace_gpu(fact, dim)
            _, fact_ms = timed(fact.compact)
            _, dim_ms = timed(dim.compact)
            record["compaction"] = dict(fact_ms=fact_ms, dimension_ms=dim_ms,
                                        fact_stats=snapshot_stats(fact), dimension_stats=snapshot_stats(dim))
            record["queries"]["compacted"] = query_samples(con, fact, dim, repeats)
            assert all(record["queries"]["compacted"][name]["result"] == record["cycles"][-1]["queries"][name]["result"]
                       for name in QUERY_SQL)
            record["gpu_trace_compacted"] = trace_gpu(fact, dim)
        record["xor_kernel"] = xor_kernel(initial["v"], edit_ids[0, 0])
    finally:
        fact.close()
        dim.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=(100_000, 1_000_000, 8_000_000))
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--update-mode", choices=("segments", "planes"), default="segments")
    parser.add_argument("--updates-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.batch, args.cycles, args.repeats) <= 0:
        parser.error("sizes and repeat counts must be positive")
    torbles.set_threads(24)
    torbles.set_build_threads(24)
    gpu = torbles.gpu_name()
    if "emulat" in gpu.lower() or "no CUDA" in gpu:
        raise RuntimeError("use the real CUDA executor for this benchmark")
    record = dict(duckdb=duckdb.__version__, torbles=torbles.__version__, gpu=gpu, threads=24,
                  input_seed=20261103, repeats=args.repeats, update_mode=args.update_mode,
                  updates_only=args.updates_only, scales={})
    for n in args.rows:
        record["scales"][n] = run_scale(n, args.batch, args.cycles, args.repeats, record["input_seed"],
                                           args.update_mode, args.updates_only)
        args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()