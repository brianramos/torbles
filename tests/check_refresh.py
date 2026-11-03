# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Check stable refresh IDs and reductions against an independent DuckDB table."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torbles.refresh import RefreshTable
from torbles import algq


UINT32_MAX = (1 << 32) - 1
VALIDATION_ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError)
NAMES = ("g", "x", "y")


def u32(values):
    return np.array(values, dtype=np.uint32)


def fixture(n):
    i = np.arange(n, dtype=np.uint32)
    return {"g": i % 11, "x": (i * 37) % 501, "y": i % 3}


def where_sql(where):
    conditions, parameters = [], []
    for name, value in (where or {}).items():
        if isinstance(value, tuple):
            conditions.append(f"{name} BETWEEN ? AND ?")
            parameters.extend(value)
        else:
            conditions.append(f"{name} = ?")
            parameters.append(value)
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), parameters


class RefreshChecks(unittest.TestCase):
    def new_table(self, columns, **options):
        table = RefreshTable(columns, **options)
        self.addCleanup(table.close)
        return table

    def check_read(self, table, ids, expected, columns=NAMES):
        actual = table.read(ids, columns)
        self.assertEqual(set(actual), set(columns))
        for index, name in enumerate(columns):
            np.testing.assert_array_equal(actual[name], [row[index] for row in expected])

    def compare_reductions(self, table, con, where):
        suffix, params = where_sql(where)
        expected_count, expected_sum = con.execute(f"SELECT count(*), sum(x) FROM live{suffix}", params).fetchone()
        expected_groups = dict(con.execute(f"SELECT g, sum(x) FROM live{suffix} GROUP BY g", params).fetchall())
        self.assertEqual(table.count(where=where), expected_count)
        self.assertEqual(table.sum("x", where=where), expected_sum)
        self.assertEqual(table.group_sum("g", "x", where=where), expected_groups)

    def test_seeded_mixed_batches_against_duckdb(self):
        initial = fixture(137)
        initial["x"][0] = UINT32_MAX
        table = self.new_table(initial)
        rng = np.random.default_rng(503)
        with duckdb.connect(":memory:") as con:
            con.execute("CREATE TABLE live(id BIGINT PRIMARY KEY, g UBIGINT, x UBIGINT, y UBIGINT)")
            con.executemany("INSERT INTO live VALUES (?, ?, ?, ?)",
                            [(i, *(int(initial[name][i]) for name in NAMES)) for i in range(137)])
            live = set(range(137))
            next_id = 137
            for step in range(24):
                with self.subTest(step=step):
                    if step % 3 == 0:
                        n = int(rng.integers(1, 12))
                        g = rng.integers(0, 11, n, dtype=np.uint32)
                        x = rng.integers(0, 1 << 32, n, dtype=np.uint32)
                        y = rng.integers(0, 3, n, dtype=np.uint32)
                        expected_ids = list(range(next_id, next_id + n))
                        self.assertEqual(list(table.append(dict(g=g, x=x, y=y))), expected_ids)
                        con.executemany("INSERT INTO live VALUES (?, ?, ?, ?)",
                                        [(row_id, int(a), int(b), int(c)) for row_id, a, b, c in zip(expected_ids, g, x, y)])
                        live.update(expected_ids)
                        next_id += n
                    elif step % 3 == 1:
                        ids = [int(i) for i in rng.choice(sorted(live), 4, replace=False)]
                        g = rng.integers(0, 11, 4, dtype=np.uint32)
                        x = rng.integers(0, 1 << 32, 4, dtype=np.uint32)
                        table.update(ids, dict(g=g, x=x))
                        con.executemany("UPDATE live SET g = ?, x = ? WHERE id = ?",
                                        [(int(a), int(b), row_id) for row_id, a, b in zip(ids, g, x)])
                    else:
                        ids = [int(i) for i in rng.choice(sorted(live), 3, replace=False)]
                        table.delete(ids)
                        con.executemany("DELETE FROM live WHERE id = ?", [(row_id,) for row_id in ids])
                        live.difference_update(ids)
                    if step in (4, 11, 19):
                        table.compact()
                    lo, hi = sorted(int(v) for v in rng.integers(0, 11, 2))
                    for where in (None, {"g": (lo, hi), "y": step % 3}, {"x": (UINT32_MAX, UINT32_MAX)}):
                        self.compare_reductions(table, con, where)
                    ids = [int(i) for i in rng.choice(sorted(live), 5, replace=False)]
                    expected = [con.execute("SELECT y, x FROM live WHERE id = ?", [i]).fetchone() for i in ids]
                    self.check_read(table, ids, expected, columns=("y", "x"))

    def test_sparse_edits_visibility_compaction_and_input_ownership(self):
        original = fixture(4099)
        table = self.new_table(original)
        original["x"][64] = UINT32_MAX
        self.check_read(table, [64], [(64 % 11, (64 * 37) % 501, 64 % 3)])
        edits = u32([UINT32_MAX, 17])
        table.update([4098, 64], {"x": edits})
        edits[:] = 0
        table.delete([0, 63, 4097])
        added = dict(g=u32([0, 9]), x=u32([42, 0]), y=u32([2, 1]))
        self.assertEqual(list(table.append(added)), [4099, 4100])
        added["x"][:] = 123
        ids = [4100, 64, 4099, 4098]
        expected = [(9, 0, 1), (64 % 11, 17, 64 % 3), (0, 42, 2), (4098 % 11, UINT32_MAX, 4098 % 3)]
        self.check_read(table, ids, expected)
        total = table.sum("x")
        groups = table.group_sum("g", "x")
        self.assertEqual(table.count(), 4098)
        self.assertEqual(table.count(where={"x": UINT32_MAX}), 1)
        with self.assertRaises(KeyError):
            table.read([4099, 0], ("x",))
        self.assertEqual(table.stats["next_id"], 4101)
        table.compact()
        self.check_read(table, ids, expected)
        self.assertEqual(table.count(), 4098)
        self.assertEqual(table.sum("x"), total)
        self.assertEqual(table.group_sum("g", "x"), groups)
        self.assertEqual(table.stats["physical_rows"], 4098)
        self.assertEqual(table.stats["override_entries"], 0)
        with self.assertRaises(KeyError):
            table.delete([63])
        self.assertEqual(list(table.append(dict(g=u32([0]), x=u32([1]), y=u32([0])))), [4101])

    def test_failed_batches_leave_every_row_and_next_id_intact(self):
        initial = fixture(8)
        table = self.new_table(initial)
        expected = list(zip(*(initial[name] for name in NAMES)))
        operations = {
            "missing append column": lambda: table.append(dict(g=u32([1]), x=u32([2]))),
            "different append lengths": lambda: table.append(dict(g=u32([1]), x=u32([2, 3]), y=u32([0]))),
            "invalid append value": lambda: table.append(dict(g=u32([1]), x=[UINT32_MAX + 1], y=u32([0]))),
            "mixed valid and missing update IDs": lambda: table.update([1, 100], {"x": u32([99, 100])}),
            "invalid update value after valid value": lambda: table.update([1, 2], {"x": [99, -1]}),
            "unknown update column": lambda: table.update([1], {"x": u32([99]), "unknown": u32([0])}),
            "duplicate update IDs": lambda: table.update([1, 1], {"x": u32([99, 100])}),
            "mixed valid and missing delete IDs": lambda: table.delete([1, 100]),
            "duplicate delete IDs": lambda: table.delete([1, 1]),
        }
        for label, operation in operations.items():
            with self.subTest(label=label):
                with self.assertRaises(VALIDATION_ERRORS):
                    operation()
                self.assertEqual(table.count(), 8)
                self.check_read(table, list(range(8)), expected)
        self.assertEqual(list(table.append(dict(g=u32([1]), x=u32([99]), y=u32([0])))), [8])

    def test_empty_segments_and_compaction_do_not_reuse_ids(self):
        table = self.new_table(fixture(0))
        self.assertEqual(list(table.append(fixture(0))), [])
        self.assertEqual(table.stats["segments"], 0)
        self.assertEqual(table.count(), 0)
        self.assertIsNone(table.sum("x"))
        self.assertEqual(table.group_sum("g", "x"), {})
        self.assertEqual(list(table.append(fixture(2))), [0, 1])
        self.check_read(table, [1, 0, 1], [(37,), (0,), (37,)], columns=("x",))
        table.delete([0, 1])
        table.compact()
        self.assertEqual(table.stats["physical_rows"], 0)
        self.assertEqual(list(table.append(fixture(1))), [2])

    def test_in_place_repeated_updates_within_words(self):
        original = fixture(131)
        table = self.new_table(original, update_mode="planes")
        expected = {name: column.copy() for name, column in original.items()}
        batches = [([63, 64, 65, 130], {"x": u32([511, 4, 5, 6]), "g": u32([15, 14, 13, 12])}),
                   ([65, 64, 66], {"x": u32([40, 30, 20])}),
                   ([64, 66], {"x": u32([30, 20])})]
        for ids, edits in batches:
            table.update(ids, edits)
            for name, values in edits.items():
                expected[name][ids] = values
            self.check_read(table, [130, 65, 66, 63, 64, 67],
                            [tuple(expected[name][i] for name in NAMES) for i in (130, 65, 66, 63, 64, 67)])
        rows = (expected["g"] >= 10) & (expected["x"] <= 500)
        where = {"g": (10, 15), "x": (0, 500)}
        self.assertEqual(table.count(where), int(rows.sum()))
        self.assertEqual(table.sum("x", where), int(expected["x"][rows].sum(dtype=np.uint64)))
        groups = {int(g): int(expected["x"][rows & (expected["g"] == g)].sum(dtype=np.uint64))
                  for g in np.unique(expected["g"][rows])}
        self.assertEqual(table.group_sum("g", "x", where), groups)
        stats = table.stats
        self.assertEqual(stats["update_mode"], "planes")
        self.assertEqual(stats["plane_update_batches"], 2)
        self.assertEqual(stats["plane_update_rows"], 7)
        self.assertEqual(stats["plane_update_values"], 11)
        self.assertEqual(stats["width_fallback_batches"], 0)
        self.assertEqual(stats["segments"], 1)
        self.assertEqual(stats["physical_rows"], 131)
        self.assertEqual(stats["override_entries"], 0)

    def test_in_place_width_fallback_is_one_atomic_replacement_batch(self):
        table = self.new_table(dict(g=u32([0, 1]), x=u32([0, 31]), y=u32([0, 1])), update_mode="planes")
        self.assertEqual(list(table.append(dict(g=u32([1, 0]), x=u32([255, 80]), y=u32([2, 0])))), [2, 3])
        table.update([3, 1], {"x": u32([128, 30])})
        self.check_read(table, [0, 1, 2, 3], [(0, 0, 0), (1, 30, 1), (1, 255, 2), (0, 128, 0)])
        self.assertEqual(table.stats["plane_update_rows"], 2)
        self.assertEqual(table.stats["physical_rows"], 4)
        snapshot = table.stats
        with self.assertRaises(KeyError):
            table.update([1, 3, 40], {"x": u32([1, 2, 3])})
        with self.assertRaises(VALIDATION_ERRORS):
            table.update([1, 3], {"x": [1, -1]})
        self.assertEqual(table.stats, snapshot)
        self.check_read(table, [1, 3], [(1, 30, 1), (0, 128, 0)])

        # The first value needs a wider plane. Both edits go to one segment.
        table.update([0, 2], {"x": u32([32, 127])})
        self.assertEqual(table.stats["width_fallback_batches"], 1)
        self.assertEqual(table.stats["plane_update_rows"], 2)
        self.assertEqual(table.stats["segments"], 3)
        self.assertEqual(table.stats["physical_rows"], 6)
        self.assertEqual(table.stats["override_entries"], 2)
        table.update([2, 0], {"x": u32([126, 33])})
        expected = [(0, 33, 0), (1, 30, 1), (1, 126, 2), (0, 128, 0)]
        self.check_read(table, [0, 1, 2, 3], expected)
        self.assertEqual(table.stats["plane_update_rows"], 4)
        self.assertEqual(table.sum("x"), 317)
        self.assertEqual(table.group_sum("g", "x"), {0: 161, 1: 156})
        self.assertEqual(table.count({"x": (127, 255)}), 1)
        table.compact()
        self.check_read(table, [0, 1, 2, 3], expected)
        self.assertEqual(table.stats["physical_rows"], 4)
        self.assertEqual(table.stats["override_entries"], 0)

    def test_warmed_gpu_applies_incremental_updates_to_retained_handle(self):
        gpu = algq.gpu_lib()
        if not hasattr(gpu, "gpu_xor_words"):
            self.skipTest("GPU executor with incremental word updates is not built")
        data = fixture(8195)
        table = self.new_table(data, update_mode="planes")
        model = {name: column.copy() for name, column in data.items()}
        visible = np.ones(8195, dtype=bool)
        table.delete([1])
        visible[1] = False
        where = {"g": (3, 15), "x": (50, 300)}
        devices = []
        execute = algq.Query._exec

        def observe(query, device, native_table, plan):
            ran = execute(query, device, native_table, plan)
            if ran:
                devices.append(query.device)
            return ran

        def compare_gpu():
            rows = visible & (model["g"] >= 3) & (model["x"] >= 50) & (model["x"] <= 300)
            self.assertEqual(table.count(where, engine="gpu"), int(rows.sum()))
            self.assertEqual(table.sum("x", where, engine="gpu"), int(model["x"][rows].sum(dtype=np.uint64)))
            grouped = {int(g): int(model["x"][rows & (model["g"] == g)].sum(dtype=np.uint64))
                       for g in np.unique(model["g"][rows])}
            self.assertEqual(table.group_sum("g", "x", where, engine="gpu"), grouped)

        with patch.object(algq.Query, "_exec", observe):
            compare_gpu()
            base = table._segments[0].table
            handle = base._storage().gpu
            self.assertTrue(handle)
            with patch.object(gpu, "gpu_xor_words", wraps=gpu.gpu_xor_words) as sync:
                for ids, g, x in (([63, 64, 65, 8194], [4, 3, 2, 1], [120, 200, 60, 99]),
                                  ([65, 64], [15, 0], [180, 75])):
                    table.update(ids, {"g": u32(g), "x": u32(x)})
                    model["g"][ids], model["x"][ids] = g, x
                    compare_gpu()
                    self.assertEqual(base._storage().gpu, handle)
                self.assertEqual(sync.call_count, 2)
            table.delete([63, 8194])
            visible[[63, 8194]] = False
            extra = dict(g=u32([9, 8]), x=u32([57, 310]), y=u32([0, 1]))
            self.assertEqual(list(table.append(extra)), [8195, 8196])
            model = {name: np.concatenate((model[name], extra[name])) for name in NAMES}
            visible = np.concatenate((visible, [True, True]))
            compare_gpu()
            self.assertEqual(base._storage().gpu, handle)
        self.assertTrue(devices and all(d.startswith("gpu") for d in devices), devices)
        ids = [8196, 64, 65, 8195]
        self.check_read(table, ids, [tuple(model[name][i] for name in NAMES) for i in ids])


    def test_failed_gpu_sync_rebuilds_from_committed_cpu_planes(self):
        gpu = algq.gpu_lib()
        if not hasattr(gpu, "gpu_xor_words"):
            self.skipTest("GPU executor with incremental word updates is not built")
        data = fixture(257)
        table = self.new_table(data, update_mode="planes")
        total = int(data["x"].sum(dtype=np.uint64))
        self.assertEqual(table.sum("x", engine="gpu"), total)
        with patch.object(gpu, "gpu_xor_words", return_value=-2) as sync:
            table.update([0], {"x": u32([100])})
            self.assertEqual(sync.call_count, 1)
        self.assertIsNone(table._segments[0].table._storage().gpu)
        self.assertEqual(table.stats["gpu_plane_evictions"], 1)
        self.assertEqual(table.sum("x"), total + 100)
        self.assertEqual(table.sum("x", engine="gpu"), total + 100)
        self.assertEqual(table.stats["segments"], 1)


if __name__ == "__main__":
    unittest.main()
