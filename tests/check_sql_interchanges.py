# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from torbles import algq, sqlgpugroups, sqlinterchanges


class InterchangeChecks(unittest.TestCase):
    def test_coordinate_requires_both_directions(self):
        raw = np.arange(256, dtype=np.uint32)
        values = (raw >> 4) * 3 + 7
        proof = sqlinterchanges.certify_coordinate(raw, values, [7, 6, 5, 4])
        self.assertEqual((proof.shift, proof.width), (4, 4))
        np.testing.assert_array_equal(proof.labels[raw >> 4], values)
        conflict = values.copy()
        conflict[1] += 1
        self.assertIsNone(sqlinterchanges.certify_coordinate(raw, conflict, [7, 6, 5, 4]))
        self.assertIsNone(sqlinterchanges.certify_coordinate(raw, values % 4, [7, 6, 5, 4]))

    def test_switch_certificate_and_cuda_fallback(self):
        raw = np.arange(1024, dtype=np.uint32)
        rng = np.random.default_rng(23)
        labels = rng.integers(0, 16, len(raw), dtype=np.uint32)
        labels[:512] = (raw[:512] ^ (raw[:512] >> 2) ^ 7) & 15
        proof = sqlinterchanges.compile_interchange(raw, labels, 10, 4)
        self.assertGreaterEqual(proof.covered, 512)
        self.assertLess(proof.covered, len(raw))
        for code, value in zip(raw.tolist(), labels.tolist()):
            mode, bias, *masks = proof.sectors[code >> proof.shift].tolist()
            if mode:
                self.assertEqual(bias ^ sum(((code & mask).bit_count() & 1) << j
                                            for j, mask in enumerate(masks)), value)
        if not hasattr(algq.gpu_lib(), "gpu_group_distinct_sector_planes"):
            return
        fk = np.tile(raw, 10)
        users = np.arange(len(fk), dtype=np.uint32) % 73
        table = algq.Table("t", {"fk": fk, "uid": users}, msb=("fk",))
        self.addCleanup(table.close)
        key, unique = ("fk", None, 0, 3, 0), ("uid", None, 0, 0, 0)
        projected = (labels, 1023, 4)
        stats = {}
        base = sqlgpugroups.group_distinct(table, [key], [], 16, table.used_bm, {},
                                          distinct=unique, rank=0, projection=projected)
        switched = sqlgpugroups.group_distinct(table, [key], [], 16, table.used_bm, stats,
                                              distinct=unique, rank=0, projection=projected, interchange=proof)
        self.assertIn("cuda_interchange_proven_sectors", stats)
        def rows(result):
            return sorted(zip(result[0][:, 0].tolist(), result[1].tolist(), result[3].tolist()))
        self.assertEqual(rows(base), rows(switched))


    def test_rank_switch_bounds_distinct_work(self):
        sizes = np.arange(1, 65) * 53
        self.assertLess(sizes[-8:].sum(), sizes.sum() // 2)
        if not hasattr(algq.gpu_lib(), "gpu_group_distinct_sector_planes"):
            return
        codes = np.repeat(np.arange(64, dtype=np.uint32), sizes)
        users = np.arange(len(codes), dtype=np.uint32) % 101
        table = algq.Table("t", {"fk": codes, "uid": users}, msb=("fk",))
        self.addCleanup(table.close)
        key, unique = ("fk", None, 0, 4, 6), ("uid", None, 0, 0, 0)
        stats = {}
        result = sqlgpugroups.group_distinct(
            table, [key], ["uid"], 8, table.used_bm, stats,
            distinct=unique, rank=0, rank_switch=True,
        )
        self.assertEqual(stats["cuda_interchange_rank_rows"], int(sizes[-8:].sum()))
        keys, counts, sums, distinct = result
        for i, code in enumerate(range(63, 55, -1)):
            values = users[codes == code]
            self.assertEqual((int(keys[i, 0]), int(counts[i]), int(sums[i, 0]), int(distinct[i])),
                             (code, len(values), int(values.sum()), len(np.unique(values))))


    def test_dense_joint_coordinate_matches_hash(self):
        from torbles.sqlencodednative import encoded_groups
        i = np.arange(131072, dtype=np.uint32)
        table = algq.Table("t", {"fk": i % 1024, "b": i // 1024 % 16, "uid": i % 257},
                           msb=("fk",))
        self.addCleanup(table.close)
        keys = [("fk", None, 4, 4, 6), ("b", None, 0, 0, 0)]
        unique = ("uid", None, 0, 0, 0)
        cpu = encoded_groups(table, keys, ["uid"], 1024, table.used_bm, distinct=unique, rank=1)
        self.assertIsNotNone(cpu)
        self.assertEqual(len(cpu[1]), 1024)
        if not hasattr(algq.gpu_lib(), "gpu_group_distinct_sector_planes"):
            return
        stats = {}
        base = sqlgpugroups.group_distinct(table, keys, ["uid"], 1024, table.used_bm, {},
                                          distinct=unique, rank=1)
        dense = sqlgpugroups.group_distinct(table, keys, ["uid"], 1024, table.used_bm, stats,
                                           distinct=unique, rank=1, rank_switch=True)
        self.assertEqual(stats["cuda_interchange_dense_bits"], 10)

        def rows(result):
            codes, counts, sums, distinct = result
            return sorted(zip(map(tuple, codes.tolist()), counts.tolist(),
                              sums[:, 0].tolist(), distinct.tolist()))

        self.assertEqual(rows(cpu), rows(base))
        self.assertEqual(rows(base), rows(dense))

    def test_distinct_rank_certificate_and_exact_retry(self):
        if not hasattr(algq.gpu_lib(), "gpu_group_distinct_sector_planes"):
            return
        sizes = np.r_[np.arange(2500, 2600, 50), np.full(14, 1200), np.full(112, 320)]
        keys = [("g", None, 0, 4, 7)]
        unique = ("u", None, 0, 0, 0)
        for certify in (True, False):
            values = np.concatenate([
                np.arange(size, dtype=np.uint32) if certify or code == 20
                else np.zeros(size, dtype=np.uint32)
                for code, size in enumerate(sizes)
            ])
            table = algq.Table("t", {"g": np.repeat(np.arange(128, dtype=np.uint32), sizes), "u": values})
            self.addCleanup(table.close)
            stats = {}
            result = sqlgpugroups.group_distinct(table, keys, [], 2, table.used_bm, stats,
                                                 distinct=unique, rank=1, rank_switch=True)
            self.assertEqual(stats["cuda_interchange_dense_bits"], 7)
            if certify:
                self.assertGreater(stats["cuda_interchange_rank_rows"], 0)
                self.assertEqual(result[0][:, 0].tolist(), [1, 0])
                self.assertEqual(result[3].tolist(), [2550, 2500])
            else:
                self.assertNotIn("cuda_interchange_rank_rows", stats)
                self.assertEqual(result[0][:, 0].tolist(), [20, 0])
                self.assertEqual(result[3].tolist(), [320, 1])


    def test_count_interchange_accepts_dense_keys_and_rejects_sparse_domains(self):
        from unittest.mock import patch
        from torbles import sqlcpuinterchanges

        rows = np.arange(131072, dtype=np.uint32)
        table = algq.Table("t", {"g": rows % 64, "value": rows % 2048, "wide": rows})
        self.addCleanup(table.close)
        scope = table.used_bm
        keys = [("g", None, 0, 0, 0)]
        cpu_stats = {}
        cpu = sqlcpuinterchanges.count_topk(table, keys, 5, scope, cpu_stats)
        self.assertEqual(cpu[1].tolist(), [2048] * 5)
        self.assertEqual(cpu_stats["cpu_interchange_histogram_codes"], 64)
        sparse = [("wide", None, 0, 0, 0)]
        self.assertIsNone(sqlcpuinterchanges.count_topk(table, sparse, 5, scope, {}))
        if not hasattr(algq.gpu_lib(), "gpu_group_dense_planes"):
            return
        with patch.dict("os.environ", {"TORBLES_INTERCHANGES": "0"}):
            baseline = sqlgpugroups.group(table, keys, ["value"], 64, scope, {})
        stats = {}
        with patch.dict("os.environ", {"TORBLES_INTERCHANGES": "auto"}):
            dense = sqlgpugroups.group(table, keys, ["value"], 64, scope, stats)
            rejected = {}
            sqlgpugroups.group(table, sparse, [], 5, scope, rejected)
        self.assertEqual(stats["cuda_interchange_dense_slots"], 64)
        self.assertNotIn("cuda_interchange_dense_slots", rejected)

        def rows(result):
            codes, counts, sums = result
            return sorted(zip(codes[:, 0].tolist(), counts.tolist(), sums[:, 0].tolist()))

        self.assertEqual(rows(baseline), rows(dense))


    def test_collision_certificate_preserves_repeated_groups_and_singletons(self):
        from unittest.mock import patch

        if not hasattr(algq.gpu_lib(), "gpu_group_collision_planes"):
            return
        i = np.arange(131072, dtype=np.uint32)
        low = i * np.uint32(2654435761)
        high = i % 257
        second = i % 29
        for values in (low, high, second):
            values[1:4] = values[0]
        table = algq.Table("t", {"lo": low, "hi": high, "b": second, "m": i}, msb=("lo",))
        self.addCleanup(table.close)
        keys = [("lo", "hi", 32, 0, 0), ("b", None, 0, 0, 0)]
        stats = {}
        with patch.dict("os.environ", {"TORBLES_INTERCHANGES": "auto"}):
            codes, counts, sums = sqlgpugroups.group(table, keys, ["m"], 8, table.used_bm, stats)
        self.assertLess(stats["cuda_interchange_collision_rows"], len(i) // 2)
        self.assertEqual(counts.tolist(), [4, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(sums[0, 0], 6)
        for code, value in zip(codes[1:], sums[1:, 0]):
            row = int(value)
            self.assertEqual(tuple(code), ((int(high[row]) << 32) | int(low[row]), int(second[row])))
