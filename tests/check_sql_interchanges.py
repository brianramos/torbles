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
