"""Reuse 0.2.0's compiled plans without retaining stale relationship proofs."""
import unittest
import pandas as pd
import torbles
from torbles.relationships import discover


class RelationshipCache(unittest.TestCase):
    def make(self):
        x = [256*i+1 for i in range(1, 9) for _ in range(7)]
        y = [256*j+7 for _ in range(8) for j in range(1, 8)]
        z = [a*b for a, b in zip(x, y)]
        db = torbles.from_frames({"t": pd.DataFrame(dict(x=x, y=y, z=z, w=[(v-7)//256 for v in z]))})
        return db, x, y, z

    def catalog(self, db, **kw):
        cat = discover(db, "t", **kw)
        self.addCleanup(cat.close)
        return cat

    def test_rewritten_query_reuses_compiled_plan(self):
        db, x, y, z = self.make()
        self.catalog(db)
        t = db["t"]
        q = db.query(t).sum(t.x*t.y)
        self.assertEqual(q.run(), sum(z))
        self.assertFalse(q.relationship_plan["optimizer_cached"])
        self.assertEqual(q.run(), sum(z))
        self.assertTrue(q.relationship_plan["optimizer_cached"])
        self.assertTrue(q.plan_cached)

    def test_changed_filter_keeps_proof_but_recompiles_query(self):
        db, x, y, z = self.make()
        self.catalog(db)
        t = db["t"]; q = db.query(t).sum(t.x*t.y)
        q.run(); q.run()
        q.where(t.x > 1000)
        self.assertEqual(q.run(), sum(v for a, v in zip(x, z) if a > 1000))
        self.assertTrue(q.relationship_plan["optimizer_cached"])
        self.assertFalse(q.plan_cached)
        q.run(); self.assertTrue(q.plan_cached)

    def test_new_proof_invalidates_search_and_compiled_replacement(self):
        db, x, y, z = self.make()
        cat = self.catalog(db, modules=("product",))
        t = db["t"]; q = db.query(t).sum(t.x*t.y)
        q.run(); q.run()
        old_cost = q.relationship_plan["after_cost"]
        cat.probe("affine", "z", "w", scale=256, bias=7)
        self.assertEqual(q.run(), sum(z))
        self.assertFalse(q.relationship_plan["optimizer_cached"])
        self.assertLess(q.relationship_plan["after_cost"], old_cost)
        self.assertEqual(q.run(), sum(z))
        self.assertTrue(q.plan_cached)

    def test_mutation_cannot_reuse_verified_replacement(self):
        db, x, y, z = self.make()
        self.catalog(db)
        t = db["t"]; q = db.query(t).sum(t.x*t.y)
        q.run(); q.run(); t.flip(0, 0); t.repair()
        self.assertEqual(q.run(), sum(z))
        self.assertEqual(q.relationship_plan["status"], "stale")

    def test_physical_only_catalog_decodes_only_if_explicitly_probed(self):
        db, x, y, z = self.make()
        cat = self.catalog(db, modules=())
        self.assertEqual(cat.stats["decoded_bytes"], 0)
        t = db["t"]; q = db.query(t).sum(t.x*t.y)
        self.assertEqual(q.run(), sum(z))
        self.assertEqual(cat.stats["decoded_bytes"], 0)
        self.assertTrue(cat.probe("product", "z", "x", "y")["usable"])
        self.assertGreater(cat.stats["decoded_bytes"], 0)
        self.assertEqual(q.run(), sum(z))


if __name__ == "__main__": unittest.main()
