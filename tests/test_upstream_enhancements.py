"""Preserve the existing 0.2.0 adaptive matrix executor under native composition.

Run: python -m unittest discover -s tests -p test_upstream_enhancements.py -v
"""
import os
import subprocess
import sys
import unittest

import pandas as pd
import torbles
from torbles.algq import A_GENERAL, A_MM, A_POLYG
from torbles.relationships import discover


class UpstreamAdaptiveMatrix(unittest.TestCase):
    @staticmethod
    def make(rows=16384):
        values = dict(x=[(i * 7) % 31 for i in range(rows)],
                      y=[(i * 13) % 63 for i in range(rows)],
                      bucket=[i % 64 for i in range(rows)])
        db = torbles.from_frames({"t": pd.DataFrame(values)})
        expected = {}
        for x, y, bucket in zip(values["x"], values["y"], values["bucket"]):
            k = (bucket,)
            expected[k] = expected.get(k, 0) + (x+y)*(x-y)
        return db, expected

    def test_dense_groupby_auto_keeps_beneficial_adaptive_matrix(self):
        db, expected = self.make()
        t = db["t"]
        q = db.query(t).group_by(t.bucket).sum((t.x+t.y)*(t.x-t.y))
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.last_plan.agg, A_MM)
        cat = discover(db, "t", ["x", "y"], modules=(), system_strategy="combine")
        self.addCleanup(cat.close)
        for mode in ("combine", "chain", "auto", "combine"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assertEqual(q.system_plan["choice"], "combine" if mode == "auto" else mode)
            self.assertEqual(q.last_plan.agg, A_GENERAL if mode == "chain" else A_MM)
            self.assertTrue(q.last_plan.out_key)
            if mode in ("combine", "auto"):
                self.assertEqual(q.last_plan.agg, A_MM)
                self.assertLess(q.system_plan["matrix_combine_cost"], q.system_plan["word_combine_cost"])
        self.assertLess(q.system_plan["matrix_combine_cost"], q.system_plan["chain_cost"])

    def test_small_group_linear_combination_prefers_word_popcounts(self):
        n = 16384
        values = dict(w=[(i * 324983) % (1 << 20) for i in range(n)],
                      bucket=[i % 2 for i in range(n)])
        db = torbles.from_frames({"t": pd.DataFrame(values)})
        t = db["t"]
        expected = {}
        for w, bucket in zip(values["w"], values["bucket"]):
            expected[(bucket,)] = expected.get((bucket,), 0) + 256*w + 7
        cat = discover(db, "t", ["w"], modules=(), system_strategy="combine")
        self.addCleanup(cat.close)
        q = db.query(t).group_by(t.bucket).sum(256*t.w+7)
        for mode in ("combine", "auto"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assertEqual(q.system_plan["choice"], "combine")
            self.assertTrue(q.last_plan.out_key)
            self.assertEqual(q.last_plan.agg, A_POLYG)
            self.assertLess(q.system_plan["word_combine_cost"], q.system_plan["matrix_combine_cost"])

    def test_ssb_shaped_dense_product_groupby_selects_matrix_automatically(self):
        n = 16384
        values = dict(price=[(i*324983) % (1 << 20) for i in range(n)],
                      discount=[i % 11 for i in range(n)], year=[i % 7 for i in range(n)])
        db = torbles.from_frames({"t": pd.DataFrame(values)})
        t = db["t"]
        expected = {}
        for p, d, y in zip(values["price"], values["discount"], values["year"]):
            expected[(y, d)] = expected.get((y, d), 0) + p*d
        cat = discover(db, "t", ["price", "discount", "year"], modules=(), system_strategy="auto")
        self.addCleanup(cat.close)
        q = db.query(t).group_by(t.year, t.discount).sum(t.price*t.discount)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["choice"], "combine")
        self.assertEqual(q.last_plan.agg, A_MM)
        self.assertLess(q.system_plan["matrix_combine_cost"], q.system_plan["word_combine_cost"])
        self.assertLess(q.system_plan["matrix_combine_cost"], q.system_plan["chain_cost"])

    def test_ungrouped_combination_keeps_direct_polynomial_executor(self):
        db, grouped = self.make(1024)
        t = db["t"]
        cat = discover(db, "t", ["x", "y"], modules=(), system_strategy="combine")
        self.addCleanup(cat.close)
        q = db.query(t).sum((t.x+t.y)*(t.x-t.y))
        self.assertEqual(q.run(), sum(grouped.values()))
        self.assertEqual(q.last_plan.agg, A_POLYG)
        self.assertFalse(q.last_plan.out_key)

    def test_forced_0_2_transpose_row_kernel_remains_reachable(self):
        # The native MM configuration is read once per process. A fresh child
        # verifies the existing forced row-kernel path alongside the new chooser.
        script = r'''
import pandas as pd, torbles
from torbles.algq import A_MM
from torbles.relationships import discover
n=16384
x=[(i*7)%31 for i in range(n)]; y=[(i*13)%63 for i in range(n)]; b=[i%64 for i in range(n)]
db=torbles.from_frames({'t':pd.DataFrame(dict(x=x,y=y,bucket=b))}); t=db['t']
cat=discover(db,'t',['x','y'],modules=(),system_strategy='combine')
q=db.query(t).group_by(t.bucket).sum((t.x+t.y)*(t.x-t.y))
expected={}
for a,c,g in zip(x,y,b): expected[(g,)]=expected.get((g,),0)+(a+c)*(a-c)
assert q.run()==expected
assert q.last_plan.agg==A_MM
assert q.system_plan['choice']=='combine'
print('forced 0.2 row kernel exact')
'''
        env = dict(os.environ, TORBLES_MM_ROWS="2", TORBLES_GROUPBY="mm")
        out = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(os.path.dirname(__file__)),
                             env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("forced 0.2 row kernel exact", out.stdout)

    def test_disabled_0_2_row_kernel_is_not_costed_as_available(self):
        script = r'''
import sys
sys.path.insert(0, 'tests')
from test_upstream_enhancements import UpstreamAdaptiveMatrix
from torbles.algq import A_POLYG
from torbles.relationships import discover
db,expected=UpstreamAdaptiveMatrix.make(8192); t=db['t']
cat=discover(db,'t',['x','y'],modules=(),system_strategy='combine')
q=db.query(t).group_by(t.bucket).sum((t.x+t.y)*(t.x-t.y))
assert q.run()==expected
assert q.system_plan['matrix_combine_cost']==-1
assert q.last_plan.agg==A_POLYG
'''
        env = dict(os.environ, TORBLES_MM_ROWS="0", TORBLES_GROUPBY="mm")
        out = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(os.path.dirname(__file__)),
                             env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


if __name__ == "__main__": unittest.main()
