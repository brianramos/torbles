"""Native discovery -> equation composition -> ordinary Torbles CPU/GPU plan.

Run from the repository: python examples/relationship_demo.py --rows 65536
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torbles
from torbles.relationships import discover


def run(rows=65536, engine="alg"):
    if rows < 8: raise ValueError("demo needs at least 8 rows")
    i = np.arange(rows, dtype=np.int64)
    x, y = 256 * (i % 63 + 1) + 1, 256 * ((i * 17) % 61 + 1) + 7
    z = x * y
    frame = pd.DataFrame(dict(x=x, y=y, z=z, w=(z - 7) // 256, bucket=i % 8))
    db = torbles.from_frames({"measurements": frame})
    # Demonstrate source-independent operation on a saved native datastore.
    with tempfile.TemporaryDirectory(prefix="torbles-native-discovery-") as folder:
        db.save(folder)
        db.close()
        db = torbles.load(folder)
        t = db["measurements"]
        torbles.set_threads(4)
        q = db.query(t).group_by(t.bucket).where(t.x > 2000).sum(t.x * t.y)
        baseline = q.run(engine)
        started = time.perf_counter()
        catalog = discover(db, "measurements", ["x", "y", "z", "w"])
        discovery_ms = (time.perf_counter() - started) * 1000
        optimized = q.run(engine)
        if optimized != baseline: raise AssertionError("rewritten result differs from baseline")
        if q.relationship_plan["status"] != "rewritten" or len(q.relationship_plan["steps"]) < 2:
            raise AssertionError("demo did not compose two native modules")
        wanted = {}
        for xv, yv, g in zip(frame.x, frame.y, frame.bucket):
            if xv > 2000: wanted[(int(g),)] = wanted.get((int(g),), 0) + int(xv) * int(yv)
        if optimized != wanted: raise AssertionError("result differs from independent Python-integer calculation")
        report = dict(rows=rows, engine=engine, source_connection=db.con is not None,
            modules=["affine", "product", "sum"], rules=catalog.rules, catalog=catalog.stats,
            query="SUM(x*y) GROUP BY bucket WHERE x > 2000",
            composed_equivalence="SUM(x*y) = SUM(z) = 256*SUM(w) + 7*COUNT(*)",
            discovery_ms=round(discovery_ms, 3), optimizer=q.relationship_plan, execution=q.system_plan,
            independently_checked=True,
            output=[dict(bucket=k[0], total=v) for k, v in sorted(optimized.items())],
            rejected_hypothesis=catalog.probe("affine", "z", "w", scale=256, bias=8),
            limitations=["observed snapshot only", "integer columns with <=32-bit stored codes",
                         "cost is estimated plane work, not a speedup measurement",
                         "bounded search; predicates and join conditions are not rewritten"])
        catalog.close()
        return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--engine", choices=["alg", "gpu"], default="alg")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run(args.rows, args.engine)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
