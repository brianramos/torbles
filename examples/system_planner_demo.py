"""Compare actual native chained execution with combined algebraic systems.

    python examples/system_planner_demo.py --output results/system_planner_demo.json
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torbles
from torbles.relationships import discover


def case(name, rows):
    if name == "dense_narrow":
        return dict(x=[i % 8 for i in range(rows)], y=[(i * 3) % 7 for i in range(rows)],
                    bucket=[i % 4 for i in range(rows)], flag=[1] * rows)
    return dict(x=[(i * 324983) % (1 << 20) for i in range(rows)],
                y=[(i * 50777) % (1 << 20) for i in range(rows)],
                bucket=[i % 4 for i in range(rows)], flag=[int(i % 64 == 0) for i in range(rows)])


def run(rows=16384):
    if rows < 128: raise ValueError("use at least 128 rows")
    torbles.set_threads(4)
    reports = []
    for name in ("dense_narrow", "selective_wide"):
        values = case(name, rows)
        db = torbles.from_frames({"t": pd.DataFrame(values)})
        t = db["t"]
        # No learned relationships are needed to test the physical choice.
        catalog = discover(db, "t", ["x", "y"], modules=())
        q = db.query(t).where(t.flag == 1).group_by(t.bucket).sum((t.x + t.y) * (t.x - t.y))
        expected = {}
        for x, y, flag, bucket in zip(values["x"], values["y"], values["flag"], values["bucket"]):
            if flag:
                k = (bucket,)
                expected[k] = expected.get(k, 0) + (x+y)*(x-y)
        report = dict(case=name, rows=rows, expression="(x+y)*(x-y)",
            combined_system="x*x-y*y", query="SUM(measure) GROUP BY bucket WHERE flag=1",
            learned_relationships=len(catalog.rules), alternatives={})
        for mode in ("chain", "combine", "auto"):
            catalog.system_strategy = mode
            q.run()  # warm the native path and compiled query cache
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                answer = q.run()
                samples.append((time.perf_counter() - start) * 1000)
                if answer != expected: raise AssertionError(f"{name}/{mode}: wrong answer")
            report["alternatives"][mode] = dict(execution=copy.deepcopy(q.system_plan),
                native_aggregate=int(q.last_plan.agg), device=q.device,
                elapsed_ms_best_of_3=round(min(samples), 4))
        wanted = "combine" if name == "dense_narrow" else "chain"
        if report["alternatives"]["auto"]["execution"]["choice"] != wanted:
            raise AssertionError(f"{name}: expected native auto choice {wanted}")
        report["independently_checked"] = True
        report["result"] = [dict(bucket=k[0], total=v) for k, v in sorted(expected.items())]
        reports.append(report)
        catalog.close()
        db.close()
    return dict(cases=reports,
        scope="whole measure: native stack chain or normalized polynomial system",
        limitations=["sampled heuristic costs, not calibrated timing predictions",
                     "no arbitrary hybrid region partitions or intermediate tables",
                     "GPU supports combined execution; general chaining is CPU only"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = json.dumps(run(args.rows), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
