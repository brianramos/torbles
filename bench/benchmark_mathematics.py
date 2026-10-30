"""Measure a parameterized query family with and without moment-state reuse.

This is a synthetic function-family experiment, separate from the mixed SSB
matrix. Both paths produce the same exact answers. Snapshot construction,
symbolic construction and first/warm batches are reported independently.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def timed(call):
    start=time.perf_counter_ns()
    result=call()
    return result,(time.perf_counter_ns()-start)/1e6


def summary(values):
    return dict(raw_ms=values,median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values))


def run_case(rows,args):
    import numpy as np
    import pandas as pd
    import torbles
    from torbles.algq import copy_query
    from torbles.mathematics import MomentBasis

    index=np.arange(rows,dtype=np.int64)
    frame=pd.DataFrame(dict(x=index%31-15,y=(index*7)%43-21,flag=index%5))
    db,load_ms=timed(lambda:torbles.from_ddl(
        "CREATE TABLE t (x BIGINT,y BIGINT,flag BIGINT)",{"t":frame}))
    table=db["t"]
    scope=db.query(table).where(table.flag>=2)
    bank,build_ms=timed(lambda:MomentBasis(scope,["x","y"],engine=args.engine))
    try:
        x,y=bank.variables["x"],bank.variables["y"]
        a,b=bank.parameter("a"),bank.parameter("b")
        function,compile_ms=timed(lambda:bank.reduce((y-a*x-b)**2))
        da=function.diff("a")
        dbeta=function.diff("b")
        daa=da.diff("a")
        dab=da.diff("b")
        integrated=function.integral("a",0,1)
        parameters=[(i%7-3,(i*3)%9-4) for i in range(args.parameters)]
        direct=[copy_query(scope).sum((table.y-aa*table.x-bb)*(table.y-aa*table.x-bb))
                for aa,bb in parameters]
        expected=[int(db.con.execute(
            "SELECT sum((y-?*x-?)*(y-?*x-?)) FROM t WHERE flag>=2",[aa,bb,aa,bb]).fetchone()[0] or 0)
                  for aa,bb in parameters]
        n,sx,sy,sxx,sxy,syy=db.con.execute(
            "SELECT count(*),sum(x),sum(y),sum(x*x),sum(x*y),sum(y*y) FROM t WHERE flag>=2").fetchone()
        sx,sy,sxx,sxy,syy=map(lambda value:int(value or 0),(sx,sy,sxx,sxy,syy))
        for aa,bb in parameters:
            assert da.evaluate(a=aa,b=bb)==2*aa*sxx+2*bb*sx-2*sxy
            assert dbeta.evaluate(a=aa,b=bb)==2*aa*sx+2*n*bb-2*sy
            assert daa.evaluate(a=aa,b=bb)==2*sxx
            assert dab.evaluate(a=aa,b=bb)==2*sx
            oracle=syy-2*bb*sy+n*bb*bb-sxy+bb*sx+Fraction(sxx,3)
            assert integrated.evaluate(b=bb)==oracle
        def direct_batch(order):
            return [direct[i].run(args.engine) or 0 for i in order]
        def state_batch(order):
            return [function.evaluate(a=parameters[i][0],b=parameters[i][1]) for i in order]
        order=list(range(args.parameters))
        first_direct,first_direct_ms=timed(lambda:direct_batch(order))
        first_state,first_state_ms=timed(lambda:state_batch(order))
        assert first_direct==expected and first_state==expected
        samples={"independent_native_queries":[],"native_function_from_snapshot":[]}
        rng=random.Random(args.seed+rows)
        for repetition in range(args.repeats):
            rng.shuffle(order)
            engines=[("independent_native_queries",direct_batch),("native_function_from_snapshot",state_batch)]
            if repetition%2: engines.reverse()
            for name,call in engines:
                values,elapsed=timed(lambda:call(order))
                assert values==[expected[i] for i in order]
                samples[name].append(elapsed)
        # Structural evidence: evaluation/calculus must not dispatch another
        # fact query after the immutable snapshot has been built.
        from unittest.mock import patch
        with patch("algq.Query.run",side_effect=AssertionError("unexpected data rescan")):
            assert state_batch(order)==[expected[i] for i in order]
            assert da.evaluate(a=2,b=1)==4*sxx+2*sx-2*sxy
        ratio=statistics.median(samples["independent_native_queries"])/statistics.median(samples["native_function_from_snapshot"])
        return dict(rows=rows,selected_rows=int(n),parameters=parameters,all_correct=True,
                    setup_ms=dict(source_and_native_load=load_ms,moment_snapshot=build_ms,symbolic_reduction=compile_ms),
                    first_ms=dict(independent_native_queries=first_direct_ms,native_function_from_snapshot=first_state_ms),
                    warm={name:summary(values) for name,values in samples.items()},
                    warm_median_ratio=ratio,snapshot_metadata=getattr(bank,"metadata",{}),
                    expected_answer_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                    post_snapshot_evaluation_fact_queries=0,
                    device_sets=sorted({getattr(query,"device","unknown") for query in direct}))
    finally:
        bank.close()
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows",default="1000,100000,1000000")
    parser.add_argument("--parameters",type=int,default=25)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--threads",type=int,default=24)
    parser.add_argument("--engine",choices=("alg","gpu"),default="alg")
    parser.add_argument("--source",type=Path,default=ROOT,
                        help="Source checkout or preserved source/library archive")
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.parameters<1 or args.repeats<1:
        parser.error("parameters and repeats must be positive")
    rows=[int(value) for value in args.rows.split(",")]
    if any(value<1 for value in rows): parser.error("row counts must be positive")
    root=args.source.resolve()
    sys.path.insert(0,str(root))
    import torbles
    torbles.set_threads(args.threads)
    if (root/".git").exists():
        source_head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    elif (root/"MEASURED_RUNTIME.json").exists():
        source_head=json.loads((root/"MEASURED_RUNTIME.json").read_text())["commit"]
    else:
        source_head="unversioned"
    result=dict(experiment="fixed_scope_exact_polynomial_query_family",engine=args.engine,threads=args.threads,
                source=str(root),source_head=source_head,
                method="Actual serial batches; alternating paired order; exact comparisons outside timers; construction and first calls separate",
                limitations=["Synthetic signed integer columns, scalar fixed predicate, degree-two data basis",
                             "One process per invocation; within-process repetitions, not independent process rounds",
                             "Baseline runs independent native aggregate queries; it has no cross-query state reuse",
                             "Snapshot materialization uses several existing native queries, not a fused one-pass reducer",
                             "Coefficient calculus executes in native CPU code even when snapshot queries use CUDA",
                             "This is not a rerun or replacement of the mixed SSB matrix"],
                cases=[])
    for count in rows:
        case=run_case(count,args)
        result["cases"].append(case)
        print(json.dumps(dict(rows=count,correct=case["all_correct"],ratio=case["warm_median_ratio"])),flush=True)
    result["all_correct"]=all(case["all_correct"] for case in result["cases"])
    result["gpu_name"]=torbles.gpu_name() if args.engine=="gpu" else None
    if args.engine=="gpu" and "emulat" in result["gpu_name"].lower():
        raise RuntimeError("GPU benchmark requires real CUDA, not CPU emulation")
    if args.engine=="gpu" and any(
            not device.startswith("gpu") for case in result["cases"]
            for device in (*case["device_sets"],*case["snapshot_metadata"]["actual_devices"])):
        raise RuntimeError("GPU benchmark encountered a CPU row fallback")
    names=("mathematics.c","mathematics.h","mathematics.py","math_domains.py",
           "math_domain.c","math_domain.h","math_modules.c","math_modules.h","math_state.c","math_state.h")
    result["native_source_sha256"]={name:hashlib.sha256((root/"torbles"/name).read_bytes()).hexdigest()
                                    for name in names if (root/"torbles"/name).exists()}
    result["library_sha256"]=hashlib.sha256((root/"torbles/libalg.so").read_bytes()).hexdigest()
    if args.engine=="gpu":
        from torbles.algq import gpu_lib
        result["gpu_library_sha256"]=hashlib.sha256(Path(gpu_lib()._name).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
