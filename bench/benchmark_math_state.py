"""Measure exact scalar moment-cache reuse and native descriptor overhead.

Each timed request constructs a MomentBasis, reads its cross moment, and closes
it. Verification, snapshot-metadata inspection, and cache-stat sampling happen
after each batch timer. Source fingerprint construction remains inside timing.
This is a synthetic bounded-cache experiment, separate from the SSB matrix.
"""
import argparse
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
SOURCE_FILES=(
    "torbles/mathematics.c","torbles/mathematics.h","torbles/mathematics.py",
    "torbles/math_domain.c","torbles/math_domain.h","torbles/math_domains.py",
    "torbles/math_modules.c","torbles/math_modules.h","torbles/math_state.c",
    "torbles/math_state.h","bench/benchmark_math_state.py")


def timed(call):
    started=time.perf_counter_ns()
    result=call()
    return result,(time.perf_counter_ns()-started)/1e6


def summarize(samples):
    return dict(raw_ms=samples,median_ms=statistics.median(samples),
                min_ms=min(samples),max_ms=max(samples))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_snapshot(bank):
    metadata=bank.metadata
    names=("row_count","cache_enabled","cache_hit","cache_store_status",
           "native_query_calls","actual_devices","domain_key_sha256",
           "row_domain",
           "engine_requested","gpu_executor","gpu_emulated",
           "coefficient_executor","construction")
    return {name:metadata[name] for name in names}


def checked_batch(run,order,expected,*,hit,query_calls,selected_rows):
    (values,banks),elapsed=timed(lambda:run(order))
    # Banks are closed inside run(). Their immutable receipt stays accessible.
    snapshots=[compact_snapshot(bank) for bank in banks]
    assert values==[expected for _ in order]
    assert all(snapshot["row_count"]==selected_rows for snapshot in snapshots)
    assert all(snapshot["cache_hit"] is hit for snapshot in snapshots)
    assert sum(snapshot["native_query_calls"] for snapshot in snapshots)==query_calls
    if hit:
        assert all(not snapshot["actual_devices"] for snapshot in snapshots)
    return dict(serial_ms=elapsed,order=list(order),native_query_calls=query_calls,
                values=values,snapshots=snapshots)


def descriptor_bench(repeats):
    from torbles.math_domains import Domain
    cases=[]
    for count in (1,16,128):
        samples={"construct":[],"equal_compare":[],"different_revision_compare":[]}
        owned_bytes=[]
        for _ in range(repeats):
            domains=clones=different=[]
            try:
                def construct():
                    return [Domain("snapshot",f"descriptor/{index:03}/scope",carrier="integer",
                                   nvars=2,ndata=2,revision=index,universe_rows=1_000_000)
                            for index in range(count)]
                domains,elapsed=timed(construct)
                samples["construct"].append(elapsed)
                clones=[domain.clone() for domain in domains]
                different=[Domain("snapshot",domain.key,carrier="integer",nvars=2,ndata=2,
                                  revision=index+1,universe_rows=1_000_000)
                           for index,domain in enumerate(domains)]
                equal,elapsed=timed(lambda:[left.same_as(right) for left,right in zip(domains,clones)])
                samples["equal_compare"].append(elapsed)
                unequal,elapsed=timed(lambda:[left.same_as(right) for left,right in zip(domains,different)])
                samples["different_revision_compare"].append(elapsed)
                assert equal==[True]*count and unequal==[False]*count
                owned_bytes.append(sum(domain.bytes for domain in domains))
            finally:
                for domain in (*domains,*clones,*different):
                    domain.close()
        cases.append(dict(descriptors=count,all_correct=True,
                          timing_ms={name:summarize(values) for name,values in samples.items()},
                          owned_original_descriptor_bytes=owned_bytes,
                          equality_checks_per_batch=count,mismatch_checks_per_batch=count,
                          actual_native_aggregate_queries=0,backend="native CPU"))
    return dict(cases=cases,
                method="Construct owned row-domain descriptors; compare with owned exact clones and same-key changed-revision descriptors",
                scope="No table scans or operator execution; false identity comparisons measure mismatch rejection only",
                cleanup="Owned clones/comparison descriptors are prepared and all descriptors closed outside timing")


def data_case(rows,args):
    import numpy as np
    import pandas as pd
    import torbles
    from torbles.mathematics import MomentBasis
    from torbles.math_domains import MomentCache

    index=np.arange(rows,dtype=np.int64)
    frame=pd.DataFrame(dict(x=index%31-15,y=(index*7)%43-21,flag=index%5))
    database,load_ms=timed(lambda:torbles.from_ddl(
        "CREATE TABLE t (x BIGINT,y BIGINT,flag BIGINT)",{"t":frame}))
    table=database["t"]
    scopes=[database.query(table).where(table.flag>=2) for _ in range(args.requests)]
    selected,expected=database.con.execute(
        "SELECT count(*),sum(x*y) FROM t WHERE flag>=2").fetchone()
    changed_count,changed_expected=database.con.execute(
        "SELECT count(*),sum(x*y) FROM t WHERE flag>=3").fetchone()
    selected,expected=int(selected),int(expected or 0)
    changed_count,changed_expected=int(changed_count),int(changed_expected or 0)
    cache,cache_create_ms=timed(lambda:MomentCache(capacity=64))
    all_devices=set()
    all_query_calls=0
    try:
        def requests(order,selected_cache,queries=scopes):
            values,banks=[],[]
            for item in order:
                bank=MomentBasis(queries[item],["x","y"],engine=args.engine,cache=selected_cache)
                try:
                    values.append(bank.moment((1,1)))
                finally:
                    bank.close()
                banks.append(bank)
            return values,banks
        def record(batch):
            nonlocal all_query_calls
            all_query_calls+=batch["native_query_calls"]
            for snapshot in batch["snapshots"]:
                all_devices.update(snapshot["actual_devices"])
        uncached=lambda order:requests(order,None)
        cached=lambda order:requests(order,cache)
        cold=checked_batch(cached,[0],expected,hit=False,query_calls=6,selected_rows=selected)
        record(cold)
        order=list(range(args.requests))
        first_uncached=checked_batch(uncached,order,expected,hit=False,
                                    query_calls=6*args.requests,selected_rows=selected)
        record(first_uncached)
        first_reused=checked_batch(cached,order,expected,hit=True,query_calls=0,selected_rows=selected)
        record(first_reused)
        batches=[]
        rng=random.Random(args.seed+rows)
        for repetition in range(args.repeats):
            rng.shuffle(order)
            paths=[("uncached",uncached), ("compatible_cache",cached)]
            if repetition%2:
                paths.reverse()
            for name,call in paths:
                before=cache.stats
                batch=checked_batch(call,order,expected,hit=name=="compatible_cache",
                                    query_calls=0 if name=="compatible_cache" else 6*args.requests,
                                    selected_rows=selected)
                after=cache.stats
                assert after["misses"]==before["misses"]
                assert after["hits"]-before["hits"]==(args.requests if name=="compatible_cache" else 0)
                batch.update(path=name,repetition=repetition,cache_stats_before=before,cache_stats_after=after)
                record(batch)
                batches.append(batch)
        # Check an incompatible predicate and two explicit invalidation paths
        # separately; these calls are not pooled into compatible warm medians.
        changed=[database.query(table).where(table.flag>=3)]
        changed_run=lambda order:requests(order,cache,changed)
        before=cache.stats
        incompatible=checked_batch(changed_run,[0],changed_expected,hit=False,
                                   query_calls=6,selected_rows=changed_count)
        incompatible.update(cache_stats_before=before,cache_stats_after=cache.stats)
        assert incompatible["cache_stats_after"]["misses"]==before["misses"]+1
        assert incompatible["snapshots"][0]["domain_key_sha256"]!=cold["snapshots"][0]["domain_key_sha256"]
        record(incompatible)
        _,clear_ms=timed(cache.clear)
        manual_refresh=checked_batch(cached,[0],expected,hit=False,query_calls=6,selected_rows=selected)
        manual_refresh["clear_ms"]=clear_ms
        assert manual_refresh["snapshots"][0]["domain_key_sha256"]==cold["snapshots"][0]["domain_key_sha256"]
        record(manual_refresh)
        # Flip+repair restores the same synthetic values but advances revision.
        # The old snapshot is deliberately retained in cache to test identity.
        table.flip(0,0)
        table.repair()
        before=cache.stats
        revision_refresh=checked_batch(cached,[0],expected,hit=False,query_calls=6,selected_rows=selected)
        revision_refresh.update(cache_stats_before=before,cache_stats_after=cache.stats)
        assert revision_refresh["cache_stats_after"]["misses"]==before["misses"]+1
        assert revision_refresh["snapshots"][0]["row_domain"]["revision"]>cold["snapshots"][0]["row_domain"]["revision"]
        record(revision_refresh)
        from unittest.mock import patch
        with patch("algq.Query.run",side_effect=AssertionError("unexpected aggregate query on cache hit")):
            structural=checked_batch(cached,order,expected,hit=True,query_calls=0,selected_rows=selected)
        record(structural)
        warm={name:summarize([batch["serial_ms"] for batch in batches if batch["path"]==name])
              for name in ("uncached","compatible_cache")}
        if args.engine=="gpu" and any(not device.startswith("gpu") for device in all_devices):
            raise RuntimeError(f"CUDA materialization encountered a CPU row fallback: {sorted(all_devices)}")
        return dict(rows=rows,selected_rows=selected,requests_per_batch=args.requests,all_correct=True,
                    setup_ms=dict(source_and_native_load=load_ms,cache_create=cache_create_ms),
                    cold_cache_first_request=cold,first_batches=dict(uncached=first_uncached,compatible_cache=first_reused),
                    warm_ms=warm,warm_median_ratio=warm["uncached"]["median_ms"]/warm["compatible_cache"]["median_ms"],
                    batches=batches,incompatible_predicate_miss=incompatible,
                    explicit_clear_refresh=manual_refresh,revision_refresh_after_repair=revision_refresh,
                    structural_no_aggregate_query_check=structural,final_cache_stats=cache.stats,
                    materialization_actual_devices=sorted(all_devices),total_recorded_native_queries=all_query_calls,
                    oracle=dict(method="Independent DuckDB SQL on original synthetic columns",
                                cross_moment=expected,incompatible_cross_moment=changed_expected))
    finally:
        cache.close()
        database.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine",choices=("alg","gpu"),default="alg")
    parser.add_argument("--rows",default="1000,100000,1000000")
    parser.add_argument("--requests",type=int,default=5)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--threads",type=int,default=24)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[int(value) for value in args.rows.split(",")]
    if not rows or any(value<5 for value in rows):
        parser.error("each synthetic row count must be at least five")
    if not 1<=args.requests<=128 or not 1<=args.repeats<=100 or args.threads<1:
        parser.error("requests must be 1..128, repeats 1..100 and threads positive")
    import torbles
    torbles.set_threads(args.threads)
    source={name:digest(ROOT/name) for name in SOURCE_FILES}
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    library=digest(ROOT/"torbles/libalg.so")
    result=dict(experiment="typed_scalar_moment_state_reuse",source_head=head,source_sha256=source,
                library_sha256=library,engine_requested=args.engine,threads=args.threads,
                repeats=args.repeats,requests_per_batch=args.requests,
                descriptor_microbench=descriptor_bench(args.repeats),cases=[],
                method="Actual serial construct/moment/close batches; same seeded paired request order; alternating path order; exact verification and metadata collection outside timers",
                limitations=[
                    "Bounded scalar non-NULL integer degree-two states; no grouped, fused or global automatic planner claim",
                    "Cache-hit timing includes source fingerprint validation and may read used-row metadata proportional to universe size",
                    "Zero fact aggregate calls does not mean zero metadata reads or constant-time cache-hit construction",
                    "One process per backend, ascending scales and within-process repetitions; no independent process confidence interval",
                    "Cache is populated before uncached first batches; shared executor/data residency is already established",
                    "Each miss builds six existing aggregate queries; five uncached requests build thirty, not a fused scan",
                    "Compatible cache hits execute native CPU cloning/lookup with no row executor; empty device lists are deliberate",
                    "Domain microbench prepares native bindings before data cases; cold cache first request is not full process startup",
                    "This synthetic experiment is separate from the mixed SSB matrix"])
    for count in rows:
        case=data_case(count,args)
        result["cases"].append(case)
        print(json.dumps(dict(rows=count,correct=True,uncached_ms=case["warm_ms"]["uncached"]["median_ms"],
                              cache_ms=case["warm_ms"]["compatible_cache"]["median_ms"])),flush=True)
    result["gpu_executor"]=torbles.gpu_name() if args.engine=="gpu" else None
    if args.engine=="gpu":
        if "emulat" in result["gpu_executor"].lower():
            raise RuntimeError("GPU benchmark requires real CUDA, not emulation")
        from torbles.algq import gpu_lib
        result["gpu_library_sha256"]=digest(Path(gpu_lib()._name))
    assert source=={name:digest(ROOT/name) for name in SOURCE_FILES}
    assert library==digest(ROOT/"torbles/libalg.so")
    assert head==subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    result["all_correct"]=all(case["all_correct"] for case in result["cases"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
