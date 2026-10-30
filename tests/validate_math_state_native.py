"""Independent native cache oracle and sanitizer/fault-injection validation.

Compiles into a new /tmp directory; never rebuilds the shared Torbles engine.
Run from the repository with the benchmark Python environment:

    python tests/validate_math_state_native.py \
        --receipt results/algebra_domains/native_state_validation.json
"""
import argparse
from collections import OrderedDict
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


ROOT=Path(__file__).resolve().parent.parent
SEED=20261030
NATIVE_SOURCES=("torbles/mathematics.c","torbles/math_domain.c","torbles/math_state.c")
TRACKED_SOURCES=NATIVE_SOURCES+(
    "torbles/mathematics.h","torbles/math_domain.h","torbles/math_state.h",
    "tests/validate_math_state_native.py","tests/math_state_native_edges.c")


class Spec(C.Structure):
    _fields_=[(name,C.c_int32) for name in (
        "abi","kind","carrier","nvars","ndata","validity","grouping","flags")]+[
        ("revision",C.c_uint64),("rows",C.c_uint64)]


class Stats(C.Structure):
    _fields_=[(name,C.c_uint64) for name in (
        "hits","misses","puts","evictions","bytes","max_bytes")]+[
        ("capacity",C.c_int32),("entries",C.c_int32)]


def bind(library,name,result,args):
    function=getattr(library,name)
    function.restype,function.argtypes=result,args
    return function


def powers(dimension):
    result=[(0,)*dimension]
    result.extend(tuple(int(index==variable) for index in range(dimension))
                  for variable in range(dimension))
    result.extend(tuple(int(index==left)+int(index==right) for index in range(dimension))
                  for left in range(dimension) for right in range(left,dimension))
    return result


def oracle(library,steps):
    ptr,int32,u64=C.c_void_p,C.c_int32,C.c_uint64
    domain=bind(library,"tr_math_domain_create",ptr,[C.POINTER(Spec),ptr,C.c_uint32,ptr,C.POINTER(int32)])
    domain_free=bind(library,"tr_math_domain_free",None,[ptr])
    bank_create=bind(library,"tr_math_bank_create",ptr,[int32,int32,ptr,ptr,C.POINTER(int32)])
    bank_free=bind(library,"tr_math_bank_free",None,[ptr])
    moment=bind(library,"tr_math_bank_moment",int32,[ptr,ptr,C.POINTER(C.c_int64)])
    create=bind(library,"tr_math_state_cache_create",ptr,[int32,u64,C.POINTER(int32)])
    put=bind(library,"tr_math_state_cache_put",int32,[ptr,ptr,ptr,C.c_uint32,ptr])
    get=bind(library,"tr_math_state_cache_get",ptr,[ptr,ptr,ptr,C.c_uint32,int32,C.POINTER(int32)])
    stats=bind(library,"tr_math_state_cache_stats",int32,[ptr,C.POINTER(Stats)])
    free=bind(library,"tr_math_state_cache_free",None,[ptr])
    assert C.sizeof(Spec)==48 and C.sizeof(Stats)==56
    rng=random.Random(SEED)
    operations=0
    for dimension in range(5):
        exponents=powers(dimension)
        flat=(C.c_uint8*(len(exponents)*dimension))(*(value for term in exponents for value in term))
        domains=[]
        try:
            for index in range(17):
                key=f"scope/{index:03}".encode()
                error=int32()
                value=domain(C.byref(Spec(1,2,1,dimension,dimension,0,0,0,index,10)),
                             key,len(key),None,C.byref(error))
                assert value and not error.value
                domains.append(value)
            for capacity in (1,2,7):
                error=int32()
                cache=create(capacity,100000,C.byref(error))
                assert cache and not error.value
                expected=OrderedDict()
                hits=misses=puts=evictions=0
                try:
                    for _ in range(steps):
                        domain_index,basis_index=rng.randrange(17),rng.randrange(3)
                        key=(domain_index,basis_index)
                        basis=f"columns/{basis_index}".encode()+b"\0exact"
                        if rng.random()<.55:
                            rows=[[rng.randrange(-8,9) for _ in range(dimension)]
                                  for _ in range(rng.randrange(5))]
                            values=[]
                            for term in exponents:
                                total=0
                                for row in rows:
                                    value=1
                                    for coordinate,power in zip(row,term):
                                        value*=coordinate**power
                                    total+=value
                                values.append(total)
                            array=(C.c_int64*len(values))(*values)
                            bank=bank_create(dimension,len(exponents),flat if dimension else None,
                                             array,C.byref(error))
                            assert bank and not error.value
                            try:
                                assert put(cache,domains[domain_index],basis,len(basis),bank)==0
                            finally:
                                bank_free(bank)
                            puts+=1
                            if key in expected:
                                expected.pop(key)
                            elif len(expected)==capacity:
                                expected.popitem(last=False)
                                evictions+=1
                            expected[key]=values
                        else:
                            bank=get(cache,domains[domain_index],basis,len(basis),dimension,C.byref(error))
                            assert not error.value
                            if key in expected:
                                assert bank
                                hits+=1
                                values=expected.pop(key)
                                expected[key]=values
                                try:
                                    for term,wanted in zip(exponents,values):
                                        code=(C.c_uint8*dimension)(*term)
                                        value=C.c_int64()
                                        assert moment(bank,code if dimension else None,C.byref(value))==0
                                        assert value.value==wanted
                                finally:
                                    bank_free(bank)
                            else:
                                assert not bank
                                misses+=1
                        measured=Stats()
                        assert stats(cache,C.byref(measured))==0
                        assert (measured.hits,measured.misses,measured.puts,measured.evictions,measured.entries)==(
                            hits,misses,puts,evictions,len(expected))
                        assert measured.bytes<=measured.max_bytes and measured.capacity==capacity
                        operations+=1
                finally:
                    free(cache)
        finally:
            for value in domains:
                domain_free(value)
    return dict(seed=SEED,operations=operations,data_dimensions=list(range(5)),capacities=[1,2,7],
                checks=["OrderedDict LRU identity/order","independent row-arithmetic moments",
                        "source input banks freed immediately after put","binary basis keys",
                        "exact statistics and retained-byte accounting"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,default=ROOT/"results/algebra_domains/native_state_validation.json")
    parser.add_argument("--steps",type=int,default=1000,help="Operations per dimension/capacity pair")
    args=parser.parse_args()
    if not 1<=args.steps<=100000:
        parser.error("steps must be between one and 100000")
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in TRACKED_SOURCES}
    scratch=Path(tempfile.mkdtemp(prefix="torbles-math-state-validation-"))
    shared=scratch/"state.so"
    common=["gcc","-Wall","-Wextra","-Werror","-I",str(ROOT/"torbles")]
    shared_command=common+["-O2","-shared","-fPIC",*[str(ROOT/name) for name in NATIVE_SOURCES],"-o",str(shared)]
    subprocess.run(shared_command,check=True)
    result=oracle(C.CDLL(str(shared)),args.steps)
    sanitizer=scratch/"state_edges"
    sanitizer_command=common+["-O1","-g","-fsanitize=address,undefined","-fno-omit-frame-pointer",
        *[str(ROOT/name) for name in NATIVE_SOURCES],str(ROOT/"tests/math_state_native_edges.c"),
        "-Wl,--wrap=malloc,--wrap=calloc","-o",str(sanitizer)]
    subprocess.run(sanitizer_command,check=True)
    env=dict(os.environ,ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
             UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    checked=subprocess.run([str(sanitizer)],env=env,check=True,capture_output=True,text=True)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in source_hashes.items())
    receipt=dict(all_passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
                 source_sha256=source_hashes,compiler=subprocess.check_output(["gcc","--version"],text=True).splitlines()[0],
                 python=sys.version,random_oracle=result,
                 sanitizer=dict(address=True,undefined=True,leak_detection=True,
                     checks=["independent returned-bank lifetime","LRU and byte eviction",
                             "three staged-allocation failures leave cache/statistics unchanged",
                             "returned-clone allocation failure leaves statistics unchanged",
                             "replacement ownership","NULL/carrier/count refusal",
                             "embedded-NUL key distinction","same-key revision distinction",
                             "clear resets allocations/counters"],output=checked.stdout.strip()),
                 commands=dict(reproduce=[sys.executable,str(Path(__file__).resolve()),"--receipt",str(args.receipt.resolve())],
                               compile_shared=shared_command,compile_sanitizer=sanitizer_command,
                               run_sanitizer=[str(sanitizer)]),scratch=str(scratch))
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(receipt=str(args.receipt),all_passed=True,random_operations=result["operations"],
                          sanitizers="ASAN/UBSAN/leaks/allocation-failure atomicity"),indent=2))


if __name__=="__main__":
    main()
