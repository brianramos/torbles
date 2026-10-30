"""Independent native exact-arithmetic oracles and ASAN/UBSAN validation.

Run from the repository root in Linux/WSL (stdlib Python and GCC are sufficient):

    python tests/validate_mathematics_native.py
    python tests/validate_mathematics_native.py --receipt /tmp/math-receipt.json

Only isolated temporary binaries are built; torbles/libalg.so is untouched.
The receipt records source/test hashes, compiler identity, seeds, counts and the
exact compile commands. The native C harness checks lifetimes and UB; Python's
arbitrary-precision Fraction independently checks the mathematical results.
"""
import argparse
import ctypes as C
import datetime
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


ROOT=Path(__file__).resolve().parent.parent
PROGRAM_SEED=20261030
BANK_SEED=24030
LOW,HIGH=-(1<<63),(1<<63)-1


class Rational(C.Structure):
    _fields_=[("numerator",C.c_int64),("denominator",C.c_int64)]


class Term(C.Structure):
    _fields_=[("exponents",C.c_uint8*8),("coefficient",Rational)]


class Native:
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        pointer,integer,error=C.c_void_p,C.c_int32,C.POINTER(C.c_int32)
        signatures={
            "constant":([C.c_int64,C.c_int64,error],pointer),
            "variable":([integer,error],pointer),
            "add":([pointer,pointer,error],pointer),
            "sub":([pointer,pointer,error],pointer),
            "mul":([pointer,pointer,error],pointer),
            "pow":([pointer,C.c_uint32,error],pointer),
            "derivative":([pointer,integer,error],pointer),
            "integrate":([pointer,integer,error],pointer),
            "substitute":([pointer,integer,C.c_int64,C.c_int64,error],pointer),
            "evaluate":([pointer,C.POINTER(Rational),C.POINTER(Rational)],integer),
            "equal":([pointer,pointer],integer),
            "term_count":([pointer],integer),
            "get_term":([pointer,integer,C.POINTER(Term)],integer),
            "free":([pointer],None),
        }
        for name,(args,result) in signatures.items():
            function=getattr(self.lib,"tr_math_poly_"+name)
            function.argtypes,function.restype=args,result
            setattr(self,name,function)
        bank_signatures={
            "create":([integer,integer,C.POINTER(C.c_uint8),C.POINTER(C.c_int64),error],pointer),
            "reduce":([pointer,pointer,error],pointer),
            "count":([pointer,C.POINTER(C.c_int64)],integer),
            "free":([pointer],None),
        }
        for name,(args,result) in bank_signatures.items():
            function=getattr(self.lib,"tr_math_bank_"+name)
            function.argtypes,function.restype=args,result
            setattr(self,"bank_"+name,function)
        self.owned=[]

    def call(self,name,*arguments):
        error=C.c_int32()
        handle=getattr(self,name)(*arguments,C.byref(error))
        assert handle and error.value==0,(name,error.value)
        self.owned.append(handle)
        return handle

    def scalar(self,value):
        value=Fraction(value)
        return self.call("constant",value.numerator,value.denominator)

    def coefficients(self,handle):
        result={}
        count=self.term_count(handle)
        assert count>=0
        for index in range(count):
            term=Term()
            assert self.get_term(handle,index,C.byref(term))==0
            assert term.coefficient.denominator>0
            result[tuple(term.exponents)]=Fraction(term.coefficient.numerator,term.coefficient.denominator)
        return result

    def close(self):
        while self.owned:
            self.free(self.owned.pop())


def symbolic_add(left,right,sign=1):
    result=dict(left)
    for exponents,coefficient in right.items():
        result[exponents]=result.get(exponents,Fraction())+sign*coefficient
        if not result[exponents]:
            del result[exponents]
    return result


def symbolic_multiply(left,right):
    result={}
    for left_exponents,left_coefficient in left.items():
        for right_exponents,right_coefficient in right.items():
            exponents=tuple(a+b for a,b in zip(left_exponents,right_exponents))
            result[exponents]=result.get(exponents,Fraction())+left_coefficient*right_coefficient
    return {key:value for key,value in result.items() if value}


def symbolic_evaluate(polynomial,values):
    return sum((coefficient*math.prod(value**power for value,power in zip(values,exponents))
                for exponents,coefficient in polynomial.items()),Fraction())


def program_oracle(native):
    rng=random.Random(PROGRAM_SEED)
    for _attempt in range(200):
        pool=[]
        for variable in range(4):
            handle=native.call("variable",variable)
            exponents=[0]*8; exponents[variable]=1
            pool.append((handle,{tuple(exponents):Fraction(1)}))
        for _ in range(3):
            value=Fraction(rng.randint(-3,3),rng.randint(1,4))
            pool.append((native.scalar(value),{} if not value else {(0,)*8:value}))
        for _ in range(8):
            left,left_expected=rng.choice(pool)
            right,right_expected=rng.choice(pool)
            operation=rng.choice(("add","sub","mul"))
            degree=lambda poly:max([sum(exponents) for exponents in poly]+[0])
            if operation=="mul" and degree(left_expected)+degree(right_expected)>4:
                operation="add"
            result=native.call(operation,left,right)
            expected=(symbolic_multiply(left_expected,right_expected) if operation=="mul"
                      else symbolic_add(left_expected,right_expected,-1 if operation=="sub" else 1))
            assert native.coefficients(result)==expected
            assert native.coefficients(left)==left_expected and native.coefficients(right)==right_expected
            pool.append((result,expected))
        handle,expected=pool[-1]
        variable=rng.randrange(4)
        derivative,integral={},{}
        for exponents,coefficient in expected.items():
            changed=list(exponents)
            if exponents[variable]:
                changed[variable]-=1
                derivative[tuple(changed)]=coefficient*exponents[variable]
            changed=list(exponents); changed[variable]+=1
            integral[tuple(changed)]=coefficient/Fraction(changed[variable])
        differentiated=native.call("derivative",handle,variable)
        integrated=native.call("integrate",handle,variable)
        assert native.coefficients(differentiated)==derivative
        assert native.coefficients(integrated)==integral
        recovered=native.call("derivative",integrated,variable)
        assert native.equal(handle,recovered)==1
        value=Fraction(rng.randint(-3,3),rng.randint(1,4))
        substituted=native.call("substitute",handle,variable,value.numerator,value.denominator)
        expected_substitution={}
        for exponents,coefficient in expected.items():
            changed=list(exponents); changed[variable]=0
            expected_substitution=symbolic_add(expected_substitution,
                {tuple(changed):coefficient*value**exponents[variable]})
        assert native.coefficients(substituted)==expected_substitution
        values=[Fraction(rng.randint(-2,2),rng.randint(1,3)) for _ in range(8)]
        arguments=(Rational*8)(*(Rational(value.numerator,value.denominator) for value in values))
        output=Rational(100,7)
        assert native.evaluate(handle,arguments,C.byref(output))==0
        assert Fraction(output.numerator,output.denominator)==symbolic_evaluate(expected,values)
    return dict(seed=PROGRAM_SEED,programs=200,arithmetic_steps=1600,
                checks="Fraction polynomial arithmetic, operand immutability, derivative, antiderivative, substitution, evaluation")


def rational_edges(native):
    numerators=(LOW,LOW+1,-HIGH,-2,-1,0,1,2,HIGH-1,HIGH)
    denominators=(LOW,-HIGH,-2,-1,1,2,HIGH)
    constants=[]
    for numerator,denominator in itertools.product(numerators,denominators):
        expected=Fraction(numerator,denominator)
        error=C.c_int32()
        handle=native.constant(numerator,denominator,C.byref(error))
        fits=LOW<=expected.numerator<=HIGH and expected.denominator<=HIGH
        if not fits:
            assert not handle and error.value==-3
            continue
        assert handle and error.value==0
        native.owned.append(handle)
        assert native.coefficients(handle)==({(0,)*8:expected} if expected else {})
        constants.append((handle,expected))
    pairs=0
    for (left,a),(right,b) in itertools.product(constants,repeat=2):
        for operation,expected in (("add",a+b),("sub",a-b),("mul",a*b)):
            error=C.c_int32()
            handle=getattr(native,operation)(left,right,C.byref(error))
            fits=LOW<=expected.numerator<=HIGH and expected.denominator<=HIGH
            if fits:
                assert handle and error.value==0
                try:
                    assert native.coefficients(handle)==({(0,)*8:expected} if expected else {})
                finally:
                    native.free(handle)
            else:
                assert not handle and error.value==-3
        pairs+=1
    assert pairs==3969
    error=C.c_int32()
    assert not native.constant(1,0,C.byref(error)) and error.value==-1
    return dict(canonical_input_constants=len(constants),pairs=pairs,binary_results=3*pairs,
                checks="Every constant result/refusal compared with arbitrary-precision Fraction")


def moment_oracle(native):
    rng=random.Random(BANK_SEED)
    banks=0
    for ndata in range(5):
        basis=[exponents for exponents in itertools.product(range(3),repeat=ndata) if sum(exponents)<=2]
        for _iteration in range(12):
            count=rng.randrange(12)
            rows=[[rng.randint(-8,8) for _ in range(ndata)] for _ in range(count)]
            moments=[sum(math.prod(value**power for value,power in zip(row,exponents)) for row in rows)
                     for exponents in basis]
            exponents=(C.c_uint8*(len(basis)*ndata))(*(power for item in basis for power in item))
            values=(C.c_int64*len(moments))(*moments)
            error=C.c_int32()
            bank=native.bank_create(ndata,len(moments),exponents,values,C.byref(error))
            assert bank and not error.value
            try:
                actual_count=C.c_int64()
                assert native.bank_count(bank,C.byref(actual_count))==0 and actual_count.value==count
                handle=native.scalar(0); expected={}
                for _ in range(8):
                    powers=rng.choice(basis)
                    coefficient=Fraction(rng.randint(-4,4),rng.randint(1,3))
                    term=native.scalar(coefficient)
                    for variable,power in enumerate(powers):
                        if power:
                            term=native.call("mul",term,native.call("pow",native.call("variable",variable),power))
                    variable=ndata+rng.randrange(8-ndata); power=rng.randrange(3)
                    if power:
                        term=native.call("mul",term,native.call("pow",native.call("variable",variable),power))
                    handle=native.call("add",handle,term)
                    residual=[0]*8; residual[variable]=power
                    expected[tuple(residual)]=expected.get(tuple(residual),Fraction())+coefficient*moments[basis.index(powers)]
                reduced=native.bank_reduce(bank,handle,C.byref(error))
                assert reduced and not error.value
                native.owned.append(reduced)
                assert native.coefficients(reduced)=={key:value for key,value in expected.items() if value}
                if ndata:
                    cubic=native.call("pow",native.call("variable",0),3)
                    assert not native.bank_reduce(bank,cubic,C.byref(error)) and error.value==-5
            finally:
                native.bank_free(bank)
            banks+=1
    exponents=(C.c_uint8*3)(0,1,2)
    error=C.c_int32()
    for invalid in ((-1,0,0),(0,1,0)):
        assert not native.bank_create(1,3,exponents,(C.c_int64*3)(*invalid),C.byref(error)) and error.value==-1
    assert banks==60
    return dict(seed=BANK_SEED,banks=banks,data_variable_counts=list(range(5)),
                checks="Signed rows, empty bags, rational coefficients, retained parameter indices, missing cubic basis and malformed count states")


def source_hashes():
    paths=("torbles/mathematics.c","torbles/mathematics.h","torbles/torble.h",
           "tests/validate_mathematics_native.py","tests/mathematics_native_edges.c")
    return {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,default=ROOT/"results/algebra_foundation/native_validation.json")
    parser.add_argument("--cc",default="gcc")
    args=parser.parse_args()
    hashes=source_hashes()
    build=Path(tempfile.mkdtemp(prefix="torbles-mathematics-validation-"))
    common=[args.cc,"-Wall","-Wextra","-Werror","-I",str(ROOT/"torbles")]
    shared=common+["-O2","-shared","-fPIC",str(ROOT/"torbles/mathematics.c"),"-o",str(build/"mathematics.so")]
    edge=common+["-O1","-g","-fsanitize=address,undefined","-fno-sanitize-recover=all",
                 str(ROOT/"tests/mathematics_native_edges.c"),str(ROOT/"torbles/mathematics.c"),"-o",str(build/"edges")]
    for command in (shared,edge):
        subprocess.run(command,check=True)
    native=Native(build/"mathematics.so")
    try:
        programs=program_oracle(native)
        rationals=rational_edges(native)
        moments=moment_oracle(native)
        owned=len(native.owned)
    finally:
        native.close()
    environment=dict(os.environ,ASAN_OPTIONS="detect_leaks=1")
    sanitized=subprocess.run([str(build/"edges")],env=environment,check=True,text=True,capture_output=True)
    assert source_hashes()==hashes,"Native inputs changed during validation"
    receipt=dict(all_passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 compiler=subprocess.check_output([args.cc,"--version"],text=True).splitlines()[0],
                 python=sys.version,source_sha256=hashes,
                 runtime_source_sha256=hashlib.sha256(json.dumps({key:value for key,value in hashes.items()
                     if key.startswith("torbles/")},sort_keys=True).encode()).hexdigest(),
                 programs=programs,rational_edges=rationals,moment_banks=moments,
                 tracked_native_handles_freed=owned,
                 sanitizers=dict(address=True,undefined=True,leak_detection=True,pairs=3969,output=sanitized.stdout.strip()),
                 commands=dict(reproduce=[sys.executable,str(Path(__file__).resolve()),"--receipt",str(args.receipt)],
                               compile_shared=shared,compile_sanitized_edges=edge,run_sanitized_edges=[str(build/"edges")]),
                 build_directory=str(build),scope="Standalone native mathematical foundation; no engine library rebuild, no claims about unimplemented capability families")
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(receipt=str(args.receipt),all_passed=True,programs=200,
                         moment_banks=60,rational_pairs=3969,sanitizers="ASAN+UBSAN+leaks"),indent=2))


if __name__=="__main__":
    main()
