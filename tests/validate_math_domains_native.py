"""Reproduce warning-clean and sanitizer checks for domains/operator dispatch.

Builds only in a fresh temporary directory; the engine libraries and source
datasets are never changed. Run with the existing validation Python in WSL.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT=Path(__file__).resolve().parent.parent
SOURCES=("torbles/mathematics.c","torbles/mathematics.h","torbles/math_domain.c",
         "torbles/math_domain.h","torbles/math_modules.c","torbles/math_modules.h",
         "tests/math_domain_native_edges.c","tests/validate_math_domains_native.py")


def run(argv, *, env=None):
    result=subprocess.run(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        raise RuntimeError(f"Native validation command failed ({result.returncode}): {argv!r}\n{result.stdout}")
    return result.stdout.strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,default=ROOT/"results/algebra_domains/native_domain_validation.json")
    parser.add_argument("--cc",default="gcc")
    args=parser.parse_args()
    build=Path(tempfile.mkdtemp(prefix="torbles-domain-validation-"))
    inputs=[str(ROOT/path) for path in ("tests/math_domain_native_edges.c","torbles/mathematics.c",
                                      "torbles/math_domain.c","torbles/math_modules.c")]
    common=[args.cc,"-std=c11","-Wall","-Wextra","-Werror","-I",str(ROOT/"torbles")]
    release=common+["-O2",*inputs,"-o",str(build/"domain_edges")]
    sanitizer=common+["-O1","-g","-fsanitize=address,undefined","-fno-omit-frame-pointer",
                      *inputs,"-o",str(build/"domain_edges_sanitized")]
    run(release)
    release_output=run([str(build/"domain_edges")])
    run(sanitizer)
    environment=dict(os.environ,ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                     UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    sanitizer_output=run([str(build/"domain_edges_sanitized")],env=environment)
    if "Typed domains and executable registry edge checks passed." not in release_output or release_output!=sanitizer_output:
        raise AssertionError("Native validation did not emit its completion marker")
    receipt=dict(all_passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
                 compiler=run([args.cc,"--version"]).splitlines()[0],python=platform.python_version(),
                 source_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES},
                 checks=["ABI sizes and alignment","owned semantic keys and origins survive caller storage changes",
                         "exact identity includes kind, carrier, validity, grouping, revision and universe",
                         "wrong context, source origin, carrier, geometry and observed-variable calculus are refused",
                         "actual native polynomial arithmetic, moment contraction, parameter integration and evaluation",
                         "aggregate/parameter composition preserves source and rejects raw observed-coordinate mixing",
                         "output remains unchanged on invalid native evaluation","unsupported module/index checks",
                         "large constant exponent shape-cost arithmetic","function survives source descriptor/bank release"],
                 sanitizers=dict(address=True,undefined=True,leak_detection=True,output=sanitizer_output),
                 release_output=release_output,
                 commands=dict(reproduce=[sys.executable,str(Path(__file__).resolve()),"--receipt",str(args.receipt.resolve())],
                               release_compile=release,sanitizer_compile=sanitizer,
                               sanitized_run=[str(build/"domain_edges_sanitized")]),
                 limitations=["Descriptor compatibility is checked over declared semantic identities, not a proof that arbitrary supplied moments came from a physical source.",
                              "Only current rational-polynomial and scalar-moment module kernels are registered; metadata for other carrier/domain kinds does not implement their mathematics.",
                              "All registry operations execute in native CPU code; no CUDA operator kernel is claimed.",
                              "Opaque native handles require valid ownership; concurrent close/execution must be serialized by callers."])
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(all_passed=True,receipt=str(args.receipt),output=sanitizer_output),indent=2))


if __name__=="__main__": main()
