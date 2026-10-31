#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# The mathematical modules' tests against the native code built with AddressSanitizer and UndefinedBehaviorSanitizer
# (the engine library and the emulated CUDA kernels, in a temporary directory; the repository's libraries are left
# alone). Any sanitizer report fails the run.
#   bash tests/sanitize_math.sh [unittest file pattern, default test_math*.py] [-k filter, e.g. lin001]
set -uo pipefail
cd "$(dirname "$0")/.."
PATTERN=${1:-'test_math*.py'}; FILTER=${2:-}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
SAN="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined"
T=torbles
cc $SAN -shared -fPIC $T/alg.c $T/relationships.c $T/relationships_optimizer.c $T/relationships_system.c \
   $T/relationships_dag.c $T/relationships_stage.c $(ls $T/math*.c | sort) -o "$TMP/libalg.so" -lpthread -lm || exit 1
g++ $SAN -fwrapv -DEMULATE -x c++ -shared -fPIC $(ls $T/math_*_gpu.cu | sort) -o "$TMP/libmathgpu_emu.so" || exit 1
ASAN_LIB=$(cc -print-file-name=libasan.so); UBSAN_LIB=$(cc -print-file-name=libubsan.so)
export LD_PRELOAD="$ASAN_LIB:$UBSAN_LIB"
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:verify_asan_link_order=0:halt_on_error=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
export TORBLES_ALG_LIB="$TMP/libalg.so" TORBLES_MATH_GPU_LIB="$TMP/libmathgpu_emu.so" TORBLES_EMU_SCRAMBLE=1
export PYTHONPATH="$PWD:$PWD/torbles:$PWD/tests"
python3 -m unittest discover -s tests -p "$PATTERN" ${FILTER:+-k "$FILTER"} 2>&1 | tee "$TMP/out.log" | tail -5
rc=${PIPESTATUS[0]}
if grep -qE "ERROR: AddressSanitizer|runtime error:|ERROR: LeakSanitizer" "$TMP/out.log"; then echo "SANITIZER REPORT"; exit 1; fi
[ $rc -eq 0 ] && echo "ALL MATCH (sanitized)" || exit $rc
