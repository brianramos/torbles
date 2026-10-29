#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# Every correctness test of torbles. Each compares the algebraic engine with the plain columnar baseline and/or DuckDB,
# across join forms, thread counts and GROUP BY modes; the GPU tests compare the GPU executor with the CPU engine
# (the real GPU when torbles/libgpu.so exists, else the CPU emulation of the same kernels).
#   bash tests/run_all.sh [quick]
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/torbles:$PWD/tests"
[ -f torbles/libgpu.so ] && export TORBLES_GPU_LIB="$PWD/torbles/libgpu.so"
missing=$(python3 -c "
import importlib.util as u
print(' '.join(m for m in ('numpy', 'pandas', 'duckdb') if u.find_spec(m) is None))")
if [ -n "$missing" ]; then   # one clear message instead of a failure per test
  echo "the tests need Python packages that $(command -v python3) does not have: $missing"
  echo "install them (pip install $missing, or pip install '.[test]') or activate the environment that has them"
  echo "(wsl/run_validation.sh makes one: source ~/torble_gpu_run/venv/bin/activate, or ~/torbles_run/venv/bin/activate)"
  exit 1
fi
python3 -c "import algq; print('torbles', algq.__version__, '| CPU kernel:', algq.mm_kernel(), '| GPU:', algq.gpu_name())" || exit 1
fails=0
run() {   # name, command...
  local name=$1; shift
  local t0=$(date +%s%N) out; out=$("$@" 2>&1); local rc=$?
  local ms=$(( ($(date +%s%N) - t0) / 1000000 )); local dt="$((ms / 1000)).$(( (ms % 1000) / 100 ))"
  local last=$(printf '%s\n' "$out" | tail -1)
  if [ $rc -eq 0 ] && printf '%s' "$last" | grep -qE "ALL MATCH|ALL FIX TESTS PASSED|NO MISMATCHES"; then printf '  %-44s ok   (%ss)\n' "$name" "$dt"
  else printf '  %-44s FAIL (%ss)\n' "$name" "$dt"; printf '%s\n' "$out" | tail -15 | sed 's/^/      /'; fails=$((fails + 1)); fi
}
echo "== CPU engine vs plain columns vs DuckDB"
run "star schema, every join form, 1/2 threads"   python3 tests/check_small.py
run "edge cases and fixes"                       python3 tests/test_fixes.py
run "order-encoded bounds (rnd as bounds)"        python3 tests/check_bounds.py
run "range / band / inequality joins"             python3 tests/check_ineq.py
run "derived surrogates"                          python3 tests/check_surrogate.py
run "additive deltas (moved dimension rows)"      python3 tests/check_moves.py
run "self-extending (64-bit) columns"             python3 tests/check_extend.py
run "compiled-plan reuse"                         python3 tests/check_plancache.py
run "values wider than 32 bits are refused"        python3 tests/check_widths.py
echo "== typed data: loaders, key design, persistence"
run "typed columns, every engine and join form"  python3 tests/check_typed.py
run "loaders: DDL, files, Frictionless, frames"   python3 tests/check_loaders.py
run "automatic key design"                        python3 tests/check_keydesign.py
run "save / load, integrity on load"              python3 tests/check_persist.py
run "sums and group codes past 64 bits"           python3 tests/check_wide.py
run "  ... with the row kernel on every tile"     env TORBLES_MM_ROWS=2 python3 tests/check_wide.py 20000
run "  ... with plane popcounts only"             env TORBLES_MM_ROWS=0 TORBLES_FACTOR=0 python3 tests/check_wide.py 20000
run "differential fuzzing vs DuckDB (40 cases)"   python3 -m torbles.fuzz --seed 2026 --cases 40
echo "== GROUP BY modes"
run "previous GROUP BY path (mask split / rows)"  env TORBLES_GROUPBY=split python3 tests/check_small.py
run "planner without native planning or cache"    env TORBLES_NATIVE_PLAN=0 TORBLES_PLAN_CACHE=0 python3 tests/check_small.py
echo "== GPU executor vs CPU engine ($( [ -n "${TORBLES_GPU_LIB:-}" ] && echo CUDA || echo emulated ))"
run "GPU per-word kernel (default)"               env TORBLES_GPU_MM=word python3 tests/check_gpu.py
run "GPU tile matrix product, __popc"             env TORBLES_GPU_MM=popc python3 tests/check_gpu.py
run "GPU tile matrix product, tensor cores"       env TORBLES_GPU_MM=mma python3 tests/check_gpu.py
run "GPU with dense group arrays (split mode)"    env TORBLES_GROUPBY=split python3 tests/check_gpu.py
if [ "${1:-}" != quick ]; then
  echo "== CPU kernel variants"
  run "matrix product, scalar popcount"           env TORBLES_POPCNT=scalar python3 tests/check_small.py
fi
echo
[ $fails -eq 0 ] && echo "ALL TESTS PASSED" || { echo "$fails TEST(S) FAILED"; exit 1; }
