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
print(' '.join(m for m in ('numpy', 'pandas', 'duckdb', 'pytz', 'pyuca') if u.find_spec(m) is None))")
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
  if [ $rc -eq 0 ] && printf '%s' "$last" | grep -qE "ALL MATCH|ALL FIX TESTS PASSED|NO MISMATCHES|^OK( \(skipped=[0-9]+\))?$"; then printf '  %-44s ok   (%ss)\n' "$name" "$dt"
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
run "tables wider than a torble: parts, groups"   python3 tests/check_many_columns.py
run "sums and group codes past 64 bits"           python3 tests/check_wide.py
run "  ... with the row kernel on every tile"     env TORBLES_MM_ROWS=2 python3 tests/check_wide.py 20000
run "  ... with plane popcounts only"             env TORBLES_MM_ROWS=0 TORBLES_FACTOR=0 python3 tests/check_wide.py 20000
run "differential fuzzing vs DuckDB (40 cases)"   python3 -m torbles.fuzz --seed 2026 --cases 40
echo "== SQL (db.sql and the Python API)"
run "MIN/MAX by plane elimination, row bitmaps"  python3 tests/check_minmax.py
run "SQL vs DuckDB, ISO forms, Python API, fuzzing" python3 tests/check_sql.py
run "  ... on the plain columnar baseline engine"   env TORBLES_SQL_ENGINE=plain python3 tests/check_sql.py
run "  ... on the GPU executor"                     env TORBLES_SQL_ENGINE=gpu python3 tests/check_sql.py
run "snowflake FK filters and group keys"        python3 tests/check_snowflake.py
run "stored hierarchy filters and join headers" python3 tests/check_join_header.py
run "shared SQL workload states"              python3 -m unittest check_sql_workload check_sql_algebra
run "native periodic SQL reductions"          python3 tests/check_sql_periodic.py
run "direct native GROUP BY states"             python3 tests/check_direct_groups.py
run "raw-plane top groups and fused AVG"        python3 tests/check_raw_plane_group.py
run "encoded exact distinct and ranked groups"  python3 -m unittest check_encoded_groups check_sql_interchanges
run "bounded bitplane row ordering"             python3 -m unittest check_rowplanes
run "dictionary predicates and projections"     python3 -m unittest check_text_planes check_text_hist
run "ISO forms: zones, collations, samples, ..."  python3 tests/check_iso.py
run "  ... on the plain columnar baseline engine"   env TORBLES_SQL_ENGINE=plain python3 tests/check_iso.py
run "  ... on the GPU executor"                     env TORBLES_SQL_ENGINE=gpu python3 tests/check_iso.py
echo "== release review 2026-11-01: regression tests (R01 also in CI: tests/check_min_deps.py on the wheel)"
run "SQL casts, overflow, collations, null-safe joins" python3 tests/check_review_sql.py
run "engine API NULL aggregates, SQL of every measure"  python3 tests/check_review_api.py
run "native table lifetimes, empty tables"            python3 -m unittest discover -s tests -p test_review_native.py
run "array algebra vs row by row and DuckDB"     python3 tests/check_arrays.py
run "  ... on the plain columnar baseline engine"   env TORBLES_SQL_ENGINE=plain python3 tests/check_arrays.py 250 6
run "  ... on the GPU executor"                     env TORBLES_SQL_ENGINE=gpu python3 tests/check_arrays.py 250 8
run "rank selection, cell bits, join partial sums" python3 tests/check_algebra.py 120 7
run "  ... on the plain columnar baseline engine"   env TORBLES_SQL_ENGINE=plain python3 tests/check_algebra.py 50 8
run "  ... on the GPU executor"                     env TORBLES_SQL_ENGINE=gpu python3 tests/check_algebra.py 50 9
run "windows whose values every row outputs"    python3 tests/check_window_output.py
run "encoded top-k and row numbers"             python3 tests/check_encoded_order.py
run "GPU bitplane top-k and SQL fallback"        python3 tests/check_mask_topk.py
run "full raw-plane order and ROW_NUMBER"      python3 tests/check_raw_plane_order.py
run "CUDA ordered windows over raw planes"       python3 tests/check_cuda_ordered_windows.py
run "algebraic refresh and compaction"         python3 tests/check_refresh.py
run "SQL INSERT, UPDATE, DELETE and MERGE"     python3 tests/check_sql_dml.py
run "durable SQL and crash recovery"           python3 tests/check_durable.py
run "the release fuzzing campaign's faults"     python3 tests/check_campaign_sql.py
echo "== GROUP BY modes"
run "native relationship discovery and composition" python3 -m unittest discover -s tests -p test_relationships.py
run "native chains vs combined algebraic systems" python3 -m unittest discover -s tests -p test_relationship_systems.py
run "proof-aware compiled plan reuse"             python3 -m unittest discover -s tests -p test_relationship_cache.py
run "0.2 adaptive matrix integration"              python3 -m unittest discover -s tests -p test_upstream_enhancements.py
run "physical cache and GPU profile elision"       python3 -m unittest discover -s tests -p test_physical_cache.py
run "hybrid native regions and shared DAGs"         python3 -m unittest discover -s tests -p test_relationship_dag.py
run "native whole-stage algebra and finalization"  python3 -m unittest discover -s tests -p test_relationship_stages.py
run "native mathematical functions and moments"   python3 -m unittest discover -s tests -p test_mathematics.py
run "typed mathematical domain contracts"        python3 -m unittest discover -s tests -p test_math_domains.py
run "native mathematical module dispatch"        python3 -m unittest discover -s tests -p test_math_modules.py
run "native compatible moment-state reuse"       python3 -m unittest discover -s tests -p test_math_state.py
for f in tests/test_math_*.py; do   # every further mathematical capability (CUDA paths: the device after make gpu)
  case "${f##*/}" in test_math_domains.py|test_math_modules.py|test_math_state.py) continue;; esac
  m=${f##*/}; run "mathematics: ${m%.py}" python3 -m unittest discover -s tests -p "$m"
done
run "previous GROUP BY path (mask split / rows)"  env TORBLES_GROUPBY=split python3 tests/check_small.py
run "planner without native planning or cache"    env TORBLES_NATIVE_PLAN=0 TORBLES_PLAN_CACHE=0 python3 tests/check_small.py
run "no row kernel for many groups"               env TORBLES_ROWPATH_GROUPS=0 python3 tests/check_small.py
run "row kernel for every GROUP BY of 2+ groups"  env TORBLES_ROWPATH_GROUPS=2 python3 tests/check_small.py
echo "== GPU executor vs CPU engine ($( [ -n "${TORBLES_GPU_LIB:-}" ] && echo CUDA || echo emulated ))"
if python3 -c 'import algq; raise SystemExit(not hasattr(algq.gpu_lib(), "gpu_scalar_plane_moments"))'; then
  run "CUDA packed SQL stages"                     python3 -m unittest check_cuda_plane_stages
fi
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
