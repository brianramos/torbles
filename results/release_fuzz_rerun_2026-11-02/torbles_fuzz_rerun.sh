#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# The torbles 0.3.0 release's fuzzing rerun, after the fixes for the release run's fuzzing reports. Run it in WSL:
#
#   bash /mnt/c/Users/brian/Documents/torbles_fuzz_rerun.sh
#
#   1. brings ~/torbles_math to alpha at 605bb4f (as torbles_release_run.sh did) and builds the CPU and CUDA libraries;
#   2. where each engine call of the 13 SSB queries in SQL runs, at SF30 and SF50, on the CPU and the GPU (the release
#      run found Q3.1 through SQL on the GPU at 3.5 ms at SF30 and 43.8 ms at SF50): torbles_sql_devices.py, next to
#      this script, on the SSB databases in ~/torbles_scaling_data, which are only read (TORBLES_SKIP_DEVICES=1 skips
#      this step);
#   3. the full test suite with the GPU tests on the GPU (tests/run_all.sh), at the lowest priority;
#   4. the fuzzing campaign again, on new seeds, at the lowest priority, for TORBLES_FUZZ_MINUTES minutes (default 60):
#      the SQL fuzzer on the alg, GPU and plain engines, twice with the algebraic paths forced onto its small tables,
#      and the engine fuzzer, every answer compared with DuckDB's. Every worker's DuckDB is capped at 4 GB, and a
#      watchdog stops the worker using the most memory if the free memory falls below 6 GB.
# Step 2 wants a quiet machine; 3 and 4 do not. Everything goes to torbles-fuzz-rerun-605bb4f next to this script.
# Other processes, their priorities and their affinity are left as they are.
set -uo pipefail
DOCS=/mnt/c/Users/brian/Documents
BUNDLE=$DOCS/torbles-0.3.0-605bb4f-full.git-bundle
BASE=605bb4f735da44aadcf008227ecc79cb872f53a6
REPO=${TORBLES_REPO:-$HOME/torbles_math}
VENV=${TORBLES_VENV:-$HOME/torble_gpu_run/venv}
MINUTES=${TORBLES_FUZZ_MINUTES:-60}
SCALE_DATA=${TORBLES_SCALE_DATA:-$HOME/torbles_scaling_data}
OUT=$DOCS/torbles-fuzz-rerun-605bb4f
mkdir -p "$OUT" || exit 1
exec > >(tee -a "$OUT/run.log") 2>&1
echo "== $(date -Is) torbles 0.3.0 fuzzing rerun on $(hostname)"
fail() { echo "STOPPED: $*"; exit 1; }
STEPS=()
mark() { STEPS+=("$1: $2"); echo "-- $1: $2 ($(date +%T))"; }

[ -f "$BUNDLE" ] || fail "missing $BUNDLE"
cd "$REPO" || fail "no $REPO"
# shellcheck disable=SC1091
source "$VENV/bin/activate" || fail "no Python environment at $VENV (set TORBLES_VENV)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "$REPO has uncommitted changes; commit or stash them first"
if ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
  git fetch -q "$BUNDLE" alpha:refs/remotes/bundle/alpha || fail "could not fetch the bundle"
  # a commit made here and not in the bundle (a receipt, say) stays on a branch of its own
  git merge-base --is-ancestor HEAD refs/remotes/bundle/alpha || git branch -f before-605bb4f HEAD
  git checkout -q -B alpha refs/remotes/bundle/alpha || fail "could not check out alpha"
fi
echo "HEAD $(git log --oneline -1)"
python3 -c "import sys, numpy, pandas, duckdb; print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| pandas', pandas.__version__, '| DuckDB', duckdb.__version__)" \
  || fail "the venv needs numpy, pandas and duckdb"
command -v nvidia-smi > /dev/null || fail "no nvidia-smi: the GPU steps need the GPU"
echo "WSL: $(nproc) cores, $(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo) GiB memory; load average: $(cut -d' ' -f1-3 /proc/loadavg)"

echo "== $(date +%T) 1. building (the CUDA libraries take several minutes)"
make > "$OUT/build_cpu.log" 2>&1 || fail "make (see build_cpu.log)"
make gpu > "$OUT/build_gpu.log" 2>&1 || fail "make gpu (see build_gpu.log)"
python3 -c "import torbles; print('torbles', torbles.__version__, '| GPU:', torbles.gpu_name())"
python3 -c "import torbles, sys; sys.exit('emulat' in torbles.gpu_name().lower())" || fail "the GPU library is the emulation"
mark build ok

if [ "${TORBLES_SKIP_DEVICES:-0}" != 1 ]; then
  echo "== $(date +%T) 2. where the SQL queries' engine calls run, SF30 and SF50, $(nproc) threads"
  ps -eo pid,ni,pcpu,etime,comm --sort=-pcpu | head -4
  SSB_DATA="$SCALE_DATA" python3 "$DOCS/torbles_sql_devices.py" "$REPO" "$OUT/sql_devices.json" "30 50" > "$OUT/sql_devices.log" 2>&1; drc=$?
  grep -E "loaded in|Q3\.1|fallback" "$OUT/sql_devices.log" | head -20
  mark "SQL engine calls by device" "rc=$drc ($(grep -c 'fallback' "$OUT/sql_devices.log") lines with a GPU fallback)"
fi

echo "== $(date +%T) 3. the full test suite, the GPU tests on the GPU"
export PYTHONDONTWRITEBYTECODE=1
nice -n 19 bash tests/run_all.sh > "$OUT/full_suite.log" 2>&1; suite_rc=$?
grep -E " FAIL " "$OUT/full_suite.log" | head -10
mark "full test suite" "$(grep -cE '^  .* ok +\(' "$OUT/full_suite.log") ok, $(grep -cE '^  .* FAIL \(' "$OUT/full_suite.log") failed (rc=$suite_rc)"

echo "== $(date +%T) 4. fuzzing for $MINUTES minutes, on new seeds"
export PYTHONPATH=$REPO:$REPO/torbles:$REPO/tests TORBLES_FUZZ_DUCKDB_MEMORY=4GB
FORCED="TORBLES_ALG_MIN_ROWS=0 TORBLES_SERIES_ROWS_PER_GROUP=1 TORBLES_SERIES_STRICT=1"
WORKERS=(
  "sql_alg_601|TORBLES_SQL_ENGINE=alg|-m torbles.sqlfuzz --seed 601"
  "sql_alg_602|TORBLES_SQL_ENGINE=alg|-m torbles.sqlfuzz --seed 602"
  "sql_gpu_603|TORBLES_SQL_ENGINE=gpu|-m torbles.sqlfuzz --seed 603"
  "sql_gpu_604|TORBLES_SQL_ENGINE=gpu|-m torbles.sqlfuzz --seed 604"
  "sql_plain_605|TORBLES_SQL_ENGINE=plain|-m torbles.sqlfuzz --seed 605"
  "sql_forced_alg_606|TORBLES_SQL_ENGINE=alg $FORCED|-m torbles.sqlfuzz --seed 606"
  "sql_forced_gpu_607|TORBLES_SQL_ENGINE=gpu $FORCED|-m torbles.sqlfuzz --seed 607"
  "engine_41||-m torbles.fuzz --seed 41"
  "engine_42||-m torbles.fuzz --seed 42"
  "engine_43||-m torbles.fuzz --seed 43"
)
mkdir -p "$OUT/fuzz"
echo "starting ${#WORKERS[@]} workers (results at about $(date -d "+$((MINUTES + 3)) min" +%H:%M))"
PIDS=(); NAMES=()
for w in "${WORKERS[@]}"; do
  IFS='|' read -r name envs cmd <<< "$w"
  # shellcheck disable=SC2086
  ( exec env $envs nice -n 19 python3 $cmd --minutes "$MINUTES" --keep-going > "$OUT/fuzz/$name.log" 2>&1 ) &
  PIDS+=($!); NAMES+=("$name")
  echo "  $name: $envs python3 $cmd"
done
# the watchdog: if the free memory falls below 6 GB, it stops the worker using the most memory (only these workers)
while :; do
  alive=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2> /dev/null && alive+=("$p"); done
  [ ${#alive[@]} -gt 0 ] || break
  sleep 10
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "${avail:-0}" -lt 6000000 ]; then
    big=$(ps -o pid=,rss= -p "$(IFS=,; echo "${alive[*]}")" 2> /dev/null | sort -k2 -n | tail -1 | awk '{print $1}')
    if [ -n "$big" ]; then
      echo "$(date +%T) free memory $((avail / 1024)) MB: stopping worker process $big"
      kill "$big"; sleep 5
    fi
  fi
done
wait "${PIDS[@]}" 2> /dev/null      # these workers only (a bare wait would also wait for the log's tee)
nfail=$(grep -hE "^=== |: ERROR building" "$OUT"/fuzz/*.log | grep -c . || true)
mark fuzzing "$nfail failure(s) reported"

echo "== $(date +%T) done; summary"
{
  echo "HEAD $(git log --oneline -1) | GPU: $(python3 -c 'import torbles; print(torbles.gpu_name())' 2>&1)"
  for s in "${STEPS[@]}"; do echo "  $s"; done
  echo
  [ -f "$OUT/sql_devices.log" ] && grep -E "loaded in|Q3\.1|fallback" "$OUT/sql_devices.log" && echo
  for name in "${NAMES[@]}"; do
    f=$OUT/fuzz/$name.log
    tot=$(grep -E "statements in|queries in" "$f" | tail -1)
    [ -n "$tot" ] || tot="stopped early; last progress: $(grep -E '^case [0-9]+: totals' "$f" | tail -1)"
    verdict=$(grep -E "NO MISMATCHES|FAILURE" "$f" | tail -1)
    echo "$name: $tot | ${verdict:-no verdict}"
  done
  echo
  echo "what the fuzzers counted as refusals and as DuckDB's (the most frequent, by worker):"
  for name in "${NAMES[@]}"; do
    echo "  $name:"; sed -n '/^refusals/,/FAILURE\|NO MISMATCHES/p' "$OUT/fuzz/$name.log" | grep -E "^ +[0-9]+ " | head -8
  done
  echo
  echo "fuzzing failures (each one prints the command that replays it):"
  grep -h -A 6 -E "^=== |: ERROR building" "$OUT"/fuzz/*.log | head -300
} | tee "$OUT/summary.txt"
echo "== done: $OUT"
