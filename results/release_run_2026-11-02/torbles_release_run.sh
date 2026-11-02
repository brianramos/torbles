#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# The torbles 0.3.0 release's last laptop run. Run it in WSL:
#
#   bash /mnt/c/Users/brian/Documents/torbles_release_run.sh
#
#   1. brings ~/torbles_math to alpha at 99cf300 (as torbles_device_run.sh does), then waits while the cross-benchmark
#      (anything running from torbles_crossbench) is timing, so that neither run disturbs the other's numbers
#      (TORBLES_NO_WAIT=1 skips the wait);
#   2. builds the CPU and CUDA libraries;
#   3. the scaling sweep for the README's poster, on all cores (wsl/run_scaling.sh): the 13 official SSB queries at
#      6M to 300M fact rows (SF1, 3, 10, 30, 50), DuckDB in memory on every thread, torbles through the Python API and
#      through SQL, on the CPU and the GPU, every answer checked against DuckDB's. The SSB databases in
#      ~/torble_gpu_run/ssbdata are only read (linked into ~/torbles_scaling_data); a scale missing there is generated
#      in ~/torbles_scaling_data (SF50 needs about 43 GB of disk);
#      then the cross-benchmark's two window queries that output all of 1M rows (bench/window_output.py);
#   4. the full test suite with the GPU tests on the GPU (tests/run_all.sh), at the lowest priority;
#   5. the fuzzing campaign, at the lowest priority, for TORBLES_FUZZ_MINUTES minutes (default 60): the SQL fuzzer on
#      the alg, GPU and plain engines, twice with the algebraic paths forced onto its small tables, and the engine
#      fuzzer on three seeds, every answer compared with DuckDB's. Every worker's DuckDB is capped at 4 GB, and a
#      watchdog stops the worker using the most memory if the free memory falls below 6 GB.
# Steps 1-3 want a quiet machine; 4 and 5 do not. Everything goes to torbles-release-run-99cf300 next to this script.
# Other processes, their priorities and their affinity are left as they are.
set -uo pipefail
DOCS=/mnt/c/Users/brian/Documents
BUNDLE=$DOCS/torbles-0.3.0-99cf300-full.git-bundle
BASE=99cf300c68e8464021d291f35055f61441bb8d18
REPO=${TORBLES_REPO:-$HOME/torbles_math}
VENV=${TORBLES_VENV:-$HOME/torble_gpu_run/venv}
MINUTES=${TORBLES_FUZZ_MINUTES:-60}
PRESERVED=$HOME/torble_gpu_run/ssbdata
SCALE_DATA=${TORBLES_SCALE_DATA:-$HOME/torbles_scaling_data}
SFS=${TORBLES_SCALES:-"1 3 10 30 50"}
OUT=$DOCS/torbles-release-run-99cf300
mkdir -p "$OUT" || exit 1
exec > >(tee -a "$OUT/run.log") 2>&1
echo "== $(date -Is) torbles 0.3.0 release run on $(hostname)"
fail() { echo "STOPPED: $*"; exit 1; }
STEPS=()
mark() { STEPS+=("$1: $2"); echo "-- $1: $2 ($(date +%T))"; }

[ -f "$BUNDLE" ] || fail "missing $BUNDLE"
if [ ! -d "$REPO/.git" ]; then
  git clone -q -b alpha "$BUNDLE" "$REPO" || fail "could not clone the bundle into $REPO"
fi
cd "$REPO" || fail "no $REPO"
# shellcheck disable=SC1091
source "$VENV/bin/activate" || fail "no Python environment at $VENV (set TORBLES_VENV)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "$REPO has uncommitted changes; commit or stash them first"
git config user.email > /dev/null || git config user.email "brianramos@gmail.com"
git config user.name > /dev/null || git config user.name "Brian Richard RAMOS"
if ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
  git rev-parse -q --verify before-final > /dev/null || git branch before-final HEAD   # the state before the first run
  git fetch -q "$BUNDLE" alpha:refs/remotes/bundle/alpha || fail "could not fetch the bundle"
  # a commit made here and not in the bundle (a receipt, say) stays on a branch of its own
  git merge-base --is-ancestor HEAD refs/remotes/bundle/alpha || git branch -f before-99cf300 HEAD
  git checkout -q -B alpha refs/remotes/bundle/alpha || fail "could not check out alpha"
fi
echo "HEAD $(git log --oneline -1)"
python3 -c "import sys, numpy, pandas, duckdb; print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| pandas', pandas.__version__, '| DuckDB', duckdb.__version__)" \
  || fail "the venv needs numpy, pandas and duckdb"
command -v nvidia-smi > /dev/null || fail "no nvidia-smi: the GPU steps need the GPU"
echo "WSL: $(nproc) cores, $(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo) GiB memory, $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ') free on disk"

echo "== $(date +%T) 1. a quiet machine (the build and the benchmark)"
if [ "${TORBLES_NO_WAIT:-0}" != 1 ]; then
  while pgrep -f torbles_crossbench > /dev/null; do
    echo "$(date +%T) the cross-benchmark is running; waiting (checking every 5 minutes; TORBLES_NO_WAIT=1 skips this):"
    pgrep -af torbles_crossbench | cut -c1-150 | head -4
    sleep 300
  done
fi

echo "load average: $(cut -d' ' -f1-3 /proc/loadavg); busiest processes now (left as they are):"
ps -eo pid,ni,pcpu,etime,comm --sort=-pcpu | head -6
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
mark "quiet machine" "load $(cut -d' ' -f1 /proc/loadavg)"

echo "== $(date +%T) 2. building (the CUDA libraries take several minutes)"
make > "$OUT/build_cpu.log" 2>&1 || fail "make (see build_cpu.log)"
make gpu > "$OUT/build_gpu.log" 2>&1 || fail "make gpu (see build_gpu.log)"
python3 -c "import torbles; print('torbles', torbles.__version__, '| GPU:', torbles.gpu_name())"
python3 -c "import torbles, sys; sys.exit('emulat' in torbles.gpu_name().lower())" || fail "the GPU library is the emulation"
mark build ok

echo "== $(date +%T) 3. the scaling sweep, SF $SFS, $(nproc) threads (results in scaling/)"
for SF in $SFS; do
  mkdir -p "$SCALE_DATA/sf$SF"
  if [ -f "$PRESERVED/sf$SF/ssb.duckdb" ] && [ ! -e "$SCALE_DATA/sf$SF/ssb.duckdb" ]; then
    ln -s "$PRESERVED/sf$SF/ssb.duckdb" "$SCALE_DATA/sf$SF/ssb.duckdb"         # read only, never written
  fi
done
t0=$(date +%s)
SSB_DATA="$SCALE_DATA" SCALING_OUT="$OUT/scaling" SCALING_THREADS=0 bash wsl/run_scaling.sh "$SFS" > "$OUT/scaling.log" 2>&1
src=$?
grep -E "^== SF|loaded in|13 SSB queries|all results|MISMATCH|failed|skipped|STOPPED" "$OUT/scaling.log" | head -60
if [ -f "$OUT/scaling/scaling.json" ]; then
  cat "$OUT/scaling/scaling_summary.txt"
  mark "scaling sweep" "rc=$src, $(( ($(date +%s) - t0) / 60 )) min, $(python3 -c "import json; print(len(json.load(open('$OUT/scaling/scaling.json'))['scale']))") scales"
else
  mark "scaling sweep" "rc=$src, no scaling.json (see scaling.log)"
fi

echo "== $(date +%T) 3b. the cross-benchmark's two full-output window queries, 1M rows, $(nproc) threads"
python3 bench/window_output.py 1000000 "" "$OUT/window_output.json" > "$OUT/window_output.log" 2>&1; wrc=$?
cat "$OUT/window_output.log" | tail -4
mark "full-output windows" "rc=$wrc, $(tail -2 "$OUT/window_output.log" | head -1)"

echo "== $(date +%T) 4. the full test suite, the GPU tests on the GPU"
export PYTHONDONTWRITEBYTECODE=1
nice -n 19 bash tests/run_all.sh > "$OUT/full_suite.log" 2>&1; suite_rc=$?
mark "full test suite" "$(grep -cE '^  .* ok +\(' "$OUT/full_suite.log") ok, $(grep -cE '^  .* FAIL \(' "$OUT/full_suite.log") failed (rc=$suite_rc)"

echo "== $(date +%T) 5. fuzzing for $MINUTES minutes"
export PYTHONPATH=$REPO:$REPO/torbles:$REPO/tests TORBLES_FUZZ_DUCKDB_MEMORY=4GB
FORCED="TORBLES_ALG_MIN_ROWS=0 TORBLES_SERIES_ROWS_PER_GROUP=1 TORBLES_SERIES_STRICT=1"
WORKERS=(
  "sql_alg_501|TORBLES_SQL_ENGINE=alg|-m torbles.sqlfuzz --seed 501"
  "sql_alg_502|TORBLES_SQL_ENGINE=alg|-m torbles.sqlfuzz --seed 502"
  "sql_gpu_503|TORBLES_SQL_ENGINE=gpu|-m torbles.sqlfuzz --seed 503"
  "sql_gpu_504|TORBLES_SQL_ENGINE=gpu|-m torbles.sqlfuzz --seed 504"
  "sql_plain_505|TORBLES_SQL_ENGINE=plain|-m torbles.sqlfuzz --seed 505"
  "sql_forced_alg_506|TORBLES_SQL_ENGINE=alg $FORCED|-m torbles.sqlfuzz --seed 506"
  "sql_forced_gpu_507|TORBLES_SQL_ENGINE=gpu $FORCED|-m torbles.sqlfuzz --seed 507"
  "engine_31||-m torbles.fuzz --seed 31"
  "engine_32||-m torbles.fuzz --seed 32"
  "engine_33||-m torbles.fuzz --seed 33"
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
nfail=$(grep -h "^=== " "$OUT"/fuzz/*.log | grep -c . || true)
mark fuzzing "$nfail failure(s) reported"

echo "== $(date +%T) done; summary"
{
  echo "HEAD $(git log --oneline -1) | GPU: $(python3 -c 'import torbles; print(torbles.gpu_name())' 2>&1)"
  for s in "${STEPS[@]}"; do echo "  $s"; done
  echo
  [ -f "$OUT/scaling/scaling_summary.txt" ] && cat "$OUT/scaling/scaling_summary.txt" && echo
  for name in "${NAMES[@]}"; do
    f=$OUT/fuzz/$name.log
    tot=$(grep -E "statements in|queries in" "$f" | tail -1)
    [ -n "$tot" ] || tot="stopped early; last progress: $(grep -E '^case [0-9]+: totals' "$f" | tail -1)"
    verdict=$(grep -E "NO MISMATCHES|FAILURE" "$f" | tail -1)
    echo "$name: $tot | ${verdict:-no verdict}"
  done
  echo
  echo "fuzzing failures (each one prints the command that replays it):"
  grep -h -A 6 "^=== " "$OUT"/fuzz/*.log | head -300
} | tee "$OUT/summary.txt"
echo "== done: $OUT"
