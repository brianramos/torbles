#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# Does the enhancements branch cost anything when no relationship catalog is attached? A matched A/B on WSL2 with an
# NVIDIA GPU:
#   bash wsl/ab_enhancements.sh [rounds, default 3] [SF, default 10]
# A = alpha, B = enhancements (A_REF / B_REF override; any commit-ish of this repository). Both are exported from git
# into separate trees on WSL's disk and built the same way (CPU engine, CPU emulation, CUDA executor). B then runs every
# correctness test (GPU tests on the GPU) and the 2,000-program DAG property check. Then the official SSB queries
# (derived-surrogate keys, every answer checked against DuckDB) run once per side per round, in fresh processes, on the
# CPU at 1 thread and at all cores and on the GPU (per-word kernel); rounds alternate the order (A B, B A, ...) so that
# drift of the machine (clocks, temperature, page cache) falls on both sides. No catalog is attached on either side.
# The table at the end (wsl/ab_table.py) gives per engine and query the best time of each side, the change, and each
# side's own round-to-round spread, which is the noise the change has to be read against.
# Needs the SSB data and DuckDB's reference results for SF from wsl/run_validation.sh (SSB_DATA, default as there).
# Without a GPU: AB_NO_GPU=1 (CPU engines only). PY=<python> uses that interpreter instead of the validation venv.
set -uo pipefail
ROUNDS=${1:-3}; SF=${2:-10}
SRC="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$HOME/torbles_run/ab"; STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$SRC/results/ab_enhancements/sf${SF}_$STAMP"; mkdir -p "$OUT"
LOG="$OUT/ab.log"
exec > >(tee "$LOG") 2>&1
die() { echo; echo "STOPPED: $*"; echo "(log: $LOG)"; exit 1; }
A_REF=${A_REF:-alpha}; B_REF=${B_REF:-enhancements}
A_SHA=$(git -C "$SRC" rev-parse --short "$A_REF") || die "no commit $A_REF in $SRC"
B_SHA=$(git -C "$SRC" rev-parse --short "$B_REF") || die "no commit $B_REF in $SRC"
echo "== A = $A_REF ($A_SHA), B = $B_REF ($B_SHA), $ROUNDS rounds, SSB SF$SF"
grep -m1 "model name" /proc/cpuinfo; CORES=$(nproc); echo "cores: $CORES"
GPU=1; [ -n "${AB_NO_GPU:-}" ] && GPU=0
if [ $GPU = 1 ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed (no GPU visible; AB_NO_GPU=1 runs the CPU part only)"
  NVCC="$(command -v nvcc || true)"; [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
  [ -z "$NVCC" ] && die "nvcc not found (install the CUDA toolkit for WSL-Ubuntu)"
fi

echo "== python"
if [ -n "${PY:-}" ]; then echo "using $PY"
else
  VENV="$HOME/torble_gpu_run/venv"; [ -x "$VENV/bin/python" ] || VENV="$HOME/torbles_run/venv"
  [ -x "$VENV/bin/python" ] || die "no venv: run wsl/run_validation.sh once first (or set PY=python3)"
  PY="$VENV/bin/python"
fi
"$PY" -c "import numpy, pandas, duckdb" || die "numpy / pandas / duckdb missing for $PY"

export SSB_DATA="${SSB_DATA:-$HOME/torble_gpu_run/ssbdata}"
D="$SSB_DATA/sf$SF"
[ -f "$D/ssb.duckdb" ] && [ -f "$D/duckdb_results.json" ] || die "no SSB SF$SF data with DuckDB results in $D: run bash wsl/run_validation.sh $SF first"

echo "== export and build both sides"
rm -rf "$WORK"; mkdir -p "$WORK"
for side in A B; do
  ref=$([ $side = A ] && echo "$A_REF" || echo "$B_REF")
  mkdir -p "$WORK/$side"; git -C "$SRC" archive "$ref" | tar -x -C "$WORK/$side" || die "git archive $ref failed"
  ( cd "$WORK/$side" && make clean >/dev/null && make >/dev/null ) || die "build of $side ($ref) failed"
  if [ $GPU = 1 ]; then ( cd "$WORK/$side" && make gpu NVCC="$NVCC" ) || die "CUDA build of $side ($ref) failed"; fi
  echo "$side: built $ref"
done

echo "== B: every correctness test (GPU tests on the GPU when built) and the DAG property check"
( cd "$WORK/B" && PATH="$(dirname "$PY"):$PATH" bash tests/run_all.sh ) || die "tests failed on B"
( cd "$WORK/B" && PYTHONPATH="$PWD:$PWD/torbles:$PWD/tests" "$PY" tests/validate_relationship_dag_random.py --attempts 2000 --seed 80492 ) \
  | tee "$OUT/dag_validation.json" | grep -q '"correct": true' || die "DAG property check failed on B"

ENGINES="cpu-mm@1,cpu-mm@$CORES"; [ $GPU = 1 ] && ENGINES="$ENGINES,gpu-word"
cat "$D/ssb.duckdb" > /dev/null                   # the database in the page cache before round 1, as for every later run
echo "== SSB, interleaved rounds: engines $ENGINES"
for r in $(seq 1 "$ROUNDS"); do
  order="A B"; [ $((r % 2)) = 0 ] && order="B A"
  for side in $order; do
    tag="_ab_${side}_r${r}"
    echo "-- round $r, $side"
    ( cd "$WORK/$side/bench/ssb" && SSB_THREADS="1,$CORES" TORBLES_ENGINES="$ENGINES" TORBLES_OUT_TAG="$tag" \
        "$PY" ssb_torbles.py "$SF" derived > "$OUT/run${tag}.log" 2>&1 ) || die "SSB run $side round $r failed (see $OUT/run${tag}.log)"
    tail -1 "$OUT/run${tag}.log" | grep -q "all results match DuckDB" || die "SSB answers differ from DuckDB on $side round $r"
    mv "$D/torbles_derived${tag}.json" "$OUT/${side}_r${r}.json"
    grep -E "ingestion|peak memory" "$OUT/run${tag}.log" | sed 's/^/   /'
  done
done

echo "== table"
"$PY" "$SRC/wsl/ab_table.py" "$OUT" "$A_REF ($A_SHA)" "$B_REF ($B_SHA)" | tee "$OUT/table.txt"
echo; echo "== done. Table: $OUT/table.txt"
