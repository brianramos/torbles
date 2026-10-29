#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# Scaling benchmark on WSL2 with an NVIDIA GPU: the 13 official SSB queries over growing data (scale factors) and,
# at SF10, over growing thread counts; DuckDB and torbles (CPU, GPU), every result checked against DuckDB. Both answer
# from memory: DuckDB runs on an in-memory copy of its database (no buffer-pool loads), with its default memory limit.
#   bash wsl/run_scaling.sh ["1 3 10 30 50"]        scale factors (default 1 3 10 30 50)
# Data are generated with the official generator (serial dbgen, as always) where missing; the .tbl files are deleted
# once DuckDB holds them. A scale factor that does not fit in memory or on disk is skipped with a note.
# Results: results/scaling/ (one directory per run; summary scaling.json).
set -uo pipefail
SFS=${1:-"1 3 10 30 50"}
SRC="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$HOME/torbles_run"
OUT="$SRC/results/scaling"; mkdir -p "$OUT"
LOG="$OUT/scaling_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
die() { echo; echo "STOPPED: $*"; echo "(log: $LOG)"; exit 1; }
echo "== torbles $(cat "$SRC/VERSION") scaling benchmark"
grep -m1 "model name" /proc/cpuinfo; CORES=$(nproc); MEM=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
echo "WSL: $CORES cores, $MEM GiB memory"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed"
NVCC="$(command -v nvcc || true)"; [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && die "nvcc not found"
VENV="$HOME/torble_gpu_run/venv"; [ -x "$VENV/bin/python" ] || VENV="$WORK/venv"
if [ ! -x "$VENV/bin/python" ]; then mkdir -p "$WORK"; python3 -m venv "$VENV" || die "python3 -m venv failed"; fi
. "$VENV/bin/activate"; pip install -q numpy pandas duckdb || die "pip install failed"

echo "== build"
rm -rf "$WORK/src"; mkdir -p "$WORK"; cp -r "$SRC" "$WORK/src"; cd "$WORK/src" || die "copy failed"
make clean >/dev/null; make >/dev/null || die "build failed"; make gpu NVCC="$NVCC" || die "CUDA build failed"
export SSB_DATA="${SSB_DATA:-$HOME/torble_gpu_run/ssbdata}" SSB_DBGEN="${SSB_DBGEN:-$HOME/torble_gpu_run/ssb-dbgen}"
if [ ! -x "$SSB_DBGEN/dbgen" ]; then
  rm -rf "$SSB_DBGEN"; git clone -q --depth 1 https://github.com/electrum/ssb-dbgen "$SSB_DBGEN" || die "git clone failed"
  PERMISSIVE='-std=gnu89 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=return-mismatch -Wno-error=return-type'
  gcc -Werror=implicit-function-declaration -Wno-error=return-mismatch -x c -c /dev/null -o /dev/null 2>/dev/null || PERMISSIVE='-std=gnu89 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types'
  ( cd "$SSB_DBGEN" && make CFLAGS="-O2 -fcommon -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DDBNAME=\\\"dss\\\" -DLINUX -DDB2 -DSSBM $PERMISSIVE" > "$WORK/dbgen_build.log" 2>&1 ) || die "building ssb-dbgen failed"
fi
SSB13="Q1.1,Q1.2,Q1.3,Q2.1,Q2.2,Q2.3,Q3.1,Q3.2,Q3.3,Q3.4,Q4.1,Q4.2,Q4.3"
cd bench/ssb
for SF in $SFS; do
  echo; echo "== SF$SF ($((SF * 6))M fact rows)"
  # memory: the torble build peaks near 0.6 GiB per SF; disk: ~0.6 GB of .tbl per SF while DuckDB loads them
  NEED_MEM=$(( SF * 6 / 10 + 4 )); FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  if [ "$NEED_MEM" -gt "$MEM" ]; then echo "skipped: needs about $NEED_MEM GiB of memory, WSL has $MEM"; continue; fi
  if [ ! -f "$SSB_DATA/sf$SF/.complete" ]; then
    NEED_DISK=$(( SF * 8 / 10 + 3 )); [ "$FREE" -lt "$NEED_DISK" ] && { echo "skipped: needs about $NEED_DISK GB of disk, $FREE GB free"; continue; }
    t0=$(date +%s); rm -rf "$SSB_DATA/sf$SF"
    python ssb_common.py "$SF" || { echo "data generation failed for SF$SF, skipped"; continue; }
    touch "$SSB_DATA/sf$SF/.complete"; rm -f "$SSB_DATA/sf$SF/"*.tbl; echo "generated and loaded in $(( $(date +%s) - t0 )) s"
  fi
  SSB_DUCK_INMEM=1 SSB_THREADS="$CORES" SSB_QUERIES="$SSB13" python ssb_duck.py "$SF" | grep "in-memory" || { echo "DuckDB failed for SF$SF"; continue; }
  TORBLES_QUERIES="$SSB13" SSB_THREADS="$CORES" TORBLES_OUT_TAG=_scaling TORBLES_ENGINES="cpu-mm@$CORES,gpu-word" \
    python ssb_torbles.py "$SF" derived | grep -E "ingestion|GPU:|all results|MISMATCH|peak memory" || { echo "torbles failed for SF$SF"; continue; }
  mkdir -p "$OUT/sf$SF"; cp "$SSB_DATA/sf$SF/duckdb_results.json" "$SSB_DATA/sf$SF/torbles_derived_scaling.json" "$OUT/sf$SF/"
done
echo; echo "== thread scaling at SF10"
if [ -f "$SSB_DATA/sf10/.complete" ]; then
  TH=""; for t in 1 2 4 8 12 16 $CORES; do [ "$t" -le "$CORES" ] && TH="$TH,$t"; done; TH=$(echo "$TH" | sed 's/^,//' | tr ',' '\n' | sort -nu | paste -sd,)
  SSB_DUCK_INMEM=1 SSB_THREADS="$TH" SSB_QUERIES="$SSB13" python ssb_duck.py 10 > /dev/null || die "DuckDB thread sweep failed"
  ENG=$(echo "$TH" | tr ',' '\n' | sed 's/^/cpu-mm@/' | paste -sd,)
  TORBLES_QUERIES="$SSB13" SSB_THREADS="$TH" TORBLES_OUT_TAG=_threads TORBLES_ENGINES="$ENG" python ssb_torbles.py 10 derived | grep -E "all results|MISMATCH"
  mkdir -p "$OUT/threads_sf10"; cp "$SSB_DATA/sf10/duckdb_results.json" "$SSB_DATA/sf10/torbles_derived_threads.json" "$OUT/threads_sf10/"
fi
python ssb_scaling.py "$OUT" | tee "$OUT/scaling_summary.txt"
echo; echo "== done. Results: $OUT"
