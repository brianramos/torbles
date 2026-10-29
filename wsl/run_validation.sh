#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# torbles validation on WSL2 (Ubuntu) with an NVIDIA GPU:
#   bash wsl/run_validation.sh [SF]        SF = SSB scale factor, default 10
# Builds the engine and the CUDA executor, runs every correctness test (GPU tests on the GPU), generates the official
# SSB data if not present (reuses ~/torble_gpu_run/ssbdata from earlier runs), and benchmarks DuckDB and torbles
# (CPU and GPU) on it, every result checked against DuckDB. Work happens on WSL's own disk (~/torbles_run); the log
# and report land in results/validation next to this repository.
set -uo pipefail
SF=${1:-10}
SRC="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$HOME/torbles_run"; OUT="$SRC/results/validation"; mkdir -p "$OUT"
LOG="$OUT/validation_sf${SF}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
die() { echo; echo "STOPPED: $*"; echo "(log: $LOG)"; exit 1; }
echo "== torbles $(cat "$SRC/VERSION")"
grep -m1 "model name" /proc/cpuinfo; CORES=$(nproc); echo "WSL: $CORES cores"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed (GPU not visible in WSL)"
NVCC="$(command -v nvcc || true)"; [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && die "nvcc not found (install the CUDA toolkit for WSL-Ubuntu)"; "$NVCC" --version | tail -1
for t in gcc g++ make git python3; do command -v $t >/dev/null || die "missing $t: sudo apt install build-essential git python3 python3-venv"; done

echo "== python environment"
VENV="$HOME/torble_gpu_run/venv"; [ -x "$VENV/bin/python" ] || VENV="$WORK/venv"
if [ ! -x "$VENV/bin/python" ]; then mkdir -p "$WORK"; python3 -m venv "$VENV" || die "python3 -m venv failed"; fi
. "$VENV/bin/activate"; pip install -q numpy pandas duckdb || die "pip install failed"

echo "== build"
rm -rf "$WORK/src"; mkdir -p "$WORK"; cp -r "$SRC" "$WORK/src"; cd "$WORK/src" || die "copy failed"
make clean >/dev/null; make || die "build failed"
make gpu NVCC="$NVCC" || die "CUDA build failed"

echo "== every correctness test (GPU tests on the GPU)"
bash tests/run_all.sh || die "tests failed"

echo "== official SSB data (SF$SF)"
export SSB_DATA="${SSB_DATA:-$HOME/torble_gpu_run/ssbdata}" SSB_DBGEN="${SSB_DBGEN:-$HOME/torble_gpu_run/ssb-dbgen}" SSB_THREADS="1,$CORES"
if [ ! -f "$SSB_DATA/sf$SF/.complete" ]; then
  if [ ! -x "$SSB_DBGEN/dbgen" ]; then
    rm -rf "$SSB_DBGEN"; git clone -q --depth 1 https://github.com/electrum/ssb-dbgen "$SSB_DBGEN" || die "git clone failed"
    PERMISSIVE='-std=gnu89 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=return-mismatch -Wno-error=return-type'
    gcc -Werror=implicit-function-declaration -Wno-error=return-mismatch -x c -c /dev/null -o /dev/null 2>/dev/null || PERMISSIVE='-std=gnu89 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=incompatible-pointer-types'
    ( cd "$SSB_DBGEN" && make CFLAGS="-O2 -fcommon -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DDBNAME=\\\"dss\\\" -DLINUX -DDB2 -DSSBM $PERMISSIVE" > "$WORK/dbgen_build.log" 2>&1 ) || die "building ssb-dbgen failed ($WORK/dbgen_build.log)"
  fi
  rm -rf "$SSB_DATA/sf$SF"; ( cd bench/ssb && python ssb_common.py "$SF" ) || die "data generation failed"
  touch "$SSB_DATA/sf$SF/.complete"; rm -f "$SSB_DATA/sf$SF/"*.tbl
fi
cd bench/ssb
echo "== DuckDB (1 and $CORES threads, in-memory copy of the database)"
SSB_DUCK_INMEM=1 python ssb_duck.py "$SF" || die "DuckDB run failed"
echo "== torbles (derived-surrogate keys): CPU 1 and $CORES threads, GPU (per-word kernel = default, tile kernel, dense groups)"
TORBLES_OUT_TAG=_validation TORBLES_ENGINES="cpu-mm@1,cpu-mm@$CORES,gpu-word,gpu-popc,gpu-split" python ssb_torbles.py "$SF" derived || die "torbles run failed"
cp "$SSB_DATA/sf$SF/duckdb_results.json" "$SSB_DATA/sf$SF/torbles_derived_validation.json" "$OUT/"
python ssb_mm_report.py "$OUT/torbles_derived_validation.json" "$OUT/duckdb_results.json" | tee "$OUT/validation_report_sf$SF.txt"
echo "== torbles with automatic keys: the database loaded with torbles.from_duckdb, keys from schema.sql, CPU and GPU"
python ssb_auto.py "$SF" cpu,gpu | tee "$OUT/auto_keys_sf$SF.txt" || die "automatic-key run failed"
cp "$SSB_DATA/sf$SF/torbles_auto.json" "$OUT/torbles_auto_sf$SF.json"
echo; echo "== done. Report: $OUT/validation_report_sf$SF.txt"
