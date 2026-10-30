#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# SSB in SQL (db.sql) against DuckDB on all cores, CPU and GPU, plus the queries that leave the engine (a few minutes):
#   bash wsl/run_sql_bench.sh [SF]        SF default 10; reuses the data and the venv of run_validation.sh
set -uo pipefail
SF=${1:-10}
SRC="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$HOME/torbles_run"; OUT="$SRC/results/validation"; mkdir -p "$OUT"
LOG="$OUT/sql_bench_sf${SF}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
die() { echo; echo "STOPPED: $*"; echo "(log: $LOG)"; exit 1; }
echo "== torbles $(cat "$SRC/VERSION"), SQL benchmark"; grep -m1 "model name" /proc/cpuinfo; echo "WSL: $(nproc) cores"
VENV="$HOME/torble_gpu_run/venv"; [ -x "$VENV/bin/python" ] || VENV="$WORK/venv"
if [ ! -x "$VENV/bin/python" ]; then mkdir -p "$WORK"; python3 -m venv "$VENV" || die "python3 -m venv failed"; fi
. "$VENV/bin/activate"; pip install -q numpy pandas duckdb pytz pyuca || die "pip install failed"
NVCC="$(command -v nvcc || true)"; [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
rm -rf "$WORK/src_sql"; mkdir -p "$WORK"; cp -r "$SRC" "$WORK/src_sql"; cd "$WORK/src_sql" || die "copy failed"
make clean >/dev/null; make >/dev/null || die "build failed"
ENG=alg; if [ -n "$NVCC" ]; then make gpu NVCC="$NVCC" >/dev/null && ENG=alg,gpu; fi
export SSB_DATA="${SSB_DATA:-$HOME/torble_gpu_run/ssbdata}"
[ -f "$SSB_DATA/sf$SF/ssb.duckdb" ] || die "no SSB SF$SF data in $SSB_DATA (run wsl/run_validation.sh $SF once first)"
cd bench/ssb && python ssb_sql.py "$SF" "$(nproc)" "$ENG" 200000 || die "benchmark failed"
echo; echo "== done (log: $LOG)"
