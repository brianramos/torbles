#!/bin/bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Ablation of the GROUP BY bridges on SSB SF1: every configuration (environment switches of the engine) runs
# ssb_auto.py once per round at 1 and 2 threads, rounds interleaved so that machine noise spreads over all of them.
#   bash ablation.sh [rounds] [out dir]        BASELINE=<checkout of the committed 0.2.0> adds it as configuration a
# then: python3 ablation_table.py <out dir>
HERE=$(cd "$(dirname "$0")" && pwd); ROUNDS=${1:-3}; OUT=${2:-$HERE/ablation_out}; mkdir -p $OUT
D=$(cd $HERE && python3 -c "from ssb_common import datadir; print(datadir(1))")
CUR=$HERE
OFF="TORBLES_FACTOR=0 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
declare -A CFG
[ -n "$BASELINE" ] && CFG[a_committed]="OLD"
CFG[b_transpose]="$OFF"
CFG[c_factor]="TORBLES_FACTOR=1 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[d_tree]="TORBLES_FACTOR=0 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=0 TORBLES_SPLIT=tree TORBLES_DIRECT=1"
CFG[e_direct]="TORBLES_FACTOR=0 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=1"
CFG[f_rows]="TORBLES_FACTOR=0 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=-1 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[g_mobius]="TORBLES_FACTOR=0 TORBLES_MOBIUS=-1 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[h_factor_rows]="TORBLES_FACTOR=1 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=-1 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[i_factor_mobius]="TORBLES_FACTOR=1 TORBLES_MOBIUS=-1 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[j_factor_rows_mobius]="TORBLES_FACTOR=1 TORBLES_MOBIUS=-1 TORBLES_MM_ROWS=-1 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[k_rows_forced]="TORBLES_FACTOR=1 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=2 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[l_mobius_forced]="TORBLES_FACTOR=1 TORBLES_MOBIUS=1 TORBLES_MM_ROWS=0 TORBLES_SPLIT=scan TORBLES_DIRECT=0"
CFG[m_nomargin]="TORBLES_FACTOR=1 TORBLES_MOBIUS=0 TORBLES_MM_ROWS=-1 TORBLES_SPLIT=scan TORBLES_DIRECT=0 TORBLES_RK_MARGIN=1"
CFG[n_defaults]="TORBLES_NOTHING=1"
for r in $(seq 1 $ROUNDS); do
  for k in $(echo "${!CFG[@]}" | tr ' ' '\n' | sort); do
    if [ "${CFG[$k]}" = OLD ]; then (cp $HERE/ssb_auto.py $BASELINE/bench/ssb/ssb_auto_ablation.py && cd $BASELINE/bench/ssb && python3 ssb_auto_ablation.py 1 cpu@1,cpu@2 > $OUT/$k.r$r.log 2>&1)
    else (cd $CUR && env ${CFG[$k]} python3 ssb_auto.py 1 cpu@1,cpu@2 > $OUT/$k.r$r.log 2>&1); fi
    cp $D/torbles_auto.json $OUT/$k.r$r.json
    grep -q MISMATCH $OUT/$k.r$r.log && echo "MISMATCH in $k round $r"
  done
  echo "round $r done"
done
