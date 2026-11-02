#!/usr/bin/env bash
# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
# The release review's protocol (torbles-release-review-2026-11-01/RELEASE_REVIEW.md) run again on the fixed alpha, on
# this laptop's GPU. Run it in WSL, with no build, test or benchmark of your own running at the same time:
#
#   bash /mnt/c/Users/brian/Documents/torbles_review_rerun.sh
#
# It brings ~/torbles_math to alpha at 7486dba (as torbles_device_run.sh does; HEAD before the first run stays on the
# branch before-final), builds the CPU and CUDA libraries, and then runs, one step after another:
#   1. bench/validate_math.py --device --all-ready, if a mathematics capability still waits for the GPU;
#   2. the full test suite, the GPU tests on the GPU (tests/run_all.sh);
#   3. the review's own reproducers (review_sql, review_native) against this tree, and a probe of GPU memory over many
#      tables uploaded, queried and released;
#   4. the wheel built from this tree, installed in a new venv outside it, with its SQL tests on the three engines;
#   5. the review's benchmarks, unchanged: benchmark_release.py (SSB SF1, SF3 and SF10 through db.sql, five rounds; the
#      mixed SQL workload at 10K, 100K and 1M rows, three rounds) and benchmark_native_ssb.py (SF1, SF3 and SF10
#      through the Python API), each worker a fresh process, one at a time.
# Logs, reports and a comparison with the review's numbers go to torbles-release-rerun-7486dba next to this script. The
# review's folder is only read. Other processes, their priorities and their affinity are left as they are.
set -uo pipefail
DOCS=/mnt/c/Users/brian/Documents
REVIEW=$DOCS/torbles-release-review-2026-11-01
BUNDLE=$DOCS/torbles-0.2.0-alpha-7486dba-full.git-bundle
BASE=7486dba58c4d3c10db73621a157db475adb0a074
REPO=${TORBLES_REPO:-$HOME/torbles_math}
VENV=${TORBLES_VENV:-$HOME/torble_gpu_run/venv}
SSBDATA=/home/brian/torble_gpu_run/ssbdata          # where the review's harness reads SSB (fixed in benchmark_release.py)
OUT=$DOCS/torbles-release-rerun-7486dba
RR=$HOME/torbles_release_rerun_7486dba             # the review's layout: copies of its scripts, candidate -> $REPO
mkdir -p "$OUT/results" || exit 1
exec > >(tee -a "$OUT/rerun.log") 2>&1
echo "== $(date -Is) release-review rerun on $(hostname)"
fail() { echo "STOPPED: $*"; exit 1; }
STEPS=()
mark() { STEPS+=("$1: $2"); echo "-- $1: $2 ($(date +%T))"; }
export PYTHONDONTWRITEBYTECODE=1

[ -f "$BUNDLE" ] || fail "missing $BUNDLE"
[ -f "$REVIEW/benchmark_release.py" ] || fail "missing the review's folder $REVIEW"
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
  git checkout -q -B alpha refs/remotes/bundle/alpha || fail "could not check out alpha"
fi
echo "HEAD $(git log --oneline -1)"
python3 -c "import sys, numpy, pandas, duckdb; print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| pandas', pandas.__version__, '| DuckDB', duckdb.__version__)" \
  || fail "the venv needs numpy, pandas and duckdb"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
echo "busiest processes now (left as they are):"; ps -eo pid,ni,pcpu,etime,comm --sort=-pcpu | head -6

echo "== building (the CUDA libraries take several minutes)"
make > "$OUT/build_cpu.log" 2>&1 || fail "make (see build_cpu.log)"
make gpu > "$OUT/build_gpu.log" 2>&1 || fail "make gpu (see build_gpu.log)"
python3 -c "import torbles; print('torbles', torbles.__version__, '| GPU:', torbles.gpu_name())"
python3 -c "import torbles, sys; sys.exit('emulat' in torbles.gpu_name().lower())" || fail "the GPU library is the emulation"

echo "== 1. mathematics capabilities still waiting for the GPU"
ready=$(python3 - <<'PY'
import hashlib, json
from pathlib import Path
d = json.load(open("docs/algebra_inventory.json"))
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
print(" ".join(c["id"] for c in d["capabilities"] if c["status"] == "in_progress" and (c.get("validation") or {}).get("cuda")
               and all(Path(f).is_file() and sha(f) == h for f, h in c["validation"]["source_sha256"].items())))
PY
)
if [ -n "$ready" ]; then
  echo "validating: $ready"
  python3 bench/validate_math.py --device --all-ready; rc=$?
  git add -A results docs/algebra_inventory.json ALGEBRA_INVENTORY.md
  git diff --cached --quiet || git commit -q -m "Device receipts for the mathematics inventory ($(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
  mark "device validation" "$([ $rc -eq 0 ] && echo passed || echo "FAILED (rc=$rc)"): $ready"
else
  mark "device validation" "nothing waiting"
fi
python3 -c "
import json; from collections import Counter
print('inventory:', dict(Counter(c['status'] for c in json.load(open('docs/algebra_inventory.json'))['capabilities'])))"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$BASE")" ]; then
  git bundle create "$OUT/torbles-rerun-receipts.git-bundle" "$BASE..HEAD" 2> /dev/null && echo "receipts: $OUT/torbles-rerun-receipts.git-bundle"
fi

echo "== 2. the full test suite, the GPU tests on the GPU"
t0=$SECONDS
bash tests/run_all.sh > "$OUT/full_suite.log" 2>&1; rc=$?
grep -E "FAIL|PASSED|FAILED" "$OUT/full_suite.log" | tail -8
mark "full test suite" "$([ $rc -eq 0 ] && echo passed || echo FAILED), $(grep -c ' ok ' "$OUT/full_suite.log") ok, $(( (SECONDS - t0) / 60 )) min"

echo "== 3. the review's reproducers against this tree"
rm -rf "$RR"; mkdir -p "$RR/review_sql" "$RR/review_native" "$RR/results" || fail "could not make $RR"
ln -s "$REPO" "$RR/candidate"
cp "$REVIEW/review_sql/release_sql_regressions.py" "$RR/review_sql/"
for f in probe_native.py probe_empty.py probe_lifetime.py; do   # their only change: the candidate's path
  sed "s#$REVIEW/candidate#$RR/candidate#g" "$REVIEW/review_native/$f" > "$RR/review_native/$f"
done
cd "$RR" || fail "no $RR"
python3 review_sql/release_sql_regressions.py > "$OUT/review_sql_regressions.log" 2>&1; rc=$?
cp review_sql/release_sql_regressions.json "$OUT/results/" 2>/dev/null
sqlres=$(python3 -c "import json; r = json.load(open('review_sql/release_sql_regressions.json')); print(r['mismatching_cases'], 'mismatching cases of', len(r['cases']), '| controls match:', r['controls_match'])" 2>&1)
mark "review SQL reproducers (R02 R03 R06)" "rc=$rc, $sqlres"
python3 review_native/probe_native.py > "$OUT/probe_native.log" 2>&1; rc=$?
mark "review engine-API probe (R04 R07)" "rc=$rc, see probe_native.log"
python3 review_native/probe_empty.py nplanes > "$OUT/probe_empty.log" 2>&1; rc=$?
python3 review_native/probe_empty.py >> "$OUT/probe_empty.log" 2>&1
mark "review empty-table probe (R08)" "nplanes rc=$rc (the review's: SIGFPE, -8), $(head -1 "$OUT/probe_empty.log") planes"
python3 review_native/probe_lifetime.py > "$OUT/results/probe_lifetime.json" 2> "$OUT/probe_lifetime.err"; rc=$?
leak=$(python3 -c "import json; r = json.load(open('$OUT/results/probe_lifetime.json')); print(r['native_in_use_delta'], 'bytes still allocated after 20 discarded tables; alive', r['alive_weakrefs'])" 2>&1)
mark "review lifetime probe (R05)" "rc=$rc, $leak (the review's: 11,057,408 bytes)"
cat > "$RR/gpu_memory_probe.py" <<'PY'
# GPU memory over tables uploaded, queried on the GPU and released, 120 times: the tables' device copies, and those of
# the query's host arrays (a looked-up attribute's ranked column), go with them, so device memory stays flat. The
# numbers are the whole device's (WSL2 does not report a process's own), so another process using the GPU moves them.
import gc, json, subprocess, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import torbles
from torbles import Query, Table
def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0])
rng = np.random.default_rng(1)
nd, n, samples, equal, devices = 5000, 400_000, [used_mib()], True, set()
for r in range(1, 121):
    d = Table("d", dict(a=rng.integers(0, 50, nd).astype(np.uint32), b=rng.integers(0, 7, nd).astype(np.uint32)))
    f = Table("f", dict(x=rng.integers(0, 1000, n).astype(np.uint32), fk=rng.integers(0, nd, n).astype(np.uint32)), refs={"fk": d})
    for q in (Query(f, join_mode="lookup").join(d, f.fk).group_by(d.a).sum(f.x),
              Query(f, join_mode="lookup").join(d, f.fk).group_by(d.b).count(),
              Query(f).where(f.x < 500).group_by(f.fk).count()):
        got = q.run("gpu"); devices.add(q.device); equal &= got == q.run("alg")
    f.close(); d.close(); del q, f, d; gc.collect()
    if r % 20 == 0: samples.append(used_mib())
print(json.dumps(dict(rounds=120, device_mib_used_every_20_rounds=samples, growth_mib=samples[-1] - samples[1],
                      gpu_equal_cpu=bool(equal), devices=sorted(map(str, devices)), gpu=torbles.gpu_name())))
PY
python3 "$RR/gpu_memory_probe.py" "$REPO" > "$OUT/results/gpu_memory_probe.json" 2> "$OUT/gpu_memory_probe.err"; rc=$?
mark "GPU memory probe" "rc=$rc, $(cat "$OUT/results/gpu_memory_probe.json" 2>/dev/null | tail -1)"

echo "== 4. the wheel, installed in a new venv outside the tree"
rm -rf "$RR/src" "$RR/dist" "$RR/wheel-venv" "$RR/t"; mkdir -p "$RR/src" "$RR/t"
git -C "$REPO" archive HEAD | tar -x -C "$RR/src"          # a clean source tree: the build cannot touch the repository's
( python3 -m pip wheel -q --no-deps -w "$RR/dist" "$RR/src" || python3 -m pip wheel -q --no-deps --no-build-isolation -w "$RR/dist" "$RR/src" ) \
  > "$OUT/wheel_build.log" 2>&1; rc=$?
if [ $rc -eq 0 ] && python3 -m venv "$RR/wheel-venv" && "$RR/wheel-venv/bin/pip" install -q "$RR"/dist/torbles-*.whl pandas pytz >> "$OUT/wheel_build.log" 2>&1; then
  cp -r "$REPO/tests" "$RR/t/"; cd "$RR/t" || fail "no $RR/t"
  "$RR/wheel-venv/bin/python" -c "import sys, torbles, numpy, duckdb; print(torbles.__file__, torbles.__version__, '| GPU:', torbles.gpu_name(), '| Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| DuckDB', duckdb.__version__)" \
    | tee "$OUT/wheel_env.log"
  "$RR/wheel-venv/bin/python" tests/check_min_deps.py > "$OUT/wheel_check_min_deps.log" 2>&1; r0=$?
  res="check_min_deps $([ $r0 -eq 0 ] && echo ok || echo FAIL)"
  for e in alg plain gpu; do
    TORBLES_SQL_ENGINE=$e "$RR/wheel-venv/bin/python" tests/check_sql.py > "$OUT/wheel_check_sql_$e.log" 2>&1
    res="$res, check_sql $e $([ $? -eq 0 ] && tail -1 "$OUT/wheel_check_sql_$e.log" | grep -q "ALL MATCH" && echo ok || echo FAIL)"
  done
  mark "installed wheel" "$res"
else
  mark "installed wheel" "NOT RUN: the wheel did not build or install (see wheel_build.log)"
fi

echo "== 5. the review's benchmarks (one worker at a time, each a fresh process)"
cd "$RR" || fail "no $RR"
cp "$REVIEW/benchmark_release.py" "$REVIEW/benchmark_native_ssb.py" "$RR/"
python3 - "$REPO" "$RR" > "$RR/source_manifest.json" <<'PY'
import hashlib, json, subprocess, sys
from datetime import date
repo, rr = sys.argv[1], sys.argv[2]
names = subprocess.run(["git", "-C", repo, "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
files = {n: hashlib.sha256(open(f"{repo}/{n}", "rb").read()).hexdigest() for n in names if n}
head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
print(json.dumps(dict(date=str(date.today()), source=repo, snapshot=f"{rr}/candidate", file_count=len(files), file_sha256=files,
    canonical_manifest_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    published_tag=None, published_head=head, note="The repository's tracked files at HEAD; the review's harness reads this hash."), indent=2))
PY
bench() {   # label, output json, command...
  local label=$1 out=$2; shift 2
  local t0=$SECONDS
  "$@" > "$OUT/bench_$label.log" 2>&1; local rc=$?
  [ -f "$out" ] && cp "$out" "$OUT/results/"
  mark "benchmark $label" "$([ $rc -eq 0 ] && tail -1 "$OUT/bench_$label.log" || echo "FAILED rc=$rc (see bench_$label.log)"), $(( SECONDS - t0 )) s"
}
for sf in 1 3 10; do
  if [ -f "$SSBDATA/sf$sf/ssb.duckdb" ]; then bench "ssb_sf$sf" results/ssb_sf$sf.json python3 benchmark_release.py --kind ssb --scale $sf --repeats 5 --output results/ssb_sf$sf.json
  else mark "benchmark ssb_sf$sf" "NOT RUN: no $SSBDATA/sf$sf/ssb.duckdb"; fi
done
for n in 10000 100000 1000000; do
  bench "mixed_$n" results/mixed_$n.json python3 benchmark_release.py --kind mixed --scale $n --repeats 3 --output results/mixed_$n.json
done
for sf in 1 3 10; do
  if [ -f "$SSBDATA/sf$sf/ssb.duckdb" ]; then bench "native_ssb_sf$sf" results/native_ssb_sf$sf.json python3 benchmark_native_ssb.py $sf
  else mark "benchmark native_ssb_sf$sf" "NOT RUN: no $SSBDATA/sf$sf/ssb.duckdb"; fi
done
cp "$RR/source_manifest.json" "$OUT/results/"

echo "== the review's numbers and this run's (medians, ms)"
python3 - "$REVIEW/results" "$OUT/results" <<'PY' | tee "$OUT/comparison.md"
import json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
print("| report | part | engine | review | rerun | rerun / review |")
print("|---|---|---|---:|---:|---:|")
for name in ("ssb_sf1", "ssb_sf3", "ssb_sf10", "native_ssb_sf1", "native_ssb_sf3", "native_ssb_sf10",
             "mixed_10000", "mixed_100000", "mixed_1000000"):
    if not (new / f"{name}.json").exists():
        print(f"| {name} | not run | | | | |"); continue
    b = json.loads((new / f"{name}.json").read_text())
    a = json.loads((old / f"{name}.json").read_text()) if (old / f"{name}.json").exists() else None
    parts = [("whole batch", b["warm_ms"], a and a["warm_ms"])] + \
            [(c, b["warm_categories_ms"][c], a and a["warm_categories_ms"].get(c)) for c in sorted(b["warm_categories_ms"])]
    for part, nb, na in parts:
        for eng, v in nb.items():
            o = (na or {}).get(eng, {}).get("median_ms")
            ratio = f"{v['median_ms'] / o:.2f}" if o else ""
            print(f"| {name} | {part} | {eng} | {o:.2f} | {v['median_ms']:.2f} | {ratio} |" if o else
                  f"| {name} | {part} | {eng} | | {v['median_ms']:.2f} | |")
PY

echo "== summary"
printf '  %s\n' "${STEPS[@]}" | tee "$OUT/summary.txt"
echo "== done: $OUT"
