# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The table of wsl/ab_enhancements.sh: A/B of two builds on the SSB queries, rounds interleaved.
    python3 wsl/ab_table.py <dir with A_r1.json, B_r1.json, ...> [label A] [label B]
Per engine and query: each side's best time over its rounds (each round's time is itself the best of 3 runs in one
process), the change B / A - 1, and each side's own spread over the rounds ((max - min) / min of its per-round times):
a change smaller than the spreads is noise. Per engine, the 13 SSB queries together: the sums of the bests, of the
first calls (planning included), and the paired ratio of the two sides' sums round by round (the same round, so the
machine's drift cancels). K1-K3 and M1-M3 are listed but not in the sums."""
import sys, os, glob, json, re, statistics as st

out = sys.argv[1]
label = {"A": sys.argv[2] if len(sys.argv) > 2 else "A", "B": sys.argv[3] if len(sys.argv) > 3 else "B"}
runs = {"A": {}, "B": {}}
for f in glob.glob(os.path.join(out, "[AB]_r*.json")):
    m = re.match(r"([AB])_r(\d+)\.json$", os.path.basename(f))
    rec = json.load(open(f))
    if not rec.get("all_match"): sys.exit(f"{f}: answers differ from DuckDB")
    runs[m.group(1)][int(m.group(2))] = rec
rounds = sorted(set(runs["A"]) & set(runs["B"]))
if not rounds: sys.exit("no round with both sides")
keys = sorted(set.intersection(*(set(runs[s][r]["times"]) for s in "AB" for r in rounds)))
engines = sorted({k.split("/", 1)[1] for k in keys}, key=lambda e: (e.startswith("gpu"), int(re.sub(r"\D", "", e.split("@")[-1]) or 0)))
queries = sorted({k.split("/", 1)[0] for k in keys}, key=lambda q: (q[0] != "Q", q))
ssb = [q for q in queries if q.startswith("Q")]

def t(s, r, q, e, what="best"): return runs[s][r]["times"][f"{q}/{e}"][what] * 1e3
def spread(xs): return (max(xs) - min(xs)) / min(xs) if min(xs) > 0 else 0.0

print(f"A = {label['A']}\nB = {label['B']}\n{len(rounds)} rounds, interleaved; times in ms (best of each round's 3 runs)")
devices = {(s, e): {runs[s][r]["times"][f"{q}/{e}"]["device"] for r in rounds for q in queries} for s in "AB" for e in engines}
verdicts = []
for e in engines:
    print(f"\n== {e}  (ran on: A {', '.join(sorted(devices[('A', e)]))}; B {', '.join(sorted(devices[('B', e)]))})")
    print(f"  {'query':6s} {'A best':>9s} {'B best':>9s} {'change':>8s} {'A spread':>9s} {'B spread':>9s}")
    for q in queries:
        a = [t("A", r, q, e) for r in rounds]; b = [t("B", r, q, e) for r in rounds]
        ch = min(b) / min(a) - 1
        flag = "" if abs(ch) <= max(spread(a), spread(b)) else "  <- beyond both spreads"
        print(f"  {q:6s} {min(a):9.2f} {min(b):9.2f} {100 * ch:+7.1f}% {100 * spread(a):8.1f}% {100 * spread(b):8.1f}%{flag}")
    sa = [sum(t("A", r, q, e) for q in ssb) for r in rounds]; sb = [sum(t("B", r, q, e) for q in ssb) for r in rounds]
    fa = [sum(t("A", r, q, e, "first") for q in ssb) for r in rounds]; fb = [sum(t("B", r, q, e, "first") for q in ssb) for r in rounds]
    sum_a = sum(min(t("A", r, q, e) for r in rounds) for q in ssb); sum_b = sum(min(t("B", r, q, e) for r in rounds) for q in ssb)
    paired = [y / x for x, y in zip(sa, sb)]
    noise = max(spread(sa), spread(sb))
    ch = sum_b / sum_a - 1
    print(f"  13 SSB queries, sum of bests: A {sum_a:.1f}  B {sum_b:.1f}  change {100 * ch:+.1f}%")
    print(f"    per round, B / A: {', '.join(f'{p:.3f}' for p in paired)}  (median {st.median(paired):.3f});"
          f" each side's own spread of its round sums: A {100 * spread(sa):.1f}%, B {100 * spread(sb):.1f}%")
    print(f"  13 SSB queries, first calls (planning included), median round: A {st.median(fa):.1f}  B {st.median(fb):.1f}"
          f"  change {100 * (st.median(fb) / st.median(fa) - 1):+.1f}%")
    side = "slower" if min(paired) > 1 else "faster" if max(paired) < 1 else None
    verdicts.append((e, ch, noise, abs(ch) <= noise, side, len(paired)))
print("\n== verdict (13 SSB queries, sum of bests; noise = the larger of the two sides' own round-to-round spreads)")
for e, ch, noise, within, side, nr in verdicts:
    print(f"  {e:14s} change {100 * ch:+6.1f}%  noise {100 * noise:5.1f}%  "
          + ("within noise: no cost measured" if within else ("B slower beyond noise" if ch > 0 else "B faster beyond noise"))
          + (f"; B {side} in each of the {nr} paired rounds" if side else "; paired rounds go both ways"))
print("  (a consistent direction over few rounds is weak evidence on its own; more rounds narrow it: bash wsl/ab_enhancements.sh 6)")
