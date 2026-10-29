# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Cross-check every query: algebraic engine (torbles) vs plain columnar engine vs DuckDB."""
import duckdb, sys, time
from star import generate, build, load_duckdb, queries
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
gen = generate(n); T, _ = build(gen); con = duckdb.connect(); load_duckdb(gen, con)
def norm(x):
    if isinstance(x, dict): return {k: v for k, v in sorted(x.items())}
    return x
ok = True
for name, q in queries(T).items():
    from algq import set_threads
    set_threads(1); a, b = q.run("alg"), q.run("plain")
    set_threads(2); a2t, b2t = q.run("alg"), q.run("plain"); set_threads(1)
    if a2t != a or b2t != b: a = ("threads differ", a2t)
    forms = {}
    for mode in ("factored", "expanded", "lookup"):
        q.join_mode = mode; forms[mode] = (q.run("alg"), q.run("plain"))
    q.join_mode = "auto"
    b = b if all(v == (a, a) for v in forms.values()) else ("join forms differ", forms)
    rows = con.execute(q.sql()).fetchall()
    dd = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    same = norm(a) == norm(b) == norm(dd)
    ok &= same
    q.run("alg")
    print(f"{name:30s} {'match' if same else 'MISMATCH'}  joins: {q.range_used}  keys: {[('fk field' if q.last_plan.key_kind[i] == 2 else 'coordinate' if q.last_plan.key_kind[i] == 3 else 'col' if q.last_plan.key_kind[i] == 0 else 'lookup') for i in range(q.last_plan.nkey)]}")
print("ALL MATCH" if ok else "FAILURES")
