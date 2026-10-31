# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-002 benchmark: parity relationship discovery, certification and substitution.

    python bench/benchmark_math_parity.py --capability FIELD-002 --out FILE [--device] [--quick]

Case 1: the packed GF(2) primitives the discovery stands on (FIELD-001's matrices): square matrices of 32 / 128 / 512
rows and columns at ranks n, n/2, n/8 and densities 0.5 / 0.05 (L U with full-rank factors, rows permuted, so the rank
is exact). Measured separately: rank, the reduced row echelon form (elimination), the nullspace basis (the proposal
step), and syndromes H x of 1k vectors (H: the relations as a parity-check matrix; CPU, CUDA with --device), against
NumPy baselines (a vectorized Gauss-Jordan on 0/1 arrays; a float32 BLAS product mod 2, exact below 2^24). Every
baseline result is compared with the native one.
Case 2: planted parity dependencies over 1k / 100k / 1M rows (1k / 100k with --quick): Boolean columns and two code
planes with three exact relations (one affine, one through a code plane) and a one-row near miss. Measured separately:
the bit columns (state construction), candidate generation (the proposals of a 256-row sample), the exact verification
of every proposal over every row (CPU; CUDA with --device; DuckDB XOR with `<>` as the baseline), the whole discovery
with its FND-007 certificates (CPU; CUDA with --device), the re-certification of the found relations, the substitution
(the rewrite of a predicate over the XOR of columns into one column, and back), and the verification of the
substituted predicate against the original on every row (CPU; CUDA with --device; DuckDB IS DISTINCT FROM as the
baseline). The near miss must be rejected with its witness row; after a codec change the substitution must be refused
with the staleness reason.
Then the refusal thresholds (discovery sample, substitution circuit), at the last admitted and the first refused point.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import coltypes
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_certificates as mc
from torbles import math_field as mf
from torbles import math_parity as mp
from torbles.math_budgets import Budget

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR


def known_rank(rng, n, k, density):
    L = (rng.random((n, k)) < density).astype(np.uint8)
    L[:k] = np.eye(k, dtype=np.uint8)
    U = (rng.random((k, n)) < density).astype(np.uint8)
    U[:, :k] = np.eye(k, dtype=np.uint8)
    A = mf.Matrix.from_bits(L) @ mf.Matrix.from_bits(U) if k else mf.Matrix(n, n)
    return mf.Matrix(n, n, A.words[rng.permutation(n)].copy())


def numpy_rank(bits):
    M = bits.copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        p = np.flatnonzero(M[r:, c])
        if not len(p):
            continue
        p = r + p[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        hit = np.flatnonzero(M[:, c])
        hit = hit[hit != r]
        M[hit] ^= M[r]
        r += 1
    return r


def case_matrices(bench):
    rng = np.random.default_rng(bench.seed)
    for n in (32, 128, 512):
        for k in (n, n // 2, n // 8):
            for density in (0.5, 0.05):
                A = known_rank(rng, n, k, density)
                p = dict(n=n, rank=k, density=density)
                got = bench.measure("rank", "cpu", A.rank, repeats=3, **p)
                bits = A.to_bits()
                base = bench.measure("rank", "numpy-elimination", lambda: numpy_rank(bits), repeats=1, **p)
                if got != k or base != k:
                    raise SystemExit(f"rank mismatch at {p}: native {got}, numpy {base}")
                bench.measure("elimination (reduced row echelon form)", "cpu", lambda: len(A.rref()[1]), repeats=3, **p)
                N = bench.measure("nullspace basis (the proposals)", "cpu", A.nullspace, repeats=3, **p)
                if N.rows != n - k or (A @ N.T).to_ints() != [0] * n:
                    raise SystemExit(f"nullspace mismatch at {p}")
        H = known_rank(rng, n, n // 2, 0.5)
        hb = H.to_bits().astype(np.float32)
        X = mf.Matrix.from_bits((rng.random((1000, n)) < 0.5).astype(np.uint8))
        q = dict(n=n, vectors=1000)
        for backend in bench.backends():
            S = bench.measure("syndromes H x of many rows", backend, lambda: H.syndromes(X, backend=backend), repeats=3,
                              **q)
        xb = X.to_bits().astype(np.float32)
        want = bench.measure("syndromes H x of many rows", "numpy-blas-mod2",
                             lambda: (np.rint(xb @ hb.T).astype(np.int64) & 1).astype(np.uint8), repeats=2, **q)
        if not np.array_equal(S.to_bits(), want):
            raise SystemExit(f"syndrome mismatch at {q}")


def planted_table(rng, rows):
    cols = {f"b{i}": rng.integers(0, 2, rows).astype(bool) for i in range(9)}
    cols["b7"][0] = cols["b8"][0] = False                                  # row 0: code 0, the codec offset
    cols["d0"] = cols["b0"] ^ cols["b1"]                                  # exact
    cols["d1"] = ~(cols["b2"] ^ cols["b3"] ^ cols["b4"])                  # exact, affine (constant 1)
    near = rows // 3 + 7
    d2 = cols["b5"] ^ cols["b6"]
    d2[near] = ~d2[near]                                                   # a one-row near miss
    cols["d2"] = d2
    code = (rng.integers(0, 1000, rows) * 4 + (cols["b7"] ^ cols["b8"]).astype(np.int64)   # plane 0 = b7 ^ b8
            + 2 * rng.integers(0, 2, rows))                                                # plane 1 random
    code[0] = 0
    frame = pd.DataFrame({**cols, "code": code})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BOOLEAN' for c in cols) + ', "code" BIGINT)'
    return torbles.from_ddl(ddl, {"t": frame}), list(cols), near


def node_sql(nodes, root, conds_sql):
    memo = {}

    def go(i):
        if i in memo:
            return memo[i]
        op, a, b = nodes[i]
        if op == CONST:
            s = ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[a]
        elif op == VAR:
            s = conds_sql[a]
        elif op == NOT:
            s = f"(NOT {go(a)})"
        else:
            s = f"({go(a)} {dict([(AND, 'AND'), (OR, 'OR'), (XOR, '<>')])[op]} {go(b)})"
        memo[i] = s
        return s
    return go(root)


def condition_sql(c):
    if c.op == "bit":
        base, bit = c.args
        return f'(((((CAST("{c.column}" AS HUGEINT) - ({base})) >> {bit}) & 1)) = 1)'
    (n, d), = c.args
    return f'(CAST("{c.column}" AS HUGEINT) * {d} {c.op} {n})'


def case_parity(bench):
    rng = np.random.default_rng(bench.seed + 1)
    for rows in ((1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)):
        db, names, near = planted_table(rng, rows)
        scope = mbk.Scope.from_table(db["t"], names + ["code"])
        specs = names + [("code", "code", 0), ("code", "code", 1)]
        p = dict(rows=rows, variables=len(specs))
        bits = bench.measure("bit columns (state construction)", "cpu", lambda: mf.Bits.from_scope(scope, specs),
                             repeats=3, phase="build", **p)
        prop = bench.measure("candidate generation (256-row sample)", "cpu", lambda: bits.propose(sample=256),
                             repeats=5, **p)
        rels = [(r["vars"], r["constant"]) for r in prop["proposals"]]
        for backend in bench.backends():
            ev = bench.measure("exact verification of every proposal", backend,
                               lambda: bits.verify(rels, backend=backend), repeats=5, proposals=len(rels), **p)
        con = db.con

        def duck():
            out = []
            for r in prop["proposals"]:
                terms = [f'"{n}"' if "." not in n else f'((("code" >> {n[-1]}) & 1) = 1)' for n in r["labels"]]
                expr = "TRUE" if r["constant"] else "FALSE"
                for t in terms:
                    expr = f"({expr} <> {t})"
                v, w = con.execute(f"SELECT COUNT(*) FILTER (WHERE {expr}), MIN(rowid) FILTER (WHERE {expr}) "
                                   "FROM t").fetchall()[0]
                out.append((int(v), None if w is None else int(w)))
            return out
        want = bench.measure("exact verification of every proposal", "duckdb", duck, repeats=2, proposals=len(rels),
                             **p)
        if want != [(e["violations"], e["witness"]) for e in ev]:
            raise SystemExit(f"DuckDB disagrees with the native verification at {p}")
        for backend in bench.backends():
            found = bench.measure("discovery with FND-007 certificates", backend,
                                  lambda: mp.discover(scope, specs, sample=256, backend=backend), repeats=3, **p)
        found = mp.discover(scope, specs, sample=256)
        labels = {(r["labels"], r["constant"]): r for r in found.relations}
        planted = {(("b0", "b1", "d0"), 0), (("b2", "b3", "b4", "d1"), 1), (("b7", "b8", "code.code0"), 0)}
        rejected = [(r["labels"], r["violations"], r["witness"]) for r in found.rejected]
        if not planted <= set(labels) or rejected != [(("b5", "b6", "d2"), 1, near)]:
            raise SystemExit(f"planted relations not recovered at {p}: {set(labels)}, {rejected}")
        if any(labels[k]["status"] != "certified" for k in planted):
            raise SystemExit(f"a planted relation is not certified at {p}")
        certs = [labels[k]["certificate"] for k in sorted(planted)]
        for backend in bench.backends():
            bench.measure("re-certification of the found relations", backend,
                          lambda: mc.reverify(certs, scope, backend=backend), repeats=3, relations=len(certs), **p)
        cert = labels[(("b0", "b1", "d0"), 0)]["certificate"]
        conds = [mp.bit_condition(scope, "b0"), mp.bit_condition(scope, "b1"), mb.condition("b2", "<>", 0)]
        x = mb.variables(3)
        rw = bench.measure("substitution: b0 XOR b1 rewritten as d0", "cpu",
                           lambda: mp.substitute(scope, conds, (x[0] ^ x[1]) & x[2], cert), repeats=5, **p)
        back = bench.measure("substitution: d0 rewritten as b0 XOR b1 (mode all)", "cpu",
                             lambda: mp.substitute(scope, [mp.bit_condition(scope, "d0"), conds[2]], x[0] & x[1], cert,
                                                   mode="all"), repeats=5, **p)
        bench.note(f"{rows} rows: (b0 XOR b1) AND b2 -> {rw.chosen}, cost {rw.costs}; d0 AND b2 -> {back.chosen} "
                   f"(mode all), cost {back.costs}; certified relations {found.info['certified']}; rounds "
                   f"{found.info['rounds']}; near miss rejected at its row {near}")
        for backend in bench.backends():
            masks = bench.measure("condition bitmaps of the rewrite (state construction)", backend,
                                  lambda: mb.Masks.from_scope(scope, rw.conditions, backend=backend), repeats=3,
                                  phase="build", **p)
            res = bench.measure("verification of the substituted predicate on every row", backend,
                                lambda: mp.verify_programs(masks, rw.original, rw.program, backend=backend), repeats=5,
                                **p)
            if not res["equivalent"] or res != mp.verify(rw, scope, backend=backend):
                raise SystemExit(f"the substitution is not equivalent at {p}")
        csql = [condition_sql(c) for c in rw.conditions]
        o = node_sql(rw.original.nodes, rw.original.root, csql)
        s = node_sql(rw.program.nodes, rw.program.root, csql)
        d = bench.measure("verification of the substituted predicate on every row", "duckdb",
                          lambda: con.execute(f"SELECT COUNT(*) FILTER (WHERE ({o}) IS DISTINCT FROM ({s})), "
                                              f"COUNT(*) FILTER (WHERE {o}) FROM t").fetchall()[0], repeats=2, **p)
        if (int(d[0]), int(d[1])) != (0, res["original"]["true"]):
            raise SystemExit(f"DuckDB disagrees with the substitution check at {p}")
        if rows == 1000:                                       # a codec change: the substitution is refused
            t = db["t"]
            old = t.codecs["code"]
            t.codecs["code"] = coltypes.IntCodec(old.affine[0] + 1)
            try:
                moved = mbk.Scope.from_table(t, names + ["code"])
                ccert = labels[(("b7", "b8", "code.code0"), 0)]["certificate"]
                try:
                    mp.substitute(moved, [mp.bit_condition(scope, ("code", "code", 0))], mb.var(0), ccert, mode="all")
                    raise SystemExit("a stale certificate authorized a substitution")
                except mc.StaleCertificateError as e:
                    bench.note(f"after a codec change of code the substitution is refused: {e.reasons}")
            finally:
                t.codecs["code"] = old


def thresholds(bench):
    rng = np.random.default_rng(bench.seed + 2)
    db, names, near = planted_table(rng, 2000)
    scope = mbk.Scope.from_table(db["t"], names + ["code"])
    sb = Budget(support=256)
    bench.measure("discovery within a 256-row sample budget", "cpu",
                  lambda: len(mp.discover(scope, names, sample=256, budget=sb).relations), repeats=1, sample=256)
    bench.refusal("discovery past a 256-row sample budget", "cpu", lambda: mp.discover(scope, names, sample=257,
                                                                                       budget=sb), sample=257)
    bench.threshold("discovery sample rows (support 256)", admitted=dict(sample=256), refused=dict(sample=257))
    found = mp.discover(scope, names, sample=256)
    cert = next(r for r in found.relations if r["labels"] == ("b0", "b1", "d0"))["certificate"]
    conds = [mp.bit_condition(scope, "d0"), mb.condition("b2", "<>", 0)]
    x = mb.variables(2)
    nn = mp.substitute(scope, conds, x[0] & x[1], cert, mode="all").info["nodes"]
    bench.measure("substitution within its circuit budget", "cpu",
                  lambda: mp.substitute(scope, conds, x[0] & x[1], cert, mode="all", budget=Budget(circuit=nn)),
                  repeats=1, nodes=nn)
    bench.refusal("substitution one node past its circuit budget", "cpu",
                  lambda: mp.substitute(scope, conds, x[0] & x[1], cert, mode="all", budget=Budget(circuit=nn - 1)),
                  nodes=nn)
    bench.threshold("rewritten program nodes (circuit budget)", admitted=dict(circuit=nn), refused=dict(circuit=nn - 1))


def main():
    bench = Bench.from_args("FIELD-002", __doc__)
    case_matrices(bench)
    case_parity(bench)
    thresholds(bench)
    bench.note("Discovery is FIELD-001's (proposals, exhaustive verification); certification is FND-007's snapshot "
               "(CUDA: tr_mgpu_cert_verify adopted); the substitution's rewrite runs on the CPU and its verification on "
               "every row has a CUDA kernel. Native CPU paths are single-threaded; CUDA is timed only with --device.")
    bench.write()


if __name__ == "__main__":
    main()
