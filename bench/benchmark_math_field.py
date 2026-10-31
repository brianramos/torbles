# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-001 benchmark: packed GF(2) matrices and planted parity dependencies over table rows.

    python bench/benchmark_math_field.py --capability FIELD-001 --out FILE [--device] [--quick]

Case 1: square packed GF(2) matrices of 32 / 128 / 512 rows and columns at ranks n, n/2, n/8 and densities 0.5 / 0.05
(built as L U with full-rank factors, rows permuted, so the rank is exact). Measured separately: rank, the reduced row
echelon form, the nullspace basis, a consistent solve (with its basis), and syndromes of 1k / 10k vectors (CPU; CUDA
with --device), against NumPy baselines (a vectorized Gauss-Jordan on 0/1 arrays; syndromes as a float32 BLAS product
mod 2, exact below 2^24). Every baseline result is compared with the native one.
Case 2: planted parity dependencies over 1k / 100k / 1M rows (1k / 100k with --quick): 12 Boolean columns and two bit
planes of a code column, with three exact relations (one affine, one through the code planes) and one one-row near
miss. Measured separately: the bit columns (state construction), candidate generation (the proposals of a 256-row
sample), the exact verification of every proposal over every row (CPU; CUDA with --device; DuckDB XOR with `<>` as the
baseline), the whole discovery (proposals, verification, refinement with the witnesses), and the substitution of a
dependent column by the XOR of the others. The near miss must be rejected with its witness row.
Then the refusal thresholds (elimination tensor, discovery sample), at the last admitted and the first refused point.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_field as mf
from torbles.math_budgets import Budget


def known_rank(rng, n, k, density):
    """an n x n matrix of rank exactly k: L (n x k, identity on top) times U (k x n, identity on the left)"""
    L = (rng.random((n, k)) < density).astype(np.uint8)
    L[:k] = np.eye(k, dtype=np.uint8)
    U = (rng.random((k, n)) < density).astype(np.uint8)
    U[:, :k] = np.eye(k, dtype=np.uint8)
    A = mf.Matrix.from_bits(L) @ mf.Matrix.from_bits(U) if k else mf.Matrix(n, n)
    perm = rng.permutation(n)
    return mf.Matrix(n, n, A.words[perm].copy())


def numpy_rank(bits):
    """Gauss-Jordan over GF(2) on a 0/1 uint8 array (NumPy vectorized row XOR): the rank"""
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
                got = bench.measure("rank", "cpu", A.rank, repeats=5, **p)
                bits = A.to_bits()
                base = bench.measure("rank", "numpy-elimination", lambda: numpy_rank(bits), repeats=2, **p)
                if got != k or base != k:
                    raise SystemExit(f"rank mismatch at {p}: native {got}, numpy {base}")
                bench.measure("reduced row echelon form", "cpu", lambda: len(A.rref()[1]), repeats=5, **p)
                bench.measure("nullspace basis", "cpu", lambda: A.nullspace().rows, repeats=5, **p)
                x0 = int(rng.integers(0, 1 << 62)) % (1 << n) if n < 62 else int.from_bytes(rng.bytes(n // 8), "little")
                b = A.matvec(x0)
                sol = bench.measure("solve (consistent, with basis)", "cpu", lambda: A.solve(b), repeats=5, **p)
                if sol["status"] != ("unique" if k == n else "family") or A.matvec(sol["x"]) != b:
                    raise SystemExit(f"solve mismatch at {p}")
            H = known_rank(rng, n, n // 2, 0.5)
            hb = H.to_bits().astype(np.float32)
            for nvec in (1000, 10_000):
                X = mf.Matrix.from_bits((rng.random((nvec, n)) < 0.5).astype(np.uint8))
                q = dict(n=n, vectors=nvec)
                for backend in bench.backends():
                    S = bench.measure("syndromes H x for many x", backend, lambda: H.syndromes(X, backend=backend),
                                      repeats=5, **q)
                xb = X.to_bits().astype(np.float32)
                want = bench.measure("syndromes H x for many x", "numpy-blas-mod2",
                                     lambda: (np.rint(xb @ hb.T).astype(np.int64) & 1).astype(np.uint8), repeats=3, **q)
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
    code = (rng.integers(0, 1000, rows) * 4 + (cols["b7"] ^ cols["b8"]).astype(np.int64)   # plane 0 = b7 ^ b8,
            + 2 * rng.integers(0, 2, rows))                                                # plane 1 random
    code[0] = 0                                                            # the smallest code: planes = bits
    frame = pd.DataFrame({**cols, "code": code})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BOOLEAN' for c in cols) + ', "code" BIGINT)'
    return torbles.from_ddl(ddl, {"t": frame}), list(cols), near


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
        bad = [e for e in ev if e["violations"]]
        con = db.con

        def duck():
            out = []
            for r in prop["proposals"]:
                terms = [f'"{n}"' if "." not in n else f'((("code" >> {n[-1]}) & 1) = 1)' for n in r["labels"]]
                expr = "TRUE" if r["constant"] else "FALSE"
                for t in terms:
                    expr = f"({expr} <> {t})"                         # XOR, SQL NULL semantics
                v, w = con.execute(f"SELECT COUNT(*) FILTER (WHERE {expr}), MIN(rowid) FILTER (WHERE {expr}) "
                                   "FROM t").fetchall()[0]
                out.append((int(v), None if w is None else int(w)))
            return out
        want = bench.measure("exact verification of every proposal", "duckdb", duck, repeats=2,
                             proposals=len(rels), **p)
        if want != [(e["violations"], e["witness"]) for e in ev]:
            raise SystemExit(f"DuckDB disagrees with the native verification at {p}")
        for backend in bench.backends():
            res = bench.measure("discovery (proposals, verification, refinement)", backend,
                                lambda: bits.discover(sample=256, backend=backend), repeats=3, **p)
        found = {(r["labels"], r["constant"]) for r in res["relations"]}
        planted = {(("b0", "b1", "d0"), 0), (("b2", "b3", "b4", "d1"), 1), (("b7", "b8", "code.code0"), 0)}
        rejected = [(r["labels"], r["violations"], r["witness"]) for r in res["rejected"]]
        ok = planted <= found and rejected == [(("b5", "b6", "d2"), 1, near)]
        bench.note(f"{rows} rows: planted relations found {planted <= found}; near miss rejected with its witness row "
                   f"{rejected == [(('b5', 'b6', 'd2'), 1, near)]}; rounds {res['rounds']}; proposals failing in the "
                   f"first round {len(bad)}")
        if not ok:
            raise SystemExit(f"planted relations not recovered at {p}: {found}, {rejected}")
        rel = (["b0", "b1"], 0)
        sub = bench.measure("substitution (d0 as b0 XOR b1)", "cpu", lambda: bits.combine(rel)[0], repeats=5, **p)
        if not np.array_equal(sub, bits.words(names.index("d0"))[0]):
            raise SystemExit("the substituted column differs from the stored one")


def thresholds(bench):
    rng = np.random.default_rng(bench.seed + 2)
    budget = Budget(tensor=512 * 512)
    A = known_rank(rng, 512, 256, 0.5)
    bench.measure("rank within a 512 x 512 tensor budget", "cpu", lambda: A.rank(budget=budget), repeats=1,
                  rows=512, cols=512)
    B = mf.Matrix.from_bits(np.vstack([A.to_bits(), np.zeros((1, 512), np.uint8)]))
    bench.refusal("rank past a 512 x 512 tensor budget", "cpu", lambda: B.rank(budget=budget), rows=513, cols=512)
    bench.threshold("elimination elements (tensor 2^18)", admitted=dict(rows=512, cols=512), refused=dict(rows=513,
                                                                                                         cols=512))
    db, names, near = planted_table(rng, 2000)
    bits = mf.Bits.from_scope(mbk.Scope.from_table(db["t"], names), names)
    sb = Budget(support=256)
    bench.measure("discovery within a 256-row sample budget", "cpu",
                  lambda: len(bits.discover(sample=256, budget=sb)["relations"]), repeats=1, sample=256)
    bench.refusal("discovery past a 256-row sample budget", "cpu", lambda: bits.discover(sample=257, budget=sb),
                  sample=257)
    bench.threshold("discovery sample rows (support 256)", admitted=dict(sample=256), refused=dict(sample=257))


def main():
    bench = Bench.from_args("FIELD-001", __doc__)
    case_matrices(bench)
    case_parity(bench)
    thresholds(bench)
    bench.note("Native CPU paths are single-threaded; elimination, rank, nullspace and solve run on the CPU only "
               "(no CUDA path); syndromes, verification and discovery's verification have CUDA kernels (timed with "
               "--device only).")
    bench.write()


if __name__ == "__main__":
    main()
