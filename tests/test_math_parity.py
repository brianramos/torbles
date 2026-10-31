# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-002: parity relationship discovery, certification and substitution (torbles/math_parity.c,
math_parity_gpu.cu, math_parity.py).

Oracles, independent of the implementation: a Gauss-Jordan elimination on Python ints (rows as bit sets) for the exact
relation space of the table rows, Python booleans on the source values for every relation and its witness, DuckDB SQL
(XOR as `<>`, SQL NULLs, COUNT(*) FILTER, MIN(rowid)) for the planted relations and for the original and the
substituted predicates (counted separately, and their disagreement with IS DISTINCT FROM), and the columns and planes a
program scans recomputed here from its nodes. Every result with a CUDA path (discovery, certification, verification of
the substitution) is also computed there (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and
compared for equality. Seeds are fixed.
"""
import unittest

import numpy as np
import pandas as pd

import algq
import torbles
from torbles import coltypes
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_certificates as mc
from torbles import math_handles as mh
from torbles import math_parity as mp
from torbles.math_budgets import Budget
from torbles.math_certificates import NotAuthorizedError, StaleCertificateError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR


# ---------------------------------------------------------------------- oracles
def py_rref(rows, cols):
    rows, r, piv = list(rows), 0, []
    for c in range(cols):
        p = next((i for i in range(r, len(rows)) if (rows[i] >> c) & 1), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> c) & 1:
                rows[i] ^= rows[r]
        piv.append(c)
        r += 1
    return rows, piv


def py_nullspace(rows, cols):
    R, piv = py_rref(rows, cols)
    out = []
    for f in range(cols):
        if f in piv:
            continue
        v = 1 << f
        for i, p in enumerate(piv):
            if (R[i] >> f) & 1:
                v |= 1 << p
        out.append(v)
    return out


def decode(table, col):
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(col), out.ctypes.data)
    return out[:table.n]


def flip_column(table, col, row):
    """flips one stored bit so that exactly column `col` changes at `row`"""
    before = {c: decode(table, c) for c in table.cols}
    for p in range(table.nplanes()):
        table.flip(p, row)
        after = {c: decode(table, c) for c in table.cols}
        if [c for c in table.cols if not np.array_equal(before[c], after[c])] == [col]:
            return after[col]
        table.flip(p, row)
    raise AssertionError(f"no single plane changes only {col} at row {row}")


def obj(values):
    return pd.Series(values, dtype=object)


def planted(rng, n, near):
    """Boolean columns with three exact relations (one affine, one through a code plane) and a one-row near miss"""
    cols = {f"b{i}": rng.integers(0, 2, n).astype(bool) for i in range(9)}
    cols["b5"][0] = cols["b6"][0] = False                  # row 0 holds code 0 (the codec offset: planes = bits)
    cols["d0"] = cols["b0"] ^ cols["b1"]
    cols["d1"] = ~(cols["b2"] ^ cols["b3"] ^ cols["b4"])
    d2 = cols["b7"] ^ cols["b8"]
    d2[near] = ~d2[near]
    cols["d2"] = d2
    code = rng.integers(0, 1000, n) * 4 + (cols["b5"] ^ cols["b6"]).astype(np.int64) + 2 * rng.integers(0, 2, n)
    code[0] = 0
    names = list(cols)
    frame = pd.DataFrame({**cols, "code": code})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BOOLEAN' for c in names) + ', "code" BIGINT)'
    return torbles.from_ddl(ddl, {"t": frame}), cols, code, names


def strip(rec):
    if isinstance(rec, list):
        return [strip(r) for r in rec]
    if isinstance(rec, dict):
        return {k: strip(v) for k, v in rec.items() if k not in ("backend", "device", "certificate")}
    return rec


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
    """the SQL of a math_bool Condition built by bit_condition / condition(name, '<>', 0) / comparisons"""
    if c.op == "bit":
        base, bit = c.args
        return f'(((((CAST("{c.column}" AS HUGEINT) - ({base})) >> {bit}) & 1)) = 1)'
    if c.op in ("is null", "is not null"):
        return f'("{c.column}" {c.op.upper()})'
    (n, d), = c.args
    return f'(CAST("{c.column}" AS HUGEINT) * {d} {c.op} {n})'


def reachable_vars(nodes, root):
    seen, out, stack = set(), set(), [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        op, a, b = nodes[i]
        if op == VAR:
            out.add(a)
        elif op == NOT:
            stack.append(a)
        elif op in (AND, OR, XOR):
            stack += [a, b]
    return out


def scan_cost(conditions, used):
    """columns and planes scanned: a BIT condition one plane (column, bit, base), any other its column; a scanned
    column subsumes its planes"""
    whole = {conditions[i].column for i in used if conditions[i].op != "bit"}
    planes = {(conditions[i].column,) + tuple(conditions[i].args) for i in used if conditions[i].op == "bit"}
    return len(whole) + len({p for p in planes if p[0] not in whole})


class Parity(unittest.TestCase):
    def found_both(self, scope, specs, **kw):
        cpu = mp.discover(scope, specs, **kw)
        gpu = mp.discover(scope, specs, backend="cuda", **kw)
        self.assertEqual(strip(cpu.relations), strip(gpu.relations))
        self.assertEqual(strip(cpu.rejected), strip(gpu.rejected))
        ci, gi = cpu.info, gpu.info
        self.assertEqual({k: v for k, v in ci.items() if k not in ("backend", "device", "verified_by")},
                         {k: v for k, v in gi.items() if k not in ("backend", "device", "verified_by")})
        self.assertEqual((ci["verified_by"], gi["verified_by"]), ("cpu", "cuda library"))
        for c, g in zip(cpu.certificates, gpu.certificates):
            self.assertEqual(c is None, g is None)
            if c is not None:
                self.assertEqual(c.claim, g.claim)
                self.assertEqual({k: v for k, v in c.evidence.items() if k not in ("backend", "device")},
                                 {k: v for k, v in g.evidence.items() if k not in ("backend", "device")})
                self.assertEqual(g.info["verified_by"], "cuda library")
        return cpu

    def verify_both(self, rw, scope):
        cpu = mp.verify(rw, scope)
        gpu = mp.verify(rw, scope, backend="cuda")
        self.assertEqual(strip(cpu), strip(gpu))
        self.assertIn(gpu["backend"], ("cuda", "cuda-emulated"))
        return cpu

    # ------------------------------------------------------------------ acceptance: planted relations, near misses
    def test_field002_recovers_planted_parity_relations_and_rejects_one_row_near_misses(self):
        rng = np.random.default_rng(8101)
        n, sample = 4000, 100
        near = 7 * (n // sample) + 3                         # not one of the sampled rows
        db, cols, code, names = planted(rng, n, near)
        scope = mbk.Scope.from_table(db["t"], names + ["code"])
        specs = names + [("code", "code", 0), ("code", "code", 1)]
        found = self.found_both(scope, specs, sample=sample)
        offset = scope.column("code")["offset"]
        self.assertEqual(offset, 0)
        bits = [cols[c] for c in names] + [((code - offset) >> 0) & 1, ((code - offset) >> 1) & 1]
        nv = len(bits)
        vecs = [1 | sum(int(bits[j][r]) << (j + 1) for j in range(nv)) for r in range(n)]
        want = py_nullspace(vecs, nv + 1)                     # the exact relation space of every row
        got = [int(r["constant"]) | sum(1 << (j + 1) for j in r["vars"]) for r in found.relations]
        self.assertEqual(got, want)
        labels = {(r["labels"], r["constant"]) for r in found.relations}
        self.assertEqual(labels, {(("b0", "b1", "d0"), 0), (("b2", "b3", "b4", "d1"), 1),
                                  (("b5", "b6", "code.code0"), 0)})
        self.assertEqual([(r["labels"], r["constant"], r["violations"], r["witness"]) for r in found.rejected],
                         [(("b7", "b8", "d2"), 0, 1, near)])            # the one-row near miss, with its row
        self.assertEqual((found.info["rounds"], found.info["initial_sample"], found.info["sample_rows"]),
                         (2, sample, sample + 1))
        for r in found.relations:                                # each relation certified, exhaustively
            self.assertEqual((r["status"], r["violations"], r["checked"], r["holds"]), ("certified", 0, n, True))
            self.assertEqual((r["certified"]["checked"], r["certified"]["violations"], r["certified"]["holds"]),
                             (n, 0, True))
            cert = r["certificate"]
            self.assertEqual((cert.kind, cert.info["claim"], cert.info["exact"], cert.info["field"]),
                             ("snapshot", "parity", True, 2))
            claim = cert.claim
            self.assertEqual(claim["names"], sorted({lab.split(".")[0] for lab in r["labels"]},
                                                    key=lambda c: scope.names.index(c)))
            self.assertEqual(claim["constant"], r["constant"])
            src = cert.source()                                  # scope, revision, codec and NULL mask recorded
            self.assertEqual((src["revision"], src["rows"], src["selected"], src["exact_selection"]),
                             (scope.revision, n, n, True))
            for c in src["columns"]:
                self.assertEqual((c["offset"], c["scale"], c["nulls"]), (scope.column(scope.names[c["index"]])["offset"],
                                                                         0, 0))
            self.assertEqual(cert.authorize_exact(scope, field=2).nulls, "forbid")
        code_cert = next(r for r in found.relations if "code.code0" in r["labels"])["certificate"]
        self.assertEqual(code_cert.claim["bits"][-1], (2, "plane", 0, offset))         # the code plane and its base
        # DuckDB: the planted relations hold, the near miss fails at its row only
        con = db.con
        for sql, viol, wit in (('(("b0" <> "b1") <> "d0")', 0, None),
                               ('((("b2" <> "b3") <> "b4") <> "d1") <> TRUE', 0, None),
                               ('(("b5" <> "b6") <> ((("code" >> 0) & 1) = 1))', 0, None),
                               ('(("b7" <> "b8") <> "d2")', 1, near)):
            v, w = con.execute(f"SELECT COUNT(*) FILTER (WHERE {sql}), MIN(rowid) FILTER (WHERE {sql}) FROM t").fetchall()[0]
            self.assertEqual((int(v), None if w is None else int(w)), (viol, wit), sql)
        # a selection that skips the near-miss row: the relation holds there and is certified on those rows
        mask = np.ones(n, bool); mask[near] = False
        sub = mbk.Scope.from_table(db["t"], names + ["code"], rows=mask)
        f2 = self.found_both(sub, specs, sample=sample)
        rel = next(r for r in f2.relations if r["labels"] == ("b7", "b8", "d2"))
        self.assertEqual((rel["status"], rel["certified"]["checked"]), ("certified", n - 1))
        self.assertEqual(rel["certificate"].source()["selected"], n - 1)
        self.assertEqual(f2.rejected, [])

    def test_field002_nulls_policies_and_uncertifiable_relations(self):
        rng = np.random.default_rng(8102)
        n = 1500
        a, b, c = (rng.integers(0, 2, n).astype(bool) for _ in range(3))
        m = [None if rng.random() < 0.1 else bool(x) for x in (a ^ b)]     # m = a XOR b where present
        z = [None if i % 97 == 5 else bool(x) for i, x in enumerate(rng.integers(0, 2, n))]
        q = (a ^ c).copy()
        bad_row = next(i for i in range(n) if z[i] is None)
        q[bad_row] = ~q[bad_row]                    # q = a XOR c except on a row where z is NULL
        w = [rng.integers(0, 2, n).astype(bool) for _ in range(8)]
        w8 = w[0] ^ w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6] ^ w[7]             # a relation over 9 columns
        frame = pd.DataFrame({"a": a, "b": b, "c": c, "m": obj(m), "z": obj(z), "q": q,
                              **{f"w{i}": w[i] for i in range(8)}, "w8": w8})
        ddl = "CREATE TABLE t (" + ", ".join(f'"{k}" BOOLEAN' for k in frame.columns) + ")"
        db = torbles.from_ddl(ddl, {"t": frame})
        scope = mbk.Scope.from_table(db["t"], list(frame.columns))
        specs = ["a", "b", "c", "m", "z", "q"]
        first_null_m = min(i for i in range(n) if m[i] is None)
        nulls_m = sum(v is None for v in m)
        forbid = self.found_both(scope, specs, sample=0)
        by = {r["labels"]: r for r in forbid.relations}
        self.assertEqual(by[("a", "b", "m")]["status"], "violated")           # FORBID: a NULL violates
        self.assertEqual(by[("a", "b", "m")]["certified"]["witness"], first_null_m)
        self.assertEqual(by[("a", "c", "q")]["status"], "violated")           # fails where z (not its own) is NULL
        self.assertEqual(by[("a", "c", "q")]["certified"]["witness"], bad_row)
        self.assertIsNone(by[("a", "b", "m")]["certificate"])
        for r in forbid.relations:                            # exact on the complete rows only (listwise)
            self.assertEqual((r["violations"], r["holds"]), (0, False))
        skip = self.found_both(scope, specs, sample=0, nulls="skip")
        sb = {r["labels"]: r for r in skip.relations}
        self.assertEqual(sb[("a", "b", "m")]["status"], "certified")          # SKIP: certified on its complete rows
        self.assertEqual((sb[("a", "b", "m")]["certified"]["excluded"], sb[("a", "b", "m")]["certified"]["checked"]),
                         (nulls_m, n - nulls_m))
        self.assertEqual(sb[("a", "b", "m")]["certificate"].authorize_exact(scope, field=2).nulls, "skip")
        self.assertEqual((sb[("a", "c", "q")]["status"], sb[("a", "c", "q")]["certified"]["witness"]),
                         ("violated", bad_row))
        wide = self.found_both(scope, [f"w{i}" for i in range(9)], sample=0, affine=False)
        self.assertEqual([(r["labels"], r["status"], r["columns"]) for r in wide.relations],
                         [(tuple(f"w{i}" for i in range(9)), "too wide", 9)])
        self.assertIsNone(wide.relations[0]["certificate"])
        empty = mbk.Scope.from_table(db["t"], ["a", "b"], rows=np.zeros(n, bool))
        e = self.found_both(empty, ["a", "b"], sample=0)     # no complete row: every relation vacuous
        self.assertEqual([(r["vars"], r["constant"], r["status"]) for r in e.relations],
                         [((), 1, "empty"), ((0,), 0, "certified"), ((1,), 0, "certified")])
        self.assertEqual(e.relations[1]["certified"]["checked"], 0)

    # ------------------------------------------------------------------ substitution
    def test_field002_substitution_chooses_the_cheaper_form_and_verifies_on_every_row(self):
        rng = np.random.default_rng(8103)
        n = 3000
        _, cols, code, names = planted(rng, n, near=11)
        v = [None if rng.random() < 0.2 else int(x) for x in rng.integers(-5, 6, n)]
        frame = pd.DataFrame({**{c: cols[c] for c in names}, "code": code, "v": obj(v)})
        db = torbles.from_ddl("CREATE TABLE t (" + ", ".join(f'"{c}" BOOLEAN' for c in names) +
                              ', "code" BIGINT, "v" BIGINT)', {"t": frame})
        scope = mbk.Scope.from_table(db["t"], names + ["code", "v"])
        specs = names + [("code", "code", 0)]
        found = mp.discover(scope, specs, sample=128)
        cert = {r["labels"]: r["certificate"] for r in found.relations}
        rel0, rel1, relc = cert[("b0", "b1", "d0")], cert[("b2", "b3", "b4", "d1")], cert[("b5", "b6", "code.code0")]
        B = {c: mp.bit_condition(scope, c) for c in names}
        plane0 = mp.bit_condition(scope, ("code", "code", 0))
        vpos = mb.condition("v", ">", 0)
        con = db.con

        def check(conds, expr, certificate, mode, chosen, cost, k_after=None):
            rw = mp.substitute(scope, conds, expr, certificate, mode=mode)
            self.assertEqual(rw.chosen, chosen, expr)
            nodes, root = rw.program.nodes, rw.program.root
            used = reachable_vars(nodes, root)
            self.assertEqual((rw.costs["original"], rw.costs["rewritten"]),
                             (scan_cost(rw.conditions, reachable_vars(rw.original.nodes, rw.original.root)),
                              scan_cost(rw.conditions, used)))
            self.assertEqual((rw.costs["original"], rw.costs["rewritten"]), cost, expr)
            if k_after is not None:
                self.assertEqual(len(rw.conditions), k_after)
            res = self.verify_both(rw, scope)
            self.assertEqual((res["equivalent"], res["mismatches"], res["points"]), (True, 0, n))
            self.assertEqual(res["original"], res["substituted"])
            csql = [condition_sql(c) for c in rw.conditions]
            o = node_sql(rw.original.nodes, rw.original.root, csql)
            s = node_sql(nodes, root, csql)
            want = con.execute(f"SELECT COUNT(*) FILTER (WHERE {o}), COUNT(*) FILTER (WHERE NOT {o}), "
                               f"COUNT(*) FILTER (WHERE ({o}) IS NULL), COUNT(*) FILTER (WHERE {s}), "
                               f"COUNT(*) FILTER (WHERE NOT {s}), COUNT(*) FILTER (WHERE ({s}) IS NULL), "
                               f"COUNT(*) FILTER (WHERE ({o}) IS DISTINCT FROM ({s})) FROM t").fetchall()[0]
            self.assertEqual(tuple(int(x) for x in want),
                             tuple(res["original"].values()) + tuple(res["substituted"].values()) + (0,))
            return rw

        x = mb.variables(4)
        # a predicate over a column that is the XOR of others: kept (cheaper), or rewritten over the others ("all")
        check([B["d0"], vpos], x[0] & x[1], rel0, "auto", "kept", (2, 2))
        rw = check([B["d0"], vpos], x[0] & x[1], rel0, "all", "rewritten", (2, 3), k_after=4)
        self.assertEqual({rw.conditions[i].column for i in reachable_vars(rw.program.nodes, rw.program.root)},
                         {"b0", "b1", "v"})
        # ... and vice versa: the XOR of the others becomes the column (cheaper: chosen automatically)
        rw = check([B["b0"], B["b1"], vpos], (x[0] ^ x[1]) | x[2], rel0, "auto", "rewritten", (3, 2), k_after=4)
        self.assertEqual({rw.conditions[i].column for i in reachable_vars(rw.program.nodes, rw.program.root)},
                         {"d0", "v"})
        # the affine relation (constant 1): NOT (b2 XOR b3 XOR b4) is d1, and b2 XOR b3 XOR b4 is NOT d1
        check([B["b2"], B["b3"], B["b4"], vpos], ~(x[0] ^ x[1] ^ x[2]) & x[3], rel1, "auto", "rewritten", (4, 2))
        check([B["b2"], B["b3"], B["b4"], vpos], (x[0] ^ (x[1] ^ x[2])) | ~x[3], rel1, "auto", "rewritten", (4, 2))
        check([B["d1"], vpos], ~x[0] | x[1], rel1, "all", "rewritten", (2, 4))
        # partial overlaps: b2 XOR d1 = NOT (b3 XOR b4): same cost, kept; b2 XOR b3 XOR b4 XOR d1 is the constant TRUE
        check([B["b2"], B["d1"]], x[0] ^ x[1], rel1, "auto", "kept", (2, 2))
        rw = check([B["b2"], B["b3"], B["b4"], B["d1"]], (x[0] ^ x[1]) ^ (x[2] ^ x[3]), rel1, "auto", "rewritten",
                   (4, 0))
        self.assertEqual(rw.program.nodes[rw.program.root], (CONST, 1, -1))
        # a code plane: bit 0 of the code is b5 XOR b6, a plane is one scan unit
        check([plane0, vpos], x[0] & x[1], relc, "auto", "kept", (2, 2))
        check([B["b5"], B["b6"], vpos], (x[0] ^ x[1]) & x[2], relc, "auto", "rewritten", (3, 2))
        check([plane0, mb.condition("code", ">", 100)], x[0] & x[1], relc, "all", "rewritten", (1, 3))
        # nothing to rewrite: a form under UNKNOWN or AND, a variable outside the relation, a repeated variable
        check([B["b0"], B["b1"], vpos], (x[0] & x[1]) | x[2], rel0, "auto", "kept", (3, 3))
        check([B["b0"], B["b1"], vpos], (x[0] & x[1]) | x[2], rel0, "all", "rewritten", (3, 4))
        check([B["b0"], vpos], (x[0] ^ mb.UNKNOWN) & x[1], rel0, "auto", "kept", (2, 2))
        check([B["b0"], B["b1"], vpos], ((x[0] ^ x[1]) ^ x[0]) & x[2], rel0, "auto", "kept", (3, 3))
        # shared forms are rewritten once, every use follows
        f = x[0] ^ x[1]
        check([B["b0"], B["b1"], vpos], (f & x[2]) | (f ^ x[2]), rel0, "auto", "rewritten", (3, 2))
        check([B["b0"], B["b1"], vpos], (f & x[2]) | ~f, rel0, "auto", "rewritten", (3, 2))

    def test_field002_codec_changes_null_masks_and_revisions_invalidate_affected_certificates(self):
        rng = np.random.default_rng(8104)
        n = 1200
        a, b = rng.integers(0, 2, n).astype(bool), rng.integers(0, 2, n).astype(bool)
        g = rng.integers(0, 2, n).astype(bool)
        x = rng.integers(0, 300, n) * 2 + (a ^ g).astype(np.int64)            # plane 0 of x = a XOR g
        x[0] = int(a[0] ^ g[0])
        x[1] = 600 + int(a[1] ^ g[1])
        nl = [bool(v) for v in rng.integers(0, 2, n)]
        e = [None if i % 50 == 7 else bool(p ^ q) for i, (p, q) in enumerate(zip(nl, b))]   # e = nl XOR b, NULLs
        frame = pd.DataFrame({"a": a, "b": b, "c": a ^ b, "g": g, "x": x, "nl": nl, "e": obj(e)})
        ddl = 'CREATE TABLE t ("a" BOOLEAN, "b" BOOLEAN, "c" BOOLEAN, "g" BOOLEAN, "x" BIGINT, "nl" BOOLEAN, "e" BOOLEAN)'
        db = torbles.from_ddl(ddl, {"t": frame})
        t = db["t"]
        names = ["a", "b", "c", "g", "x", "nl", "e"]
        scope = mbk.Scope.from_table(t, names)
        found = self.found_both(scope, ["a", "b", "c", "g", ("x", "code", 0), "nl", "e"], sample=0, nulls="skip")
        cert = {r["labels"]: r["certificate"] for r in found.relations}
        abc, agx, nbe = cert[("a", "b", "c")], cert[("a", "g", "x.code0")], cert[("b", "nl", "e")]
        dev = mp.discover(scope, ["a", "b", "c", "g", ("x", "code", 0), "nl", "e"], sample=0, nulls="skip",
                          backend="cuda")
        dcert = {r["labels"]: r["certificate"] for r in dev.relations}
        self.assertEqual(dcert[("a", "g", "x.code0")].info["verified_by"], "cuda library")
        for cc in (abc, agx, nbe):
            self.assertTrue(cc.fresh(scope).fresh)
        y = mb.variables(3)
        conds_ab = [mp.bit_condition(scope, "a"), mp.bit_condition(scope, "b"), mb.condition("g", "<>", 0)]
        conds_x = [mp.bit_condition(scope, ("x", "code", 0)), mb.condition("b", "<>", 0)]
        self.assertEqual(mp.substitute(scope, conds_ab, (y[0] ^ y[1]) & y[2], abc).chosen, "rewritten")
        self.assertEqual(mp.substitute(scope, conds_x, y[0] & y[1], agx, mode="all").chosen, "rewritten")
        with self.assertRaises(NotAuthorizedError) as ctx:       # SKIP left the NULL rows of e out: no rewrite
            mp.substitute(scope, [mp.bit_condition(scope, "e")], y[0], nbe, mode="all")
        self.assertEqual(ctx.exception.reason, "nulls")
        # a codec change of x: only the certificate over x's plane is affected, with the exact reason
        old = t.codecs["x"]
        t.codecs["x"] = coltypes.IntCodec(old.affine[0] + 5)
        try:
            moved = mbk.Scope.from_table(t, names)
            self.assertEqual(agx.fresh(moved).reasons, ("columns",))
            self.assertEqual(dcert[("a", "g", "x.code0")].fresh(moved).reasons, ("columns",))
            self.assertTrue(abc.fresh(moved).fresh)
            self.assertTrue(dcert[("a", "b", "c")].fresh(moved).fresh)
            with self.assertRaises(StaleCertificateError) as ctx:
                mp.substitute(moved, conds_x, y[0] & y[1], agx, mode="all")
            self.assertEqual(ctx.exception.reasons, ("columns",))
            self.assertEqual(mp.substitute(moved, conds_ab, (y[0] ^ y[1]) & y[2], abc).chosen, "rewritten")
        finally:
            t.codecs["x"] = old
        self.assertTrue(agx.fresh(mbk.Scope.from_table(t, names)).fresh)
        # a revision (one stored bit flipped and repaired: identical data): every certificate of the table is stale
        t.flip(0, 9); t.repair()
        now = mbk.Scope.from_table(t, names)
        for cc in (abc, agx, nbe):
            self.assertEqual(cc.fresh(now).reasons, ("revision",))
        with self.assertRaises(StaleCertificateError) as ctx:
            mp.substitute(now, conds_ab, (y[0] ^ y[1]) & y[2], abc)
        self.assertEqual(ctx.exception.reasons, ("revision",))
        with self.assertRaises(StaleStateError):                # the scope of the old revision itself
            mp.substitute(scope, conds_ab, (y[0] ^ y[1]) & y[2], abc)
        renewed = mp.discover(now, ["a", "b", "c", "g", ("x", "code", 0)], sample=0, backend="cuda")
        fresh_abc = next(r for r in renewed.relations if r["labels"] == ("a", "b", "c"))["certificate"]
        self.assertTrue(fresh_abc.fresh(now).fresh)
        rw = mp.substitute(now, conds_ab, (y[0] ^ y[1]) & y[2], fresh_abc)
        self.assertTrue(self.verify_both(rw, now)["equivalent"])
        # a NULL mask change (e becomes NULL at one row): the affected certificate reports null_domain too
        row = 3
        self.assertIsNotNone(e[row])
        flip_column(t, t.nullable["e"], row)
        after = mbk.Scope.from_table(t, names)
        self.assertEqual(set(nbe.fresh(after).reasons), {"revision", "null_domain"})
        self.assertEqual(fresh_abc.fresh(after).reasons, ("revision",))
        with self.assertRaises(StaleCertificateError) as ctx:
            mp.substitute(after, conds_ab, (y[0] ^ y[1]) & y[2], fresh_abc)
        self.assertEqual(ctx.exception.reasons, ("revision",))
        # array scopes: a NULL mask change affects exactly the certificates over that column
        av, bv = rng.integers(0, 2, 800), rng.integers(0, 2, 800)
        mv = (av ^ bv)
        present = rng.random(800) < 0.9
        cols = {"a": av, "b": bv, "c": av ^ bv, "m": mv, "k": rng.integers(0, 2, 800)}
        arr = mbk.Scope.from_arrays(cols, valid={"m": present})
        fa = mp.discover(arr, ["a", "b", "c", "m"], sample=0, nulls="skip")
        ca = {r["labels"]: r["certificate"] for r in fa.relations}
        less = present.copy(); less[int(np.flatnonzero(present)[0])] = False
        arr2 = mbk.Scope.from_arrays(cols, valid={"m": less})
        self.assertTrue(ca[("a", "b", "c")].fresh(arr2).fresh)
        self.assertIn("null_domain", ca[("a", "b", "m")].fresh(arr2).reasons)
        # a sampled hypothesis or an approximate claim never authorizes a substitution; a non-parity claim neither
        hyp = mc.hypothesis(after, mc.parity(["a", "b", "c"]), size=50, seed=3)
        with self.assertRaises(NotAuthorizedError) as ctx:
            mp.substitute(after, conds_ab, (y[0] ^ y[1]) & y[2], hyp)
        self.assertEqual((ctx.exception.reason, ctx.exception.detail), ("certificate", "statistical"))
        other = mc.snapshot(after, mc.not_null("a"))
        with self.assertRaises(NotAuthorizedError) as ctx:
            mp.substitute(after, conds_ab, (y[0] ^ y[1]) & y[2], other)
        self.assertEqual(ctx.exception.reason, "claim")

    # ------------------------------------------------------------------ the verification kernel
    def test_field002_verification_matches_cpu_and_finds_mismatches(self):
        rng = np.random.default_rng(8105)
        n = 20_000
        p = rng.integers(0, 2, n).astype(bool)
        q = rng.integers(0, 2, n).astype(bool)
        r = [None if i % 31 == 0 else bool(v) for i, v in enumerate(rng.integers(0, 2, n))]
        frame = pd.DataFrame({"p": p, "q": q, "r": obj(r)})
        db = torbles.from_ddl('CREATE TABLE t ("p" BOOLEAN, "q" BOOLEAN, "r" BOOLEAN)', {"t": frame})
        for sel in (np.ones(n, bool), rng.random(n) < 0.3):
            scope = mbk.Scope.from_table(db["t"], ["p", "q", "r"], rows=sel)
            conds = [mb.condition(c, "<>", 0) for c in ("p", "q", "r")]
            masks = mb.Masks.from_scope(scope, conds)
            x = mb.variables(3)
            a = mb.Program((x[0] & x[1]) | x[2], 3)
            b = mb.Program((x[0] | x[2]) & (x[1] | x[2]), 3)            # distributivity: equal, also under UNKNOWN
            c = mb.Program((x[0] & x[1]) | (x[2] & x[0]), 3)            # differs on some rows
            for left, right, equal in ((a, b, True), (a, c, False)):
                cpu = mp.verify_programs(masks, left, right)
                gpu = mp.verify_programs(masks, left, right, backend="cuda")
                self.assertEqual(strip(cpu), strip(gpu))
                rows = np.flatnonzero(sel)

                def val(prog_id, i):
                    pp, qq, rr = bool(p[i]), bool(q[i]), r[i]
                    if prog_id == 0:
                        t1 = pp and qq
                        return True if (t1 or rr is True) else (None if rr is None else False)
                    t2 = pp and qq
                    t3 = (None if pp else False) if rr is None else (rr and pp)    # Kleene AND
                    if t2 or t3 is True:
                        return True
                    return None if t3 is None else False
                va = [val(0, i) for i in rows]
                vb = va if equal else [val(1, i) for i in rows]
                diff = [int(i) for i, s, t_ in zip(rows, va, vb) if s is not t_]
                self.assertEqual((cpu["mismatches"], cpu["witness"], cpu["equivalent"]),
                                 (len(diff), diff[0] if diff else None, not diff))
                self.assertEqual(cpu["original"], dict(true=va.count(True), false=va.count(False),
                                                       unknown=va.count(None)))

    def test_field002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(8106)
        db, cols, code, names = planted(rng, 2000, near=5)
        scope = mbk.Scope.from_table(db["t"], names + ["code"])
        specs = names + [("code", "code", 0)]

        def edge(fn, resource, need):
            fn(Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need))
        need = mp.discover_need(scope, specs, sample=64)
        cols_ = len(specs) + 1
        self.assertEqual((need["variables"], need["support"], need["iterations"], need["output"]),
                         (cols_, 64, cols_ + 1, cols_ + cols_ * (cols_ + 1) // 2))
        for resource in ("variables", "support", "iterations", "output", "bytes"):
            edge(lambda b: mp.discover(scope, specs, sample=64, budget=b), resource, need[resource])
        for resource in ("support", "output"):
            edge(lambda b: mp.discover(scope, specs, sample=64, backend="cuda", budget=b), resource, need[resource])
        found = mp.discover(scope, specs, sample=64)
        cert = next(r for r in found.relations if r["labels"] == ("b0", "b1", "d0"))["certificate"]
        x = mb.variables(3)
        conds = [mp.bit_condition(scope, "d0"), mb.condition("b2", "<>", 0), mb.condition("b3", "<>", 0)]
        rw = mp.substitute(scope, conds, (x[0] & x[1]) | x[2], cert, mode="all")
        nn = rw.info["nodes"]
        edge(lambda b: mp.substitute(scope, conds, (x[0] & x[1]) | x[2], cert, mode="all", budget=b), "circuit", nn)
        masks = mb.Masks.from_scope(scope, rw.conditions)
        for backend in ("cpu", "cuda"):
            vn = mp.verify_need(masks, rw.original, rw.program, backend=backend)
            self.assertEqual(vn["circuit"], len(rw.original.nodes) + nn)
            for resource in ("circuit", "bytes"):
                edge(lambda b: mp.verify_programs(masks, rw.original, rw.program, backend=backend, budget=b),
                     resource, vn[resource])

    def test_field002_malformed_input_and_limits(self):
        rng = np.random.default_rng(8107)
        db, cols, code, names = planted(rng, 500, near=5)
        scope = mbk.Scope.from_table(db["t"], names + ["code"])
        found = mp.discover(scope, names, sample=0)
        cert = found.relations[0]["certificate"]
        x = mb.variables(2)
        conds = [mb.condition("b0", "<>", 0), mb.condition("b1", "<>", 0)]
        with self.assertRaises(ValueError):
            mp.substitute(scope, conds, x[0] ^ x[1], cert, mode="sometimes")
        with self.assertRaises(ValueError):
            mp.discover(scope, names, nulls="propagate")
        with self.assertRaises(ValueError):
            mp.discover(scope, ["nope"])
        with self.assertRaises(TypeError):
            mp.substitute(scope, conds, x[0] ^ x[1], "a certificate")
        with self.assertRaises(UnsupportedError):                      # 64 conditions and an appended bit
            many = [mb.condition("code", ">", i) for i in range(64)]
            mp.substitute(scope, many, mb.var(0) ^ mb.var(63), cert, mode="all")
        three = mb.Program(mb.var(0) ^ mb.var(2), 3)
        with self.assertRaises(ValueError):                            # a program over another number of conditions
            mp.substitute(scope, conds, three, cert)
        masks = mb.Masks.from_scope(scope, conds)
        from torbles.math_domains import DomainError
        with self.assertRaises(DomainError):
            mp.verify_programs(masks, mb.Program(x[0], 2), three)

    def test_field002_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_parity.")}
        self.assertEqual(mine, {"tr_parity_spec", "tr_parity_relation", "tr_parity_found_info",
                                "tr_parity_rewrite_info", "tr_parity_check"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_parity.")}
        self.assertEqual(cuda, {"tr_parity_check"})


if __name__ == "__main__":
    unittest.main()
