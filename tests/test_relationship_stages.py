"""Whole-stage identities must execute real reductions and preserve bag semantics."""
import ctypes
import json
import random
import unittest

import duckdb
import numpy as np
import pandas as pd
import torbles
from torbles.algq import (A_BIL, A_COUNT, A_LIN, A_MM, A_POLYG, Col, Const, QPlan,
                         lower_measure)
from torbles.relationships import _lib, discover


class RelationshipStages(unittest.TestCase):
    def make(self, n=257, values=None, *, stage="compose", columns=None):
        if values is None:
            # Only d is eligible as a hidden key: q has a >16-value domain.
            values = dict(q=[100+i % 31 for i in range(n)], d=[i % 5 for i in range(n)],
                          flag=[i % 3 for i in range(n)], g=[i % 7 for i in range(n)])
        frame = pd.DataFrame({name: pd.Series(items, dtype=object) for name, items in values.items()})
        ddl = "CREATE TABLE t ("+",".join('"'+name+'" BIGINT' for name in values)+")"
        db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)
        cat = discover(db, "t", columns or list(values), modules=(), stage_strategy=stage,
                       max_rows=max(len(frame), 1_000_000))
        self.addCleanup(cat.close)
        return db, db["t"], cat, values, frame

    @staticmethod
    def c3(t):
        q, d = Col(t, "q"), Col(t, "d")
        u = (q+d)*(q-d)+d*d
        return u*d+u*d

    @staticmethod
    def c4(t):
        q, d = Col(t, "q"), Col(t, "d")
        return (q+d)*(q+d)*d

    @staticmethod
    def expected(values, kind="c3", *, keep=lambda i: True, group=None, both=False):
        out = {} if group else [0, 0]
        for i, (q, d) in enumerate(zip(values["q"], values["d"])):
            if not keep(i) or q is None or d is None:
                continue
            v = 2*q*q*d if kind == "c3" else (q+d)*(q+d)*d
            state = out.setdefault((values[group][i],), [0, 0]) if group else out
            state[0] += v
            state[1] += 1
        def finish(state):
            value = state[0] if state[1] else None
            return (value, state[1]) if both else value
        return {key: finish(state) for key, state in out.items()} if group else finish(out)

    def duck(self, frame, sql):
        con = duckdb.connect()
        try:
            con.register("t", frame)
            return con.execute(sql).fetchall()
        finally:
            con.close()

    def assert_stages(self, query, scans, *, hidden=None, engine="cpu"):
        info = query.stage_plan
        self.assertEqual(info["choice"], "stages", info)
        self.assertEqual(info["scan_count"], scans)
        self.assertEqual(len(info["scan_devices"]), scans)
        self.assertEqual(info["finalizer_device"], "cpu(native)")
        self.assertFalse(info["row_fallback"])
        self.assertGreater(info["workspace_bytes"], 0)
        self.assertGreaterEqual(info["candidate_count"], 1)
        self.assertIn(info["partition_kind"], (1, 2))
        self.assertGreater(info["input_degree"], info["residual_degree"])
        self.assertLessEqual(info["residual_degree"], 2)
        if hidden is not None:
            self.assertEqual(bool(info["hidden_key"]), hidden)
        if engine == "gpu":
            self.assertEqual(info["scan_devices"], ["gpu"]*scans)
            self.assertEqual(query.device, "gpu+cpu-finalize")
        else:
            self.assertTrue(all(device.startswith("cpu") for device in info["scan_devices"]), info)
            self.assertEqual(query.device, "cpu(stages)")

    def test_c3_hidden_refinement_runs_one_native_reduction(self):
        db, t, _, values, frame = self.make()
        query = db.query(t).sum(self.c3(t))
        want = self.expected(values)
        self.assertEqual(self.duck(frame, "SELECT sum(2*CAST(q AS BIGINT)*q*d) FROM t"), [(want,)])
        self.assertEqual(query.run(), want)
        self.assert_stages(query, 1, hidden=True)
        self.assertEqual(query.stage_plan["key_col"], t.idx("d"))
        self.assertLessEqual(query.stage_plan["key_domain"], 16)

    def test_c4_compares_whole_stage_partitions_and_reuses_counts(self):
        db, t, _, values, frame = self.make(513)
        query = db.query(t).where(t.flag == 1).group_by(t.d).sum(self.c4(t))
        query._both = True
        want = self.expected(values, "c4", keep=lambda i: values["flag"][i] == 1, group="d", both=True)
        sql = "SELECT d,sum((CAST(q AS BIGINT)+d)*(q+d)*d),count(*) FROM t WHERE flag=1 GROUP BY d ORDER BY d"
        self.assertEqual(self.duck(frame, sql), [(key[0], *want[key]) for key in sorted(want)])
        self.assertEqual(query.run(), want)
        scans = query.stage_plan["scan_count"]
        self.assertIn(scans, (1, 2))
        self.assert_stages(query, scans, hidden=False)
        self.assertGreaterEqual(query.stage_plan["candidate_count"], 2)
        self.assertEqual(scans, 1 if query.stage_plan["partition_kind"] == 2 else 2)
        self.assertLessEqual(query.stage_plan["key_domain"], 64)

    def test_nondivisible_constant_keeps_the_two_residual_candidate(self):
        db, t, _, values, frame = self.make(129)
        query = db.query(t).group_by(t.d).sum(self.c4(t)+1)
        want = self.expected(values, "c4", group="d", both=True)
        want = {key: total+count for key, (total, count) in want.items()}
        sql = "SELECT d,sum((CAST(q AS BIGINT)+d)*(q+d)*d+1) FROM t GROUP BY d ORDER BY d"
        self.assertEqual(self.duck(frame, sql), [(key[0], want[key]) for key in sorted(want)])
        for engine in ("alg", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(query.run(engine), want)
                self.assert_stages(query, 2, hidden=False, engine="gpu" if engine == "gpu" else "cpu")
                self.assertEqual(query.stage_plan["partition_kind"], 1)

    def test_negative_key_offset_common_factor_keeps_residual_constant(self):
        values = dict(q=[100+i % 31 for i in range(129)], d=[-2+i % 5 for i in range(129)])
        db, t, _, _, frame = self.make(values=values)
        query = db.query(t).group_by(t.d).sum(self.c4(t))
        want = self.expected(values, "c4", group="d")
        sql = "SELECT d,sum((CAST(q AS BIGINT)+d)*(q+d)*d) FROM t GROUP BY d ORDER BY d"
        self.assertEqual(self.duck(frame, sql), [(key[0], want[key]) for key in sorted(want)])
        self.assertEqual(query.run("gpu"), want)
        self.assert_stages(query, 1, hidden=False, engine="gpu")
        self.assertEqual(query.stage_plan["partition_kind"], 2)
        self.assertGreaterEqual(query.stage_plan["candidate_count"], 2)
        scan = _lib.tr_stage_scan(query.last_plan._stage_handle.handle, 0).contents
        self.assertEqual(scan.p_const, 98*98)

    def test_gpu_scans_and_cpu_finalization_are_reported_separately(self):
        db, t, _, values, _ = self.make()
        queries = [(db.query(t).sum(self.c3(t)), self.expected(values), 1, True),
                   (db.query(t).group_by(t.d).sum(self.c4(t)), self.expected(values, "c4", group="d"), None, False)]
        for query, want, scans, hidden in queries:
            with self.subTest(scans=scans):
                self.assertEqual(query.run("gpu"), want)
                scans = scans or query.stage_plan["scan_count"]
                self.assert_stages(query, scans, hidden=hidden, engine="gpu")

    def test_signed_codec_offsets_and_equivalent_spelling_are_composed(self):
        values = dict(q=[-27+i % 31 for i in range(257)], d=[-2+i % 5 for i in range(257)],
                      flag=[i % 2 for i in range(257)])
        db, t, _, _, frame = self.make(values=values)
        # Compact spelling stays below the 48-instruction source budget even
        # when both operands have affine offsets. It is not the C3 AST pattern.
        query = db.query(t).sum(2*t.q*t.q*t.d)
        want = self.expected(values)
        self.assertEqual(self.duck(frame, "SELECT sum(2*CAST(q AS BIGINT)*q*d) FROM t"), [(want,)])
        for engine in ("alg", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(query.run(engine), want)
                self.assert_stages(query, 1, hidden=True, engine="gpu" if engine == "gpu" else "cpu")

    def test_duplicate_rows_contribute_with_original_multiplicity(self):
        original = dict(q=[101, 102, 103, 104], d=[0, 1, 2, 3])
        repeated = {name: values*7 for name, values in original.items()}
        db, t, _, values, _ = self.make(values=repeated)
        query = db.query(t).sum(self.c3(t))
        query._both = True
        want = self.expected(values, both=True)
        self.assertEqual(query.run(), want)
        self.assertEqual(want, (7*self.expected(original), 28))
        self.assert_stages(query, 1, hidden=True)

    def test_zero_measure_is_zero_with_positive_seen_count(self):
        values = dict(q=[100+i % 31 for i in range(129)], d=[0]*129)
        db, t, _, _, _ = self.make(values=values)
        query = db.query(t).sum(self.c3(t))
        query._both = True
        self.assertEqual(query.run(), (0, 129))
        self.assertEqual(query.run("gpu"), (0, 129))

    def test_empty_filter_and_empty_table_have_no_phantom_groups(self):
        db, t, _, _, _ = self.make()
        scalar = db.query(t).where(t.q < -999).sum(self.c3(t))
        scalar._both = True
        grouped = db.query(t).where(t.q < -999).group_by(t.d).sum(self.c4(t))
        for engine in ("alg", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(scalar.run(engine), (None, 0))
                self.assertEqual(grouped.run(engine), {})
        empty_db, empty_t, _, _, _ = self.make(values=dict(q=[], d=[]))
        self.assertIsNone(empty_db.query(empty_t).sum(2*empty_t.q*empty_t.q*empty_t.d).run())

    def test_stage_chain_and_auto_switches_restore_the_original_query(self):
        db, t, cat, values, _ = self.make()
        query = db.query(t).sum(self.c3(t))
        want = self.expected(values)
        for strategy in ("compose", "off", "auto", "compose", "off", "compose"):
            with self.subTest(strategy=strategy):
                cat.stage_strategy = strategy
                cat.system_strategy = "chain"
                self.assertEqual(query.run(), want)
                if strategy == "compose":
                    self.assert_stages(query, 1, hidden=True)
                elif strategy == "off":
                    self.assertNotEqual(query.stage_plan["choice"], "stages")
                    self.assertEqual(query.system_plan["choice"], "chain")
        self.assertEqual(query.stage_plan["cache_hit"], 1)

    def test_warm_cache_scope_change_and_revision_do_not_reuse_wrong_results(self):
        db, t, _, values, _ = self.make()
        query = db.query(t).sum(self.c3(t))
        self.assertEqual(query.run(), self.expected(values))
        self.assertEqual(query.stage_plan["cache_hit"], 0)
        self.assertEqual(query.run(), self.expected(values))
        self.assertEqual(query.stage_plan["cache_hit"], 1)
        query.where(t.flag == 1)
        want = self.expected(values, keep=lambda i: values["flag"][i] == 1)
        self.assertEqual(query.run(), want)
        self.assertEqual(query.stage_plan["cache_hit"], 0)
        t.flip(0, 0)
        t.repair()
        self.assertEqual(query.run(), want)
        self.assertEqual(query.stage_plan.get("cache_hit", 0), 0)

    def test_nullable_canceled_dependency_refuses_staging(self):
        values = dict(q=[100+i % 31 for i in range(129)], d=[i % 5 for i in range(129)],
                      r=[None if i % 11 == 0 else i % 7 for i in range(129)])
        db, t, _, _, _ = self.make(values=values)
        valid = Col(t, t.nullable["r"]) == 1
        u = (t.q+t.r)*(t.q-t.r)+t.r*t.r
        query = db.query(t).where(valid).sum(u*t.d+u*t.d)
        self.assertEqual(query.run(), self.expected(values, keep=lambda i: values["r"][i] is not None))
        self.assertNotEqual(query.stage_plan["choice"], "stages")
        self.assertIn("nullable", str(query.stage_plan["reason"]))

    def test_sql_null_groups_keep_presence_and_null_sum(self):
        values = dict(q=[None, None, 101, 102, 101, 103], d=[1, 1, 2, 2, 2, None],
                      g=[0, 0, 1, 1, 1, None])
        db, _, _, _, frame = self.make(values=values)
        sql = "SELECT g,sum(2*CAST(q AS BIGINT)*q*d),count(*) FROM t GROUP BY g ORDER BY g NULLS FIRST"
        want = self.duck(frame, sql)
        self.assertEqual(want, [(None, None, 1), (0, None, 2), (1, 123224, 3)])
        for engine in ("alg", "plain", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(db.sql(sql, engine).rows, want)

    def test_high_cardinality_hidden_keys_and_sparse_outputs_are_preserved(self):
        values = dict(q=[100+i % 31 for i in range(513)], d=[i % 17 for i in range(513)],
                      g=[i*65537 for i in range(513)])
        db, t, _, _, _ = self.make(values=values)
        scalar = db.query(t).sum(self.c3(t))
        self.assertEqual(scalar.run("gpu"), self.expected(values))
        self.assertNotEqual(scalar.stage_plan["choice"], "stages")
        self.assertTrue(scalar.device.startswith("cpu"), scalar.device)
        grouped = db.query(t).group_by(t.g).sum(self.c4(t))
        self.assertEqual(grouped.run(), self.expected(values, "c4", group="g"))
        self.assertNotEqual(grouped.stage_plan["choice"], "stages")
        self.assertTrue(grouped.last_plan.out_key)
        self.assertEqual(grouped.last_plan.out_n, 513)

    def test_mutable_external_scopes_bypass_native_stage_preparation(self):
        db, t, cat, _, _ = self.make(stage="off")
        query = db.query(t).sum(self.c3(t))
        cat.system_strategy = "chain"
        query.run()
        source_plan = QPlan.from_buffer_copy(bytes(query.last_plan))
        source = cat._capture_system_source(source_plan)
        cat.stage_strategy = "compose"
        bitmap = np.ones((t.n+63)//64, dtype=np.uint64)
        ranges = np.array([0, 1], dtype=np.uint64)
        for scope in ("xrows", "ranges", "semi", "wide", "attribute"):
            with self.subTest(scope=scope):
                plan = QPlan.from_buffer_copy(bytes(source_plan))
                if scope == "xrows": plan.xmode, plan.xrows = 1, bitmap.ctypes.data
                elif scope == "ranges": plan.nranges, plan.ranges = 1, ranges.ctypes.data
                elif scope == "semi": plan.nsemi, plan.semi_bm[0] = 1, bitmap.ctypes.data
                elif scope == "wide": plan.wide = 1
                else: plan.nattr = 1
                selected, info = cat._prepare_stage(plan, "cpu", source, cat._stage_handle())
                self.assertFalse(selected, info)
                self.assertEqual(info["choice"], "preserve")

    def test_non_general_source_aggregates_are_not_reinterpreted_as_sum(self):
        db, t, cat, _, _ = self.make(stage="off")
        cat.system_strategy = "chain"
        query = db.query(t).sum(self.c3(t))
        query.run()
        original = QPlan.from_buffer_copy(bytes(query.last_plan))
        cat.stage_strategy = "compose"
        for aggregate in (A_COUNT, A_LIN, A_BIL, A_POLYG, A_MM):
            with self.subTest(aggregate=aggregate):
                plan = QPlan.from_buffer_copy(bytes(original))
                plan.agg = aggregate
                source = cat._capture_system_source(plan)
                selected, info = cat._prepare_stage(plan, "cpu", source, cat._stage_handle())
                self.assertFalse(selected, info)
                self.assertEqual(info["choice"], "preserve")
                self.assertIn("unsupported", info["reason"])
                self.assertEqual(plan.agg, aggregate)
                self.assertEqual(plan.total_sum, original.total_sum)

    def test_semijoin_fallback_keeps_orphan_and_null_foreign_keys_out(self):
        n = 129
        values = dict(q=[100+i % 31 for i in range(n)], d=[i % 5 for i in range(n)],
                      fk=[[0, 1, 2, 99, None][i % 5] for i in range(n)])
        fact = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in values.items()})
        dimension = pd.DataFrame(dict(id=[0, 1, 2], active=[0, 1, 1]))
        db = torbles.from_ddl(
            "CREATE TABLE dim (id BIGINT PRIMARY KEY, active BIGINT); "
            "CREATE TABLE t (q BIGINT,d BIGINT,fk BIGINT REFERENCES dim(id))",
            {"dim": dimension, "t": fact}, keep_plain=True)
        t, dim = db["t"], db["dim"]
        cat = discover(db, "t", ["q", "d"], modules=(), stage_strategy="compose")
        self.addCleanup(cat.close)
        query = db.query(t).join(dim).where(dim.active == 1).sum(self.c3(t))
        query.join_mode = "lookup"
        want = self.expected(values, keep=lambda i: values["fk"][i] in (1, 2))
        con = duckdb.connect()
        try:
            con.register("t", fact)
            con.register("dim", dimension)
            self.assertEqual(con.execute("SELECT sum(2*CAST(t.q AS BIGINT)*t.q*t.d) "
                                         "FROM t JOIN dim ON t.fk=dim.id WHERE dim.active=1").fetchall(), [(want,)])
        finally:
            con.close()
        for engine in ("alg", "gpu", "plain"):
            with self.subTest(engine=engine):
                self.assertEqual(query.run(engine), want)
                self.assertNotEqual(query.stage_plan["choice"], "stages")

    def test_missing_column_metadata_preserves_ordinary_execution(self):
        db, t, cat, values, _ = self.make(columns=["q"])
        query = db.query(t).sum(self.c3(t))
        self.assertEqual(query.run(), self.expected(values))
        self.assertNotEqual(query.stage_plan["choice"], "stages")

    def test_closing_selected_catalog_clears_stage_trace_and_keeps_execution(self):
        db, t, cat, values, _ = self.make()
        query = db.query(t).sum(self.c3(t))
        self.assertEqual(query.run(), self.expected(values))
        self.assert_stages(query, 1, hidden=True)
        cat.close()
        self.assertEqual(query.run(), self.expected(values))
        self.assertNotEqual(query.stage_plan["choice"], "stages")

    def test_plain_backend_preserves_original_and_result_width_guards_remain(self):
        db, t, _, values, _ = self.make()
        query = db.query(t).sum(self.c3(t))
        self.assertEqual(query.run("plain"), self.expected(values))
        self.assertNotEqual(query.stage_plan["choice"], "stages")
        huge = dict(q=[(1 << 62)+i % 31 for i in range(33)], d=[i % 5 for i in range(33)])
        huge_db, huge_t, _, _, _ = self.make(values=huge)
        self.assertGreater(self.expected(huge), (1 << 127)-1)
        guarded = huge_db.query(huge_t).sum(2*huge_t.q*huge_t.q*huge_t.d)
        guarded._nobridge = True
        with self.assertRaises(OverflowError):
            guarded.run()
        # The existing exact Python histogram bridge remains a legitimate
        # fallback beyond the native width bound; staged int64 must not wrap it.
        bridged = huge_db.query(huge_t).sum(2*huge_t.q*huge_t.q*huge_t.d)
        self.assertEqual(bridged.run(), self.expected(huge))

    def test_native_finalizer_rejects_overflow_instead_of_wrapping(self):
        db, t, cat, values, _ = self.make(stage="off")
        query = db.query(t).sum(self.c3(t))
        query.run()
        plan = QPlan.from_buffer_copy(bytes(query.last_plan))
        source = cat._capture_system_source(plan)
        owner = cat._stage_handle()
        cat.stage_strategy = "compose"
        selected, info = cat._prepare_stage(plan, "cpu", source, owner)
        self.assertTrue(selected, info)
        self.assertEqual(info["key_col"], t.idx("d"))
        self.assertEqual(_lib.tr_stage_reset(owner.handle), 0)
        scan = _lib.tr_stage_scan(owner.handle, 0).contents
        sums = ctypes.cast(scan.out_sum, ctypes.POINTER(ctypes.c_int64))
        counts = ctypes.cast(scan.out_cnt, ctypes.POINTER(ctypes.c_uint64))
        # Inject a boundary aggregate state rather than constructing billions of
        # rows. d=2 multiplies this residual beyond int64 in the native fold.
        sums[2], counts[2] = (1 << 63)-1, 1
        self.assertEqual(_lib.tr_stage_finish(owner.handle, ctypes.byref(plan)), -3)
        self.assertEqual(query.run(), self.expected(values))

    def test_bad_stage_strategy_is_rejected_at_the_api_boundary(self):
        _, _, cat, _, _ = self.make()
        with self.assertRaises(ValueError): cat.stage_strategy = "fuse_everything"

    def test_seeded_generic_polynomial_stage_composition(self):
        rng = random.Random(20261030)
        fixtures = []
        for qoff, yoff, doff, domain, rows in ((0, 0, 0, 5, 65), (100, 50, 0, 5, 65),
                                                (-20, -11, -2, 5, 65), (0, 0, 0, 17, 65),
                                                (0, 0, 0, 5, 0)):
            values = dict(q=[qoff+i % 31 for i in range(rows)],
                          y=[yoff+(i*5) % 23 for i in range(rows)],
                          d=[doff+i % domain for i in range(rows)], flag=[i % 3 for i in range(rows)])
            db, table, cat, _, frame = self.make(values=values)
            cat.system_strategy = "chain"
            oracle = duckdb.connect()
            oracle.register("t", frame)
            self.addCleanup(oracle.close)
            fixtures.append((db, table, cat, values, oracle))
        report = dict(seed=20261030, attempted=300, executed=0, oversized_source_budget=0,
                      accepted=0, refused=0, accepted_scan_counts={}, count_only=0,
                      accepted_degree_four=0, empty_cases=0, duckdb_checked=0, gpu_checked=0,
                      accepted_partition_counts={}, gpu_partition_counts={},
                      common_nonzero_constant_executions=0, refused_reasons={})

        def size(node):
            return 1+size(node.l)+size(node.r) if hasattr(node, "l") else 1

        def power(node, exponent):
            result = Const(1)
            for _ in range(exponent): result = result*node
            return result

        for case in range(report["attempted"]):
            fixture = fixtures[3] if case % 31 == 0 else fixtures[4] if case % 47 == 0 else fixtures[case % 3]
            db, table, cat, values, oracle = fixture
            grouped = case % 2 == 0 and case % 31 != 0
            both = case % 13 == 0
            family = case % 5
            a, b = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(2))
            c = rng.choice((-2, -1, 0, 1, 2))
            p, k = rng.choice((1, 2)), rng.randrange(5)
            x, y, d = table.q, table.y, table.d
            rx, ry = x*x, y*y
            if family == 0:
                expression = a*power(d, p)*rx+c*power(d, k)
                evaluate = lambda q,y,d,a=a,p=p,c=c,k=k: a*d**p*q*q+c*d**k
                text = f"({a})*power_d_{p}*q*q+({c})*power_d_{k}"
            elif family == 1:
                # Proportional key coefficient vectors should share a scan.
                expression = a*power(d, p)*rx+b*power(d, p)*ry+c*power(d, k)
                evaluate = lambda q,y,d,a=a,b=b,p=p,c=c,k=k: a*d**p*q*q+b*d**p*y*y+c*d**k
                text = f"({a})*power_d_{p}*q*q+({b})*power_d_{p}*y*y+({c})*power_d_{k}"
            elif family == 2:
                expression = a*d*rx+b*d*d*(x*y)+c*power(d, k)
                evaluate = lambda q,y,d,a=a,b=b,c=c,k=k: a*d*q*q+b*d*d*q*y+c*d**k
                text = f"({a})*d*q*q+({b})*d*d*q*y+({c})*power_d_{k}"
            elif family == 3:
                expression = (x+d)*(x+d)*d+c*power(d, k)
                evaluate = lambda q,y,d,c=c,k=k: (q+d)*(q+d)*d+c*d**k
                text = f"(q+d)*(q+d)*d+({c})*power_d_{k}"
            else:
                expression = a*power(d, 3)+b*power(d, 4)+c*d
                evaluate = lambda q,y,d,a=a,b=b,c=c: a*d**3+b*d**4+c*d
                text = f"({a})*power_d_3+({b})*power_d_4+({c})*d"
            if case % 37 == 0:
                expression = 0*x*d*d
                evaluate = lambda q,y,d: 0
                text = "0*q*d*d"
            for exponent in range(5):
                sql_power = "1" if not exponent else "("+"*".join(["d"]*exponent)+")"
                text = text.replace(f"power_d_{exponent}", sql_power)
            lowered, _ = lower_measure(expression, table)
            if size(lowered) > 48:
                report["oversized_source_budget"] += 1
                continue
            report["executed"] += 1
            empty = case % 19 == 0
            filtered = not empty and case % 7 == 0
            keep = (lambda i: False) if empty else (lambda i: values["flag"][i] == 1) if filtered else (lambda i: True)
            predicate = (table.q < -1_000_000) if empty else (table.flag == 1) if filtered else None
            states = {} if grouped else {(): [0, 0]}
            for i, (qv, yv, dv) in enumerate(zip(values["q"], values["y"], values["d"])):
                if not keep(i): continue
                state = states.setdefault((dv,) if grouped else (), [0, 0])
                state[0] += evaluate(qv, yv, dv)
                state[1] += 1
            def finish(state):
                value = state[0] if state[1] else None
                return (value, state[1]) if both else value
            want = {key: finish(state) for key, state in states.items()} if grouped else finish(states[()])
            if not any(state[1] for state in states.values()): report["empty_cases"] += 1

            def query():
                result = db.query(table)
                if predicate is not None: result = result.where(predicate)
                if grouped: result = result.group_by(table.d)
                result = result.sum(expression)
                result._both = both
                # Exercise the native stage compiler, including count-only
                # folds, rather than the earlier Python one-residual factoring.
                # The legacy control receives this same controlled shape.
                result._factored = True
                return result

            cat.stage_strategy = "compose"
            composed = query()
            with self.subTest(case=case, family=family, grouped=grouped, both=both):
                self.assertEqual(composed.run(), want)
                info = dict(composed.stage_plan)
                if info["choice"] == "stages":
                    report["accepted"] += 1
                    scans = info["scan_count"]
                    report["accepted_scan_counts"][str(scans)] = report["accepted_scan_counts"].get(str(scans), 0)+1
                    partition = str(info["partition_kind"])
                    report["accepted_partition_counts"][partition] = report["accepted_partition_counts"].get(partition, 0)+1
                    if info["partition_kind"] == 2:
                        scan = _lib.tr_stage_scan(composed.last_plan._stage_handle.handle, 0).contents
                        report["common_nonzero_constant_executions"] += scan.p_const != 0
                    report["count_only"] += info["residual_degree"] == 0
                    report["accepted_degree_four"] += info["input_degree"] == 4
                    self.assertGreaterEqual(scans, 1)
                    self.assertLessEqual(scans, 4)
                    self.assertEqual(info["finalizer_device"], "cpu(native)")
                    if case % 23 == 0 and values["q"]:
                        self.assertEqual(composed.run("gpu"), want)
                        gpu_info = composed.stage_plan
                        self.assert_stages(composed, gpu_info["scan_count"], engine="gpu")
                        partition = str(gpu_info["partition_kind"])
                        report["gpu_partition_counts"][partition] = report["gpu_partition_counts"].get(partition, 0)+1
                        if gpu_info["partition_kind"] == 2:
                            scan = _lib.tr_stage_scan(composed.last_plan._stage_handle.handle, 0).contents
                            report["common_nonzero_constant_executions"] += scan.p_const != 0
                        report["gpu_checked"] += 1
                else:
                    report["refused"] += 1
                    reason = str(info["reason"])
                    report["refused_reasons"][reason] = report["refused_reasons"].get(reason, 0)+1
                cat.stage_strategy = "off"
                self.assertEqual(query().run(), want)
                if case % 23 == 0 or (family == 4 and case < 15):
                    where = " WHERE q < -1000000" if empty else " WHERE flag=1" if filtered else ""
                    sql = f"SELECT {'d,' if grouped else ''}sum({text})"+(",count(*)" if both else "")+" FROM t"+where
                    if grouped: sql += " GROUP BY d"
                    rows = oracle.execute(sql).fetchall()
                    actual = {tuple(row[:1]): (row[1],row[2]) if both else row[1] for row in rows} if grouped else tuple(rows[0]) if both else rows[0][0]
                    self.assertEqual(actual, want)
                    report["duckdb_checked"] += 1
        self.assertEqual(report["executed"]+report["oversized_source_budget"], report["attempted"])
        self.assertEqual(report["accepted"]+report["refused"], report["executed"])
        self.assertGreaterEqual(report["accepted"], 120)
        self.assertGreaterEqual(report["accepted_scan_counts"].get("1", 0), 30)
        self.assertGreaterEqual(report["accepted_scan_counts"].get("2", 0), 15)
        self.assertGreaterEqual(report["count_only"], 20)
        self.assertGreaterEqual(report["accepted_degree_four"], 30)
        self.assertGreater(report["common_nonzero_constant_executions"], 0)
        self.assertGreater(report["refused"], 0)
        print("STAGE_PROPERTY_COUNTS "+json.dumps(report,sort_keys=True),flush=True)


if __name__ == "__main__": unittest.main()
