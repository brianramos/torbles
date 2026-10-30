"""Opt-in native relationship discovery and query optimization prototype.

Candidate generation, exact verification, equation composition and cost search run
in C. This module marshals handles/codec offsets and renders native evidence.
"""
import ctypes as C
import functools
import threading
from collections import OrderedDict
from . import algq

MAX_TERMS, MAX_STEPS = 32, 16
MODULES = {"affine": 1, "product": 2, "sum": 4}
NAMES = {v: k for k, v in MODULES.items()}


class Column(C.Structure):
    _fields_ = [("col", C.c_int32), ("valid", C.c_int32), ("offset", C.c_int64)]


class Term(C.Structure):
    _fields_ = [("a", C.c_int32), ("b", C.c_int32), ("coefficient", C.c_int64)]


class Polynomial(C.Structure):
    _fields_ = [("n", C.c_int32), ("terms", Term * MAX_TERMS)]


class Rule(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("module", "out", "left", "right", "usable")] + [
        ("scale", C.c_int64), ("bias", C.c_int64)] + [
        (n, C.c_uint64) for n in ("checked", "excluded", "violations", "witness")] + [("rhs", Polynomial)]


class Trace(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("changed", "steps", "explored", "truncated")] + [
        ("before_cost", C.c_double), ("after_cost", C.c_double),
        ("rule", C.c_int32 * MAX_STEPS), ("direction", C.c_int32 * MAX_STEPS)]


class Stats(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("rows", "used_rows", "decoded_bytes")] + [
        (n, C.c_int32) for n in ("candidates", "rules", "truncated", "refused")]


class SystemTrace(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("choice", "original_agg", "selected_agg", "reason",
                 "sample_words", "nonempty_words", "chain_ops", "stack_peak")] + [
        (n, C.c_uint64) for n in ("sampled_rows", "passing_rows")] + [
        (n, C.c_double) for n in ("groups_per_word", "chain_cost", "combine_cost", "word_combine_cost", "matrix_combine_cost")] + [
        ("cache_hit", C.c_int32), ("gpu_profile_skipped", C.c_int32)] + [
        (n, C.c_int32) for n in ("dag_nodes", "hybrid_ops", "hybrid_registers", "hybrid_regions")] + [
        ("hybrid_cost", C.c_double)]


def _locked(fn):
    @functools.wraps(fn)
    def call(self, *args, **kw):
        with self._lock:
            return fn(self, *args, **kw)
    return call


_lib = algq.lib
for _name, _restype, _args in (
    ("tr_discover", C.c_void_p, [C.c_void_p, C.POINTER(Column), C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.POINTER(C.c_int32)]),
    ("tr_free", None, [C.c_void_p]),
    ("tr_rule_count", C.c_int32, [C.c_void_p]),
    ("tr_get_rule", C.c_int32, [C.c_void_p, C.c_int32, C.POINTER(Rule)]),
    ("tr_stats", C.c_int32, [C.c_void_p, C.POINTER(Stats)]),
    ("tr_scope_valid", C.c_int32, [C.c_void_p, C.c_void_p, C.c_void_p]),
    ("tr_probe", C.c_int32, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int64, C.c_int64, C.POINTER(Rule)]),
    ("tr_optimize", C.c_int32, [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(Polynomial), C.POINTER(Polynomial), C.POINTER(Trace)]),
    ("tr_plan_system", C.c_int32, [C.c_void_p, C.POINTER(algq.QPlan), C.c_int32, C.c_int32, C.POINTER(SystemTrace)]),
    ("tr_plan_system_cached", C.c_int32, [C.c_void_p, C.c_void_p, C.POINTER(algq.QPlan), C.c_int32, C.c_int32, C.POINTER(SystemTrace)]),
    ("tr_system_source_create", C.c_void_p, [C.POINTER(algq.QPlan)]),
    ("tr_system_source_free", None, [C.c_void_p]),
    ("tr_plan_system_from_source", C.c_int32, [C.c_void_p, C.c_void_p, C.POINTER(algq.QPlan), C.c_void_p, C.c_int32, C.c_int32, C.POINTER(SystemTrace)]),
):
    _fn = getattr(_lib, _name)
    _fn.restype, _fn.argtypes = _restype, _args


def _check(rc):
    if rc < 0:
        raise ValueError({-1: "unsupported relationship input", -2: "relationship allocation failed",
                          -3: "relationship arithmetic exceeds exact native limits",
                          -4: "relationship catalog is stale; rediscover after mutation"}.get(rc, f"relationship error {rc}"))


def _i64(n):
    original = n
    n = int(n)
    if original != n:
        raise ValueError("prototype relationship parameters must be exact integers")
    if not -(1 << 63) <= n < (1 << 63):
        raise ValueError("prototype relationship coefficients and offsets must fit int64")
    return n


class _SystemSource:
    """Own immutable native measure IR for one compiled query plan."""
    def __init__(self, plan):
        self.handle = _lib.tr_system_source_create(C.byref(plan))
        if not self.handle:
            raise MemoryError("native source program unavailable")

    def __del__(self):
        if getattr(self, "handle", None):
            _lib.tr_system_source_free(self.handle)
            self.handle = None


class Catalog:
    """Table-scoped native equations. close() detaches and frees native buffers."""
    def __init__(self, table, columns=None, *, modules=("affine", "product", "sum"),
                 max_candidates=256, max_rows=1_000_000, attach=True, system_strategy="auto"):
        self.table, self._handle = table, None
        self._lock = threading.RLock()
        self._rewrite_cache = OrderedDict()
        self._proof_generation = 0
        self.system_strategy = system_strategy
        if table.n > max_rows:
            raise ValueError(f"table has {table.n} rows; prototype max_rows={max_rows}")
        if max_candidates < 0 or max_candidates > 100_000:
            raise ValueError("max_candidates must be between 0 and 100000")
        def supported(name):
            cd = table.codecs.get(name)
            return (cd is not None and cd.kind == "int" and name not in getattr(table, "wide", {})
                    and table.width(name) <= 32 and "#" not in name)
        self.columns = tuple(columns) if columns is not None else tuple(c for c in table.codecs if supported(c))
        if not self.columns or len(self.columns) > 12 or len(set(self.columns)) != len(self.columns):
            raise ValueError("choose 1 to 12 distinct integer columns")
        for name in self.columns:
            if name not in table.cols or not supported(name):
                raise ValueError(f"{name!r}: prototype supports integer columns with <=32-bit stored codes only")
        self.offsets = {c: _i64(table.codecs[c].offset) for c in self.columns}
        desc = (Column * len(self.columns))(*(Column(table.idx(c), table.idx(table.nullable[c])
                if c in table.nullable else -1, self.offsets[c]) for c in self.columns))
        mask = 0
        for m in modules:
            if m not in MODULES: raise ValueError(f"unknown relationship module {m!r}")
            mask |= MODULES[m]
        error = C.c_int32()
        self._handle = _lib.tr_discover(table.at, desc, len(desc), table.used_bm.ctypes.data,
                                         mask, max_candidates, C.byref(error))
        _check(error.value)
        if not self._handle: raise MemoryError("native relationship catalog unavailable")
        if attach:
            previous = getattr(table, "_relationships", None)
            if previous is not None: previous.close()
            table._relationships = self

    @_locked
    def close(self):
        if getattr(self.table, "_relationships", None) is self: del self.table._relationships
        if self._handle:
            _lib.tr_free(self._handle)
            self._handle = None
        self._rewrite_cache.clear()

    def __del__(self):
        if getattr(self, "_handle", None): _lib.tr_free(self._handle)

    def _open(self):
        if not self._handle: raise ValueError("relationship catalog is closed")

    @property
    def system_strategy(self):
        return self._system_strategy

    @system_strategy.setter
    def system_strategy(self, value):
        if value not in ("auto", "chain", "combine", "hybrid"):
            raise ValueError("system_strategy must be auto, chain, combine, or hybrid")
        with self._lock: self._system_strategy = value

    @_locked
    def _plan_system(self, plan, backend, source=None):
        """Ask C to choose a physical execution form, not a rewrite ordering."""
        if not self._handle: return {"choice": "preserve", "reason": "closed"}
        trace = SystemTrace()
        target = {"gpu": 1, "plain": 2}.get(backend, 0)
        mode = {"auto": 0, "chain": 1, "combine": 2, "hybrid": 3}[self._system_strategy]
        if source is None:
            rc = _lib.tr_plan_system_cached(self._handle, self.table.at, C.byref(plan), target, mode, C.byref(trace))
        else:
            rc = _lib.tr_plan_system_from_source(self._handle, self.table.at, C.byref(plan), source.handle, target, mode, C.byref(trace))
        _check(rc)
        reasons = {0: "cost", 1: "forced", 2: "unsupported", 3: "GPU capability",
                   4: "wide arithmetic", 5: "plain baseline", 6: "empty sample", 7: "invalid request"}
        info = {name: getattr(trace, name) for name, _ in SystemTrace._fields_}
        info.update(choice={0: "preserve", 1: "chain", 2: "combine", 3: "hybrid"}.get(trace.choice, "preserve"),
                    reason=reasons.get(trace.reason, "unsupported"), backend=backend,
                    strategy=self._system_strategy, cost_unit="relative sampled work, not elapsed time",
                    profile_ran=bool(trace.sample_words and not trace.cache_hit),
                    intermediate_storage="native evaluation stack" if trace.choice == 1 else
                        "combined plane algebra; no row intermediate table" if trace.choice == 2 else
                        "native scalar DAG registers" if trace.choice == 3 else "existing plan")
        return info

    @staticmethod
    def _capture_system_source(plan):
        return _SystemSource(plan)

    def _describe(self, r):
        t = self.table
        out, left = t.cols[r.out], t.cols[r.left]
        right = t.cols[r.right] if r.right >= 0 else None
        if r.module == MODULES["affine"]: expr = f"{out} = {r.scale}*{left} + {r.bias}"
        else: expr = f"{out} = {left} {'*' if r.module == MODULES['product'] else '+'} {right}"
        d = dict(module=NAMES[r.module], expression=expr, checked=int(r.checked), excluded=int(r.excluded),
                 violations=int(r.violations), usable=bool(r.usable), scope="observed table snapshot")
        if r.violations and r.witness < t.n:
            names = dict.fromkeys([out, left] + ([right] if right else []))
            d["witness"] = {"stored_row": int(r.witness), "values": {
                c: int(t.get_rows(c, [r.witness])[0]) + self.offsets[c] for c in names}}
        return d

    @property
    @_locked
    def rules(self):
        self._open()
        result = []
        count = _lib.tr_rule_count(self._handle)
        _check(count)
        for i in range(count):
            r = Rule(); _check(_lib.tr_get_rule(self._handle, i, C.byref(r)))
            result.append(self._describe(r))
        return result

    @property
    @_locked
    def stats(self):
        self._open()
        value = Stats()
        _check(_lib.tr_stats(self._handle, C.byref(value)))
        return {name: getattr(value, name) for name, _ in Stats._fields_}

    @_locked
    def probe(self, module, out, left, right=None, *, scale=1, bias=0):
        """Verify an explicit hypothesis natively, returning its first counterexample.

        A completely verified non-degenerate equation is registered automatically.
        Partial/empty/violated evidence cannot be used by the optimizer.
        """
        self._open()
        if module not in MODULES: raise ValueError(f"unknown relationship module {module!r}")
        for c in [out, left] + ([right] if right is not None else []):
            if c not in self.columns: raise ValueError(f"{c!r} is not a catalog column")
        if module != "affine" and right is None: raise ValueError("product/sum need right=")
        r = Rule()
        old_count = _lib.tr_rule_count(self._handle)
        _check(_lib.tr_probe(self._handle, MODULES[module], self.table.idx(out), self.table.idx(left),
               self.table.idx(right) if right is not None else -1, _i64(scale), _i64(bias), C.byref(r)))
        if _lib.tr_rule_count(self._handle) != old_count:
            self._proof_generation += 1
            self._rewrite_cache.clear()
        return self._describe(r)

    @_locked
    def _rewrite_codes(self, measure):
        """Marshal lowered Torbles IR; C chooses and composes all rewrites."""
        if not self._handle: return None, {"status": "closed"}
        count = _lib.tr_rule_count(self._handle)
        if count == -4: return None, {"status": "stale", "reason": "table snapshot changed"}
        _check(count)
        if not count:
            return None, {"status": "unchanged", "reason": "no verified relationships", "explored": 0}
        if any(self.table.codecs[c].offset != self.offsets[c] for c in self.columns):
            return None, {"status": "stale", "reason": "column codec changed"}
        if not _lib.tr_scope_valid(self._handle, self.table.at, self.table.used_bm.ctypes.data):
            return None, {"status": "stale", "reason": "table row scope changed"}
        key = id(measure)
        cached = self._rewrite_cache.get(key)
        if cached is not None and cached[0] is measure and cached[1] == self._proof_generation:
            self._rewrite_cache.move_to_end(key)
            return cached[2], dict(cached[3], optimizer_cached=True)
        replacement, info = self._search_codes(measure)
        info["optimizer_cached"] = False
        self._rewrite_cache[key] = (measure, self._proof_generation, replacement, dict(info))
        self._rewrite_cache.move_to_end(key)
        while len(self._rewrite_cache) > 64: self._rewrite_cache.popitem(last=False)
        return replacement, info

    def _search_codes(self, measure):
        """Uncached native search; caller owns the catalog lock and scope checks."""
        poly = algq.poly2(measure, self.table)
        if poly is None or len(poly) > MAX_TERMS:
            return None, {"status": "unsupported", "reason": "measure is not a bounded degree-2 fact polynomial"}
        inp = Polynomial(); inp.n = len(poly)
        try:
            for i, (cols, coef) in enumerate(poly.items()):
                if int(coef) != coef: raise ValueError("non-integer coefficient")
                if any(c not in self.columns for c in cols):
                    return None, {"status": "unsupported", "reason": "measure uses columns outside catalog"}
                inp.terms[i] = Term(self.table.idx(cols[0]) if cols else -1,
                                   self.table.idx(cols[1]) if len(cols) == 2 else -1, _i64(coef))
        except (ValueError, OverflowError):
            return None, {"status": "unsupported", "reason": "coefficient exceeds int64"}
        if any(self.table.codecs[c].offset != self.offsets[c] for c in self.columns):
            return None, {"status": "stale", "reason": "column codec changed"}
        out, trace = Polynomial(), Trace()
        rc = _lib.tr_optimize(self._handle, self.table.at, self.table.used_bm.ctypes.data,
                             C.byref(inp), C.byref(out), C.byref(trace))
        if rc == -4: return None, {"status": "stale", "reason": "table snapshot changed"}
        _check(rc)
        info = dict(status="rewritten" if trace.changed else "unchanged", before_cost=trace.before_cost,
                    after_cost=trace.after_cost, explored=trace.explored, search_truncated=bool(trace.truncated),
                    steps=[dict(rule=int(trace.rule[i]), direction="expand" if trace.direction[i] > 0 else "contract")
                           for i in range(trace.steps)], cost_unit="estimated plane work per 64-row word")
        if not trace.changed: return None, info
        expr = None
        for term in out.terms[:out.n]:
            if term.a < 0:
                node = algq.Const(int(term.coefficient))
            else:
                node = algq.Col(self.table, self.table.cols[term.a])
                if term.b >= 0: node = node * algq.Col(self.table, self.table.cols[term.b])
                if term.coefficient != 1: node = int(term.coefficient) * node
            expr = node if expr is None else expr + node
        if expr is None: expr = algq.Const(0)
        def bytecode_size(e):
            return 1 if isinstance(e, (algq.Col, algq.Const)) else 1 + bytecode_size(e.l) + bytecode_size(e.r)
        if bytecode_size(expr) > 48:
            return None, dict(info, status="fallback", reason="replacement exceeds engine measure bytecode budget")
        info["raw_expression"] = algq.expr_sql(expr)
        return expr, info


def discover(db, table, columns=None, **options):
    """Discover and optionally attach a native catalog to this table only."""
    return Catalog(db[table] if isinstance(table, str) else table, columns, **options)
