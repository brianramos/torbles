# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Persist exact SQL moment bases and derive new scalar queries."""

from copy import copy
from fractions import Fraction
from decimal import localcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import sqlexec as E
import sqlir as I
import sqltypes as T

from . import sqlsnapshot, sqlworkload


_LOW, _HIGH = -(1 << 128), 1 << 128
_ALL = ((_LOW, _HIGH),)
_MAX_FILE = 16 << 20


def _union(left, right):
    out = []
    for low, high in sorted(left + right):
        if out and low <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], high))
        else:
            out.append((low, high))
    return tuple(out)


def _intersection(left, right):
    return _union(tuple((max(a, c), min(b, d)) for a, b in left for c, d in right
                        if max(a, c) <= min(b, d)), ())


def _complement(regions):
    cuts = ((_LOW - 1, _LOW - 1),) + regions + ((_HIGH + 1, _HIGH + 1),)
    return tuple((end + 1, start - 1) for (_, end), (start, _) in zip(cuts, cuts[1:]) if end + 1 < start)


def _column(col, source):
    if not isinstance(col, I.Col) or col.b is not source or not col.t.integer or col.nullable:
        return False
    if hasattr(source, "writable"):
        return col.name in source.writable.columns
    table = source.table
    return getattr(table.codecs.get(col.name), "kind", None) == "int" and col.name not in table.nullable


def _range(col, source):
    if not _column(col, source):
        return None
    if hasattr(source, "writable"):
        return T.INT_RANGE[col.t.id]
    return E.stored_range(source.table, col.name)


def _scope(expr, source):
    if expr is None or isinstance(expr, I.Lit) and expr.v is True:
        return None, _ALL
    if isinstance(expr, I.Lit) and expr.v is False:
        return None, ()
    if not isinstance(expr, I.Func):
        return None
    if expr.fn == "not" and len(expr.args) == 1:
        inner = _scope(expr.args[0], source)
        return None if inner is None else (inner[0], _complement(inner[1]))
    if expr.fn in ("and", "or") and expr.args:
        axis, regions = None, _ALL if expr.fn == "and" else ()
        op = _intersection if expr.fn == "and" else _union
        for arg in expr.args:
            child = _scope(arg, source)
            if child is None or axis is not None and child[0] not in (None, axis):
                return None
            axis, regions = axis or child[0], op(regions, child[1])
            if len(regions) > 32:
                return None
        return axis, regions
    if expr.fn not in ("=", "<>", "<", "<=", ">", ">=") or len(expr.args) != 2:
        return None
    col, lit = map(E._strip_cast, expr.args)
    op = expr.fn
    if isinstance(col, I.Lit):
        col, lit = lit, col
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)
    if not _column(col, source) or not isinstance(lit, I.Lit) or type(lit.v) is not int:
        return None
    v = lit.v
    ranges = {"=": ((v, v),), "<>": ((_LOW, v - 1), (v + 1, _HIGH)),
              "<": ((_LOW, v - 1),), "<=": ((_LOW, v),), ">": ((v + 1, _HIGH),), ">=": ((v, _HIGH),)}
    return col.name, _intersection(_ALL, ranges[op])


def _linear(expr, source):
    """Integer coefficients, with SQL intermediate overflow checked separately."""
    e = E._strip_cast(expr)
    if _column(e, source):
        return {e.name: 1}
    if isinstance(e, I.Lit) and type(e.v) is int:
        return {None: e.v}
    if not isinstance(e, I.Func) or e.fn not in ("+", "-", "*", "neg"):
        return None
    parts = [_linear(arg, source) for arg in e.args]
    if any(part is None for part in parts):
        return None
    if e.fn == "neg" and len(parts) == 1:
        return {col: -value for col, value in parts[0].items()}
    if len(parts) != 2:
        return None
    left, right = parts
    if e.fn == "*":
        if set(right) <= {None}:
            left, right = right, left
        if set(left) <= {None}:
            return {col: left.get(None, 0) * value for col, value in right.items()}
        return None
    scale = -1 if e.fn == "-" else 1
    return {col: left.get(col, 0) + scale * right.get(col, 0) for col in left.keys() | right.keys()}


def _finalizer(expr):
    if isinstance(expr, (I.Agg, I.Lit)):
        return True
    if isinstance(expr, I.Cast):
        return (expr.t.integer or expr.t.id == "DOUBLE") and _finalizer(expr.arg)
    return (isinstance(expr, I.Func) and expr.fn in ("+", "-", "*", "neg")
            and (expr.t.integer or expr.t.id == "DOUBLE") and all(map(_finalizer, expr.args)))


def _shape(query):
    if not isinstance(query, I.Select):
        return None
    source = query.from_
    if (not isinstance(source, I.Binding) or source.table is None and not hasattr(source, "writable")
            or source.q is not None
            or source.sample is not None or source.outer_null or not query.aggregate
            or query.group or query.sets is not None or query.having is not None or query.qualify is not None
            or query.distinct or query.distinct_on is not None or query.order or query.limit is not None or query.offset is not None
            or query.sample is not None or query.with_ties or query.percent or not query.items):
        return None
    scope = _scope(query.where, source)
    if scope is None:
        return None
    axis, regions = scope
    if not regions or regions == _ALL:
        axis = None
    if not all(_finalizer(expr) for expr, _ in query.items):
        return None
    aggregates = E.collect([expr for expr, _ in query.items], I.Agg)
    if not aggregates:
        return None
    projections, basis = [], set()
    for agg in aggregates.values():
        if agg.distinct or agg.filter is not None or agg.order or agg.params:
            return None
        if agg.name == "count_star":
            projections.append((agg, {}))
            continue
        if (agg.name not in ("sum", "avg", "count") or len(agg.args) != 1 or not agg.args[0].t.integer
                or agg.name == "sum" and agg.t.id not in T.INT_RANGE):
            return None
        coefficients = _linear(agg.args[0], source)
        if coefficients is None or E.expr_range(agg.args[0], lambda col: _range(col, source)) is None:
            return None
        if agg.name != "count":
            basis.update(col for col in coefficients if col is not None)
        projections.append((agg, coefficients))
    table = source.writable if hasattr(source, "writable") else source.table
    return (table.name, axis, regions), projections, basis


def _read_state(db, query, columns, engine):
    """Execute the basis through the existing packed scalar-moments path."""
    source = query.from_
    columns = sorted(columns)
    count, sums = None, {}
    for start in range(0, max(1, len(columns)), 2):
        chunk = columns[start:start + 2]
        scan = copy(query)
        scan.items = [(I.Agg("count_star", [], T.BIGINT), "__rows")]
        scan.items += [(I.Agg("sum", [source.col(col)], T.HUGEINT), col) for col in chunk]
        n, *values = E.run(db, scan, engine).rows[0]
        count = n
        sums.update((col, int(value) if value is not None else 0) for col, value in zip(chunk, values))
    return count, sums


def _derived(states, key, columns):
    table, axis, target = key
    if not target:
        return (0, dict.fromkeys(columns, 0)), "empty_scope"
    direct = states.get(key)
    if direct is not None and columns <= direct[1].keys():
        return direct, "basis"
    candidates = [(domain, state) for (name, col, domain), state in states.items()
                  if name == table and col in (None, axis) and columns <= state[1].keys()]
    if axis is None and target == _ALL:
        # A full scope can also be covered by two disjoint regions of one axis.
        axes = {col for name, col, _ in states if name == table and col is not None}
        for candidate in sorted(axes):
            hit = _derived_axis(states, table, candidate, target, columns)
            if hit is not None:
                return hit
        return None
    return _combine(candidates, target, columns)


def _derived_axis(states, table, axis, target, columns):
    return _combine([(domain, state) for (name, col, domain), state in states.items()
                     if name == table and col in (None, axis) and columns <= state[1].keys()], target, columns)


def _combine(candidates, target, columns):
    for i, (left, a) in enumerate(candidates):
        for j, (right, b) in enumerate(candidates):
            if i == j:
                continue
            if _intersection(left, right) == right and _intersection(left, _complement(right)) == target:
                return (a[0] - b[0], {c: a[1][c] - b[1][c] for c in columns}), "scope_difference"
            if i < j and not _intersection(left, right) and _union(left, right) == target:
                return (a[0] + b[0], {c: a[1][c] + b[1][c] for c in columns}), "scope_union"
    return _span(candidates, target, columns)


def _span(candidates, target, columns):
    """Reduce an interval indicator against saved indicators over exact rationals."""
    if not 3 <= len(candidates) <= 64:
        return None
    fields = sorted(columns)

    def boundary(regions):
        return {point: sign for low, high in regions for point, sign in ((low, 1), (high + 1, -1))}

    def eliminate(vector, moments, rows):
        for pivot, (other, state) in rows.items():
            scale = vector.get(pivot, 0)
            if scale == 0:
                continue
            for point, value in other.items():
                remainder = vector.get(point, 0) - scale * value
                if remainder:
                    vector[point] = remainder
                else:
                    vector.pop(point, None)
            moments = [value - scale * delta for value, delta in zip(moments, state)]
        return moments

    rows = {}
    for regions, (count, sums) in candidates:
        vector = boundary(regions)
        moments = eliminate(vector, [count, *(sums[col] for col in fields)], rows)
        if not vector:
            continue
        pivot = min(vector)
        divisor = Fraction(vector[pivot])
        rows[pivot] = ({point: value / divisor for point, value in vector.items()},
                       [value / divisor for value in moments])
    want = boundary(target)
    result = eliminate(want, [0] * (len(fields) + 1), rows)
    if want or any(value != int(value) for value in result):
        return None
    count, *sums = [-int(value) for value in result]
    return (count, dict(zip(fields, sums))), "scope_span"


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


class Catalog:
    """Reusable recipes and exact aggregate bases tied to a verified source."""

    def __init__(self, engine="alg"):
        if engine not in ("alg", "gpu"):
            raise ValueError("algebra catalog engine must be 'alg' or 'gpu'")
        self.engine = engine
        self._lock = RLock()
        self._snapshot = None
        self._recipes = []
        self._states = {}

    def _learn(self, db, sql, states):
        for query in sqlworkload._bind_reads(db, sql):
            plan = _shape(query)
            if plan is None:
                continue
            key, _, columns = plan
            old = states.get(key)
            missing = columns if old is None else columns - old[1].keys()
            if old is not None and not missing:
                continue
            got = _derived(states, key, columns)
            if got is not None:
                states[key] = got[0]
                continue
            count, sums = _read_state(db, query, missing, self.engine)
            states[key] = (count, {**(old[1] if old else {}), **sums})

    def learn(self, db, sql):
        """Learn the supported exact basis from a read-only SQL workload."""
        with self._lock, db._sql_lock:
            snapshot = sqlsnapshot.token(db)
            if snapshot is None:
                raise ValueError("learning requires a versioned Torbles source")
            recipes = list(dict.fromkeys([*self._recipes, sql]))
            states = dict(self._states) if snapshot == self._snapshot else {}
            for recipe in recipes if snapshot != self._snapshot else [sql]:
                self._learn(db, recipe, states)
            if snapshot != sqlsnapshot.token(db):
                raise RuntimeError("Torbles source changed during algebra learning")
            self._snapshot, self._recipes, self._states = snapshot, recipes, states
        return self

    def refresh(self, db):
        """Re-evaluate the learned bases on a changed source."""
        with self._lock, db._sql_lock:
            snapshot = sqlsnapshot.token(db)
            if snapshot is None:
                raise ValueError("refresh requires a versioned Torbles source")
            states = {}
            for recipe in self._recipes:
                self._learn(db, recipe, states)
            if snapshot != sqlsnapshot.token(db):
                raise RuntimeError("Torbles source changed during algebra refresh")
            self._snapshot, self._states = snapshot, states
        return self

    def query(self, db, sql, *, refresh=True):
        """Derive one SELECT when the saved basis suffices, or execute it normally."""
        with self._lock, db._sql_lock:
            queries = sqlworkload._bind_reads(db, sql)
            if len(queries) != 1:
                raise ValueError("query expects exactly one read-only statement")
            query, = queries
            plan = _shape(query)
            if plan is not None and self._recipes:
                snapshot = sqlsnapshot.token(db)
                if refresh and snapshot is not None and snapshot != self._snapshot:
                    try:
                        self.refresh(db)
                    except T.SqlError:
                        return E.run(db, query, self.engine)
                if snapshot is not None and snapshot == self._snapshot:
                    key, projections, columns = plan
                    got = _derived(self._states, key, columns)
                    if got is not None:
                        (count, sums), reason = got
                        rel = E.Rel(1)
                        for agg, coefficients in projections:
                            if agg.name in ("count_star", "count"):
                                value = count
                            elif not count:
                                value = None
                            else:
                                total = sum(coeff * (count if col is None else sums[col])
                                            for col, coeff in coefficients.items())
                                if agg.name == "avg":
                                    value = total / count
                                else:
                                    lo, hi = T.INT_RANGE[agg.t.id]
                                    if not lo <= total <= hi:
                                        return E.run(db, query, self.engine)
                                    value = total
                            rel.cols[agg.key()] = [value]
                        cx = E.Cx(db)
                        with localcontext(T.CTX):
                            values = [E.ev(expr, rel, cx)[0] for expr, _ in query.items]
                        if not any(isinstance(value, E.Err) for value in values) and snapshot == sqlsnapshot.token(db):
                            result = E.Relation([name for _, name in query.items],
                                                [expr.t for expr, _ in query.items], [tuple(values)])
                            result.stats = dict(cx.stats, algebra_derivation=reason)
                            return result
            return E.run(db, query, self.engine)

    def save(self, path):
        """Atomically replace a checksum-protected catalog file."""
        path = Path(path)
        with self._lock:
            states = [[table, axis, regions, count, sums]
                      for (table, axis, regions), (count, sums) in self._states.items()]
            record = {"version": 1, "engine": self.engine, "snapshot": self._snapshot,
                      "recipes": self._recipes, "states": states}
            data = _json({"sha256": sha256(_json(record)).hexdigest(), "catalog": record})
        if len(data) > _MAX_FILE:
            raise ValueError("algebra catalog exceeds the format size limit")
        tmp = None
        try:
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                tmp = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Load a catalog. Its checkpoint is admitted only by a matching source."""
        with open(path, "rb") as stream:
            data = stream.read(_MAX_FILE + 1)
        if len(data) > _MAX_FILE:
            raise ValueError("algebra catalog exceeds the format size limit")
        try:
            payload = json.loads(data)
            record = payload["catalog"]
            if payload["sha256"] != sha256(_json(record)).hexdigest() or record["version"] != 1:
                raise ValueError("catalog checksum or version mismatch")
            snapshot, recipes = record["snapshot"], record["recipes"]
            if ((snapshot is not None and not isinstance(snapshot, str)) or not isinstance(recipes, list)
                    or any(not isinstance(recipe, str) for recipe in recipes)):
                raise ValueError("invalid catalog metadata")
            result = cls(record.get("engine", "alg"))
            result._snapshot, result._recipes = snapshot, recipes
            for table, axis, regions, count, sums in record["states"]:
                regions = tuple(tuple(region) for region in regions)
                if (not isinstance(table, str) or axis is not None and not isinstance(axis, str)
                        or type(count) is not int or count < 0 or not isinstance(sums, dict)
                        or any(type(v) is not int for v in sums.values())
                        or any(len(region) != 2 or any(type(v) is not int for v in region)
                               or not _LOW <= region[0] <= region[1] <= _HIGH for region in regions)
                        or regions != _union(regions, ()) or axis is None and regions not in ((), _ALL)):
                    raise ValueError("invalid aggregate basis")
                result._states[(table, axis, regions)] = (count, sums)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid algebra catalog") from exc
        return result
