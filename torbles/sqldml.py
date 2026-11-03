# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Execute SQL mutations through the writable bitplane catalog."""

import decimal

import numpy as np

from . import sqlfront, sqlfuncs as F, sqlir as I, sqltypes as T, sqlexec as E
from .database import _q


_U32 = T.SqlType("UINTEGER")


def _columns(table, names, *, all_columns=False):
    by_name = {c.lower(): c for c in table.columns}
    result = []
    for name in names:
        if name.lower() not in by_name:
            raise T.SqlError(f"unknown target column {name!r}")
        result.append(by_name[name.lower()])
    if len(set(result)) != len(result):
        raise T.SqlError("duplicate target column")
    if all_columns and set(result) != set(table.columns):
        raise T.SqlError("INSERT requires every writable column")
    return result


def _pack(names, columns, types):
    result = {}
    for name, values, src in zip(names, columns, types, strict=True):
        if src != _U32:
            values = [v if isinstance(v, F.Err) else T.cast_value(v, src, _U32) for v in values]
        for value in values:
            if isinstance(value, F.Err):
                raise T.SqlError(value.msg)
            if value is None:
                raise T.SqlError(f"NULL is not allowed in writable column {name!r}")
        result[name] = np.asarray(values, dtype=np.uint32)
    return result


def _insert(db, table, node, engine):
    names = _columns(table, node.columns or table.columns, all_columns=True)
    text = node.select if node.select is not None else "VALUES " + ", ".join(
        "(" + ", ".join(row) + ")" for row in node.values)
    query = sqlfront.bind(db, text)[0]
    result = E.run(db, query, engine)
    if len(result.types) != len(names):
        raise T.SqlError("INSERT source has a different column count")
    columns = list(zip(*result.rows)) if result.rows else [[] for _ in names]
    return len(table.insert(_pack(names, columns, result.types)))


def _lossless_column(expr):
    while isinstance(expr, I.Cast):
        if expr.try_ or not expr.arg.t.integer or not expr.t.integer:
            break
        lo, hi = T.INT_RANGE[expr.arg.t.id]
        a, b = T.INT_RANGE[expr.t.id]
        if a > lo or b < hi:
            break
        expr = expr.arg
    return expr


def _other_key(expr, target, primary_key, source):
    """Return the source expression in a lossless equality on the target PK."""
    if not isinstance(expr, I.Func) or expr.fn != "=" or len(expr.args) != 2:
        return None
    for a, b in (expr.args, expr.args[::-1]):
        col = _lossless_column(a)
        other = _lossless_column(b)
        valid = isinstance(other, I.Lit) if source is None else isinstance(other, I.Col) and other.b is source
        if isinstance(col, I.Col) and col.b is target and col.name == primary_key and b.t.integer and valid:
            return b
    return None


def _lookup(table, values):
    ids = np.full(len(values), -1, np.int64)
    keys, positions = [], []
    for i, value in enumerate(values):
        if isinstance(value, F.Err):
            raise T.SqlError(value.msg)
        if value is not None and 0 <= value <= 0xffffffff:
            keys.append(value)
            positions.append(i)
    if positions:
        ids[positions] = table.lookup(np.asarray(keys, dtype=np.uint32))
    return ids


def _target_rel(binding, table, ids, needed):
    names = list(dict.fromkeys(c.name for c in needed))
    cols = table.store.read(ids, names)
    rel = E.Rel(len(ids))
    for name, values in cols.items():
        rel.cols.put(binding.col(name).key(), E.ColArr(values))
    return rel


def _source_rel(binding, needed, cx):
    cols = {c.key(): c for c in needed}
    rel = E._mat_binding(binding, cols, [], cx)
    if rel.cnt.count(1) != len(rel.cnt):
        rel = rel.take(np.repeat(np.arange(rel.n), rel.cnt))
        rel.cnt = [1] * rel.n
    return rel


def _expressions(names, exprs, rel, cx):
    values = [E.ev(e, rel, cx) for e in exprs]
    return _pack(names, values, [e.t for e in exprs])


def _bound_change(db, table, node, assignment_names, insert_names):
    target_alias = _q(node.alias or table.name)
    table_from = _q(table.name) + (f" AS {target_alias}" if node.alias else "")
    select = [f"{target_alias}.{_q(table.primary_key)}"]
    select += [f"({e})" for _, e in node.assignments]
    if insert_names:
        select += [f"({e})" for e in node.values[0]]
    if node.source is not None:
        table_from += ", " + node.source.sql
    where = node.on if node.kind == "merge" else node.where
    sql = f"SELECT {', '.join(select)} FROM {table_from}" + (f" WHERE {where}" if where else "")
    bound = sqlfront.bind(db, sql)[0]
    expressions = [e for e, _ in bound.items]
    if bound.aggregate or E.collect(expressions, I.Win):
        raise T.SqlError("DML values cannot contain top-level aggregates or windows")
    binds = E.block_bindings(bound.from_)
    n = len(assignment_names)
    return bound, binds[0], binds[1] if len(binds) > 1 else None, expressions[1:1+n], expressions[1+n:]


def _needed(expressions, bindings):
    expressions = [e for e in expressions if e is not None]
    cols = [c for e in expressions for c in I.cols_of(e)]
    for subquery in E.subqueries_in(expressions):
        cols.extend(E.outer_refs_into(subquery.q, bindings))
    return cols


def _keyed_source(table, target, source, cond, assignments, inserts, cx):
    key = _other_key(cond, target, table.primary_key, source)
    if key is None:
        return None
    expressions = [key, *assignments, *inserts]
    need = _needed(expressions, {target, source})
    relation = _source_rel(source, (c for c in need if c.b is source), cx)
    ids = _lookup(table, E.ev(key, relation, cx))
    return relation, ids, [c for c in need if c.b is target]


def _change(db, table, node, engine):
    assignments = _columns(table, (name for name, _ in node.assignments))
    if table.primary_key in assignments:
        raise T.SqlError("the writable primary key is immutable")
    inserts = _columns(table, node.columns or (), all_columns=True) if node.kind == "merge" and node.values else []
    bound, target, source, update_exprs, insert_exprs = _bound_change(db, table, node, assignments, inserts)
    cx = E.Cx(db, engine)
    if inserts and any(c.b is target for c in _needed(insert_exprs, {target})):
        raise T.SqlError("a MERGE insert expression cannot reference the target")
    keyed = _keyed_source(table, target, source, bound.where, update_exprs, insert_exprs, cx) if source else None
    if keyed is not None:
        rel, ids, needed = keyed
        matched, missing = np.flatnonzero(ids >= 0), np.flatnonzero(ids < 0)
        found = ids[matched]
        if (assignments or node.matched_delete) and len(np.unique(found)) != len(found):
            raise T.SqlError("multiple source rows match the same target row")
        if node.kind == "delete":
            return table.delete(np.unique(found)), True
        updates = None
        if assignments:
            vals = _target_rel(target, table, found, needed)
            paired = E._combine(vals, rel, np.arange(len(found)), matched)
            updates = _expressions(assignments, update_exprs, paired, cx)
        if node.kind == "update":
            return table.update(found, updates), True
        appended = _expressions(inserts, insert_exprs, rel.take(missing), cx) if inserts else None
        return table.merge(found if assignments else (), updates, appended,
                           delete_ids=found if node.matched_delete else None), True
    if node.kind == "merge":
        raise T.SqlError("MERGE ON requires an equality on the target primary key")

    # Bind the original predicate and assignments once. An ordinary SQL SELECT
    # handles general predicates and correlated subqueries before any mutation.
    indexed = False
    if source is None:
        for term in E._conjuncts(bound.where):
            key = _other_key(term, target, table.primary_key, None)
            if key is not None:
                ids = _lookup(table, E.ev(key, E.Rel(1), cx))
                needed = _needed([bound.where, *(e for e, _ in bound.items)], {target})
                target._dml_rel = _target_rel(target, table, ids[ids >= 0], needed)
                indexed = True
                break
    rows = E.run(db, bound, engine).rows
    found = _lookup(table, [r[0] for r in rows])
    if node.kind == "delete":
        return table.delete(np.unique(found)), indexed
    if len(np.unique(found)) != len(found):
        raise T.SqlError("multiple source rows match the same target row")
    values = list(zip(*(r[1:] for r in rows))) if rows else [[] for _ in assignments]
    return table.update(found, _pack(assignments, values, [e.t for e in update_exprs])), indexed


def run(db, node, engine):
    table = next((v for k, v in db._writable.items() if k.lower() == node.target.lower()), None)
    if table is None:
        raise T.SqlError(f"table {node.target!r} is not writable")
    F.CLOCK["now"] = None
    with decimal.localcontext(T.CTX):
        try:
            if node.kind == "insert":
                count, indexed = _insert(db, table, node, engine), False
            else:
                count, indexed = _change(db, table, node, engine)
        except (ValueError, OverflowError, KeyError, TypeError) as error:
            raise T.SqlError(str(error)) from error
    result = E.Relation(["Count"], [T.BIGINT], [(int(count),)])
    result.stats = dict(writable_affected_rows=int(count), writable_key_lookup=int(indexed))
    return result
