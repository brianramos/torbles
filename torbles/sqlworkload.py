# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Compose compatible read-only SQL roots over a shared exact moment state."""

from copy import copy

import sqlexec as E
import sqlfront
import sqlir as I
import sqltypes as T


def _bind_reads(db, sql):
    T.SESSION["zone"] = getattr(db, "_sql_zone", None)
    queries = []
    for statement in sqlfront.split_statements(sql):
        nodes, text = sqlfront.parse(statement)
        flags = dict(sqlfront.FLAGS)
        queries.extend(sqlfront.Binder(db, text, flags).bind_statement(node) for node in nodes)
    return queries


def _family(query, bindings):
    if not isinstance(query, I.Select):
        return None
    source = query.from_
    if (not isinstance(source, I.Binding) or source.table is None
            or source.sample is not None or source.q is not None or source.outer_null
            or not query.aggregate or query.group or query.sets is not None
            or query.having is not None or query.qualify is not None or query.distinct
            or query.order or query.limit is not None or query.offset is not None or query.sample is not None
            or query.percent or query.with_ties or not query.items):
        return None
    table = source.table
    if not E.ALG_MIN_ROWS <= table.n < 1 << 32:
        return None
    measures = set()
    for agg, _ in query.items:
        if (not isinstance(agg, I.Agg) or agg.distinct or agg.filter is not None
                or agg.order or agg.params):
            return None
        if agg.name == "count_star":
            continue
        if agg.name not in ("sum", "avg") or len(agg.args) != 1:
            return None
        arg = agg.args[0]
        if (not isinstance(arg, I.Col) or arg.b is not source or not arg.t.integer
                or arg.nullable or getattr(table.codecs.get(arg.name), "kind", None) != "int"
                or arg.name in table.nullable):
            return None
        measures.add(arg.name)
    if len(measures) > 2:
        return None
    binding = bindings.setdefault(id(table), source)

    def rebind(node):
        if isinstance(node, I.Col) and node.b is source:
            return I.Col(binding, node.name, node.t, node.nullable)
        return None

    where = query.where
    if where is not None:
        if I.has(where, I.Subq) or I.has(where, I.Outer) or not E._deterministic(where):
            return None
        if any(col.b is not source for col in I.cols_of(where)):
            return None
        where = I.subst(where, rebind)
    shape = copy(query)
    shape.from_, shape.where = binding, where
    items = [(I.subst(agg, rebind), name) for agg, name in query.items]
    return (id(table), None if where is None else where.key()), shape, items, measures


def _combined(db, entries, engine):
    _, query, _ = entries[0]
    query.items = []
    slots, projections = {}, []
    for index, _, items in entries:
        select = []
        for agg, name in items:
            key = (agg.key(), str(agg.t))
            if key not in slots:
                slots[key] = len(query.items)
                query.items.append((agg, f"__workload_{slots[key]}"))
            select.append((slots[key], name, agg.t))
        projections.append((index, select))
    result = E.run(db, query, engine)
    values = result.rows[0]
    for root, (index, select) in enumerate(projections):
        out = E.Relation([name for _, name, _ in select], [typ for _, _, typ in select],
                         [tuple(values[slot] for slot, _, _ in select)])
        out.stats = dict(result.stats, engine_queries=result.stats["engine_queries"] if root == 0 else 0,
                         workload_shared_statements=len(entries), workload_output_slots=len(slots))
        yield index, out


def run(db, sql, engine):
    queries = _bind_reads(db, sql)
    results = [None] * len(queries)
    families, bindings = {}, {}
    if engine in ("alg", "gpu") and not db._writable:
        for index, query in enumerate(queries):
            family = _family(query, bindings)
            if family is None:
                continue
            key, shape, items, measures = family
            batches = families.setdefault(key, [])
            if not batches or len(batches[-1][0] | measures) > 2:
                batches.append((set(), []))
            columns, batch = batches[-1]
            columns.update(measures)
            batch.append((index, shape, items))
    starts = {batch[0][0]: batch for batches in families.values() for _, batch in batches if len(batch) > 1}
    for index, query in enumerate(queries):
        if results[index] is not None:
            continue
        if index in starts:
            try:
                for slot, result in _combined(db, starts[index], engine):
                    results[slot] = result
            except T.SqlError:
                # Run the original statements in order so an error belongs to its own root.
                pass
        if results[index] is None:
            results[index] = E.run(db, query, engine)
    return results
