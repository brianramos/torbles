# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Plan exact sparse COUNT DISTINCT and bounded encoded GROUP BY."""

import sqlir as I
import sqltypes as T

from torbles.sqlencodednative import (BY_COUNT, BY_DISTINCT, BY_KEY, BY_UNORDERED,
                                      GROUP_LIMIT, PREFIX_LIMIT, encoded_groups)
from torbles.sqlgroupkeys import plane_group_key


def _moved_key(expr, binding, fact):
    """Project a proven total generated reference to a dimension equality code."""
    if (not isinstance(expr, I.Col) or expr.b is not binding or expr.nullable
            or T.coll_key(expr.t) is not None
            or expr.t.id not in (*T.INT_RANGE, "VARCHAR", "BOOLEAN", "DATE")):
        return None
    name = expr.name
    determinant = getattr(fact, "moved", {}).get(name)
    if determinant is None or determinant not in fact.cols or determinant in fact.wide:
        return None
    dim = fact._refdim.get(determinant)
    surrogate = fact.surr.get(determinant)
    if (dim is None or getattr(dim, "generated", None) != (fact.name, determinant)
            or fact.orphans.get(determinant, 0) or fact.fk_nulls.get(determinant, 0)
            or (surrogate is not None and surrogate.version)
            or name not in dim.codecs or name in dim.nullable or name in dim.wide):
        return None
    codec = dim.codecs[name]
    if codec.kind == "ref":
        return None
    mask = (1 << 32) - 1 if surrogate is None else surrogate.mask
    return (determinant, None, 0, 3, 0, codec.decode), (dim.decoded(name), mask, dim.width(name))


def run(executor, preds, aggs):
    from sqlexec import ALG_MIN_ROWS, COLUMNAR, VEC_GROUPS, Rel, _strip_cast
    from torbles.sqlgroupplans import bitmap, single_fact

    s, cx, binding = executor.s, executor.cx, executor.fact
    fact = executor.tab[binding]
    if (not COLUMNAR or not VEC_GROUPS or cx.engine != "alg" or not single_fact(executor)
            or not ALG_MIN_ROWS <= fact.n < 1 << 32 or s.sets is not None or len(s.group) > 3
            or s.having is not None or s.qualify is not None or s.distinct or s.distinct_on is not None
            or s.percent or s.with_ties or len(s.order) > 1):
        return None
    group_keys = {expr.key() for expr in s.group}
    if any(expr.key() not in group_keys and expr.key() not in aggs for expr, _ in s.items):
        return None
    keys, projection = [], None
    for expr in s.group:
        spec = plane_group_key(expr, binding, fact)
        if spec is None:
            moved = _moved_key(expr, binding, fact) if projection is None else None
            if moved is None:
                return None
            spec, projection = moved
        keys.append(spec)

    distinct, measures, plans = None, [], {}
    for key, agg in aggs.items():
        if agg.filter is not None or agg.order:
            return None
        if agg.distinct:
            if agg.name != "count" or len(agg.args) != 1:
                return None
            spec = plane_group_key(agg.args[0], binding, fact)
            if spec is None or (distinct is not None and distinct != spec[:-1]):
                return None
            distinct = spec[:-1]
            plans[key] = ("distinct", 0, 0)
        elif agg.name == "count_star":
            plans[key] = ("count", 0, 0)
        elif agg.name in ("sum", "avg") and len(agg.args) == 1 and agg.args[0].t.integer:
            arg = _strip_cast(agg.args[0])
            if not isinstance(arg, I.Col) or arg.b is not binding or arg.nullable:
                return None
            b, name = executor.canon(arg)
            codec = fact.codecs.get(name)
            if (b is not binding or name not in fact.cols or name in fact.nullable
                    or codec is None or codec.kind != "int" or name in fact.wide or name in fact.sidecars):
                return None
            offset = int(codec.offset)
            max_code = (1 << int(fact.width(name))) - 1
            if fact.n * max_code >= 1 << 64:
                return None
            if agg.name == "sum":
                low, high = T.INT_RANGE[agg.t.id]
                if min(0, offset) * fact.n < low or max(0, offset + max_code) * fact.n > high:
                    return None
            if name not in measures:
                if len(measures) == 2:
                    return None
                measures.append(name)
            plans[key] = (agg.name, measures.index(name), offset)
        else:
            return None

    lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
    off = s.offset.v if isinstance(s.offset, I.Lit) else s.offset
    off = 0 if off is None else off
    if not keys:
        if distinct is None or s.order or lim is not None or off or any(kind != "distinct" for kind, _, _ in plans.values()):
            return None
        cap, rank = 1, BY_DISTINCT
    else:
        if type(lim) is not int or type(off) is not int or lim <= 0 or off < 0 or lim + off > GROUP_LIMIT:
            return None
        cap, rank = lim + off, BY_KEY
        if (not s.order and distinct is None and not measures and cap <= PREFIX_LIMIT
                and all(key[3] == 0 for key in keys)):
            rank = BY_UNORDERED
        if s.order:
            if distinct is None and projection is None:   # retain existing COUNT-ordered route
                return None
            spec, desc, _ = s.order[0]
            expr = s.items[spec[1]][0] if spec[0] == "pos" else spec[1]
            if not desc or expr.key() not in plans:
                return None
            kind = plans[expr.key()][0]
            if kind == "distinct":
                rank = BY_DISTINCT
            elif kind == "count":
                rank = BY_COUNT
            else:
                return None

    scope = bitmap(executor, preds)
    if scope is None:
        return None
    result = encoded_groups(fact, [key[:-1] for key in keys], measures, cap, scope,
                            distinct=distinct, rank=rank, projection=projection)
    if result is None:
        return None
    codes, counts, sums, uniques = result
    counts = counts[off:].tolist()
    rel = Rel(len(counts))
    for i, (expr, key) in enumerate(zip(s.group, keys)):
        rel.cols[expr.key()] = key[-1](codes[off:, i])
    for key, (kind, index, offset) in plans.items():
        if kind == "distinct":
            rel.cols[key] = uniques[off:].tolist()
        elif kind == "count":
            rel.cols[key] = counts
        else:
            exact = [code + offset * n for code, n in zip(sums[off:, index].tolist(), counts)]
            rel.cols[key] = exact if kind == "sum" else [v / n for v, n in zip(exact, counts)]
    rel.plane_ordered = True
    cx.stats["engine_queries"] += 1
    cx.stats["encoded_group_rows"] = cx.stats.get("encoded_group_rows", 0) + fact.n
    cx.stats["encoded_group_groups"] = cx.stats.get("encoded_group_groups", 0) + rel.n
    return rel

