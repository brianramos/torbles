# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Bounded reductions over one fact's encoded keys."""

import ctypes as C

import numpy as np

import algq
import sqlir as I
import sqltypes as T
from .sqlgroupkeys import plane_group_key


def single_fact(planner):
    s = planner.s
    return (s.from_ is planner.fact and not planner.left and planner.fact_sample is None
            and s.sample is None and set(planner.joins) <= set(planner.hidden.values()))


def bounds(s):
    limit = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
    offset = s.offset.v if isinstance(s.offset, I.Lit) else s.offset
    offset = 0 if offset is None else offset
    if type(limit) is not int or type(offset) is not int or limit <= 0 or offset < 0:
        return None
    return limit, offset


def order_expr(s):
    if len(s.order) != 1:
        return None, False
    spec, desc, _ = s.order[0]
    return (s.items[spec[1]][0] if spec[0] == "pos" else spec[1]), desc


def count_order(s):
    expr, desc = order_expr(s)
    return (desc and isinstance(expr, I.Agg) and not expr.distinct and expr.filter is None
            and expr.name in ("count_star", "count"))


def bitmap(planner, preds, *, engine=None):
    fact = planner.tab[planner.fact]
    engine = planner.cx.engine if engine is None else engine
    if not preds and not planner.joins:
        return fact.used_bm
    query = algq.Query(fact, join_mode="auto")
    for binding, fk in planner.joins.items():
        query.join(planner.tab[binding], fk)
    for pred in preds:
        query.where(pred)
    query._rows = query._rowbits = True
    try:
        result = query._run_codes(engine) & fact.used_bm
        if engine == "gpu" and query.device == "gpu":
            planner.cx.stats["cuda_scope_bitmaps"] = planner.cx.stats.get("cuda_scope_bitmaps", 0) + 1
    except (TypeError, ValueError, OverflowError):
        return None
    planner.cx.stats["engine_queries"] += 1
    return result


def _packed_count(part, scope):
    from .sqlscalarplanes import _moments

    count = C.c_uint64()
    if _moments is not None and _moments(part.at, scope.ctypes.data, 0, None, C.byref(count)) == 0:
        return count.value
    return sum(int(word).bit_count() for word in scope)


def cold_wide_group_scope(planner, preds):
    """Price a mandatory wide equality before its first full-part CUDA upload."""
    fact = planner.tab[planner.fact]
    parts = {
        fact._part_of(name)
        for clause in preds
        for pred in algq.conjuncts(clause)
        if isinstance(pred, algq.Cmp) and pred.op == "EQ" and pred.col.table is fact
        and pred.col.name in fact.wide
        for name in (pred.col.name, fact.wide[pred.col.name][1])
    }
    cold = [part for part in parts if part._storage().gpu is None]
    if not cold:
        return None

    scope = bitmap(planner, preds, engine="alg")
    if scope is None:
        return None
    selected = _packed_count(cold[0], scope)
    columns = sum(part.width(name) for part in cold for name in part.cols)
    upload_bytes = columns * ((fact.n + 63) // 64) * 8
    # Bound encoded group state by row count, sparsity, and the transfer it
    # replaces. Charge 256 bytes per selected row for keys and hash scratch.
    limit = min(200_000, fact.n // 64, upload_bytes // 256)
    stats = planner.cx.stats
    stats["cold_wide_scope_cpu"] = stats.get("cold_wide_scope_cpu", 0) + 1
    stats["cold_wide_scope_rows"] = stats.get("cold_wide_scope_rows", 0) + selected
    stats["cold_wide_group_row_budget"] = limit
    return scope, selected, limit, upload_bytes


def layout(planner, groups):
    """Keep exact equality keys and defer safe functions of those keys."""
    from . import sqlexec as E

    fact = planner.tab[planner.fact]
    keys, deferred = [], []
    for expr in groups:
        spec = plane_group_key(expr, planner.fact, fact)
        if spec is None:
            deferred.append(expr)
        else:
            keys.append((expr, spec))
    stored = {expr.key() for expr, _ in keys if isinstance(expr, I.Col)}

    def span(col):
        codec = fact.codecs.get(col.name)
        if getattr(codec, "kind", None) == "ref":
            dim = fact._refdim.get(col.name)
            return E.stored_range(dim, codec.pk) if dim is not None else None
        return E.stored_range(fact, col.name)

    for expr in deferred:
        if isinstance(expr, I.Lit) and not isinstance(expr.v, E.Err):
            continue
        columns = I.cols_of(expr)
        if (not expr.t.integer or not columns or any(col.key() not in stored for col in columns)
                or E.expr_range(expr, span) is None):
            return None
    return (keys, deferred) if keys else None


def _plain_count(agg):
    return agg.name == "count_star" and not agg.distinct and agg.filter is None and not agg.order


def _basic(planner, aggs, engines=("alg",)):
    s = planner.s
    return (single_fact(planner) and planner.cx.engine in engines and s.aggregate
            and s.sets is None and s.having is None and s.qualify is None
            and not s.distinct and s.distinct_on is None and not s.percent and not s.with_ties
            and all(expr.key() in aggs or any(expr.key() == g.key() for g in s.group)
                    for expr, _ in s.items))


def _count_relation(planner, aggs, rows, stat):
    from . import sqlexec as E

    rel = E.Rel(len(rows))
    for i, expr in enumerate(planner.s.group):
        rel.cols[expr.key()] = [keys[i] for _, keys in rows]
    counts = [count for count, _ in rows]
    for key in aggs:
        rel.cols[key] = counts
    rel.plane_ordered = True
    planner.cx.stats[stat] = len(rows)
    return rel


def _rank_packed_counts(planner, scope, specs, aggs, limit, offset):
    """Rank native group states before decoding any group keys."""
    fact = planner.tab[planner.fact]
    query = algq.Query(fact)
    query._raw = True
    query._xrows = (scope, 1)
    try:
        codes, _, counts = query.group_by(*(algq.Col(fact, spec[0]) for spec in specs)).count()._run_codes(planner.cx.engine)
    except (TypeError, ValueError, OverflowError):
        return None
    if planner.cx.engine == "gpu" and query.device == "gpu":
        planner.cx.stats["cuda_packed_offset"] = len(counts)
    keep = np.argsort(-counts, kind="stable")[offset:offset + limit]
    decoded = [spec[-1](codes[keep, i]) for i, spec in enumerate(specs)]
    planner.cx.stats["engine_queries"] += 1
    return _count_relation(planner, aggs, list(zip(counts[keep].tolist(), zip(*decoded))),
                           "plane_group_packed_offset")


def _gpu_minute(fact, spec, scope, offset, limit):
    fn = getattr(algq.gpu_lib(), "gpu_epoch_minute_histogram", None)
    if fn is None:
        return None
    low, high, shift, transform, base, decode = spec
    minute = 60_000_000
    domain = (fact.maxcode(low) + base % minute) // minute + 1
    if domain > 1 << 20:
        return None
    source = algq._plan_table(fact)
    key = algq._PlaneGroupKey(source.idx(low), -1 if high is None else source.idx(high), shift, transform, base)
    counts = np.empty(domain, np.uint32)
    fn.restype = C.c_int
    fn.argtypes = [algq.vp, algq.vp, C.POINTER(algq._PlaneGroupKey), C.c_uint32, algq.vp]
    with algq._GPU_LOCK:
        try:
            handle = algq._run_on(source).gpu_handle()
        except RuntimeError:
            return None
        if fn(handle, scope.ctypes.data, key, domain, counts.ctypes.data):
            return None
    keys = np.flatnonzero(counts)[offset:offset + limit]
    return list(zip(counts[keys].tolist(), zip(decode(keys))))


def special_count(planner, preds, aggs):
    """Count-bounded CASE branches, ordered time buckets, and empty offsets."""
    from .sqlencodednative import BY_KEY, encoded_groups

    s = planner.s
    limit_offset = bounds(s)
    if (not limit_offset or not _basic(planner, aggs, ("alg", "gpu")) or not s.group
            or not aggs or not all(map(_plain_count, aggs.values()))):
        return None
    limit, offset = limit_offset
    gpu = planner.cx.engine == "gpu"
    fact = planner.tab[planner.fact]
    descriptors = [plane_group_key(expr, planner.fact, fact) for expr in s.group]
    if offset and count_order(s) and all(isinstance(expr, I.Col) for expr in s.group):
        packed_rank = limit + offset > 1024 and all(
            spec is not None and spec[1] is None and spec[3] == 0 for spec in descriptors
        )
        cold_scope = cold_wide_group_scope(planner, preds) if gpu and packed_rank else None
        scope = cold_scope[0] if cold_scope is not None else bitmap(planner, preds)
        if scope is None:
            return None
        from . import sqlexec as E

        selected = cold_scope[1] if cold_scope is not None else sum(int(word).bit_count() for word in scope)
        if selected <= offset:
            return _count_relation(planner, aggs, [], "plane_group_empty_offset")
        if fact.n >= E.ALG_MIN_ROWS and selected <= min(200_000, fact.n // 8):
            if packed_rank:
                ranked = _rank_packed_counts(planner, scope, descriptors, aggs, limit, offset)
                if ranked is not None:
                    return ranked
            if gpu:
                return None
            ids = np.flatnonzero(np.unpackbits(scope.view(np.uint8), bitorder="little")[:fact.n])
            atoms = {expr.key(): expr for expr in s.group}
            source = planner.rows_rel(atoms, [], False, ids=ids.astype(np.uint64))
            out = E.aggregate(s, source, aggs, {}, planner.cx)
            out.grouped = True
            planner.cx.stats["plane_group_selective_offset"] = selected
            return out
    else:
        scope = None
    if limit + offset > 1024:
        return None
    order, desc = order_expr(s)
    if (len(descriptors) == 1 and descriptors[0] is not None and descriptors[0][3] == 2
            and order is not None and order.key() == s.group[0].key() and not desc):
        if scope is None:
            scope = bitmap(planner, preds)
        if scope is None:
            return None
        spec = descriptors[0]
        if gpu:
            rows = _gpu_minute(fact, spec, scope, offset, limit)
            if rows is None:
                return None
            planner.cx.stats["engine_queries"] += 1
            planner.cx.stats["cuda_minute_groups"] = len(rows)
            return _count_relation(planner, aggs, rows, "plane_group_ordered_bucket")
        result = encoded_groups(fact, [spec[:-1]], [], limit + offset, scope, rank=BY_KEY)
        if result is None:
            return None
        codes, counts, _, _ = result
        values = spec[-1](codes[offset:, 0])
        planner.cx.stats["engine_queries"] += 1
        return _count_relation(planner, aggs, list(zip(counts[offset:].tolist(), zip(values))),
                               "plane_group_ordered_bucket")
    missing = [i for i, spec in enumerate(descriptors) if spec is None]
    if not count_order(s) or len(missing) != 1:
        return None
    pos = missing[0]
    case = s.group[pos]
    if (not isinstance(case, I.Case) or len(case.whens) != 1
            or not isinstance(case.else_, I.Lit) or type(case.else_.v) is not str
            or T.coll_key(case.t) is not None):
        return None
    condition, then = case.whens[0]
    stored = {expr.key() for expr, spec in zip(s.group, descriptors)
              if spec is not None and isinstance(expr, I.Col)}
    if any(col.key() not in stored for col in I.cols_of(condition)):
        return None
    yes = plane_group_key(then, planner.fact, fact)
    pred = planner.engine_pred(condition)
    if yes is None or pred is None:
        return None
    scope = bitmap(planner, preds) if scope is None else scope
    yes_scope = bitmap(planner, pred)
    if scope is None or yes_scope is None:
        return None
    branches = []
    for mask, include_then in ((scope & yes_scope, True), (scope & ~yes_scope, False)):
        specs = list(descriptors)
        specs[pos] = yes if include_then else None
        physical = [spec for spec in specs if spec is not None]
        if not physical:
            return None
        keys = [spec[:-1] for spec in physical]
        if gpu:
            from . import sqlgpugroups
            result = sqlgpugroups.group(fact, keys, [], limit + offset, mask, planner.cx.stats)
        else:
            result = fact._plane_group(keys, [], limit + offset, mask)
        if result is None:
            return None
        codes, counts, _ = result
        branches.append((physical, include_then, codes, counts))
        planner.cx.stats["engine_queries"] += 1

    # The condition is determined by the other group keys. The branches
    # are disjoint, so ranking counts needs neither decoding nor a merge by key.
    all_counts = np.concatenate([counts for _, _, _, counts in branches]).astype(np.int64)
    ranked = np.argsort(-all_counts, kind="stable")[offset:offset + limit]
    rows = [None] * len(ranked)
    start = 0
    for physical, include_then, codes, counts in branches:
        end = start + len(counts)
        spots = np.flatnonzero((ranked >= start) & (ranked < end))
        ids = ranked[spots] - start
        columns = [spec[-1](codes[ids, i]) for i, spec in enumerate(physical)]
        if not include_then:
            columns.insert(pos, [case.else_.v] * len(ids))
        for spot, count, values in zip(spots, counts[ids].tolist(), zip(*columns)):
            rows[spot] = (count, values)
        start = end
    return _count_relation(planner, aggs, rows, "plane_group_case_branches")


def late_count(planner, preds, aggs):
    """Rank by count, then gather arguments only for the selected groups."""
    from . import sqlexec as E

    s = planner.s
    limit_offset = bounds(s)
    if (not limit_offset or sum(limit_offset) > 1024 or not _basic(planner, aggs, ("alg", "gpu"))
            or not count_order(s) or len(s.group) != 1):
        return None
    order, _ = order_expr(s)
    if not _plain_count(order):
        return None
    fact = planner.tab[planner.fact]
    if fact.n < E.ALG_MIN_ROWS:
        return None
    group = s.group[0]
    key = plane_group_key(group, planner.fact, fact)
    if key is None or key[1] is not None or key[3] != 0:
        return None
    atoms = {group.key(): group}
    for agg in aggs.values():
        if _plain_count(agg):
            continue
        if (agg.name not in ("count", "sum", "avg", "min", "max") or agg.filter is not None
                or agg.order or len(agg.args) != 1 or not isinstance(agg.args[0], I.Col)):
            return None
        arg = agg.args[0]
        if arg.b is not planner.fact or T.coll_key(arg.t) is not None:
            return None
        if agg.name in ("sum", "avg"):
            value_range = planner.value_range(arg)
            if not arg.t.integer or value_range is None:
                return None
            if agg.name == "sum":
                lo, hi = T.INT_RANGE[agg.t.id]
                if min(0, value_range[0]) * fact.n < lo or max(0, value_range[1]) * fact.n > hi:
                    return None
        atoms[arg.key()] = arg
    scope = bitmap(planner, preds)
    if scope is None:
        return None
    limit, offset = limit_offset
    if planner.cx.engine == "gpu":
        from . import sqlgpugroups

        result = sqlgpugroups.group(fact, [key[:-1]], [], limit + offset, scope, planner.cx.stats)
    else:
        result = fact._plane_group([key[:-1]], [], limit + offset, scope)
    if result is None:
        return None
    codes, counts, _ = result
    if len(counts) <= offset:
        return _count_relation(planner, aggs, [], "plane_group_late_states")
    winners = codes[offset:, 0]
    domain = fact.maxcode(key[0]) + 1
    if domain > 1 << 22:
        return None
    selected = np.zeros(domain, dtype=bool)
    selected[winners] = True
    code_set = algq.CodeSet(algq.Col(fact, key[0]), selected)
    chosen = bitmap(planner, [code_set])
    if chosen is None:
        return None
    chosen &= scope
    ids = np.flatnonzero(np.unpackbits(chosen.view(np.uint8), bitorder="little")[:fact.n]).astype(np.uint64)
    source = planner.rows_rel(atoms, [], False, ids=ids)
    # OFFSET was applied to the native ranking. The caller's original
    # ORDER BY must not drop the same groups a second time.
    out = E.aggregate(s, source, aggs, {}, planner.cx)
    count_key = order.key()
    ranked = sorted(range(out.n), key=lambda i: out.cols[count_key][i], reverse=True)
    out = out.take(ranked)
    out.plane_ordered = True
    planner.cx.stats["engine_queries"] += 1
    planner.cx.stats["plane_group_late_states"] = len(ids)
    return out
