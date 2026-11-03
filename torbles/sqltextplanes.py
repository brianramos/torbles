# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Dictionary maps and bitplane reductions for text predicates and projections."""

import ctypes as C
import re

import numpy as np

from . import algq, sqlfuncs as F, sqlir as I, sqltypes as T
from .sqlgroupkeys import plane_group_key


_BYTE_LENGTH = {"strlen", "octet_length"}
_CHAR_LENGTH = {"length", "len", "char_length", "character_length"}


def _dict_column(expr, planner):
    if not isinstance(expr, I.Col) or expr.t.id != "VARCHAR" or T.coll_key(expr.t) is not None:
        return None
    binding, name = planner.canon(expr)
    table = planner.tab[binding]
    codec = table.codecs.get(name)
    if (codec is None or codec.kind != "dict" or codec.sql_type != "VARCHAR" or name not in table.cols
            or name in table.wide or name in getattr(table, "moved", {})):
        return None
    return binding, table, name, codec


def _cache(table, attr, key, build, limit):
    cache = table.__dict__.setdefault(attr, {})
    cached = cache.get(key)
    if cached is not None:
        return cached
    value = build()
    if len(cache) >= limit:
        cache.pop(next(iter(cache), None), None)
    cache[key] = value
    return value


def _like_match(pattern, escape, insensitive):
    if not escape and not insensitive and len(pattern) >= 2 and pattern[0] == pattern[-1] == "%":
        literal = pattern[1:-1]
        if "%" not in literal and "_" not in literal:
            return lambda text: literal in text
    regex = F.like_regex(pattern, escape, insensitive)
    return lambda text: regex.fullmatch(text) is not None


def dictionary_predicate(planner, expr):
    """Build a code-domain bitmap for a literal LIKE or NOT LIKE on a text column."""
    negated = isinstance(expr, I.Func) and expr.fn == "not" and len(expr.args) == 1
    term = expr.args[0] if negated else expr
    if not isinstance(term, I.Func) or term.fn not in ("like", "ilike") or len(term.args) not in (2, 3):
        return None
    col, *args = term.args
    if not all(isinstance(arg, I.Lit) and isinstance(arg.v, str) for arg in args):
        return None
    source = _dict_column(col, planner)
    if source is None:
        return None
    binding, table, name, codec = source
    is_fact = binding is planner.fact
    if not is_fact and (binding not in planner.hidden.values() or not getattr(table, "generated", None)):
        return None
    pattern = args[0].v
    insensitive = term.fn == "ilike"
    try:
        escape = F._esc(args[1].v) if len(args) == 2 else None
        matcher = _like_match(pattern, escape, insensitive)
    except T.SqlError:
        return None

    validity = table.nullable.get(name)

    def make():
        matches = np.fromiter((matcher(text) != negated for text in codec.values), np.bool_, count=len(codec.values))
        if is_fact:
            return algq.CodeSet(algq.Col(table, name), matches, sql=I.render(expr))
        codes = table.decoded(name)
        present = table.used & (codes < len(matches))
        if validity is not None:
            present &= table.decoded(validity) != 0
        rows = np.zeros(table.n, bool)
        rows[present] = matches[codes[present]]
        columns = (name,) if validity is None else (name, validity)
        return algq.RowMask(table, rows, cols=columns, sql=I.render(expr))

    cache_key = (id(table), is_fact, name, codec, pattern, escape, insensitive, negated)
    predicate = _cache(table, "_text_predicates", cache_key, make, 8)
    stats = planner.cx.stats
    stats["dictionary_text_predicates"] = stats.get("dictionary_text_predicates", 0) + 1
    if not is_fact:
        stats["dictionary_dimension_predicates"] = stats.get("dictionary_dimension_predicates", 0) + 1
    if is_fact and validity is not None:
        return [predicate, algq.Cmp(algq.Col(table, validity), "EQ", 1)]
    return [predicate]


def _count_having(expr, aggs, engine):
    """An optional safe early HAVING bound on COUNT(*)."""
    if not isinstance(expr, I.Func) or expr.fn not in (">", ">=", "<", "<=", "=") or len(expr.args) != 2:
        return None
    left, right = map(engine._strip_cast, expr.args)
    op = expr.fn
    if isinstance(left, I.Lit):
        left, right = right, left
        op = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}[op]
    if (not isinstance(left, I.Agg) or left.name != "count_star" or left.key() not in aggs
            or not isinstance(right, I.Lit) or type(right.v) is not int or not -(1 << 63) <= right.v < 1 << 63):
        return None
    return {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "=": np.equal}[op], right.v


def _rows(fact, preds, joins):
    bitmap = fact.used_bm
    if preds or joins:
        query = algq.Query(fact, join_mode="auto")
        for table, fk in joins:
            query.join(table, fk)
        for pred in preds:
            query.where(pred)
        query._rows = query._rowbits = True
        bitmap = query._run_codes("alg") & bitmap
    selected = np.unpackbits(bitmap.view(np.uint8), bitorder="little")[:fact.n]
    return selected if not selected.all() else None


def _read(fact, name, selected, ids):
    if ids is not None:
        return fact.get_rows(name, ids)
    values = fact._decode_column(name)
    return values if selected is None else values[selected]


def _text_codes(fact, table, name, fk, selected, ids):
    validity = table.nullable.get(name)
    if fk is None:
        codes = _read(fact, name, selected, ids)
        valid = None if validity is None else _read(fact, validity, selected, ids) != 0
    else:
        rows = _read(fact, fk, selected, ids)
        mask = fact.kmask(fk)
        if mask:
            rows = rows & mask
        codes = table.decoded(name)[rows]
        valid = None if validity is None else table.decoded(validity)[rows] != 0
    return codes, valid


def _regex_group(expr, col, table, source):
    if not isinstance(expr, I.Func) or expr.fn != "regexp_replace" or len(expr.args) not in (3, 4):
        return None
    if (expr.args[0].key() != col.key() or T.coll_key(expr.t) is not None
            or not all(isinstance(arg, I.Lit) and isinstance(arg.v, str) for arg in expr.args[1:])):
        return None
    name, codec = source
    pattern, replacement, *flags = (arg.v for arg in expr.args[1:])
    flags = flags[0] if flags else ""

    def build():
        regex = F._rx(pattern, flags)
        replacement_value = re.sub(r"\\(\d)", r"\\g<\1>", replacement)
        count = 0 if "g" in flags else 1
        keys = {}

        def code(text):
            value = regex.sub(replacement_value, text, count=count)
            return keys.setdefault(value, len(keys))

        mapping = np.fromiter((code(text) for text in codec.values), np.uint32, count=len(codec.values))
        return mapping, list(keys)

    try:
        return _cache(table, "_text_groups", (name, codec, pattern, replacement, flags), build, 2)
    except (T.SqlError, re.error, IndexError):
        return None


def _length_map(fact, source, byte_length):
    name, codec = source

    def build():
        values = codec.values
        if byte_length:
            lens = (len(text) if text.isascii() else len(text.encode("utf-8")) for text in values)
        else:
            lens = map(len, values)
        return np.fromiter(lens, np.uint64, count=len(values))

    return _cache(fact, "_text_lengths", (name, codec, byte_length), build, 4)


_lookup_sums = getattr(algq.lib, "at_plane_lookup_sums", None)
if _lookup_sums is not None:
    _lookup_sums.restype = C.c_int
    _lookup_sums.argtypes = [algq.vp, algq.vp, C.c_int32, C.c_int32, C.c_uint32,
                            algq.vp, C.c_uint64, algq.vp, C.c_uint32, algq.vp]


def _contract_length(planner, preds, aggs, group, key, source, table, fk, length, having):
    """Derive candidate group masks, then contract projected lengths over them."""
    from . import sqlexec as E, sqlgroupplans as P

    fact = planner.tab[planner.fact]
    name, codec = source
    if (_lookup_sums is None or key is None or key[1] is not None or having is None
            or name in table.nullable or any(a.name == "min" for a in aggs.values())):
        return None
    comparison, boundary = having
    if comparison is np.greater:
        boundary += 1
    elif comparison is not np.greater_equal:
        return None
    if boundary <= 0 or fact.n // boundary > 128:
        return None
    value = name if fk is None else fk
    if value in fact.wide or (fk is not None and fact.fk_nulls.get(fk, 0)):
        return None
    scope = P.bitmap(planner, preds)
    if scope is None:
        return None
    cap = fact.n // boundary
    if cap:
        states = fact._plane_group([key[:-1]], [], cap, scope)
        if states is None:
            return None
        codes, counts, _ = states
        keep = counts >= boundary
        codes = np.ascontiguousarray(codes[keep, 0], dtype=np.uint32)
        counts = counts[keep]
    else:
        codes = np.empty(0, dtype=np.uint32)
        counts = np.empty(0, dtype=np.uint64)
    sums = np.zeros(len(codes), dtype=np.uint64)
    if len(codes):
        byte_length = length.fn in _BYTE_LENGTH

        def make_lookup():
            lengths = _length_map(table, source, byte_length)
            if int(lengths.max(initial=0)) >= 1 << 32:
                return None
            if fk is not None:
                stored = table.decoded(name)
                if int(stored.max(initial=0)) >= len(lengths):
                    return None
                lengths = lengths[stored]
            return lengths.astype(np.uint32)

        try:
            lookup = _cache(table, "_text_lengths", ("lookup", name, codec, byte_length, fk is not None),
                            make_lookup, 4)
        except UnicodeError:
            return None
        if lookup is None:
            return None
        view = algq._plan_table(fact)
        key_idx, value_idx = view.idx(key[0]), view.idx(value)
        mask = fact.kmask(fk) if fk is not None else 0
        if _lookup_sums(algq._run_on(view).at, scope.ctypes.data, key_idx, value_idx,
                        mask or (1 << 32) - 1, lookup.ctypes.data, len(lookup),
                        codes.ctypes.data, len(codes), sums.ctypes.data):
            return None
    result = E.Rel(len(codes))
    result.grouped = True
    result.cols[group.key()] = key[-1](codes)
    for agg_key, agg in aggs.items():
        vals = counts if agg.name == "count_star" else np.asarray(
            [total / count for total, count in zip(sums.tolist(), counts.tolist())], dtype=np.float64)
        result.cols.put(agg_key, E.ColArr(vals))
    planner.cx.stats["engine_queries"] += bool(cap) + bool(len(codes))
    planner.cx.stats["dictionary_plane_groups"] = len(codes)
    planner.cx.stats["dictionary_plane_selected_rows"] = int(counts.sum())
    return result


def project_group(planner, preds, aggs):
    """Group one stored text projection with exact length sums and dictionary MIN."""
    from . import sqlexec as E

    s, cx, fact = planner.s, planner.cx, planner.tab[planner.fact]
    if (not E.COLUMNAR or not E.VEC_GROUPS or cx.engine != "alg" or s.from_ is not planner.fact
            or planner.left or planner.fact_sample is not None or s.sample is not None
            or not E.ALG_MIN_ROWS <= fact.n < 1 << 32 or s.sets is not None or len(s.group) != 1
            or s.qualify is not None or s.distinct or s.distinct_on is not None
            or any(a.distinct or a.filter is not None or a.order for a in aggs.values())):
        return None
    averages = [a for a in aggs.values() if a.name == "avg"]
    if len(averages) != 1 or len(averages[0].args) != 1:
        return None
    length = averages[0].args[0]
    if not isinstance(length, I.Func) or length.fn not in _BYTE_LENGTH | _CHAR_LENGTH or len(length.args) != 1:
        return None
    col = length.args[0]
    source = _dict_column(col, planner)
    if source is None:
        return None
    binding, table, name, codec = source
    source = name, codec
    fk = None if binding is planner.fact else planner.joins.get(binding)
    if binding is not planner.fact and (
        fk is None or not getattr(table, "generated", None) or fact.orphans.get(fk, 0)
        or getattr(fact.surr.get(fk), "version", 0)
    ):
        return None
    if any(b not in planner.hidden.values() for b in planner.joins):
        return None
    joins = [(planner.tab[b], fk_name) for b, fk_name in planner.joins.items()]
    if any(a.name not in ("avg", "count_star") and
           (a.name != "min" or len(a.args) != 1 or a.args[0].key() != col.key())
           for a in aggs.values()):
        return None
    group = s.group[0]
    output_keys = {group.key(), *aggs}
    expressions = [x for x, _ in s.items] + [spec[1] for spec, _, _ in s.order if spec[0] == "expr"]
    if any(x.key() not in output_keys for x in expressions):
        return None
    having = _count_having(s.having, aggs, E)
    if s.having is not None and having is None:
        return None
    key = plane_group_key(group, planner.fact, fact)
    if key is None and (
        not isinstance(group, I.Func) or group.fn != "regexp_replace" or col.nullable or name in table.nullable
    ):
        return None
    if key is not None and key[3] != 0:
        return None
    if key is None:
        from . import sqltexthist
        histogram = sqltexthist.project_group(planner, preds, aggs, group, col, source, table, fk,
                                              length, having, joins)
        if histogram is not None:
            return histogram
    contracted = _contract_length(planner, preds, aggs, group, key, source, table, fk, length, having)
    if contracted is not None:
        return contracted
    try:
        selected = _rows(fact, preds, joins)
    except (TypeError, ValueError, OverflowError):
        return None
    if selected is not None:
        n = int(np.count_nonzero(selected))
        ids = np.flatnonzero(selected).astype(np.uint64) if n * 8 < fact.n else None
        selected = selected.astype(bool, copy=False)
    else:
        n, ids = fact.n, None
    text_codes, valid = None, None
    if key is None:
        regex = _regex_group(group, col, table, source)
        if regex is None:
            return None
        mapping, values = regex
        text_codes, valid = _text_codes(fact, table, name, fk, selected, ids)
        raw = mapping[text_codes]
        group_size = len(values)

        def decode(codes):
            return [values[i] for i in codes.tolist()]

    else:
        low, high, shift, _, _, decode = key
        raw = _read(fact, low, selected, ids)
        if high is not None:
            raw = raw.astype(np.uint64) | (_read(fact, high, selected, ids).astype(np.uint64) << shift)
        group_size = int(raw.max(initial=0)) + 1

    # Keep dense group codes when the code domain is reasonable; compact sparse key domains.
    if group_size > max(1 << 20, n * 2):
        distinct, inverse = np.unique(raw, return_inverse=True)
        group_size = len(distinct)
        inverse = inverse.reshape(-1)
    else:
        distinct = None
        inverse = raw.astype(np.int64, copy=False)
    counts = np.bincount(inverse, minlength=group_size)
    eligible = counts != 0
    if having is not None:
        comparison, boundary = having
        eligible &= comparison(counts, boundary)
    groups = np.flatnonzero(eligible)
    keep_counts = counts[groups]
    result = E.Rel(len(groups))
    result.grouped = True
    logical = groups if distinct is None else distinct[groups]
    result.cols[group.key()] = decode(logical)

    avg_values, avg_valid, minima = [], None, []
    wants_min = any(a.name == "min" for a in aggs.values())
    if len(groups):
        if text_codes is None:
            text_codes, valid = _text_codes(fact, table, name, fk, selected, ids)
        chosen = eligible[inverse]
        if valid is not None:
            chosen &= valid
        projected = text_codes[chosen]
        targets = inverse[chosen]
        try:
            lens = _length_map(table, source, length.fn in _BYTE_LENGTH)
        except UnicodeError:
            return None
        if n * int(lens.max(initial=0)) >= 1 << 53:
            return None                      # float bincount weights must sum to exact integers
        totals = np.bincount(targets, weights=lens[projected].astype(np.float64), minlength=group_size)
        observations = keep_counts if valid is None else np.bincount(targets, minlength=group_size)[groups]
        avg_valid = None if valid is None else observations != 0
        avg_values = [v / c if c else 0.0 for v, c in zip(
            totals[groups].astype(np.int64).tolist(), observations.tolist()
        )]
        if wants_min:
            min_codes = np.full(group_size, np.iinfo(np.uint32).max, dtype=np.uint32)
            np.minimum.at(min_codes, targets, projected)
            minima = codec.decode(min_codes[groups])
    for agg_key, agg in aggs.items():
        if agg.name == "count_star":
            result.cols.put(agg_key, E.ColArr(keep_counts))
        elif agg.name == "avg":
            result.cols.put(agg_key, E.ColArr(np.asarray(avg_values, np.float64), avg_valid))
        else:
            result.cols[agg_key] = minima
    cx.stats["engine_queries"] += bool(preds or joins)
    cx.stats["dictionary_projection_rows"] = cx.stats.get("dictionary_projection_rows", 0) + n
    if fk is not None:
        cx.stats["dictionary_dimension_projection_rows"] = cx.stats.get("dictionary_dimension_projection_rows", 0) + n
    return result

