# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Stored and derived keys for native SQL group selection."""

import sqlir as I
import sqltypes as T


def _stored_key(col, binding, table):
    if not isinstance(col, I.Col) or col.b is not binding or col.nullable:
        return None
    name = col.name
    if name not in table.cols or name in table.nullable or table.fk_nulls.get(name, 0):
        return None
    codec = table.codecs.get(name)
    if codec is None:
        return None
    wide = table.wide.get(name)
    return name, (wide[1] if wide else None), (wide[0] if wide else 0), codec


def _reference_decoder(table, name, codec):
    # A generated dimension has exactly one row per original key. Its
    # current surrogate is therefore an equality code for that key.
    dim = table._refdim.get(name)
    if (dim is None or getattr(dim, "generated", None) != (table.name, name) or codec.pk != name
            or table.orphans.get(name, 0) or name + "#raw" in table.cols
            or name in dim.wide or name in dim.nullable):
        return None
    surrogate = table.surr.get(name)
    if surrogate is not None and surrogate.version:
        return None
    key_codec = dim.codecs[name]

    def decode(codes):
        rows = codes if surrogate is None else codes & surrogate.mask
        return key_codec.decode(dim.get_rows(name, rows))

    return decode


def plane_group_key(expr, binding, table):
    """Return physical low/high planes, transform, offset and result decoder."""
    if isinstance(expr, I.Col):
        key = _stored_key(expr, binding, table)
        if key is None or T.coll_key(expr.t) is not None:
            return None
        name, high, shift, codec = key
        if expr.t.id not in T.INT_RANGE and expr.t.id not in ("VARCHAR", "BOOLEAN", "DATE"):
            return None
        decode = _reference_decoder(table, name, codec) if codec.kind == "ref" else codec.decode
        return (name, high, shift, 0, 0, decode) if decode is not None else None

    if not isinstance(expr, I.Func):
        return None
    transform = 1
    if (expr.fn == "date_trunc" and len(expr.args) == 2 and isinstance(expr.args[0], I.Lit)
            and type(expr.args[0].v) is str and expr.args[0].v.lower() in ("minute", "minutes", "min", "m")):
        col, transform = expr.args[1], 2
    elif expr.fn == "minute" and len(expr.args) == 1:
        col = expr.args[0]
    elif (expr.fn == "date_part" and len(expr.args) == 2 and isinstance(expr.args[0], I.Lit)
          and isinstance(expr.args[0].v, str) and expr.args[0].v.lower() in ("minute", "minutes", "min", "m")):
        col = expr.args[1]
    else:
        return None
    key = _stored_key(col, binding, table)
    if key is None:
        return None
    name, high, shift, codec = key
    if (codec.kind != "timestamp" or codec.zoned
            or not -(1 << 63) <= codec.offset <= codec.offset + table.maxcode(name) < 1 << 63):
        return None
    if transform == 1:
        return name, high, shift, 1, codec.offset, lambda codes: codes.tolist()

    def decode(codes):
        remainder = codec.offset % 60_000_000
        return codec.decode([int(code) * 60_000_000 - remainder for code in codes])

    return name, high, shift, 2, codec.offset, decode
