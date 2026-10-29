# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Column types: how a typed value is stored as an unsigned integer code in the bit planes, and back.

Every codec is order-preserving, so a comparison with a constant becomes an interval of codes (one comparator
polynomial on the planes), and GROUP BY works on codes and decodes the groups at the end.

    IntCodec        integers, signed or not:      code = v - min
    DecimalCodec    DECIMAL(p, s):                code = v * 10^s - min        (exact; sums come back as Decimal)
    DateCodec       DATE:                         code = days since 1970-01-01 - min
    TimestampCodec  TIMESTAMP:                    code = microseconds since the epoch - min
    BoolCodec       BOOLEAN:                      code = 0 / 1
    DictCodec       strings (and floats):         code = rank among the distinct values, in sort order

A code that needs more than 32 bits is stored in a self-extending column (Table(extend=...)) and keeps its meaning.
The affine codecs (all but DictCodec) can also be summed: value = (code + offset) / 10^scale."""
import datetime as _dt, json, math
from bisect import bisect_left, bisect_right
from decimal import Decimal
from fractions import Fraction
import numpy as np

EPOCH = _dt.date(1970, 1, 1)
EPOCH_TS = _dt.datetime(1970, 1, 1)


def _frac(v):
    """an exact rational for a numeric constant (int, float, Decimal, Fraction, numeric string)"""
    if isinstance(v, bool): return Fraction(int(v))
    if isinstance(v, (int, np.integer)): return Fraction(int(v))
    if isinstance(v, Fraction): return v
    if isinstance(v, Decimal):
        if not v.is_finite(): raise ValueError(f"{v} is not a finite number")
        return Fraction(v)
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v): raise ValueError(f"{v} is not a finite number")
        return Fraction(Decimal(repr(float(v))))          # the shortest decimal that round-trips: 0.1 is 1/10
    if isinstance(v, str): return Fraction(Decimal(v))
    raise TypeError(f"expected a number, got {type(v).__name__} {v!r}")


def _int_bounds(op, x):
    """integer interval [lo, hi] (None = unbounded) of the integers i with  i <op> x  (x rational); None if empty"""
    fl, ce = math.floor(x), math.ceil(x)
    if op == "EQ": return (fl, fl) if fl == x else None
    if op == "LT": return (None, ce - 1)
    if op == "LE": return (None, fl)
    if op == "GT": return (fl + 1, None)
    if op == "GE": return (ce, None)
    raise ValueError(op)


class Codec:
    kind = "?"
    affine = None                    # (offset, scale) when value = (code + offset) / 10^scale, else None
    sql_type = ""

    def interval(self, op, v):
        """codes c with  value(c) <op> v, as an inclusive interval (lo, hi) with None for unbounded; None if empty.
        op: EQ LT LE GT GE"""
        raise NotImplementedError

    def literal(self, v):
        """v as a SQL literal of this column's type (for the SQL rendering of a query)"""
        raise NotImplementedError

    def decode(self, codes):
        """codes (int64 array) -> list of Python values"""
        raise NotImplementedError

    def to_json(self): return dict(kind=self.kind, **self._state())
    def _state(self): return {}


class IntCodec(Codec):
    kind = "int"

    def __init__(self, offset=0, sql_type="BIGINT"):
        self.offset, self.sql_type = int(offset), sql_type
        self.affine = (self.offset, 0)

    def interval(self, op, v):
        b = _int_bounds(op, _frac(v))
        if b is None: return None
        lo, hi = b
        return (None if lo is None else lo - self.offset, None if hi is None else hi - self.offset)

    def literal(self, v):
        f = _frac(v)
        return str(f.numerator) if f.denominator == 1 else format(Decimal(f.numerator) / Decimal(f.denominator), "f")

    def decode(self, codes):
        a = np.asarray(codes)
        if a.dtype != object and abs(self.offset) < 2**62 and (not a.size or int(np.abs(a).max()) < 2**62):
            return (a.astype(np.int64) + self.offset).tolist()
        return [int(c) + self.offset for c in a.tolist()]
    def _state(self): return dict(offset=self.offset, sql_type=self.sql_type)


class BoolCodec(Codec):
    kind = "bool"; sql_type = "BOOLEAN"
    affine = (0, 0)

    def interval(self, op, v):
        if not isinstance(v, (bool, np.bool_, int, np.integer)): raise TypeError(f"expected a boolean, got {v!r}")
        b = _int_bounds(op, Fraction(int(bool(v))))
        return b

    def literal(self, v): return "TRUE" if v else "FALSE"
    def decode(self, codes): return [bool(c) for c in np.asarray(codes).tolist()]


class DecimalCodec(Codec):
    kind = "decimal"

    def __init__(self, scale, offset=0, sql_type=None):
        self.scale, self.offset = int(scale), int(offset)
        self.sql_type = sql_type or f"DECIMAL(18, {self.scale})"
        self.affine = (self.offset, self.scale)

    def interval(self, op, v):
        b = _int_bounds(op, _frac(v) * 10 ** self.scale)
        if b is None: return None
        lo, hi = b
        return (None if lo is None else lo - self.offset, None if hi is None else hi - self.offset)

    def literal(self, v):
        f = _frac(v)
        d = Decimal(f.numerator) / Decimal(f.denominator) if f.denominator != 1 else Decimal(f.numerator)
        return format(d, "f")

    def decode(self, codes):
        s = self.scale
        return [Decimal((int(int(c) + self.offset < 0), tuple(map(int, str(abs(int(c) + self.offset)))), -s)) for c in np.asarray(codes).tolist()]

    def _state(self): return dict(scale=self.scale, offset=self.offset, sql_type=self.sql_type)


def _as_date(v):
    if isinstance(v, _dt.datetime): return v.date()
    if isinstance(v, _dt.date): return v
    if isinstance(v, np.datetime64): return _dt.date.fromisoformat(str(v.astype("datetime64[D]")))
    if isinstance(v, str): return _dt.date.fromisoformat(v)
    raise TypeError(f"expected a date, got {v!r}")


class DateCodec(Codec):
    kind = "date"; sql_type = "DATE"

    def __init__(self, offset=0):
        self.offset = int(offset); self.affine = None        # dates are compared and grouped, not summed

    def interval(self, op, v):
        d = _as_date(v); x = Fraction((d - EPOCH).days)
        if isinstance(v, _dt.datetime) and v != _dt.datetime.combine(d, _dt.time()):
            x += Fraction(1, 2)                                   # a time within the day: strictly between two dates
        b = _int_bounds(op, x)
        if b is None: return None
        lo, hi = b
        return (None if lo is None else lo - self.offset, None if hi is None else hi - self.offset)

    def literal(self, v):
        if isinstance(v, _dt.datetime): return f"TIMESTAMP '{v.isoformat(sep=' ')}'"
        return f"DATE '{_as_date(v).isoformat()}'"

    def decode(self, codes):
        return [EPOCH + _dt.timedelta(days=int(c) + self.offset) for c in np.asarray(codes).tolist()]

    def _state(self): return dict(offset=self.offset)


def _as_us(v):
    if isinstance(v, str): v = _dt.datetime.fromisoformat(v)
    if isinstance(v, np.datetime64): return Fraction(int(v.astype("datetime64[us]").astype(np.int64)))
    if isinstance(v, _dt.datetime):
        if v.tzinfo is not None: v = v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        d = v - EPOCH_TS
        return Fraction(d.days * 86_400_000_000 + d.seconds * 1_000_000 + d.microseconds)
    if isinstance(v, _dt.date): return Fraction((v - EPOCH).days * 86_400_000_000)
    raise TypeError(f"expected a timestamp, got {v!r}")


class TimestampCodec(Codec):
    kind = "timestamp"; sql_type = "TIMESTAMP"

    def __init__(self, offset=0):
        self.offset = int(offset); self.affine = None

    def interval(self, op, v):
        b = _int_bounds(op, _as_us(v))
        if b is None: return None
        lo, hi = b
        return (None if lo is None else lo - self.offset, None if hi is None else hi - self.offset)

    def literal(self, v):
        if isinstance(v, str): v = _dt.datetime.fromisoformat(v)
        if isinstance(v, _dt.datetime): return f"TIMESTAMP '{v.replace(tzinfo=None).isoformat(sep=' ')}'"
        return f"TIMESTAMP '{_as_date(v).isoformat()} 00:00:00'"

    def decode(self, codes):
        return [EPOCH_TS + _dt.timedelta(microseconds=int(c) + self.offset) for c in np.asarray(codes).tolist()]

    def _state(self): return dict(offset=self.offset)


class DictCodec(Codec):
    """code = rank of the value among the column's distinct values (sorted: strings by code point, which is the
    byte order of UTF-8 and so the order DuckDB compares VARCHAR in). Numeric dictionaries compare like SQL: DOUBLE
    columns as doubles, integer and decimal columns exactly."""
    kind = "dict"
    FLOATS = ("DOUBLE", "FLOAT", "REAL")

    def __init__(self, values, sql_type="VARCHAR"):
        self.values = list(values); self.sql_type = sql_type
        self.numeric = sql_type not in ("VARCHAR",)
        self.floating = sql_type.upper() in self.FLOATS
        self._keys = self.values if not self.numeric or self.floating else [_frac(v) for v in self.values]

    def _key(self, v):
        if self.numeric:
            if isinstance(v, str): raise TypeError(f"expected a number for a {self.sql_type} column, got {v!r}")
            if self.floating: return float(v)
            return _frac(v)
        if not isinstance(v, str): raise TypeError(f"expected a string for a {self.sql_type} column, got {v!r}")
        return v

    def interval(self, op, v):
        k, vals = self._key(v), self._keys
        if op == "EQ":
            i = bisect_left(vals, k)
            return (i, i) if i < len(vals) and vals[i] == k else None
        if op == "LT": return (None, bisect_left(vals, k) - 1)
        if op == "LE": return (None, bisect_right(vals, k) - 1)
        if op == "GT": return (bisect_right(vals, k), None)
        if op == "GE": return (bisect_left(vals, k), None)
        raise ValueError(op)

    def prefix_interval(self, prefix):
        """codes of the strings starting with prefix: one interval of the sorted dictionary"""
        if self.numeric: raise TypeError("startswith() needs a string column")
        lo = bisect_left(self.values, prefix)
        if not prefix: return (0, None)
        hi = bisect_left(self.values, prefix[:-1] + chr(ord(prefix[-1]) + 1)) if ord(prefix[-1]) < 0x10FFFF else len(self.values)
        return (lo, hi - 1) if hi > lo else None

    def literal(self, v):
        if self.floating: return repr(float(v))
        if self.numeric:
            f = _frac(v)
            return str(f.numerator) if f.denominator == 1 else format(Decimal(f.numerator) / Decimal(f.denominator), "f")
        return "'" + str(v).replace("'", "''") + "'"

    def decode(self, codes):
        vals = self.values; return [vals[int(c)] for c in np.asarray(codes).tolist()]

    def _state(self): return dict(sql_type=self.sql_type)          # the values are stored next to the manifest


class RankCodec(Codec):
    """a column of a referenced table whose values span more than 32 bits: code = rank of its exact integer
    representation (the base codec's, offset 0) among the distinct ones. Order-preserving and exact for every type
    (integers, decimals, dates, timestamps); constants are converted by the base codec, decoded values likewise."""
    kind = "rank"

    def __init__(self, base, raw):
        self.base, self.raw = base, [int(x) for x in raw]; self.sql_type = base.sql_type
        self.affine = None

    def interval(self, op, v):
        b = self.base.interval(op, v)
        if b is None: return None
        lo, hi = b
        return (None if lo is None else bisect_left(self.raw, lo), None if hi is None else bisect_right(self.raw, hi) - 1)

    def literal(self, v): return self.base.literal(v)
    def decode(self, codes): return self.base.decode(np.asarray([self.raw[int(c)] for c in np.asarray(codes).tolist()], object))
    def _state(self): return dict(base=self.base.to_json())


def from_json(d, values=None):
    k = d["kind"]
    if k == "int": return IntCodec(d["offset"], d.get("sql_type", "BIGINT"))
    if k == "bool": return BoolCodec()
    if k == "decimal": return DecimalCodec(d["scale"], d["offset"], d.get("sql_type"))
    if k == "date": return DateCodec(d["offset"])
    if k == "timestamp": return TimestampCodec(d["offset"])
    if k == "dict": return DictCodec(values if values is not None else [], d.get("sql_type", "VARCHAR"))
    if k == "rank": return RankCodec(from_json(d["base"]), values if values is not None else [])
    raise ValueError(f"unknown column type {k}")


def pack_strings(values):
    """a list of str -> (utf-8 bytes, int64 offsets): a compact, pickle-free form for saving dictionaries"""
    enc = [s.encode("utf-8") for s in values]
    offs = np.zeros(len(enc) + 1, np.int64)
    if enc: offs[1:] = np.cumsum([len(b) for b in enc])
    return np.frombuffer(b"".join(enc), np.uint8).copy(), offs


def unpack_strings(blob, offs):
    b = bytes(np.asarray(blob, np.uint8)); o = np.asarray(offs, np.int64).tolist()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]
