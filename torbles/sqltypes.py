# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SQL data types of the SQL front end: the type of every expression, implicit and explicit casts, and how values are
represented in Python.

Values are Python objects (NULL is None): BOOLEAN bool, integers int (exact: arithmetic never wraps), DECIMAL
decimal.Decimal (exact), FLOAT / DOUBLE float, VARCHAR str, DATE datetime.date, TIME datetime.time, TIMESTAMP
datetime.datetime, INTERVAL Interval (months, days, microseconds; returned as datetime.timedelta like DuckDB), lists
for ARRAY_AGG. Where ISO/IEC 9075 leaves a rule to the implementation (the type of a division, the order of NULLs,
rounding of casts), the rule is DuckDB's, so that torbles and DuckDB can be compared value for value."""
import datetime as _dt, decimal, math, re
from decimal import Decimal

CTX = decimal.Context(prec=80, rounding=decimal.ROUND_HALF_UP, Emax=999999, Emin=-999999)   # exact for 38-digit values


class SqlError(Exception):
    """an error in a query (binding, types, a runtime error such as a failed cast)"""


class SqlType:
    """a data type; coll: the explicit collation of a character string (COLLATE), not part of the type's identity"""
    __slots__ = ("id", "p", "s", "child", "coll")

    def __init__(self, id, p=None, s=None, child=None, coll=None):
        self.id, self.p, self.s, self.child, self.coll = id, p, s, child, coll

    def __eq__(self, o): return isinstance(o, SqlType) and (self.id, self.p, self.s, self.child) == (o.id, o.p, o.s, o.child)
    def __hash__(self): return hash((self.id, self.p, self.s, self.child))
    def __repr__(self):
        if self.id == "DECIMAL": return f"DECIMAL({self.p},{self.s})"
        if self.id == "DECFLOAT": return f"DECFLOAT({self.p})"
        if self.id == "LIST": return f"{self.child!r}[]"
        if self.id == "MULTISET": return f"{self.child!r} MULTISET"
        if self.id == "ROW" and self.child is not None:
            return "ROW(" + ", ".join(f"{_field_name(n)} {t!r}" if n else repr(t) for n, t in self.child) + ")"
        if self.id in ("TIMESTAMP", "TIME", "TIMESTAMPTZ", "TIMETZ") and self.p is not None:   # fractional seconds precision
            base = "TIME" if self.id.startswith("TIME") and not self.id.startswith("TIMESTAMP") else "TIMESTAMP"
            return f"{base}({self.p})" + (" WITH TIME ZONE" if self.id.endswith("TZ") else "")
        if self.id == "INTERVAL" and self.s: return f"INTERVAL {self.s}"
        if self.id == "VARCHAR" and self.p is not None: return f"VARCHAR({self.p})" + (f" COLLATE {self.coll}" if self.coll else "")
        if self.id in _DISPLAY: return _DISPLAY[self.id]
        if self.coll: return f"{self.id} COLLATE {self.coll}"
        return self.id
    __str__ = __repr__

    # classification
    @property
    def integer(self): return self.id in INT_RANGE
    @property
    def numeric(self): return self.id in INT_RANGE or self.id in ("DECIMAL", "FLOAT", "DOUBLE", "DECFLOAT")
    @property
    def exact(self): return self.id in INT_RANGE or self.id == "DECIMAL"
    @property
    def floating(self): return self.id in ("FLOAT", "DOUBLE")
    @property
    def temporal(self): return self.id in ("DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ", "TIMETZ")
    @property
    def zoned(self): return self.id in ("TIMESTAMPTZ", "TIMETZ")


_DISPLAY = {"TIMESTAMPTZ": "TIMESTAMP WITH TIME ZONE", "TIMETZ": "TIME WITH TIME ZONE"}


INT_RANGE = {"TINYINT": (-2**7, 2**7 - 1), "SMALLINT": (-2**15, 2**15 - 1), "INTEGER": (-2**31, 2**31 - 1),
             "BIGINT": (-2**63, 2**63 - 1), "HUGEINT": (-2**127, 2**127 - 1), "UTINYINT": (0, 2**8 - 1),
             "USMALLINT": (0, 2**16 - 1), "UINTEGER": (0, 2**32 - 1), "UBIGINT": (0, 2**64 - 1), "UHUGEINT": (0, 2**128 - 1)}
INT_DIGITS = {"TINYINT": 3, "SMALLINT": 5, "INTEGER": 10, "BIGINT": 19, "HUGEINT": 38, "UTINYINT": 3, "USMALLINT": 5,
              "UINTEGER": 10, "UBIGINT": 20, "UHUGEINT": 38}
_INT_ORDER = ["TINYINT", "UTINYINT", "SMALLINT", "USMALLINT", "INTEGER", "UINTEGER", "BIGINT", "UBIGINT", "HUGEINT", "UHUGEINT"]

BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, HUGEINT = (SqlType(x) for x in ("BOOLEAN", "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT"))
UTINYINT, USMALLINT, UINTEGER, UBIGINT, UHUGEINT = (SqlType(x) for x in ("UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"))
FLOAT, DOUBLE, VARCHAR, DATE, TIME, TIMESTAMP, INTERVAL, NULL, BLOB = (SqlType(x) for x in ("FLOAT", "DOUBLE", "VARCHAR", "DATE", "TIME", "TIMESTAMP", "INTERVAL", "NULL", "BLOB"))
TIMESTAMPTZ, TIMETZ = SqlType("TIMESTAMPTZ"), SqlType("TIMETZ")
JSON = SqlType("JSON")
def DECIMAL(p=18, s=3): return SqlType("DECIMAL", min(int(p), 38), int(s))
def DECFLOAT(p=34): return SqlType("DECFLOAT", 16 if int(p) <= 16 else 34)


class DecFloat(Decimal):
    """a DECFLOAT value (ISO T076, IEEE 754 decimal floating point): a Decimal that knows it is one"""
    __slots__ = ()


_DF_CTX = {16: decimal.Context(prec=16, rounding=decimal.ROUND_HALF_EVEN, Emax=384, Emin=-383, capitals=1, traps=[decimal.InvalidOperation, decimal.Overflow]),
           34: decimal.Context(prec=34, rounding=decimal.ROUND_HALF_EVEN, Emax=6144, Emin=-6143, capitals=1, traps=[decimal.InvalidOperation, decimal.Overflow])}


def decfloat(v, p=34):
    """v rounded to DECFLOAT(p); past the exponent range, an error (ISO: numeric value out of range)"""
    ctx = _DF_CTX[16 if p <= 16 else 34]
    d = v if isinstance(v, Decimal) else Decimal(repr(v)) if isinstance(v, float) else Decimal(v)
    try: r = ctx.plus(d) if d.is_finite() else d
    except decimal.Overflow: raise SqlError(f"data exception: numeric value out of range for DECFLOAT({p})")
    if r.is_finite() and r != 0 and (r.adjusted() > ctx.Emax or r.adjusted() < ctx.Emin - ctx.prec + 1):
        raise SqlError(f"data exception: numeric value out of range for DECFLOAT({p})")
    return DecFloat(r)
def LIST(t, maxcard=None): return SqlType("LIST", p=maxcard, child=t)
def datetime_type(tid, p):
    """TIMESTAMP / TIME [WITH TIME ZONE] with p fractional digits of seconds (6, microseconds, is the default)"""
    if not 0 <= p <= 9: raise SqlError(f"invalid fractional seconds precision {p}")
    return SqlType(tid, p=p if p < 6 else None)


def _interval_number(v, t):
    """CAST(<interval of one field> AS <number>) (ISO 6.13): the value in that field"""
    q = (t.s or "").split("(")[0].strip().upper() if t is not None and t.id == "INTERVAL" else ""
    if q not in ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"):
        raise SqlError(f"Conversion Error: cannot cast {t} to a number (an interval of one field only)")
    if q == "YEAR": return v.months // 12 if v.months % 12 == 0 else Decimal(v.months) / 12
    if q == "MONTH": return v.months
    us = v.days * 86_400_000_000 + v.micros
    per = {"DAY": 86_400_000_000, "HOUR": 3_600_000_000, "MINUTE": 60_000_000, "SECOND": 1_000_000}[q]
    return us // per if us % per == 0 else Decimal(us) / per


def round_frac(v, p):
    """a datetime rounded to p fractional digits of seconds, halves away from zero (DuckDB's TIMESTAMP_S / _MS)"""
    if p is None or v is None: return v
    q = 10 ** (6 - p)
    if isinstance(v, _dt.time):
        us = ((v.hour * 60 + v.minute) * 60 + v.second) * 1_000_000 + v.microsecond
        us = (us + q // 2) // q * q % 86_400_000_000
        return _dt.time(us // 3_600_000_000, us // 60_000_000 % 60, us // 1_000_000 % 60, us % 1_000_000, tzinfo=v.tzinfo)
    base = _dt.datetime(1970, 1, 1, tzinfo=v.tzinfo)
    d = v - base
    us = (d.days * 86_400 + d.seconds) * 1_000_000 + d.microseconds
    r = (abs(us) + q // 2) // q * q
    return base + _dt.timedelta(microseconds=r if us >= 0 else -r)
def ROW_T(fields):
    """a row type (T051): fields [(name, type)]; a field of a row value constructor has the name ''"""
    return SqlType("ROW", child=tuple((str(n), t) for n, t in fields))


_TYPE_WORD = re.compile(r"(?i)(DOUBLE|CHARACTER|CHAR|NATIONAL|BINARY|TIMESTAMP|TIME|BIT|NCHAR|INTERVAL)")


def _field_name(n): return n if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) and not _TYPE_WORD.fullmatch(n) else '"' + n.replace('"', '""') + '"'


def _split_top(text):
    parts, d, cur, q = [], 0, "", False
    for ch in text:
        if ch == '"': q = not q
        if not q and ch in "([": d += 1
        if not q and ch in ")]": d -= 1
        if ch == "," and d == 0 and not q: parts.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts


def _row_type(text):
    """ROW(a INTEGER, b VARCHAR) / STRUCT(...) / TUPLE(INTEGER, ...) as written by DuckDB or in SQL"""
    m = re.fullmatch(r"\s*(ROW|STRUCT|TUPLE)\s*\((.*)\)\s*", text, re.S | re.I)
    if not m: return None
    fields = []
    for part in _split_top(m.group(2)):
        fq = re.fullmatch(r'"((?:[^"]|"")*)"\s+(.+)', part, re.S)
        fm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_$]*)\s+(.+)", part, re.S)
        if fq: fields.append((fq.group(1).replace('""', '"'), type_from_name(fq.group(2))))
        elif fm and not _TYPE_WORD.fullmatch(fm.group(1)): fields.append((fm.group(1), type_from_name(fm.group(2))))
        else: fields.append(("", type_from_name(part)))
    return ROW_T(fields)
def MULTISET(t): return SqlType("MULTISET", child=t)

_NAMES = {"BOOL": "BOOLEAN", "BOOLEAN": "BOOLEAN", "LOGICAL": "BOOLEAN", "TINYINT": "TINYINT", "INT1": "TINYINT",
          "SMALLINT": "SMALLINT", "INT2": "SMALLINT", "SHORT": "SMALLINT", "INTEGER": "INTEGER", "INT": "INTEGER", "INT4": "INTEGER",
          "SIGNED": "INTEGER", "INT32": "INTEGER", "BIGINT": "BIGINT", "INT8": "BIGINT", "LONG": "BIGINT", "INT64": "BIGINT",
          "HUGEINT": "HUGEINT", "INT128": "HUGEINT", "UTINYINT": "UTINYINT", "USMALLINT": "USMALLINT", "UINTEGER": "UINTEGER",
          "UBIGINT": "UBIGINT", "UHUGEINT": "UHUGEINT", "REAL": "FLOAT", "FLOAT4": "FLOAT", "FLOAT": "FLOAT", "DOUBLE": "DOUBLE",
          "FLOAT8": "DOUBLE", "DOUBLE PRECISION": "DOUBLE", "VARCHAR": "VARCHAR", "TEXT": "VARCHAR", "STRING": "VARCHAR",
          "CHAR": "VARCHAR", "BPCHAR": "VARCHAR", "CHARACTER": "VARCHAR", "CHARACTER VARYING": "VARCHAR", "NVARCHAR": "VARCHAR",
          "DATE": "DATE", "TIME": "TIME", "TIMESTAMP": "TIMESTAMP", "DATETIME": "TIMESTAMP", "TIMESTAMP WITHOUT TIME ZONE": "TIMESTAMP",
          "TIME WITHOUT TIME ZONE": "TIME", "TIMESTAMPTZ": "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
          "TIMETZ": "TIMETZ", "TIME WITH TIME ZONE": "TIMETZ", "TIMESTAMP_US": "TIMESTAMP",
          "TIMESTAMP_S": "TIMESTAMP", "TIMESTAMP_MS": "TIMESTAMP", "TIMESTAMP_NS": "TIMESTAMP",
          "INTERVAL": "INTERVAL", "BLOB": "BLOB", "BYTEA": "BLOB", "VARBINARY": "BLOB", "NULL": "NULL", "BINARY": "BLOB",
          "BINARY VARYING": "BLOB", "BINARY LARGE OBJECT": "BLOB", "CLOB": "VARCHAR", "NCLOB": "VARCHAR", "NCHAR": "VARCHAR",
          "CHARACTER LARGE OBJECT": "VARCHAR", "CHAR LARGE OBJECT": "VARCHAR", "NATIONAL CHARACTER": "VARCHAR",
          "NATIONAL CHAR": "VARCHAR", "NATIONAL CHARACTER VARYING": "VARCHAR", "NATIONAL CHAR VARYING": "VARCHAR",
          "NCHAR VARYING": "VARCHAR", "NATIONAL CHARACTER LARGE OBJECT": "VARCHAR", "NCHAR LARGE OBJECT": "VARCHAR",
          "CHAR VARYING": "VARCHAR", "JSON": "JSON"}


def type_from_name(name, args=()):
    """a type from its SQL name and parameters: DECIMAL(10, 2), NUMERIC, VARCHAR(20), DOUBLE PRECISION, ..."""
    raw = str(name).strip()
    if raw.endswith("[]"): return LIST(type_from_name(raw[:-2]))
    rt = _row_type(raw)
    if rt is not None: return rt
    u = re.sub(r"\s+", " ", raw.upper())
    u = re.sub(r"\s+COLLATE\s+\S+$", "", u)
    m = re.fullmatch(r"__(TIMESTAMP|TIMESTAMPTZ|TIME|TIMETZ)_P(\d+)", u) or \
        re.fullmatch(r"(TIMESTAMP|TIME|TIMESTAMPTZ|TIMETZ) ?\( ?(\d+) ?\)( WITH TIME ZONE| WITHOUT TIME ZONE)?", u)
    if m:                                                  # TIMESTAMP(p), TIME(p) [WITH TIME ZONE]: fractional digits kept
        tid = m.group(1) + ("TZ" if m.lastindex == 3 and m.group(3) == " WITH TIME ZONE" and not m.group(1).endswith("TZ") else "")
        return datetime_type(tid, int(m.group(2)))
    if args and u in ("TIMESTAMP", "TIME", "TIMESTAMPTZ", "TIMETZ", "TIMESTAMP WITH TIME ZONE", "TIME WITH TIME ZONE"):
        return datetime_type(_NAMES[u], int(args[0]))
    if u in ("TIMESTAMP_S", "TIMESTAMP_MS"): return datetime_type("TIMESTAMP", 0 if u == "TIMESTAMP_S" else 3)
    if u.endswith("[]"): return LIST(type_from_name(u[:-2]))
    if u.endswith(" MULTISET"): return MULTISET(type_from_name(u[:-9]))
    m = re.fullmatch(r"DECFLOAT\s*(?:\(\s*(\d+)\s*\))?", u)
    if m: return DECFLOAT(int(args[0]) if args else int(m.group(1) or 34))
    m = re.fullmatch(r"(DECIMAL|NUMERIC|DEC)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?", u)
    if m or u in ("DECIMAL", "NUMERIC", "DEC"):
        a = [int(x) for x in args] if args else [int(x) for x in (m.group(2), m.group(3)) if x is not None] if m else []
        if not a: return DECIMAL(18, 3)
        return DECIMAL(a[0], a[1] if len(a) > 1 else 0)
    u = re.sub(r"\s*\(\s*\d+\s*\)", "", u)                  # VARCHAR(20), CHAR(3), TIMESTAMP(6): the length is not kept
    u = re.sub(r"\s*\(.*\)$", "", u)
    if u in _NAMES: return SqlType(_NAMES[u])
    raise SqlError(f"type {name} is not supported")


def from_codec(cd):
    """the SQL type of a stored torbles column"""
    if cd is None: return BIGINT
    st = getattr(cd, "sql_type", "") or ""
    try: return type_from_name(st)
    except SqlError: pass
    k = getattr(cd, "kind", "")
    return {"bool": BOOLEAN, "date": DATE, "timestamp": TIMESTAMP, "dict": VARCHAR}.get(k, BIGINT)


# ------------------------------------------------------------------ collations
def _noaccent(x):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", x) if not unicodedata.combining(c))


def _nfc(x):
    import unicodedata
    return unicodedata.normalize("NFC", x)


_COLLATIONS = {"BINARY": None, "C": None, "POSIX": None, "UCS_BASIC": None, "SQL_TEXT": None, "SQL_IDENTIFIER": None,
               "SQL_CHARACTER": None, "NOCASE": str.lower, "NOACCENT": _noaccent, "NFC": _nfc}


_UCA = []


def _uca_key(v):
    """the Unicode Collation Algorithm's sort key (DUCET, all levels) as a string whose code point order is the key's"""
    if not _UCA:
        try: import pyuca
        except ImportError:
            raise SqlError("COLLATE UNICODE needs the pyuca package (pip install 'torbles[unicode]')")
        _UCA.append(pyuca.Collator())
    return "".join(chr(w + 1) for w in _UCA[0].sort_key(v))


def collation(name):
    """a collation's name -> its key function on strings (None: code point order). NOCASE, NOACCENT, NFC and their
    combinations (NOCASE.NOACCENT) as in DuckDB; UCS_BASIC (ISO), BINARY, C, POSIX: code point order; UNICODE (UCA):
    the Unicode Collation Algorithm with the default table, after any of the others it is combined with"""
    parts = [x.strip().upper() for x in str(name).split(".") if x.strip()]
    fns, uca = [], False
    for x in parts:
        if x in ("UNICODE", "UCA"): uca = True; continue
        if x not in _COLLATIONS:
            raise SqlError(f"unknown collation {name!r} (NOCASE, NOACCENT, NFC, UNICODE, BINARY, UCS_BASIC, C, POSIX)")
        if _COLLATIONS[x] is not None: fns.append(_COLLATIONS[x])
    if uca: _uca_key(""); fns.append(_uca_key)
    if not fns: return None
    def key(v, fns=tuple(fns)):
        for f in fns: v = f(v)
        return v
    return key


def coll_key(t):
    """the comparison key function of a type's collation (None: values compare as they are)"""
    c = getattr(t, "coll", None)
    return collation(c) if c else None


def with_coll(t, name):
    return SqlType(t.id, t.p, t.s, t.child, name.upper())


# ------------------------------------------------------------------ intervals
class Interval:
    """an SQL interval: months, days and microseconds kept apart (a month is not a fixed number of days)"""
    __slots__ = ("months", "days", "micros")

    def __init__(self, months=0, days=0, micros=0): self.months, self.days, self.micros = int(months), int(days), int(micros)
    def _key(self): return self.months * 30 * 86_400_000_000 + self.days * 86_400_000_000 + self.micros     # DuckDB's order
    def __eq__(self, o): return isinstance(o, Interval) and self._key() == o._key()
    def __lt__(self, o): return self._key() < o._key()
    def __le__(self, o): return self._key() <= o._key()
    def __gt__(self, o): return self._key() > o._key()
    def __ge__(self, o): return self._key() >= o._key()
    def __hash__(self): return hash(self._key())
    def __neg__(self): return Interval(-self.months, -self.days, -self.micros)
    def __add__(self, o): return Interval(self.months + o.months, self.days + o.days, self.micros + o.micros)
    def __sub__(self, o): return self + (-o)
    def __repr__(self): return f"Interval({self.months}, {self.days}, {self.micros})"
    def to_timedelta(self): return _dt.timedelta(days=self.months * 30 + self.days, microseconds=self.micros)
    def __str__(self):
        parts = []
        y, mo = divmod(abs(self.months), 12); sg = "-" if self.months < 0 else ""
        if y: parts.append(f"{sg}{y} year" + ("s" if y != 1 else ""))
        if mo: parts.append(f"{sg}{mo} month" + ("s" if mo != 1 else ""))
        if self.days: parts.append(f"{self.days} day" + ("s" if abs(self.days) != 1 else ""))
        if self.micros or not parts:
            us = abs(self.micros); h, us = divmod(us, 3_600_000_000); m, us = divmod(us, 60_000_000); s, us = divmod(us, 1_000_000)
            t = f"{'-' if self.micros < 0 else ''}{h:02d}:{m:02d}:{s:02d}" + (f".{us:06d}" if us else "")
            parts.append(t)
        return " ".join(parts)


def add_months(d, n):
    """date or timestamp plus n months; the day is clamped to the last day of the target month"""
    y, m = divmod(d.month - 1 + n, 12); y += d.year; m += 1
    import calendar
    day = min(d.day, calendar.monthrange(y, m)[1])
    return d.replace(year=y, month=m, day=day)


def add_interval(d, iv, sign=1):
    """date / timestamp +- interval (a DATE operand gives a TIMESTAMP, as in DuckDB)"""
    if not isinstance(d, _dt.datetime):
        d = _dt.datetime(d.year, d.month, d.day)
    if iv.months: d = add_months(d, sign * iv.months)
    return d + _dt.timedelta(days=sign * iv.days, microseconds=sign * iv.micros)


# ------------------------------------------------------------------ unification (the common supertype)
def _int_supertype(a, b):
    if a.id == b.id: return a
    ia, ib = _INT_ORDER.index(a.id), _INT_ORDER.index(b.id)
    signed_a, signed_b = not a.id.startswith("U"), not b.id.startswith("U")
    if signed_a == signed_b: return a if ia > ib else b
    s, u = (a, b) if signed_a else (b, a)                 # signed with unsigned: a signed type wide enough for both
    for name in ("SMALLINT", "INTEGER", "BIGINT", "HUGEINT"):
        if INT_RANGE[name][1] >= INT_RANGE[u.id][1] and INT_RANGE[name][0] <= INT_RANGE[s.id][0]: return SqlType(name)
    return HUGEINT


def as_decimal(t):
    """an exact numeric type as DECIMAL(p, s)"""
    if t.id == "DECIMAL": return t
    return DECIMAL(INT_DIGITS[t.id], 0)


def supertype(a, b, literal_a=False, literal_b=False):
    """the type both a and b convert to implicitly (CASE branches, COALESCE, comparisons, set operations); a string
    literal takes the other side's type"""
    if a == b:
        if a.id == "VARCHAR" and a.coll != b.coll:        # an explicit collation wins (ISO 9.5); two differ: a conflict
            if a.coll and b.coll: raise SqlError(f"collation conflict: {a.coll} and {b.coll}")
            return a if a.coll else b
        return a
    if a.id == "NULL": return b
    if b.id == "NULL": return a
    if literal_a and a.id == "VARCHAR" and b.id != "VARCHAR": return b
    if literal_b and b.id == "VARCHAR" and a.id != "VARCHAR": return a
    if a.integer and b.integer: return _int_supertype(a, b)
    if a.numeric and b.numeric and "DECFLOAT" in (a.id, b.id):
        return DECFLOAT(max(x.p for x in (a, b) if x.id == "DECFLOAT") if not (a.floating or b.floating) else 34)
    if a.numeric and b.numeric:
        if a.floating or b.floating: return DOUBLE
        da, db = as_decimal(a), as_decimal(b)
        s = max(da.s, db.s); w = max(da.p - da.s, db.p - db.s)
        return DECIMAL(min(38, w + s), s)
    if a.id == b.id == "INTERVAL": return INTERVAL         # two qualifiers: an interval of any fields
    if a.id == b.id and a.temporal:                        # the same datetime type, two precisions: the finer one
        return SqlType(a.id, p=None if a.p is None or b.p is None else max(a.p, b.p))
    if {a.id, b.id} == {"DATE", "TIMESTAMP"}: return TIMESTAMP
    if {a.id, b.id} in ({"DATE", "TIMESTAMPTZ"}, {"TIMESTAMP", "TIMESTAMPTZ"}): return TIMESTAMPTZ
    if {a.id, b.id} == {"TIME", "TIMETZ"}: return TIMETZ
    if a.id == "VARCHAR" or b.id == "VARCHAR":
        c = (a.coll if a.id == "VARCHAR" else None) or (b.coll if b.id == "VARCHAR" else None)
        return SqlType("VARCHAR", coll=c) if c else VARCHAR
    if {a.id, b.id} == {"JSON", "VARCHAR"}: return VARCHAR
    if a.id == "LIST" and b.id == "LIST": return LIST(supertype(a.child, b.child))
    if a.id == "MULTISET" and b.id == "MULTISET": return MULTISET(supertype(a.child, b.child))
    if a.id == "ROW" and b.id == "ROW" and a.child is not None and b.child is not None:
        if len(a.child) != len(b.child): raise SqlError(f"cannot combine row types of degrees {len(a.child)} and {len(b.child)}")
        return ROW_T([(na or nb, supertype(ta, tb)) for (na, ta), (nb, tb) in zip(a.child, b.child)])
    if a.id == "BOOLEAN" and b.numeric: return b
    if b.id == "BOOLEAN" and a.numeric: return a
    raise SqlError(f"cannot combine types {a} and {b}")


def supertype_all(types, literals=None):
    out = NULL
    literals = literals or [False] * len(types)
    lit_out = True
    for t, lit in zip(types, literals):
        out = supertype(out, t, lit_out, lit); lit_out = lit_out and lit
    return out


# ------------------------------------------------------------------ casts
def _dec_quant(v, s):
    """v rounded half away from zero to s decimal places (DuckDB's rounding of decimals)"""
    return v.quantize(Decimal(1).scaleb(-s), rounding=decimal.ROUND_HALF_UP, context=CTX)


def _check_int(v, t, err):
    lo, hi = INT_RANGE[t.id]
    if not lo <= v <= hi:
        if err: raise SqlError(f"Conversion Error: {v} is out of range for {t}")
        return None
    return v


def _round_half_even_float(f):
    return int(round(f))                                 # Python's round(): half to even, like nearbyint


def _parse_int_str(x):
    s = x.strip()
    if re.fullmatch(r"[+-]?\d+", s): return int(s)
    try: d = Decimal(s)
    except decimal.InvalidOperation: raise SqlError(f"Conversion Error: could not convert string '{x}' to an integer")
    if not d.is_finite(): raise SqlError(f"Conversion Error: could not convert string '{x}' to an integer")
    return int(d.to_integral_value(rounding=decimal.ROUND_HALF_UP))


_DATE_RE = re.compile(r"\s*(-?\d{1,6})-(\d{1,2})-(\d{1,2})(?:\s*\(BC\))?\s*$")
_ZONE = r"(Z|UTC|GMT|[+-]\d{1,2}(?::?\d{2})?|[A-Za-z][A-Za-z0-9_+\-]*(?:/[A-Za-z0-9_+\-]+)+)"
_TS_RE = re.compile(r"\s*(-?\d{1,6})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?)?\s*" + _ZONE + r"?\s*$")
_TIME_RE = re.compile(r"\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?\s*" + _ZONE + r"?\s*$")


# ------------------------------------------------------------------ time zones
# A value WITH TIME ZONE is an instant together with its time zone displacement (ISO/IEC 9075-2 4.6): a Python
# datetime / time whose tzinfo is a fixed offset. Values WITHOUT TIME ZONE meet zoned ones through the SQL-session
# time zone (SET TIME ZONE; by default the machine's zone, like DuckDB's TimeZone setting).
SESSION = {"zone": None}


def _default_zone():
    import os
    for name in (os.environ.get("TZ", "").lstrip(":"),):
        if name:
            try: return zone_of(name)
            except SqlError: pass
    try:
        link = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in link: return zone_of(link.split("zoneinfo/", 1)[1])
    except OSError: pass
    return _dt.timezone.utc


def session_zone():
    if SESSION["zone"] is None: SESSION["zone"] = _default_zone()
    return SESSION["zone"]


def zone_of(spec):
    """a time zone from 'UTC', 'Z', '+02:00', '-0530', 'Europe/Paris', or an interval (a displacement)"""
    if isinstance(spec, _dt.tzinfo): return spec
    if isinstance(spec, Interval):
        if spec.months or spec.days: raise SqlError("a time zone displacement must be an interval of hours and minutes")
        us = spec.micros
        if abs(us) > 18 * 3_600_000_000 or us % 60_000_000: raise SqlError(f"invalid time zone displacement {spec}")
        return _dt.timezone(_dt.timedelta(microseconds=us))
    s = str(spec).strip()
    if s.upper() in ("Z", "UTC", "GMT", "ZULU", "UCT", "UNIVERSAL", "ETC/UTC"): return _dt.timezone.utc
    m = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?", s, re.I)
    if m:
        h, mi = int(m.group(2)), int(m.group(3) or 0)
        if h > 18 or mi > 59: raise SqlError(f"invalid time zone displacement {s!r}")
        off = _dt.timedelta(hours=h, minutes=mi)
        return _dt.timezone(-off if m.group(1) == "-" else off)
    try:
        import zoneinfo
        return zoneinfo.ZoneInfo(s)
    except Exception:
        raise SqlError(f"unknown time zone {s!r}")


def fixed(aware):
    """a zoned value with its displacement as a fixed offset (named zones resolve to their offset at that instant)"""
    if isinstance(aware.tzinfo, _dt.timezone): return aware
    return aware.replace(tzinfo=_dt.timezone(aware.utcoffset()))


def localize(naive, zone=None):
    """a TIMESTAMP (or TIME) taken as local time in zone (default: the session's) -> its zoned value"""
    z = zone_of(zone) if zone is not None else session_zone()
    if isinstance(naive, _dt.time):
        off = _dt.datetime.combine(_dt.date.today(), naive).replace(tzinfo=z).utcoffset()
        return naive.replace(tzinfo=_dt.timezone(off))
    return fixed(naive.replace(tzinfo=z))


def in_zone(aware, zone=None):
    """the same instant with the displacement of zone (default: the session's)"""
    z = zone_of(zone) if zone is not None else session_zone()
    if isinstance(aware, _dt.time):
        d = _dt.datetime.combine(_dt.date(2000, 1, 3), aware.replace(tzinfo=None)) - aware.utcoffset()
        d = d.replace(tzinfo=_dt.timezone.utc).astimezone(z)
        return d.timetz().replace(tzinfo=_dt.timezone(d.utcoffset()))
    return fixed(aware.astimezone(z))


def local(aware):
    """a zoned value as local time of the session (TIMESTAMP / TIME WITHOUT TIME ZONE)"""
    return in_zone(aware).replace(tzinfo=None)


def parse_date(s):
    m = _DATE_RE.match(s) or _TS_RE.match(s)
    if not m: raise SqlError(f"Conversion Error: invalid date field format: \"{s}\"")
    try: return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError: raise SqlError(f"Conversion Error: date field value out of range: \"{s}\"")


def _parse_ts(s):
    m = _TS_RE.match(s)
    if not m: raise SqlError(f"Conversion Error: invalid timestamp field format: \"{s}\"")
    frac = (m.group(7) or "").ljust(6, "0")[:6]
    try:
        v = _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0), int(m.group(5) or 0),
                         int(m.group(6) or 0), int(frac or 0))
    except ValueError: raise SqlError(f"Conversion Error: timestamp field value out of range: \"{s}\"")
    return v, m.group(8)


def has_zone(s):
    """does a datetime string carry a time zone (ISO: its literal is WITH TIME ZONE)?"""
    m = _TS_RE.match(s) or _TIME_RE.match(s)
    return bool(m and m.group(m.re.groups))


def parse_timestamp(s):
    """TIMESTAMP (WITHOUT TIME ZONE) from text; a time zone in the text is ignored (DuckDB, PostgreSQL)"""
    return _parse_ts(s)[0]


def parse_timestamptz(s):
    v, z = _parse_ts(s)
    if z: return fixed(v.replace(tzinfo=zone_of(z)))
    return localize(v)


def _parse_tm(s):
    m = _TIME_RE.match(s)
    if not m: raise SqlError(f"Conversion Error: invalid time field format: \"{s}\"")
    frac = (m.group(4) or "").ljust(6, "0")[:6]
    try: v = _dt.time(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), int(frac or 0))
    except ValueError: raise SqlError(f"Conversion Error: time field value out of range: \"{s}\"")
    return v, m.group(5)


def parse_time(s): return _parse_tm(s)[0]


def parse_timetz(s):
    v, z = _parse_tm(s)
    if z:
        zz = zone_of(z)
        return v.replace(tzinfo=zz if isinstance(zz, _dt.timezone) else _dt.timezone(_dt.datetime.combine(_dt.date.today(), v).replace(tzinfo=zz).utcoffset()))
    return localize(v)


def fmt_offset(off):
    """+HH or +HH:MM (DuckDB's and ISO's text of a displacement)"""
    us = off // _dt.timedelta(microseconds=1); sg = "-" if us < 0 else "+"; us = abs(us)
    h, mi = us // 3_600_000_000, us // 60_000_000 % 60
    return f"{sg}{h:02d}" + (f":{mi:02d}" if mi else "")


_IV_UNITS = {"year": (12, 0, 0), "years": (12, 0, 0), "y": (12, 0, 0), "month": (1, 0, 0), "months": (1, 0, 0), "mon": (1, 0, 0),
             "mons": (1, 0, 0), "day": (0, 1, 0), "days": (0, 1, 0), "d": (0, 1, 0), "week": (0, 7, 0), "weeks": (0, 7, 0),
             "hour": (0, 0, 3_600_000_000), "hours": (0, 0, 3_600_000_000), "h": (0, 0, 3_600_000_000),
             "minute": (0, 0, 60_000_000), "minutes": (0, 0, 60_000_000), "min": (0, 0, 60_000_000), "m": (0, 0, 60_000_000),
             "second": (0, 0, 1_000_000), "seconds": (0, 0, 1_000_000), "sec": (0, 0, 1_000_000), "s": (0, 0, 1_000_000),
             "millisecond": (0, 0, 1000), "milliseconds": (0, 0, 1000), "ms": (0, 0, 1000),
             "microsecond": (0, 0, 1), "microseconds": (0, 0, 1), "us": (0, 0, 1),
             "decade": (120, 0, 0), "decades": (120, 0, 0), "century": (1200, 0, 0), "centuries": (1200, 0, 0),
             "millennium": (12000, 0, 0), "millennia": (12000, 0, 0), "quarter": (3, 0, 0), "quarters": (3, 0, 0)}


def parse_interval(s):
    """'1 year 2 months 3 days 04:05:06', '3 days', '04:05:06', 'P1Y2M' is not supported"""
    toks = s.strip().lower().replace(",", " ").split()
    mo = d = us = 0; i = 0
    while i < len(toks):
        t = toks[i]
        m = re.fullmatch(r"([+-]?)(\d+):(\d{1,2})(?::(\d{1,2})(?:\.(\d{1,6}))?)?", t)
        if m:
            sg = -1 if m.group(1) == "-" else 1
            us += sg * (int(m.group(2)) * 3_600_000_000 + int(m.group(3)) * 60_000_000 + int(m.group(4) or 0) * 1_000_000
                        + int((m.group(5) or "").ljust(6, "0") or 0)); i += 1; continue
        try: n = Decimal(t)
        except decimal.InvalidOperation: raise SqlError(f"Conversion Error: could not convert string '{s}' to INTERVAL")
        unit = toks[i + 1] if i + 1 < len(toks) else None
        if unit is None or unit not in _IV_UNITS: raise SqlError(f"Conversion Error: could not convert string '{s}' to INTERVAL")
        a, b, c = _IV_UNITS[unit]
        if a and n != int(n): raise SqlError(f"Conversion Error: could not convert string '{s}' to INTERVAL")
        mo += int(n) * a; d += int(n) * b if b and n == int(n) else 0
        if b and n != int(n): us += int(n * b * 86_400_000_000)
        us += int(n * c); i += 2
    return Interval(mo, d, us)


def fmt_double(f):
    """DOUBLE as text, like DuckDB: the shortest round-trip form; exponent past 1e15 or below 1e-5"""
    if f != f: return "nan"
    if f in (math.inf, -math.inf): return "inf" if f > 0 else "-inf"
    if f == int(f) and abs(f) < 1e15: return f"{int(f)}.0" if f or not math.copysign(1, f) < 0 else "-0.0"
    r = repr(f)
    if "e" in r:
        mant, ex = r.split("e"); ex = int(ex)
        return f"{mant}e{'+' if ex >= 0 else '-'}{abs(ex):02d}"
    return r


def to_text(v, t):
    """a value as VARCHAR (CAST(x AS VARCHAR), ||)"""
    if v is None: return None
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, int): return str(v)
    if isinstance(v, DecFloat) or (t is not None and t.id == "DECFLOAT" and isinstance(v, Decimal)):
        return str(Decimal(v)).replace("Infinity", "Infinity")         # DECFLOAT: its own text (1.5E+20, NaN, -Infinity)
    if isinstance(v, Decimal):
        if t is not None and t.id == "DECIMAL": v = _dec_quant(v, t.s)
        return format(v, "f")
    if isinstance(v, float): return fmt_double(v)
    if isinstance(v, _dt.datetime):
        s = f"{v.year:04d}-{v.month:02d}-{v.day:02d} {v.hour:02d}:{v.minute:02d}:{v.second:02d}"
        if v.microsecond: s += f".{v.microsecond:06d}".rstrip("0")
        if v.tzinfo is not None: s += fmt_offset(v.utcoffset())
        return s
    if isinstance(v, _dt.date): return v.isoformat()
    if isinstance(v, _dt.time):
        s = v.strftime("%H:%M:%S")
        if v.microsecond: s += f".{v.microsecond:06d}".rstrip("0")
        if v.tzinfo is not None: s += fmt_offset(v.utcoffset())
        return s
    if isinstance(v, Interval): return str(v)
    if isinstance(v, tuple) or (isinstance(v, list) and t is not None and t.id == "ROW"):
        fs = t.child if t is not None and t.id == "ROW" and t.child is not None else [("", None)] * len(v)
        txt = ["NULL" if x is None else to_text(x, ft) for x, (_, ft) in zip(v, fs)]
        if all(n for n, _ in fs): return "{" + ", ".join(f"'{n}': {x}" for (n, _), x in zip(fs, txt)) + "}"
        return "(" + ", ".join(txt) + ")"
    if isinstance(v, list): return "[" + ", ".join("NULL" if x is None else to_text(x, t.child if t is not None and t.id in ("LIST", "MULTISET") else None) for x in v) + "]"
    if isinstance(v, (bytes, bytearray)): return blob_text(v)
    return str(v)


def blob_text(b):
    """a binary string as text (DuckDB): printable ASCII as it is, every other octet as \\xHH"""
    return "".join(chr(c) if 32 <= c < 127 and c != 92 else f"\\x{c:02X}" for c in b)


def parse_blob(s):
    """text -> a binary string: \\xHH escapes and ASCII characters (DuckDB's VARCHAR -> BLOB)"""
    out, i = bytearray(), 0
    while i < len(s):
        if s[i] == "\\" and i + 4 <= len(s) and s[i + 1] in "xX" and re.fullmatch(r"[0-9A-Fa-f]{2}", s[i + 2:i + 4]):
            out.append(int(s[i + 2:i + 4], 16)); i += 4; continue
        if ord(s[i]) > 127: raise SqlError(f"Conversion Error: non-ASCII character {s[i]!r} in a VARCHAR -> BLOB conversion (use \\xHH)")
        out.append(ord(s[i])); i += 1
    return bytes(out)


def cast_value(v, src, dst, err=True):
    """CAST(v AS dst) for a value of type src (TRY_CAST: err=False returns None on failure)"""
    if v is None: return None
    try:
        return _cast(v, src, dst, err)
    except SqlError:
        if err: raise
        return None
    except (ValueError, OverflowError, decimal.InvalidOperation, ArithmeticError) as e:
        if err: raise SqlError(f"Conversion Error: cannot cast {v!r} to {dst}: {e}")
        return None


def _cast(v, src, dst, err):
    d = dst.id
    if src is not None and src == dst: return v
    if dst.temporal and dst.p is not None:
        return round_frac(_cast(v, src, SqlType(d), err), dst.p)
    if d == "ROW":                                         # a row: field by field, same degree (ISO 6.13)
        if not isinstance(v, (list, tuple)) or (src is not None and src.id not in ("ROW", "NULL")): raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
        if dst.child is None: return tuple(v)
        if len(v) != len(dst.child): raise SqlError(f"Conversion Error: cannot cast a row of degree {len(v)} to {dst}")
        st = src.child if src is not None and src.id == "ROW" and src.child is not None else [("", None)] * len(v)
        return tuple(None if x is None else _cast(x, s_, t_, err) for x, (_, s_), (_, t_) in zip(v, st, dst.child))
    if d == "VARCHAR":
        r = to_text(v, src)                            # to VARCHAR(n): the first n characters (ISO 6.13; a warning, not an error)
        return r[:dst.p] if dst.p is not None and r is not None and len(r) > dst.p else r
    if isinstance(v, Interval) and (dst.integer or d in ("DECIMAL", "DOUBLE", "FLOAT", "DECFLOAT")):
        v = _interval_number(v, src)                       # a single-field interval: the value of its field
        src = DECIMAL(38, 6) if isinstance(v, Decimal) else BIGINT
        return _cast(v, src, dst, err)
    if dst.integer:
        if isinstance(v, bool): x = int(v)
        elif isinstance(v, int): x = v
        elif isinstance(v, Decimal): x = int(v.to_integral_value(rounding=decimal.ROUND_HALF_UP))
        elif isinstance(v, float):
            if not math.isfinite(v): raise SqlError(f"Conversion Error: {v} cannot be cast to {dst}")
            x = _round_half_even_float(v)
        elif isinstance(v, str): x = _parse_int_str(v)
        else: raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
        return _check_int(x, dst, True)
    if d == "DECIMAL":
        if isinstance(v, bool): x = Decimal(int(v))
        elif isinstance(v, (int, Decimal)): x = Decimal(v)
        elif isinstance(v, float):
            if not math.isfinite(v): raise SqlError(f"Conversion Error: {v} cannot be cast to {dst}")
            x = Decimal(repr(v))
        elif isinstance(v, str):
            try: x = Decimal(v.strip())
            except decimal.InvalidOperation: raise SqlError(f"Conversion Error: could not convert string '{v}' to {dst}")
        else: raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
        q = _dec_quant(x, dst.s)
        if abs(q) >= Decimal(10) ** (dst.p - dst.s): raise SqlError(f"Conversion Error: {v} does not fit in {dst}")
        return q
    if d == "DECFLOAT":
        if isinstance(v, bool): raise SqlError(f"Conversion Error: cannot cast BOOLEAN to {dst}")
        if isinstance(v, (int, Decimal, float)): return decfloat(v, dst.p)
        if isinstance(v, str):
            t = v.strip()
            if re.fullmatch(r"[+-]?(inf|infinity|nan|snan)", t, re.I): return DecFloat(t.replace("inf", "Infinity").replace("INF", "Infinity") if t.lower() in ("inf", "+inf", "-inf") else t)
            try: return decfloat(Decimal(t), dst.p)
            except decimal.InvalidOperation: raise SqlError(f"Conversion Error: could not convert string '{v}' to {dst}")
        raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
    if d in ("DOUBLE", "FLOAT"):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("nan", "+nan", "-nan"): return math.nan
            if s in ("inf", "+inf", "infinity", "+infinity"): return math.inf
            if s in ("-inf", "-infinity"): return -math.inf
            try: return float(Decimal(s))
            except decimal.InvalidOperation: raise SqlError(f"Conversion Error: could not convert string '{v}' to {dst}")
        if isinstance(v, (bool, int, Decimal, float)):
            f = float(v)
            if d == "FLOAT":
                import struct
                f = struct.unpack("f", struct.pack("f", f))[0]
            return f
        raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
    if d == "BOOLEAN":
        if isinstance(v, bool): return v
        if isinstance(v, (int, Decimal, float)): return v != 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "t", "1", "yes", "y", "on"): return True
            if s in ("false", "f", "0", "no", "n", "off"): return False
            raise SqlError(f"Conversion Error: could not convert string '{v}' to BOOLEAN")
        raise SqlError(f"Conversion Error: cannot cast {src} to BOOLEAN")
    if d == "DATE":
        if isinstance(v, _dt.datetime): return (local(v) if v.tzinfo is not None else v).date()
        if isinstance(v, _dt.date): return v
        if isinstance(v, str): return parse_date(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to DATE")
    if d == "TIMESTAMP":
        if isinstance(v, _dt.datetime): return local(v) if v.tzinfo is not None else v
        if isinstance(v, _dt.date): return _dt.datetime(v.year, v.month, v.day)
        if isinstance(v, str): return parse_timestamp(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to TIMESTAMP")
    if d == "TIMESTAMPTZ":
        if isinstance(v, _dt.datetime): return fixed(v) if v.tzinfo is not None else localize(v)
        if isinstance(v, _dt.date): return localize(_dt.datetime(v.year, v.month, v.day))
        if isinstance(v, str): return parse_timestamptz(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to TIMESTAMP WITH TIME ZONE")
    if d == "TIME":
        if isinstance(v, _dt.datetime): return (local(v) if v.tzinfo is not None else v).time()
        if isinstance(v, _dt.time): return local(v) if v.tzinfo is not None else v
        if isinstance(v, str): return parse_time(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to TIME")
    if d == "TIMETZ":
        if isinstance(v, _dt.datetime):
            a = fixed(v) if v.tzinfo is not None else localize(v)
            return a.timetz()
        if isinstance(v, _dt.time): return fixed(v) if v.tzinfo is not None else localize(v)
        if isinstance(v, str): return parse_timetz(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to TIME WITH TIME ZONE")
    if d == "INTERVAL":
        if isinstance(v, Interval): return v
        if isinstance(v, str): return parse_interval(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to INTERVAL")
    if d == "JSON":
        if src is not None and src.id == "JSON": return v
        import sqljson
        if isinstance(v, str):
            try: sqljson.parse(v)
            except sqljson.JErr as e: raise SqlError(f"Conversion Error: {e}")
            return v
        try: return sqljson.dumps(sqljson.from_sql(v, src))
        except sqljson.JErr as e: raise SqlError(f"Conversion Error: {e}")
    if d == "BLOB":
        if isinstance(v, (bytes, bytearray)): return bytes(v)
        if isinstance(v, str): return parse_blob(v)
        raise SqlError(f"Conversion Error: cannot cast {src} to BLOB")
    if d in ("LIST", "MULTISET"):
        if isinstance(v, list):
            out = [cast_value(x, src.child if src is not None and src.id in ("LIST", "MULTISET") else None, dst.child, err) for x in v]
            if d == "LIST" and dst.p is not None and len(out) > dst.p:
                raise SqlError(f"data exception: array data, right truncation ({len(out)} elements for {dst.child} ARRAY[{dst.p}])")
            return out
        raise SqlError(f"Conversion Error: cannot cast {src} to {dst}")
    if d == "NULL": return None
    raise SqlError(f"cast to {dst} is not supported")


def py_type(v):
    """the SQL type of a Python value (constants of the Python API)"""
    if v is None: return NULL
    if isinstance(v, bool): return BOOLEAN
    if isinstance(v, int):
        for name in ("INTEGER", "BIGINT", "HUGEINT", "UHUGEINT"):
            lo, hi = INT_RANGE[name]
            if lo <= v <= hi: return SqlType(name)
        raise SqlError(f"integer {v} is too large")
    if isinstance(v, DecFloat): return DECFLOAT(34)
    if isinstance(v, Decimal):
        t = v.as_tuple(); s = max(0, -t.exponent); p = max(len(t.digits), s + 1)
        return DECIMAL(max(p, 1), s)
    if isinstance(v, float): return DOUBLE
    if isinstance(v, str): return VARCHAR
    if isinstance(v, _dt.datetime): return TIMESTAMP if v.tzinfo is None else TIMESTAMPTZ
    if isinstance(v, _dt.date): return DATE
    if isinstance(v, _dt.time): return TIME if v.tzinfo is None else TIMETZ
    if isinstance(v, (Interval, _dt.timedelta)): return INTERVAL
    if isinstance(v, (bytes, bytearray)): return BLOB
    raise SqlError(f"no SQL type for {type(v).__name__}")


def out_value(v, t=None):
    """a value as returned to the caller (intervals as datetime.timedelta, rows with named fields as dicts and other
    rows as tuples, like DuckDB's Python API)"""
    if isinstance(v, Interval): return v.to_timedelta()
    if isinstance(v, tuple):
        fs = t.child if t is not None and t.id == "ROW" and t.child is not None else [("", None)] * len(v)
        vals = [out_value(x, ft) for x, (_, ft) in zip(v, fs)]
        return dict(zip([n for n, _ in fs], vals)) if fs and all(n for n, _ in fs) else tuple(vals)
    if isinstance(v, list):
        if t is not None and t.id == "ROW":
            fs = t.child or [("", None)] * len(v)
            vals = [out_value(x, ft) for x, (_, ft) in zip(v, fs)]
            return dict(zip([n for n, _ in fs], vals)) if fs and all(n for n, _ in fs) else tuple(vals)
        ch = t.child if t is not None and t.id in ("LIST", "MULTISET") else None
        return [out_value(x, ch) for x in v]
    return v


def sort_key(v):
    """a total order key for values of one type (NULLs are handled by the caller)"""
    if isinstance(v, float) and v != v: return (1, 0)                # NaN sorts after every number (DuckDB)
    if isinstance(v, (list, tuple)): return (0, tuple(((1, 0) if x is None else (0, sort_key(x))) for x in v))
    return (0, v)
