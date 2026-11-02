# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""torbles 0.3.0: algebraic queries over torbles.

Queries are built as algebraic expressions (operator overloading), compiled into:
  - Boolean polynomials over bit planes (filters), one program per table,
  - semi-joins (a dimension's filter becomes a bitmap; the fact's foreign key selects bits from it),
  - sums as popcount algebra (linear and bilinear forms) or, for GROUP BY, measure bytecode on surviving rows.
The same query renders to SQL, so DuckDB can run it for cross-checking.

    lo, d, c = db["lineorder"], db["dates"], db["customer"]
    q = (Query(lo).join(d, lo.date).join(c, lo.cust)
           .where((c.region == 2) & d.year.between(1, 5))
           .group_by(c.nation, d.year)
           .sum(lo.price * (100 - lo.discount)))
    q.run("alg");  q.sql()
"""
import ctypes, numpy as np
from pathlib import Path

MAXCOL = 31                      # columns per torble, hidden ones included (torble.h)
P = dict(TRUE=0, EQ=1, NE=2, LT=3, LE=4, GT=5, GE=6, BETWEEN=7, AND=8, OR=9, NOT=10, PREFIX=11, BIT=12, CUBE=13, CLT=14, CGE=15, XOR=16)
A_COUNT, A_LIN, A_BIL, A_GENERAL, A_POLYG, A_DOM, A_MM, A_ROWS, A_DIST = range(9)
M = dict(FCOL=0, ATTR=1, CONST=2, ADD=3, SUB=4, MUL=5, SAVE=6, LOAD=7)

class PInstr(ctypes.Structure): _fields_ = [("op", ctypes.c_int32), ("col", ctypes.c_int32), ("a", ctypes.c_uint32), ("b", ctypes.c_uint32)]
class MInstr(ctypes.Structure): _fields_ = [("op", ctypes.c_int32), ("arg", ctypes.c_int32), ("c", ctypes.c_int64)]
MAXKEY = 12                       # GROUP BY keys per plan (torble.h); a 64-bit fact column takes two
class QPlan(ctypes.Structure):
    _fields_ = [("nprog", ctypes.c_int32), ("prog", PInstr * 1024),
                ("nsemi", ctypes.c_int32), ("semi_col", ctypes.c_int32 * 8), ("semi_bm", ctypes.c_void_p * 8),
                ("agg", ctypes.c_int32), ("nlin", ctypes.c_int32), ("lin_col", ctypes.c_int32 * 8), ("lin_coef", ctypes.c_int64 * 8),
                ("bil_a", ctypes.c_int32), ("bil_b", ctypes.c_int32),
                ("nmeas", ctypes.c_int32), ("meas", MInstr * 48),
                ("nattr", ctypes.c_int32), ("attr_fkcol", ctypes.c_int32 * 8), ("attr_vals", ctypes.c_void_p * 8),
                ("nkey", ctypes.c_int32), ("key_kind", ctypes.c_int32 * MAXKEY), ("key_arg", ctypes.c_int32 * MAXKEY), ("key_radix", ctypes.c_uint64 * MAXKEY), ("key_nbits", ctypes.c_int32 * MAXKEY),
                ("ngroups", ctypes.c_uint64), ("out_sum", ctypes.c_void_p), ("out_cnt", ctypes.c_void_p),
                ("total_sum", ctypes.c_int64), ("total_cnt", ctypes.c_uint64),
                ("semi_n", ctypes.c_uint64 * 8), ("attr_n", ctypes.c_uint64 * 8),
                ("npl", ctypes.c_int32), ("pl_col", ctypes.c_int32 * 8), ("pl_coef", ctypes.c_int64 * 8),
                ("npb", ctypes.c_int32), ("pb_a", ctypes.c_int32 * 8), ("pb_b", ctypes.c_int32 * 8), ("pb_coef", ctypes.c_int64 * 8),
                ("p_const", ctypes.c_int64), ("nranges", ctypes.c_uint64), ("ranges", ctypes.c_void_p),
                ("key_mask", ctypes.c_uint32 * MAXKEY),
                ("dom_col", ctypes.c_int32), ("ndom", ctypes.c_int32), ("dom_t", ctypes.c_void_p),
                ("semi_kmask", ctypes.c_uint32 * 8), ("attr_kmask", ctypes.c_uint32 * 8),
                ("xrows", ctypes.c_void_p), ("xmode", ctypes.c_int32),
                ("out_cap", ctypes.c_uint64), ("out_n", ctypes.c_uint64), ("out_key", ctypes.c_void_p), ("mm_stats", ctypes.c_int64 * 4),
                ("wide", ctypes.c_int32), ("pl_coef_hi", ctypes.c_int64 * 8), ("pb_coef_hi", ctypes.c_int64 * 8), ("p_const_hi", ctypes.c_int64),
                ("out_sum_hi", ctypes.c_void_p), ("total_sum_hi", ctypes.c_int64), ("widekey", ctypes.c_int32), ("out_key_hi", ctypes.c_void_p),
                ("ext_on", ctypes.c_int32), ("ext_col", ctypes.c_int32), ("out_rows", ctypes.c_void_p),
                ("dist_bits", ctypes.c_void_p), ("dist_radix", ctypes.c_uint64)]
KEY_MAX = (1 << 128) - 2          # group codes are 128-bit (all ones marks an empty slot): at most 2^128 - 2 groups

M64 = (1 << 64) - 1
def _s64(x):
    """the low 64 bits of an integer, as a signed 64-bit field (the engine's arithmetic is modular)"""
    x &= M64
    return x - (1 << 64) if x >> 63 else x
def _hi64(x):
    """the high word of a 128-bit two's complement integer, signed"""
    return _s64(x >> 64)
def _join128(lo, hi):
    """(lo, hi) words of a 128-bit sum -> a Python int"""
    return (int(hi) << 64) + (int(lo) & M64)

def _join128_arr(lo, hi):
    """_join128 of arrays: an int64 array when every sum fits 64 bits (hi is the sign extension of lo), else Python ints"""
    lo, hi = np.asarray(lo), np.asarray(hi)
    l64 = lo.astype(np.int64, copy=False) if lo.dtype != np.uint64 else lo.view(np.int64)
    h64 = hi.astype(np.int64, copy=False) if hi.dtype != np.uint64 else hi.view(np.int64)
    if np.array_equal(h64, np.where(l64 < 0, -1, 0)): return l64
    return np.array([_join128(a, b) for a, b in zip(lo.tolist(), hi.tolist())], object)

def _unique_rows(cols):
    """the distinct rows of equal-length int64 columns, in sorted order, and each row's index among them -> (list of
    columns, inverse): one lexsort (np.unique(axis=0) sorts rows as opaque byte strings, many times slower)"""
    n = len(cols[0])
    o = np.lexsort(cols[::-1])
    new = np.ones(n, bool)
    if n:
        new[1:] = np.logical_or.reduce([c[o][1:] != c[o][:-1] for c in cols])
    gid = np.cumsum(new) - 1
    inv = np.empty(n, np.int64); inv[o] = gid
    return [c[o][new] for c in cols], inv

def _digits128(lo, hi, radix):
    """the mixed-radix digits (radixes below 2^32) of 128-bit codes given as (lo, hi) 64-bit words, by long division
    on 32-bit limbs: each step divides a value below r * 2^32 <= 2^64, so uint64 arithmetic is exact"""
    lo = np.asarray(lo).astype(np.uint64); hi = np.asarray(hi).astype(np.uint64)
    m32, s32 = np.uint64(0xFFFFFFFF), np.uint64(32)
    limbs = [hi >> s32, hi & m32, lo >> s32, lo & m32]          # most significant first
    cols = []
    for r in reversed(radix):
        r = np.uint64(int(r)); rem = np.zeros(len(lo), np.uint64)
        for j in range(4):
            cur = (rem << s32) | limbs[j]
            limbs[j] = cur // r; rem = cur % r
        cols.append(rem)
    return cols[::-1]

def interval(e, fact):
    """the range [lo, hi] of an integer expression over stored codes on one row: codes are in [0, colmax]"""
    if isinstance(e, Const): return int(e.v), int(e.v)
    if isinstance(e, Col): return 0, int(e.table.maxcode(e.name) if hasattr(e.table, "maxcode") else e.table.colmax(e.name))
    (a, b), (c, d) = interval(e.l, fact), interval(e.r, fact)
    if e.op == "+": return a + c, b + d
    if e.op == "-": return a - d, b - c
    ps = (a * c, a * d, b * c, b * d)
    return min(ps), max(ps)

SUM_BITS = 127                    # 128-bit two's complement sums: exact while |sum| < 2^127
def sum_width(measure, fact):
    """0: every sum of the measure fits 64 bits (|sum| < 2^63), 1: 128 bits (< 2^127). Each group's sum and the total
    are at most rows x max |value of one row| (interval arithmetic on the expression, not on its expansion: the engine's
    arithmetic is modular, so the expansion's terms may wrap as long as the result fits)."""
    if measure is None: return 0
    lo, hi = interval(measure, fact)
    bound = max(1, fact.n) * max(abs(lo), abs(hi))
    if bound >= 1 << SUM_BITS:
        raise OverflowError(f"SUM may exceed 128 bits on {fact.n:,} rows (one row up to about 2^{max(abs(lo), abs(hi)).bit_length()}); "
                            "torbles refuses rather than risk a wrong answer")
    return int(bound >= 1 << 63)

__version__ = "0.3.0"
_libpath = Path(__import__("os").environ.get("TORBLES_ALG_LIB") or Path(__file__).with_name("libalg.so"))   # (an instrumented build: tests/sanitize_math.sh)
if not _libpath.exists(): raise ImportError(f"{_libpath} not built: run `make` in the repository root")
lib = ctypes.CDLL(str(_libpath))
vp, u64 = ctypes.c_void_p, ctypes.c_uint64
for name, res, args in [("at_create", vp, [u64, ctypes.c_int, vp, vp]), ("pt_create", vp, [u64, ctypes.c_int, vp]),
                        ("at_save", ctypes.c_int, [vp, ctypes.c_char_p]), ("at_load", vp, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]),
                        ("at_bytes", u64, [vp]), ("at_width", ctypes.c_int, [vp, ctypes.c_int]), ("at_nplanes", ctypes.c_int, [vp]),
                        ("at_flip", None, [vp, ctypes.c_int, u64]), ("at_validate", u64, [vp, vp, u64]),
                        ("at_filter_bitmap", u64, [vp, vp, ctypes.c_int, vp]), ("at_filter_bitmap_masked", u64, [vp, vp, ctypes.c_int, vp, vp]), ("bm_ids", u64, [vp, u64, vp]), ("pt_filter_bitmap", u64, [vp, vp, ctypes.c_int, vp]),
                        ("at_decode_column", None, [vp, ctypes.c_int, vp]),
                        ("at_set_threads", None, [ctypes.c_int]), ("at_set_virtual", None, [vp, ctypes.c_int, ctypes.c_int, vp, vp]), ("at_rows_semi", u64, [vp, ctypes.c_int, ctypes.c_uint32, vp, u64, vp]), ("at_get_rows", None, [vp, ctypes.c_int, vp, u64, vp]), ("at_tail_offsets", None, [vp, ctypes.c_int, ctypes.c_uint32, vp, vp, u64, vp]), ("at_get_threads", ctypes.c_int, []), ("at_repair", u64, [vp]), ("at_query", ctypes.c_int64, [vp, ctypes.POINTER(QPlan)]), ("at_mm_kernel", ctypes.c_char_p, []), ("at_version", ctypes.c_char_p, []),
                        ("at_set_build_threads", None, [ctypes.c_int]), ("at_get_build_threads", ctypes.c_int, []),
                        ("tb_map", None, [vp, u64, vp, u64, vp]),
                        ("jp_terms", ctypes.c_int64, [vp, ctypes.c_int, vp, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, vp, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]), ("tb_date_surrogate", None, [vp, u64, vp]),
                        ("tb_refok", u64, [vp, u64, ctypes.c_uint32, u64, vp, vp, ctypes.c_uint32, vp]), ("pt_query", ctypes.c_int64, [vp, ctypes.POINTER(QPlan)])]:
    f = getattr(lib, name); f.restype = res; f.argtypes = args
# the release of a Table's storage (Table.close, or its finalizer): a torble's planes or a loaded file's mapping
# (at_release), then its header (at_free), and the plain baseline's column table
lib.at_free.restype, lib.at_free.argtypes = None, [vp]
lib.at_release.restype, lib.at_release.argtypes = None, [vp]
lib.pt_free.restype, lib.pt_free.argtypes = None, [vp]

# optional GPU executor (gpu.cu): libgpu.so (CUDA) or, for testing, libgpu_emu.so (the same kernels run on the CPU)
_libg = None
def gpu_lib():
    global _libg
    if _libg is None:
        import os
        name = os.environ.get("TORBLES_GPU_LIB") or str(Path(__file__).with_name("libgpu.so"))
        if not Path(name).exists(): name = str(Path(__file__).with_name("libgpu_emu.so"))
        g = ctypes.CDLL(name)
        if not hasattr(g, "gpu_query_tagged"):
            raise RuntimeError(f"{name} is older than this torbles (no gpu_query_tagged): rebuild it (make, make gpu)")
        g.gpu_open.restype = vp; g.gpu_open.argtypes = [vp]
        g.gpu_query.restype = ctypes.c_int; g.gpu_query.argtypes = [vp, ctypes.POINTER(QPlan), ctypes.POINTER(ctypes.c_double)]
        u64p = ctypes.POINTER(ctypes.c_uint64)       # the device copies of host arrays (see _gpu_tag)
        g.gpu_query_tagged.restype = ctypes.c_int; g.gpu_query_tagged.argtypes = [vp, ctypes.POINTER(QPlan), u64p, ctypes.POINTER(ctypes.c_double)]
        g.gpu_cache_evict.restype = ctypes.c_int; g.gpu_cache_evict.argtypes = [vp, u64p, ctypes.c_int]
        g.gpu_cache_entries.restype = ctypes.c_int; g.gpu_cache_entries.argtypes = [vp, u64p, u64p, ctypes.c_int]
        g.gpu_name.restype = ctypes.c_char_p; g.gpu_close.restype = None; g.gpu_close.argtypes = [vp]
        g.gpu_mm_mode.restype = ctypes.c_int; g.gpu_mm_mode.argtypes = [ctypes.c_int]
        g.gpu_mm_name.restype = ctypes.c_char_p
        if os.environ.get("TORBLES_GPU_MM"): g.gpu_mm_mode({"word": 0, "popc": 1, "mma": 2}[os.environ["TORBLES_GPU_MM"]])
        _libg = g
    return _libg
def gpu_name(): return gpu_lib().gpu_name().decode()
def gpu_mm(mode=None):
    """GPU GROUP BY kernel: "word" (one thread per 64-row word, default), or the tile matrix product with "popc"
    (__popc) or "mma" (tensor-core b1 MMA, if its self-test passed); returns the kernel's name"""
    if mode is not None: gpu_lib().gpu_mm_mode({"word": 0, "popc": 1, "mma": 2}[mode])
    return gpu_lib().gpu_mm_name().decode()

# GROUP BY evaluation: "mm" = one AND-popcount matrix product per tile (A_MM, sparse output), "split" = the earlier
# per-word mask splitting / per-row path with dense group arrays. TORBLES_GROUPBY overrides the default.
import os as _os
GROUPBY = _os.environ.get("TORBLES_GROUPBY", "mm")
ROWPATH_GROUPS = int(_os.environ.get("TORBLES_ROWPATH_GROUPS", 1 << 14))        # GROUP BY with at least this many possible groups: the row kernel, not the matrix product (0: off)
ROWPATH_DENSE = 1 << 17          # (with a filter: only while the keys' code domain is at most this)
DIST_PER_ROW = 64                # ... and at most this many per fact row (a larger domain: the cells as groups)
DIST_CELLS = 1 << 28              # COUNT(DISTINCT) by cell bits (A_DIST): at most this many (group, value) cells (32 MB)
# compiled plans are reused when the same query runs again on the same data version (TORBLES_PLAN_CACHE=0: off)
PLAN_CACHE = _os.environ.get("TORBLES_PLAN_CACHE", "1") != "0"
# bridge "factor through the key": measure factors that are GROUP BY key columns are constant per group and are
# multiplied in after the sum (TORBLES_FACTOR=0: off)
FACTOR = _os.environ.get("TORBLES_FACTOR", "1") != "0"
NATIVE_PLAN = _os.environ.get("TORBLES_NATIVE_PLAN", "1") != "0"      # join-polynomial terms planned in C (jp_terms)
def set_groupby(mode):
    global GROUPBY
    assert mode in ("mm", "split"); GROUPBY = mode
def mm_kernel(): return lib.at_mm_kernel().decode()
if lib.at_version().decode().replace("-", "") != __version__: raise ImportError(f"libalg.so is version {lib.at_version().decode()}, algq.py is {__version__}: rebuild (make)")

def set_build_threads(n):
    """threads used to build torbles and for the native ingestion helpers (default: all CPUs)"""
    lib.at_set_build_threads(int(n))
def par_map(col, lut):
    """lut[col] for a uint32 column and an int64 lookup table (keys outside the table map to 0), on all cores"""
    col = np.ascontiguousarray(col, np.uint32); lut = np.ascontiguousarray(lut, np.int64); out = np.empty(len(col), np.uint32)
    if len(lut) and (int(lut.max()) > 0xFFFFFFFF or int(lut.min()) < -1):        # -1: a key absent from the map
        raise ValueError("par_map: mapped values must fit in 32 bits (a wider key needs a self-extending column)")
    lib.tb_map(col.ctypes.data, len(col), lut.ctypes.data, len(lut), out.ctypes.data); return out
def par_date_surrogate(col):
    """yyyymmdd -> (y - 1992) << 9 | m << 5 | d, on all cores"""
    col = np.ascontiguousarray(col, np.uint32); out = np.empty(len(col), np.uint32)
    lib.tb_date_surrogate(col.ctypes.data, len(col), out.ctypes.data); return out

def set_threads(n):
    """threads used by queries and validation (both the torbles engine and the plain columnar baseline)"""
    lib.at_set_threads(int(n))

# ------------------------------------------------------------------ expressions
class Expr:
    __hash__ = object.__hash__
    def _w(self, o): return o if isinstance(o, Expr) else Const(o)
    def __add__(self, o): return Arith("+", self, self._w(o))
    def __radd__(self, o): return Arith("+", self._w(o), self)
    def __sub__(self, o): return Arith("-", self, self._w(o))
    def __rsub__(self, o): return Arith("-", self._w(o), self)
    def __mul__(self, o): return Arith("*", self, self._w(o))
    def __rmul__(self, o): return Arith("*", self._w(o), self)
class Const(Expr):
    def __init__(self, v):
        self.raw = v                                   # as written (the SQL rendering keeps it)
        if isinstance(v, (bool, int, np.integer)): self.v = int(v)
        else:
            from coltypes import _frac
            f = _frac(v); self.v = int(f) if f.denominator == 1 else None   # None: not an integer (a decimal constant)
class Arith(Expr):
    def __init__(self, op, l, r): self.op, self.l, self.r = op, l, r
_SQLOP = dict(EQ="=", NE="<>", LT="<", LE="<=", GT=">", GE=">=")
import re as _re
_RESERVED = set("""
    all analyse analyze and anti any array as asc asof asymmetric at authorization between bigint binary bit
    boolean both by case cast char character check coalesce collate collation column columns concurrently
    constraint create cross dec decimal default deferrable desc describe distinct do else end except exists
    extract false fetch float for foreign freeze from full generated glob group grouping grouping_id having
    ilike in initially inner inout int integer intersect interval into is isnull join lambda lateral leading
    left like limit map national natural nchar none not notnull null nullif numeric offset on only or order
    out outer overlaps overlay pivot pivot_longer pivot_wider placing position positional precision primary
    qualify real references returning right row select semi setof show similar smallint some struct
    substring summarize symmetric table tablesample then time timestamp to trailing treat trim true try_cast
    tuple union unique unpack unpivot using values varchar variadic verbose when where window with
    xmlattributes xmlconcat xmlelement xmlexists xmlforest xmlnamespaces xmlparse xmlpi xmlroot xmlserialize
    xmltable
""".split())                                       # keywords that cannot be bare identifiers (DuckDB, standard SQL)
def _ident(name):
    """an SQL identifier: as is when plain, else double-quoted"""
    return name if _re.fullmatch(r"[a-z_][a-z0-9_]*", name) and name not in _RESERVED else '"' + name.replace('"', '""') + '"'
def _valid_base(table, name):
    """the column whose validity plane is column `name` of table (its c#v), or None"""
    for c, v in (getattr(table, "nullable", None) or {}).items():
        if v == name: return c
    return None
def _qc(col):
    """a column in the source's SQL; a validity plane is not a source column: it is the test that its column is not NULL"""
    base = _valid_base(col.table, col.name)
    if base is not None: return f"({_qc(Col(col.table, base))} IS NOT NULL)"
    return f"{_ident(col.table.name)}.{_ident(col.name)}"
class Col(Expr):
    def __init__(self, table, name): self.table, self.name = table, name
    def codec(self): return getattr(self.table, "codecs", {}).get(self.name)
    def _valid(self, p):
        """a comparison on a column with NULLs: FALSE for its NULL rows (the engine API is two-valued; db.sql applies
        SQL's three-valued logic)"""
        vc = getattr(self.table, "nullable", {}).get(self.name)
        if vc is None: return p
        v = Cmp(Col(self.table, vc), "EQ", 1)
        q = BoolOp("AND", v, p)
        # the SQL says so too: the comparison alone is UNKNOWN on a NULL row, which NOT keeps (the engine's FALSE does not)
        s = getattr(p, "sql", None)
        q.sql = None if s is None else f"({pred_sql(v)} AND {s})"
        return q
    def _cmp(self, op, v): return self._valid(self._cmp0(op, v))
    def _cmp0(self, op, v):
        cd = self.codec()
        if cd is None:
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 <= int(v) <= 0xFFFFFFFF:
                return Cmp(self, op, int(v))
            from coltypes import IntCodec                 # out-of-range or non-integer constants: exact semantics
            cd = IntCodec(0, "BIGINT")
        return _typed_cmp(self, cd, op, v)
    def __eq__(self, v): return self._cmp("EQ", v)
    def __ne__(self, v): return self._cmp("NE", v)
    def __lt__(self, v): return self._cmp("LT", v)
    def __le__(self, v): return self._cmp("LE", v)
    def __gt__(self, v): return self._cmp("GT", v)
    def __ge__(self, v): return self._cmp("GE", v)
    def between(self, a, b):
        cd = self.codec()
        if cd is None and all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) and 0 <= int(x) <= 0xFFFFFFFF for x in (a, b)):
            return Between(self, int(a), int(b))
        if cd is None:
            from coltypes import IntCodec; cd = IntCodec(0, "BIGINT")
        lo, hi = cd.interval("GE", a), cd.interval("LE", b)
        iv = None if lo is None or hi is None else (lo[0], hi[1])
        p = _code_pred(self, iv)
        p.sql = f"{_qc(self)} BETWEEN {cd.literal(a)} AND {cd.literal(b)}"
        return self._valid(p)
    def isin(self, vals):
        vals = list(vals)
        if not vals: raise ValueError("isin() needs at least one value")
        p = self._cmp0("EQ", vals[0])
        for v in vals[1:]: p = p | self._cmp0("EQ", v)
        cd = self.codec()
        if cd is not None:
            p.sql = f"{_qc(self)} IN ({', '.join(cd.literal(v) for v in vals)})"
        return self._valid(p)                          # (one validity term for the whole list, as in its SQL)
    def startswith(self, prefix):
        """string prefix match: one interval of the sorted dictionary"""
        cd = self.codec()
        if cd is None or not hasattr(cd, "prefix_interval"): raise TypeError(f"{self.table.name}.{self.name} is not a string column")
        p = _code_pred(self, cd.prefix_interval(prefix))
        p.sql = f"starts_with({_qc(self)}, {cd.literal(prefix)})"
        return self._valid(p)
class Pred:
    sql = None                                         # the predicate in the source's terms (typed columns)
    def __and__(self, o): return BoolOp("AND", self, o)
    def __or__(self, o): return BoolOp("OR", self, o)
    def __invert__(self): return Not(self)
class Cmp(Pred):
    def __init__(self, col, op, v): self.col, self.op, self.v = col, op, v
class Between(Pred):
    def __init__(self, col, a, b): self.col, self.a, self.b = col, a, b
class BoolOp(Pred):
    def __init__(self, op, l, r): self.op, self.l, self.r = op, l, r
class Not(Pred):
    def __init__(self, x): self.x = x
class RowMask(Pred):
    """a dimension filter given as its rows: mask[i] is True when dimension row i passes (any predicate, evaluated on
    the dimension's distinct values). It joins like every dimension filter: as the bitmap of a join polynomial or a
    semi-join. cols: the attributes the predicate reads (they guide the join polynomial's form)."""
    def __init__(self, table, mask, cols=(), sql=None):
        self.table, self.cols, self.sql = table, set(cols), sql
        m = np.zeros(((table.n + 63) // 64 + 1) * 64, np.uint8); m[:table.n] = np.asarray(mask, bool)[:table.n]
        self.bm = np.packbits(m, bitorder="little").view(np.uint64)
class RefOk(Pred):
    """the fact rows whose foreign key fk has (ok=True) or has not (ok=False) a matching dimension row: one plane, the
    fk's bit of the hidden _refok column"""
    def __init__(self, table, fk, ok=True, sql=None): self.table, self.fk, self.ok, self.sql = table, fk, ok, sql
class CodeSet(Pred):
    """a filter on one fact column given as the set of its codes that pass (any predicate of that column, evaluated on
    the codes it can hold): a semi-join of the column against a bitmap over its code domain. Only as an AND-term."""
    def __init__(self, col, mask, sql=None):
        self.col, self.sql = col, sql
        mask = np.asarray(mask, bool); self.n = len(mask)
        m = np.zeros(((self.n + 63) // 64 + 1) * 64, np.uint8); m[:self.n] = mask
        self.bm = np.packbits(m, bitorder="little").view(np.uint64); self.frac = float(mask.mean()) if self.n else 0.0

def _maxcode(col):
    t = col.table
    return t.maxcode(col.name) if hasattr(t, "maxcode") else t.colmax(col.name)
def _code_pred(col, iv):
    """the predicate  code in [lo, hi]  on a column's codes (None bounds: unbounded; iv None: empty), simplified
    against the codes that exist (0 .. the column's largest code): FALSE, TRUE, =, <=, >= or BETWEEN"""
    FALSE = Cmp(col, "LT", 0)
    if iv is None: return FALSE
    mx = _maxcode(col); lo, hi = iv
    lo = 0 if lo is None or lo < 0 else int(lo); hi = mx if hi is None or hi > mx else int(hi)
    if lo > hi: return FALSE
    if lo == 0 and hi == mx: return Not(FALSE)
    if lo == hi: return Cmp(col, "EQ", lo)
    if lo == 0: return Cmp(col, "LE", hi)
    if hi == mx: return Cmp(col, "GE", lo)
    return Between(col, lo, hi)
def _typed_cmp(col, cd, op, v):
    if op == "NE":
        p = Not(_typed_cmp(col, cd, "EQ", v))
    else:
        p = _code_pred(col, cd.interval(op, v))
    p.sql = f"{_qc(col)} {_SQLOP[op]} {cd.literal(v)}"
    return p

def id_ranges(bm, n):
    """contiguous runs of set bits in a row bitmap -> [(lo, hi)] inclusive"""
    bits = np.unpackbits(bm.view(np.uint8), bitorder="little")[:n].astype(np.int8)
    edges = np.diff(np.concatenate([[0], bits, [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    if len(starts) > 16: return None
    return [(int(a), int(b)) for a, b in zip(starts, ends)]

def cover_blocks(sel, used, b, limit=32):
    """Cover the selected ids with aligned blocks [p*2^(b-L), (p+1)*2^(b-L)) (ids absent from `used` are don't-care).
    Each block is a monomial on the first L planes of an MSB-first key. Returns [(p, L)] or None if > limit."""
    cnt = lambda a, lo, hi: int(np.searchsorted(a, hi, "right") - np.searchsorted(a, lo, "left"))
    out, stack = [], [(0, 0)]
    while stack:
        p, L = stack.pop()
        lo, hi = p << (b - L), ((p + 1) << (b - L)) - 1
        ns = cnt(sel, lo, hi)
        if ns == 0: continue
        if ns == cnt(used, lo, hi):
            out.append((p, L))
            if len(out) > limit: return None
            continue
        stack += [(2 * p + 1, L + 1), (2 * p, L + 1)]
    return sorted(out)

def dc_runs(sel_mask, used_mask, limit=16):
    """contiguous id ranges covering the selected ids, unused ids treated as don't-care"""
    bits = (sel_mask | ~used_mask).astype(np.int8)
    edges = np.diff(np.concatenate([[0], bits, [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    cs = np.concatenate([[0], np.cumsum(sel_mask)])
    keep = [(int(a), int(b)) for a, b in zip(starts, ends) if cs[b + 1] - cs[a] > 0]
    return keep if len(keep) <= limit else None

class Surrogate:
    """A foreign key stored as a derived surrogate s of the dimension's own key: the dimension row is s & (2^w - 1)
    (the official key, kept verbatim in the low w bits: a projection), and s_of_row[k] is the surrogate of row k
    (e.g. (coordinates of row k) << w | k). Joins substitute pk := s mod 2^w; the high bits carry derived coordinates."""
    def __init__(self, w, s_of_row):
        self.w, self.mask = int(w), (1 << int(w)) - 1
        self.s_of_row = np.asarray(s_of_row, np.int64).copy()     # the current derivation of every row
        if len(self.s_of_row) and int(self.s_of_row.max()) > 0xFFFFFFFF:
            raise ValueError(f"Surrogate: derived surrogates need {int(self.s_of_row.max()).bit_length()} bits; foreign keys are 32-bit")
        self.stale, self.version = {}, 0                           # row -> earlier surrogates still stored in facts
    def move(self, rows, new_s):
        """A dimension change (e.g. a customer moves city): the derivation of these rows changes. Nothing is rewritten:
        facts keep the surrogate they were written with, and the old values become stale points. Queries correct
        for them additively: filters by XOR (F2) of the points whose predicate flips, groups by regrouping the
        stale rows (integers)."""
        for r, v in zip(np.asarray(rows, np.int64).tolist(), np.asarray(new_s, np.int64).tolist()):
            if not 0 <= v <= 0xFFFFFFFF: raise ValueError("Surrogate.move: surrogates are 32-bit")
            old = int(self.s_of_row[r])
            if old != v:
                self.stale.setdefault(r, [])
                if old not in self.stale[r]: self.stale[r].append(old)
                self.s_of_row[r] = v
        self.version += 1
    def stale_arrays(self):
        """(stale surrogate values, their dimension rows)"""
        pts = [(v, r) for r, vs in self.stale.items() for v in vs]
        if not pts: return np.zeros(0, np.int64), np.zeros(0, np.int64)
        a = np.array(pts, np.int64); return a[:, 0], a[:, 1]
    def compacted(self):
        """after the fact has been rewritten with current surrogates: no stale points left"""
        self.stale = {}; self.version += 1

class PlaneKey(object):
    """GROUP BY key read directly from planes of a fact column (a projection: the code of the planes in mask)"""
    def __init__(self, table, name, mask): self.table, self.name, self.mask = table, name, int(mask)

class IdSpace:
    """The values a fact's foreign key takes for a dimension: the dimension's row ids, or their surrogates.
    ids are sorted; rows[i] is the dimension row of ids[i]."""
    def __init__(self, dim, sg=None):
        self.dim, self.sg, self.n = dim, sg, dim.n
        rows = dim.used_ids
        if sg is None: self.ids, self.rows = rows.astype(np.int64), rows
        else:
            sv = sg.s_of_row[rows]; o = np.argsort(sv, kind="stable"); self.ids, self.rows = sv[o], rows[o]
    def row_of(self, x): return x if self.sg is None else (x & self.sg.mask)
    def values(self, attr): return self.dim.data[attr][self.rows]
    def attr_at(self, attr, x):
        """for candidate ids x: (exists, attribute values where they exist)"""
        col = self.dim.data[attr]
        if self.sg is None:
            ok = x < self.n; ok[ok] = self.dim.used[x[ok]]
            v = np.zeros(len(x), col.dtype); v[ok] = col[x[ok]]; return ok, v
        p = np.searchsorted(self.ids, x); p = np.minimum(p, len(self.ids) - 1)
        ok = self.ids[p] == x if len(self.ids) else np.zeros(len(x), bool)
        v = np.zeros(len(x), col.dtype); v[ok] = col[self.rows[p[ok]]]; return ok, v

def _sp(x):
    if isinstance(x, IdSpace): return x
    if not hasattr(x, "_space0"): x._space0 = IdSpace(x)
    return x._space0

import weakref
# per-space caches, keyed by the space object itself (weakly): an id() key could be reused by a new space after the
# old one is freed and return the old space's answer
def _space_memo(cache, space): 
    d = cache.get(space)
    if d is None: d = cache[space] = {}
    return d
_prefix_cache = weakref.WeakKeyDictionary()
def prefix_key(space, attr, b):
    """smallest L such that `attr` is a function of the first L bits (MSB first) of the b-bit id; with the map"""
    space = _sp(space); ck = (attr, b); memo = _space_memo(_prefix_cache, space)
    if ck not in memo:
        m = space.ids < (1 << b); ids = space.ids[m]; a = space.values(attr)[m]; res = None
        if not len(ids): memo[ck] = (0, {}); return memo[ck]          # no id fits b bits: nothing can join
        for L in range(0, b + 1):
            g = ids >> (b - L)
            starts = np.concatenate([[0], np.nonzero(np.diff(g))[0] + 1])
            if np.array_equal(np.maximum.reduceat(a, starts), np.minimum.reduceat(a, starts)):
                res = (L, dict(zip(g[starts].tolist(), a[starts].tolist()))); break
        memo[ck] = res
    return memo[ck]

def poly_fact(e, fact):
    """expand a measure into a polynomial of any degree in fact columns: {sorted tuple of column names: coef}; None if
    it involves another table's columns"""
    if isinstance(e, Const): return {(): e.v} if e.v is not None else None
    if isinstance(e, Col): return {(e.name,): 1} if e.table is fact else None
    if not isinstance(e, Arith): return None
    l, r = poly_fact(e.l, fact), poly_fact(e.r, fact)
    if l is None or r is None: return None
    out = {}
    if e.op in "+-":
        for k, v in l.items(): out[k] = out.get(k, 0) + v
        for k, v in r.items(): out[k] = out.get(k, 0) + (v if e.op == "+" else -v)
    else:
        for ka, va in l.items():
            for kb, vb in r.items():
                k = tuple(sorted(ka + kb)); out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v}

def factor_through_keys(measure, keys, fact):
    """The bridge between the GROUP BY key and the measure. Within a group, a key column is a constant, so every
    monomial of the measure splits into (key part) x (rest): SUM_g(measure) = sum over rest-monomials R of
    P_R(key_g) * SUM_g(R), with P_R a polynomial in the key columns. When at most one rest-monomial is not constant,
    one pass gives SUM_g(R) and COUNT_g, and the polynomials are applied per group afterwards: a product of a measure
    column with a key column costs the measure's planes, not their pairs.
    -> (R or None, P_R, P_1, {key column: position}) or None when nothing factors"""
    keycols = {k.name: i for i, k in enumerate(keys) if isinstance(k, Col) and k.table is fact}
    if not keycols or measure is None: return None
    poly = poly_fact(measure, fact)
    if poly is None: return None
    parts = {}
    for mono, c in poly.items():
        K = tuple(x for x in mono if x in keycols); R = tuple(x for x in mono if x not in keycols)
        d = parts.setdefault(R, {}); d[K] = d.get(K, 0) + c
    if not any(K for d in parts.values() for K in d): return None
    rest = [R for R in parts if R]
    if len(rest) > 1: return None
    R = rest[0] if rest else None
    return R, (parts[R] if R else {}), parts.get((), {}), keycols

def _poly_at(P, keycols, key):
    v = 0
    for K, c in P.items():
        t = c
        for x in K: t *= key[keycols[x]]
        v += t
    return v

def _set_poly(plan, poly, wide, fact):
    """a degree-2 polynomial's terms into the plan: coefficients as their low 64 bits (the engine's arithmetic is
    modular) and, for 128-bit plans, their high words"""
    fact_idx = fact.idx
    lins = [(m[0], c) for m, c in poly.items() if len(m) == 1]; bils = [(m, c) for m, c in poly.items() if len(m) == 2]
    c0 = poly.get((), 0); plan.p_const = _s64(c0); plan.p_const_hi = _hi64(c0) if wide else 0
    plan.npl, plan.npb = len(lins), len(bils)
    for i, (col, c) in enumerate(lins):
        plan.pl_col[i], plan.pl_coef[i] = fact_idx(col), _s64(c); plan.pl_coef_hi[i] = _hi64(c) if wide else 0
    for i, ((a, b), c) in enumerate(bils):
        plan.pb_a[i], plan.pb_b[i], plan.pb_coef[i] = fact_idx(a), fact_idx(b), _s64(c); plan.pb_coef_hi[i] = _hi64(c) if wide else 0

def poly2(e, fact):
    """expand an expression into a polynomial of degree <= 2 in fact columns: {(): c0, (x,): c, (a, b): c}"""
    if isinstance(e, Const): return {(): e.v}
    if isinstance(e, Col): return {(e.name,): 1} if e.table is fact else None
    l, r = poly2(e.l, fact), poly2(e.r, fact)
    if l is None or r is None: return None
    out = {}
    if e.op in "+-":
        for k, v in l.items(): out[k] = out.get(k, 0) + v
        for k, v in r.items(): out[k] = out.get(k, 0) + (v if e.op == "+" else -v)
    else:
        for ka, va in l.items():
            for kb, vb in r.items():
                k = tuple(sorted(ka + kb))
                if len(k) > 2: return None
                out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v}

_coord_cache = weakref.WeakKeyDictionary()
def attr_coords(space, attr, b, maxbits=24):
    """Is `attr` a coordinate of the id? Find the id bits its value depends on (bit k matters if flipping it changes
    the value for some pair of existing ids), then check the attribute really is a function of those bits.
    Works on row ids or on derived surrogates (IdSpace). Returns (bits, most significant first; {code: value}) or None."""
    space = _sp(space); ck = (attr, b); memo = _space_memo(_coord_cache, space)
    if ck not in memo:
        m = space.ids < (1 << b); ids = space.ids[m]; v = space.values(attr)[m]; res = None
        if not len(ids): memo[ck] = ([], {}); return memo[ck]
        hint = getattr(space.sg, "coords", {}).get(attr) if space.sg is not None else None
        if hint is not None and hint and max(hint) < b and len(hint) <= maxbits:
            # the key design knows which bits hold the attribute (a derived surrogate's fields): verify and use them
            code = np.zeros(len(ids), np.int64)
            for k in hint: code = (code << 1) | ((ids >> k) & 1)
            order = np.argsort(code, kind="stable"); cs, vs = code[order], v[order]
            starts = np.concatenate([[0], np.nonzero(np.diff(cs))[0] + 1])
            if np.array_equal(np.maximum.reduceat(vs, starts), np.minimum.reduceat(vs, starts)):
                memo[ck] = (list(hint), dict(zip(cs[starts].tolist(), vs[starts].tolist())))
                pk = prefix_key(space, attr, b)
                if pk is not None and pk[0] < len(hint): memo[ck] = (list(range(b - 1, b - 1 - pk[0], -1)), pk[1])
                return memo[ck]
        R = []
        for k in range(b):
            ok, pv = space.attr_at(attr, ids ^ (1 << k))
            if np.any(v[ok] != pv[ok]): R.append(k)
        R.sort(reverse=True)
        if len(R) <= maxbits and len(ids):
            code = np.zeros(len(ids), np.int64)
            for k in R: code = (code << 1) | ((ids >> k) & 1)
            order = np.argsort(code, kind="stable"); cs, vs = code[order], v[order]
            starts = np.concatenate([[0], np.nonzero(np.diff(cs))[0] + 1])
            if np.array_equal(np.maximum.reduceat(vs, starts), np.minimum.reduceat(vs, starts)):
                res = (R, dict(zip(cs[starts].tolist(), vs[starts].tolist())))
        # Single-bit flips find nothing where ids are sparse (e.g. a surrogate whose low bits are a unique key: flipping
        # a coordinate bit never lands on another existing id). Also test directly for the smallest prefix of the id
        # that determines the attribute (a function check, no flips); keep the smaller valid answer.
        pk = prefix_key(space, attr, b)
        if pk is not None and pk[0] <= maxbits and (res is None or pk[0] < len(res[0])):
            res = (list(range(b - 1, b - 1 - pk[0], -1)), pk[1])
        memo[ck] = res
    return memo[ck]

_space_cache = weakref.WeakKeyDictionary()
def coord_space(space, R, b):
    """the codes (projections onto id bits R) that occur among existing ids, with one representative id per code"""
    space = _sp(space); key = (tuple(R), b); memo = _space_memo(_space_cache, space)
    if key not in memo:
        ids = space.ids[space.ids < (1 << b)]
        code = np.zeros(len(ids), np.int64)
        for k in R: code = (code << 1) | ((ids >> k) & 1)
        uniq, first = np.unique(code, return_index=True)
        memo[key] = (uniq, ids[first])
    return memo[key]

# ------------------------------------------------------------------ join polynomials
# Cost model in nanoseconds per 64-row word on one thread, measured by calibrate.py on this machine:
#   lit: one plane literal (an AND/ANDN of a decoded plane), instr: one instruction of the filter program,
#   lookup_row: one live row checked against a dimension bitmap (extract the fk, test the bit).
COST = dict(lit=0.65, instr=2.7, lookup_row=9.5)

def _best_tz(lo, hi):
    """the value in [lo, hi] with the most trailing zero bits (0 if lo <= 0)"""
    if lo <= 0: return 0
    for t in range(int(hi).bit_length(), -1, -1):
        c = (hi >> t) << t
        if c >= lo: return c
    return hi

def _tz(v, nb): return nb if v == 0 else (v & -v).bit_length() - 1

def _blocks(A, B):
    """the aligned blocks that exactly cover [A, B): [(start, k)] with block size 2^k"""
    out, cur = [], A
    while cur < B:
        k = (cur & -cur).bit_length() - 1 if cur else B.bit_length() - 1
        while cur + (1 << k) > B: k -= 1
        out.append((cur, k)); cur += 1 << k
    return out

class JoinPoly:
    """A dimension filter seen from the fact: the Boolean polynomial f(fk) with f(pk) = [the dimension row passes].
    The equi-join is the substitution pk := fk, so f is evaluated directly on the fk's bit planes.
    Variables: the id bits the filtered attributes depend on when they are coordinates (attr_coords), else all id
    bits. In those variables f is a sum of interval terms over the codes that occur (unused codes are don't-care,
    so each interval can widen into the gaps around it, which buys bounds with trailing zeros: fewer planes).
    Every interval is the comparator polynomial [code >= A][code < B], evaluated in one of two forms:
      factored  one pass over the code's planes per bound (P_CGE / P_CLT; a bound's trailing zero bits are skipped)
      expanded  its aligned blocks, one cube (AND of literals) each (P_CUBE)
    The complement may be cheaper (f = NOT g). The whole f may instead be evaluated by lookup (bitmap semi-join);
    that is what an f with no structure in these variables costs, and the planner picks it when it is cheaper."""
    def __init__(self, space, attrs, bm, b):
        space = _sp(space); self.b = b
        R = None
        if attrs:
            Rs = set()
            for a in attrs:
                c = attr_coords(space, a, b)
                if c is None: Rs = None; break
                Rs |= set(c[0])
            if Rs is not None and len(Rs) <= 24: R = sorted(Rs, reverse=True)
        if R is None:
            R = list(range(b - 1, -1, -1)); codes = space.ids[space.ids < (1 << b)]; reps = codes
        else:
            codes, reps = coord_space(space, R, b)
        self.R, self.nb, self.codes = R, len(R), codes
        rr = space.row_of(reps)                         # the filter bitmap is over dimension rows
        self.sel = ((bm[rr >> 6] >> (rr & 63).astype(np.uint64)) & np.uint64(1)).astype(bool) if len(reps) else np.zeros(0, bool)
        self.space = "code" if len(R) < b else "id"

    def _terms(self, sel, form, budget, invert=False):
        """interval terms of the selected codes with their chosen form; None if over budget (native: jp_terms)"""
        if NATIVE_PLAN:
            if not hasattr(self, "_codes64"):
                self._codes64 = np.ascontiguousarray(self.codes, np.int64); self._sel8 = np.ascontiguousarray(self.sel).view(np.uint8)
            cap = 4096; cost = ctypes.c_double(0)
            while True:
                buf = np.empty(cap, np.int64)
                w = lib.jp_terms(self._sel8.ctypes.data, int(invert), self._codes64.ctypes.data, len(self._codes64), self.nb,
                                 {"factored": 0, "expanded": 1}.get(form, 2), float(budget), COST["lit"], COST["instr"],
                                 buf.ctypes.data, cap, ctypes.byref(cost))
                if w != -2 or cap > (1 << 24): break
                cap *= 8
            if w == -1: return None
            if w >= 0:
                terms, v, i = [], buf[:w].tolist(), 0
                while i < w:
                    pick, np_ = v[i], v[i + 1]; parts = [(v[i + 2 + 2 * k], v[i + 3 + 2 * k]) for k in range(np_)]
                    if pick == 0: terms.append(("factored", [("CGE" if op == 0 else "CLT", K) for op, K in parts]))
                    else: terms.append(("expanded", parts))
                    i += 2 + 2 * np_
                return terms, cost.value
        if invert: sel = ~sel
        nb, codes, full = self.nb, self.codes, 1 << self.nb
        e = np.diff(np.concatenate([[0], sel.astype(np.int8), [0]]))
        st, en = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0] - 1
        if len(st) * COST["instr"] > budget: return None
        terms, cost = [], COST["instr"] * max(0, len(st) - 1)          # the ORs joining the terms
        for i, j in zip(st.tolist(), en.tolist()):
            lo_min = int(codes[i - 1]) + 1 if i > 0 else 0
            hi_max = int(codes[j + 1]) if j + 1 < len(codes) else full
            A, B = _best_tz(lo_min, int(codes[i])), _best_tz(int(codes[j]) + 1, hi_max)
            bounds = [(op, K) for op, K, need in (("CGE", A, A > 0), ("CLT", B, B < full)) if need]
            cf = sum(COST["lit"] * (nb - _tz(K, nb)) + COST["instr"] for _, K in bounds) + COST["instr"] * (len(bounds) == 2)
            blocks = _blocks(A, B)
            ce = sum(COST["lit"] * (nb - k) + COST["instr"] for _, k in blocks) + COST["instr"] * (len(blocks) - 1)
            pick = form if form in ("factored", "expanded") else ("factored" if cf < ce else "expanded")
            terms.append((pick, bounds if pick == "factored" else blocks)); cost += cf if pick == "factored" else ce
            if cost > budget: return None
        return terms, cost

    def plan(self, form="auto", lookup_cost=None):
        """-> (cost, complement, terms) for the cheapest evaluation, or None when lookup is cheaper"""
        budget = float("inf") if form != "auto" else (lookup_cost if lookup_cost is not None else 64 * COST["lookup_row"])
        if not self.sel.any(): return (0.0, False, [])                  # FALSE
        if self.nb == 0 or self.sel.all(): return (0.0, True, [])       # TRUE (complement of the empty sum)
        best = None
        for comp in (False, True):
            r = self._terms(self.sel, form, budget, invert=comp)
            if r is None: continue
            cost = r[1] + COST["instr"] * comp
            if best is None or cost < best[0]: best = (cost, comp, r[0])
        if form != "auto" and best is not None and sum(len(t[1]) for t in best[2]) > 400: return None
        return best

    def program(self, fkc, planned):
        """filter-program instructions for a planned polynomial (leaves one word on the stack)"""
        _, comp, terms = planned
        nb, b, R = self.nb, self.b, self.R
        plane = lambda i: b - 1 - R[i]                                   # code bit i (0 = most significant)
        out = []
        if not terms: out += [PInstr(P["TRUE"], 0, 0, 0), PInstr(P["NOT"], 0, 0, 0)]     # the empty sum is FALSE
        for t, (form, parts) in enumerate(terms):
            sub = []
            if form == "factored":
                for op, K in parts:
                    tz = _tz(K, nb); top = nb - tz
                    mk = 0
                    for i in range(top): mk |= 1 << plane(i)
                    sub.append(PInstr(P[op], fkc, K >> tz, mk))
                if len(sub) == 2: sub.append(PInstr(P["AND"], 0, 0, 0))
                if not sub: sub.append(PInstr(P["TRUE"], 0, 0, 0))
            else:
                for q, (start, k) in enumerate(parts):
                    L = nb - k
                    if L == 0: sub.append(PInstr(P["TRUE"], 0, 0, 0))
                    else:
                        mk = vk = 0; v = start >> k
                        for i in range(L):
                            mk |= 1 << plane(i)
                            if (v >> (L - 1 - i)) & 1: vk |= 1 << plane(i)
                        sub.append(PInstr(P["CUBE"], fkc, vk, mk))
                    if q: sub.append(PInstr(P["OR"], 0, 0, 0))
            out += sub
            if t: out.append(PInstr(P["OR"], 0, 0, 0))
        if comp: out.append(PInstr(P["NOT"], 0, 0, 0))
        return out

    def evaluate(self, planned, vals):
        """the planned polynomial at given fk values (to find the stale points whose predicate flips)"""
        _, comp, terms = planned
        vals = np.asarray(vals, np.int64); code = np.zeros(len(vals), np.int64)
        for k in self.R: code = (code << 1) | ((vals >> k) & 1)
        res = np.zeros(len(vals), bool)
        for form, parts in terms:
            if form == "factored":
                t = np.ones(len(vals), bool)
                for op, K in parts: t &= (code >= K) if op == "CGE" else (code < K)
            else:
                t = np.zeros(len(vals), bool)
                for start, k in parts: t |= (code >> k) == (start >> k)
            res |= t
        return ~res if comp else res

    @staticmethod
    def describe(planned):
        _, comp, terms = planned
        if not terms: return "TRUE" if comp else "FALSE"
        nf = sum(len(p) for f, p in terms if f == "factored"); nc = sum(len(p) for f, p in terms if f == "expanded")
        kind = "comparators" if not nc else "cubes" if not nf else "cubes+comparators"
        return kind, nf + nc

def pred_cols(p):
    if isinstance(p, (Cmp, Between, CodeSet)): return {p.col.name}
    if isinstance(p, RowMask): return set(p.cols)
    if isinstance(p, BoolOp): return pred_cols(p.l) | pred_cols(p.r)
    if isinstance(p, Not): return pred_cols(p.x)
    return set()

def tables_of(p):
    if isinstance(p, (Cmp, Between, CodeSet)): return {p.col.table}
    if isinstance(p, RefOk): return {p.table}
    if isinstance(p, RowMask): return {p.table}
    if isinstance(p, BoolOp): return tables_of(p.l) | tables_of(p.r)
    if isinstance(p, Not): return tables_of(p.x)
    if isinstance(p, Col): return {p.table}
    if isinstance(p, Arith): return tables_of(p.l) | tables_of(p.r)
    return set()
def conjuncts(p):
    if p is None: return []
    if isinstance(p, BoolOp) and p.op == "AND": return conjuncts(p.l) + conjuncts(p.r)
    return [p]

# ------------------------------------------------------------------ tables
# Lifetimes. A Table owns its native storage (a _Storage): the torble (built planes, or the private mapping of a loaded
# file), the sidecar torbles its self-extending columns point into, the plain baseline's column table (which points at
# the Table's own uint32 arrays) and the GPU copy (decoded copies of the planes: it keeps nothing of the host). Every
# borrower holds a strong reference to the Table: a Query its fact and dimensions (its compiled plans hold no table
# pointer: the torble is passed at each run), a relationship Catalog and the math modules' objects (Scope, Column,
# Dimension, Layout, ...) their table. The math modules' native states copy what they read and know the table only by
# address; a relationship catalog keeps the torble's address and decodes through it after checking its revision. So the
# finalizer, which runs once no Table object over the storage is left, never frees memory in use. close() frees the
# storage at once but the torble's header, which stays (emptied, of a new revision) until the finalizer: a catalog
# then finds a stale table where it would read freed memory, and every use through the Table raises TableClosedError.
class TableClosedError(RuntimeError):
    """the table was closed (Table.close, Database.close): its native storage has been released"""


class _Storage:
    """the native handles of one table (see _release): at is None once the table is closed; header is the torble's
    address until the finalizer frees it; gpu its GPU copy (glib: the GPU library); gtags the tags of its live arrays
    that GPU handles may hold device copies of (see _gpu_tag)"""
    __slots__ = ("at", "header", "pt", "sides", "gpu", "glib", "gtags")
    def __init__(self): self.at = self.header = self.pt = self.gpu = self.glib = None; self.sides, self.gtags = [], set()


# Device copies of host arrays. Besides its planes (uploaded once per table: gpu_handle), a GPU query reads host arrays:
# the value arrays of its looked-up dimension attributes (a dimension's decoded or ranked column) and its delta plane.
# gpu.cu keeps device copies of them in the fact's handle for later queries, keyed by address, length and a tag: a
# serial given here to one array object, never reused. An address and a length do not name an array: once an array is
# freed, another can be allocated there with other values, by a table built later or by the same table (the ranked
# columns of a table's copy, which sqlexec binds when a query joins a table twice, die with the copy; a delta plane
# dies with its query). An array's copies are evicted from every handle when it is collected, and those of a table's
# arrays when its storage is released (close, or the finalizer). An array not found among the columns of an open joined
# table, other than the query's delta plane, has no tag (0): it is uploaded for its query only. The lock serializes the
# GPU calls: an eviction can run in a finalizer, in whichever thread drops the last reference, during another thread's
# GPU query.
import itertools as _itertools, threading as _threading
_GPU_LIVE = set()                       # the storages that hold a GPU handle
_GPU_TAGS = {}                          # id(array) -> (weak reference to the array, its tag)
_GPU_SERIAL = _itertools.count(1)
_GPU_LOCK = _threading.RLock()


def _gpu_evict(tags, live=_GPU_LIVE, lock=_GPU_LOCK):
    """drop the device copies of these tags from every GPU handle"""
    if not tags: return
    arr = (ctypes.c_uint64 * len(tags))(*tags)
    with lock:
        for s in list(live):
            if s.gpu: s.glib.gpu_cache_evict(s.gpu, arr, len(tags))


def _gpu_forget(key, tag, owner, tags=_GPU_TAGS, evict=_gpu_evict):
    """the array of this tag was collected: no query presents the tag again, and its copies are dropped"""
    if tags.get(key, (None, 0))[1] == tag: del tags[key]
    if owner is not None: owner.gtags.discard(tag)
    evict((tag,))


def _gpu_tag(arr, owner=None):
    """the tag of host array arr in the GPU caches (see above); owner: the storage whose release drops its copies"""
    key = id(arr); e = _GPU_TAGS.get(key)
    if e is not None and e[0]() is arr: return e[1]
    tag = next(_GPU_SERIAL); _GPU_TAGS[key] = (weakref.ref(arr), tag)
    weakref.finalize(arr, _gpu_forget, key, tag, owner).atexit = False   # at exit the handles are closed: nothing to drop
    if owner is not None: owner.gtags.add(tag)
    return tag


def _release(s, final=True, at_free=lib.at_free, at_release=lib.at_release, pt_free=lib.pt_free,
             evict=_gpu_evict, live=_GPU_LIVE, lock=_GPU_LOCK):
    """free a table's storage: the device copies of its arrays in other tables' GPU handles (see _gpu_tag), its GPU
    copy, the plain baseline's table, the torble's planes (or mapping), then the sidecars (the torble points into
    them); with final (the finalizer), also the torble's header, which close() keeps. Each handle is cleared before it
    is freed. The functions are bound here, for a release at interpreter exit."""
    s.at = None
    tags, s.gtags = s.gtags, set()
    evict(tags)
    with lock:
        live.discard(s)
        gpu, s.gpu = s.gpu, None
        if gpu: s.glib.gpu_close(gpu)
    pt, s.pt = s.pt, None
    if pt: pt_free(pt)
    if s.header: at_release(s.header)
    sides, s.sides = s.sides, []
    for side in sides:
        if side: at_free(side)
    if final:
        header, s.header = s.header, None
        if header: at_free(header)


class _Native:
    """A table's storage and its finalizer, shared by the Table objects over it (copy.copy(table) is another Table over
    the same storage: sqlexec binds a table twice that way). The finalizer (weakref.finalize: it also runs at interpreter
    exit) frees the storage when the last of them is collected, once."""
    def __init__(self):
        self.s = _Storage()
        self.release = weakref.finalize(self, _release, self.s)


class _ClosedColumns(dict):
    """the plain columns of a closed table (released with it): reading one raises TableClosedError"""
    def __init__(self, name): super().__init__(); self.name = name
    def __missing__(self, c): raise TableClosedError(f"table {self.name!r} is closed: column {c!r} is released")


class Table:
    def __init__(self, name, columns, msb=(), used=None, refs=None, cluster=None, extend=None, virtual=None):
        """msb: columns stored most-significant-bit first (hierarchical foreign keys);
        used: for a dimension indexed by hierarchical id, which ids exist;
        refs: {fk column: dimension Table}. Whether each foreign key points at an existing row is recorded at load
        time in the hidden column _refok (one bit per foreign key, stored and protected like every other column);
        every join requires that bit, so orphaned references never join."""
        self.name = name
        self.data = {}
        self.codecs = {}                 # typed columns: {col: coltypes codec}; empty for plain uint32 tables
        self._wmax = {}
        # self-extending columns: extend={col: B}. A value v (up to B + 32 bits) is stored as base = v mod 2^B for every
        # row, a flag plane (v >= 2^B), and hi = v >> B only for the flagged rows, in a sidecar torble addressed by rank
        # (a popcount over the flag plane: no stored pointer). Queries see the logical value base + 2^B hi.
        self.wide, virtual, virtual_arg = {}, {}, virtual
        if extend:
            columns = dict(columns)
            for col, B in extend.items():
                v = np.asarray(columns[col], np.uint64); B = int(B)
                self._wmax[col] = int(v.max()) if v.size else 0
                hi = v >> np.uint64(B)
                if hi.size and int(hi.max()) > 0xFFFFFFFF: raise ValueError(f"{name}.{col}: values need more than {B} + 32 bits")
                flag = hi != 0
                columns[col] = (v & np.uint64((1 << B) - 1)).astype(np.uint32)
                columns[col + "#x"] = flag.astype(np.uint32)
                columns[col + "#hi"] = hi.astype(np.uint32)                 # dense copy: plain-engine baseline only
                self.wide[col] = (B, col + "#hi", col + "#x")
                virtual[col + "#hi"] = (col + "#x", hi[flag].astype(np.uint32), 0)
        if virtual_cols := virtual_arg:
            for col, (fc, vmsb) in virtual_cols.items():      # user-declared virtual columns: values only where fc != 0
                fl = np.asarray(columns[fc]).astype(bool)
                virtual[col] = (fc, np.asarray(columns[col], np.uint32)[fl].astype(np.uint32), 1 if vmsb else 0)
        for k, v in columns.items():
            a = np.asarray(v)
            if a.dtype != np.uint32:
                if a.size and (a.min() < 0 or a.max() > 0xFFFFFFFF): raise ValueError(f"{name}.{k}: values must fit in uint32")
                a = a.astype(np.uint32)
            self.data[k] = np.ascontiguousarray(a)
        self.refbit, self.surr = {}, {}
        if refs:
            ok = np.zeros(len(next(iter(self.data.values()))), np.uint32)
            self._refdim = {}
            for bit, (fk, ref) in enumerate(refs.items()):
                dim, sg = ref if isinstance(ref, tuple) else (ref, None)
                self._refdim[fk] = dim
                v = self.data[fk]
                if sg is None or not len(sg.stale_arrays()[0]):     # native, all cores
                    sor = None if sg is None else np.ascontiguousarray(sg.s_of_row, np.int64)
                    lib.tb_refok(v.ctypes.data, len(v), 0 if sg is None else sg.mask, dim.n if sor is None else min(dim.n, len(sor)), dim.used_bm.ctypes.data,
                                 None if sor is None else sor.ctypes.data, bit, ok.ctypes.data)
                    self.refbit[fk] = bit; self.surr[fk] = sg; continue
                k = v if sg is None else (v & np.uint32(sg.mask))
                inside = k < dim.n
                valid = np.zeros(len(v), bool); valid[inside] = dim.used[k[inside]]
                if sg is not None:                     # a derived surrogate must agree with its row's derivation
                    cur = sg.s_of_row[k[inside]] == v[inside]
                    pts, _ = sg.stale_arrays()             # (or with an earlier version of it, written before a change)
                    if len(pts): cur |= np.isin(v[inside].astype(np.int64), pts)
                    valid[inside] &= cur
                ok |= valid.astype(np.uint32) << bit; self.refbit[fk] = bit; self.surr[fk] = sg
                del k, inside, valid
            self.data["_refok"] = ok
            self.orphans = {fk: int(((ok >> b) & 1 == 0).sum()) for fk, b in self.refbit.items()}
        self.cols = list(self.data)
        if len(self.cols) > MAXCOL:
            raise ValueError(f"{name}: {len(self.cols)} columns (hidden ones included); a torble holds at most {MAXCOL}")
        # physical-order hint (e.g. facts arrive in time order). Only speed depends on it: GROUP BY on this column
        # splits each 64-row word by key, which takes one step per distinct key present in the word.
        self.cluster = cluster
        if cluster is not None:
            v = self.data[cluster]; self.cluster_breaks = int(np.count_nonzero(v[1:] != v[:-1]))
            self.cluster_sorted = bool(np.all(v[1:] >= v[:-1]))   # sorted: the physical order is an index on it
        self.msb_cols = set(msb)
        self.n = len(next(iter(self.data.values())))
        self.used = np.ones(self.n, bool) if used is None else np.asarray(used, bool)
        self._used_ids = None                          # ids in use, computed when first needed (dimensions)
        bits = np.zeros(((self.n + 63) // 64 + 1) * 64, np.uint8); bits[:self.n] = self.used
        self.used_bm = np.packbits(bits, bitorder='little').view(np.uint64)   # ids in use, as a bitmap
        zeros = np.zeros(self.n, np.uint32) if virtual else None    # virtual columns keep no planes of their own
        aptrs = (ctypes.c_void_p * len(self.cols))(*[(zeros if c in virtual else self.data[c]).ctypes.data for c in self.cols])
        ptrs = (ctypes.c_void_p * len(self.cols))(*[self.data[c].ctypes.data for c in self.cols])
        flags = (ctypes.c_int32 * len(self.cols))(*[1 if c in msb else 0 for c in self.cols])
        self._own(lib.at_create(self.n, len(self.cols), aptrs, flags), lib.pt_create(self.n, len(self.cols), ptrs))
        if not self.pt: raise MemoryError(f"{name}: the plain baseline's table could not be allocated")
        self.sidecars = {}
        for c, (fc, compact, vmsb) in virtual.items():
            fl = self.data[fc].astype(bool); nw = (self.n + 63) // 64
            per = np.zeros(nw * 64, np.uint32); per[:self.n] = fl
            rank = np.concatenate([[0], np.cumsum(per.reshape(nw, 64).sum(1))])[:nw].astype(np.uint32)
            body = compact if len(compact) else np.zeros(1, np.uint32)
            side = self._own_side(lib.at_create(len(body), 1, (ctypes.c_void_p * 1)(body.ctypes.data), (ctypes.c_int32 * 1)(vmsb)))
            lib.at_set_virtual(self.at, self.idx(c), self.idx(fc), side, rank.ctypes.data)
            self.sidecars[c] = dict(side=side, rank=rank, rows=len(compact), body=body)
        del zeros
        self._decoded = {}; self._colmax = {}
    # ---- ownership and release (see _Native)
    def _own(self, at, pt=None):
        """this new Table takes the torble at (and the plain baseline's table pt): from here on they are freed with it"""
        nat = _Native(); nat.s.at, nat.s.header, nat.s.pt = at, at, pt
        if not at:
            nat.release(); raise MemoryError(f"{self.__dict__.get('name', 'table')}: the torble could not be allocated")
        self.__dict__["_native"] = nat
    def _own_side(self, side):
        """a sidecar torble of this table's self-extending columns: freed after the torble, which points into it"""
        self._storage().sides.append(side)
        if not side: raise MemoryError(f"{self.name}: a sidecar torble could not be allocated")
        return side
    def _storage(self):
        nat = self.__dict__.get("_native")
        if nat is None or nat.s.at is None:
            raise TableClosedError(f"table {self.__dict__.get('name', '?')!r} is closed: its native storage has been released"
                                   if nat is not None else f"table {self.__dict__.get('name', '?')!r} has no native storage")
        return nat.s
    @property
    def at(self):
        """the native torble (TableClosedError once the table is closed)"""
        return self._storage().at
    @property
    def pt(self):
        """the plain baseline's native column table, or None (dropped: drop_plain; loaded tables have none)"""
        return self._storage().pt
    @property
    def closed(self):
        nat = self.__dict__.get("_native")
        return nat is None or nat.s.at is None
    def close(self):
        """Release the native storage now: the planes (a loaded table's file mapping), the sidecars, the plain baseline's
        columns, the GPU copy and the device copies of its columns held by the GPU handles of the tables it was joined
        to; the relationship catalog attached to the table is closed with it. Any later use of the storage, through this
        Table, its copies (copy.copy shares it) or the objects built on it (queries, scopes, ...), raises
        TableClosedError; a catalog kept apart (attach=False) reports a stale table. Closing again does nothing.
        Only the torble's header (an empty torble) stays until the last Table object over it is collected; without
        close(), everything is freed then. Not while another thread is running a query on the table."""
        nat = self.__dict__.get("_native")
        if nat is None: return
        catalog = self.__dict__.get("_relationships")
        if catalog is not None: catalog.close()        # the table's own catalog (it detaches itself)
        if nat.s.at is not None: _release(nat.s, final=False)
        self.sidecars, self._decoded, self.data = {}, {}, _ClosedColumns(self.__dict__.get("name", "?"))
        for a in ("_ranked", "_deltas"): self.__dict__.pop(a, None)
    def __deepcopy__(self, memo):
        """a table is not copied by deepcopy (a copy of a query or predicate keeps its tables: their identity and their
        storage's single owner)"""
        return self
    def __getattr__(self, k):
        if k in ("cols", "data") or k.startswith("_"): raise AttributeError(k)
        if k in self.cols: return Col(self, k)
        raise AttributeError(k)
    def __getitem__(self, k):
        """a column by name (for names that are not Python identifiers: t["Région"])"""
        if k in self.cols: return Col(self, k)
        raise KeyError(f"{self.name} has no column {k!r}")
    def idx(self, c): return self.cols.index(c)
    def width(self, c): return lib.at_width(self.at, self.idx(c))
    def nrows(self): return len(self.used_ids)
    def decoded(self, c):          # dimension attribute as integers, decoded from the torus planes
        if c not in self._decoded:
            out = np.empty(self.n, dtype=np.uint32); lib.at_decode_column(self.at, self.idx(c), out.ctypes.data)
            self._decoded[c] = out
        return self._decoded[c]
    def space(self, fk, dim):
        """the id space of the dimension joined through foreign key fk (row ids or derived surrogates)"""
        if not hasattr(self, "_spaces"): self._spaces = {}
        sg = getattr(self, "surr", {}).get(fk)
        ck = (fk, id(dim), None if sg is None else sg.version)
        if ck not in self._spaces:
            self._spaces[ck] = _sp(dim) if sg is None else IdSpace(dim, sg)
        return self._spaces[ck]
    def delta(self, fk, dim):
        """the delta plane of foreign key fk: one bit per fact row, set where the referenced dimension row has moved
        since the fact was written (found once per change by a scan of the fk planes; cached per change version)"""
        sg = self.surr[fk]
        if not hasattr(self, "_deltas"): self._deltas = {}
        ck = (fk, sg.version)
        if ck not in self._deltas:
            bm = np.zeros((dim.n + 63) // 64 + 1, np.uint64)
            for r in sg.stale: bm[r >> 6] |= np.uint64(1) << np.uint64(r & 63)
            out = np.zeros((self.n + 63) // 64, np.uint64)
            cnt = lib.at_rows_semi(self.at, self.idx(fk), sg.mask, bm.ctypes.data, dim.n, out.ctypes.data)
            self._deltas[ck] = (out, int(cnt))
        return self._deltas[ck]
    def kmask(self, fk):
        sg = getattr(self, "surr", {}).get(fk); return 0 if sg is None else sg.mask
    @property
    def used_ids(self):
        if self._used_ids is None: self._used_ids = np.nonzero(self.used)[0]
        return self._used_ids
    def gpu_handle(self):
        """upload the value planes to the GPU once (decoded there by a prefix-XOR kernel); the torus stays here. The
        copy is part of the table's storage: released with it (close, or the finalizer)."""
        s = self._storage()
        if s.gpu is None:
            import time as _t; t0 = _t.perf_counter()
            g = gpu_lib(); h = g.gpu_open(s.at); self.gpu_upload_s = _t.perf_counter() - t0
            if not h: raise RuntimeError("GPU upload failed")
            s.gpu, s.glib = h, g; _GPU_LIVE.add(s)
        return s.gpu
    def colmax(self, c):
        if c not in self._colmax: self._colmax[c] = int(self.data[c].max()) if self.n else 0
        return self._colmax[c]
    def maxcode(self, c):
        """the largest code stored in column c (the logical value for a self-extending column)"""
        return self._wmax[c] if c in getattr(self, "_wmax", {}) else self.colmax(c)
    def fks_to(self, dim):
        """the declared foreign keys of this table that reference dim"""
        return [fk for fk in getattr(self, "refbit", {}) if getattr(self, "_refdim", {}).get(fk) is dim]
    def plain_handle(self):
        """the plain columnar engine's table (a test baseline): only kept when the database was built with keep_plain"""
        if self.pt is None:
            raise RuntimeError(f"engine 'plain' is not available for {self.name}: build the database with keep_plain=True")
        return self.pt
    def drop_plain(self):
        """free the plain uint32 columns once the torble is built (the bit planes are the data; the plain columnar
        engine, a test baseline, is unavailable afterwards). Keeps per-column maxima and the clustering column."""
        for c in self.cols: self.colmax(c)
        keep = {self.cluster} if self.cluster else set()
        s = self._storage(); pt, s.pt = s.pt, None
        if pt: lib.pt_free(pt)                     # before its columns go: it points at them
        self.data = {c: v for c, v in self.data.items() if c in keep}
    def ranked(self, c, engine="alg"):
        """(distinct values in use, each id's value as a rank among them): a dense GROUP BY key, cached"""
        ck = (c, engine)
        if not hasattr(self, "_ranked"): self._ranked = {}
        if ck not in self._ranked:
            vals = self.decoded(c) if engine == "alg" else self.data[c]
            u = np.unique(vals[self.used_ids]); self._ranked[ck] = (u, np.searchsorted(u, vals).astype(np.uint32))
        return self._ranked[ck]
    def torus_bytes(self): return lib.at_bytes(self.at)
    def sidecar_bytes(self):
        self._storage()                            # a copy of a closed table still lists its released sidecars
        return sum(lib.at_bytes(v["side"]) for v in getattr(self, "sidecars", {}).values())
    def get_rows(self, c, rows):
        """values of column c at the given rows, straight from the planes (virtual columns included)"""
        rows = np.ascontiguousarray(rows, np.uint64); out = np.empty(len(rows), np.uint32)
        if len(rows) and int(rows.max()) >= self.n:          # the planes hold n rows (none, for an empty table)
            raise IndexError(f"{self.name}: row {int(rows.max())} of a table of {self.n} rows")
        lib.at_get_rows(self.at, self.idx(c), rows.ctypes.data, len(rows), out.ctypes.data); return out
    def plain_bytes(self): return 4 * self.n * len(self.cols)
    def repair(self): return lib.at_repair(self.at)
    def nplanes(self): return lib.at_nplanes(self.at)
    def flip(self, plane, row):
        """fault injection: flip the bit of row in the plane-th stored plane"""
        if not 0 <= int(plane) < self.nplanes() or not 0 <= int(row) < self.n:   # the planes hold n rows (none if empty)
            raise IndexError(f"{self.name}: no stored bit at plane {int(plane)} (of {self.nplanes()}), row {int(row)} (of {self.n})")
        lib.at_flip(self.at, int(plane), int(row))
    def validate(self, maxbad=8):
        bad = (ctypes.c_uint64 * maxbad)(); n = lib.at_validate(self.at, bad, maxbad); return n, list(bad[:min(n, maxbad)])

def compile_pred(p, table, prog):
    if isinstance(p, RefOk):
        prog.append(PInstr(P["BIT"], table.idx("_refok"), table.refbit[p.fk], 0))
        if not p.ok: prog.append(PInstr(P["NOT"], 0, 0, 0))
        return
    if isinstance(p, Cmp): prog.append(PInstr(P[p.op], table.idx(p.col.name), p.v, 0))
    elif isinstance(p, Between): prog.append(PInstr(P["BETWEEN"], table.idx(p.col.name), p.a, p.b))
    elif isinstance(p, BoolOp): compile_pred(p.l, table, prog); compile_pred(p.r, table, prog); prog.append(PInstr(P[p.op], 0, 0, 0))
    elif isinstance(p, Not): compile_pred(p.x, table, prog); prog.append(PInstr(P["NOT"], 0, 0, 0))
    else: raise TypeError(p)
_OPF = dict(EQ=lambda a, b: a == b, NE=lambda a, b: a != b, LT=lambda a, b: a < b, LE=lambda a, b: a <= b,
            GT=lambda a, b: a > b, GE=lambda a, b: a >= b)
def pred_sql(p):
    if getattr(p, "sql", None): return p.sql
    if isinstance(p, (Cmp, Between)):
        base = _valid_base(p.col.table, p.col.name)
        if base is not None:                       # on a validity plane (codes 0 / 1): its column IS [NOT] NULL
            hold = tuple(c for c in (0, 1) if (p.a <= c <= p.b if isinstance(p, Between) else _OPF[p.op](c, p.v)))
            b = _qc(Col(p.col.table, base))
            return {(1,): f"{b} IS NOT NULL", (0,): f"{b} IS NULL", (0, 1): "TRUE", (): "FALSE"}[hold]
    if isinstance(p, Cmp): return f"{_qc(p.col)} {dict(EQ='=', NE='<>', LT='<', LE='<=', GT='>', GE='>=')[p.op]} {p.v}"
    if isinstance(p, Between): return f"{_qc(p.col)} BETWEEN {p.a} AND {p.b}"
    if isinstance(p, BoolOp): return f"({pred_sql(p.l)} {p.op} {pred_sql(p.r)})"
    if isinstance(p, Not): return f"(NOT {pred_sql(p.x)})"
def expr_sql(e):
    if isinstance(e, Col): return _qc(e)
    if isinstance(e, Const):
        if e.v is not None: return str(e.v)
        from coltypes import _frac
        f = _frac(e.raw); from decimal import Decimal as _D
        return format(_D(f.numerator) / _D(f.denominator), "f")
    return f"({expr_sql(e.l)} {e.op} {expr_sql(e.r)})"
def _requires(p, table, vc):
    """does predicate p hold only on rows where validity plane vc of table is 1? (by its form: a conjunct vc = 1, or
    every disjunct requiring it, as a comparison on a nullable column or db.sql's IS NOT NULL give)"""
    if isinstance(p, Cmp):
        return p.col.table is table and p.col.name == vc and ((p.op == "EQ" and p.v == 1) or (p.op == "GE" and p.v == 1) or (p.op == "GT" and p.v == 0))
    if isinstance(p, Between): return p.col.table is table and p.col.name == vc and p.a >= 1
    if isinstance(p, BoolOp) and p.op == "AND": return _requires(p.l, table, vc) or _requires(p.r, table, vc)
    if isinstance(p, BoolOp) and p.op == "OR": return _requires(p.l, table, vc) and _requires(p.r, table, vc)
    return False
def _expr_cols(e, out=None):
    """the columns an expression reads"""
    out = [] if out is None else out
    if isinstance(e, Col): out.append(e)
    elif isinstance(e, Arith): _expr_cols(e.l, out); _expr_cols(e.r, out)
    return out

def dominance(M, H, nb):
    """sum over x < t of M[:, x] H[:, t] (a matrix: fact groups x dimension groups), by the comparator polynomial
    [x < t] = sum_j (1 + x_j) t_j E_j (MSB first): at level j, x and t agree above bit j and x_j = 0, t_j = 1, so
    the even blocks of size 2^j of M pair with the odd blocks of H. Exact integer arithmetic."""
    out = np.zeros((M.shape[0], H.shape[0]), M.dtype)
    Mj, Hj = M, H
    for j in range(nb):
        out = out + Mj[:, 0::2] @ Hj[:, 1::2].T
        Mj, Hj = Mj[:, 0::2] + Mj[:, 1::2], Hj[:, 0::2] + Hj[:, 1::2]
    return out

# ------------------------------------------------------------------ typed measures
from decimal import Decimal
def _has_fraction(e):
    if isinstance(e, Const): return e.v is None
    if isinstance(e, Arith): return _has_fraction(e.l) or _has_fraction(e.r)
    return False

def _bridge_col(c):
    """how a measure column takes part in the histogram bridge: None (summable as is), "hide" (no affine code but exact
    values, integers or decimals: summed per value), "no" (cannot take part: DOUBLE, strings, dates, ...)"""
    cd = c.codec()
    if cd is None or cd.affine is not None: return None
    base = getattr(cd, "base", None)
    if base is not None: return "hide" if getattr(base, "affine", None) is not None else "no"      # rank of int / decimal
    if getattr(cd, "kind", "") == "dict" and cd.numeric and not cd.floating: return "hide"
    return "no"

def _bridge_reason(q, e):
    """why a failed query can go through the histogram bridge: "codec" (a column without an affine code), "wide" (128
    bits needed, measure not a degree-2 polynomial of fact columns), "bound" (could pass 128 bits); None: it cannot"""
    if q.measure is None or getattr(q, "_nobridge", False) or getattr(q, "_codes", False) or getattr(q, "rjoin", None) is not None: return None
    msg = str(e)
    if isinstance(e, TypeError) and "cannot be summed" in msg:
        cols = []
        def walk(x):
            if isinstance(x, Col): cols.append(x)
            elif isinstance(x, Arith): walk(x.l); walk(x.r)
        walk(q.measure)
        kinds = [_bridge_col(c) for c in cols]
        return "codec" if "hide" in kinds and "no" not in kinds else None
    if isinstance(e, OverflowError) and "polynomials of degree <= 2" in msg: return "wide"
    if isinstance(e, OverflowError) and "SUM may exceed 128 bits" in msg: return "bound"
    return None

def _expand(e):
    """a measure as exact monomials over its columns: {(column ids, sorted): Fraction}; a column id is (id(table), name)"""
    from fractions import Fraction
    from coltypes import _frac
    if isinstance(e, Const): return {(): _frac(e.raw)} if e.raw is not None else {(): Fraction(0)}
    if isinstance(e, Col): return {((id(e.table), e.name),): Fraction(1)}
    l, r = _expand(e.l), _expand(e.r); out = {}
    if e.op in "+-":
        for k, v in l.items(): out[k] = out.get(k, 0) + v
        for k, v in r.items(): out[k] = out.get(k, 0) + (v if e.op == "+" else -v)
    else:
        for ka, va in l.items():
            for kb, vb in r.items():
                k = tuple(sorted(ka + kb, key=lambda x: (x[1], x[0]))); out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v}

def _dec_scale_of(e):
    """the decimal scale of a typed measure, as SQL types it: products add scales, sums take the larger"""
    if isinstance(e, Const):
        from coltypes import _frac
        f, sc = _frac(e.raw), 0
        while f.denominator != 1 and sc < 38: f *= 10; sc += 1
        return sc
    if isinstance(e, Col):
        cd = e.codec()
        if cd is None: return 0
        if cd.affine is not None: return cd.affine[1]
        base = getattr(cd, "base", None)
        if base is not None and getattr(base, "affine", None) is not None: return base.affine[1]
        import re
        m = re.match(r"\s*DECIMAL\s*\(\s*\d+\s*,\s*(\d+)\s*\)", cd.sql_type.upper())
        return int(m.group(1)) if m else 0
    a, b = _dec_scale_of(e.l), _dec_scale_of(e.r)
    return a + b if e.op == "*" else max(a, b)

class CodeGroups:
    """a GROUP BY result as columns, in the keys' code space: cols (one code array per key), vals (one value per group:
    an int64 / object array of sums or counts, or a list of MinMax), cnt (the row counts, for (sum, count) results)"""
    __slots__ = ("cols", "vals", "cnt")
    def __init__(self, cols, vals, cnt=None): self.cols, self.vals, self.cnt = cols, vals, cnt


class Groups:
    """a typed GROUP BY result as columns (Query.run_columns): keys [(codes, valid mask or None, codec or None)] (the codes
    order and compare like the values; codec.decode gives the values), vals [one value per group, as run() gives them]"""
    __slots__ = ("keys", "vals")
    def __init__(self, keys, vals): self.keys, self.vals = keys, vals
    def __len__(self): return len(self.vals)


def _code_groups_dict(r, q):
    """a CodeGroups result as the dict _finish gives otherwise: {key code tuple: value} ((sum, count) for _both)"""
    keys = list(zip(*[np.asarray(c).tolist() for c in r.cols])) if r.cols else [()] * len(r.vals)
    vs = r.vals if isinstance(r.vals, list) else r.vals.tolist()
    if r.cnt is not None: vs = list(zip(vs, r.cnt.tolist()))
    return dict(zip(keys, vs))

def _bits_key(fact, name, lo, nb, engine):
    """a GROUP BY key on the code bits [lo, lo + nb) of a fact column -> (PlaneKey, whether its values come back
    bit-reversed). A plane key's mask names planes, and the engines number them differently: the plain baseline counts
    from the most significant code bit; the plane engines count physical planes (code bit j, or bit w - 1 - j for a
    column stored most significant bit first) and read the lowest masked plane as the key's top bit."""
    w = int(fact.width(name)); bits = range(lo, lo + nb)
    if engine == "plain": return PlaneKey(fact, name, sum(1 << (w - 1 - b) for b in bits)), False
    msb = name in getattr(fact, "msb_cols", ())
    return PlaneKey(fact, name, sum(1 << ((w - 1 - b) if msb else b) for b in bits)), not msb

def _codes_array(c):
    """key codes as an array for a codec's decode: int64, or Python integers past it (64-bit unsigned codes)"""
    try: return np.asarray(c, np.int64)
    except OverflowError: return np.asarray(c, object)

def _exact_dec(v, scale):
    """the integer v / 10^scale as a Decimal, every digit kept (128-bit sums have up to 39; the default context
    rounds to 28)"""
    return Decimal(f"{int(v)}E-{scale}")          # read from text: exact, no context rounding

def copy_query(q):
    import copy
    c = copy.copy(q); c._plan_cache = {}
    for a in ("_codeq", "_lowered_measure", "_fq", "_observed"):
        if a in c.__dict__: del c.__dict__[a]
    return c

def lower_measure(e, fact):
    """A measure over typed columns -> (an integer expression over stored codes, decimal scale). A column with an affine
    codec is (code + offset) / 10^s; constants are exact decimals. Sums and differences are brought to a common scale,
    products add scales. The engine sums the integer expression exactly; the caller divides by 10^scale. Raises when
    a column cannot be summed (strings, dates, floats) or when the sum could pass 128 bits."""
    from fractions import Fraction
    from coltypes import _frac
    def lo(e):
        if isinstance(e, Const):
            f = _frac(e.raw); sc = 0
            while f.denominator != 1:
                f *= 10; sc += 1
                if sc > 18: raise ValueError(f"constant {e.raw!r} has more than 18 decimal places")
            return Const(int(f)), sc
        if isinstance(e, Col):
            cd = e.codec()
            if cd is None: return e, 0
            if cd.affine is None:
                hint = (f"; load it as an exact DECIMAL: types={{'{e.table.name}.{e.name}': 'DECIMAL(18, 2)'}}"
                        if cd.sql_type.upper() in ("DOUBLE", "FLOAT", "REAL") else "")
                raise TypeError(f"{e.table.name}.{e.name} ({cd.sql_type}) cannot be summed or used in arithmetic{hint}")
            off, sc = cd.affine
            return (Arith("+", e, Const(off)) if off else e), sc
        if isinstance(e, Arith):
            (a, sa), (b, sb) = lo(e.l), lo(e.r)
            if e.op == "*": return Arith("*", a, b), sa + sb
            S = max(sa, sb)
            if sa < S: a = Arith("*", a, Const(10 ** (S - sa)))
            if sb < S: b = Arith("*", b, Const(10 ** (S - sb)))
            return Arith(e.op, a, b), S
        raise TypeError(f"unsupported measure term {e!r}")
    low, scale = lo(e)
    sum_width(low, fact)                       # refuses a sum that could pass 128 bits
    return low, scale

# ------------------------------------------------------------------ queries
class MinMax(tuple):
    """(min, max) of a group (Query.minmax)"""

def _merge_vals(a, b):
    """two partial results of one group: sums add, (sum, count) pairs add, (min, max) pairs combine"""
    if isinstance(a, MinMax): return MinMax((min(a[0], b[0]), max(a[1], b[1])))
    if isinstance(a, tuple): return tuple((x + y) if x is not None and y is not None else (x if y is None else y) for x, y in zip(a, b))
    return a + b

class Query:
    def __init__(self, fact, join_mode="prefix"):
        """join_mode: how a dimension filter (a polynomial f in the fk's bit variables, see JoinPoly) is evaluated.
             "auto" (alias "prefix")  cheapest by the cost model: per interval term factored or expanded, or lookup;
                                      GROUP BY on coordinate attributes reads them from the fk planes
             "factored" (alias "between")  every term as comparators, in the id's own variables
             "expanded"               every term as cubes
             "lookup" (alias "bitmap")     bitmap semi-join"""
        self.fact, self.joins, self.pred, self.keys, self.measure = fact, {}, None, [], None
        self.join_mode, self.range_used = join_mode, []
    def join(self, dim, fk=None):
        if fk is None:
            cands = self.fact.fks_to(dim)
            if len(cands) != 1:
                raise ValueError(f"{self.fact.name} has {len(cands)} foreign keys to {dim.name}" + (f" ({', '.join(cands)}): name one, join({dim.name}, fk)" if cands else ""))
            fk = Col(self.fact, cands[0])
        if isinstance(fk, str): fk = Col(self.fact, fk)
        if fk.name not in getattr(self.fact, "refbit", {}):
            raise ValueError(f"{self.fact.name}.{fk.name} is not a declared reference; declare refs= when building the table")
        self.joins[dim] = fk.name; return self
    def join_range(self, dim, x, lo=None, hi=None):
        """range / band / interval join: fact column x BETWEEN lo AND hi, where lo and hi are expressions over the
        dimension's columns (e.g. promo.start, event.day - 3); either may be None (unbounded). One per query."""
        if lo is None and hi is None: raise ValueError("a range join needs at least one bound")
        self.rjoin = (dim, x, None if lo is None else Expr._w(x, lo), None if hi is None else Expr._w(x, hi)); return self
    def join_ineq(self, dim, x, op, y):
        """inequality join: fact column x <op> y, op in < <= > >=, y an expression over the dimension's columns"""
        y = Expr._w(x, y)
        lo, hi = {"<": (None, y - 1), "<=": (None, y), ">": (y + 1, None), ">=": (y, None)}[op]
        self.rjoin = (dim, x, lo, hi); self.rjoin_sql = (op, y); return self
    def where(self, p): self.pred = p if self.pred is None else (self.pred & p); return self
    def group_by(self, *cols): self.keys = list(cols); return self
    # Aggregates skip NULLs as in SQL: SUM, MIN / MAX and COUNT(e) read the rows where every column of their argument is
    # not NULL; a group is there when it has rows, so one whose argument is NULL on all of them gives None (COUNT(e): 0).
    def sum(self, e): self.measure = e; self._ext = None; self._count = None; return self
    def count(self, e=None):
        """COUNT(*), or with a column (or an expression) COUNT(e): the rows where it is not NULL"""
        self.measure = None; self._ext = None; self._count = e if e is None or isinstance(e, Expr) else Const(e); return self
    def minmax(self, col):
        """MIN and MAX of a fact column per group, by plane elimination: run() gives (min, max), or {key: (min, max)}"""
        if not (isinstance(col, Col) and col.table is self.fact): raise ValueError("minmax() takes a column of the fact table")
        if col.name in getattr(self.fact, "wide", {}): raise ValueError(f"{col.name}: MIN / MAX of a self-extending column")
        self.measure = None; self._ext = col; self._count = None; return self
    def rank_codes(self, col, ranks, engine="alg", bits=8):
        """Rank selection on the planes: the codes at the given 0-based ranks (ascending) among the codes of fact column
        col on the rows that pass the filters and joins (NULL rows left out), without reading any row. Radix selection
        from the top plane down, `bits` planes per level: each level is one engine GROUP BY over those planes, restricted
        to the rows whose higher planes hold the prefix chosen so far (a code interval), and the cumulative counts of its
        2^bits buckets give the next digits of each rank's code. A negative rank counts from the top (-1: the largest).
        -> ({rank: code}, number of passing non-NULL rows), ranks as passed; ranks past that number are left out."""
        fact = self.fact
        if not (isinstance(col, Col) and col.table is fact): raise ValueError("rank_codes() takes a column of the fact table")
        if col.name in getattr(fact, "wide", {}): raise ValueError(f"{col.name}: rank selection on a self-extending column")
        w = int(fact.width(col.name))
        base = copy_query(self); base.keys = []; base.measure = None; base._ext = None; base._codes = True
        for a in ("_both", "_rows", "_cols_owner", "_dom"): base.__dict__.pop(a, None)
        vc = getattr(fact, "nullable", {}).get(col.name)
        if vc is not None:                                   # NULL rows have no rank
            nn = Cmp(Col(fact, vc), "EQ", 1)
            base.pred = nn if base.pred is None else BoolOp("AND", base.pred, nn)
        eng = "plain" if engine == "plain" else engine
        def counts(lo_hi, nb, sh):
            """passing rows per value of the code's bits [sh, sh + nb), on the code interval lo_hi (or all rows)"""
            q = copy_query(base); q._raw = True
            if sh or nb < w: key, reverse = _bits_key(fact, col.name, sh, nb, eng); q.keys = [key]
            else: q.keys = [Col(fact, col.name)]; reverse = False
            if lo_hi is not None:
                extra = Between(Col(fact, col.name), lo_hi[0], lo_hi[1])
                q.pred = extra if q.pred is None else BoolOp("AND", q.pred, extra)
            km, _, cn = q._run_codes(eng)
            k = np.asarray(km[:, 0], np.int64) if len(cn) else np.zeros(0, np.int64)
            if reverse and len(k):
                rev = np.zeros(len(k), np.int64)
                for b in range(nb): rev |= ((k >> b) & 1) << (nb - 1 - b)
                k = rev
            full = np.zeros(1 << nb, np.int64); np.add.at(full, k, np.asarray(cn, np.int64))
            return np.cumsum(full)
        if w == 0: return {}, 0
        nb0 = min(bits, w)
        cum = counts(None, nb0, w - nb0); total = int(cum[-1])
        ranks_in = list(ranks)
        ranks = [int(r) + total if int(r) < 0 else int(r) for r in ranks]          # negative: counted from the top
        want = sorted({r for r in ranks if 0 <= r < total})
        if not want: return {}, total
        ranks0 = list(ranks_in)
        state = {}                                           # rank -> (prefix, rank within the prefix's rows)
        for r in want:
            d = int(np.searchsorted(cum, r, side="right")); state[r] = (d, r - (int(cum[d - 1]) if d else 0))
        done = nb0
        while done < w:
            nb = min(bits, w - done); sh = w - done - nb
            nxt = {}                                         # this level's prefixes, from the previous level's only
            for p in sorted({pr for pr, _ in state.values()}):
                c = counts((p << (w - done), ((p + 1) << (w - done)) - 1), nb, sh)
                for r, (pr, rr) in state.items():
                    if pr != p: continue
                    d = int(np.searchsorted(c, rr, side="right")); nxt[r] = ((p << nb) | d, rr - (int(c[d - 1]) if d else 0))
            state = nxt; done += nb
        back = {}
        for r0 in ranks0:
            r = int(r0) + total if int(r0) < 0 else int(r0)
            if r in state: back[r0] = state[r][0]
        return back, total

    def rows(self, engine="alg"):
        """the ids of the fact rows that pass the filters and joins (a row bitmap from the engine)"""
        q = copy_query(self); q.keys = []; q.measure = None; q._ext = None; q._rows = True
        return q._run_codes("plain" if engine == "plain" else "alg")

    def _sql_aggs(self):
        """the aggregate in SQL, one expression per column of its value: SUM(e) (and COUNT(e) for a (sum, count)
        result), MIN(c) and MAX(c), COUNT(e) or COUNT(*)"""
        ext, cnt = getattr(self, "_ext", None), getattr(self, "_count", None)
        if ext is not None: c = expr_sql(ext); return [f"MIN({c})", f"MAX({c})"]
        if self.measure is not None:
            m = expr_sql(self.measure)
            return [f"SUM({m})"] + ([f"COUNT({m})"] if getattr(self, "_both", False) else [])
        return [f"COUNT({expr_sql(cnt)})" if cnt is not None else "COUNT(*)"]

    def sql(self):
        keys = [expr_sql(k) for k in self.keys]
        F = _ident(self.fact.name)
        s = f"SELECT {', '.join(keys + self._sql_aggs())} FROM {F}"
        for dim, fk in self.joins.items(): s += f" JOIN {_ident(dim.name)} ON {F}.{_ident(fk)} = {_ident(dim.name)}.{_ident(getattr(dim, 'pk', None) or 'id')}"
        if getattr(self, "rjoin", None) is not None:
            dim, x, lo, hi = self.rjoin
            if getattr(self, "rjoin_sql", None) is not None: on = f"{expr_sql(x)} {self.rjoin_sql[0]} {expr_sql(self.rjoin_sql[1])}"
            elif lo is not None and hi is not None: on = f"{expr_sql(x)} BETWEEN {expr_sql(lo)} AND {expr_sql(hi)}"
            else: on = f"{expr_sql(x)} >= {expr_sql(lo)}" if lo is not None else f"{expr_sql(x)} <= {expr_sql(hi)}"
            s += f" JOIN {_ident(dim.name)} ON {on}"
        if self.pred is not None: s += " WHERE " + " AND ".join(pred_sql(c) for c in conjuncts(self.pred))
        if keys: s += " GROUP BY " + ", ".join(keys)
        return s

    def sql_preagg(self):
        """the same range / inequality join for a SQL engine, rewritten with the aggregation pushed below the join
        (the fact side grouped by (fact keys, x) first): the algebra torbles uses, handed to the SQL engine"""
        dim, x, lo, hi = self.rjoin
        dpreds = [c for c in conjuncts(self.pred) if tables_of(c) == {dim}]
        others = [c for c in conjuncts(self.pred) if tables_of(c) != {dim}]
        fkeys = [k for k in self.keys if k.table is not dim]
        aggs = self._sql_aggs()
        sel = [f"{expr_sql(k)} AS k{i}" for i, k in enumerate(fkeys)] + [f"{expr_sql(x)} AS kx"] + [f"{a} AS v{i}" for i, a in enumerate(aggs)]
        # the partial aggregates combine: extremes by MIN / MAX, sums and counts by SUM (a count of no pair is 0)
        if getattr(self, "_ext", None) is not None: outer = ["MIN(m.v0)", "MAX(m.v1)"]
        else: outer = [f"SUM(m.v{i})" if a.startswith("SUM(") else f"COALESCE(SUM(m.v{i}), 0)" for i, a in enumerate(aggs)]
        F = _ident(self.fact.name)
        inner = f"SELECT {', '.join(sel)} FROM {F}"
        for d, fk in self.joins.items(): inner += f" JOIN {_ident(d.name)} ON {F}.{_ident(fk)} = {_ident(d.name)}.{_ident(getattr(d, 'pk', None) or 'id')}"
        if others: inner += " WHERE " + " AND ".join(pred_sql(c) for c in others)
        inner += " GROUP BY " + ", ".join([expr_sql(k) for k in fkeys] + [expr_sql(x)])
        if getattr(self, "rjoin_sql", None) is not None: on = f"m.kx {self.rjoin_sql[0]} {expr_sql(self.rjoin_sql[1])}"
        elif lo is not None and hi is not None: on = f"m.kx BETWEEN {expr_sql(lo)} AND {expr_sql(hi)}"
        else: on = f"m.kx >= {expr_sql(lo)}" if lo is not None else f"m.kx <= {expr_sql(hi)}"
        it = iter(range(len(fkeys)))
        keys = [expr_sql(k) if k.table is dim else f"m.k{next(it)}" for k in self.keys]
        s = f"WITH m AS ({inner}) SELECT {', '.join(keys + outer)} FROM m JOIN {_ident(dim.name)} ON {on}"
        if dpreds: s += " WHERE " + " AND ".join(pred_sql(c) for c in dpreds)
        if keys: s += " GROUP BY " + ", ".join(keys)
        return s

    def _linear(self, e):
        """return [(col, coef)] if e is a linear form in fact columns, else None"""
        if isinstance(e, Col) and e.table is self.fact: return [(e.name, 1)]
        if isinstance(e, Arith) and e.op in "+-":
            l, r = self._linear(e.l), self._linear(e.r)
            if l is None or r is None: return None
            return l + [(c, -k if e.op == "-" else k) for c, k in r]
        if isinstance(e, Arith) and e.op == "*":
            if isinstance(e.l, Const): r = self._linear(e.r); return None if r is None else [(c, k * e.l.v) for c, k in r]
            if isinstance(e.r, Const): l = self._linear(e.l); return None if l is None else [(c, k * e.r.v) for c, k in l]
        return None

    def _exec(self, dev, table, plan):
        """run a finished plan: on the GPU when asked (False if it cannot: the caller re-plans for the CPU), or on a
        CPU engine"""
        for d in self.joins: d._storage()     # a joined table closed since the plan was compiled: TableClosedError
        self.device, self.kernel_ms = ("plain" if dev == "plain" else "cpu"), None
        import time as _time
        catalog = getattr(table, "_relationships", None)
        self.system_plan_ms = 0.0
        self.stage_plan_ms = 0.0
        self.stage_plan = {"choice": "preserve", "reason": "disabled", "scan_devices": []}
        if catalog is None:
            self.system_plan = {"choice": "preserve", "reason": "disabled"}
        else:
            t_system = _time.perf_counter()
            source = getattr(plan, "_system_source", None)
            if source is None:
                source = plan._system_source = catalog._capture_system_source(plan)
            if catalog.stage_strategy != "off":
                t_stage = _time.perf_counter()
                owner = getattr(plan, "_stage_handle", None)
                if owner is None:
                    owner = plan._stage_handle = catalog._stage_handle()
                selected, self.stage_plan = catalog._prepare_stage(plan, dev, source, owner)
                self.stage_plan_ms = (_time.perf_counter() - t_stage) * 1e3
                if selected:
                    self.system_plan = {"choice": "stages", "reason": "native stage composition"}
                    t0 = _time.perf_counter()
                    try:
                        return self._exec_stages(dev, table, plan, owner)
                    finally:
                        self.exec_ms = (_time.perf_counter() - t0) * 1e3
            t_system = _time.perf_counter()
            self.system_plan = catalog._plan_system(plan, dev, source=source)
            self.system_plan_ms = (_time.perf_counter() - t_system) * 1e3
        t0 = _time.perf_counter()
        try: return self._exec_loop(dev, table, plan)
        finally: self.exec_ms = (_time.perf_counter() - t0) * 1e3     # the engine call alone (no planning)

    def _exec_stages(self, dev, table, plan, owner):
        """Dispatch legacy scans; all decomposition and final arithmetic stay native."""
        import time as _time
        owner.reset()
        devices, scan_ms, kernel_ms = [], [], []
        for scan in owner.scans():
            start = _time.perf_counter()
            if not self._exec_loop(dev, table, scan):
                self.stage_plan.update(scan_devices=devices, row_fallback=(dev == "gpu"),
                                       failure=getattr(self, "_gpu_fail", "unsupported scan"))
                return False
            scan_ms.append((_time.perf_counter() - start) * 1e3)
            devices.append(self.device)
            if self.kernel_ms is not None:
                kernel_ms.append(self.kernel_ms)
        start = _time.perf_counter()
        owner.finish(plan)
        self.stage_plan.update(scan_devices=devices, scan_exec_ms=scan_ms,
                               finalizer_device="cpu(native)",
                               finalizer_ms=(_time.perf_counter() - start) * 1e3,
                               row_fallback=(dev == "gpu" and any(d != "gpu" for d in devices)))
        self.device = "gpu+cpu-finalize" if devices and all(d == "gpu" for d in devices) else "cpu(stages)"
        self.kernel_ms = sum(kernel_ms) if kernel_ms else None
        return True

    def _gpu_tags(self, plan):
        """the tags of the plan's host arrays in the GPU caches (see _gpu_tag): an attribute's value array is one of the
        decoded or ranked columns of an open joined table, which owns it; the delta plane is this query's. Any other
        array has no tag (0)."""
        tags = (ctypes.c_uint64 * 9)()
        want = {}
        for i in range(plan.nattr): want.setdefault((plan.attr_vals[i], plan.attr_n[i] * 4), []).append(i)
        if want:
            for t in self.joins:
                if t.closed: continue
                d = t.__dict__
                for a in [*d.get("_decoded", {}).values(), *(r[1] for r in d.get("_ranked", {}).values())]:
                    for i in want.get((a.ctypes.data, a.nbytes), ()): tags[i] = _gpu_tag(a, t._native.s)
        xr = getattr(self, "_xrows", None)
        if plan.xmode and xr is not None and plan.xrows == xr[0].ctypes.data: tags[8] = _gpu_tag(xr[0])
        return tags

    def _exec_loop(self, dev, table, plan):
        tags = self._gpu_tags(plan) if dev == "gpu" else None
        while True:
            if dev == "gpu":
                ms = ctypes.c_double(0); h = table.gpu_handle()
                with _GPU_LOCK: rc = gpu_lib().gpu_query_tagged(h, ctypes.byref(plan), tags, ctypes.byref(ms))
                if rc != 0:
                    self._gpu_fail = "unsupported plan" if rc == -1 else "CUDA error"
                    return False
                self.device, self.kernel_ms = "gpu", ms.value
            else:
                rc = (lib.pt_query if dev == "plain" else lib.at_query)(table.plain_handle() if dev == "plain" else table.at, ctypes.byref(plan))
                if rc == -2: raise OverflowError("SUM needs 128 bits, which this engine evaluates for polynomials of degree <= 2 in fact columns")
                if rc != 0: raise RuntimeError(f"plan not supported by the engine (agg {plan.agg}, keys {plan.nkey}, rc {rc}, dev {dev})")
            if not plan.out_key or plan.out_n <= plan.out_cap: return True
            self._mm_alloc(plan, int(plan.out_n) * 2)     # more groups than the output arrays hold: grow, rerun

    def _mm_alloc(self, plan, cap):
        self._mm_out = (np.zeros(cap, np.uint64), np.zeros(cap, np.int64), np.zeros(cap, np.uint64), np.zeros(cap, np.int64), np.zeros(cap, np.uint64))
        plan.out_cap, plan.out_n = cap, 0
        self._mm_bind(plan)

    def _mm_bind(self, plan):
        plan.out_key, plan.out_sum, plan.out_cnt, plan.out_sum_hi, plan.out_key_hi = (a.ctypes.data for a in self._mm_out)

    def _moved(self):
        surr = getattr(self.fact, "surr", {})
        return {dim: fk for dim, fk in self.joins.items() if surr.get(fk) is not None and surr[fk].stale}

    def _run_delta(self, engine, moved):
        """Dimension rows changed since some facts were written (derived surrogates with stale versions). Additive:
        pass A = every row except the delta plane, evaluated on the stored surrogates (cubes, comparators, plane keys);
        pass B = only the delta rows (their words alone are scanned), the moved dimensions read by lookup of current
        values. The fact itself is never rewritten. Result: A + B."""
        import copy
        fact = self.fact; D = None
        for dim, fk in moved.items():
            d, _ = fact.delta(fk, dim); D = d.copy() if D is None else (D | d)
        A = copy.copy(self); A._inpass = True; A._xrows = (D, 2)
        rA = A._run_codes(engine)
        B = copy.copy(self); B._inpass = True; B._xrows = (D, 1)
        B._lookup_dims = set(moved); B._force_lookup = set(moved)
        words = np.nonzero(D)[0].astype(np.uint64)
        B._word_ranges = np.stack([words, words + np.uint64(1)], 1).ravel() if len(words) else np.zeros(2, np.uint64)
        rB = B._run_codes(engine)
        self.device, self.kernel_ms = A.device, getattr(A, "kernel_ms", None)
        self.last_plan, self.range_used, self.delta_rows = A.last_plan, A.range_used, int(sum(bin(int(x)).count("1") for x in D[words.astype(np.int64)])) if len(words) else 0
        self.system_plan = {"choice": "partitioned", "passes": [getattr(A, "system_plan", {}), getattr(B, "system_plan", {})]}
        self.stage_plan = {"choice": "partitioned", "passes": [getattr(A, "stage_plan", {}), getattr(B, "stage_plan", {})]}
        self.stage_plan_ms = getattr(A, "stage_plan_ms", 0) + getattr(B, "stage_plan_ms", 0)
        if getattr(self, "_rows", False): return np.union1d(rA, rB)
        if self.keys:
            out = dict(rA)
            for key, v in rB.items(): out[key] = v if key not in out else _merge_vals(out[key], v)
            return out
        if getattr(self, "_ext", None) is not None:
            if rA is None or rB is None: return rA if rB is None else rB
            return MinMax((min(rA[0], rB[0]), max(rA[1], rB[1])))
        if getattr(self, "_both", False): return _merge_vals(rA, rB)
        if self.measure is None: return rA + rB
        return None if rA is None and rB is None else (rA or 0) + (rB or 0)

    def _run_wide(self, engine):
        """queries over self-extending columns, lowered to their physical parts: a value is base + 2^B hi, so measures
        become polynomials in base and hi, comparisons with constants become lexicographic comparisons on (hi, base),
        and a GROUP BY key becomes two keys, recombined in the result"""
        import copy
        fact, W = self.fact, self.fact.wide
        def lc(e):
            if isinstance(e, Col) and e.table is fact and e.name in W:
                B, hi, _ = W[e.name]; return Arith("+", Col(fact, e.name), Arith("*", Const(1 << B), Col(fact, hi)))
            if isinstance(e, Arith): return Arith(e.op, lc(e.l), lc(e.r))
            return e
        def lp(p):
            if isinstance(p, BoolOp): return BoolOp(p.op, lp(p.l), lp(p.r))
            if isinstance(p, Not): return Not(lp(p.x))
            if isinstance(p, (Cmp, Between)) and p.col.table is fact and p.col.name in W:
                B, hi, _ = W[p.col.name]; base, H = Col(fact, p.col.name), Col(fact, hi); m = (1 << B) - 1
                FALSE = Cmp(base, "LT", 0)                        # code-level predicates (no typed operators here)
                def lt(K):                                        # value < K
                    if K <= 0: return FALSE
                    kh, kl = K >> B, K & m
                    if kh > 0xFFFFFFFF: return Not(FALSE)
                    return Cmp(H, "LT", kh) if kl == 0 else BoolOp("OR", Cmp(H, "LT", kh), BoolOp("AND", Cmp(H, "EQ", kh), Cmp(base, "LT", kl)))
                def eq(K):
                    if K < 0 or (K >> B) > 0xFFFFFFFF: return FALSE
                    return BoolOp("AND", Cmp(H, "EQ", K >> B), Cmp(base, "EQ", K & m))
                if isinstance(p, Between): return BoolOp("AND", Not(lt(p.a)), lt(p.b + 1))
                v = p.v
                return {"LT": lambda: lt(v), "LE": lambda: lt(v + 1), "GT": lambda: Not(lt(v + 1)), "GE": lambda: Not(lt(v)),
                        "EQ": lambda: eq(v), "NE": lambda: Not(eq(v))}[p.op]()
            return p
        if getattr(self, "_distinct", False) and any(isinstance(k, Col) and k.table is fact and k.name in W for k in self.keys):
            raise ValueError("COUNT(DISTINCT) cells: not on self-extending GROUP BY columns")
        q = copy.copy(self); q._lowered = True; q._codes = True
        q.pred = lp(self.pred) if self.pred is not None else None
        q.measure = lc(self.measure) if self.measure is not None else None
        spans, keys = [], []
        for k in self.keys:
            if isinstance(k, Col) and k.table is fact and k.name in W:
                B, hi, _ = W[k.name]; keys += [Col(fact, k.name), Col(fact, hi)]; spans.append(B)
            else: keys.append(k); spans.append(None)
        q.keys = keys
        cols_wanted = self is getattr(self, "_cols_owner", None)
        if cols_wanted: q._cols_owner = q
        try: r = q._run_codes(engine)
        finally:
            if cols_wanted: q._cols_owner = None
        self.device, self.kernel_ms = getattr(q, "device", None), getattr(q, "kernel_ms", None)
        self.last_plan, self.range_used = q.last_plan, q.range_used
        for a in ("delta_rows", "_lookup_keys", "relationship_plan", "system_plan", "relationship_ms", "system_plan_ms", "stage_plan", "stage_plan_ms"):
            if hasattr(q, a): setattr(self, a, getattr(q, a))
        if not self.keys or (getattr(self, "_distinct", False) and not any(B is not None for B in spans)): return r
        if getattr(self, "_raw", False):             # (key matrix, sums, counts): each value = base + 2^B hi again
            km, vs, cn = r; cols, j = [], 0
            for B in spans:
                if B is None: cols.append(np.asarray(km[:, j], np.int64)); j += 1; continue
                lo_, hi_ = np.asarray(km[:, j], np.int64), np.asarray(km[:, j + 1], np.int64); j += 2
                if len(hi_) and B + int(hi_.max()).bit_length() >= 63: raise ValueError("64-bit GROUP BY codes in a raw result")
                cols.append(lo_ + (hi_ << B))
            return (np.stack(cols, 1) if cols else np.zeros((len(cn), 0), np.int64)), vs, cn
        if isinstance(r, CodeGroups):                # columns: value = base + 2^B hi, by array arithmetic
            it, out_cols, ok = iter(r.cols), [], True
            for B in spans:
                c = np.asarray(next(it))
                if B is None: out_cols.append(c); continue
                h = np.asarray(next(it))
                lo_, hi_ = c.astype(np.int64), h.astype(np.int64)
                if len(lo_) and (int(lo_.min()) < 0 or int(lo_.max()) >> B): ok = False; break   # not one value per (base, hi)
                if B + (int(hi_.max()).bit_length() if len(hi_) else 0) < 63: out_cols.append(lo_ + (hi_ << B))
                else: out_cols.append(np.array([a + (b << B) for a, b in zip(lo_.tolist(), hi_.tolist())], object))
            if ok:
                out_cols += list(it)                 # the validity keys of nullable GROUP BY columns
                return CodeGroups(out_cols, r.vals, r.cnt)
            r = _code_groups_dict(r, q)
        out = {}
        for key, v in r.items():
            it, nk = iter(key), []
            for B in spans: nk.append(next(it) + (next(it) << B) if B is not None else next(it))
            k = tuple(nk)
            out[k] = v if k not in out else _merge_vals(out[k], v)
        return out

    def run(self, engine="alg"):
        """execute: "alg" (the algebraic CPU engine), "gpu" (the CUDA executor, or its emulation), or "plain" (the
        columnar baseline). GROUP BY returns {key tuple: value}, else a number (COUNT) or the SUM (None if no row).
        Typed columns: keys come back as their values (str, Decimal, date, ...), decimal sums as Decimal."""
        ids = [(id(k.table), k.name) if isinstance(k, Col) else id(k) for k in self.keys]
        if len(set(ids)) < len(ids) and not getattr(self, "_codes", False):
            # the same GROUP BY column twice: group by each once, repeat it in the result
            import copy
            first = {}                                     # column -> its position among the distinct keys
            for x in ids: first.setdefault(x, len(first))
            q = copy.copy(self); q._plan_cache = {}
            for a in ("_codeq", "_lowered_measure", "_fq"):
                if a in q.__dict__: del q.__dict__[a]
            q.keys = [k for i, k in enumerate(self.keys) if ids.index(ids[i]) == i]
            r = q.run(engine)
            for a in ("device", "kernel_ms", "last_plan", "range_used", "system_plan", "stage_plan", "stage_plan_ms"):
                if hasattr(q, a): setattr(self, a, getattr(q, a))
            pos = [first[x] for x in ids]
            return {tuple(k[p] for p in pos): v for k, v in r.items()}
        need = self._null_terms()
        if need: return self._run_observed(engine, need)
        try:
            if not getattr(self, "_codes", False) and self._typed():
                return self._run_typed(engine)
            return self._run_codes(engine)
        except (TypeError, OverflowError) as e:
            why = _bridge_reason(self, e)
            if why is None: raise
            return self._run_bridge(engine, why)

    def _null_terms(self):
        """the validity planes (table, column) of the nullable columns the aggregate reads (SUM's measure, MIN / MAX's
        column, COUNT(e)'s argument; an expression is NULL where any of its columns is) that the filter does not already
        require: the rows where one is 0 are not observations of the aggregate"""
        e = self.measure if self.measure is not None else getattr(self, "_ext", None)
        if e is None:
            e = getattr(self, "_count", None)
            rdim = self.rjoin[0] if getattr(self, "rjoin", None) is not None else None
            for c in _expr_cols(e):                  # (COUNT(e) reads nothing else of these tables: SUM would fail here too)
                if c.table is not self.fact and c.table not in self.joins and c.table is not rdim:
                    raise ValueError(f"{c.table.name} is not joined")
        out = []
        for c in _expr_cols(e):
            vc = getattr(c.table, "nullable", {}).get(c.name)
            if vc is None or any(t is c.table and v == vc for t, v in out): continue
            if self.pred is not None and _requires(self.pred, c.table, vc): continue
            out.append((c.table, vc))
        return out

    def _run_observed(self, engine, need):
        """An aggregate over nullable columns reads only its observations, the rows where the validity planes `need` are
        1, while the groups stay those of the rows that pass the filter: a WHERE on the validity alone would drop a group
        whose argument is NULL on every row, which SQL keeps with a NULL aggregate (COUNT(e): 0). So two engine queries,
        each an ordinary one: the aggregate under the filter AND the validity (no NULL placeholder is summed, compared or
        counted), and with GROUP BY the groups, COUNT(*) under the filter alone; every group of the first is a group of
        the second (its rows are among them), the others get the aggregate of no observation."""
        objs = (self.pred, self.measure, getattr(self, "_ext", None), getattr(self, "_count", None), getattr(self, "rjoin", None)) \
            + tuple(self.keys) + tuple(self.joins)
        key = (tuple(self.joins.values()), self.join_mode, getattr(self, "_both", False), getattr(self, "range_mode", "auto"),
               tuple(vc for _, vc in need))
        c = self.__dict__.get("_observed")
        if c is None or c["key"] != key or len(c["objs"]) != len(objs) or any(a is not b for a, b in zip(c["objs"], objs)):
            vp = self.pred                                   # (kept, so that the queries reuse their compiled plans)
            for t, vc in need:
                v = Cmp(Col(t, vc), "EQ", 1)
                vp = v if vp is None else BoolOp("AND", vp, v)
            obs = copy_query(self); obs.pred = vp
            pres = None
            if self.keys:
                pres = copy_query(self); pres.measure = None; pres._ext = None; pres._count = None; pres._both = False
            c = self._observed = dict(key=key, objs=objs, obs=obs, pres=pres)
        obs, pres = c["obs"], c["pres"]
        r = obs.run(engine)
        for a in ("device", "kernel_ms", "exec_ms", "plan_ms", "plan_cached", "last_plan", "range_used", "range_path", "scanned", "delta_rows",
                  "relationship_plan", "system_plan", "relationship_ms", "system_plan_ms", "stage_plan", "stage_plan_ms"):
            if hasattr(obs, a): setattr(self, a, getattr(obs, a))
        if pres is None: return r
        groups = pres.run(engine)
        if getattr(self, "_ext", None) is not None or self.measure is not None:
            none = (None, 0) if getattr(self, "_both", False) and self.measure is not None else None
        else: none = 0                                       # COUNT(e)
        out = {g: (r[g] if g in r else none) for g in groups}
        for g, v in r.items(): out.setdefault(g, v)
        return out

    def _run_bridge(self, engine, why):
        """The histogram bridge: SUM(m) = sum over the values v of some columns H of m(v) restricted to the rows with
        H = v. The measure is expanded into monomials; the columns of H leave each monomial as exact values: H joins
        the GROUP BY (hidden keys), the engine sums each remaining monomial (the residual: fact columns, degree <= 2)
        per group, and the result is put together in exact arithmetic,
            SUM[k] = sum over the groups (k, v) of  sum_m  c_m * prod_(h in m) v_h * SUM_r(m)[k, v]   (r() = COUNT).
        H holds the columns the engine cannot sum (no affine code: a referenced table's column wider than 32 bits),
        the referenced tables' columns, and the fact columns past degree 2 (past degree 1 when even the degree-2
        residuals could pass 128 bits). The products and sums in Python are exact: no 128-bit limit on the result."""
        from fractions import Fraction
        from coltypes import _frac
        mono = _expand(self.measure)                          # {(column ids...): Fraction}, columns by (table, name)
        cols = {}
        def walk(e):
            if isinstance(e, Col): cols[(id(e.table), e.name)] = e
            elif isinstance(e, Arith): walk(e.l); walk(e.r)
        walk(self.measure)
        fact = self.fact
        # always summed per value: columns without an affine code, and the referenced tables' columns (the engine's
        # 128-bit sums are polynomials in fact columns)
        hidden0 = {k for k, c in cols.items() if _bridge_col(c) == "hide" or c.table is not fact}
        last = None
        for cap in ((2, 1) if why != "bound" else (1,)):        # fact columns kept per monomial
            hidden = set(hidden0)
            while True:
                over = [m for m in mono if len([x for x in m if x not in hidden]) > cap]
                if not over: break
                cnt = {}
                for m in over:
                    for x in m:
                        if x not in hidden: cnt[x] = cnt.get(x, 0) + 1
                def span(x):                                    # prefer the column with the fewest values
                    t = cols[x].table
                    return t.maxcode(cols[x].name) if hasattr(t, "maxcode") else t.colmax(cols[x].name)
                hidden.add(max(cnt, key=lambda x: (cnt[x], -span(x), x[1])))
            try: return self._bridge_eval(engine, mono, cols, hidden)
            except OverflowError as e:                          # a residual still past 128 bits: one fact column less
                if "SUM may exceed 128 bits" not in str(e): raise
                last = e
        raise last

    def _bridge_eval(self, engine, mono, cols, hidden):
        from fractions import Fraction
        from coltypes import _frac
        H = sorted(hidden, key=lambda x: (x[1], x[0]))
        if len(self.keys) + len(H) > MAXKEY: raise OverflowError("the measure needs more hidden GROUP BY columns than the engine has keys")
        resid = {}
        for m, c in mono.items():
            r = tuple(sorted((x for x in m if x not in hidden), key=lambda x: (x[1], x[0])))
            hv = tuple(x for x in m if x in hidden)
            resid.setdefault(r, []).append((hv, c))
        import copy
        nk = len(self.keys)
        def sub(meas):
            q = copy.copy(self); q._nobridge = True; q._plan_cache = {}; q._both = False
            for a in ("_codeq", "_lowered_measure", "_fq"):
                if a in q.__dict__: del q.__dict__[a]
            q.keys = list(self.keys) + [cols[h] for h in H]; q.measure = meas
            out = q.run(engine)
            for a in ("device", "kernel_ms", "last_plan", "range_used", "system_plan", "stage_plan", "stage_plan_ms"):
                if hasattr(q, a): setattr(self, a, getattr(q, a))
            if not q.keys: return {} if out is None or out == 0 and meas is None else {(): out}   # one group: ()
            return out
        sums = {}
        for r in resid:
            if not r: continue
            m = None
            for x in r: m = cols[x] if m is None else Arith("*", m, cols[x])
            sums[r] = sub(m)
        if () in resid or not sums: counts = sub(None)     # the groups (k, v) and their rows (COUNT: the constant term)
        else:                                               # a group with rows has every residual's sum
            counts = {}
            for sr in sums.values():
                for g in sr: counts[g] = 1
        hpos = {h: nk + i for i, h in enumerate(H)}
        out, floats = {}, False
        for g, n in counts.items():
            if not n: continue
            acc = Fraction(0)
            for r, terms in resid.items():
                sr = n if not r else sums[r].get(g)
                if sr is None: continue
                if isinstance(sr, float): floats = True
                f = _frac(sr)
                for hv, c in terms:
                    p = c
                    for h in hv:
                        v = g[hpos[h]]
                        if isinstance(v, float): floats = True
                        p *= _frac(v)
                    acc += p * f
            k = g[:nk]
            out[k] = out.get(k, Fraction(0)) + acc
        scale = _dec_scale_of(self.measure)
        both = getattr(self, "_both", False)
        if both:                                            # the row counts per group, for (sum, count) results
            cnt = {}
            for g, n in (counts.items() if (() in resid or not sums) else sub(None).items()):
                cnt[g[:nk]] = cnt.get(g[:nk], 0) + n
        def fin(v):
            if floats: return float(v)
            if not scale: return int(v) if v.denominator == 1 else float(v)
            x = v * 10 ** scale
            return _exact_dec(int(x), scale) if x.denominator == 1 else Decimal(v.numerator) / Decimal(v.denominator)
        if not self.keys:
            if both: return (fin(out[()]) if () in out else None, cnt.get((), 0))
            return fin(out[()]) if () in out else None
        if both: return {k: (fin(v), cnt.get(k, 0)) for k, v in out.items()}
        return {k: fin(v) for k, v in out.items()}

    def _typed(self):
        tabs = {self.fact} | set(self.joins) | {k.table for k in self.keys if isinstance(k, Col)}
        if self.measure is not None: tabs |= tables_of(self.measure)
        if any(getattr(t, "codecs", None) for t in tabs): return True
        return self.measure is not None and _has_fraction(self.measure)

    def run_columns(self, engine="alg"):
        """run() with GROUP BY keys, the groups as columns: a Groups of the keys' codes, validity and codecs (decode them
        only for the rows needed) and one value per group, as run() gives it"""
        ids = [(id(k.table), k.name) if isinstance(k, Col) else id(k) for k in self.keys]
        if self.keys and len(set(ids)) == len(ids) and not getattr(self, "_codes", False) and self._typed() and not self._null_terms():
            try: return self._run_typed(engine, columns=True)
            except (TypeError, OverflowError): pass          # the bridges and NULL-skipping aggregates: through run()
        r = self.run(engine)
        keys = list(r.keys())
        cols = list(zip(*keys)) if keys else [[] for _ in self.keys]
        return Groups([(None, None, list(c)) for c in cols], [r[k] for k in keys])

    def _run_typed(self, engine, columns=False):
        import copy
        lm = getattr(self, "_lowered_measure", None)
        if lm is None or lm[0] is not self.measure:
            low, scale = (None, 0) if self.measure is None else lower_measure(self.measure, self.fact)
            lm = self._lowered_measure = (self.measure, low, scale)
        q = getattr(self, "_codeq", None)
        if q is None or q._src_measure is not self.measure:
            q = self._codeq = copy.copy(self); q._codes = True; q._src_measure = self.measure
            q._plan_cache = {}
        # a GROUP BY column with NULLs: its validity is grouped too, and its NULL rows get the key None
        nkeys = [Col(k.table, k.table.nullable[k.name]) for k in self.keys
                 if isinstance(k, Col) and k.name in getattr(k.table, "nullable", {})]
        vpos = [i for i, k in enumerate(self.keys) if isinstance(k, Col) and k.name in getattr(k.table, "nullable", {})]
        q.measure, q.pred, q.keys, q.joins, q.join_mode = lm[1], self.pred, list(self.keys) + nkeys, self.joins, self.join_mode
        q._ext = getattr(self, "_ext", None); q._both = getattr(self, "_both", False)
        q.range_mode = getattr(self, "range_mode", "auto")
        q._cols_owner = q if columns else None
        try: r = q._run_codes(engine)
        finally: q._cols_owner = None
        for a in ("device", "kernel_ms", "exec_ms", "plan_ms", "plan_cached", "last_plan", "range_used", "scanned", "delta_rows",
                  "relationship_plan", "system_plan", "relationship_ms", "system_plan_ms", "stage_plan", "stage_plan_ms"):
            if hasattr(q, a): setattr(self, a, getattr(q, a))
        scale = lm[2]
        val = (lambda v: v if v is None or not scale else _exact_dec(v, scale)) if self.measure is not None else (lambda v: v)
        ext = getattr(self, "_ext", None)
        if ext is not None:                          # MIN / MAX codes -> values of the column's type
            ecd = ext.codec()
            val = (lambda v: v) if ecd is None else (lambda v: v if v is None else MinMax(ecd.decode(_codes_array(list(v)))))
        if getattr(self, "_both", False) and self.measure is not None:
            v0 = val; val = lambda v: v if v is None else (v0(v[0]), v[1])
        if not self.keys: return val(r)
        decs = []
        for k in self.keys:
            cd = k.codec() if isinstance(k, Col) else None
            decs.append(cd)
        if columns: return self._groups(r, decs, vpos, val, scale, ext)
        if not r: return {}
        keys = list(r.keys()); cols = list(zip(*keys))
        valid = cols[len(self.keys):]; cols = cols[:len(self.keys)]
        cols = [list(c) if cd is None else cd.decode(_codes_array(c)) for c, cd in zip(cols, decs)]
        for i, vcol in zip(vpos, valid): cols[i] = [x if ok else None for x, ok in zip(cols[i], vcol)]
        return {tuple(kk): val(r[k0]) for k0, kk in zip(keys, zip(*cols))}

    def _groups(self, r, decs, vpos, val, scale, ext):
        """the typed result as columns (Groups): codes kept, values as run() gives them"""
        nk = len(self.keys)
        if isinstance(r, CodeGroups):
            cols, vals = r.cols, r.vals
            both = r.cnt is not None
        else:                                          # a dict from another path (wide, delta, factored ...)
            keys = list(r.keys())
            cols = [_codes_array(c) for c in zip(*keys)] if keys else [np.zeros(0, np.int64) for _ in range(nk + len(vpos))]
            vals = [r[k] for k in keys]
            both = False
        if both and not scale and ext is None: vl = list(zip(vals.tolist(), r.cnt.tolist()))
        elif both and ext is None: vl = [(_exact_dec(v, scale), c) for v, c in zip(vals.tolist(), r.cnt.tolist())]
        elif both: vl = [val((v, c)) for v, c in zip(vals.tolist(), r.cnt.tolist())]
        elif isinstance(vals, np.ndarray) and not scale and not ext: vl = vals.tolist()
        elif isinstance(vals, np.ndarray) and not ext: vl = [_exact_dec(v, scale) for v in vals.tolist()]
        elif ext is not None and not both and isinstance(vals, list) and all(v is not None for v in vals):
            ecd = ext.codec()                        # MIN / MAX codes: every minimum, then every maximum, decoded at once
            if ecd is None: vl = vals
            else:
                lo_ = ecd.decode(_codes_array([v[0] for v in vals])); hi_ = ecd.decode(_codes_array([v[1] for v in vals]))
                vl = [MinMax((a, b)) for a, b in zip(lo_, hi_)]
        else: vl = [val(v) for v in (vals.tolist() if isinstance(vals, np.ndarray) else vals)]
        valid = {i: np.asarray(cols[nk + j]) != 0 for j, i in enumerate(vpos)}
        keys = [(np.asarray(cols[i]), valid.get(i), decs[i]) for i in range(nk)]
        return Groups(keys, vl)

    def _run_codes(self, engine="alg"):
        if not getattr(self, "_relationship_done", False):
            self.relationship_plan = {"status": "disabled"}
            self.relationship_ms = 0.0
        if getattr(self.fact, "wide", None) and not getattr(self, "_lowered", False): return self._run_wide(engine)
        if getattr(self, "rjoin", None) is not None:
            if getattr(self, "_distinct", False): raise ValueError("COUNT(DISTINCT) cells: not over a range join")
            return self._run_range(engine)
        # Experimental opt-in: native modules verify equations; native search composes
        # them over this lowered code polynomial before the ordinary planner runs.
        catalog = getattr(self.fact, "_relationships", None)
        if catalog is not None and self.measure is not None and not getattr(self, "_relationship_done", False) \
                and getattr(self, "_ext", None) is None:
            import time as _time
            t_relationship = _time.perf_counter()
            replacement, info = catalog._rewrite_codes(self.measure)
            self.relationship_ms = (_time.perf_counter() - t_relationship) * 1e3
            self.relationship_plan = info
            if replacement is not None:
                cached = getattr(self, "_relationship_query", None)
                if cached is not None and cached[0] is catalog and cached[1] is replacement:
                    q = cached[2]
                else:
                    q = copy_query(self)
                    q.__dict__.pop("_relationship_query", None)
                    self._relationship_query = (catalog, replacement, q)
                # Retain 0.2.0's compiled-plan/factor cache, while refreshing the
                # query scope. Its normal identity signature detects changes.
                q.pred, q.keys, q.joins, q.join_mode = self.pred, self.keys, self.joins, self.join_mode
                for name in ("_xrows", "_word_ranges", "_lookup_dims", "_force_lookup", "_inpass", "_both", "_raw", "_dom", "_ext", "range_mode"):
                    if hasattr(self, name): setattr(q, name, getattr(self, name))
                    else: q.__dict__.pop(name, None)
                q.measure, q._relationship_done = replacement, True
                cols_wanted = self is getattr(self, "_cols_owner", None)
                q._cols_owner = q if cols_wanted else None         # columns (CodeGroups) when the original was asked for them
                try:
                    result = q._run_codes(engine)
                except (ValueError, OverflowError, TypeError) as error:
                    self.relationship_plan = dict(info, status="fallback", reason=str(error))
                else:
                    for name in ("device", "kernel_ms", "exec_ms", "plan_ms", "plan_cached", "last_plan", "range_used", "scanned", "delta_rows", "system_plan", "system_plan_ms", "stage_plan", "stage_plan_ms"):
                        if hasattr(q, name): setattr(self, name, getattr(q, name))
                    return result
        self._lookup_keys = set(getattr(self, "_force_lookup", ()))
        if not getattr(self, "_inpass", False):
            moved = self._moved()
            if moved:                                # only dimensions this query filters or groups on need the split
                used = {t for c in conjuncts(self.pred) for t in tables_of(c)} | {k.table for k in self.keys if not isinstance(k, PlaneKey)}
                moved = {d: fk for d, fk in moved.items() if d in used}
                if moved:
                    if getattr(self, "_distinct", False): raise ValueError("COUNT(DISTINCT) cells: not over moved dimension rows")
                    return self._run_delta(engine, moved)
        if FACTOR and self.keys and self.measure is not None and not getattr(self, "_factored", False) \
                and not getattr(self, "_raw", False) and getattr(self, "_dom", None) is None:
            fq = getattr(self, "_fq", None)
            if fq is None or fq[0] is not self.measure or fq[1] != tuple(map(id, self.keys)):
                f = factor_through_keys(self.measure, self.keys, self.fact); q2 = None
                if f is not None:
                    import copy
                    q2 = copy.copy(self); q2._factored = True; q2._both = True; q2._plan_cache = {}
                    R = f[0]; m = None
                    for c in (R or ()): m = Col(self.fact, c) if m is None else Arith("*", m, Col(self.fact, c))
                    q2.measure = m
                fq = self._fq = (self.measure, tuple(map(id, self.keys)), q2, f)
            if fq[2] is not None:
                q2, (R, PR, P1, keycols) = fq[2], fq[3]
                q2.pred, q2.joins, q2.join_mode = self.pred, self.joins, self.join_mode
                for a_ in ("_xrows", "_word_ranges", "_lookup_dims", "_force_lookup", "_inpass"):
                    if hasattr(self, a_): setattr(q2, a_, getattr(self, a_))
                cols_wanted = self is getattr(self, "_cols_owner", None)
                if cols_wanted: q2._cols_owner = q2
                try: r = q2._run_codes(engine)
                finally:
                    if cols_wanted: q2._cols_owner = None
                for a_ in ("device", "kernel_ms", "exec_ms", "plan_ms", "plan_cached", "last_plan", "range_used", "scanned", "system_plan", "system_plan_ms", "stage_plan", "stage_plan_ms"):
                    if hasattr(q2, a_): setattr(self, a_, getattr(q2, a_))
                if isinstance(r, CodeGroups):        # columns: the key polynomials on the key code arrays (exact integers)
                    n = np.asarray(r.cnt if r.cnt is not None else r.vals)
                    def pol(P):
                        acc = np.zeros(len(n), object)
                        for K, c in P.items():
                            t = np.full(len(n), c, object)
                            for x in K: t = t * np.asarray(r.cols[keycols[x]]).astype(object)
                            acc = acc + t
                        return acc
                    v = pol(P1) * n.astype(object)
                    if R: v = v + pol(PR) * np.asarray(r.vals).astype(object)
                    return CodeGroups(r.cols, v, n if getattr(self, "_both", False) else None)
                out = {}
                for key, (sm, n) in r.items():
                    v = (_poly_at(PR, keycols, key) * sm if R else 0) + _poly_at(P1, keycols, key) * n
                    out[key] = (v, n) if getattr(self, "_both", False) else v
                return out
        self._dev = engine; engine = "alg" if engine == "gpu" else engine     # the GPU runs the algebraic plan
        import time as _time
        t_plan = _time.perf_counter()
        self._wide = sum_width(self.measure, self.fact)     # 64- or 128-bit sums (refuses past 128 bits)
        sig = self._plan_sig(engine)
        if sig is not None:                          # same query, same data version: reuse the compiled plan
            c = getattr(self, "_plan_cache", {}).get(sig[0])
            if c is not None and len(c["objs"]) == len(sig[1]) and all(a is b for a, b in zip(c["objs"], sig[1])):
                self.plan_ms, self.plan_cached = (_time.perf_counter() - t_plan) * 1e3, True
                return self._finish(c)
        fact, plan = self.fact, QPlan()
        keep = []                                           # keep ctypes buffers alive
        groups = {}
        for c in conjuncts(self.pred):
            ts = tables_of(c)
            if len(ts) != 1: raise ValueError("each AND-term must refer to one table")
            groups.setdefault(next(iter(ts)), []).append(c)
        # fact filter: one Boolean polynomial over the fact's bit planes (code sets: semi-joins on the column)
        prog = []; codesets = [c for c in groups.get(fact, []) if isinstance(c, CodeSet)]
        for i, c in enumerate([c for c in groups.get(fact, []) if not isinstance(c, CodeSet)]):
            compile_pred(c, fact, prog)
            if i: prog.append(PInstr(P["AND"], 0, 0, 0))
        # inner-join semantics: every joined foreign key must point at an existing row (one plane: its _refok bit)
        for dim, fkname in self.joins.items():
            prog.append(PInstr(P["BIT"], fact.idx("_refok"), fact.refbit[fkname], 0))
            if len(prog) > 1: prog.append(PInstr(P["AND"], 0, 0, 0))
        had_fact, nrange, plan_false = bool(prog), 0, False
        row_ranges = None
        self.range_used = []
        # dimension filters -> bitmaps -> (range predicates on the fk planes | semi-joins, most selective first)
        semis = []
        for cs in codesets:
            keep.append(cs.bm)
            semis.append((cs.frac, fact.idx(cs.col.name), cs.bm, cs.n, 0))
        for dim, preds in groups.items():
            if dim is fact: continue
            if dim not in self.joins: raise ValueError(f"{dim.name} is not joined")
            dp = []
            for i, c in enumerate([c for c in preds if not isinstance(c, RowMask)]):
                compile_pred(c, dim, dp)
                if i: dp.append(PInstr(P["AND"], 0, 0, 0))
            arr = (PInstr * len(dp))(*dp)
            bm = np.zeros((dim.n + 63) // 64 + 1, dtype=np.uint64); keep.append(bm)
            if engine == "alg": cnt = lib.at_filter_bitmap_masked(dim.at, arr, len(dp), bm.ctypes.data, dim.used_bm.ctypes.data)
            else: cnt = lib.pt_filter_bitmap(dim.plain_handle(), arr, len(dp), bm.ctypes.data); bm[:len(dim.used_bm)] &= dim.used_bm[:len(bm)]
            rms = [c for c in preds if isinstance(c, RowMask)]
            if rms:                                    # rows given by mask: AND them in
                for rm in rms: bm &= rm.bm[:len(bm)]
                cnt = int(np.unpackbits(bm.view(np.uint8)).sum())
            fkn = self.joins[dim]; fkc = fact.idx(fkn); b = fact.width(fkn); space = fact.space(fkn, dim)
            on_cluster = fact.cluster == fkn and getattr(fact, "cluster_sorted", False) and space.sg is None
            if on_cluster:
                # the fact is sorted by this foreign key: the selected ids are row ranges, found by binary search
                sel = np.empty(max(1, int(cnt)), np.uint32); ns_ = lib.bm_ids(bm.ctypes.data, len(bm), sel.ctypes.data); sel = sel[:ns_]
                rr = []                                          # no dimension row selected -> nothing to scan
                if len(sel):
                    pos = np.searchsorted(dim.used_ids, sel); brk = np.nonzero(np.diff(pos) != 1)[0]
                    firsts, lasts = sel[np.concatenate([[0], brk + 1])], sel[np.concatenate([brk, [len(sel) - 1]])]
                    col = fact.data[fact.cluster]
                    rr = [(int(np.searchsorted(col, a, "left")), int(np.searchsorted(col, b_, "right"))) for a, b_ in zip(firsts, lasts)]
                row_ranges = rr if row_ranges is None else [(max(a, c_), min(b_, d_)) for a, b_ in row_ranges for c_, d_ in rr if max(a, c_) < min(b_, d_)]
            # the join as one polynomial on the fk's planes: f(fk), f = the dimension filter (see JoinPoly)
            mode = {"prefix": "auto", "between": "factored", "bitmap": "lookup"}.get(self.join_mode, self.join_mode)
            planned = None
            if mode != "lookup" and fkn in fact.msb_cols:
                attrs = set().union(*[pred_cols(c) for c in preds]) if mode != "factored" else set()   # factored: id variables
                poly = JoinPoly(space, attrs, bm, b)
                planned = poly.plan(mode)
            if dim in getattr(self, "_lookup_dims", ()): planned = None     # rows of moved keys: current values by lookup
            if planned is not None:
                prog.extend(poly.program(fkc, planned))
                if had_fact or nrange: prog.append(PInstr(P["AND"], 0, 0, 0))
                nrange += 1
                d_ = JoinPoly.describe(planned)
                self.range_used.append((dim.name,) + (d_ if isinstance(d_, tuple) else (d_, 0)) + (poly.space,))
            else:
                semis.append((cnt / max(dim.n, 1), fkc, bm, dim.n, fact.kmask(fkn)))    # (an empty dimension: 0)
        if getattr(self, "_xrows", None) is not None:
            xr, xm = self._xrows; keep.append(xr); plan.xrows = xr.ctypes.data; plan.xmode = xm
        if getattr(self, "_word_ranges", None) is not None and row_ranges is None:
            wr = self._word_ranges; keep.append(wr); plan.nranges = len(wr) // 2; plan.ranges = wr.ctypes.data; self.scanned = None
        elif row_ranges is not None:
            # the 64-row words of the row ranges, ascending and disjoint: two ranges rounded out to words can share
            # one, which must be scanned once (the GPU executors scanned it once per range and counted its rows twice)
            ws = []
            for a, b_ in sorted(row_ranges):
                if b_ <= a: continue
                lo, hi = a // 64, (b_ + 63) // 64
                if ws and lo <= ws[-1][1]: ws[-1][1] = max(ws[-1][1], hi)
                else: ws.append([lo, hi])
            wr = np.array(ws or [(0, 0)], dtype=np.uint64).ravel()
            keep.append(wr); plan.nranges = len(wr) // 2; plan.ranges = wr.ctypes.data
            self.scanned = float(sum(int(wr[2 * i + 1] - wr[2 * i]) for i in range(len(wr) // 2))) / max(1, (fact.n + 63) // 64)
        else: self.scanned = 1.0
        if len(prog) > 1024: raise ValueError("filter program too long")
        plan.nprog = len(prog)
        for i, ins in enumerate(prog): plan.prog[i] = ins
        semis.sort(key=lambda s: s[0])
        plan.nsemi = len(semis)
        for i, (_, col, bm, dn, km) in enumerate(semis):
            plan.semi_col[i] = col; plan.semi_bm[i] = bm.ctypes.data; plan.semi_n[i] = dn; plan.semi_kmask[i] = km
        # dimension attributes used in keys or the measure
        attrs = {}
        def attr_id(col):
            key = (col.table, col.name)
            if key not in attrs:
                vals = col.table.decoded(col.name) if engine == "alg" else col.table.data[col.name]
                attrs[key] = (len(attrs), fact.idx(self.joins[col.table]), vals, fact.kmask(self.joins[col.table]))
            return attrs[key][0]
        if getattr(self, "_dom", None) is not None:
            # dominance slices for a range join: rows with t_(k-1) <= x < t_k, measure by popcount algebra
            xname, ts = self._dom
            poly = poly2(self.measure, fact) if self.measure is not None else {(): 0}
            plan.agg = A_DOM; plan.dom_col = fact.idx(xname); plan.ndom = len(ts); plan.dom_t = ts.ctypes.data; keep.append(ts)
            if poly is None: raise ValueError("a range join's measure must be a polynomial of degree <= 2 in fact columns")
            _set_poly(plan, poly, self._wide, fact)
            out_sum = np.zeros(len(ts), np.int64); out_cnt = np.zeros(len(ts), np.uint64); out_hi = np.zeros(len(ts), np.int64)
            keep += [out_sum, out_cnt, out_hi]
            plan.ngroups, plan.nkey = len(ts), 1                # one group per slice (merged across threads like groups)
            plan.out_sum, plan.out_cnt, plan.out_sum_hi = out_sum.ctypes.data, out_cnt.ctypes.data, out_hi.ctypes.data
            if not self._exec(self._dev, fact, plan):
                r = self._run_codes("alg"); self.device = "cpu (gpu: " + self._gpu_fail + ")"; return r
            self.last_plan = plan
            if self._wide:
                return (np.array([_join128(a, b) for a, b in zip(out_sum, out_hi)], object), out_cnt.astype(np.int64),
                        _join128(plan.total_sum, plan.total_sum_hi), int(plan.total_cnt))
            return out_sum.copy(), out_cnt.astype(np.int64), int(plan.total_sum), int(plan.total_cnt)
        if getattr(self, "_rows", False):         # the passing rows as a bitmap
            plan.agg = A_ROWS; plan.nkey = 0
            out_rows = np.zeros((fact.n + 63) // 64 + 1, np.uint64); keep.append(out_rows); plan.out_rows = out_rows.ctypes.data
            plan.nattr = 0
            self._exec("plain" if engine == "plain" else "cpu", fact, plan); self.last_plan = plan
            return np.nonzero(np.unpackbits(out_rows.view(np.uint8), bitorder="little")[:fact.n])[0].astype(np.uint64)
        ext = getattr(self, "_ext", None)
        if ext is not None: plan.ext_on, plan.ext_col = 1, fact.idx(ext.name)
        # aggregation
        lin = self._linear(self.measure) if self.measure is not None else None
        wide = self._wide; plan.wide = wide
        if ext is not None and not self.keys: plan.agg = A_GENERAL; plan.nkey = 0     # MIN / MAX of the whole result
        elif wide and not self.keys:
            # 128-bit SUM without GROUP BY: the measure over each word's passing rows by popcount algebra, folded in
            # 128 bits (the measure must be a polynomial of degree <= 2 in fact columns)
            poly = poly2(self.measure, fact)
            if poly is None or len([m for m in poly if len(m) == 1]) > 8 or len([m for m in poly if len(m) == 2]) > 8:
                raise OverflowError("SUM needs 128 bits, which the engine evaluates for polynomials of degree <= 2 in fact columns")
            plan.agg = A_POLYG; plan.nkey = 0; _set_poly(plan, poly, True, fact)
        elif self.keys or (self.measure is not None and lin is None and not self._bilinear()):
            plan.agg = A_GENERAL
            meas = []
            def emit(e):
                if isinstance(e, Col):
                    meas.append(MInstr(M["FCOL"], fact.idx(e.name), 0) if e.table is fact else MInstr(M["ATTR"], attr_id(e), 0))
                elif isinstance(e, Const): meas.append(MInstr(M["CONST"], 0, e.v))
                else: emit(e.l); emit(e.r); meas.append(MInstr(M[{"+": "ADD", "-": "SUB", "*": "MUL"}[e.op]], 0, 0))
            if self.measure is not None: emit(self.measure)
            plan.nmeas = len(meas)
            for i, m in enumerate(meas): plan.meas[i] = m
            # GROUP BY keys: hierarchy attributes are read as fields of the fk (region read of its leading planes)
            # when the attribute is a function of an id prefix; fall back to lookups if the group space gets too big
            poly = poly2(self.measure, fact) if self.measure is not None else {(): 0}
            mm_ok = (engine == "alg" and bool(self.keys) and poly is not None and len([m for m in poly if len(m) == 1]) <= 8
                     and len([m for m in poly if len(m) == 2]) <= 8
                     and sum(fact.width(m[0]) * (fact.width(m[1]) if len(m) == 2 else 1) for m in poly if m) <= 1024)
            if ext is not None: mm_ok = False              # MIN / MAX: mask splitting or row by row, never the matrix product
            mm = (GROUPBY == "mm" or wide) and mm_ok      # 128-bit sums: the matrix product (or mask splitting)
            # the matrix product's output is sparse (no dense key domain): its group codes are 128-bit (64-bit while the
            # domain fits), so the radix product must stay below 2^128 - 1; the dense arrays of the other path need 2^22
            dom_limit = KEY_MAX if mm else (1 << 22)
            sparse_general = False
            if len(self.keys) > MAXKEY:
                raise ValueError(f"group domain too large: more than {MAXKEY} GROUP BY keys (a 64-bit fact column takes two)")
            cand = []
            for k in self.keys:
                if isinstance(k, PlaneKey):
                    pc = bin(k.mask).count("1"); cand.append([k, 1 << pc, ("plane", pc, None), True]); continue
                nat = k.table.colmax(k.name) + 1
                info = None
                if k.table is not fact and self.join_mode in ("prefix", "auto") and k.table not in self._lookup_keys \
                        and self.joins[k.table] in fact.msb_cols:
                    # a coordinate or prefix field is read from the fk's planes in the order of a column stored most
                    # significant bit first; another fk's planes hold its bits from the lowest, so the codes would be its
                    # bits reversed (wrong groups, codes past the dimension): its keys are looked up instead
                    bw = fact.width(self.joins[k.table])
                    sp_ = fact.space(self.joins[k.table], k.table)
                    co, pk = attr_coords(sp_, k.name, bw), prefix_key(sp_, k.name, bw)
                    if co is not None and (pk is None or len(co[0]) < pk[0]): info = ("coord", len(co[0]), co)
                    elif pk is not None: info = ("prefix", pk[0], pk)
                if k.table is not fact:                     # a looked-up key is rank-compressed: radix = distinct values
                    nat = len(k.table.ranked(k.name, engine)[0])
                cand.append([k, nat, info, info is not None])
            def _dom():
                p_ = 1
                for c in cand: p_ *= (1 << c[2][1]) if c[3] else int(c[1])
                return p_
            # many groups (by the keys' distinct counts): the row kernel, its plane keys transposed once per 64-row
            # word, gathers each row's group directly; the matrix product's cost grows with the groups of each tile.
            # Only while its dense group arrays (one entry per code combination of the keys as the plan reads them)
            # are no larger than the table, and with a filter only while they stay small: a filter may leave few rows
            # and groups, which the matrix product's sparse output handles without arrays over the whole code domain
            rowpath = False
            if mm and not wide and ext is None and ROWPATH_GROUPS and self._dev != "gpu":
                est = 1
                for c in cand: est *= int(c[1])
                ng_ = _dom()
                if est >= ROWPATH_GROUPS and ng_ <= min(fact.n, 1 << 22) and (self.pred is None or ng_ <= ROWPATH_DENSE):
                    mm, rowpath, dom_limit = False, True, 1 << 22
            while _dom() > dom_limit:
                fields = [c for c in cand if c[3] and not isinstance(c[0], PlaneKey)]
                if not fields and not mm and dom_limit < KEY_MAX:   # too many groups for dense arrays
                    if mm_ok: mm = True                      # the sparse matrix product,
                    else: sparse_general = True              # or the general path with sparse output (CPU)
                    dom_limit = KEY_MAX
                    for c in cand: c[3] = c[2] is not None
                    continue
                if not fields:
                    raise ValueError("group domain too large: the GROUP BY keys need more than "
                                     f"{dom_limit.bit_length() - 1} bits together ({', '.join(f'{c[0].name} {max(1, int(c[1] - 1).bit_length())}' for c in cand if not isinstance(c[0], PlaneKey))})")
                worst = max(fields, key=lambda c: (1 << c[2][1]) / c[1])
                worst[3] = False
            radix, kmaps = [], []
            for i, (k, nat, info, use_field) in enumerate(cand):
                kmap = None
                if isinstance(k, PlaneKey):
                    plan.key_kind[i], plan.key_arg[i] = 3, fact.idx(k.name); plan.key_mask[i] = k.mask; r = nat
                elif k.table is fact: plan.key_kind[i], plan.key_arg[i] = 0, fact.idx(k.name); r = nat
                elif use_field and info[0] == "coord":
                    # coordinate projection: the attribute is read from its id bits on the fk planes
                    bw = fact.width(self.joins[k.table]); R, kmap = info[2]
                    plan.key_kind[i], plan.key_arg[i] = 3, fact.idx(self.joins[k.table])
                    plan.key_mask[i] = sum(1 << (bw - 1 - kk) for kk in R); r = 1 << len(R)
                elif use_field:
                    plan.key_kind[i], plan.key_arg[i], plan.key_nbits[i] = 2, fact.idx(self.joins[k.table]), info[1]
                    r, kmap = 1 << info[1], info[2][1]
                else:                                       # lookup of the value's rank among the dimension's values
                    u, ranks = k.table.ranked(k.name, engine)
                    ak = (k.table, k.name, "rank")
                    if ak not in attrs: attrs[ak] = (len(attrs), fact.idx(self.joins[k.table]), ranks, fact.kmask(self.joins[k.table]))
                    plan.key_kind[i], plan.key_arg[i] = 1, attrs[ak][0]; r = len(u)
                    kmap = dict(enumerate(u.tolist()))
                plan.key_radix[i] = r; radix.append(r); kmaps.append(kmap)
            plan.nkey = len(self.keys)
            # keys that are functions of the physical clustering column + a degree-2 measure in fact columns:
            # GROUP BY by mask splitting and popcount algebra (no per-row integers)
            # mask splitting takes one step per distinct key in a 64-row word: the clustering key changes rarely,
            # a coordinate key has at most 2^bits values; keys read through dimension lookups are not on planes
            nw = max(1, (fact.n + 63) // 64); steps = 1.0; on_planes = True
            for (k, nat, info, use_p), r in zip(cand, radix):
                clustered = fact.cluster is not None and ((k.table is fact and k.name == fact.cluster) or
                                                          (k.table is not fact and use_p and self.joins[k.table] == fact.cluster))
                if k.table is not fact and not use_p: on_planes = False
                steps *= (1 + fact.cluster_breaks / nw) if clustered else min(64, r, nat)
            # mask splitting + popcounts: dense group arrays, or sparse output for 128-bit sums (the matrix product holds at
            # most 1024 measure planes: a product of two 64-bit columns is 4096 plane pairs, folded per mask here)
            polyg_ok = (not sparse_general or wide) and poly is not None and len([m for m in poly if len(m) == 1]) <= 8 and len([m for m in poly if len(m) == 2]) <= 8
            if mm or (polyg_ok and ((on_planes and steps <= (12 if not rowpath else 4)) or self._dev == "gpu" or (wide and on_planes))):
                plan.agg = A_MM if mm else A_POLYG
                _set_poly(plan, poly, wide, fact)
            elif wide and poly is not None and len([m for m in poly if len(m) == 1]) <= 8 and len([m for m in poly if len(m) == 2]) <= 8:
                _set_poly(plan, poly, wide, fact)               # the general path: row by row in 128 bits (lookup keys)
            elif wide:
                raise OverflowError("SUM needs 128 bits, which the engine evaluates for polynomials of degree <= 2 in fact columns")
            dist = getattr(self, "_distinct", False)
            if dist:                                          # COUNT(DISTINCT last key): dense cells, a bit each
                if sparse_general or self._dev == "gpu" or engine != "alg" or wide or ext is not None or self.measure is not None:
                    raise ValueError("COUNT(DISTINCT) cells: not on this plan")
                if any(m is not None and len(set(m.values())) != len(m) for m in kmaps):
                    raise ValueError("COUNT(DISTINCT) cells: keys whose codes merge")
                mm = False
            ng = 1
            for r_ in radix: ng *= int(r_)                    # exact (may pass 2^64)
            # 128-bit group codes from 2^64 groups on: a 64-bit code stays below 2^64 - 1, which the GPU executor's
            # group table (and 0.2.0's 64-bit tables) use as the empty slot
            plan.ngroups = min(ng, (1 << 64) - 1); plan.widekey = int(ng >= (1 << 64))
            if not dist and not mm and not sparse_general and plan.agg == A_GENERAL and ng > 4 * max(fact.n, 1 << 14):
                sparse_general = True                         # dense arrays far larger than the table: its groups by hashing
            if dist:
                if ng > min(DIST_CELLS, max(1 << 16, DIST_PER_ROW * fact.n)):   # bits to clear and count: about the rows, not more
                    raise ValueError("group domain too large: COUNT(DISTINCT) cells")
                R = int(radix[-1]); bits = np.zeros(ng // 64 + 2, np.uint64)
                out_sum = np.zeros(ng // R, np.int64); out_cnt = np.zeros(ng // R, np.uint64); keep += [bits, out_sum, out_cnt]
                plan.out_sum, plan.out_cnt = out_sum.ctypes.data, out_cnt.ctypes.data
                plan.agg = A_DIST; plan.nmeas = 0; plan.dist_bits = bits.ctypes.data; plan.dist_radix = R
            elif mm or sparse_general: self._mm_alloc(plan, min(ng, 1 << 14))
            else:
                if ng > 1 << 25: raise ValueError("group domain too large")
                out_sum = np.zeros(ng, dtype=np.int64); out_cnt = np.zeros(ng, dtype=np.uint64); keep += [out_sum, out_cnt]
                plan.out_sum, plan.out_cnt = out_sum.ctypes.data, out_cnt.ctypes.data
                if wide or ext is not None: out_hi = np.zeros(ng, dtype=np.int64); keep.append(out_hi); plan.out_sum_hi = out_hi.ctypes.data
                if ext is not None: out_sum.fill(np.iinfo(np.int64).max)
        elif self.measure is None: plan.agg = A_COUNT
        elif lin is not None:
            plan.agg = A_LIN; plan.nlin = len(lin)
            for i, (c, k) in enumerate(lin): plan.lin_col[i], plan.lin_coef[i] = fact.idx(c), k
        else:
            plan.agg = A_BIL; plan.bil_a, plan.bil_b = fact.idx(self.measure.l.name), fact.idx(self.measure.r.name)
        if len(attrs) > len(plan.attr_fkcol):          # the plan's slots (torble.h): one per looked-up dimension column
            raise ValueError(f"the query reads {len(attrs)} dimension columns through lookups, more than the "
                             f"{len(plan.attr_fkcol)} one engine query holds")
        plan.nattr = len(attrs)
        for _ak, (i, fk, vals, km) in attrs.items():
            plan.attr_fkcol[i] = fk; plan.attr_vals[i] = vals.ctypes.data; plan.attr_n[i] = len(vals); plan.attr_kmask[i] = km; keep.append(vals)
        sparse = bool(self.keys) and bool(plan.out_key)
        ctx = dict(plan=plan, keep=keep, radix=radix if self.keys else None, kmaps=kmaps if self.keys else None,
                   dist=(bits, R) if self.keys and plan.agg == A_DIST else None,
                   dense=((out_sum, out_cnt) + ((out_hi,) if wide or ext is not None else ())) if self.keys and not sparse else None, sparse=sparse, wide=wide, ext=ext is not None,
                   widekey=bool(self.keys) and bool(plan.widekey),
                   mm_out=getattr(self, "_mm_out", None) if sparse else None,
                   range_used=self.range_used, scanned=getattr(self, "scanned", None), fresh=True)
        if sig is not None:
            ctx["objs"] = sig[1]
            if not hasattr(self, "_plan_cache"): self._plan_cache = {}
            self._plan_cache[sig[0]] = ctx
        self.plan_ms, self.plan_cached = (_time.perf_counter() - t_plan) * 1e3, False
        return self._finish(ctx)

    def _plan_sig(self, engine):
        """cache key of the compiled plan (None: do not cache): engine, GROUP BY mode, join form, derived-surrogate
        versions; plus the query's parts, compared by identity (a changed query has new parts)"""
        if not PLAN_CACHE or getattr(self, "_rows", False): return None
        for a in ("_inpass", "_xrows", "_word_ranges", "_dom", "_lookup_dims", "_force_lookup"):
            if getattr(self, a, None): return None
        surr = getattr(self.fact, "surr", {})
        key = (engine, self._dev, GROUPBY, self.join_mode, getattr(self, "_ext", None) is not None, getattr(self, "_rows", False),
               getattr(self, "_distinct", False), ROWPATH_GROUPS, ROWPATH_DENSE, tuple(sorted((k, sg.version) for k, sg in surr.items() if sg is not None)),
               tuple((d.name, f) for d, f in self.joins.items()), len(self.keys), getattr(self, "_wide", 0), self.fact.n)
        objs = (self.pred, self.measure, getattr(self, "_ext", None), getattr(self, "_rows", False)) + tuple(self.keys) + tuple(self.joins)
        return key, objs

    def _finish(self, ctx):
        """execute a compiled plan (fresh or cached) and turn its output into the result"""
        plan, radix, kmaps = ctx["plan"], ctx["radix"], ctx["kmaps"]
        if not ctx.pop("fresh", False):             # a cached plan: clear its outputs
            if ctx["dense"] is not None and ctx.get("touched") is not None:     # only the entries the last run set
                for a in ctx["dense"]: a[ctx["touched"]] = 0
                if ctx.get("ext"): ctx["dense"][0][ctx["touched"]] = np.iinfo(np.int64).max
            if ctx.get("dist") is not None: ctx["dist"][0].fill(0)
            if ctx.get("sparse"):
                self._mm_out = ctx["mm_out"]; plan.out_n = 0
                self._mm_bind(plan); plan.out_cap = len(self._mm_out[0])
            self.range_used, self.scanned = ctx["range_used"], ctx["scanned"]
        if not self._exec(self._dev, self.fact, plan):
            r = self._run_codes("alg"); self.device = "cpu (gpu: " + self._gpu_fail + ")"; return r
        if ctx.get("sparse"): ctx["mm_out"] = self._mm_out          # grown if the groups did not fit
        self.last_plan = plan
        wide = ctx.get("wide", 0)
        if ctx["dense"] is not None: out_sum, out_cnt = ctx["dense"][:2]
        ext = ctx.get("ext", False)
        if not self.keys and ext:    # MIN / MAX of nothing is NULL
            return MinMax((int(plan.total_sum), int(plan.total_sum_hi))) if plan.total_cnt else None
        if not self.keys:            # SQL semantics: COUNT of nothing is 0, SUM of nothing is NULL
            if self.measure is None: return int(plan.total_cnt)
            if not plan.total_cnt: return (None, 0) if getattr(self, "_both", False) else None
            v = _join128(plan.total_sum, plan.total_sum_hi) if wide else int(plan.total_sum)
            return (v, int(plan.total_cnt)) if getattr(self, "_both", False) else v
        wkey = ctx.get("widekey", False)
        if ctx.get("dist") is not None:     # COUNT(DISTINCT last key) per group of the other keys: (key matrix, counts, counts)
            nz = np.nonzero(out_cnt)[0]; ctx["touched"] = nz
            cols, rem = [], nz.astype(np.uint64)
            for r in reversed(radix[:-1]): cols.append(rem % np.uint64(r)); rem = rem // np.uint64(r)
            cols = [c_.astype(np.int64) for c_ in cols[::-1]]
            for i, m in enumerate(kmaps[:-1]):
                if m is not None:
                    lut = np.full(radix[i], -1, np.int64); lut[np.fromiter(m.keys(), np.int64)] = np.fromiter(m.values(), np.int64)
                    cols[i] = lut[cols[i]]
            cn = out_cnt[nz].astype(np.int64)
            return (np.stack(cols, 1) if cols else np.zeros((len(nz), 0), np.int64)), cn, cn
        if ctx.get("sparse"):               # sparse output: (group code, sum, count) per group present
            n = int(plan.out_n); gk, gs, gc, gh, gkh = (a[:n] for a in self._mm_out)
            o = np.lexsort((gk, gkh)) if wkey else np.argsort(gk, kind="stable")      # group codes, in order
            gk, gs, gc, gh, gkh = gk[o], gs[o], gc[o], gh[o], gkh[o]
        else:
            nz = np.nonzero(out_cnt)[0]; gk, gs, gc = nz, out_sum[nz], out_cnt[nz]; ctx["touched"] = nz
            gh = ctx["dense"][2][nz] if wide or ext else None
        if wide: gs = _join128_arr(gs, gh)                                     # exact 128-bit sums
        def digits():
            """the mixed-radix digits of the group codes, one array per key (128-bit codes: Python integers)"""
            if not wkey:
                cols, rem = [], np.asarray(gk, np.uint64)
                for r in reversed(radix): cols.append(rem % np.uint64(r)); rem = rem // np.uint64(r)
                return cols[::-1]
            if all(int(r_) < 1 << 32 for r_ in radix): return _digits128(gk, gkh, radix)
            cols = [[] for _ in radix]
            for lo_, hi_ in zip(gk.tolist(), gkh.tolist()):
                c = (int(hi_) << 64) | int(lo_)
                for i in range(len(radix) - 1, -1, -1): c, d = divmod(c, int(radix[i])); cols[i].append(d)
            return [np.array(c_, np.uint64) for c_ in cols]
        if getattr(self, "_raw", False):    # arrays instead of a dict: (key matrix, sums, counts)
            cols = [c_.astype(np.int64) for c_ in digits()]
            for i, m in enumerate(kmaps):
                if m is not None:
                    lut = np.full(radix[i], -1, np.int64); lut[np.fromiter(m.keys(), np.int64)] = np.fromiter(m.values(), np.int64)
                    cols[i] = lut[cols[i]]
            km = np.stack(cols, 1) if cols else np.zeros((len(gk), 0), np.int64)
            cn = np.asarray(gc, np.int64)
            return km, ((gs if wide else np.array(gs, np.int64)) if self.measure is not None else cn), cn
        # keys in bulk: mixed-radix digits by array arithmetic, value maps by lookup tables (cached on the plan)
        cols = digits()
        luts = ctx.setdefault("luts", {})
        for i, m in enumerate(kmaps):
            if m is None: continue
            if i not in luts:
                kk = np.fromiter(m.keys(), np.int64, len(m)); vv = np.fromiter(m.values(), np.int64, len(m))
                lut = np.full(int(kk.max()) + 1 if len(kk) else 1, -1, np.int64); lut[kk] = vv; luts[i] = lut
            cols[i] = luts[i][cols[i]]
        if self is getattr(self, "_cols_owner", None):
            # columns, not a dict (Query.run_columns): one row per group code; with lookup keys (several codes of one
            # value) the rows that share a key are merged: counts and sums add, extremes take the min / max
            cna = np.asarray(gc, np.int64)
            if ext: va = None; mn, mx = np.asarray(gs, np.int64), np.asarray(gh, np.int64)
            elif self.measure is None: va = cna
            else: va = gs if wide else np.asarray(gs, np.int64)
            cols = list(cols)
            if any(m is not None for m in kmaps) and len(cna) > 1:
                kc = [np.asarray(c).astype(np.int64) for c in cols]
                uk, inv = _unique_rows(kc)
                if len(uk[0]) < len(cna):
                    g = len(uk[0]); c2 = np.zeros(g, np.int64); np.add.at(c2, inv, cna)
                    if ext:
                        a = np.full(g, np.iinfo(np.int64).max, np.int64); np.minimum.at(a, inv, mn)
                        b = np.full(g, np.iinfo(np.int64).min, np.int64); np.maximum.at(b, inv, mx); mn, mx = a, b
                    elif self.measure is None: va = c2
                    elif va.dtype == object or wide:         # 128-bit plans: the merged sums may pass 64 bits
                        v2 = [0] * g
                        for i, v in zip(inv.tolist(), va.tolist()): v2[i] += v
                        va = np.array(v2, object)
                    else: v2 = np.zeros(g, np.int64); np.add.at(v2, inv, va); va = v2
                    cna = c2; cols = uk
            if ext: va = [MinMax(p) for p in zip(mn.tolist(), mx.tolist())]
            return CodeGroups(cols, va, cna if getattr(self, "_both", False) else None)
        keys = list(zip(*[c.tolist() for c in cols])) if cols else [()] * len(gk)
        cn = np.asarray(gc, np.int64).tolist()
        vs = (gs.tolist() if wide else np.asarray(gs, np.int64).tolist()) if self.measure is not None else cn
        if ext: vs = [MinMax(p) for p in zip(np.asarray(gs, np.int64).tolist(), np.asarray(gh, np.int64).tolist())]
        if getattr(self, "_both", False):
            out = {}
            for key, v, c in zip(keys, vs, cn): o = out.get(key, (0, 0)); out[key] = (o[0] + v, o[1] + c)
            return out
        if len(set(keys)) == len(keys): return dict(zip(keys, vs))
        out = {}                                    # several codes of one value (lookup keys): their groups merge
        for key, v in zip(keys, vs): out[key] = v if key not in out else (_merge_vals(out[key], v) if ext else out[key] + v)
        return out

    def _run_range(self, engine):
        """Range / inequality join with aggregation, through the comparator polynomial.
        SUM over pairs (x, y) of m(x) [lo(y) <= x <= hi(y)] splits by the dimension group g(y) and the fact group k:
            [lo <= x <= hi] = [x < hi + 1] - [x < lo]
        so the answer is the dominance form  R[k, g] = sum_{x < t} M[k, x] H[g, t]  with
            M[k, x] = the measure aggregated by (k, x) on the bit-plane engine (all other joins and filters applied),
            H[g, t] = +1 at t = hi(y) + 1 and -1 at t = lo(y) for every passing dimension row y in group g.
        Coordinates are rank-transformed (order-preserving, so the comparator is unchanged) and the dominance sum is
        evaluated by the comparator polynomial [x < t] = sum_j (1 + x_j) t_j E_j: at level j, x and t agree above
        bit j and differ at it, which pairs the even blocks of size 2^j of M with the odd blocks of H (dominance())."""
        dim, x, lo, hi = self.rjoin
        if not (isinstance(x, Col) and x.table is self.fact): raise ValueError("the range join's x must be a fact column")
        if getattr(self, "_ext", None) is not None: raise ValueError("MIN / MAX over a range or inequality join: not supported")
        both = getattr(self, "_both", False) and self.measure is not None      # (sum, count) per group
        dpreds = [c for c in conjuncts(self.pred) if tables_of(c) == {dim}]
        others = [c for c in conjuncts(self.pred) if tables_of(c) != {dim}]
        fkeys = [k for k in self.keys if k.table is not dim]; dkeys = [k for k in self.keys if k.table is dim]
        inner = Query(self.fact, self.join_mode); inner.joins = dict(self.joins); inner._codes = True
        for c in others: inner.where(c)
        inner.measure = self.measure
        # dimension side: passing rows, their bounds and group keys
        dp = []
        for i, c in enumerate(dpreds):
            compile_pred(c, dim, dp)
            if i: dp.append(PInstr(P["AND"], 0, 0, 0))
        if dp:
            arr = (PInstr * len(dp))(*dp); bm = np.zeros((dim.n + 63) // 64 + 1, dtype=np.uint64)
            if engine in ("alg", "gpu"): lib.at_filter_bitmap_masked(dim.at, arr, len(dp), bm.ctypes.data, dim.used_bm.ctypes.data)
            else: lib.pt_filter_bitmap(dim.plain_handle(), arr, len(dp), bm.ctypes.data); bm[:len(dim.used_bm)] &= dim.used_bm[:len(bm)]
            ids = np.nonzero(np.unpackbits(bm.view(np.uint8), bitorder="little")[:dim.n])[0]
        else: ids = dim.used_ids
        col = (lambda c: dim.decoded(c)) if engine in ("alg", "gpu") else (lambda c: dim.data[c])
        def ev(e):
            if isinstance(e, Const): return np.full(len(ids), e.v, np.int64)
            if isinstance(e, Col):
                if e.table is not dim: raise ValueError("range bounds must be dimension expressions")
                return col(e.name)[ids].astype(np.int64)
            l, r = ev(e.l), ev(e.r)
            return l + r if e.op == "+" else l - r if e.op == "-" else l * r
        lov = ev(lo) if lo is not None else None; hiv = ev(hi) if hi is not None else None
        if lov is not None and hiv is not None:
            ok = lov <= hiv; lov, hiv, ids = lov[ok], hiv[ok], ids[ok]
        gk = np.stack([col(k.name)[ids].astype(np.int64) for k in dkeys], 1) if dkeys else np.zeros((len(ids), 0), np.int64)
        # Two evaluations of the same sum. (a) group the fact by (fact keys, x) and apply the comparator polynomial
        # to the aggregates (dominance()); good when x is clustered or has few values. (b) the comparator on x's own
        # planes, one slice per distinct bound: good when x has many values and the bounds are few.
        ends = np.unique(np.concatenate(([lov] if lov is not None else []) + ([hiv + 1] if hiv is not None else [])))
        xmax = self.fact.colmax(x.name)
        ts = ends[(ends > 0) & (ends <= xmax)]
        poly_ok = self.measure is None or poly2(self.measure, self.fact) is not None
        force = getattr(self, "range_mode", "auto")
        can_slice = not fkeys and poly_ok
        self.range_path = ("slices" if can_slice and force == "slices" else "groups" if force != "auto" else
                           "slices" if (can_slice and len(ts) <= 64 and x.name != self.fact.cluster and xmax >= 4096) else "groups")
        self.inner = inner
        if self.range_path == "slices":
            inner._dom = (x.name, ts.astype(np.uint32))
            ss, sc, tot_s, tot_c = inner._run_codes(engine); self.device, self.kernel_ms = inner.device, inner.kernel_ms
            Ds = np.concatenate([np.array([0], object), np.cumsum(np.asarray(ss, object))])        # D(t) at t = 0, ts... (exact)
            Dc = np.concatenate([[0], np.cumsum(sc)])
            def D(arr, tot, t):                                # D(t) for any bound: 0 below the data, the total above it
                return np.where(t <= 0, 0, np.where(t > xmax, tot, arr[np.searchsorted(ts, t, "right")]))
            gu, ginv = np.unique(gk, axis=0, return_inverse=True) if len(gk) else (np.zeros((0, len(dkeys)), np.int64), np.zeros(0, np.int64))
            ginv = np.asarray(ginv).ravel()
            hi_t = hiv + 1 if hiv is not None else np.full(len(ids), xmax + 1)
            vs = D(Ds, tot_s, hi_t) - (D(Ds, tot_s, lov) if lov is not None else 0)
            vc = D(Dc, tot_c, hi_t) - (D(Dc, tot_c, lov) if lov is not None else 0)
            Rs = np.zeros(len(gu), object); Rc = np.zeros(len(gu), np.int64)
            np.add.at(Rs, ginv, vs.astype(object)); np.add.at(Rc, ginv, vc)
            out = {tuple(gu[g].tolist()): ((int(Rs[g]), int(Rc[g])) if both else int(Rs[g]) if self.measure is not None else int(Rc[g]))
                   for g in range(len(gu)) if Rc[g] > 0}
            if self.keys: return out
            tc = int(Rc.sum())
            if self.measure is None: return tc
            return (int(Rs.sum()) if tc else None, tc) if both else int(Rs.sum()) if tc else None
        inner.keys = fkeys + [x]; inner._raw = True
        km_, ms_, mc_ = inner._run_codes(engine); self.device, self.kernel_ms = inner.device, inner.kernel_ms
        # fact side: (k, x) -> (sum, count)
        mk, mx, ms, mc = km_[:, :-1], km_[:, -1], (ms_ if ms_.dtype == object else ms_.astype(np.int64)), mc_
        # order-preserving coordinates: x values, lo, hi + 1, and +infinity for an unbounded top
        parts = [mx] + ([lov] if lov is not None else []) + ([hiv + 1] if hiv is not None else [])
        coords = np.unique(np.concatenate(parts)) if sum(len(p) for p in parts) else np.zeros(0, np.int64)
        INF = len(coords); nb = max(1, int(INF).bit_length()); size = 1 << nb
        ku, kinv = np.unique(mk, axis=0, return_inverse=True) if len(mk) else (np.zeros((0, len(fkeys)), np.int64), np.zeros(0, np.int64))
        gu, ginv = np.unique(gk, axis=0, return_inverse=True) if len(gk) else (np.zeros((0, len(dkeys)), np.int64), np.zeros(0, np.int64))
        kinv, ginv = np.asarray(kinv).ravel(), np.asarray(ginv).ravel()
        xr = np.searchsorted(coords, mx)
        H = np.zeros((len(gu), size), np.int64)
        np.add.at(H, (ginv, np.searchsorted(coords, hiv + 1) if hiv is not None else np.full(len(ids), INF)), 1)
        if lov is not None: np.add.at(H, (ginv, np.searchsorted(coords, lov)), -1)
        Mc = np.zeros((len(ku), size), np.int64); np.add.at(Mc, (kinv, xr), mc)
        big = ms.dtype == object or (len(ms) and max(-int(ms.min()), int(ms.max())) * max(1, len(ms)) * max(1, int(np.abs(H).sum())) >= 1 << 62)
        Ms = np.zeros((len(ku), size), object if big else np.int64); np.add.at(Ms, (kinv, xr), ms.astype(object) if big else ms)
        Rs, Rc = dominance(Ms, H if not big else H.astype(object), nb), dominance(Mc, H, nb)
        out = {}
        for a in range(len(ku)):
            for g in range(len(gu)):
                if Rc[a, g] > 0:
                    fk_, dk_ = iter(ku[a].tolist()), iter(gu[g].tolist())
                    key = tuple(next(dk_) if k.table is dim else next(fk_) for k in self.keys)
                    out[key] = (int(Rs[a, g]), int(Rc[a, g])) if both else int(Rs[a, g]) if self.measure is not None else int(Rc[a, g])
        if self.keys: return out
        tot_c = int(Rc.sum()) if Rc.size else 0
        if self.measure is None: return tot_c
        return (int(Rs.sum()) if tot_c else None, tot_c) if both else int(Rs.sum()) if tot_c else None

    def _bilinear(self):
        e = self.measure
        return (isinstance(e, Arith) and e.op == "*" and isinstance(e.l, Col) and isinstance(e.r, Col)
                and e.l.table is self.fact and e.r.table is self.fact)
    def plan_kind(self):
        if self.keys:
            a = getattr(getattr(self, "last_plan", None), "agg", None)
            return {A_POLYG: "mask split + popcounts", A_MM: "AND-popcount matrix product"}.get(a, "general")
        if getattr(self, "_ext", None) is not None: return "MIN / MAX (plane elimination)"
        if self.measure is None: return "count (popcount)"
        if self._linear(self.measure) is not None: return "linear (popcount algebra)"
        if self._bilinear(): return "bilinear (popcount algebra)"
        return "general"
