# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Bring your own data: load tables from DuckDB, Parquet, CSV or data frames into torbles, with typed columns and an
automatically designed key for every foreign key.

    import torbles
    db = torbles.from_duckdb("ssb.duckdb", foreign_keys={"lineorder.lo_custkey": "customer", ...})
    lo, c = db["lineorder"], db["customer"]
    db.query(lo).join(c).where(c.c_region == "ASIA").group_by(c.c_nation).sum(lo.lo_revenue).run()
    db.save("ssb.torbles");  db = torbles.load("ssb.torbles")

DuckDB reads the sources and does the heavy, parallel conversions (dictionary ranks, decimal scaling, key lookups).
Keys come from standard schema declarations: PRIMARY KEY / FOREIGN KEY constraints (SQL DDL, ISO/IEC 9075), read
through the standard INFORMATION_SCHEMA views of the source database; a DDL script for files (from_ddl); or a
Frictionless Data Table Schema (primaryKey / foreignKeys) next to CSV or Parquet files (from_frictionless). Keys given
as arguments are added. Guessing keys from names and values is opt-in (infer_keys=True) and reported, because a wrong
guess silently changes what joins. Each referenced table is stored sorted by its primary key; each foreign key stores
the referenced row's derived surrogate (see keydesign.py). torbles does not enforce foreign keys: a reference without a
match (or NULL) never joins, as in an SQL inner join. A column with NULLs gets a validity plane (its hidden column
c#v, 1 where the value is not NULL); fill_nulls= replaces NULLs by a value instead."""
import json, os, re, time
from threading import RLock
import numpy as np
import algq
from algq import Table, Query, Surrogate
import coltypes as ct
import keydesign

FORMAT = 1
_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "HUGEINT", "UHUGEINT"}
_TS_TYPES = {"TIMESTAMP", "TIMESTAMP_S", "TIMESTAMP_MS"}
_TSTZ_TYPES = {"TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}


def _q(name): return '"' + str(name).replace('"', '""') + '"'


class LoadError(ValueError):
    pass


def _physical(data, extend, refs):
    """the torble columns of a table's columns, the hidden ones included: validity planes (in data), the flags and
    high parts of the columns past 32 bits (two each), the references' bits (one column)"""
    return len(data) + 2 * len(extend) + bool(refs)


def _same_per_value(data, nullable, C, first, inv, rows):
    """is column C (with its validity plane) constant on the rows that share a value (inv: each row's value, first: a row
    of each value)?"""
    v = data[C][rows]
    if not np.array_equal(v[first][inv], v): return False
    vc = nullable.get(C)
    if vc is None: return True
    f = data[vc][rows]
    return np.array_equal(f[first][inv], f)


def fd_candidates(data, nullable, candidates, n, sample=1 << 16):
    """The determinants worth checking, from about `sample` rows: {D: (the columns D determines there, an estimate of
    D's number of values)}, for D never NULL, with at most half as many values as rows and two determined columns or
    more. The rows are a stride of the table when D's values repeat in it; otherwise (a key of many values, each on a
    few rows: a stride would see most of them once, and every column would look determined) every row of a sample of
    D's values, chosen by a hash of the value, so that each value is seen with all its rows. A dependency on every row
    holds on any subset of the rows, so nothing is missed; what the sample suggests, fd_verify checks on every row."""
    if n < 4: return {}
    step = max(1, n // sample)
    out = {}
    for D in candidates:
        if D in nullable: continue
        v = data[D]
        rows, scale = slice(0, n, step), 1
        u, first, inv = np.unique(v[rows], return_index=True, return_inverse=True)
        if step > 1 and 2 * len(u) > len(np.asarray(inv).ravel()):       # few repeats in a stride: sample values
            h = np.asarray(v, np.uint32) * np.uint32(2654435761)           # Knuth's multiplicative hash (wraps)
            rows, scale = np.flatnonzero(h < np.uint32((1 << 32) // step)), step
            u, first, inv = np.unique(v[rows], return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        if not len(inv) or 2 * len(u) > len(inv): continue
        deps = [C for C in candidates if C != D and _same_per_value(data, nullable, C, first, inv, rows)]
        if len(deps) >= 2: out[D] = (deps, len(u) * scale)
    return out


def fd_verify(data, nullable, D, deps, n):
    """the columns of deps that D determines on every row, and D's values: (columns, (values, a row of each, each
    row's value)); no columns when D has more than half as many values as rows"""
    u, first, inv = np.unique(data[D], return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    if 2 * len(u) > n: return [], (u, first, inv)
    return [C for C in deps if _same_per_value(data, nullable, C, first, inv, slice(None))], (u, first, inv)


def fd_groups(data, nullable, candidates, n, sample=1 << 16):
    """The columns each column determines: {determinant: (columns, its number of values)}. A column C is determined by
    D when the rows with the same value of D have the same value of C, or are all NULL in C (codes and validity planes
    compared: data holds the codes, nullable maps a column to its validity plane's name in data). Only the candidates
    are considered, as determinants (never NULL, and with at most half as many values as rows: there is storage to
    save) and as determined columns; a determinant is kept with two columns or more. A strided sample of the rows rules
    out most pairs (fd_candidates) before the check on every row (fd_verify)."""
    out = {}
    for D, (deps, _) in fd_candidates(data, nullable, candidates, n, sample).items():
        deps, (u, _, _) = fd_verify(data, nullable, D, deps, n)
        if len(deps) >= 2: out[D] = (deps, len(u))
    return out


class RefCodec(ct.Codec):
    """a foreign key column: it stores surrogates, so it is joined, not compared or grouped on directly"""
    kind = "ref"; affine = None

    def __init__(self, dim, pk): self.dim, self.pk, self.sql_type = dim, pk, "foreign key"
    def interval(self, op, v):
        raise TypeError(f"a foreign key stores the referenced row's surrogate: join {self.dim} and filter on {self.dim}.{self.pk}")
    def literal(self, v): return str(v)
    def decode(self, codes):
        raise TypeError(f"a foreign key stores the referenced row's surrogate: join {self.dim} and group by {self.dim}.{self.pk}")
    def _state(self): return dict(dim=self.dim, pk=self.pk)


class Database:
    """A set of torbles built from typed source tables, with their keys. Index it by table name for the tables
    (algq.Table, columns as attributes), build queries with query(), run them with .run(); .sql() renders the same
    query in the source's schema (verify() runs it on the source and compares)."""

    def __init__(self, con=None):
        import duckdb
        self._own = con is None
        self.con = con if con is not None else duckdb.connect()
        self._add = {}                                # name -> dict(primary_key, foreign_keys, columns)
        self.tables, self.meta, self.notes = {}, {}, []
        self._writable, self._sql_lock = {}, RLock()

    def add_writable_table(self, name, columns, *, primary_key="id"):
        """Register an in-memory uint32 table for SQL reads and atomic DML statements."""
        from torbles.sqlwrite import WritableTable

        with self._sql_lock:
            if self.con is None:
                raise RuntimeError("database is closed")
            if not isinstance(name, str) or not name:
                raise ValueError("writable table name must be a nonempty string")
            if any(k.lower() == name.lower() for k in (*self.tables, *self._writable, *self._add)):
                raise ValueError(f"table {name!r} already exists")
            table = WritableTable(name, columns, primary_key)
            if len({c.lower() for c in table.columns}) != len(table.columns):
                table.close()
                raise ValueError("writable column names must be unique ignoring case")
            self._writable[name] = table
        return self

    # ------------------------------------------------------------------ sources
    def add_table(self, name, source=None, primary_key=None, foreign_keys=None, columns=None):
        """source: a pandas / polars DataFrame, a pyarrow Table, a dict of arrays, a DuckDB relation, a file path
        (.parquet, .csv, .tsv, .json, globs allowed), or None for a table already in the connection.
        foreign_keys: {column: "table" or "table.column"}; columns: the columns to load (default: all)."""
        if any(k.lower() == name.lower() for k in self._writable):
            raise ValueError(f"table {name!r} already exists as a writable table")
        if source is not None:
            if isinstance(source, (str, os.PathLike)):
                p = str(source); low = p.lower()
                fn = "read_parquet" if ".parquet" in low or low.endswith(".pq") else \
                     "read_json_auto" if low.endswith((".json", ".ndjson", ".jsonl")) else "read_csv_auto"
                self.con.execute(f"CREATE OR REPLACE TABLE {self._t(name)} AS SELECT * FROM {fn}('{p.replace(chr(39), chr(39) * 2)}')")
            elif hasattr(source, "create") and hasattr(source, "query") and hasattr(source, "columns"):
                source.create(name)                                      # a DuckDB relation
            else:
                if isinstance(source, dict):
                    import pandas as pd; source = pd.DataFrame(source)
                tmp = f"__torbles_src_{len(self._add)}"
                self.con.register(tmp, source)
                self.con.execute(f"CREATE OR REPLACE TABLE {self._t(name)} AS SELECT * FROM {_q(tmp)}")
                self.con.unregister(tmp)
        self._add[name] = dict(primary_key=primary_key, foreign_keys=dict(foreign_keys or {}), columns=columns)
        return self

    # ------------------------------------------------------------------ build
    def build(self, infer_keys=False, fill_nulls=None, keep_plain=False, threads=None, verbose=False, types=None, groups=True):
        """Read every added table, design the keys, build the torbles. Returns self.
        types: {"table.column": "SQL type"} loads a column as another type (e.g. a DOUBLE amount as DECIMAL(18, 2), to
        sum it exactly); fill_nulls: {"table.column": value}; infer_keys: guess undeclared keys (reported in notes).
        A table wider than one torble (31 columns, the hidden ones included) is split. With groups, first the columns
        that one column determines (every row with the same value of it has the same values of them) go to a dimension
        of their own, stored once per value, that column becoming its foreign key: the groups whose dimensions are
        smallest first (cheap to join), while the table is wider than a torble, for tables that no foreign key
        references. The columns left over go to row-aligned parts (see algq.Table). Queries read either kind as
        columns of the table."""
        t0 = time.perf_counter()
        if threads: algq.set_build_threads(threads); algq.set_threads(threads)
        con, names = self.con, list(self._add)
        fill = dict(fill_nulls or {})
        self._src = {}
        cols = {n: self._columns(n) for n in names}
        for spec, typ in (types or {}).items():                      # typed copies of the tables with overrides
            t, _, c = spec.partition(".")
            if t not in cols or c not in cols[t]: raise LoadError(f"types: no column {spec}")
            cols[t][c] = typ
        for t in {spec.partition(".")[0] for spec in (types or {})}:
            sel = ", ".join(f"CAST({_q(c)} AS {ty}) AS {_q(c)}" if f"{t}.{c}" in types else _q(c) for c, ty in cols[t].items())
            self.con.execute(f"CREATE OR REPLACE TEMP TABLE {_q('__torbles_typed_' + t)} AS SELECT {sel} FROM {_q(t)}")
            self._src[t] = _q("__torbles_typed_" + t)
        pks, fks = self._keys(names, cols, infer_keys)
        dims = {d for f in fks.values() for d, _ in f.values()}
        for d in dims:
            if d not in pks: raise LoadError(f"{d} is referenced by a foreign key but has no single-column primary key")
        order = _topo(names, fks)
        stats = {n: self._stats(n, cols[n], pks.get(n), fks.get(n, {}), fill) for n in order}
        designs, rowmaps = {}, {}
        for name in order:
            tb = time.perf_counter()
            pk, fk = pks.get(name), fks.get(name, {})
            st = stats[name]; n = st["rows"]
            data, extend, codecs, info, nullable = {}, {}, {}, {}, {}
            order_by = f" ORDER BY {_q(pk)}" if name in dims else ""
            for c, typ in cols[name].items():
                if c in fk: continue
                codes, cd, why = self._encode(name, c, typ, st, order_by, name in dims, fill)
                if cd is None: self.notes.append(f"{name}.{c}: not loaded ({why})"); continue
                if st.get("nulls", {}).get(c):           # a validity plane: 1 where the value is not NULL
                    v = self.con.execute(f"SELECT CAST({_q(c)} IS NOT NULL AS UINTEGER) AS v FROM {self._t(name)}"
                                         + (order_by if order_by else " ORDER BY rowid")).fetchnumpy()["v"]
                    data[c + "#v"] = np.ascontiguousarray(np.asarray(v), np.uint32); nullable[c] = c + "#v"
                    codecs[c + "#v"] = ct.BoolCodec()
                if codes.dtype == np.uint64 and codes.size and int(codes.max()) > 0xFFFFFFFF:
                    if name in dims: raise LoadError(f"{name}.{c}: internal error, dimension codes wider than 32 bits")
                    extend[c] = 32
                    data[c] = codes
                else: data[c] = np.ascontiguousarray(codes, np.uint32)
                codecs[c] = cd; info[c] = dict(type=typ, codec=cd.kind)
            refs = {}
            for c, (dname, dpk) in fk.items():
                dt, ddes = self.tables[dname], designs[dname]
                codes = self._fk_codes(name, c, cols[name][c], dname, dpk, cols[dname][dpk], ddes, rowmaps[dname])
                data[c] = codes
                refs[c] = (dt, ddes["surrogate"]) if ddes["surrogate"] is not None else dt
                codecs[c] = RefCodec(dname, dpk); info[c] = dict(type=cols[name][c], codec="ref", references=f"{dname}.{dpk}")
                # references without a match (NULL or dangling): the key's own values are kept as a column c#raw, so
                # that the foreign key can be read as a value (the stored surrogate only names rows that exist)
                nn = self.con.execute(f"SELECT count(*) FROM {self._t(name)} f WHERE NOT EXISTS (SELECT 1 FROM {self._t(dname)} d "
                                      f"WHERE d.{_q(dpk)} = f.{_q(c)})").fetchone()[0]
                if nn:
                    rc = c + "#raw"
                    lo, hi = self.con.execute(f"SELECT min({_raw(c, cols[name][c]) or _q(c)}), max({_raw(c, cols[name][c]) or _q(c)}) FROM {self._t(name)}").fetchone() \
                        if _raw(c, cols[name][c]) is not None else (None, None)
                    st2 = dict(rows=st["rows"], nonnull={rc: st["nonnull"][c]}, min={rc: lo}, max={rc: hi})
                    rcodes, rcd, why = self._encode_as(name, c, cols[name][c], st2, rc, fill)
                    if rcd is not None:
                        if rcodes.dtype == np.uint64 and rcodes.size and int(rcodes.max()) > 0xFFFFFFFF: extend[rc] = 32; data[rc] = rcodes
                        else: data[rc] = np.ascontiguousarray(rcodes, np.uint32)
                        codecs[rc] = rcd
                        if st["nonnull"][c] < st["rows"]:
                            v = self.con.execute(f"SELECT CAST({_q(c)} IS NOT NULL AS UINTEGER) AS v FROM {self._t(name)} ORDER BY rowid").fetchnumpy()["v"]
                            data[rc + "#v"] = np.ascontiguousarray(np.asarray(v), np.uint32); nullable[rc] = rc + "#v"; codecs[rc + "#v"] = ct.BoolCodec()
            if not data: raise LoadError(f"{name}: no column could be loaded")
            moved, made = {}, {}                       # column -> its determinant; determinant -> its dimension
            if groups and name not in dims and _physical(data, extend, refs) > algq.MAXCOL:
                cand = [c for c in data if "#" not in c and c not in fk and c != pk and c not in extend and c in codecs]
                fds = fd_candidates(data, nullable, cand, n)      # from a sample; each checked on every row when its turn comes
                for D in sorted(fds, key=lambda D: (fds[D][1], -len(fds[D][0]), cand.index(D))):   # smallest first
                    if _physical(data, extend, refs) <= algq.MAXCOL or len(refs) >= 32: break   # (32 references' bits)
                    if D not in data or D in moved: continue
                    deps, vals = fd_verify(data, nullable, D, [C for C in fds[D][0] if C in data and C not in made], n)
                    if len(deps) < 2: continue
                    made[D] = self._group(name, D, deps, data, codecs, info, nullable, refs, designs, vals)
                    moved.update({C: D for C in deps})
            cluster = next((c for c in list(fk) + [c for c in data if c not in fk]
                            if c not in extend and "#" not in c and c not in nullable and n > 1 and bool(np.all(data[c][1:] >= data[c][:-1]))), None)
            t = Table(name, data, msb=tuple(fk) + tuple(made), refs=refs or None, extend=extend or None, cluster=cluster)
            t.codecs, t.pk, t.sqlname = codecs, pk, name
            t.nullable = nullable                       # column -> its validity column
            t.moved, t.groups = moved, made              # columns stored once per value of a determinant (_group)
            t.col_order = [c for c in cols[name] if c in data or c in moved]     # the declared order (SELECT *)
            t.fk_nulls = {c: int(st["rows"] - st["nonnull"][c]) for c in fk}     # NULL references (never join)
            self.tables[name] = t
            if name in dims:
                rowmaps[name] = self._pk_rows(name, pk, cols[name][pk], n)
                designs[name] = self._design(name, t, pk, fk, n)
            if name not in dims and not keep_plain: t.drop_plain()
            self.meta[name] = dict(rows=n, primary_key=pk, foreign_keys={c: f"{d}.{p}" for c, (d, p) in fk.items()},
                                   columns=info, cluster=cluster, design=_design_json(designs.get(name)),
                                   build_s=round(time.perf_counter() - tb, 3))
            if made: self.meta[name]["groups"] = {D: dict(table=g, columns=[c for c, d in moved.items() if d == D]) for D, g in made.items()}
            if t.parts is not None: self.meta[name]["parts"] = [list(p.cols) for p in t.parts]
            if verbose: print(f"  {name}: {n:,} rows, {len(data)} columns, {time.perf_counter() - tb:.2f} s", flush=True)
        self.build_s = time.perf_counter() - t0
        self._designs = designs
        return self

    def _group(self, name, D, deps, data, codecs, info, nullable, refs, designs, vals):
        """Columns deps of table `name`, each determined by column D (fd_verify), go to a dimension of their own,
        stored once per value of D in D's order and keyed by it (its key designed like any dimension's), and D becomes
        the foreign key to it: its codes in the table become the dimension's surrogates. Every value of D has its row,
        so every reference joins. Built from the table's codes (the same codecs); vals: D's values, a row of each and
        each row's value (fd_verify). Returns the dimension's name."""
        t0 = time.perf_counter()
        u, first, inv = vals
        g = f"{name}#{D}"
        gdata, gcodecs, gnull = {D: u.astype(np.uint32)}, {D: codecs[D]}, {}
        for C in deps:
            gdata[C] = np.ascontiguousarray(data[C][first]); gcodecs[C] = codecs[C]
            vc = nullable.get(C)
            if vc is not None: gdata[vc] = np.ascontiguousarray(data[vc][first]); gcodecs[vc] = codecs[vc]; gnull[C] = vc
        gt = Table(g, gdata, cluster=D)
        gt.codecs, gt.pk, gt.sqlname, gt.nullable = gcodecs, D, g, gnull
        gt.col_order, gt.fk_nulls, gt.generated = [D] + list(deps), {}, (name, D)
        self.tables[g] = gt
        des = designs[g] = self._design(g, gt, D, {}, len(u))
        lut = des["s_of_row"] if des["kind"] == "derived" else np.arange(len(u) + 1, dtype=np.int64)
        data[D] = np.ascontiguousarray(np.asarray(lut, np.int64)[inv], np.uint32)
        refs[D] = (gt, des["surrogate"]) if des["surrogate"] is not None else gt
        ginfo = {D: dict(info[D])}
        ginfo.update({C: dict(info[C]) for C in deps})
        codecs[D] = RefCodec(g, D); info[D] = dict(info[D], codec="ref", references=f"{g}.{D}")
        for C in deps:
            del data[C]; codecs.pop(C, None); info[C] = dict(info[C], stored_in=g)
            vc = nullable.pop(C, None)
            if vc is not None: del data[vc]; codecs.pop(vc, None)
        self.meta[g] = dict(rows=len(u), primary_key=D, foreign_keys={}, columns=ginfo, cluster=D, design=_design_json(des),
                            generated_for=name, build_s=round(time.perf_counter() - t0, 3))
        self.notes.append(f"{name}: {', '.join(deps)} stored once per value of {D} ({len(u):,} values), in {g}")
        return g

    def _t(self, name):
        """the SQL name to read a source table from"""
        return getattr(self, "_src", {}).get(name, _q(name))

    def _columns(self, name):
        rows = self.con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                                "ORDER BY ordinal_position", [name]).fetchall()
        if not rows: raise LoadError(f"no table {name} in the source")
        want = self._add[name]["columns"]
        out = {c: t for c, t in rows if want is None or c in want}
        if want is not None:
            missing = [c for c in want if c not in out]
            if missing: raise LoadError(f"{name}: no column(s) {', '.join(missing)}")
        return out

    def _keys(self, names, cols, infer):
        pks, fks = {}, {n: {} for n in names}
        for tname, typ, ccols, rt, rcols in self._declared_keys():
            if tname not in cols: continue
            if typ == "PRIMARY KEY" and len(ccols) == 1: pks[tname] = ccols[0]
            if typ == "PRIMARY KEY" and len(ccols) > 1: self.notes.append(f"{tname}: composite primary key ({', '.join(ccols)}) not used")
            if typ == "FOREIGN KEY" and rt in cols:
                if len(ccols) == 1: fks[tname][ccols[0]] = (rt, rcols[0] if rcols else None)
                else: self.notes.append(f"{tname}: composite foreign key ({', '.join(ccols)}) not used")
        for n in names:
            a = self._add[n]
            if a["primary_key"]: pks[n] = a["primary_key"]
            for c, ref in a["foreign_keys"].items():
                if "." in c and c.split(".", 1)[0] in cols: n2, c = c.split(".", 1)
                else: n2 = n
                d, _, dc = ref.partition(".")
                fks[n2][c] = (d, dc or None)
        for n in names:
            for c, (d, dc) in list(fks[n].items()):
                if c not in cols[n]:
                    if self._add[n]["columns"] is not None: del fks[n][c]; continue      # a column left out
                    raise LoadError(f"{n}.{c}: no such column")
                if d not in cols: raise LoadError(f"{n}.{c} references {d}, which was not added")
                if dc and pks.get(d, dc) != dc: raise LoadError(f"{n}.{c} references {d}.{dc}, but {d}'s primary key is {pks[d]}")
                if dc: pks[d] = dc
        if infer:
            for n in names:
                if n not in pks:
                    k = self._infer_pk(n, cols[n])
                    if k: pks[n] = k; self.notes.append(f"{n}: primary key {k} (inferred: unique, non-null)")
            for n in names:
                for c, typ in cols[n].items():
                    if c in fks[n] or c == pks.get(n): continue
                    for d, dk in pks.items():
                        if d == n or not _name_match(c, dk, d): continue
                        if _base_type(cols[d][dk]) != _base_type(typ): continue
                        miss = self.con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {_q(c)} AS v FROM {self._t(n)} WHERE {_q(c)} IS NOT NULL) x "
                                                f"WHERE NOT EXISTS (SELECT 1 FROM {self._t(d)} y WHERE y.{_q(dk)} = x.v)").fetchone()[0]
                        tot = self.con.execute(f"SELECT count(DISTINCT {_q(c)}) FROM {self._t(n)}").fetchone()[0]
                        if tot and miss <= 0.01 * tot:
                            fks[n][c] = (d, dk); self.notes.append(f"{n}.{c} -> {d}.{dk} (inferred: names and values match)")
                            break
        fks = {n: {c: (d, pks.get(d)) for c, (d, dc) in f.items()} for n, f in fks.items()}
        for n, f in fks.items():
            for c, (d, dk) in f.items():
                if dk is None: raise LoadError(f"{n}.{c} references {d}, which has no primary key")
                if _base_type(cols[d][dk]) != _base_type(cols[n][c]):
                    raise LoadError(f"{n}.{c} ({cols[n][c]}) references {d}.{dk} ({cols[d][dk]}): types differ")
        return pks, fks

    def _declared_keys(self):
        """(table, PRIMARY KEY | FOREIGN KEY, columns, referenced table, referenced columns), from the standard
        INFORMATION_SCHEMA views (TABLE_CONSTRAINTS, KEY_COLUMN_USAGE, REFERENTIAL_CONSTRAINTS)"""
        out = []
        try:
            rows = self.con.execute("""
                SELECT tc.table_name, tc.constraint_type, tc.constraint_name, kcu.column_name, kcu.ordinal_position,
                       ref.table_name AS rt, ref.column_name AS rc
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name AND kcu.table_name = tc.table_name
                 AND kcu.constraint_schema = tc.constraint_schema
                LEFT JOIN information_schema.referential_constraints rc
                  ON rc.constraint_name = tc.constraint_name AND rc.constraint_schema = tc.constraint_schema
                LEFT JOIN information_schema.key_column_usage ref
                  ON ref.constraint_name = rc.unique_constraint_name AND ref.constraint_schema = rc.unique_constraint_schema
                 AND ref.ordinal_position = coalesce(kcu.position_in_unique_constraint, kcu.ordinal_position)
                WHERE tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY') AND tc.table_schema = current_schema()
                ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position""").fetchall()
            grp = {}
            for t, typ, cn, col, pos, rt, rc in rows:
                g = grp.setdefault((t, cn), [t, typ, [], None, []])
                g[2].append(col)
                if rt is not None: g[3] = rt; g[4].append(rc)
            out = [tuple(g) for g in grp.values()]
        except Exception:                                  # a source without the standard views: DuckDB's own catalog
            try:
                out = [tuple(r) for r in self.con.execute("SELECT table_name, constraint_type, constraint_column_names, "
                       "referenced_table, referenced_column_names FROM duckdb_constraints() "
                       "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')").fetchall()]
            except Exception:
                out = []
        return out

    def _infer_pk(self, n, cols):
        cands = [c for c, t in cols.items() if _base_type(t) in ("int", "str", "date")]
        cands.sort(key=lambda c: (not re.search(r"(key|_?id)$", c, re.I), list(cols).index(c)))
        for c in cands[:6]:
            tot, nn, d = self.con.execute(f"SELECT count(*), count({_q(c)}), count(DISTINCT {_q(c)}) FROM {self._t(n)}").fetchone()
            if tot and nn == tot and d == tot and re.search(r"(key|id)$", c, re.I): return c
        return None

    def _stats(self, name, cols, pk, fk, fill):
        sel = ["count(*)"]
        for c, typ in cols.items():
            sel.append(f"count({_q(c)})")
            r = _raw(c, typ)
            if r is not None and c not in fk: sel += [f"min({r})", f"max({r})"]
        row = self.con.execute(f"SELECT {', '.join(sel)} FROM {self._t(name)}").fetchone()
        st = dict(rows=int(row[0]), nonnull={}, min={}, max={}); i = 1
        for c, typ in cols.items():
            st["nonnull"][c] = int(row[i]); i += 1
            if _raw(c, typ) is not None and c not in fk:
                st["min"][c], st["max"][c] = row[i], row[i + 1]; i += 2
            nulls = st["rows"] - st["nonnull"][c]
            st.setdefault("nulls", {})[c] = nulls if (c not in fk and f"{name}.{c}" not in fill and c not in fill) else 0
        if pk is not None:
            d, nn = self.con.execute(f"SELECT count(DISTINCT {_q(pk)}), count({_q(pk)}) FROM {self._t(name)}").fetchone()
            if d != st["rows"] or nn != st["rows"]:
                raise LoadError(f"{name}.{pk} is not a primary key: {st['rows'] - nn} NULL(s), {nn - d} duplicate(s)")
        return st

    def _encode(self, name, c, typ, st, order_by, is_dim, fill):
        """-> (codes uint64, codec, None) or (None, None, why)"""
        ok, why = _supported(typ)
        if not ok: return None, None, why
        base = _base_type(typ); col = _q(c)
        fv = fill.get(f"{name}.{c}", fill.get(c))
        if fv is not None:
            lit = _fill_literal(typ, fv); col = f"coalesce({col}, {lit})"
        if base in ("str", "float", "blob", "json", "row"):
            vals = [r[0] for r in self.con.execute(f"SELECT DISTINCT {col} AS v FROM {self._t(name)} WHERE {col} IS NOT NULL ORDER BY v").fetchall()]
            if base == "float": vals = [float(v) for v in vals]
            # NULLs sort last, so the ranks of the values do not depend on them; a NULL's code is 0 (its validity says NULL)
            codes = self.con.execute(f"SELECT CAST(CASE WHEN {col} IS NULL THEN 0 ELSE dense_rank() OVER (ORDER BY {col} NULLS LAST) - 1 END AS UINTEGER) AS c FROM {self._t(name)}"
                                     + (order_by if order_by else " ORDER BY rowid")).fetchnumpy()["c"]
            if base == "blob": vals = [bytes(v) for v in vals]
            if base == "row":                              # a row type (T051): values are lists of the fields' values
                import sqltypes as T
                rt = T.type_from_name(typ)
                return np.asarray(codes), ct.DictCodec([_row_value(v, rt) for v in vals], str(rt)), None
            return np.asarray(codes), ct.DictCodec(vals, {"str": "VARCHAR", "blob": "BLOB", "json": "JSON"}.get(base, "DOUBLE")), None
        raw = _raw(c, typ) if fv is None else _raw_expr(col, typ)
        lo, hi = st["min"].get(c), st["max"].get(c)
        if fv is not None:
            lo2, hi2 = self.con.execute(f"SELECT min({raw}), max({raw}) FROM {self._t(name)}").fetchone(); lo, hi = lo2, hi2
        lo, hi = (0, 0) if lo is None else (int(lo), int(hi))
        if 0 < lo and hi <= 0xFFFFFFFF and hi.bit_length() == (hi - lo).bit_length():
            lo = 0                       # the value itself: no narrower with an offset, and no offset terms in sums
        span = hi - lo
        if span >= 1 << 64: return None, None, "value range wider than 64 bits"
        if is_dim and span > 0xFFFFFFFF:
            # a dimension stays within 32-bit codes: rank the distinct values instead (order-preserving)
            vals = [int(r[0]) for r in self.con.execute(f"SELECT DISTINCT {raw} AS v FROM {self._t(name)} WHERE {raw} IS NOT NULL ORDER BY v").fetchall()]
            codes = self.con.execute(f"SELECT CAST(CASE WHEN {raw} IS NULL THEN 0 ELSE dense_rank() OVER (ORDER BY {raw} NULLS LAST) - 1 END AS UINTEGER) AS c FROM {self._t(name)}{order_by}").fetchnumpy()["c"]
            cd = ct.RankCodec(_codec(typ, 0), vals)
            return np.asarray(codes), cd, None
        target = "UINTEGER" if span <= 0xFFFFFFFF else "UBIGINT"
        inner = f"CAST({raw} AS BIGINT) - ({lo})" if -(1 << 62) < lo < (1 << 62) and -(1 << 62) < hi < (1 << 62) else f"CAST({raw} AS HUGEINT) - ({lo})"
        codes = self.con.execute(f"SELECT CAST(coalesce({inner}, 0) AS {target}) AS c FROM {self._t(name)}{order_by}").fetchnumpy()["c"]
        return np.asarray(codes), _codec(typ, lo), None

    def _encode_as(self, name, c, typ, st, key, fill):
        """codes of column c under another name (key) of the stats"""
        st = dict(st); st["min"] = dict(st["min"]); st["max"] = dict(st["max"])
        st["min"][c] = st["min"].get(key); st["max"][c] = st["max"].get(key)
        return self._encode(name, c, typ, st, "", False, fill)

    def _pk_rows(self, name, pk, typ, n):
        """how a key value finds its row: (kind, data). Rows are in key order, so row = rank of the key."""
        if _base_type(typ) in ("int", "date", "bool", "ts", "tstz"):
            r = _raw(pk, typ)
            k = self.con.execute(f"SELECT CAST({r} AS BIGINT) AS k FROM {self._t(name)} ORDER BY {_q(pk)}").fetchnumpy()["k"]
            return ("int", np.asarray(k, np.int64))
        return ("sql", None)

    def _design(self, name, t, pk, fk, n):
        attrs = {c: t.data[c].astype(np.int64) for c in t.cols if c not in fk and c != pk and not c.startswith("_") and c in t.codecs
                 and "#" not in c and c not in getattr(t, "nullable", {})}
        des = keydesign.design(n, attrs)
        des["surrogate"] = Surrogate(des["w"], des["s_of_row"][:n]) if des["kind"] == "derived" else None
        if des["surrogate"] is not None: des["surrogate"].coords = des.get("coords", {})
        return des

    def _fk_codes(self, name, c, typ, dname, dpk, dtyp, des, rowmap):
        """the stored foreign key: the referenced row's surrogate (or row); a missing / NULL reference gets the code
        of row n (no such row: it never joins)"""
        n = len(self.tables[dname].used)
        lut_final = des["s_of_row"] if des["kind"] == "derived" else np.arange(n + 1, dtype=np.int64)
        kind, keys = rowmap
        if kind == "int":
            lo, hi = (int(keys[0]), int(keys[-1])) if len(keys) else (0, -1)
            r = _raw(c, typ)
            if len(keys) and hi - lo <= 4 * n + 1024:
                L = hi - lo + 1
                lut = np.full(L + 1, lut_final[n], np.int64); lut[keys - lo] = lut_final[:n]
                idx = self.con.execute(f"SELECT CAST(CASE WHEN {r} BETWEEN {lo} AND {hi} THEN CAST({r} AS BIGINT) - ({lo}) ELSE {L} END AS UINTEGER) AS i "
                                       f"FROM {self._t(name)}").fetchnumpy()["i"]
                return algq.par_map(np.asarray(idx, np.uint32), lut)
            sent = (lo - 1) if lo > -(1 << 62) else (hi + 1)
            v = np.asarray(self.con.execute(f"SELECT coalesce(CAST({r} AS BIGINT), {sent}) AS v FROM {self._t(name)}").fetchnumpy()["v"], np.int64)
            pos = np.minimum(np.searchsorted(keys, v), max(0, len(keys) - 1))
            hit = (keys[pos] == v) if len(keys) else np.zeros(len(v), bool)
            rows = np.where(hit, pos, n)
            return lut_final[rows].astype(np.uint32)
        # other key types: matched in DuckDB, in the fact's row order
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE __torbles_rows AS SELECT {_q(dpk)} AS k, CAST(row_number() OVER (ORDER BY {_q(dpk)}) - 1 AS BIGINT) AS r FROM {self._t(dname)}")
        rows = self.con.execute(f"SELECT coalesce(m.r, {n}) AS r FROM {self._t(name)} f LEFT JOIN __torbles_rows m ON f.{_q(c)} = m.k ORDER BY f.rowid").fetchnumpy()["r"]
        self.con.execute("DROP TABLE __torbles_rows")
        return lut_final[np.asarray(rows, np.int64)].astype(np.uint32)

    # ------------------------------------------------------------------ use
    def __getitem__(self, name):
        return self._writable[name] if name in self._writable else self.tables[name]
    def __repr__(self):
        return "Database(" + ", ".join(f"{n}: {t.n:,} rows" for n, t in self.tables.items()) + ")"
    def __contains__(self, name): return name in self.tables or name in self._writable
    def __iter__(self): return iter((*self.tables, *self._writable))

    def query(self, fact):
        """a query over this database's tables (fact: a table or its name)"""
        return Query(self.tables[fact] if isinstance(fact, str) else fact)

    def table(self, name, alias=None):
        """a table for the Python SQL API: columns as typed expressions (see sqlapi; F holds the functions)"""
        import sqlapi
        return sqlapi.TableRef(self, name, alias)

    def select(self, *items):
        """a SELECT built in Python (sqlapi.Query): .where() .group_by() .having() .order_by() .limit() .run()"""
        import sqlapi
        return sqlapi.Query(self, items)

    def sql(self, text, engine="alg"):
        """Run SQL (ISO/IEC 9075 queries; DuckDB's syntax is accepted too) on the torbles. Returns a Relation
        (.columns, .rows, .fetch_columns(), .df()). Joins of a fact with its dimensions, filters and aggregates run on the bit planes;
        see docs/SQL.md for how each part of a query is evaluated and for the functions supported."""
        import sqlfront, sqlexec
        if not self._writable:
            out = None
            for q in sqlfront.bind(self, text):
                out = sqlexec.run(self, q, engine)
            return out
        from torbles import sqldml, sqldmlparse
        with self._sql_lock:
            out = None
            for statement in sqlfront.split_statements(text):
                if sqldmlparse.is_dml(statement):
                    out = sqldml.run(self, sqldmlparse.parse_dml(statement), engine)
                else:
                    for q in sqlfront.bind(self, statement):
                        out = sqlexec.run(self, q, engine)
            return out

    def sql_batch(self, text, engine="alg"):
        """Run a read-only SQL workload and return one Relation per statement."""
        from torbles import sqlworkload
        with self._sql_lock:
            return sqlworkload.run(self, text, engine)

    def sql_result(self, q):
        """run the query's SQL rendering on the source (DuckDB) and return it in torbles' result form"""
        if self.con is None:
            raise RuntimeError("no source to run the SQL on: pass con= (a DuckDB connection holding the source tables) to torbles.load")
        rows = self.con.execute(q.sql()).fetchall()
        na = len(q._sql_aggs())                       # the aggregate's columns, last in each row
        ext = getattr(q, "_ext", None) is not None
        def val(a):
            if ext: return None if a[0] is None and a[1] is None else algq.MinMax(a)     # MIN / MAX of nothing: None
            return tuple(a) if na > 1 else a[0]
        if q.keys: return {tuple(r[:-na]): val(r[-na:]) for r in rows}
        return val(rows[0][-na:]) if rows else None

    def verify(self, q, engine="alg"):
        """(equal, torbles result, source result): the query on torbles and its SQL on the source"""
        a, b = q.run(engine), self.sql_result(q)
        return a == b, a, b

    def describe(self):
        """a readable summary: tables, column types, the key design chosen for every dimension"""
        out = []
        for name, m in self.meta.items():
            t = self.tables[name]
            out.append(f"{name}: {m['rows']:,} rows" + (f", primary key {m['primary_key']}" if m["primary_key"] else "")
                       + (f", clustered by {m['cluster']}" if m.get("cluster") else "")
                       + (f", {len(t.parts)} row-aligned parts of {', '.join(str(len(p.cols)) for p in t.parts)} columns" if t.parts else "")
                       + (f", made of {m['generated_for']}'s columns" if m.get("generated_for") else ""))
            for c, i in m["columns"].items():
                w = t.width(c) if c in t.cols else 0
                extra = f" -> {i['references']}" if i.get("references") else ""
                if i.get("stored_in"): extra = f" stored once per {t.moved[c]}, in {i['stored_in']}"; w = self.tables[i["stored_in"]].width(c)
                out.append(f"    {c:28s} {i['type']:16s} {i['codec']:9s} {w:2d} bits{extra}")
            d = m.get("design")
            if d:
                if d["kind"] == "derived":
                    out.append(f"    key design: derived surrogate, {d['bits']} bits = " +
                               " | ".join(f"{k} {v}" for k, v in d["fields"].items()) + f" | row {d['w']}")
                    if d["covered"]: out.append(f"      attributes on the key's bits: {', '.join(d['covered'])}")
                else:
                    out.append(f"    key design: plain row number ({d['w']} bits): {d.get('reason', '')}")
        if self.notes: out += ["notes:"] + ["    " + x for x in self.notes]
        return "\n".join(out)

    def close(self):
        """release every table's native storage (planes or file mappings, sidecars, plain columns, GPU copies: using
        a table afterwards raises algq.TableClosedError) and the DuckDB connection, when this Database opened it.
        Closing again does nothing."""
        with self._sql_lock:
            for t in self._writable.values():
                t.close()
            self._writable.clear()
            for t in self.tables.values():
                t.close()
            if self._own and self.con is not None:
                self.con.close()
            self.con = None

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Write the database to a directory: one file per torble (its bit planes, memory-mapped on load) plus a
        manifest (schema, column types, key designs) and the small arrays (dictionaries, surrogates)."""
        with self._sql_lock:
            if self._writable:
                raise ValueError("persistence of writable tables is not supported")
            save(self, path)


# ---------------------------------------------------------------------- helpers
def _base_type(t):
    u = t.upper()
    if u in _INT_TYPES: return "int"
    if u.startswith("DECIMAL") or u.startswith("NUMERIC"): return "dec"
    if u == "DATE": return "date"
    if u in _TS_TYPES: return "ts"
    if u in _TSTZ_TYPES: return "tstz"
    if u == "BOOLEAN": return "bool"
    if u in ("VARCHAR", "TEXT", "STRING") or u.startswith("VARCHAR"): return "str"
    if u in ("BLOB", "BYTEA", "VARBINARY", "BINARY"): return "blob"
    if u == "JSON": return "json"
    if u.startswith(("STRUCT(", "ROW(")): return "row"
    if u in _FLOAT_TYPES: return "float"
    return "other"


def _supported(t):
    b = _base_type(t)
    if b == "row":
        import sqltypes as T
        try: rt = T.type_from_name(t)
        except T.SqlError as e: return False, str(e)
        def ok(x):
            if x.id == "ROW": return x.child is not None and all(ok(ft) for _, ft in x.child)
            return x.id not in ("LIST", "MULTISET", "INTERVAL", "NULL")
        return (True, None) if ok(rt) else (False, f"type {t}: a row's fields are stored when they are scalars or rows")
    if b != "other": return True, None
    u = t.upper()
    if u.startswith("TIMESTAMP"): return False, f"type {t}: use TIMESTAMP or TIMESTAMP WITH TIME ZONE (microseconds)"
    return False, f"type {t} is not supported"


def _raw(c, typ): return _raw_expr(_q(c), typ)


def _raw_expr(col, typ):
    """SQL for the column's exact integer representation (None: dictionary-coded types)"""
    b = _base_type(typ)
    if b == "int": return col
    if b == "bool": return f"CAST({col} AS INTEGER)"
    if b == "date": return f"({col} - DATE '1970-01-01')"
    if b == "ts": return f"epoch_us(CAST({col} AS TIMESTAMP))"
    if b == "tstz": return f"epoch_us({col})"
    if b == "dec":
        s = _dec_scale(typ)
        # |v| < 10^(p - s), so v * 10^s < 10^p <= 10^38: exact in DECIMAL(38, s) and in HUGEINT
        return f"CAST(CAST({col} AS DECIMAL(38, {s})) * {10 ** s} AS HUGEINT)" if s else f"CAST({col} AS HUGEINT)"
    return None


def _dec_scale(typ):
    m = re.search(r"\(\s*\d+\s*,\s*(\d+)\s*\)", typ)
    return int(m.group(1)) if m else 3


def _codec(typ, lo):
    b = _base_type(typ)
    if b == "int": return ct.IntCodec(lo, typ.upper())
    if b == "bool": return ct.BoolCodec() if lo == 0 else ct.IntCodec(lo, "BOOLEAN")
    if b == "date": return ct.DateCodec(lo)
    if b == "ts": return ct.TimestampCodec(lo)
    if b == "tstz": return ct.TimestampCodec(lo, zoned=True)
    if b == "dec": return ct.DecimalCodec(_dec_scale(typ), lo, typ.upper())
    raise ValueError(typ)


def _int_values(vals, typ):
    """distinct raw integers of a wide dimension column -> its values, for a dictionary codec"""
    b = _base_type(typ)
    if b == "dec":
        from decimal import Decimal
        s = _dec_scale(typ); return [Decimal((int(int(v) < 0), tuple(map(int, str(abs(int(v))))), -s)) for v in vals]
    if b == "date": return [ct.EPOCH + __import__("datetime").timedelta(days=v) for v in vals]
    if b == "ts": return [ct.EPOCH_TS + __import__("datetime").timedelta(microseconds=v) for v in vals]
    if b == "tstz":
        import datetime as _d
        return [(ct.EPOCH_TS + _d.timedelta(microseconds=v)).replace(tzinfo=_d.timezone.utc) for v in vals]
    return vals


def _row_value(v, t):
    """a DuckDB value of a STRUCT / LIST type as torbles holds it: a row as the list of its fields' values"""
    if v is None: return None
    if t.id == "ROW" and t.child is not None:
        if isinstance(v, dict): v = [v.get(n) for n, _ in t.child] if all(n for n, _ in t.child) else list(v.values())
        return tuple(_row_value(x, ft) for x, (_, ft) in zip(v, t.child))
    if t.id in ("LIST", "MULTISET"): return [_row_value(x, t.child) for x in v]
    if isinstance(v, (bytes, bytearray, memoryview)): return bytes(v)
    return v


def _enc_value(v, t):
    """a row value as JSON (saved dictionaries): scalars as text in their SQL form"""
    if v is None: return None
    if t.id == "ROW": return [_enc_value(x, ft) for x, (_, ft) in zip(v, t.child)]
    if t.id in ("LIST", "MULTISET"): return [_enc_value(x, t.child) for x in v]
    if isinstance(v, bool) or (isinstance(v, int) and t.integer): return v
    if t.id == "BLOB": return bytes(v).hex()
    import sqltypes as T
    return T.to_text(v, t)


def _dec_value(x, t):
    if x is None: return None
    if t.id == "ROW": return tuple(_dec_value(a, ft) for a, (_, ft) in zip(x, t.child))
    if t.id in ("LIST", "MULTISET"): return [_dec_value(a, t.child) for a in x]
    if isinstance(x, (bool, int)): return x
    if t.id == "BLOB": return bytes.fromhex(x)
    import sqltypes as T
    return T.cast_value(x, T.VARCHAR, t)


def _fill_literal(typ, v):
    b = _base_type(typ)
    if b in ("str", "json"): return "'" + str(v).replace("'", "''") + "'"
    if b == "blob": return f"X'{bytes(v).hex()}'" if isinstance(v, (bytes, bytearray)) else "'" + str(v).replace("'", "''") + "'::BLOB"
    if b == "date": return f"DATE '{v}'"
    if b == "ts": return f"TIMESTAMP '{v}'"
    if b == "tstz": return f"TIMESTAMPTZ '{v}'"
    if b == "bool": return "TRUE" if v else "FALSE"
    return f"CAST({v} AS {typ})"


def _name_match(col, pk, dim):
    """does a column's name say it references dim's key pk? (lo_custkey ~ c_custkey, customer_id ~ customer.id)"""
    c, p, d = col.lower(), pk.lower(), dim.lower()
    strip = lambda s: s.split("_", 1)[1] if "_" in s else s
    if c == p or strip(c) == strip(p) and len(strip(p)) > 2: return True
    for suf in ("_id", "id", "_key", "key", "_sk", "_fk"):
        if c == d + suf or c == d.rstrip("s") + suf: return p in ("id", "key", d + suf, d.rstrip("s") + suf) or strip(p) in ("id", "key")
    return False


def _topo(names, fks):
    """tables in build order: every referenced table before the tables that reference it"""
    done, out = set(), []
    def visit(n, path):
        if n in done: return
        if n in path: raise LoadError("foreign keys form a cycle: " + " -> ".join(path + [n]))
        for d, _ in fks.get(n, {}).values():
            if d != n: visit(d, path + [n])
        done.add(n); out.append(n)
    for n in names: visit(n, [])
    return out


def _design_json(d):
    if not d: return None
    keep = {k: v for k, v in d.items() if k in ("kind", "w", "chain", "fields", "covered", "bits", "dense", "reason")}
    return keep


def keys_from_ddl(ddl):
    """the keys a standard SQL schema declares (text or a .sql file): ({table: primary key column},
    {"table.column": "table.column"}). Composite keys are skipped (torbles joins on single columns)."""
    import duckdb
    if isinstance(ddl, (str, os.PathLike)) and str(ddl).strip().lower().endswith(".sql") and os.path.exists(str(ddl)):
        ddl = open(ddl).read()
    scratch = duckdb.connect(); scratch.execute(ddl)
    keys = Database(scratch)._declared_keys(); scratch.close()
    pks = {t: c[0] for t, typ, c, rt, rc in keys if typ == "PRIMARY KEY" and len(c) == 1}
    fks = {f"{t}.{c[0]}": (f"{rt}.{rc[0]}" if rc else rt) for t, typ, c, rt, rc in keys if typ == "FOREIGN KEY" and len(c) == 1}
    return pks, fks


def from_duckdb(source, tables=None, primary_keys=None, foreign_keys=None, columns=None, schema=None, **build_opts):
    """Build a Database from a DuckDB database (a path or an open connection). tables: names (default: every base
    table); schema: a standard SQL DDL script (text or .sql file) whose PRIMARY KEY / FOREIGN KEY declarations give the
    keys, for a database that does not declare them itself; primary_keys: {table: column}; foreign_keys:
    {"table.column": "table" or "table.column"}; columns: {table: [columns]}. Keys declared in the database are used."""
    import duckdb
    if schema is not None:
        sp, sf = keys_from_ddl(schema)
        primary_keys = {**sp, **(primary_keys or {})}; foreign_keys = {**sf, **(foreign_keys or {})}
    con = duckdb.connect(str(source), read_only=True) if isinstance(source, (str, os.PathLike)) else source
    db = Database(con); db._own = isinstance(source, (str, os.PathLike))
    if tables is None:
        tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
                                            "AND table_schema = current_schema() ORDER BY table_name").fetchall()]
    fkt = {}
    for k, v in (foreign_keys or {}).items():
        t, _, c = k.partition(".")
        if not c: raise ValueError(f"foreign key {k!r}: write it as 'table.column'")
        fkt.setdefault(t, {})[c] = v
    for t in tables:
        db.add_table(t, None, primary_key=(primary_keys or {}).get(t), foreign_keys=fkt.get(t), columns=(columns or {}).get(t))
    return db.build(**build_opts)


def from_frames(frames, primary_keys=None, foreign_keys=None, **build_opts):
    """Build a Database from {name: DataFrame / pyarrow Table / dict of arrays / file path}"""
    db = Database()
    fkt = {}
    for k, v in (foreign_keys or {}).items():
        t, _, c = k.partition("."); fkt.setdefault(t, {})[c] = v
    for name, src in frames.items():
        db.add_table(name, src, primary_key=(primary_keys or {}).get(name), foreign_keys=fkt.get(name))
    return db.build(**build_opts)


def from_ddl(ddl, data, **build_opts):
    """Build a Database from a standard SQL schema (CREATE TABLE statements with PRIMARY KEY / FOREIGN KEY ... REFERENCES,
    as text or a .sql file) and the data: {table: file path / DataFrame / pyarrow Table / dict of arrays}. Columns
    are cast to the declared types. The keys are read back through INFORMATION_SCHEMA; they are not enforced while
    loading (a reference without a match never joins)."""
    import duckdb
    if isinstance(ddl, (str, os.PathLike)) and str(ddl).strip().lower().endswith(".sql") and os.path.exists(str(ddl)):
        ddl = open(ddl).read()
    scratch = duckdb.connect()
    scratch.execute(ddl)                                  # parsed by DuckDB: the schema and its constraints
    db = Database()
    tmp = Database(scratch)
    keys = tmp._declared_keys()
    types = {}
    for t, c, typ in scratch.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                                     "WHERE table_schema = current_schema() ORDER BY table_name, ordinal_position").fetchall():
        types.setdefault(t, {})[c] = typ
    scratch.close()
    missing = [t for t in types if t not in data]
    if missing: raise LoadError(f"no data for table(s) {', '.join(missing)} declared in the schema")
    for t, cols_ in types.items():
        db.add_table(f"__src_{t}", data[t])
        have = {c for c, in db.con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ?", [f"__src_{t}"]).fetchall()}
        lost = [c for c in cols_ if c not in have]
        if lost: raise LoadError(f"{t}: the data has no column(s) {', '.join(lost)} declared in the schema")
        sel = ", ".join(f"CAST({_q(c)} AS {typ}) AS {_q(c)}" for c, typ in cols_.items())
        db.con.execute(f"CREATE TABLE {_q(t)} AS SELECT {sel} FROM {_q('__src_' + t)}")
        db.con.execute(f"DROP TABLE {_q('__src_' + t)}"); del db._add[f"__src_{t}"]
    pk = {t: c[0] for t, typ, c, rt, rc in keys if typ == "PRIMARY KEY" and len(c) == 1}
    fk = {}
    for t, typ, c, rt, rc in keys:
        if typ == "FOREIGN KEY" and len(c) == 1: fk.setdefault(t, {})[c[0]] = f"{rt}.{rc[0]}" if rc else rt
    for t in types:
        db.add_table(t, None, primary_key=pk.get(t), foreign_keys=fk.get(t))
    return db.build(**build_opts)


_FRICTIONLESS_TYPES = {"integer": "BIGINT", "string": "VARCHAR", "date": "DATE", "datetime": "TIMESTAMP",
                       "boolean": "BOOLEAN", "number": None, "year": "INTEGER"}


def from_frictionless(package, **build_opts):
    """Build a Database from a Frictionless Data Package (datapackage.json, a path or a dict): every tabular resource
    (CSV or Parquet) with its Table Schema; primaryKey and foreignKeys give the keys, field types are applied where
    they are exact (number keeps the type read from the file: DECIMAL or DOUBLE)."""
    if isinstance(package, (str, os.PathLike)):
        root = os.path.dirname(os.path.abspath(str(package)))
        with open(package) as f: package = json.load(f)
    else: root = os.getcwd()
    db = Database()
    fks = {}
    for res in package.get("resources", []):
        name, sch = res["name"], res.get("schema", {})
        path = res["path"] if isinstance(res["path"], str) else res["path"][0]
        if not os.path.isabs(path): path = os.path.join(root, path)
        db.add_table(f"__src_{name}", path)
        casts = []
        for fld in sch.get("fields", []):
            t = _FRICTIONLESS_TYPES.get(fld.get("type", "string"))
            casts.append(f"CAST({_q(fld['name'])} AS {t}) AS {_q(fld['name'])}" if t else _q(fld["name"]))
        sel = ", ".join(casts) if casts else "*"
        db.con.execute(f"CREATE TABLE {_q(name)} AS SELECT {sel} FROM {_q('__src_' + name)}")
        db.con.execute(f"DROP TABLE {_q('__src_' + name)}"); del db._add[f"__src_{name}"]
        pk = sch.get("primaryKey")
        if isinstance(pk, list): pk = pk[0] if len(pk) == 1 else None
        for fkd in sch.get("foreignKeys", []):
            f = fkd["fields"]; f = f if isinstance(f, list) else [f]
            ref = fkd["reference"]; rf = ref.get("fields"); rf = rf if isinstance(rf, list) else [rf]
            if len(f) == 1: fks.setdefault(name, {})[f[0]] = f"{ref.get('resource') or name}.{rf[0]}"
        db._add[name] = dict(primary_key=pk, foreign_keys=fks.get(name, {}), columns=None)
    for name, f in fks.items(): db._add[name]["foreign_keys"] = f
    return db.build(**build_opts)


# ---------------------------------------------------------------------- persistence
def save(db, path):
    import ctypes
    snapshot = None
    if getattr(db, "_sql_snapshot", None) is not None:
        from torbles import sqlsnapshot
        snapshot = sqlsnapshot.token(db)
    files = [] if snapshot is not None else None

    def save_torble(at, filename):
        if algq.lib.at_save(at, filename.encode()) != 0:
            raise OSError(f"cannot write {filename}")
        if files is not None:
            files.append(os.path.relpath(filename, path))

    def save_array(filename, array):
        np.save(filename, array)
        if files is not None:
            files.append(os.path.relpath(filename, path))

    os.makedirs(path, exist_ok=True)
    man = dict(format=FORMAT, version=algq.__version__, tables={}, notes=db.notes, order=list(db.tables))
    sgs = {}
    for name, t in db.tables.items():
        base = os.path.join(path, _fname(name))
        for suf, tb in ([("", t)] if t.parts is None else [(f".part{i}", p) for i, p in enumerate(t.parts)]):
            save_torble(tb.at, base + suf + ".tbl")
        arrays = {}
        side = {}
        for c, sc in getattr(t, "sidecars", {}).items():
            fn = base + "." + _fname(c) + ".side.tbl"
            save_torble(sc["side"], fn)
            arrays[f"rank:{c}"] = sc["rank"]; side[c] = dict(rows=int(sc["rows"]))
        codecs = {}
        for c, cd in t.codecs.items():
            codecs[c] = cd.to_json()
            if cd.kind == "rank":
                man.setdefault("intdicts", {}).setdefault(name, {})[f"rank:{c}"] = [str(x) for x in cd.raw]
            if cd.kind == "dict":
                if cd.numeric: arrays[f"dict:{c}"] = np.asarray([float(v) if not isinstance(v, int) else v for v in cd.values], object if any(isinstance(v, int) for v in cd.values) else np.float64)
                else:
                    vs = [v.decode("latin-1") for v in cd.values] if cd.sql_type == "BLOB" else cd.values
                    if cd.sql_type.startswith("ROW"):
                        import sqltypes as T
                        rt = T.type_from_name(cd.sql_type); vs = [json.dumps(_enc_value(v, rt)) for v in cd.values]
                    blob, offs = ct.pack_strings(vs); arrays[f"dictblob:{c}"] = blob; arrays[f"dictoffs:{c}"] = offs
        refs = {}
        for fk, bit in t.refbit.items():
            sg = t.surr.get(fk); dim = t._refdim[fk]
            sid = None
            if sg is not None:
                sid = sgs.get(id(sg))
                if sid is None:
                    sid = sgs[id(sg)] = f"sg{len(sgs)}"
                    save_array(os.path.join(path, sid + ".npy"), np.asarray(sg.s_of_row, np.int64))
                    if sg.stale: raise ValueError("save: a surrogate has stale points (moved rows); compact the fact first")
            refs[fk] = dict(dim=dim.name, bit=bit, surrogate=sid, w=None if sg is None else sg.w,
                            coords=None if sg is None else {k: [int(x) for x in v] for k, v in getattr(sg, "coords", {}).items()})
        if not t.used.all(): arrays["used"] = np.packbits(t.used, bitorder="little")
        for k, v in arrays.items():
            if v.dtype == object:
                vals = [int(x) for x in v]; man.setdefault("intdicts", {}).setdefault(name, {})[k] = [str(x) for x in vals]
            else: save_array(base + "." + _fname(k) + ".npy", v)
        m = db.meta.get(name, {})
        man["tables"][name] = dict(
            file=_fname(name) + ".tbl" if t.parts is None else None, rows=int(t.n), cols=t.cols, msb=sorted(t.msb_cols), cluster=t.cluster,
            parts=None if t.parts is None else [dict(file=_fname(name) + f".part{i}.tbl", cols=p.cols, msb=sorted(p.msb_cols))
                                                for i, p in enumerate(t.parts)],
            cluster_breaks=getattr(t, "cluster_breaks", None), cluster_sorted=getattr(t, "cluster_sorted", None),
            colmax={c: int(t.colmax(c)) for c in t.cols if c in t._colmax or c in t.data},
            wmax=getattr(t, "_wmax", {}), wide={c: list(v) for c, v in getattr(t, "wide", {}).items()}, sidecars=side,
            codecs=codecs, refs=refs, pk=getattr(t, "pk", None), orphans=getattr(t, "orphans", {}),
            nullable=getattr(t, "nullable", {}), fk_nulls=getattr(t, "fk_nulls", {}), col_order=getattr(t, "col_order", None),
            moved=getattr(t, "moved", {}), groups=getattr(t, "groups", {}), generated=getattr(t, "generated", None),
            used_all=bool(t.used.all()), meta=m, dense_dim=any(t is ot._refdim.get(f) for ot in db.tables.values() for f in getattr(ot, "_refdim", {})))
        man["tables"][name]["meta"] = json.loads(json.dumps(m, default=str))
    if snapshot is not None:
        record = sqlsnapshot.save_record(db, path, man, snapshot, files)
        if record is not None:
            man["sql_snapshot"] = record
    with open(os.path.join(path, "manifest.json.tmp"), "w") as f: json.dump(man, f, indent=1, default=str)
    os.replace(os.path.join(path, "manifest.json.tmp"), os.path.join(path, "manifest.json"))


def _fname(s): return re.sub(r"[^A-Za-z0-9_.-]", lambda m: f"%{ord(m.group(0)):02x}", str(s))


def load(path, validate=True, con=None):
    """Open a saved database. The bit planes are memory-mapped (privately: nothing is written back); with validate,
    every torus face is checked on load and single-bit faults are repaired in memory (the report is in db.integrity).
    con: a DuckDB connection holding the source tables, for sql_result() / verify() (optional)."""
    with open(os.path.join(path, "manifest.json")) as f: man = json.load(f)
    if man.get("format") != FORMAT: raise LoadError(f"{path}: format {man.get('format')}, this torbles reads {FORMAT}")
    db = Database.__new__(Database)
    db._writable, db._sql_lock = {}, RLock()
    db._own, db.con, db._add, db.tables, db.meta, db.notes, db.integrity = False, con, {}, {}, {}, man.get("notes", []), {}
    try:
        before = None
        if validate and "sql_snapshot" in man:
            from torbles import sqlsnapshot
            before = sqlsnapshot.load_start(path, man, validate)
        files = [] if before is not None else None
        _load_tables(db, path, man, validate, files)
        if files is not None:
            sqlsnapshot.load_finish(db, path, man, files, before)
        return db
    except BaseException:
        db.close()                                # what was mapped so far is released now, not with the traceback
        raise


def _load_tables(db, path, man, validate, files=None):
    import ctypes
    sgs = {}
    err = ctypes.c_int(0)

    def load_torble(filename):
        if files is not None:
            files.append(os.path.relpath(filename, path))
        return algq.lib.at_load(filename.encode(), ctypes.byref(err))

    def load_array(filename):
        if files is not None:
            files.append(os.path.relpath(filename, path))
        return np.load(filename)

    for name in man["order"]:
        m = man["tables"][name]; base = os.path.join(path, _fname(name))
        t = Table.__new__(Table); t.name = name
        parts = []                                 # a table wider than one torble: its parts, one file each
        try:
            for i, pm in enumerate(m.get("parts") or [dict(file=m["file"], cols=m["cols"], msb=m["msb"])]):
                at = load_torble(os.path.join(path, pm["file"]))
                if not at: raise LoadError(f"{name}: cannot load {pm['file']}: " + {1: "cannot open or map the file", 2: "not a torble file",
                                           3: "written by an incompatible version", 4: "header damaged", 5: "file truncated"}.get(err.value, str(err.value)))
                h = t if not m.get("parts") else Table.__new__(Table)
                if h is not t: h.name = f"{name}#{i}"
                h._own(at)                         # the mapping is the table's from here on (unmapped with it)
                h.n, h.cols, h.msb_cols, h._vflag = int(m["rows"]), list(pm["cols"]), set(pm["msb"]), {}
                parts.append(h)
        except BaseException:
            for h in parts: h.close()
            raise
        if m.get("parts"):
            for h in parts:
                h.cluster, h.data, h._colmax, h._wmax, h.wide, h._decoded, h._used_ids = None, {}, {}, {}, {}, {}, None
                h.sidecars, h.codecs, h.refbit, h.surr, h._refdim, h.nullable = {}, {}, {}, {}, {}, {}
            t._set_parts(parts)
        db.tables[name] = t                        # (and a failed load closes it with the others)
        t.n, t.cols, t._vflag = int(m["rows"]), list(m["cols"]), {}
        t.msb_cols = set(m["msb"]); t.cluster = m["cluster"]; t.data = {}
        if m["cluster"] is not None: t.cluster_breaks, t.cluster_sorted = m["cluster_breaks"], m["cluster_sorted"]
        t._colmax = {c: int(v) for c, v in m["colmax"].items()}; t._wmax = {c: int(v) for c, v in m["wmax"].items()}
        t.wide = {c: tuple(v) for c, v in m["wide"].items()}
        t._decoded, t._used_ids = {}, None
        if m["used_all"]: t.used = np.ones(t.n, bool)
        else: t.used = np.unpackbits(load_array(base + ".used.npy"), bitorder="little")[:t.n].astype(bool)
        bits = np.zeros(((t.n + 63) // 64 + 1) * 64, np.uint8); bits[:t.n] = t.used
        t.used_bm = np.packbits(bits, bitorder="little").view(np.uint64)
        for h in (t.parts or ()): h.used, h.used_bm = t.used, t.used_bm
        t.sidecars = {}
        for c, sc in m["sidecars"].items():
            side = load_torble(base + "." + _fname(c) + ".side.tbl")
            if not side: raise LoadError(f"{name}.{c}: cannot load its sidecar")
            h = t._part_of(c)                      # the torble that holds the column: the table or one of its parts
            h._own_side(side)
            rank = load_array(base + "." + _fname(f"rank:{c}") + ".npy")
            flagcol = t.wide[c[:-len("#hi")]][2] if c.endswith("#hi") else None
            algq.lib.at_set_virtual(h.at, h.cols.index(c), h.cols.index(flagcol), side, rank.ctypes.data)
            h.sidecars[c] = t.sidecars[c] = dict(side=side, rank=rank, rows=sc["rows"], body=None)
            h._vflag[c] = t._vflag[c] = flagcol
        # codecs
        t.codecs = {}
        for c, j in m["codecs"].items():
            if j["kind"] == "ref": t.codecs[c] = RefCodec(j["dim"], j["pk"]); continue
            vals = None
            if j["kind"] == "rank": vals = [int(x) for x in man["intdicts"][name][f"rank:{c}"]]
            if j["kind"] == "dict":
                fb = base + "." + _fname(f"dictblob:{c}") + ".npy"
                if os.path.exists(fb):
                    vals = ct.unpack_strings(load_array(fb), load_array(base + "." + _fname(f"dictoffs:{c}") + ".npy"))
                    if j.get("sql_type") == "BLOB": vals = [v.encode("latin-1") for v in vals]
                    if (j.get("sql_type") or "").startswith("ROW"):
                        import sqltypes as T
                        rt = T.type_from_name(j["sql_type"]); vals = [_dec_value(json.loads(v), rt) for v in vals]
                elif f"dict:{c}" in man.get("intdicts", {}).get(name, {}):
                    vals = [int(x) for x in man["intdicts"][name][f"dict:{c}"]]
                    vals = _int_values(vals, j.get("sql_type", "BIGINT"))
                else: vals = load_array(base + "." + _fname(f"dict:{c}") + ".npy").tolist()
            t.codecs[c] = ct.from_json(j, vals)
        t.pk, t.sqlname = m["pk"], name
        # references
        t.refbit, t.surr, t._refdim = {}, {}, {}
        for fk, r in m["refs"].items():
            dim = db.tables[r["dim"]]
            sg = None
            if r["surrogate"] is not None:
                sg = sgs.get(r["surrogate"])
                if sg is None:
                    sg = sgs[r["surrogate"]] = Surrogate(r["w"], load_array(os.path.join(path, r["surrogate"] + ".npy")))
                    sg.coords = r.get("coords") or {}
            t.refbit[fk], t.surr[fk], t._refdim[fk] = int(r["bit"]), sg, dim
        t.orphans = {k: int(v) for k, v in m.get("orphans", {}).items()}
        t.nullable = dict(m.get("nullable", {})); t.fk_nulls = {k: int(v) for k, v in m.get("fk_nulls", {}).items()}
        if m.get("col_order"): t.col_order = list(m["col_order"])
        t.moved, t.groups = dict(m.get("moved") or {}), dict(m.get("groups") or {})
        if m.get("generated"): t.generated = tuple(m["generated"])
        if validate:                              # faces first: nothing is decoded from unchecked planes
            rep = dict(rows_with_faults=0, repaired=0, unrepaired=0)
            for part, at_ in ([("", t.at)] if t.parts is None else [(f"part{i}", h.at) for i, h in enumerate(t.parts)]) \
                             + [(c, sc["side"]) for c, sc in t.sidecars.items()]:
                nbad = algq.lib.at_validate(at_, None, 0)
                fixed = algq.lib.at_repair(at_) if nbad else 0
                left = algq.lib.at_validate(at_, None, 0) if nbad else 0
                rep["rows_with_faults"] += int(nbad); rep["repaired"] += int(fixed); rep["unrepaired"] += int(left)
                if left: raise LoadError(f"{name}{'.' + part if part else ''}: {left} row(s) fail the torus face checks and could "
                                         "not be repaired (more than one bit lost)")
            db.integrity[name] = rep
        if m.get("dense_dim"):                    # a referenced table: its plain columns serve the planner (small)
            for c in t.cols:
                if c not in t.sidecars: t.data[c] = t.decoded(c)
        elif t.cluster is not None and t.cluster not in t.sidecars:
            t.data[t.cluster] = t.decoded(t.cluster)
        db.meta[name] = m.get("meta", {})
    return db
