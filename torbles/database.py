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
match (or NULL) never joins, as in an SQL inner join. NULLs elsewhere are refused (fill_nulls= replaces them)."""
import json, os, re, time
import numpy as np
import algq
from algq import Table, Query, Surrogate
import coltypes as ct
import keydesign

FORMAT = 1
_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "HUGEINT", "UHUGEINT"}
_TS_TYPES = {"TIMESTAMP", "TIMESTAMP_S", "TIMESTAMP_MS"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}


def _q(name): return '"' + str(name).replace('"', '""') + '"'


class LoadError(ValueError):
    pass


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

    # ------------------------------------------------------------------ sources
    def add_table(self, name, source=None, primary_key=None, foreign_keys=None, columns=None):
        """source: a pandas / polars DataFrame, a pyarrow Table, a dict of arrays, a DuckDB relation, a file path
        (.parquet, .csv, .tsv, .json, globs allowed), or None for a table already in the connection.
        foreign_keys: {column: "table" or "table.column"}; columns: the columns to load (default: all)."""
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
    def build(self, infer_keys=False, fill_nulls=None, keep_plain=False, threads=None, verbose=False, types=None):
        """Read every added table, design the keys, build the torbles. Returns self.
        types: {"table.column": "SQL type"} loads a column as another type (e.g. a DOUBLE amount as DECIMAL(18, 2), to
        sum it exactly); fill_nulls: {"table.column": value}; infer_keys: guess undeclared keys (reported in notes)."""
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
            data, extend, codecs, info = {}, {}, {}, {}
            order_by = f" ORDER BY {_q(pk)}" if name in dims else ""
            for c, typ in cols[name].items():
                if c in fk: continue
                codes, cd, why = self._encode(name, c, typ, st, order_by, name in dims, fill)
                if cd is None: self.notes.append(f"{name}.{c}: not loaded ({why})"); continue
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
            if not data: raise LoadError(f"{name}: no column could be loaded")
            if len(data) + bool(refs) > algq.MAXCOL:
                raise LoadError(f"{name}: {len(data)} columns; a torble holds {algq.MAXCOL - 1} (with foreign keys) - "
                                f"choose the columns to load with add_table(..., columns=[...])")
            cluster = next((c for c in list(fk) + [c for c in data if c not in fk]
                            if c not in extend and n > 1 and bool(np.all(data[c][1:] >= data[c][:-1]))), None)
            t = Table(name, data, msb=tuple(fk), refs=refs or None, extend=extend or None, cluster=cluster)
            t.codecs, t.pk, t.sqlname = codecs, pk, name
            self.tables[name] = t
            if name in dims:
                rowmaps[name] = self._pk_rows(name, pk, cols[name][pk], n)
                designs[name] = self._design(name, t, pk, fk, n)
            if name not in dims and not keep_plain: t.drop_plain()
            self.meta[name] = dict(rows=n, primary_key=pk, foreign_keys={c: f"{d}.{p}" for c, (d, p) in fk.items()},
                                   columns=info, cluster=cluster, design=_design_json(designs.get(name)),
                                   build_s=round(time.perf_counter() - tb, 3))
            if verbose: print(f"  {name}: {n:,} rows, {len(data)} columns, {time.perf_counter() - tb:.2f} s", flush=True)
        self.build_s = time.perf_counter() - t0
        self._designs = designs
        return self

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
            if nulls and c not in fk and f"{name}.{c}" not in fill and c not in fill:
                if _supported(typ)[0]:
                    raise LoadError(f"{name}.{c} has {nulls:,} NULL value(s): NULLs are not supported yet. Pass "
                                    f"fill_nulls={{'{name}.{c}': value}} or leave the column out (columns=[...])")
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
        if base in ("str", "float"):
            if base == "float":
                nan = self.con.execute(f"SELECT count(*) FROM {self._t(name)} WHERE isnan({_q(c)}) OR isinf({_q(c)})").fetchone()[0]
                if nan: return None, None, f"{nan} NaN / infinite value(s)"
            vals = [r[0] for r in self.con.execute(f"SELECT DISTINCT {col} AS v FROM {self._t(name)} ORDER BY v").fetchall()]
            if base == "float": vals = [float(v) for v in vals]
            codes = self.con.execute(f"SELECT CAST(dense_rank() OVER (ORDER BY {col}) - 1 AS UINTEGER) AS c FROM {self._t(name)}"
                                     + (order_by if order_by else " ORDER BY rowid")).fetchnumpy()["c"]
            return np.asarray(codes), ct.DictCodec(vals, "VARCHAR" if base == "str" else "DOUBLE"), None
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
            vals = [int(r[0]) for r in self.con.execute(f"SELECT DISTINCT {raw} AS v FROM {self._t(name)} ORDER BY v").fetchall()]
            codes = self.con.execute(f"SELECT CAST(dense_rank() OVER (ORDER BY {raw}) - 1 AS UINTEGER) AS c FROM {self._t(name)}{order_by}").fetchnumpy()["c"]
            cd = ct.RankCodec(_codec(typ, 0), vals)
            return np.asarray(codes), cd, None
        target = "UINTEGER" if span <= 0xFFFFFFFF else "UBIGINT"
        inner = f"CAST({raw} AS BIGINT) - ({lo})" if -(1 << 62) < lo < (1 << 62) and -(1 << 62) < hi < (1 << 62) else f"CAST({raw} AS HUGEINT) - ({lo})"
        codes = self.con.execute(f"SELECT CAST({inner} AS {target}) AS c FROM {self._t(name)}{order_by}").fetchnumpy()["c"]
        return np.asarray(codes), _codec(typ, lo), None

    def _pk_rows(self, name, pk, typ, n):
        """how a key value finds its row: (kind, data). Rows are in key order, so row = rank of the key."""
        if _base_type(typ) in ("int", "date", "bool", "ts"):
            r = _raw(pk, typ)
            k = self.con.execute(f"SELECT CAST({r} AS BIGINT) AS k FROM {self._t(name)} ORDER BY {_q(pk)}").fetchnumpy()["k"]
            return ("int", np.asarray(k, np.int64))
        return ("sql", None)

    def _design(self, name, t, pk, fk, n):
        attrs = {c: t.data[c].astype(np.int64) for c in t.cols if c not in fk and c != pk and not c.startswith("_") and c in t.codecs}
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
    def __getitem__(self, name): return self.tables[name]
    def __repr__(self):
        return "Database(" + ", ".join(f"{n}: {t.n:,} rows" for n, t in self.tables.items()) + ")"
    def __contains__(self, name): return name in self.tables
    def __iter__(self): return iter(self.tables)

    def query(self, fact):
        """a query over this database's tables (fact: a table or its name)"""
        return Query(self.tables[fact] if isinstance(fact, str) else fact)

    def sql_result(self, q):
        """run the query's SQL rendering on the source (DuckDB) and return it in torbles' result form"""
        if self.con is None:
            raise RuntimeError("no source to run the SQL on: pass con= (a DuckDB connection holding the source tables) to torbles.load")
        rows = self.con.execute(q.sql()).fetchall()
        if q.keys: return {tuple(r[:-1]): r[-1] for r in rows}
        return rows[0][0] if rows else None

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
                       + (f", clustered by {m['cluster']}" if m.get("cluster") else ""))
            for c, i in m["columns"].items():
                w = t.width(c) if c in t.cols else 0
                extra = f" -> {i['references']}" if i.get("references") else ""
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
        if self._own and self.con is not None: self.con.close()
        self.con = None

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Write the database to a directory: one file per torble (its bit planes, memory-mapped on load) plus a
        manifest (schema, column types, key designs) and the small arrays (dictionaries, surrogates)."""
        save(self, path)


# ---------------------------------------------------------------------- helpers
def _base_type(t):
    u = t.upper()
    if u in _INT_TYPES: return "int"
    if u.startswith("DECIMAL") or u.startswith("NUMERIC"): return "dec"
    if u == "DATE": return "date"
    if u in _TS_TYPES: return "ts"
    if u == "BOOLEAN": return "bool"
    if u in ("VARCHAR", "TEXT", "STRING") or u.startswith("VARCHAR"): return "str"
    if u in _FLOAT_TYPES: return "float"
    return "other"


def _supported(t):
    b = _base_type(t)
    if b != "other": return True, None
    u = t.upper()
    if u.startswith("TIMESTAMP"): return False, f"type {t}: use TIMESTAMP (microseconds, no time zone)"
    return False, f"type {t} is not supported"


def _raw(c, typ): return _raw_expr(_q(c), typ)


def _raw_expr(col, typ):
    """SQL for the column's exact integer representation (None: dictionary-coded types)"""
    b = _base_type(typ)
    if b == "int": return col
    if b == "bool": return f"CAST({col} AS INTEGER)"
    if b == "date": return f"({col} - DATE '1970-01-01')"
    if b == "ts": return f"epoch_us(CAST({col} AS TIMESTAMP))"
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
    return vals


def _fill_literal(typ, v):
    b = _base_type(typ)
    if b == "str": return "'" + str(v).replace("'", "''") + "'"
    if b == "date": return f"DATE '{v}'"
    if b == "ts": return f"TIMESTAMP '{v}'"
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
    os.makedirs(path, exist_ok=True)
    man = dict(format=FORMAT, version=algq.__version__, tables={}, notes=db.notes, order=list(db.tables))
    sgs = {}
    for name, t in db.tables.items():
        base = os.path.join(path, _fname(name))
        if algq.lib.at_save(t.at, (base + ".tbl").encode()) != 0: raise OSError(f"cannot write {base}.tbl")
        arrays = {}
        side = {}
        for c, sc in getattr(t, "sidecars", {}).items():
            fn = base + "." + _fname(c) + ".side.tbl"
            if algq.lib.at_save(sc["side"], fn.encode()) != 0: raise OSError(f"cannot write {fn}")
            arrays[f"rank:{c}"] = sc["rank"]; side[c] = dict(rows=int(sc["rows"]))
        codecs = {}
        for c, cd in t.codecs.items():
            codecs[c] = cd.to_json()
            if cd.kind == "rank":
                man.setdefault("intdicts", {}).setdefault(name, {})[f"rank:{c}"] = [str(x) for x in cd.raw]
            if cd.kind == "dict":
                if cd.numeric: arrays[f"dict:{c}"] = np.asarray([float(v) if not isinstance(v, int) else v for v in cd.values], object if any(isinstance(v, int) for v in cd.values) else np.float64)
                else:
                    blob, offs = ct.pack_strings(cd.values); arrays[f"dictblob:{c}"] = blob; arrays[f"dictoffs:{c}"] = offs
        refs = {}
        for fk, bit in t.refbit.items():
            sg = t.surr.get(fk); dim = t._refdim[fk]
            sid = None
            if sg is not None:
                sid = sgs.get(id(sg))
                if sid is None:
                    sid = sgs[id(sg)] = f"sg{len(sgs)}"
                    np.save(os.path.join(path, sid + ".npy"), np.asarray(sg.s_of_row, np.int64))
                    if sg.stale: raise ValueError("save: a surrogate has stale points (moved rows); compact the fact first")
            refs[fk] = dict(dim=dim.name, bit=bit, surrogate=sid, w=None if sg is None else sg.w,
                            coords=None if sg is None else {k: [int(x) for x in v] for k, v in getattr(sg, "coords", {}).items()})
        if not t.used.all(): arrays["used"] = np.packbits(t.used, bitorder="little")
        for k, v in arrays.items():
            if v.dtype == object:
                vals = [int(x) for x in v]; man.setdefault("intdicts", {}).setdefault(name, {})[k] = [str(x) for x in vals]
            else: np.save(base + "." + _fname(k) + ".npy", v)
        m = db.meta.get(name, {})
        man["tables"][name] = dict(
            file=_fname(name) + ".tbl", rows=int(t.n), cols=t.cols, msb=sorted(t.msb_cols), cluster=t.cluster,
            cluster_breaks=getattr(t, "cluster_breaks", None), cluster_sorted=getattr(t, "cluster_sorted", None),
            colmax={c: int(t.colmax(c)) for c in t.cols if c in t._colmax or c in t.data},
            wmax=getattr(t, "_wmax", {}), wide={c: list(v) for c, v in getattr(t, "wide", {}).items()}, sidecars=side,
            codecs=codecs, refs=refs, pk=getattr(t, "pk", None), orphans=getattr(t, "orphans", {}),
            used_all=bool(t.used.all()), meta=m, dense_dim=any(t is ot._refdim.get(f) for ot in db.tables.values() for f in getattr(ot, "_refdim", {})))
        man["tables"][name]["meta"] = json.loads(json.dumps(m, default=str))
    with open(os.path.join(path, "manifest.json.tmp"), "w") as f: json.dump(man, f, indent=1, default=str)
    os.replace(os.path.join(path, "manifest.json.tmp"), os.path.join(path, "manifest.json"))


def _fname(s): return re.sub(r"[^A-Za-z0-9_.-]", lambda m: f"%{ord(m.group(0)):02x}", str(s))


def load(path, validate=True, con=None):
    """Open a saved database. The bit planes are memory-mapped (privately: nothing is written back); with validate,
    every torus face is checked on load and single-bit faults are repaired in memory (the report is in db.integrity).
    con: a DuckDB connection holding the source tables, for sql_result() / verify() (optional)."""
    import ctypes
    with open(os.path.join(path, "manifest.json")) as f: man = json.load(f)
    if man.get("format") != FORMAT: raise LoadError(f"{path}: format {man.get('format')}, this torbles reads {FORMAT}")
    db = Database.__new__(Database)
    db._own, db.con, db._add, db.tables, db.meta, db.notes, db.integrity = False, con, {}, {}, {}, man.get("notes", []), {}
    sgs = {}
    err = ctypes.c_int(0)
    for name in man["order"]:
        m = man["tables"][name]; base = os.path.join(path, _fname(name))
        at = algq.lib.at_load(os.path.join(path, m["file"]).encode(), ctypes.byref(err))
        if not at: raise LoadError(f"{name}: cannot load {m['file']}: " + {1: "cannot open or map the file", 2: "not a torble file",
                                   3: "written by an incompatible version", 4: "header damaged", 5: "file truncated"}.get(err.value, str(err.value)))
        t = Table.__new__(Table)
        t.name, t.at, t.pt, t.n, t.cols = name, at, None, int(m["rows"]), list(m["cols"])
        t.msb_cols = set(m["msb"]); t.cluster = m["cluster"]; t.data = {}
        if m["cluster"] is not None: t.cluster_breaks, t.cluster_sorted = m["cluster_breaks"], m["cluster_sorted"]
        t._colmax = {c: int(v) for c, v in m["colmax"].items()}; t._wmax = {c: int(v) for c, v in m["wmax"].items()}
        t.wide = {c: tuple(v) for c, v in m["wide"].items()}
        t._decoded, t._used_ids = {}, None
        if m["used_all"]: t.used = np.ones(t.n, bool)
        else: t.used = np.unpackbits(np.load(base + ".used.npy"), bitorder="little")[:t.n].astype(bool)
        bits = np.zeros(((t.n + 63) // 64 + 1) * 64, np.uint8); bits[:t.n] = t.used
        t.used_bm = np.packbits(bits, bitorder="little").view(np.uint64)
        t.sidecars = {}
        for c, sc in m["sidecars"].items():
            side = algq.lib.at_load((base + "." + _fname(c) + ".side.tbl").encode(), ctypes.byref(err))
            if not side: raise LoadError(f"{name}.{c}: cannot load its sidecar")
            rank = np.load(base + "." + _fname(f"rank:{c}") + ".npy")
            flagcol = t.wide[c.split("#")[0]][2] if c.endswith("#hi") else None
            algq.lib.at_set_virtual(t.at, t.cols.index(c), t.cols.index(flagcol), side, rank.ctypes.data)
            t.sidecars[c] = dict(side=side, rank=rank, rows=sc["rows"], body=None)
        # codecs
        t.codecs = {}
        for c, j in m["codecs"].items():
            if j["kind"] == "ref": t.codecs[c] = RefCodec(j["dim"], j["pk"]); continue
            vals = None
            if j["kind"] == "rank": vals = [int(x) for x in man["intdicts"][name][f"rank:{c}"]]
            if j["kind"] == "dict":
                fb = base + "." + _fname(f"dictblob:{c}") + ".npy"
                if os.path.exists(fb): vals = ct.unpack_strings(np.load(fb), np.load(base + "." + _fname(f"dictoffs:{c}") + ".npy"))
                elif f"dict:{c}" in man.get("intdicts", {}).get(name, {}):
                    vals = [int(x) for x in man["intdicts"][name][f"dict:{c}"]]
                    vals = _int_values(vals, j.get("sql_type", "BIGINT"))
                else: vals = np.load(base + "." + _fname(f"dict:{c}") + ".npy").tolist()
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
                    sg = sgs[r["surrogate"]] = Surrogate(r["w"], np.load(os.path.join(path, r["surrogate"] + ".npy")))
                    sg.coords = r.get("coords") or {}
            t.refbit[fk], t.surr[fk], t._refdim[fk] = int(r["bit"]), sg, dim
        t.orphans = {k: int(v) for k, v in m.get("orphans", {}).items()}
        if validate:                              # faces first: nothing is decoded from unchecked planes
            rep = dict(rows_with_faults=0, repaired=0, unrepaired=0)
            for part, at_ in [("", t.at)] + [(c, sc["side"]) for c, sc in t.sidecars.items()]:
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
        db.tables[name] = t
        db.meta[name] = m.get("meta", {})
    return db
