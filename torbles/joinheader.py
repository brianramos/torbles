# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Bounded exact join paths in the database manifest."""

from collections.abc import MutableMapping
import hashlib
import json
from pathlib import Path
import struct
import weakref

import keydesign
import numpy as np

from .sqlsnapshot import _revision


HEADER = "join_hierarchy"
_MAX_HOPS = 3
_MAX_ROWS = 1 << 20
_MAX_BITS = 12
_MAX_ATTRS = 12
_MAX_RECORDS = 64
_MAX_HEADER_BYTES = 2 << 20
_MAX_TEXT_BYTES = 256
_TBL_HEADER = 584
_TBL_HEADER_LENGTH = 560


class LazyColumns(MutableMapping):
    """A referenced table's normal data mapping, decoding only columns used."""

    def __init__(self, table, names):
        self._table = weakref.ref(table)
        self._columns = dict.fromkeys(names)

    def __getitem__(self, key):
        value = self._columns[key]
        if value is None:
            value = self._columns[key] = self._table().decoded(key)
        return value

    def __setitem__(self, key, value):
        self._columns[key] = value

    def __delitem__(self, key):
        del self._columns[key]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __contains__(self, key):
        return key in self._columns


def _link(db, table, fk):
    target = table._refdim.get(fk)
    codec = table.codecs.get(fk)
    bit = table.refbit.get(fk)
    if (target is None or db.tables.get(target.name) is not target or not target.pk
            or getattr(target, "generated", None) or getattr(codec, "kind", None) != "ref"
            or codec.dim != target.name or codec.pk != target.pk
            or type(bit) is not int or not 0 <= bit < 32 or "_refok" not in table.cols):
        return None
    sg = table.surr.get(fk)
    if sg is not None and (sg.stale or not 0 < sg.w <= 32 or sg.mask != (1 << sg.w) - 1
                           or len(sg.s_of_row) != target.n):
        return None
    return target, sg, bit


def _cells(target, sg):
    if sg is None or not 0 < target.n <= _MAX_ROWS:
        return None
    rows = np.flatnonzero(target.used)
    if not len(rows):
        return None
    ids = sg.s_of_row[rows]
    if ids.min() < 0 or ids.max() >= 1 << 32 or not np.array_equal(ids & sg.mask, rows):
        return None
    bits = list(range(int(ids.max()).bit_length() - 1, sg.w - 1, -1))
    if not 0 < len(bits) <= _MAX_BITS:
        return None
    order = np.argsort(ids >> sg.w, kind="stable")
    cells = (ids >> sg.w)[order]
    starts = np.r_[0, np.flatnonzero(cells[1:] != cells[:-1]) + 1]
    return bits, cells[starts].tolist(), rows[order], starts


def _raw(table, name, cache):
    key = table.name, name
    if key not in cache:
        # Read the current physical codes, without any planner decode cache.
        cache[key] = table._part_of(name)._decode_column(name)
    return cache[key]


def _pull(source, fk, target, sg, bit, rows, cache):
    ok = _raw(source, "_refok", cache)[rows]
    if np.any(((ok >> bit) & 1) == 0):
        return None
    codes = _raw(source, fk, cache)[rows]
    ids = codes if sg is None else codes & np.uint32(sg.mask)
    if np.any(ids >= target.n) or not np.all(target.used[ids]):
        return None
    if sg is not None and not np.array_equal(sg.s_of_row[ids], codes):
        return None
    return ids


def _scalar(value):
    return (value is None or type(value) in (bool, int) or
            type(value) is str and len(value.encode("utf-8", "surrogatepass")) <= _MAX_TEXT_BYTES)


def project(codes, bits):
    result = np.zeros(len(codes), np.int64)
    for bit in bits:
        result = (result << 1) | ((codes >> bit) & 1)
    return result


def _attributes(target, codes, rows, starts, cache):
    attrs = {}
    for name in target.cols:
        codec = target.codecs.get(name)
        if (name.startswith("_") or "#" in name or name in target.wide or
                getattr(codec, "kind", None) not in ("int", "bool", "dict", "rank")):
            continue
        vals = _raw(target, name, cache)[rows].astype(np.int64)
        validity = target.nullable.get(name)
        if validity is not None:
            vals[_raw(target, validity, cache)[rows] == 0] = -1
        if not np.array_equal(np.minimum.reduceat(vals, starts), np.maximum.reduceat(vals, starts)):
            continue
        values = vals[starts]
        dependencies = keydesign.cell_bits(codes, values)
        _, indices = np.unique(project(codes, dependencies), return_index=True)
        values = values[indices]
        decoded = codec.decode(np.maximum(values, 0))
        decoded = [None if code < 0 else value for code, value in zip(values, decoded)]
        if all(map(_scalar, decoded)):
            attrs[name] = {"bits": dependencies, "values": decoded}
            if len(attrs) == _MAX_ATTRS:
                break
    return attrs


def build(db):
    """Find total stored paths and derive exact cell projections for saving."""
    records, cache = [], {}
    remaining = _MAX_HEADER_BYTES

    def visit(fact, table, path, names, bits, codes, rows, starts):
        nonlocal remaining
        if len(records) >= _MAX_RECORDS:
            return
        item = dict(fact=fact.name, path=list(path), tables=list(names), target=table.name,
                    bits=bits, codes=codes, attrs={})
        # Membership is useful even when no target column is a function of
        # the first edge's cell. Keep it when the attribute budget runs out.
        needed = len(_json(item))
        if needed > remaining:
            return
        remaining -= needed
        for attr, vals in _attributes(table, np.asarray(codes, np.int64), rows, starts, cache).items():
            needed = len(_json([attr, vals]))
            if needed <= remaining:
                item["attrs"][attr] = vals
                remaining -= needed
        records.append(item)
        if len(path) == _MAX_HOPS:
            return
        for fk in table.refbit:
            link = _link(db, table, fk)
            if link is None:
                continue
            target, sg, bit = link
            if target.name in names or not 0 < target.n <= _MAX_ROWS:
                continue
            mapped = _pull(table, fk, target, sg, bit, rows, cache)
            if mapped is not None:
                visit(fact, target, (*path, fk), (*names, target.name), bits, codes, mapped, starts)

    for fact in db.tables.values():
        if len(records) >= _MAX_RECORDS:
            break
        for fk in fact.refbit:
            link = _link(db, fact, fk)
            if link is None:
                continue
            target, sg, _ = link
            if target is fact:
                continue
            cells = _cells(target, sg)
            if cells is not None:
                bits, codes, rows, starts = cells
                visit(fact, target, (fk,), (fact.name, target.name), bits, codes, rows, starts)
    return records


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _names(records):
    return {name for item in records for name in item["tables"]}


def _manifest_hash(manifest, records, files):
    names = _names(records)
    tables = {name: manifest["tables"][name] for name in names}
    dictionaries = {name: manifest.get("intdicts", {}).get(name) for name in names}
    # Normalize the writer's default JSON conversion before hashing.
    normalized = json.loads(json.dumps([tables, dictionaries], default=str))
    return hashlib.sha256(_json([2, normalized, records, files])).hexdigest()


def _file_name(name):
    return (type(name) is str and bool(name) and name not in (".", "..")
            and not any(c in name for c in ("/", chr(92), ":", chr(0))))


def _stamps(path, files):
    result = {}
    for name in files:
        st = (path / name).stat()
        result[name] = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    return result


def _fingerprint(path, size):
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", size))
    mapped = path.suffix == ".tbl"
    with path.open("rb") as stream:
        length = size
        if path.suffix == ".tbl":
            head = stream.read(_TBL_HEADER)
            if (len(head) == _TBL_HEADER and head[:8] == b"TORBLE\x01\x00"
                    and struct.unpack_from("=II", head, 8) == (2, 0x01020304)):
                mapped = False
                length, = struct.unpack_from("=Q", head, _TBL_HEADER_LENGTH)
                if not _TBL_HEADER <= length <= size:
                    raise ValueError("invalid topology header length")
            stream.seek(0)
        # Native v2 checks every payload CRC and face before this proof is
        # admitted. Its complete directory already commits to those CRCs.
        while length:
            block = stream.read(min(length, 1 << 20))
            if not block:
                raise ValueError("truncated hierarchy source")
            digest.update(block)
            length -= len(block)
    return digest.hexdigest(), mapped


def _expected(records, table_files):
    return set().union(*(table_files[name] for name in _names(records)))


def save_record(path, manifest, records, table_files):
    files = _expected(records, table_files)
    path = Path(path)
    before = _stamps(path, files)
    hashes = {name: _fingerprint(path / name, before[name][2])[0] for name in sorted(files)}
    if _stamps(path, files) != before:
        raise RuntimeError("source changed during hierarchy save")
    return dict(version=2, records=records, files=hashes, sha256=_manifest_hash(manifest, records, hashes))


def _shape(item):
    if not isinstance(item, dict):
        return False
    fact, path, tables, target = (item.get(key) for key in ("fact", "path", "tables", "target"))
    bits, codes, attrs = (item.get(key) for key in ("bits", "codes", "attrs"))
    if not (isinstance(path, list) and 1 <= len(path) <= _MAX_HOPS and all(type(fk) is str for fk in path)
            and isinstance(tables, list) and len(tables) == len(path) + 1
            and all(type(name) is str for name in tables) and len(set(tables)) == len(tables)
            and fact == tables[0] and target == tables[-1]
            and isinstance(bits, list) and 0 < len(bits) <= _MAX_BITS
            and all(type(bit) is int and 0 <= bit < 32 for bit in bits)
            and bits == sorted(set(bits), reverse=True)
            and isinstance(codes, list) and 0 < len(codes) <= 1 << len(bits)
            and all(type(code) is int and 0 <= code < 1 << len(bits) for code in codes)
            and codes == sorted(set(codes)) and isinstance(attrs, dict) and len(attrs) <= _MAX_ATTRS):
        return False
    for attr, spec in attrs.items():
        if type(attr) is not str or not isinstance(spec, dict):
            return False
        keep, vals = spec.get("bits"), spec.get("values")
        if not (isinstance(keep, list) and all(type(bit) is int and 0 <= bit < len(bits) for bit in keep)
                and keep == sorted(set(keep), reverse=True) and isinstance(vals, list)
                and 1 <= len(vals) <= min(len(codes), 1 << len(keep)) and all(map(_scalar, vals))):
            return False
    return True


def load_start(path, manifest):
    header = manifest.get(HEADER)
    if not isinstance(header, dict) or type(header.get("version")) is not int or header["version"] != 2:
        return None
    records, files = header.get("records"), header.get("files")
    if (not isinstance(records, list) or not 0 < len(records) <= _MAX_RECORDS or not all(map(_shape, records))
            or not isinstance(files, dict) or not files or not all(map(_file_name, files))
            or any(type(digest) is not str or len(digest) != 64 for digest in files.values())):
        return None
    try:
        if (len(_json(records)) > _MAX_HEADER_BYTES + _MAX_RECORDS + 2
                or _manifest_hash(manifest, records, files) != header.get("sha256")):
            return None
        return _stamps(Path(path), files)
    except (KeyError, OSError, ValueError):
        return None


def _native(table):
    out = []
    for part in (table.parts or (table,)):
        at = part._raw_at
        out.append((part._native, at, _revision(at), tuple(part.cols)))
    return tuple(out)


def _bind(db, item, table_cache, sg_cache):
    tables = tuple(db.tables[name] for name in item["tables"])
    links = tuple(_link(db, source, fk) for source, fk in zip(tables, item["path"]))
    if any(link is None or link[0] is not target for link, target in zip(links, tables[1:])):
        return None
    cells = _cells(tables[1], links[0][1])
    if cells is None or cells[:2] != (item["bits"], item["codes"]):
        return None
    codes = np.asarray(item["codes"], np.int64)
    if any(attr not in tables[-1].codecs or len(np.unique(project(codes, spec["bits"]))) != len(spec["values"])
           for attr, spec in item["attrs"].items()):
        return None
    stamps = []
    for i, table in enumerate(tables):
        key = table.name, bool(i)
        if key not in table_cache:
            occupied = (table.used.copy(), table.used_bm.copy()) if i else None
            table_cache[key] = table, table.n, _native(table), occupied
        stamps.append(table_cache[key])
    sgs = []
    for _, sg, _ in links:
        if sg is not None and sg not in sg_cache:
            sg_cache[sg] = sg, sg.w, sg.version, sg.s_of_row.copy()
        if sg is not None:
            sgs.append(sg_cache[sg])
    codecs = tuple((attr, tables[-1].codecs[attr]) for attr in item["attrs"])
    return item, tables, links, stamps, sgs, codecs


def load_finish(db, path, manifest, table_files, before):
    if before is None or any(info["rows_with_faults"] for info in db.integrity.values()):
        return
    header = manifest[HEADER]
    records, files = header["records"], header["files"]
    path = Path(path)
    try:
        if set(files) != _expected(records, table_files) or _stamps(path, files) != before:
            return
        mapped_files = {}
        for name, expected in files.items():
            actual, mapped = _fingerprint(path / name, before[name][2])
            if actual != expected:
                return
            if mapped:
                mapped_files[name] = before[name]
        if _stamps(path, files) != before:
            return
        found, tables, surrogates = {}, {}, {}
        for item in records:
            key = item["fact"], tuple(item["path"])
            if key in found:
                return
            bound = _bind(db, item, tables, surrogates)
            if bound is None:
                return
            mapped = {name: mapped_files[name] for name in _expected([item], table_files) if name in mapped_files}
            found[key] = bound, mapped
        db._join_header = path.resolve(), found
    except (KeyError, OSError, ValueError):
        return


def record(db, fact, path):
    """Return a current exact stored join proof, or None."""
    if isinstance(fact, str):
        fact = db.tables.get(fact)
    if fact is None:
        return None
    path = tuple(path)
    location, proofs = getattr(db, "_join_header", (None, {}))
    entry = proofs.get((fact.name, path))
    if entry is None:
        return None
    bound, mapped_files = entry
    # A v1 private mapping can still see in-place file edits. Encoded v2
    # snapshots own their validated bytes and need no filesystem recheck.
    try:
        if mapped_files and _stamps(location, mapped_files) != mapped_files:
            return None
    except OSError:
        return None
    item, tables, links, stamps, sgs, codecs = bound
    if tables[0] is not fact or any(db.tables.get(table.name) is not table for table in tables):
        return None
    if any(table.closed or table.n != rows or _native(table) != native or
           (occupied is not None and (not np.array_equal(table.used, occupied[0]) or
                                      not np.array_equal(table.used_bm, occupied[1])))
           for table, rows, native, occupied in stamps):
        return None
    if any(sg.w != width or sg.version != version or sg.stale or not np.array_equal(sg.s_of_row, old)
           for sg, width, version, old in sgs):
        return None
    if any(_link(db, source, fk) != link for source, fk, link in zip(tables, path, links)):
        return None
    if any(tables[-1].codecs.get(attr) is not codec for attr, codec in codecs):
        return None
    return item
