# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Source identity for reusable SQL workload states."""

from contextlib import ExitStack
import ctypes as C
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from . import algq


def _revision(at):
    fn = algq.lib.tr_math_table_revision
    fn.argtypes = [C.c_void_p]
    fn.restype = C.c_uint64
    return int(fn(at))


def _array_digest(array):
    data = memoryview(array).cast("B") if array.flags.c_contiguous else array.tobytes()
    return array.shape, array.dtype.str, hashlib.sha256(data).digest()


def _occupancy(table):
    used, bitmap = table.used, table.used_bm
    rows = table.n
    full, tail = divmod(rows, 64)
    limit = full + bool(tail)
    all_bits = np.uint64((1 << 64) - 1)
    used_key = ("all", rows) if used.shape == (rows,) and used.dtype == np.bool_ and used.all() else _array_digest(used)
    if (bitmap.ndim == 1 and bitmap.dtype == np.uint64 and len(bitmap) >= limit
            and np.bitwise_and.reduce(bitmap[:full], initial=all_bits) == all_bits
            and (not tail or int(bitmap[full]) == (1 << tail) - 1)
            and not bitmap[limit:].any()):
        bitmap_key = ("all", bitmap.shape)
    else:
        bitmap_key = _array_digest(bitmap)
    return used_key, bitmap_key


def _native_anchor(part):
    at = part.at
    return part._native, at, _revision(at), tuple(part.cols)


def _anchor(db):
    tables = []
    members = {id(table) for table in db.tables.values()}
    for name, table in db.tables.items():
        parts = tuple(_native_anchor(part) for part in (table.parts or (table,)))
        sidecars = tuple((column, side["side"], _revision(side["side"]), _array_digest(side["rank"]))
                         for column, side in sorted(getattr(table, "sidecars", {}).items()))
        surrogates = tuple((column, surrogate, surrogate.version)
                           for column, surrogate in sorted(getattr(table, "surr", {}).items()) if surrogate is not None)
        references = tuple(sorted(getattr(table, "_refdim", {}).items()))
        if any(id(dimension) not in members for _, dimension in references):
            return None
        exact_codecs = tuple(sorted((name, codec.offset) for name, codec in table.codecs.items()
                                    if codec.kind == "int"))
        tables.append((name, table, table.n, parts, sidecars, surrogates, references, exact_codecs,
                       _occupancy(table)))
    return tuple(tables)


def _writable_anchor(db, locks):
    stores = [(name, table, table.store) for name, table in db._writable.items()]
    for _, _, store in stores:
        locks.enter_context(store._lock)
    writable = []
    for name, table, store in stores:
        if store._closed:
            return None
        segments = tuple(
            (segment, _native_anchor(segment.table), segment.start, segment.stop,
             segment.rows, segment.live, (id(segment.bitmap), segment.bitmap))
            for segment in store._segments
        )
        # Logical-ID overrides are only edited by replacement, deletion and
        # compaction, each of which also changes the segment/visibility anchor.
        writable.append((name, table, store, store._live, store._next_id,
                         tuple(store._origins), tuple(store._starts), len(store._overrides), segments))
    return tuple(writable)


def token(db):
    """Return the current source token, or None for unsupported/closed sources."""
    with db._sql_lock, ExitStack() as locks:
        try:
            proof = getattr(db, "_sql_snapshot_proof", None)
            if proof is not None and _stamps(proof[0], proof[1]) != proof[1]:
                return None
            immutable = _anchor(db)
            writable = _writable_anchor(db, locks)
        except (OSError, algq.TableClosedError):
            return None
        if immutable is None or writable is None:
            return None
        anchor = immutable, writable
        previous = getattr(db, "_sql_snapshot", None)
        if previous is not None and anchor == previous[0]:
            return previous[1]
        value = uuid.uuid4().hex
        db._sql_snapshot = anchor, value
        return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_digest(manifest, value, files):
    # Round-trip the writer's conversion of notes and other metadata to JSON.
    normalized = json.loads(json.dumps(
        {key: item for key, item in manifest.items() if key != "sql_snapshot"}, default=str))
    return hashlib.sha256(_json([normalized, value, files])).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamps(path, names):
    stamps = {}
    for name in names:
        st = (Path(path) / name).stat()
        stamps[name] = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    return stamps


def save_record(db, path, manifest, value, filenames):
    if db._writable:
        return None
    for table in db.tables.values():
        packed = np.packbits(table.used, bitorder="little")
        bitmap = table.used_bm.view(np.uint8)
        if not np.array_equal(bitmap[:len(packed)], packed) or np.any(bitmap[len(packed):]):
            # Persistence reconstructs the bitmap from used. Only carry over a
            # token when that reconstruction represents the same row domain.
            return None
    names = set(filenames)
    before = _stamps(path, names)
    files = {name: _file_digest(Path(path) / name) for name in sorted(names)}
    if before != _stamps(path, names) or token(db) != value:
        raise RuntimeError("source changed during SQL snapshot save")
    return dict(version=1, token=value, files=files, sha256=_manifest_digest(manifest, value, files))


def load_start(path, manifest, validate):
    """Record file stamps before mapping. An invalid proof permits ordinary loading."""
    record = manifest.get("sql_snapshot")
    if not validate or not isinstance(record, dict):
        return None
    value, files = record.get("token"), record.get("files")
    if (type(record.get("version")) is not int or record["version"] != 1 or
            not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value) or
            not isinstance(files, dict) or
            any(not name or name in (".", "..") or "/" in name or chr(92) in name or ":" in name
                for name in files)):
        return None
    try:
        if _manifest_digest(manifest, value, files) != record.get("sha256"):
            return None
        return _stamps(path, files)
    except (OSError, ValueError):
        return None


def load_finish(db, path, manifest, filenames, before):
    if before is None or any(item["rows_with_faults"] for item in db.integrity.values()):
        return
    record = manifest["sql_snapshot"]
    files = record["files"]
    if set(filenames) != set(files):
        return
    try:
        if before != _stamps(path, files):
            return
        if any(_file_digest(Path(path) / name) != expected for name, expected in files.items()):
            return
        if before == _stamps(path, files):
            anchor = _anchor(db)
            if anchor is not None:
                db._sql_snapshot = (anchor, ()), record["token"]
                db._sql_snapshot_proof = Path(path).resolve(), before
    except (OSError, ValueError):
        return
