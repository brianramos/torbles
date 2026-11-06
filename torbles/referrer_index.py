# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Versioned, immutable dictionary projections and counts for archived tables."""

import ctypes as C
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from uuid import uuid4

import numpy as np

from . import algq, coltypes

DEFAULT_DIR = ".referrer-index"
_VERSION = 1
_FIELDS = (("mapping", np.uint32), ("lengths", np.uint32), ("spans", np.uint64), ("counts", np.uint32))
_DOMAIN_LIMIT = 1 << 27


class ReferrerArrays:
    def __init__(self, arrays):
        self.arrays = arrays
        self.cuda = None


@dataclass(frozen=True)
class _Binding:
    state: ReferrerArrays
    codec: object
    storage: object
    revision: int
    visible: np.ndarray

    def matches(self, table, column):
        return (table.codecs.get(column) is self.codec and table.used_bm is self.visible
                and table._part_of(column)._storage() is self.storage and self.storage.revision == self.revision)


def _stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _local(root, name):
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("invalid referrer index filename")
    return root / name


def _source_files(archive, manifest, name, column):
    from .database import _fname

    meta = manifest["tables"][name]
    parts = meta.get("parts")
    fact = next(p["file"] for p in parts if column in p["cols"]) if parts else meta["file"]
    base = _fname(name)
    files = ("manifest.json", fact, base + "." + _fname(f"dictblob:{column}") + ".npy",
             base + "." + _fname(f"dictoffs:{column}") + ".npy")
    if not meta["used_all"]:
        files += (base + ".used.npy",)
    return {file: _stamp(_local(archive, file)) for file in files}


def _snapshot(table, column):
    codec = table.codecs.get(column)
    dictionary = getattr(codec, "values", None)
    if (not isinstance(dictionary, coltypes.PackedStrings) or column in table.nullable
            or not 0 < len(dictionary) <= _DOMAIN_LIMIT or table.n >= 1 << 32):
        raise ValueError(f"{table.name}.{column} needs a non-null packed VARCHAR dictionary")

    vp = algq.vp
    prepare = algq.lib.at_referrer_prepare
    prepare.restype = C.c_int
    prepare.argtypes = [vp, C.c_uint64, vp, C.c_uint32, vp, vp, vp]
    histogram = algq.lib.at_referrer_code_counts
    histogram.restype = C.c_int
    histogram.argtypes = [vp, vp, C.c_int32, C.c_uint32, vp]
    arrays = tuple(np.empty(len(dictionary), dtype) for _, dtype in _FIELDS)
    mapping, lengths, spans, counts = arrays
    start = perf_counter()
    if prepare(dictionary.blob.ctypes.data, dictionary.blob.nbytes, dictionary.offsets.ctypes.data,
               len(dictionary), mapping.ctypes.data, lengths.ctypes.data, spans.ctypes.data):
        raise RuntimeError("could not prepare referrer dictionary projection")
    projected = perf_counter()
    part = table._part_of(column)
    if histogram(part.at, table.used_bm.ctypes.data, part.idx(column), len(dictionary), counts.ctypes.data):
        raise RuntimeError("could not prepare referrer code counts")
    if int(counts.sum(dtype=np.uint64)) != np.count_nonzero(table.used):
        raise ValueError("referrer counts disagree with archive visibility")
    return arrays, dict(project_ms=(projected - start) * 1000, count_ms=(perf_counter() - projected) * 1000)


def build_archive_referrers(archive, columns, *, output=None):
    """Build the registered projection and exact visible code counts from a saved archive.

    The index is published after all files have been written. An external output
    directory can index an existing archive without writing to that archive.
    """
    from .database import load

    archive = Path(archive)
    output = Path(output) if output is not None else archive / DEFAULT_DIR
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((archive / "manifest.json").read_text())
    columns = tuple(columns)
    source_stamps = {(name, column): _source_files(archive, manifest, name, column) for name, column in columns}
    generation = uuid4().hex
    records, receipts, created = [], [], []
    db = load(archive, validate=False, packed_text=True, referrer_index=False)
    start = perf_counter()
    try:
        for number, (name, column) in enumerate(columns):
            source = source_stamps[name, column]
            arrays, timing = _snapshot(db[name], column)
            files = {}
            write_start = perf_counter()
            for (field, _), array in zip(_FIELDS, arrays, strict=True):
                filename = f"{generation}-{number}-{field}.npy"
                path = output / filename
                created.append(path)
                with path.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                files[filename] = _stamp(path)
            if source != _source_files(archive, manifest, name, column):
                raise RuntimeError("archive changed while preparing referrer index")
            records.append(dict(table=name, column=column, files=files, source=source))
            receipts.append(dict(table=name, column=column, codes=len(arrays[0]),
                                 bytes=sum(array.nbytes for array in arrays),
                                 write_ms=(perf_counter() - write_start) * 1000, **timing))
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            created.append(temporary)
            json.dump(dict(version=_VERSION, records=records), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "index.json")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        db.close()
    return dict(records=receipts, build_ms=(perf_counter() - start) * 1000)


def read_archive_referrers(archive, manifest, index):
    """Check the index's archive identity before opening native data handles."""
    archive, index = Path(archive), Path(index)
    try:
        catalog = json.loads((index / "index.json").read_text())
        if catalog["version"] != _VERSION or not isinstance(catalog["records"], list):
            return ()
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    valid = []
    for record in catalog["records"]:
        try:
            if record["source"] == _source_files(archive, manifest, record["table"], record["column"]):
                valid.append(record)
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            continue
    return valid


def load_archive_referrers(db, archive, manifest, index, records):
    """Attach compatible mmap state. Stale or incomplete sidecars use ordinary execution."""
    archive, index = Path(archive), Path(index)
    for record in records:
        try:
            name, column = record["table"], record["column"]
            table = db[name]
            # A load-time repair changes the mapped planes without changing
            # the archive files from which the persisted counts were derived.
            if db.integrity.get(name, {}).get("repaired"):
                continue
            codec = table.codecs[column]
            if (not isinstance(codec.values, coltypes.PackedStrings) or column in table.nullable
                    or table.n >= 1 << 32 or record["source"] != _source_files(archive, manifest, name, column)):
                continue
            domain = len(codec.values)
            files = record["files"]
            # Field suffixes identify arrays independently of manifest object ordering.
            arrays = []
            for field, dtype in _FIELDS:
                filename, = (f for f in files if f.endswith(f"-{field}.npy"))
                path = _local(index, filename)
                if _stamp(path) != files[filename]:
                    raise ValueError("referrer index file changed")
                a = np.load(path, mmap_mode="r", allow_pickle=False)
                if a.dtype != dtype or a.shape != (domain,) or _stamp(path) != files[filename]:
                    raise ValueError("invalid referrer index array")
                arrays.append(a)
            part = table._part_of(column)
            storage = part._storage()
            table.__dict__.setdefault("_referrer_indexes", {})[column] = _Binding(
                ReferrerArrays(tuple(arrays)), codec, storage, storage.revision, table.used_bm,
            )
        except (OSError, AttributeError, KeyError, TypeError, ValueError, StopIteration):
            continue


def inherit_referrer_indexes(table, sources):
    """Rebind indexes when opening an equivalent row-aligned composite table."""
    for source in sources:
        for column, binding in getattr(source, "_referrer_indexes", {}).items():
            if (table.codecs.get(column) is not binding.codec
                    or table._part_of(column)._storage() is not binding.storage
                    or binding.storage.revision != binding.revision
                    or not np.array_equal(table.used_bm, binding.visible)):
                continue
            table.__dict__.setdefault("_referrer_indexes", {})[column] = _Binding(
                binding.state, binding.codec, binding.storage, binding.revision, table.used_bm,
            )


def index_for(table, column):
    binding = getattr(table, "_referrer_indexes", {}).get(column)
    return binding.state if binding is not None and binding.matches(table, column) else None
