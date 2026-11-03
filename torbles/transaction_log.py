# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Durable checkpoint and append-only transaction frames for writable SQL tables."""

from collections.abc import Mapping
import fcntl
import json
import os
from pathlib import Path
import struct
import tempfile
from threading import RLock
import uuid
from zipfile import BadZipFile
import zlib

import numpy as np

from .algq import MAXCOL


_FORMAT = 1
_WAL_MAGIC = b"TBWAL001"
_FRAME_MAGIC = b"TBR1"
_FRAME = struct.Struct("<4sQII")
_MAX_RECORD = 1 << 30
_READ_CHUNK = 1 << 20
_U32_MAX = (1 << 32) - 1


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_file(path, write):
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _unique_key(values):
    if len(values) < 2:
        return True
    compare = np.greater if values[0] < values[-1] else np.less
    for start in range(0, len(values) - 1, _READ_CHUNK):
        stop = min(start + _READ_CHUNK, len(values) - 1)
        if not compare(values[start + 1:stop + 1], values[start:stop]).all():
            return len(np.unique(values)) == len(values)
    return True


def _snapshot(tables, identity):
    if not isinstance(tables, Mapping):
        raise TypeError("checkpoint tables must be a mapping")
    metadata = {"format": _FORMAT, "identity": identity.hex, "tables": []}
    arrays = {}
    names = set()
    for name, (columns, key) in tables.items():
        if (not isinstance(name, str) or not name or not isinstance(columns, Mapping)
                or not isinstance(key, str) or key not in columns):
            raise ValueError("invalid checkpoint table or primary key")
        if len(columns) > MAXCOL:
            raise ValueError(f"checkpoint tables hold at most {MAXCOL} columns")
        if name.lower() in names:
            raise ValueError("checkpoint table names must be unique ignoring case")
        names.add(name.lower())
        column_names = set()
        schema = {"name": name, "primary_key": key, "columns": [], "rows": None}
        for column, values in columns.items():
            array = np.asarray(values)
            if not isinstance(column, str) or not column or array.ndim != 1:
                raise ValueError("invalid checkpoint column")
            if column.lower() in column_names:
                raise ValueError("checkpoint column names must be unique ignoring case")
            column_names.add(column.lower())
            if array.size and (array.dtype.kind not in "biu" or int(array.min()) < 0 or int(array.max()) > _U32_MAX):
                raise ValueError(f"checkpoint column {column!r} must contain uint32 values")
            if schema["rows"] is None:
                schema["rows"] = len(array)
            elif schema["rows"] != len(array):
                raise ValueError("checkpoint columns must have the same length")
            schema["columns"].append(column)
            array = np.ascontiguousarray(array, dtype="<u4")
            if column == key and not _unique_key(array):
                raise ValueError(f"duplicate primary key in checkpoint table {name!r}")
            arrays[f"c{len(arrays)}"] = array
        metadata["tables"].append(schema)
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode("utf-8"), np.uint8)
    return arrays


def _metadata(archive):
    try:
        raw = archive["metadata"]
        if raw.ndim != 1 or raw.dtype != np.uint8:
            raise ValueError
        meta = json.loads(raw.tobytes())
        if not isinstance(meta, dict) or meta["format"] != _FORMAT:
            raise ValueError
        identity = uuid.UUID(hex=meta["identity"])
        schemas = meta["tables"]
        if not isinstance(schemas, list):
            raise ValueError
        names = set()
        ncols = 0
        for schema in schemas:
            name, pk, cols, n = (schema[k] for k in ("name", "primary_key", "columns", "rows"))
            if (not isinstance(name, str) or not name or name.lower() in names or not isinstance(cols, list)
                    or not 0 < len(cols) <= MAXCOL or any(not isinstance(c, str) or not c for c in cols)
                    or len({c.lower() for c in cols}) != len(cols) or not isinstance(pk, str) or pk not in cols
                    or type(n) is not int or n < 0):
                raise ValueError
            names.add(name.lower())
            ncols += len(cols)
        expected = {"metadata", *(f"c{i}" for i in range(ncols))}
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError
        return identity, schemas
    except (AttributeError, KeyError, TypeError, ValueError, BadZipFile) as exc:
        raise ValueError("invalid transaction checkpoint metadata") from exc


def _write_all(fd, data):
    remaining = memoryview(data)
    while remaining:
        count = os.write(fd, remaining)
        if not count:
            raise OSError("short transaction WAL write")
        remaining = remaining[count:]


def _frame_info(header):
    magic, length, checksum, head_crc = _FRAME.unpack(header)
    if magic != _FRAME_MAGIC or length > _MAX_RECORD or zlib.crc32(header[:-4]) != head_crc:
        raise ValueError("corrupt transaction WAL frame header")
    return length, checksum


def _frame_header(payload):
    prefix = struct.pack("<4sQI", _FRAME_MAGIC, len(payload), zlib.crc32(payload))
    return prefix + struct.pack("<I", zlib.crc32(prefix))


class Journal:
    """Exclusive local checkpoint and WAL handle.

    Supplying tables creates a new checkpoint. An opener verifies complete
    frames before returning and removes a torn final frame. An append is
    durable when it returns successfully. An I/O failure poisons that opener
    so the caller must reopen and recover before attempting another append.
    """

    def __init__(self, path, *, tables=None):
        self.path = Path(path)
        self._lock = RLock()
        self._lock_fd = self._wal_fd = None
        self._poisoned = False
        if tables is not None:
            try:
                self.path.mkdir()
            except FileExistsError:
                if not self.path.is_dir():
                    raise
            else:
                _fsync_dir(self.path.parent)
        base, wal = self.path / "base.npz", self.path / "wal"
        try:
            self._lock_fd = os.open(self.path / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if tables is not None:
                if base.exists():
                    raise FileExistsError("transaction checkpoint already exists")
                if wal.exists():
                    with wal.open("rb") as file:
                        orphan = file.read(len(_WAL_MAGIC) + 17)
                    if len(orphan) != len(_WAL_MAGIC) + 16 or not orphan.startswith(_WAL_MAGIC):
                        raise FileExistsError("transaction WAL already exists without its checkpoint")
                identity = uuid.uuid4()
                arrays = _snapshot(tables, identity)
                # Publish the checkpoint last. A header-only WAL without a
                # checkpoint can only be the result of interrupted creation.
                _atomic_file(wal, lambda file: file.write(_WAL_MAGIC + identity.bytes))
                _atomic_file(base, lambda file: np.savez(file, **arrays))
            else:
                with np.load(base, allow_pickle=False) as archive:
                    identity, _ = _metadata(archive)
            self._identity = identity
            wal_header = _WAL_MAGIC + identity.bytes
            self._first = len(wal_header)
            self._wal_fd = os.open(wal, os.O_RDWR | os.O_APPEND)
            if os.pread(self._wal_fd, self._first, 0) != wal_header:
                raise ValueError("invalid transaction WAL header or checkpoint identity")
            self._end = self._recover()
        except BaseException:
            self.close()
            raise

    def _check_open(self):
        if self._wal_fd is None:
            raise RuntimeError("transaction journal is closed")
        if self._poisoned:
            raise RuntimeError("transaction journal must be reopened after an I/O failure")

    def _recover(self):
        fd = self._wal_fd
        end = os.fstat(fd).st_size
        pos = self._first
        while end - pos >= _FRAME.size:
            header = os.pread(fd, _FRAME.size, pos)
            if len(header) != _FRAME.size:
                raise OSError("short transaction WAL header read")
            length, checksum = _frame_info(header)
            payload = pos + _FRAME.size
            if length > end - payload:
                break
            crc = 0
            for offset in range(0, length, _READ_CHUNK):
                size = min(_READ_CHUNK, length - offset)
                chunk = os.pread(fd, size, payload + offset)
                if len(chunk) != size:
                    raise OSError("short transaction WAL payload read")
                crc = zlib.crc32(chunk, crc)
            if crc != checksum:
                raise ValueError("corrupt transaction WAL frame payload")
            pos = payload + length
        if pos != end:
            os.ftruncate(fd, pos)
            os.fsync(fd)
        return pos

    def load_base(self):
        """Read the original checkpoint as {table: (uint32 columns, primary key)}."""
        with self._lock:
            self._check_open()
            with np.load(self.path / "base.npz", allow_pickle=False) as archive:
                identity, schemas = _metadata(archive)
                if identity != self._identity:
                    raise ValueError("transaction checkpoint changed while open")
                result = {}
                i = 0
                for schema in schemas:
                    columns = {}
                    for name in schema["columns"]:
                        array = archive[f"c{i}"]
                        if array.shape != (schema["rows"],) or array.dtype != np.dtype("<u4"):
                            raise ValueError("invalid transaction checkpoint column")
                        columns[name] = np.ascontiguousarray(array, dtype=np.uint32)
                        i += 1
                    result[schema["name"]] = columns, schema["primary_key"]
                return result

    def records(self):
        """Yield the checked transaction payloads present when iteration begins."""
        with self._lock:
            self._check_open()
            end = self._end
        pos = self._first
        while pos < end:
            with self._lock:
                self._check_open()
                header = os.pread(self._wal_fd, _FRAME.size, pos)
                if len(header) != _FRAME.size:
                    raise OSError("short transaction WAL header read")
                length, checksum = _frame_info(header)
                pos += _FRAME.size
                if length > end - pos:
                    raise ValueError("transaction WAL changed after recovery")
                payload = os.pread(self._wal_fd, length, pos)
                if len(payload) != length or zlib.crc32(payload) != checksum:
                    raise ValueError("corrupt transaction WAL frame payload")
                pos += length
            yield payload

    def append(self, payload):
        """Write and fsync one transaction payload."""
        if not isinstance(payload, bytes) or len(payload) > _MAX_RECORD:
            raise ValueError("transaction payload must be bytes no larger than 1 GiB")
        header = _frame_header(payload)
        with self._lock:
            self._check_open()
            try:
                _write_all(self._wal_fd, header)
                _write_all(self._wal_fd, payload)
                os.fsync(self._wal_fd)
            except BaseException:
                self._poisoned = True
                raise
            self._end += len(header) + len(payload)

    @property
    def wal_bytes(self):
        """Physical WAL bytes, including its checkpoint header."""
        with self._lock:
            self._check_open()
            return self._end

    def close(self):
        with self._lock:
            wal, lock = self._wal_fd, self._lock_fd
            self._wal_fd = self._lock_fd = None
            try:
                if wal is not None:
                    os.close(wal)
            finally:
                if lock is not None:
                    os.close(lock)
