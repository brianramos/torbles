# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Logical redo for validated mutations of writable uint32 tables."""

import hashlib
import struct

import numpy as np


_MAGIC = b"TBR1"
_HEADER = struct.Struct("<4sI16sQQQH")
_CODE = np.dtype("<u4")
_COLUMN = np.dtype("<u2")


def _schema(table):
    digest = hashlib.blake2s(digest_size=16)
    for name in (table.primary_key, *table.columns):
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def attach(db, callback):
    """Capture one binary record after each successful nonempty mutation."""
    if callback is not None and not callable(callback):
        raise TypeError("redo callback must be callable or None")
    with db._sql_lock:
        for table in db._writable.values():
            with table._lock:
                table._redo_callback = callback


def _encoded(array, dtype=_CODE):
    return np.asarray(array, dtype=dtype).tobytes()


def encode(table, update_keys=(), updates=None, inserts=None, delete_keys=()):
    """Encode primary keys and assigned values from a validated mutation."""
    nupdate = len(update_keys) if updates else 0
    ninsert = len(inserts[table.primary_key]) if inserts else 0
    ndelete = len(delete_keys)
    if not (nupdate or ninsert or ndelete):
        return None
    selected = [name for name in table.columns if nupdate and name in updates]
    table_name = table.name.encode("utf-8")
    parts = [
        _HEADER.pack(_MAGIC, len(table_name), _schema(table), nupdate, ninsert, ndelete, len(selected)),
        table_name,
        _encoded([table.columns.index(name) for name in selected], _COLUMN),
    ]
    if nupdate:
        parts.append(_encoded(update_keys))
        parts.extend(_encoded(updates[name]) for name in selected)
    if ninsert:
        parts.extend(_encoded(inserts[name]) for name in table.columns)
    if ndelete:
        parts.append(_encoded(delete_keys))
    return b"".join(parts)


def replay(db, payload):
    """Apply one redo record to the corresponding writable table without evaluating SQL."""
    if not isinstance(payload, bytes) or len(payload) < _HEADER.size:
        raise ValueError("invalid redo record header")
    magic, name_size, signature, nupdate, ninsert, ndelete, ncols = _HEADER.unpack_from(payload)
    offset = _HEADER.size + name_size
    if magic != _MAGIC or offset > len(payload) or not name_size or not (nupdate or ninsert or ndelete):
        raise ValueError("invalid redo record header")
    name = payload[_HEADER.size:offset].decode("utf-8")
    with db._sql_lock:
        table = db._writable.get(name)
        if table is None or _schema(table) != signature:
            raise ValueError(f"redo table or schema mismatch for {name!r}")
        with table._lock:
            if table._redo_callback is not None:
                raise ValueError("replay requires capture to be detached")
            if not 0 <= ncols < len(table.columns) or bool(ncols) != bool(nupdate):
                raise ValueError("invalid redo update columns")
            total = offset + ncols * _COLUMN.itemsize
            total += (nupdate * (1 + ncols) + ninsert * len(table.columns) + ndelete) * _CODE.itemsize
            if total != len(payload):
                raise ValueError("invalid redo record length")
            positions = np.frombuffer(payload, dtype=_COLUMN, count=ncols, offset=offset).tolist()
            if len(set(positions)) != ncols or any(i >= len(table.columns) for i in positions):
                raise ValueError("invalid redo column positions")
            names = [table.columns[i] for i in positions]
            if table.primary_key in names:
                raise ValueError("redo cannot update a primary key")
            offset += ncols * _COLUMN.itemsize

            def read(n):
                nonlocal offset
                values = np.frombuffer(payload, dtype=_CODE, count=n, offset=offset)
                offset += n * _CODE.itemsize
                return values

            updated_keys = read(nupdate)
            updates = {col: read(nupdate) for col in names}
            inserts = {col: read(ninsert) for col in table.columns} if ninsert else None
            deleted_keys = read(ndelete)
            updated_ids = table.lookup(updated_keys)
            deleted_ids = table.lookup(deleted_keys)
            if (updated_ids < 0).any() or (deleted_ids < 0).any():
                raise ValueError("redo references an absent primary key")
            return table.merge(updated_ids, updates, inserts, delete_ids=deleted_ids)
