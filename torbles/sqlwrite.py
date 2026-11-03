# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Writable SQL table storage over the in-memory refresh engine."""

import numpy as np

from .refresh import RefreshTable, _columns, _vector, _U32_MAX


class WritableTable:
    """A fixed, nonnullable uint32 schema with one immutable primary key.

    Row IDs are the refresh store's logical IDs and survive replacement
    segments. All mutations and scans use the same lock as native queries.
    The store may be queried or compacted directly, while mutations go
    through this adapter so the primary-key index stays in sync.
    """

    def __init__(self, name, columns, primary_key):
        data = _columns(columns)
        if primary_key not in data:
            raise ValueError("primary key must name a table column")
        self.name, self.columns, self.primary_key = name, tuple(data), primary_key
        keys = data[primary_key]
        self._base_n = len(keys)
        self._base_start = None
        self._base_keys = self._base_rows = self._base_deleted = None
        if len(keys):
            # The common identity or offset key needs neither a copy nor an
            # index with one object per row. Bound the temporary comparisons.
            increasing = True
            for start in range(0, len(keys) - 1, 1 << 20):
                stop = min(start + (1 << 20), len(keys) - 1)
                if not (keys[start + 1:stop + 1] > keys[start:stop]).all():
                    increasing = False
                    break
            if increasing and int(keys[-1]) - int(keys[0]) + 1 == len(keys):
                self._base_start = int(keys[0])
            elif increasing:
                self._base_keys = keys.copy()
            else:
                order = np.argsort(keys, kind="stable")
                sorted_keys = keys[order]
                if (sorted_keys[1:] == sorted_keys[:-1]).any():
                    raise ValueError("duplicate primary key in initial rows")
                self._base_keys = sorted_keys
                self._base_rows = order.astype(np.uint32)
        self._appended = {}
        self.store = RefreshTable(data, name=name, update_mode="planes")
        self._lock = self.store._lock

    def _lookup(self, keys):
        if self._base_start is not None:
            ids = keys.astype(np.int64) - self._base_start
            ids[(ids < 0) | (ids >= self._base_n)] = -1
        else:
            ids = np.full(len(keys), -1, np.int64)
            if self._base_keys is not None:
                positions = np.searchsorted(self._base_keys, keys)
                candidates = np.flatnonzero(positions < self._base_n)
                candidates = candidates[self._base_keys[positions[candidates]] == keys[candidates]]
                hits = positions[candidates]
                ids[candidates] = hits if self._base_rows is None else self._base_rows[hits]
        if self._base_deleted is not None:
            hits = np.flatnonzero(ids >= 0)
            logical = ids[hits].astype(np.uint64)
            dead = (self._base_deleted[logical >> 6] >> (logical & 63)) & 1
            ids[hits[dead != 0]] = -1
        if self._appended:
            misses = np.flatnonzero(ids < 0)
            ids[misses] = np.fromiter((self._appended.get(key, -1) for key in keys[misses].tolist()),
                                      dtype=np.int64, count=len(misses))
        return ids

    def lookup(self, keys):
        """Return int64 logical row IDs, with -1 for absent or deleted keys."""
        with self._lock:
            self.store._check_open()
            return self._lookup(_vector(keys, np.uint32, _U32_MAX, "primary keys", "biu"))

    def _insert_data(self, columns):
        data = _columns(columns)
        if data.keys() != set(self.columns):
            raise ValueError("insert requires exactly the table's columns")
        keys = data[self.primary_key]
        if len(np.unique(keys)) != len(keys) or (self._lookup(keys) >= 0).any():
            raise ValueError("duplicate primary key in insert")
        if self.store._next_id + len(keys) > 1 << 63:
            raise OverflowError("writable row IDs exceed int64")
        return data, keys.tolist()

    def _append(self, data, keys):
        start = self.store._next_id
        try:
            self._appended.update(zip(keys, range(start, start + len(keys))))
            return self.store._append(data)
        except BaseException:
            for key in keys:
                self._appended.pop(key, None)
            raise

    def _undo_append(self, assigned, keys):
        if not keys:
            return
        self.store._next_id = assigned.start
        self.store._live -= len(keys)
        segment = self.store._segments.pop()
        self.store._origins.pop()
        self.store._starts.pop()
        for key in keys:
            self._appended.pop(key, None)
        segment.table.close()

    def insert(self, columns):
        """Insert a batch and return the contiguous logical row IDs assigned."""
        with self._lock:
            self.store._check_open()
            return self._append(*self._insert_data(columns))

    def _update_data(self, ids, columns):
        data = _columns({} if columns is None else columns)
        self.store._check_columns(data)
        if self.primary_key in data:
            raise ValueError("primary key is immutable")
        ids, groups = self.store._locate(ids, distinct=True)
        if any(len(column) != len(ids) for column in data.values()):
            raise ValueError("replacement columns must match the number of row IDs")
        return ids, data, groups

    def update(self, ids, columns):
        """Update distinct live rows and return the number affected."""
        with self._lock:
            self.store._check_open()
            ids, data, groups = self._update_data(ids, columns)
            self.store._update_prepared(ids, data, groups)
            return len(ids) if data else 0

    def _delete_data(self, ids):
        ids, groups = self.store._locate(ids, distinct=True)
        base = ids[ids < self._base_n]
        appended = ids[ids >= self._base_n]
        keys = self.store.read(appended, self.primary_key)[self.primary_key].tolist() if len(appended) else []
        bitmap = self._base_deleted
        if len(base) and bitmap is None:
            bitmap = np.zeros((self._base_n + 63) // 64, np.uint64)
        index = bitmap, (base >> 6).astype(np.intp), np.left_shift(np.uint64(1), base & 63), keys
        return ids, self.store._prepare_delete(ids, groups), index

    def _delete(self, prepared, index):
        self.store._apply_delete(prepared)
        bitmap, words, bits, keys = index
        if len(words):
            np.bitwise_or.at(bitmap, words, bits)
            self._base_deleted = bitmap
        for key in keys:
            del self._appended[key]

    def delete(self, ids):
        """Delete distinct live rows and return the number affected."""
        with self._lock:
            self.store._check_open()
            ids, prepared, index = self._delete_data(ids)
            self._delete(prepared, index)
            return len(ids)

    def merge(self, ids, update_cols, insert_cols, *, delete_ids=None):
        """Apply disjoint updates/deletes and new rows as one locked batch."""
        with self._lock:
            self.store._check_open()
            ids, data, groups = self._update_data(ids, update_cols)
            deleted, prepared, index = self._delete_data(() if delete_ids is None else delete_ids)
            if len(ids) and len(deleted) and len(np.intersect1d(ids, deleted, assume_unique=True)):
                raise ValueError("cannot update and delete the same row in one merge")
            insert, keys = self._insert_data(insert_cols) if insert_cols is not None else (None, ())
            assigned = self._append(insert, keys) if insert is not None else None
            try:
                self.store._update_prepared(ids, data, groups)
            except BaseException:
                self._undo_append(assigned, keys)
                raise
            self._delete(prepared, index)
            return (len(ids) if data else 0) + len(deleted) + len(keys)

    def scan(self, columns=None):
        """Read the requested visible columns in logical row ID order."""
        with self._lock:
            self.store._check_open()
            names = self.columns if columns is None else (columns,) if isinstance(columns, str) else tuple(columns)
            self.store._check_columns(names)
            result = {name: np.empty(self.store._live, np.uint32) for name in names}
            if not result:
                return result
            ids = np.empty(self.store._live, np.uint64) if self.store._overrides else None
            offset = 0
            for segment in self.store._segments:
                if not segment.live:
                    continue
                rows = segment.active_rows()
                target = slice(offset, offset + segment.live)
                if ids is not None:
                    ids[target] = segment.logical_ids(rows)
                sparse = rows is not None and segment.live * 4 < segment.rows
                if sparse:
                    rows = rows.astype(np.uint64, copy=False)
                for name, out in result.items():
                    if sparse:
                        out[target] = segment.table.get_rows(name, rows)
                    else:
                        values = segment.table._decode_column(name)
                        out[target] = values if rows is None else values[rows]
                offset += segment.live
            if ids is not None:
                order = np.argsort(ids, kind="stable")
                return {name: column[order] for name, column in result.items()}
            return result

    def close(self):
        with self._lock:
            self.store.close()
            self._appended.clear()
            self._base_keys = self._base_rows = self._base_deleted = None
