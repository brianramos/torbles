# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""In-memory refresh of fixed uint32 relations over native bitplanes."""

from bisect import bisect_right
from collections.abc import Mapping
from numbers import Integral
from threading import RLock

import numpy as np

from . import algq


_U32_MAX = (1 << 32) - 1


def _vector(values, dtype, maximum, what, kinds="iu"):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional")
    if array.size and (array.dtype.kind not in kinds or int(array.min()) < 0 or int(array.max()) > maximum):
        raise ValueError(f"{what} must contain integers in [0, {maximum}]")
    return np.ascontiguousarray(array, dtype=dtype)


def _columns(values):
    if not isinstance(values, Mapping):
        raise TypeError("columns must be a mapping of names to arrays")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("column names must be nonempty strings")
    data = {name: _vector(column, np.uint32, _U32_MAX, name, "biu") for name, column in values.items()}
    if len({len(column) for column in data.values()}) > 1:
        raise ValueError("all columns must have the same length")
    return data


def _bound(value):
    if not isinstance(value, Integral) or not 0 <= value <= _U32_MAX:
        raise ValueError("filter bounds must be uint32 integers")
    return int(value)


class _Segment:
    """A native table and one mutable bit per physical row."""

    def __init__(self, name, data, start, stop, ids=None):
        table = algq.Table(name, data)
        try:
            table.drop_plain()
            self._words = table.used_bm.copy()
            self.bitmap = self._words.view()
        except BaseException:
            table.close()
            raise
        self.table, self.start, self.stop, self.ids = table, start, stop, ids
        self.rows = self.live = table.n

    def visible(self, row):
        return bool((int(self.bitmap[row >> 6]) >> (row & 63)) & 1)

    def clear(self, rows):
        rows = np.asarray(rows, np.uint64)
        bits = np.left_shift(np.uint64(1), rows & 63)
        np.bitwise_and.at(self._words, (rows >> 6).astype(np.intp), ~bits)
        # GPU delta caches use ndarray identity. Give each visibility revision a
        # new view so device reads refresh without copying the host bitmap.
        self.bitmap = self._words.view()
        self.live -= len(rows)

    def active_rows(self):
        if self.live == self.rows:
            return None
        return np.flatnonzero(np.unpackbits(self.bitmap.view(np.uint8), bitorder="little")[:self.rows])

    def logical_ids(self, rows):
        if self.ids is None:
            return (np.arange(self.rows, dtype=np.uint64) if rows is None else rows.astype(np.uint64)) + self.start
        return self.ids if rows is None else self.ids[rows]


class RefreshTable:
    """A fixed-schema relation whose batches operate on native bitplane segments.

    Initial rows have IDs 0 through N-1. Append returns the contiguous IDs assigned to
    its rows. Update preserves IDs and accepts any subset of columns. Read preserves the
    requested order, including repeats. A mutation requires distinct live IDs. Deleted IDs
    never reappear. Count, sum and grouped sum accept ANDed equalities and inclusive
    intervals, expressed as a mapping from column names to integers or (low, high) pairs.
    Each public operation observes one consistent snapshot under the table lock.
    With update_mode="planes", updates XOR the changed value bits at their current
    physical locations. A batch that needs a wider column uses a replacement segment.
    """

    def __init__(self, columns, *, name="refresh", update_mode="segments"):
        if update_mode not in ("segments", "planes"):
            raise ValueError("update_mode must be 'segments' or 'planes'")
        data = _columns(columns)
        if not 0 < len(data) <= algq.MAXCOL:
            raise ValueError(f"refresh requires between 1 and {algq.MAXCOL} columns")
        self.columns, self.name = tuple(data), name
        self._lock = RLock()
        self._closed = False
        self._update_mode = update_mode
        self._plane_batches = self._plane_rows = self._plane_values = 0
        self._width_fallback_batches = self._width_fallback_rows = 0
        self._segments, self._origins, self._starts = [], [], []
        self._overrides = {}
        self._live = self._next_id = self._last_compact_rewritten_bytes = 0
        self._append(data)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("refresh table is closed")

    def _check_columns(self, names):
        for name in names:
            if name not in self.columns:
                raise KeyError(name)

    def _append(self, data):
        n = len(next(iter(data.values())))
        start, stop = self._next_id, self._next_id + n
        if stop > 1 << 64:
            raise OverflowError("logical row IDs exceed uint64")
        if n:
            segment = _Segment(self.name, data, start, stop)
            self._segments.append(segment)
            self._origins.append(segment)
            self._starts.append(start)
            self._live += n
            self._next_id = stop
        return range(start, stop)

    def append(self, columns):
        """Append a batch with all schema columns and return its range of logical IDs."""
        with self._lock:
            self._check_open()
            data = _columns(columns)
            if data.keys() != set(self.columns):
                raise ValueError("append requires exactly the table's columns")
            return self._append(data)

    def _locate(self, ids, *, distinct=False):
        ids = _vector(ids, np.uint64, (1 << 64) - 1, "row IDs")
        if distinct and len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate row IDs in mutation")
        if not len(ids):
            return ids, {}
        if not self._overrides and self._origins and self._origins[0].ids is None:
            return ids, self._locate_implicit(ids)
        groups = {}
        for i, logical in enumerate(ids.tolist()):
            if logical >= self._next_id:
                raise KeyError(logical)
            item = self._overrides.get(logical)
            if item is not None:
                segment, row = item
            else:
                index = bisect_right(self._starts, logical) - 1
                if index < 0:
                    raise KeyError(logical)
                segment = self._origins[index]
                if segment.ids is None:
                    row = logical - segment.start
                else:
                    row = int(np.searchsorted(segment.ids, np.uint64(logical)))
                    if row >= segment.rows or int(segment.ids[row]) != logical:
                        raise KeyError(logical)
                if logical >= segment.stop or not segment.visible(row):
                    raise KeyError(logical)
            positions, rows = groups.setdefault(segment, ([], []))
            positions.append(i)
            rows.append(row)
        return ids, groups

    def _locate_implicit(self, ids):
        outside = (ids < self._starts[0]) | (ids >= self._next_id)
        if outside.any():
            raise KeyError(int(ids[outside][0]))
        if len(self._origins) == 1:
            pieces = [(self._origins[0], slice(None))]
        else:
            origin = np.searchsorted(np.asarray(self._starts, np.uint64), ids, side="right") - 1
            order = np.argsort(origin, kind="stable")
            stops = np.flatnonzero(np.diff(origin[order])) + 1
            pieces = [(self._origins[origin[pos[0]]], pos) for pos in np.split(order, stops)]
        groups = {}
        for segment, positions in pieces:
            logical = ids[positions]
            rows = logical - segment.start
            if segment.live != segment.rows:
                visible = (segment.bitmap[rows >> 6] >> (rows & 63)) & 1
                if not visible.all():
                    raise KeyError(int(logical[visible == 0][0]))
            groups[segment] = positions, rows
        return groups

    @staticmethod
    def _gather(names, n, groups):
        result = {name: np.empty(n, np.uint32) for name in names}
        for segment, (positions, rows) in groups.items():
            physical = np.asarray(rows, np.uint64)
            for name in names:
                result[name][positions] = segment.table.get_rows(name, physical)
        return result

    def read(self, ids, columns=None):
        """Return selected rows as a dictionary of uint32 arrays in requested ID order."""
        with self._lock:
            self._check_open()
            names = self.columns if columns is None else (columns,) if isinstance(columns, str) else tuple(columns)
            self._check_columns(names)
            ids, groups = self._locate(ids)
            return self._gather(names, len(ids), groups)

    def delete(self, ids):
        """Delete a batch of live IDs."""
        with self._lock:
            self._check_open()
            ids, groups = self._locate(ids, distinct=True)
            for segment, (_, rows) in groups.items():
                segment.clear(rows)
            for logical in ids.tolist():
                self._overrides.pop(logical, None)
            self._live -= len(ids)

    def update(self, ids, columns):
        """Replace specified columns for a batch of distinct live IDs."""
        with self._lock:
            self._check_open()
            data = _columns(columns)
            self._check_columns(data)
            ids, groups = self._locate(ids, distinct=True)
            if any(len(column) != len(ids) for column in data.values()):
                raise ValueError("replacement columns must match the number of row IDs")
            if not len(ids) or not data:
                return
            if self._update_mode == "planes" and self._update_planes(data, groups):
                return
            unchanged = tuple(name for name in self.columns if name not in data)
            previous = self._gather(unchanged, len(ids), groups)
            segment = _Segment(self.name, {**previous, **data}, 0, 0, ids.copy())
            for old, (_, rows) in groups.items():
                old.clear(rows)
            self._segments.append(segment)
            self._overrides.update((logical, (segment, row)) for row, logical in enumerate(ids.tolist()))
            if self._update_mode == "planes":
                self._width_fallback_batches += 1
                self._width_fallback_rows += len(ids)

    def _update_planes(self, data, groups):
        """Prepare every segment's XOR patch before committing any native changes."""
        patches = []
        changed_rows = changed_values = 0
        for segment, (positions, rows) in groups.items():
            table = segment.table
            rows = np.asarray(rows, np.uint64)
            touched = np.zeros(len(rows), bool)
            columns, physical, deltas = [], [], []
            for name, column in data.items():
                new = column[positions]
                if int(new.max()) >= 1 << int(table.width(name)):
                    return False
                delta = table.get_rows(name, rows) ^ new
                changed = delta != 0
                if changed.any():
                    touched |= changed
                    selected = rows[changed]
                    columns.append(np.full(len(selected), table.idx(name), np.int32))
                    physical.append(selected)
                    deltas.append(delta[changed])
                    changed_values += len(selected)
            if columns:
                changed_rows += int(touched.sum())
                patches.append((table, np.concatenate(columns), np.concatenate(physical), np.concatenate(deltas)))
        applied = []
        try:
            for table, columns, rows, deltas in patches:
                table._xor_rows(columns, rows, deltas)
                applied.append((table, columns, rows, deltas))
        except BaseException:
            for table, columns, rows, deltas in reversed(applied):
                table._xor_rows(columns, rows, deltas)
            raise
        if applied:
            self._plane_batches += 1
            self._plane_rows += changed_rows
            self._plane_values += changed_values
        return True

    def _filters(self, where, engine):
        if engine not in ("alg", "gpu"):
            raise ValueError("engine must be 'alg' or 'gpu'")
        if where is None:
            return ()
        if not isinstance(where, Mapping):
            raise TypeError("where must be a mapping of column names to bounds")
        self._check_columns(where)
        terms = []
        for name, value in where.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError("an interval must contain two bounds")
                low, high = map(_bound, value)
                if low > high:
                    raise ValueError("an interval's lower bound exceeds its upper bound")
            else:
                low = high = _bound(value)
            terms.append((name, low, high))
        return terms

    @staticmethod
    def _query(segment, terms):
        table = segment.table
        query = algq.Query(table)
        if segment.live != segment.rows:
            query._xrows = (segment.bitmap, 1)
        for name, low, high in terms:
            col = table[name]
            query.where(algq.Cmp(col, "EQ", low) if low == high else algq.Between(col, low, high))
        return query

    def count(self, where=None, *, engine="alg"):
        """Count visible rows satisfying the filter."""
        with self._lock:
            self._check_open()
            terms = self._filters(where, engine)
            if not terms:
                return self._live
            return sum(int(self._query(seg, terms).count().run(engine)) for seg in self._segments if seg.live)

    def sum(self, column, where=None, *, engine="alg"):
        """Return an exact integer sum, or None when no visible rows match."""
        with self._lock:
            self._check_open()
            self._check_columns((column,))
            terms = self._filters(where, engine)
            result = None
            for seg in self._segments:
                if not seg.live:
                    continue
                value = self._query(seg, terms).sum(seg.table[column]).run(engine)
                if value is not None:
                    result = (0 if result is None else result) + int(value)
            return result

    def group_sum(self, key, column, where=None, *, engine="alg"):
        """Return exact sums keyed by distinct visible values of one column."""
        with self._lock:
            self._check_open()
            self._check_columns((key, column))
            terms = self._filters(where, engine)
            result = {}
            for seg in self._segments:
                if not seg.live:
                    continue
                query = self._query(seg, terms).group_by(seg.table[key]).sum(seg.table[column])
                for (group,), value in query.run(engine).items():
                    group = int(group)
                    result[group] = result.get(group, 0) + int(value)
            return result

    def compact(self):
        """Rewrite the live rows into one bitplane segment, preserving logical IDs."""
        with self._lock:
            self._check_open()
            if not self._overrides and len(self._segments) <= 1 and all(
                seg.rows == seg.live for seg in self._segments
            ):
                self._last_compact_rewritten_bytes = 0
                return
            replacement = None
            if self._live:
                has_holes = self._live != self._next_id
                data = {name: np.empty(self._live, np.uint32) for name in self.columns}
                ids = np.empty(self._live, np.uint64) if has_holes else None
                offset = 0
                for seg in self._segments:
                    if not seg.live:
                        continue
                    rows = seg.active_rows()
                    logical = seg.logical_ids(rows)
                    if ids is None:
                        positions = logical
                    else:
                        positions = slice(offset, offset + seg.live)
                        ids[positions] = logical
                        offset += seg.live
                    for name in self.columns:
                        codes = seg.table._decode_column(name)
                        data[name][positions] = codes if rows is None else codes[rows]
                if ids is not None:
                    order = np.argsort(ids, kind="stable")
                    ids = ids[order]
                    data = {name: column[order] for name, column in data.items()}
                replacement = _Segment(self.name, data, 0, self._next_id, ids)
            old = self._segments
            self._segments = [replacement] if replacement is not None else []
            self._origins = self._segments.copy()
            self._starts = [0] if replacement is not None else []
            self._overrides.clear()
            self._last_compact_rewritten_bytes = self._live * len(self.columns) * np.dtype(np.uint32).itemsize
            for seg in old:
                seg.table.close()

    @property
    def stats(self):
        """Snapshot row counts, native torus bytes, visibility and logical-ID index sizes."""
        with self._lock:
            self._check_open()
            return dict(
                segments=len(self._segments),
                live_rows=self._live,
                physical_rows=sum(seg.rows for seg in self._segments),
                bitmap_bytes=sum(seg.bitmap.nbytes for seg in self._segments),
                torus_bytes=sum(seg.table.torus_bytes() for seg in self._segments),
                id_index_bytes=sum(seg.ids.nbytes for seg in self._segments if seg.ids is not None),
                override_entries=len(self._overrides),
                last_compact_rewritten_bytes=self._last_compact_rewritten_bytes,
                next_id=self._next_id,
                update_mode=self._update_mode,
                plane_update_batches=self._plane_batches,
                plane_update_rows=self._plane_rows,
                plane_update_values=self._plane_values,
                width_fallback_batches=self._width_fallback_batches,
                width_fallback_rows=self._width_fallback_rows,
                gpu_plane_syncs=sum(getattr(seg.table, "_gpu_xor_batches", 0) for seg in self._segments),
                gpu_plane_evictions=sum(getattr(seg.table, "_gpu_xor_evictions", 0) for seg in self._segments),
            )

    def close(self):
        """Release all segments and their device copies."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for seg in self._segments:
                seg.table.close()
            self._segments.clear()
            self._origins.clear()
            self._starts.clear()
            self._overrides.clear()

    def __enter__(self):
        with self._lock:
            self._check_open()
            return self

    def __exit__(self, *exc):
        self.close()
