# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Large objects (CLOB / BLOB) as self-extending records.

A record is its length plus its bytes. The first H bytes (default 16) are K = H/4 limbs, 32-bit big-endian (so byte
order = integer order), stored as bit planes: limb 0 for every row, limb k only for rows longer than 4k bytes (a
virtual column: a flag plane + a sidecar torble addressed by rank, scattered back with PDEP; see torble.h). Bytes past
H go to one contiguous payload heap. The heap offset of a row is not stored: it is dir[w] (one number per 64 rows) plus
the tail lengths of the rows before it in its word, derived from the length planes (at_tail_offsets).

Prefix filters, lengths, types (magic bytes) and groupings on leading bytes are polynomials on the planes and never
touch the payload; fetching an object reads its limbs from the planes and its tail from the heap; substring search
scans the payload (as every other store does). Replacing objects is additive: a delta plane marks the replaced rows
and a small delta store holds their new contents (base minus delta rows, plus the delta store)."""
import numpy as np, ctypes, re
from algq import Table, Query, Col, Cmp, Between, Not, lib, PlaneKey

class BlobStore:
    def __init__(self, objs, H=16, name="blobs"):
        self.H, self.K, self.name = H, H // 4, name
        n = self.n = len(objs)
        L = np.fromiter((len(o) for o in objs), np.uint64, n)
        head = np.frombuffer(b"".join(o[:H].ljust(H, b"\0") for o in objs), ">u4").reshape(n, self.K).astype(np.uint32)
        self.heap = b"".join(o[H:] for o in objs)
        tl = np.maximum(L.astype(np.int64) - H, 0)
        cs = np.concatenate([[0], np.cumsum(tl)])
        nw = (n + 63) // 64
        self.dir = cs[np.arange(nw) * 64].astype(np.uint64)                  # one offset per 64 rows
        cols = {"len": L.astype(np.uint32), "limb0": head[:, 0]}
        virtual = {}
        for k in range(1, self.K):
            cols[f"f{k}"] = (L > 4 * k).astype(np.uint32)
            cols[f"limb{k}"] = head[:, k]
            virtual[f"limb{k}"] = (f"f{k}", True)
        self.t = Table(name, cols, msb=tuple(f"limb{k}" for k in range(self.K)), virtual=virtual)
        self.t.drop_plain()                                                   # the planes are the data
        self.replaced, self.delta, self.xrows = {}, None, None

    # ------------------------------------------------------------ storage
    def bytes_used(self):
        return dict(planes=int(self.t.torus_bytes()), sidecars=int(self.t.sidecar_bytes()), payload=len(self.heap),
                    directory=int(self.dir.nbytes))

    # ------------------------------------------------------------ fetch
    def _get_base(self, rows):
        rows = np.ascontiguousarray(rows, np.uint64)
        L = self.t.get_rows("len", rows).astype(np.int64)
        limbs = np.stack([self.t.get_rows(f"limb{k}", rows) for k in range(self.K)], 1).astype(">u4")
        heads = limbs.tobytes(); off = np.empty(len(rows), np.uint64)
        lib.at_tail_offsets(self.t.at, self.t.idx("len"), self.H, self.dir.ctypes.data, rows.ctypes.data, len(rows), off.ctypes.data)
        H, out, heap = self.H, [], self.heap
        for i in range(len(rows)):
            l = int(L[i]); h = heads[i * H: i * H + min(l, H)]
            out.append(h + heap[int(off[i]): int(off[i]) + l - H] if l > H else h)
        return out
    def get(self, rows):
        rows = np.asarray(rows, np.int64)
        if not self.replaced: return self._get_base(rows)
        out = [None] * len(rows); base_i = [i for i, r in enumerate(rows.tolist()) if r not in self.replaced]
        for i, o in zip(base_i, self._get_base(rows[base_i])): out[i] = o
        di = [i for i in range(len(rows)) if out[i] is None]
        if di:
            for i, o in zip(di, self.delta._get_base(np.array([self.replaced[int(rows[i])] for i in di]))): out[i] = o
        return out

    # ------------------------------------------------------------ predicates on the planes
    def prefix(self, s):
        """starts with s (len(s) <= H): limb equalities / a range on the last partial limb, and len >= len(s)"""
        t = self.t; p = Cmp(Col(t, "len"), "GE", len(s))
        for k in range(0, (len(s) + 3) // 4):
            chunk = s[4 * k: 4 * k + 4]; v = int.from_bytes(chunk.ljust(4, b"\0"), "big"); c = Col(t, f"limb{k}")
            if len(chunk) == 4: p = p & Cmp(c, "EQ", v)
            else: p = p & Between(c, v, v | ((1 << (8 * (4 - len(chunk)))) - 1))
        return p
    def longer(self, n): return Cmp(Col(self.t, "len"), "GT", n)

    # ------------------------------------------------------------ queries (base + delta store, additively)
    def run(self, build, engine="alg"):
        """build(store) -> Query on store.t; with replacements: base without the delta plane + the delta store"""
        q = build(self)
        if self.xrows is not None: q._xrows = (self.xrows, 2)
        r = q.run(engine)
        if self.delta is None: return r
        rd = self.delta.run(build, engine)
        if isinstance(r, dict):
            out = dict(r)
            for k, v in rd.items(): out[k] = out.get(k, 0) + v
            return out
        if q.measure is None: return r + rd
        return None if r is None and rd is None else (r or 0) + (rd or 0)

    def count_prefix(self, s, engine="alg"): return self.run(lambda st: Query(st.t).where(st.prefix(s)).count(), engine)
    def length_stats(self, big, engine="alg"):
        return (self.run(lambda st: Query(st.t).sum(Col(st.t, "len")), engine),
                self.run(lambda st: Query(st.t).where(st.longer(big)).count(), engine))
    def group_leading(self, limb, nbytes, engine="alg"):
        """GROUP BY nbytes leading bytes of a limb (a projection of its top planes): {bytes: (count, total length)}"""
        W = self.t.width(f"limb{limb}")                                 # msb layout: plane j holds bit W-1-j, and
        nb = W - (32 - 8 * nbytes)                                       # bits above W-1 are zero (not stored)
        mask = (1 << nb) - 1 if nb > 0 else 0                            # the planes of bits 31..32-8n that exist
        mk = lambda st: PlaneKey(st.t, f"limb{limb}", mask if st is self else st._lead_mask(limb, nbytes))
        cnt = self.run(lambda st: Query(st.t).group_by(mk(st)).count(), engine)
        tot = self.run(lambda st: Query(st.t).group_by(mk(st)).sum(Col(st.t, "len")), engine)
        return {int(k[0]).to_bytes(nbytes, "big"): (cnt[k], tot.get(k, 0)) for k in cnt}

    def _lead_mask(self, limb, nbytes):
        nb = self.t.width(f"limb{limb}") - (32 - 8 * nbytes); return (1 << nb) - 1 if nb > 0 else 0

    # ------------------------------------------------------------ payload scan
    def _contains_base(self, pat, exclude=None):
        """rows containing pat: matches inside a tail (heap scan), plus matches within the head or across head|tail"""
        n, H, m = self.n, self.H, len(pat)
        rows_all = np.arange(n, dtype=np.uint64); off = np.empty(n, np.uint64)
        lib.at_tail_offsets(self.t.at, self.t.idx("len"), H, self.dir.ctypes.data, rows_all.ctypes.data, n, off.ctypes.data)
        L = np.empty(n, np.uint32); lib.at_decode_column(self.t.at, self.t.idx("len"), L.ctypes.data)
        L = L.astype(np.int64); off = off.astype(np.int64); tl = np.maximum(L - H, 0)
        hit = np.zeros(n, bool)
        pos = np.array([mm.start() for mm in re.finditer(re.escape(pat), self.heap)], np.int64)
        if len(pos):
            r = np.searchsorted(off, pos, "right") - 1
            ok = pos + m <= off[r] + tl[r]; hit[r[ok]] = True
        # head windows: the first H bytes from the planes and up to m - 1 bytes of the tail
        limbs = np.empty((n, self.K), np.uint32)
        for k in range(self.K):
            col = np.empty(n, np.uint32); lib.at_decode_column(self.t.at, self.t.idx(f"limb{k}"), col.ctypes.data); limbs[:, k] = col
        heads = np.frombuffer(limbs.astype(">u4").tobytes(), np.uint8).reshape(n, H).astype(np.int16)
        width = H + m - 1; win = np.full((n, width), -1, np.int16); win[:, :H] = heads
        heap = np.frombuffer(self.heap, np.uint8)
        for j in range(m - 1):
            has = tl > j; win[has, H + j] = heap[off[has] + j]
        valid = np.arange(width)[None, :] < L[:, None]; win[~valid] = -1
        p = np.frombuffer(pat, np.uint8).astype(np.int16)
        for sft in range(width - m + 1):
            hit |= np.all(win[:, sft: sft + m] == p[None, :], axis=1)
        if exclude is not None: hit[list(exclude)] = False
        return hit
    def count_contains(self, pat):
        c = int(self._contains_base(pat, self.replaced.keys()).sum())
        return c + (self.delta.count_contains(pat) if self.delta is not None else 0)

    # ------------------------------------------------------------ replacing objects (additive)
    def replace(self, rows, objs):
        """new contents for these rows: the base is not rewritten; a delta plane marks them and a small delta store
        (itself a BlobStore) holds the new contents"""
        old = {r: self.delta._get_base(np.array([i]))[0] for r, i in self.replaced.items()} if self.delta is not None else {}
        for r, o in zip(np.asarray(rows).tolist(), objs): old[int(r)] = o
        keys = sorted(old); self.replaced = {r: i for i, r in enumerate(keys)}
        self.delta = BlobStore([old[r] for r in keys], self.H, self.name + "_delta")
        x = np.zeros((self.n + 63) // 64, np.uint64)
        for r in keys: x[r >> 6] |= np.uint64(1) << np.uint64(r & 63)
        self.xrows = x
