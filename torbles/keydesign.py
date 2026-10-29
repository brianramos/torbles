# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Automatic key design: find the hierarchy in a dimension and encode it into the foreign keys that reference it.

A join is cheap when the dimension attributes a query filters or groups on are fields of the foreign key's bits:
then a filter is a few cubes or comparators on the fact's planes, and a GROUP BY reads its key straight from them.
That is what the hand-designed SSB keys did (region | nation | city | key). Here it is found from the data:

1. Candidates: the dimension's attributes with few distinct values compared with its rows (codes, order-preserving).
2. Functional dependencies: A -> B when every value of A has a single value of B (city -> nation -> region). Tested on
   the full data by sorting on A and checking B is constant on every run.
3. The hierarchy is the longest chain of dependencies A1 <- A2 <- ... <- Ak whose finest level determines the most
   attributes (every attribute the finest level determines becomes a function of the key's bits, not only the chain).
4. The key: each level is ranked within its parent (order-preserving), the ranks concatenated into a cell code, and the
   derived surrogate is  cell << w | row  where row is the dimension row (the rank of its primary key, so the key itself
   stays a projection of the low w bits). If the fields do not fit 32 bits with the row, the cell is replaced by its
   dense rank (still order-preserving: every prefix of the hierarchy stays one interval); then levels are dropped from
   the fine end; with none left, the foreign key is the plain row number.

The result is only a layout: the planner (JoinPoly, attr_coords) finds which bits each attribute depends on by itself."""
import numpy as np

MAX_CANDIDATE_DISTINCT = 1 << 20


def _fd_matrix(codes, names):
    """fd[a][b] = True when attribute a determines attribute b (on every row)"""
    fd = {a: {} for a in names}
    for a in names:
        o = np.argsort(codes[a], kind="stable"); sa = codes[a][o]
        starts = np.concatenate([[0], np.nonzero(np.diff(sa))[0] + 1]) if len(sa) else np.zeros(0, np.int64)
        for b in names:
            if a == b: fd[a][b] = True; continue
            sb = codes[b][o]
            fd[a][b] = bool(len(sa) == 0 or np.array_equal(np.maximum.reduceat(sb, starts), np.minimum.reduceat(sb, starts)))
    return fd


def level_rank(parent, lv):
    """rank of each row's value within its parent cell (order-preserving): (rank per row, bits needed)"""
    key = (parent.astype(np.int64) << 32) | lv.astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    first = np.r_[True, (u[1:] >> 32) != (u[:-1] >> 32)] if len(u) else np.zeros(0, bool)
    pos = np.arange(len(u))
    rank = pos - np.maximum.accumulate(np.where(first, pos, 0)) if len(u) else pos
    return rank[np.asarray(inv).ravel()], max(1, int(rank.max()).bit_length()) if len(u) else 1


def cell_bits(cell, attr):
    """the bits of the cell code an attribute depends on (most significant first), or None if the attribute is not a
    function of them. Cells are dense (ranks within parents), so flipping a bit usually lands on another cell: bit k
    matters when some pair of cells differing only in k has different values. The result is then verified."""
    uc, idx = np.unique(cell, return_index=True); val = np.asarray(attr)[idx]
    if not len(uc): return []
    # the attribute must be a function of the cell at all
    o = np.argsort(cell, kind="stable"); cs, vs = cell[o], np.asarray(attr)[o]
    st = np.concatenate([[0], np.nonzero(np.diff(cs))[0] + 1])
    if not np.array_equal(np.maximum.reduceat(vs, st), np.minimum.reduceat(vs, st)): return None
    nb = max(1, int(uc.max()).bit_length()); R = []
    for k in range(nb):
        f = uc ^ (1 << k); pos = np.minimum(np.searchsorted(uc, f), len(uc) - 1); ex = uc[pos] == f
        if np.any(val[ex] != val[pos[ex]]): R.append(k)
    R.sort(reverse=True)
    code = np.zeros(len(uc), np.int64)
    for k in R: code = (code << 1) | ((uc >> k) & 1)
    o = np.argsort(code, kind="stable"); c2, v2 = code[o], val[o]
    st = np.concatenate([[0], np.nonzero(np.diff(c2))[0] + 1])
    if np.array_equal(np.maximum.reduceat(v2, st), np.minimum.reduceat(v2, st)): return R
    return list(range(nb - 1, -1, -1))                # not separable: every cell bit


def cells(levels):
    """the hierarchy as concatenated codes, each level ranked within its parent: (cell per row, field widths)"""
    n = len(levels[0]); parent = np.zeros(n, np.int64); widths = []
    for lv in levels:
        r, w = level_rank(parent, lv); widths.append(w); parent = (parent << w) | r
        if parent.size and int(parent.max()) >= 1 << 31: return None, None        # too wide to rank further
    return parent, widths


def design(n, attrs, budget=32, max_candidate=None):
    """n dimension rows (row r = primary-key rank r); attrs: {name: int64 codes per row (order-preserving)}.
    Returns a dict:
        kind     "derived" (s = cell << w | row) or "plain" (the foreign key is the row)
        w        bits of the row part (row n, one past the last, marks a missing reference)
        chain    the hierarchy, coarsest first; fields: {name: bits}; covered: attributes the finest level determines
        s_of_row int64 surrogates of rows 0..n (derived only; s_of_row[n] = n: the code of a missing reference)"""
    w = max(1, int(n).bit_length())
    limit = max_candidate or max(2, min(MAX_CANDIDATE_DISTINCT, n // 2))
    cand = {}
    for a, c in attrs.items():
        c = np.asarray(c, np.int64)
        if len(c) != n or n == 0: continue
        d = len(np.unique(c))
        if 2 <= d <= limit: cand[a] = (c, d)
    plain = dict(kind="plain", w=w, chain=[], fields={}, covered=[], reason="")
    if not cand:
        plain["reason"] = "no attribute with few distinct values"; return plain
    names = sorted(cand, key=lambda a: (cand[a][1], a))
    fd = _fd_matrix({a: cand[a][0] for a in names}, names)
    # equivalence classes (a <-> b): keep one representative; the others are determined by it anyway
    rep = {}
    for a in names:
        rep[a] = next((r for r in rep.values() if fd[a][r] and fd[r][a]), a)
    nodes = sorted({rep[a] for a in names}, key=lambda a: (cand[a][1], a))
    # longest chain ending at each node (coarse -> fine), through dependencies between representatives
    best = {}
    for f in nodes:                                         # increasing cardinality: parents come first
        up = [b for b in nodes if b != f and fd[f][b] and cand[b][1] < cand[f][1]]
        prev = max((best[b] for b in up), key=lambda ch: (len(ch), [-cand[x][1] for x in ch]), default=[])   # longest; coarsest on ties
        best[f] = prev + [f]
    def covered(f): return [a for a in names if fd[f][a]]
    order = sorted(nodes, key=lambda f: (-len(covered(f)), -len(best[f]), cand[f][1], f))
    tried = []
    for f in order:
        chain = list(best[f])
        while chain:
            levels = [cand[a][0] for a in chain]
            cell, widths = cells(levels)
            if cell is not None and sum(widths) + w <= budget:
                fields = dict(zip(chain, widths)); dense = False
            else:
                lv = cell if cell is not None else None
                if lv is None:                                  # rank the full tuple directly
                    key = np.zeros(n, np.int64)
                    for x in levels: key = key * (int(x.max()) + 1) + x
                    lv = key
                u, inv = np.unique(lv, return_inverse=True); cell = np.asarray(inv, np.int64).ravel()
                cb = max(1, int(len(u) - 1).bit_length())
                if cb + w > budget: chain = chain[:-1]; continue
                fields = {"cell rank (" + " > ".join(chain) + ")": cb}; dense = True
            s = np.empty(n + 1, np.int64); s[:n] = (cell << w) | np.arange(n, dtype=np.int64); s[n] = n
            cov = covered(chain[-1])
            coords = {}
            for a in cov:                                     # the surrogate bits each attribute depends on
                R = cell_bits(cell, cand[a][0])
                if R is not None: coords[a] = [w + k for k in R]
            return dict(kind="derived", w=w, chain=chain, fields=fields, dense=dense, covered=cov, coords=coords,
                        bits=int(s.max()).bit_length(), s_of_row=s)
        tried.append(f)
    plain["reason"] = f"the hierarchy does not fit {budget} bits next to the {w}-bit row"
    return plain
