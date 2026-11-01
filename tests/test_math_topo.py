# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TOPO-001 (declared occupancy complexes and local topology) and TOPO-002 (bounded cycles, homology and
persistence) over engine tables and explicit chain complexes (torbles/math_topo.c, math_topo_gpu.cu, math_topo.py).

Oracles, independent of the implementation: hand-enumerated cells, incidences and Euler counts of tiny grids; a
Python enumeration of the closure of the occupied cells (Khalimsky coordinates) with the product orientation;
Gaussian elimination written here over GF(2) (Python ints as bit rows), GF(p) and Q (Fractions); the components of
the dual graph by a graph search written here; brute-force star counts and boundary flags; a Python standard column reduction for persistence
pairs, and the persistent Betti numbers beta_k^{i,j} = rank[Z_k(K_i) | B_k(K_j)] - rank B_k(K_j) computed by ranks
alone. Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled
thread order) and compared for equality. Seeds are fixed.
"""
import ctypes as C
import itertools
import random
import unittest
from fractions import Fraction
from math import prod

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_field as mf
from torbles import math_handles as mh
from torbles import math_registry as reg
from torbles import math_topo as mt
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError, DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

P61 = (1 << 61) - 1


# ---------------------------------------------------------------------- oracles: cells and incidences
def oracle_complex(occ, ext):
    """the closure of the occupied positions occ (0-based tuples) of an extent ext: {k: [Khalimsky tuples]} in
    increasing linear index, the index of each cell, and the incidences (faces, signs) per k"""
    d = len(ext)
    K = [2 * n + 1 for n in ext]
    stride = [prod(K[:a]) for a in range(d)]
    cells = set()
    for p in occ:
        t = [2 * i + 1 for i in p]
        for off in itertools.product((-1, 0, 1), repeat=d):
            cells.add(tuple(t[a] + off[a] for a in range(d)))

    def lin(c):
        return sum(c[a] * stride[a] for a in range(d))
    by = {k: [] for k in range(d + 1)}
    for c in cells:
        by[sum(x & 1 for x in c)].append(c)
    for k in by:
        by[k].sort(key=lin)
    index = {k: {c: i for i, c in enumerate(by[k])} for k in by}
    inc = {}
    for k in range(1, d + 1):
        faces, signs = [], []
        for c in by[k]:
            f, s = [], []
            for i, a in enumerate([a for a in range(d) if c[a] & 1]):
                minus = tuple(c[b] - (b == a) for b in range(d))
                plus = tuple(c[b] + (b == a) for b in range(d))
                f += [index[k - 1][minus], index[k - 1][plus]]
                s += [-(-1) ** i, (-1) ** i]
            faces.append(f)
            signs.append(s)
        inc[k] = (faces, signs)
    return dict(cells=by, index=index, inc=inc, lin=lin, ext=ext, occ=set(occ))


def dense_boundary(oc, k):
    """d_k as a list of integer rows (n_{k-1} x n_k)"""
    rows, cols = len(oc["cells"][k - 1]), len(oc["cells"][k])
    D = [[0] * cols for _ in range(rows)]
    faces, signs = oc["inc"][k]
    for j in range(cols):
        for f, s in zip(faces[j], signs[j]):
            D[f][j] += s
    return D


# ---------------------------------------------------------------------- oracles: exact linear algebra
def rank_q(M):
    """rank over Q of a list of integer rows (Fractions, Gaussian elimination)"""
    A = [[Fraction(x) for x in r] for r in M]
    rank, cols = 0, len(A[0]) if A else 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(A)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        for i in range(len(A)):
            if i != rank and A[i][c] != 0:
                f = A[i][c] / A[rank][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[rank])]
        rank += 1
    return rank


def rank_p(M, p):
    """rank over GF(p) (p prime; p = 2 included) of integer rows"""
    A = [[x % p for x in r] for r in M]
    rank, cols = 0, len(A[0]) if A else 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = pow(A[rank][c], p - 2, p)
        A[rank] = [x * inv % p for x in A[rank]]
        for i in range(len(A)):
            if i != rank and A[i][c]:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[rank])]
        rank += 1
    return rank


def rank_gf2_bits(rows):
    """rank over GF(2) of Python ints read as bit rows (an elimination of its own)"""
    basis = {}
    for r in rows:
        while r:
            top = r.bit_length() - 1
            if top not in basis:
                basis[top] = r
                break
            r ^= basis[top]
    return len(basis)


def rank_field(M, field):
    if not M or not M[0]:
        return 0
    if field == "q":
        return rank_q(M)
    if field == "gf2":
        return rank_gf2_bits([sum(1 << j for j, x in enumerate(r) if x % 2) for r in M])
    return rank_p(M, field)


def transpose(M, rows, cols):
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def betti_oracle(sizes, mats, field):
    """b_k = n_k - rank d_k - rank d_{k+1} from dense matrices mats[k] = d_k (k = 1 .. top)"""
    top = len(sizes) - 1
    ranks = [0] + [rank_field(mats[k], field) if sizes[k - 1] and sizes[k] else 0 for k in range(1, top + 1)] + [0]
    return [sizes[k] - ranks[k] - ranks[k + 1] for k in range(top + 1)], ranks


def nullspace_q(M, cols):
    """a basis of {x : M x = 0} over Q (lists of Fractions)"""
    A = [[Fraction(x) for x in r] for r in M]
    pivots, rank = [], 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(A)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        A[rank] = [x / A[rank][c] for x in A[rank]]
        for i in range(len(A)):
            if i != rank and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[rank])]
        pivots.append(c)
        rank += 1
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * cols
        v[f] = Fraction(1)
        for i, c in enumerate(pivots):
            v[c] = -A[i][f]
        basis.append(v)
    return basis


def nullspace_mod(M, cols, p):
    A = [[x % p for x in r] for r in M]
    pivots, rank = [], 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = pow(A[rank][c], p - 2, p)
        A[rank] = [x * inv % p for x in A[rank]]
        for i in range(len(A)):
            if i != rank and A[i][c]:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[rank])]
        pivots.append(c)
        rank += 1
    basis = []
    for f in [c for c in range(cols) if c not in pivots]:
        v = [0] * cols
        v[f] = 1
        for i, c in enumerate(pivots):
            v[c] = (-A[i][f]) % p
        basis.append(v)
    return basis


# ---------------------------------------------------------------------- oracles: persistence
def reference_pairs(sizes, mats, levels, ties, field):
    """the standard column reduction written here: (pairs, essential) sorted as the native contract says"""
    top = len(sizes) - 1
    p = None if field in ("q", "gf2") else field
    order, pos = {}, {}
    for k in range(top + 1):
        cand = [i for i in range(sizes[k]) if levels[k][i] != -1]
        cand.sort(key=lambda i: (levels[k][i], i if ties == "index" else -i))
        order[k] = cand
        pos[k] = {c: q for q, c in enumerate(cand)}
    pairs, killed = [], {k: set() for k in range(top + 1)}
    positive = {k: set() for k in range(top + 1)}
    positive[0] = set(range(len(order[0])))
    for k in range(1, top + 1):
        owner = {}
        for jp, j in enumerate(order[k]):
            col = {}
            for i in range(sizes[k - 1]):
                x = mats[k][i][j]
                if field == "gf2":
                    x %= 2
                elif p:
                    x %= p
                if x:
                    col[pos[k - 1][i]] = Fraction(x) if field == "q" else x
            while col:
                low = max(col)
                if low not in owner:
                    break
                other = owner[low]
                if field == "gf2":
                    for r in other:
                        if r in col:
                            del col[r]
                        else:
                            col[r] = 1
                else:
                    f = col[low] / other[low] if field == "q" else col[low] * pow(other[low], p - 2, p) % p
                    for r, y in other.items():
                        v = col.get(r, 0) - f * y
                        if field != "q":
                            v %= p
                        if v:
                            col[r] = v
                        else:
                            col.pop(r, None)
            if col:
                low = max(col)
                owner[low] = dict(col)
                b = order[k - 1][low]
                pairs.append((k - 1, levels[k - 1][b], levels[k][j], b, j))
                killed[k - 1].add(low)
            else:
                positive[k].add(jp)
    essential = []
    for k in range(top + 1):
        for jp in sorted(positive[k] - killed[k]):
            c = order[k][jp]
            essential.append((k, levels[k][c], c))
    return sorted(pairs), sorted(essential)


def persistent_betti_all(sizes, mats, levels, m, field):
    """{(k, i, j): beta_k^{i,j}} for i <= j < m, by ranks alone: rank[Z_k(K_i) | B_k(K_j)] - rank B_k(K_j)"""
    top = len(sizes) - 1
    pp = None if field == "q" else (2 if field == "gf2" else field)

    def rk(vectors):
        if not vectors:
            return 0
        return rank_q(vectors) if pp is None else rank_p([[int(x) for x in v] for v in vectors], pp)

    def entered(k, lvl):
        return [c for c in range(sizes[k]) if levels[k][c] != -1 and levels[k][c] <= lvl]
    out = {}
    for k in range(top + 1):
        Z = {}
        for i in range(m):                                  # Z_k(K_i): absolute k-cell coordinates
            ini = entered(k, i)
            if k >= 1 and ini:
                Dk = [[mats[k][r][c] for c in ini] for r in entered(k - 1, i)]
                basis = (nullspace_q(Dk, len(ini)) if pp is None else nullspace_mod(Dk, len(ini), pp)) if Dk else \
                    [[int(a == b) for a in range(len(ini))] for b in range(len(ini))]
            else:
                basis = [[int(a == b) for a in range(len(ini))] for b in range(len(ini))]
            Z[i] = [{c: z[q] for q, c in enumerate(ini) if z[q]} for z in basis]
        for j in range(m):
            inj = entered(k, j)
            B = [[mats[k + 1][r][c] for r in inj] for c in entered(k + 1, j)] if k < top else []
            rB = rk(B)
            for i in range(j + 1):
                Zv = [[z.get(c, 0) for c in inj] for z in Z[i]]
                out[(k, i, j)] = rk(Zv + B) - rB
    return out


# ---------------------------------------------------------------------- oracles: local invariants, dual graph
def oracle_local(oc):
    """per cell (dimension-major): star counts, link Euler characteristic, boundary flag"""
    d = len(oc["ext"])
    allcells = {c: k for k in oc["cells"] for c in oc["cells"][k]}
    out = []
    for k in range(d + 1):
        for c in oc["cells"][k]:
            star = [0, 0, 0, 0]
            for t, kt in allcells.items():
                if all(abs(t[a] - c[a]) <= 1 and (c[a] % 2 == 0 or t[a] == c[a]) for a in range(d)):
                    star[kt] += 1
            link = sum((-1) ** (j - k - 1) * star[j] for j in range(k + 1, d + 1))
            boundary = False
            if k < d:
                deg = [a for a in range(d) if c[a] % 2 == 0]
                for eps in itertools.product((-1, 1), repeat=len(deg)):
                    t = list(c)
                    for a, e in zip(deg, eps):
                        t[a] += e
                    inside = all(1 <= t[a] <= 2 * oc["ext"][a] - 1 for a in range(d))
                    if not inside or tuple((x - 1) // 2 for x in t) not in oc["occ"]:
                        boundary = True
            out.append((star, link, boundary, k))
    return out


class DualGraph:
    """an undirected simple graph on the occupied top cells, with its edges and connected components (a depth-first
    search): the oracle of the dual-graph adjacency"""

    def __init__(self, nodes):
        self.adj = {p: set() for p in nodes}

    def add_edge(self, p, q):
        self.adj[p].add(q)
        self.adj[q].add(p)

    def number_of_edges(self):
        return sum(len(v) for v in self.adj.values()) // 2

    def edges(self):
        return [(p, q) for p in self.adj for q in self.adj[p] if p < q]

    def connected_components(self):
        left, out = set(self.adj), []
        while left:
            start = left.pop()
            comp, stack = {start}, [start]
            while stack:
                for q in self.adj[stack.pop()]:
                    if q in left:
                        left.discard(q)
                        comp.add(q)
                        stack.append(q)
            out.append(comp)
        return out


def dual_graph(occ, d, adjacency):
    G = DualGraph(occ)
    for p in occ:
        for off in itertools.product((-1, 0, 1), repeat=d):
            nz = sum(1 for o in off if o)
            if not nz or (adjacency in (4, 6) and nz != 1) or (adjacency == 18 and nz > 2):
                continue
            q = tuple(a + o for a, o in zip(p, off))
            if q in occ:
                G.add_edge(p, q)
    return G


# ---------------------------------------------------------------------- tables
def obj(values):
    return pd.Series(values, dtype=object)


def make_table(points, values=None, extra=None, order=None):
    """a torbles table of coordinate tuples (None entries: NULL), an optional value column v and extra columns"""
    d = len(points[0]) if points else 2
    names = ["x", "y", "z"][:d]
    cols = {n: obj([p[a] if p[a] is None else int(p[a]) for p in points]) for a, n in enumerate(names)}
    if values is not None:
        cols["v"] = obj(values)
    for name, vals in (extra or {}).items():
        cols[name] = obj(vals)
    order = order or list(cols)
    ddl = "CREATE TABLE t (" + ", ".join(f"{n} BIGINT" for n in order) + ")"
    return torbles.from_ddl(ddl, {"t": pd.DataFrame({n: cols[n] for n in order})})


def scope_of(db, d=2, value=False, extra=()):
    names = ["x", "y", "z"][:d] + (["v"] if value else []) + list(extra)
    return mbk.Scope.from_table(db["t"], names)


def grid_of(ext, adjacency=None, value=None, lo=None):
    d = len(ext)
    lo = lo or [0] * d
    return mt.Grid(["x", "y", "z"][:d], [(lo[a], lo[a] + ext[a] - 1) for a in range(d)],
                   adjacency or (8 if d == 2 else 26), value=value)


def random_occ(rng, ext, q):
    return {p for p in itertools.product(*[range(n) for n in ext]) if rng.random() < q}


def complex_of(occ, ext, adjacency=None, values=None, backend_check=True):
    """the native complex of an occupancy (CPU, checked equal to the device build) and its oracle"""
    pts = sorted(occ)
    db = make_table(pts or [tuple([None] * len(ext))], values if values is not None else None)
    scope = scope_of(db, len(ext), value=values is not None)
    grid = grid_of(ext, adjacency, value="v" if values is not None else None)
    cx = mt.Complex.build(scope, grid)
    if backend_check:
        gx = mt.Complex.build(scope, grid, backend="cuda")
        assert cx.equal(gx), "device build differs"
        assert gx.backend in ("cuda", "cuda-emulated")
    return cx, oracle_complex(occ, ext), (db, scope, grid)


def shape_2d(rows):
    """occupied positions of a picture: rows of strings, '#' occupied, row 0 at y = 0"""
    return {(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "#"}


def complex_matrices(cx):
    sizes = cx.counts
    mats = {k: cx.boundary(k).dense().tolist() for k in range(1, len(sizes))}
    return sizes, mats


class Topo(unittest.TestCase):
    def check_against_oracle(self, cx, oc):
        d = len(oc["ext"])
        self.assertEqual(cx.counts, [len(oc["cells"][k]) for k in range(d + 1)])
        for k in range(d + 1):
            cells = cx.cells(k)
            want_anchor = [[c[a] // 2 + 0 for a in range(d)] for c in oc["cells"][k]]
            want_axes = [sum(1 << a for a in range(d) if c[a] & 1) for c in oc["cells"][k]]
            lo = [cx.grid.extent[a][0] for a in range(d)]
            self.assertEqual(cells["anchor"].tolist(), [[x + lo[a] for a, x in enumerate(r)] for r in want_anchor], k)
            self.assertEqual(cells["axes"].tolist(), want_axes, k)
            self.assertEqual(cells["linear"].tolist(), [oc["lin"](c) for c in oc["cells"][k]], k)
            if k >= 1:
                faces, signs = cx.incidence(k)
                self.assertEqual(faces.tolist(), oc["inc"][k][0], k)
                self.assertEqual(signs.tolist(), oc["inc"][k][1], k)
        n = [len(oc["cells"][k]) for k in range(d + 1)]
        self.assertEqual(cx.euler, sum((-1) ** k * n[k] for k in range(d + 1)))

    def local_both(self, cx):
        cpu, gpu = cx.local(), cx.local(backend="cuda")
        for key in ("star", "link_euler", "boundary", "degree", "dim"):
            self.assertEqual(cpu[key].tolist(), gpu[key].tolist(), key)
        self.assertEqual(cpu["totals"], gpu["totals"])
        self.assertIn(gpu["backend"], ("cuda", "cuda-emulated"))
        return cpu

    def check_both(self, cx):
        cpu, gpu = cx.check(), cx.check(backend="cuda")
        for key in ("z", "gf2", "invalid", "terms"):
            self.assertEqual(cpu[key], gpu[key], key)
        return cpu

    # ================================================================== TOPO-001
    def test_topo001_hand_enumerated_cells_incidences_and_euler_counts(self):
        # one unit square at (0, 0): 4 vertices, 4 edges, 1 square
        cx, oc, _ = complex_of({(0, 0)}, (1, 1))
        self.assertEqual(cx.counts, [4, 4, 1])
        self.assertEqual(cx.euler, 1)
        self.assertEqual(cx.cells(0)["anchor"].tolist(), [[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertEqual(cx.cells(1)["anchor"].tolist(), [[0, 0], [0, 0], [1, 0], [0, 1]])
        self.assertEqual(cx.cells(1)["axes"].tolist(), [1, 2, 2, 1])          # bottom, left, right, top
        faces, signs = cx.incidence(1)
        self.assertEqual(faces.tolist(), [[0, 1], [0, 2], [1, 3], [2, 3]])
        self.assertEqual(signs.tolist(), [[-1, 1]] * 4)                        # d [a, b] = b - a
        faces, signs = cx.incidence(2)
        self.assertEqual(faces.tolist(), [[1, 2, 0, 3]])                       # left, right, bottom, top
        self.assertEqual(signs.tolist(), [[-1, 1, 1, -1]])                     # d = right - left - top + bottom
        self.check_against_oracle(cx, oc)
        # a domino (two squares sharing an edge), an L of three, a pinch (two squares meeting at a corner)
        for occ, ext, n in (({(0, 0), (1, 0)}, (2, 1), [6, 7, 2]),
                            ({(0, 0), (1, 0), (0, 1)}, (2, 2), [8, 10, 3]),
                            ({(0, 0), (1, 1)}, (2, 2), [7, 8, 2]),
                            ({(0, 0), (2, 0)}, (3, 1), [8, 8, 2])):        # two disjoint squares: chi = 2
            cx, oc, _ = complex_of(occ, ext)
            self.assertEqual(cx.counts, n, occ)
            self.assertEqual(cx.euler, n[0] - n[1] + n[2])
            self.check_against_oracle(cx, oc)
        # one unit cube: 8, 12, 6, 1; its six faces in the canonical order and signs of the product orientation
        cx, oc, _ = complex_of({(0, 0, 0)}, (1, 1, 1))
        self.assertEqual(cx.counts, [8, 12, 6, 1])
        self.assertEqual(cx.euler, 1)
        self.assertEqual(cx.cells(2)["axes"].tolist(), [3, 5, 6, 6, 5, 3])
        faces, signs = cx.incidence(3)
        self.assertEqual(faces.tolist(), [[2, 3, 1, 4, 0, 5]])                # x-, x+, y-, y+, z-, z+
        self.assertEqual(signs.tolist(), [[-1, 1, 1, -1, -1, 1]])
        self.check_against_oracle(cx, oc)
        # two cubes sharing a face: 12, 20, 11, 2 (chi = 1); a 2x2x2 block: 27, 54, 36, 8 (chi = 1)
        cx, oc, _ = complex_of({(0, 0, 0), (1, 0, 0)}, (2, 1, 1))
        self.assertEqual(cx.counts, [12, 20, 11, 2])
        cx, oc, _ = complex_of(set(itertools.product(range(2), repeat=3)), (2, 2, 2))
        self.assertEqual(cx.counts, [27, 54, 36, 8])
        self.check_against_oracle(cx, oc)

    def test_topo001_random_grids_match_the_enumerated_closure(self):
        rng = random.Random(1201)
        for trial in range(10):
            d = 2 if trial < 6 else 3
            ext = tuple(rng.randint(1, 7 if d == 2 else 4) for _ in range(d))
            occ = random_occ(rng, ext, rng.choice((0.25, 0.5, 0.8)))
            cx, oc, (db, scope, grid) = complex_of(occ, ext)
            self.check_against_oracle(cx, oc)
            c_cpu, c_gpu = mt.counts(scope, grid), mt.counts(scope, grid, backend="cuda")
            for key in ("rows", "cells", "boundary", "euler", "boundary_euler", "positions", "candidates"):
                self.assertEqual(c_cpu[key], c_gpu[key], key)
            self.assertEqual(c_cpu["cells"], cx.counts)
            self.assertEqual(c_cpu["euler"], cx.euler)
        # grids of several device chunks (TR_TOPO_CHUNK candidates each): the prefix offsets of the device build
        for ext, q in (((24, 17), 0.4), ((7, 6, 5), 0.3), ((40, 6), 0.9)):
            self.assertGreater(prod(2 * n + 1 for n in ext), 4 * 256)
            cx, oc, _ = complex_of(random_occ(rng, ext, q), ext)
            self.check_against_oracle(cx, oc)

    def test_topo001_duplicates_nulls_and_rows_outside_the_extent_are_counted_apart(self):
        pts = [(0, 0), (0, 0), (1, 0), (None, 2), (2, None), (5, 0), (-1, 1), (1, 1), (1, 1), (1, 1)]
        vals = [3, None, 4, 1, 1, 9, 9, 7, 2, None]
        db = make_table(pts, vals)
        scope = scope_of(db, 2, value=True)
        grid = grid_of((3, 2), value="v")
        for backend in ("cpu", "cuda"):
            cx = mt.Complex.build(scope, grid, backend=backend)
            self.assertEqual(cx.rows, dict(selected=10, null_rows=2, outside_rows=2, valid_rows=6, occupied=3,
                                           duplicates=3, value_nulls=2, valued=3))
            self.assertEqual(cx.counts, [8, 10, 3])
            v = cx.values()
            self.assertEqual(v["min"].tolist(), [3, 4, 2])                     # top cells (0,0), (1,0), (1,1)
            self.assertEqual(v["max"].tolist(), [3, 4, 7])
        # a selection: only the selected rows count
        mask = np.array([True, False, True, True, False, False, False, True, False, False])
        cx = mt.Complex.build(mbk.Scope.from_table(db["t"], ["x", "y"], rows=mask), grid_of((3, 2)))
        self.assertEqual(cx.rows["selected"], 4)
        self.assertEqual((cx.rows["null_rows"], cx.rows["valid_rows"], cx.rows["occupied"]), (1, 3, 3))
        # an empty extent occupancy: no cells at all
        db = make_table([(9, 9)])
        cx = mt.Complex.build(scope_of(db), grid_of((3, 3)))
        self.assertEqual((cx.counts, cx.euler, cx.rows["outside_rows"]), ([0, 0, 0], 0, 1))

    def test_topo001_adjacency_components_match_a_graph_search(self):
        rng = random.Random(1202)
        for trial in range(8):
            d = 2 if trial < 5 else 3
            ext = tuple(rng.randint(2, 8 if d == 2 else 4) for _ in range(d))
            occ = random_occ(rng, ext, 0.35)
            cx, oc, _ = complex_of(occ, ext)
            tops = oc["cells"][d]
            for adjacency in mt.ADJACENCY[d]:
                a = cx.adjacency(adjacency, pairs=True)
                G = dual_graph(occ, d, adjacency)
                self.assertEqual((a["edges"], a["components"]), (G.number_of_edges(), len(G.connected_components())))
                index = {tuple((x - 1) // 2 for x in c): i for i, c in enumerate(tops)}
                want = sorted(tuple(sorted((index[p], index[q]))) for p, q in G.edges())
                self.assertEqual([tuple(p) for p in a["pairs"].tolist()], want)
                for comp in G.connected_components():
                    ids = sorted(index[p] for p in comp)
                    self.assertEqual({int(a["labels"][i]) for i in ids}, {ids[0]})
            # with 8 / 26 adjacency the components are exactly H_0 of the union
            h = mt.homology(cx, field="gf2", cycles=())
            self.assertEqual(cx.adjacency()["components"] if occ else 0, h.betti[0])
            loc = self.local_both(cx)
            self.assertEqual(loc["totals"]["dual_edges"], cx.adjacency()["edges"])

    def test_topo001_boundary_cells_and_local_star_link_counts_match_brute_force(self):
        rng = random.Random(1203)
        cases = [(shape_2d(["##", "#."]), (2, 2)), (shape_2d(["#.", ".#"]), (2, 2)),
                 (shape_2d(["###", "#.#", "###"]), (3, 3))]
        for _ in range(4):
            ext = (rng.randint(2, 6), rng.randint(2, 6))
            cases.append((random_occ(rng, ext, 0.5), ext))
        for _ in range(2):
            ext = (3, 3, rng.randint(2, 3))
            cases.append((random_occ(rng, ext, 0.5), ext))
        cases.append(({p for p in itertools.product(range(2), repeat=3) if sum(p) % 2 == 0}, (2, 2, 2)))   # checkerboard
        for occ, ext in cases:
            cx, oc, _ = complex_of(occ, ext)
            loc = self.local_both(cx)
            want = oracle_local(oc)
            self.assertEqual(loc["star"].tolist(), [w[0] for w in want])
            self.assertEqual(loc["link_euler"].tolist(), [w[1] for w in want])
            self.assertEqual(loc["boundary"].tolist(), [w[2] for w in want])
            self.assertEqual(loc["dim"].tolist(), [w[3] for w in want])
            d = len(ext)
            nb = [sum(1 for w in want if w[3] == k and w[2]) for k in range(d + 1)]
            self.assertEqual(loc["totals"]["boundary"], nb)
            self.assertEqual(cx.info["boundary"], nb)
            self.assertEqual(cx.info["boundary_euler"], sum((-1) ** k * nb[k] for k in range(d + 1)))
        # the pinch: the shared vertex has two arcs as its link (chi 2), an interior vertex a circle (chi 0)
        cx, oc, _ = complex_of(shape_2d(["#.", ".#"]), (2, 2))
        i = oc["index"][0][(2, 2)]
        self.assertEqual((cx.local()["star"][i].tolist(), int(cx.local()["link_euler"][i])), ([1, 4, 2, 0], 2))
        cx, oc, _ = complex_of(shape_2d(["###", "###", "###"]), (3, 3))
        i = oc["index"][0][(2, 2)]
        self.assertEqual((cx.local()["star"][i].tolist(), int(cx.local()["link_euler"][i]), bool(cx.local()["boundary"][i])),
                         ([1, 4, 4, 0], 0, False))

    def test_topo001_storage_torus_and_code_order_is_never_the_geometry(self):
        rng = np.random.default_rng(1204)
        occ = random_occ(random.Random(1204), (6, 5), 0.45)
        pts = sorted(occ)
        base = make_table(pts, values=list(range(len(pts))))
        cx = mt.Complex.build(scope_of(base, value=True), grid_of((6, 5), value="v"))
        # (1) the physical row order is not geometry: shuffled rows, the same complex
        perm = rng.permutation(len(pts))
        shuffled = make_table([pts[i] for i in perm], values=[int(i) for i in perm])
        self.assertTrue(cx.equal(mt.Complex.build(scope_of(shuffled, value=True), grid_of((6, 5), value="v"))))
        # (2) the torus layout (the stored column order) is not geometry: the same columns declared in another order
        moved = make_table(pts, values=list(range(len(pts))), order=["v", "y", "x"])
        self.assertNotEqual(base["t"].idx("x"), moved["t"].idx("x"))
        self.assertTrue(cx.equal(mt.Complex.build(scope_of(moved, value=True), grid_of((6, 5), value="v"))))
        # (3) the stored codes are not geometry: one far row outside the extent shifts every code of x
        shifted = make_table(pts + [(-1000, 0)], values=list(range(len(pts))) + [0])
        s0, s1 = scope_of(base, value=True), scope_of(shifted, value=True)
        self.assertNotEqual(s0.column("x")["offset"], s1.column("x")["offset"])
        other = mt.Complex.build(s1, grid_of((6, 5), value="v"))
        self.assertEqual((other.counts, other.euler, other.rows["outside_rows"]), (cx.counts, cx.euler, 1))
        for k in range(3):
            self.assertEqual(other.cells(k)["anchor"].tolist(), cx.cells(k)["anchor"].tolist())
        # (4) bit planes are not adjacency: x = 7 and x = 8 differ in four planes and touch; 0 and 8 differ in one
        # plane and do not
        line = make_table([(7, 0), (8, 0), (0, 0)])
        c4 = mt.Complex.build(scope_of(line), mt.Grid(["x", "y"], [(0, 8), (0, 0)], 4))
        a = c4.adjacency(pairs=True)
        self.assertEqual((a["components"], a["pairs"].tolist()), (2, [[1, 2]]))   # tops 0 (x=0), 1 (x=7), 2 (x=8)
        self.assertEqual(mt.homology(c4, field="gf2").betti[0], 2)
        # (5) the declared roles are the geometry: (y, x) gives the transposed complex
        t = mt.Complex.build(scope_of(base, value=True), mt.Grid(["y", "x"], [(0, 4), (0, 5)], 8))
        oc = oracle_complex({(y, x) for x, y in occ}, (5, 6))
        self.check_against_oracle(t, oc)
        # (6) only the declared columns are read: another value in a column that is not declared changes nothing
        a1 = make_table(pts, extra={"w": list(range(len(pts)))})
        a2 = make_table(pts, extra={"w": [7] * len(pts)})
        self.assertTrue(mt.Complex.build(scope_of(a1, extra=("w",)), grid_of((6, 5))).equal(
            mt.Complex.build(scope_of(a2, extra=("w",)), grid_of((6, 5)))))
        # (7) a layout built from other columns is refused: the complex's FND-007 record names its columns
        s_yx = mbk.Scope.from_table(base["t"], ["y", "x", "v"])
        f = cx.fresh(s_yx)
        self.assertFalse(f.fresh)
        self.assertIn("columns", f.reasons)
        with self.assertRaises(StaleStateError) as ctx:
            cx.require_fresh(s_yx)
        self.assertIn("columns", ctx.exception.reasons)
        self.assertIn("columns", cx.fresh(mbk.Scope.from_table(base["t"], ["x"])).reasons)   # a column missing
        self.assertTrue(cx.fresh(scope_of(base, value=True)).fresh)                          # the same layout
        # (8) only exact integer columns are coordinates: decimal, Boolean, text and a repeated column are refused
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, d DECIMAL(10,2), b BOOLEAN, s VARCHAR)",
                              {"t": pd.DataFrame({"x": obj([0, 1]), "y": obj([0, 0]), "d": obj(["1.50", "2.00"]),
                                                  "b": obj([True, False]), "s": obj(["a", "b"])})})
        sc = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        with self.assertRaises(DomainError):
            mt.Complex.build(sc, mt.Grid(["x", "d"], [(0, 1), (0, 300)], 8))         # native: decimal scale 2
        with self.assertRaises(DomainError):
            mt.Complex.build(sc, mt.Grid(["x", "b"], [(0, 1), (0, 1)], 8))           # Boolean codec
        with self.assertRaises(UnsupportedError):
            mbk.Scope.from_table(db["t"], ["s"])                                      # text has no logical value
        with self.assertRaises(ValueError):
            mt.Grid(["x", "x"], [(0, 1), (0, 1)], 8)
        with self.assertRaises(ValueError):
            mt.Grid(["x", "y"], [(0, 1), (0, 1)], 6)                                  # a 3D adjacency on a 2D grid
        arr = mbk.Scope.from_arrays({"x": [0, 1], "y": [0, 0]}, scales={"y": 1})
        with self.assertRaises(DomainError):
            mt.Complex.build(arr, mt.Grid(["x", "y"], [(0, 1), (0, 1)], 8))           # an array column of scale 1

    def test_topo001_euler_counts_without_the_complex_agree_on_cpu_and_cuda(self):
        rng = random.Random(1205)
        for d, ext in ((2, (16, 12)), (3, (5, 4, 6))):
            occ = random_occ(rng, ext, 0.3)
            pts = sorted(occ)
            db = make_table(pts + [(None,) * d] + [tuple([-5] * d)])
            scope = scope_of(db, d)
            grid = grid_of(ext)
            cpu, gpu = mt.counts(scope, grid), mt.counts(scope, grid, backend="cuda")
            oc = oracle_complex(occ, ext)
            n = [len(oc["cells"][k]) for k in range(d + 1)]
            self.assertEqual(cpu["cells"], n)
            self.assertEqual(cpu["euler"], sum((-1) ** k * n[k] for k in range(d + 1)))
            self.assertEqual((cpu["rows"]["null_rows"], cpu["rows"]["outside_rows"]), (1, 1))
            for key in ("rows", "cells", "boundary", "euler", "boundary_euler"):
                self.assertEqual(cpu[key], gpu[key])
            self.assertEqual(gpu["backend"], "cuda-emulated")

    def test_topo001_budgets_at_the_limit_and_one_past(self):
        occ = random_occ(random.Random(1206), (6, 6), 0.5)
        db = make_table(sorted(occ), values=list(range(len(occ))))
        scope, grid = scope_of(db, value=True), grid_of((6, 6), value="v")
        for backend in ("cpu", "cuda"):
            need = mt.need(scope, grid, backend=backend)
            whole = {}                                     # each phase checks its own needs: the largest decides
            for phase in ("scan", "build", "adopt"):
                for r, v in need.get(phase, {}).items():
                    whole[r] = max(whole.get(r, 0), v)
            self.assertEqual("adopt" in need, backend == "cuda")
            for resource in ("variables", "states", "output", "tensor", "bytes"):
                limit = whole[resource]
                mt.Complex.build(scope, grid, backend=backend, budget=Budget(**{resource: limit}))
                with self.assertRaises(AlgebraBudgetError, msg=(backend, resource)) as ctx:
                    mt.Complex.build(scope, grid, backend=backend, budget=Budget(**{resource: limit - 1}))
                self.assertEqual(ctx.exception.resource, resource)
        # the counts alone need the scan only
        n = mt.need(scope, grid)["scan"]
        mt.counts(scope, grid, budget=Budget(states=36, bytes=n["bytes"]))
        with self.assertRaises(AlgebraBudgetError):
            mt.counts(scope, grid, budget=Budget(states=35))
        with self.assertRaises(AlgebraBudgetError):
            mt.counts(scope, grid, budget=Budget(bytes=n["bytes"] - 1))
        # a grid past the representation limit, with a budget that admits it
        huge = Budget(states=1 << 40, bytes=1 << 62)
        with self.assertRaises(UnsupportedError):
            mt.counts(scope, mt.Grid(["x", "y"], [(0, 1 << 15), (0, 1 << 14)], 8), budget=huge)
        # the default budget refuses a 2^25-position extent before allocating anything
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mt.Complex.build(scope, mt.Grid(["x", "y"], [(0, (1 << 13) - 1), (0, (1 << 12) - 1)], 8))
        self.assertEqual(ctx.exception.resource, "states")
        # local invariants and the dual graph: their outputs
        cx = mt.Complex.build(scope, grid)
        total = sum(cx.counts)
        cx.local(budget=Budget(output=total))
        with self.assertRaises(AlgebraBudgetError):
            cx.local(budget=Budget(output=total - 1))
        with self.assertRaises(AlgebraBudgetError):
            cx.local(backend="cuda", budget=Budget(output=total - 1))
        cx.adjacency(budget=Budget(output=cx.counts[2]))
        with self.assertRaises(AlgebraBudgetError):
            cx.adjacency(budget=Budget(output=cx.counts[2] - 1))

    def test_topo001_stale_scope_is_refused(self):
        db = make_table([(0, 0), (1, 0), (1, 1)])
        t = db["t"]
        scope = scope_of(db)
        cx = mt.Complex.build(scope, grid_of((2, 2)))
        self.assertTrue(cx.fresh().fresh)
        t.flip(0, 0)                                       # a mutation: a new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mt.Complex.build(scope, grid_of((2, 2)), backend=backend)
            with self.assertRaises(StaleStateError):
                mt.counts(scope, grid_of((2, 2)), backend=backend)
        for call in (lambda: cx.counts, lambda: cx.cells(0), lambda: cx.local(), lambda: cx.adjacency(),
                     lambda: cx.check(), lambda: cx.boundary(1), lambda: cx.fresh()):
            with self.assertRaises(StaleStateError):
                call()
        fresh_scope = scope_of(db)
        f = cx.fresh(fresh_scope)
        self.assertFalse(f.fresh)
        self.assertIn("revision", f.reasons)

    def test_topo001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_topo.")}
        self.assertEqual(mine, {"tr_topo_grid", "tr_topo_rows", "tr_topo_counts", "tr_topo_view", "tr_topo_info",
                                "tr_topo_local", "tr_topo_local_totals", "tr_topo_adjacency", "tr_topo_dd",
                                "tr_topo_field", "tr_topo_filtration", "tr_topo_homology_info", "tr_topo_class",
                                "tr_topo_persistence_info", "tr_topo_pair", "tr_topo_chain_info", "tr_topo_states"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_topo.")}
        self.assertEqual(cuda, {"tr_topo_grid", "tr_topo_rows", "tr_topo_counts", "tr_topo_states", "tr_topo_view",
                                "tr_topo_local", "tr_topo_local_totals", "tr_topo_dd"})
        self.assertTrue(all(r["match"] for r in report["checked"]))

    # ================================================================== TOPO-002
    def test_topo002_boundary_of_boundary_is_zero_over_z_and_gf2(self):
        rng = random.Random(1301)
        for trial in range(8):
            d = 2 if trial < 4 else 3
            ext = tuple(rng.randint(1, 6 if d == 2 else 4) for _ in range(d))
            cx, oc, _ = complex_of(random_occ(rng, ext, 0.6), ext)
            res = self.check_both(cx)
            self.assertTrue(res["z"]["holds"] and res["gf2"]["holds"])
            self.assertEqual(res["z"]["checked"][2:d + 1], cx.counts[2:])
            self.assertEqual(res["terms"], sum(2 * k * 2 * (k - 1) * cx.counts[k] for k in range(2, d + 1)))
            for k in range(2, d + 1):                                # the oracle: the integer product is zero
                if cx.counts[k] and cx.counts[k - 2]:
                    A, B = cx.boundary(k - 1).dense(), cx.boundary(k).dense()
                    self.assertFalse((A @ B).any())
                    G1, G2 = cx.boundary(k - 1, "gf2"), cx.boundary(k, "gf2")
                    self.assertEqual(G1.rows, cx.counts[k - 2])
                    self.assertFalse((G1 @ G2).to_bits().any())      # FIELD-001's product over GF(2)
                    self.assertEqual(G2.to_bits().tolist(), (np.abs(B) % 2).tolist())
                    R = cx.boundary(k, 5)
                    self.assertEqual(R.values.tolist(), [x % 5 for x in cx.boundary(k).values.tolist()])
        # the checker sees corrupted incidences: a flipped sign breaks Z (not GF(2)); a wrong face breaks both
        cx, oc, _ = complex_of(shape_2d(["##", "##"]), (2, 2))
        v = cx.view()
        faces, signs = cx.incidence(2)
        bad_sign = signs.copy()
        bad_sign[2, 1] = -bad_sign[2, 1]
        bad_face = faces.copy()
        bad_face[3, 0] = faces[0, 0]
        for f, s, z, two, cell in ((faces, bad_sign, False, True, 2), (bad_face, signs, False, False, 3)):
            fa, sa = np.ascontiguousarray(f, dtype=np.uint32), np.ascontiguousarray(s, dtype=np.int8)
            w = mt._View.from_buffer_copy(v)
            w.face[2], w.sign[2] = fa.ctypes.data, sa.ctypes.data
            cpu, gpu = mt.check_view(w), mt.check_view(w, backend="cuda")
            self.assertEqual((cpu["z"], cpu["gf2"]), (gpu["z"], gpu["gf2"]))
            self.assertEqual((cpu["z"]["holds"], cpu["gf2"]["holds"]), (z, two))
            self.assertEqual(cpu["z"]["witness"][2], cell)
        # an explicit "chain complex" whose composition is not zero is refused
        with self.assertRaises(DomainError):
            mt.ChainComplex([[[1, 1]], [[1], [1]]])

    def test_topo002_tiny_known_complexes_yield_expected_ranks_and_betti_numbers(self):
        ring = shape_2d(["###", "#.#", "###"])
        known = [
            ("annulus", ring, (3, 3), [1, 1, 0]),
            ("two squares", {(0, 0), (2, 0)}, (3, 1), [2, 0, 0]),
            ("two diamonds", shape_2d([".#...#.", "#.#.#.#", ".#...#."]), (7, 3), [2, 2, 0]),
            ("empty 2D", set(), (3, 3), [0, 0, 0]),
            ("full 2D", set(itertools.product(range(4), range(3))), (4, 3), [1, 0, 0]),
            ("shell with a cavity", {p for p in itertools.product(range(3), repeat=3) if p != (1, 1, 1)}, (3, 3, 3),
             [1, 0, 1, 0]),
            ("3D ring", {(x, y, 0) for x, y in ring}, (3, 3, 1), [1, 1, 0, 0]),
            ("solid torus", {(x, y, z) for x, y in ring for z in range(2)}, (3, 3, 2), [1, 1, 0, 0]),
            ("empty 3D", set(), (2, 2, 2), [0, 0, 0, 0]),
            ("full 3D", set(itertools.product(range(3), range(2), range(2))), (3, 2, 2), [1, 0, 0, 0]),
        ]
        for name, occ, ext, betti in known:
            cx, oc, _ = complex_of(occ, ext)
            sizes, mats = complex_matrices(cx)
            for field in ("gf2", 3, P61, "q"):
                h = mt.homology(cx, field=field)
                self.assertEqual(h.betti, betti, (name, field))
                want, ranks = betti_oracle(sizes, mats, field)
                self.assertEqual(h.betti, want, (name, field))
                self.assertEqual(h.ranks, ranks, (name, field))
                self.assertEqual(sum((-1) ** k * b for k, b in enumerate(h.betti)), cx.euler)   # Euler-Poincare
            self.assertEqual(mt.homology(cx, field="gf2", method="packed").betti, betti, name)
            self.assertEqual(len(mt.homology(cx, field="q", cycles=(1,)).cycles[1]), betti[1], name)
        # explicit chain complexes where the field matters (integral torsion)
        rp2 = mt.ChainComplex([[[0]], [[2]]])                 # one cell per dimension, d_2 = 2, d_1 = 0
        klein = mt.ChainComplex([[[0, 0]], [[0], [2]]])        # word a b a^-1 b: d_2 = (0, 2)
        torus = mt.ChainComplex([[[0, 0]], [[0], [0]]])
        for cc, expect in ((rp2, {"gf2": [1, 1, 1], 3: [1, 0, 0], "q": [1, 0, 0]}),
                           (klein, {"gf2": [1, 2, 1], 3: [1, 1, 0], "q": [1, 1, 0]}),
                           (torus, {"gf2": [1, 2, 1], 3: [1, 2, 1], "q": [1, 2, 1]})):
            for field, b in expect.items():
                self.assertEqual(mt.homology(cc, field=field).betti, b, (cc, field))
            self.assertEqual(mt.homology(cc, field="gf2", method="packed").betti, expect["gf2"])
        # the 6-vertex RP^2 (triangulated, oriented over Z): GF(2) 1, 1, 1; Q 1, 0, 0
        tris = [tuple(sorted(t)) for t in ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 2, 4),
                                           (2, 3, 5), (3, 4, 1), (4, 5, 2), (5, 1, 3))]
        edges = sorted({tuple(sorted(e)) for t in tris for e in itertools.combinations(t, 2)})
        eidx = {e: i for i, e in enumerate(edges)}
        d1 = [[0] * len(edges) for _ in range(6)]
        for j, (a, b) in enumerate(edges):
            d1[a][j], d1[b][j] = -1, 1
        d2 = [[0] * len(tris) for _ in edges]
        for j, (a, b, c) in enumerate(tris):
            d2[eidx[(b, c)]][j] += 1
            d2[eidx[(a, c)]][j] -= 1
            d2[eidx[(a, b)]][j] += 1
        rp2s = mt.ChainComplex([d1, d2])
        self.assertEqual(mt.homology(rp2s, field="gf2").betti, [1, 1, 1])
        self.assertEqual(mt.homology(rp2s, field="q").betti, [1, 0, 0])
        self.assertEqual(mt.homology(rp2s, field=3).betti, [1, 0, 0])
        self.assertEqual(betti_oracle([6, len(edges), len(tris)], {1: d1, 2: d2}, "gf2")[0], [1, 1, 1])

    def test_topo002_random_ranks_and_betti_match_exact_python_elimination(self):
        rng = random.Random(1302)
        for trial in range(8):
            d = 2 if trial < 5 else 3
            ext = tuple(rng.randint(2, 6 if d == 2 else 3) for _ in range(d))
            cx, oc, _ = complex_of(random_occ(rng, ext, rng.choice((0.25, 0.5, 0.75))), ext)
            sizes, mats = complex_matrices(cx)
            for k in range(1, d + 1):
                self.assertEqual(mats[k], dense_boundary(oc, k))
            for field in ("gf2", 5, "q"):
                want, ranks = betti_oracle(sizes, mats, field)
                for method in (("reduction", "packed") if field == "gf2" else ("reduction",)):
                    h = mt.homology(cx, field=field, method=method)
                    self.assertEqual((h.betti, h.ranks), (want, ranks), (trial, field, method))
            for k in range(1, d + 1):                     # FIELD-001's own rank of the packed matrix
                self.assertEqual(cx.boundary(k, "gf2").rank(), rank_field(mats[k], "gf2"))

    def test_topo002_cycle_representatives_form_a_homology_basis(self):
        rng = random.Random(1303)
        cases = [(shape_2d(["#####", "#.#.#", "#####"]), (5, 3)),                       # two holes
                 (shape_2d([".#...#.", "#.#.#.#", ".#...#."]), (7, 3)),
                 ({p for p in itertools.product(range(3), repeat=3) if p != (1, 1, 1)}, (3, 3, 3))]
        for _ in range(3):
            ext = (rng.randint(3, 6), rng.randint(3, 6))
            cases.append((random_occ(rng, ext, 0.6), ext))
        for occ, ext in cases:
            cx, oc, _ = complex_of(occ, ext)
            sizes, mats = complex_matrices(cx)
            top = len(sizes) - 1
            for field in ("gf2", 7, "q"):
                h = mt.homology(cx, field=field, cycles=tuple(range(1, top + 1)))
                for k in range(1, top + 1):
                    reps = h.cycles.get(k, [])
                    self.assertEqual(len(reps), h.betti[k], (field, k))
                    p = 2 if field == "gf2" else (None if field == "q" else field)
                    vecs = []
                    for r in reps:
                        z = [0] * sizes[k]
                        for c, a in zip(r["cells"], r["coefficients"]):
                            z[c] = a
                        self.assertIn(r["generator"], r["cells"])
                        g = r["coefficients"][r["cells"].index(r["generator"])]
                        self.assertEqual(g, 1 if p else g)
                        if p is None:
                            self.assertGreater(g, 0)
                        dz = [sum(mats[k][i][j] * z[j] for j in range(sizes[k])) for i in range(sizes[k - 1])]
                        self.assertTrue(all((x % p == 0) if p else x == 0 for x in dz), (field, k))   # a cycle
                        vecs.append(z)
                    # independent modulo the boundaries: rank [B | Z] = rank B + b_k
                    B = transpose(mats[k + 1], sizes[k], sizes[k + 1]) if k < top else []

                    def rk(M):
                        return 0 if not M else (rank_q(M) if p is None else rank_p(M, p))
                    self.assertEqual(rk(B + vecs), rk(B) + len(vecs), (field, k))
        # the annulus' H_1 generator is an edge cycle around the hole
        cx, oc, _ = complex_of(shape_2d(["###", "#.#", "###"]), (3, 3))
        h = mt.homology(cx, field="gf2")
        self.assertEqual(len(h.cycles[1]), 1)
        edges = cx.cells(1)
        deg = {}
        for c in h.cycles[1][0]["cells"]:
            a, axes = edges["anchor"][c].tolist(), int(edges["axes"][c])
            for end in (tuple(a), (a[0] + (axes == 1), a[1] + (axes == 2))):
                deg[end] = deg.get(end, 0) + 1
        self.assertTrue(all(v == 2 for v in deg.values()))       # every vertex of the cycle meets two of its edges

    def test_topo002_persistence_pairs_match_the_python_reduction_and_rank_invariants(self):
        rng = random.Random(1304)
        for trial in range(9):
            d = 2 if trial < 7 else 3
            ext = tuple(rng.randint(2, 5) for _ in range(2)) if d == 2 else (2, 2, rng.randint(2, 3))
            occ = random_occ(rng, ext, 0.7)
            if not occ:
                continue
            pts = sorted(occ)
            vals = [rng.randint(0, 9) for _ in pts]
            value_of = dict(zip(pts, vals))
            cx, oc, _ = complex_of(occ, ext, values=vals)
            sizes, mats = complex_matrices(cx)
            thresholds = sorted(rng.sample(range(0, 10), rng.randint(2, 4)))
            for direction in ("sublevel", "superlevel"):
                ts = thresholds if direction == "sublevel" else thresholds[::-1]
                # the oracle's levels: a top cell enters at the first admitting threshold, a face with its earliest
                # coface (the closure of the declared levels)
                levels = {d: []}
                for c in oc["cells"][d]:
                    x = value_of[tuple((t - 1) // 2 for t in c)]
                    ok = [l for l, t in enumerate(ts) if (x <= t if direction == "sublevel" else x >= t)]
                    levels[d].append(ok[0] if ok else -1)
                for k in range(d - 1, -1, -1):
                    levels[k] = [-1] * sizes[k]
                    for j, f in enumerate(oc["inc"][k + 1][0]):
                        l = levels[k + 1][j]
                        for x in f:
                            if l != -1 and (levels[k][x] == -1 or l < levels[k][x]):
                                levels[k][x] = l
                m = len(ts)
                betti = {fld: persistent_betti_all(sizes, mats, levels, m, fld) for fld in ("gf2", "q")}
                diagrams = {}
                for ties in ("index", "reverse"):
                    for field in ("gf2", 3, "q"):
                        p = mt.persistence(cx, field=field, ties=ties, thresholds=ts, direction=direction)
                        self.assertEqual([p.levels(k).tolist() for k in range(d + 1)], [levels[k] for k in range(d + 1)])
                        rp, re = reference_pairs(sizes, mats, levels, ties, field)
                        got = sorted((q["dim"], q["birth"], q["death"], q["birth_cell"], q["death_cell"]) for q in p.pairs)
                        self.assertEqual(got, rp, (trial, field, ties, direction))
                        self.assertEqual(sorted((e["dim"], e["birth"], e["cell"]) for e in p.essential), re)
                        self.assertEqual(p.info["zero_length"], sum(1 for q in p.pairs if q["birth"] == q["death"]))
                        diagrams[(ties, field)] = sorted(p.diagram(), key=lambda t: (t[0], t[1] is None, t[1] or 0))
                        if field in betti:              # persistent Betti numbers of the pairs equal the rank formula
                            for (k, i, j), want in betti[field].items():
                                have = sum(1 for q in p.pairs if q["dim"] == k and q["birth"] <= i and q["death"] > j) + \
                                    sum(1 for e in p.essential if e["dim"] == k and e["birth"] <= i)
                                self.assertEqual(have, want, (trial, field, ties, k, i, j))
                        for level in range(m):
                            self.assertEqual(p.betti(level), [betti["q"][(k, level, level)] for k in range(d + 1)])
                # the diagram (pairs with birth < death, essential classes) does not depend on the tie rule
                self.assertEqual(len({tuple(x) for x in diagrams.values()}), 1)
        # explicit levels on an explicit chain: the filtration of a triangle's vertices, edges and face
        tri = mt.ChainComplex([[[-1, -1, 0], [1, 0, -1], [0, 1, 1]], [[1], [-1], [1]]])
        p = mt.persistence(tri, field="q", ties="index", levels=[[0, 0, 1], [1, 2, 3], [4]], nlevels=5)
        self.assertEqual(sorted(p.diagram(), key=lambda t: (t[0], t[1] is None, t[1] or 0)),
                         [(0, 1), (0, None), (1, 2), (3, 4)])
        with self.assertRaises(DomainError):                    # a face after its coface: not a filtration
            mt.persistence(tri, field="q", ties="index", levels=[[0, 0, 3], [1, 2, 3], [4]], nlevels=5)
        with self.assertRaises(ValueError):                     # a level past the declared bound
            mt.persistence(tri, field="q", ties="index", levels=[[0, 0, 1], [1, 2, 3], [5]], nlevels=5)

    def test_topo002_field_orientation_ties_matrix_size_and_reduction_budgets_are_explicit(self):
        occ = random_occ(random.Random(1305), (6, 6), 0.6)
        vals = list(range(len(occ)))
        cx, oc, _ = complex_of(occ, (6, 6), values=vals)
        with self.assertRaises(TypeError):
            mt.homology(cx)                                      # no default field
        with self.assertRaises(TypeError):
            mt.persistence(cx, field="gf2", thresholds=[3])      # no default tie rule
        with self.assertRaises(ValueError):
            mt.persistence(cx, field="gf2", ties="lexicographic", thresholds=[3])
        with self.assertRaises(UnsupportedError):
            mt.homology(cx, field="z")                           # integral homology (torsion) is not claimed
        with self.assertRaises(ValueError):
            mt.homology(cx, field=15)                            # not a prime (verified natively)
        with self.assertRaises(UnsupportedError):
            mt.homology(cx, field=(1 << 62) + 135)               # GF(p) needs p < 2^62
        with self.assertRaises(ValueError):
            mt.homology(cx, field=5, method="packed")            # FIELD-001's packed path is GF(2)
        with self.assertRaises(ValueError):
            mt.homology(cx, field="gf2", method="packed", cycles=(1,))   # representatives need the reduction
        with self.assertRaises(ValueError):
            mt.persistence(cx, field="gf2", ties="index", thresholds=[5, 3])   # sublevel thresholds increase
        self.assertEqual(mt.homology(cx, field=2).field, "gf2")
        # orientation: the signs make d o d = 0 over Z; with every sign +1 the "boundary" is no chain map over Q
        faces, signs = cx.incidence(2)
        self.assertTrue(set(np.unique(signs).tolist()) <= {-1, 1})
        D1, D2 = cx.boundary(1).dense(), cx.boundary(2).dense()
        self.assertFalse((D1 @ D2).any())
        self.assertTrue((np.abs(D1) @ np.abs(D2)).any())
        # matrix size (states, tensor), levels (support) and the reduction's steps and bytes: limit and one past
        h = mt.homology(cx, field="q", cycles=(1,))
        need = mt.homology_need(cx, field="q")
        self.assertEqual(need["states"], sum(cx.counts))
        self.assertEqual(need["tensor"], 2 * cx.counts[1] + 4 * cx.counts[2])
        for resource in ("states", "tensor", "support"):
            mt.homology(cx, field="q", budget=Budget(**{resource: need[resource]}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mt.homology(cx, field="q", budget=Budget(**{resource: need[resource] - 1}))
            self.assertEqual(ctx.exception.resource, resource)
        for field in ("gf2", 3, "q"):
            h = mt.homology(cx, field=field, cycles=(1,))
            steps, peak, out = h.steps, h.peak_bytes, h.info["classes"] + h.info["entries"]
            self.assertGreater(steps, 0)
            again = mt.homology(cx, field=field, cycles=(1,), budget=Budget(iterations=steps, bytes=peak, output=out))
            self.assertEqual((again.betti, again.cycles), (h.betti, h.cycles))
            for resource, limit in (("iterations", steps - 1), ("bytes", peak - 1), ("output", out - 1)):
                with self.assertRaises(AlgebraBudgetError) as ctx:      # nothing is returned: no partial claim
                    mt.homology(cx, field=field, cycles=(1,), budget=Budget(**{resource: limit}))
                self.assertEqual(ctx.exception.resource, resource)
                if resource == "iterations":
                    self.assertEqual((ctx.exception.need, ctx.exception.limit), (steps, steps - 1))
        packed = mt.homology_need(cx, field="gf2", method="packed")
        self.assertEqual(packed["tensor"], max(cx.counts[0] * cx.counts[1], cx.counts[1] * cx.counts[2]))
        with self.assertRaises(AlgebraBudgetError):
            mt.homology(cx, field="gf2", method="packed", budget=Budget(tensor=packed["tensor"] - 1))
        ts = [2, 9, 15, 30]
        p = mt.persistence(cx, field="gf2", ties="index", thresholds=ts)
        pn = mt.persistence_need(cx, field="gf2", ties="index", thresholds=ts)
        self.assertEqual(pn["support"], 4)
        mt.persistence(cx, field="gf2", ties="index", thresholds=ts,
                       budget=Budget(support=4, iterations=p.info["steps"], bytes=p.info["peak_bytes"],
                                     output=len(p.pairs) + len(p.essential)))
        for resource, limit in (("support", 3), ("iterations", p.info["steps"] - 1), ("bytes", p.info["peak_bytes"] - 1),
                                ("output", len(p.pairs) + len(p.essential) - 1)):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mt.persistence(cx, field="gf2", ties="index", thresholds=ts, budget=Budget(**{resource: limit}))
            self.assertEqual(ctx.exception.resource, resource)
        # explicit chain complexes: their own matrix-size budget
        with self.assertRaises(AlgebraBudgetError):
            mt.ChainComplex([[[1, -1]]], budget=Budget(tensor=1))
        # Q refuses an entry past int64 instead of wrapping (column 1 - column 0 holds 2^63); GF(p) has no such limit
        big = 1 << 62
        cc = mt.ChainComplex([[[-big, big], [0, 1], [1, 1]]])
        with self.assertRaises(OverflowError):
            mt.homology(cc, field="q")
        self.assertEqual(mt.homology(cc, field=3).betti, betti_oracle([3, 2], {1: [[-big, big], [0, 1], [1, 1]]}, 3)[0])
        self.assertEqual(mt.homology(cc, field="gf2").betti, mt.homology(cc, field="gf2", method="packed").betti)

    def test_topo002_no_homology_claim_follows_from_popcount(self):
        # a diamond (four squares around a hole) and a separate square, against a diagonal chain of five squares:
        # the same occupancy popcount, the same V, E, F and Euler characteristic, different Betti numbers
        a = shape_2d([".#.....", "#.#..#.", ".#....."])
        b = {(i, i) for i in range(5)}
        ca, oa, _ = complex_of(a, (7, 3))
        cb, ob, _ = complex_of(b, (5, 5))
        self.assertEqual((ca.rows["occupied"], ca.counts, ca.euler), (cb.rows["occupied"], cb.counts, cb.euler))
        self.assertEqual((ca.counts, ca.euler), ([16, 20, 5], 1))
        ha, hb = mt.homology(ca, field="gf2"), mt.homology(cb, field="gf2")
        self.assertEqual((ha.betti, hb.betti), ([2, 1, 0], [1, 0, 0]))
        for cx, oc, want in ((ca, oa, [2, 1, 0]), (cb, ob, [1, 0, 0])):
            sizes, mats = complex_matrices(cx)
            self.assertEqual(betti_oracle(sizes, mats, "q")[0], want)
        # their packed boundary matrices have the same popcounts (every edge column 2 bits) and different ranks
        A, B = ca.boundary(1, "gf2"), cb.boundary(1, "gf2")
        self.assertEqual(A.column_counts(), B.column_counts())
        self.assertNotEqual(A.rank(), B.rank())
        # the counts carry no homology: no Betti number in a complex's info or in the Euler counts
        self.assertFalse(any("betti" in k for k in ca.info))
        self.assertFalse(hasattr(ca, "betti"))
        self.assertFalse(any("betti" in k for k in mt.counts(ca.scope, ca.grid)))
        # and the registry refuses a rank step over the popcount carrier: counts form no field
        with self.assertRaises(CarrierError) as ctx:
            reg.check_chain([("popcount", "rowset"), ("solve", "count")], input="rowset", contract="exact")
        self.assertEqual(ctx.exception.refusal["law"], "inverse")
        reg.check_chain([("convert", "integer"), ("residue", "gf2"), ("solve", "gf2")], input="code", contract="exact")

    def test_topo002_stale_complex_is_refused_by_homology_and_persistence(self):
        db = make_table([(0, 0), (1, 0), (1, 1)], values=[1, 2, 3])
        scope = scope_of(db, value=True)
        cx = mt.Complex.build(scope, grid_of((2, 2), value="v"))
        h = mt.homology(cx, field="gf2")
        self.assertTrue(h.fresh().fresh)
        db["t"].flip(0, 1)
        with self.assertRaises(StaleStateError):
            mt.homology(cx, field="gf2")
        with self.assertRaises(StaleStateError):
            mt.persistence(cx, field="gf2", ties="index", thresholds=[1, 2])
        with self.assertRaises(StaleStateError):
            mt.homology_need(cx, field="q")

    def test_topo002_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        rows = [r for r in report["checked"] if r["structure"].startswith("torbles.math_topo.")]
        self.assertTrue(rows and all(r["match"] for r in rows))
        for native in ("tr_topo_field", "tr_topo_filtration", "tr_topo_pair", "tr_topo_class", "tr_topo_dd"):
            self.assertIn(native, {r["native"] for r in rows})


if __name__ == "__main__":
    unittest.main()
