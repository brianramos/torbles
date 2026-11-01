# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TOPO-001 / TOPO-002 benchmark: declared occupancy complexes, local topology, homology and persistence.

    python bench/benchmark_math_topo.py --capability TOPO-001 | TOPO-002 --out FILE [--device] [--quick]

Case 1 (complexes): explicitly declared 2D grids (64^2, 128^2, 256^2; 32^2 with --quick) and 3D grids (16^3, 24^3, 32^3; 8^3)
with four occupancy patterns: empty, full, 25% occupied (uniform, seeded) and known small holes (tiled rings in 2D,
tiled hollow shells in 3D: one hole / cavity per tile). Reported separately: the scope (state construction from the
table), the Euler counts alone (mt.counts), the complex with its cells and incidences (first and warm), then the
local invariants (star / link counts, boundary flags), the dual graph and the boundary-composition check on the built
complex. Baseline: NumPy (the cells per dimension from shifted ORs of the occupancy array) and SciPy (ndimage.label
components with 8 / 26 connectivity); both are compared with the native counts. SciPy is not a declared dependency:
without it the ndimage.label baseline is skipped and noted (the known components of the patterns still check).
Case 2 (homology and persistence): the same holes patterns at four sizes (2D 32^2 .. 192^2, 3D 8^3 .. 24^3: the
boundary-matrix size), each field
(GF(2) by the reduction and through FIELD-001's packed matrices, GF(2^61 - 1), Q), the H_1 (H_2) representatives,
compared with the known Betti numbers of the patterns (holes / cavities per tile); persistence of the value column
at 4 / 16 / 64 declared thresholds; homology computed twice on one complex (first and warm). Then the measured
limits: the reduction's exact step count admitted at its value and refused one below it (iterations), the packed
path's tensor refusal under the default budget, and the scan's states refusal, each at the last admitted and the
first refused point. CUDA is timed only with --device (cells, incidences, Euler counts, local invariants, d o d);
the column reductions are CPU only. Both capabilities run both cases (the inventory gives them the same benchmark
cases).
"""
import itertools

import numpy as np
import pandas as pd
try:
    from scipy import ndimage
except ImportError:                                               # not a declared dependency: that baseline skipped
    ndimage = None

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_topo as mt
from torbles.math_budgets import Budget

P61 = (1 << 61) - 1


def pattern(kind, shape, rng):
    """an occupancy array and its known Betti numbers (None when not known by construction)"""
    d = len(shape)
    if kind == "empty":
        return np.zeros(shape, dtype=bool), [0] * (d + 1)
    if kind == "full":
        return np.ones(shape, dtype=bool), [1] + [0] * d
    if kind == "25%":
        return rng.random(shape) < 0.25, None
    occ = np.zeros(shape, dtype=bool)                  # holes: tiles of 4^d, a ring / shell of 3^d in each corner
    tiles = [n // 4 for n in shape]
    count = 0
    for t in itertools.product(*[range(m) for m in tiles]):
        sl = tuple(slice(4 * a, 4 * a + 3) for a in t)
        occ[sl] = True
        occ[tuple(4 * a + 1 for a in t)] = False
        count += 1
    known = [count, count, 0] if d == 2 else [count, 0, count, 0]
    return occ, known


def table_of(occ, rng):
    pts = np.argwhere(occ)
    d = occ.ndim
    names = ["x", "y", "z"][:d]
    frame = pd.DataFrame({n: pts[:, a].astype(np.int64) for a, n in enumerate(names)})
    frame["v"] = rng.integers(0, 1000, len(pts)).astype(np.int64)
    ddl = "CREATE TABLE t (" + ", ".join(f"{n} BIGINT" for n in names + ["v"]) + ")"
    return torbles.from_ddl(ddl, {"t": frame}), names


def numpy_cells(occ):
    """cells per dimension by shifted ORs: the cell spanning the axes S at anchor a exists when a top cell with
    t_a = a_a (a in S) and t_a in {a_a - 1, a_a} (a not in S) is occupied"""
    d = occ.ndim
    counts = [0] * (d + 1)
    for S in itertools.product((0, 1), repeat=d):        # S[a] = 1: the cell spans axis a
        pad = np.pad(occ, [(0, 0) if S[a] else (1, 1) for a in range(d)])
        acc = None
        for shift in itertools.product(*[((0,) if S[a] else (0, 1)) for a in range(d)]):
            sl = tuple(slice(s, s + (occ.shape[a] if S[a] else occ.shape[a] + 1)) for a, s in enumerate(shift))
            acc = pad[sl] if acc is None else (acc | pad[sl])
        counts[sum(S)] += int(acc.sum())
    return counts


def summary(bench, case, params, **result):
    """the results of a case next to its timings (not a timing)"""
    bench.records.append(dict(case=case, backend="cpu", phase="result", params=params, result=result))


def grid_of(names, occ, value=True):
    return mt.Grid(names, [(0, n - 1) for n in occ.shape], 8 if occ.ndim == 2 else 26, value="v" if value else None)


def complexes_case(bench, rng):
    sizes2 = [(32, 32)] if bench.quick else [(64, 64), (128, 128), (256, 256)]
    sizes3 = [(8, 8, 8)] if bench.quick else [(16, 16, 16), (24, 24, 24), (32, 32, 32)]
    for shape in sizes2 + sizes3:
        for kind in ("empty", "full", "25%", "holes"):
            occ, known = pattern(kind, shape, rng)
            db, names = table_of(occ, rng)
            params = dict(shape=list(shape), pattern=kind, rows=int(occ.sum()))
            scope = bench.measure("scope", "cpu", lambda: mbk.Scope.from_table(db["t"], names + ["v"]), repeats=2,
                                  phase="build", **params)
            grid = grid_of(names, occ)
            want = numpy_cells(occ)
            for backend in bench.backends():
                c = bench.measure("euler counts", backend, lambda: mt.counts(scope, grid, backend=backend), repeats=3,
                                  phase="run", **params)
                assert c["cells"] == want, (shape, kind, c["cells"], want)
                cx = bench.measure("complex (cells, incidences)", backend,
                                   lambda: mt.Complex.build(scope, grid, backend=backend), repeats=3, phase="build",
                                   **params)
                assert cx.counts == want
                loc = bench.measure("local invariants", backend, lambda: cx.local(backend=backend), repeats=3,
                                    phase="run", **params)
                dd = bench.measure("d o d check", backend, lambda: cx.check(backend=backend), repeats=3, phase="run",
                                   **params)
                assert dd["z"]["holds"] and dd["gf2"]["holds"]
            bench.measure("numpy cells (baseline)", "numpy", lambda: numpy_cells(occ), repeats=3, phase="run", **params)
            adj = bench.measure("dual graph components", "cpu", lambda: cx.adjacency(), repeats=3, phase="run", **params)
            comps = None
            if ndimage is not None:
                structure = np.ones((3,) * occ.ndim, dtype=bool)
                comps = bench.measure("scipy ndimage.label (baseline)", "scipy",
                                      lambda: ndimage.label(occ, structure=structure)[1], repeats=3, phase="run",
                                      **params)
                assert adj["components"] == comps
            if known is not None:
                assert adj["components"] == known[0]
            summary(bench, "complex summary", params, cells=cx.counts, euler=cx.euler, components=adj["components"],
                    boundary=loc["totals"]["boundary"], numpy_cells=want,
                    scipy_components=None if comps is None else int(comps))
            cx.close()


def homology_case(bench, rng):
    sizes = [((16, 16), (8, 8, 8))] if bench.quick else [((32, 32), (8, 8, 8)), ((64, 64), (12, 12, 12)),
                                                         ((96, 96), (16, 16, 16)), ((192, 192), (24, 24, 24))]
    fields = [("gf2", "reduction"), ("gf2", "packed"), (P61, "reduction"), ("q", "reduction")]
    for pair in sizes:
        for shape in pair:
            occ, known = pattern("holes", shape, rng)
            db, names = table_of(occ, rng)
            scope = mbk.Scope.from_table(db["t"], names + ["v"])
            grid = grid_of(names, occ)
            cx = bench.measure("complex for homology", "cpu", lambda: mt.Complex.build(scope, grid), repeats=1,
                               phase="build", shape=list(shape))
            cells = sum(cx.counts)
            top = len(shape)
            for field, method in fields:
                params = dict(shape=list(shape), field=str(field), method=method, cells=cells)
                try:
                    mt.homology_need(cx, field=field, method=method)
                    cyc = () if method == "packed" else tuple(range(1, top + 1))
                    h = bench.measure("homology (ranks, Betti, cycles)", "cpu",
                                      lambda: mt.homology(cx, field=field, method=method, cycles=cyc), repeats=2,
                                      phase="run", **params)
                except Exception as e:                   # the packed path past the default tensor budget
                    if type(e).__name__ != "AlgebraBudgetError":
                        raise
                    bench.refusal("homology (ranks, Betti, cycles)", "cpu",
                                  lambda: mt.homology(cx, field=field, method=method, cycles=()), **params)
                    continue
                assert h.betti == known, (shape, field, method, h.betti, known)
                if cyc:
                    assert all(len(h.cycles.get(k, [])) == known[k] for k in cyc)
                summary(bench, "homology summary", params, betti=h.betti, known=known, ranks=h.ranks, steps=h.steps,
                        peak_bytes=h.peak_bytes)
            for m in (4, 16, 64):
                ts = sorted(int(x) for x in np.linspace(0, 999, m))
                ts = sorted(set(ts))
                p = bench.measure("persistence (sublevel thresholds)", "cpu",
                                  lambda: mt.persistence(cx, field="gf2", ties="index", thresholds=ts), repeats=2,
                                  phase="run", shape=list(shape), levels=len(ts), cells=cells)
                assert p.info["betti"] == known
                summary(bench, "persistence summary", dict(shape=list(shape), levels=len(ts)), pairs=len(p.pairs),
                        essential=len(p.essential), zero_length=p.info["zero_length"], steps=p.info["steps"],
                        peak_bytes=p.info["peak_bytes"], betti=p.info["betti"])
            for backend in bench.backends():
                bench.measure("d o d check (homology complex)", backend, lambda: cx.check(backend=backend),
                              repeats=2, phase="run", shape=list(shape), cells=cells)
            cx.close()


def limits_case(bench, rng):
    occ, known = pattern("holes", (24, 24) if bench.quick else (48, 48), rng)
    db, names = table_of(occ, rng)
    scope = mbk.Scope.from_table(db["t"], names + ["v"])
    cx = mt.Complex.build(scope, grid_of(names, occ))
    h = mt.homology(cx, field="q", cycles=(1,))
    steps = h.steps
    bench.measure("homology at the step limit", "cpu",
                  lambda: mt.homology(cx, field="q", cycles=(1,), budget=Budget(iterations=steps)), repeats=1,
                  phase="run", iterations=steps)
    bench.refusal("homology one step below", "cpu",
                  lambda: mt.homology(cx, field="q", cycles=(1,), budget=Budget(iterations=steps - 1)),
                  iterations=steps - 1)
    bench.threshold("reduction steps (Q, H_1 cycles): the iterations budget", admitted=dict(iterations_limit=steps),
                    refused=dict(iterations_limit=steps - 1), steps_needed=steps, shape=list(occ.shape), betti=h.betti)
    # the packed GF(2) path under the default tensor budget (2^26 matrix elements): the largest admitted square grid
    admitted, refused = None, None
    for n in (16, 24, 32, 40, 48, 56, 64, 72, 80):
        o = np.ones((n, n), dtype=bool)
        d2, nm = table_of(o, rng)
        c2 = mt.Complex.build(mbk.Scope.from_table(d2["t"], nm), grid_of(nm, o, value=False))
        need = mt.homology_need(c2, field="gf2", method="packed")["tensor"]
        if need <= Budget().limits["tensor"]:
            admitted = dict(n=n, tensor=need)
        else:
            refused = dict(n=n, tensor=need)
            bench.refusal("packed GF(2) homology", "cpu", lambda: mt.homology(c2, field="gf2", method="packed"),
                          n=n, tensor=need)
            break
    bench.threshold("packed GF(2) ranks (default tensor budget)", admitted=admitted, refused=refused)
    # the scan: positions against the default states budget (2^24)
    small = mt.Grid(names, [(0, (1 << 12) - 1), (0, (1 << 12) - 1)], 8)
    big = mt.Grid(names, [(0, (1 << 12) - 1), (0, (1 << 12))], 8)
    bench.measure("Euler counts of a 4096^2 extent", "cpu", lambda: mt.counts(scope, small), repeats=1, phase="run",
                  positions=1 << 24)
    bench.refusal("Euler counts of a 4096 x 4097 extent", "cpu", lambda: mt.counts(scope, big),
                  positions=(1 << 12) * ((1 << 12) + 1))
    bench.threshold("scan states (default budget)", admitted=dict(positions=1 << 24),
                    refused=dict(positions=(1 << 12) * ((1 << 12) + 1)))


def main():
    bench = Bench.from_args("TOPO-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    complexes_case(bench, rng)
    homology_case(bench, rng)
    limits_case(bench, rng)
    bench.note("CUDA timings only with --device; the column reductions (homology, persistence) run on the CPU only.")
    bench.note("Baselines: NumPy shifted ORs for the cells per dimension, SciPy ndimage.label for the components; "
               "both checked equal to the native counts. Known Betti numbers: holes per tile of the patterns.")
    if ndimage is None:
        bench.note("SciPy is not installed: the ndimage.label baseline was skipped (components checked against the "
                   "patterns' known counts where they are known).")
    bench.write()


if __name__ == "__main__":
    main()
