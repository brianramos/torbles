# Declared occupancy complexes, homology and persistence (TOPO-001, TOPO-002)

`torbles/math_topo.h/.c` (native CPU), `math_topo_gpu.cu` (CUDA kernels: occupancy, cells and incidences, Euler
counts, local invariants, boundary-composition checks), `math_topo.py` (opt-in Python API). TOPO-001 builds a cubical
complex from user-declared geometry over the rows of a FND-004 scope; TOPO-002 reduces its boundary matrices (or
those of an explicit integral chain complex) over a declared field: ranks, Betti numbers, cycle bases and the
persistence of a bounded filtration. The header comment of `math_topo.h` is the normative contract; this page
summarizes it.

```python
from torbles import math_bank as mbk, math_topo as mt
scope = mbk.Scope.from_table(db["t"], ["x", "y", "v"], rows=mask)
grid = mt.Grid(["x", "y"], extent=[(0, 15), (0, 15)], adjacency=8, value="v")   # 2D: 4 / 8; 3D: 6 / 18 / 26
cx = mt.Complex.build(scope, grid, backend="cpu")       # or backend="cuda"; mt.counts(scope, grid) without the complex
cx.rows; cx.counts; cx.euler                            # row accounting, [V, E, F(, C)], V - E + F (- C)
cx.cells(k)                                             # anchors (lower corners), axes masks, Khalimsky linear indices
cx.incidence(k)                                         # faces (n_k x 2k) and orientation signs
cx.boundary(k)                                          # d_k over Z (rows, cols, values; .dense()); cx.boundary(k, p): mod p
cx.boundary(k, "gf2")                                   # FIELD-001's packed Matrix (rank, rref, nullspace, @ ...)
cx.local(backend="cuda")                                # star / link counts, boundary flags, dual degrees + totals
cx.adjacency(4, pairs=True)                             # dual graph: edges, components, labels, pairs
cx.check(backend="cuda")                                # d o d = 0 over Z and GF(2) from the stored incidences
cx.fresh(scope2)                                        # FND-007 freshness: a scope over other columns -> "columns"
h = mt.homology(cx, field="gf2" | "q" | p, method="reduction" | "packed", cycles=(1,))
h.ranks, h.betti, h.cycles[1]                           # H_1 basis as edge cycles: generator, cells, coefficients
p = mt.persistence(cx, field="gf2", ties="index" | "reverse", thresholds=[1, 5, 9], direction="sublevel")
p = mt.persistence(cx, field="q", ties="index", levels=per_top_cell, nlevels=m)
p.pairs, p.essential, p.diagram(1), p.betti(level), p.levels(k), p.info
cc = mt.ChainComplex([d1, d2]); mt.homology(cc, field=3); mt.persistence(cc, ..., levels=[[..], [..], [..]], nlevels=m)
mt.need(scope, grid, backend=...), mt.homology_need(...), mt.persistence_need(...)
```

## Declared geometry (TOPO-001)

- `Grid(coords, extent, adjacency, value=None)`: 2 or 3 distinct coordinate columns of the scope (axis order as
  given), the extent `(lo, hi)` of the top cells per axis (inclusive, logical units), the dual-graph adjacency and an
  optional value column for filtrations. Coordinates must be exact integer columns: an `int` codec (or an array column
  of scale 0); a decimal (its logical integer is a scaled storage form), Boolean or text column is refused
  (`DomainError` / `UnsupportedError`). Only the declared columns' logical values are read: the physical row order,
  the stored codes (logical - offset), their bit planes and the torus layout of the columns are never geometry (the
  tests permute rows and columns, shift every code, and show that 7 and 8 touch while 0 and 8, one bit plane apart, do
  not).
- Cell convention: a selected row whose coordinates are present and inside the extent occupies the closed unit top
  cell `[x, x+1] x [y, y+1] (x [z, z+1])`. Rows with a NULL coordinate (`null_rows`, decided first) or a present
  coordinate outside the extent (`outside_rows`) occupy nothing; rows of one cell collapse (`duplicates = valid_rows -
  occupied`). The complex is the closure of the occupied top cells.
- Cells: Khalimsky coordinates `c_a` in `0 .. 2 n_a` (odd: the cell spans `[x, x+1]` along axis a); the k-cells are
  numbered in increasing linear index `L = c_0 + K_0 (c_1 + K_1 c_2)`, `K_a = 2 n_a + 1`. `cells(k)` gives each cell's
  anchor `lo_a + floor(c_a / 2)` and its axes.
- Incidences and orientation: a k-cell spanning `a_0 < .. < a_{k-1}` has 2k faces; slot 2i is the minus face
  (`c_{a_i} - 1`, sign `-(-1)^i`), slot 2i + 1 the plus face (sign `(-1)^i`): `d = sum_i (-1)^i (face_i^+ - face_i^-)`,
  so `d o d = 0` over Z. Euler characteristic `chi = sum (-1)^k n_k`, exact.
- Boundary cells: a cell of dimension < dim lies on the topological boundary of the union iff one of the
  `2^(dim-k)` top positions around it is unoccupied or outside the extent; they form a subcomplex (`boundary`,
  `boundary_euler` in the counts).
- Local invariants (per cell, dimension-major): `star[j]` = the j-cells of the complex having the cell as a face, the
  link counts (`link[i] = star[k+1+i]`: the link has one simplex per coface), `link_euler = sum_{j>k} (-1)^(j-k-1)
  star[j]` (2D: 0 interior vertex, 1 boundary arc, 2 a pinch), the boundary flag, and for top cells the dual degree.
- Dual graph: occupied top cells, adjacent when sharing an edge (4) / face (6), an edge (18), a vertex (8, 26); the
  components are labelled by their smallest top index. With 8 / 26 they are the components of the union (`b_0`).
- None of these is a homology claim: a complex and its counts carry no Betti number (two complexes with the same
  popcount, V, E, F and chi differ in their Betti numbers; the registry refuses a rank step over the popcount carrier).

## Homology and persistence (TOPO-002)

- Field (declared, no default): `"gf2"`; a prime `p` (3 <= p < 2^62, verified by FND-009's deterministic
  Miller-Rabin; `2` is GF(2)); `"q"`. `"z"` (integral homology with torsion) is `UnsupportedError`.
- `method="reduction"` (every field): the standard column reduction; GF(2) XOR of sorted columns, GF(p) mod p
  (Barrett, FND-009), Q fraction-free with primitive integer columns (a column scaled by nonzero integers and divided
  by its content changes no pivot: the pairs and ranks of elimination over Q, as LIN-002's Bareiss elimination is),
  entries checked to fit int64 with 128-bit intermediates (`OverflowError` past it, never wrapped). `method="packed"`
  (GF(2)): the ranks through FIELD-001 (`tr_field_rank` of each packed `d_k`), no representatives.
- Results: `ranks[k] = rank d_k` (k = 0 .. top + 1), `betti[k] = n_k - rank_k - rank_{k+1}`, and with `cycles` a basis of
  H_k as k-cycles (the V columns of the essential positive cells: `d z = 0`, independent modulo the boundaries);
  normalized: GF(p) generator coefficient 1, Q primitive with a positive generator coefficient, entries by cell index.
  For subcomplexes of 2D / 3D grids every field gives the same Betti numbers (no torsion); explicit chain complexes
  show the field: RP^2 is 1, 1, 1 over GF(2) and 1, 0, 0 over Q.
- Explicit chain complexes (`ChainComplex`): integer matrices `d_1 .. d_top` (top <= 7), `d_{k-1} d_k = 0` verified over
  Z at creation (`DomainError` otherwise; an int128 overflow of the check is `OverflowError`).
- Filtrations (bounded, declared): `thresholds` on the complex's value column (sublevel: a top cell enters at the
  first level l with `min(v) <= t_l`, t strictly increasing; superlevel: `max(v) >= t_l`, t strictly decreasing;
  thresholds are exact values of the column's scale), or explicit `levels` with `nlevels` (per top cell of a complex,
  per cell of each dimension of a chain; `None` / -1 never enters). Lower cells enter with their earliest occupied
  top coface; a top cell without a present value or past the last threshold never enters (`info["never"]`). Explicit
  chain levels must be a filtration (`DomainError`). Order: (level, dimension, index) with the declared tie rule
  (`"index"` increasing, `"reverse"` decreasing): both are filtrations, the pairs are deterministic, and the diagram
  (pairs with birth < death, essential classes) does not depend on the rule.
- Pairs (`p.pairs`): dim, birth and death levels, birth and death cells; zero-length pairs are reported and counted;
  `p.essential`: classes alive at the last level (nothing is claimed past it). `p.betti(level)` gives the Betti numbers
  of the filtration's complex at a level.

## CUDA

`tr_mgpu_topo_counts` (occupancy by atomic OR, row and cell counts per thread added once), `tr_mgpu_topo_build`
(occupancy, per-position value extremes by atomic min / max, per-chunk cell counts in their own slots, their exclusive
prefix on the host, cells written in increasing order with one writer per slot, incidences one work item per cell by
binary search, top-cell values), adopted by `tr_topo_adopt` after a full validation (canonical occupancy, increasing
cells of their dimension inside the complex, every face at its slot with its sign, every lower cell a face, the row
accounting); `tr_mgpu_topo_local` (one work item per cell); `tr_mgpu_topo_dd` (one work item per k-cell, counts by
atomics, witnesses by atomic minimum). Equal to the CPU bit for bit (emulated with a scrambled thread order in the
tests). The column reductions (homology, persistence) run on the CPU only.

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| scan (`counts`, a build's first phase) | variables = dim; states = positions `prod n_a`; bytes = occupancy, rank directory and valued bitmaps (CUDA: + the selection and column copies) |
| build (after the scan) | output = cells; tensor = incidences `sum 2k n_k`; bytes = the complex (8 per cell, 5 per incidence, 16 per top cell with a value, bitmaps) and its FND-007 record; CUDA: bytes = the device copies, then the adopted complex as on the CPU (`mt.need` reports every phase) |
| local / adjacency / check / boundary | output = entries written; check: tensor = composition terms; CUDA: bytes = copies |
| homology / persistence | states = cells (matrix rows and columns); tensor = boundary entries (packed: the largest `n_{k-1} n_k`); support = levels (1 for homology); bytes = fixed arrays; during the reduction: iterations = column additions, bytes = every column, V column and scratch held; after it: output = pairs + essential + classes + entries |
| explicit chain | states = cells, tensor = entries, bytes = copies and the d o d scratch |

A reduction past its step or byte budget is refused when the step or allocation would be needed (deterministic;
`.need` is `limit + 1` steps or the bytes then held) and returns nothing: no partial pairs, ranks or cycles.
`h.steps`, `h.peak_bytes`, `p.info` report what a request used. Representation limits (`UnsupportedError`):
positions > 2^28, more than 7 chain dimensions, 2^32 cells or entries in a chain dimension, p >= 2^62. Malformed input
(`ValueError`): bad extents or adjacency, thresholds not strictly monotone, levels past `nlevels`, a composite p.
`StaleStateError`: the scope changed (FND-004), or `require_fresh(scope)` against another layout (FND-007 reasons).

## Limitations

The extent is dense (states = positions, at most 2^28); the cell enumeration scans every candidate of the Khalimsky
grid; the reductions are the plain standard algorithm (no clearing / twist, no cohomology), single-threaded and CPU
only; Q entries are bounded by int64 (refused past it); integral homology (torsion coefficients) is not computed;
FND-007 has no claim kind for homology, so results are tied to the complex's FND-007 record (freshness with exact
reasons) rather than issued as certificates; the CUDA kernels are compiled with nvcc for sm_80 and verified in the
emulation only (no device run yet).
