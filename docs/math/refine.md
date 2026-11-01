# Adaptive refinement and change localization (MULTI-002)

`torbles/math_refine.h` (every contract; the region bounds and rollups shared with CUDA), `math_refine.c`,
`math_refine_gpu.cu`, `math_refine.py` (opt-in). Requests about the groups of one level of a MULTI-001 hierarchy are
answered from the coarsest level down: a region whose sound bound decides every descendant is final, only undecided
regions are refined, and a region the refinement budget does not reach is reported unresolved with its bound. DELTA-001
updates of the leaves are rolled up along the affected paths only, with a complete rebuild when that is not possible.

```python
from fractions import Fraction
from torbles import math_multi as mm, math_delta as md, math_refine as mr
h = mm.Hierarchy([day, month, year], ["1", "x", "x^2"])      # MULTI-001: certified maps, exact banks per level
st = mr.State(h)                                             # cells, UNC-001 inputs and links of every level
r = st.threshold(0, "sum:x", ">", 1000)                      # which leaves have SUM(x) > 1000
r.answer, r.unresolved, r.entries, r.stats, r.complete       # entries: decision and bound of every final region
st.ratio(1, "sum:x", "count:x", ">=", Fraction(5, 2))        # AVG(x) >= 5/2 (D = 0: undefined, never in the answer)
st.topk(0, {"sum:x": 1, "rows": -3}, 10, expand_limit=50, backend="cuda")
st.fine("topk", 0, "sum:x", k=10)                            # the complete fine-grain answer (the baseline)
st.bounds("threshold", 0, "sum:x", 2)                        # every region bound of level 2 (CPU or CUDA)
st2 = st.update(md.Delta.from_arrays(...), backend="cuda")   # localized; levels=[...] enables the rebuild fallback
st2.changes, st2.info["changed"], st2.info["work"], st2.tight(1, g)
```

## Requests and region bounds

An aggregate is `A = sum c_i atom_i` with int64 coefficients over the atoms `sum:<term>`, `count:<term>` and `rows`
of the hierarchy's FND-004 basis (the bank's logical integers), at most 4 atoms (and 4 for a ratio's denominator).
THRESHOLD `A op T`, RATIO `A / D op T` (decided as `C = q A - p D op 0`, T = p / q, the operator flipped where D < 0;
D = 0 undefined), TOPK (largest or smallest A, ties by the smaller group id).

For a region R above the target level, every present descendant t satisfies `A(t) in [lo, hi]`: the atoms are split
into validity classes (SQL: the term's column set; listwise: one class; ROWS: every row); for a class with n valid
rows in R, exact total E and per-row interval `[elo, ehi]` of its row expression (UNC-001's exact abstract
evaluation of `sum c_i term_i` on R's inputs: affine forms cancel correlated terms, so `SUM(x) - SUM(x)` is exactly 0),
t holding k of the n rows: `A_V(t) <= max_k min(k ehi, E - (n - k) elo)` and symmetrically below, evaluated exactly
(256-bit integers) at the endpoints and at the crossing of the two lines. When every atom is valid on every row of R
the atoms form one class. At the target level the bound is the exact value. Inputs are rolled up exactly (min of
mins, max of maxes, gcd(m1, m2, min1 - min2)).

Waves run level by level, coarse to fine, in increasing group id: THRESHOLD / RATIO regions are final when the bound
proves every descendant in (ALL) or out (NONE); TOPK prunes a region whose upper bound is below tau, the k-th largest
witness of the wave (a region's lower bound, or the mean of its present target descendants, since the region's own
exact A is their sum). Other regions are expanded into their present children, at most `expand_limit` expansions
(0: no limit); a region left when the limit is reached is UNRESOLVED with its sound bound and its descendants are
reported in `unresolved`, never guessed. Target groups are TRUE / FALSE / UNDEFINED, or TOPK's IN (rank certain within
k) / OUT / UNRESOLVED. The entries partition the present target groups. `fine` answers from the complete states of
the target level. Bounds come from `tr_refine_eval_bounds` (CPU) or `tr_mgpu_refine_bounds` (one work item per
region on the gathered cells, the UNC-001 evaluation in its own device scratch), equal bit for bit; `tr_refine_supply`
adopts them.

## Change localization

`update(delta)`: DELTA-001 applies the delta to the leaves' maintained state (compatibility, the overflow proof, the
nonnegative promise); each touched leaf's exact change (rows, counts, 192-bit sums, present values, the inserted
values' extremes and gcd, the deleted multiplicity) is summed into its ancestors level by level, one work item per
affected region (CPU or `tr_mgpu_refine_rollup`, adopted after a per-level conservation check). Every changed region is
reported with its exact change; the new state's level 0 equals DELTA-001's cells bit for bit. Extrema after deletions
remain enclosures (the group is no longer `tight`) and the gcd remains a valid congruence. The maps are carried
(status `carried`, as MULTI-001 without levels): a touched leaf keeps its parent. Not compatible locally (the delta's
schema, the overflow proof, a leaf that becomes present without a parent): with `levels=` (the scopes after the
update) a complete rebuild with maps certified on them (strategy `rebuild`, the reason kept; for a new group the
rebuilt leaves must equal the old leaves plus the delta), else `RebuildRequired` / `OverflowError`. A contradicted
nonnegative measure raises `NegativeMultiplicityError`, never rebuilt around. `info` reports the cells visited
(`work`) and those of a complete rollup (`rebuild_work`).

## Budgets and refusals

Checked before allocation: build: support, variables, states, bytes (covering its UNC-001 and DELTA-001 steps,
`State.need`); request: support = atoms, circuit = class program nodes, output = entries, iterations = expansions,
bytes (`request_need`); CUDA bounds: output = regions, bytes = device copies (`bounds_need`); update: the largest of
DELTA-001's apply, the update arrays and the new state (`update_need`; CUDA adds the rollup's device copies).
`UnsupportedError`: more atoms or program nodes than represented, a ratio whose combined coefficients leave int64, a
signed measure (the bounds count rows), more than 32 columns.
