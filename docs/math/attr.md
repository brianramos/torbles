# Sparse interaction and contribution attribution (ATTR-001)

`torbles/math_attr.h` (every contract; the coalition evaluator, the exhaustive weights and the bounded
exploration / search drivers are inline and shared by the CPU and the CUDA library), `math_attr.c`,
`math_attr_gpu.cu`, `math_attr.py` (opt-in). A cooperative game is declared explicitly (its players, its value
function v(S), its baseline v(empty)); attribution indices are exact rationals computed from the game's Moebius
(Harsanyi) transform, sparse when the provenance polynomial is.

```python
from torbles import math_attr as ma
g = ma.Game.provenance(G, tuples=[0])                  # PROV-001: v(S) = COUNT(*) of G's tuple 0, S = the present tuples
g = ma.Game.provenance(J, weights="a", players="sources", semiring="count")   # SUM(a), one player per source relation
g = ma.Game.provenance(G, semiring="bool", players=[("R", 0), "S", {("R", 1), ("R", 2)}], others="absent")
g = ma.Game.conjunction(hist)        # LATTICE-002: v(S) = COUNT(*) WHERE the conditions of S joined by AND
g = ma.Game.disjunction(hist)        # v(S) = COUNT(*) WHERE the conditions of S joined by OR (v(empty) = 0)
g = ma.Game.predicate(masks, program, baseline=[True, False, ...])   # absent conditions replaced by constants
g = ma.Game.table(values)            # 2^n declared exact values, values[S] = v(S) (bit i of S: player i)
g.values([(), [("R", 0)], 0b101], backend="cuda")   # coalitions: iterables of players, or masks (bit i: player i)
g.info; g.convention; g.players; g.need("exhaustive", backend="cuda"); g.support_need("expansion")
s = g.support("auto" | "expansion" | "witnesses" | "transform"); s.terms; s.info   # the nonzero m(T)
s = ma.Support.from_terms(n, {(0, 3): 6, (5,): -2})
ma.shapley(g | s, method="auto" | "support" | "exhaustive", backend="cpu" | "cuda"); ma.banzhaf(...)
ma.interactions(g | s, order=2, index="shapley" | "banzhaf" | "moebius" | "shapley_taylor", min_order=1)
ma.interactions_need(s, order=2, index="shapley")    # tensor = output = the (set, term) candidates
ma.exhaustive(g, backend="cuda")     # both values, the exact numerators over n! and 2^(n-1)
ma.explore(g, max_coalitions=5000)   # exact, or certified bounds lo <= index <= hi, with the explored states
ma.search(g, "repair" | "explanation", "<=", 0, max_coalitions=10_000, max_solutions=16)
```

## The game and its convention (part of every result)

| constructor | players | v(S) | baseline v(empty) |
|---|---|---|---|
| `provenance` | source tuples of the plan's cone (`"tuples"`), source relations (`"sources"`), or declared groups; the other tuples fixed `present` / `absent` | sum_j w_j a_j(S), a_j the annotation of output tuple j in N (multiplicity: COUNT(*), weights give SUM(col)) or in B (existence) under the 0/1 valuation where exactly the players' tuples of S are present | every player's tuples deleted |
| `conjunction` | the histogram's conditions | points TRUE on every condition of S (zeta-up of the `true_sets` projection of a three-valued histogram: SQL WHERE; `complete` of an `apart` one: the complete points only) | no condition applied: every point |
| `disjunction` | the conditions | points TRUE on at least one condition of S | 0 |
| `predicate` | the program's conditions | points where the BOOL-001 program is TRUE with every condition outside S replaced by its declared baseline constant (SQL three-valued logic) | every condition at its constant |
| `table` | 0..n-1 | as declared | as declared |

Every result has `convention` (`game`, `players`, `baseline`, `value`, `index`, `weighting`), `causal: False`
and `interpretation`. **An algebraic attribution has no automatic causal interpretation**: it splits
v(P) - v(empty) under the declared game, baseline and weighting; it is not the effect of an intervention on the data,
and another baseline (e.g. the predicate's absent conditions TRUE instead of FALSE) or weighting attributes the same
data differently.

## Indices (exact `Fraction`s, formed from native canonical 128-bit rationals)

With m(T) = sum over S subset of T of (-1)^|T - S| v(S): Shapley phi_i = sum over T containing i of m(T)/|T|
(= the weighted marginal contributions |S|!(n-|S|-1)!/n!), Banzhaf beta_i = sum m(T)/2^(|T|-1), Moebius m(S),
Shapley interaction I(S) = sum over T superset of S of m(T)/(|T|-|S|+1), Banzhaf interaction m(T)/2^(|T|-|S|),
Shapley-Taylor of order k (m(S) below k, sum m(T)/C(|T|, k) at k). `interactions` lists every set with a nonzero
index up to the order; every set not listed is exactly 0. Identities checked natively (`identities`): efficiency
(sum of the Shapley values = v(P) - v(empty)), Shapley-Taylor efficiency, the Moebius sum (= v(P)), and for a support
built from a game its reconstruction of the game's own v(P), v(empty) (`reconstruction`).

## Sparse supports

`expansion` (N): the plan's cone expanded bottom-up in the multilinear algebra over players (x^2 = x under 0/1
valuations; a player's tuples merge; PLUS adds, TIMES unites sets and multiplies coefficients): its size is the
polynomial's player-set support, so a join of 40 tuples has one term per join pair, never 2^40 coalitions.
`witnesses` (B): per output the minimal player sets of its expansion, then v = 1 - prod (1 - x_W) expanded
(inclusion-exclusion, at most 2^w terms). `transform`: the LATTICE-002 fast Moebius transform of the 2^n values.
The canonical order of terms and sets is by size, then the largest player first (LATTICE-002's order).

## Enumeration, bounded exploration, minimum search (n <= 62)

- `values`: v at given coalitions. `exhaustive` (n <= 32): every coalition once, player i receiving
  c_i(S) v(S) with c_i(S) = (s-1)!(n-s)! (i in S) or -s!(n-1-s)! over n! (Banzhaf: +-1 over 2^(n-1)); exact in
  128-bit two's complement after the host proves 2 n! bound < 2^127 and 2^n bound < 2^127 (else `OverflowError`).
- `explore(max_coalitions)`: whole levels (coalitions of one size) while they fit the budget; v(empty), v(P) are the
  game's (`known` = `explored` + 2). **Moebius regime** (provenance in N with weights >= 0: m(T) >= 0 is certified):
  level n-1, then levels 1 .. g-1 (g the degree bound: the most player variables in a derivation); with levels 0..d
  known every m(T) with |T| <= d is exact and U_i = v(P) - v(P - i) - (the known mass on sets containing i) >= 0 lies
  on sets of sizes d+1..g, so phi_i is in [K_i + U_i/g, K_i + U_i/(d+1)] (Banzhaf with 2^(g-1), 2^d): exact once
  d + 1 >= g (a join of two relations: after 2n evaluations, a three-way join after 2n + C(n, 2)). **Range regime** (otherwise): levels
  n-1, 1, n-2, 2, ...; each unexplored level widens phi_i by +-(hi - lo)/n, with [lo, hi] the certified range of v
  (provenance: each annotation between its values at empty and P; tables: the exact extremes), and a monotone game's
  indices are clamped to [0, v(P) - v(empty)]. The result is `"exact"` (`values`) only when every bound is a point,
  else `"bounds"` (no `values`); the efficiency check then reads sum lo <= v(P) - v(empty) <= sum hi.
- `search(kind, op, threshold)`: the minimum coalitions C with v(C) op t (`explanation`: C alone suffices) or
  v(P - C) op t (`repair`: removing C suffices), levels 0, 1, ... in increasing mask order within `max_coalitions`
  evaluations. `found`: the minimum size and the number of minimum solutions are exact; `none`: proven (every level
  explored, or a monotone game whose extreme value fails); `incomplete`: certified `min_size` (lo, hi) (no solution
  below lo; hi a known solution's size, or None) and, when the minimum is known, `multiplicity` (lo, hi).

## CUDA

`tr_mgpu_attr_values`: one work item per coalition (a 0/1 valuation of the cone with the shared `tr_attr_eval`, or a
table lookup) into its own slot, launch chunks within 64 MiB of scratch. `tr_mgpu_attr_exhaustive`: work items are
aligned slices of 64 masks; a slice sums its weighted marginal-contribution terms locally (one counter per low
player, one for the high players whose membership is constant on the slice, one player-independent) and adds each
with one `tr_atomic_add_wide`; numerators are adopted natively (`tr_attr_exhaustive_adopt`). `explore` and `search`
run the shared drivers with the values kernel as their batch evaluator. CPU, emulation and device agree bit for bit
(emulated with a scrambled thread order in the tests). Supports and interactions are CPU only.

## Budgets and refusals

| request | needs (checked before allocation) |
|---|---|
| table game | variables = n (players), tensor = 2^n, bytes = the object; lattice: + iterations = k; predicate: tensor = 2^k x words, circuit = scheduled nodes |
| provenance game | variables = n, circuit = cone nodes, support = children, output = outputs, bytes (the plan: PROV-001's own needs) |
| values / exhaustive | output = coalitions / iterations = 2^n, output = 2n; bytes = scratch (CUDA: every device copy, accumulators, scratch chunk: `Game.need(..., backend="cuda")`) |
| support | circuit, support = the term bound (per node: sum / product of the children's, at most 2^(players reached)), degree, bytes; witnesses: tensor = terms^2 + sum 2^w; transform: tensor = 2^n, iterations |
| shapley / banzhaf / interactions | support = terms, output; interactions: tensor = output = the (set, term) candidates |
| explore / search | iterations = max_coalitions (the declared exploration budget; the result is incomplete, never refused, when the space is larger), tensor = the Moebius inversion, output = 4n / max_solutions, bytes |

`AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError` (an exact value outside the 128-bit carriers:
never wrapped), `UnsupportedError` (n > 24 for tables, > 32 exhaustive, > 62 enumerated, > 2^24 players, more than 62
witnesses of one output; the sparse path on CUDA), `DomainError` (a program over another number of conditions, a
projection the histogram did not record), `StaleStateError` (a game over a mutated table), `ValueError`, `TypeError`.

## Limitations

Coalitions of enumeration paths are 64-bit masks (n <= 62); the sparse path has no such limit but its indices are
128-bit rationals (an overflow is reported). Provenance games inherit PROV-001's operators (no aggregates other than
multiplicity-weighted sums, no negation). Supports, interactions and the Moebius inversion are single-threaded CPU
code. Explore's bounds are certified, not tight: the range regime's widening is the worst case over the declared
range; only provenance games in N have the Moebius regime. Search explores levels in a fixed order (no pruning by
monotonicity beyond the extreme-value proof). No FND-007 certificate is emitted: the identities are checked natively
and reported per result (FND-007's claim kinds are row relations of a scope or polynomial laws of at most 8
variables, not attribution identities).
