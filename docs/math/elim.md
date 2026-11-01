# Bounded elimination and generator selection (GEOM-002)

`torbles/math_elim.h` (every contract; the row check of a pair is inline and shared with CUDA), `math_elim.c`,
`math_elim_gpu.cu`, `math_elim.py` (opt-in: `import torbles` does not load it). Certified relations among the columns
of one scope (FND-007 certificates, e.g. DISC-002 identities, read as GEOM-001 reads them) decide which columns are
generators and which are reconstructed exactly from them, and which form of a query expression (as written, with
some columns substituted by their reconstructions, or the canonical normal form) is cheapest on the measured data.
Nothing is assumed from the algebra alone: every reconstruction and every form is checked on every selected row.

```python
from torbles import math_elim as me, math_geom as mg
sel = me.select(scope, relations)                   # bounded search by size: the smallest generator set
sel.generators, sel.eliminated                      # ["x", "y"], {"z": x*y, "s": x + y, "w": x^2 - y}
sel.info                                            # minimum, minimal, undecided, solves, max_degree, max_terms, ...
sel.candidates                                      # every set tried: status, reason, failed, costs, refusal
sel.columns                                         # per variable: role, read path, width, reconstruction cost, check
sel.certificate("z")                                # FND-007 derived certificate D (z - f) = sum D h_k r_k
sel.check(backend="cuda")                           # every reconstruction on every row: values, NULL mask, range
me.select(scope, relations, mode="order", order=["z", "s", "w"])          # one elimination order (one solve)
F = me.forms(scope, sel, q)                         # direct, local substitutions, normal form of q
F.candidates, F.chosen, F.best, F.polynomial(i), F.certificate(i)
me.check(scope, [("z", f), (q, p)], backend="cuda") # pairs (target, form); a column target also checks its range
me.cost(scope, p); me.choose(scope, [q, p], priority="bits")
```

## Generators

Variables V are scope columns (default: the relations' columns; at most 8). A set G of V is a generator set when every
eliminated x of E = V \ G has a reconstruction x = f_x(G), a polynomial in G alone with x - f_x in the relations'
ideal I. This is decided exactly by elimination: under lex with E before G, GEOM-001's reduced Groebner basis gives
the normal form of x, which lies in Q[G] exactly when such an f_x exists (the elimination property), and then f_x is
that normal form. Relation columns that are not variables are eliminated first.

- `mode="search"`: sizes k = 0, 1, ... in increasing order; every k-subset of the relation variables (a variable in no
  relation is always a generator), in lexicographic order of positions, is one solve (a completion, then the
  reductions of E). The first size with a set whose reconstructions all exist and all check is the minimum; with
  `rank_all` every set of that size is costed and the cheapest by the priority (`total`, `scan`, `work`, `bits`) is
  chosen, ties to the earliest; otherwise the first. `max_size` bounds the sizes tried; nothing feasible leaves G = V
  (reason `direct`).
- `mode="order"`: one solve under the given lex sequence (the rest after it, in V's order); E is the variables whose
  normal form differs from themselves, reads only the others and checks; G is the rest. `minimal` is false.
- A set the exact arithmetic cannot decide within the budget is undecided, never guessed: past the degree or support
  budget (or GEOM-001's per-solve limit of 2^20 steps) it is rejected with reason `budget` and its refusal (resource,
  need, limit), past int64 or a representation limit with reason `unclaimable`; the search goes on.
  `info["minimal"]` holds only when every smaller set was decided infeasible (`undecided`, `undecided_size`).
- Candidate reasons: `cheapest`, `costlier` (admissible, not chosen), `not reconstructible` (a normal form leaves G),
  `null mismatch`, `value mismatch`, `range`, `undecided` (row check), `budget`, `unclaimable`, `direct`.

## Checks (values, NULL domains, ranges)

A pair (T, F) of a target and a form with integer forms A = D T, B = D F (one D > 0) is evaluated on every selected
row with SQL NULL propagation: both NULL agree; one NULL is a NULL-domain mismatch (`null_lost`: the target present,
the form NULL; `null_extra`: the converse); both present compare A and B exactly in W 64-bit words (W <= 8) under a
bound proven from the scope's extremes (rows past it are `undecided`, never wrapped; a view whose extremes exceed the
proven magnitudes is `DomainError`). A column target is a range pair: B is also compared with D lo and D hi of that
column's extremes (`below`, `above`; `at_lo`, `at_hi`: the range is attained). It holds when nothing mismatches,
nothing is undecided and a ranged pair attains both ends. `first` is the smallest mismatching row of any kind
(`first_null`, `first_value`, `first_range` per kind). A relation certified with `nulls="skip"` holds on complete rows
only, so a column NULL where its generators are present is not reconstructed by them (the set is rejected with
`null mismatch`). A reconstruction is kept only when its check holds.

## Costs

Relative work units of a bit-plane engine, measured on the scope (not time). A column's width is the bit length of
max - min of its selected present values (at least 1) plus one validity plane when it has NULLs. A polynomial p with
integer form D p costs scan = ceil(N / 64) x the widths of the columns it reads and work = N x (terms +
multiplications) x words, words = ceil((actual bits + 1) / 64) from the largest |D p| measured on the compared rows;
total = scan + work. A generator set costs storage = ceil(N / 64) x its generators' widths plus each reconstruction's
scan and work. An eliminated column's read path is `reconstruct` only when its reconstruction is strictly cheaper
than scanning the stored column, else `direct`. `choose` ranks by the priority, then the other measures; ties go to
the earliest candidate.

## Forms of a query expression

`forms(scope, sel, q)`: candidate 0 is q as written; then every local substitution q_S for the nonempty sets S of
eliminated variables q reads (|S| <= `max_eliminate`, increasing size then lexicographic): GEOM-001's division of q
by the reconstruction certificates of S (lex S first, not completed: exactly the substitution), certified by FND-007
through those certificates (depth 2); with `normal_form`, the normal form under the selection's elimination ideal
(canonical, certified from the relations). Each alternative is checked against q on every selected row and costed on
its own integer form; the cheapest admissible by the priority is chosen and ties go to the earliest, so the direct
form is kept whenever no alternative is strictly cheaper (term explosions such as w^6 -> (x + 1)^6) or every
alternative is rejected. The normal form is a candidate like any other, never assumed cheapest (x y - z + 3 w: the
local form 3 w beats the normal form 3 x + 3 on the stored widths).

## Exactness

Coefficients are GEOM-001's canonical int64 rationals; past int64 a set is undecided, a form `unclaimable`, never
rounded. A certificate's kind is the weakest relation's and its domain theirs (a snapshot relation makes a
reconstruction exact on that snapshot's rows only; it goes stale with its table). Row evaluation is DISC-002's proven
W-word arithmetic.

## CUDA

`tr_mgpu_elim_check(view, pairs, budget, out, refusal)`: work items over chunks of 8 pairs x row slices of at least
64 selection words; each accumulates locally (`tr_elim_pair_row`, shared with the CPU) and adds its counters with one
atomic each, atomic minima for the first rows and atomic maxima for the bit lengths: no shared memory, no barrier,
results independent of thread order and equal to `tr_elim_check_view` bit for bit. The host validates the pair view
and budgets the device copies before launching. Selection, completion, reduction and certification run on the CPU.

## Budgets and refusals

iterations = solves (the sets of a search; the alternatives of `forms`, counted before any solve), refused at the solve
past the limit (need = limit + 1), never truncated. degree and support bound every completion and reduction: a search
set past them is undecided (above), an `order` selection is refused, a form candidate is rejected with its refusal and
the direct form kept. variables = |V|; output and bytes bound the pairs and allocations (checked before allocation).
`AlgebraBudgetError` names the resource, need and limit. More than 8 variables, 16 relations or 256 pairs per batch
is `UnsupportedError`; relations GEOM-001 refuses are `DomainError`.

## Limitations

- Reconstructions are polynomials in the generators (no rational or algebraic functions); the search is exponential in
  |V| (at most 255 sets for 8 variables) and its lex completions can be large: dense or implicitization-like sets are
  often undecided, and then minimality is not claimed.
- Coefficients are int64 rationals (GEOM-001); the cost model is relative work units on one scope, not a planner's
  timing model.
- The CUDA kernel covers the batched row checks only.
