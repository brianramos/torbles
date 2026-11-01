# Scoped query equivalence and counterexamples (META-001)

`torbles/math_equiv.h` (every contract; the per-word evaluation of a question is inline and shared with CUDA),
`math_equiv.c`, `math_equiv_gpu.cu`, `math_equiv.py` (opt-in). Query parts (BOOL-001 predicates, polynomial
measures, queries) are shared natively; questions of equivalence, implication and containment between them are
decided on a declared domain: the rows of a scope (a snapshot) or a declared universal domain. A verdict is proven,
refuted with a concrete witness, or unknown with its reason, and every proof is labelled with where it holds.

```python
from torbles import math_bank as mbk, math_geom as mg, math_equiv as meq
p0 = meq.cond("x", "<", 5) & meq.cond("y", ">=", 0)                     # SQL three-valued logic
p1 = ~(meq.cond("x", ">=", 5) | meq.cond("y", "<", 0))
f, g = mg.poly({(1, 1): 1}, ["x", "y"]), mg.poly({(1, 0): 1, (0, 1): 1}, ["x", "y"])
s = meq.Set(["x", "y"], predicates=[p0, p1], measures=[f, g], queries=[(0, 0), (1, 1), (0, None)])
s.info(); s.ids()          # distinct conditions, the shared DAG, distinct measures / queries (common subexpressions)
qs = [meq.predicates(0, 1), meq.predicates(0, 1, relation="implies"), meq.predicates(0, 1, semantics="kleene"),
      meq.measures(0, 1, nulls="sql"), meq.queries(0, 1, relation="implies")]
r = s.check(scope, qs, backend="cuda", relations=[certs])   # snapshot (or "cpu"); relations: FND-007 certificates
r[3]["verdict"], r[3]["label"], r[3]["witness"], r[3]["point"], r[3]["evidence"]
r.certificate(0); r.authorize(0, scope2)                  # a snapshot proof authorizes only on its exact scope
u = s.decide(qs, meq.domain({"x": (-10, 10), "y": (-3, 3)}, nullable=["y"]))          # universal over a box
e = s.decide(qs, meq.domain({"x": [1, 2, 3], "y": (0, 1)}, mode="enumerate"))        # exhaustive over the product
answers, stats = s.run(scope, mode="shared")              # COUNT / exact SUM; stats: scans actually performed
g2, trace = meq.rewrite(catalog, table_scope, f)          # relationships.Catalog (tr_optimize), lifted back exactly
meq.measure_from_math(first_slice_expression, ["x", "y"])  # a first-slice polynomial as a measure
```

## Parts and questions

- Predicates: BOOL-001 conditions on scope columns (exact constants, BETWEEN, IS [NOT] NULL, BIT, column
  comparisons) combined with AND / OR / NOT / XOR (Kleene, SQL's), hash-consed: equal conditions (reduced constants)
  are one of at most 64, structurally equal nodes (commutative operands ordered) are one node of a shared DAG.
- Measures: polynomials over the scope's exact logical integers with rational coefficients, canonical (equal
  measures are one); a measure is NULL on a row where a column it reads is NULL.
- Queries: (filter or none, measure or none = COUNT); a query's rows are those where its filter is TRUE.
- Questions: predicates equivalent (FILTER: the same TRUE rows; KLEENE: the same truth value) or implies
  (containment of the TRUE rows); measures equivalent with NULL semantics SQL (one side NULL differs) or SKIP (such
  rows excluded); queries equivalent or implies (rows, then measures on the shared rows).

## Domains, verdicts and labels

- Snapshot (`check`): every selected row decided exactly (the conditions once as BOOL-001 masks, each question's
  cone word-parallel, measures by GEOM-001's exact pair candidate P = D (f - g)). Proven: label snapshot, an FND-007
  record of the scope and, for measures, the snapshot certificate of P = 0 (policy SKIP). Refuted: the smallest
  differing row and its values. With relation certificates, a measure question is first reduced by GEOM-001
  (complete Groebner basis): NF(f - g) = 0 gives label relations (the derived certificate; SKIP semantics or equal
  supports only); the rows still refute.
- Universal (`decide`), never reading rows: enumerate (every point of the declared product: label domain); decide
  (ranges only): predicates per NULL pattern of the nullable columns read, the TRUE / FALSE sets as reduced ordered
  BDDs over the conditions, every satisfying cube of the difference checked for an integer point (bounds and
  difference constraints x - y <= c by Bellman-Ford, disjunctions branched); BDD equality with no theory step over
  nullable columns is label universal. Measures: P = 0 canonically is universal (FND-007 RING_IDENTITY law); else a
  NULL-pattern mismatch (SQL) or the Nullstellensatz grid (deg_v P + 1 values per variable near 0, or the whole range
  when smaller) gives a counterexample, or proves P vanishes on the box (label domain). Every counterexample is a
  row of a synthetic array scope decided by the snapshot path (a point found by the theory is evaluated on a one-row
  scope before it is reported; grid and enumeration points are the rows evaluated).
- Unknown, never a guess: BIT conditions or column comparisons across scales in a cube that decides the answer,
  value-set columns for predicates in decide mode, words past 512 bits (unsupported); a budget reached (resource,
  need, limit); undecided rows (snapshot).
- Authorization (`authorize`): universal always; snapshot only on a scope whose FND-007 record matches exactly;
  domain only on a scope whose selected values (columns read) lie in the domain with the same scales; relations as
  FND-007 authorizes the derived certificate.

## Families

`run`: every query's COUNT, measure rows present and exact SUM (D f summed in W + 1 words; the sum is value / D).
SHARED: one scan (each distinct condition once, the shared DAG once per word, each distinct measure once per row);
SEPARATE: one scan per query. Both give identical answers; the statistics count the scans, condition rows, node
words and measure rows actually evaluated.

## Exactness, CUDA, budgets

Counts are integers, measures exact W-word values (widths proven from the scope's extremes). `tr_mgpu_equiv_check`:
work items (question, word block) run the same inline code, add counters with integer atomics and take the first
witness with an atomic minimum (the minimum row id); the host validates the view, budgets the device copies
(`tr_equiv_device_need`), and the CPU adopts the evidence after consistency checks (counts, cells, the witness's word
re-evaluated). Budgets: sets (circuit, support, output, bytes; variables = distinct conditions), checks (output,
bytes; BOOL-001, GEOM-001 and FND-007 check theirs), decisions (support, bytes, circuit = BDD nodes, iterations =
patterns + cubes + branches), runs (states, output, bytes). Representation limits (TR_MATH_UNSUPPORTED): 64
distinct conditions, 4096 parts or questions, 128 cone nodes per question, measure pairs of one check over at most 8
columns and 256 pairs, 20 nullable columns per universal question. The relationship optimizer's rewrite holds where
its catalog's rules were verified: it is a candidate, checked here like any measure (relationships_stage.c, an
executor transformation, is not used).
