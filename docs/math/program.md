# Query families as a multi-output program (META-002)

`torbles/math_program.h` (the contract; the finalization slot evaluator is shared with CUDA), `math_program.c`,
`math_program_gpu.cu` (the finalization kernel, and the fused pass kept on the device), `math_program.py` (opt-in).
A family of related queries over one FND-004 scope is compiled into one program: canonical conditions and
predicates, shared FND-006 states (one traversal of the rows for the whole family), and per-query finalization;
every query keeps its own domain, validity, group presence, values and explanation.

```python
from torbles import math_bank as mbk, math_bool as mb, math_program as mp
scope = mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"], group_by=["k"])
fam = mp.Family(scope, conditions=[mb.condition("x", ">", 0), mb.condition("x", ">=", 1), mb.condition("d", "is null")],
                groupings={"c": mbk.Scope.from_table(db["f"], [], group_by=[("c", db["dim"], "id")]),
                           "all": mp.rollup(mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"])),
                           "two": mp.rollup(map=ids, groups=2)})
pos, pos1, dn = mb.variables(3)
fam.query("q1", [mp.count(), mp.sum("x"), mp.avg({"x": 2, "x*y": -1, "1": 3})], where=pos)
fam.query("q2", [mp.count("y"), mp.histogram(["y"])], where=pos1, group="c")        # x >= 1 is x > 0: one class
fam.query("q3", [mp.sum("x*d")], where=~~pos & ~dn, group="all", scalar=True)        # no GROUP BY
prog = fam.compile(prove=True, lifetime="run")       # or lifetime="snapshot"
prog.info; prog.query_info("q2"); prog.explain("q2"); prog.need("fused", "cuda", "gpu")
res = prog.run(backend="cuda", finalize="gpu")       # plan="fused" | "repeated" | None (FND-006's planner)
res["q1"].rows()        # [{"key": (k,), "count(*)": 44, "sum(x)": 10387, "avg(...)": Fraction(9181, 10)}, ...]
res["q1"].explanation   # predicate class and proof, states and their sharers, per-measure paths, memory, dispatch
prog.compare("q1", "q3"); prog.snapshot_compare("q1", "q3")
```

## Queries and semantics

A query is `SELECT <keys>, <measures> FROM scope WHERE <predicate> GROUP BY <grouping>`: the predicate is a
math_bool expression over the family's BOOL-001 conditions (SQL three-valued; WHERE keeps the TRUE rows), the
grouping one of FND-006's (the scope's own, another scope of the same rows, a rollup), `scalar=True` (no GROUP BY:
one result row, the grouping must have one group; the default when the scope has one group). Measures:
`count()` (COUNT(*)), `count(p)`, `sum(p)`, `avg(p)` of an integer polynomial p (`"x"`, `"x*y^2"`, `"1"`,
`{monomial: int coefficient}`, `[(coefficient, monomial)]`; equal monomials merged, their coefficients added exactly
and the total required to fit int64, zeros dropped; logical values: a DECIMAL factor in units of 10^-scale) and `histogram(columns, weight=, repr=, domain=)` (DIST-001 cells per group).
SQL semantics: a row counts for p iff every column written in p is present (SQL evaluates the expression to NULL
otherwise, also when the column's terms cancel or have a zero coefficient: `x - x` and `0 * d + x` are NULL where
x, resp. d, is NULL); COUNT(p) is those rows, SUM(p) their exact sum or None, AVG(p) the exact `Fraction`
SUM / COUNT or None; COUNT(*) the group's rows. A grouped query lists its present groups (rows > 0) with their
decoded keys; a scalar query its single row, also when empty (COUNT 0, SUM None).

## Compilation

1. Canonical conditions, valid on every database of the schema: comparisons with constants become intervals of
   logical integers (`x > 0`, `x >= 1`, `x > 0.5` on an integer column: the same `BETWEEN 1 AND 2^63-1`; `x = 0.5`
   on an integer column: the empty interval; `x <> 0.5`: TRUE on every present value), `NE q` for an integer q,
   column comparisons ordered (`x > y` is `y < x`); equal ones are one condition.
2. Canonical predicates: hash-consed nodes rewritten only by identities of SQL's Kleene logic: double negation,
   constants (AND: FALSE absorbs, TRUE drops; OR dually; XOR: FALSE drops, TRUE negates, UNKNOWN absorbs), De Morgan
   (NOT moved to the leaves, folded into the complementary condition where one exists: `NOT x >= L` is `x <= L-1`,
   `NOT IS NULL` is `IS NOT NULL`), associative-commutative flattening with sorted operands, AND / OR idempotence.
   Never excluded middle (`x OR NOT x` is UNKNOWN where x is). A predicate whose canonical form is TRUE is no filter.
   Each node is negated once (shared subexpressions too: the work is linear in the DAG). Queries with the same
   canonical node form one class (proof `canonical`).
3. `prove=True`: the bounded universal search below over the pairs of distinct canonical predicates with the same
   condition support; proven pairs join one class (proof `search`); pairs left unknown or budget-limited stay apart
   (`info["search_unknown"]`, `info["search_budget"]`). Iterations (assignments evaluated) are budgeted.
4. States: per (class, grouping) one FND-006 SQL-mode bank with the monomials of every measure that reads no column
   or whose canonical monomials each read exactly its written columns (its per-term validity is then the measure's)
   and the rows for COUNT(*); per (class, grouping, written column set) one JOINT-mode bank for the other polynomials
   (listwise over exactly that set; a written column no canonical monomial reads gets a degree-1 term); one
   histogram per (class, grouping, spec). The FND-006 plan computes them all (fused: one traversal; or the repeated
   plan).
5. Finalization: per measure its state, its terms (index in the state's basis, coefficient), the term whose count is
   its valid rows, its groups and value slots.

## Execution and devices

`run(backend, finalize, plan, devices, force, budget)`: the FND-006 pass (the planner's choice, or `plan=`) on
`"cpu"` or `"cuda"`, then the finalization on `"cpu"` or `"gpu"`: one value per (measure, group) slot, SUM as an exact
256-bit contraction of the shared 128-bit sums. `cuda` + `gpu` with the fused plan keeps the states on the device
between the fused kernel and the finalization kernel (`tr_mgpu_program_run`: two launches); a CPU-built state can
be finalized on the device (`tr_mgpu_program_finalize`) and a device-built one on the CPU. CPU, emulation and device
give identical values (the slot evaluator is one inline function; no atomics). `res.how` and every
`explanation["devices"]` record the plan, the pass device and the finalization device (`"cuda-emulated"` in the
emulation); `explanation["measures"]` records per measure the device whose pass built its state and the one that
finalized its values (a histogram's cells are its state as the pass built them: no finalization).

## Explanations, costs, lifetimes

`prog.explain(name)` / `res[name].explanation`: the predicate's canonical node, class, proof and class size, the
conditions it reads; the states it reads (kind, mode, terms, grouping, the other queries sharing each, their exact
bytes after the run); its value slots and bytes; per measure its path; dispatch (the shared pass: plan, scans,
kernel launches, native calls, queries sharing it; the finalization device and slots); memory (per state: bytes and
readers). Lifetimes: `"run"` keeps nothing between runs (every run traverses again); `"snapshot"` keeps the last
run's states and values for later runs of the same program while its scopes are fresh (`res.reused`, scans 0,
whatever the arguments of the later run); a stale scope raises `StaleStateError` and drops them; `drop()` ends the
lifetime. Programs never share states with each other and no cache is consulted implicitly.

## Equivalence: universal search and snapshot check

`prog.compare(a, b, budget=)` decides the two queries' WHERE predicates over the declared universe (a column is
NULL only where the schema allows it: a table column with a validity column, an array column whose validity array
marks a selected value NULL (the scope drops an all-present validity array); any int64 logical value otherwise):
`equivalent` by canonical form, or by search (every assignment of TRUE / FALSE / UNKNOWN to the conditions read on
which [a = TRUE] and [b = TRUE] differ is infeasible: per column, NULL or a value in the intervals, holes and points
the assignment demands); `different` with a witness row (`{column: value or None}`) and the query that accepts it;
`unknown` when a differing assignment involves a BIT or column-comparison condition (their feasibility is not
decided); `budget` when 3^k assignments exceed the iterations budget. It also reports `grouping_equal`,
`measures_equal` (equal values on every database: COUNT(p) by p's written columns, its coefficients ignored, and
COUNT of a polynomial over no column is COUNT(*); SUM and AVG by the written columns and the canonical polynomial;
histograms by the normalized spec) and `equivalent_queries` (all three). A filter outcome is about the WHERE clauses:
queries with different filters can still return equal results (e.g. a difference only on rows a measure skips).
`prog.snapshot_compare(a, b)`: both predicates on this snapshot's selected rows in one pass: the rows each keeps, the
rows where they differ and the first one; `equal` holds for this snapshot only, never as a universal claim.

## Budgets and refusals

`prog.need(plan, backend, finalize)` is what `run` checks before it allocates anything (natively, `tr_prog_need`):
FND-006's need for the pass (docs/math/reduce.md) with output = the larger of its and the value slots and bytes =
its plus the finalization's; it bounds every later check of the run but the occupied cells of sparse histograms
(FND-006: checked once counted). A run at its need succeeds and one unit less of any resource refuses first (tested
for every resource and every pass / finalization combination).

| check | needs |
|---|---|
| compile, before allocating | circuit = conditions + nodes, output = queries, support = polynomial terms |
| after canonicalization | circuit = canonical conditions + nodes; iterations = the proof search's assignments (a pair over budget is left apart) |
| once the tables exist | output = queries + value slots, bytes = the program (its tables, the FND-006 plan) + 72 per slot |
| FND-006 compile | its own (docs/math/reduce.md) |
| a run (`prog.need`) | the pass's need; output = max(its, slots); bytes = its + the finalization's |
| CPU finalization (`tr_prog_finalize`) | output = slots, bytes = the state views + 72 per slot |
| GPU finalization (`tr_mgpu_program_finalize`) | output = slots, bytes = its device tables, the view, the values and the states copied in (`tr_mgpu_program_run`: the fused pass's device need + the tables, view and values) |
| `compare` | iterations = 3^k for the k conditions read (`budget` outcome past it) |

Refusals: `AlgebraBudgetError`, `ValueError` (malformed input, a scalar query over several groups; `.failed` = the
query), `OverflowError` (FND-006's proof, a coefficient sum out of int64), `UnsupportedError` (4096 queries, 16384
measures, 65536 terms, 1024 terms per measure, 64 canonical conditions), `StaleStateError`, `KeyError` (an unknown
query).

## Relationship with META-001

META-001 (`math_equiv`, docs/math/equiv.md) decides equivalence, implication and containment of predicates,
measures and queries (BDDs with integer difference logic, Nullstellensatz grids, snapshot certificates) and runs
query families with a shared scan for COUNT and SUM. META-002 compiles a family into shared FND-006 states with
GROUP BY, AVG and histograms, its own canonical forms and a bounded search used to merge predicate classes; its
`compare` is that search (WHERE predicates only, measures compared canonically), not META-001's decision procedures.

## Limitations

The universal search decides interval, NE and NULL conditions per column; BIT and column comparisons make it
`unknown`. Compile-time merging by search compares only predicates with the same canonical support (a predicate
equivalent to one over fewer conditions stays apart; `compare` still decides the pair over the union). The canonical
form does not distribute AND over OR or absorb (`a AND (a OR b)`). Witness rows respect declared nullability only:
not keys, references, type ranges (a DECIMAL's precision) or other schema constraints. Measures are COUNT / SUM /
AVG of integer polynomials and histograms (no MIN, MAX, variance or ratio finalizers); coefficients are int64.
Grouped queries over huge group domains allocate one value slot per group of their grouping (budgeted). Snapshot
lifetimes are kept in the Python object (reuse is decided by the native freshness checks of the scopes); the CPU
paths are single-threaded.
