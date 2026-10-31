# Observed-domain constraint counts and witnesses (COUNT-001)

`torbles/math_count.h/.c` (native CPU), `math_count_gpu.cu` (CUDA kernels: counts with atomics, chunked extraction),
`math_count.py` (opt-in Python API). For a BOOL-001 predicate or a set of constraints, the exact number of points that
satisfy, violate or leave UNKNOWN each constraint, every satisfying point (bounded by the output budget), the smallest
violating points (counterexample witnesses), and a certificate saying exactly which points were covered. The header
comment of `math_count.h` is the normative contract; this page summarizes it.

```python
from torbles import math_bank as mbk, math_bool as mb, math_count as mc
x = mb.variables(3)
cs = mc.Constraints(3, [mc.implies(x[0], x[1]), mc.equiv(x[1], x[0] | x[2]), mc.excludes(x[0], x[2]),
                        mc.holds(x[0] ^ x[2])])          # one shared DAG; a predicate is a holds constraint
cs = mc.Constraints.from_nodes(3, [("var", 0), ("var", 1)], [("implies", 0, 1)])
dom = mc.Domain.observe(scope, [mb.condition("a", ">", 5), mb.condition("b", "is null"),
                                mb.condition("c", "bit", 3)], backend="cpu")     # or "cuda": device condition words
uni = mc.Domain.universe(3)                              # the 2^3 assignments: another, labelled result
ans = mc.count(dom, cs, rows=True, witnesses=5, backend="cpu")                  # or "cuda"
ans.results       # per constraint: constraint, a, b, points, satisfied, violated, unknown, witness, rows, witnesses
ans.certificate   # kind, domain, points, rows, support, complete, observed, universe, unobserved, covers_universe
ans.source()      # the FND-007 record of the rows covered (None for the universe)
ans.fresh(scope); ans.authorize("snapshot", scope); ans.authorize("universe", scope); ans.decide(...)
mc.predicate(dom, expr, rows=True); mc.domain_need(scope, conds, backend); mc.count_need(dom, cs, rows=True)
```

## Constraints and their truth values

SQL three-valued logic (Kleene's connectives, BOOL-001): at every point a constraint is satisfied (TRUE), violated
(FALSE) or UNKNOWN, and UNKNOWN is counted apart, never as satisfied or violated.

| constraint | value | violated where |
|---|---|---|
| `holds(a)` | a | a is FALSE |
| `implies(a, b)` | NOT a OR b | a TRUE and b FALSE (a NULL premise: UNKNOWN unless b is TRUE) |
| `equiv(a, b)` | NOT (a XOR b) | both known and different |
| `excludes(a, b)` | NOT (a AND b) | both TRUE |

Operands are nodes of one predicate DAG over the k conditions; a subexpression shared by several constraints is
evaluated once per word (CPU). satisfied + violated + unknown = points for every constraint.

## Domains

- **Observed** (`Domain.observe`): the selected rows of an FND-004 scope, each once; the conditions are BOOL-001 row
  predicates evaluated into value and valid bitmaps. The domain keeps an FND-007 record of the scope over the columns its
  conditions read: lifetime token, revision, rows, selected rows with their exact bits, each column's encoding (stored
  column, offset, scale) and NULL domain. Points are physical row numbers.
- **Universe** (`Domain.universe(k)`): the 2^k assignments (bit i of an assignment is condition i), enumerated; small k
  only (default support budget 2^20: k <= 20). No data is read. Points are assignments.

## Results

`witness` is the smallest violating point (None without one); with `rows=True`, `rows` holds every satisfying point in
increasing order; `witnesses` holds the min(w, violated) smallest violating points. The satisfying points are refused
past the output budget (AlgebraBudgetError with the exact need: constraints + satisfying points + witnesses), never
truncated.

## The certificate

- Observed: kind `snapshot`. `points` = the selected rows covered (the record keeps their bits: `source()`), `support`
  = the conditions the constraints read (s of them), `complete` = covered rows where every support condition is known,
  `observed` = distinct assignments of the support among them, `universe` = 2^s, `unobserved` = universe - observed.
  The rows say nothing about the unobserved assignments: they were not checked. `authorize("universe")` is refused
  (`NotAuthorizedError`, reason `unobserved`, with the number) unless every assignment of the support occurs among the
  complete covered rows (a constraint's value at a point depends only on its support there).
- Universe: kind `law` (a property of the predicates by enumeration): always fresh, no scope;
  `authorize("snapshot")` is refused (reason `domain`): it covers no table rows.
- Freshness: the record against the current scope (FND-007 exact match): identity, revision, rows, selection,
  null_domain, columns, with the first reason and the recorded and current quantities. `authorize("snapshot", scope)`
  needs a fresh observed certificate (`StaleCertificateError` with `.reasons` otherwise).
- The coverage (complete, observed) is computed on the host from the domain's condition words in both paths (a 2^s
  bitmap for s <= 24, else a sort of the patterns).

## Execution

CPU: one pass over the 64-point words (the DAG's scheduled nodes once per word, each constraint's words, popcounts, the
first violating point), then, when points are requested and admitted, a second pass writing them. CUDA:
`tr_mgpu_count_counts` (work items (constraint, word slice) evaluate the constraint's cone, count with popcounts, one
atomic per counter and an atomic minimum for the witness); `tr_mgpu_count_extract` (per (constraint, chunk of 16 words):
the chunk's satisfying and violating counts in their own slots, their exclusive prefix on the host checked against the
counts, then each chunk writes its points at their ranks); `tr_count_adopt` checks the readback (constraints, counts
partitioning the points, the layout, every point a selected row in increasing order, the first witness) and builds the
answer. Results equal the CPU's bit for bit (emulated with a scrambled thread order in the tests).

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| constraint set | variables = k, circuit = nodes, bytes = the object and the walk scratch (the cones: 4 bytes per entry, once counted) |
| observed domain | variables = k, bytes = BOOL-001's masks (or the device copies), the conditions and a bound of the record |
| universe | variables = k, support = 2^k |
| count | variables, circuit, support (universe), output = constraints, states = 2^s (the coverage bitmap, s <= 24), bytes = node words, results, coverage scratch, answer; after the counts: output = constraints + points returned, bytes + 8 per point |
| CUDA | bytes = the device copies, every work item's node words (16 per node), accumulators; extraction: chunk counts and prefixes, outputs |

Representation limits (`UnsupportedError`): k > 64 (observed) or > 63 (universe), more than 2^24 nodes, more than 2^16
constraints. Malformed input (`ValueError`): bad nodes or constraints, an empty set, a readback that is not this
domain's. A set over another k than the domain: `DomainError`. A domain whose scope changed: `StaleStateError` before
every use.

## Limitations

Counting is over the observed rows or by enumeration of at most 2^20 assignments (default budget); symbolic counting
(decision diagrams, weighted models) is COUNT-002's. Each selected row counts once (no per-group counts, no weights);
the conditions are BOOL-001's (exact-valued columns only); the certificate is COUNT-001's own type built on FND-007's
record and staleness reasons (FND-007 has no claim kind for counts), and it is not serialized; dimension changes are not
recorded (groups are not read); the CPU paths are single-threaded.
