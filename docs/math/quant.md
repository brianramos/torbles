# Quantified group predicates and relational division (QUANT-001, QUANT-002)

`torbles/math_quant.h` (every contract; the answer rules, the predicate word evaluator and the device sizes shared with
CUDA), `math_quant.c`, `math_quant_gpu.cu`, `math_quant.py` (opt-in). Exact per-group states over the child rows of a
FND-004 scope: for QUANT-001 the rows where a BOOL-001 predicate is TRUE, FALSE and UNKNOWN, from which every SQL
quantifier follows; for QUANT-002 the set of categories each group holds, tested against a required set (relational
division). The header comment of `math_quant.h` is the normative contract; this page summarizes it.

```python
from torbles import math_bank as mbk, math_bool as mb, math_quant as mq
lines = mbk.Scope.from_table(db["lines"], ["qty", "price"], group_by=[("oid", db["orders"], "id")])  # children by parent
orders = mbk.Scope.from_table(db["orders"], [], group_by=["id"])              # the parent rows
q = mq.Quantifier.build(lines, [mb.condition("qty", ">", 5), mb.condition("price", "is null")],
                        mb.var(0) | mb.var(1), parents=orders, backend="cpu")  # or "cuda"; parents=None / "ids"
q.state(g)            # {"present", "rows", "true", "false", "unknown", "parents"}  exact ints
q.answer("all", g)    # True / False / None (UNKNOWN) / mq.ABSENT;  q.answers("some") -> uint8 codes 0 F, 1 T, 2 U, 3 ABSENT
q.table(["some", "all", "exists"]); q.tally("none"); q.info; q.identity; q.same(other); q.matches(lines, conds, pred)
q.merge(other_partition)                       # the bag union: counts add exactly
cats = mbk.Scope.from_table(db["lines"], [], group_by=["item"])               # the same rows, grouped by category
cov = mq.Coverage.build(lines, cats, ["apple", "pear"], parents=orders, representation="auto", backend="cpu")
cov.state(g)          # {"present", "rows", "nulls", "parents", "distinct", "matched", "missing", "extra"}
cov.categories(g, "missing"); cov.codes(g, "extra"); cov.answer("divides", g)   # divides / exactly / overlaps / disjoint
cov.require(["fig"], backend="cuda")           # another required set over the same presence states (no rescan)
cov.masks()           # a math_field.Matrix: groups x categories bits (FIELD-001 packed, canonical)
mq.Dictionary.from_scope(cats), mq.Dictionary.codes(D, null=..., offset=..., scale=...), mq.Dictionary.keys(values, null=...)
mq.groups_need(...), mq.cover_need(..., representation="masks" | "sets", backend=...), mq.dictionary_need(...)
```

## Groups, parents and presence

- The child rows are the selected rows of the child scope; a row's group is its FND-004 group id (the parent key,
  e.g. a foreign key's dimension attribute). Every selected row counts once (bag multiplicity).
- Presence is declared, apart from the counts: `parents=None` (GROUPED: a group exists iff it has child rows, SQL GROUP
  BY), `"ids"` (every group id is a parent), or a parent Scope (PARENTS: a group exists iff the parent scope has a
  selected row in it; `parents` counts them). Parent group g is child group g: the parent grouping has the same key
  codecs and radices (the first key's radix may be smaller: the child's NULL code lies past the parent ids); the Python
  layer checks this from the key specs, natively the parent scope may not have more groups.
- A present group without child rows is **empty** (vacuous answers); an **absent** group answers `ABSENT`, never FALSE
  or UNKNOWN; child rows of absent groups (orphans) are counted (`info["orphans"]`), never answered.
- NULL keys (a NULL key, an unmatched foreign key, a NULL dimension attribute: FND-004's NULL group): a group of its own
  under GROUPED; under PARENTS with a primary-key parent no parent (SQL's correlated `c.key = p.key`); with a nullable
  parent key, the parents whose key is NULL are its parents (IS NOT DISTINCT FROM over the children LEFT JOIN the
  dimension).

## QUANT-001: quantifiers

The predicate is a BOOL-001 program over native row conditions of the child scope, evaluated per row with SQL's
three-valued logic. Per group t, f, u = the rows where it is TRUE, FALSE, UNKNOWN, n = t + f + u:

| quantifier | answer | SQL |
|---|---|---|
| `some` | TRUE if t > 0, else UNKNOWN if u > 0, else FALSE | `TRUE = ANY (SELECT p FROM c WHERE ...)` |
| `all` | FALSE if f > 0, else UNKNOWN if u > 0, else TRUE | `TRUE = ALL (SELECT p ...)` |
| `none` / `not_all` | NOT some / NOT all | `NOT (TRUE = ANY ...)`, `NOT (TRUE = ALL ...)` |
| `exists` / `not_exists` | t > 0 / t = 0 (two-valued) | `[NOT] EXISTS (SELECT 1 ... AND p)` |
| `no_counterexample` | f = 0 (UNKNOWN does not refute) | `NOT EXISTS (... AND NOT p)` |
| `all_true` | t = n (UNKNOWN refutes) | `NOT EXISTS (... AND p IS NOT TRUE)` |
| `bool_or` / `bool_and` | UNKNOWN if t + f = 0, else t > 0 / f = 0 | `bool_or(p)`, `bool_and(p)` |

Conventions: vacuous truth for an empty group (some FALSE, all TRUE, none TRUE, not_all FALSE, exists FALSE,
no_counterexample TRUE, all_true TRUE, bool_or / bool_and UNKNOWN as SQL aggregates over no input); an all-NULL group is
not an empty group (some, all, none, not_all, bool_or, bool_and UNKNOWN; exists FALSE; no_counterexample TRUE; all_true
FALSE); `x OR NOT x` is UNKNOWN where x is (SQL evaluation). Multiplicity counts rows but never erases coverage: a
duplicated row doubles the counts and changes no answer (every answer depends only on which counts are zero); states
combine by integer addition (`merge`), never by a parity.

## QUANT-002: coverage and division

- Categories: a second scope over the same rows (same table, version and selection bits) whose group ids are the
  category codes (a string column's dictionary codes, an integer column's codes, a dimension attribute). The dictionary
  (the codec) gives the codes their meaning: `codes` (key = code + offset, value = key / 10^scale) or `keys` (one exact
  string per code); `Dictionary.from_scope` reads the category key's codec. One code may be the NULL category.
- A NULL category never satisfies a requirement and is never extra (its rows are counted in `nulls`); duplicated
  categories count once for coverage (sets, OR masks) and every row counts in `rows`.
- Required set: values resolved natively against the dictionary (duplicates merged); values outside the dictionary and
  `None` (a NULL requirement: SQL's `c.cat = NULL` never matches) are missing in every group.
- Per group: `distinct` (present categories), `matched` (present & required), `missing` (required & not present, plus
  the outside keys and the NULL requirement), `extra` (present & not required). Answers: `divides` (missing = 0: the
  classic double NOT EXISTS; vacuously TRUE for an empty required set, also for an empty group), `exactly` (divides and
  no extra), `overlaps` (some required category present), `disjoint`.
- Representations: `masks` (per group D bits, a canonical FIELD-001 matrix) or `sets` (sorted distinct codes per
  group). `auto` chooses by cost (the bytes each stores) among those within the budget (`tr_math_budget_choose`: masks
  are the expanded representation, sets the factored one); a forced representation over budget is refused. The CUDA
  path builds masks.
- Identity: exact bytes, the presence part (domain, the FND-004 descriptors of children, categories and parents, the
  dictionary: kind, size, NULL code, offset and scale or every key) and the requirement part (sorted required codes,
  sorted outside keys, the NULL flag). Another required set or another dictionary is another state (`same`, `matches`);
  `require` derives the state of another set from the same presence states and equals the state a build gives.

## Execution and CUDA

- CPU, one pass over the selection words: QUANT-001 evaluates per 64-row word the conditions the root reads and the
  program (BOOL-001's word evaluator on a one-word domain), then adds popcounts to the word's group when all its
  selected rows share one (contiguous groups) and attributes row by row otherwise. QUANT-002 ORs each row's bit into
  its group's mask, or groups the codes, sorts and deduplicates each group; the coverage test is AND-NOT and popcount
  against the required mask (sets: a binary search of the required codes).
- CUDA (`tr_mgpu_quant_groups`, `tr_mgpu_quant_cover`, `tr_mgpu_quant_cover_test`): work items are slices of
  consecutive words; a run of rows of one group is accumulated locally and added with 64-bit atomics when the group
  changes (a contiguous word costs four atomics, scattered groups about one per row); mask words are combined with
  atomic OR (pending bits of one word kept locally); the coverage test is one work item per group with its own output
  slots; parent rows have their own scan. The host adopts the device arrays after checks (per group TRUE + FALSE +
  UNKNOWN = rows and the totals; canonical masks, popcounts equal to `distinct`, the coverage identities). The
  emulation (scrambled thread order) equals the CPU bit for bit in the tests; the device run is pending.

## Budgets and refusals

Checked before anything is allocated (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| quantifier state | variables = conditions; circuit = program nodes; states = output = groups; bytes = object + 40 per group + identity + descriptor copies + scan scratch (16 per condition and node); CUDA: the device copies (columns read, ids, program, per-item scratch, counts) |
| dictionary | support = codes; bytes = object (+ keys, offsets, index and sort scratch) |
| coverage state | support = categories; states = groups x categories (masks) or the selected rows (sets); output = groups; bytes = object + 56 per group + masks (8 per word) or sets (offsets, 4 per row, cursors) + the required set and its identity bound + the dictionary copy; CUDA: the device copies |
| merge / require | the new object |

Malformed input (`ValueError`): unknown quantifiers or questions, conditions over missing columns, a non-aligned
parent grouping, duplicate dictionary keys, a key for the NULL code. `DomainError`: a predicate over another number of
conditions, more parent groups than child groups, categories over other rows, a dictionary of another size, merging
states of other predicates, dictionaries, required sets or domains. `TypeError`: a float category. `UnsupportedError`:
sets on the device, a category scope with several keys, a float or BLOB dictionary. `StaleStateError`: a scope's table
changed (checked before every use).

## Limitations

The category of a row is one group id of a single-key category scope (a multi-key category is not supported, nor
float or BLOB dictionaries, dates or timestamps as categories); merged states record the identities of their parts (a
merged state never matches a build); the dictionary `codes` key is the 8-byte little-endian logical value; the CPU
paths are single-threaded; the sets representation and the merges have no CUDA path; DIST-001 histograms are not used
(the states here are this module's own).
