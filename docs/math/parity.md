# Parity relationship discovery and substitution (FIELD-002)

`torbles/math_parity.h/.c` (native CPU), `math_parity_gpu.cu` (CUDA kernel: verification of a substitution on every
row), `math_parity.py` (opt-in Python API). Exact XOR dependencies among Boolean columns and code bit planes are found
with FIELD-001, each verified relation is certified as an FND-007 snapshot certificate, and a predicate that reads one
side of a certified relation is rewritten over the other side when the certificate is fresh, the cheaper form chosen by
the columns and planes it scans. The header comment of `math_parity.h` is the normative contract; this page summarizes
it.

```python
from torbles import math_bank as mbk, math_bool as mb, math_parity as mp
scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "code"])
found = mp.discover(scope, ["a", "b", "c", ("code", "code", 0)], sample=256, affine=True, nulls="forbid",
                    backend="cpu")                       # or "cuda": FIELD-001's and FND-007's kernels, adopted
found.relations   # vars, labels, constant, FIELD-001 evidence, status, columns, certified (FND-007 evidence), certificate
found.rejected    # the proposals that failed: one-row near misses with violations 1 and their witness row
found.info        # FIELD-001's statistics (rounds, sample rows, rank, ...), certified, verified_by
conds = [mp.bit_condition(scope, "a"), mp.bit_condition(scope, "b"), mb.condition("d", ">", 5)]
x = mb.variables(3)
rw = mp.substitute(scope, conds, (x[0] ^ x[1]) & x[2], found.relations[0]["certificate"], mode="auto")
rw.chosen, rw.costs, rw.conditions, rw.original, rw.program, rw.info
mp.verify(rw, scope, backend="cuda")     # points, mismatches, witness, equivalent, original / substituted counts
mp.verify_programs(masks, a, b); mp.discover_need(scope, specs); mp.verify_need(masks, a, b, backend=...)
```

## Discovery and certificates

- Bit variables are FIELD-001's specs: a Boolean column (`name`: logical != 0), `(name, "code", b)` (bit plane b of the
  stored code: base = the codec offset), `(name, "bit", b, base)`.
- Discovery is FIELD-001's (proposals: the canonical nullspace of a deterministic sample; exhaustive verification over
  every complete row; failed proposals rejected with their witness row joining the sample). `relations` is the canonical
  basis of the relations holding on every complete selected row.
- Each relation becomes an FND-007 PARITY claim (its scope columns, its bits with kind, plane and base, the constant,
  the NULL policy `nulls`) verified exhaustively and certified (`tr_cert_snapshot`; on CUDA `tr_mgpu_cert_verify` and
  `tr_cert_adopt`). The certificate records the table identity and revision, the rows, the exact selection, each
  column's encoding and NULL domain. Status: `certified`; `violated` (FND-007 found violations: a NULL under
  `forbid`, or a row where this relation's own columns are present while another variable is NULL); `too wide` (more
  than 8 columns: no claim); `empty` (the vacuous contradiction 0 = 1 on no complete row).
- Invalidation is FND-007's: a codec change (offset, scale, stored column) makes the certificates over that column
  stale (`columns`), a NULL mask change those over that column (`null_domain`), a revision every certificate of the
  table (`revision`); certificates over other columns stay fresh (a table's revision moves with any mutation).

## Substitution

- The relation XOR_S x = c over the predicate's conditions: a relation bit is the condition `col <> 0` (Boolean) or
  `bit b of (col - base)` (a plane) of the claim's column; equal conditions of the predicate are used, missing ones
  appended (at most 64 conditions).
- Refused before anything is rewritten: a stale certificate (`StaleCertificateError` with FND-007's `.reasons`), a
  hypothesis, approximate or incomplete claim (`NotAuthorizedError`, reason `certificate`, `.detail`), a certificate
  that skipped NULL rows (reason `nulls`: the rewrite would change UNKNOWN rows), a claim that is not parity (`claim`).
- A pure XOR form is VAR x, TRUE / FALSE, NOT of a pure form, or XOR of two pure forms with disjoint variables (no
  variable cancels, so the form is UNKNOWN exactly where one of its variables is). Each maximal form (the root or an
  operand of a node that is not pure) meeting S is a candidate; its rewrite is the form plus (S, c): a variable that is
  the XOR of others becomes that XOR, an XOR of the others becomes the variable, a complete relation becomes a constant.
  Exact on every selected row: the variables added or removed are relation bits, known there.
- Cost: the columns and planes scanned by the reachable conditions (a BIT condition its plane, another condition its
  column; a scanned column subsumes its planes). `mode="auto"` keeps a rewrite only when the cost strictly drops
  (greedy, in node order): `chosen` is `rewritten` or `kept`; `mode="all"` rewrites every candidate.
- `verify`: both programs over the rewrite's conditions on every selected row; `mismatches` counts rows with another
  value or UNKNOWN on one side only, `witness` the smallest. CUDA: one work item per 64-row word (a strided slice past
  64 MiB of node words), both root bitmaps, their XOR, popcounts, one atomic per counter, an atomic minimum.

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| discover | variables = nvars + affine, support = sample (0: selected rows), iterations = columns + 1, output = the relation and rejection capacities, bytes = FIELD-001's bits and discovery and the found object; then FND-007's snapshot needs per relation |
| substitute | variables = k', bytes = the analysis scratch; after the choice circuit = the rewritten nodes, bytes + 16 per node |
| verify | variables = k, circuit = both programs' nodes, bytes = their node words (CUDA: copies, node words per work item, accumulators) |

Representation limits (`UnsupportedError`): more than 64 bit variables or discovery columns, more than 64 conditions
after appending the relation bits. Malformed input (`ValueError`): bad specs, modes or NULL policies, a program over
another number of conditions. Programs over another domain: `DomainError`. A stale scope: `StaleStateError`.

## Limitations

Rewrites only XOR forms with disjoint variables (an XOR that cancels a variable, or XOR under UNKNOWN, is left alone)
and only with one relation per call; the choice is greedy (not an optimum over all subsets of candidates); the cost is
the number of columns and planes, not their widths; the rewrite runs on the CPU (its verification has a CUDA kernel);
certificates record no dimension (parity claims do not read groups), and the substitution checks freshness without
dimensions, so a certificate made elsewhere that recorded one (a scope with lookups) is refused as stale; discovery is
limited by FIELD-001 (64 columns, listwise over complete rows); the CPU paths are single-threaded.
