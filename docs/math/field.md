# GF(2) vectors, matrices and parity dependencies (FIELD-001)

`torbles/math_field.h/.c` (native CPU), `math_field_gpu.cu` (CUDA kernels: batched syndromes, exhaustive relation
verification), `math_field.py` (opt-in Python API). Packed GF(2) linear algebra with field laws kept apart from integer
counts, and exact parity (XOR) dependencies among Boolean columns and bit planes of a data scope. The header comment of
`math_field.h` is the normative contract; this page summarizes it.

```python
from torbles import math_bank as mbk, math_field as mf
A = mf.Matrix.from_bits([[1, 0, 1], [0, 1, 1]])      # or Matrix.from_ints(rows, cols), Matrix(rows, cols, words)
A.rank(); R, pivots = A.rref(); N = A.nullspace()    # canonical RREF and nullspace basis ((cols - rank) x cols)
A.solve(b)        # {"status": "unique" | "family" | "none", "rank", "x", "basis", "certificate"}
A @ B; A + B; A.T; A.matvec(x); H.syndromes(X, backend="cuda")   # H x_j for every row x_j of X
A.row_sum(); A.column_counts()                       # the GF(2) sum of the rows vs the integer counts per column
mf.parity(v, n); mf.dot(a, b, n); mf.weight(v, n)    # field elements 0 / 1 vs the integer Hamming weight
scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "code"], rows=mask)
bits = mf.Bits.from_scope(scope, ["a", "b", ("c", "nonzero"), ("code", "code", 0), ("code", "bit", 63)])
bits.verify([(["a", "b", "c"], 1)], mode="sql", backend="cuda")   # checked, excluded, violations, witness, holds
bits.propose(sample=256)                             # candidate generation only
bits.discover(sample=256, affine=True, backend="cuda")   # {"relations", "rejected", "complete", "rank", "rounds", ...}
bits.combine((["a", "b"], 0))                        # the substitution: words of a XOR b on its checked rows
```

## Encoding

A matrix is rows x cols bits, row-major, 64 columns per uint64 word (bit j of a row: bit j % 64 of word j / 64),
`stride >= ceil(cols / 64)` words per row. It must be canonical: every bit past `cols` (the tail of a row's last word
and the words up to the stride) is zero, otherwise every operation refuses it (`ValueError`). Vectors are one row
(Python ints with bit j = entry j, 0/1 sequences or uint64 words). Empty matrices (0 rows or 0 columns) are valid.
Rows and columns are at most 2^31 (`Matrix` refuses larger shapes with `ValueError`; the native limit is
`TR_MATH_UNSUPPORTED`).

## Field laws, not integer arithmetic

Addition is XOR, multiplication AND: `A + B`, `row_sum` (XOR of the rows), `dot` (parity of popcount(a AND b)) and
products are GF(2) results; `weight` (popcount) and `column_counts` are integers. They agree only mod 2:
`parity(v) = weight(v) mod 2`, `row_sum` bit j = `column_counts[j]` mod 2, and
`weight(a ^ b) = weight(a) + weight(b) - 2 weight(a & b)`. Over table rows, a relation XOR = 0 holds where the integer
sum of its bits is even, not where it is 0.

## Elimination and solutions

Gauss-Jordan on packed rows, columns in increasing order (pivot: the first row at or below the current one with the
bit set). The reduced row echelon form is unique; pivots increase; the nullspace basis is canonical (one vector per
free column f: bit f, and bit pivots[i] = R[i][f]). `solve` eliminates `[A | b]` (and `[A | b | I]` for the
certificate): `none` (with y: y^T A = 0, y^T b = 1), `unique`, or `family` (x with every free variable 0, plus the
nullspace basis). Exhaustively checked on every small binary system; exact by construction (bits only).

## Parity dependencies over rows

- Bit variables (`Bits.from_scope`): a column name or `(name, "nonzero")` (logical != 0: Boolean columns),
  `(name, "bit", b[, base])` (bit b of (logical - base) mod 2^64), `(name, "code", b)` (bit plane b of the stored code:
  base = the codec offset). NULL values make the variable NULL on that row; only selected rows count.
- A relation (variables, constant) `XOR x_j = constant` is checked on the selected rows where its variables are
  present (`mode="sql"`, SQL's `a <> b <> ...`) or where every variable is (`"listwise"`); other selected rows are
  excluded. Evidence: checked, excluded, violations, the smallest violating row (witness), holds (no violation and no
  exclusion: an exact rewrite over the selection).
- Discovery: every relation (affine: with a constant) holding on every complete selected row (every variable
  present). Proposals: the canonical nullspace basis of a deterministic sample (`sample` complete rows at an even
  stride; 0 = all) over vectors `[1, x_0, ..]` (the constant first, so a dependent variable is expressed by earlier ones
  and the constant). Exact verification of every proposal over every complete row; a failing proposal is reported in
  `rejected` with its evidence (a one-row near miss: violations 1 and its witness) and its witness row joins the
  sample (it lies outside the sample's row space, so the rank grows: at most columns + 1 rounds). The final proposals
  all hold, so `relations` is exactly the canonical basis of the complete rows' relation space. With NULLs, the found
  relations are exact on the complete rows only (`holds` is false: excluded > 0). With no complete row every vector
  holds vacuously (including the contradiction `0 = 1` when affine).

## CUDA

`tr_mgpu_field_syndromes`: one work item per (vector, 64-bit syndrome word), one output slot each.
`tr_mgpu_field_verify`: work items (relation, word slice) XOR the relation's words per 64-row word, count with
popcounts and add checked / excluded / violations with one atomic each and the witness with an atomic minimum.
`tr_mgpu_field_discover`: the same discovery loop (shared inline code of the header) with the device verification; the
bits are copied once. Elimination, rank, nullspace and solve have no CUDA path. Results equal the CPU's bit for bit
(emulated with a scrambled thread order in the tests).

## Budgets and refusals

Checked before allocation (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`):

| request | needs |
|---|---|
| rank / rref / nullspace | tensor = rows x cols; bytes = the scratch copy (rref: none); output = rows x cols (rref) or nullity x cols (nullspace, checked after elimination, before writing) |
| solve | tensor = rows x (cols + 1 + rows with the certificate); bytes = scratch; output = cols + nullity x cols + rows |
| mul / transpose / syndromes | tensor = the elements involved; output = the result's elements; CUDA syndromes: bytes = device copies |
| bits | variables = nvars; bytes = object + 8 nw (1 + 2 nvars) |
| verify | output = relations; CUDA: bytes = device copies |
| discover / propose | variables = nvars + affine; support = sample rows (0: selected rows); iterations = columns + 1 (propose: 1); output = columns + columns (columns + 1) / 2; bytes = proposals (+ device copies) |

Malformed input (non-canonical matrices or vectors, shape mismatch, overlapping outputs, bad specs or relations):
`ValueError`. A stale scope (the table changed): `StaleStateError` before every use. Representation limits: more than
64 bit variables, discovery over more than 64 columns: `UnsupportedError`.

## For the next capabilities

SPEC-002 (Walsh spectra) and QUANT-002 (relational division) use the matrix primitives on caller words
(`tr_field_matrix` views: `tr_field_rank`, `_rref`, `_nullspace`, `_solve`, `_mul`, `_syndromes`, the inline
`tr_field_row_dot`, `tr_field_syndrome_word`); FIELD-002 (parity discovery and substitution) builds on
`tr_field_bits` (`tr_field_bits_get_view`), `tr_field_verify`, `tr_field_propose`, `tr_field_discover` and its inline
loop `tr_field_discover_run` (any verifier callback), the single-word echelon helpers (`tr_field_basis_insert`,
`tr_field_basis_relations`) and `tr_field_bits_combine` (the substituted column).

## Limitations

Plain O(rows x cols x words) Gauss-Jordan (no Method of Four Russians); dense matrices only; discovery is over at most
64 columns and uses the listwise domain; certificate invalidation under codec changes and revisions is FIELD-002's;
the CPU paths are single-threaded.
