# Proof certificates and evidence scope (FND-007)

`torbles/math_certificates.h` (every contract; the claim row semantics shared with CUDA), `math_certificates.c`,
`math_certificates_gpu.cu` (exhaustive re-verification on the device), `math_certificates.py` (opt-in). One native
certificate type for four kinds of claims that are never confused; the kind decides every later check.

```python
from torbles import math_bank as mbk, math_certificates as mc
scope = mbk.Scope.from_table(db["f"], ["x", "a", "s", "y", "p"], rows=mask)
cert = mc.snapshot(scope, mc.affine("a", "x", 3, -7))           # exhaustive; backend="cuda": the kernel verifies
cert.fresh(scope)          # Freshness(fresh, reasons, first, column, dimension, recorded, current)
cert.authorize_exact(scope)  # Authorization(kind, claim, field, nulls) or NotAuthorizedError / StaleCertificateError
cert.decide(scope, field=0)  # the decision as a dict, never raising
mc.hypothesis(scope, claim, size=100, seed=1, method="uniform")   # "stride", "prefix": never exact
mc.schema(dim_scope, mc.unique("id"), enforcement="load")         # an enforced schema fact ("write": every write checked)
mc.law(mc.polynomial({...}, ["u", "v"]))                           # "ring_identity"; "gfp_function" with field=p
mc.derive(claim, [premises], [cofactors], scope)                   # P = sum q_i G_i in Z[x]: weakest premise's kind
mc.verify(scope, [claims], backend="cuda")                         # exhaustive evidence of many claims at once
mc.reverify([certs], current_scope, backend="cuda")                # re-certification after an invalidation
cert.to_bytes(); mc.Certificate.from_bytes(data, names)            # canonical, address-free (FND-011 wraps it)
mc.Dimension(dim_table, "id"); mc.code_scope(table, ["fk"]); mc.need(scope, claims, backend=...); mc.describe()
```

## Kinds

| kind | evidence | fresh | authorizes an exact rewrite |
|---|---|---|---|
| `law` | decided natively: `ring_identity` (the canonical form in Z[x] is zero: every commutative ring), `gfp_function` (reduced by x^p = x and mod p it is zero: vanishes on GF(p)^n) | always | yes, in the carriers the law holds in |
| `schema` | a declared constraint (`unique`, `not_null`, `reference`) verified on every selected row; `enforcement` load (verified at load, as database.py does for a PRIMARY KEY) or write (the storage checks every write) | load: identity, revision, rows, columns, dimensions, current selection a subset of the verified one; write: identity, columns, dimension keys | yes when fresh |
| `snapshot` | exhaustive over every selected row of an exact scope, in a stated carrier, complete and without violation | every part of the recorded scope equal | yes when fresh and exact |
| `hypothesis` | the claim held on a sample (`uniform` with replacement and a seed, `stride`, `prefix`; population and distinct rows recorded) | as a snapshot | never (`statistical`) |

## Claims (one typed payload; one table entry per kind, `mc.describe()`)

| kind | statement (col0 is the output where there is one) | Python |
|---|---|---|
| affine, sum, product | col0 = scale col1 + bias, col1 + col2, col1 col2 (DISC-001) | `affine`, `sum_of`, `product` |
| parity | XOR of bit variables = constant over GF(2): `"nonzero"`, `("bit", b, base)`, `("code", b)` (FIELD-001) | `parity` |
| refinement | col0 = map(col1), the map total on the data (MULTI-001 as a dependency of key values) | `refinement` |
| coverage | every present group holds every required category (<= 64; NULL never counts) (QUANT-002) | `coverage` |
| polynomial | sum c_t prod col_v^e = 0, canonical terms; `field=p`: a GF(p) claim (DISC-002, GEOM-001) | `polynomial` |
| unique, not_null, reference | a key; no NULL; a foreign-key code naming a used dimension row | `unique`, `not_null`, `reference` |

- Carrier of the check: exact 128-bit integers (a polynomial row outside int128 is undecided: `OverflowError`, never
  wrapped) or GF(p) (`modulus=p`). Exact iff tolerance 0 and the check matches where the claim holds (Z checked in
  128 bits, GF(p) checked mod p). `tolerance > 0` (|residual| <= tolerance) and a Z claim checked mod p are
  **approximate** and never authorize.
- NULL policy (`nulls=`): `forbid` (a NULL violates), `skip` (excluded: the claim covers the complete rows; the
  authorization says `nulls="skip"`), `propagate` (output NULL exactly when an input is).
- Evidence: checked, excluded, violations, the smallest violating row (coverage: group; unique: the smallest row
  that is not its value's first occurrence), overflow, holds. The CPU loop and the kernel share `tr_cert_row`.

## The recorded scope and freshness

A snapshot records (`cert.source()`): the table's lifetime token, revision, rows, selected rows, the FND-004 version
digest, the exact selection bits, each claim column's encoding and NULL domain (NULL count and digest of the NULL
rows), the grouping descriptor for group claims, and every dimension it reaches through (the scope's lookups, a
reference's dimension: token, revision, rows, used rows, key uniqueness and NULLs, decided natively). No address.
`fresh` reports every reason that applies, the first in this order: `identity`, `revision`, `rows`, `selection`,
`null_domain`, `columns`, `grouping`, `join_revision` (a dimension changed or is absent), `join_key` (its key is no
longer unique or NULL-free). Only recorded dimensions are checked: a change of another one does not apply. An array
scope's identity is the digest of its selected present values.

## Authorization

`authorize_exact(scope, field=0)`: a law in a target it holds in (`ring_identity` everywhere, `gfp_function` only in
its GF(p)), a fresh schema fact, or a fresh exact snapshot (a Z claim also authorizes a GF(p) target). Refusal reasons:
`statistical`, `approximate`, `carrier`, `no scope`, `stale` (`StaleCertificateError` with `.reasons`), `premise`.

## CUDA

`tr_mgpu_cert_verify`: work items (claim, contiguous slice of rows) count locally and add with one atomic per
counter, the witness with an atomic minimum; unique keeps first occurrences by atomic minimum over the column's
[min, max] domain and counts the rest in a second launch; coverage marks present groups and ORs category bits into
per-group masks, then one launch over (claim, group slice). `tr_cert_adopt` turns the evidence into a snapshot after
consistency checks; the certificate says `verified_by="cuda library"`. Equal to the CPU bit for bit (emulated with a
scrambled thread order in the tests).

## Budgets and refusals

| request | needs |
|---|---|
| claims | variables = columns, degree = polynomial degree, support = terms / map entries / required values / dimension rows / 64, output = claims |
| verification (CPU) | bytes = the largest scratch (coverage: 8 per group + bitmap; unique: 16 per selected row), states = coverage groups |
| verification (CUDA) | bytes = device copies (selection, ids, the columns read, claims and payloads, accumulators, masks, 8 per value of a unique column's domain) |
| certificates, records | bytes = objects, payload copy, selection bits, grouping bytes; dimensions: 8 (12 with validity) per row |

Refusals: `AlgebraBudgetError` (before allocation), `ValueError` (malformed claim, scope or bytes; non-canonical
bytes), `UnsupportedError` (more than 8 columns, 64 bits, 64 required values, 4096 terms, 8 dimensions),
`ClaimViolated` (a `DomainError` with `.evidence`), `DomainError` (not an instance of a law; not a combination),
`OverflowError`, `CarrierError`, `StaleStateError`.

## For the next capabilities

Emit: `tr_cert_snapshot` / `tr_cert_hypothesis` / `tr_cert_schema` / `tr_cert_law` over a `tr_scope` with a
`tr_cert_claim`; batch evidence with `tr_cert_verify` (CPU) or `tr_mgpu_cert_verify` + `tr_cert_adopt` (CUDA);
`tr_cert_derive` for ideal-membership steps (GEOM-001 reductions with cofactors); `tr_cert_poly_canonical` and the
GF(p) law for Boolean / parity identities (FIELD-002). Consume: `tr_cert_fresh` (exact reasons),
`tr_cert_authorize_exact` (with the target field) before any rewrite (META-001, PRE-001, UNC-001, COUNT-001),
`tr_cert_get_claim` / `tr_cert_get_record` to read them, `tr_cert_record_scope` / `tr_cert_record_compare` for an
address-free scope identity of other states (FND-011 uses it), `tr_cert_dim_describe` for join multiplicity.

## Limitations

The polynomial check is exact only within 128 bits (wider rows are refused, or checked mod p as an approximate
claim); derivations combine polynomial claims only (no inequalities or implications between other kinds); a
hypothesis records counts, not a confidence bound; the version digest and lifetime token are process-specific, so a
deserialized snapshot is never fresh in another process; references are checked against row-id foreign keys only
(a derived surrogate is refused, as in FND-004); the CUDA unique check needs the column's [min, max] domain within
the bytes budget (the CPU sorts).
