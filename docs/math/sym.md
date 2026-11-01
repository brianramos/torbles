# Closed-basis actions and certified data symmetries (SYM-001, SYM-002)

`torbles/math_sym.h` (every contract; the 512-bit arithmetic, the request evaluator and the symmetry checks are inline
and shared with CUDA), `math_sym.c`, `math_sym_gpu.cu`, `math_sym.py` (opt-in). SYM-001 answers questions about
transformed data from retained moment states without rescanning: the exact matrix M_T of a transformation T on a
polynomial basis closed under it. SYM-002 certifies that a declared group preserves the multiset of rows and then
answers invariant queries on orbit representatives.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_sym as ms, math_tensor as mt
scope = mbk.Scope.from_table(db["p"], ["x", "y", "c"], group_by=["k"])
geo = ms.geometry(scope, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})   # the declared coordinates
B = ms.basis(geo, degree=2)                       # or terms=["1", "x", "y", "x*y"], homogeneous=True, columns=[...]
R = ms.rotate("x", "y", cos=Fraction(3, 5), sin=Fraction(4, 5), center=(1, 0))   # degrees=30: certified intervals
A = ms.action(B, ms.compose(R, ms.translate(x="1.25")), geo)   # NotClosedError when B is not closed under the map
A.matrix(); A.numerators(); A.compose(other); A.equal(other); A.info; A.index; ms.identity_action(B)
st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)    # or ms.state(mt.moments(...), geo)
st.transformed(A, g); st.query([dict(action=0, group=g, functional=0, mean=True)], actions=[A],
                               functionals=[{"x^2": 1, "y^2": 1}], backend="cuda")
G = ms.group(geo, [ms.rotate("x", "y", degrees=90), ms.reflect(x=0), ms.values("c", {0: 1, 1: 0})])
cert = ms.certify(scope, geo, G, nulls="skip", backend="cuda")   # SymmetryViolated(.evidence, .witness, .kind)
cert.fresh(scope); cert.evidence; cert.info
orb = cert.orbits(scope); orb.summary()           # [(representative, orbit size, multiplicity)]
orb.query([ms.total({"x^2": 1, "y^2": 1}), ms.count({"x^2": 1, "y^2": 1}, "<=", 25)])   # NotInvariantError
ms.evaluate(scope, geo, queries)                  # the exhaustive evaluation over every selected row
```

## Declared geometry

A geometry lists the coordinates and nothing else is inferred. `numeric` coordinates are measured quantities: the
value is the scope's exact decimal value (logical / 10^scale, the scale read natively from the scope), the unit a
label. `categorical` coordinates are codes with no metric (a dictionary code, a foreign key, any category): their
order and adjacency mean nothing. A code column (a codec without an affine logical value) cannot be declared
numeric (`DomainError`). Natively: a transformation naming a scope column that is not declared, any metric operation
on a categorical coordinate (translation, scaling, reflection, rotation, affine), a rotation of two coordinates of
different units, a permutation across kinds or units, and a basis or bank term over a categorical coordinate are all
refused (`DomainError`). Abstract geometries without data (`ms.abstract(names, units=, scales=)`) serve bases and
actions alone (DYN-001).

## SYM-001: transformations, bases, actions

- Transformations are declared in value units with exact rationals (int, `Fraction`, `Decimal`, decimal strings;
  floats are refused): `translate`, `scale`, `reflect` (x -> 2a - x), `permute` ({target: source}), `affine`
  (A x + b over named coordinates), `rotate` (about a center; an exact rational point (cos, sin) of the unit circle,
  c^2 + s^2 = 1 verified exactly, or an angle in degrees: multiples of 90 exact, anything else certified enclosures of
  cos and sin by a Taylor series with its truncation bound in outward-rounded binary64 arithmetic), `compose(T, S)` =
  T o S, `identity`. Natively each becomes a map in logical units, x' = D A D^-1 x + D b with D = diag(10^scale),
  stored per row as (sum a_ij x_j + b_i) / q_i in int128 (`OverflowError` past it) or as interval rows.
- Bases: up to 4096 distinct monomials of total degree <= 8 over up to 32 coordinates, in order: every monomial of
  degree <= d (`homogeneous`: exactly d) over the numeric coordinates or a subset, graded in FND-004's order, or
  explicit terms. A basis is closed under T when every (T x)^a lies in its span.
- Actions: phi(T x) = M_T phi(x), so moments(T(rows)) = M_T moments(rows) for any weights. Row a is the exact
  multinomial expansion of prod_i ((sum_j a_ij x_j + b_i) / q_i)^{a_i}; a term outside the basis is `NotClosedError`
  with the basis term and the missing monomial. EXACT actions are int128 numerators over one common denominator D,
  canonical (gcd 1), so `equal` compares them exactly; a coordinate permutation gives an index permutation of the
  basis (`info["permutation"]`, `index`). INTERVAL actions (irrational rotations) hold enclosures of every entry,
  `approximate=True`, `certified=True`, `precision=53`; their closure is decided structurally (an entry enclosure that
  is not exactly [0, 0] is a monomial of the image), conservatively. `compose` is M_T M_S (exact products in 512 bits,
  reduced with FND-009 big integers, narrowed to int128 or `OverflowError`; an interval operand gives interval
  products: enclosures are carried through). The action of a composed map and the product of the actions are
  computed independently and agree exactly; M_T M_T^-1 is the identity.
- States: the retained moments of an FND-004 bank (its terms over declared numeric coordinates; JOINT, or SQL when
  every term of every group was valid on all its rows, otherwise `DomainError`: moments of different rows) or of a
  TENSOR-001 moment tensor (LISTWISE, every axis the same columns, at most once the constant; with it the basis of
  degree <= order, without it the homogeneous monomials of the order, closed under linear maps only). Per group: rows,
  contributing rows and the int128 moments. A state and its geometry come from one scope; an action of another
  geometry is refused on it.
- Requests (batched): (action or none, group, a basis term or a functional of int64 coefficients over the basis,
  mean). Exact answers are F . (N m_g) / D as 512-bit numerators over 512-bit denominators (`int` or `Fraction`);
  `mean` divides by the constant term's moment (rows or weight sum; empty: `None`; no constant term:
  `UnsupportedError`). Interval actions answer enclosures (lo, hi).

## SYM-002: certificates and orbits

- Generators: maps that send the integer domain into itself, integral in logical units and invertible (det A != 0,
  decided exactly): coordinate permutations, reflections about half-integer points, rotations by multiples of 90
  degrees (of coordinates of one scale), integral affine maps; and `values(column, {value: image})`, a permutation of
  finitely many values of one coordinate (numeric or categorical). Non-integral, singular and interval maps are
  refused (`DomainError`); at most 16 generators, 65,536 moved values each.
- The support: the sorted distinct tuples of every declared coordinate over the selected rows, with multiplicities and
  smallest rows; `nulls="skip"` excludes rows with a NULL coordinate, `"forbid"` refuses them (`DomainError` with
  `.row`). It keeps an FND-007 record of the scope (identity, revision, rows, exact selection, NULL domains, columns).
- Verification, exact and exhaustive: generator g preserves the multiset iff mult(g(v)) = mult(v) for every support
  tuple v (g injective). Per generator: rows and tuples checked, violating rows, missing tuples (image absent: a missing
  orbit member) and broken ones (image of another multiplicity), the witness = the smallest violating row with its
  kind. Every generator holding gives a certificate of FND-007 kind `snapshot` (`verified_by` cpu or the CUDA
  library); otherwise `SymmetryViolated` with the evidence and no certificate. `fresh(scope)` is FND-007's exact
  comparison with every reason; a stale certificate builds no orbits and answers no query (`StaleCertificateError`).
- Orbits: union-find over v ~ g(v) on the support: per orbit the representative (smallest tuple), size and
  multiplicity (equal for every member by the certificate). Queries are polynomials over the coordinates with int64
  coefficients (degree <= 8, <= 4096 terms): `total` (the sum over the rows) and `count` (rows with p OP c). Under the
  reduction a query must be proven invariant as a law: p o g = p as polynomials for an affine generator (the exact
  expansion of p(A x + b)), p(x_c = a) = p(x_c = sigma(a)) as polynomials for every value a a value permutation moves;
  then the answer is the sum over orbits of size x multiplicity x f(representative), one evaluation per orbit. A query
  not proven invariant is refused (`NotInvariantError`, the generator named), including invariant predicates of
  non-invariant polynomials (x^2 - y^2 = 0 under a quarter turn): the check is a sufficient condition. `evaluate`
  answers any query exhaustively over every selected row: the reference the reduction is checked against.

## CUDA

- `st.query(..., backend="cuda")` (`tr_mgpu_sym_query`): the state's moments, every action's matrix (and index) and
  the functionals are copied once per batch; one work item per request runs the shared `tr_sym_eval_request` (exact
  512-bit products and sums, or interval products and sums in the same order with the explicit-fma helpers of
  FND-009): the CPU results bit for bit.
- `ms.certify(..., backend="cuda")` (`tr_mgpu_sym_verify`): work items (generator, tuple slice) map each tuple
  (`tr_sym_image`, exact int128 with every coordinate checked into int64), find the image by binary search of the
  sorted support (exact comparisons, no hashing), compare the multiplicities, add one atomic per counter and take the
  smallest violating row by an atomic minimum; the host finishes the evidence (`tr_sym_evidence_finish`) and
  `tr_sym_adopt` checks it (counts, the witness's tuple and kind) before certifying. Equal to the CPU bit for bit.

## Budgets and refusals

| request | needs |
|---|---|
| basis | variables, degree, support = monomials, bytes |
| action | variables, degree, support, tensor = nb^2, bytes = the matrix and the expansion scratch (bounded per row from the coordinates its expansion reaches) |
| composition | support, tensor = nb^2, iterations = nb^3, bytes (the 512-bit products) |
| state | variables, support, states = groups x nb, bytes |
| requests | output = requests, bytes = results (CUDA: every device copy) |
| group | variables, support = moved values, bytes |
| support | variables, support = selected rows, bytes = tuples, sort index and scratch |
| verification | variables, output = generators, bytes (CUDA: the device copies) |
| orbits | support = distinct tuples, bytes |
| queries | variables, degree, support = terms, iterations = value-permutation substitutions, output = queries, bytes = the invariance expansions |

Refusals raise before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`), `OverflowError` (an int128
map, action or composed entry; a query value past int128 at a point), `UnsupportedError` (past 32 coordinates,
degree 8, 4096 monomials or terms, 16 generators; MEAN without a constant term; a value permutation in an action),
`DomainError` (undeclared or categorical coordinates, units, a basis not closed: `NotClosedError`, moments of
different rows, a state of another scope, a group that is not a symmetry: `SymmetryViolated`, a query not proven
invariant: `NotInvariantError`, non-integral or singular generators), `ValueError` (malformed input), `TypeError`
(a float parameter), `StaleCertificateError` / `StaleStateError`.

## For the next capabilities

DYN-001 builds on the public maps (`tr_sym_map_make`, `tr_sym_map_compose`, `tr_sym_map_identity`), bases
(`tr_sym_basis_full`, `tr_sym_basis_create`) and actions (`tr_sym_action_make`, `tr_sym_action_compose`,
`tr_sym_action_identity`, `tr_sym_action_equal`, `tr_sym_action_get_view`): operator powers by repeated
composition, closure checks of observables, and the batched requests on retained states.

## Limitations

Exact entries are int128 (a large translation at a high degree, or many compositions, are refused, not widened);
irrational rotations are binary64 enclosures only (no multiprecision intervals) and only rotations are irrational
(other transformations are exact rationals); interval closure is structural (an exact cancellation is not seen);
invariance under the reduction is a sufficient polynomial condition (invariant predicates of non-invariant
polynomials are refused); orbits are those of the generated group on the observed support, not on the whole domain;
NULL coordinates are skipped or forbidden (not a category of their own); the support sort, the orbit union-find and
the queries on representatives run on the CPU; the CPU loops are single-threaded.
