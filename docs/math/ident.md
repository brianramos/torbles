# Moment-Gram polynomial identity discovery (DISC-002)

`torbles/math_ident.h` (every contract; the basis evaluation, the W-word candidate evaluation and the row tests are
inline and shared with CUDA), `math_ident.c`, `math_ident_gpu.cu`, `math_ident.py` (opt-in). Over a FND-004 scope and
k <= 8 integer columns, the Gram matrix of a bounded monomial basis gives every polynomial identity of that basis the
data satisfy (its nullspace), each then verified exhaustively on every row and certified by FND-007.

```python
from torbles import math_bank as mbk, math_ident as mi
scope = mbk.Scope.from_table(db["t"], ["x", "y", "z", "w"])
found = mi.discover(scope, ["x", "y", "z"], degree=2)               # plan, Gram, nullspace, identities, certificates
found.report      # rank, nullity, points S, forced = max(0, B - S), generators / consequences / known, labels_exact
for i in found.identities: i["text"], i["status"], i["underdetermined"], i["consequence"], i["certificate"]
found.discoveries()                                                # verified, determined, not known, not implied
p = mi.plan(scope, ["x", "y"], degree=3, weight="w", width="auto")   # basis, moments, "exact" or "modular"
g = mi.gram(scope, p, backend="cuda")                               # or source="bank" (FND-004) / "lin" (LIN-001)
k = g.kernel(backend="cuda")                                        # LIN-002 (exact) or CRT + reconstruction (modular)
ids = mi.identities(scope, g, k, known=[{(1, 0, 0): 1, (0, 2, 0): -1}])   # declared relations: restatements labelled
ids = mi.candidates(scope, ["x", "y"], [{(2, 0): 1, (0, 1): -1}], gram=g)  # given polynomials as candidates
ids.verify(backend="cuda"); ids.certify(backend="cuda")             # exhaustive evidence; FND-007 snapshots
mi.quadratic(g, ids, 0)            # c^T G c exactly (and the binary64 value a floating miner would compute)
mi.hypothesis(scope, g, ids, 0, tolerance=1e-12, size=1000)        # a numerical near zero: a HYPOTHESIS at most
mi.support(scope, ["x", "y"]); mi.need(scope, ["x", "y"], degree=2)
(g1 + g2).sign, (g1 - g2).sign                                      # "nonnegative", "signed" (a deletion)
```

## Domain

- The Gram rows are the selected rows where every variable (and the weight) is present. G = sum_r w_r m(x_r) m(x_r)^T
  over the basis m_1 < ... < m_B (every monomial of degree <= d, or an explicit list; increasing graded lexicographic
  order; B <= 63). Under nonnegative weights c^T G c = sum_r w_r (c . m(x_r))^2, so G c = 0 exactly when c . m
  vanishes on every Gram row of positive weight: the nullspace is the set of identities of the basis.
- Signed weights (a negative weight, or `g1 - g2`, e.g. a DELTA-001 deletion) make the Gram "signed": a zero
  quadratic form there can be cancellation, so it never certifies an identity (`sos: "signed"`); such a candidate is
  only a proposal until the exhaustive verification decides it.
- Verification and certificates are over the identity's own support columns: a row where one of them is NULL is
  excluded (FND-007 policy `skip`); every other selected row is evaluated.

## Exactness

- `exact` plans prove every moment fits 128 bits before reading (N W prod M_v^e <= 2^127 - 1 from the scope's
  extremes); the moments are exact signed 128-bit sums and the nullspace is LIN-002's (modular elimination, CRT within
  the Hadamard bound, exact verification G c = 0): rank and vectors exact (`exact_rank`).
- `modular` plans (unproven bounds, or `width="modular"`) keep every moment modulo the recorded LIN-002 primes; the
  kernel is the CRT and rational reconstruction of the primes agreeing on the best echelon key. Its rank is a lower
  bound of the exact rank and every vector only a proposal: the exhaustive verification decides each one; a vector
  whose reconstruction fails is `unresolved`.
- Identities are primitive integer polynomials (int64 coefficients, else `unrepresentable`). Each is evaluated on
  every row in W 64-bit words modulo 2^(64W), W the smallest width whose proven bound sum |c| prod M^e <= 2^(64W-1) - 1
  holds (W <= 8, else undecided: rows counted as overflow): a zero residue is a zero value. One violating row rejects
  an identity (status `rejected`, the smallest violating row as the witness; no certificate).
- Labels are exact. `points` S: the distinct points of the identity's support over its domain (64-bit fingerprints in a
  radix sort, equal fingerprints compared value by value); an identity with more terms than points (T > S) exists on
  any S points and is `underdetermined`, never a discovery (`forced = max(0, T - S)`; the basis level reports
  max(0, B - S)). `known`: in the span of the declared relations' multiples within the basis (a restated
  definition); `consequence`: in the span of those and of the earlier generators' multiples (fraction-free 128-bit
  elimination; `labels_exact` false when it overflowed, every label then false).
- `quadratic` gives c^T G c exactly (big integers; residues for a modular Gram) and the binary64 value with its scale
  and relative size, marked approximate. `hypothesis` turns a numerical near zero into an FND-007 hypothesis on a
  sample (never an exact authorization), and refuses one that is not near zero.

## Certificates

A verified identity becomes an FND-007 snapshot certificate of its POLYNOMIAL claim (policy skip, carrier I128): the
evidence is adopted when W = 2, else FND-007 re-checks it in its own 128-bit carrier, which may refuse
(`certificate_code` -3: the identity is verified but not certifiable there). Certificates go stale with the table's
revision, selection or NULL domains (`fresh`, `authorize_exact`); every state of a changed scope raises
`StaleStateError`.

## CUDA

`tr_mgpu_ident_gram` (work items over moment chunks, residue slots and row slices; local exact 128-bit sums or residue
sums, one two-word atomic per moment) and `tr_mgpu_ident_verify` (candidate chunks x row slices, local counters, one
atomic per counter, the witness by an atomic minimum) give the CPU's states and evidence bit for bit; the exact
nullspace can run its LIN-002 elimination on the device (`kernel(backend="cuda")`). Results name `cuda` or
`cuda-emulated` and the device.

## Budgets and refusals

Checked before allocation (`Budget`): variables k, degree d, support B, tensor B^2, states moments x primes,
iterations (primes), bytes (plan, Gram, kernel, candidates, device copies, and the distinct counts: 33 bytes
per row, 49 with a weight, plus 512 KiB). B > 63, k > 8, a degree past 16 or more than 32 primes is
`UnsupportedError`; `width="exact"` on an unproven plan is `OverflowError`; a bank or LIN state of another scope or
shape, or combining Grams of different plans, is `DomainError`.

## Limitations

- A relation violated on a few rows makes its products with any polynomial vanishing on those rows exact identities of
  one degree more (a near miss q = x + y at one row gives (q - x - y) (x - x_r) = 0): they hold on every row and are
  reported as discoveries; nothing factors them.
- An identity implied only through multiples past the basis degree is labelled a generator (GEOM-001 reduces such
  consequences exactly).
- The basis is bounded by LIN-002's 63 unknowns; larger bases are refused, not approximated. Columns must be integer
  (or fixed-point logical) values of the scope.
