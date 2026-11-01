# Conservation and stability across revisions (DISC-003)

`torbles/math_conserve.h` (every contract), `math_conserve.c`, `math_conserve_gpu.cu`, `math_conserve.py` (opt-in).
A tracker keeps the DISC-002 Gram of a plan current under DELTA-001 signed updates, re-verifies a set of identities
incrementally (only the delta's rows are evaluated), records per identity which revisions it was certified on and
where it first failed, and keeps four evidence classes apart. Declared transformations are checked for conserving a
quantity symbolically (a universal FND-007 law) or on data (a snapshot certificate or the violating rows).

```python
from torbles import math_bank as mbk, math_ident as mi, math_delta as md, math_conserve as mcons
s0 = mbk.Scope.from_arrays({"x": x, "y": y, "s": x + y})
ids = mi.discover(s0, ["x", "y", "s"], degree=1, certify=False).identities       # or mi.candidates(...)
tr = mcons.track(s0, ["x", "y", "s"], degree=1, identities=ids)    # step 0: Gram, exhaustive verification, certificates
tr = tr.update(md.Delta.between(s0, s1), scope=s1, backend="cuda") # bound revision: incremental, certified on s1
tr = tr.update(md.Delta.from_arrays({...}, weights=[1, -1]))       # unbound: identities HELD, never certified
tr.record(i)        # statuses per step, certified runs and streak, first failure (step, revision, witness row, values)
tr.steps(); tr.gram(s1) == mi.gram(s1, mi.plan(s1, ["x", "y", "s"], degree=1))   # the incremental Gram is exact
tr.classes(i, given=[schema_or_hypothesis_certs]); tr.authorize(i, s1)   # only a fresh snapshot authorizes
tr.locate(i, s1)    # the smallest violating physical row of a failure recorded as a delta row
law = mcons.conserve({(1, 0, 0): 1, (0, 1, 0): 1},                 # q = x + y under x' = x + t, y' = y - t
                     {"x": {(1, 0, 0): 1, (0, 0, 1): 1}, "y": {(0, 1, 0): 1, (0, 0, 1): -1}}, ["x", "y", "t"])
law.conserved, law.certificate, law.residual, law.counterexample
law.check(scope, {"x": "x", "y": "y", "t": "t"}, backend="cuda")  # on the rows: snapshot certificate or witnesses
```

## Revisions

- Step 0 verifies every tracked identity exhaustively and certifies the holding ones (FND-007 snapshots). An update
  evaluates only the delta's rows (weights +1 insert, -1 delete): per identity the checked, excluded (a NULL support
  column, policy SKIP), violating and undecided weights, inserted and deleted apart, and the smallest violating
  inserted and deleted rows; the evidence is updated by these sums. The Gram is updated exactly
  (G' = G + sum w m m^T over the complete delta rows), proven against 128 bits from the delta's extremes first.
- A DIFF (`Delta.between(current, new)`) given with the new scope gives a BOUND revision: the evidence is then the new
  scope's exhaustive evidence (complete at the build, a DIFF lists exactly the rows whose contribution changed), and a
  holding identity gets a snapshot certificate of the new scope (adopted from the incremental evidence). Any other
  delta gives an UNBOUND revision: holding identities are HELD (a record), never certified. `rebind(scope)`
  recomputes everything from a scope (the complete recomputation; also the benchmark's baseline).
- Statuses per revision: certified, verified (FND-007 refused the certificate), held, failed, undecided (the
  evaluation width could not be proven: W <= 8 words), untracked. A perturbation that breaks an identity records the
  step, the table revision, the number of violating rows and the witness (the delta row and its values).

## Evidence classes (never converted)

declared (schema guarantees: declared identities, FND-007 schema certificates passed in), snapshot (the tracker's own
certificate of one revision), stability (the persistence record: what held on the listed revisions, not a proof about
the next one), hypothesis (DISC-002 / FND-007 hypothesis certificates passed in). Only a fresh snapshot authorizes an
exact rewrite; the others are refused with their class (stability, statistical, declared, failed, none).

## Conservation laws

P = q(T(x)) - q(x) is expanded exactly. P = 0: T conserves q on every commutative ring (FND-007 RING_IDENTITY law, no
scope needed). P != 0: a counterexample assignment from the grid prod_v S_v with |S_v| = deg_v P + 1 (a nonzero P has a
non-root there), evaluated exactly. On data, P = 0 is verified on every selected row (snapshot certificate) or the
violating rows are listed (the perturbations).

## Exactness, CUDA, budgets

All arithmetic is exact (signed 128-bit Gram sums, W-word identity evaluation, int64 coefficients; past a proven
bound: TR_MATH_OVERFLOW or undecided rows, never wrapped). `tr_mgpu_cons_delta` evaluates a delta's rows (Gram
increments with two-word atomics, identity evidence with integer atomics and atomic minima for the witnesses) and the
tracker adopts its result after consistency checks; results equal the CPU bit for bit. Budgets (checked before
allocation): variables, support (basis monomials), states (moments + identities), output, bytes; at most 256 tracked
identities and the DISC-002 plan limits (8 variables, 63 basis monomials: TR_MATH_UNSUPPORTED past them).
Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR (an inapplicable delta, a delete of
rows that are not there), _STALE.
