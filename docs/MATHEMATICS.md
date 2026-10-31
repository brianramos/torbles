# Native mathematical query functions

This opt-in module starts the broader [algebra inventory](../ALGEBRA_INVENTORY.md).
It implements checked rational polynomial functions, parameter calculus, and
immutable scalar moment snapshots. The remaining inventory capabilities retain
their individual implementation and validation status.

```python
from fractions import Fraction
import pandas as pd
import torbles
from torbles.mathematics import Context, MomentBasis

db = torbles.from_frames({"t": pd.DataFrame({
    "x": [-2, 1, 3], "y": [-1, 4, 2], "keep": [1, 0, 1],
})})
t = db["t"]
scope = db.query(t).where(t.keep == 1)

with MomentBasis(scope, ["x", "y"]) as basis:
    x, y = basis.variables["x"], basis.variables["y"]
    theta = basis.parameter("theta")
    loss = basis.reduce((y - theta*x)**2)

    assert loss.evaluate(theta=2) == 25
    assert loss.diff("theta").evaluate(theta=2) == 36
    assert loss.diff("theta").diff("theta").evaluate() == 26
    assert loss.integral("theta", 0, 1).evaluate() == Fraction(4, 3)

# Functions own their coefficient programs independently of the moment bank.
assert loss.evaluate(theta=2) == 25
loss.close()
```

The selected rows produce `loss(theta) = 13*theta**2 - 16*theta + 5`.
Materialization builds the complete degree-two data basis: count, each linear
moment, and every square/cross-product. Two data coordinates require six native
aggregate queries. Later function reduction, evaluation, differentiation,
integration and substitution execute in C without another fact scan. Python
constructs objects, marshals arguments, dispatches existing aggregate queries and
formats results; it does not perform polynomial simplification or calculus.

## Formal functions without a dataset

```python
ctx = Context()
a, b = ctx.symbols("a", "b")
f = (a + b)**3
assert f.diff("a").evaluate(a=2, b=1) == 27
assert f.integral("a", 0, 1).evaluate(b=0) == Fraction(1, 4)
assert f.substitute("b", 0) == a**3
```

Inputs are exact integers, `Fraction`, or finite `Decimal` constants. Ordinary
floating-point constants are refused. Results are integers when integral and
`Fraction` otherwise. Expressions provide `terms`, `degree`, `support` and
`term_count`; native coefficient inspection exposes the actual formal program.

## Scope and semantics

- A context has at most eight formal variables. A polynomial has total degree
  at most eight and at most 128 nonzero terms. Canonical rational numerators and
  positive denominators must fit signed int64. Native intermediate checks may
  conservatively refuse a calculation before a later cancellation; results
  never silently wrap or approximate.
- A moment bank initially accepts one to four non-null integer fact columns
  with affine codecs and stored codes no wider than 32 bits. Logical signed
  offsets are handled through existing typed native queries. Decimal, nullable,
  wide, grouped, range-join and moved-key delta moment states remain outside
  this slice. Formal coefficients can still be rational.
- The data-variable degree after expansion must fit the stored degree-two
  basis. Parameter powers may use the larger formal polynomial budget. A
  missing moment is an explicit error; terms are never silently discarded.
- Empty selected input defines the zero empirical linear functional:
  `basis.reduce(f)` is zero. This mathematical convention differs from SQL
  `SUM` over no rows, which returns NULL. The snapshot records this convention.
- Aggregate functions differentiate and integrate free parameters. Their data
  coordinates are fixed observations. No snapshot relationship is silently
  promoted into a law about continuous coordinate dependence.
- The source query is copied before materialization. Supported predicates and
  ordinary declared equijoins retain the existing engine's selection and bag
  semantics. Every basis term uses the same selected row domain.
- Callers must serialize materialization with source mutation. Before/after
  source signatures detect supported revision, address and domain changes;
  this is not an MVCC or transactional snapshot implementation.
- A finished snapshot is immutable. Later mutations do not refresh its values;
  the recorded source identity/revision describes the represented snapshot.
  Closing a bank prevents further reductions through it; previously produced
  functions remain independently owned and usable.
- Serialize explicit `close()` with operations on the same object. Concurrent
  closure and native execution are outside this initial ownership contract.
- `engine="gpu"` requests existing CUDA aggregate queries when building the
  snapshot. Actual devices are recorded. Coefficient algebra and calculus run
  in native CPU code. This slice adds no GPU calculus kernel or fused reducer.

`AlgebraBudgetError`, `OverflowError`, `MemoryError`, and input/scope errors are
reported separately. Native constructors leave their inputs unchanged on error.

## Typed operators and compatible state reuse

```python
from torbles.math_domains import MomentCache, modules

with MomentCache(capacity=16) as cache:
    with MomentBasis(scope, ["x", "y"], cache=cache) as first:
        assert first.metadata["native_query_calls"] == 6
    with MomentBasis(scope, ["x", "y"], cache=cache) as reused:
        assert reused.metadata["cache_hit"]
        assert reused.metadata["native_query_calls"] == 0
        x, y = reused.variables.values()
        theta = reused.parameter("theta")
        response = reused.reduce((y-theta*x)**2)
        assert response.module_plan["executed_via_registry"]

# Returned banks and functions own their state independently of the cache.
assert response.evaluate(theta=2) == 25
```

The compatible scalar cache compares full bounded descriptor bytes, including
the selected scope, ordered data coordinates, codecs, validity convention,
physical source identity, revision and membership fingerprint. Equivalent fresh
query objects over that same declared scope can reuse a state. Different
predicates, coordinate order, source lifetimes, revisions or codec identities
miss. Each Table gets a stable lifetime token, so allocator address reuse does
not identify a new source as an older one. Query text and source fingerprints
are conservative identities; the cache does not prove arbitrary predicate
equivalence.

The native cache has one to 128 entries, an explicit retained-byte budget and
LRU eviction. `cache.stats` reports hits, misses, puts, evictions, bytes and
capacity; `clear()` releases entries and resets counters. An entry too large for
the cache can be declined while the newly materialized result remains usable.
Returned banks are owned copies and survive input-bank release, eviction, clear
and cache closure. Calls on the mutable cache are serialized by its Python
wrapper; direct C callers must serialize access themselves.

Cache hits still compute and compare source/domain fingerprints, including
membership bitmaps and any supported sidecar rank metadata. Their complete
construction cost can therefore depend on source-domain metadata size despite
issuing zero aggregate queries. Materialization/source mutation must still be
externally serialized; this is not MVCC.

Arithmetic, powers, differentiation, integration, substitution, evaluation and
moment contraction now run through an executable native registry. `modules()`
lists the two actual built-ins: checked rational polynomials and scalar integer
moment contraction. `expression.domain` and `basis.row_domain` return owned
typed descriptors; `expression.module_plan` records the actual native dispatch,
CPU backend, operator and a relative shape-work estimate. That estimate is not
elapsed time, a calibrated latency model, or an automatic selection policy.

Domains distinguish physical bags, selected snapshots, encoded assignments,
formal parameters and declared geometry, with explicit carrier, coordinate
roles, validity, grouping, revision and source origin. The registry rejects
unsupported domain/carrier/operator combinations and mixing aggregate functions
with unreduced observed coordinates. Naming a Boolean, GF(2), assignment or
geometry descriptor does not implement its mathematical operators or certify a
data relationship. Direct native callers remain responsible for pairing the
correct bank with its declared source descriptor.

Each formal Context has a stable nominal identity independent of later symbol
names. Separate contexts are not implicitly aligned, even when their moment
values originated in the same cache. Native descriptors own copied keys and
origins; semantic keys are bounded to 64 KiB each. Cache retained-byte accounting
includes descriptor and basis keys; one extra bounded payload may be staged to
keep failed puts atomic.

A cache hit records `actual_devices=()` and no GPU executor, even if the caller
requested `engine="gpu"`; no GPU work occurred on that hit. Misses retain the
actual inherited aggregate-device evidence. All coefficient transforms remain
native CPU operations. Grouped/nullable banks, refinement and rollup maps,
fused reducers, and optimizer-driven build/reuse selection remain inventory work.

## Validation and measurement

`tests/test_mathematics.py` checks independent arithmetic and database answers,
calculus, source scopes, refusal paths, ownership and reuse. The domain, module
and state tests additionally check typed rejection, actual registry dispatch,
cache invalidation, ownership and metadata isolation. Standalone native harnesses
check rational edges, descriptor contracts, randomized LRU behavior, allocation
failures, and address/undefined/leak sanitizer behavior.
`bench/benchmark_mathematics.py` compares repeated parameterized loss queries
with native evaluation from a materialized basis, while reporting basis build,
symbolic construction, first calls and warm batches separately. Its baseline is
independent native aggregate queries; the experiment does not represent a
general speedup for every workload and is separate from the earlier SSB matrix.

`bench/benchmark_math_state.py` compares full construct/read/close requests with
and without the compatible scalar cache, including source fingerprint cost.
It records cold construction, seven paired warm batches, invalidation/refill,
actual aggregate-query counts and descriptor construction/comparison costs.
`benchmark_mathematics.py --source PATH` also permits an identical response
function harness against the preserved earlier runtime to expose dispatch costs.

The next dependencies are grouped/validity-aware state, shared native multi-output
accumulation, additional typed operators, and optimizer-driven construction/reuse
selection. They remain explicit inventory items with their own validation gates.

## Further capabilities

Each capability beyond the first slice has its own page, with its exact scope, budgets and refusals; the conventions
they share (native C, a CUDA kernel per capability, oracles, receipts) are in [MATH_CONVENTIONS.md](MATH_CONVENTIONS.md),
and their status is the inventory's.

- [Budgets](math/budgets.md) (FND-010): resource limits, exact size formulas, representation choice, output pre-pass.
- [Relationship verification](math/disc.md) (DISC-001): affine, sum and product hypotheses verified on the CPU or in CUDA.
- [Validation gates](math/gates.md) (FND-012): receipts, the inventory check, error codes, device identity.
- [Scopes and state banks](math/bank.md) (FND-004): selected rows, exact grouped moments, rollups, the state cache.
- [Handles and ABI checks](math/handles.md) (FND-013): owned handles, deferred close, struct layout checks.
- [Numeric carriers](math/carriers.md) (FND-009): big integers and rationals, GF(p), CRT, intervals, truncated series.
- [Module registry](math/registry.md) (FND-005): module descriptors, law and carrier checks, explanations.
- [Plan choice](math/planner.md) (FND-008): costed candidates, sampling that never proves emptiness, forced controls.
- [Jets](math/jet.md) (JET-001, JET-002): exact directional derivatives, gradients, Hessians and higher partials of circuits and aggregates.
- [Inverse answers](math/inv.md) (INV-001): exact affine solution sets, certified polynomial roots, finite-grid search.
- [Non-polynomial separation](math/calc.md) (CALC-003): phase and exponential families with certified coefficient enclosures.
- [Moment matrices and solves](math/lin.md) (LIN-001, LIN-002): grouped Gram systems, exact solves, inertia, covariance, certified eigensystems.
- [Quadratic minimizers](math/opt.md) (OPT-001): exact unique, nonunique, unbounded and box-constrained optima.
- [Hierarchies](math/multi.md) (MULTI-001): certified refinement maps, exact per-level rollups, contrasts.
- [Signed updates](math/delta.md) (DELTA-001): insert, delete, weight-change and move deltas on exact states.
- [Predicate cofactors](math/bool.md) (BOOL-001): cofactors, XOR and signed differences under SQL three-valued logic.
- [GF(2) primitives](math/field.md) (FIELD-001): packed elimination, nullspaces, syndromes, parity dependencies.
- [Histograms and quantiles](math/dist.md) (DIST-001, DIST-002): grouped scalar/joint histograms, exact CDF, quantiles, range integrals.
- [Loss curves](math/resp.md) (RESP-002): exact piecewise absolute, hinge and squared-hinge losses with slopes and minimizers.
- [Group quantifiers](math/quant.md) (QUANT-001, QUANT-002): some/all/none with SQL NULL semantics, relational division.
- [Boolean lattices](math/lattice.md) (LATTICE-001, LATTICE-002): 2^k occupancy histograms, zeta/Moebius, inclusion-exclusion.
- [Walsh spectra](math/spec.md) (SPEC-002): exact Walsh-Hadamard transforms, Parseval, parity correlations.
- [Predicate influence](math/influence.md) (BOOL-002): influence and interactions under uniform and empirical measures.
- [Information statistics](math/info.md) (DIST-003): entropy, mutual information, contingency statistics with certified bounds.
- [Pairwise summaries](math/pair.md) (PAIR-001): Gini difference, AUC, KS, Wasserstein and Cramer-von Mises from prefixes.
- [Generating polynomials](math/gen.md) (GEN-001): exact coefficients, moments, products and truncated series of supports.
- [Certificates](math/certificates.md) (FND-007): laws, schema facts, snapshot proofs and hypotheses, never confused.
- [Persistence](math/persist.md) (FND-011): versioned, checksummed records of mathematical states.
- [Convolutions and series](math/conv.md) (GEN-002): declared-independence products, per-key products, bounded walks.
- [Provenance](math/prov.md) (PROV-001): factorized derivations in the Boolean, counting and polynomial semirings.
- [Bilinear updates](math/bilinear.md) (DELTA-002): exact join and matrix-product deltas with all cross terms.
- [Group envelopes](math/env.md) (ENV-001): exact winning groups and breakpoints of affine/quadratic families.
- [Sparse interaction and contribution attribution](math/attr.md) (ATTR-001): declared cooperative games over provenance (PROV-001), occupancy histograms (LATTICE-002), predicates (BOOL-001) or given values; exact Shapley, Banzhaf, Moebius, Shapley / Banzhaf interaction and Shapley-Taylor indices from sparse Moebius supports, with the sum identities checked; budgeted exploration and minimum explanation / repair search returning exact results or certified bounds with the explored states; the convention is part of every result and an attribution has no automatic causal interpretation.
