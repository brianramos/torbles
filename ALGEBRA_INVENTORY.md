# Torbles algebraic capability and implementation inventory

Generated from [docs/algebra_inventory.json](docs/algebra_inventory.json). Edit that canonical file and run `python bench/render_algebra_inventory.py`; use `--check` to validate coverage, dependencies, evidence and this rendered view.

Inventory date: 2026-10-30. Source baseline: `aaacb0f`. Working branch: `alpha`.

The torbles repository (branch alpha). Moved here from the prototype (prototype/mixed-scaling, 7affc3e) on 2026-10-31; earlier receipts under results/algebra_foundation and results/algebra_domains were measured there.

All 32 mathematical families discussed with the user, including shared native foundations. This is an implementation backlog with explicit bounded capabilities, not a claim that all families are implemented.

Coverage: **32 families, 81 actionable capabilities, including 13 shared foundations**.

| Status | Capabilities |
|---|---:|
| existing | 0 |
| partial | 0 |
| planned | 0 |
| in_progress | 63 |
| validated | 18 |

A planned target or acceptance case is not an implementation or a passed test. Prior native-stage benchmarks cover their recorded workloads only. A capability becomes validated only within its stated bounds and with linked test and benchmark receipts.

## Status meanings

- **existing**: A scoped capability already exists in the preserved prototype, supported by source/test references; not a fresh validation receipt.
- **partial**: Existing building blocks or narrower behavior exist; the full listed capability remains incomplete.
- **planned**: No implementation claim. Listed files are proposed targets and acceptance/benchmark cases are requirements.
- **in_progress**: Implementation is being worked on; no validated completion claim.
- **validated**: Implemented within stated bounds and accompanied by linked successful test and benchmark evidence. From 2026-10-31 a validated capability also has its own CUDA kernel with a receipt from an actual device run (validation.cuda); the six capabilities validated before then carry validation.cuda_standard instead.

## Mathematical contracts

- Every operation declares physical-row/snapshot, encoded-assignment, formal-parameter or geometric domain.
- NULL validity, bag multiplicity and group presence are separate state components.
- Snapshot-certified relations are distinct from universal laws, enforced schema guarantees and statistical hypotheses.
- Ordinary parameter differentiation, Boolean finite differences and finite update deltas use different laws.
- Data coordinate dependencies must be declared before differentiating through snapshot relations.
- All exactness, arithmetic widths, approximation assumptions, search/output budgets and refusal paths are explicit.
- Python may construct/marshal native programs and display evidence; computational mathematics is implemented natively for claimed native capabilities.
- Actual CUDA work, native CPU finalization, CPU fallback and emulation are reported separately.
- Existing benchmark evidence does not validate or predict unimplemented capability performance.

## Benchmark contracts

- Every benchmark case here is a planned experiment unless a capability links a measured receipt.
- Record source/library hashes, seed, hardware, actual scan/kernel/device path, row/support/state/output sizes and workspace.
- Separate data/basis construction, first execution, warm reuse and validation; report one-shot cases as well as amortized reuse.
- Use matched semantic work and independent correctness oracles; preserve all raw observations and failed/over-budget cases.
- Use actual CUDA only for CUDA claims; distinguish native CPU finalization, CPU row fallback and emulation.
- Leave unrelated user processes and their affinity/priority untouched. Keep dev and preserved data sources read-only.

## First implementation slice

Status: **validated**.

Checked native rational polynomial functions and immutable scalar degree-two moment contraction. Bounds: <=8 formal variables, total degree<=8, <=128 terms, int64 rational representation; one to four nonnullable INTEGER data variables with affine codecs/stored codes<=32 bits. Scope materialization uses existing native reductions; coefficient algebra and calculus run in native CPU code.

Capabilities: [FND-002](#fnd-002), [FND-003](#fnd-003), [CALC-001](#calc-001), [CALC-002](#calc-002), [RESP-001](#resp-001)

API guide: [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Backend semantics:

- engine=gpu requests existing actual CUDA aggregate queries for moment materialization where supported, with actual devices recorded.
- Function reduction, rational polynomial arithmetic, evaluation, differentiation and integration run in native CPU code.
- No new GPU calculus kernel, generic heterogeneous IR or fused multi-output reducer is claimed.

Explicit exclusions:

- No general typed heterogeneous stage DAG.
- No fused single-pass moment reducer claim.
- No nullable, decimal-column, wide, grouped, range-join or moved-key delta moment states in this slice.
- No general transcendental, probability, graph, topology, inverse or optimization implementation claim.

First-slice validation:

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)
- Source integrity: [results/algebra_domains/dev_integrity.json](results/algebra_domains/dev_integrity.json)

## Current continuation slice

Typed domains, executable modules and compatible scalar state reuse

Status: **validated bounded implementation; broader capabilities remain partial**.

Owned native domain contracts and actual polynomial/moment operator dispatch, plus bounded exact compatible scalar-cache reuse. Existing calculus functions are revalidated through typed dispatch. See [current benchmark and validation evidence](results/algebra_domains/README.md).

Capabilities: [FND-001](#fnd-001), [FND-004](#fnd-004), [FND-005](#fnd-005), [FND-013](#fnd-013)

Remaining work:

- Grouped/nullable states and explicit rollup/alignment remain unimplemented.
- No fused reducer, new CUDA coefficient kernel or automatic build/reuse planner.

## Delivery waves

### W0: Inventory and scalar exact foundation

- Inventory covers all 32 requested families without implied implementation.
- Checked rational polynomial handles and immutable scalar moment snapshots.
- Native scalar polynomial reduction, evaluation, differentiation and definite integration with explicit budgets.

### W1: Reusable function families

- Typed state identity, grouping/validity contracts and basis reuse.
- Polynomial response functions, jets, bounded inverse/optimization and explicit function-domain contracts.
- Separate construction, cold and reuse timings; no fused-pass claim until a fused kernel exists.

### W2: Finite domains and Boolean transforms

- Histograms, prefix states, quantifiers and finite-loss response curves.
- Boolean/GF(2)/subset transforms with domain budgets.
- Hierarchical state refinement and exact distributed merges.

### W3: Proofs, payloads and constraints

- Certified relation reduction, preimages and bounded model counting.
- Typed provenance, uncertainty and update-state algebras.
- Exact inverse constraints and explanation searches with explicit search limits.

### W4: Operators and tensor families

- Moment matrices, contractions, spectral bases and finite-feature kernels.
- Symmetry/dynamics actions over closed bases.
- Native fused CPU and separately validated CUDA candidates where they pay.

### W5: Structured graph, inference and geometry

- Graph layouts and bounded fixed-point/path operators.
- Factorized probability and provenance composition.
- Declared geometric complexes, boundary maps and bounded topology.

## Family coverage

| Family | Scope | Capability IDs |
|---|---|---|
| CALC: Continuous calculus | Derivatives, mixed partials, gradients, Hessians, integrals and Taylor expansions. | [CALC-001](#calc-001), [CALC-002](#calc-002), [CALC-003](#calc-003) |
| JET: Derivative algebras | Dual numbers and truncated multivariate series for simultaneous values and derivatives. | [JET-001](#jet-001), [JET-002](#jet-002) |
| RESP: Whole response functions | Return complete parameterized aggregate or piecewise loss functions. | [RESP-001](#resp-001), [RESP-002](#resp-002) |
| INV: Inverse queries | Solve for parameter values or changes producing a requested answer. | [INV-001](#inv-001), [INV-002](#inv-002) |
| OPT: Optimization | Optimize query-defined objectives under explicitly declared domains and constraints. | [OPT-001](#opt-001), [OPT-002](#opt-002) |
| GEN: Generating functions | Distribution, count and path generating polynomials or formal series. | [GEN-001](#gen-001), [GEN-002](#gen-002) |
| DIST: Distribution algebra | Histograms, tails, quantiles, entropy and information measures. | [DIST-001](#dist-001), [DIST-002](#dist-002), [DIST-003](#dist-003) |
| PAIR: Pairwise statistics | Distance, ranking and pairwise summaries without unconditional pair enumeration. | [PAIR-001](#pair-001), [PAIR-002](#pair-002) |
| LIN: Linear algebra | Moment matrices, regression, covariance, projections and eigensystems. | [LIN-001](#lin-001), [LIN-002](#lin-002), [LIN-003](#lin-003) |
| TENSOR: Tensor algebra | Sparse/factored moment tensors and contraction planning. | [TENSOR-001](#tensor-001), [TENSOR-002](#tensor-002) |
| KERNEL: Kernel calculations | Finite-feature embeddings and certified approximate kernel computations. | [KERNEL-001](#kernel-001), [KERNEL-002](#kernel-002) |
| SPEC: Spectral queries | Fourier, Walsh, parity correlation and frequency-response representations. | [SPEC-001](#spec-001), [SPEC-002](#spec-002) |
| MULTI: Multiscale analysis | Hierarchy states, contrasts, adaptive refinement and localized changes. | [MULTI-001](#multi-001), [MULTI-002](#multi-002) |
| SYM: Symmetry queries | Actions of translations, rotations, permutations and certified symmetries. | [SYM-001](#sym-001), [SYM-002](#sym-002) |
| DYN: Repeated dynamics | Finite closed observables and repeated operator application. | [DYN-001](#dyn-001), [DYN-002](#dyn-002) |
| BOOL: Boolean calculus | Predicate cofactors, signed/GF(2) finite differences and influence. | [BOOL-001](#bool-001), [BOOL-002](#bool-002) |
| LATTICE: Subset-lattice algebra | Zeta/Mobius transforms, exclusive regions and inclusion-exclusion. | [LATTICE-001](#lattice-001), [LATTICE-002](#lattice-002) |
| QUANT: Quantified queries | Existential/universal predicates and category coverage. | [QUANT-001](#quant-001), [QUANT-002](#quant-002) |
| COUNT: Constraint counting | Observed-domain and symbolic-domain model counts and witnesses. | [COUNT-001](#count-001), [COUNT-002](#count-002) |
| FIELD: Finite-field algebra | GF(2) dependencies, parity, ranks, codes and syndromes. | [FIELD-001](#field-001), [FIELD-002](#field-002) |
| GEOM: Algebraic geometry | Certified ideal reduction, elimination and generator selection. | [GEOM-001](#geom-001), [GEOM-002](#geom-002) |
| DISC: Relationship discovery | Polynomial identities, conserved quantities and hidden coordinates. | [DISC-001](#disc-001), [DISC-002](#disc-002), [DISC-003](#disc-003) |
| PRE: Preimage queries | Push conditions on derived values back to generators. | [PRE-001](#pre-001), [PRE-002](#pre-002) |
| ENV: Parametric envelopes | Winning groups, exact breakpoints and parameter regions. | [ENV-001](#env-001), [ENV-002](#env-002) |
| GRAPH: Graph and fixed-point queries | Incidence operators, frontiers, paths and bounded recursive computations. | [GRAPH-001](#graph-001), [GRAPH-002](#graph-002) |
| PROV: Provenance | Factorized explanation and multiplicity expressions. | [PROV-001](#prov-001), [PROV-002](#prov-002) |
| ATTR: Attribution and repair | Contributions, interaction indices and bounded minimal changes. | [ATTR-001](#attr-001), [ATTR-002](#attr-002) |
| PROB: Probability and inference | Small-support distributions, partition functions and factorized inference. | [PROB-001](#prob-001), [PROB-002](#prob-002) |
| DELTA: Update calculus | Signed changes, cross terms and maintained algebraic states. | [DELTA-001](#delta-001), [DELTA-002](#delta-002) |
| UNC: Certified uncertainty | Bounds, abstract interpretation and explicit approximation budgets. | [UNC-001](#unc-001), [UNC-002](#unc-002) |
| META: Queries about queries | Equivalence, implication, containment and shared query-family computation. | [META-001](#meta-001), [META-002](#meta-002) |
| TOPO: Finite geometry and topology | Declared occupancy complexes, boundaries, cycles and homology. | [TOPO-001](#topo-001), [TOPO-002](#topo-002) |

## Shared native foundations

<a id="fnd-001"></a>

### FND-001: Typed domain and algebra contracts

Status: **validated**. Delivery wave: **W0**.

Owned native descriptors distinguish physical bags, selected snapshots, full encoded assignments, formal parameter contexts and declared geometry; record carrier, observed/free coordinate layout, validity, grouping, revision, universe size and exact bounded semantic identity. Native composition checks apply to the registered rational polynomial and scalar integer moment operators.

Exactness: Semantic contracts precede optimization; snapshot-certified facts have no universal or causal authority.

Dependencies: None.

Native CPU: Implemented owned native descriptors and exact bounded identity/context checks, used by the executable polynomial and scalar moment registry.

GPU: Descriptor checks execute on native CPU. CUDA-built snapshot banks preserve their row-domain metadata; no CUDA descriptor kernel is claimed.

Bounds:

- At most eight formal slots and four observed data coordinates; each exact semantic key is 1..65536 bytes.
- Owned origin depth is at most one, formal to a physical or selected row domain. Unsupported contracts fail explicitly.

Implemented files:

- [torbles/math_domain.c](torbles/math_domain.c)
- [torbles/math_domain.h](torbles/math_domain.h)
- [tests/test_math_domains.py](tests/test_math_domains.py)

Acceptance tests:

- Reject combining states with different selected-row or validity domains unless an explicit exact alignment exists.
- Distinguish free parameters from observed data coordinates and reject undeclared derivative dependencies.

Benchmark cases:

- Measure descriptor construction/comparison cost for 1, 16 and 128 reusable states.
- Compare exact domain-match reuse with incompatible predicate/partition rejection.

Additional limits:

- Descriptors label caller-declared identity; they are not unforgeable data, causal or geometric proofs.
- Only polynomial and scalar moment operators execute in this slice; other declared domain/carrier labels do not imply kernels.
- At most eight slots/four data coordinates; semantic keys <=64KiB and at most one owned row origin.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="fnd-002"></a>

### FND-002: Checked rational polynomial kernel

Status: **validated**. Delivery wave: **W0**.

Immutable native polynomials: at most eight variables, total degree eight, 128 terms; canonical rational coefficients with checked signed-int64 numerator and positive int64 denominator. Arithmetic and calculus stay in C.

Exactness: Exact within checked rational-int64 representation and explicit polynomial budgets; arbitrary precision remains planned.

Dependencies: None.

Native CPU: Implemented and validated in native C within the declared first-slice rational, variable, degree and term budgets.

GPU: Coefficient algebra, calculus and function evaluation are native CPU work. A new GPU calculus/function kernel is not implemented.

Bounds:

- variables <= 8
- total degree <= 8
- nonzero terms <= 128
- normalized numerator and denominator fit the declared int64 domain

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [docs/MATHEMATICS.md](docs/MATHEMATICS.md): Current first-slice API and bounded domain/backend semantics.

Implemented files:

- [torbles/mathematics.c](torbles/mathematics.c)
- [torbles/mathematics.h](torbles/mathematics.h)
- [torbles/mathematics.py](torbles/mathematics.py)
- [tests/test_mathematics.py](tests/test_mathematics.py)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Acceptance tests:

- Match Fraction-based arithmetic for signed coefficients, cancellation, powers and normalization.
- Detect coefficient, degree, term-count and denominator overflow without publishing partial results; free handles safely.

Benchmark cases:

- Separate native handle construction from 10,000 repeated evaluations of sparse 1/8-variable polynomials.
- Compare expression expansion and retained function evaluation across term-count budgets.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="fnd-003"></a>

### FND-003: Immutable scalar moment snapshot

Status: **validated**. Delivery wave: **W0**.

Materialize count and degree-at-most-two moments for one to four nonnullable INTEGER fact variables with affine codecs and stored codes<=32 bits from a fixed scalar query scope. Construction orchestrates existing native reductions; native code owns the immutable state and its polynomial contraction. Empty selection defines the zero empirical linear functional rather than SQL SUM's NULL convention.

Exactness: Exact within first-slice rational-int64 state limits; immutable values represent a recorded snapshot, not a live maintained view.

Dependencies: [FND-002](#fnd-002)

Native CPU: Validated: existing native aggregate queries materialize the complete scalar degree-two basis; C owns immutable state and exact contraction. Construction remains several native queries rather than a fused traversal.

GPU: Moment materialization may request existing actual CUDA aggregate queries and records actual devices. Bank contraction and all coefficient/calculus work remain native CPU; no new GPU math kernel.

Bounds:

- data variables: one to four at the public API
- data monomial degree <= 2
- nonnullable INTEGER fact columns with affine codecs and stored codes <=32 bits
- scalar fixed scope only; nullable/decimal/wide/grouped/range-join/moved-key delta states refused
- Multiple native reductions are expected; not a fused one-pass implementation.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Query reductions, logical codecs and histogram bridge are existing lowerings.
- [torbles/alg.c](torbles/alg.c): Bit-plane SUM and grouped native reductions exist.
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md): Current first-slice API and bounded domain/backend semantics.

Implemented files:

- [torbles/mathematics.c](torbles/mathematics.c)
- [torbles/mathematics.h](torbles/mathematics.h)
- [torbles/mathematics.py](torbles/mathematics.py)
- [tests/test_mathematics.py](tests/test_mathematics.py)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Acceptance tests:

- Match independent row arithmetic for count, first moments, squares and cross moments under selective predicates, signed affine codecs and duplicate rows.
- Reject unsupported NULL/wide/group/scope cases and out-of-range moments explicitly; snapshot evaluation remains independent of later table mutation.
- Empty selected input reduces any supported function to zero; independently owned result functions remain valid after closing the bank.

Benchmark cases:

- Report construction time and actual native reduction count separately from 10,000 coefficient-only function evaluations.
- Compare repeated row-query loss evaluation with building the same moments once and reusing them.

Additional limits:

- Materialization must be serialized with source mutation; before/after identity checks are not an MVCC/transactional snapshot.
- Completed state is immutable and does not auto-refresh when source data changes.
- Formal rational coefficients are supported; decimal data columns remain outside the initial moment-bank scope.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="fnd-004"></a>

### FND-004: Reusable state bank, partitions and invalidation

Status: **validated**. Delivery wave: **W1**.

Cache immutable state banks keyed by exact domain, revision, codec, validity, basis and grouping grain; support compatible refinement/rollup maps and deliberate refresh.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-003](#fnd-003)

Native CPU: Native C: data scopes (math_scope.c: selected rows, exact logical values, validity, dense group ids, decoded once from the engine table or taken from arrays) and grouped exact moment banks (math_bank.c: bag multiplicity, per-monomial valid counts and 128-bit sums, SQL or listwise validity, chunked build and merge, exact rollup, an exact-descriptor LRU cache). The first slice's scalar moment cache (math_state.c) is unchanged.

GPU: CUDA kernels math_bank_gpu.cu (tr_mgpu_bank_build, tr_mgpu_bank_rollup): the bank and its rollup on the device with exactly the CPU states, device copies budgeted before allocation.

Bounds:

- Scope: at most 32 value columns and 2^32-1 dense groups (the product of the key radices); states = groups x monomials and the decoded bytes are budgeted (math_budgets.h) before allocation.
- Bank: at most 1024 monomials of total degree <= 4; every product and sum is proven to fit 127 bits from the column extremes and row counts before accumulation, else refused (TR_MATH_OVERFLOW).
- Cache: capacity 1..128 with a retained-byte budget; keys are exact descriptors (table identity and revision, selection bits, columns, keys and lookups, basis, validity mode).

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships_system.c](torbles/relationships_system.c): Existing cache preserves live filters/output addresses but is not a cross-query moment-state bank.
- [tests/test_physical_cache.py](tests/test_physical_cache.py): Existing physical-decision cache tests.
- [torbles/math_state.c](torbles/math_state.c): Bounded exact-descriptor scalar moment cache with owned clones, LRU and retained-byte accounting.
- [tests/test_math_state.py](tests/test_math_state.py): Compatible reuse, source invalidation, lifetime, budget and metadata controls.

Implemented files:

- [torbles/math_scope.h](torbles/math_scope.h)
- [torbles/math_scope.c](torbles/math_scope.c)
- [torbles/math_bank.h](torbles/math_bank.h)
- [torbles/math_bank.c](torbles/math_bank.c)
- [torbles/math_bank_gpu.cu](torbles/math_bank_gpu.cu)
- [torbles/math_bank.py](torbles/math_bank.py)
- [tests/test_math_bank.py](tests/test_math_bank.py)
- [bench/benchmark_math_bank.py](bench/benchmark_math_bank.py)

Acceptance tests:

- A different predicate, NULL policy, revision or group grain cannot hit the same live cache entry.
- Refined group states merge exactly to coarse states with explicit empty/all-NULL group presence.

Benchmark cases:

- Benchmark compatible warm reuse versus refresh and incompatible-domain rejection.
- Measure state bytes and cache eviction at bounded capacities.

Additional limits:

- CPU accumulation is single-threaded (only decoding uses the engine's threads); dense groups take memory for absent groups too.
- The overflow proof is conservative (max magnitude per factor times rows), so some requests whose sums would fit are refused.
- Dimension lookups are copied at build; the native freshness check covers the fact table, and the Python layer tracks dimension revisions.
- Not supported: lookups through derived surrogate keys, self-extending columns as group keys, date/timestamp/string columns as values.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

Partial capability only: receipts validate the bounded implemented scalar/typed subset described in existing evidence and limitations; broader acceptance cases remain unimplemented. Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/math/FND-004/tests_test_math_bank.json](results/math/FND-004/tests_test_math_bank.json)
- tests: [results/math/FND-004/sanitized_test_math_bank.json](results/math/FND-004/sanitized_test_math_bank.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)
- benchmarks: [results/math/FND-004/benchmark_cpu_math_bank_vm.json](results/math/FND-004/benchmark_cpu_math_bank_vm.json)
- benchmarks: [results/math/FND-004/benchmark_device_math_bank_WIN_OHVNN0FUC09.json](results/math/FND-004/benchmark_device_math_bank_WIN_OHVNN0FUC09.json)

<a id="fnd-005"></a>

### FND-005: Native module and operator registry

Status: **in_progress**. Delivery wave: **W1**.

Modules declare carriers, laws, valid transforms, sufficient state, proof dependencies, native kernels and complete construction/reuse costs.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-002](#fnd-002)

Native CPU: Native C (math_registry.c): a constant table of 37 module descriptors covering the first-slice built-ins, the executors of the registry's own request kind and every implemented family (key, capability, state kind, carriers, laws relied on and guaranteed, proof dependencies, devices, CPU and CUDA entry points, cost coefficients where calibrated); carriers include GF(2), GF(p), big integers and rationals, real, rational-endpoint and complex intervals, truncated series and Kleene logic, with explicit lift / residue / null_encode conversions; a composition check that names the missing law or conversion; explanation records (module, representation, entry point, device, refusal).

GPU: CUDA kernels math_registry_gpu.cu: the direct-scan and histogram executors of the registry's request kind and the lattice/Walsh transforms dispatched through the registry, reporting device or emulation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/math_modules.c](torbles/math_modules.c): Executable dispatch for two actual built-ins: Qpolynomial transforms and scalar moment contraction.
- [tests/test_math_modules.py](tests/test_math_modules.py): Native-domain/carrier checks and actual dispatch, including bypass-route spies.

Implementation targets (proposed until the capability is validated):

- `torbles/math_registry.h`
- `torbles/math_registry.c`
- `torbles/math_registry_gpu.cu`
- `torbles/math_registry.py`
- `tests/test_math_registry.py`
- `bench/benchmark_math_registry.py`
- `torbles/math_modules.c`
- `torbles/math_modules.h`
- `tests/test_math_modules.py`

Acceptance tests:

- Reject an operator composition when required laws or carrier conversions are absent.
- Expose actual chosen module, representation, native devices and refusal reason in explanations.

Benchmark cases:

- Measure registry dispatch and bounded candidate search separately from execution.
- Compare direct scans, sparse polynomial, histogram and factored-state candidates on identical requests.

Additional limits:

- Descriptors are written by hand: a test checks that every named entry point exists in the libraries, every kernel file and tr_mgpu_* entry is described and every capability id exists, not that descriptions are complete.
- Only the registry's own request kind and the LATTICE-002/SPEC-002 transforms execute through the registry; other families run through their own APIs, and their cost coefficients are not calibrated.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

Partial capability only: receipts validate the bounded implemented scalar/typed subset described in existing evidence and limitations; broader acceptance cases remain unimplemented. Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/math/FND-005/tests_test_math_registry.json](results/math/FND-005/tests_test_math_registry.json)
- tests: [results/math/FND-005/sanitized_test_math_registry.json](results/math/FND-005/sanitized_test_math_registry.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)
- benchmarks: [results/math/FND-005/benchmark_cpu_math_registry_vm.json](results/math/FND-005/benchmark_cpu_math_registry_vm.json)
- benchmarks: [results/math/FND-005/benchmark_device_math_registry_WIN_OHVNN0FUC09.json](results/math/FND-005/benchmark_device_math_registry_WIN_OHVNN0FUC09.json)

<a id="fnd-006"></a>

### FND-006: Fused multi-output basis reduction

Status: **in_progress**. Delivery wave: **W4**.

Accumulate multiple basis states in one traversal while sharing predicates, decoded planes, group masks and overlapping products; preserve a competing repeated-reduction plan.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-003](#fnd-003), [FND-005](#fnd-005)

Native CPU: Native C (math_reduce.c): a request over a FND-004 scope (BOOL-001 conditions, a shared Kleene predicate DAG, up to 16 groupings: the scope's own, another scope of exactly the same rows, or either mapped onto coarse groups; up to 1024 outputs, each a FND-004 moment basis in SQL or listwise validity or a DIST-001 histogram, with its own predicate and grouping) is compiled (equal conditions and nodes merged, distinct monomials collected as shared products, every term's 127-bit bound proven from the scope's extremes before anything runs) and executed by a fused plan, one traversal of the selection words in which each condition and node is evaluated once per word and each product at most once per row for every output term that uses it, every output keeping its own rows, validity counts and group presence; or by the preserved repeated plan, the independent reductions one by one (BOOL-001 masks, a derived scope per output, tr_bank_build / tr_dist_hist_build). Both give identical states, equal to independent FND-004 / DIST-001 reductions and DuckDB in the tests. Each plan's need (tr_reduce_need) is checked before anything is allocated and bounds every later check of the run (the repeated plan's from each derived scope's shape, with FND-004's and DIST-001's own need formulas for the device builds), except the occupied cells of sparse histograms, checked once counted. Candidates (fused, repeated) x (CPU, CUDA) are costed with FND-008's components (tr_reg_candidate, fusion credited as FND-008 does), budget-checked and chosen, and a force (fused | repeated [@cpu | @cuda], TORBLES_MATH_REDUCE_FORCE) runs exactly that plan or refuses. Runs are atomic: an overflow proof, a budget (a sparse state known after counting, one output of the repeated plan), a device or adoption failure exposes no output.

GPU: CUDA kernel math_reduce_gpu.cu (tr_mgpu_reduce_fused): the fused pass on the device, one work item per contiguous slice of selection words producing every output's contribution of its rows with the same per-word helpers as the CPU loop; bank outputs accumulate locally while the group stays the same and flush with integer atomics (128-bit sums with the two-word atomic), histogram cells add one atomic per row; device copies (only the columns read, each group-id array once) and per-item scratch budgeted before allocation; no barriers or shared memory; the arenas are adopted after consistency checks, equal bit for bit to the CPU (scrambled emulation). tr_mgpu_reduce_fused_device keeps the states on the device for META-002's finalization kernel. The repeated plan on CUDA runs FND-004's and DIST-001's own CUDA builds per derived scope.

Bounds:

- Checked before allocation (math_budgets.h): compile checks circuit = conditions + nodes and output = outputs before allocating, then the plan's bytes; a run checks its plan's need (tr_reduce_need: variables, degree, support, circuit, output, states, bytes; on CUDA the device copies and per-item scratch, or FND-004's / DIST-001's device needs per derived scope) before it allocates anything; the need bounds every later check except the occupied cells of sparse histograms, checked once counted.
- Representation limits (UnsupportedError): 1024 outputs, 16 groupings, 64 conditions, 65536 nodes, 65536 bank terms, 4096 distinct products; FND-004 / DIST-001 limits per output.
- Refusals are atomic (no output of a refused run exists): AlgebraBudgetError (resource, need, limit), OverflowError (an unprovable 127-bit term or weight bound, before anything runs), ValueError, DomainError, StaleStateError.

Implementation targets (proposed until the capability is validated):

- `torbles/math_reduce.h`
- `torbles/math_reduce.c`
- `torbles/math_reduce_gpu.cu`
- `torbles/math_reduce.py`
- `tests/test_math_fused_reduce.py`
- `bench/benchmark_math_reduce.py`

Acceptance tests:

- Every output and validity count matches independent native reductions for duplicated, empty and selectively grouped data.
- CPU and actual CUDA variants prove execution devices and handle overflow/partial failures atomically.

Benchmark cases:

- Benchmark 1, 3, 6 and 15 basis outputs across selectivity/group-domain regimes.
- Record actual scan/kernel count, decoded bytes, workspace, construction and first/reuse times.

Additional limits:

- Fused histogram outputs count in a dense scratch over the base scope's domain: a domain too wide is over budget for the fused plan (the repeated plan on the CPU sorts; DIST-001's device build has the same bound).
- The overflow proof uses the base scope's extremes for every output: a filtered output whose own rows would prove a tighter bound is still refused.
- The repeated plan's needs are bounds from the derived scopes' shapes (the base scope's rows, int64 columns, a sparse histogram's cells bounded by the selected rows): conservative for selective outputs.
- The repeated CUDA plan evaluates predicates and builds the derived scopes on the host and is orchestrated from Python through native calls; sparse fused outputs are compacted on the host.
- Plan costs are hand-set relative estimates (cost basis estimate), not calibrated times; the CPU paths are single-threaded.
- Measured on the CPU and the emulated kernels only here; the device receipt needs an actual GPU run.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/FND-006/tests_test_math_fused_reduce.json](results/math/FND-006/tests_test_math_fused_reduce.json)
- tests: [results/math/FND-006/sanitized_test_math_fused_reduce.json](results/math/FND-006/sanitized_test_math_fused_reduce.json)
- benchmarks: [results/math/FND-006/benchmark_cpu_math_reduce_vm.json](results/math/FND-006/benchmark_cpu_math_reduce_vm.json)

<a id="fnd-007"></a>

### FND-007: Proof certificates and evidence scope

Status: **in_progress**. Delivery wave: **W3**.

Unify universal laws, enforced schema facts, exhaustive snapshot certificates and statistical hypotheses. Certificates include exact domain/revision and arithmetic carrier.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001)

Native CPU: Native C (math_certificates.c): one certificate type for universal laws, enforced schema facts, exhaustive snapshot certificates (table identity and revision, rows, exact selection, NULL policy, join multiplicity, carrier) and statistical hypotheses, never confused; typed claims (DISC-001 rules, FIELD-001 parity relations, MULTI-001 refinement maps, QUANT-002 coverage, polynomial relations); freshness checks with the exact staleness reason; exact authorization only for laws, schema facts and fresh complete snapshots; derivations P = sum q_i G_i checked in Z[x].

GPU: CUDA kernel math_certificates_gpu.cu: exhaustive re-verification of one or many snapshot certificates over all selected rows (violations and the smallest witness with atomics), adopted after exact checks, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships.c](torbles/relationships.c): Existing exhaustive affine/sum/product verification and snapshot scope checks.
- [tests/test_relationship_cache.py](tests/test_relationship_cache.py): Existing stale proof/compiled-plan reuse checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_certificates.h`
- `torbles/math_certificates.c`
- `torbles/math_certificates_gpu.cu`
- `torbles/math_certificates.py`
- `tests/test_math_certificates.py`
- `bench/benchmark_math_certificates.py`

Acceptance tests:

- A sampled or approximate relationship cannot authorize an exact rewrite.
- Mutation, NULL-domain changes and join multiplicity changes invalidate applicable snapshot certificates.

Benchmark cases:

- Measure certificate checking separately from candidate discovery.
- Compare certificate reuse, invalidation and full re-verification over growing row counts.

Additional limits:

- Polynomial checks are exact within 128 bits (wider rows refused, or checked modulo p and recorded as approximate); derivations combine polynomial claims only; hypotheses record sample counts, not a confidence bound.
- Source tokens are process-specific: a persisted snapshot is never fresh in another process (it stays readable).

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/FND-007/tests_test_math_certificates.json](results/math/FND-007/tests_test_math_certificates.json)
- tests: [results/math/FND-007/sanitized_test_math_certificates.json](results/math/FND-007/sanitized_test_math_certificates.json)
- benchmarks: [results/math/FND-007/benchmark_cpu_math_certificates_vm.json](results/math/FND-007/benchmark_cpu_math_certificates_vm.json)

<a id="fnd-008"></a>

### FND-008: Complete-plan choice and cost evidence

Status: **in_progress**. Delivery wave: **W1**.

Cost chaining, algebraic combination, basis construction/reuse, fusion and materialization independently. Preserve explicit scan and CPU/GPU finalization accounting.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-005](#fnd-005)

Native CPU: Native C (math_planner.c): costed candidate plans with separate scan, construction, reuse, fusion, materialization and CPU/GPU finalization components, budget refusals, the cheapest admissible choice, full evidence of every candidate, sampled selectivity that is never a proof of emptiness, and forced controls that run the named representation or are refused.

GPU: CUDA kernel math_planner_gpu.cu: sampled and exact counts of selected rows over bitmap words for selectivity estimates, equal to the CPU counts.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships_system.c](torbles/relationships_system.c): Existing bounded chain/combine/hybrid physical chooser.
- [torbles/relationships_stage.c](torbles/relationships_stage.c): Existing bounded stage alternatives and native finalization.
- [wsl/ab_table.py](wsl/ab_table.py): Existing matched A/B with each side's own round-to-round noise (the prior mixed-scaling comparison: Recorded in the prototype it comes from (prototype/mixed-scaling, results/mixed_scaling); not copied into this repository.)

Implementation targets (proposed until the capability is validated):

- `torbles/math_planner.h`
- `torbles/math_planner.c`
- `torbles/math_planner_gpu.cu`
- `torbles/math_planner.py`
- `tests/test_math_planner.py`
- `bench/benchmark_math_planner.py`
- `torbles/math_registry.c`
- `torbles/math_registry_gpu.cu`

Acceptance tests:

- A zero sampled match never becomes an empty-result proof.
- Selected/rejected candidates retain cost evidence; forced controls execute their declared valid representation.

Benchmark cases:

- Use held-out query shapes and volume scales; report misses as well as gains.
- Retain raw batch distributions, setup, state bytes and actual backend paths.

Additional limits:

- Cached states are reused on the CPU only, not kept in device memory.
- The planner covers the registry's request kind and the DISC-001 verifier; later families register their candidates.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/FND-008/tests_test_math_planner.json](results/math/FND-008/tests_test_math_planner.json)
- tests: [results/math/FND-008/sanitized_test_math_planner.json](results/math/FND-008/sanitized_test_math_planner.json)
- benchmarks: [results/math/FND-008/benchmark_cpu_math_planner_vm.json](results/math/FND-008/benchmark_cpu_math_planner_vm.json)

<a id="fnd-009"></a>

### FND-009: Numeric carriers and exactness conversions

Status: **validated**. Delivery wave: **W1**.

Extend checked rationals to explicit wide integers, arbitrary precision, real/complex numerical values, finite fields, intervals and truncated series without silently mixing laws.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-002](#fnd-002), [FND-005](#fnd-005)

Native CPU: Native C (math_carriers.c): arbitrary-precision integers and canonical rationals (limb budgets, floor/truncating division, gcd, shifts, powers, integer square roots), GF(p) for verified primes p < 2^62 with Barrett reduction, CRT and rational reconstruction, outward-rounded real and complex intervals, truncated rational power series (product, reciprocal, composition, exp, log), and an explicit conversion/combination matrix.

GPU: CUDA kernels math_carriers_gpu.cu (tr_mgpu_carriers_gfp_vec/dot/matvec/reduce_i64, enclose_q64, sum_i64, dot_wide): batched GF(p) arithmetic, exact 128-bit sums and dot products, enclosures of int64 rationals; equal bit for bit to the CPU functions.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_carriers.h](torbles/math_carriers.h)
- [torbles/math_carriers.c](torbles/math_carriers.c)
- [torbles/math_carriers_gpu.cu](torbles/math_carriers_gpu.cu)
- [torbles/math_carriers.py](torbles/math_carriers.py)
- [tests/test_math_carriers.py](tests/test_math_carriers.py)
- [bench/benchmark_math_carriers.py](bench/benchmark_math_carriers.py)

Acceptance tests:

- Round-trip lossless conversions preserve values; narrowing and incompatible-carrier operations fail explicitly.
- Numerical transforms disclose precision and certified approximation state instead of claiming exact rational answers.

Benchmark cases:

- Benchmark coefficient bit width and carrier-conversion cost independently of row count.
- Compare native fixed-width, wide and arbitrary-precision paths on matched expressions.

Additional limits:

- Schoolbook multiplication and quadratic decimal conversion: slow past a few thousand bits.
- Interval endpoints are doubles; no transcendental interval functions; complex division refused when the divisor's squared modulus rounds to zero.
- Series have rational coefficients only (no GF(p) series); composition is cubic in the order.
- Batched checked evaluations may write part of their output array before refusing; the Python layer discards it.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/FND-009/tests_test_math_carriers.json](results/math/FND-009/tests_test_math_carriers.json)
- tests: [results/math/FND-009/sanitized_test_math_carriers.json](results/math/FND-009/sanitized_test_math_carriers.json)
- benchmarks: [results/math/FND-009/benchmark_cpu_math_carriers_vm.json](results/math/FND-009/benchmark_cpu_math_carriers_vm.json)
- benchmarks: [results/math/FND-009/benchmark_device_math_carriers_WIN_OHVNN0FUC09.json](results/math/FND-009/benchmark_device_math_carriers_WIN_OHVNN0FUC09.json)

<a id="fnd-010"></a>

### FND-010: Search, representation and output budgets

Status: **validated**. Delivery wave: **W0**.

Every new module bounds variable count, degree, support, tensor size, circuit size, iteration count and output cardinality; fallbacks remain explicit and costed.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001)

Native CPU: Native C: exact saturating size formulas, the ordered budget check and the costed expanded/factored choice (inline in math_budgets.h, exported by math_budgets.c), and the exact distinct-cell pre-pass that refuses before any output.

GPU: CUDA kernel math_budgets_gpu.cu (tr_mgpu_budgets_count_cells): the distinct-cell pre-pass on the device, with the device copies of its inputs budgeted before allocation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships_dag.c](torbles/relationships_dag.c): Existing DAG/instruction bounds.
- [torbles/relationships_stage.c](torbles/relationships_stage.c): Existing degree, residual-scan and group-domain bounds.

Implemented files:

- [torbles/math_budgets.h](torbles/math_budgets.h)
- [torbles/math_budgets.c](torbles/math_budgets.c)
- [torbles/math_budgets_gpu.cu](torbles/math_budgets_gpu.cu)
- [torbles/math_budgets.py](torbles/math_budgets.py)
- [tests/test_math_budgets.py](tests/test_math_budgets.py)
- [bench/benchmark_math_budgets.py](bench/benchmark_math_budgets.py)

Acceptance tests:

- Reject over-budget requests deterministically before partial result exposure.
- Boundary cases at each limit preserve exact valid existing execution paths.

Benchmark cases:

- Measure expansion versus factored fallback at polynomial, histogram and Boolean domain boundaries.
- Record resource/refusal thresholds rather than extrapolating missing measurements.

Additional limits:

- The first-slice polynomial kernel keeps its fixed limits (8 variables, degree 8, 128 terms); the budget module states them (Budget.polynomial_kernel) and refuses at the same boundaries but does not replace them.
- Costs given to the representation choice are relative work units supplied by the calling capability, not measured times.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/FND-010/tests_test_math_budgets.json](results/math/FND-010/tests_test_math_budgets.json)
- tests: [results/math/FND-010/sanitized_test_math_budgets.json](results/math/FND-010/sanitized_test_math_budgets.json)
- benchmarks: [results/math/FND-010/benchmark_cpu_math_budgets_vm.json](results/math/FND-010/benchmark_cpu_math_budgets_vm.json)
- benchmarks: [results/math/FND-010/benchmark_device_math_budgets_WIN_OHVNN0FUC09.json](results/math/FND-010/benchmark_device_math_budgets_WIN_OHVNN0FUC09.json)

<a id="fnd-011"></a>

### FND-011: State serialization and interoperability

Status: **in_progress**. Delivery wave: **W3**.

Versioned persistence for typed mathematical states, including basis, domain/proof identity, endianness, exactness, checksums and owned metadata; no live addresses serialized.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: Native C (math_persist.c): versioned little-endian records of FND-004 banks (with their scope descriptor), DIST-001 histograms, LIN-001 moment systems, FND-009 big integers, big rationals and GF(p) elements, and FND-007 certificates, each with type, carrier, basis, source identity and revision, selection digest, metadata and a checksum, no live addresses; loads refuse truncated, corrupted, unknown-version and wrong-type records with distinct reasons; reloaded states keep their recorded source and are never fresh views of a changed one.

GPU: CUDA kernel math_persist_gpu.cu: the order-independent position-weighted checksum of 64-bit words on the device, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/database.py](torbles/database.py): Existing format-1 database persistence, not mathematical-state persistence.
- [tests/check_persist.py](tests/check_persist.py): Existing source datastore round-trip/integrity tests.

Implementation targets (proposed until the capability is validated):

- `torbles/math_persist.h`
- `torbles/math_persist.c`
- `torbles/math_persist_gpu.cu`
- `torbles/math_persist.py`
- `tests/test_math_persist.py`
- `bench/benchmark_math_persist.py`

Acceptance tests:

- Round-trip each supported carrier/state and reject truncated, corrupted or unknown-version files.
- Reloaded immutable states cannot be misrepresented as refreshed views of a changed source.

Benchmark cases:

- Report serialization/load/validation time and bytes separately from mathematical evaluation.
- Compare persisted versus rebuilt state under fixed data and changing query families.

Additional limits:

- Little-endian records only (a foreign byte order is refused, not converted); histograms and moments rehydrate only into their recorded scope.
- A reloaded moment state declares the accumulation width it used (128 or 192 bits).

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/FND-011/tests_test_math_persist.json](results/math/FND-011/tests_test_math_persist.json)
- tests: [results/math/FND-011/sanitized_test_math_persist.json](results/math/FND-011/sanitized_test_math_persist.json)
- benchmarks: [results/math/FND-011/benchmark_cpu_math_persist_vm.json](results/math/FND-011/benchmark_cpu_math_persist_vm.json)

<a id="fnd-012"></a>

### FND-012: Differential validation and benchmark gates

Status: **in_progress**. Delivery wave: **W0**.

Per-capability oracles, algebraic property checks, scope/NULL/bag/refusal tests and measured build/cold/reuse benchmarks. Existing suite remains a regression gate.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: None.

Native CPU: The gates run natively built code against independent oracles (Python integers and Fractions, sympy, networkx, DuckDB); bench/validate_math.py runs each capability's tests, sanitizers, nvcc and benchmark and writes its receipts; the renderer refuses unbacked claims.

GPU: CUDA core kernel math_core_gpu.cu (exact 128-bit accumulation, min, max; device identity); device receipts come only from an actual device (--device refuses the emulation), and emulated runs are labeled.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [tests/run_all.sh](tests/run_all.sh): Existing quick/full suite entry point.
- [tests/test_relationship_stages.py](tests/test_relationship_stages.py): Existing stage correctness/property tests.
- [tests/check_algebra.py](tests/check_algebra.py): Existing switch-on/off and DuckDB differential checks of algebraic paths (the prior mixed-scaling validation: Recorded in the prototype it comes from (prototype/mixed-scaling, results/mixed_scaling); not copied into this repository.)

Implementation targets (proposed until the capability is validated):

- `tests/test_math_gates.py`
- `tests/test_mathematics.py`
- `tests/mathematics_native_edges.c`
- `tests/validate_mathematics_native.py`
- `tests/sanitize_math.sh`
- `torbles/math_core_gpu.cu`
- `torbles/math_gpu.py`
- `torbles/math_errors.py`
- `bench/validate_math.py`
- `bench/render_algebra_inventory.py`
- `bench/math_bench.py`
- `bench/benchmark_math_gates.py`
- `bench/benchmark_mathematics.py`

Acceptance tests:

- All supported exact results match independent arithmetic or a stated trusted oracle; negative cases fail safely.
- No new capability can be marked validated without linked test evidence and measured benchmark evidence.

Benchmark cases:

- Run every new module's declared benchmark cases with raw observations and source/library hashes.
- Keep existing mixed workload as regression controls without claiming new module coverage from it.

Additional limits:

- The gate verifies that receipts exist, pass and match the measured sources; whether an oracle is independent of the implementation is reviewed per capability, not decided by the gate.

Validation receipts:

Measured runtime: `00f63ab97ae8685403b2b44d7d37d366626ebf88`.

- tests: [results/math/FND-012/tests_test_math_gates.json](results/math/FND-012/tests_test_math_gates.json)
- tests: [results/math/FND-012/sanitized_test_math_gates.json](results/math/FND-012/sanitized_test_math_gates.json)
- benchmarks: [results/math/FND-012/benchmark_cpu_math_gates_vm.json](results/math/FND-012/benchmark_cpu_math_gates_vm.json)
- benchmarks: [results/math/FND-012/benchmark_device_math_gates_WIN_OHVNN0FUC09.json](results/math/FND-012/benchmark_device_math_gates_WIN_OHVNN0FUC09.json)

<a id="fnd-013"></a>

### FND-013: State ownership, concurrency and native ABI

Status: **validated**. Delivery wave: **W1**.

Opaque owned handles, immutable sharing, explicit release, deterministic closed-handle errors and serialized mutable caches; ABI version/size checks cover all ctypes boundaries.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005)

Native CPU: Native C: a caller-owned handle table (slot + generation ids, C11 atomic reference counts, deferred free on close during a live operation, per-payload mutex for mutable caches, live-allocation counts) and an ABI registry (TR_MATH_ABI / TR_MGPU_ABI entries in linker sections) that Python checks against every ctypes Structure of the torbles.math* modules.

GPU: CUDA kernel math_handles_gpu.cu: one immutable sparse integer polynomial shared by all threads, evaluated exactly (128-bit, overflow flagged per point) at many points; equal to the CPU path and to Python integers.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/math_domain.h](torbles/math_domain.h): Fixed versioned descriptor ABI and owned key/origin contracts.
- [torbles/math_domains.py](torbles/math_domains.py): Owned Python wrappers and serialized mutable cache operations.
- [tests/test_math_domains.py](tests/test_math_domains.py): Descriptor ownership, ABI/layout and closed-handle controls.

Implemented files:

- [torbles/math_handles.h](torbles/math_handles.h)
- [torbles/math_handles.c](torbles/math_handles.c)
- [torbles/math_handles_gpu.cu](torbles/math_handles_gpu.cu)
- [torbles/math_handles.py](torbles/math_handles.py)
- [tests/test_math_handles.py](tests/test_math_handles.py)
- [tests/math_handles_native_threads.c](tests/math_handles_native_threads.c)
- [bench/benchmark_math_handles.py](bench/benchmark_math_handles.py)

Acceptance tests:

- Repeated create/release and shared immutable evaluations do not leak or reuse freed buffers.
- Concurrent reads are safe under documented ownership; mutation/close cannot race a live native operation.

Benchmark cases:

- Measure handle/state lifetime memory under repeated workloads.
- Compare concurrent immutable evaluation and serialized cache updates without changing user process priorities.

Additional limits:

- The first-slice Python wrappers (Domain, MomentCache, expressions) are not yet table-owned; their close remains externally serialized.
- The ABI check compares size, alignment and version, not field order; engine ctypes structs (algq.py, relationships.py) are outside it.
- Freeing a table must not race other calls on it natively; the Python wrapper waits for operations in flight.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

Partial capability only: receipts validate the bounded implemented scalar/typed subset described in existing evidence and limitations; broader acceptance cases remain unimplemented. Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/math/FND-013/tests_test_math_handles.json](results/math/FND-013/tests_test_math_handles.json)
- tests: [results/math/FND-013/sanitized_test_math_handles.json](results/math/FND-013/sanitized_test_math_handles.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)
- benchmarks: [results/math/FND-013/benchmark_cpu_math_handles_vm.json](results/math/FND-013/benchmark_cpu_math_handles_vm.json)
- benchmarks: [results/math/FND-013/benchmark_device_math_handles_WIN_OHVNN0FUC09.json](results/math/FND-013/benchmark_device_math_handles_WIN_OHVNN0FUC09.json)

## CALC: Continuous calculus

Derivatives, mixed partials, gradients, Hessians, integrals and Taylor expansions.

<a id="calc-001"></a>

### CALC-001: Polynomial parameter derivatives

Status: **validated**. Delivery wave: **W0**.

Differentiate immutable formal parameter polynomials, including repeated and mixed partials, within declared rational/degree budgets. Gradient/Hessian components can be constructed from these derivative programs; a general native vector/tensor calculus API remains a later capability.

Exactness: Exact bounded rational polynomial differentiation; no claim about arbitrary smooth functions.

Dependencies: [FND-002](#fnd-002), [FND-012](#fnd-012)

Native CPU: Implemented and validated in native C within the declared first-slice rational, variable, degree and term budgets.

GPU: Coefficient algebra, calculus and function evaluation are native CPU work. A new GPU calculus/function kernel is not implemented.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [docs/MATHEMATICS.md](docs/MATHEMATICS.md): Current first-slice API and bounded domain/backend semantics.

Implemented files:

- [torbles/mathematics.c](torbles/mathematics.c)
- [torbles/mathematics.py](torbles/mathematics.py)
- [tests/test_mathematics.py](tests/test_mathematics.py)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Acceptance tests:

- Compare first through fourth and mixed derivatives with independent Fraction coefficient arithmetic.
- Reject unknown variables and keep observed-coordinate dependencies explicit; constants differentiate to exact zero.

Benchmark cases:

- Vary one to eight parameters, 8/32/128 terms and derivative order 1/2/4; report construction and coefficient-transform cost separately.
- Compare 1/100/10,000 evaluations or bound pairs after one retained-state build; include zero polynomials and degree/coefficient refusal boundaries.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="calc-002"></a>

### CALC-002: Polynomial parameter integration

Status: **validated**. Delivery wave: **W0**.

Indefinite and definite integration in a declared formal variable, with exact rational coefficients and exact bound substitution.

Exactness: Exact rational polynomial integration; continuous parameter integration is distinct from summing an empirical row measure.

Dependencies: [FND-002](#fnd-002), [FND-012](#fnd-012)

Native CPU: Implemented and validated in native C within the declared first-slice rational, variable, degree and term budgets.

GPU: Coefficient algebra, calculus and function evaluation are native CPU work. A new GPU calculus/function kernel is not implemented.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [docs/MATHEMATICS.md](docs/MATHEMATICS.md): Current first-slice API and bounded domain/backend semantics.

Implemented files:

- [torbles/mathematics.c](torbles/mathematics.c)
- [torbles/mathematics.py](torbles/mathematics.py)
- [tests/test_mathematics.py](tests/test_mathematics.py)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Acceptance tests:

- Check derivative(antiderivative) equals the source polynomial where budget permits.
- Check zero/reversed/fractional bounds and integration-degree/coefficient overflow without partial handles.

Benchmark cases:

- Vary one to eight parameters, 8/32/128 terms and derivative order 1/2/4; report construction and coefficient-transform cost separately.
- Compare 1/100/10,000 evaluations or bound pairs after one retained-state build; include zero polynomials and degree/coefficient refusal boundaries.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="calc-003"></a>

### CALC-003: Non-polynomial finite-separation calculus

Status: **validated**. Delivery wave: **W1**.

Closed families such as fixed-frequency phase shifts and separable exponentials return compact parameter functions and their derivatives/integrals.

Exactness: Separation identities are algebraically exact; transcendental coefficient evaluation requires a numerical precision contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: Native C (math_calc.c): retained data coefficients of the fixed-frequency sine phase family (2^k coefficients for k phase parameters) and of the separable exponential family per group, with exact derivatives and integrals in the parameters as coefficient transforms, and rigorous enclosures of every transcendental coefficient and evaluated value; another frequency, family or scope is a new construction and a derivative in the frequency is refused.

GPU: CUDA kernel math_calc_gpu.cu: the coefficient sums over rows, accumulated as exact scaled integers with 128-bit atomics, equal bit for bit to the CPU enclosures.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_calc.h](torbles/math_calc.h)
- [torbles/math_calc.c](torbles/math_calc.c)
- [torbles/math_calc_gpu.cu](torbles/math_calc_gpu.cu)
- [torbles/math_calc.py](torbles/math_calc.py)
- [tests/test_math_calc.py](tests/test_math_calc.py)
- [bench/benchmark_math_calc.py](bench/benchmark_math_calc.py)

Acceptance tests:

- Fixed-frequency sine phase sweeps match direct numerical evaluation at declared precision.
- Changing frequency or a family outside the retained basis triggers new construction or an explicit refusal.

Benchmark cases:

- Vary one to eight parameters, 8/32/128 terms and derivative order 1/2/4; report construction and coefficient-transform cost separately.
- Compare 1/100/10,000 evaluations or bound pairs after one retained-state build; include zero polynomials and degree/coefficient refusal boundaries.

Additional limits:

- Sine and cosine enclosures are absolute (about 2^-113 per row times 
- w
- ), so tiny values have large relative width.
- After a definite integral the coefficients are double intervals without exact bounds.
- Only the phase and separable exponential families are retained; evaluation points are doubles with 
- phi
-  < 2^62; at most 16 phase factors.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/CALC-003/tests_test_math_calc.json](results/math/CALC-003/tests_test_math_calc.json)
- tests: [results/math/CALC-003/sanitized_test_math_calc.json](results/math/CALC-003/sanitized_test_math_calc.json)
- benchmarks: [results/math/CALC-003/benchmark_cpu_math_calc_vm.json](results/math/CALC-003/benchmark_cpu_math_calc_vm.json)
- benchmarks: [results/math/CALC-003/benchmark_device_math_calc_WIN_OHVNN0FUC09.json](results/math/CALC-003/benchmark_device_math_calc_WIN_OHVNN0FUC09.json)

## JET: Derivative algebras

Dual numbers and truncated multivariate series for simultaneous values and derivatives.

<a id="jet-001"></a>

### JET-001: Dual-number directional derivatives

Status: **validated**. Delivery wave: **W1**.

Native pair payloads carry a value and first directional derivative with typed sum/product/chain laws.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009)

Native CPU: Native C (math_jet.c): immutable budgeted polynomial circuits (constants, parameters, data columns, add, sub, mul, integer powers, rational scaling) and plans (order, 0..64 seed directions, expansion center); Taylor (Hasse) jets with exact int64 numerators over plan-fixed denominators and checked 128-bit intermediates; aggregates of jets over FND-004 scopes with SQL NULL semantics, exact merges, and carrier conversions of every component or a refusal.

GPU: CUDA kernel math_jet_gpu.cu: the aggregate jet over the selected rows of each group (one work item per row slice, the same inline evaluation code, exact 128-bit sums via tr_atomic_add_wide), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_jet.h](torbles/math_jet.h)
- [torbles/math_jet.c](torbles/math_jet.c)
- [torbles/math_jet_gpu.cu](torbles/math_jet_gpu.cu)
- [torbles/math_jet.py](torbles/math_jet.py)
- [tests/test_math_jet.py](tests/test_math_jet.py)
- [bench/benchmark_math_jet.py](bench/benchmark_math_jet.py)

Acceptance tests:

- Match symbolic derivatives of bounded polynomial circuits for several seeds and directions.
- Do not discard derivative components when merging aggregate states or converting carriers.

Benchmark cases:

- Vary one to eight seed directions and jet orders 1/2/3 on shared polynomial circuits; measure state bytes and arithmetic operations.
- Compare a joint value/derivative pass with independently transformed circuits; include sparse gradients and a dense worst-supported interaction case.

Additional limits:

- Evaluation uses int64 numerators over fixed int64 denominators; many decimal or rational factors refuse early (no big-rational or GF(p) evaluation engine, only exact conversions of results).
- Circuits have one output and the expansion center is fixed per plan.
- Only the aggregate runs on CUDA; plan compilation, readouts and conversions are CPU.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/JET-001/tests_test_math_jet.json](results/math/JET-001/tests_test_math_jet.json)
- tests: [results/math/JET-001/sanitized_test_math_jet.json](results/math/JET-001/sanitized_test_math_jet.json)
- benchmarks: [results/math/JET-001/benchmark_cpu_math_jet_vm.json](results/math/JET-001/benchmark_cpu_math_jet_vm.json)
- benchmarks: [results/math/JET-001/benchmark_device_math_jet_WIN_OHVNN0FUC09.json](results/math/JET-001/benchmark_device_math_jet_WIN_OHVNN0FUC09.json)

<a id="jet-002"></a>

### JET-002: Higher-order and multivariate jets

Status: **validated**. Delivery wave: **W1**.

Truncated series payloads produce Hessians, Hessian-vector products and mixed partials with explicit factorial or Hasse-basis conventions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [JET-001](#jet-001)

Native CPU: Native C (math_jet.c): immutable budgeted polynomial circuits (constants, parameters, data columns, add, sub, mul, integer powers, rational scaling) and plans (order, 0..64 seed directions, expansion center); Taylor (Hasse) jets with exact int64 numerators over plan-fixed denominators and checked 128-bit intermediates; aggregates of jets over FND-004 scopes with SQL NULL semantics, exact merges, and carrier conversions of every component or a refusal.

GPU: CUDA kernel math_jet_gpu.cu: the aggregate jet over the selected rows of each group (one work item per row slice, the same inline evaluation code, exact 128-bit sums via tr_atomic_add_wide), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_jet.h](torbles/math_jet.h)
- [torbles/math_jet.c](torbles/math_jet.c)
- [torbles/math_jet_gpu.cu](torbles/math_jet_gpu.cu)
- [torbles/math_jet.py](torbles/math_jet.py)
- [tests/test_math_jet.py](tests/test_math_jet.py)
- [bench/benchmark_math_jet.py](bench/benchmark_math_jet.py)

Acceptance tests:

- Compare second/third-order coefficients against exact polynomial derivatives.
- Bound jet dimension/order and test factorial conventions at zero and nonzero expansion centers.

Benchmark cases:

- Vary one to eight seed directions and jet orders 1/2/3 on shared polynomial circuits; measure state bytes and arithmetic operations.
- Compare a joint value/derivative pass with independently transformed circuits; include sparse gradients and a dense worst-supported interaction case.

Additional limits:

- Evaluation uses int64 numerators over fixed int64 denominators; many decimal or rational factors refuse early (no big-rational or GF(p) evaluation engine, only exact conversions of results).
- Circuits have one output and the expansion center is fixed per plan.
- Only the aggregate runs on CUDA; plan compilation, readouts and conversions are CPU.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/JET-002/tests_test_math_jet.json](results/math/JET-002/tests_test_math_jet.json)
- tests: [results/math/JET-002/sanitized_test_math_jet.json](results/math/JET-002/sanitized_test_math_jet.json)
- benchmarks: [results/math/JET-002/benchmark_cpu_math_jet_vm.json](results/math/JET-002/benchmark_cpu_math_jet_vm.json)
- benchmarks: [results/math/JET-002/benchmark_device_math_jet_WIN_OHVNN0FUC09.json](results/math/JET-002/benchmark_device_math_jet_WIN_OHVNN0FUC09.json)

## RESP: Whole response functions

Return complete parameterized aggregate or piecewise loss functions.

<a id="resp-001"></a>

### RESP-001: Polynomial aggregate response functions

Status: **validated**. Delivery wave: **W0**.

Reduce a data/parameter polynomial against immutable moments once, returning a native parameter polynomial rather than rerunning each parameter value.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-003](#fnd-003), [CALC-001](#calc-001), [CALC-002](#calc-002), [FND-012](#fnd-012)

Native CPU: Implemented and validated in native C within the declared first-slice rational, variable, degree and term budgets.

GPU: Coefficient algebra, calculus and function evaluation are native CPU work. A new GPU calculus/function kernel is not implemented.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [docs/MATHEMATICS.md](docs/MATHEMATICS.md): Current first-slice API and bounded domain/backend semantics.

Implemented files:

- [torbles/mathematics.c](torbles/mathematics.c)
- [torbles/mathematics.py](torbles/mathematics.py)
- [tests/test_mathematics.py](tests/test_mathematics.py)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

Acceptance tests:

- Squared-loss response, gradient and integral match direct exact row evaluation for signed data and duplicate rows.
- Missing moments, unsupported data degrees, incompatible contexts and out-of-range coefficients fail explicitly.

Benchmark cases:

- Compare 1, 10, 100 and 10,000 parameter evaluations with direct native queries; report moment construction and actual scan count.
- Show repeated evaluation no longer touches data rows; retain one-shot cases where state construction loses.

Validation receipts:

Measured runtime: `bae446434669e110608f41ece4187a911334e477`.

Current typed runtime passed all 48 focused tests and the full actual-CUDA quick suite. Native edge/sanitizer gates, 15,000 independent LRU operations and 200 independent public-API value/derivative checks passed. Six sequential synthetic benchmark processes retain all raw samples and source/library identities. Cache reuse saves repeated construction; typed coefficient dispatch adds measurable overhead. This validates only the declared bounded polynomial/scalar behavior, not the remaining inventory extensions.

- tests: [results/algebra_domains/validation.json](results/algebra_domains/validation.json)
- tests: [results/algebra_domains/focused.log](results/algebra_domains/focused.log)
- tests: [results/algebra_domains/native_domain_validation.json](results/algebra_domains/native_domain_validation.json)
- tests: [results/algebra_domains/native_state_validation.json](results/algebra_domains/native_state_validation.json)
- tests: [results/algebra_domains/independent_audit.json](results/algebra_domains/independent_audit.json)
- tests: [results/algebra_foundation/native_validation.json](results/algebra_foundation/native_validation.json)
- tests: [results/algebra_domains/benchmark_audit.json](results/algebra_domains/benchmark_audit.json)
- benchmarks: [results/algebra_domains/cache_cpu.json](results/algebra_domains/cache_cpu.json)
- benchmarks: [results/algebra_domains/cache_gpu.json](results/algebra_domains/cache_gpu.json)
- benchmarks: [results/algebra_domains/response_baseline_cpu.json](results/algebra_domains/response_baseline_cpu.json)
- benchmarks: [results/algebra_domains/response_current_cpu.json](results/algebra_domains/response_current_cpu.json)
- benchmarks: [results/algebra_domains/response_baseline_gpu.json](results/algebra_domains/response_baseline_gpu.json)
- benchmarks: [results/algebra_domains/response_current_gpu.json](results/algebra_domains/response_current_gpu.json)

<a id="resp-002"></a>

### RESP-002: Piecewise absolute and hinge loss curves

Status: **in_progress**. Delivery wave: **W2**.

Sorted histogram/prefix moments return complete absolute, hinge and squared-hinge loss functions over observed breakpoints.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-002](#dist-002)

Native CPU: Native C (math_resp.c): complete piecewise absolute, hinge (both orientations) and squared-hinge loss functions of a parameter from the sorted support and its prefix moments: exact values and one-sided slopes at and between every breakpoint, minimizer sets with a tie convention, empty and degenerate cases specified.

GPU: CUDA kernel math_resp_gpu.cu: batched evaluation of many parameter requests against the retained prefix state, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_resp.h`
- `torbles/math_resp.c`
- `torbles/math_resp_gpu.cu`
- `torbles/math_resp.py`
- `tests/test_math_resp.py`
- `bench/benchmark_math_resp.py`
- `torbles/math_dist.c`

Acceptance tests:

- Evaluate values and one-sided slopes at/between every breakpoint against direct row sums.
- Duplicate observations, empty input and degenerate minima preserve the specified tie convention.

Benchmark cases:

- Use squared, absolute and hinge loss families over 1k/100k/1M rows with 1/10/100/10,000 parameter requests; separate basis/prefix construction from reuse.
- Vary distinct breakpoints or basis rank from 2 to the supported limit and include one-shot requests where retained-state construction loses.

Additional limits:

- The piecewise function and minimizers are CPU; one direct pass is cheaper than building the state for a single request (measured).
- At most 2^20 breakpoints per group with the default budget.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/RESP-002/tests_test_math_resp.json](results/math/RESP-002/tests_test_math_resp.json)
- tests: [results/math/RESP-002/sanitized_test_math_resp.json](results/math/RESP-002/sanitized_test_math_resp.json)
- benchmarks: [results/math/RESP-002/benchmark_cpu_math_resp_vm.json](results/math/RESP-002/benchmark_cpu_math_resp_vm.json)

## INV: Inverse queries

Solve for parameter values or changes producing a requested answer.

<a id="inv-001"></a>

### INV-001: Linear and univariate polynomial inverse answers

Status: **validated**. Delivery wave: **W1**.

Solve bounded exact linear equations first; extend to certified intervals/algebraic roots for univariate response targets.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [FND-009](#fnd-009)

Native CPU: Native C (math_inv.c): exact affine target inverses (unique point, empty with the reason, whole domain, or an affine family with a witness, directions and exact parameter ranges, by exact elimination and Fourier-Motzkin bounds), certified real-root isolation of univariate polynomial targets in declared bounds (square-free decomposition, exact rational roots, Sturm-certified isolating intervals with multiplicities, refinement to a width), and exhaustive finite-grid search with an explicit incomplete result.

GPU: CUDA kernels math_inv_gpu.cu: the finite-grid sweep (exact evaluation at every grid point, matches collected order-independently) and batched exact sign evaluation for root refinement, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_inv.h](torbles/math_inv.h)
- [torbles/math_inv.c](torbles/math_inv.c)
- [torbles/math_inv_gpu.cu](torbles/math_inv_gpu.cu)
- [torbles/math_inv.py](torbles/math_inv.py)
- [tests/test_math_inv.py](tests/test_math_inv.py)
- [bench/benchmark_math_inv.py](bench/benchmark_math_inv.py)

Acceptance tests:

- Exact linear target inverses detect unique, empty and whole-domain solution sets.
- Polynomial root isolation returns all roots in declared bounds and never labels a numerical candidate exact.

Benchmark cases:

- Use linear and bounded-degree univariate targets with zero/one/multiple/no solutions and domains containing exact boundary roots; report solving separately from state construction.
- Sweep parameter dimension 1/2/3 and finite domains of 8/16/32 values per coordinate, retaining incomplete-search cases and proof bounds.

Additional limits:

- Sign evaluation and refinement use at most 128 moduli (values up to about 7,800 bits) and raise OverflowError beyond.
- The grid search takes first-slice polynomials (int64 rational coefficients, degree <= 8, at most 8 unknowns); solve_affine takes up to 64 unknowns.
- Sturm sequences and Fourier-Motzkin elimination are exact but grow quickly with degree or free parameters; both are budgeted.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/INV-001/tests_test_math_inv.json](results/math/INV-001/tests_test_math_inv.json)
- tests: [results/math/INV-001/sanitized_test_math_inv.json](results/math/INV-001/sanitized_test_math_inv.json)
- benchmarks: [results/math/INV-001/benchmark_cpu_math_inv_vm.json](results/math/INV-001/benchmark_cpu_math_inv_vm.json)
- benchmarks: [results/math/INV-001/benchmark_device_math_inv_WIN_OHVNN0FUC09.json](results/math/INV-001/benchmark_device_math_inv_WIN_OHVNN0FUC09.json)

<a id="inv-002"></a>

### INV-002: Constrained minimal interventions

Status: **in_progress**. Delivery wave: **W3**.

Find bounded parameter or feature interventions that attain a target while respecting declared linear/convex constraints and an explicit cost.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [INV-001](#inv-001), [UNC-001](#unc-001)

Native CPU: Native minimal interventions: the change d of declared inputs (x = x0 + d) of minimal cost (weighted L1, Linf, squared L2 with rational weights, or the weighted number of changed inputs) such that the function attains a target f(x) op t (op =, !=, <, <=, >, >=) or, as a membership change, the complement of the predicate's value at x0 (decided exactly). Every result is labelled an algebraic intervention on the function as given (no causal model; causal semantics are refused as UNSUPPORTED). Linear targets (coefficients or first-slice expressions of degree <= 1, e.g. RESP-001 functions; n <= 16) under A x <= b, E x = e, boxes on x and d and the norm balls ||d||_1 <= r, ||d||_inf <= r: exact LPs (L1, Linf) and the exact projection QP (L2) by OPT-002's simplex and active-set methods, KKT-verified; L0 by supports in increasing weight (an exact feasibility LP each), every minimal support reported; strict targets on their closure with attainment decided by an LP over the optimal set (else UNATTAINED with the exact infimum); uniqueness by 2n LPs over the optimal set with a second optimal change when not unique. Univariate polynomial targets: the roots of f - t in the domain by INV-001 (exact rational roots or certified isolating intervals, refinable), the feasible set as points and open intervals decided by exact signs, the nearest feasible points on both sides compared exactly (a tie of irrational distances decided by a common root of (f - t)(x0 + y) and (f - t)(x0 - y)). Finite grids (1..8 coordinates, at most 2^24 points, any first-slice polynomial target with up to 15 polynomial constraints and linear rows): every condition as an integer polynomial in cleared numerators whose sign is decided from residues modulo verified primes (INV-001) and Garner's digits; integer cost tables over a common denominator; grid blocks pruned soundly by outward-rounded interval enclosures (FND-009); the remaining points ordered by (exact cost, grid index) and scanned in chunks; the minimum and every tie of the minimal cost listed; past the iteration budget INCOMPLETE with a proven lower bound (every cheaper point examined or pruned), the best found as upper bound and a resume rank; the output budget truncates the ties explicitly. Aggregates over the selected rows of an FND-004 scope: COUNT(*), COUNT(v), SUM(v), AVG(v) WHERE x + d op c (v may be x, then shifted too) with SQL NULL semantics, piecewise in d with exact breakpoints c - x_k from a state of sorted distinct values with prefix sums built once; every piece decided exactly so rows entering and leaving the predicate are part of the objective, never the frozen row set; staleness refused through the scope. Budgets (variables, degree, support, tensor, states, iterations, output, bytes) are checked before allocation.

GPU: tr_mgpu_interv_grid: the batched exhaustive evaluation of a chunk of a grid plan's cost order. Launch 1: one work item per point evaluates every condition by its exact sign (residues modulo the verified primes, Garner's digits compared with (M - 1) / 2) and its integer cost, sets its bit of the feasibility bitmap by atomic OR and lowers the minimal feasible cost by atomic minimum; launch 2: one work item per point sets its tie bit (feasible and of the matched cost) by atomic OR. The results do not depend on thread order and equal the CPU's tr_interv_grid_eval bit for bit (the same inline per-point functions of math_interv.h); Grid.search(backend='cuda') runs every chunk of the exact search on it with the same answers. The plan, the order, the search loop, the LPs, the roots and the aggregates run on the CPU. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_interv.h`
- `torbles/math_interv.c`
- `torbles/math_interv_gpu.cu`
- `torbles/math_interv.py`
- `tests/test_math_interv.py`
- `bench/benchmark_math_interv.py`

Acceptance tests:

- Brute-force small discrete domains certify the minimum and all tie cases.
- Separate algebraic intervention from causal claims; retain predicate-membership changes in the objective.

Benchmark cases:

- Use linear and bounded-degree univariate targets with zero/one/multiple/no solutions and domains containing exact boundary roots; report solving separately from state construction.
- Sweep parameter dimension 1/2/3 and finite domains of 8/16/32 values per coordinate, retaining incomplete-search cases and proof bounds.

Additional limits:

- Answers are algebraic interventions on the function as given: no causal model, no claim about changing inputs in the world.
- Linear targets must be linear in the changed inputs; polynomial targets in several inputs are searched only over explicit finite grids (at most 8 coordinates and 2^24 points).
- Convex constraints are limited to linear rows, boxes and the L1 / Linf norm balls; general convex sets are not supported.
- The L0 cost enumerates supports (2^n, budgeted) and the L1 / Linf uniqueness test costs 2n further exact LPs, which is slow at n = 16.
- Aggregates shift one feature with one comparison predicate and support COUNT, SUM and AVG; changes at irrational roots are reported as isolating intervals.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `6ff26a49d485b3aad9a0daa2606659900a2c2a09`.

- tests: [results/math/INV-002/tests_test_math_interv.json](results/math/INV-002/tests_test_math_interv.json)
- tests: [results/math/INV-002/sanitized_test_math_interv.json](results/math/INV-002/sanitized_test_math_interv.json)
- benchmarks: [results/math/INV-002/benchmark_cpu_math_interv_vm.json](results/math/INV-002/benchmark_cpu_math_interv_vm.json)

## OPT: Optimization

Optimize query-defined objectives under explicitly declared domains and constraints.

<a id="opt-001"></a>

### OPT-001: Quadratic objective minimizers

Status: **validated**. Delivery wave: **W1**.

Minimize scalar or small multivariate quadratic response functions with singular/indefinite/domain cases explicit.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [LIN-001](#lin-001)

Native CPU: Native C (math_opt.c, with the exact solver of math_lin.c): exact minimizers of quadratics q(t) = t'At/2 - b't + c: positive definite (unique), positive semidefinite and consistent (minimal value and an affine family), inconsistent or indefinite (unbounded below with a certificate direction), and box-constrained optima by exact face enumeration with exact KKT checks; infeasible boxes refused.

GPU: CUDA kernel math_opt_gpu.cu: the face filter of the box-constrained problem (each work item one active set), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_opt.h](torbles/math_opt.h)
- [torbles/math_opt.c](torbles/math_opt.c)
- [torbles/math_opt_gpu.cu](torbles/math_opt_gpu.cu)
- [torbles/math_opt.py](torbles/math_opt.py)
- [tests/test_math_opt.py](tests/test_math_opt.py)
- [bench/benchmark_math_opt.py](bench/benchmark_math_opt.py)
- [torbles/math_lin.c](torbles/math_lin.c)

Acceptance tests:

- Compare positive-definite solutions with independent exact linear solves.
- Return unbounded, nonunique and constrained-boundary optima correctly rather than dividing by zero.

Benchmark cases:

- Benchmark positive-definite, singular and indefinite quadratics in 1/2/4/8 parameters, with and without box constraints; report matrix/state build and solve separately.
- Compare repeated optimization over 1k/100k/1M source rows using shared states against full objective recomputation; retain infeasible/unbounded cases.

Additional limits:

- Boxes up to 8 parameters (3^n faces); one-sided bounds and general linear constraints are not supported.
- Unconstrained solves and the exact face phase run on the CPU.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/OPT-001/tests_test_math_opt.json](results/math/OPT-001/tests_test_math_opt.json)
- tests: [results/math/OPT-001/sanitized_test_math_opt.json](results/math/OPT-001/sanitized_test_math_opt.json)
- benchmarks: [results/math/OPT-001/benchmark_cpu_math_opt_vm.json](results/math/OPT-001/benchmark_cpu_math_opt_vm.json)
- benchmarks: [results/math/OPT-001/benchmark_device_math_opt_WIN_OHVNN0FUC09.json](results/math/OPT-001/benchmark_device_math_opt_WIN_OHVNN0FUC09.json)

<a id="opt-002"></a>

### OPT-002: Bounded convex and polynomial optimization

Status: **in_progress**. Delivery wave: **W3**.

Use supported convex structure or bounded certified search on small polynomial systems; report feasibility and optimality evidence.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [OPT-001](#opt-001), [INV-001](#inv-001), [UNC-001](#unc-001)

Native CPU: Native exact optimization in FND-009 big rationals. Convex problems: minimize x^T H x / 2 + g^T x + c0 (H symmetric PSD, decided by LIN-002's exact inertia; H = 0 an LP) subject to A x <= b, E x = e and per-variable bounds (either side optional), declared for n <= 8 (representation 64 variables, 256 rows), also from an OPT-001 objective (matrices, a LIN-001 moment state, a first-slice polynomial). LPs by a tableau simplex over x = u - v with slacks, phase 1 with artificials and Bland's rule; QPs by a primal active-set method from the phase-1 vertex with exact equality-constrained steps (LIN-002 solves, zero-curvature descent directions from LIN-002's inconsistency certificate), ratio tests and smallest-index drops, its step budget refused as NOT_CONVERGED. Every answer is verified exactly before it is returned: OPTIMAL with multipliers satisfying stationarity, feasibility and complementary slackness, uniqueness decided (2n LPs over the optimal set, skipped when H is definite or every nonbasic reduced cost is positive) and a second optimal point when not unique; INFEASIBLE with a Farkas certificate; UNBOUNDED with a feasible anchor and a recession ray of negative curvature or of zero curvature and negative rate. An indefinite H: on an affine feasible set the reduced Hessian N^T H N decides it exactly (a curvature ray, or the convex reduced problem's optimum or rate ray); otherwise a negative direction of N^T H N or an LP ray that is a recession direction certifies unboundedness; otherwise a bounded feasible polyhedron of dimension <= 4 after exact elimination of the equalities is handed to the branch and bound (box from 2 dim LPs); else UNSUPPORTED. Polynomial problems (n <= 4, degree <= 8 per variable, dense size <= 256 per polynomial, <= 64 constraints <=, >=, =, a finite rational box): certified best-first branch and bound over dyadic boxes; batches of boxes enclosed by the natural interval extension intersected with the centered form in FND-009 outward-rounded arithmetic; boxes pruned only when a constraint is proven violated, the lower bound exceeds the incumbent, or it cannot improve the incumbent by more than the tolerance; incumbents exact (box corners and midpoints whose constraints hold exactly); the answer is [lower, upper] with upper attained exactly at a feasible witness, COMPLETE only within the declared tolerance, else INCOMPLETE with the reason (node budget, time limit, depth limit); infeasibility certified by leaf boxes each violating a constraint on its enclosure; one-variable unconstrained problems by INV-001's roots of the derivative. Budgets (variables, degree, support, tensor, iterations, output, bytes) are checked before allocation.

GPU: tr_mgpu_polyopt_boxes: one work item per box of a branch-and-bound batch evaluates the enclosures of the objective and of every constraint (natural interval extension and centered form, FND-009's contraction-safe directed double operations) and writes its own output slot (status keep / infeasible / bound, the objective enclosure, the violated constraint, whether the box is inside every constraint); it runs the inline functions of math_polyopt.h, so it equals the CPU's tr_polyopt_boxes bit for bit, and backend='cuda' sends every batch of the search to it with the same answers. The simplex, the active set, the uniqueness and hand-off LPs, the exact incumbents and the search control stay on the CPU. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_polyopt.h`
- `torbles/math_polyopt.c`
- `torbles/math_polyopt_gpu.cu`
- `torbles/math_polyopt.py`
- `tests/test_math_polyopt.py`
- `bench/benchmark_math_polyopt.py`

Acceptance tests:

- Compare small exhaustive feasible sets and interval-certified optima.
- A resource-limited search returns an explicit incomplete/certified-bound result, not a false global optimum.

Benchmark cases:

- Benchmark positive-definite, singular and indefinite quadratics in 1/2/4/8 parameters, with and without box constraints; report matrix/state build and solve separately.
- Compare repeated optimization over 1k/100k/1M source rows using shared states against full objective recomputation; retain infeasible/unbounded cases.

Additional limits:

- Exact big-rational pivots: the convex method is meant for n <= 8 variables and tens of rows; the uniqueness test costs 2n further LPs.
- An indefinite QP over an unbounded polyhedron is decided only when the feasible set is affine or a recession direction of negative curvature is found (the reduced Hessian's negative direction or an LP ray); otherwise it is refused (copositivity is not decided).
- Indefinite QPs are handed to the branch and bound only when bounded and of dimension <= 4 after the equalities; larger ones are refused.
- The branch and bound is exponential in n and log(1/tolerance) and uses first-order enclosures (no relaxations): tight tolerances on flat or degenerate minima end incomplete with certified bounds.
- With equality constraints a witness exists only where a dyadic box corner or midpoint satisfies them exactly; otherwise feasibility stays unknown and upper is absent.
- The one-variable method's INV-001 evaluations are refused past the iteration budget instead of returning an incomplete result.
- The CUDA kernel has been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `6ff26a49d485b3aad9a0daa2606659900a2c2a09`.

- tests: [results/math/OPT-002/tests_test_math_polyopt.json](results/math/OPT-002/tests_test_math_polyopt.json)
- tests: [results/math/OPT-002/sanitized_test_math_polyopt.json](results/math/OPT-002/sanitized_test_math_polyopt.json)
- benchmarks: [results/math/OPT-002/benchmark_cpu_math_polyopt_vm.json](results/math/OPT-002/benchmark_cpu_math_polyopt_vm.json)

## GEN: Generating functions

Distribution, count and path generating polynomials or formal series.

<a id="gen-001"></a>

### GEN-001: Finite-support distribution generating polynomials

Status: **in_progress**. Delivery wave: **W2**.

Represent count/weight distributions as sparse or bounded dense generating polynomials; support coefficient extraction and moment operators.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: Native C (math_gen.c): generating polynomials of 1-D and 2-D finite supports with exact (big-integer) coefficients keeping bag multiplicity and declared shifts for negative values; coefficient extraction, exact evaluation at rationals, raw and factorial moments through the Euler operator and derivatives, products, powers, truncated products and conversion to FND-009 truncated series, with support growth, coefficient pairs and bytes budgeted before allocation.

GPU: CUDA kernels math_gen_gpu.cu: the exact 128-bit convolution of two coefficient arrays and batched exact evaluation, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_gen.h`
- `torbles/math_gen.c`
- `torbles/math_gen_gpu.cu`
- `torbles/math_gen.py`
- `tests/test_math_gen.py`
- `bench/benchmark_math_gen.py`
- `torbles/math_dist.c`

Acceptance tests:

- G(1), coefficients and Euler-derivative moments match counts and exact row moments.
- Negative values use declared shifts/Laurent support; coefficients retain bag multiplicity.

Benchmark cases:

- Build sparse/dense generating states for support sizes 8/64/1,024 and two-dimensional support; measure coefficient, moment and evaluation requests separately.
- Compare convolution/truncated-series orders 4/16/64 with explicit tiny products; record support growth, intermediate bytes and budget refusals.

Additional limits:

- At most 2 variables; the device kernels take one variable, int64 inputs and 128-bit outputs (larger coefficients are CPU).
- Powers multiply sequentially (quadratic, no FFT); the byte budget counts big-integer coefficients without allocator overhead.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/GEN-001/tests_test_math_gen.json](results/math/GEN-001/tests_test_math_gen.json)
- tests: [results/math/GEN-001/sanitized_test_math_gen.json](results/math/GEN-001/sanitized_test_math_gen.json)
- benchmarks: [results/math/GEN-001/benchmark_cpu_math_gen_vm.json](results/math/GEN-001/benchmark_cpu_math_gen_vm.json)

<a id="gen-002"></a>

### GEN-002: Convolution and bounded formal series

Status: **in_progress**. Delivery wave: **W3**.

Products encode independent/full-product combinations; truncated series support declared bounded coefficients and path lengths.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GEN-001](#gen-001), [FND-009](#fnd-009)

Native CPU: Native C (math_conv.c, on GEN-001): products of generating states for independent draws and full Cartesian products with support growth budgeted; every factor records its source, and a product of factors from the same keyed scope without an independence declaration is refused (DomainError) in favour of the per-key conditional product or the joint state; bounded series with declared coefficient bounds and walk tables over path lengths, divergent walk sums refused.

GPU: CUDA kernels math_conv_gpu.cu: batched exact convolutions, per-key conditional products in parallel over keys, and walk tables, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_conv.h`
- `torbles/math_conv.c`
- `torbles/math_conv_gpu.cu`
- `torbles/math_conv.py`
- `tests/test_math_conv.py`
- `bench/benchmark_math_conv.py`
- `torbles/math_gen.c`

Acceptance tests:

- Compare convolution coefficients with enumerated small Cartesian products.
- Key-conditioned joins or dependent events cannot use an unconditional product of marginal distributions.

Benchmark cases:

- Build sparse/dense generating states for support sizes 8/64/1,024 and two-dimensional support; measure coefficient, moment and evaluation requests separately.
- Compare convolution/truncated-series orders 4/16/64 with explicit tiny products; record support growth, intermediate bytes and budget refusals.

Additional limits:

- At most two variables (GEN-001's limit); per-key coefficients below 2^127; keyed families are 1-D and dense per key.
- Independence declarations and cross-table key correspondence are trusted, not verified against the data.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/GEN-002/tests_test_math_conv.json](results/math/GEN-002/tests_test_math_conv.json)
- tests: [results/math/GEN-002/sanitized_test_math_conv.json](results/math/GEN-002/sanitized_test_math_conv.json)
- benchmarks: [results/math/GEN-002/benchmark_cpu_math_conv_vm.json](results/math/GEN-002/benchmark_cpu_math_conv_vm.json)

## DIST: Distribution algebra

Histograms, tails, quantiles, entropy and information measures.

<a id="dist-001"></a>

### DIST-001: Reusable scalar and joint histograms

Status: **in_progress**. Delivery wave: **W2**.

Native sparse/dense support states with counts/weights, explicit NULL/presence states and support budgets.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004)

Native CPU: Native C (math_dist.c): grouped scalar and joint (2..3 column) histograms over FND-004 scopes with exact counts, optional exact weight sums, explicit per-column NULL categories and group presence, in dense (bounded domain) or sparse (sorted CSR support) form, budgeted; exact dense/sparse conversions, merges, marginals and rollups.

GPU: CUDA kernels math_dist_gpu.cu: histogram construction on the device (atomic cell counts over the bounded code domain, compaction in several launches), equal to the CPU state.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing GROUP BY and histogram bridge can produce distributions, but no general immutable native distribution-state API.
- [torbles/sqlfuncs.py](torbles/sqlfuncs.py): Existing SQL histogram finalizers do not establish the proposed native state abstraction.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dist.h`
- `torbles/math_dist.c`
- `torbles/math_dist_gpu.cu`
- `torbles/math_dist.py`
- `tests/test_math_dist.py`
- `bench/benchmark_math_dist.py`

Acceptance tests:

- Match grouped native counts for duplicates, NULL categories and empty selections.
- Dense/sparse conversions preserve every represented cell and refuse over-budget joint domains.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

Additional limits:

- Device sparse builds need the dense counting scratch over the value range to fit the budget (no device sort); merges, rollups, marginals and conversions are CPU.
- At most 3 coordinates and 2^32-1 scope rows; the CPU sort path uses 128 bytes of scratch per row.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/DIST-001/tests_test_math_dist.json](results/math/DIST-001/tests_test_math_dist.json)
- tests: [results/math/DIST-001/sanitized_test_math_dist.json](results/math/DIST-001/sanitized_test_math_dist.json)
- benchmarks: [results/math/DIST-001/benchmark_cpu_math_dist_vm.json](results/math/DIST-001/benchmark_cpu_math_dist_vm.json)

<a id="dist-002"></a>

### DIST-002: Prefix distribution calculus and exact quantiles

Status: **in_progress**. Delivery wave: **W2**.

Sorted histogram prefix counts/moments answer CDF, tails, quantiles, trimmed sums and range integrals.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: Native C (math_dist.c): prefix counts, weights and moments over the sorted support: exact CDF and tails at and between observed values with declared endpoint inclusion, exact quantiles (PERCENTILE_DISC and PERCENTILE_CONT as exact rationals), trimmed sums and range integrals of value^k, ties and empty input specified.

GPU: CUDA kernel math_dist_gpu.cu: batched CDF, quantile and range requests (one work item per request, a binary search over the uploaded prefix arrays), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing dominance and range lowerings are reusable building blocks.
- [torbles/sqlfuncs.py](torbles/sqlfuncs.py): Existing SQL percentile/quantile finalizers; shared native prefix-state API remains planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dist.h`
- `torbles/math_dist.c`
- `torbles/math_dist_gpu.cu`
- `torbles/math_dist.py`
- `tests/test_math_dist.py`
- `bench/benchmark_math_dist.py`

Acceptance tests:

- Every rank and threshold at/between observed values matches expanded-row reference calculations.
- Specify interpolation, empty input, ties and inclusion of range endpoints explicitly.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

Additional limits:

- Weighted quantiles need nonnegative weights and treat each row as repeated by its weight in the column's stored integer units.
- Prefix construction is CPU; results are exact rationals returned as (numerator, denominator).

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/DIST-002/tests_test_math_dist.json](results/math/DIST-002/tests_test_math_dist.json)
- tests: [results/math/DIST-002/sanitized_test_math_dist.json](results/math/DIST-002/sanitized_test_math_dist.json)
- benchmarks: [results/math/DIST-002/benchmark_cpu_math_dist_vm.json](results/math/DIST-002/benchmark_cpu_math_dist_vm.json)

<a id="dist-003"></a>

### DIST-003: Entropy, mutual information and contingency transforms

Status: **in_progress**. Delivery wave: **W2**.

Compute information and contingency statistics on retained count support, preserving joint dependence and numerical precision.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [FND-009](#fnd-009)

Native CPU: Native C (math_info.c): marginal and contingency tables from DIST-001 joint histogram states (no row reads), NULL as a category or excluded per declared coordinate; entropy, conditional entropy, mutual and conditional mutual information, G and chi-square statistics, Cramer's V, as exact rational bounds from certified fixed-point logarithms summed exactly in 256 bits, plus outward-rounded doubles; chi-square and V^2 also as exact fractions; statistics over coordinates not held jointly are refused.

GPU: CUDA kernel math_info_gpu.cu: the per-cell certified terms and their exact 256-bit sums, equal bit for bit to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_info.h`
- `torbles/math_info.c`
- `torbles/math_info_gpu.cu`
- `torbles/math_info.py`
- `tests/test_math_info.py`
- `bench/benchmark_math_info.py`
- `torbles/math_dist.c`

Acceptance tests:

- Compare entropy/MI with explicit normalized joint tables including structural zeros.
- Do not infer a joint distribution from moments or independent marginals without declared assumptions.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

Additional limits:

- A group's total weight below 2^64; at most 3 coordinates; marginal tables, exact chi-square and request answers are CPU; no p-values; results in bits are enclosed as doubles only (exact bounds are in nats).

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/DIST-003/tests_test_math_info.json](results/math/DIST-003/tests_test_math_info.json)
- tests: [results/math/DIST-003/sanitized_test_math_info.json](results/math/DIST-003/sanitized_test_math_info.json)
- benchmarks: [results/math/DIST-003/benchmark_cpu_math_info_vm.json](results/math/DIST-003/benchmark_cpu_math_info_vm.json)

## PAIR: Pairwise statistics

Distance, ranking and pairwise summaries without unconditional pair enumeration.

<a id="pair-001"></a>

### PAIR-001: Sorted-distribution pairwise summaries

Status: **in_progress**. Delivery wave: **W2**.

Compute supported absolute-distance sums, weighted rank/AUC and one-dimensional distribution distances from sorted counts/prefix states.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-002](#dist-002)

Native CPU: Native C (math_pair.c): pairwise summaries from sorted counts and prefix states without enumerating pairs: absolute-difference sums (Gini mean difference), Mann-Whitney U and AUC with ties counted half, Kolmogorov-Smirnov, 1-Wasserstein and two-sample Cramer-von Mises, exact rationals with self-pair, ordering, normalization and empty-class conventions explicit.

GPU: CUDA kernel math_pair_gpu.cu: batched summary requests over groups or class pairs from the retained prefix states (one work item per request), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pair.h`
- `torbles/math_pair.c`
- `torbles/math_pair_gpu.cu`
- `torbles/math_pair.py`
- `tests/test_math_pair.py`
- `bench/benchmark_math_pair.py`
- `torbles/math_dist.c`

Acceptance tests:

- Match enumerated pair calculations for tiny tied/weighted inputs.
- Specify self-pair exclusion, ordering, normalization and class-empty conventions.

Benchmark cases:

- For scalar supports 8/64/1,024, compare pairwise summaries with explicit pairs on small data and retained-prefix computation on larger repeated inputs.
- Vary ties, class imbalance, diagonal exclusion and weights; separate support sorting from 1/100/10,000 summary requests.

Additional limits:

- Each class's total mass below 2^64 (Cramer-von Mises needs a further size bound; overflow refused); KS two-sided only; with weights a self-pair is a weight unit paired with itself.
- One thread per request: a request over near-row-sized support is long on the device.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/PAIR-001/tests_test_math_pair.json](results/math/PAIR-001/tests_test_math_pair.json)
- tests: [results/math/PAIR-001/sanitized_test_math_pair.json](results/math/PAIR-001/sanitized_test_math_pair.json)
- benchmarks: [results/math/PAIR-001/benchmark_cpu_math_pair_vm.json](results/math/PAIR-001/benchmark_cpu_math_pair_vm.json)

<a id="pair-002"></a>

### PAIR-002: Structured pairwise query planning

Status: **in_progress**. Delivery wave: **W4**.

Choose support aggregation, finite-feature factorization or bounded direct pairs according to the exact statistic.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PAIR-001](#pair-001), [KERNEL-001](#kernel-001)

Native CPU: Native C (math_pairplan.c): pairwise statistics S = sum over a declared pair set of w_i w_j k(x_i, x_j) over the elements of FND-004 scope groups (selected rows with a present weight; listwise, or NULL a category for the delta kernel), with the pair set exact (within one group: ordered or unordered pairs, self-pairs included or excluded; between two groups: each cross pair once, or both orientations), the weighted pair count P and optional means S / P; kernels |x - y| (L1), Mann-Whitney [x > y] + 1/2 [x = y], KERNEL-001's polynomial, monomial and delta kernels, declared similarity tables (symmetric or not, a default elsewhere), L-infinity bands and squared distances; RBF, Laplace and other kernels without an exact value refused, never approximated. tr_pairplan_make plans a batch: an exact profile (per group n, sum w, sum |w|, sum w^2, from one scope pass or a retained table), up to seven candidates (support aggregation and direct pairs on the CPU and CUDA, KERNEL-001 finite features dense and sparse, PAIR-001 sorted prefix), each with its validity (unsupported structure, negative weight masses, unproven magnitude bound), its budget needs (iterations = kernel evaluations: the declared pair budget; support, states, output, bytes, device bytes) and exact amounts per FND-008 cost component under a stated work model, costed and budget-checked by tr_planner_choose with the family's rates; the cheapest admissible candidate runs, forces run exactly their match or refuse, a plan without an admissible candidate is refused with its evidence. Support aggregation sorts the elements into distinct value tuples with multiplicity m, W = sum w and Q = sum w^2 and weights a diagonal entry pair by W^2 - Q, W^2 or (W^2 -+ Q) / 2; direct pairs evaluate the same exact evaluator over one entry per row; the feature and prefix strategies issue KERNEL-001 SUM and PAIR-001 ABS / Mann-Whitney requests with the corrections their conventions need. Every strategy returns the same exact num / den in 1024-bit words after a magnitude proof (bits of sum |w| per side plus the kernel's bound <= 1020; requests proven within 126 bits are summed in 128-bit words, the same integers); retained support, rows, prefix and feature states are reused with exact sizes.

GPU: CUDA kernels math_pairplan_gpu.cu: a support or rows table uploaded once (tr_mgpu_pairplan_upload), then per batch (tr_mgpu_pairplan_query) one work item per (request, 32 x 32 tile of entry pairs) running the shared tr_pairplan_tile (128- or 1024-bit sums as on the CPU) into a local 1024-bit sum added to the request's accumulator with a sixteen-word atomic carry chain (exact modulo 2^1024, hence exact under the host's magnitude proof), visits and overflow flags by atomics, then one work item per request running the shared tr_pairplan_finish (pair counts, means); the host checks every request and the budget (output, iterations, device bytes) before copying; equal to the CPU evaluation bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pairplan.h`
- `torbles/math_pairplan.c`
- `torbles/math_pairplan_gpu.cu`
- `torbles/math_pairplan.py`
- `tests/test_math_pairplan.py`
- `bench/benchmark_math_pairplan.py`

Acceptance tests:

- Unsupported arbitrary similarities retain a correct costed direct plan or fail explicitly.
- Verify pair multiplicities and diagonal treatment across support compression.

Benchmark cases:

- For scalar supports 8/64/1,024, compare pairwise summaries with explicit pairs on small data and retained-prefix computation on larger repeated inputs.
- Vary ties, class imbalance, diagonal exclusion and weights; separate support sorting from 1/100/10,000 summary requests.

Additional limits:

- Planned statistics are sums over pairs of w_i w_j k(x_i, x_j): PAIR-001's distribution distances (KS, 1-Wasserstein, Cramer-von Mises) are not pair sums and stay PAIR-001's own API; no FND-007 certificate is issued (the capability computes values, it proves no data relation).
- The finite-feature and sorted-prefix strategies run on the CPU only (their CUDA paths remain KERNEL-001's and PAIR-001's own APIs); support tables are sorted on the host for both devices.
- Distinct supports in a plan without a retained state are exact upper bounds (min of the elements and the cells of the value box); the rates are hand-set relative estimates, not calibrated times, and the work model's per-evaluation operation counts are stated, not measured.
- At most 8 feature columns; Mann-Whitney over one column; an unordered pair set needs a symmetric kernel (refused per request otherwise); results are limited to 1024 bits (an unprovable magnitude rejects the support and direct candidates); the sorted prefix needs nonnegative weights and class masses below 2^64.
- The CUDA tile kernel uses 255 registers with a 576-byte stack frame and register spills (1024-bit arithmetic); its work items are tiles, so a single small request has little parallelism; the CPU loops are single-threaded.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/PAIR-002/tests_test_math_pairplan.json](results/math/PAIR-002/tests_test_math_pairplan.json)
- tests: [results/math/PAIR-002/sanitized_test_math_pairplan.json](results/math/PAIR-002/sanitized_test_math_pairplan.json)
- benchmarks: [results/math/PAIR-002/benchmark_cpu_math_pairplan_vm.json](results/math/PAIR-002/benchmark_cpu_math_pairplan_vm.json)

## LIN: Linear algebra

Moment matrices, regression, covariance, projections and eigensystems.

<a id="lin-001"></a>

### LIN-001: Grouped moment matrices and regression systems

Status: **validated**. Delivery wave: **W1**.

Assemble exact counts, sums, X^T X and X^T y from a compatible basis and grain; general grouped/wide banks extend the initial scalar slice.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-003](#fnd-003), [FND-004](#fnd-004)

Native CPU: Native C (math_lin.c): exact grouped counts, sums, X^T X, X^T y and y^T y (with or without an intercept) for up to 16 features from a scope, by a fused pass or from the FND-004 bank, under a declared validity policy: common-listwise (a true Gram matrix) or pairwise (per-entry counts).

GPU: CUDA kernel math_lin_gpu.cu: the fused moment scan per group with exact wide atomics, equal to the CPU states.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/sqlexec.py](torbles/sqlexec.py): SQL statistical aggregates lower polynomial moments through existing native queries.
- [torbles/alg.c](torbles/alg.c): Existing linear/bilinear masked reductions support matrix-state assembly.

Implemented files:

- [torbles/math_lin.h](torbles/math_lin.h)
- [torbles/math_lin.c](torbles/math_lin.c)
- [torbles/math_lin_gpu.cu](torbles/math_lin_gpu.cu)
- [torbles/math_lin.py](torbles/math_lin.py)
- [tests/test_math_lin.py](tests/test_math_lin.py)
- [bench/benchmark_math_lin.py](bench/benchmark_math_lin.py)

Acceptance tests:

- Match explicit matrix products for signed, duplicated and selectively grouped rows.
- Pairwise validity and common-listwise validity are distinct declared policies; matrix dimensions/output budgets are enforced.

Benchmark cases:

- Use feature dimensions 2/4/8/16, groups 1/16/256 and 1k/100k/1M rows; report moment/matrix assembly, bytes and solve/decomposition time separately.
- Include well-conditioned, nearly singular and exactly rank-deficient matrices; compare repeated model/ridge/deletion requests against complete reconstruction.

Additional limits:

- At most 16 features; dense groups (absent groups use memory).
- Pairwise states built from a bank lack per-pair sums, so covariance on them is refused; the pairwise covariance/correlation may be indefinite (tested), the raw pairwise Gram is PSD.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/LIN-001/tests_test_math_lin.json](results/math/LIN-001/tests_test_math_lin.json)
- tests: [results/math/LIN-001/sanitized_test_math_lin.json](results/math/LIN-001/sanitized_test_math_lin.json)
- benchmarks: [results/math/LIN-001/benchmark_cpu_math_lin_vm.json](results/math/LIN-001/benchmark_cpu_math_lin_vm.json)
- benchmarks: [results/math/LIN-001/benchmark_device_math_lin_WIN_OHVNN0FUC09.json](results/math/LIN-001/benchmark_device_math_lin_WIN_OHVNN0FUC09.json)

<a id="lin-002"></a>

### LIN-002: Solves, projections, covariance and eigensystems

Status: **validated**. Delivery wave: **W1**.

Small native matrix operators finalize retained states, with exact solves where supported and explicit numerical contracts for decompositions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [FND-009](#fnd-009)

Native CPU: Native C (math_lin.c): exact rational solves (modular elimination over several primes, CRT, rational reconstruction and exact verification of A x = b), exact rank, solution sets of rank-deficient systems (unique, none, affine family), exact inertia (positive definite, semidefinite, indefinite), exact covariance and correlation-squared, exact projections and residual sums; numerical symmetric eigensystems (cyclic Jacobi) returned with residual, orthogonality and eigenvalue error bounds.

GPU: CUDA kernel math_lin_gpu.cu: batched modular solves, one work item per (system, prime), with the CPU doing CRT, reconstruction and exact verification.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implemented files:

- [torbles/math_lin.h](torbles/math_lin.h)
- [torbles/math_lin.c](torbles/math_lin.c)
- [torbles/math_lin_gpu.cu](torbles/math_lin_gpu.cu)
- [torbles/math_lin.py](torbles/math_lin.py)
- [tests/test_math_lin.py](tests/test_math_lin.py)
- [bench/benchmark_math_lin.py](bench/benchmark_math_lin.py)

Acceptance tests:

- Exact rational solves match Fraction elimination; rank-deficient and indefinite cases are explicit.
- Numerical PCA/projection residuals meet declared tolerances; no stability claim follows from exact input moments.

Benchmark cases:

- Use feature dimensions 2/4/8/16, groups 1/16/256 and 1k/100k/1M rows; report moment/matrix assembly, bytes and solve/decomposition time separately.
- Include well-conditioned, nearly singular and exactly rank-deficient matrices; compare repeated model/ridge/deletion requests against complete reconstruction.

Additional limits:

- Solves up to 64 x 63; the CRT and verification step on the CPU costs about an exact solve, so the device pays off only with many systems.
- Eigensystems are floating point with certified residual and eigenvalue bounds; no claim on eigenvector accuracy beyond the reported bounds.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/LIN-002/tests_test_math_lin.json](results/math/LIN-002/tests_test_math_lin.json)
- tests: [results/math/LIN-002/sanitized_test_math_lin.json](results/math/LIN-002/sanitized_test_math_lin.json)
- benchmarks: [results/math/LIN-002/benchmark_cpu_math_lin_vm.json](results/math/LIN-002/benchmark_cpu_math_lin_vm.json)
- benchmarks: [results/math/LIN-002/benchmark_device_math_lin_WIN_OHVNN0FUC09.json](results/math/LIN-002/benchmark_device_math_lin_WIN_OHVNN0FUC09.json)

<a id="lin-003"></a>

### LIN-003: Leave-group-out and low-rank model updates

Status: **in_progress**. Delivery wave: **W4**.

Subtract compatible group states and use checked low-rank update identities with singularity fallbacks.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [LIN-002](#lin-002), [DELTA-001](#delta-001)

Native CPU: Native C (math_linupd.c): immutable exact solved models of LIN-002's normal equations of a fit of a LIN-001 moment state (value units, exact ridge on the features, feature subsets; A, A0, b, y^T y, sum y, n, the design) or of any square rational system, with LIN-002's canonical verified solution set (unique / family with the canonical nullspace / none with a certificate) and statistics, and (on request) A^-1 and det(A) from fraction-free Gauss-Jordan elimination of [M | diag(L) | c] over big integers with A A^-1 = I verified exactly. Exact Sherman-Morrison-Woodbury updates A + U V^T (rank <= 64): the capacitance K = I + V^T A^-1 U and its determinant (rank one: 1 + v^T A^-1 u) formed exactly, det(K) != 0 giving A'^-1 and x' (verified A' x' = b'), det(K) = 0 or no kept inverse following the declared contract (an exact rebuild with LIN-002's canonical singular solution set, or SingularError); row deletions and insertions (U = -/+ Z^T, V = Z^T, y^T y, sum y, n updated; one row: 1 -/+ h), ridge changes (the rank-p Woodbury update or a rebuild from A0, equal), subtraction or addition of a compatible group state. Leave-one-group-out plans: the group states copied, their 192-bit total and every complement total - g formed exactly, the exact integer complement systems (row scaling den 10^(s_u + T)) with their Hadamard bounds; every system solved exactly (LIN-002's Bareiss), or finished from modular residues: agreeing primes, Garner CRT of the det-scaled reduced row echelon form, exact verification against the integer system, more primes when short. A binary64 variant: Gauss-Jordan with partial pivoting on the midpoint of A's tight enclosure, certified by interval arithmetic (alpha >= ||I - R A||_inf, bound = ||R (b - A x~)||_inf / (1 - alpha), kappa >= ||A|| ||R||), refused when alpha >= 1 or the bound exceeds the tolerance.

GPU: CUDA kernel math_linupd_gpu.cu (tr_mgpu_linupd_logo): batched leave-one-group-out solves, one work item per (group, prime in [2^61, 2^62)): the complement normal system of its group formed modulo its prime directly from the copied group states and their total (192-bit words reduced, row scaling and ridge modulo p), then LIN-002's modular Gauss-Jordan elimination (tr_lin_modular_rref) into its own output block (header and det-scaled reduced row echelon form); one output slot per work item, equal bit for bit to the CPU reference tr_linupd_logo_residues (emulated with a scrambled thread order in the tests); the host finish (prime choice, CRT within the Hadamard bounds, exact verification) gives the same canonical models as the CPU path.

Bounds:

- Fit / system: variables = n (<= 63), tensor = n (2n + 1), output = n (n + 1), plus LIN-002's normal-equation needs k (k + 1); updates: tensor = n r + r^2 + n^2 (r <= 64), output = n (n + 1).
- Leave-one-group-out plan: states = groups x slots, tensor = systems x n (n + 1), output = systems, bytes = the copies; each system as a fit; CUDA: tensor = systems x primes x (4 + n (n + 1)), bytes = the device copies and residues (at most 1024 primes).
- Floating variant: tensor = n^2, bytes = 48 n^2; refused with NotConvergedError when a pivot vanishes, alpha >= 1 or the certified bound exceeds tol.

Implementation targets (proposed until the capability is validated):

- `torbles/math_linupd.h`
- `torbles/math_linupd.c`
- `torbles/math_linupd_gpu.cu`
- `torbles/math_linupd.py`
- `tests/test_math_linupd.py`
- `bench/benchmark_math_linupd.py`

Acceptance tests:

- Compare deletion/leave-group-out models with complete reconstruction.
- Detect update denominators or condition failures and rebuild/fail according to the numerical contract.

Benchmark cases:

- Use feature dimensions 2/4/8/16, groups 1/16/256 and 1k/100k/1M rows; report moment/matrix assembly, bytes and solve/decomposition time separately.
- Include well-conditioned, nearly singular and exactly rank-deficient matrices; compare repeated model/ridge/deletion requests against complete reconstruction.

Additional limits:

- Woodbury updates use canonical big rationals (O(n^2 r) rational operations, a gcd each): in the benchmark (20,000 rows) a one-row Woodbury deletion took 2.4 to 14 times as long as the exact rebuild of the updated normal equations with its inverse (3 to 17 unknowns) and was faster than the complete reconstruction from the rows only for small systems (5x at 3 unknowns, within 10% at 9 and 17); a ridge change touching every feature was 3 to 5 times slower than its rebuild from A0. A fraction-free (adjugate) inverse representation that would avoid most gcds is not implemented.
- Leave-group-out subtracts moment states and solves each complement system again; there is no low-rank downdate from Gram matrices (moment states do not keep the rows).
- Row deletions need the deleted rows and cannot detect rows that were never in the data; rows updates are for common-listwise regression models only.
- The CUDA kernel covers the modular LOGO eliminations; the host forms the exact integer systems (Hadamard bounds, verification) and runs the CRT, comparable in cost to the exact CPU solve; updates and the floating variant are CPU code.
- No FND-007 certificate is emitted: FND-007's claim kinds are row relations of a scope or polynomial laws, and a fitted model proves neither; every exact solution is verified natively (A x = b, A A^-1 = I, the CRT reconstruction against the integer system) and flagged so.

Validation receipts:

Measured runtime: `cacb92841d6cae9a8a136a680c29f3cecf01b657`.

- tests: [results/math/LIN-003/tests_test_math_linupd.json](results/math/LIN-003/tests_test_math_linupd.json)
- tests: [results/math/LIN-003/sanitized_test_math_linupd.json](results/math/LIN-003/sanitized_test_math_linupd.json)
- benchmarks: [results/math/LIN-003/benchmark_cpu_math_linupd_vm.json](results/math/LIN-003/benchmark_cpu_math_linupd_vm.json)

## TENSOR: Tensor algebra

Sparse/factored moment tensors and contraction planning.

<a id="tensor-001"></a>

### TENSOR-001: Sparse and factored moment tensors

Status: **in_progress**. Delivery wave: **W4**.

Retain declared multi-index states without forcing dense Cartesian expansion.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [FND-004](#fnd-004)

Native CPU: Native C (math_tensor.c): exact retained multi-index states over an FND-004 scope, per group: moment tensors T_g[i] = sum w prod_k x[col_k[i_k]] of order 0..8 over per-axis column lists (the constant 1 allowed; listwise, or SQL with per-entry contributing-row counts; axes sharing one column list accumulate their sorted multi-indices only) and joint count tensors over categorical axes (scope or explicit domains, NULL a slot of its own, weighted or not), with the selected and contributing rows kept apart from NULL validity; dense, sparse (CSR, strictly increasing multi-indices) and factored (declared rank-one sums with int64 factors, built without expansion) forms with exact conversions preserving order, dimensions, axes, groups, rows, counts and entries; marginalization, slicing, permutation, outer products, ungrouping and entry access in any form; int128 entries proven before any row is read (N x max|w| x prod_k max|x| <= 2^127 - 1) and every derived sum checked; a chunked builder (init, accumulate, merge, finish) that gives the same state bit for bit for any chunking; budgets (variables, degree, tensor, states, support, bytes) checked before allocation.

GPU: CUDA kernels math_tensor_gpu.cu (tr_mgpu_tensor_build): dense moment work items (entry, row slice) accumulate locally and add with the exact two-word atomic (symmetric tensors on their canonical entries, then a copy launch); count work items add one atomic per row; sparse builds insert every emission into an open-addressing table of at least twice the capacity (a local 64-bit compare-and-swap claims a slot) whose occupied slots the host sorts into the canonical CSR; the readback is adopted by tr_tensor_from_states after shape, row, order and bound checks; equal to the CPU bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_tensor.h`
- `torbles/math_tensor.c`
- `torbles/math_tensor_gpu.cu`
- `torbles/math_tensor.py`
- `tests/test_math_tensor.py`
- `bench/benchmark_math_tensor.py`

Acceptance tests:

- Match enumerated tensor entries and marginalizations for small joint domains.
- Sparse/dense/factored conversions preserve ordering, dimensions, validity and exact coefficients.

Benchmark cases:

- Vary tensor order 2/3/4, support sparsity 1%/10%/dense and dimensions 2/4/8; record actual intermediate shapes and bytes.
- Compare multiple contraction orders, sparse and dense states on identical tensors; include cases where the cheapest order changes and over-budget intermediates are refused.

Additional limits:

- Entries are limited to 128 bits (refused beyond); orders up to 8, at most 64 columns per axis, dimensions below 2^32 and fewer than 2^64 cells per group.
- A factored form's bound is a conservative sum over its terms, so some factored operations refuse what the dense form computes; count tensors have no factored form.
- The CPU loops are single-threaded; the device sparse build sorts its hash table on the host.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/TENSOR-001/tests_test_math_tensor.json](results/math/TENSOR-001/tests_test_math_tensor.json)
- tests: [results/math/TENSOR-001/sanitized_test_math_tensor.json](results/math/TENSOR-001/sanitized_test_math_tensor.json)
- benchmarks: [results/math/TENSOR-001/benchmark_cpu_math_tensor_vm.json](results/math/TENSOR-001/benchmark_cpu_math_tensor_vm.json)

<a id="tensor-002"></a>

### TENSOR-002: Costed contraction planning

Status: **in_progress**. Delivery wave: **W4**.

Choose variable/contraction order, kernel, intermediate shape and materialization lifetime for bounded tensor networks.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TENSOR-001](#tensor-001), [FND-008](#fnd-008)

Native CPU: Native C (math_tensor.c): networks of 1..8 TENSOR-001 tensors with named indices (at most 16 labels; a grouped tensor's group label first) contracted by Einstein summation along orders of pairwise steps: every order enumerated when there are at most max_orders (default 64), otherwise a bounded search (the greedy orders, then depth-first ones); each order with the CPU dense, CPU sparse (operands matched on their shared labels, products sorted and reduced) and CUDA dense kernels; exact per-step labels, dimensions, cells, summed combinations, stored entries (sparse: exact upper bounds), bytes, scratch, live bytes by materialization lifetime, the peak, the output bytes and a magnitude bound per step proven before execution; exact amounts per FND-008 cost component costed and budget-checked by tr_planner_choose (the family's own rates passed as the planner's override table), the cheapest admissible candidate chosen and rejected candidates kept as evidence with their refusals; execution of any admissible candidate returns the exact tensor with the actual per-step entries, bytes and peak.

GPU: CUDA kernel math_tensor_gpu.cu (tr_mgpu_tensor_contract): the dense candidate's pairwise steps, one work item per result entry evaluating the shared tr_tensor_dense_entry (exact int128 products and sums; every step bound re-proven on the host before copying), intermediates kept on the device and freed after their consumer; equal to the CPU kernels bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_tensor.h`
- `torbles/math_tensor.c`
- `torbles/math_tensor_gpu.cu`
- `torbles/math_tensor.py`
- `tests/test_math_tensor.py`
- `bench/benchmark_math_tensor.py`

Acceptance tests:

- Every candidate order agrees with a tiny direct contraction oracle.
- Check complete workspace/output costs and refuse plans whose intermediates exceed budgets.

Benchmark cases:

- Vary tensor order 2/3/4, support sparsity 1%/10%/dense and dimensions 2/4/8; record actual intermediate shapes and bytes.
- Compare multiple contraction orders, sparse and dense states on identical tensors; include cases where the cheapest order changes and over-budget intermediates are refused.

Additional limits:

- No factored contraction kernel: factored operands are expanded (costed and budgeted); the CUDA contraction is dense only (sparse candidates run on the CPU).
- Sparse sizes in a plan are exact upper bounds, not the data's actual supports; the rates are hand-set estimates in relative work units (not calibrated times); traces within one tensor are not supported.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/TENSOR-002/tests_test_math_tensor.json](results/math/TENSOR-002/tests_test_math_tensor.json)
- tests: [results/math/TENSOR-002/sanitized_test_math_tensor.json](results/math/TENSOR-002/sanitized_test_math_tensor.json)
- benchmarks: [results/math/TENSOR-002/benchmark_cpu_math_tensor_vm.json](results/math/TENSOR-002/benchmark_cpu_math_tensor_vm.json)

## KERNEL: Kernel calculations

Finite-feature embeddings and certified approximate kernel computations.

<a id="kernel-001"></a>

### KERNEL-001: Exact finite-feature kernel embeddings

Status: **in_progress**. Delivery wave: **W4**.

Polynomial or explicitly finite-feature kernels reduce means, comparisons and selected pair sums to retained feature states.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TENSOR-001](#tensor-001)

Native CPU: Native C (math_kernel.c): kernels with an explicit finite feature map, polynomial (x.y + c)^d (d <= 8, |c|^d <= 2^127 - 1; linear and homogeneous included), declared monomial sums sum_t w_t x^a_t y^a_t and delta kernels (categorical match, NULL a category of its own), reduced to TENSOR-001 states: the listwise moment tensor of order D >= d over (1, x_1..x_n), or the count tensor, with per group W = sum w, Q = sum w^2 and the declared kernel's self-pair sum sum w^2 k(x, x) (1024 bits, proven to fit before any row is read); exact kernel sums over selected pairs (across groups, within a group with or without self-pairs, ordered or unordered) and their weighted pair counts, kernel mean inner products, biased and unbiased MMD^2 and witness values, as exact rationals in 1024-bit words (overflow refused per request, never wrapped), evaluated over the C(n + d, d) sorted multi-indices with their multiplicities; any kernel the state retains (family, degree <= D, a subset of its columns) is evaluated on the same state, an absent degree, dimension or family is refused (UNSUPPORTED), and RBF, Laplace and other kernels are refused, never approximated; budgets (the tensor's needs, 288 bytes a group, evaluation items, output) checked before allocation.

GPU: CUDA kernels math_kernel_gpu.cu: the feature state (tr_mgpu_kernel_build: TENSOR-001's device build of the feature tensor, then one work item per row slice summing W, Q and Dg locally and adding them with exact atomics, the 1024-bit sums by a local 16-word atomic carry chain), adopted by tr_kernel_from_states after consistency checks; batched requests (tr_mgpu_kernel_upload once per state and kernel, then tr_mgpu_kernel_query: one work item per request running the shared evaluator tr_kernel_eval_request, one output slot each); equal to the CPU bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_kernel.h`
- `torbles/math_kernel.c`
- `torbles/math_kernel_gpu.cu`
- `torbles/math_kernel.py`
- `tests/test_math_kernel.py`
- `bench/benchmark_math_kernel.py`

Acceptance tests:

- Match explicit kernel matrices on small inputs including weights and diagonal exclusions.
- Verify required feature degree and dimensions; an absent feature cannot silently be approximated.

Benchmark cases:

- Compare exact polynomial features at dimensions 2/4/8 and degrees 1/2/4 against direct small kernel matrices; separate feature-state construction and reuse.
- For explicitly approximate kernels vary feature counts 16/64/256 and fixed seeds; record observed error against a small exact numerical oracle alongside time and memory.

Additional limits:

- Exact finite-feature kernels only: RBF, Laplace and other kernels without an exact finite feature map are refused (declared approximations are KERNEL-002's).
- The feature tensor stores (n + 1)^D cells per group (dense, or sparse at the data's support), so the tensor budget bounds the dimension and degree; listwise only (a row with a NULL feature column does not contribute).
- Answers are limited to 1024 bits (refused beyond) and are unreduced num / den natively; the CUDA query kernel has a 1984-byte stack frame with register spills and runs one work item per request.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/KERNEL-001/tests_test_math_kernel.json](results/math/KERNEL-001/tests_test_math_kernel.json)
- tests: [results/math/KERNEL-001/sanitized_test_math_kernel.json](results/math/KERNEL-001/sanitized_test_math_kernel.json)
- benchmarks: [results/math/KERNEL-001/benchmark_cpu_math_kernel_vm.json](results/math/KERNEL-001/benchmark_cpu_math_kernel_vm.json)

<a id="kernel-002"></a>

### KERNEL-002: Approximate spectral feature kernels

Status: **in_progress**. Delivery wave: **W4**.

Offer RBF/other selected kernels through explicit finite-feature approximation, seeds, precision and error accounting.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SPEC-001](#spec-001), [UNC-002](#unc-002)

Native CPU: Native C (math_kapprox.c, the per-row code shared with CUDA in math_kapprox.h): Gaussian exp(-gamma |x - y|^2), Laplace exp(-gamma |x - y|_1) and Cauchy prod 1 / (1 + gamma^2 (x_j - y_j)^2) kernels (exact int64-rational gamma, d <= 16 columns of one decimal scale, an optional weight) through declared finite-feature approximations. Taylor (Gaussian): E(x) E(y) sum_{n <= N} (2 gamma x.y)^n / n! (N <= 12) as features E(x) u^a over the C(d + N, N) multi-indices with KERNEL-001's sorted-multi-index multiplicities, per row t = gamma |x|^2 / ln 2 from the exact 192-bit sum of squares and a 64.320 fixed-point constant (577-bit 1/ln 2), E = 2^-t by CALC-003's certified exp, exact u = X 2^-b, midpoint-radius monomials, outward integers summed exactly in 128 bits; the declared domain |x|^2 <= radius2 (declared or the data's largest, exact; rows and points outside refused) and the certified per-pair bound e^-U T_N(U) = P(Poisson(U) > N), U = 2 gamma radius2, as an exact rational. Random Fourier features (all three): D <= 2^20 frequencies from a native seeded generator (splitmix64 words in UNC-002's stream convention, 128-bit uniform cells; Box-Muller, tangent and exponential transforms evaluated with certified fixed-point ln (atanh), integer square roots and CALC-003 sin / cos after SPEC-001's exact quarter-turn reduction, rounded to dyadics with each sample's enclosure checked), deduplicated with multiplicities, accumulated through SPEC-001's fixed spectral basis (plan, exact sums, adoption checks); per pair, with probability >= 1 - delta under the stated independence assumption, |k^ - k| <= 2 sqrt(ln(2 / delta) / (2 D)) + beta, beta the deterministic bias 2 P_bad + 2 rho sqrt(d) eta of the dyadic frequencies (a coupling with exact samples), all exact rationals. Per group the selected and counted rows, W = sum w, A = sum |w| (128 bits), Q = sum w^2 (192 bits). Answers (KERNEL-001's pair semantics: sums across and within groups with or without self-pairs, ordered or unordered, kernel means, biased and unbiased MMD^2, witness values at logical points inside the domain): exact rational enclosures of the approximant (midpoint-radius products in KERNEL-001's 384-bit accumulators, FND-009 big rationals) with UNC-002 records: norm abs, assumptions (domain, no overflow, independence for random features), error terms (remainder eps F; probabilistic h R with the failure probability and quantization beta F, F the weight factor, R the Hoeffding range), their exact sum, the bound [lo - error, hi + error], label certified or probabilistic; a requested error budget below the per-pair bound is refused (NOT_CONVERGED). tr_kapprox_authorize_exact always refuses (FND-007 reason approximate, statistical for random features). Persistence in FND-011's record format (header, FND-007 source record, the spec with seed, count and failure, derived constants, frequencies and SPEC-001 reduction constants, metadata, payload, FND-011's checksum) with FND-011's refusal order; a reload gives identical answers and bytes, with a scope it must be fresh and the plan rebuilt from the recorded seed must agree; FND-011's typed loader refuses the record (an unknown type, exact 0). Exact polynomial kernels are answered by KERNEL-001. Budgets checked before allocation.

GPU: tr_mgpu_kapprox_sums (math_kapprox_gpu.cu): a counts kernel (work item = row slice: rows, counted rows, W and A with two-word 128-bit atomics, Q with a three-word atomic carry chain, local while the group stays the same; the largest |x|^2 of the slice in its own output slot), for Taylor a prefactor kernel (one slot per row: exp(-gamma |x|^2) = 2^e m by the shared inline code) and a feature kernel (work item = (item, row slice): the monomial recomputed in the CPU's multiplication order, outward integers accumulated locally per group and added with tr_atomic_add_wide), for random features the counts kernel then SPEC-001's tr_mgpu_fourier_sums (certified cos / sin of omega.x per row summed as exact scaled integers); the host validates the views and budgets every device copy, and tr_kapprox_from_sums adopts the readback after consistency checks (and SPEC-001's adoption checks). Equal to the CPU bit for bit (emulated with a scrambled thread order in the tests); compiled with nvcc for sm_80 without warnings; not yet run on a device.

Bounds:

- Build: variables = d (<= 16) + weight, degree = N (<= 12), support = Taylor items C(d + N, N) or the D (<= 2^20) frequencies, states = groups x items, tensor = selected rows x items (random features: SPEC-001's rows x frequencies), iterations = D d generator draws, bytes = the plan, the sums and the state (SPEC-001's own needs); CUDA: every device copy (tr_kapprox_device_bytes).
- Requests: output = requests, tensor = points x items (point features), bytes = the point features; every big rational of an answer is FND-009-budgeted; save / load: bytes = the record (twice for a load).
- Refused before allocation: AlgebraBudgetError; NotConvergedError when the per-pair bound exceeds the error budget; UnsupportedError for d > 16, N > 12, D > 2^20, Taylor features of Laplace or Cauchy kernels, frequencies past 2^60, a Gaussian factor below 2^-(2^24); DomainError for columns of different scales and data or points outside the declared domain.

Implementation targets (proposed until the capability is validated):

- `torbles/math_kapprox.h`
- `torbles/math_kapprox.c`
- `torbles/math_kapprox_gpu.cu`
- `torbles/math_kapprox.py`
- `tests/test_math_kapprox.py`
- `bench/benchmark_math_kapprox.py`

Acceptance tests:

- Compare approximation error with direct small kernel calculations under declared assumptions.
- Persist feature parameters/seeds and prevent approximate states authorizing exact algebraic rewrites.

Benchmark cases:

- Compare exact polynomial features at dimensions 2/4/8 and degrees 1/2/4 against direct small kernel matrices; separate feature-state construction and reuse.
- For explicitly approximate kernels vary feature counts 16/64/256 and fixed seeds; record observed error against a small exact numerical oracle alongside time and memory.

Additional limits:

- Deterministic (Taylor) features for the Gaussian kernel only, with N <= 12; the bound P(Poisson(U) > N) is useful only while U = 2 gamma radius2 is moderate, and the per-answer weight factors are worst cases (A_a A_b for a sum).
- Random-feature bounds hold per answer with probability >= 1 - delta (m answers together: m delta) under the assumption that the seeded generator's words are i.i.d. uniform; Hoeffding's ranges are conservative (observed errors at most about 0.14 of the stated bound in the tests and benchmark).
- Only the row accumulation runs on CUDA; answers, point features and the frequency generator (host code with FND-009 big integers, 20 to 50 us a coordinate) run on the CPU.
- Kernel columns must share one decimal scale; witness points are logical values inside the declared domain; at most 16 dimensions and 2^20 frequencies.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.
- No FND-007 certificate is emitted: an approximate kernel quantity is neither a row relation nor a polynomial law; the answers carry their own error records.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/KERNEL-002/tests_test_math_kapprox.json](results/math/KERNEL-002/tests_test_math_kapprox.json)
- tests: [results/math/KERNEL-002/sanitized_test_math_kapprox.json](results/math/KERNEL-002/sanitized_test_math_kapprox.json)
- benchmarks: [results/math/KERNEL-002/benchmark_cpu_math_kapprox_vm.json](results/math/KERNEL-002/benchmark_cpu_math_kapprox_vm.json)

## SPEC: Spectral queries

Fourier, Walsh, parity correlation and frequency-response representations.

<a id="spec-001"></a>

### SPEC-001: Fixed spectral basis state

Status: **in_progress**. Delivery wave: **W4**.

Aggregate chosen Fourier/complex features once; phase actions and derivatives reuse that fixed frequency basis.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: Native C (math_fourier.c, the per-row reduction shared with CUDA in math_fourier.h): a basis plan over an FND-004 scope (d <= 16 columns, an optional weight column, unit radian / pi / turn, a set of m distinct exact int64-rational frequency vectors, sorted for exact lookup) with 64.192 fixed-point reduction constants (radians from a 512-bit 2/pi, pi and turn units exact when dyadic) and CALC-003's proven sum scale; per counted row the angle reduced exactly in 384-bit two's complement (quadrant exact, fraction to 128 bits; exact integer angles give exact sin / cos), CALC-003's certified 120-bit sin / cos, each weighted enclosure rounded outward to integers and summed exactly in 128 bits per (group, frequency); per group the selected and counted rows and the exact weight sum. Exact bounds [lo, hi] / (2^s 10^wscale) of C and S; for one radian dimension the sums equal CALC-003's phase family bit for bit. Phase actions without rescanning: translations tau and phases phi reduced exactly from the exact frequencies and the exact doubles, derivative orders in the translation (times i c omega_j) and the phase (times i), normalization by the exact weight sum (the empirical characteristic function), all as outward-rounded complex intervals; derivative states whose exact bounds are exact transforms (rotation by i^m, omega^alpha as big rationals, the unit factor c^|alpha| reported apart). Identity: columns, weight, unit, the exact frequency set and the scope's exact descriptor; require / Retained answer subsets and build new basis states for new frequencies; a frequency outside the basis is refused. The Boolean case: 2^k Walsh frequencies over 0 / 1 columns give SPEC-002's exact W(S) and the exact histogram through SPEC-002's inverse transform with its full-domain cost; incomplete Walsh sets, non-Boolean data and inexact coefficients are refused.

GPU: CUDA kernels math_fourier_gpu.cu (tr_mgpu_fourier_sums): a counts kernel (a slice of rows per work item: selected rows, counted rows and the exact weight sum per group, local while the group stays the same, one atomic add per counter, two-word 128-bit atomics for the weight sums) and a feature kernel (work item = (frequency, slice of rows): the same inline exact reduction and certified per-row enclosures as the CPU, accumulated locally per group and added with exact two-word 128-bit atomics tr_atomic_add_wide under the host's proof that no partial sum leaves 2^126); the readback is adopted natively after consistency checks; equal to the CPU bit for bit (emulated with a scrambled thread order in the tests). Actions, derivatives and the Walsh inverse run on the CPU.

Bounds:

- Build: variables = d (<= 16), support = m frequencies (<= 2^22), states = groups x m, tensor = selected rows x m, bytes = the plan, the sums and the state (CUDA: every device copy, tr_fourier_device_bytes).
- Actions: degree = |alpha| + beta with the state's (<= 64), output = requests x selected frequencies, tensor = output x (d + 1), bytes = the selected coefficients; derivative: degree, states, bytes; Walsh inverse: variables = k, tensor = 2^k, iterations = k, bytes = 40 x 2^k.
- Refused before allocation: AlgebraBudgetError; UnsupportedError for d > 16, a reduction constant >= 2^64, |tau| or |phi| >= 2^62; DomainError for missing spectral support, a zero weight sum (characteristic function) or a non-Boolean Walsh inverse.

Implementation targets (proposed until the capability is validated):

- `torbles/math_fourier.h`
- `torbles/math_fourier.c`
- `torbles/math_fourier_gpu.cu`
- `torbles/math_fourier.py`
- `tests/test_math_fourier.py`
- `bench/benchmark_math_fourier.py`

Acceptance tests:

- Fixed-frequency phase transforms match direct evaluation within numerical precision.
- New frequencies request new basis state; missing spectral support is not treated as zero.

Benchmark cases:

- Use fixed frequency sets of 1/8/32 elements and 1/100/10,000 phase/evaluation requests; distinguish new-frequency construction from fixed-basis actions.
- For Boolean transforms vary k=3/6/10/12 within budget, reporting all 2^k states, normalization and forward/inverse timings rather than claiming row-count-only cost.

Additional limits:

- One certified 120-bit sin / cos per (row, frequency), about 2 us on the CPU (no harmonic recurrences or angle-addition reuse); frequencies are int64 rationals; at most 16 dimensions.
- Actions return outward-rounded double rectangles (no exact bounds of rotations by transcendental angles); only derivative states carry exact rational transforms.
- Only the feature sums run on CUDA; actions, derivatives, lookups and the Walsh inverse are CPU code.
- The only inverse transform is the Boolean (Walsh) one; no general discrete Fourier inversion or band-limited reconstruction is offered, and nothing is ever reconstructed from an incomplete basis.
- No FND-007 certificate is emitted: FND-007's claim kinds are row relations of a scope or polynomial laws, and a basis proves neither; its exact scaled bounds (and the Walsh case's exact points) are the results' own evidence.

Validation receipts:

Measured runtime: `cacb92841d6cae9a8a136a680c29f3cecf01b657`.

- tests: [results/math/SPEC-001/tests_test_math_fourier.json](results/math/SPEC-001/tests_test_math_fourier.json)
- tests: [results/math/SPEC-001/sanitized_test_math_fourier.json](results/math/SPEC-001/sanitized_test_math_fourier.json)
- benchmarks: [results/math/SPEC-001/benchmark_cpu_math_fourier_vm.json](results/math/SPEC-001/benchmark_cpu_math_fourier_vm.json)

<a id="spec-002"></a>

### SPEC-002: Walsh spectra and parity correlations

Status: **in_progress**. Delivery wave: **W2**.

Exact Walsh-Hadamard transforms of bounded Boolean histograms expose parity and interaction coefficients.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LATTICE-001](#lattice-001), [FIELD-001](#field-001)

Native CPU: Native C (math_spec.c): exact Walsh-Hadamard transforms of the occupancy histogram (integer forward coefficients, exact rational inverse under a declared normalization), Parseval identities, parity correlations of condition subsets, constant parities cross-checked with FIELD-001, and coefficient truncation reported as an explicit approximation with its exact residual energy; the full 2^k cost is reported.

GPU: CUDA kernel math_spec_gpu.cu: the butterfly passes of the fast Walsh-Hadamard transform, one launch per pass, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_spec.h`
- `torbles/math_spec.c`
- `torbles/math_spec_gpu.cu`
- `torbles/math_spec.py`
- `tests/test_math_spec.py`
- `bench/benchmark_math_spec.py`

Acceptance tests:

- Forward/inverse transforms and Parseval identities agree under a declared normalization.
- Report full 2^k domain cost; coefficient truncation becomes an explicit approximation.

Benchmark cases:

- Use fixed frequency sets of 1/8/32 elements and 1/100/10,000 phase/evaluation requests; distinguish new-frequency construction from fixed-basis actions.
- For Boolean transforms vary k=3/6/10/12 within budget, reporting all 2^k states, normalization and forward/inverse timings rather than claiming row-count-only cost.

Additional limits:

- Correlations, truncation, evaluation and constant parities are CPU only.
- Transforms are exact in 128 bits (refused beyond); Parseval and truncation energies must fit 256 bits.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/SPEC-002/tests_test_math_spec.json](results/math/SPEC-002/tests_test_math_spec.json)
- tests: [results/math/SPEC-002/sanitized_test_math_spec.json](results/math/SPEC-002/sanitized_test_math_spec.json)
- benchmarks: [results/math/SPEC-002/benchmark_cpu_math_spec_vm.json](results/math/SPEC-002/benchmark_cpu_math_spec_vm.json)

## MULTI: Multiscale analysis

Hierarchy states, contrasts, adaptive refinement and localized changes.

<a id="multi-001"></a>

### MULTI-001: Hierarchy rollups and contrasts

Status: **in_progress**. Delivery wave: **W2**.

Use certified refinement maps for reusable coarse/fine counts, moments and Haar-like contrasts.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004)

Native CPU: Native C (math_multi.c): hierarchies of 2..8 levels over FND-004 scopes, refinement maps certified exhaustively on the data (a violation is a refusal with its witness row) or marked declared, per-level banks rolled up exactly from the finest level with explicit absent and all-NULL groups, localized updates with a full-rollup fallback, and exact Haar-like contrasts (child against parent, siblings).

GPU: CUDA kernels math_multi_gpu.cu: the exhaustive certification of a refinement map (minimum and maximum coarse code per fine group with atomics, the smallest violating row) and batched contrasts; the finest bank uses the FND-004 kernel.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/keydesign.py](torbles/keydesign.py): Existing discovered hierarchy/coordinate layouts.
- [torbles/algq.py](torbles/algq.py): Existing group-key projections and grouped reductions; reusable contrast states are planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_multi.h`
- `torbles/math_multi.c`
- `torbles/math_multi_gpu.cu`
- `torbles/math_multi.py`
- `tests/test_math_multi.py`
- `bench/benchmark_math_multi.py`

Acceptance tests:

- Child states merge exactly into parents with explicit absent/all-NULL groups.
- Reject incompatible hierarchy maps and preserve logical codec semantics.

Benchmark cases:

- Use balanced and skew hierarchies with 2/4/8 levels, groups 8/64/1,024 and repeated coarse/fine requests; report retained state bytes and rollup/refinement work.
- Compare localized changes in 1%/10%/all hierarchy leaves with complete reconstruction, including cases where adaptive refinement provides no gain.

Additional limits:

- Rollups and localized updates run on the CPU; contrasts are in logical units (the decimal scale is reported, not divided out).
- Map bindings are exact copies of the id arrays for array scopes (memory proportional to rows).
- After an update without new levels the maps are carried, not re-proven; re-certification rescans all rows.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/MULTI-001/tests_test_math_multi.json](results/math/MULTI-001/tests_test_math_multi.json)
- tests: [results/math/MULTI-001/sanitized_test_math_multi.json](results/math/MULTI-001/sanitized_test_math_multi.json)
- benchmarks: [results/math/MULTI-001/benchmark_cpu_math_multi_vm.json](results/math/MULTI-001/benchmark_cpu_math_multi_vm.json)

<a id="multi-002"></a>

### MULTI-002: Adaptive refinement and change localization

Status: **in_progress**. Delivery wave: **W4**.

Refine hierarchy regions only where exact or permitted approximate bounds require more detail.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [MULTI-001](#multi-001), [UNC-001](#unc-001)

Native CPU: Native refinement states of a MULTI-001 hierarchy: per level the exact FND-004 cells (rows, per term count and 128-bit sum), the UNC-001 abstract input of every group and column (rows, present values, min, max, gcd of differences) built from the leaves' scope and rolled up exactly along the maps, the parent and children links, and the DELTA-001 maintained state of the leaves (nonnegative measure). Requests on a target level: THRESHOLD A op T, RATIO A / D op T (decided as q A - p D op 0 with the operator flipped where D < 0, D = 0 undefined) and TOPK (largest or smallest, ties by the smaller id), A and D integer combinations of up to four SUM / COUNT / ROWS atoms of the basis. Region bounds: the atoms are split into validity classes (SQL term column sets, listwise, every row; one merged class when every atom is valid on every row of the region); per class the exact total E of the region, its valid rows n and the per-row interval [elo, ehi] of the class expression from UNC-001's exact abstract evaluation (affine forms cancel correlated terms) give every present descendant max_k min(k ehi, E - (n - k) elo) above and the symmetric bound below, evaluated exactly in 256-bit integers at the endpoints and the crossing; infinite sides are kept. Waves run coarse to fine in increasing group id: ALL / NONE regions are final, TOPK prunes regions whose upper bound is below the k-th largest witness (lower bounds and the means of the regions' present target descendants), the others are expanded up to an expansion limit, past which they are UNRESOLVED with their bounds; target groups are exact (TRUE / FALSE / UNDEFINED, TOPK IN / OUT with certain ranks, or UNRESOLVED). The entries partition the present target groups; the answer and unresolved bitmaps are their descendants. The complete fine-grain answer of the same request is the baseline. Change localization: DELTA-001 applies a signed delta to the leaves (compatibility, overflow proof, nonnegative promise); each touched leaf's exact change (rows, counts, 192-bit sums, present values, inserted extremes and gcd, deleted multiplicity) is summed into its ancestors only, level by level, the new state checked bit for bit against DELTA-001's cells at level 0, every changed region reported with its exact change, extrema after deletions kept as enclosures (not tight), maps carried; a delta of another schema, an overflow refusal or a leaf that becomes present without a parent falls back to a complete rebuild from new levels (maps certified; for a new group the rebuilt leaves must equal the old ones plus the delta) or is refused; a contradicted nonnegative promise is refused with DELTA-001's violation. Budgets (support, variables, circuit, iterations, output, states, bytes) are checked before allocation and cover the UNC-001 and DELTA-001 steps.

GPU: tr_mgpu_refine_bounds: the host gathers the cells and inputs of a wave's regions; one work item per region evaluates the inline tr_refine_region_bound (exact 256-bit integers, the UNC-001 exact abstract evaluation of each class expression in the item's own slots of a device scratch, chunks of 1024 regions per launch) and writes its own output slot. tr_mgpu_refine_rollup: one launch per level, one work item per affected region summing its affected children's exact changes (128-bit counts, 192-bit sums, present counts, inserted extremes and gcd) into its own slots; the host adopts it after a per-level conservation check. No atomics or barriers: every result is a function of its inputs only. The host validates every index the kernels follow and budgets the device copies. Equal to the CPU bit for bit (tested in the emulation with scrambled thread order); compiled with nvcc for sm_80 without warnings; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_refine.h`
- `torbles/math_refine.c`
- `torbles/math_refine_gpu.cu`
- `torbles/math_refine.py`
- `tests/test_math_refine.py`
- `bench/benchmark_math_refine.py`

Acceptance tests:

- Compare adaptive results with complete fine-grain states on small trees.
- Every unexpanded region carries a sound bound or is explicitly unresolved.

Benchmark cases:

- Use balanced and skew hierarchies with 2/4/8 levels, groups 8/64/1,024 and repeated coarse/fine requests; report retained state bytes and rollup/refinement work.
- Compare localized changes in 1%/10%/all hierarchy leaves with complete reconstruction, including cases where adaptive refinement provides no gain.

Additional limits:

- Region bounds use per-column UNC-001 inputs (no joint information between columns) and a descendant may hold a single row of its region, so lower bounds are weak: gains come from heavy-hitter thresholds and top-k with k at most the heavy regions; uniform data or a median threshold expands almost every region (measured: more work than the fine-grain answer).
- Aggregates are integer combinations of at most four SUM / COUNT / ROWS atoms per side (AVG through RATIO); no MIN / MAX, distinct or quantile requests; the decisions run on the host and tr_refine_supply trusts the bounds it is given (tr_refine_eval_bounds or tr_mgpu_refine_bounds).
- Nonnegative multiplicities only: a signed measure is refused (the bounds count rows).
- Localized updates carry the maps (status carried): a row inserted into a leaf is assumed to share the leaf's parent; a delete of a row that is not there but matches present rows is not detectable (DELTA-001); extrema after deletions are enclosures until a rebuild.
- A localized update copies the old state's arrays before patching the affected regions (memory and copy time proportional to the state; the rollup work proportional to the affected paths); a rebuild reports changes only when every level keeps its group count.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `6ff26a49d485b3aad9a0daa2606659900a2c2a09`.

- tests: [results/math/MULTI-002/tests_test_math_refine.json](results/math/MULTI-002/tests_test_math_refine.json)
- tests: [results/math/MULTI-002/sanitized_test_math_refine.json](results/math/MULTI-002/sanitized_test_math_refine.json)
- benchmarks: [results/math/MULTI-002/benchmark_cpu_math_refine_vm.json](results/math/MULTI-002/benchmark_cpu_math_refine_vm.json)

## SYM: Symmetry queries

Actions of translations, rotations, permutations and certified symmetries.

<a id="sym-001"></a>

### SYM-001: Closed-basis transformation actions

Status: **in_progress**. Delivery wave: **W4**.

Translate polynomial coordinates, rotate declared coordinates, and permute features by exact or precision-declared basis-action matrices.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [TENSOR-001](#tensor-001)

Native CPU: Native C (math_sym.c): a declared geometry (scope columns with a kind, NUMERIC or CATEGORICAL, and a unit label; decimal scales read from the FND-004 scope; nothing else inferred: an undeclared column, any metric operation on a categorical code, a rotation across units and a basis over a code are refused, DOMAIN_ERROR); transformations declared in value units with exact int64 rationals (translation, scaling, reflection, coordinate permutation, affine maps over named coordinates, rotations about a center by an exact rational point of the unit circle with c^2 + s^2 = 1 verified exactly, or by an angle in degrees: multiples of 90 exact, other angles certified enclosures of cos and sin from a Taylor series with its truncation bound in outward-rounded binary64 arithmetic) turned into maps in logical units (D A D^-1, D b with D = diag(10^scale), int128 rows (a x + b) / q reduced by their gcd, overflow refused) that compose exactly; monomial bases of up to 4096 terms of degree <= 8 over up to 32 coordinates (full, homogeneous or explicit); the action M_T (phi(T x) = M_T phi(x), hence moments(T(rows)) = M_T moments(rows)) by the exact multinomial expansion of every basis monomial, with closure decided and its witness (basis term, missing monomial) reported, as canonical int128 numerators over one common denominator (an index permutation for a coordinate permutation) or as interval enclosures; composition M_T M_S in 512-bit products reduced with FND-009 big integers, identity actions and exact equality; retained per-group moment states from FND-004 banks (JOINT, or SQL when every term is valid on every row) and TENSOR-001 LISTWISE moment tensors (with the constant: degree <= order; without: homogeneous), and batched requests (an action, a group, a basis term or an int64 functional, an optional mean over the constant term's moment) answered as exact 512-bit rationals or certified intervals by the evaluator shared with CUDA; budgets (variables, degree, support, tensor = nb^2, iterations = nb^3 for compositions, states, output, bytes) checked before allocation.

GPU: CUDA kernel math_sym_gpu.cu (tr_mgpu_sym_query): the retained moments, every action's matrix (and permutation index) and the functionals copied once per batch, then one work item per request running the shared evaluator tr_sym_eval_request (exact 512-bit two's complement products and sums of int128 numerators and moments, or outward-rounded interval products and sums with the explicit-fma helpers of FND-009, in the CPU's order), one output slot per request; equal to the CPU bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_sym.h`
- `torbles/math_sym.c`
- `torbles/math_sym_gpu.cu`
- `torbles/math_sym.py`
- `tests/test_math_sym.py`
- `bench/benchmark_math_sym.py`

Acceptance tests:

- Composition and identity actions match direct transformed-row evaluation.
- Only declared coordinate geometry is used; dictionary/code adjacency has no inferred physical metric.

Benchmark cases:

- Use polynomial translations, declared 2D rotations and feature permutations on bases of 3/6/15/32 states; compare direct transformed-row evaluation with basis actions.
- Vary transformation requests 1/100/10,000 and certified orbit sizes 2/4/8/16; include one broken-multiplicity example that must invalidate orbit reuse.

Additional limits:

- Exact entries are int128 numerators over one int128 denominator: a large translation at a high degree, or many compositions, are refused (OverflowError), not widened to big rationals.
- Only rotations by angles that are not multiples of 90 degrees are approximate: binary64 interval enclosures (53-bit endpoints, no multiprecision); their closure is decided structurally (an entry enclosure that is not exactly [0, 0] counts as a monomial of the image), so a basis closed only through an exact cancellation is refused.
- States are FND-004 banks (degree <= 4) and TENSOR-001 LISTWISE moment tensors (degree <= 8) without per-entry counts; NULL handling is listwise; the action construction and composition run on the CPU, single-threaded.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/SYM-001/tests_test_math_sym.json](results/math/SYM-001/tests_test_math_sym.json)
- tests: [results/math/SYM-001/sanitized_test_math_sym.json](results/math/SYM-001/sanitized_test_math_sym.json)
- benchmarks: [results/math/SYM-001/benchmark_cpu_math_sym_vm.json](results/math/SYM-001/benchmark_cpu_math_sym_vm.json)

<a id="sym-002"></a>

### SYM-002: Certified data symmetries and orbit reductions

Status: **in_progress**. Delivery wave: **W4**.

Verify multiplicity-preserving symmetries, construct orbit summaries and reduce work under those scoped certificates.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SYM-001](#sym-001), [FND-007](#fnd-007)

Native CPU: Native C (math_sym.c): declared groups of up to 16 generators that map the integer domain into itself (SYM-001 maps integral in logical units and invertible, det A != 0 decided exactly by elimination modulo 2^61 - 1 and fraction-free Bareiss in int128; e.g. coordinate permutations, reflections, rotations by multiples of 90 degrees) or value permutations of one coordinate (a bijection of finitely many values); the support of a scope (the sorted distinct tuples of every declared coordinate over the selected rows with multiplicities and smallest rows, NULL coordinates skipped or forbidden) with an FND-007 record of the scope; exhaustive exact verification that every generator preserves the multiset (mult(g(v)) = mult(v) for every support tuple, images computed exactly in int128 and found by binary search) with per-generator evidence (rows, tuples, violating rows, missing orbit members, broken multiplicities, the smallest violating row, its tuple, kind and image multiplicity); a certificate of FND-007 kind snapshot only when every generator holds, its freshness by FND-007's exact record comparison; orbit summaries under a fresh certificate (union-find, the smallest tuple as representative, sizes and multiplicities); polynomial SUM and COUNT queries proven invariant as laws (p o g = p by the exact expansion of p(A x + b); equal substitutions for every moved value) and evaluated once per orbit, non-invariant queries refused (DOMAIN_ERROR with the generator), and the exhaustive evaluation over every selected row as the check; budgets checked before allocation.

GPU: CUDA kernel math_sym_gpu.cu (tr_mgpu_sym_verify): the sorted support, multiplicities, first rows and the generators copied to the device; work items (generator, tuple slice) map each tuple exactly (tr_sym_image), find its image by binary search of the sorted tuples (exact comparisons, no hashing), compare multiplicities (tr_sym_check_tuple), add one atomic per counter and take the smallest violating row by an atomic minimum; the host finishes the evidence (tr_sym_evidence_finish) and tr_sym_adopt checks it before certifying; equal to the CPU bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_sym.h`
- `torbles/math_sym.c`
- `torbles/math_sym_gpu.cu`
- `torbles/math_sym.py`
- `tests/test_math_sym.py`
- `bench/benchmark_math_sym.py`

Acceptance tests:

- Broken multiplicity or missing orbit members invalidate the symmetry certificate.
- Orbit-reduced answers match exhaustive original-domain evaluation.

Benchmark cases:

- Use polynomial translations, declared 2D rotations and feature permutations on bases of 3/6/15/32 states; compare direct transformed-row evaluation with basis actions.
- Vary transformation requests 1/100/10,000 and certified orbit sizes 2/4/8/16; include one broken-multiplicity example that must invalidate orbit reuse.

Additional limits:

- Invariance under the reduction is a sufficient condition (polynomial identity under each generator): an invariant predicate of a non-invariant polynomial (x^2 - y^2 = 0 under a quarter turn, c = 3 under a value permutation fixing 3) is refused.
- Orbits are those of the generated group on the observed support; generators must be integral, invertible maps or value permutations (no interval or non-integral maps); NULL coordinates are skipped or forbidden, not a category of their own.
- The support sort, the orbit union-find, the invariance proofs and the queries on representatives run on the CPU, single-threaded; only the verification has a CUDA kernel.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/SYM-002/tests_test_math_sym.json](results/math/SYM-002/tests_test_math_sym.json)
- tests: [results/math/SYM-002/sanitized_test_math_sym.json](results/math/SYM-002/sanitized_test_math_sym.json)
- benchmarks: [results/math/SYM-002/benchmark_cpu_math_sym_vm.json](results/math/SYM-002/benchmark_cpu_math_sym_vm.json)

## DYN: Repeated dynamics

Finite closed observables and repeated operator application.

<a id="dyn-001"></a>

### DYN-001: Finite closed-observable dynamics

Status: **in_progress**. Delivery wave: **W4**.

Represent declared dynamics as a matrix action on a basis proven closed under composition.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SYM-001](#sym-001)

Native CPU: Native C (math_dyn.c): declared dynamics x' = F(x) on the n declared coordinates of a SYM-001 geometry in logical units: exact affine maps (SYM-001 maps: translations, scalings, reflections, permutations, affine maps, rotations by exact rational points of the unit circle or multiples of 90 degrees, compositions; interval and value maps refused) and polynomial maps (per output coordinate a sum of exact int64-rational terms in value units of total degree <= 8, optionally polynomial of degree <= 8 in one declared parameter, converted exactly to logical units with one int128 denominator per coordinate; categorical coordinates neither read nor written); the transfer operator K on a SYM-001 monomial basis with phi(F(x)) = K phi(x), hence moments(F(rows)) = K moments(rows): SYM-001's tr_sym_action_make for affine maps, an exact int128 expansion of every prod_i F_i(x)^a_i for polynomial ones, closure decided exactly and completely with the witness (basis term, missing monomial) of the first non-closed term, canonical int128 numerators over one denominator; truncation only under an explicit contract (a declared box bounding every coordinate the basis' evolution reads: the kept terms, the certified bound eps_a = sum of |c| prod max(|lo|, |hi|)^b over the dropped terms as exact FND-009 rationals, F-invariance of the box decided by exact rational interval arithmetic), labelled approximate (UNC-002's label value 5), refused by tr_dyn_authorize_exact (FND-007 reason approximate) and by every exact use; evolution of retained SYM-001 states (FND-004 banks, TENSOR-001 moment tensors; the operator's scope, basis and exact geometry) for t steps without touching rows: integer numerators N^t m_0 over D^t by stepping, exact repeated squaring or modular residues with Chinese remaindering within the proven bound ||N||_inf^t max|m_0| and a check prime (every method the same integers, AUTO by estimated costs: each method's structural work, sparsity included, times unit costs fitted to timed runs), truncated evolutions with the certified componentwise bound E_t = |K^| E_(t-1) + W eps under checked premises (unweighted state, the state's scope with its selected extremes inside the box, F-invariance for more than one step); the explicit exact row evolution of every selected complete row as the reference (one common denominator per step); one exact 512-bit step K m_g per group (SYM-001's evaluator); budgets (variables, degree, support, tensor = (p + 1) nb^2, iterations = steps, output, states, bytes from proven bit bounds) checked before allocation.

GPU: CUDA kernel math_dyn_gpu.cu (tr_mgpu_dyn_apply): the operators' int128 numerators and every group's int128 moments copied once, then one work item per (operator, group, basis row) running tr_dyn_apply_entry (SYM-001's 512-bit row evaluator tr_sym_row_exact), one output slot per entry; equal to tr_dyn_apply_view_eval bit for bit. Multi-step evolutions on the device use DYN-002's modular kernels.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dyn.h`
- `torbles/math_dyn.c`
- `torbles/math_dyn_gpu.cu`
- `torbles/math_dyn.py`
- `tests/test_math_dyn.py`
- `bench/benchmark_math_dyn.py`

Acceptance tests:

- Repeated affine transforms of polynomial moments agree with explicit row evolution.
- Non-closure is detected; truncation requires an explicit approximation/error contract.

Benchmark cases:

- Apply declared affine or closed-observable dynamics for 0/1/10/1,000 steps on 2/4/8-dimensional states; report operator construction and powers separately.
- Compare repeated row evolution with retained operator powers; include nonclosed maps, coefficient growth and numerical stability/budget refusals.

Additional limits:

- Exact transfer operators are int128 numerators over an int128 denominator (SYM-001's limit); a large expansion coefficient is refused (OverflowError), not widened.
- The truncation bound is a worst case over the declared box (often far above the actual error), needs an unweighted state and certifies one step unless the box is proven F-invariant; parametric maps cannot be truncated.
- The native row evolution of a nonlinear map shares one denominator Q_(t+1) = L Q_t^d, so its proven bound grows doubly exponentially and long nonlinear row evolutions are refused; the closure expansion, the truncation bounds and every big-integer product run on the CPU, single-threaded.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/DYN-001/tests_test_math_dyn.json](results/math/DYN-001/tests_test_math_dyn.json)
- tests: [results/math/DYN-001/sanitized_test_math_dyn.json](results/math/DYN-001/sanitized_test_math_dyn.json)
- benchmarks: [results/math/DYN-001/benchmark_cpu_math_dyn_vm.json](results/math/DYN-001/benchmark_cpu_math_dyn_vm.json)

<a id="dyn-002"></a>

### DYN-002: Repeated operator powers and semigroups

Status: **in_progress**. Delivery wave: **W4**.

Evaluate many steps, selected recurrences or parameterized evolution from retained finite operators.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DYN-001](#dyn-001), [LIN-002](#lin-002)

Native CPU: Native C (math_dyn.c): exact powers K^n of transfer operators as integer numerators N^n over D^n (K^0 = I) by exact repeated squaring in big integers (sign-magnitude limb vectors with positive and negative accumulators, every capacity from the proven bound ||N^n||_max <= ||N||_inf^n) or by residues modulo LIN-002's primes in [2^61, 2^62) with Garner's Chinese remaindering within that bound and the check of every entry modulo a further prime (device residues adopted only after the check), AUTO by estimated costs (structural work, sparsity included, times fitted unit costs); composition and exact equality by cross products; semigroup identities K^0 = I, K^1 = K, K^m K^n = K^(m+n) verified exactly; the exact characteristic polynomial from chi_N modulo primes (Hessenberg reduction and its recurrence) reconstructed within the Hadamard bound C(nb, k) ||N||_inf^k and checked; recurrences of s_t = c^T K^t v (c int64, v a state group's moments or int128 values): the characteristic recurrence, or the minimal one by Berlekamp-Massey over FND-009 rationals on 2 nb terms (integral by Gauss, checked, verified on every computed term), any term s_T by square and multiply of x^T modulo the recurrence polynomial in Z[x]; parameterized evolution: convex mixtures sum w_j K_j (exact weights >= 0 summing to 1: the expected moments of random map choices), operators polynomial in a declared parameter evolved as polynomials in it (stepping) and substituted at exact rationals; floating powers K^n and K^n m_g in outward-rounded binary64 interval arithmetic from tight enclosures, every value with a rigorous bound, refused as unstable (NOT_CONVERGED, with the exponent certified and the error) when the bound exceeds the declared absolute or relative tolerance or leaves the double range; budgets (iterations = the step count, output, bytes from the proven coefficient growth, tensor, support) checked before allocation.

GPU: CUDA kernels math_dyn_gpu.cu (tr_mgpu_dyn_powmod, tr_mgpu_dyn_evolvemod): the operators' numerators reduced modulo every prime on the device (one work item per (operator, prime, entry)), the binary exponentiation driven by the host with one launch per product (one work item per (operator, prime, entry) running tr_dyn_mulmod_entry), then the moments of every group reduced and multiplied (one work item per (operator, prime, group, row) running tr_dyn_applymod_entry); every output slot has one writer, so the residues equal tr_dyn_powmod_view_eval / tr_dyn_evolvemod_view_eval bit for bit; the host reconstructs and checks them (tr_dyn_power_adopt, tr_dyn_evolution_adopt).

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dyn.h`
- `torbles/math_dyn.c`
- `torbles/math_dyn_gpu.cu`
- `torbles/math_dyn.py`
- `tests/test_math_dyn.py`
- `bench/benchmark_math_dyn.py`

Acceptance tests:

- Zero/one/many-step composition and semigroup identities match small direct calculations.
- Bound coefficient growth, iteration count and output size; unstable numerical powers are diagnosed.

Benchmark cases:

- Apply declared affine or closed-observable dynamics for 0/1/10/1,000 steps on 2/4/8-dimensional states; report operator construction and powers separately.
- Compare repeated row evolution with retained operator powers; include nonclosed maps, coefficient growth and numerical stability/budget refusals.

Additional limits:

- Big-integer products are schoolbook (no Karatsuba or FFT) and the a priori bound ||N||_inf^n can be far above the actual sizes (cancellations are not foreseen), so some exact powers are refused by the bytes budget that would fit; the CRT, the characteristic polynomial, Berlekamp-Massey and the polynomial reduction run on the CPU, single-threaded; Garner's O(k^2) reconstruction makes CPU modular powers slower than squaring except for dense operators at large exponents with few groups, and AUTO's unit costs are estimates fitted on one machine.
- Floating powers are binary64 interval enclosures only (no multiprecision): the wrapping effect makes the bounds of non-normal or expanding operators grow, which the contract reports as unstable rather than returning them.
- Parametric evolutions step (no squaring of polynomial matrices) and have no CUDA path; mixtures need convex rational weights over one basis; truncated operators have no powers, recurrences or device path.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/DYN-002/tests_test_math_dyn.json](results/math/DYN-002/tests_test_math_dyn.json)
- tests: [results/math/DYN-002/sanitized_test_math_dyn.json](results/math/DYN-002/sanitized_test_math_dyn.json)
- benchmarks: [results/math/DYN-002/benchmark_cpu_math_dyn_vm.json](results/math/DYN-002/benchmark_cpu_math_dyn_vm.json)

## BOOL: Boolean calculus

Predicate cofactors, signed/GF(2) finite differences and influence.

<a id="bool-001"></a>

### BOOL-001: Predicate cofactors and finite differences

Status: **in_progress**. Delivery wave: **W2**.

Native Boolean cofactors produce XOR sensitivity or signed changes with an explicit encoded-variable or logical-feature domain.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012)

Native CPU: Native C (math_bool.c): predicate DAGs over k conditions (AND, OR, NOT, XOR, constants, shared subexpressions) on an explicit domain (the 2^k assignments, or native row predicates over FND-004 scope columns as bitmaps); cofactors, XOR derivatives, signed differences and mixed differences with their measures, under SQL three-valued logic (UNKNOWN rows counted separately, never as false).

GPU: CUDA kernel math_bool_gpu.cu: word-parallel evaluation of the DAG and its cofactors over row bitmaps with a validity word for UNKNOWN, popcount accumulation, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing Boolean predicate programs and JoinPoly factorization.
- [torbles/alg.c](torbles/alg.c): Existing mask execution; a first-class cofactor/difference API is planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_bool.h`
- `torbles/math_bool.c`
- `torbles/math_bool_gpu.cu`
- `torbles/math_bool.py`
- `tests/test_math_bool.py`
- `bench/benchmark_math_bool.py`

Acceptance tests:

- Compare cofactors and first/mixed differences with complete small truth tables.
- Distinguish XOR derivative from signed difference and preserve SQL unknown/NULL semantics.

Benchmark cases:

- Use predicate DAGs with 4/8/16 conditions and varying shared subexpressions; compare independent cofactor evaluation with shared-mask execution.
- Measure signed and XOR first/mixed differences under uniform assignment and empirical row measures; include sparse support and NULL-conditioned scopes.

Additional limits:

- One measure over all selected rows (no per-group measures); conditions on exact-valued columns only (no string or date predicates); constants fit int64 rationals.
- Difference order at most 20; the assignment domain at most 2^20 points with the default budget.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/BOOL-001/tests_test_math_bool.json](results/math/BOOL-001/tests_test_math_bool.json)
- tests: [results/math/BOOL-001/sanitized_test_math_bool.json](results/math/BOOL-001/sanitized_test_math_bool.json)
- benchmarks: [results/math/BOOL-001/benchmark_cpu_math_bool_vm.json](results/math/BOOL-001/benchmark_cpu_math_bool_vm.json)

<a id="bool-002"></a>

### BOOL-002: Influence and predicate interactions

Status: **in_progress**. Delivery wave: **W2**.

Count dependency masks and bounded interactions under declared empirical or full-Boolean measures.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [BOOL-001](#bool-001), [DIST-001](#dist-001)

Native CPU: Native C (math_influence.c): influence of each condition of a BOOL-001 predicate (the measure where its XOR derivative is 1), total influence and bounded mixed interactions, under the uniform assignment measure and the empirical row measure as separate results, labelled as coordinate sensitivity of the predicate, not a causal effect.

GPU: CUDA kernel math_influence_gpu.cu: word-parallel cofactor evaluation for every condition over row bitmaps with popcount accumulation, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_influence.h`
- `torbles/math_influence.c`
- `torbles/math_influence_gpu.cu`
- `torbles/math_influence.py`
- `tests/test_math_influence.py`
- `bench/benchmark_math_influence.py`

Acceptance tests:

- Match exhaustive influence calculations under uniform and empirical weights.
- Do not label coordinate sensitivity a causal intervention or confuse stored rows with all assignments.

Benchmark cases:

- Use predicate DAGs with 4/8/16 conditions and varying shared subexpressions; compare independent cofactor evaluation with shared-mask execution.
- Measure signed and XOR first/mixed differences under uniform assignment and empirical row measures; include sparse support and NULL-conditioned scopes.

Additional limits:

- No per-group or weighted measures.
- With more than 20 conditions the uniform measure is refused by the default budget (measures=('empirical',) avoids it).

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/BOOL-002/tests_test_math_influence.json](results/math/BOOL-002/tests_test_math_influence.json)
- tests: [results/math/BOOL-002/sanitized_test_math_influence.json](results/math/BOOL-002/sanitized_test_math_influence.json)
- benchmarks: [results/math/BOOL-002/benchmark_cpu_math_influence_vm.json](results/math/BOOL-002/benchmark_cpu_math_influence_vm.json)

## LATTICE: Subset-lattice algebra

Zeta/Mobius transforms, exclusive regions and inclusion-exclusion.

<a id="lattice-001"></a>

### LATTICE-001: Bounded Boolean histogram state

Status: **in_progress**. Delivery wave: **W2**.

Group shared predicates into a bounded k-condition occupancy histogram with exact bag counts.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [BOOL-001](#bool-001)

Native CPU: Native C (math_lattice.c): the bounded k-condition occupancy histogram of BOOL-001 conditions over the selected rows of a scope: exact bag counts of every one of the 2^k cells, condition order kept, UNKNOWN outcomes in a declared three-valued encoding (never silently false), k and 2^k states budgeted.

GPU: CUDA kernel math_lattice_gpu.cu: the histogram scan (cell id per row from the condition bits, atomic or per-thread local counts), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_lattice.h`
- `torbles/math_lattice.c`
- `torbles/math_lattice_gpu.cu`
- `torbles/math_lattice.py`
- `tests/test_math_lattice.py`
- `bench/benchmark_math_lattice.py`

Acceptance tests:

- All 2^k small-domain cells match enumerated predicate outcomes.
- Enforce condition/state budgets and preserve condition order/NULL conventions.

Benchmark cases:

- Vary Boolean condition count k=3/6/10/12, predicate overlap and occupancy sparsity; record 2^k state bytes and native histogram scans.
- Compare 1/100/all intersection/coalition requests with one histogram plus zeta/Mobius transforms, retaining cases where full-domain construction loses.

Additional limits:

- One histogram per domain (no per-group histograms); at most 30 axes; on the device, histograms of more than 64 cells use one atomic per row.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/LATTICE-001/tests_test_math_lattice.json](results/math/LATTICE-001/tests_test_math_lattice.json)
- tests: [results/math/LATTICE-001/sanitized_test_math_lattice.json](results/math/LATTICE-001/sanitized_test_math_lattice.json)
- benchmarks: [results/math/LATTICE-001/benchmark_cpu_math_lattice_vm.json](results/math/LATTICE-001/benchmark_cpu_math_lattice_vm.json)

<a id="lattice-002"></a>

### LATTICE-002: Zeta, Mobius and inclusion-exclusion transforms

Status: **in_progress**. Delivery wave: **W2**.

Transform occupancy counts into all intersections, exclusive regions and bounded coalition interactions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LATTICE-001](#lattice-001)

Native CPU: Native C (math_lattice.c): zeta (all intersections, subset and superset variants) and Moebius transforms of the occupancy histogram, inclusion-exclusion counts and bounded coalition interactions, exact with checked 128-bit intermediates or FND-009 big integers, never wrapping.

GPU: CUDA kernel math_lattice_gpu.cu: the k butterfly passes of the fast zeta/Moebius transforms, one launch per pass, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_lattice.h`
- `torbles/math_lattice.c`
- `torbles/math_lattice_gpu.cu`
- `torbles/math_lattice.py`
- `tests/test_math_lattice.py`
- `bench/benchmark_math_lattice.py`

Acceptance tests:

- Zeta/Mobius round trips and inclusion-exclusion match direct set counts.
- Signed intermediate states use a compatible carrier and cannot silently wrap.

Benchmark cases:

- Vary Boolean condition count k=3/6/10/12, predicate overlap and occupancy sparsity; record 2^k state bytes and native histogram scans.
- Compare 1/100/all intersection/coalition requests with one histogram plus zeta/Mobius transforms, retaining cases where full-domain construction loses.

Additional limits:

- Direct coefficients, inclusion-exclusion counts and the big-integer carrier are CPU only.
- Measured: answering 1 or 100 requests directly beats building the full transform at k = 16; the transform wins for all 2^k requests.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/LATTICE-002/tests_test_math_lattice.json](results/math/LATTICE-002/tests_test_math_lattice.json)
- tests: [results/math/LATTICE-002/sanitized_test_math_lattice.json](results/math/LATTICE-002/sanitized_test_math_lattice.json)
- benchmarks: [results/math/LATTICE-002/benchmark_cpu_math_lattice_vm.json](results/math/LATTICE-002/benchmark_cpu_math_lattice_vm.json)

## QUANT: Quantified queries

Existential/universal predicates and category coverage.

<a id="quant-001"></a>

### QUANT-001: Existential and universal group predicates

Status: **in_progress**. Delivery wave: **W2**.

Carry selected-row, qualifying-row and seen/unknown states to implement some/all/none conditions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: Native C (math_quant.c): per group of an FND-004 scope (also through a dimension lookup), the child row count and the rows where a BOOL-001 predicate is TRUE, FALSE or UNKNOWN, with declared group presence (child-row groups, every group id, or a parent scope); some / all / none / not-all / EXISTS / bool_or / bool_and answers in SQL three-valued logic with vacuous truth, empty, absent and all-NULL groups specified.

GPU: CUDA kernel math_quant_gpu.cu: the per-group state scan (64-row words with popcounts when a word lies in one group, atomic adds otherwise), checked for consistency before adoption, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/sqlexec.py](torbles/sqlexec.py): Existing SQL EXISTS/aggregate execution paths; typed native quantified states remain planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_quant.h`
- `torbles/math_quant.c`
- `torbles/math_quant_gpu.cu`
- `torbles/math_quant.py`
- `tests/test_math_quant.py`
- `bench/benchmark_math_quant.py`

Acceptance tests:

- Compare some/all/none with explicit child rows for empty, all-NULL and mixed groups.
- Specify vacuous truth and SQL unknown conventions; multiplicity does not erase coverage.

Benchmark cases:

- Use 1/100/10,000 groups with 1/8/64 children and 1%/50%/100% qualifying rows; include empty, duplicate and all-NULL child sets.
- Vary required category sets of 4/16/64 elements; compare count/mask coverage states against materialized matching children.

Additional limits:

- Parent and child groupings must align (same key codecs and radices; checked from the key specs in Python).
- A nullable parent key matches the child NULL group (IS NOT DISTINCT FROM semantics), declared and tested.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/QUANT-001/tests_test_math_quant.json](results/math/QUANT-001/tests_test_math_quant.json)
- tests: [results/math/QUANT-001/sanitized_test_math_quant.json](results/math/QUANT-001/sanitized_test_math_quant.json)
- benchmarks: [results/math/QUANT-001/benchmark_cpu_math_quant_vm.json](results/math/QUANT-001/benchmark_cpu_math_quant_vm.json)

<a id="quant-002"></a>

### QUANT-002: Relational division and required-category coverage

Status: **in_progress**. Delivery wave: **W2**.

Retain exact required/present category sets or bounded masks per group.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [QUANT-001](#quant-001), [FIELD-001](#field-001)

Native CPU: Native C (math_quant.c): per group, the exact set of present categories as packed bit masks (FIELD-001 format) or sorted sets, chosen within the budget; division, exact match, overlap and disjointness against a required set resolved through the category dictionary, missing and extra categories per group; NULL categories never satisfy a requirement, duplicates count once; the state identity records the required set and the dictionary.

GPU: CUDA kernels math_quant_gpu.cu: category masks built with atomic or, coverage per group by AND-NOT and popcount, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_quant.h`
- `torbles/math_quant.c`
- `torbles/math_quant_gpu.cu`
- `torbles/math_quant.py`
- `tests/test_math_quant.py`
- `bench/benchmark_math_quant.py`

Acceptance tests:

- Missing, extra, duplicated and NULL categories match set-based reference results.
- Changing the required set or category dictionary changes state identity.

Benchmark cases:

- Use 1/100/10,000 groups with 1/8/64 children and 1%/50%/100% qualifying rows; include empty, duplicate and all-NULL child sets.
- Vary required category sets of 4/16/64 elements; compare count/mask coverage states against materialized matching children.

Additional limits:

- A category is one group id of a single-key scope; multi-key, float, BLOB, date or timestamp categories are refused.
- The sorted-set representation and merges are CPU only.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/QUANT-002/tests_test_math_quant.json](results/math/QUANT-002/tests_test_math_quant.json)
- tests: [results/math/QUANT-002/sanitized_test_math_quant.json](results/math/QUANT-002/sanitized_test_math_quant.json)
- benchmarks: [results/math/QUANT-002/benchmark_cpu_math_quant_vm.json](results/math/QUANT-002/benchmark_cpu_math_quant_vm.json)

## COUNT: Constraint counting

Observed-domain and symbolic-domain model counts and witnesses.

<a id="count-001"></a>

### COUNT-001: Observed-domain constraint counts and witnesses

Status: **in_progress**. Delivery wave: **W3**.

Return satisfying stored rows/counts and counterexample witnesses with explicit snapshot scope.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [BOOL-001](#bool-001), [FND-007](#fnd-007)

Native CPU: Native C (math_count.c): constraint sets over one shared BOOL-001 predicate DAG (holds, implies A -> B, equivalence A <-> B, exclusion NOT (A AND B); per-constraint cones), observed domains (BOOL-001 condition bitmaps over the selected rows of an FND-004 scope, with an FND-007 record of the scope's identity, revision, exact selection, column encodings and NULL domains) and the 2^k assignment universe; exact satisfied / violated / UNKNOWN counts under SQL three-valued logic, every satisfying point (refused past the output budget with the exact need), the smallest violating points as witnesses, and a certificate (a snapshot of exactly the covered rows, or a law by enumeration) with the coverage of the support's assignments (complete rows, distinct observed assignments against 2^s), FND-007 freshness with exact staleness reasons, and an authorization that refuses universe statements while assignments were unobserved.

GPU: CUDA kernels math_count_gpu.cu: the counts and the smallest violating point of every constraint (work items (constraint, word slice) evaluate the constraint's cone per 64-row word with bitwise three-valued logic, count with popcounts, add with one atomic per counter and an atomic minimum), and the bounded extraction of the satisfying and smallest violating points (per (constraint, chunk) counts in their own slots, their exclusive prefix on the host checked against the counts, then rank-addressed writes: one writer per output slot); the readback is adopted natively after consistency checks; equal to the CPU bit for bit.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/alg.c](torbles/alg.c): Existing row masks/counts can evaluate stored-row predicates.
- [torbles/relationships.c](torbles/relationships.c): Existing violation witnesses and exhaustive relation checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_count.h`
- `torbles/math_count.c`
- `torbles/math_count_gpu.cu`
- `torbles/math_count.py`
- `tests/test_math_count.py`
- `bench/benchmark_math_count.py`

Acceptance tests:

- Implication/equivalence violations match exhaustive mask evaluation.
- Certificates report rows actually covered and never imply unobserved assignments were checked.

Benchmark cases:

- Count and extract witnesses on 1k/100k/1M observed rows for structured and irregular predicates; distinguish the observed snapshot from the full assignment universe.
- For symbolic counts use tiny brute-force-checked formulas and bounded larger circuits with variable order/width controls; report compilation size and budget refusals.

Additional limits:

- Counts over the observed selected rows, or by enumeration of at most 2^20 assignments with the default budget; symbolic (decision-diagram, weighted) counting is COUNT-002's.
- One count over all selected rows (no per-group counts, no weights); conditions on exact-valued columns only (BOOL-001).
- The certificate is COUNT-001's own type built on FND-007's record and staleness reasons (FND-007 has no claim kind for counts); it is not serialized; its coverage is computed on the host in both paths.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/COUNT-001/tests_test_math_count.json](results/math/COUNT-001/tests_test_math_count.json)
- tests: [results/math/COUNT-001/sanitized_test_math_count.json](results/math/COUNT-001/sanitized_test_math_count.json)
- benchmarks: [results/math/COUNT-001/benchmark_cpu_math_count_vm.json](results/math/COUNT-001/benchmark_cpu_math_count_vm.json)

<a id="count-002"></a>

### COUNT-002: Symbolic full-domain algebraic model counting

Status: **in_progress**. Delivery wave: **W5**.

Add a bounded circuit/decision-diagram/variable-elimination backend for counts and witnesses over declared assignment domains.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [COUNT-001](#count-001)

Native CPU: Native C (math_symcount.c, the exact L-limb arithmetic and the count functions inline in math_symcount.h): a decision-diagram builder (hash-consed reduced ordered BDD nodes over a declared order, iterative Shannon apply for AND / OR / XOR with a lossy computed table, node / apply-step / byte budgets checked at every growth and refused at the first node past the limit) and its immutable diagrams renumbered from the deepest level; BOOL-001 programs over k <= 64 variables compiled two-valued (one root) or three-valued (the TRUE and FALSE indicator functions under Kleene's connectives) under the natural, a given or the DFS first-appearance order; exact model counts and weighted model counts over the whole 2^k assignment universe (rational literal weights scaled to integers over per-variable common denominators, counts in L 64-bit limbs with the bound prod (a + b) < 2^(64 L) proven before counting, reduced to canonical rationals with FND-009 big integers), TRUE + FALSE + UNKNOWN = TOTAL; the first m TRUE or FALSE assignments in lexicographic order of the levels; a brute-force enumeration of the 2^k assignments (64 per word with BOOL-001's evaluator) for verification; a breadth-first bounded Shannon expansion with possible-value sets whose open branches give certified lower and upper bounds per class (exact when none is open); finite-domain factor systems (domains <= 2^16, scopes <= 31 variables, nonnegative rational tables scaled per factor, zeros as hard constraints) counted by variable elimination under a given, min-fill or min-degree order with the width (the largest joint table) checked first and its exact need reported, model counting by indicators, witnesses decoded without backtracking in lexicographic order of the reverse elimination order, and a factor brute force. Every result is labelled as a count of the declared assignment universe, distinct from COUNT-001's observed rows.

GPU: CUDA kernels math_symcount_gpu.cu: tr_mgpu_symcount_levels counts a compiled diagram with one launch per level from the deepest, one work item per node writing its exact L-limb weighted count into its own slot from its children's (the shared tr_symc_node_count, the skipped levels' weight sums multiplied in), then one work item per root; tr_mgpu_symcount_brute enumerates the 2^k assignments in work items over slices of 64-point words (BOOL-001's word evaluator, popcounts of the TRUE / FALSE / UNKNOWN words, the weights of the TRUE and FALSE assignments as H(word) times the low variables' weights) with local sums added by one 64-bit atomic per count and a multi-word atomic add per L-limb sum (exact modulo 2^(64 L), exact by the host's bound). The readbacks are adopted natively after consistency checks (tr_symc_dd_adopt, tr_symc_brute_adopt); raw words equal the CPU's bit for bit.

Bounds:

- Compilation: variables = k (programs), circuit = every node the builder holds (a refusal names the first node past the limit), iterations = Shannon apply steps, bytes = the node, unique-table, computed-table and stack arrays at every growth.
- Counts: circuit = diagram nodes, output = roots, bytes = nodes x L x 8 (CUDA: every device copy, the node values and the widest level's scratch); witnesses: output = m.
- Brute force: variables = k, support = 2^k, circuit = program nodes, bytes (CUDA: per work item node words and sums); bounded search: iterations = max_evaluations (exact or certified bounds, never refused for a larger space), bytes = the branch queue.
- Factors: variables = n, support = entries, tensor = the largest table; elimination: tensor = the width (exact need), iterations = the products, bytes = every message at L limbs per entry, output = 2 + witnesses; factor brute force: support = prod d_i.

Implementation targets (proposed until the capability is validated):

- `torbles/math_symcount.h`
- `torbles/math_symcount.c`
- `torbles/math_symcount_gpu.cu`
- `torbles/math_symcount.py`
- `tests/test_math_symcount.py`
- `bench/benchmark_math_symcount.py`

Acceptance tests:

- Compare all models of tiny formulas with brute force, including weighted/dependent factors.
- Refuse or return explicit incompleteness when circuit width/search budgets are exhausted.

Benchmark cases:

- Count and extract witnesses on 1k/100k/1M observed rows for structured and irregular predicates; distinguish the observed snapshot from the full assignment universe.
- For symbolic counts use tiny brute-force-checked formulas and bounded larger circuits with variable order/width controls; report compilation size and budget refusals.

Additional limits:

- Programs over at most 64 variables (63 for the brute force); no complement edges, dynamic reordering or garbage collection inside one compilation: the node budget counts every node held, and the order is given or the DFS heuristic.
- Weights are nonnegative exact rationals with int64 numerators and denominators (scaled per variable or factor below 2^126); a formula with dependent weights is counted as a factor system (CNF clauses as 0/1 tables), not as a weighted diagram.
- The bounded search is breadth first in a fixed order without caching of equal residual formulas: its certified bounds can be loose; variable elimination, search and witness decoding run single-threaded on the CPU (the level count and the brute force have kernels).
- Counts are of the declared universe only; statements about observed rows remain COUNT-001's certificates (no FND-007 certificate is emitted for a symbolic count).

Validation receipts:

Measured runtime: `cacb92841d6cae9a8a136a680c29f3cecf01b657`.

- tests: [results/math/COUNT-002/tests_test_math_symcount.json](results/math/COUNT-002/tests_test_math_symcount.json)
- tests: [results/math/COUNT-002/sanitized_test_math_symcount.json](results/math/COUNT-002/sanitized_test_math_symcount.json)
- benchmarks: [results/math/COUNT-002/benchmark_cpu_math_symcount_vm.json](results/math/COUNT-002/benchmark_cpu_math_symcount_vm.json)

## FIELD: Finite-field algebra

GF(2) dependencies, parity, ranks, codes and syndromes.

<a id="field-001"></a>

### FIELD-001: GF(2) vector and matrix primitives

Status: **in_progress**. Delivery wave: **W2**.

Native parity, rank, elimination, nullspaces and syndromes over packed bits with field laws separate from integer arithmetic.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009)

Native CPU: Native C (math_field.c): packed GF(2) vectors and matrices (parity, products, rank, reduced row echelon form, nullspace bases, solving with unique / none / affine-family answers, syndromes), kept separate from integer popcounts; parity dependencies among Boolean columns or bit planes of a scope found from a sample and verified exhaustively, with one-row near misses reported with their witness row.

GPU: CUDA kernels math_field_gpu.cu: batched syndromes and matrix-vector products over many vectors, and the exhaustive verification of a parity relation over all rows (violations and the smallest violating row), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/alg.c](torbles/alg.c): Existing XOR torus integrity faces; not a general GF(2) matrix API.
- [tests/test_fixes.py](tests/test_fixes.py): Existing fault/integrity checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_field.h`
- `torbles/math_field.c`
- `torbles/math_field_gpu.cu`
- `torbles/math_field.py`
- `tests/test_math_field.py`
- `bench/benchmark_math_field.py`

Acceptance tests:

- Compare elimination/rank/nullspace with exhaustive small binary systems.
- Check dimensional/encoding boundaries and distinguish parity from integer popcount sums.

Benchmark cases:

- Benchmark packed GF(2) matrices of 32/128/512 rows and columns at multiple ranks/densities; compare elimination, rank and syndrome operations separately.
- Discover planted parity dependencies over 1k/100k/1M rows and one-row near misses; record candidate generation, exact verification and substitution time.

Additional limits:

- Dense Gauss-Jordan elimination only; rank, elimination and solving have no CUDA path.
- Discovery handles at most 64 columns (63 variables and the constant); with NULLs, relations are exact on the complete rows only (holds is false).

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/FIELD-001/tests_test_math_field.json](results/math/FIELD-001/tests_test_math_field.json)
- tests: [results/math/FIELD-001/sanitized_test_math_field.json](results/math/FIELD-001/sanitized_test_math_field.json)
- benchmarks: [results/math/FIELD-001/benchmark_cpu_math_field_vm.json](results/math/FIELD-001/benchmark_cpu_math_field_vm.json)

<a id="field-002"></a>

### FIELD-002: Parity relationship discovery and substitution

Status: **in_progress**. Delivery wave: **W3**.

Discover bounded exact XOR dependencies, verify them over the scoped data, and substitute equivalent predicate circuits.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [FND-007](#fnd-007)

Native CPU: Native C (math_parity.c): FIELD-001's discovery (sampled proposals, exhaustive verification, one-row near misses rejected with their witness rows) with every found relation turned into an FND-007 PARITY claim (its scope columns, bits with plane and base, the constant, the NULL policy) and certified as a snapshot (identity, revision, exact selection, column encodings, NULL domains); substitution of predicate circuits: the maximal pure XOR forms meeting a fresh certified relation are rewritten by adding the relation over GF(2) (a column that is the XOR of others becomes that XOR and vice versa), the cheaper form chosen by the columns and planes scanned, refused with FND-007's exact staleness reason after codec, NULL-mask or revision changes; exhaustive verification of the substituted predicate against the original on every selected row.

GPU: CUDA kernel math_parity_gpu.cu: the verification of a substituted predicate against the original (one work item per 64-row word: both programs' value and valid bitmaps, their XOR, popcounts, one atomic per counter and an atomic minimum for the witness), equal to the CPU; the device discovery and certification use FIELD-001's tr_mgpu_field_discover and FND-007's tr_mgpu_cert_verify, adopted natively (tr_parity_adopt).

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_parity.h`
- `torbles/math_parity.c`
- `torbles/math_parity_gpu.cu`
- `torbles/math_parity.py`
- `tests/test_math_parity.py`
- `bench/benchmark_math_parity.py`

Acceptance tests:

- Recover planted parity relations and reject one-row near misses.
- Codec changes, NULL masks and revisions invalidate affected parity certificates.

Benchmark cases:

- Benchmark packed GF(2) matrices of 32/128/512 rows and columns at multiple ranks/densities; compare elimination, rank and syndrome operations separately.
- Discover planted parity dependencies over 1k/100k/1M rows and one-row near misses; record candidate generation, exact verification and substitution time.

Additional limits:

- Rewrites only XOR forms whose variables are disjoint, with one relation per call, greedily; the cost is the number of columns and planes scanned, not their widths.
- The rewrite itself runs on the CPU; discovery inherits FIELD-001's limits (at most 64 columns, listwise over the complete rows).
- Certificates record no dimension; a certificate that skipped NULL rows never authorizes a substitution.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/FIELD-002/tests_test_math_parity.json](results/math/FIELD-002/tests_test_math_parity.json)
- tests: [results/math/FIELD-002/sanitized_test_math_parity.json](results/math/FIELD-002/sanitized_test_math_parity.json)
- benchmarks: [results/math/FIELD-002/benchmark_cpu_math_parity_vm.json](results/math/FIELD-002/benchmark_cpu_math_parity_vm.json)

## GEOM: Algebraic geometry

Certified ideal reduction, elimination and generator selection.

<a id="geom-001"></a>

### GEOM-001: Certified sparse ideal reduction

Status: **in_progress**. Delivery wave: **W3**.

Verify f-g=sum(h_i*r_i) certificates over a declared carrier/domain and choose cheaper equivalent query representations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DISC-002](#disc-002), [FND-007](#fnd-007)

Native CPU: Native ideals, reductions, certificates, costs, row agreement, batched checks and derivatives over exact int64 rationals (an overflow is refused, never rounded): relations are up to 16 FND-007 certificates of one NULL policy (POLYNOMIAL claims, AFFINE / SUM / PRODUCT claims read as FND-007 reads them; laws, schema facts, tolerance and GF(p) claims refused) over at most 8 columns, recorded on one table (or array scope) through one column layout (checked on their FND-007 records), under LEX, DEGLEX or DEGREVLEX with a declared variable significance; the ideal is the reduced Groebner basis by Buchberger's algorithm (normal selection, product criterion, interreduction), each element with exact cofactors over the relations, or the monic relations themselves; reduction by the multivariate division algorithm gives f = g + sum h_i r_i with the remainder irreducible; its certificate P = D (f - g) with cofactors D h_i (D the lcm of every denominator) is derived by FND-007 (tr_cert_derive checks P - sum D h_i r_i = 0 in Z[x]); its kind is the weakest relation's and it authorizes only on the relations' own scope. Batched checks of certificates P = sum q_i G_i expand every product into a dense graded index with signed 128-bit sums under a proven bound (else undecided) and report the nonzero slots and the smallest nonzero monomial. Agreement evaluates the integer forms of f, g and D (f - g) exactly on every selected row (DISC-002's W-word evaluation; rows with a NULL column of f or g excluded): disagreements, the smallest disagreeing row and the largest bit lengths. Costs: terms, columns read, degree, naive multiplications, bound bits on the scope and measured bits; the cheaper form is chosen by a declared priority. Derivatives are formal, or total only through declared dependencies v = phi (refused undeclared, dependent or cyclic), and declarations are verified on every row. Budgets (iterations as S-pairs plus division steps, support, degree, output, states, bytes) are checked before allocation or as reached, refused never truncated.

GPU: tr_mgpu_geom_agree: work items (pair chunk, row slice) evaluate D (f - g), D_f f and D_g g exactly on each selected row of their slice with DISC-002's inline W-word evaluation, count locally and add one atomic per counter, the smallest disagreeing row by atomic minimum and the largest bit lengths by atomic maxima. tr_mgpu_geom_check: one work item per product term (every term of P and every pair of terms of each cofactor and premise) adds its signed 128-bit value into its certificate's dense slot with the two-word atomic (the host proved every partial sum within 2^127 - 1, so the sum does not depend on the order), then a scan counts the nonzero slots per certificate and keeps the smallest by atomic minimum; a rank outside its slots (a malformed view) is refused. Both give the CPU's results bit for bit; the host validates the views and budgets the device copies before launching. Completion and reduction run on the CPU. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_geom.h`
- `torbles/math_geom.c`
- `torbles/math_geom_gpu.cu`
- `torbles/math_geom.py`
- `tests/test_math_geom.py`
- `bench/benchmark_math_geom.py`

Acceptance tests:

- Certificate verification catches wrong coefficients, domains and dependencies.
- Equivalent candidates agree on certified rows; derivative operators must separately respect declared dependencies.

Benchmark cases:

- Use small relation components with 2/4/8 variables, degrees 1/2/3 and sparse/dense certificates; separate discovery, certificate checking and reduction.
- Compare direct expressions with alternative generator/elimination forms using actual resulting bit widths and scans; retain term-explosion and cost-regression cases.

Additional limits:

- Coefficients are int64 rationals: a reduction or certificate whose coefficients pass int64 is refused (no arbitrary-precision coefficients); a certificate past 4096 terms has no FND-007 claim.
- Completion is Buchberger's algorithm with the product criterion only (no F4/F5 or sugar strategy); lex bases can grow quickly and are refused by the budget.
- Relations: at most 16 certificates over at most 8 columns with one NULL policy (FND-007's derivation and claim limits); at most 128 basis elements.
- The CUDA kernels cover the row agreement and the batched certificate checks; completion, reduction and derivatives run on the CPU.
- Costs are bounds and widths measured on one scope, not a query planner's cost model.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `2e0e328cf05243328e3a4e64b529f54463387459`.

- tests: [results/math/GEOM-001/tests_test_math_geom.json](results/math/GEOM-001/tests_test_math_geom.json)
- tests: [results/math/GEOM-001/sanitized_test_math_geom.json](results/math/GEOM-001/sanitized_test_math_geom.json)
- benchmarks: [results/math/GEOM-001/benchmark_cpu_math_geom_vm.json](results/math/GEOM-001/benchmark_cpu_math_geom_vm.json)

<a id="geom-002"></a>

### GEOM-002: Bounded elimination and generator selection

Status: **in_progress**. Delivery wave: **W4**.

Find small generator sets or sparse local elimination forms without assuming a canonical normal form is cheapest.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GEOM-001](#geom-001), [FND-008](#fnd-008)

Native CPU: Native generator selection, reconstructions, local elimination forms, row checks and costs over an FND-004 scope and GEOM-001 relations (up to 16 FND-007 certificates over at most 8 variables). A generator set G has a reconstruction x = f_x(G) for every eliminated variable exactly when, under lex with the eliminated variables first, GEOM-001's reduced Groebner basis reduces x into Q[G] (the elimination property); f_x is that normal form and its FND-007 derived certificate D (x - f_x) = sum D h_k r_k is GEOM-001's reduction certificate. Two modes: one elimination under a given lex sequence (one solve), or a bounded search by increasing size over every subset of the relation variables in lexicographic order (variables in no relation always generators), one completion per set, ranked by a declared priority (total, scan, work, bits; ties to the earliest) among the sets of the minimum size; a set whose completion passes the degree or support budget, GEOM-001's per-solve limit or int64 is recorded as undecided (with its refusal) and the search goes on, minimality claimed only when every smaller set was decided. A reconstruction is kept only when its row check holds: on every selected row the target column and the reconstruction are evaluated exactly in W <= 8 64-bit words under a bound proven from the scope's extremes (DISC-002's evaluation), with SQL NULL propagation (null_lost / null_extra NULL-domain mismatches), value mismatches, the target column's range (below, above, both ends attained) and the smallest mismatching row of each kind. Costs are measured on the scope: widths from max - min of the present values plus a validity plane, scan = ceil(N / 64) x widths read, work = N x (terms + multiplications) x words from the actual bit length of every evaluated row, a generator set's storage plus its reconstructions, each column's read path (reconstruct only when strictly cheaper than its stored scan). Local elimination forms of a query q: q as written, every substitution of a nonempty set of the eliminated variables q reads (GEOM-001 division by their reconstruction certificates, certified by FND-007 at depth 2) and optionally the normal form under the selection's elimination ideal; each checked against q on every row and costed on its own integer form, the cheapest admissible chosen and ties kept by the direct form, so a term explosion or cost regression keeps q. Budgets: iterations = solves (refused at the limit + 1), degree and support bound every completion and reduction (an order selection refused, a form rejected with its refusal), variables, output and bytes checked before allocation.

GPU: tr_mgpu_elim_check: work items (a chunk of 8 pairs, a row slice of at least 64 selection words) run the inline tr_elim_pair_row shared with the CPU on each selected row of their slice for every pair of their chunk: the target's and the form's NULL domains, their exact W-word integer forms under the bound the host proved, the form against the target column's range; they count locally and add one atomic per nonzero counter, the smallest mismatching rows by atomic minimum and the largest bit lengths by atomic maximum, so the results (and the first mismatching row) do not depend on the order threads run in and equal tr_elim_check_view bit for bit. The host validates the pair view (its structure and the view's extremes within the magnitudes the bounds were proven for) and budgets the device copies before launching. Selection, completion, reduction and certification run on the CPU. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_elim.h`
- `torbles/math_elim.c`
- `torbles/math_elim_gpu.cu`
- `torbles/math_elim.py`
- `tests/test_math_elim.py`
- `bench/benchmark_math_elim.py`

Acceptance tests:

- Reconstruction covers every supported variable and preserves ranges/NULL domains.
- Enforce degree/term/solve budgets and retain the direct candidate when expansion loses.

Benchmark cases:

- Use small relation components with 2/4/8 variables, degrees 1/2/3 and sparse/dense certificates; separate discovery, certificate checking and reduction.
- Compare direct expressions with alternative generator/elimination forms using actual resulting bit widths and scans; retain term-explosion and cost-regression cases.

Additional limits:

- Reconstructions are polynomials in the generators (no rational, algebraic or implicit reconstructions); a column that is only algebraic over the others stays a generator.
- The search is exponential in the number of variables (at most 255 sets for 8) and its lex completions grow quickly: with dense or implicitization-like sets many are undecided under the budget or int64, and minimality is then not claimed.
- Coefficients are GEOM-001's int64 rationals (no arbitrary precision); a reconstruction or form past int64 is undecided or unclaimable, never rounded.
- A reconstruction is exact where its relations hold: a snapshot relation makes it exact on that snapshot's rows only (FND-007 staleness applies).
- Costs are relative work units of the bit-plane engine measured on one scope (widths, scans, word counts), not a planner's timing model.
- The CUDA kernel covers the batched row checks; selection, completion, reduction and certification run on the CPU.
- The CUDA kernel has been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `4487999185c3750d020db8f5048dacf383236504`.

- tests: [results/math/GEOM-002/tests_test_math_elim.json](results/math/GEOM-002/tests_test_math_elim.json)
- tests: [results/math/GEOM-002/sanitized_test_math_elim.json](results/math/GEOM-002/sanitized_test_math_elim.json)
- benchmarks: [results/math/GEOM-002/benchmark_cpu_math_elim_vm.json](results/math/GEOM-002/benchmark_cpu_math_elim_vm.json)

## DISC: Relationship discovery

Polynomial identities, conserved quantities and hidden coordinates.

<a id="disc-001"></a>

### DISC-001: Existing bounded affine, sum and product discovery

Status: **validated**. Delivery wave: **W0**.

Current native exhaustive modules discover/probe affine, sum and product relations with bounded candidate budgets and snapshot evidence.

Exactness: Exact exhaustive snapshot evidence for supported non-NULL logical affine columns; not a universal law or predictor.

Dependencies: None.

Native CPU: Existing bounded native discovery (relationships.c); math_disc.c adds the order-independent exhaustive verifier (tr_disc_verify) and the affine proposal, both native.

GPU: CUDA kernel math_disc_gpu.cu (tr_mgpu_disc_verify): exhaustive verification of one hypothesis on the device, identical evidence to the CPU verifier and to tr_probe; proposals stay on the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships.c](torbles/relationships.c): Native affine/product/sum proposals and exact verification.
- [torbles/relationships_optimizer.c](torbles/relationships_optimizer.c): Existing bounded equation composition/search.
- [tests/test_relationships.py](tests/test_relationships.py): Recorded relationship tests; rerun required for new changes.

Implemented files:

- [torbles/relationships.h](torbles/relationships.h)
- [torbles/relationships.c](torbles/relationships.c)
- [torbles/relationships_optimizer.c](torbles/relationships_optimizer.c)
- [torbles/relationships.py](torbles/relationships.py)
- [torbles/math_disc.h](torbles/math_disc.h)
- [torbles/math_disc.c](torbles/math_disc.c)
- [torbles/math_disc_gpu.cu](torbles/math_disc_gpu.cu)
- [torbles/math_disc.py](torbles/math_disc.py)
- [tests/test_math_disc.py](tests/test_math_disc.py)
- [tests/test_relationships.py](tests/test_relationships.py)
- [bench/benchmark_math_disc.py](bench/benchmark_math_disc.py)

Acceptance tests:

- Preserve existing tests for signed offsets, sum/product/square probes and one-row violations.
- Refuse partial/nullable/stale proofs as exact rewrites.

Benchmark cases:

- Use 2/4/8 observed variables over 1k/100k/1M rows with planted affine/product/conservation relations plus one-row near misses; report proposal and exhaustive verification separately.
- Vary candidate/basis degree budgets and revision stability; measure exact Gram/nullspace state growth and refusals rather than extrapolating unsupported bases.

Additional limits:

- The CUDA verifier copies one hypothesis's columns to the device per call; there is no device-resident catalog, and discovery (proposal enumeration) is CPU only.
- Evidence is exact for the observed snapshot of non-NULL logical integer columns with <=32-bit stored codes; it is not a universal law or predictor.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

- tests: [results/math/DISC-001/tests_test_math_disc.json](results/math/DISC-001/tests_test_math_disc.json)
- tests: [results/math/DISC-001/sanitized_test_math_disc.json](results/math/DISC-001/sanitized_test_math_disc.json)
- benchmarks: [results/math/DISC-001/benchmark_cpu_math_disc_vm.json](results/math/DISC-001/benchmark_cpu_math_disc_vm.json)
- benchmarks: [results/math/DISC-001/benchmark_device_math_disc_WIN_OHVNN0FUC09.json](results/math/DISC-001/benchmark_device_math_disc_WIN_OHVNN0FUC09.json)

<a id="disc-002"></a>

### DISC-002: Moment-Gram polynomial identity discovery

Status: **in_progress**. Delivery wave: **W3**.

Use exact Gram/nullspace relationships of a bounded basis to propose and certify polynomial identities under nonnegative row weights.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [LIN-002](#lin-002), [FND-007](#fnd-007)

Native CPU: Native plan, Gram, nullspace, identities and certificates over an FND-004 scope and up to 8 integer columns: the canonical basis (every monomial of degree <= d or an explicit list, at most 63, increasing graded lexicographic order) and its distinct moment products; the representation decided before reading: EXACT when N W prod M_v^e <= 2^127 - 1 is proven from the scope's extremes (exact signed 128-bit moment sums), else MODULAR (residues modulo the recorded LIN-002 primes). The Gram G = sum w_r m(x_r) m(x_r)^T over the rows where every variable and the weight are present, from a fused pass, an FND-004 joint bank or LIN-001 moments (all equal), merged or subtracted (a difference is signed), recording its domain, weight signs and extremes. The nullspace: LIN-002's modular batch, CRT and exact verification G c = 0 for an exact Gram (exact rank and canonical vectors, one per free column), or per prime the reduced echelon form, the CRT and rational reconstruction of the primes agreeing on the best key for a modular Gram (rank a lower bound, unresolved vectors flagged). Each vector becomes a primitive integer polynomial (int64 coefficients or unrepresentable) with exact labels: distinct points S of its support columns (64-bit fingerprints in a radix sort, equal fingerprints compared value by value), underdetermined when its terms outnumber S (never a discovery), known (in the span of the declared relations' multiples within the basis) or a consequence (also of the earlier generators' multiples; fraction-free 128-bit elimination), and the Gram's sum-of-squares verdict (certified only for an exact Gram under nonnegative weights; refused as signed when a weight is negative or the Gram is a difference). Every candidate (kernel vectors or given polynomials) is verified on every selected row (a row with a NULL support column excluded) by an exact evaluation in W 64-bit words under a proven bound (W <= 8, else undecided), giving FND-007 evidence with the smallest violating row; a verified identity becomes an FND-007 snapshot certificate (the evidence adopted when W = 2, else FND-007's own check, which may refuse past 128 bits). The exact quadratic form c^T G c (big integers) and its binary64 value; a numerical near zero yields at most an FND-007 hypothesis on a sample. Budgets (variables, degree, support, tensor, states, iterations, bytes including the distinct-count scratch and device copies) are checked before allocation.

GPU: tr_mgpu_ident_gram: work items (moment chunk, residue slot, row slice) evaluate the basis on each Gram row of their slice (exactly, or modulo their prime), accumulate their chunk's moments locally (proven 128-bit sums, or sums of residues) and add them with one two-word atomic each; the items of chunk 0 and slot 0 count the rows by weight sign and keep the weight extremes by atomic add, min and max. tr_mgpu_ident_verify: work items (candidate chunk, row slice) evaluate every candidate of the chunk on each row of the slice with the inline W-word evaluation, count locally and add one atomic per counter, the smallest violating row by atomic minimum. Both run the inline functions of math_ident.h on device copies of the columns and give the CPU's states and evidence bit for bit; the exact nullspace can run its LIN-002 elimination on the device (tr_mgpu_lin_eliminate). The host checks the plan's or candidates' structure and the budget of the device copies before launching. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_ident.h`
- `torbles/math_ident.c`
- `torbles/math_ident_gpu.cu`
- `torbles/math_ident.py`
- `tests/test_math_ident.py`
- `bench/benchmark_math_ident.py`

Acceptance tests:

- Recover exact small polynomial dependencies and reject a single violating row.
- Signed delta weights cannot use zero sum-of-squares as an identity certificate; numerical near-zero is only a hypothesis.

Benchmark cases:

- Use 2/4/8 observed variables over 1k/100k/1M rows with planted affine/product/conservation relations plus one-row near misses; report proposal and exhaustive verification separately.
- Vary candidate/basis degree budgets and revision stability; measure exact Gram/nullspace state growth and refusals rather than extrapolating unsupported bases.

Additional limits:

- A relation violated on a few rows makes its products with polynomials vanishing on those rows exact identities of one degree more (a near miss q = x + y at row r gives (q - x - y)(x - x_r) = 0); they hold on every row and are reported as discoveries, nothing factors them.
- Labels are decided within the basis's degree bound: an identity implied only through multiples past it is labelled a generator (GEOM-001 reduces such consequences exactly).
- At most 8 integer columns, 63 basis monomials (LIN-002's exact systems), degree 16 and 32 primes; larger requests are refused, never approximated.
- A modular Gram's rank is a lower bound and its vectors are proposals until the exhaustive verification; a vector whose rational reconstruction fails is unresolved.
- An identity verified in more than two 64-bit words may be refused a certificate by FND-007's 128-bit check (verified, not certified).
- The numerical quadratic form is binary64 and only ever supports a hypothesis certificate.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `2e0e328cf05243328e3a4e64b529f54463387459`.

- tests: [results/math/DISC-002/tests_test_math_ident.json](results/math/DISC-002/tests_test_math_ident.json)
- tests: [results/math/DISC-002/sanitized_test_math_ident.json](results/math/DISC-002/sanitized_test_math_ident.json)
- benchmarks: [results/math/DISC-002/benchmark_cpu_math_ident_vm.json](results/math/DISC-002/benchmark_cpu_math_ident_vm.json)

<a id="disc-003"></a>

### DISC-003: Conservation and stability across revisions

Status: **in_progress**. Delivery wave: **W3**.

Track scoped invariants across declared transformations/updates, recording persistence separately from snapshot truth or predictive evidence.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DISC-002](#disc-002), [DELTA-001](#delta-001)

Native CPU: tr_cons_tracker_make builds the exact DISC-002 Gram of an unweighted EXACT plan over a FND-004 scope, verifies up to 256 DISC-002 identities exhaustively and certifies the holding ones (FND-007 snapshots); tr_cons_job_make / tr_cons_job_run evaluate only the rows of a DELTA-001 delta: the Gram increment sum w m m^T over complete rows (each moment proven against 128 bits from the delta's extremes before anything is added) and, per identity, the checked / excluded / violating / undecided weights of inserted and deleted rows with the smallest violating row of each (W-word exact evaluation, W proven from the extremes); tr_cons_tracker_update adds them (a DELTA-001 companion state refuses deltas that do not apply and deletes of rows that are not there), certifies holding identities again after a DIFF bound to the new scope (the evidence is then that scope's exhaustive evidence by induction; tr_cert_adopt) and only holds them after other deltas; tr_cons_tracker_rebind is the complete recomputation. Per identity a persistence record (certified steps, runs, streak, the first failure's step, revision, violations and witness row with its values) and four evidence classes kept apart (declared, snapshot, stability, hypothesis); only a fresh snapshot authorizes an exact rewrite (tr_cons_authorize). tr_cons_conserve expands P = q(T(x)) - q(x) exactly: zero is a universal FND-007 RING_IDENTITY law, else a counterexample point from the Nullstellensatz grid; tr_cons_law_candidates / tr_cons_law_witnesses check P = 0 on the rows (snapshot certificate or the violating rows).

GPU: tr_mgpu_cons_delta evaluates a job's delta rows on the device: work items (moment chunk, row slice) add the exact Gram increments with two-word atomics (tr_atomic_add_wide; the host proved the 128-bit bound), work items (identity chunk, row slice) evaluate every identity exactly (DISC-002's W-word evaluation) and add their evidence counters with integer atomics and the smallest violating inserted / deleted rows with atomic minima; no barriers or shared memory, results equal to tr_cons_job_run bit for bit, adopted by tr_cons_tracker_update after consistency checks.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_conserve.h`
- `torbles/math_conserve.c`
- `torbles/math_conserve_gpu.cu`
- `torbles/math_conserve.py`
- `tests/test_math_conserve.py`
- `bench/benchmark_math_conserve.py`

Acceptance tests:

- Known conserved transformations preserve certified quantities; perturbations create witnesses.
- Schema guarantees, observed revision stability and sampled hypotheses remain separate evidence classes.

Benchmark cases:

- Use 2/4/8 observed variables over 1k/100k/1M rows with planted affine/product/conservation relations plus one-row near misses; report proposal and exhaustive verification separately.
- Vary candidate/basis degree budgets and revision stability; measure exact Gram/nullspace state growth and refusals rather than extrapolating unsupported bases.

Additional limits:

- plans are unweighted and EXACT (no modular or weighted Gram); at most 8 variables, 63 basis monomials (DISC-002) and 256 tracked identities
- identities are certified again only on bound revisions (a DIFF of the current scope given with the new scope); ARRAYS and SCOPE deltas only hold them; a same-version table DIFF compares the selection only (DELTA-001)
- the tracker re-verifies the identities it was given; new identities at a revision come from DISC-002's nullspace of the tracker's Gram (tr_cons_tracker_gram), not from the tracker
- an identity whose evaluation needs more than 8 words on a delta is undecided until a rebind; a conservation counterexample search needs a grid evaluation within 8 words
- transformations are polynomial maps (no division, no piecewise or conditional maps); a stability record states only what held on the revisions it lists
- CUDA covers the delta evaluation only (certification, records and laws run on the CPU); it was run in the CPU emulation here, not on a physical GPU

Validation receipts:

Measured runtime: `3833d4d1d856591a0f2080e9ecb42e5545209163`.

- tests: [results/math/DISC-003/tests_test_math_conserve.json](results/math/DISC-003/tests_test_math_conserve.json)
- tests: [results/math/DISC-003/sanitized_test_math_conserve.json](results/math/DISC-003/sanitized_test_math_conserve.json)
- benchmarks: [results/math/DISC-003/benchmark_cpu_math_conserve_vm.json](results/math/DISC-003/benchmark_cpu_math_conserve_vm.json)

## PRE: Preimage queries

Push conditions on derived values back to generators.

<a id="pre-001"></a>

### PRE-001: Affine and monotone-piece preimages

Status: **in_progress**. Delivery wave: **W3**.

Transform predicates on derived values into exact generator intervals with integer rounding, sign and NULL guards.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-007](#fnd-007), [UNC-001](#unc-001)

Native CPU: Native transformation of a predicate y op c (op <, <=, =, <>, >=, >, or IN a finite set) on a derived value y = f(x) into the exact set of stored codes of the generator column, for its codec (logical = code + offset, x = logical / 10^scale, codes over a declared domain, e.g. 0 .. 2^width - 1 of an engine column): affine maps a x + b (a negative slope flips the inequality; integer rounding by floor and ceil of the exact threshold), squares a x^2 + b (symmetric intervals by integer square roots, zero, one or two points for equalities, both branches always), residue maps x mod m (sets of residues of L mod m 10^scale, then residue classes of codes, or plain intervals when the domain is not larger than the modulus) and piecewise maps of affine and square pieces with closed domains (each piece solved in its code range, all unioned; a piece whose image lies on one side is decided by UNC-001's exact abstract evaluation). Thresholds are FND-009 big rationals, so nothing overflows or rounds. A zero slope or modulus is refused (DomainError) and pieces whose code ranges leave a gap (a missing branch, DomainError) or overlap (ValueError) are refused, never approximated. Results are sorted, disjoint spans with the empty and whole-domain flags, the exact code count and the NULL guard (a NULL value never satisfies), with one FND-007 law (ring identity of the codec substitution D f((code + offset) / T) - (P code + Q), decided canonically zero by FND-007) per branch. The scan sets the rows of a scope column whose present codes are in the set and, with the direct check, evaluates the original predicate exactly on every selected row (UNC-001's row evaluator) and counts mismatches. Budgets (support, output, bytes) are checked before allocation.

GPU: tr_mgpu_pre_scan: one work item per 64-row word evaluates the transformed predicate on the stored codes (span membership by binary search) and writes its own bitmap word; with the direct check it also evaluates the original predicate on each selected row with UNC-001's exact row evaluator and adds the counts (NULL rows, in-set rows, true rows, mismatches, undecided rows, rows outside the domain) by atomics, the first mismatching and undecided rows by atomic minimum. It runs the inline functions of math_pre.h on the set's plain view with device copies, equal to the CPU bit for bit. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing typed comparisons and dimension predicate substitution.
- [torbles/coltypes.py](torbles/coltypes.py): Existing logical codec intervals; general derived-function preimages remain planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pre.h`
- `torbles/math_pre.c`
- `torbles/math_pre_gpu.cu`
- `torbles/math_pre.py`
- `tests/test_math_pre.py`
- `bench/benchmark_math_pre.py`

Acceptance tests:

- All comparison operators agree on boundary, negative-scale, fractional-threshold and empty cases.
- Require nonzero divisors and all branches for noninvertible maps.

Benchmark cases:

- Use affine positive/negative slopes, monotone pieces, squares and residue maps over bounded integer domains; compare all predicate boundaries with direct evaluation.
- Vary preimage branch counts 1/4/16/64 and predicate selectivity; compare transformed plane predicates with the correct direct/lookup candidate and retain losing cases.

Additional limits:

- Maps are affine, a x^2 + b (vertex at 0), x mod m with an integer m and pieces of affine and square maps; no general polynomial or composed maps.
- Residue spans have no bit-plane program (a code lookup is the engine candidate); self-extending (wide) columns are not supported.
- The law certificates cover the algebraic substitution of the codec; the integer rounding is proved by the exact computation and the exhaustive tests, not by a certificate; a branch whose coefficients leave int64 has none.
- The direct check evaluates in int64 rationals: a row it cannot decide is reported (overflow), not checked.
- The CUDA kernel has been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `246664aa793bcecefac4cd954e6366a1a85271f8`.

- tests: [results/math/PRE-001/tests_test_math_pre.json](results/math/PRE-001/tests_test_math_pre.json)
- tests: [results/math/PRE-001/sanitized_test_math_pre.json](results/math/PRE-001/sanitized_test_math_pre.json)
- benchmarks: [results/math/PRE-001/benchmark_cpu_math_pre_vm.json](results/math/PRE-001/benchmark_cpu_math_pre_vm.json)

<a id="pre-002"></a>

### PRE-002: Finite roots, residues and periodic preimages

Status: **in_progress**. Delivery wave: **W3**.

Represent all bounded preimage branches as intervals, residue classes, finite sets or Boolean cubes.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PRE-001](#pre-001), [FIELD-001](#field-001)

Native CPU: Native exact preimages over PRE-001's codec (logical = code + offset, value = logical / 10^scale, codes over the declared domain) and predicates (y op c, c an int64 rational, or y IN up to 4096 values). Polynomial maps of degree <= 16 with rational coefficients: the integer polynomial N(L) = K p(L / 10^scale) in FND-009 big integers, its critical points (the real roots of N') isolated by INV-001's Sturm sequences and refined to width 1, the integers between them strictly monotone pieces, and on each piece two binary searches with exact big-integer evaluations give the level sets (intervals; the exact integer roots for equality; every IN value through the same pieces). Power and polynomial residues (sum c_i x^i) mod m, 1 <= m <= 2^62: the solutions per prime power of m (trial division, each prime power enumerated up to 2^24 residues) combined by the CRT for equality and IN, every residue enumerated for the other operators (m <= 2^24). Affine residues (a x + b) mod m: the linear congruences solved by gcd and modular inverses, residue classes of the period m / gcd(a, m). GF(2) bit maps XOR / AND / OR with a constant on the w low bits of logical - base (FIELD-001's bit variables): the images of the dyadic blocks of {y op c} as Boolean cubes of the stored code, or residue runs mod 2^(H+1) when the base is not the code's. Results are unions of one representation: sorted disjoint intervals (finite sets as points), residue classes of one period (a domain no larger than the period as intervals) or disjoint cubes; residues of a power-of-two period convert to low-bit cubes for bit-plane programs. Past the output budget (branches), the iterations or bytes budget or a representation limit the set falls back to a lookup bitmap (residues mod the period or the domain's codes, at most 2^26 bits, computed exactly) or to the direct exact evaluation of the predicate (polynomials in W <= 8 words under a bound proven over the domain, residues in 128-bit modular arithmetic), never dropping a solution; fallback none refuses with the exceeded resource. Each polynomial or periodic set carries an FND-007 LAW certificate of its transformation (the codec substitution, the periodicity identities) when its coefficients fit int64. The scan marks the selected rows whose present value's code is in the set and can evaluate the original predicate on every row and count mismatches; the verification checks every code of a domain of at most 2^26 codes against the direct predicate. A NULL value never satisfies.

GPU: tr_mgpu_preimg_scan: one work item per 64-row selection word evaluates membership of each present row's code (interval binary search, residue runs, cubes, a lookup bitmap or the direct predicate, the inline functions of math_preimg.h shared with the CPU), optionally the direct predicate and its comparison, writes its own bitmap word and adds its counts with one atomic each, the smallest mismatching and undecided rows by atomic minimum. tr_mgpu_preimg_verify: one work item per 64 codes of the domain counts codes, members, true codes and mismatches with one atomic each and keeps the smallest mismatching code by atomic minimum. No barrier, no shared memory; results equal the CPU's bit for bit. The host validates the set's view and budgets the device copies before launching; the set itself is built on the CPU. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_preimg.h`
- `torbles/math_preimg.c`
- `torbles/math_preimg_gpu.cu`
- `torbles/math_preimg.py`
- `tests/test_math_preimg.py`
- `bench/benchmark_math_preimg.py`

Acceptance tests:

- Compare all preimages with exhaustive small integer domains for squares/moduli/periods.
- Branch explosion triggers a budgeted fallback rather than dropping solutions.

Benchmark cases:

- Use affine positive/negative slopes, monotone pieces, squares and residue maps over bounded integer domains; compare all predicate boundaries with direct evaluation.
- Vary preimage branch counts 1/4/16/64 and predicate selectivity; compare transformed plane predicates with the correct direct/lookup candidate and retain losing cases.

Additional limits:

- Periodic and bit maps are defined on integer domains only (scale 0); polynomial maps have degree <= 16 and int64 rational coefficients and thresholds.
- Composite moduli are factored by trial division and prime powers enumerated up to 2^24 residues; past that (or past the CRT class limit) the set falls back to a lookup or the direct evaluation.
- Residues of a period that is not a power of two have no bit-plane program: the lookup or the direct evaluation is the candidate; a plane program is limited to 1024 engine instructions.
- The exhaustive verification covers domains of at most 2^26 codes with values within int64.
- Branch construction, critical-point isolation and certificates run on the CPU; CUDA covers the scan and the verification.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `4487999185c3750d020db8f5048dacf383236504`.

- tests: [results/math/PRE-002/tests_test_math_preimg.json](results/math/PRE-002/tests_test_math_preimg.json)
- tests: [results/math/PRE-002/sanitized_test_math_preimg.json](results/math/PRE-002/sanitized_test_math_preimg.json)
- benchmarks: [results/math/PRE-002/benchmark_cpu_math_preimg_vm.json](results/math/PRE-002/benchmark_cpu_math_preimg_vm.json)

## ENV: Parametric envelopes

Winning groups, exact breakpoints and parameter regions.

<a id="env-001"></a>

### ENV-001: Univariate affine/quadratic group envelopes

Status: **in_progress**. Delivery wave: **W3**.

Return winning groups and exact parameter breakpoints with a declared tie policy.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [INV-001](#inv-001)

Native CPU: Native C (math_env.c): the lower or upper envelope of one affine or quadratic function per group over a parameter interval: the winning group per piece and exact breakpoints (rational, or certified isolating intervals via INV-001 for algebraic ones), with a declared tie policy, coincident pieces reported, empty groups never winning and unbounded intervals handled exactly.

GPU: CUDA kernel math_env_gpu.cu: batched exact lookups of the winning group at many parameter values, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_env.h`
- `torbles/math_env.c`
- `torbles/math_env_gpu.cu`
- `torbles/math_env.py`
- `tests/test_math_env.py`
- `bench/benchmark_math_env.py`

Acceptance tests:

- Compare envelope winners against all group functions at every interval and boundary.
- Handle identical functions, empty groups and unbounded parameter intervals exactly.

Benchmark cases:

- Build 8/64/1,024 group functions and request 1/100/10,000 parameter values; compare an envelope build plus lookups with repeated all-group evaluation.
- Vary coincident curves, breakpoint count and parameter dimension; report exact region/output size and over-budget region cases.

Additional limits:

- One parameter, degree at most 2; the build grows roughly with groups^2 when every group wins (budgeted).
- The envelope is built on the CPU; irrational breakpoints are certified intervals refined on request; envelope values only at rational inputs.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/ENV-001/tests_test_math_env.json](results/math/ENV-001/tests_test_math_env.json)
- tests: [results/math/ENV-001/sanitized_test_math_env.json](results/math/ENV-001/sanitized_test_math_env.json)
- benchmarks: [results/math/ENV-001/benchmark_cpu_math_env_vm.json](results/math/ENV-001/benchmark_cpu_math_env_vm.json)

<a id="env-002"></a>

### ENV-002: Bounded multivariate parameter regions

Status: **in_progress**. Delivery wave: **W4**.

Represent supported linear/convex regions where groups or plans dominate; bound region count and dimension.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [ENV-001](#env-001), [INV-002](#inv-002)

Native CPU: Native C (math_region.c; the lookup functions inline in math_region.h): for groups whose functions are affine in a parameter vector of dimension d = 1..4 with exact rational coefficients (FND-009 big rationals), over a declared box with up to 64 further linear rows (P nonempty with an interior, decided by an exact OPT-002 LP), identical functions merged into classes; a kd-tree candidate index on a dyadic lattice with exact corner bounds and pairwise-difference pruning; per candidate class the region R_c = {theta in P : f_c <= f_c' for every c'} computed exactly from its index neighbours (Q_c = R_c whenever R_c is nonempty) by the double description method in homogeneous integer coordinates with the combinatorial adjacency test: nonempty regions of full or lower dimension with exact vertices, irredundant primitive integer inequalities and implicit equalities (exact ranks), the sources of every facet (box faces, rows, the classes tied along it), and each region's exact LP evidence (OPT-002 simplex: the maximum margin and its point) checked against the enumeration; vertices deduplicated with their tied classes and walls; boundaries (wall, cell, inner) with the classes tied on them; a declared tie policy (smallest or largest member group) names the winner; the maximizer by negation. Point location of many values (int64 arrays or big rationals) by the kd-tree, by every region's description or by every class, all exact through residues modulo INV-001's verified primes with Garner signs and a proven bound. Quadratic coefficients are refused unless the explicit piecewise-linear contract is declared: per cell an affine function with an exact remainder enclosure |q - L| <= eps, the largest eps checked against the declared error budget, the result labelled approximate (an exact use refused through FND-007). Budgets are checked before allocation or counted exactly and refused at limit + 1; nothing is truncated.

GPU: math_region_gpu.cu: tr_mgpu_region_locate copies a lookup job's view (the classes' residues, the kd-tree, the regions' inequalities, the cells) to the device and answers one parameter value per work item with the same inline tr_region_answer_one as the CPU (index, regions or all modes: status, kind, cell, tied classes, winner, class, region), each into its own output slots, so the answers do not depend on the order of the threads and equal the CPU bit for bit (emulated with a scrambled thread order in the tests); the output and device-byte budgets are checked on the host before launching. The region build is CPU only.

Bounds:

- Build, before allocation: variables = d, degree (1, or 2 under the contract), states = cells x groups, bytes = the function tables.
- Build, counted exactly and refused at limit + 1 with that need (nothing partial, never truncated): output = nonempty regions, support = region-vertex incidences, circuit = index nodes, iterations = index nodes + clipping steps + LP pivots, bytes = the tables as they grow.
- Lookups: output = values, bytes = the job (CUDA: plus every device copy, checked together before launching).

Implementation targets (proposed until the capability is validated):

- `torbles/math_region.h`
- `torbles/math_region.c`
- `torbles/math_region_gpu.cu`
- `torbles/math_region.py`
- `tests/test_math_region.py`
- `bench/benchmark_math_region.py`

Acceptance tests:

- Exhaustive small grids agree with region membership and tie boundaries.
- Unsupported nonlinear regions are not approximated without an explicit permitted contract.

Benchmark cases:

- Build 8/64/1,024 group functions and request 1/100/10,000 parameter values; compare an envelope build plus lookups with repeated all-group evaluation.
- Vary coincident curves, breakpoint count and parameter dimension; report exact region/output size and over-budget region cases.

Additional limits:

- Affine functions in d <= 4 only; curved (quadratic) regions exist only as the labelled piecewise-linear approximation with a certified error, never as exact regions.
- The build is single-threaded CPU code whose work grows with the regions, their vertices and their index neighbours (measured: 1,024 tangent planes in d = 2 about 11 s, 64 in d = 4 about 6 s on the 2-core development machine); the CUDA path is point location only.
- At most 64 rows besides the box, a certifying LP of at most 255 rows, lookup values within 64 moduli (about 3,900 bits).
- No FND-007 certificate kind describes a region: the LP evidence and the vertex enumeration are cross-checked natively, and FND-007 is used to refuse the exact use of an approximate result.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/ENV-002/tests_test_math_region.json](results/math/ENV-002/tests_test_math_region.json)
- tests: [results/math/ENV-002/sanitized_test_math_region.json](results/math/ENV-002/sanitized_test_math_region.json)
- benchmarks: [results/math/ENV-002/benchmark_cpu_math_region_vm.json](results/math/ENV-002/benchmark_cpu_math_region_vm.json)

## GRAPH: Graph and fixed-point queries

Incidence operators, frontiers, paths and bounded recursive computations.

<a id="graph-001"></a>

### GRAPH-001: Declared graph layouts and reductions

Status: **in_progress**. Delivery wave: **W5**.

Build explicit incidence/adjacency ownership and native degree, frontier, neighborhood and matrix-vector operations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [LIN-001](#lin-001)

Native CPU: Native C (math_graph.c; contracts and the per-vertex pulls shared with CUDA in math_graph.h): immutable layouts built from the selected edge rows of an FND-004 scope (table or arrays; key columns of one decimal scale, an optional weight column with a declared unit, Fraction weights put on their least common denominator natively; NULL endpoints or weights skipped and counted, or refused; an optional vertex scope adds isolated vertices). The vertex dictionary is the radix-sorted distinct keys (dense id = rank); canonical edges carry multiplicities (deduplication declared, with refuse / min / max / sum weight combination; loops kept or dropped; an undirected edge appears in both directions and a self-loop once); CSR and CSC are built by stable counting sorts, with the canonical edge list and the vertex-incidence lists (oriented incidence roles). The identity is the declaration, the exact FND-004 descriptors (snapshot revision included) and the dictionary, compared byte for byte; a stale layout is refused natively and every state of another identity is refused. Exact degrees (out / in / total / loops in multiplicity, weight and distinct modes, checked int128), one-step frontier expansion with exclusion, neighborhood aggregates, and pull products y = A^T x or A x over declared semirings: counting (+, x) in checked int128, Boolean (or, and), GF(2) (equal to FIELD-001's matvec of the exported packed A mod 2), (min, +) and (max, +) in checked int64 units of the weight denominator, binary64 (+, x) as a fixed-order fma accumulation with a rigorous per-vertex a priori error bound; NULL inputs contribute no term and validity is reported. Budgets (layout bytes, support, states, output; tensor, output and bytes per request) are checked before allocation.

GPU: math_graph_gpu.cu: degrees and every semiring product with one work item per vertex running the shared inline pull (entries in their stored order) into its own output slot (Boolean / GF(2) words by atomic OR, an overflow by atomic minimum, the binary64 coefficients computed on the host and copied); frontier expansion with one work item per frontier word pushing by atomic OR, the count by atomic add. Device copies are budgeted; results equal the CPU's bit for bit (scrambled emulation in the tests). Compiled with nvcc for sm_80; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_graph.h`
- `torbles/math_graph.c`
- `torbles/math_graph_gpu.cu`
- `torbles/math_graph.py`
- `tests/test_math_graph.py`
- `bench/benchmark_math_graph.py`

Acceptance tests:

- Compare directed/undirected/self-loop/multiedge semantics with tiny graph references.
- Vertex dictionaries and snapshot revisions remain part of layout/state identity.

Benchmark cases:

- Use chain, star, disconnected and cyclic graphs with 1k/100k/1M edges; record adjacency/incidence build separately from frontier and reduction iterations.
- Vary frontier density and bounded path length 1/4/16/32; compare supported semirings with small exact graph oracles and retain nontermination/overflow refusals.

Additional limits:

- The layout build (key sort, CSR / CSC, incidence lists) runs on the CPU only, single-threaded; a device entry copies the forms it reads on every call (no device-resident layout).
- Keys are exact-valued integer or decimal columns (no string keys); one graph per scope (a grouped scope is refused); at most 2^32 - 2 vertices and 2^32 - 1 entries.
- The binary64 products carry an a priori error bound for exact double inputs, not an enclosure; GF(2) products of weights need integral weights.
- No FND-007 claim kind states graph facts, so layouts and products emit no certificates.

Validation receipts:

Measured runtime: `4487999185c3750d020db8f5048dacf383236504`.

- tests: [results/math/GRAPH-001/tests_test_math_graph.json](results/math/GRAPH-001/tests_test_math_graph.json)
- tests: [results/math/GRAPH-001/sanitized_test_math_graph.json](results/math/GRAPH-001/sanitized_test_math_graph.json)
- benchmarks: [results/math/GRAPH-001/benchmark_cpu_math_graph_vm.json](results/math/GRAPH-001/benchmark_cpu_math_graph_vm.json)

<a id="graph-002"></a>

### GRAPH-002: Bounded fixed points, paths and graph energies

Status: **in_progress**. Delivery wave: **W5**.

Add reachability, bounded path counts, selected shortest-path semirings and iterative energies with explicit termination/iteration rules.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GRAPH-001](#graph-001), [GEN-002](#gen-002)

Native CPU: Native C on GRAPH-001 layouts: reachability R* / R+ from a source set with BFS levels and the rounds to the fixed point (a max_rounds bound refuses with NotConverged), the transitive closure per source (64 sources per lane word, FIELD-001 rows); walk counts w_(l+1) = A^T w_l over multiplicities for lengths <= L as exact unsigned multiword integers sized by a declared bound (every row, total, length total and the grand total checked, the first violation reported), the unbounded count only when no cycle is reachable from the sources (Kahn's order on the reachable part, else NoSolution); synchronous frontier Bellman-Ford in (min, +) / (max, +) with checked int64 sums, where a change in round n is reported with the improving cycle extracted from the round predecessors and its weight (Unbounded), and Dijkstra with lexicographic (distance, hops) keys for nonnegative weights, both giving the same canonical hops and predecessors; the exact Dirichlet energy x^T L x in 320-bit arithmetic over the canonical edges; PageRank (dangling and teleport sets, damping in (0, 1]) and power iteration in binary64 with fixed blocked sums and explicit fma, stopping at the declared tolerance or refusing at the iteration cap (NotConverged), with rigorous outward-rounded a posteriori bounds (||r - r*||_1, residual norms, an eigenvalue radius for undirected layouts).

GPU: math_graph_gpu.cu: reach, closure, walk counts and Bellman-Ford as one launch sequence per round or step (candidates from atomic touch counters, the shared pull per candidate into its own slot, then the commit; violations and witnesses by atomic minimum, walk length totals by exact multiword atomic adds, the host reading each round's counters); the energy by edge slices with 320-bit local sums and multiword atomic adds; PageRank / power iteration with block sums (one work item per block), their total (one work item) and the per-vertex pull update each iteration. The certificates, the walk horizon and the cycle extraction run on the host. Results and statistics equal the CPU's bit for bit (scrambled emulation in the tests). Compiled with nvcc for sm_80; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_graph.h`
- `torbles/math_graph.c`
- `torbles/math_graph_gpu.cu`
- `torbles/math_graph.py`
- `tests/test_math_graph.py`
- `bench/benchmark_math_graph.py`

Acceptance tests:

- Compare tiny cycles, disconnected graphs and negative/zero weights under the stated semiring.
- Unbounded walk counts and nonconvergent recurrences cannot silently return a finite answer.

Benchmark cases:

- Use chain, star, disconnected and cyclic graphs with 1k/100k/1M edges; record adjacency/incidence build separately from frontier and reduction iterations.
- Vary frontier density and bounded path length 1/4/16/32; compare supported semirings with small exact graph oracles and retain nontermination/overflow refusals.

Additional limits:

- Dijkstra and the iteration certificates run on the CPU only; Bellman-Ford rounds are synchronous and every device round reads its counters back to the host.
- Walk counts count walks (multiplicities, not weights; weighted path sums are GRAPH-001's count and tropical products); bounds of at most 2048 bits; an unbounded count needs no reachable cycle at all (finiteness is not refined per vertex).
- PageRank and power iteration are binary64 with a posteriori bounds; damping 1 gives no bound (inf); power iteration needs nonnegative weights and converges only with a dominant eigenvalue.
- No FND-007 claim kind states graph facts: a refused cycle carries its own witness (vertices and weight) and paths their predecessor tree.

Validation receipts:

Measured runtime: `4487999185c3750d020db8f5048dacf383236504`.

- tests: [results/math/GRAPH-002/tests_test_math_graph.json](results/math/GRAPH-002/tests_test_math_graph.json)
- tests: [results/math/GRAPH-002/sanitized_test_math_graph.json](results/math/GRAPH-002/sanitized_test_math_graph.json)
- benchmarks: [results/math/GRAPH-002/benchmark_cpu_math_graph_vm.json](results/math/GRAPH-002/benchmark_cpu_math_graph_vm.json)

## PROV: Provenance

Factorized explanation and multiplicity expressions.

<a id="prov-001"></a>

### PROV-001: Factorized provenance payloads

Status: **in_progress**. Delivery wave: **W3**.

Native shared-DAG annotations distinguish Boolean existence, bag multiplicity and polynomial provenance.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009)

Native CPU: Native C (math_prov.c): a shared factorized derivation DAG per store for source, select, union, join and group (global group included), with explicit NULL and group presence; results tagged by semiring: Boolean existence with minimal witnesses, bag multiplicities, and expanded N[X] polynomials; node, child, byte and expansion budgets checked before anything is added.

GPU: CUDA kernel math_prov_gpu.cu: the DAG evaluated under many valuations at once (one work item per valuation, e.g. source deletions), equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_prov.h`
- `torbles/math_prov.c`
- `torbles/math_prov_gpu.cu`
- `torbles/math_prov.py`
- `tests/test_math_prov.py`
- `bench/benchmark_math_prov.py`

Acceptance tests:

- Small union/join/group examples match explicit derivation polynomials and multiplicities.
- Shared derivations stay factored; annotation limits and NULL/group presence are explicit.

Benchmark cases:

- Use small joins with shared versus independent derivations and increasing duplicate multiplicities; record factorized DAG nodes, annotation bytes and evaluation time.
- Compare 1/100/1,000 source-deletion or explanation requests against complete query reconstruction; include annotation-budget overflow and correlated derivations.

Additional limits:

- No SUM/MIN aggregates, difference, outer joins or recursion; integer values only; one owner per store (not thread-safe).
- The expansion budget may refuse polynomials that would fit; witnesses are minimized in quadratic time.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/PROV-001/tests_test_math_prov.json](results/math/PROV-001/tests_test_math_prov.json)
- tests: [results/math/PROV-001/sanitized_test_math_prov.json](results/math/PROV-001/sanitized_test_math_prov.json)
- benchmarks: [results/math/PROV-001/benchmark_cpu_math_prov_vm.json](results/math/PROV-001/benchmark_cpu_math_prov_vm.json)

<a id="prov-002"></a>

### PROV-002: Explanation specialization and deletion propagation

Status: **in_progress**. Delivery wave: **W5**.

Specialize retained provenance into source explanations and supported deletion effects without recomputing derivations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROV-001](#prov-001), [DELTA-002](#delta-002)

Native CPU: Native C (math_explain.c; the propagation tr_expl_set and its update laws inline in math_explain.h): an immutable explanation state copied from a PROV-001 plan's cone (nothing re-derived from the sources) with reverse edges (one parent entry per child entry), the answers of every node, the cone node of every variable and the baseline under a declared base valuation (N multiplicities in unsigned 128 bits and B existence with PROV-001's shared evaluators, and the per-node support counts B needs); deletion propagation of change sets (deletions, restorations, multiplicities) visiting only the nodes above the changed tuples in increasing order through a min-heap with an early cutoff, PLUS by the linear law (old N minus the changed children's old values plus their new ones), TIMES by the product of the children's new values (DELTA-002's finite difference dA B + A dB + dA dB), B by counting maintenance of the true (OR) or false (AND) children, with per set every answer's N and B, an overflow status redone in B alone, the nodes recomputed and the answers changed; explanations relative to the base on COUNT-002's decision diagram of the answer's monotone lineage (absent sources FALSE for why, present ones TRUE for why-not): the lineage, the relevant sources (the support of the existence function), the counterfactual sources, the minimal sufficient sets (minimal models on the diagram, refused past max_sets), a smallest sufficient set, the minimal why-not restorations, with found / absent / present / underivable status; answer probabilities under independent tuple probabilities as exact rationals (the weighted count of the lineage diagram with (p, 1 - p): shared variables exact) within the diagram budget, or certified outward-rounded bounds on the DAG without compilation (exact independence on variable-disjoint children, Harris-FKG bounds for increasing events otherwise).

GPU: CUDA kernel math_explain_gpu.cu: tr_mgpu_explain_propagate runs one work item per change set with the shared tr_expl_set on its own scratch row (the min-heap, the PLUS / TIMES / counting update laws, the early cutoff) and its own rows of answers, status, nodes recomputed and answers changed, restoring the touched scratch entries so rows are clean for the next launch chunk (chunks within 64 MiB of scratch); the host checks the view, the change sets and the budget first; equal to the CPU byte for byte. Exact probabilities can count the lineage diagram with COUNT-002's level kernels (tr_mgpu_symcount_levels).

Bounds:

- State: circuit = cone nodes, support = child entries, output = answers, bytes = the copies, reverse edges, answer lists and baseline.
- Propagation: output = change sets x answers, bytes = the scratch (65 per cone node; CUDA: every device copy, the outputs and the scratch chunk).
- Lineage diagrams: COUNT-002's builder (circuit = nodes held, iterations = apply steps, bytes); necessary sources: iterations = levels x diagram nodes; minimal sets: support = the set entries held, output = max_sets (refused, never truncated); probability bounds: bytes = the support bitsets and intervals.

Implementation targets (proposed until the capability is validated):

- `torbles/math_explain.h`
- `torbles/math_explain.c`
- `torbles/math_explain_gpu.cu`
- `torbles/math_explain.py`
- `tests/test_math_explain.py`
- `bench/benchmark_math_explain.py`

Acceptance tests:

- Delete each tiny input source and compare supported answer changes with complete reruns.
- Correlated derivations cannot be treated as independent probability events.

Benchmark cases:

- Use small joins with shared versus independent derivations and increasing duplicate multiplicities; record factorized DAG nodes, annotation bytes and evaluation time.
- Compare 1/100/1,000 source-deletion or explanation requests against complete query reconstruction; include annotation-budget overflow and correlated derivations.

Additional limits:

- PROV-001's operators only (selection, union, equi-join, group-by; no aggregates other than multiplicities, no negation, outer joins or recursion); for a global group (always present) the existence explained and propagated is COUNT(*) > 0.
- Probabilities assume independent source tuples; the exact path is bounded by the lineage diagram, whose size depends on the variable order (DFS first appearance by default); past it the bounds are certified but can be loose on heavily shared lineages.
- Explanations and probability bounds are per answer and single-threaded on the CPU; minimal sets are enumerated explicitly from the diagram; propagation is batched on the device but each change set is sequential within its work item.

Validation receipts:

Measured runtime: `cacb92841d6cae9a8a136a680c29f3cecf01b657`.

- tests: [results/math/PROV-002/tests_test_math_explain.json](results/math/PROV-002/tests_test_math_explain.json)
- tests: [results/math/PROV-002/sanitized_test_math_explain.json](results/math/PROV-002/sanitized_test_math_explain.json)
- benchmarks: [results/math/PROV-002/benchmark_cpu_math_explain_vm.json](results/math/PROV-002/benchmark_cpu_math_explain_vm.json)

## ATTR: Attribution and repair

Contributions, interaction indices and bounded minimal changes.

<a id="attr-001"></a>

### ATTR-001: Sparse interaction and contribution attribution

Status: **in_progress**. Delivery wave: **W3**.

Compute declared marginal/coalition contributions and sparse-decomposition attribution indices.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROV-001](#prov-001), [LATTICE-002](#lattice-002)

Native CPU: Native C (math_attr.c, the shared drivers in math_attr.h): immutable games declared as 2^n exact int128 coalition values (given; LATTICE-002 zeta coalition values of a LATTICE-001 histogram for conjunctions and disjunctions; a BOOL-001 program evaluated word-parallel with the absent conditions replaced by declared constants under SQL three-valued logic) or as a PROV-001 plan with weights, a semiring (N multiplicities, B existence) and a player of every source variable (others fixed present or absent), with certified facts recorded at creation (exact range, monotonicity, a nonnegative Moebius sign for N with weights >= 0, a degree bound). Sparse Moebius supports: the multilinear expansion of the plan's cone over player sets (x^2 = x, players' variables merged, budgeted by per-node term bounds), inclusion-exclusion over the minimal witnesses for B, the LATTICE-002 fast Moebius transform for 2^n values, or declared terms. Exact 128-bit canonical rationals for Shapley and Banzhaf values, Moebius coefficients, Shapley and Banzhaf interaction indices and Shapley-Taylor indices up to an order (every unlisted set exactly zero), with the efficiency, Shapley-Taylor efficiency, Moebius-sum and reconstruction identities checked natively. Enumeration (n <= 62): coalition values, exhaustive weighted marginal contributions (numerators over n! and 2^(n-1), wrapping 128-bit sums made exact by a proven bound), bounded exploration of whole levels within max_coalitions (Moebius regime: exact small coefficients and certified bounds from the per-player unknown mass and the degree bound; range regime: certified widening per unexplored level and the monotone clamp), and minimum explanation / repair search by increasing size (found with exact size and multiplicity, none proven, or incomplete with certified bounds and the explored states). Overflow is reported, never wrapped; every budget is checked before allocation.

GPU: math_attr_gpu.cu: tr_mgpu_attr_values evaluates many coalitions at once, one work item per coalition (a 0/1 valuation of the provenance cone with the shared tr_attr_eval, or a table lookup) into its own slot, in launch chunks within 64 MiB of scratch; tr_mgpu_attr_exhaustive enumerates every coalition in aligned slices of 64 masks, each slice summing its weighted marginal-contribution terms locally (one counter per low player, one for the high players whose membership is constant on the slice, one player-independent counter) and adding each with one tr_atomic_add_wide (exact modulo 2^128, exact because the host proves 2 n! bound < 2^127 and 2^n bound < 2^127 first); tr_mgpu_attr_explore and tr_mgpu_attr_search run the shared drivers of math_attr.h with the values kernel as their batch evaluator. Results equal the CPU bit for bit (emulated with a scrambled thread order in the tests); supports and interactions are CPU only.

Bounds:

- Games: variables = players, tensor = 2^n values (predicate: x domain words), circuit = cone or program nodes, output = plan outputs, bytes = the object and scratch; table games n <= 24.
- Supports: support = the per-node term bound of the expansion (sum / product of the children's, at most 2^(players reached)), degree, bytes = every memo bound and the largest raw product; witnesses: tensor = terms^2 + sum 2^w (checked before the products); transform: tensor = 2^n, iterations.
- Indices: support = terms, output = players; interactions: tensor = output = the (set, term) candidates.
- Enumeration: values output = coalitions; exhaustive iterations = 2^n (n <= 32); explore / search iterations = max_coalitions (a smaller space returns exact results, a larger one certified bounds, never a refusal), tensor = the Moebius inversion, output; CUDA bytes = every device copy, the accumulators and the scratch chunk.

Implementation targets (proposed until the capability is validated):

- `torbles/math_attr.h`
- `torbles/math_attr.c`
- `torbles/math_attr_gpu.cu`
- `torbles/math_attr.py`
- `tests/test_math_attr.py`
- `bench/benchmark_math_attr.py`

Acceptance tests:

- Match brute-force small coalition values and contribution sum identities.
- State the baseline/game/weighting convention; algebraic attribution has no automatic causal interpretation.

Benchmark cases:

- Use 3/6/10/12 conditions with sparse versus dense interactions; compare supported contribution indices against exhaustive small coalition calculations.
- Vary explanation/repair search budgets and minimum-solution multiplicities; retain incomplete cases with their certified bounds and actual explored states.

Additional limits:

- Coalitions of the enumeration paths are 64-bit masks (n <= 62; exhaustive n <= 32, table games n <= 24); the sparse support path takes up to 2^24 players but its indices are 128-bit rationals (an overflow is reported, never wrapped).
- Provenance games inherit PROV-001's operators: selection, union, equi-join, group-by; values are multiplicity-weighted sums (COUNT, SUM with weights, existence counts), no MIN / MAX or negation.
- Explore's bounds are certified, not tight: only provenance games in N (weights >= 0) have the Moebius regime; the range regime widens by the worst case over the certified range per unexplored level; levels are explored whole in a fixed order.
- The minimum search explores levels in increasing size without pruning beyond the monotone extreme-value proof; the number of minimum solutions is exact only when the minimum level completes.
- Supports, interactions and the Moebius inversion are single-threaded CPU code; the CUDA paths are coalition values, the exhaustive accumulation and the batch evaluation of explore and search.
- No FND-007 certificate is emitted: the sum identities are checked natively and reported per result; FND-007's claim kinds are row relations of a scope or polynomial laws of at most 8 variables.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/ATTR-001/tests_test_math_attr.json](results/math/ATTR-001/tests_test_math_attr.json)
- tests: [results/math/ATTR-001/sanitized_test_math_attr.json](results/math/ATTR-001/sanitized_test_math_attr.json)
- benchmarks: [results/math/ATTR-001/benchmark_cpu_math_attr_vm.json](results/math/ATTR-001/benchmark_cpu_math_attr_vm.json)

<a id="attr-002"></a>

### ATTR-002: Bounded explanation and repair search

Status: **in_progress**. Delivery wave: **W5**.

Search for redundant predicates, minimal distinguishing sets and least-cost feasible repairs with certificates where tractable.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [ATTR-001](#attr-001), [INV-002](#inv-002)

Native CPU: Native C (math_repair.c; the evaluators, bounds and search drivers inline in math_repair.h): immutable subset problems over at most 62 elements with an exact feasibility test per subset: hitting sets of inclusion-minimal masks (declared; the minimal difference sets of the distinct three-valued or SQL-selection patterns of two row sets of a BOOL-001 domain, or of a program's selected and other points: minimal distinguishing sets and membership explanations; the minimal failure sets of a conjunction's rejected points: keep sets whose complements are jointly redundant), a BOOL-001 program with the conditions outside S at declared baseline constants that must select exactly the same points (SQL three-valued logic, the exact count of differing points), an ATTR-001 game (v(S) or v(P - S) against an exact threshold), or deletions of declared rows under COUNT(*), COUNT, SUM or AVG targets with SQL NULL semantics (exact 128-bit sums, magnitude bounds proven at construction). Searches over positive exact rational costs scaled to integers: exhaustive in increasing (cost, mask) order, or best-first branch and bound with admissible bounds (a disjoint-mask packing for hitting sets, the exactly rounded fractional relaxation for deletions) and sound pruning only, in index, cost or exact ATTR-001 Shapley / Banzhaf order; results found (the exact minimum, every tie counted and listed up to a limit), none (proven) or incomplete when the declared state budget stops the search (proven lower and upper bounds, multiplicity bounds when the minimum is known, explored states and open nodes). Redundancy of every allowed condition with exact counts and witness rows. Continuous repairs through INV-002: value edits by its exact LP or projection, single-threshold changes by its aggregate state, the new constants computed natively.

GPU: math_repair_gpu.cu: tr_mgpu_repair_eval evaluates a batch of candidate subsets, one work item per candidate writing its exact counts and value into its own slot (hitting masks, word-parallel BOOL-001 programs in three-valued logic with baselines, ATTR-001 game values, deletion aggregates); tr_mgpu_repair_scan marks the feasible candidates by atomic OR, takes the smallest feasible cost by atomic minimum and then the ties of that cost by atomic OR; tr_mgpu_repair_search runs the shared exhaustive and branch-and-bound drivers of math_repair.h with these kernels as its evaluators. Results do not depend on the order of the threads and equal the CPU bit for bit (emulated with a scrambled thread order in the tests); the output and device-byte budgets are checked on the host before launching. Continuous repairs and the ATTR-001 orders are CPU only.

Bounds:

- Problems: variables = elements; tensor = pattern pairs (separations, explanations); support = the masks kept (declared masks: the copy before minimization); circuit = program nodes; bytes = the copies.
- Evaluation: output = candidates, bytes (CUDA: every device copy and the scratch chunk).
- Search: iterations = max_states (a larger space gives an incomplete result with proven bounds, never a refusal), output = max_solutions, bytes = the frontier, the batches and the sorted order; Shapley / Banzhaf orders add ATTR-001's needs. Redundancy: output = elements x (1 + witnesses). Thresholds: variables = conditions, output = changeable conditions.

Implementation targets (proposed until the capability is validated):

- `torbles/math_repair.h`
- `torbles/math_repair.c`
- `torbles/math_repair_gpu.cu`
- `torbles/math_repair.py`
- `tests/test_math_repair.py`
- `bench/benchmark_math_repair.py`

Acceptance tests:

- Compare small searches with exhaustive minima including all ties.
- Budget exhaustion is reported as incomplete with any proven lower/upper bounds.

Benchmark cases:

- Use 3/6/10/12 conditions with sparse versus dense interactions; compare supported contribution indices against exhaustive small coalition calculations.
- Vary explanation/repair search budgets and minimum-solution multiplicities; retain incomplete cases with their certified bounds and actual explored states.

Additional limits:

- Subsets are 64-bit masks: at most 62 elements; a weighted exhaustive search sorts its 2^m subsets (m <= 20).
- The search is exponential in the worst case: admissible bounds exist for hitting sets and linear deletion targets only; circuits, games and != targets branch with the generic bound (the cheapest undecided element), so large ones end incomplete under a state budget, with proven bounds.
- Continuous repairs (value edits, threshold changes) are CPU only, through INV-002, and change one declared threshold at a time; the CUDA path is the batched candidate evaluation and scan.
- No FND-007 certificate kind is emitted for minimality: optimality is proven by the exhaustive order or by the bounds and redundancy by exact counts; every result is algebraic, never causal.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/ATTR-002/tests_test_math_repair.json](results/math/ATTR-002/tests_test_math_repair.json)
- tests: [results/math/ATTR-002/sanitized_test_math_repair.json](results/math/ATTR-002/sanitized_test_math_repair.json)
- benchmarks: [results/math/ATTR-002/benchmark_cpu_math_repair_vm.json](results/math/ATTR-002/benchmark_cpu_math_repair_vm.json)

## PROB: Probability and inference

Small-support distributions, partition functions and factorized inference.

<a id="prob-001"></a>

### PROB-001: Small-support partition and likelihood functions

Status: **in_progress**. Delivery wave: **W4**.

Evaluate declared finite-support probability models, normalizers, likelihood gradients and Hessians from retained statistics.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [CALC-003](#calc-003), [FND-009](#fnd-009)

Native CPU: Native C (math_prob.c; the per-point terms inline in math_prob.h): declared supports (1..3 coordinates, logical int64 values or NULL, sorted natively into DIST-001 cell order, duplicates refused, exact-content identity); models with features T stored as int64 numerators over per-feature lcm denominators: categorical (weights or exact probabilities per request, one-hot natural parameters when no features) and exponential families with a rational base measure h >= 0 (zero everywhere refused). Retained statistics: a DIST-001 histogram (dense or sparse, count or nonnegative weight masses, declared listwise NULL exclusion) aligned to the declared support without reading a row: per group the mass of every declared point (zeros kept), N, rows, dropped mass; an observed cell outside the support is refused naming it; the state records the support, the histogram's scope digest and GEN-002's data source and grouping. Products of two states (the CROSS JOIN within groups over the product support) only under a GEN-002 declaration covering a shared source (tr_conv_decl_covers; reasons needs_declaration, not_covered, grouping_differs, keys_differ). Plans validate every request (own status) and compute CALC-003's 64.192 reduction constants theta_j / (D_j ln 2) from floor(2^512 / ln 2); sums per request: categorical exact 128-bit W, W T, W T T sums (bit-length proof, else OverflowError) and DIST-003-logarithm bounds of sum n ln W at a proven scale; exponential families an exact maximum of the binary exponents n_i + bitlen(H_i), then CALC-003's tr_calc_exp2_reduce per point, (H_i / 2^bitlen) m_i 2^(n_i + bitlen - n_ref) in 120-bit midpoint-radius fixed point, outward floor / ceil integers of S_0, S_j, S_jl at scales s = min(120, 124 - bitlen(d)) - b_j - b_l that prove every partial sum below 2^126. The finish (shared by both backends, sums checked against the proofs) gives Z (exact form [zlo, zhi] 2^zexp / zden and outward doubles, [DBL_MAX, inf] past the range), log Z and the transcendental part of the log-likelihood as exact 256-bit bounds at 2^-120 (a 128-bit extension of DIST-003's logarithm: hi - lo <= E + 149 ulps), the log-likelihood as -inf with a flag when a zero-probability point has mass (never a finite number), and the gradient Tobs - N E[T] and Hessian -N Cov[T] with exact 256-bit numerators from the corners of the sums' boxes (exact rationals for categorical models); tr_prob_exact gives the exact rational bounds of every entry with FND-009 big integers. Every budget is checked before allocation.

GPU: math_prob_gpu.cu, tr_mgpu_prob_sums: one work item per (request, 64 support points). Exponential families: launch 1 computes n_i + bitlen(H_i) (the integer part of CALC-003's reduction) with a local maximum and one tr_atomic_max_i64 per request; launch 2 the CALC-003 exponentials, the outward integers of every wanted sum term and local 128-bit sums flushed with one exact two-word tr_atomic_add_wide per bound. Categorical models: one launch of the exact sums and the DIST-003 log bounds with the same atomics. Exponent overflows and zero-probability points with mass are flags set by atomic OR. Every term is an exact integer and every partial sum is proven below 2^126, so the device sums, reference exponents and flags equal the CPU's bit for bit whatever the thread order (emulated with a scrambled order in the tests); the host finishes both with the same tr_prob_finish.

Bounds:

- Support: variables = k, support = d, bytes; models: variables = m, support = d, tensor = d m, bytes; statistics: support = d, states = G d, bytes (with the support and the source descriptor); products: support = da db, states = G da db, bytes.
- Evaluation: output = n x (3 + gradient entries + Hessian entries), tensor = n d, states = n x sums, bytes = the plan (680 bytes of reduction constants per exponential request, 8 d per categorical request), the sums and the finish scratch; CUDA: bytes = every device copy, states = n x sums; exact bounds: bytes of every big integer.
- Refusals before allocation (AlgebraBudgetError with resource, need, limit); sizes past 64 bits are refused whatever the budget; at most 3 coordinates, 16 features, 2^32 points (UnsupportedError).

Implementation targets (proposed until the capability is validated):

- `torbles/math_prob.h`
- `torbles/math_prob.c`
- `torbles/math_prob_gpu.cu`
- `torbles/math_prob.py`
- `tests/test_math_prob.py`
- `bench/benchmark_math_prob.py`

Acceptance tests:

- Normalization, gradients and Hessians match enumerated small distributions.
- Zero mass, support changes and dependent events preserve the declared model semantics.

Benchmark cases:

- Use support sizes 8/64/1,024 and 1/100/10,000 parameter requests; compare retained partition/likelihood states with repeated support enumeration.
- For factorized inference vary graph size and elimination width on brute-force-checked tiny instances and bounded larger models; report intermediate support and precision.

Additional limits:

- Features, base measures and parameters are rationals whose common denominators and numerators fit int64 (OverflowError otherwise); categorical sums need bitlen(d) + bitlen(max W) + bitlen(max |feature product|) <= 126, a conservative bit-length proof that can refuse admissible requests.
- Exponents are limited to CALC-003's 2^(2^24) range (OverflowError per request beyond); double enclosures of Z leave the double range as [DBL_MAX, inf] or [0, 2^-1074] (the exact form and log Z stay finite); the double enclosures of gradients and Hessians add several ulps to the exact bounds.
- Interval bounds of the gradient and Hessian treat the sums as independent boxes (dependency not exploited): rigorous, about 2^-100 relative to the moments, wider in relative terms when a covariance nearly cancels; a variance's upper bound is clamped at 0.
- Weight masses are logical integers (scaled by 10^wscale) with a group total below 2^64; the statistics are dense per group over the declared support (G x d masses).
- Products of statistics combine two states of 1..2 coordinates into at most 3, within groups (the CROSS JOIN); the independence declaration is the caller's statement, never verified against the data, and states without a data source (merged histograms) cannot be multiplied.
- No FND-007 certificate is emitted: the support containment is checked natively at construction and reported as a refusal (FND-007 has no claim kind for membership in a finite set).
- The benchmark's second case (factorized inference: graph size, elimination width, intermediate support and precision) belongs to PROB-002 and is not measured here.

Validation receipts:

Measured runtime: `40e4d463b0e30e7a2e78fde12ae5c7831010b78f`.

- tests: [results/math/PROB-001/tests_test_math_prob.json](results/math/PROB-001/tests_test_math_prob.json)
- tests: [results/math/PROB-001/sanitized_test_math_prob.json](results/math/PROB-001/sanitized_test_math_prob.json)
- benchmarks: [results/math/PROB-001/benchmark_cpu_math_prob_vm.json](results/math/PROB-001/benchmark_cpu_math_prob_vm.json)

<a id="prob-002"></a>

### PROB-002: Factorized probabilistic inference

Status: **in_progress**. Delivery wave: **W5**.

Use bounded factor graphs/circuits and costed elimination to compute marginals or selected best assignments.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROB-001](#prob-001), [TENSOR-002](#tensor-002), [COUNT-002](#count-002)

Native CPU: Native C (math_factor.c; the elimination step tr_factor_cell inline in math_factor.h): factor graphs of n <= 4096 variables with declared finite supports (1..2^16 logical values or the NULL category, distinct, with a decimal scale) and factors over scopes of <= 31 variables built from explicit nonnegative rationals, one group of a dense or sparse TENSOR-001 tensor (count axes mapped by value with equal scales, index axes by index; a nonzero cell outside the declared values refused with the factor, axis and value), a PROB-001 categorical model with one request's weights or exact probabilities (support points mapped by value, every other cell 0) or one group of PROB-001 retained statistics; each factor scaled to integers over the lcm of its denominators (< 2^126); data factors record their scope's source (GEN-002's rule) and two of one source are admitted only under a covering GEN-002 independence declaration (refused otherwise with the pair, the source digest and the reason). Query shapes (kept variables in order, evidence variables, sum / max / k-best, R requests) are planned by simulating bucket elimination for each candidate order (an exact dynamic program over the subsets of the eliminated variables, at most 14 by default and 64 free variables, minimizing the total joint cells among the orders whose every step fits the degree / tensor / support budgets and proving infeasibility for every order otherwise; min-fill, min-degree, min-weight, weighted min-fill, natural, reverse and a given order), each on the CPU or the device, with exact per-candidate width (the induced width), largest joint table, largest intermediate table, products, arena, host and device bytes and limbs, costed by component and chosen by FND-008's tr_planner_choose (the cheapest admissible, the first on a tie; a force runs exactly its candidate or refuses); refusals carry the evidence of every candidate, whether they are proven for every order (the DP or a minor-min-width / first-step / output lower bound past its limit) and the exact minima or lower bounds of width, joint and support. Programs run every step as dense messages in a per-request arena: sums (and, for max, the maxima, the minimum index reaching them and the number of maximizing completions) of products of operand entries over the eliminated variable's values in exact L-limb integers (COUNT-002's arithmetic; L proven from the factors' maxima and the domain sizes), evidence by per-request base offsets; answers are adopted natively from the CPU or device arena: Z_e and Z_e / D, P(e) = Z_e / Z from the retained normalizer, marginal tables P(q | e) over any kept variables, conditional tables P(q | r, e) (undefined where P(r | e) = 0), the maximum weight, its probability, the exact number of maximizers and every tied maximizer (decoded through the tight values, sorted lexicographically, refused past the output budget with the exact count), and the k best by lists of the top k with ties per message cell decoded through backpointers (canonical order: weight decreasing, then lexicographic; the ties beyond k counted); all values canonical rationals with FND-009 big integers. Alternative backends of the weighted model count Z_e: COUNT-002's variable elimination and brute force over the restricted tables, a COUNT-002 decision diagram of the parameter encoding of binary graphs counted with literal weights, and TENSOR-002's costed contraction of the dense scaled tables.

GPU: CUDA kernel math_factor_gpu.cu: tr_mgpu_factor_eliminate runs an elimination program with one launch per step (later steps read earlier messages) and one work item per (request, output cell) evaluating the shared tr_factor_cell into its own slots: exact L-limb sums of the products of 128-bit scaled input entries and L-limb messages over the eliminated variable's values and, for max programs, the maximum, its minimum index (argmax ties broken by the minimum index) and the count of maximizing completions; no atomics on values (an overflow flag by atomic OR, excluded by the host's L proof), 7 L words of local scratch per item (L <= 16 on the device, else the CUDA candidate is rejected); the host checks the program, every request's evidence bases and the budget (bytes = every device copy, states = R x arena) before allocating; the raw arena equals the CPU's bit for bit in any thread order and is adopted by the same native answer code.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_factor.h`
- `torbles/math_factor.c`
- `torbles/math_factor_gpu.cu`
- `torbles/math_factor.py`
- `tests/test_math_factor.py`
- `bench/benchmark_math_factor.py`

Acceptance tests:

- Compare tiny correlated graphs with exhaustive joint enumeration.
- Treewidth/intermediate-support budgets govern execution; marginals are never multiplied as an undeclared independence assumption.

Benchmark cases:

- Use support sizes 8/64/1,024 and 1/100/10,000 parameter requests; compare retained partition/likelihood states with repeated support enumeration.
- For factorized inference vary graph size and elimination width on brute-force-checked tiny instances and bounded larger models; report intermediate support and precision.

Additional limits:

- Factors are exact nonnegative rational tables (explicit, count tensors, categorical models, retained masses): exponential-family factors (certified enclosures) are refused, and requests past every budget are refused, never answered with certified bounds or an extrapolation.
- The exact subset DP that proves a refusal for every order runs for at most 14 eliminated variables by default (dp_limit up to 24) and 64 free variables, within the iterations budget; beyond it the heuristic candidates may miss an order that fits, and a refusal is then proven only when a lower bound (minor-min-width treewidth bound, the first step's table, the output table) exceeds its limit.
- Refusals report FND-008's first over-budget candidate (resource, need, limit); the plan evidence lists every candidate's needs, and the minima are exact only when the DP ran.
- Every message of a program is kept in the arena (the decoders read them): memory is the whole arena per request; the CPU loops are single-threaded; the CUDA kernel holds at most 16 limbs (1024-bit values) per work item.
- Evidence fixes one declared value per observed variable (no soft evidence); the maximum and the k best are over every unobserved variable (no marginal MAP); the k best and every decode run on the CPU.
- Data factors take one group's state; the independence declaration is the caller's statement, never verified against the data; products of PROB-001 statistics are refused as factors.
- Alternative backends are bounded by their own capabilities: the decision diagram needs binary variables and at most 64 program variables (variables plus parameters), TENSOR-002 at most 8 tables over 16 variables with scopes of 8 and every contraction step's bound below 2^127, COUNT-002 elimination int64 rational entries.
- Cost rates are hand-set relative estimates (not calibrated); the CUDA timings need --device on an actual GPU.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/PROB-002/tests_test_math_factor.json](results/math/PROB-002/tests_test_math_factor.json)
- tests: [results/math/PROB-002/sanitized_test_math_factor.json](results/math/PROB-002/sanitized_test_math_factor.json)
- benchmarks: [results/math/PROB-002/benchmark_cpu_math_factor_vm.json](results/math/PROB-002/benchmark_cpu_math_factor_vm.json)

## DELTA: Update calculus

Signed changes, cross terms and maintained algebraic states.

<a id="delta-001"></a>

### DELTA-001: Signed linear state updates

Status: **in_progress**. Delivery wave: **W2**.

Merge insert/delete/weight-change deltas into compatible counts and moment states with revision/proof invalidation.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: Native C (math_delta.c): insert, delete, weight-change and group-move deltas merged exactly into FND-004 bank states (counts, per-monomial counts and 128-bit sums, with an overflow proof), producing a new immutable state whose source identity records the update so old proofs and cache entries are not reused; a negative multiplicity in a state that promises a nonnegative measure is refused with the group and need.

GPU: CUDA kernel math_delta_gpu.cu: a delta batch applied to the states on the device (signed atomic adds of counts and 128-bit sums), equal to the CPU, with the nonnegativity check exact before the new state is exposed.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing moved-dimension delta-plane split executes additively.
- [tests/check_moves.py](tests/check_moves.py): Existing delta/move comparisons; no general maintained mathematical-state bank.

Implementation targets (proposed until the capability is validated):

- `torbles/math_delta.h`
- `torbles/math_delta.c`
- `torbles/math_delta_gpu.cu`
- `torbles/math_delta.py`
- `tests/test_math_delta.py`
- `bench/benchmark_math_delta.py`

Acceptance tests:

- Random signed update sequences match complete state reconstruction after every step.
- Reject invalid negative multiplicities when the public state promises a nonnegative empirical measure.

Benchmark cases:

- Apply insert/delete/weight-change batches of 1/10/1,000 rows to 1k/100k/1M-row states; separate initial build from each maintained update.
- Vary one versus both changing sides of bilinear joins and group-move frequency; compare every update with complete reconstruction, retaining rebuild/refusal cases.

Additional limits:

- The nonnegativity check sees aggregates only: deleting a row that is absent but matches the group, validity and values of present rows cannot be detected.
- The overflow proof is conservative (a net-zero swap of two large rows may be refused; a rebuild is the fallback); each update copies the whole state.
- Bilinear (join) deltas are DELTA-002.

Validation receipts:

Measured runtime: `85e58b523446b835f1f9d4c462e6a43402704d06`.

- tests: [results/math/DELTA-001/tests_test_math_delta.json](results/math/DELTA-001/tests_test_math_delta.json)
- tests: [results/math/DELTA-001/sanitized_test_math_delta.json](results/math/DELTA-001/sanitized_test_math_delta.json)
- benchmarks: [results/math/DELTA-001/benchmark_cpu_math_delta_vm.json](results/math/DELTA-001/benchmark_cpu_math_delta_vm.json)

<a id="delta-002"></a>

### DELTA-002: Bilinear/join and nonlinear update laws

Status: **in_progress**. Delivery wave: **W3**.

Carry all finite cross terms and typed nonlinear-finalizer state; distinguish update calculus from an infinitesimal derivative.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DELTA-001](#delta-001)

Native CPU: Native C (math_bilinear.c, on DELTA-001 and FND-004): maintained per-key bilinear join states updated on either or both sides with all finite cross terms, Delta(AB) = Delta(A) B + A Delta(B) + Delta(A) Delta(B), exact with proven bounds; changing join multiplicities through the per-key states; nonlinear finalizers with their own retained state (per-value multiplicities for DISTINCT, the multiset or a rebuild for MIN/MAX, sufficient sums for ratios and variances); documented as finite differences of exact states, not derivatives.

GPU: CUDA kernels math_bilinear_gpu.cu: batched exact cross-term updates per key with wide atomics, and the matrix-product delta, equal to the CPU.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_bilinear.h`
- `torbles/math_bilinear.c`
- `torbles/math_bilinear_gpu.cu`
- `torbles/math_bilinear.py`
- `tests/test_math_bilinear.py`
- `bench/benchmark_math_bilinear.py`

Acceptance tests:

- Verify Delta(AB)=Delta(A)B+A Delta(B)+Delta(A)Delta(B) on exact small joins/matrices.
- DISTINCT, MIN/MAX and changing join multiplicities request their own retained state rather than inheriting SUM deletion rules.

Benchmark cases:

- Apply insert/delete/weight-change batches of 1/10/1,000 rows to 1k/100k/1M-row states; separate initial build from each maintained update.
- Vary one versus both changing sides of bilinear joins and group-move frequency; compare every update with complete reconstruction, retaining rebuild/refusal cases.

Additional limits:

- Two-sided joins only, over dense key ids shared by both sides (no key dictionary); side states dense over keys x groups.
- COUNT DISTINCT and MIN/MAX are maintained for single-table value states (a join result raises DomainError); the min/max-only state refuses with RebuildRequired when the last extreme is deleted.

Validation receipts:

Measured runtime: `81a5294ad96d447c8ffe8361cdd27c24b64f4863`.

- tests: [results/math/DELTA-002/tests_test_math_bilinear.json](results/math/DELTA-002/tests_test_math_bilinear.json)
- tests: [results/math/DELTA-002/sanitized_test_math_bilinear.json](results/math/DELTA-002/sanitized_test_math_bilinear.json)
- benchmarks: [results/math/DELTA-002/benchmark_cpu_math_bilinear_vm.json](results/math/DELTA-002/benchmark_cpu_math_bilinear_vm.json)

## UNC: Certified uncertainty

Bounds, abstract interpretation and explicit approximation budgets.

<a id="unc-001"></a>

### UNC-001: Intervals, affine forms and congruence reasoning

Status: **in_progress**. Delivery wave: **W3**.

Native abstract states propagate sound bounds, correlations, residues, nullness and exact-zero facts for pruning and overflow decisions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009), [FND-007](#fnd-007)

Native CPU: Native abstract states over the groups of an FND-004 scope (per group and column: selected rows, present rows, logical min and max, gcd of the differences) and expression programs (up to 64 nodes over up to 32 columns: column, constant, +, -, *, negation, division by a nonzero constant, min, max, abs, square, mod by a positive constant, the six comparisons, AND, OR, NOT, IS NULL, IS NOT NULL). Each numeric node carries a reduced product: nullness (SQL propagation), an interval with exact int64-rational bounds (overflow widens soundly and is flagged) or FND-009 outward-rounded doubles, an affine form over shared noise symbols (one per column, one approximation symbol per node holding its nonlinear remainder, rounding errors and condensed terms, so x - x = 0 and a node minus itself is 0) with exact rational or rigorously bounded double coefficients truncated to a chosen precision, a rank limit (terms condensed into the node's symbol) and an error budget (the form collapses past it), a rational congruence r + mZ with Granger's rules and the CRT meet (GF(p) as m = p), and a zero fact; boolean nodes carry their set of SQL outcomes. Comparisons decide d = a - b as a form intersected with the interval difference and the class of d. Per group a boolean root is decided ALL, NONE, UNKNOWN or EMPTY; the filter keeps ALL groups, prunes NONE groups and evaluates the rows of UNKNOWN groups exactly (int64 rationals with 128-bit intermediates, SQL NULLs; an undecidable row is refused as overflow). Also: every node's value for a group (trace), the symbolic affine form, the overflow decision of a SUM over a group in a given accumulator width and unit, FND-007 LAW certificates (ring identities D E - D L = 0 of exact affine identities, canonicalized by FND-007) and SNAPSHOT certificates of proved NOT NULL, congruence modulo a prime (GF(p) polynomial claims) and constant facts, re-verified by FND-007 on every selected row; freshness through the scope's FND-007 record. Budgets (circuit, variables, support, states, output, bytes) are checked before allocation.

GPU: tr_mgpu_unc_eval: one work item per group evaluates the program through every node with its values in its own slots of a device scratch (chunks of 4096 groups) and writes its own result slot; tr_mgpu_unc_rows: one work item per 64-row word writes its own bitmap word (rows of groups decided by the proof set or cleared, the others evaluated exactly) and adds its counts by atomics, the first overflowing row by atomic minimum. Both run the inline functions of math_unc.h (the double operations are FND-009's contraction-safe directed operations), so they equal the CPU bit for bit; the host validates programs and inputs and budgets the device copies before launching. Compiled with nvcc for sm_80 and tested in the emulation with scrambled thread order; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_unc.h`
- `torbles/math_unc.c`
- `torbles/math_unc_gpu.cu`
- `torbles/math_unc.py`
- `tests/test_math_unc.py`
- `bench/benchmark_math_unc.py`

Acceptance tests:

- Concrete exhaustive small values remain enclosed by each abstract operation.
- Affine correlation preserves x-x=0 where independent intervals would lose it; unsound pruning is prohibited.

Benchmark cases:

- Use correlated cancellation, interval thresholds and modular constraints with 1%/50%/99% ambiguous regions; report proof/pruning work and concrete fallback work separately.
- Vary coefficient precision, approximation rank and error budget; compare certified bounds with exhaustive small or high-precision oracles and record unproved cases.

Additional limits:

- Comparison outcomes are combined as sets: AND / OR of correlated predicates (x < 5 AND x >= 5) are not decided.
- The inputs are per column (no joint information between columns); the correlation kept is the algebraic one of shared nodes and columns.
- Congruence tightening of interval bounds is applied in the exact carrier only; binary64 intervals keep full precision while form coefficients may be truncated.
- Exact arithmetic is int64 rationals with 128-bit intermediates: past it bounds widen (flagged) and concrete rows are refused as overflow, never wrapped.
- FND-007 has no inequality claim kind: interval bounds and pruning decisions are tied to the scope's record (freshness), not issued as certificates; law certificates need polynomial nodes over at most 8 columns and int64 coefficients.
- MIN / MAX propagate NULL (unlike SQL LEAST / GREATEST); division and mod take constant divisors only.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `246664aa793bcecefac4cd954e6366a1a85271f8`.

- tests: [results/math/UNC-001/tests_test_math_unc.json](results/math/UNC-001/tests_test_math_unc.json)
- tests: [results/math/UNC-001/sanitized_test_math_unc.json](results/math/UNC-001/sanitized_test_math_unc.json)
- benchmarks: [results/math/UNC-001/benchmark_cpu_math_unc_vm.json](results/math/UNC-001/benchmark_cpu_math_unc_vm.json)

<a id="unc-002"></a>

### UNC-002: Explicit approximation and error certificates

Status: **in_progress**. Delivery wave: **W4**.

Approximate transforms carry declared norms, assumptions, error budgets and deterministic/randomized provenance.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [UNC-001](#unc-001)

Native CPU: Native immutable approximation objects that record the declared norm (absolute, relative, L2, Frobenius, spectral, max entry), the assumptions (domain interval, derivative and Lipschitz bounds, certified range, independence of random probes, no overflow), the error budget, the error spent term by term (kind, the id of the approximation it came from, scaled or not; the error is their exact sum) and the provenance (deterministic, or randomized with the seed and the failure probability, label probabilistic). Series of exp, log, sin, cos, sqrt and 1/x on [a, b] with exact rational endpoints: Taylor coefficients at the midpoint enclosed in 224-bit dyadic interval arithmetic (argument reduction and squaring / double angles, log by atanh, sqrt by integer square roots), the Lagrange remainder (and the integral form for log, sqrt, 1/x), then either monomials in t = (x - mid) / rad or Chebyshev economization of a Taylor series of degree min(n + 24, 2n + 8) with the dropped coefficients added; representatives on 192-bit dyadics with the enclosure radius added; certified range, relative bound when the function has no zero, L2 bound. Exact polynomials; sums (errors add), scalings, products (|f| e_g + |g| e_f + e_f e_g, truncation past degree 64 as its own term) and compositions (the inner range widened by its error must lie in the outer interval; L_outer e_inner + e_outer) never drop a term; quantization of the coefficients to multiples of 2^-bits with the induced error; binary64 coefficients in x with the coefficient rounding, the Horner bound gamma_2n sum |a_j| B^j and an underflow term. Low rank of integer matrices up to 64 x 64: cross approximation by fraction-free Bareiss elimination with complete pivoting (the exact Schur complement residual: exact Frobenius square, Frobenius, min(F, sqrt(norm1 norminf)) spectral, max-entry bounds; exact factorization at full rank) and a seeded Rademacher sketch with exact rational projection whose Frobenius bound 2 max_i ||E w_i|| holds except with probability (13/16)^probes. A requested error budget that the certified bound exceeds is refused (TR_MATH_NOT_CONVERGED), never reported as certified. Per-row evaluation over an FND-004 scope column (binary64 enclosure of x checked inside the domain, interval Clenshaw or Horner with the coefficient enclosures widened by the error and clamped to the range; NULL, out-of-domain and unselected rows reported apart) with exact group sums of floor / ceil of the enclosures in 2^-F units in 128 bits after an overflow proof; approximate predicates f(x) op T over a UNC-001 state (groups decided ALL / NONE by the binary64 UNC-001 proof of the polynomial with dyadic coefficients and the threshold moved by the error, the other rows by their enclosure, undecidable rows counted undecided in their own bitmap); tr_approx_authorize_exact always refuses with the FND-007 reason approximate (statistical for a probabilistic label); FND-007 snapshot certificates of the approximate relationship y ~ p(x) as a POLYNOMIAL claim with a tolerance (certified error, coefficient rounding, data tolerance), which FND-007 refuses to authorize. Budgets (degree, support, tensor, iterations, output, bytes) are checked before allocation.

GPU: tr_mgpu_approx_eval: one work item per 64-row word evaluates each selected row with the inline row evaluator of math_approx.h (the binary64 enclosure of x, interval Clenshaw / Horner with the coefficient enclosures, the certified error and range clamp; or the binary64 Horner value with its bound), writes the rows' own output slots, and adds floor(lo 2^F) and ceil(hi 2^F) of its rows into a local 128-bit accumulator per run of equal group ids, flushed with the two-word 128-bit atomic add (counts with 64-bit atomics), so the sums do not depend on the order of the threads; the host proves the sums fit, validates the view and budgets the device copies before launching. Equal to the CPU bit for bit (tested in the emulation with scrambled thread order); compiled with nvcc for sm_80 without warnings; not yet run on a device.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_approx.h`
- `torbles/math_approx.c`
- `torbles/math_approx_gpu.cu`
- `torbles/math_approx.py`
- `tests/test_math_approx.py`
- `bench/benchmark_math_approx.py`

Acceptance tests:

- Measured small-oracle error satisfies the advertised applicable bound.
- Approximate candidates cannot authorize exact relationship rewrites or silently drop uncertainty during composition.

Benchmark cases:

- Use correlated cancellation, interval thresholds and modular constraints with 1%/50%/99% ambiguous regions; report proof/pruning work and concrete fallback work separately.
- Vary coefficient precision, approximation rank and error budget; compare certified bounds with exhaustive small or high-precision oracles and record unproved cases.

Additional limits:

- Functions are exp, log, sin, cos, sqrt, 1/x and exact polynomials of one variable on a bounded rational interval (|a|, |b| <= 1024 for exp, sin, cos); no other special functions and no multivariate approximations.
- Chebyshev series are economized Taylor expansions at the midpoint, not best (minimax) approximations; near a singularity (log or 1/x close to 0, long intervals of sin / cos) the certified bound may exceed a budget the true error meets, which is refused.
- Coefficient representatives are 192-bit dyadics: certified errors do not go below about 2^-190 times the coefficients' size.
- Low-rank methods are exact methods for small integer matrices (up to 64 x 64): cross approximation is not the optimal SVD truncation, the spectral bound is min(F, sqrt(norm1 norminf)), and the sketch bound is probabilistic (a conservative Paley-Zygmund argument), labelled so and never certified.
- The approximate filter runs on the CPU and proves series only (the binary64 form is evaluated per row); rows whose binary64 enclosure of x leaves the domain (a non-dyadic endpoint) are refused as out of domain.
- Compositions need the inner range plus its error inside the outer interval and an internal degree <= 256; products are truncated to degree 64 with a truncation term.
- FND-007 relationship snapshots need scaled integer coefficients (degree <= 16, decimal scales sx n + sy <= 18); they are approximate claims with a tolerance, never exact laws.
- The CUDA kernel has been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `6ff26a49d485b3aad9a0daa2606659900a2c2a09`.

- tests: [results/math/UNC-002/tests_test_math_approx.json](results/math/UNC-002/tests_test_math_approx.json)
- tests: [results/math/UNC-002/sanitized_test_math_approx.json](results/math/UNC-002/sanitized_test_math_approx.json)
- benchmarks: [results/math/UNC-002/benchmark_cpu_math_approx_vm.json](results/math/UNC-002/benchmark_cpu_math_approx_vm.json)

## META: Queries about queries

Equivalence, implication, containment and shared query-family computation.

<a id="meta-001"></a>

### META-001: Scoped query equivalence and counterexamples

Status: **in_progress**. Delivery wave: **W3**.

Prove supported equivalence/implication/containment on a declared snapshot or universal domain, or return concrete counterexamples.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-007](#fnd-007), [BOOL-001](#bool-001), [GEOM-001](#geom-001)

Native CPU: tr_equiv_set_make hash-conses predicates (BOOL-001 conditions with reduced constants, at most 64 distinct, and nodes with ordered commutative operands: one shared DAG), canonical rational measures and queries, and reports the common subexpressions; tr_equiv_check_make / tr_equiv_check_run decide questions (predicate equivalence under FILTER or Kleene semantics, implication, measure equivalence under SQL or SKIP NULL semantics, query equivalence and containment) on every selected row of a FND-004 scope: the conditions once as BOOL-001 masks, each question's cone word-parallel, measures through GEOM-001's exact pair candidate P = D (f - g); tr_equiv_check_decide labels the verdicts SNAPSHOT (an FND-007 record of the scope; for measures the snapshot certificate of P = 0 adopted from the evidence) or REFUTED with the smallest differing row, reduces measure questions modulo relation certificates first (GEOM-001 complete Groebner basis: label RELATIONS with the derived certificate) and adopts device evidence only after consistency checks (counts, truth cells, the witness re-evaluated). tr_equiv_decide never reads rows: exhaustive enumeration of a declared product, or decision procedures (per NULL pattern, reduced ordered BDDs of the TRUE / FALSE sets and every cube of their difference checked for an integer point by bounds and Bellman-Ford difference constraints with branching; measures by exact normal forms, a FND-007 RING_IDENTITY law when P = 0, NULL-pattern mismatches and Nullstellensatz grids), every counterexample re-evaluated on a synthetic one-row scope; over-budget or unsupported questions are UNKNOWN with the reason. tr_equiv_authorize: universal always, snapshot only on the exact recorded scope, domain only on a scope inside the domain, relations as FND-007 authorizes. tr_equiv_family_run computes COUNT and exact SUM per query with one shared scan or one scan per query and counts the work actually done; tr_equiv_rewrite lowers a degree <= 2 integer measure to stored codes, runs relationships_optimizer.c's tr_optimize and lifts the result back exactly as a candidate to check.

GPU: tr_mgpu_equiv_check runs a check's questions on the device: work items (question, word block) evaluate the question's cone on the BOOL-001 condition words (the masks themselves from tr_mgpu_bool_masks) with Kleene's connectives and, for measures, GEOM-001's exact W-word pair candidates at the rows both sides read, add their counters (checked, excluded, differing, undecided, NULL mismatches, NULL rows, nine truth cells) with integer atomics and the first witness with an atomic minimum (the minimum row id); no barriers or shared memory, evidence equal to tr_equiv_check_run bit for bit, adopted by tr_equiv_check_decide after consistency checks.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships_optimizer.c](torbles/relationships_optimizer.c): Existing bounded equivalent-polynomial search under verified rules.
- [torbles/relationships_stage.c](torbles/relationships_stage.c): Existing exact bounded stage decompositions; no general query theorem prover.

Implementation targets (proposed until the capability is validated):

- `torbles/math_equiv.h`
- `torbles/math_equiv.c`
- `torbles/math_equiv_gpu.cu`
- `torbles/math_equiv.py`
- `tests/test_math_equiv.py`
- `bench/benchmark_math_equiv.py`

Acceptance tests:

- Deliberately different predicates/measures produce a witness on the applicable domain.
- Snapshot proof labels do not become universal model claims; unsupported universal proofs remain unknown.

Benchmark cases:

- Compare families of 1/8/32/128 related queries with shared and disjoint filters/measures; report proof/common-subexpression construction and actual shared scans.
- Use equivalent, one-row-different and universally-different query pairs; separate snapshot checking from universal proof search and retain unknown/budget-limited outcomes.

Additional limits:

- decide mode handles predicates over integer-range columns only (finite value sets need enumerate mode); BIT conditions and comparisons of columns of different scales are outside the difference-logic theory (unknown when a deciding cube needs them); at most 20 nullable columns per universal question; the BDD order is the condition order (no reordering: large predicates can reach the circuit budget and answer unknown)
- measures are polynomials over logical integers; equivalence modulo relations is scoped to where the relation certificates hold (snapshot or derived certificates), never universal; query measures on cube boxes with column comparisons are unknown
- representation limits: 64 distinct conditions per set, 4096 parts or questions, 128 cone nodes per question, measure pairs of one check over at most 8 columns and 256 pairs, evaluation widths up to 512 bits
- families compute COUNT, present-measure counts and exact SUM only (no MIN / MAX / AVG / GROUP BY); the family runs, universal decisions and relation reductions run on the CPU only
- the relationship optimizer bridge takes integer measures of degree <= 2 over catalog columns (tr_optimize's limits: 32 terms, 12 columns, 32-bit codes); relationships_stage.c (an executor transformation) is not used
- snapshot verdicts are not maintained across revisions (re-check on the new scope); CUDA was run in the CPU emulation here, not on a physical GPU

Validation receipts:

Measured runtime: `3833d4d1d856591a0f2080e9ecb42e5545209163`.

- tests: [results/math/META-001/tests_test_math_equiv.json](results/math/META-001/tests_test_math_equiv.json)
- tests: [results/math/META-001/sanitized_test_math_equiv.json](results/math/META-001/sanitized_test_math_equiv.json)
- benchmarks: [results/math/META-001/benchmark_cpu_math_equiv_vm.json](results/math/META-001/benchmark_cpu_math_equiv_vm.json)

<a id="meta-002"></a>

### META-002: Query families as a multi-output program

Status: **in_progress**. Delivery wave: **W4**.

Compile related predicates/scenarios/measures jointly, sharing decode, masks and states while retaining per-output domains and explanations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-006](#fnd-006), [FND-008](#fnd-008)

Native CPU: Native C (math_program.c): a family of queries (a BOOL-001 predicate, a FND-006 grouping, scalar or grouped, measures COUNT(*), COUNT / SUM / AVG of integer polynomials, DIST-001 histograms) over one scope is compiled into one program: conditions canonicalized to integer intervals (x > 0 and x >= 1 are one condition), predicates hash-consed under identities of SQL's Kleene logic only (constants, double negation, De Morgan with complementary conditions, associative-commutative flattening, idempotence; each node negated once; a TRUE predicate is no filter), optionally merged by a bounded universal search; per (predicate class, grouping) one SQL-mode FND-006 bank for the polynomials whose canonical monomials each read exactly the measure's written columns, one listwise bank per other written column set (SQL validity of the written expression kept when terms cancel), one histogram per spec, all built by one FND-006 plan (one fused traversal); per-query finalization (exact 256-bit contractions of the shared 128-bit sums, SQL NULLs, present groups or the scalar row). A run checks its whole need (tr_prog_need: the pass and the finalization) first. Every query records its predicate class and proof, the states it reads and their sharers, its value slots and bytes, the pass and finalization devices per measure; states live for one run, or (declared) for later runs of the same program on its fresh scope. Universal comparison of two queries' predicates: equivalent (canonical form or the 3^k search with per-column feasibility), different with a witness row, unknown (BIT, column comparisons) or budget-limited, with grouping and measure equality by value semantics; snapshot comparison: the rows where they differ on this snapshot.

GPU: CUDA kernel math_program_gpu.cu: the per-output finalization, one work item per (measure, group) slot with the shared inline slot evaluator (no atomics, no barriers), over states copied from the host (tr_mgpu_program_finalize) or left on the device by FND-006's fused kernel (tr_mgpu_program_run: the fused pass and the finalization in two launches without a host round trip of the states); device copies budgeted first; equal to the CPU values bit for bit (scrambled emulation).

Bounds:

- Compile: circuit = conditions + nodes, output = queries, support = polynomial terms before allocating; circuit = canonical conditions + nodes after canonicalization; iterations = the proof search's assignments (a pair past the budget stays apart, recorded); output = queries + value slots and bytes = the program and 72 bytes per slot once the tables exist; FND-006's own checks.
- A run checks its whole need first (tr_prog_need: the FND-006 pass and the finalization, output = the larger, bytes = their sum); compare: iterations = 3^k for the k conditions read (outcome budget past it).
- Representation limits (UnsupportedError): 4096 queries, 16384 measures, 65536 terms, 1024 terms per measure, 64 canonical conditions.

Implementation targets (proposed until the capability is validated):

- `torbles/math_program.h`
- `torbles/math_program.c`
- `torbles/math_program_gpu.cu`
- `torbles/math_program.py`
- `tests/test_math_program.py`
- `bench/benchmark_math_program.py`

Acceptance tests:

- Joint outputs match independently executed queries including incompatible/overlapping groups.
- Memory/dispatch costs and per-output CPU/GPU paths are recorded; no implicit cross-query state reuse outside declared lifetimes.

Benchmark cases:

- Compare families of 1/8/32/128 related queries with shared and disjoint filters/measures; report proof/common-subexpression construction and actual shared scans.
- Use equivalent, one-row-different and universally-different query pairs; separate snapshot checking from universal proof search and retain unknown/budget-limited outcomes.

Additional limits:

- The universal search decides interval, NE and NULL conditions column by column; BIT and column comparisons leave a differing assignment undecided (unknown). META-001 (math_equiv) holds the decision procedures for measures, implication and containment; META-002's compare covers WHERE predicates only, with measures compared canonically.
- Compile-time merging by search compares only predicates with the same canonical condition support; the canonical form does not distribute AND over OR or absorb.
- Witness rows respect declared nullability only (no keys, references, type ranges or other schema constraints).
- Measures are COUNT, SUM and AVG of int64-coefficient polynomials and histograms (no MIN, MAX, variance or ratios of different measures).
- Snapshot lifetimes are kept in the Python program object (reuse decided by the native scope freshness checks); the CPU paths are single-threaded.
- Measured on the CPU and the emulated kernels only here; the device receipt needs an actual GPU run.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/META-002/tests_test_math_program.json](results/math/META-002/tests_test_math_program.json)
- tests: [results/math/META-002/sanitized_test_math_program.json](results/math/META-002/sanitized_test_math_program.json)
- benchmarks: [results/math/META-002/benchmark_cpu_math_program_vm.json](results/math/META-002/benchmark_cpu_math_program_vm.json)

## TOPO: Finite geometry and topology

Declared occupancy complexes, boundaries, cycles and homology.

<a id="topo-001"></a>

### TOPO-001: Declared occupancy complexes and local topology

Status: **in_progress**. Delivery wave: **W5**.

Construct explicit grid/cubical incidence from user-declared geometry; compute supported boundaries, adjacency and Euler-characteristic summaries.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [MULTI-001](#multi-001)

Native CPU: Native C (math_topo.c) over an FND-004 scope: a declared grid (2 or 3 integer coordinate columns read as exact logical values, decimal and repeated coordinates refused; an extent of top cells per axis; 4 / 8 or 6 / 18 / 26 dual-graph adjacency; an optional value column). Selected rows inside the extent occupy closed unit top cells (NULL coordinates and rows outside the extent counted apart, duplicates collapsed); the complex is their closure, enumerated over the Khalimsky grid of the extent: cells per dimension in a canonical order (increasing linear index of the declared coordinates), their anchors and axes, the 2k signed faces of every k-cell (product orientation, d o d = 0 over Z), the exact Euler characteristic, the boundary cells of the union (a subcomplex) and their Euler characteristic, per-cell local invariants (star counts per dimension, link counts and link Euler characteristic, boundary flag, dual degree of top cells), the dual graph (edges, union-find components labelled by their smallest top cell, adjacent pairs), the per-top-cell extremes of the value column, the Euler counts without the complex, and an FND-007 record of the scope (coordinate and value columns, exact selection) with exact staleness reasons (a scope over other columns is refused: columns). A device build is adopted after a full validation (canonical occupancy, increasing cells of their dimension inside the complex, every face at its slot with its sign, every lower cell a face, row accounting). Budgets (variables, states = positions, output = cells, tensor = incidences, bytes) are checked before allocation; the data-dependent ones after the occupancy scan, before the arrays exist.

GPU: CUDA kernels math_topo_gpu.cu: tr_mgpu_topo_counts (one work item per row slice: occupancy by atomic OR, row accounting per thread added once; one per candidate slice: cells and boundary cells per dimension, local counts added once) and tr_mgpu_topo_build (occupancy, per-position value extremes by atomic min / max, per-chunk cell counts of 256 candidates in their own slots, their exclusive prefix on the host, cells written in increasing order with one writer per slot, incidences one work item per cell by binary search, top-cell values), adopted natively by tr_topo_adopt; tr_mgpu_topo_local (one work item per cell into its own slot, totals by atomics). They run the inline functions of math_topo.h, so they equal the CPU bit for bit (emulated with a scrambled thread order in the tests). Compiled with nvcc for sm_80 and verified in the emulation only; not yet run on a device.

Bounds:

- Scan (Euler counts, a build's first phase): variables = grid dimension (2 or 3), states = top-cell positions of the declared extent, bytes = the occupancy, rank-directory and valued bitmaps (CUDA: + the selection and column copies), checked before allocation.
- Build: output = cells, tensor = signed incidences, bytes = the complex (8 per cell, 5 per incidence, 16 per top cell with a value column) and its FND-007 record, checked after the occupancy scan and before the arrays exist; CUDA: the device copies, then the adopted complex.
- Local invariants, dual graph and boundary exports budget their outputs (cells, labels and pairs, entries); at most 2^28 top-cell positions (UnsupportedError past it).

Implementation targets (proposed until the capability is validated):

- `torbles/math_topo.h`
- `torbles/math_topo.c`
- `torbles/math_topo_gpu.cu`
- `torbles/math_topo.py`
- `tests/test_math_topo.py`
- `bench/benchmark_math_topo.py`

Acceptance tests:

- Small occupied grids match hand-enumerated cells, incidences and Euler counts.
- Storage torus/code order is never assumed to be the data's physical geometry.

Benchmark cases:

- Use explicitly declared 2D/3D occupancy grids with empty/full/25%-occupied patterns and known small holes; report complex/incidence construction separately from local invariants.
- Vary boundary-matrix size, coefficient field and filtration levels within budgets; compare small known Betti/cycle results and record reduction/iteration limits.

Additional limits:

- The extent is dense: states = top-cell positions, at most 2^28; the cells are enumerated by scanning every candidate of the Khalimsky grid (no sparse enumeration for huge sparse extents).
- Unit cells of integer coordinates only (no declared spacing, no decimal coordinates, no periodic / toroidal boundary conditions); one complex per scope (no per-group complexes).
- Union-find components, boundary-matrix export and value extremes run on the CPU; the CPU paths are single-threaded.
- The complex is tied to its scope through an FND-007 record (freshness with exact reasons); FND-007 has no claim kind for complexes, so no certificate is issued.
- The CUDA kernels have been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/TOPO-001/tests_test_math_topo.json](results/math/TOPO-001/tests_test_math_topo.json)
- tests: [results/math/TOPO-001/sanitized_test_math_topo.json](results/math/TOPO-001/sanitized_test_math_topo.json)
- benchmarks: [results/math/TOPO-001/benchmark_cpu_math_topo_vm.json](results/math/TOPO-001/benchmark_cpu_math_topo_vm.json)

<a id="topo-002"></a>

### TOPO-002: Bounded cycles, homology and persistence

Status: **in_progress**. Delivery wave: **W5**.

Add boundary matrices, cycle/rank calculations and explicitly bounded filtrations with a declared coefficient field.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TOPO-001](#topo-001), [LIN-002](#lin-002)

Native CPU: Native C (math_topo.c): the boundary matrices d_k of a TOPO-001 complex (signed coordinates over Z, residues mod p, or FIELD-001 packed GF(2) matrices) or of an explicit integral chain complex (up to 7 dimensions, d_{k-1} d_k = 0 verified over Z at creation); d o d = 0 checked over Z and GF(2) from the stored incidences (violations and smallest violating cell per dimension); over a declared field: GF(2) (ranks through FIELD-001's tr_field_rank of the packed matrices, or the sparse reduction), GF(p) for a verified prime 3 <= p < 2^62 (FND-009 Barrett arithmetic), Q exactly (fraction-free reduction with primitive integer columns, entries checked to fit int64 with 128-bit intermediates, refused past it). The standard column reduction gives ranks, Betti numbers b_k = n_k - rank d_k - rank d_{k+1} and, with V tracking, a basis of H_k as k-cycles (H_1 as edge cycles; GF(p) generator coefficient 1, Q primitive with a positive generator). Persistence of an explicitly bounded filtration: top cells enter at the first declared threshold admitting the value column's minimum (sublevel) or maximum (superlevel), or at explicit levels (per top cell, or per cell of a chain, validated as a filtration); lower cells with their earliest coface; cells past the last level never enter. The total order is (level, dimension, index) with a declared tie rule (index increasing or decreasing); pairs (dimension, birth and death levels and cells, zero-length counted) and essential classes are sorted canonically; Betti numbers at every level. Budgets: states = cells, tensor = boundary entries (packed: n_{k-1} n_k), support = levels, iterations = column additions and bytes = every column held, checked deterministically during the reduction (a refusal returns nothing), output after it.

GPU: CUDA kernel math_topo_gpu.cu: tr_mgpu_topo_dd, the batched boundary-composition check d_{k-1} d_k = 0 over Z and over GF(2) (one work item per k-cell of every dimension 2 .. dim, counts by atomics, the smallest violating cell by atomic minimum), equal to tr_topo_check_view bit for bit (emulated with a scrambled thread order, including corrupted incidences). The column reductions (ranks, cycle bases, persistence) run on the CPU only. Compiled with nvcc for sm_80 and verified in the emulation only; not yet run on a device.

Bounds:

- Homology and persistence: states = cells (the boundary matrix's rows and columns), tensor = boundary entries (packed GF(2): the largest n_{k-1} n_k), support = filtration levels (1 for homology), bytes = the fixed arrays, checked before allocation.
- During the reduction: iterations = column additions and bytes = every reduced column, V column and scratch buffer held, refused deterministically at the first step or allocation past the limit (nothing is returned); output = pairs, essential classes, cycle classes and entries, checked before the result is written.
- Explicit chain complexes: states = cells, tensor = entries, bytes = the copies and the d o d scratch; at most 7 dimensions and 2^32 cells or entries per dimension; GF(p) needs a verified prime p < 2^62.

Implementation targets (proposed until the capability is validated):

- `torbles/math_topo.h`
- `torbles/math_topo.c`
- `torbles/math_topo_gpu.cu`
- `torbles/math_topo.py`
- `tests/test_math_topo.py`
- `bench/benchmark_math_topo.py`

Acceptance tests:

- Boundary-of-boundary is zero and tiny known complexes yield expected ranks/Betti numbers.
- Field, orientation, filtration ties, matrix size and reduction budgets are explicit; no free homology claim follows from popcount.

Benchmark cases:

- Use explicitly declared 2D/3D occupancy grids with empty/full/25%-occupied patterns and known small holes; report complex/incidence construction separately from local invariants.
- Vary boundary-matrix size, coefficient field and filtration levels within budgets; compare small known Betti/cycle results and record reduction/iteration limits.

Additional limits:

- The reduction is the plain standard algorithm (no clearing / twist, no cohomology, no apparent pairs), single-threaded and CPU only; the packed GF(2) path is dense (n_{k-1} n_k bits, refused past the tensor budget).
- Q entries are bounded by int64 (with 128-bit intermediates): a larger entry is refused (OverflowError), never wrapped; integral homology (torsion coefficients, Smith normal form) is not computed.
- Cycle representatives are the reduction's V columns (cycles, not shortest or simple cycles); persistence returns pairs and levels, no representative cycles per pair.
- Filtrations are lower-star style on the top cells of a complex (or explicit levels); FND-007 has no claim kind for homology, so results are tied to the complex's FND-007 record rather than issued as certificates.
- The CUDA kernel has been compiled with nvcc and verified in the emulation only; no device run yet.

Validation receipts:

Measured runtime: `60d74e4d7c4e6672892f7f30e57396c90fdc1123`.

- tests: [results/math/TOPO-002/tests_test_math_topo.json](results/math/TOPO-002/tests_test_math_topo.json)
- tests: [results/math/TOPO-002/sanitized_test_math_topo.json](results/math/TOPO-002/sanitized_test_math_topo.json)
- benchmarks: [results/math/TOPO-002/benchmark_cpu_math_topo_vm.json](results/math/TOPO-002/benchmark_cpu_math_topo_vm.json)
