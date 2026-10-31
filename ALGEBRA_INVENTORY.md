# Torbles algebraic capability and implementation inventory

Generated from [docs/algebra_inventory.json](docs/algebra_inventory.json). Edit that canonical file and run `python bench/render_algebra_inventory.py`; use `--check` to validate coverage, dependencies, evidence and this rendered view.

Inventory date: 2026-10-30. Source baseline: `aaacb0f`. Working branch: `alpha`.

The torbles repository (branch alpha). Moved here from the prototype (prototype/mixed-scaling, 7affc3e) on 2026-10-31; earlier receipts under results/algebra_foundation and results/algebra_domains were measured there.

All 32 mathematical families discussed with the user, including shared native foundations. This is an implementation backlog with explicit bounded capabilities, not a claim that all families are implemented.

Coverage: **32 families, 81 actionable capabilities, including 13 shared foundations**.

| Status | Capabilities |
|---|---:|
| existing | 0 |
| partial | 11 |
| planned | 49 |
| in_progress | 2 |
| validated | 19 |

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

Status: **validated**. Delivery wave: **W1**.

Modules declare carriers, laws, valid transforms, sufficient state, proof dependencies, native kernels and complete construction/reuse costs.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-002](#fnd-002)

Native CPU: Native C (math_registry.c): a constant table of module descriptors (carriers, required and guaranteed laws, operators with carriers, state kind, proof dependencies, CPU and CUDA entry points, cost coefficients) covering the two first-slice built-ins, FND-010, DISC-001 and four executors of one request kind (direct scan, code histogram, sparse power sums, bit-plane counts) plus a proven-empty answer; a composition check that names the missing law or conversion; explanation records of what ran.

GPU: CUDA kernels math_registry_gpu.cu: the direct-scan and histogram executors of the request (exact 128-bit sums, min, max of integer polynomials of a column), reporting device or emulation in the explanation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/math_modules.c](torbles/math_modules.c): Executable dispatch for two actual built-ins: Qpolynomial transforms and scalar moment contraction.
- [tests/test_math_modules.py](tests/test_math_modules.py): Native-domain/carrier checks and actual dispatch, including bypass-route spies.

Implemented files:

- [torbles/math_registry.h](torbles/math_registry.h)
- [torbles/math_registry.c](torbles/math_registry.c)
- [torbles/math_registry_gpu.cu](torbles/math_registry_gpu.cu)
- [torbles/math_registry.py](torbles/math_registry.py)
- [tests/test_math_registry.py](tests/test_math_registry.py)
- [bench/benchmark_math_registry.py](bench/benchmark_math_registry.py)
- [torbles/math_modules.c](torbles/math_modules.c)
- [torbles/math_modules.h](torbles/math_modules.h)
- [tests/test_math_modules.py](tests/test_math_modules.py)

Acceptance tests:

- Reject an operator composition when required laws or carrier conversions are absent.
- Expose actual chosen module, representation, native devices and refusal reason in explanations.

Benchmark cases:

- Measure registry dispatch and bounded candidate search separately from execution.
- Compare direct scans, sparse polynomial, histogram and factored-state candidates on identical requests.

Additional limits:

- One request kind (sum/min/max of up to four integer polynomials of one column of 32-bit codes); later families add their descriptors.
- Cost coefficients are hand-set estimates checked (not calibrated) by the benchmark; the CUDA ones are guesses until measured on a device.
- Power-sum and bit-plane representations compute modulo 2^128 and are admitted only when the exact result is proven to fit.

Validation receipts:

Measured runtime: `2cabc7ed424807bc9d2b8382d71e3ac29fcfee96`.

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

Status: **planned**. Delivery wave: **W4**.

Accumulate multiple basis states in one traversal while sharing predicates, decoded planes, group masks and overlapping products; preserve a competing repeated-reduction plan.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-003](#fnd-003), [FND-005](#fnd-005)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: New CUDA fused reducer must be implemented and independently benchmarked; not inherited from the stage MVP.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_reduce.c`
- `torbles/math_reduce_gpu.cu`
- `tests/test_math_fused_reduce.py`

Acceptance tests:

- Every output and validity count matches independent native reductions for duplicated, empty and selectively grouped data.
- CPU and actual CUDA variants prove execution devices and handle overflow/partial failures atomically.

Benchmark cases:

- Benchmark 1, 3, 6 and 15 basis outputs across selectivity/group-domain regimes.
- Record actual scan/kernel count, decoded bytes, workspace, construction and first/reuse times.

<a id="fnd-007"></a>

### FND-007: Proof certificates and evidence scope

Status: **partial**. Delivery wave: **W3**.

Unify universal laws, enforced schema facts, exhaustive snapshot certificates and statistical hypotheses. Certificates include exact domain/revision and arithmetic carrier.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships.c](torbles/relationships.c): Existing exhaustive affine/sum/product verification and snapshot scope checks.
- [tests/test_relationship_cache.py](tests/test_relationship_cache.py): Existing stale proof/compiled-plan reuse checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_certificates.c`
- `torbles/math_certificates.h`
- `tests/test_math_certificates.py`

Acceptance tests:

- A sampled or approximate relationship cannot authorize an exact rewrite.
- Mutation, NULL-domain changes and join multiplicity changes invalidate applicable snapshot certificates.

Benchmark cases:

- Measure certificate checking separately from candidate discovery.
- Compare certificate reuse, invalidation and full re-verification over growing row counts.

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

Measured runtime: `77472b9f66941ee337eccd5bff8590eeb91f696d`.

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

Status: **planned**. Delivery wave: **W3**.

Versioned persistence for typed mathematical states, including basis, domain/proof identity, endianness, exactness, checksums and owned metadata; no live addresses serialized.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/database.py](torbles/database.py): Existing format-1 database persistence, not mathematical-state persistence.
- [tests/check_persist.py](tests/check_persist.py): Existing source datastore round-trip/integrity tests.

Implementation targets (proposed until the capability is validated):

- `torbles/math_persist.c`
- `torbles/math_persist.h`
- `tests/test_math_persist.py`

Acceptance tests:

- Round-trip each supported carrier/state and reject truncated, corrupted or unknown-version files.
- Reloaded immutable states cannot be misrepresented as refreshed views of a changed source.

Benchmark cases:

- Report serialization/load/validation time and bytes separately from mathematical evaluation.
- Compare persisted versus rebuilt state under fixed data and changing query families.

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

Measured runtime: `3983bf5eec82829968c0e6f2308c3abf04990798`.

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

Status: **planned**. Delivery wave: **W2**.

Sorted histogram/prefix moments return complete absolute, hinge and squared-hinge loss functions over observed breakpoints.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-002](#dist-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_resp.c`
- `torbles/math_resp.h`
- `tests/test_math_resp.py`

Acceptance tests:

- Evaluate values and one-sided slopes at/between every breakpoint against direct row sums.
- Duplicate observations, empty input and degenerate minima preserve the specified tie convention.

Benchmark cases:

- Use squared, absolute and hinge loss families over 1k/100k/1M rows with 1/10/100/10,000 parameter requests; separate basis/prefix construction from reuse.
- Vary distinct breakpoints or basis rank from 2 to the supported limit and include one-shot requests where retained-state construction loses.

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

Status: **planned**. Delivery wave: **W3**.

Find bounded parameter or feature interventions that attain a target while respecting declared linear/convex constraints and an explicit cost.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [INV-001](#inv-001), [UNC-001](#unc-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_inv.c`
- `torbles/math_inv.h`
- `tests/test_math_inv.py`

Acceptance tests:

- Brute-force small discrete domains certify the minimum and all tie cases.
- Separate algebraic intervention from causal claims; retain predicate-membership changes in the objective.

Benchmark cases:

- Use linear and bounded-degree univariate targets with zero/one/multiple/no solutions and domains containing exact boundary roots; report solving separately from state construction.
- Sweep parameter dimension 1/2/3 and finite domains of 8/16/32 values per coordinate, retaining incomplete-search cases and proof bounds.

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

Status: **planned**. Delivery wave: **W3**.

Use supported convex structure or bounded certified search on small polynomial systems; report feasibility and optimality evidence.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [OPT-001](#opt-001), [INV-001](#inv-001), [UNC-001](#unc-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_opt.c`
- `torbles/math_opt.h`
- `tests/test_math_opt.py`

Acceptance tests:

- Compare small exhaustive feasible sets and interval-certified optima.
- A resource-limited search returns an explicit incomplete/certified-bound result, not a false global optimum.

Benchmark cases:

- Benchmark positive-definite, singular and indefinite quadratics in 1/2/4/8 parameters, with and without box constraints; report matrix/state build and solve separately.
- Compare repeated optimization over 1k/100k/1M source rows using shared states against full objective recomputation; retain infeasible/unbounded cases.

## GEN: Generating functions

Distribution, count and path generating polynomials or formal series.

<a id="gen-001"></a>

### GEN-001: Finite-support distribution generating polynomials

Status: **planned**. Delivery wave: **W2**.

Represent count/weight distributions as sparse or bounded dense generating polynomials; support coefficient extraction and moment operators.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_gen.c`
- `torbles/math_gen.h`
- `tests/test_math_gen.py`

Acceptance tests:

- G(1), coefficients and Euler-derivative moments match counts and exact row moments.
- Negative values use declared shifts/Laurent support; coefficients retain bag multiplicity.

Benchmark cases:

- Build sparse/dense generating states for support sizes 8/64/1,024 and two-dimensional support; measure coefficient, moment and evaluation requests separately.
- Compare convolution/truncated-series orders 4/16/64 with explicit tiny products; record support growth, intermediate bytes and budget refusals.

<a id="gen-002"></a>

### GEN-002: Convolution and bounded formal series

Status: **planned**. Delivery wave: **W3**.

Products encode independent/full-product combinations; truncated series support declared bounded coefficients and path lengths.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GEN-001](#gen-001), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_gen.c`
- `torbles/math_gen.h`
- `tests/test_math_gen.py`

Acceptance tests:

- Compare convolution coefficients with enumerated small Cartesian products.
- Key-conditioned joins or dependent events cannot use an unconditional product of marginal distributions.

Benchmark cases:

- Build sparse/dense generating states for support sizes 8/64/1,024 and two-dimensional support; measure coefficient, moment and evaluation requests separately.
- Compare convolution/truncated-series orders 4/16/64 with explicit tiny products; record support growth, intermediate bytes and budget refusals.

## DIST: Distribution algebra

Histograms, tails, quantiles, entropy and information measures.

<a id="dist-001"></a>

### DIST-001: Reusable scalar and joint histograms

Status: **partial**. Delivery wave: **W2**.

Native sparse/dense support states with counts/weights, explicit NULL/presence states and support budgets.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing GROUP BY and histogram bridge can produce distributions, but no general immutable native distribution-state API.
- [torbles/sqlfuncs.py](torbles/sqlfuncs.py): Existing SQL histogram finalizers do not establish the proposed native state abstraction.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dist.c`
- `torbles/math_dist.h`
- `tests/test_math_dist.py`

Acceptance tests:

- Match grouped native counts for duplicates, NULL categories and empty selections.
- Dense/sparse conversions preserve every represented cell and refuse over-budget joint domains.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

<a id="dist-002"></a>

### DIST-002: Prefix distribution calculus and exact quantiles

Status: **partial**. Delivery wave: **W2**.

Sorted histogram prefix counts/moments answer CDF, tails, quantiles, trimmed sums and range integrals.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing dominance and range lowerings are reusable building blocks.
- [torbles/sqlfuncs.py](torbles/sqlfuncs.py): Existing SQL percentile/quantile finalizers; shared native prefix-state API remains planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dist.c`
- `torbles/math_dist.h`
- `tests/test_math_dist.py`

Acceptance tests:

- Every rank and threshold at/between observed values matches expanded-row reference calculations.
- Specify interpolation, empty input, ties and inclusion of range endpoints explicitly.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

<a id="dist-003"></a>

### DIST-003: Entropy, mutual information and contingency transforms

Status: **planned**. Delivery wave: **W2**.

Compute information and contingency statistics on retained count support, preserving joint dependence and numerical precision.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dist.c`
- `torbles/math_dist.h`
- `tests/test_math_dist.py`

Acceptance tests:

- Compare entropy/MI with explicit normalized joint tables including structural zeros.
- Do not infer a joint distribution from moments or independent marginals without declared assumptions.

Benchmark cases:

- Use 1k/100k/1M rows, support sizes 8/64/1,024 and joint dimensions 1/2/3; report sparse/dense construction, state bytes and native scan count.
- Run 1/100/10,000 CDF, quantile, tail or information requests from shared support; include duplicates, structural zeros and near-row-sized support.

## PAIR: Pairwise statistics

Distance, ranking and pairwise summaries without unconditional pair enumeration.

<a id="pair-001"></a>

### PAIR-001: Sorted-distribution pairwise summaries

Status: **planned**. Delivery wave: **W2**.

Compute supported absolute-distance sums, weighted rank/AUC and one-dimensional distribution distances from sorted counts/prefix states.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-002](#dist-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pair.c`
- `torbles/math_pair.h`
- `tests/test_math_pair.py`

Acceptance tests:

- Match enumerated pair calculations for tiny tied/weighted inputs.
- Specify self-pair exclusion, ordering, normalization and class-empty conventions.

Benchmark cases:

- For scalar supports 8/64/1,024, compare pairwise summaries with explicit pairs on small data and retained-prefix computation on larger repeated inputs.
- Vary ties, class imbalance, diagonal exclusion and weights; separate support sorting from 1/100/10,000 summary requests.

<a id="pair-002"></a>

### PAIR-002: Structured pairwise query planning

Status: **planned**. Delivery wave: **W4**.

Choose support aggregation, finite-feature factorization or bounded direct pairs according to the exact statistic.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PAIR-001](#pair-001), [KERNEL-001](#kernel-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pair.c`
- `torbles/math_pair.h`
- `tests/test_math_pair.py`

Acceptance tests:

- Unsupported arbitrary similarities retain a correct costed direct plan or fail explicitly.
- Verify pair multiplicities and diagonal treatment across support compression.

Benchmark cases:

- For scalar supports 8/64/1,024, compare pairwise summaries with explicit pairs on small data and retained-prefix computation on larger repeated inputs.
- Vary ties, class imbalance, diagonal exclusion and weights; separate support sorting from 1/100/10,000 summary requests.

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

Status: **planned**. Delivery wave: **W4**.

Subtract compatible group states and use checked low-rank update identities with singularity fallbacks.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [LIN-002](#lin-002), [DELTA-001](#delta-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_lin.c`
- `torbles/math_lin.h`
- `tests/test_math_lin.py`

Acceptance tests:

- Compare deletion/leave-group-out models with complete reconstruction.
- Detect update denominators or condition failures and rebuild/fail according to the numerical contract.

Benchmark cases:

- Use feature dimensions 2/4/8/16, groups 1/16/256 and 1k/100k/1M rows; report moment/matrix assembly, bytes and solve/decomposition time separately.
- Include well-conditioned, nearly singular and exactly rank-deficient matrices; compare repeated model/ridge/deletion requests against complete reconstruction.

## TENSOR: Tensor algebra

Sparse/factored moment tensors and contraction planning.

<a id="tensor-001"></a>

### TENSOR-001: Sparse and factored moment tensors

Status: **planned**. Delivery wave: **W4**.

Retain declared multi-index states without forcing dense Cartesian expansion.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [FND-004](#fnd-004)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_tensor.c`
- `torbles/math_tensor.h`
- `tests/test_math_tensor.py`

Acceptance tests:

- Match enumerated tensor entries and marginalizations for small joint domains.
- Sparse/dense/factored conversions preserve ordering, dimensions, validity and exact coefficients.

Benchmark cases:

- Vary tensor order 2/3/4, support sparsity 1%/10%/dense and dimensions 2/4/8; record actual intermediate shapes and bytes.
- Compare multiple contraction orders, sparse and dense states on identical tensors; include cases where the cheapest order changes and over-budget intermediates are refused.

<a id="tensor-002"></a>

### TENSOR-002: Costed contraction planning

Status: **planned**. Delivery wave: **W4**.

Choose variable/contraction order, kernel, intermediate shape and materialization lifetime for bounded tensor networks.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TENSOR-001](#tensor-001), [FND-008](#fnd-008)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_tensor.c`
- `torbles/math_tensor.h`
- `tests/test_math_tensor.py`

Acceptance tests:

- Every candidate order agrees with a tiny direct contraction oracle.
- Check complete workspace/output costs and refuse plans whose intermediates exceed budgets.

Benchmark cases:

- Vary tensor order 2/3/4, support sparsity 1%/10%/dense and dimensions 2/4/8; record actual intermediate shapes and bytes.
- Compare multiple contraction orders, sparse and dense states on identical tensors; include cases where the cheapest order changes and over-budget intermediates are refused.

## KERNEL: Kernel calculations

Finite-feature embeddings and certified approximate kernel computations.

<a id="kernel-001"></a>

### KERNEL-001: Exact finite-feature kernel embeddings

Status: **planned**. Delivery wave: **W4**.

Polynomial or explicitly finite-feature kernels reduce means, comparisons and selected pair sums to retained feature states.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TENSOR-001](#tensor-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_kernel.c`
- `torbles/math_kernel.h`
- `tests/test_math_kernel.py`

Acceptance tests:

- Match explicit kernel matrices on small inputs including weights and diagonal exclusions.
- Verify required feature degree and dimensions; an absent feature cannot silently be approximated.

Benchmark cases:

- Compare exact polynomial features at dimensions 2/4/8 and degrees 1/2/4 against direct small kernel matrices; separate feature-state construction and reuse.
- For explicitly approximate kernels vary feature counts 16/64/256 and fixed seeds; record observed error against a small exact numerical oracle alongside time and memory.

<a id="kernel-002"></a>

### KERNEL-002: Approximate spectral feature kernels

Status: **planned**. Delivery wave: **W4**.

Offer RBF/other selected kernels through explicit finite-feature approximation, seeds, precision and error accounting.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SPEC-001](#spec-001), [UNC-002](#unc-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_kernel.c`
- `torbles/math_kernel.h`
- `tests/test_math_kernel.py`

Acceptance tests:

- Compare approximation error with direct small kernel calculations under declared assumptions.
- Persist feature parameters/seeds and prevent approximate states authorizing exact algebraic rewrites.

Benchmark cases:

- Compare exact polynomial features at dimensions 2/4/8 and degrees 1/2/4 against direct small kernel matrices; separate feature-state construction and reuse.
- For explicitly approximate kernels vary feature counts 16/64/256 and fixed seeds; record observed error against a small exact numerical oracle alongside time and memory.

## SPEC: Spectral queries

Fourier, Walsh, parity correlation and frequency-response representations.

<a id="spec-001"></a>

### SPEC-001: Fixed spectral basis state

Status: **planned**. Delivery wave: **W4**.

Aggregate chosen Fourier/complex features once; phase actions and derivatives reuse that fixed frequency basis.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_spec.c`
- `torbles/math_spec.h`
- `tests/test_math_spec.py`

Acceptance tests:

- Fixed-frequency phase transforms match direct evaluation within numerical precision.
- New frequencies request new basis state; missing spectral support is not treated as zero.

Benchmark cases:

- Use fixed frequency sets of 1/8/32 elements and 1/100/10,000 phase/evaluation requests; distinguish new-frequency construction from fixed-basis actions.
- For Boolean transforms vary k=3/6/10/12 within budget, reporting all 2^k states, normalization and forward/inverse timings rather than claiming row-count-only cost.

<a id="spec-002"></a>

### SPEC-002: Walsh spectra and parity correlations

Status: **planned**. Delivery wave: **W2**.

Exact Walsh-Hadamard transforms of bounded Boolean histograms expose parity and interaction coefficients.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LATTICE-001](#lattice-001), [FIELD-001](#field-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_spec.c`
- `torbles/math_spec.h`
- `tests/test_math_spec.py`

Acceptance tests:

- Forward/inverse transforms and Parseval identities agree under a declared normalization.
- Report full 2^k domain cost; coefficient truncation becomes an explicit approximation.

Benchmark cases:

- Use fixed frequency sets of 1/8/32 elements and 1/100/10,000 phase/evaluation requests; distinguish new-frequency construction from fixed-basis actions.
- For Boolean transforms vary k=3/6/10/12 within budget, reporting all 2^k states, normalization and forward/inverse timings rather than claiming row-count-only cost.

## MULTI: Multiscale analysis

Hierarchy states, contrasts, adaptive refinement and localized changes.

<a id="multi-001"></a>

### MULTI-001: Hierarchy rollups and contrasts

Status: **partial**. Delivery wave: **W2**.

Use certified refinement maps for reusable coarse/fine counts, moments and Haar-like contrasts.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/keydesign.py](torbles/keydesign.py): Existing discovered hierarchy/coordinate layouts.
- [torbles/algq.py](torbles/algq.py): Existing group-key projections and grouped reductions; reusable contrast states are planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_multi.c`
- `torbles/math_multi.h`
- `tests/test_math_multi.py`

Acceptance tests:

- Child states merge exactly into parents with explicit absent/all-NULL groups.
- Reject incompatible hierarchy maps and preserve logical codec semantics.

Benchmark cases:

- Use balanced and skew hierarchies with 2/4/8 levels, groups 8/64/1,024 and repeated coarse/fine requests; report retained state bytes and rollup/refinement work.
- Compare localized changes in 1%/10%/all hierarchy leaves with complete reconstruction, including cases where adaptive refinement provides no gain.

<a id="multi-002"></a>

### MULTI-002: Adaptive refinement and change localization

Status: **planned**. Delivery wave: **W4**.

Refine hierarchy regions only where exact or permitted approximate bounds require more detail.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [MULTI-001](#multi-001), [UNC-001](#unc-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_multi.c`
- `torbles/math_multi.h`
- `tests/test_math_multi.py`

Acceptance tests:

- Compare adaptive results with complete fine-grain states on small trees.
- Every unexpanded region carries a sound bound or is explicitly unresolved.

Benchmark cases:

- Use balanced and skew hierarchies with 2/4/8 levels, groups 8/64/1,024 and repeated coarse/fine requests; report retained state bytes and rollup/refinement work.
- Compare localized changes in 1%/10%/all hierarchy leaves with complete reconstruction, including cases where adaptive refinement provides no gain.

## SYM: Symmetry queries

Actions of translations, rotations, permutations and certified symmetries.

<a id="sym-001"></a>

### SYM-001: Closed-basis transformation actions

Status: **planned**. Delivery wave: **W4**.

Translate polynomial coordinates, rotate declared coordinates, and permute features by exact or precision-declared basis-action matrices.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [TENSOR-001](#tensor-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_sym.c`
- `torbles/math_sym.h`
- `tests/test_math_sym.py`

Acceptance tests:

- Composition and identity actions match direct transformed-row evaluation.
- Only declared coordinate geometry is used; dictionary/code adjacency has no inferred physical metric.

Benchmark cases:

- Use polynomial translations, declared 2D rotations and feature permutations on bases of 3/6/15/32 states; compare direct transformed-row evaluation with basis actions.
- Vary transformation requests 1/100/10,000 and certified orbit sizes 2/4/8/16; include one broken-multiplicity example that must invalidate orbit reuse.

<a id="sym-002"></a>

### SYM-002: Certified data symmetries and orbit reductions

Status: **planned**. Delivery wave: **W4**.

Verify multiplicity-preserving symmetries, construct orbit summaries and reduce work under those scoped certificates.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SYM-001](#sym-001), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_sym.c`
- `torbles/math_sym.h`
- `tests/test_math_sym.py`

Acceptance tests:

- Broken multiplicity or missing orbit members invalidate the symmetry certificate.
- Orbit-reduced answers match exhaustive original-domain evaluation.

Benchmark cases:

- Use polynomial translations, declared 2D rotations and feature permutations on bases of 3/6/15/32 states; compare direct transformed-row evaluation with basis actions.
- Vary transformation requests 1/100/10,000 and certified orbit sizes 2/4/8/16; include one broken-multiplicity example that must invalidate orbit reuse.

## DYN: Repeated dynamics

Finite closed observables and repeated operator application.

<a id="dyn-001"></a>

### DYN-001: Finite closed-observable dynamics

Status: **planned**. Delivery wave: **W4**.

Represent declared dynamics as a matrix action on a basis proven closed under composition.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [SYM-001](#sym-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dyn.c`
- `torbles/math_dyn.h`
- `tests/test_math_dyn.py`

Acceptance tests:

- Repeated affine transforms of polynomial moments agree with explicit row evolution.
- Non-closure is detected; truncation requires an explicit approximation/error contract.

Benchmark cases:

- Apply declared affine or closed-observable dynamics for 0/1/10/1,000 steps on 2/4/8-dimensional states; report operator construction and powers separately.
- Compare repeated row evolution with retained operator powers; include nonclosed maps, coefficient growth and numerical stability/budget refusals.

<a id="dyn-002"></a>

### DYN-002: Repeated operator powers and semigroups

Status: **planned**. Delivery wave: **W4**.

Evaluate many steps, selected recurrences or parameterized evolution from retained finite operators.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DYN-001](#dyn-001), [LIN-002](#lin-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_dyn.c`
- `torbles/math_dyn.h`
- `tests/test_math_dyn.py`

Acceptance tests:

- Zero/one/many-step composition and semigroup identities match small direct calculations.
- Bound coefficient growth, iteration count and output size; unstable numerical powers are diagnosed.

Benchmark cases:

- Apply declared affine or closed-observable dynamics for 0/1/10/1,000 steps on 2/4/8-dimensional states; report operator construction and powers separately.
- Compare repeated row evolution with retained operator powers; include nonclosed maps, coefficient growth and numerical stability/budget refusals.

## BOOL: Boolean calculus

Predicate cofactors, signed/GF(2) finite differences and influence.

<a id="bool-001"></a>

### BOOL-001: Predicate cofactors and finite differences

Status: **partial**. Delivery wave: **W2**.

Native Boolean cofactors produce XOR sensitivity or signed changes with an explicit encoded-variable or logical-feature domain.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing Boolean predicate programs and JoinPoly factorization.
- [torbles/alg.c](torbles/alg.c): Existing mask execution; a first-class cofactor/difference API is planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_bool.c`
- `torbles/math_bool.h`
- `tests/test_math_bool.py`

Acceptance tests:

- Compare cofactors and first/mixed differences with complete small truth tables.
- Distinguish XOR derivative from signed difference and preserve SQL unknown/NULL semantics.

Benchmark cases:

- Use predicate DAGs with 4/8/16 conditions and varying shared subexpressions; compare independent cofactor evaluation with shared-mask execution.
- Measure signed and XOR first/mixed differences under uniform assignment and empirical row measures; include sparse support and NULL-conditioned scopes.

<a id="bool-002"></a>

### BOOL-002: Influence and predicate interactions

Status: **planned**. Delivery wave: **W2**.

Count dependency masks and bounded interactions under declared empirical or full-Boolean measures.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [BOOL-001](#bool-001), [DIST-001](#dist-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_bool.c`
- `torbles/math_bool.h`
- `tests/test_math_bool.py`

Acceptance tests:

- Match exhaustive influence calculations under uniform and empirical weights.
- Do not label coordinate sensitivity a causal intervention or confuse stored rows with all assignments.

Benchmark cases:

- Use predicate DAGs with 4/8/16 conditions and varying shared subexpressions; compare independent cofactor evaluation with shared-mask execution.
- Measure signed and XOR first/mixed differences under uniform assignment and empirical row measures; include sparse support and NULL-conditioned scopes.

## LATTICE: Subset-lattice algebra

Zeta/Mobius transforms, exclusive regions and inclusion-exclusion.

<a id="lattice-001"></a>

### LATTICE-001: Bounded Boolean histogram state

Status: **planned**. Delivery wave: **W2**.

Group shared predicates into a bounded k-condition occupancy histogram with exact bag counts.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [BOOL-001](#bool-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_lattice.c`
- `torbles/math_lattice.h`
- `tests/test_math_lattice.py`

Acceptance tests:

- All 2^k small-domain cells match enumerated predicate outcomes.
- Enforce condition/state budgets and preserve condition order/NULL conventions.

Benchmark cases:

- Vary Boolean condition count k=3/6/10/12, predicate overlap and occupancy sparsity; record 2^k state bytes and native histogram scans.
- Compare 1/100/all intersection/coalition requests with one histogram plus zeta/Mobius transforms, retaining cases where full-domain construction loses.

<a id="lattice-002"></a>

### LATTICE-002: Zeta, Mobius and inclusion-exclusion transforms

Status: **planned**. Delivery wave: **W2**.

Transform occupancy counts into all intersections, exclusive regions and bounded coalition interactions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LATTICE-001](#lattice-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_lattice.c`
- `torbles/math_lattice.h`
- `tests/test_math_lattice.py`

Acceptance tests:

- Zeta/Mobius round trips and inclusion-exclusion match direct set counts.
- Signed intermediate states use a compatible carrier and cannot silently wrap.

Benchmark cases:

- Vary Boolean condition count k=3/6/10/12, predicate overlap and occupancy sparsity; record 2^k state bytes and native histogram scans.
- Compare 1/100/all intersection/coalition requests with one histogram plus zeta/Mobius transforms, retaining cases where full-domain construction loses.

## QUANT: Quantified queries

Existential/universal predicates and category coverage.

<a id="quant-001"></a>

### QUANT-001: Existential and universal group predicates

Status: **partial**. Delivery wave: **W2**.

Carry selected-row, qualifying-row and seen/unknown states to implement some/all/none conditions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/sqlexec.py](torbles/sqlexec.py): Existing SQL EXISTS/aggregate execution paths; typed native quantified states remain planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_quant.c`
- `torbles/math_quant.h`
- `tests/test_math_quant.py`

Acceptance tests:

- Compare some/all/none with explicit child rows for empty, all-NULL and mixed groups.
- Specify vacuous truth and SQL unknown conventions; multiplicity does not erase coverage.

Benchmark cases:

- Use 1/100/10,000 groups with 1/8/64 children and 1%/50%/100% qualifying rows; include empty, duplicate and all-NULL child sets.
- Vary required category sets of 4/16/64 elements; compare count/mask coverage states against materialized matching children.

<a id="quant-002"></a>

### QUANT-002: Relational division and required-category coverage

Status: **planned**. Delivery wave: **W2**.

Retain exact required/present category sets or bounded masks per group.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [QUANT-001](#quant-001), [FIELD-001](#field-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_quant.c`
- `torbles/math_quant.h`
- `tests/test_math_quant.py`

Acceptance tests:

- Missing, extra, duplicated and NULL categories match set-based reference results.
- Changing the required set or category dictionary changes state identity.

Benchmark cases:

- Use 1/100/10,000 groups with 1/8/64 children and 1%/50%/100% qualifying rows; include empty, duplicate and all-NULL child sets.
- Vary required category sets of 4/16/64 elements; compare count/mask coverage states against materialized matching children.

## COUNT: Constraint counting

Observed-domain and symbolic-domain model counts and witnesses.

<a id="count-001"></a>

### COUNT-001: Observed-domain constraint counts and witnesses

Status: **partial**. Delivery wave: **W3**.

Return satisfying stored rows/counts and counterexample witnesses with explicit snapshot scope.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [BOOL-001](#bool-001), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/alg.c](torbles/alg.c): Existing row masks/counts can evaluate stored-row predicates.
- [torbles/relationships.c](torbles/relationships.c): Existing violation witnesses and exhaustive relation checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_count.c`
- `torbles/math_count.h`
- `tests/test_math_count.py`

Acceptance tests:

- Implication/equivalence violations match exhaustive mask evaluation.
- Certificates report rows actually covered and never imply unobserved assignments were checked.

Benchmark cases:

- Count and extract witnesses on 1k/100k/1M observed rows for structured and irregular predicates; distinguish the observed snapshot from the full assignment universe.
- For symbolic counts use tiny brute-force-checked formulas and bounded larger circuits with variable order/width controls; report compilation size and budget refusals.

<a id="count-002"></a>

### COUNT-002: Symbolic full-domain algebraic model counting

Status: **planned**. Delivery wave: **W5**.

Add a bounded circuit/decision-diagram/variable-elimination backend for counts and witnesses over declared assignment domains.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [COUNT-001](#count-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_count.c`
- `torbles/math_count.h`
- `tests/test_math_count.py`

Acceptance tests:

- Compare all models of tiny formulas with brute force, including weighted/dependent factors.
- Refuse or return explicit incompleteness when circuit width/search budgets are exhausted.

Benchmark cases:

- Count and extract witnesses on 1k/100k/1M observed rows for structured and irregular predicates; distinguish the observed snapshot from the full assignment universe.
- For symbolic counts use tiny brute-force-checked formulas and bounded larger circuits with variable order/width controls; report compilation size and budget refusals.

## FIELD: Finite-field algebra

GF(2) dependencies, parity, ranks, codes and syndromes.

<a id="field-001"></a>

### FIELD-001: GF(2) vector and matrix primitives

Status: **partial**. Delivery wave: **W2**.

Native parity, rank, elimination, nullspaces and syndromes over packed bits with field laws separate from integer arithmetic.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/alg.c](torbles/alg.c): Existing XOR torus integrity faces; not a general GF(2) matrix API.
- [tests/test_fixes.py](tests/test_fixes.py): Existing fault/integrity checks.

Implementation targets (proposed until the capability is validated):

- `torbles/math_field.c`
- `torbles/math_field.h`
- `tests/test_math_field.py`

Acceptance tests:

- Compare elimination/rank/nullspace with exhaustive small binary systems.
- Check dimensional/encoding boundaries and distinguish parity from integer popcount sums.

Benchmark cases:

- Benchmark packed GF(2) matrices of 32/128/512 rows and columns at multiple ranks/densities; compare elimination, rank and syndrome operations separately.
- Discover planted parity dependencies over 1k/100k/1M rows and one-row near misses; record candidate generation, exact verification and substitution time.

<a id="field-002"></a>

### FIELD-002: Parity relationship discovery and substitution

Status: **planned**. Delivery wave: **W3**.

Discover bounded exact XOR dependencies, verify them over the scoped data, and substitute equivalent predicate circuits.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_field.c`
- `torbles/math_field.h`
- `tests/test_math_field.py`

Acceptance tests:

- Recover planted parity relations and reject one-row near misses.
- Codec changes, NULL masks and revisions invalidate affected parity certificates.

Benchmark cases:

- Benchmark packed GF(2) matrices of 32/128/512 rows and columns at multiple ranks/densities; compare elimination, rank and syndrome operations separately.
- Discover planted parity dependencies over 1k/100k/1M rows and one-row near misses; record candidate generation, exact verification and substitution time.

## GEOM: Algebraic geometry

Certified ideal reduction, elimination and generator selection.

<a id="geom-001"></a>

### GEOM-001: Certified sparse ideal reduction

Status: **planned**. Delivery wave: **W3**.

Verify f-g=sum(h_i*r_i) certificates over a declared carrier/domain and choose cheaper equivalent query representations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DISC-002](#disc-002), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_geom.c`
- `torbles/math_geom.h`
- `tests/test_math_geom.py`

Acceptance tests:

- Certificate verification catches wrong coefficients, domains and dependencies.
- Equivalent candidates agree on certified rows; derivative operators must separately respect declared dependencies.

Benchmark cases:

- Use small relation components with 2/4/8 variables, degrees 1/2/3 and sparse/dense certificates; separate discovery, certificate checking and reduction.
- Compare direct expressions with alternative generator/elimination forms using actual resulting bit widths and scans; retain term-explosion and cost-regression cases.

<a id="geom-002"></a>

### GEOM-002: Bounded elimination and generator selection

Status: **planned**. Delivery wave: **W4**.

Find small generator sets or sparse local elimination forms without assuming a canonical normal form is cheapest.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GEOM-001](#geom-001), [FND-008](#fnd-008)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_geom.c`
- `torbles/math_geom.h`
- `tests/test_math_geom.py`

Acceptance tests:

- Reconstruction covers every supported variable and preserves ranges/NULL domains.
- Enforce degree/term/solve budgets and retain the direct candidate when expansion loses.

Benchmark cases:

- Use small relation components with 2/4/8 variables, degrees 1/2/3 and sparse/dense certificates; separate discovery, certificate checking and reduction.
- Compare direct expressions with alternative generator/elimination forms using actual resulting bit widths and scans; retain term-explosion and cost-regression cases.

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

Status: **planned**. Delivery wave: **W3**.

Use exact Gram/nullspace relationships of a bounded basis to propose and certify polynomial identities under nonnegative row weights.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [LIN-001](#lin-001), [LIN-002](#lin-002), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_disc.c`
- `torbles/math_disc.h`
- `tests/test_math_disc.py`

Acceptance tests:

- Recover exact small polynomial dependencies and reject a single violating row.
- Signed delta weights cannot use zero sum-of-squares as an identity certificate; numerical near-zero is only a hypothesis.

Benchmark cases:

- Use 2/4/8 observed variables over 1k/100k/1M rows with planted affine/product/conservation relations plus one-row near misses; report proposal and exhaustive verification separately.
- Vary candidate/basis degree budgets and revision stability; measure exact Gram/nullspace state growth and refusals rather than extrapolating unsupported bases.

<a id="disc-003"></a>

### DISC-003: Conservation and stability across revisions

Status: **planned**. Delivery wave: **W3**.

Track scoped invariants across declared transformations/updates, recording persistence separately from snapshot truth or predictive evidence.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DISC-002](#disc-002), [DELTA-001](#delta-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_disc.c`
- `torbles/math_disc.h`
- `tests/test_math_disc.py`

Acceptance tests:

- Known conserved transformations preserve certified quantities; perturbations create witnesses.
- Schema guarantees, observed revision stability and sampled hypotheses remain separate evidence classes.

Benchmark cases:

- Use 2/4/8 observed variables over 1k/100k/1M rows with planted affine/product/conservation relations plus one-row near misses; report proposal and exhaustive verification separately.
- Vary candidate/basis degree budgets and revision stability; measure exact Gram/nullspace state growth and refusals rather than extrapolating unsupported bases.

## PRE: Preimage queries

Push conditions on derived values back to generators.

<a id="pre-001"></a>

### PRE-001: Affine and monotone-piece preimages

Status: **partial**. Delivery wave: **W3**.

Transform predicates on derived values into exact generator intervals with integer rounding, sign and NULL guards.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-007](#fnd-007), [UNC-001](#unc-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing typed comparisons and dimension predicate substitution.
- [torbles/coltypes.py](torbles/coltypes.py): Existing logical codec intervals; general derived-function preimages remain planned.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pre.c`
- `torbles/math_pre.h`
- `tests/test_math_pre.py`

Acceptance tests:

- All comparison operators agree on boundary, negative-scale, fractional-threshold and empty cases.
- Require nonzero divisors and all branches for noninvertible maps.

Benchmark cases:

- Use affine positive/negative slopes, monotone pieces, squares and residue maps over bounded integer domains; compare all predicate boundaries with direct evaluation.
- Vary preimage branch counts 1/4/16/64 and predicate selectivity; compare transformed plane predicates with the correct direct/lookup candidate and retain losing cases.

<a id="pre-002"></a>

### PRE-002: Finite roots, residues and periodic preimages

Status: **planned**. Delivery wave: **W3**.

Represent all bounded preimage branches as intervals, residue classes, finite sets or Boolean cubes.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PRE-001](#pre-001), [FIELD-001](#field-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_pre.c`
- `torbles/math_pre.h`
- `tests/test_math_pre.py`

Acceptance tests:

- Compare all preimages with exhaustive small integer domains for squares/moduli/periods.
- Branch explosion triggers a budgeted fallback rather than dropping solutions.

Benchmark cases:

- Use affine positive/negative slopes, monotone pieces, squares and residue maps over bounded integer domains; compare all predicate boundaries with direct evaluation.
- Vary preimage branch counts 1/4/16/64 and predicate selectivity; compare transformed plane predicates with the correct direct/lookup candidate and retain losing cases.

## ENV: Parametric envelopes

Winning groups, exact breakpoints and parameter regions.

<a id="env-001"></a>

### ENV-001: Univariate affine/quadratic group envelopes

Status: **planned**. Delivery wave: **W3**.

Return winning groups and exact parameter breakpoints with a declared tie policy.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [RESP-001](#resp-001), [INV-001](#inv-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_env.c`
- `torbles/math_env.h`
- `tests/test_math_env.py`

Acceptance tests:

- Compare envelope winners against all group functions at every interval and boundary.
- Handle identical functions, empty groups and unbounded parameter intervals exactly.

Benchmark cases:

- Build 8/64/1,024 group functions and request 1/100/10,000 parameter values; compare an envelope build plus lookups with repeated all-group evaluation.
- Vary coincident curves, breakpoint count and parameter dimension; report exact region/output size and over-budget region cases.

<a id="env-002"></a>

### ENV-002: Bounded multivariate parameter regions

Status: **planned**. Delivery wave: **W4**.

Represent supported linear/convex regions where groups or plans dominate; bound region count and dimension.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [ENV-001](#env-001), [INV-002](#inv-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_env.c`
- `torbles/math_env.h`
- `tests/test_math_env.py`

Acceptance tests:

- Exhaustive small grids agree with region membership and tie boundaries.
- Unsupported nonlinear regions are not approximated without an explicit permitted contract.

Benchmark cases:

- Build 8/64/1,024 group functions and request 1/100/10,000 parameter values; compare an envelope build plus lookups with repeated all-group evaluation.
- Vary coincident curves, breakpoint count and parameter dimension; report exact region/output size and over-budget region cases.

## GRAPH: Graph and fixed-point queries

Incidence operators, frontiers, paths and bounded recursive computations.

<a id="graph-001"></a>

### GRAPH-001: Declared graph layouts and reductions

Status: **planned**. Delivery wave: **W5**.

Build explicit incidence/adjacency ownership and native degree, frontier, neighborhood and matrix-vector operations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [LIN-001](#lin-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_graph.c`
- `torbles/math_graph.h`
- `tests/test_math_graph.py`

Acceptance tests:

- Compare directed/undirected/self-loop/multiedge semantics with tiny graph references.
- Vertex dictionaries and snapshot revisions remain part of layout/state identity.

Benchmark cases:

- Use chain, star, disconnected and cyclic graphs with 1k/100k/1M edges; record adjacency/incidence build separately from frontier and reduction iterations.
- Vary frontier density and bounded path length 1/4/16/32; compare supported semirings with small exact graph oracles and retain nontermination/overflow refusals.

<a id="graph-002"></a>

### GRAPH-002: Bounded fixed points, paths and graph energies

Status: **planned**. Delivery wave: **W5**.

Add reachability, bounded path counts, selected shortest-path semirings and iterative energies with explicit termination/iteration rules.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [GRAPH-001](#graph-001), [GEN-002](#gen-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_graph.c`
- `torbles/math_graph.h`
- `tests/test_math_graph.py`

Acceptance tests:

- Compare tiny cycles, disconnected graphs and negative/zero weights under the stated semiring.
- Unbounded walk counts and nonconvergent recurrences cannot silently return a finite answer.

Benchmark cases:

- Use chain, star, disconnected and cyclic graphs with 1k/100k/1M edges; record adjacency/incidence build separately from frontier and reduction iterations.
- Vary frontier density and bounded path length 1/4/16/32; compare supported semirings with small exact graph oracles and retain nontermination/overflow refusals.

## PROV: Provenance

Factorized explanation and multiplicity expressions.

<a id="prov-001"></a>

### PROV-001: Factorized provenance payloads

Status: **planned**. Delivery wave: **W3**.

Native shared-DAG annotations distinguish Boolean existence, bag multiplicity and polynomial provenance.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_prov.c`
- `torbles/math_prov.h`
- `tests/test_math_prov.py`

Acceptance tests:

- Small union/join/group examples match explicit derivation polynomials and multiplicities.
- Shared derivations stay factored; annotation limits and NULL/group presence are explicit.

Benchmark cases:

- Use small joins with shared versus independent derivations and increasing duplicate multiplicities; record factorized DAG nodes, annotation bytes and evaluation time.
- Compare 1/100/1,000 source-deletion or explanation requests against complete query reconstruction; include annotation-budget overflow and correlated derivations.

<a id="prov-002"></a>

### PROV-002: Explanation specialization and deletion propagation

Status: **planned**. Delivery wave: **W5**.

Specialize retained provenance into source explanations and supported deletion effects without recomputing derivations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROV-001](#prov-001), [DELTA-002](#delta-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_prov.c`
- `torbles/math_prov.h`
- `tests/test_math_prov.py`

Acceptance tests:

- Delete each tiny input source and compare supported answer changes with complete reruns.
- Correlated derivations cannot be treated as independent probability events.

Benchmark cases:

- Use small joins with shared versus independent derivations and increasing duplicate multiplicities; record factorized DAG nodes, annotation bytes and evaluation time.
- Compare 1/100/1,000 source-deletion or explanation requests against complete query reconstruction; include annotation-budget overflow and correlated derivations.

## ATTR: Attribution and repair

Contributions, interaction indices and bounded minimal changes.

<a id="attr-001"></a>

### ATTR-001: Sparse interaction and contribution attribution

Status: **planned**. Delivery wave: **W3**.

Compute declared marginal/coalition contributions and sparse-decomposition attribution indices.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROV-001](#prov-001), [LATTICE-002](#lattice-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_attr.c`
- `torbles/math_attr.h`
- `tests/test_math_attr.py`

Acceptance tests:

- Match brute-force small coalition values and contribution sum identities.
- State the baseline/game/weighting convention; algebraic attribution has no automatic causal interpretation.

Benchmark cases:

- Use 3/6/10/12 conditions with sparse versus dense interactions; compare supported contribution indices against exhaustive small coalition calculations.
- Vary explanation/repair search budgets and minimum-solution multiplicities; retain incomplete cases with their certified bounds and actual explored states.

<a id="attr-002"></a>

### ATTR-002: Bounded explanation and repair search

Status: **planned**. Delivery wave: **W5**.

Search for redundant predicates, minimal distinguishing sets and least-cost feasible repairs with certificates where tractable.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [ATTR-001](#attr-001), [INV-002](#inv-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_attr.c`
- `torbles/math_attr.h`
- `tests/test_math_attr.py`

Acceptance tests:

- Compare small searches with exhaustive minima including all ties.
- Budget exhaustion is reported as incomplete with any proven lower/upper bounds.

Benchmark cases:

- Use 3/6/10/12 conditions with sparse versus dense interactions; compare supported contribution indices against exhaustive small coalition calculations.
- Vary explanation/repair search budgets and minimum-solution multiplicities; retain incomplete cases with their certified bounds and actual explored states.

## PROB: Probability and inference

Small-support distributions, partition functions and factorized inference.

<a id="prob-001"></a>

### PROB-001: Small-support partition and likelihood functions

Status: **planned**. Delivery wave: **W4**.

Evaluate declared finite-support probability models, normalizers, likelihood gradients and Hessians from retained statistics.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DIST-001](#dist-001), [CALC-003](#calc-003), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_prob.c`
- `torbles/math_prob.h`
- `tests/test_math_prob.py`

Acceptance tests:

- Normalization, gradients and Hessians match enumerated small distributions.
- Zero mass, support changes and dependent events preserve the declared model semantics.

Benchmark cases:

- Use support sizes 8/64/1,024 and 1/100/10,000 parameter requests; compare retained partition/likelihood states with repeated support enumeration.
- For factorized inference vary graph size and elimination width on brute-force-checked tiny instances and bounded larger models; report intermediate support and precision.

<a id="prob-002"></a>

### PROB-002: Factorized probabilistic inference

Status: **planned**. Delivery wave: **W5**.

Use bounded factor graphs/circuits and costed elimination to compute marginals or selected best assignments.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [PROB-001](#prob-001), [TENSOR-002](#tensor-002), [COUNT-002](#count-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_prob.c`
- `torbles/math_prob.h`
- `tests/test_math_prob.py`

Acceptance tests:

- Compare tiny correlated graphs with exhaustive joint enumeration.
- Treewidth/intermediate-support budgets govern execution; marginals are never multiplied as an undeclared independence assumption.

Benchmark cases:

- Use support sizes 8/64/1,024 and 1/100/10,000 parameter requests; compare retained partition/likelihood states with repeated support enumeration.
- For factorized inference vary graph size and elimination width on brute-force-checked tiny instances and bounded larger models; report intermediate support and precision.

## DELTA: Update calculus

Signed changes, cross terms and maintained algebraic states.

<a id="delta-001"></a>

### DELTA-001: Signed linear state updates

Status: **partial**. Delivery wave: **W2**.

Merge insert/delete/weight-change deltas into compatible counts and moment states with revision/proof invalidation.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-004](#fnd-004), [FND-009](#fnd-009)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/algq.py](torbles/algq.py): Existing moved-dimension delta-plane split executes additively.
- [tests/check_moves.py](tests/check_moves.py): Existing delta/move comparisons; no general maintained mathematical-state bank.

Implementation targets (proposed until the capability is validated):

- `torbles/math_delta.c`
- `torbles/math_delta.h`
- `tests/test_math_delta.py`

Acceptance tests:

- Random signed update sequences match complete state reconstruction after every step.
- Reject invalid negative multiplicities when the public state promises a nonnegative empirical measure.

Benchmark cases:

- Apply insert/delete/weight-change batches of 1/10/1,000 rows to 1k/100k/1M-row states; separate initial build from each maintained update.
- Vary one versus both changing sides of bilinear joins and group-move frequency; compare every update with complete reconstruction, retaining rebuild/refusal cases.

<a id="delta-002"></a>

### DELTA-002: Bilinear/join and nonlinear update laws

Status: **planned**. Delivery wave: **W3**.

Carry all finite cross terms and typed nonlinear-finalizer state; distinguish update calculus from an infinitesimal derivative.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [DELTA-001](#delta-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_delta.c`
- `torbles/math_delta.h`
- `tests/test_math_delta.py`

Acceptance tests:

- Verify Delta(AB)=Delta(A)B+A Delta(B)+Delta(A)Delta(B) on exact small joins/matrices.
- DISTINCT, MIN/MAX and changing join multiplicities request their own retained state rather than inheriting SUM deletion rules.

Benchmark cases:

- Apply insert/delete/weight-change batches of 1/10/1,000 rows to 1k/100k/1M-row states; separate initial build from each maintained update.
- Vary one versus both changing sides of bilinear joins and group-move frequency; compare every update with complete reconstruction, retaining rebuild/refusal cases.

## UNC: Certified uncertainty

Bounds, abstract interpretation and explicit approximation budgets.

<a id="unc-001"></a>

### UNC-001: Intervals, affine forms and congruence reasoning

Status: **planned**. Delivery wave: **W3**.

Native abstract states propagate sound bounds, correlations, residues, nullness and exact-zero facts for pruning and overflow decisions.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-009](#fnd-009), [FND-007](#fnd-007)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_unc.c`
- `torbles/math_unc.h`
- `tests/test_math_unc.py`

Acceptance tests:

- Concrete exhaustive small values remain enclosed by each abstract operation.
- Affine correlation preserves x-x=0 where independent intervals would lose it; unsound pruning is prohibited.

Benchmark cases:

- Use correlated cancellation, interval thresholds and modular constraints with 1%/50%/99% ambiguous regions; report proof/pruning work and concrete fallback work separately.
- Vary coefficient precision, approximation rank and error budget; compare certified bounds with exhaustive small or high-precision oracles and record unproved cases.

<a id="unc-002"></a>

### UNC-002: Explicit approximation and error certificates

Status: **planned**. Delivery wave: **W4**.

Approximate transforms carry declared norms, assumptions, error budgets and deterministic/randomized provenance.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [UNC-001](#unc-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_unc.c`
- `torbles/math_unc.h`
- `tests/test_math_unc.py`

Acceptance tests:

- Measured small-oracle error satisfies the advertised applicable bound.
- Approximate candidates cannot authorize exact relationship rewrites or silently drop uncertainty during composition.

Benchmark cases:

- Use correlated cancellation, interval thresholds and modular constraints with 1%/50%/99% ambiguous regions; report proof/pruning work and concrete fallback work separately.
- Vary coefficient precision, approximation rank and error budget; compare certified bounds with exhaustive small or high-precision oracles and record unproved cases.

## META: Queries about queries

Equivalence, implication, containment and shared query-family computation.

<a id="meta-001"></a>

### META-001: Scoped query equivalence and counterexamples

Status: **partial**. Delivery wave: **W3**.

Prove supported equivalence/implication/containment on a declared snapshot or universal domain, or return concrete counterexamples.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-007](#fnd-007), [BOOL-001](#bool-001), [GEOM-001](#geom-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Existing evidence (scoped building blocks, not a completion claim for extensions):

- [torbles/relationships_optimizer.c](torbles/relationships_optimizer.c): Existing bounded equivalent-polynomial search under verified rules.
- [torbles/relationships_stage.c](torbles/relationships_stage.c): Existing exact bounded stage decompositions; no general query theorem prover.

Implementation targets (proposed until the capability is validated):

- `torbles/math_meta.c`
- `torbles/math_meta.h`
- `tests/test_math_meta.py`

Acceptance tests:

- Deliberately different predicates/measures produce a witness on the applicable domain.
- Snapshot proof labels do not become universal model claims; unsupported universal proofs remain unknown.

Benchmark cases:

- Compare families of 1/8/32/128 related queries with shared and disjoint filters/measures; report proof/common-subexpression construction and actual shared scans.
- Use equivalent, one-row-different and universally-different query pairs; separate snapshot checking from universal proof search and retain unknown/budget-limited outcomes.

<a id="meta-002"></a>

### META-002: Query families as a multi-output program

Status: **planned**. Delivery wave: **W4**.

Compile related predicates/scenarios/measures jointly, sharing decode, masks and states while retaining per-output domains and explanations.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FND-006](#fnd-006), [FND-008](#fnd-008)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_meta.c`
- `torbles/math_meta.h`
- `tests/test_math_meta.py`

Acceptance tests:

- Joint outputs match independently executed queries including incompatible/overlapping groups.
- Memory/dispatch costs and per-output CPU/GPU paths are recorded; no implicit cross-query state reuse outside declared lifetimes.

Benchmark cases:

- Compare families of 1/8/32/128 related queries with shared and disjoint filters/measures; report proof/common-subexpression construction and actual shared scans.
- Use equivalent, one-row-different and universally-different query pairs; separate snapshot checking from universal proof search and retain unknown/budget-limited outcomes.

## TOPO: Finite geometry and topology

Declared occupancy complexes, boundaries, cycles and homology.

<a id="topo-001"></a>

### TOPO-001: Declared occupancy complexes and local topology

Status: **planned**. Delivery wave: **W5**.

Construct explicit grid/cubical incidence from user-declared geometry; compute supported boundaries, adjacency and Euler-characteristic summaries.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [FIELD-001](#field-001), [MULTI-001](#multi-001)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_topo.c`
- `torbles/math_topo.h`
- `tests/test_math_topo.py`

Acceptance tests:

- Small occupied grids match hand-enumerated cells, incidences and Euler counts.
- Storage torus/code order is never assumed to be the data's physical geometry.

Benchmark cases:

- Use explicitly declared 2D/3D occupancy grids with empty/full/25%-occupied patterns and known small holes; report complex/incidence construction separately from local invariants.
- Vary boundary-matrix size, coefficient field and filtration levels within budgets; compare small known Betti/cycle results and record reduction/iteration limits.

<a id="topo-002"></a>

### TOPO-002: Bounded cycles, homology and persistence

Status: **planned**. Delivery wave: **W5**.

Add boundary matrices, cycle/rank calculations and explicitly bounded filtrations with a declared coefficient field.

Exactness: Exact semantics must be specified for the stated domain; floating or approximate implementations require an explicit precision/error contract.

Dependencies: [FND-001](#fnd-001), [FND-005](#fnd-005), [FND-010](#fnd-010), [FND-012](#fnd-012), [TOPO-001](#topo-001), [LIN-002](#lin-002)

Native CPU: New native CPU state/transform implementation required; Python may marshal and display but must not silently supply the mathematical implementation.

GPU: Planned separately; a CPU implementation or emulation is not a CUDA implementation.

Bounds:

- Bound input dimensionality, support/state size and output size.
- Reject unsupported or over-budget requests explicitly; preserve any existing valid execution path.

Implementation targets (proposed until the capability is validated):

- `torbles/math_topo.c`
- `torbles/math_topo.h`
- `tests/test_math_topo.py`

Acceptance tests:

- Boundary-of-boundary is zero and tiny known complexes yield expected ranks/Betti numbers.
- Field, orientation, filtration ties, matrix size and reduction budgets are explicit; no free homology claim follows from popcount.

Benchmark cases:

- Use explicitly declared 2D/3D occupancy grids with empty/full/25%-occupied patterns and known small holes; report complex/incidence construction separately from local invariants.
- Vary boundary-matrix size, coefficient field and filtration levels within budgets; compare small known Betti/cycle results and record reduction/iteration limits.
