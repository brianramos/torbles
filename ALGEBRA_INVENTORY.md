# Torbles algebraic capability and implementation inventory

Generated from [docs/algebra_inventory.json](docs/algebra_inventory.json). Edit that canonical file and run `python bench/render_algebra_inventory.py`; use `--check` to validate coverage, dependencies, evidence and this rendered view.

Inventory date: 2026-10-30. Source baseline: `aaacb0f`. Working branch: `alpha`.

The torbles repository (branch alpha). Moved here from the prototype (prototype/mixed-scaling, 7affc3e) on 2026-10-31; earlier receipts under results/algebra_foundation and results/algebra_domains were measured there.

All 32 mathematical families discussed with the user, including shared native foundations. This is an implementation backlog with explicit bounded capabilities, not a claim that all families are implemented.

Coverage: **32 families, 81 actionable capabilities, including 13 shared foundations**.

| Status | Capabilities |
|---|---:|
| existing | 0 |
| partial | 1 |
| planned | 32 |
| in_progress | 30 |
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

Measured runtime: `00f63ab97ae8685403b2b44d7d37d366626ebf88`.

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

Measured runtime: `00f63ab97ae8685403b2b44d7d37d366626ebf88`.

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

Status: **planned**. Delivery wave: **W3**.

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
