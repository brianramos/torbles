# Module and operator registry (FND-005)

`torbles/math_registry.h` (tables and inline helpers shared with the CUDA library), `math_registry.c`,
`math_registry_gpu.cu`, `math_registry.py`. A native, constant registry of the mathematical modules: what each one
computes in, which laws it relies on and guarantees, which operators it implements in which carriers on which devices,
what state suffices, which proofs it needs, its CPU and CUDA entry points and its cost coefficients; a composition
check over carriers, laws and conversions; the executors of one request kind through four representations; and two
families' transforms run through the registry with explanations.

```python
from torbles import math_registry as reg
reg.modules()          # id, key, name, capability, family, representation, state, carriers, laws relied on /
                       # guaranteed, operators, implements [(operator, carrier)], implementations (with devices),
                       # proofs, devices, cpu/cuda entries, costs, cost_basis
reg.module("lin.solve"); reg.module("FIELD-001"); reg.module(6)  # by key, by a capability one module holds, by id
reg.implementations()  # every (module, operator, carrier, devices) row
reg.check_chain([("convert", "rational"), ("divide", "rational")], input="integer")   # -> "rational"
reg.check_chain([("divide", "integer")], input="integer")      # CarrierError: needs law division, integer lacks it
reg.check_chain([("fold", "gf2"), ("fold", "integer")], input="gf2")   # CarrierError: gf2 -> integer only by 'lift'
reg.explain_chain(steps, input="integer", implemented=True)    # also "modules": the module implementing each step
col = reg.Column(db["t"], "x")                                 # decoded natively (codes, validity), budgeted
req = reg.Request(col, [[1, 0, 3], [0, 1]], aggregate="sum", row_ids=db.query(t).where(t.y < 5).rows())
reg.execute(req, "histogram", backend="cuda", finalize="gpu")  # {"values", "selected", "valid", "nulls", "explanation"}
st = reg.build_state(req, "power", support=[0, 1, 2, 3])        # reusable: reg.execute(req2, "power", state=st)
reg.candidates(req, devices=("cpu", "cuda"))                    # the bounded candidate search (math_planner uses it)
reg.verify(catalog, "sum", "s", "x", "y", backend="cuda")       # DISC-001 through the registry, with an explanation
reg.transform(counts, "mobius", carrier="int128", input="count", backend="cuda")   # LATTICE-002 / SPEC-002
```

## Modules

Ids 1 .. 9 are the representations of the registry's own request kinds (module id == representation id); the FND-008
planner costs their candidates (its coefficient tables have rows 0 .. 9). Ids 10 .. 34 are the implemented families:
representation `None`, requests `family` (each answers through its own API, `torbles/math_<family>.py`), no planner
candidates, cost coefficients all zero (`cost_basis` `"none"`: nothing is calibrated for them).

| id | key | module | capability | state | CPU entry (first) | CUDA |
|---|---|---|---|---|---|---|
| 1 | formal | checked rational polynomials | FND-002 | formal polynomial | `tr_math_module_apply` | — |
| 2 | moments | scalar integer moments | FND-003 | moment bank | `tr_math_module_reduce` | — |
| 3 | cells | distinct cell pre-pass | FND-010 | cell bitmap | `tr_math_budget_count_cells` | `tr_mgpu_budgets_count_cells` |
| 4 | verify | relation verification | DISC-001 | evidence | `tr_reg_verify`, `tr_disc_verify` | `tr_mgpu_registry_verify`, `tr_mgpu_disc_verify` |
| 5 | scan | direct scan | FND-005 | none | `tr_reg_execute` | `tr_mgpu_registry_execute` |
| 6 | histogram | code histogram | FND-005 | counts per code | `tr_reg_execute`, `tr_reg_state_build` | `tr_mgpu_registry_execute`, `_state` |
| 7 | power | sparse power sums | FND-005 | sums mod 2^128 | same | same |
| 8 | bitplane | bit-plane factored counts | FND-005 | counts per plane set | same | same |
| 9 | empty | proven empty selection | FND-008 | exact zero count | `tr_reg_execute` | — |
| 10 | scope | data scope | FND-004 | scope | `tr_scope_from_table` | — |
| 11 | bank | grouped moment bank | FND-004 | grouped moments | `tr_bank_build` | `tr_mgpu_bank_build`, `_rollup` |
| 12 | carriers | numeric carriers | FND-009 | carrier values | `tr_carrier_conversion` | 7 (`tr_mgpu_carriers_*`) |
| 13 | handles | handles and sparse polynomials | FND-013 | handle table | `tr_math_handles_create` | `tr_mgpu_handles_evaluate`, `_abi` |
| 14 | lin.moments | grouped moment matrices | LIN-001 | gram matrices | `tr_lin_moments_build` | `tr_mgpu_lin_moments` |
| 15 | lin.solve | exact solves and eigensystems | LIN-002 | gram matrices | `tr_lin_solve` | `tr_mgpu_lin_eliminate` |
| 16 | opt | quadratic minimizers | OPT-001 | quadratic objective | `tr_opt_quad_create` | `tr_mgpu_opt_faces` |
| 17 | jet.dual | dual-number directional derivatives | JET-001 | jet aggregates | `tr_jet_circuit_create` | `tr_mgpu_jet_aggregate` |
| 18 | jet.taylor | multivariate Taylor jets | JET-002 | jet aggregates | same | same |
| 19 | inv | inverse answers | INV-001 | formal polynomial | `tr_inv_linear_solve` | `tr_mgpu_inv_signs`, `_sweep` |
| 20 | calc | finite-separation calculus | CALC-003 | separated coefficients | `tr_calc_plan_create` | `tr_mgpu_calc_sums` |
| 21 | multi | hierarchy rollups and contrasts | MULTI-001 | hierarchy | `tr_multi_map_certify` | `tr_mgpu_multi_certify`, `_contrasts` |
| 22 | delta | signed state updates | DELTA-001 | maintained cells | `tr_delta_from_arrays` | `tr_mgpu_delta_apply` |
| 23 | bool | predicate cofactors and differences | BOOL-001 | condition masks | `tr_bool_program_create` | `tr_mgpu_bool_masks`, `_differences` |
| 24 | field | GF(2) algebra and parity relations | FIELD-001 | gf2 bits | `tr_field_rank` | `tr_mgpu_field_syndromes`, `_verify`, `_discover` |
| 25 | dist.histogram | reusable histograms | DIST-001 | joint histogram | `tr_dist_hist_build` | `tr_mgpu_dist_hist_build` |
| 26 | dist.prefix | prefix distribution calculus | DIST-002 | prefix sums | `tr_dist_prefix_build` | `tr_mgpu_dist_query`, `_prefix_*` (5) |
| 27 | resp | piecewise loss curves | RESP-002 | prefix sums | `tr_resp_eval` | `tr_mgpu_resp_eval` |
| 28 | quant.groups | quantified group predicates | QUANT-001 | quantifier counts | `tr_quant_groups_build` | `tr_mgpu_quant_groups` |
| 29 | quant.cover | relational division | QUANT-002 | coverage | `tr_quant_cover_build` | `tr_mgpu_quant_cover`, `_cover_test` |
| 30 | lattice.histogram | occupancy histograms | LATTICE-001 | occupancy histogram | `tr_lattice_hist_build` | `tr_mgpu_lattice_hist` |
| 31 | lattice.transform | subset-lattice transforms | LATTICE-002 | lattice vector | `tr_lattice_transform` | `tr_mgpu_lattice_transform`, `tr_mgpu_registry_transform` |
| 32 | spec.walsh | Walsh spectra and parity correlations | SPEC-002 | walsh spectrum | `tr_spec_walsh` | `tr_mgpu_spec_walsh`, `_inverse`, `tr_mgpu_registry_transform` |
| 33 | influence | influence and interactions | BOOL-002 | condition masks | `tr_influence_compute` | `tr_mgpu_influence_compute` |
| 34 | planner.sample | selectivity sample | FND-008 | none | `tr_planner_sample` | `tr_mgpu_planner_sample` |
| 35 | info | entropy, mutual information, contingency | DIST-003 | contingency tables | `tr_info_query` | `tr_mgpu_info_sums` |
| 36 | pair | sorted-distribution pairwise summaries | PAIR-001 | prefix sums | `tr_pair_query` | `tr_mgpu_pair_query` |
| 37 | gen | generating polynomials | GEN-001 | generating polynomial | `tr_gen_from_hist` | `tr_mgpu_gen_conv`, `tr_mgpu_gen_eval` |
| 38 | certificates | proof certificates and evidence scope | FND-007 | certificate | `tr_cert_snapshot` | `tr_mgpu_cert_verify` |
| 39 | persist | state serialization | FND-011 | serialized record | `tr_persist_load` | `tr_mgpu_persist_checksum` |
| 40 | conv | convolution and bounded formal series | GEN-002 | generating polynomial | `tr_conv_product` | `tr_mgpu_conv_batch`, `tr_mgpu_conv_conditional`, `tr_mgpu_conv_walks` |
| 41 | prov | factorized provenance payloads | PROV-001 | provenance dag | `tr_prov_evaluate` | `tr_mgpu_prov_evaluate` |
| 42 | bilinear | bilinear/join and nonlinear update laws | DELTA-002 | join terms | `tr_bilin_join_update` | `tr_mgpu_bilin_cross`, `tr_mgpu_bilin_matrix` |
| 43 | env | univariate group envelopes | ENV-001 | envelope | `tr_env_build` | `tr_mgpu_env_lookup` |
| 44 | attr | contribution attribution | ATTR-001 | cooperative game | `tr_attr_game_prov` | `tr_mgpu_attr_values`, `tr_mgpu_attr_exhaustive`, `tr_mgpu_attr_explore`, `tr_mgpu_attr_search` |
| 45 | count | observed-domain constraint counts | COUNT-001 | constraint counts | `tr_count_set_create` | `tr_mgpu_count_counts`, `tr_mgpu_count_extract` |
| 46 | parity | parity discovery and substitution | FIELD-002 | parity relations | `tr_parity_discover` | `tr_mgpu_parity_verify` |
| 47 | unc | certified abstract states | UNC-001 | abstract state | `tr_unc_program_check` | `tr_mgpu_unc_eval`, `tr_mgpu_unc_rows` |
| 48 | pre | exact preimages of derived predicates | PRE-001 | code set | `tr_pre_codec_for_column` | `tr_mgpu_pre_scan` |
| 49 | ident | moment-Gram identity discovery | DISC-002 | Gram | `tr_ident_plan_make` | `tr_mgpu_ident_gram`, `tr_mgpu_ident_verify` |
| 50 | geom | certified ideal reduction | GEOM-001 | ideal | `tr_geom_ideal_make` | `tr_mgpu_geom_agree`, `tr_mgpu_geom_check` |
| 51 | tensor.state | sparse and factored moment tensors | TENSOR-001 | moment tensors | `tr_tensor_build` | `tr_mgpu_tensor_build` |
| 52 | tensor.contract | costed contraction planning | TENSOR-002 | contraction plans | `tr_tensor_net_plan` | `tr_mgpu_tensor_contract` |
| 53 | kernel | exact finite-feature kernel embeddings | KERNEL-001 | kernel features | `tr_kernel_build` | `tr_mgpu_kernel_build`, `tr_mgpu_kernel_upload`, `tr_mgpu_kernel_query`, `tr_mgpu_kernel_bytes`, `tr_mgpu_kernel_free` |
| 54 | symcount | symbolic model counting | COUNT-002 | decision diagram | `tr_symc_compile` | `tr_mgpu_symcount_levels`, `tr_mgpu_symcount_brute` |
| 55 | explain | explanations and deletion propagation | PROV-002 | explanation state | `tr_expl_create` | `tr_mgpu_explain_propagate` |
| 56 | fourier | fixed spectral basis | SPEC-001 | spectral basis | `tr_fourier_plan_create` | `tr_mgpu_fourier_sums` |
| 57 | lin.update | leave-group-out and low-rank updates | LIN-003 | solved system | `tr_linupd_fit` | `tr_mgpu_linupd_logo` |
| 58 | prob | partition and likelihood functions | PROB-001 | support masses | `tr_prob_support_create` | `tr_mgpu_prob_sums` |
| 59 | sym.action | closed-basis transformation actions | SYM-001 | basis actions | `tr_sym_geometry_make` | `tr_mgpu_sym_query` |
| 60 | sym.symmetry | certified symmetries and orbits | SYM-002 | symmetry certificates | `tr_sym_group_create` | `tr_mgpu_sym_verify` |
| 61 | approx | explicit approximations | UNC-002 | approximation | `tr_approx_series` | `tr_mgpu_approx_eval` |
| 62 | refine | adaptive refinement, localized updates | MULTI-002 | refinement state | `tr_refine_build` | `tr_mgpu_refine_bounds`, `tr_mgpu_refine_rollup` |
| 63 | interv | constrained minimal interventions | INV-002 | intervention plan | `tr_interv_linear` | `tr_mgpu_interv_grid` |
| 64 | polyopt | convex and polynomial optimization | OPT-002 | optimization program | `tr_polyopt_qp_create` | `tr_mgpu_polyopt_boxes` |
| 65 | graph.layout | declared graph layouts | GRAPH-001 | graph layout | `tr_graph_layout_build` | `tr_mgpu_graph_degrees`, `tr_mgpu_graph_expand`, `tr_mgpu_graph_spmv` |
| 66 | graph.paths | bounded fixed points, paths and energies | GRAPH-002 | graph fixed point | `tr_graph_reach` | `tr_mgpu_graph_reach`, `tr_mgpu_graph_closure`, `tr_mgpu_graph_walks`, `tr_mgpu_graph_paths`, `tr_mgpu_graph_energy`, `tr_mgpu_graph_pagerank`, `tr_mgpu_graph_power` |
| 67 | elim | bounded elimination and generators | GEOM-002 | elimination | `tr_elim_select` | `tr_mgpu_elim_check` |
| 68 | preimg | roots, residues and periodic preimages | PRE-002 | code set | `tr_preimg_preimage` | `tr_mgpu_preimg_scan`, `tr_mgpu_preimg_verify` |
| 69 | topo.complex | declared occupancy complexes | TOPO-001 | cubical complex | `tr_topo_build` | `tr_mgpu_topo_counts`, `tr_mgpu_topo_build`, `tr_mgpu_topo_local` |
| 70 | topo.homology | bounded cycles, homology, persistence | TOPO-002 | chain reduction | `tr_topo_check` | `tr_mgpu_topo_dd` |

`reg.module(key)` gives the whole descriptor (up to 12 entry points per device, the laws, proofs and implementation
rows). The first-slice built-ins (1, 2) are described, not changed: their operator sets are read from the validated
`math_modules.c` table, and `tr_math_module_apply` / `_reduce` keep dispatching them. `reg.execute`, `reg.verify`
and the transforms call the entry points the descriptors name. A descriptor's operators and implementation rows come
from one table, `tr_reg_impls` in `math_registry.h` (`{op, carrier, devices, module}`), which the CUDA library reads
too. Proof bits: domain, snapshot, scope, bound, exact_count, catalog, cell_domain, refinement (a certified map),
prime (verified moduli), identity (a reused state's exact construction), handle (a live retained handle),
null_policy (a declared NULL policy or encoding).

**Adding a family** (e.g. DIST-003, PAIR-001, GEN-001): take the next id before `TR_REG_M_END` in `math_registry.h`
(the static assertion in `math_registry.c` requires one descriptor per id), add its implementation rows to
`tr_reg_impls` (at most 32 per module), add one `FAMILY(...)` descriptor to `modules[]` in `math_registry.c` (key,
family = the `math_<family>` file stem, capability, name, state kind, devices, finalize, carriers, laws relied on and
guaranteed, proofs, up to 12 CPU and 12 CUDA entry names of at most 39 characters), and new carriers, laws, operators
or conversions where the family needs them (with their names in `math_registry.c`). `test_fnd005_registry_agrees_with_the_code`
then checks the entry names against both libraries, that every `torbles/math_<fam>_gpu.cu` is described and every
`tr_mgpu_*` entry it defines is named, the capability id against `docs/algebra_inventory.json`, and the rows against
the descriptor.

## Carriers, laws, conversions, compositions

Carriers: `integer` (checked int64), `rational` (checked int64 rationals), `boolean`, `gf2`, `count` (multiplicities,
histogram and occupancy cell counts), `int128`, `mod128` (residues mod 2^128), `interval` (outward rounded), `float64`,
`minmax`, `code` (stored codes), `rowset`, `bigint` (exact integers beyond 128 bits: FND-009 big integers, the 192- /
256-bit words of the families), `bigq` (exact rationals beyond int64), `gfp` (GF(p), p prime < 2^62), `cinterval`
(complex rectangles of tight intervals), `series` (truncated power series over Q: FND-009 series, JET jets), `kleene`
(SQL three-valued logic, FALSE < UNKNOWN < TRUE), `qinterval` (intervals with exact rational endpoints). Laws, as
implemented by every implementation in the carrier: associative, commutative, identity, inverse, distributive,
idempotent (a+a=a), mul_idempotent (a·a=a), ordered (its comparison is total and compatible with +), division
(nonzero elements invertible), char0 (n·1 distinct), exact, enclosure, complement (a+b=1 and a·b=0 for some b:
Boolean, GF(2), row sets; not Kleene, where U OR NOT U = U), char2 (a+a=0: GF(2) only; XOR is not integer addition).
GF(p) is a field without order or char0 (p = 2 is admitted, so GF(2)'s extra laws are not GF(p)'s); `qinterval` sums
of exact endpoints are associative, `interval` and `cinterval` sums are not (rounding); no interval carrier has
inverse (no exact cancellation: [1, 2] - [1, 2] = [-1, 1]) or is exact.

Conversions applied at any step: `exact` (lossless), `checked` (lossless or refused: `code -> integer`, narrowings,
`integer -> count` (a negative multiplicity is refused), `mod128 -> int128` under a proven bound, `float64 -> bigq`
(finite doubles), `cinterval -> interval` (im = [0, 0])) and `enclosing` (`integer/rational/int128/bigint/bigq ->
interval`, `qinterval -> interval`: not lossless, admitted only under `contract="enclosure"`). Conversions applied only
by their own operator (`via`): `lift` (`gf2 -> integer/count`, `gfp -> integer/bigint`: a representative, not a
homomorphism), `reduction` by `residue` (`integer/int128/bigint/rational/bigq -> gfp`, `integer/count -> gf2`: onto
the residues, not injective) and `encoding` by `null_encode` (`kleene -> boolean`: UNKNOWN as FALSE, or the unknown
points kept apart; `kleene -> code`: the digits 0 FALSE, 1 TRUE, 2 UNKNOWN). No conversion leaves an enclosure carrier
for an exact one, a series for a scalar, or GF(p) for anything but its lift.

`check_chain(steps, input, contract="exact"|"enclosure", implemented=False)`: steps are `(operator, carrier)`; the value
starts in `input`. Accepted only when each step's operator exists, its fixed input carrier (histogram, bits, and,
popcount, mark_cells, verify, crt, ratrecon, coefficients, quantify, test, refine, group) is the step's, the carrier
change is a declared conversion admitted at this step (see above), and the step's carrier has every law the operator
requires (plus `exact`, or `exact`/`enclosure` under the enclosure contract); with `implemented=True` some module must
also implement every step (`explain_chain` then names each step's module). Otherwise `CarrierError` (or
`OperatorError` for an unknown operator) with `.refusal = {kind: "law" | "conversion" | "carrier" | "implementation" |
"operator", step, operator, carrier, law, conversion, via}` for the first failing step (`via`: the operator that alone
declares a conversion the step lacks). Operators added for the families and what they require: lift, residue,
null_encode (nothing; their conversions only), multiply (distributive), euclid (a ring with inverse, order, char0),
solve (a field), inertia and minimize (an ordered field), roots (an ordered field of char0; output qinterval), crt
(gfp -> bigint), ratrecon (gfp -> bigq), enclose (enclosure), coefficients (series -> bigq), differentiate (nothing),
predicate (ordered; output kleene), logic and cofactor (a distributive lattice), difference (a fold with inverse: the
signed difference in the integers, the XOR derivative in GF(2)), quantify (count -> kleene), test (count -> boolean),
occupancy (complement: two-valued cells partition the points; output count), zeta, rollup (a fold), mobius, update
(a fold with inverse), walsh (a fold with inverse and char0), prefix (a fold, ordered), refine and group (code ->
code). `reg.chain(representation, aggregate)` gives each representation's own chain; POWER refuses MIN/MAX (powers in
minmax: no distributive law) and BITPLANE too (no conversion count -> minmax).

## Requests and representations

A request aggregates one to four integer polynomials f_j (degree <= 8, int64 coefficients, low to high) of the
logical value x = code + offset (a DECIMAL's scaled integer) over the selected valid rows: `sum` (exact, any size),
`min` / `max` (int64; `None` when no selected row is valid). `selected`, `valid` and `nulls` are reported separately;
an empty selection sums to 0 (with `valid == 0`). Representations, each on the CPU and the CUDA device:

- `scan`: every row's f_j by Horner's rule in checked int64 (128-bit products), folded in 128 bits.
- `histogram`: counts per code (state: `ncodes` counts), contracted as sum count_c f(c + offset); CUDA finalizes on the
  device (`finalize="gpu"`) or copies the counts back and finalizes on the CPU (`"cpu"`).
- `power`: S_k = sum x^k mod 2^128 for the polynomials' exponents (sparse), then sum a_k S_k.
- `bitplane`: for every set S of at most D bit planes, the rows whose code has all of S; the sum is sum_S w_S count_S
  with the Moebius weights w_S = sum_{T in S} (-1)^{|S|-|T|} f(T + offset), mod 2^128.
- `empty` (CPU): no scan; it recounts the selection exactly and refuses unless it is empty.

Exactness: `scan` / `histogram` refuse (`OverflowError`) when a selected value's evaluation leaves int64 at a Horner
step (both refuse on the same values); their sums cannot overflow. `power` / `bitplane` run only when
rows x sum_k |a_k| M^k < 2^127 is proven before execution (M the largest |x| of the code domain), and the residue is
then the exact sum; otherwise `OverflowError` with refusal kind `bound`. A logical domain outside int64, or a stored
code >= ncodes in a selected valid row, is refused first (`OverflowError`, `ValueError`).

## Transforms through the registry (LATTICE-002, SPEC-002)

`reg.transform(values, op, carrier="int128", input=None, direction="up", backend="cpu", budget=None)`: 2^k exact
integers (k <= 30), op `zeta` / `mobius` (`direction` `up`: over supersets, `down`: over subsets) or `walsh`. Natively
(`tr_reg_transform`, `tr_mgpu_registry_transform`): the composition `[(op, carrier)]` from the values' carrier
`input` (default `carrier`) is checked; the module implementing op in carrier on the backend is taken from
`tr_reg_impls` (LATTICE-002 for zeta / Moebius in `count`, `integer` = checked int64, `int128`, `bigint` on the CPU
only; SPEC-002 for Walsh in `int128`); the module's budget is checked (variables k, tensor 2^k, iterations k, bytes
16·2^k); every value must lie in `input` (else `ValueError`) and convert to `carrier` (else `OverflowError`, kind
`overflow`); for `count` the total of the inputs must be <= 2^64 - 1, which bounds every zeta output (else
`OverflowError`, kind `bound`); then the module's entry point runs and `out` is written only on success. Refusals:
`mobius` / `walsh` in `count` (law inverse), `walsh` in `gf2` (law char0: GF(2) has characteristic 2), an input
without a conversion (`kleene -> int128`) or with an explicit one only (`gf2 -> integer`, via `lift`), no
implementation (`walsh` in `integer`; `bigint` on CUDA), `bigint` on the CPU (`UnsupportedError`: big integers are
FND-009 handles, `math_lattice.transform(carrier="bigint")`), an output leaving int64 (`OverflowError` from the
module). The values equal the families' own `math_lattice.transform` / `math_spec.walsh`.

## Explanations, states, budgets

Every execution returns (or attaches to its exception as `.explanation`) what ran: `module`, `module_name`,
`capability`, `family`, `key`, `representation` (None for a family module), `entry` (the native function that
computed: `tr_reg_execute`, `tr_disc_verify`, `tr_lattice_transform`, `tr_mgpu_spec_walsh`, ...), `device` (`"cpu"`,
`"cuda"` on a device, `"cuda-emulated"` in the emulation), `device_name`, `finalize`, `reused`, `status`, `code`,
`refusal` (kind, step, operator, carrier, law, conversion, via), `budget` (resource, need, limit), `rows_scanned`,
`words_scanned`, `state_elements`, `state_bytes`, `device_bytes`, `native_calls`. Native layouts (bytes): request 384,
result 96, explanation 176, state info 104, candidate 376, module 1832, refusal 32, chain 56, conversion 16,
implementation row 16 (ABI version 2 for the explanation, module, conversion and implementation row).

A `State` (CPU-built, or built on the device and imported) records its scope exactly (source token, revision, rows,
code domain, validity, a copy of the selection); `execute(..., state=st)` contracts it without a scan, and refuses with
`StaleStateError` when the scope differs (an engine mutation makes the `Column` itself stale) and `UnsupportedError`
when it does not suffice (exponents outside a power state's support, a degree above a plane state's).

Budgets are checked before allocation (`AlgebraBudgetError`): `degree` (and the kernel limit 8), `support` (terms),
`output` (polynomials), `states` (histogram cells, power sums, plane sets), `circuit` (plane Moebius terms
sum_S 2^|S|), `bytes` (state: 8 per count, 16 per power sum, 4 per plane-set mask; on CUDA plus the device copies of
codes, validity and selection, 160 accumulator bytes and the host copy of a CPU finalization; a Column 8 or 4 bytes
per row), `iterations` (candidates searched); transforms: the module's (above).

## Limitations

The descriptors are written by hand: the agreement test proves that every entry point named exists in its library,
that every CUDA source's entry points are named, that every family with kernels is described and that the capability
ids exist, not that a descriptor's laws or carriers are complete (they are checked against the composition rules and
the brute-force carrier models, and reviewed against each family's header). The laws of a carrier are those every
implementation in it guarantees (math_num.h's stepped intervals have no identity, FND-009's tight ones do; the
`interval` carrier claims neither). The family modules' cost coefficients are zeros: the planner has no candidates for
them and nothing is calibrated. Only LATTICE-002 and SPEC-002 run through the registry (other families through their
own APIs); the registry's transform takes 128-bit values (big-integer transforms through `math_lattice`).
