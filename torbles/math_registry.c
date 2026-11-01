/* FND-005: the native module and operator registry; see math_registry.h. SPDX-License-Identifier: Apache-2.0 */
#include "math_registry.h"
#include "math_modules.h"
#include "math_lattice.h"
#include "math_spec.h"
#include "torble.h"
#include "math_handles.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_reg_step, 1);          /* the ctypes boundary (math_handles.h) */
TR_MATH_ABI(tr_reg_chain, 1);
TR_MATH_ABI(tr_reg_operator, 1);
TR_MATH_ABI(tr_reg_conversion, 2);
TR_MATH_ABI(tr_reg_refusal, 1);
TR_MATH_ABI(tr_reg_request, 1);
TR_MATH_ABI(tr_reg_result, 1);
TR_MATH_ABI(tr_reg_explain, 2);
TR_MATH_ABI(tr_reg_state_info, 1);
TR_MATH_ABI(tr_reg_candidate, 1);
TR_MATH_ABI(tr_reg_impl, 2);
TR_MATH_ABI(tr_reg_module, 2);

extern void at_decode_column(const atable_t *, int, uint32_t *);

_Static_assert(sizeof(tr_reg_refusal) == 32, "refusal ABI size");
_Static_assert(sizeof(tr_reg_chain) == 56, "chain ABI size");
_Static_assert(sizeof(tr_reg_conversion) == 16, "conversion ABI size");
_Static_assert(sizeof(tr_reg_request) == 384, "request ABI size");
_Static_assert(sizeof(tr_reg_result) == 96, "result ABI size");
_Static_assert(sizeof(tr_reg_explain) == 176, "explanation ABI size");
_Static_assert(sizeof(tr_reg_state_info) == 104, "state info ABI size");
_Static_assert(sizeof(tr_reg_candidate) == 376, "candidate ABI size");
_Static_assert(sizeof(tr_reg_impl) == 16, "implementation row ABI size");
_Static_assert(sizeof(tr_reg_module) == 1832, "module ABI size");
_Static_assert(TR_REG_E_INVALID == TR_MATH_INVALID && TR_REG_E_MEMORY == TR_MATH_MEMORY &&
               TR_REG_E_OVERFLOW == TR_MATH_OVERFLOW && TR_REG_E_BUDGET == TR_MATH_BUDGET &&
               TR_REG_E_CARRIER == TR_MATH_CARRIER_ERROR && TR_REG_E_OPERATOR == TR_MATH_OPERATOR_ERROR &&
               TR_REG_E_UNSUPPORTED == TR_MATH_UNSUPPORTED, "codes");
_Static_assert(TR_REG_OPERATORS <= 64 * TR_REG_OPERATOR_WORDS && TR_REG_CARRIERS <= 32 && TR_REG_LAWS <= 32 &&
               TR_REG_PROOFS <= 32, "masks");
_Static_assert(TR_REG_TRANSFORM_MAX_K == TR_LATTICE_MAX_AXES && TR_REG_TRANSFORM_MAX_K == TR_SPEC_MAX_K,
               "transform sizes");
#define SAME(a, b) ((int)(a) == (int)(b))
_Static_assert(SAME(TR_REG_C_INTEGER, TR_MATH_INTEGER) && SAME(TR_REG_C_RATIONAL, TR_MATH_RATIONAL) &&
               SAME(TR_REG_C_BOOLEAN, TR_MATH_BOOLEAN) && SAME(TR_REG_C_GF2, TR_MATH_GF2), "carriers of math_domain.h");
_Static_assert(SAME(TR_OP_POLY_ADD, TR_MATH_OP_ADD) && SAME(TR_OP_POLY_EVALUATE, TR_MATH_OP_EVALUATE) &&
               SAME(TR_OP_BANK_REDUCE, TR_MATH_OP_REDUCE) && SAME(TR_REG_R_FORMAL, TR_MATH_MODULE_POLYNOMIAL) &&
               SAME(TR_REG_R_MOMENTS, TR_MATH_MODULE_MOMENTS), "operators and modules of math_modules.h");
#undef SAME

/* ------------------------------------------------------------------ names */
static const char *const carrier_names[TR_REG_CARRIERS] = {
    "any", "integer", "rational", "boolean", "gf2", "count", "int128", "mod128", "interval", "float64", "minmax",
    "code", "rowset", "bigint", "bigq", "gfp", "cinterval", "series", "kleene", "qinterval"};
static const char *const law_names[TR_REG_LAWS] = {
    "associative", "commutative", "identity", "inverse", "distributive", "idempotent", "mul_idempotent", "ordered",
    "division", "char0", "exact", "enclosure", "complement", "char2"};
static const char *const operator_names[TR_REG_OPERATORS] = {
    "none", "poly_add", "poly_sub", "poly_mul", "poly_pow", "poly_derivative", "poly_integrate", "poly_substitute",
    "poly_evaluate", "bank_reduce", "convert", "eval", "fold", "merge", "scale_count", "distinct_fold", "subtract",
    "divide", "order", "powers", "contract", "histogram", "bits", "and", "popcount", "mark_cells", "verify", "empty",
    "lift", "residue", "null_encode", "multiply", "euclid", "solve", "inertia", "minimize", "roots", "crt",
    "ratrecon", "enclose", "coefficients", "differentiate", "predicate", "logic", "cofactor", "difference",
    "quantify", "test", "occupancy", "zeta", "mobius", "walsh", "rollup", "refine", "group", "update", "prefix",
    "attribute",
    "compile", "model_count",
    "propagate", "explain", "probability",
    "normalize",
    "fixpoint", "relax",
    "boundary", "homology", "persistence",
    "eliminate",
    "subset_search"};
static const char *const state_names[TR_REG_STATE_KINDS] = {
    "none", "formal polynomial", "moment bank", "cell bitmap", "evidence", "histogram", "power sums", "plane counts",
    "empty proof", "scope", "grouped moments", "carrier values", "handle table", "gram matrices",
    "quadratic objective", "separated coefficients", "jet aggregates", "hierarchy", "maintained cells",
    "condition masks", "gf2 bits", "joint histogram", "prefix sums", "quantifier counts", "coverage",
    "occupancy histogram", "lattice vector", "walsh spectrum", "contingency tables", "generating polynomial",
    "certificate", "serialized record", "provenance dag", "join terms", "envelope",
    "cooperative game",
    "constraint counts", "parity relations",
    "abstract state", "code set",
    "ideal",
    "moment tensors", "contraction plans",
    "kernel features",
    "decision diagram",
    "explanation state",
    "spectral basis",
    "solved system",
    "support masses",
    "basis actions", "symmetry certificates",
    "approximation",
    "refinement state",
    "intervention plan",
    "optimization program",
    "graph layout", "graph fixed point",
    "elimination",
    "cubical complex", "chain reduction",
    "pairwise plans",
    "tracker",
    "query parts",
    "factor graph",
    "approximate kernel features",
    "transfer operators", "operator powers",
    "fused outputs",
    "query program",
    "parameter regions",
    "subset problem"};
static const char *const proof_names[TR_REG_PROOFS] = {
    "domain", "snapshot", "scope", "bound", "exact_count", "catalog", "cell_domain", "refinement", "prime",
    "identity", "handle", "null_policy"};
static const char *const request_names[TR_REG_REQUEST_KINDS] = {
    "formal", "moments", "cells", "verify", "polysum", "family"};
static const char *const conversion_kind_names[TR_REG_CONV_KINDS] = {
    "none", "exact", "checked", "enclosing", "lift", "reduction", "encoding"};
static const char *const cost_basis_names[2] = {"none", "estimate"};
static const char *const refusal_names[TR_REG_REFUSALS] = {
    "none", "law", "conversion", "operator", "carrier", "implementation", "budget", "overflow", "bound", "stale",
    "proof", "device", "invalid", "unsupported", "memory"};

static int32_t copy_name(const char *s, char *out, int32_t cap) {
    if (!out || cap <= 0) return TR_MATH_OK;
    size_t n = strlen(s);
    if (n >= (size_t)cap) n = (size_t)cap - 1;
    memcpy(out, s, n); out[n] = 0;
    return TR_MATH_OK;
}
int32_t tr_reg_carrier_info(int32_t carrier, uint32_t *laws, char *name, int32_t cap) {
    if (carrier < 0 || carrier >= TR_REG_CARRIERS) return TR_MATH_INVALID;
    if (laws) *laws = tr_reg_carrier_laws[carrier];
    return copy_name(carrier_names[carrier], name, cap);
}
int32_t tr_reg_law_name(int32_t bit, char *name, int32_t cap) {
    if (bit < 0 || bit >= TR_REG_LAWS) return TR_MATH_INVALID;
    return copy_name(law_names[bit], name, cap);
}
int32_t tr_reg_operator_info(int32_t op, tr_reg_operator *out, char *name, int32_t cap) {
    if (op < 1 || op >= TR_REG_OPERATORS) return TR_MATH_INVALID;
    if (out) *out = tr_reg_operators[op];
    return copy_name(operator_names[op], name, cap);
}
int32_t tr_reg_name(int32_t table, int32_t index, char *name, int32_t cap) {
    const char *const *names;
    int32_t n;
    switch (table) {
    case TR_REG_NAMES_STATES: names = state_names; n = TR_REG_STATE_KINDS; break;
    case TR_REG_NAMES_PROOFS: names = proof_names; n = TR_REG_PROOFS; break;
    case TR_REG_NAMES_REQUESTS: names = request_names; n = TR_REG_REQUEST_KINDS; break;
    case TR_REG_NAMES_CONVERSIONS: names = conversion_kind_names; n = TR_REG_CONV_KINDS; break;
    case TR_REG_NAMES_COST_BASES: names = cost_basis_names; n = 2; break;
    case TR_REG_NAMES_REFUSALS: names = refusal_names; n = TR_REG_REFUSALS; break;
    default: return TR_MATH_INVALID;
    }
    if (index < 0 || index >= n) return TR_MATH_INVALID;
    return copy_name(names[index], name, cap);
}
int32_t tr_reg_conversion_get(int32_t index, tr_reg_conversion *out) {
    if (!out || index < 0 || index >= TR_REG_CONVERSIONS) return TR_MATH_INVALID;
    *out = tr_reg_conversions[index];
    return TR_MATH_OK;
}
int32_t tr_reg_chain_get(int32_t repr, int32_t slot, tr_reg_chain *out) {
    if (!out || repr < 1 || repr >= TR_REG_REPRESENTATIONS || slot < 0 || slot > 2) return TR_MATH_INVALID;
    *out = tr_reg_chains[repr][slot];
    return TR_MATH_OK;
}
int32_t tr_reg_impl_count(void) { return TR_REG_IMPL_ROWS; }
int32_t tr_reg_impl_get(int32_t index, tr_reg_impl *out) {
    if (!out || index < 0 || index >= TR_REG_IMPL_ROWS) return TR_MATH_INVALID;
    *out = tr_reg_impls[index];
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ module descriptors */
#define CB(c) (1u << (TR_REG_C_##c))
#define DATA_PROOFS (TR_PROOF_SCOPE | TR_PROOF_SNAPSHOT)
/* cost coefficients per component, in the order WORDS, SCAN, CHAIN, COMBINE, CONSTRUCT, REUSE, FUSION, MATERIALIZE,
 * TRANSFER, CPU_FINAL, GPU_FINAL, FIXED. CPU rates of modules 1 .. 9 are hand-set estimates (relative units of about
 * a nanosecond on the 2-core development machine), checked on held-out shapes by bench/benchmark_math_planner.py,
 * which also fits replacements on its training shapes; their CUDA rates are uncalibrated estimates for a discrete
 * device (launch and synchronization in FIXED, PCIe copies in TRANSFER). FUSION is the fraction of the shared
 * components' cost a fused pass saves. None of them is a time. The family modules (ids >= TR_REG_M_SCOPE) have no
 * planner candidates and no amounts: their coefficients are zeros, nothing is calibrated (cost_basis NONE). */
#define CUDA_RATES(chain, construct, gpu_final) {0.004, 0.01, chain, 2.0, construct, 0.6, 1.0, 0.05, 0.12, 1.2, gpu_final, 25000.0}
#define BUILTIN(id_, key_, family_) .abi = TR_REG_ABI, .id = id_, .representation = id_, .key = key_, \
    .family = family_, .cost_basis = TR_REG_COST_ESTIMATE
#define FAMILY(id_, key_, family_, cap_, name_) .abi = TR_REG_ABI, .id = id_, .representation = 0, .key = key_, \
    .family = family_, .capability = cap_, .name = name_, .requests = TR_REG_REQ_FAMILY, .cost_basis = TR_REG_COST_NONE
static const tr_reg_module modules[] = {
    {BUILTIN(TR_REG_R_FORMAL, "formal", "mathematics"), .state_kind = TR_REG_STATE_FORMAL,
     .requests = TR_REG_REQ_FORMAL, .devices = TR_REG_ON_CPU,
     .carriers = CB(RATIONAL),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_DIVISION | TR_LAW_CHAR0 | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_DOMAIN,
     .cost = {{0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 200.0}, {0}},
     .name = "checked rational polynomials", .capability = "FND-002",
     .cpu_entry = {"tr_math_module_apply", "tr_math_module_evaluate"}},
    {BUILTIN(TR_REG_R_MOMENTS, "moments", "mathematics"), .state_kind = TR_REG_STATE_MOMENT_BANK,
     .requests = TR_REG_REQ_MOMENTS, .devices = TR_REG_ON_CPU,
     .carriers = CB(CODE) | CB(INTEGER) | CB(RATIONAL),
     .laws_relied = TR_REG_RING | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_DOMAIN | TR_PROOF_SNAPSHOT | TR_PROOF_SCOPE,
     .cost = {{1.0, 1.5, 2.5, 0, 1.0, 0.5, 1.0, 0.05, 0, 2.0, 0, 20000.0}, {0}},
     .name = "scalar integer moments", .capability = "FND-003",
     .cpu_entry = {"tr_math_module_reduce", "tr_math_bank_create"}},
    {BUILTIN(TR_REG_R_CELLS, "cells", "budgets"), .state_kind = TR_REG_STATE_CELL_BITMAP,
     .requests = TR_REG_REQ_CELLS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(CODE) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDEMPOTENT | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_CELL_DOMAIN,
     .cost = {{1.0, 1.5, 1.5, 0, 0.3, 0, 1.0, 0.05, 0, 0.5, 0, 300.0}, CUDA_RATES(0.02, 0.01, 0.01)},
     .name = "distinct cell pre-pass", .capability = "FND-010",
     .cpu_entry = {"tr_math_budget_count_cells"}, .cuda_entry = {"tr_mgpu_budgets_count_cells"}},
    {BUILTIN(TR_REG_R_VERIFY, "verify", "disc"), .state_kind = TR_REG_STATE_EVIDENCE,
     .requests = TR_REG_REQ_VERIFY, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(CODE) | CB(INTEGER) | CB(COUNT) | CB(MINMAX),
     .laws_relied = TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_CATALOG,
     .cost = {{1.0, 1.2, 2.0, 0, 0, 0, 1.0, 0.05, 0, 1.0, 0, 300.0}, CUDA_RATES(0.02, 0.01, 0.01)},
     .name = "relation verification", .capability = "DISC-001",
     .cpu_entry = {"tr_reg_verify", "tr_disc_verify"}, .cuda_entry = {"tr_mgpu_registry_verify", "tr_mgpu_disc_verify"}},
    {BUILTIN(TR_REG_R_SCAN, "scan", "registry"), .state_kind = TR_REG_STATE_NONE,
     .requests = TR_REG_REQ_POLYSUM, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(MINMAX),
     .laws_relied = TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = 0,
     .cost = {{0.6, 1.6, 2.2, 0, 0, 0, 1.0, 0.05, 0, 1.0, 0, 400.0}, CUDA_RATES(0.02, 0.01, 0.01)},
     .name = "direct scan", .capability = "FND-005",
     .cpu_entry = {"tr_reg_execute"}, .cuda_entry = {"tr_mgpu_registry_execute"}},
    {BUILTIN(TR_REG_R_HISTOGRAM, "histogram", "registry"), .state_kind = TR_REG_STATE_HISTOGRAM,
     .requests = TR_REG_REQ_POLYSUM, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(CODE) | CB(INTEGER) | CB(COUNT) | CB(INT128) | CB(MINMAX),
     .laws_relied = TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_CELL_DOMAIN,
     .cost = {{0.6, 1.6, 1.8, 0, 0.35, 0.5, 1.0, 0.05, 0, 0.9, 0, 600.0}, CUDA_RATES(0.03, 0.01, 0.02)},
     .name = "code histogram", .capability = "FND-005",
     .cpu_entry = {"tr_reg_execute", "tr_reg_state_build"}, .cuda_entry = {"tr_mgpu_registry_execute", "tr_mgpu_registry_state"}},
    {BUILTIN(TR_REG_R_POWER, "power", "registry"), .state_kind = TR_REG_STATE_POWER_SUMS,
     .requests = TR_REG_REQ_POLYSUM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(INTEGER) | CB(MOD128) | CB(INT128),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND,
     .cost = {{0.6, 1.6, 2.4, 0, 1.0, 0.5, 1.0, 0.05, 0, 4.0, 0, 500.0}, CUDA_RATES(0.03, 0.01, 0.01)},
     .name = "sparse power sums", .capability = "FND-005",
     .cpu_entry = {"tr_reg_execute", "tr_reg_state_build"}, .cuda_entry = {"tr_mgpu_registry_execute", "tr_mgpu_registry_state"}},
    {BUILTIN(TR_REG_R_BITPLANE, "bitplane", "registry"), .state_kind = TR_REG_STATE_PLANE_COUNTS,
     .requests = TR_REG_REQ_POLYSUM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(ROWSET) | CB(COUNT) | CB(MOD128) | CB(INT128),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_MUL_IDEMPOTENT | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_CELL_DOMAIN,
     .cost = {{0.6, 1.6, 0.5, 2.0, 1.0, 0.5, 1.0, 0.05, 0, 1.5, 0, 600.0}, CUDA_RATES(0.01, 0.01, 0.01)},
     .name = "bit-plane factored counts", .capability = "FND-005",
     .cpu_entry = {"tr_reg_execute", "tr_reg_state_build"}, .cuda_entry = {"tr_mgpu_registry_execute", "tr_mgpu_registry_state"}},
    {BUILTIN(TR_REG_R_EMPTY, "empty", "registry"), .state_kind = TR_REG_STATE_EMPTY_PROOF,
     .requests = TR_REG_REQ_POLYSUM, .devices = TR_REG_ON_CPU,
     .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(MINMAX),
     .laws_relied = TR_LAW_IDENTITY, .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_EXACT_COUNT,
     .cost = {{0.3, 0, 0, 0, 0, 0, 1.0, 0.05, 0, 0, 0, 200.0}, {0}},
     .name = "proven empty selection", .capability = "FND-008",
     .cpu_entry = {"tr_reg_execute"}},
    /* ---- the implemented families: their own APIs, no planner candidates, zero cost coefficients ---- */
    {FAMILY(TR_REG_M_SCOPE, "scope", "scope", "FND-004", "data scope"), .state_kind = TR_REG_STATE_SCOPE,
     .devices = TR_REG_ON_CPU, .carriers = CB(CODE) | CB(INTEGER) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = DATA_PROOFS,
     .cpu_entry = {"tr_scope_from_table", "tr_scope_from_arrays", "tr_scope_get_view", "tr_scope_fresh",
                   "tr_scope_rollup_map", "tr_scope_key_write", "tr_scope_need_table"}},
    {FAMILY(TR_REG_M_BANK, "bank", "bank", "FND-004", "grouped moment bank"), .state_kind = TR_REG_STATE_GROUPED_MOMENTS,
     .devices = TR_REG_ON_BOTH, .finalize = 1, .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(COUNT),
     .laws_relied = TR_REG_RING | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_REFINEMENT | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_bank_build", "tr_bank_init", "tr_bank_accumulate", "tr_bank_merge", "tr_bank_finish",
                   "tr_bank_rollup", "tr_bank_cache_get", "tr_bank_from_states"},
     .cuda_entry = {"tr_mgpu_bank_build", "tr_mgpu_bank_rollup"}},
    {FAMILY(TR_REG_M_CARRIERS, "carriers", "carriers", "FND-009", "numeric carriers"), .state_kind = TR_REG_STATE_VALUES,
     .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(RATIONAL) | CB(BIGQ) | CB(GFP) | CB(INTERVAL) |
                 CB(CINTERVAL) | CB(SERIES) | CB(FLOAT64),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE, .proofs = TR_PROOF_PRIME | TR_PROOF_BOUND,
     .cpu_entry = {"tr_carrier_conversion", "tr_bigint_mul", "tr_bigint_divmod", "tr_bigq_div", "tr_carrier_crt",
                   "tr_carrier_interval_op", "tr_series_mul", "tr_gfp_vec", "tr_gfp_matvec", "tr_carrier_enclose_q64_vec",
                   "tr_carrier_sum_i64", "tr_carrier_dot_wide"},
     .cuda_entry = {"tr_mgpu_carriers_gfp_vec", "tr_mgpu_carriers_gfp_dot", "tr_mgpu_carriers_gfp_matvec",
                    "tr_mgpu_carriers_gfp_reduce_i64", "tr_mgpu_carriers_enclose_q64", "tr_mgpu_carriers_sum_i64",
                    "tr_mgpu_carriers_dot_wide"}},
    {FAMILY(TR_REG_M_HANDLES, "handles", "handles", "FND-013", "handles and sparse polynomials"),
     .state_kind = TR_REG_STATE_HANDLE_TABLE, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(INT128),
     .laws_relied = TR_REG_RING | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_HANDLE,
     .cpu_entry = {"tr_math_handles_create", "tr_math_handles_ipoly_create", "tr_math_handles_ipoly_evaluate",
                   "tr_math_handles_retain", "tr_math_handles_release", "tr_math_handles_close", "tr_math_handles_abi"},
     .cuda_entry = {"tr_mgpu_handles_evaluate", "tr_mgpu_handles_abi"}},
    {FAMILY(TR_REG_M_LIN_MOMENTS, "lin.moments", "lin", "LIN-001", "grouped moment matrices"),
     .state_kind = TR_REG_STATE_GRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(COUNT),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_lin_moments_build", "tr_lin_moments_from_bank", "tr_lin_moments_from_states",
                   "tr_lin_moments_combine", "tr_lin_moments_get_states", "tr_lin_moments_need"},
     .cuda_entry = {"tr_mgpu_lin_moments"}},
    {FAMILY(TR_REG_M_LIN_SOLVE, "lin.solve", "lin", "LIN-002", "exact solves and eigensystems"),
     .state_kind = TR_REG_STATE_GRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGINT) | CB(BIGQ) | CB(GFP) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE, .proofs = TR_PROOF_BOUND | TR_PROOF_PRIME,
     .cpu_entry = {"tr_lin_solve", "tr_lin_system", "tr_lin_fit", "tr_lin_inertia", "tr_lin_covariance",
                   "tr_lin_eigen", "tr_lin_pca", "tr_lin_batch_fits", "tr_lin_batch_residues", "tr_lin_batch_finish"},
     .cuda_entry = {"tr_mgpu_lin_eliminate"}},
    {FAMILY(TR_REG_M_OPT, "opt", "opt", "OPT-001", "quadratic minimizers"), .state_kind = TR_REG_STATE_QUADRATIC,
     .devices = TR_REG_ON_BOTH, .finalize = 1, .carriers = CB(BIGQ) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_DOMAIN,
     .cpu_entry = {"tr_opt_quad_create", "tr_opt_quad_from_moments", "tr_opt_quad_from_poly", "tr_opt_minimize",
                   "tr_opt_faces"},
     .cuda_entry = {"tr_mgpu_opt_faces"}},
    {FAMILY(TR_REG_M_JET_DUAL, "jet.dual", "jet", "JET-001", "dual-number directional derivatives"),
     .state_kind = TR_REG_STATE_JETS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(SERIES) | CB(BIGQ) | CB(GFP) | CB(INTERVAL),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_CHAR0 | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_jet_circuit_create", "tr_jet_plan_create", "tr_jet_agg_build", "tr_jet_agg_merge",
                   "tr_jet_agg_add", "tr_jet_values_from_agg", "tr_jet_values_convert", "tr_jet_plan_need"},
     .cuda_entry = {"tr_mgpu_jet_aggregate"}},
    {FAMILY(TR_REG_M_JET_TAYLOR, "jet.taylor", "jet", "JET-002", "multivariate Taylor jets"),
     .state_kind = TR_REG_STATE_JETS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(SERIES) | CB(BIGQ) | CB(GFP) | CB(INTERVAL),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_CHAR0 | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_jet_circuit_create", "tr_jet_plan_create", "tr_jet_agg_build", "tr_jet_agg_merge",
                   "tr_jet_agg_read", "tr_jet_values_from_agg", "tr_jet_values_convert", "tr_jet_plan_need"},
     .cuda_entry = {"tr_mgpu_jet_aggregate"}},
    {FAMILY(TR_REG_M_INV, "inv", "inv", "INV-001", "inverse answers"), .state_kind = TR_REG_STATE_FORMAL,
     .devices = TR_REG_ON_BOTH, .finalize = 3, .carriers = CB(BIGINT) | CB(BIGQ) | CB(GFP) | CB(QINTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE, .proofs = TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_PRIME,
     .cpu_entry = {"tr_inv_linear_solve", "tr_inv_linear_from_polys", "tr_inv_isolate", "tr_inv_isolate_poly",
                   "tr_inv_roots_refine", "tr_inv_signs_eval", "tr_inv_sweep_prepare", "tr_inv_sweep_eval"},
     .cuda_entry = {"tr_mgpu_inv_signs", "tr_mgpu_inv_sweep"}},
    {FAMILY(TR_REG_M_CALC, "calc", "calc", "CALC-003", "finite-separation calculus"),
     .state_kind = TR_REG_STATE_SEPARATED, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(QINTERVAL) | CB(INTERVAL),
     .laws_relied = TR_REG_FOLD | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_calc_plan_create", "tr_calc_build", "tr_calc_sums", "tr_calc_from_sums", "tr_calc_derive",
                   "tr_calc_integrate", "tr_calc_evaluate", "tr_calc_require"},
     .cuda_entry = {"tr_mgpu_calc_sums"}},
    {FAMILY(TR_REG_M_MULTI, "multi", "multi", "MULTI-001", "hierarchy rollups and contrasts"),
     .state_kind = TR_REG_STATE_HIERARCHY, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(CODE) | CB(INT128) | CB(COUNT) | CB(BIGINT) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_REFINEMENT | TR_PROOF_IDENTITY | TR_PROOF_BOUND,
     .cpu_entry = {"tr_multi_map_certify", "tr_multi_map_declare", "tr_multi_map_verify", "tr_multi_map_compose",
                   "tr_multi_map_from_cert", "tr_multi_hier_build", "tr_multi_hier_update", "tr_multi_contrasts"},
     .cuda_entry = {"tr_mgpu_multi_certify", "tr_mgpu_multi_contrasts"}},
    {FAMILY(TR_REG_M_DELTA, "delta", "delta", "DELTA-001", "signed state updates"),
     .state_kind = TR_REG_STATE_MAINTAINED, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(COUNT),
     .laws_relied = TR_REG_FOLD | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_delta_from_arrays", "tr_delta_from_scope", "tr_delta_between", "tr_delta_state_from_bank",
                   "tr_delta_state_apply", "tr_delta_state_from_cells", "tr_delta_state_to_bank", "tr_delta_state_need"},
     .cuda_entry = {"tr_mgpu_delta_apply"}},
    {FAMILY(TR_REG_M_BOOL, "bool", "bool", "BOOL-001", "predicate cofactors and differences"),
     .state_kind = TR_REG_STATE_CONDITION_MASKS, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(KLEENE) | CB(BOOLEAN) | CB(GF2) | CB(COUNT) | CB(INT128),
     .laws_relied = TR_REG_LATTICE | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR2 | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_bool_program_create", "tr_bool_masks_build", "tr_bool_masks_assignments",
                   "tr_bool_masks_adopt", "tr_bool_differences", "tr_bool_differences_view", "tr_bool_direct",
                   "tr_bool_differences_need"},
     .cuda_entry = {"tr_mgpu_bool_masks", "tr_mgpu_bool_differences"}},
    {FAMILY(TR_REG_M_FIELD, "field", "field", "FIELD-001", "GF(2) algebra and parity relations"),
     .state_kind = TR_REG_STATE_GF2_BITS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(GF2) | CB(CODE) | CB(ROWSET) | CB(KLEENE) | CB(BOOLEAN) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_CHAR2 | TR_LAW_COMPLEMENT | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_field_rank", "tr_field_rref", "tr_field_nullspace", "tr_field_solve", "tr_field_syndromes",
                   "tr_field_bits_build", "tr_field_verify", "tr_field_discover"},
     .cuda_entry = {"tr_mgpu_field_syndromes", "tr_mgpu_field_verify", "tr_mgpu_field_discover"}},
    {FAMILY(TR_REG_M_DIST_HIST, "dist.histogram", "dist", "DIST-001", "reusable histograms"),
     .state_kind = TR_REG_STATE_JOINT_HISTOGRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(INTEGER) | CB(COUNT) | CB(INT128),
     .laws_relied = TR_REG_FOLD | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_CELL_DOMAIN | TR_PROOF_REFINEMENT,
     .cpu_entry = {"tr_dist_hist_build", "tr_dist_hist_init", "tr_dist_hist_accumulate", "tr_dist_hist_finish",
                   "tr_dist_hist_merge", "tr_dist_hist_rollup", "tr_dist_hist_marginal", "tr_dist_hist_from_states"},
     .cuda_entry = {"tr_mgpu_dist_hist_build"}},
    {FAMILY(TR_REG_M_DIST_PREFIX, "dist.prefix", "dist", "DIST-002", "prefix distribution calculus"),
     .state_kind = TR_REG_STATE_PREFIX_SUMS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(COUNT) | CB(INT128) | CB(BIGQ),
     .laws_relied = TR_REG_FOLD | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_DIVISION | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT, .proofs = DATA_PROOFS | TR_PROOF_BOUND,
     .cpu_entry = {"tr_dist_prefix_build", "tr_dist_prefix_get_view", "tr_dist_query", "tr_dist_prefix_need",
                   "tr_dist_prefix_device_bytes"},
     .cuda_entry = {"tr_mgpu_dist_query", "tr_mgpu_dist_prefix_upload", "tr_mgpu_dist_prefix_free",
                    "tr_mgpu_dist_prefix_bytes", "tr_mgpu_dist_prefix_device_view"}},
    {FAMILY(TR_REG_M_RESP, "resp", "resp", "RESP-002", "piecewise loss curves"),
     .state_kind = TR_REG_STATE_PREFIX_SUMS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_BOUND,
     .cpu_entry = {"tr_resp_eval", "tr_resp_pieces", "tr_resp_minimize"}, .cuda_entry = {"tr_mgpu_resp_eval"}},
    {FAMILY(TR_REG_M_QUANT_GROUPS, "quant.groups", "quant", "QUANT-001", "quantified group predicates"),
     .state_kind = TR_REG_STATE_QUANTIFIER, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(KLEENE) | CB(CODE) | CB(COUNT),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_quant_groups_build", "tr_quant_groups_merge", "tr_quant_groups_answer", "tr_quant_groups_tally",
                   "tr_quant_groups_adopt", "tr_quant_groups_need"},
     .cuda_entry = {"tr_mgpu_quant_groups"}},
    {FAMILY(TR_REG_M_QUANT_COVER, "quant.cover", "quant", "QUANT-002", "relational division"),
     .state_kind = TR_REG_STATE_COVERAGE, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(CODE) | CB(ROWSET) | CB(COUNT) | CB(BOOLEAN),
     .laws_relied = TR_REG_LATTICE | TR_LAW_COMPLEMENT | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_quant_cover_build", "tr_quant_cover_require", "tr_quant_cover_answer", "tr_quant_cover_merge",
                   "tr_quant_cover_masks", "tr_quant_cover_adopt", "tr_quant_dict_codes", "tr_quant_cover_need"},
     .cuda_entry = {"tr_mgpu_quant_cover", "tr_mgpu_quant_cover_test"}},
    {FAMILY(TR_REG_M_LATTICE_HIST, "lattice.histogram", "lattice", "LATTICE-001", "occupancy histograms"),
     .state_kind = TR_REG_STATE_OCCUPANCY, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(KLEENE) | CB(BOOLEAN) | CB(CODE) | CB(COUNT),
     .laws_relied = TR_REG_LATTICE | TR_LAW_COMPLEMENT | TR_LAW_EXACT,
     .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_lattice_hist_build", "tr_lattice_hist_from_counts", "tr_lattice_hist_merge",
                   "tr_lattice_hist_vector", "tr_lattice_hist_vector_i128", "tr_lattice_hist_need"},
     .cuda_entry = {"tr_mgpu_lattice_hist"}},
    {FAMILY(TR_REG_M_LATTICE_TRANSFORM, "lattice.transform", "lattice", "LATTICE-002", "subset-lattice transforms"),
     .state_kind = TR_REG_STATE_LATTICE_VECTOR, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(COUNT) | CB(INTEGER) | CB(INT128) | CB(BIGINT),
     .laws_relied = TR_REG_FOLD | TR_LAW_INVERSE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_BOUND,
     .cpu_entry = {"tr_lattice_transform", "tr_lattice_transform_bigint", "tr_lattice_coefficients",
                   "tr_lattice_counts", "tr_lattice_order_sets", "tr_lattice_transform_need", "tr_reg_transform"},
     .cuda_entry = {"tr_mgpu_lattice_transform", "tr_mgpu_registry_transform"}},
    {FAMILY(TR_REG_M_SPEC_WALSH, "spec.walsh", "spec", "SPEC-002", "Walsh spectra and parity correlations"),
     .state_kind = TR_REG_STATE_SPECTRUM, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(GF2),
     .laws_relied = TR_REG_FIELD | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = 0,
     .cpu_entry = {"tr_spec_walsh", "tr_spec_walsh_inverse", "tr_spec_parseval", "tr_spec_correlations",
                   "tr_spec_coefficients", "tr_spec_truncate", "tr_spec_constant_parities", "tr_reg_transform"},
     .cuda_entry = {"tr_mgpu_spec_walsh", "tr_mgpu_spec_walsh_inverse", "tr_mgpu_registry_transform"}},
    {FAMILY(TR_REG_M_INFLUENCE, "influence", "influence", "BOOL-002", "influence and interactions"),
     .state_kind = TR_REG_STATE_CONDITION_MASKS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(KLEENE) | CB(BOOLEAN) | CB(GF2) | CB(INTEGER) | CB(COUNT) | CB(INT128),
     .laws_relied = TR_REG_LATTICE | TR_LAW_INVERSE | TR_LAW_CHAR2 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_influence_compute", "tr_influence_sets", "tr_influence_need"},
     .cuda_entry = {"tr_mgpu_influence_compute"}},
    {FAMILY(TR_REG_M_PLANNER, "planner.sample", "planner", "FND-008", "selectivity sample"),
     .state_kind = TR_REG_STATE_NONE, .devices = TR_REG_ON_BOTH, .finalize = 1, .carriers = CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_FOLD | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_EXACT_COUNT,
     .cpu_entry = {"tr_planner_sample", "tr_planner_polysum", "tr_planner_verify", "tr_planner_choose",
                   "tr_planner_force_parse"},
     .cuda_entry = {"tr_mgpu_planner_sample"}},
    {FAMILY(TR_REG_M_DIST_INFO, "info", "info", "DIST-003", "entropy, mutual information, contingency"),
     .state_kind = TR_REG_STATE_CONTINGENCY, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(COUNT) | CB(QINTERVAL) | CB(INTERVAL) | CB(BIGQ),
     .laws_relied = TR_REG_FOLD | TR_LAW_DIVISION | TR_LAW_ENCLOSURE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_info_support_build", "tr_info_support_get_view", "tr_info_support_group", "tr_info_state_build",
                   "tr_info_sums", "tr_info_query", "tr_info_exact", "tr_info_ln_bounds", "tr_info_term_bounds",
                   "tr_info_support_need", "tr_info_device_need"},
     .cuda_entry = {"tr_mgpu_info_sums"}},
    {FAMILY(TR_REG_M_PAIR, "pair", "pair", "PAIR-001", "sorted-distribution pairwise summaries"),
     .state_kind = TR_REG_STATE_PREFIX_SUMS, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(COUNT) | CB(INT128) | CB(BIGQ),
     .laws_relied = TR_REG_RING | TR_LAW_ORDERED | TR_LAW_DIVISION | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND,
     .cpu_entry = {"tr_pair_query"},
     .cuda_entry = {"tr_mgpu_pair_query"}},
    {FAMILY(TR_REG_M_GEN, "gen", "gen", "GEN-001", "generating polynomials"),
     .state_kind = TR_REG_STATE_GENERATING, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGINT) | CB(INT128) | CB(BIGQ) | CB(SERIES),
     .laws_relied = TR_REG_RING | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND,
     .cpu_entry = {"tr_gen_from_hist", "tr_gen_from_terms", "tr_gen_coeff", "tr_gen_eval", "tr_gen_moment",
                   "tr_gen_euler", "tr_gen_derivative", "tr_gen_mul", "tr_gen_pow", "tr_gen_truncate",
                   "tr_gen_to_series", "tr_gen_mul_need"},
     .cuda_entry = {"tr_mgpu_gen_conv", "tr_mgpu_gen_eval"}},
    {FAMILY(TR_REG_M_CERT, "certificates", "certificates", "FND-007", "proof certificates and evidence scope"),
     .state_kind = TR_REG_STATE_CERTIFICATE, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(INT128) | CB(COUNT) | CB(CODE),
     .laws_relied = TR_REG_RING | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_cert_snapshot", "tr_cert_hypothesis", "tr_cert_schema", "tr_cert_law", "tr_cert_verify",
                   "tr_cert_adopt", "tr_cert_fresh", "tr_cert_authorize_exact", "tr_cert_derive", "tr_cert_reverify",
                   "tr_cert_get_claim", "tr_cert_verify_need"},
     .cuda_entry = {"tr_mgpu_cert_verify"}},
    {FAMILY(TR_REG_M_PERSIST, "persist", "persist", "FND-011", "state serialization"),
     .state_kind = TR_REG_STATE_RECORD, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(MOD128) | CB(BIGINT) | CB(BIGQ) | CB(GFP),
     .laws_relied = TR_REG_FOLD | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT, .proofs = TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_persist_bank", "tr_persist_histogram", "tr_persist_moments", "tr_persist_certificate",
                   "tr_persist_bigint", "tr_persist_bigq", "tr_persist_gfp", "tr_persist_load", "tr_persist_validate",
                   "tr_persist_fresh", "tr_persist_checksum", "tr_persist_save_need"},
     .cuda_entry = {"tr_mgpu_persist_checksum"}},
    {FAMILY(TR_REG_M_CONV, "conv", "conv", "GEN-002", "convolution and bounded formal series"),
     .state_kind = TR_REG_STATE_GENERATING, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(SERIES) | CB(COUNT),
     .laws_relied = TR_REG_RING | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_conv_factor_from_hist", "tr_conv_declare", "tr_conv_check_product", "tr_conv_product",
                   "tr_conv_product_need", "tr_conv_conditional", "tr_conv_joint_sum", "tr_conv_batch_run",
                   "tr_conv_series_mul", "tr_conv_series_walks", "tr_conv_walks_run", "tr_conv_series_check"},
     .cuda_entry = {"tr_mgpu_conv_batch", "tr_mgpu_conv_conditional", "tr_mgpu_conv_walks"}},
    {FAMILY(TR_REG_M_PROV, "prov", "prov", "PROV-001", "factorized provenance payloads"),
     .state_kind = TR_REG_STATE_PROVENANCE, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BOOLEAN) | CB(COUNT) | CB(BIGINT),
     .laws_relied = TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT | TR_LAW_EXACT,  /* B is a lattice; N is not */
     .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_prov_store_create", "tr_prov_source", "tr_prov_select", "tr_prov_union", "tr_prov_join",
                   "tr_prov_group", "tr_prov_plan_build", "tr_prov_evaluate", "tr_prov_expand", "tr_prov_witnesses",
                   "tr_prov_need"},
     .cuda_entry = {"tr_mgpu_prov_evaluate"}},
    {FAMILY(TR_REG_M_BILINEAR, "bilinear", "bilinear", "DELTA-002", "bilinear/join and nonlinear update laws"),
     .state_kind = TR_REG_STATE_JOIN_TERMS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(COUNT),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_bilin_join_build", "tr_bilin_join_update", "tr_bilin_join_finalize", "tr_bilin_cross_eval",
                   "tr_bilin_matrix_delta", "tr_bilin_values_from_scope", "tr_bilin_values_apply",
                   "tr_bilin_values_finalize", "tr_bilin_job_build", "tr_bilin_job_update", "tr_bilin_update_need",
                   "tr_bilin_join_need"},
     .cuda_entry = {"tr_mgpu_bilin_cross", "tr_mgpu_bilin_matrix"}},
    {FAMILY(TR_REG_M_ENV, "env", "env", "ENV-001", "univariate group envelopes"),
     .state_kind = TR_REG_STATE_ENVELOPE, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(QINTERVAL) | CB(RATIONAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND,
     .cpu_entry = {"tr_env_build", "tr_env_from_bank", "tr_env_from_polys", "tr_env_value", "tr_env_region_get",
                   "tr_env_point_get", "tr_env_point_refine", "tr_env_lookup_create", "tr_env_lookup_eval",
                   "tr_env_need", "tr_env_lookup_need"},
     .cuda_entry = {"tr_mgpu_env_lookup"}},
    {FAMILY(TR_REG_M_ATTR, "attr", "attr", "ATTR-001", "contribution attribution"),
     .state_kind = TR_REG_STATE_GAME, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(COUNT) | CB(BOOLEAN) | CB(KLEENE) | CB(INT128) | CB(MOD128) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_attr_game_prov", "tr_attr_game_lattice", "tr_attr_game_predicate", "tr_attr_game_table",
                   "tr_attr_support_build", "tr_attr_support_values", "tr_attr_interactions", "tr_attr_values",
                   "tr_attr_exhaustive", "tr_attr_explore", "tr_attr_search", "tr_attr_enum_need"},
     .cuda_entry = {"tr_mgpu_attr_values", "tr_mgpu_attr_exhaustive", "tr_mgpu_attr_explore", "tr_mgpu_attr_search"}},
    {FAMILY(TR_REG_M_COUNT, "count", "count", "COUNT-001", "observed-domain constraint counts"),
     .state_kind = TR_REG_STATE_CONSTRAINT_COUNTS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(KLEENE) | CB(BOOLEAN) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_EXACT_COUNT | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_count_set_create", "tr_count_observe", "tr_count_observe_adopt", "tr_count_universe",
                   "tr_count_run", "tr_count_counts_view", "tr_count_plan", "tr_count_adopt",
                   "tr_count_answer_fresh", "tr_count_answer_authorize", "tr_count_run_need", "tr_count_domain_need"},
     .cuda_entry = {"tr_mgpu_count_counts", "tr_mgpu_count_extract"}},
    {FAMILY(TR_REG_M_PARITY, "parity", "parity", "FIELD-002", "parity discovery and substitution"),
     .state_kind = TR_REG_STATE_PARITY_RELATIONS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(GF2) | CB(CODE) | CB(ROWSET) | CB(INTEGER) | CB(KLEENE) | CB(BOOLEAN) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_REG_LATTICE | TR_LAW_CHAR2 | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_parity_discover", "tr_parity_adopt", "tr_parity_claim", "tr_parity_discover_need",
                   "tr_parity_found_certificate", "tr_parity_substitute", "tr_parity_rewrite_program",
                   "tr_parity_verify", "tr_parity_verify_view", "tr_parity_verify_need"},
     .cuda_entry = {"tr_mgpu_parity_verify"}},
    {FAMILY(TR_REG_M_UNC, "unc", "unc", "UNC-001", "certified abstract states"),
     .state_kind = TR_REG_STATE_ABSTRACT, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(QINTERVAL) | CB(INTERVAL) | CB(RATIONAL) | CB(INTEGER) | CB(KLEENE) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_LATTICE | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_ENCLOSURE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY | TR_PROOF_PRIME,
     .cpu_entry = {"tr_unc_program_check", "tr_unc_state_from_scope", "tr_unc_state_from_inputs", "tr_unc_eval",
                   "tr_unc_eval_need", "tr_unc_trace", "tr_unc_rows", "tr_unc_filter", "tr_unc_sum_fits",
                   "tr_unc_cong_meet", "tr_unc_identity", "tr_unc_certify"},
     .cuda_entry = {"tr_mgpu_unc_eval", "tr_mgpu_unc_rows"}},
    {FAMILY(TR_REG_M_PRE, "pre", "pre", "PRE-001", "exact preimages of derived predicates"),
     .state_kind = TR_REG_STATE_CODE_SET, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(BIGINT) | CB(CODE) | CB(RATIONAL) | CB(KLEENE) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_pre_codec_for_column", "tr_pre_preimage", "tr_pre_set_get_info", "tr_pre_set_spans",
                   "tr_pre_set_contains", "tr_pre_set_get_view", "tr_pre_set_certificate", "tr_pre_scan"},
     .cuda_entry = {"tr_mgpu_pre_scan"}},
    {FAMILY(TR_REG_M_IDENT, "ident", "ident", "DISC-002", "moment-Gram identity discovery"),
     .state_kind = TR_REG_STATE_GRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INT128) | CB(GFP) | CB(BIGQ) | CB(BIGINT) | CB(INTEGER) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_PRIME | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_ident_plan_make", "tr_ident_gram_build", "tr_ident_gram_from_bank", "tr_ident_gram_from_lin",
                   "tr_ident_gram_combine", "tr_ident_kernel_build", "tr_ident_propose", "tr_ident_candidates",
                   "tr_ident_verify_view", "tr_ident_certify", "tr_ident_hypothesis", "tr_ident_support"},
     .cuda_entry = {"tr_mgpu_ident_gram", "tr_mgpu_ident_verify"}},
    {FAMILY(TR_REG_M_GEOM, "geom", "geom", "GEOM-001", "certified ideal reduction"),
     .state_kind = TR_REG_STATE_IDEAL, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(RATIONAL) | CB(INTEGER) | CB(INT128) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_geom_ideal_make", "tr_geom_reduce", "tr_geom_reduction_claims", "tr_geom_certify",
                   "tr_geom_cost_of", "tr_geom_choose", "tr_geom_pairs_make", "tr_geom_agree_view",
                   "tr_geom_checks_make", "tr_geom_checks_run", "tr_geom_derivative", "tr_geom_dependency_check"},
     .cuda_entry = {"tr_mgpu_geom_agree", "tr_mgpu_geom_check"}},
    {FAMILY(TR_REG_M_TENSOR_STATE, "tensor.state", "tensor", "TENSOR-001", "sparse and factored moment tensors"),
     .state_kind = TR_REG_STATE_MOMENT_TENSOR, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(INT128) | CB(COUNT) | CB(CODE),
     .laws_relied = TR_REG_RING | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_CELL_DOMAIN | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_tensor_build", "tr_tensor_init", "tr_tensor_accumulate", "tr_tensor_merge_builder",
                   "tr_tensor_finish", "tr_tensor_need", "tr_tensor_from_states", "tr_tensor_convert",
                   "tr_tensor_marginal", "tr_tensor_slice", "tr_tensor_permute", "tr_tensor_outer"},
     .cuda_entry = {"tr_mgpu_tensor_build"}},
    {FAMILY(TR_REG_M_TENSOR_CONTRACT, "tensor.contract", "tensor", "TENSOR-002", "costed contraction planning"),
     .state_kind = TR_REG_STATE_CONTRACTION_PLAN, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(INT128),
     .laws_relied = TR_REG_RING | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_BOUND,
     .cpu_entry = {"tr_tensor_net_plan", "tr_tensor_net_create", "tr_tensor_net_free", "tr_tensor_plan_get_info",
                   "tr_tensor_plan_get", "tr_tensor_plan_free", "tr_tensor_net_execute", "tr_tensor_net_program",
                   "tr_tensor_net_input_dense", "tr_tensor_default_rates"},
     .cuda_entry = {"tr_mgpu_tensor_contract"}},
    {FAMILY(TR_REG_M_KERNEL, "kernel", "kernel", "KERNEL-001", "exact finite-feature kernel embeddings"),
     .state_kind = TR_REG_STATE_KERNEL_FEATURES, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(COUNT) | CB(CODE),
     .laws_relied = TR_REG_FIELD | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_kernel_build", "tr_kernel_check", "tr_kernel_need", "tr_kernel_from_states",
                   "tr_kernel_get_info", "tr_kernel_get_group", "tr_kernel_get_tensor", "tr_kernel_eval_make",
                   "tr_kernel_eval_get_view", "tr_kernel_query", "tr_kernel_view_device_bytes", "tr_kernel_free"},
     .cuda_entry = {"tr_mgpu_kernel_build", "tr_mgpu_kernel_upload", "tr_mgpu_kernel_query", "tr_mgpu_kernel_bytes",
                    "tr_mgpu_kernel_free"}},
    {FAMILY(TR_REG_M_SYMCOUNT, "symcount", "symcount", "COUNT-002", "symbolic model counting"),
     .state_kind = TR_REG_STATE_DECISION_DIAGRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BOOLEAN) | CB(KLEENE) | CB(COUNT) | CB(BIGINT) | CB(BIGQ),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_symc_compile", "tr_symc_mgr_apply", "tr_symc_dd_count", "tr_symc_dd_levels",
                   "tr_symc_dd_adopt", "tr_symc_dd_witnesses", "tr_symc_brute", "tr_symc_brute_adopt",
                   "tr_symc_search", "tr_symc_factors_create", "tr_symc_factors_count", "tr_symc_factors_brute"},
     .cuda_entry = {"tr_mgpu_symcount_levels", "tr_mgpu_symcount_brute"}},
    {FAMILY(TR_REG_M_EXPLAIN, "explain", "explain", "PROV-002", "explanations and deletion propagation"),
     .state_kind = TR_REG_STATE_EXPLANATION, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BOOLEAN) | CB(COUNT) | CB(BIGQ) | CB(INTERVAL),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = TR_PROOF_BOUND,
     .cpu_entry = {"tr_expl_create", "tr_expl_create_need", "tr_expl_get_view", "tr_expl_propagate",
                   "tr_expl_propagate_need", "tr_expl_lineage", "tr_expl_explain", "tr_expl_probability_weights",
                   "tr_expl_probability_bounds"},
     .cuda_entry = {"tr_mgpu_explain_propagate"}},
    {FAMILY(TR_REG_M_FOURIER, "fourier", "fourier", "SPEC-001", "fixed spectral basis"),
     .state_kind = TR_REG_STATE_SPECTRAL_BASIS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(COUNT) | CB(INT128) | CB(QINTERVAL) | CB(BIGQ) | CB(CINTERVAL),
     .laws_relied = TR_REG_FOLD | TR_LAW_INVERSE | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_ENCLOSURE | TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_fourier_plan_create", "tr_fourier_build", "tr_fourier_sums", "tr_fourier_from_sums",
                   "tr_fourier_act", "tr_fourier_derive", "tr_fourier_exact", "tr_fourier_find",
                   "tr_fourier_require", "tr_fourier_walsh_inverse", "tr_fourier_device_need"},
     .cuda_entry = {"tr_mgpu_fourier_sums"}},
    {FAMILY(TR_REG_M_LINUPD, "lin.update", "linupd", "LIN-003", "leave-group-out and low-rank updates"),
     .state_kind = TR_REG_STATE_SOLVED_SYSTEM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(COUNT) | CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(GFP) | CB(INTERVAL) | CB(FLOAT64),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = TR_PROOF_BOUND | TR_PROOF_PRIME | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_linupd_fit", "tr_linupd_system", "tr_linupd_update", "tr_linupd_rows", "tr_linupd_ridge",
                   "tr_linupd_subtract", "tr_linupd_float_solve", "tr_linupd_logo_plan", "tr_linupd_logo_solve",
                   "tr_linupd_logo_residues", "tr_linupd_logo_finish", "tr_linupd_equal"},
     .cuda_entry = {"tr_mgpu_linupd_logo"}},
    {FAMILY(TR_REG_M_PROB, "prob", "prob", "PROB-001", "partition and likelihood functions"),
     .state_kind = TR_REG_STATE_SUPPORT_MASSES, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(COUNT) | CB(INT128) | CB(BIGQ) | CB(QINTERVAL) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_CELL_DOMAIN | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_prob_support_create", "tr_prob_model_create", "tr_prob_stats_build", "tr_prob_stats_product",
                   "tr_prob_stats_require", "tr_prob_plan_create", "tr_prob_sums", "tr_prob_finish", "tr_prob_exact",
                   "tr_prob_plan_need", "tr_prob_stats_need", "tr_prob_ln_bounds"},
     .cuda_entry = {"tr_mgpu_prob_sums"}},
    {FAMILY(TR_REG_M_SYM_ACTION, "sym.action", "sym", "SYM-001", "closed-basis transformation actions"),
     .state_kind = TR_REG_STATE_BASIS_ACTION, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(INT128) | CB(BIGINT) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_sym_geometry_make", "tr_sym_basis_full", "tr_sym_basis_create", "tr_sym_map_make",
                   "tr_sym_map_compose", "tr_sym_action_make", "tr_sym_action_compose", "tr_sym_action_identity",
                   "tr_sym_action_equal", "tr_sym_state_from_bank", "tr_sym_state_from_tensor", "tr_sym_query"},
     .cuda_entry = {"tr_mgpu_sym_query"}},
    {FAMILY(TR_REG_M_SYM_SYMMETRY, "sym.symmetry", "sym", "SYM-002", "certified symmetries and orbits"),
     .state_kind = TR_REG_STATE_SYMMETRY, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(INT128) | CB(COUNT) | CB(CODE),
     .laws_relied = TR_REG_RING | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_EXACT_COUNT | TR_PROOF_NULL_POLICY | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_sym_group_create", "tr_sym_support_build", "tr_sym_verify_view", "tr_sym_certify",
                   "tr_sym_adopt", "tr_sym_cert_fresh", "tr_sym_orbits_build", "tr_sym_poly_invariant",
                   "tr_sym_orbit_query", "tr_sym_rows_query"},
     .cuda_entry = {"tr_mgpu_sym_verify"}},
    {FAMILY(TR_REG_M_APPROX, "approx", "approx", "UNC-002", "explicit approximations"),
     .state_kind = TR_REG_STATE_APPROXIMATION, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(BIGINT) | CB(QINTERVAL) | CB(INTERVAL) | CB(FLOAT64) | CB(INT128) | CB(COUNT) | CB(RATIONAL) | CB(KLEENE) | CB(ROWSET),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_approx_series", "tr_approx_poly", "tr_approx_lowrank", "tr_approx_add", "tr_approx_mul",
                   "tr_approx_compose", "tr_approx_quantize", "tr_approx_float", "tr_approx_eval_rows",
                   "tr_approx_filter", "tr_approx_authorize_exact", "tr_approx_relation"},
     .cuda_entry = {"tr_mgpu_approx_eval"}},
    {FAMILY(TR_REG_M_REFINE, "refine", "refine", "MULTI-002", "adaptive refinement, localized updates"),
     .state_kind = TR_REG_STATE_REFINEMENT, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(COUNT) | CB(INT128) | CB(BIGINT) | CB(RATIONAL) | CB(QINTERVAL) | CB(KLEENE) | CB(ROWSET),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_REFINEMENT | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_refine_build", "tr_refine_begin", "tr_refine_supply", "tr_refine_run", "tr_refine_get_answer",
                   "tr_refine_eval_bounds", "tr_refine_fine", "tr_refine_update_state", "tr_refine_update_prepare",
                   "tr_refine_update_rollup", "tr_refine_update_finish", "tr_refine_get_changes"},
     .cuda_entry = {"tr_mgpu_refine_bounds", "tr_mgpu_refine_rollup"}},
    {FAMILY(TR_REG_M_INTERV, "interv", "interv", "INV-002", "constrained minimal interventions"),
     .state_kind = TR_REG_STATE_INTERVENTION, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(BIGINT) | CB(GFP) | CB(QINTERVAL) | CB(INTERVAL) | CB(INTEGER) | CB(MINMAX) | CB(INT128) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_PRIME | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_interv_linear", "tr_interv_linear_poly", "tr_interv_univariate", "tr_interv_grid_prepare",
                   "tr_interv_grid_eval", "tr_interv_grid_search", "tr_interv_agg_build", "tr_interv_agg_solve",
                   "tr_interv_agg_value", "tr_interv_result_get_info", "tr_interv_result_solution",
                   "tr_interv_result_free"},
     .cuda_entry = {"tr_mgpu_interv_grid"}},
    {FAMILY(TR_REG_M_POLYOPT, "polyopt", "polyopt", "OPT-002", "convex and polynomial optimization"),
     .state_kind = TR_REG_STATE_PROGRAM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(INTERVAL) | CB(QINTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = TR_PROOF_DOMAIN | TR_PROOF_BOUND,
     .cpu_entry = {"tr_polyopt_qp_create", "tr_polyopt_qp_from_quad", "tr_polyopt_qp_solve", "tr_polyopt_qp_value",
                   "tr_polyopt_poly_create", "tr_polyopt_poly_value", "tr_polyopt_boxes", "tr_polyopt_minimize",
                   "tr_polyopt_result_get_info", "tr_polyopt_result_part", "tr_polyopt_result_leaf",
                   "tr_polyopt_result_free"},
     .cuda_entry = {"tr_mgpu_polyopt_boxes"}},
    {FAMILY(TR_REG_M_GRAPH_LAYOUT, "graph.layout", "graph", "GRAPH-001", "declared graph layouts"),
     .state_kind = TR_REG_STATE_GRAPH_LAYOUT, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(COUNT) | CB(INT128) | CB(BOOLEAN) | CB(GF2) | CB(MINMAX) | CB(INTERVAL) | CB(ROWSET),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_CHAR2 | TR_LAW_ENCLOSURE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_graph_layout_build", "tr_graph_layout_need", "tr_graph_layout_get_view",
                   "tr_graph_layout_ids", "tr_graph_layout_same", "tr_graph_layout_fresh",
                   "tr_graph_layout_incidence", "tr_graph_degrees", "tr_graph_expand", "tr_graph_spmv",
                   "tr_graph_gf2_matrix", "tr_graph_need"},
     .cuda_entry = {"tr_mgpu_graph_degrees", "tr_mgpu_graph_expand", "tr_mgpu_graph_spmv"}},
    {FAMILY(TR_REG_M_GRAPH_PATHS, "graph.paths", "graph", "GRAPH-002", "bounded fixed points, paths and energies"),
     .state_kind = TR_REG_STATE_GRAPH_FIXPOINT, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BOOLEAN) | CB(ROWSET) | CB(COUNT) | CB(BIGINT) | CB(MINMAX) | CB(INT128) | CB(INTERVAL) | CB(INTEGER),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_ENCLOSURE | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_IDENTITY | TR_PROOF_BOUND,
     .cpu_entry = {"tr_graph_reach", "tr_graph_closure", "tr_graph_walks_need", "tr_graph_walks", "tr_graph_paths",
                   "tr_graph_energy", "tr_graph_pagerank", "tr_graph_power", "tr_graph_rationals"},
     .cuda_entry = {"tr_mgpu_graph_reach", "tr_mgpu_graph_closure", "tr_mgpu_graph_walks", "tr_mgpu_graph_paths",
                    "tr_mgpu_graph_energy", "tr_mgpu_graph_pagerank", "tr_mgpu_graph_power"}},
    {FAMILY(TR_REG_M_ELIM, "elim", "elim", "GEOM-002", "bounded elimination and generators"),
     .state_kind = TR_REG_STATE_ELIMINATION, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(RATIONAL) | CB(INTEGER) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_elim_select", "tr_elim_selection_get_info", "tr_elim_selection_candidates",
                   "tr_elim_selection_form", "tr_elim_selection_certify", "tr_elim_selection_pairs",
                   "tr_elim_pairs_make", "tr_elim_check_view", "tr_elim_cost_of", "tr_elim_choose",
                   "tr_elim_forms_make", "tr_elim_forms_certify"},
     .cuda_entry = {"tr_mgpu_elim_check"}},
    {FAMILY(TR_REG_M_PREIMG, "preimg", "preimg", "PRE-002", "roots, residues and periodic preimages"),
     .state_kind = TR_REG_STATE_CODE_SET, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(BIGINT) | CB(QINTERVAL) | CB(CODE) | CB(RATIONAL) | CB(INTEGER) | CB(GF2) | CB(KLEENE) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_preimg_preimage", "tr_preimg_set_get_info", "tr_preimg_set_branches",
                   "tr_preimg_set_contains", "tr_preimg_set_get_view", "tr_preimg_set_lookup", "tr_preimg_set_cubes",
                   "tr_preimg_set_certificate", "tr_preimg_scan", "tr_preimg_verify"},
     .cuda_entry = {"tr_mgpu_preimg_scan", "tr_mgpu_preimg_verify"}},
    {FAMILY(TR_REG_M_TOPO_COMPLEX, "topo.complex", "topo", "TOPO-001", "declared occupancy complexes"),
     .state_kind = TR_REG_STATE_CUBICAL_COMPLEX, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(CODE) | CB(ROWSET) | CB(COUNT),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_CELL_DOMAIN | TR_PROOF_EXACT_COUNT | TR_PROOF_NULL_POLICY | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_topo_build", "tr_topo_adopt", "tr_topo_count", "tr_topo_scan_needs", "tr_topo_build_needs",
                   "tr_topo_get_info", "tr_topo_cells", "tr_topo_local_get", "tr_topo_adjacency_get",
                   "tr_topo_values", "tr_topo_fresh", "tr_topo_equal"},
     .cuda_entry = {"tr_mgpu_topo_counts", "tr_mgpu_topo_build", "tr_mgpu_topo_local"}},
    {FAMILY(TR_REG_M_TOPO_HOMOLOGY, "topo.homology", "topo", "TOPO-002", "bounded cycles, homology, persistence"),
     .state_kind = TR_REG_STATE_CHAIN_REDUCTION, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(GF2) | CB(GFP) | CB(RATIONAL) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_PRIME | TR_PROOF_BOUND | TR_PROOF_IDENTITY,
     .cpu_entry = {"tr_topo_check", "tr_topo_check_view", "tr_topo_boundary", "tr_topo_boundary_packed",
                   "tr_topo_chain_create", "tr_topo_reduce_need", "tr_topo_homology_build",
                   "tr_topo_homology_classes", "tr_topo_persistence_build", "tr_topo_persistence_pairs",
                   "tr_topo_persistence_betti", "tr_topo_persistence_levels"},
     .cuda_entry = {"tr_mgpu_topo_dd"}},
    {FAMILY(TR_REG_M_PAIR_PLAN, "pair.plan", "pairplan", "PAIR-002", "structured pairwise query planning"),
     .state_kind = TR_REG_STATE_PAIR_PLAN, .devices = TR_REG_ON_BOTH, .finalize = 2,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(COUNT),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_pairplan_make", "tr_pairplan_plan_get_info", "tr_pairplan_plan_get",
                   "tr_pairplan_plan_kernel_view", "tr_pairplan_plan_free", "tr_pairplan_execute",
                   "tr_pairplan_eval_table", "tr_pairplan_retain", "tr_pairplan_state_get_info",
                   "tr_pairplan_state_table_view", "tr_pairplan_state_free", "tr_pairplan_default_rates"},
     .cuda_entry = {"tr_mgpu_pairplan_upload", "tr_mgpu_pairplan_query", "tr_mgpu_pairplan_bytes",
                    "tr_mgpu_pairplan_free"}},
    {FAMILY(TR_REG_M_CONSERVE, "conserve", "conserve", "DISC-003", "conservation across revisions"),
     .state_kind = TR_REG_STATE_TRACKER, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INT128) | CB(INTEGER) | CB(COUNT) | CB(BOOLEAN),
     .laws_relied = TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_cons_tracker_make", "tr_cons_job_make", "tr_cons_job_run", "tr_cons_tracker_update",
                   "tr_cons_tracker_rebind", "tr_cons_tracker_record", "tr_cons_tracker_classes",
                   "tr_cons_authorize", "tr_cons_tracker_gram", "tr_cons_conserve", "tr_cons_law_candidates",
                   "tr_cons_law_witnesses"},
     .cuda_entry = {"tr_mgpu_cons_delta"}},
    {FAMILY(TR_REG_M_EQUIV, "equiv", "equiv", "META-001", "scoped query equivalence"),
     .state_kind = TR_REG_STATE_QUERY_PARTS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(KLEENE) | CB(BOOLEAN) | CB(ROWSET) | CB(COUNT) | CB(INTEGER) | CB(RATIONAL) | CB(BIGINT),
     .laws_relied = TR_REG_FIELD | TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY | TR_PROOF_DOMAIN | TR_PROOF_CATALOG,
     .cpu_entry = {"tr_equiv_set_make", "tr_equiv_check_make", "tr_equiv_check_run", "tr_equiv_check_decide",
                   "tr_equiv_decide", "tr_equiv_authorize", "tr_equiv_family_run", "tr_equiv_rewrite",
                   "tr_equiv_poly_from_math", "tr_equiv_result_get", "tr_equiv_result_point",
                   "tr_equiv_result_certificate"},
     .cuda_entry = {"tr_mgpu_equiv_check"}},
    {FAMILY(TR_REG_M_FACTOR, "factor", "factor", "PROB-002", "factorized inference"),
     .state_kind = TR_REG_STATE_FACTOR_GRAPH, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(COUNT) | CB(RATIONAL) | CB(BIGINT) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_CELL_DOMAIN,
     .cpu_entry = {"tr_factor_graph_create", "tr_factor_plan_create", "tr_factor_prog_create",
                   "tr_factor_prog_bases", "tr_factor_eliminate", "tr_factor_answer_create", "tr_factor_kbest",
                   "tr_factor_alternative", "tr_factor_plan_get", "tr_factor_prog_need", "tr_factor_graph_table",
                   "tr_factor_answer_value"},
     .cuda_entry = {"tr_mgpu_factor_eliminate"}},
    {FAMILY(TR_REG_M_KAPPROX, "kapprox", "kapprox", "KERNEL-002", "approximate spectral feature kernels"),
     .state_kind = TR_REG_STATE_KERNEL_APPROX, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(COUNT) | CB(INT128) | CB(QINTERVAL) | CB(BIGINT) | CB(BIGQ) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_IDENTITY | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_kapprox_plan_create", "tr_kapprox_build", "tr_kapprox_sums", "tr_kapprox_from_sums",
                   "tr_kapprox_query", "tr_kapprox_need", "tr_kapprox_generate", "tr_kapprox_save",
                   "tr_kapprox_load", "tr_kapprox_fresh", "tr_kapprox_authorize_exact", "tr_kapprox_get_info"},
     .cuda_entry = {"tr_mgpu_kapprox_sums"}},
    {FAMILY(TR_REG_M_DYN_OPERATOR, "dyn.operator", "dyn", "DYN-001", "closed-observable dynamics"),
     .state_kind = TR_REG_STATE_TRANSFER_OPERATOR, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(INT128) | CB(BIGINT) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_dyn_map_from_sym", "tr_dyn_map_polynomial", "tr_dyn_operator_make", "tr_dyn_operator_need",
                   "tr_dyn_operator_check_geometry", "tr_dyn_authorize_exact", "tr_dyn_evolve", "tr_dyn_rows_evolve",
                   "tr_dyn_apply_view_eval", "tr_dyn_evolution_authorize_exact", "tr_dyn_need", "tr_dyn_export"},
     .cuda_entry = {"tr_mgpu_dyn_apply"}},
    {FAMILY(TR_REG_M_DYN_POWERS, "dyn.powers", "dyn", "DYN-002", "operator powers and recurrences"),
     .state_kind = TR_REG_STATE_OPERATOR_POWERS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(RATIONAL) | CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(GFP) | CB(INTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = DATA_PROOFS | TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_PRIME,
     .cpu_entry = {"tr_dyn_power_make", "tr_dyn_power_adopt", "tr_dyn_power_compose", "tr_dyn_power_equal",
                   "tr_dyn_semigroup", "tr_dyn_powmod_view_eval", "tr_dyn_evolvemod_view_eval",
                   "tr_dyn_evolution_adopt", "tr_dyn_charpoly", "tr_dyn_recurrence_make", "tr_dyn_recurrence_term",
                   "tr_dyn_float_power"},
     .cuda_entry = {"tr_mgpu_dyn_powmod", "tr_mgpu_dyn_evolvemod"}},
    {FAMILY(TR_REG_M_REDUCE, "reduce", "reduce", "FND-006", "fused multi-output reduction"),
     .state_kind = TR_REG_STATE_FUSED_OUTPUTS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(CODE) | CB(INTEGER) | CB(INT128) | CB(COUNT) | CB(KLEENE),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_CELL_DOMAIN | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_reduce_compile", "tr_reduce_choose", "tr_reduce_run", "tr_reduce_need", "tr_reduce_derive",
                   "tr_reduce_stage_create", "tr_reduce_stage_finish", "tr_reduce_stage_put_bank",
                   "tr_reduce_stage_put_hist", "tr_reduce_result_bank", "tr_reduce_result_hist", "tr_reduce_get_view"},
     .cuda_entry = {"tr_mgpu_reduce_fused", "tr_mgpu_reduce_fused_device", "tr_mgpu_reduce_states_arena",
                    "tr_mgpu_reduce_states_copy", "tr_mgpu_reduce_states_free"}},
    {FAMILY(TR_REG_M_PROGRAM, "program", "program", "META-002", "query family programs"),
     .state_kind = TR_REG_STATE_QUERY_PROGRAM, .devices = TR_REG_ON_BOTH, .finalize = 3,
     .carriers = CB(INTEGER) | CB(INT128) | CB(BIGINT) | CB(BIGQ) | CB(COUNT) | CB(KLEENE),
     .laws_relied = TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_DOMAIN | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_prog_compile", "tr_prog_finalize", "tr_prog_need", "tr_prog_compare",
                   "tr_prog_snapshot_compare", "tr_prog_final_view_get", "tr_prog_final_need", "tr_prog_query_get",
                   "tr_prog_measure_get", "tr_prog_output_get", "tr_prog_get_info", "tr_prog_reduce_plan"},
     .cuda_entry = {"tr_mgpu_program_finalize", "tr_mgpu_program_run"}},
    {FAMILY(TR_REG_M_REGION, "region", "region", "ENV-002", "multivariate parameter regions"),
     .state_kind = TR_REG_STATE_PARAMETER_REGIONS, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(BIGQ) | CB(BIGINT) | CB(GFP) | CB(QINTERVAL),
     .laws_relied = TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT | TR_LAW_ENCLOSURE, .laws_guaranteed = TR_LAW_EXACT | TR_LAW_ENCLOSURE,
     .proofs = TR_PROOF_DOMAIN | TR_PROOF_BOUND | TR_PROOF_PRIME,
     .cpu_entry = {"tr_region_build", "tr_region_from_polys", "tr_region_need", "tr_region_get_info",
                   "tr_region_region_get", "tr_region_vertex_get", "tr_region_boundary_get",
                   "tr_region_lookup_create", "tr_region_lookup_create_q64", "tr_region_lookup_eval",
                   "tr_region_lookup_need", "tr_region_authorize"},
     .cuda_entry = {"tr_mgpu_region_locate"}},
    {FAMILY(TR_REG_M_REPAIR, "repair", "repair", "ATTR-002", "explanation and repair search"),
     .state_kind = TR_REG_STATE_SUBSET_PROBLEM, .devices = TR_REG_ON_BOTH, .finalize = 1,
     .carriers = CB(INTEGER) | CB(COUNT) | CB(BOOLEAN) | CB(KLEENE) | CB(ROWSET) | CB(INT128) | CB(BIGQ),
     .laws_relied = TR_REG_FIELD | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT, .laws_guaranteed = TR_LAW_EXACT,
     .proofs = DATA_PROOFS | TR_PROOF_BOUND | TR_PROOF_NULL_POLICY,
     .cpu_entry = {"tr_repair_masks", "tr_repair_separate", "tr_repair_explain", "tr_repair_keep",
                   "tr_repair_program", "tr_repair_game", "tr_repair_delete", "tr_repair_prepare",
                   "tr_repair_search", "tr_repair_redundancy", "tr_repair_edit", "tr_repair_threshold"},
     .cuda_entry = {"tr_mgpu_repair_eval", "tr_mgpu_repair_scan", "tr_mgpu_repair_search"}},
};
#undef BUILTIN
#undef FAMILY
_Static_assert(sizeof modules / sizeof modules[0] == TR_REG_M_END - 1, "one descriptor per module id");

/* the implementation rows of the module, the operator mask they make; for the two first-slice built-ins the operator
 * set of their validated table (math_modules.c), so the registry never claims more than tr_math_module_apply
 * dispatches */
static void describe(int32_t index, tr_reg_module *out) {
    *out = modules[index];
    out->nimpl = 0;
    for (int32_t w = 0; w < TR_REG_OPERATOR_WORDS; w++) out->operators[w] = 0;
    for (int32_t i = 0; i < TR_REG_IMPL_ROWS; i++) {
        const tr_reg_impl *row = &tr_reg_impls[i];
        if (row->module != out->id) continue;
        if (out->nimpl < TR_REG_IMPLS) out->impl[out->nimpl++] = *row;
        out->operators[row->op / 64] |= UINT64_C(1) << (row->op % 64);
    }
    if (out->id == TR_REG_R_FORMAL || out->id == TR_REG_R_MOMENTS) {
        for (int32_t i = 0; i < tr_math_module_count(); i++) {
            tr_math_module_info info;
            if (tr_math_module_get(i, &info) == TR_MATH_OK && info.id == out->id) {
                out->operators[0] = info.operators;
                for (int32_t w = 1; w < TR_REG_OPERATOR_WORDS; w++) out->operators[w] = 0;
            }
        }
    }
}
int32_t tr_reg_module_count(void) { return (int32_t)(sizeof modules / sizeof modules[0]); }
int32_t tr_reg_module_get(int32_t index, tr_reg_module *out) {
    if (!out || index < 0 || index >= tr_reg_module_count()) return TR_MATH_INVALID;
    describe(index, out);
    return TR_MATH_OK;
}
int32_t tr_reg_module_by_id(int32_t id, tr_reg_module *out) {
    for (int32_t i = 0; i < tr_reg_module_count(); i++)
        if (modules[i].id == id) return tr_reg_module_get(i, out);
    return TR_MATH_INVALID;
}
int32_t tr_reg_find(int32_t op, int32_t carrier, int32_t devices) { return tr_reg_find_impl(op, carrier, devices); }
int32_t tr_reg_check_chain(const tr_reg_step *steps, int32_t n, int32_t input, int32_t contract, int32_t require_impl,
                           int32_t *output, tr_reg_refusal *r) {
    int32_t rc = tr_reg_check_laws(steps, n, input, contract, output, r);
    if (rc || !require_impl) return rc;
    for (int32_t i = 0; i < n; i++)
        if (!tr_reg_find(steps[i].op, steps[i].carrier, TR_REG_ON_CPU | TR_REG_ON_CUDA))
            return tr_reg_refuse(r, TR_REG_REFUSE_IMPLEMENTATION, TR_MATH_CARRIER_ERROR, i, steps[i].op,
                                 steps[i].carrier, 0, 0, 0);
    return TR_MATH_OK;
}

int32_t tr_reg_code_domain(const uint32_t *codes, uint64_t rows, uint64_t *ncodes, int32_t *width) {
    if (!ncodes || !width || (rows && !codes)) return TR_MATH_INVALID;
    uint32_t m = 0;
    for (uint64_t i = 0; i < rows; i++) if (codes[i] > m) m = codes[i];
    *ncodes = (uint64_t)m + 1;
    *width = m ? 32 - __builtin_clz(m) : 0;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ engine columns */
struct tr_reg_column_s {
    uint64_t rows, revision, ncodes, bytes;
    int64_t offset;
    int32_t col, valid_col, width, reserved;
    uint32_t *codes, *validity;
};
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
tr_reg_column *tr_reg_column_create(const atable_t *t, int32_t col, int32_t valid_col, int64_t offset,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    set_error(error, TR_MATH_OK);
    if (!t || col < 0 || col >= t->k || valid_col < -1 || valid_col >= t->k || valid_col == col) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (t->width[col] > 32 || (valid_col >= 0 && t->width[valid_col] > 32)) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    uint64_t bytes = tr_budget_add(tr_budget_mul(t->n, valid_col >= 0 ? 8 : 4), sizeof(struct tr_reg_column_s));
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, bytes, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_reg_column *c = calloc(1, sizeof *c);
    size_t n = t->n ? (size_t)t->n : 1;
    if (c) c->codes = malloc(n * sizeof *c->codes);
    if (c && c->codes && valid_col >= 0) c->validity = malloc(n * sizeof *c->validity);
    if (!c || !c->codes || (valid_col >= 0 && !c->validity)) { tr_reg_column_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (t->n) {
        at_decode_column(t, col, c->codes);
        if (valid_col >= 0) at_decode_column(t, valid_col, c->validity);
    }
    c->rows = t->n; c->revision = t->revision; c->offset = offset; c->col = col; c->valid_col = valid_col;
    c->bytes = bytes;
    tr_reg_code_domain(c->codes, c->rows, &c->ncodes, &c->width);
    return c;
}
int32_t tr_reg_column_request(const tr_reg_column *c, const atable_t *t, tr_reg_request *r) {
    if (!c || !t || !r) return TR_MATH_INVALID;
    if (t->revision != c->revision || t->n != c->rows) return TR_MATH_STALE;
    r->rows = c->rows; r->codes = c->codes; r->validity = c->validity; r->offset = c->offset;
    r->ncodes = c->ncodes; r->width = c->width; r->source = (uint64_t)(uintptr_t)c; r->revision = c->revision;
    return TR_MATH_OK;
}
uint64_t tr_reg_column_bytes(const tr_reg_column *c) { return c ? c->bytes : 0; }
void tr_reg_column_free(tr_reg_column *c) {
    if (!c) return;
    free(c->codes); free(c->validity); free(c);
}

/* ------------------------------------------------------------------ states and the CPU executors */
struct tr_reg_state_s {
    tr_reg_state_info info;
    uint64_t words, data_words, nsets;
    uint64_t *selection;          /* the scope (a built or imported state only); NULL: every row */
    uint32_t *sets;               /* BITPLANE plane sets */
    uint64_t *data;               /* HISTOGRAM counts, POWER residues (lo, hi), BITPLANE counts */
};
static void state_release(struct tr_reg_state_s *s) { free(s->selection); free(s->sets); free(s->data); }
static uint64_t effective(const tr_reg_request *r, uint64_t w) {
    return (r->selected ? r->selected[w] : ~(uint64_t)0) & tr_reg_rowmask(r->rows, w);
}
static int32_t highest_bit(uint32_t m) { return m ? 31 - __builtin_clz(m) : -1; }

/* scans the selected rows once for a state representation (or SCAN when st == NULL: res gets the fold). Flags:
 * a code >= ncodes in a selected valid row (invalid), a checked evaluation outside int64 (overflow). */
static int32_t scan_rows(const tr_reg_request *r, int32_t repr, struct tr_reg_state_s *st, tr_reg_result *res,
                         tr_reg_explain *ex) {
    uint64_t words = tr_reg_words(r), selected = 0, valid = 0, nulls = 0;
    int invalid = 0, overflow = 0;
    __int128 sum[TR_REG_POLYS] = {0, 0, 0, 0};
    int64_t ext[TR_REG_POLYS];
    for (int32_t j = 0; j < TR_REG_POLYS; j++) ext[j] = r->aggregate == TR_REG_MIN ? INT64_MAX : INT64_MIN;
    tr_u128 power[TR_REG_DEGREE + 1];
    int32_t top = st ? highest_bit(st->info.support) : -1, width = r->width;
    for (int32_t k = 0; k <= TR_REG_DEGREE; k++) power[k] = 0;
    for (uint64_t w = 0; w < words; w++) {
        uint64_t m = effective(r, w), mvalid = 0, planes[32];
        if (!m) continue;
        if (repr == TR_REG_R_BITPLANE) for (int32_t j = 0; j < width; j++) planes[j] = 0;
        while (m) {
            int b = __builtin_ctzll(m);
            m &= m - 1;
            uint64_t row = w * 64 + (uint64_t)b;
            selected++;
            if (r->validity && !r->validity[row]) { nulls++; continue; }
            valid++;
            uint32_t c = r->codes[row];
            if ((uint64_t)c >= r->ncodes) { invalid = 1; continue; }
            int64_t x = (int64_t)((__int128)c + r->offset);
            switch (repr) {
            case TR_REG_R_SCAN:
                for (int32_t j = 0; j < r->npolys; j++) {
                    int64_t v;
                    if (!tr_reg_eval(r->coef[j], r->degree[j], x, &v)) { overflow = 1; continue; }
                    if (r->aggregate == TR_REG_SUM) sum[j] += v;
                    else if (r->aggregate == TR_REG_MIN) { if (v < ext[j]) ext[j] = v; }
                    else if (v > ext[j]) ext[j] = v;
                }
                break;
            case TR_REG_R_HISTOGRAM: st->data[c]++; break;
            case TR_REG_R_POWER: {
                tr_u128 xr = (tr_u128)(__int128)x, p = 1;
                for (int32_t k = 1; k <= top; k++) { p *= xr; if ((st->info.support >> k) & 1) power[k] += p; }
                break;
            }
            case TR_REG_R_BITPLANE:
                mvalid |= (uint64_t)1 << b;
                for (uint32_t bits = c; bits; bits &= bits - 1) planes[__builtin_ctz(bits)] |= (uint64_t)1 << b;
                break;
            }
        }
        if (repr == TR_REG_R_BITPLANE && mvalid)
            for (uint64_t s = 0; s < st->nsets; s++) {
                uint64_t v = mvalid;
                for (uint32_t S = st->sets[s]; S && v; S &= S - 1) v &= planes[__builtin_ctz(S)];
                st->data[s] += (uint64_t)__builtin_popcountll(v);
            }
    }
    if (ex) { ex->rows_scanned = selected; ex->words_scanned = words; }
    res->selected = selected; res->valid = valid; res->nulls = nulls;
    if (st) { st->info.selected = selected; st->info.valid = valid; st->info.nulls = nulls; }
    if (repr == TR_REG_R_POWER) {
        int32_t idx = 0;
        for (int32_t k = 0; k <= TR_REG_DEGREE; k++) {
            if (!((st->info.support >> k) & 1)) continue;
            tr_u128 v = k == 0 ? (tr_u128)valid : power[k];
            st->data[2 * idx] = (uint64_t)v; st->data[2 * idx + 1] = (uint64_t)(v >> 64);
            idx++;
        }
    }
    if (invalid) return TR_MATH_INVALID;
    if (overflow) return TR_MATH_OVERFLOW;
    if (repr == TR_REG_R_SCAN)
        for (int32_t j = 0; j < r->npolys; j++)
            res->value[j] = r->aggregate == TR_REG_SUM ? tr_reg_i128_of((tr_u128)sum[j]) : tr_reg_i128_of_i64(ext[j]);
    return TR_MATH_OK;
}

/* allocates the arrays of a state representation (support/D resolved) */
static int32_t state_alloc(const tr_reg_request *r, int32_t repr, uint32_t support, int32_t D, struct tr_reg_state_s *st) {
    memset(st, 0, sizeof *st);
    st->info.representation = repr; st->info.support = support; st->info.degree = D;
    st->info.width = r->width; st->info.has_validity = r->validity != NULL;
    st->info.rows = r->rows; st->info.ncodes = r->ncodes; st->info.offset = r->offset;
    st->info.source = r->source; st->info.revision = r->revision;
    st->words = tr_reg_words(r);
    uint64_t elements = tr_reg_state_elements(r, repr, support, D);
    st->info.elements = elements;
    st->data_words = repr == TR_REG_R_POWER ? 2 * elements : elements;
    st->info.bytes = st->data_words * 8;
    st->data = calloc(st->data_words ? (size_t)st->data_words : 1, 8);
    if (!st->data) return TR_MATH_MEMORY;
    if (repr == TR_REG_R_BITPLANE) {
        st->nsets = elements;
        st->sets = malloc((size_t)(elements ? elements : 1) * sizeof *st->sets);
        if (!st->sets) return TR_MATH_MEMORY;
        tr_reg_plane_sets_fill(r->width, D, st->sets, elements);
        st->info.bytes += elements * 4;
    }
    return TR_MATH_OK;
}
static int32_t contract(const tr_reg_request *r, const struct tr_reg_state_s *st, tr_reg_result *res, uint64_t *work) {
    int32_t rc = TR_MATH_OK;
    *work = 0;
    if (st->info.representation == TR_REG_R_HISTOGRAM) rc = tr_reg_hist_contract(r, st->data, res, work);
    else if (st->info.representation == TR_REG_R_POWER) tr_reg_power_contract(r, st->info.support, st->data, res);
    else tr_reg_plane_contract(r, st->sets, st->data, st->nsets, res, work);
    res->selected = st->info.selected; res->valid = st->info.valid; res->nulls = st->info.nulls;
    return rc;
}
static uint32_t resolve_support(const tr_reg_request *r, uint32_t support) {
    return (support ? support : tr_reg_support(r)) | 1u;
}
static int32_t resolve_degree(const tr_reg_request *r, int32_t D) { return D > 0 ? D : tr_reg_max_degree(r); }
static int is_state_repr(int32_t repr) {
    return repr == TR_REG_R_HISTOGRAM || repr == TR_REG_R_POWER || repr == TR_REG_R_BITPLANE;
}
/* a reused state suffices: same aggregate family, POWER exponents covered, BITPLANE degree covered */
static int sufficient(const tr_reg_state_info *in, const tr_reg_request *r) {
    if (in->representation == TR_REG_R_HISTOGRAM) return 1;
    if (r->aggregate != TR_REG_SUM) return 0;
    if (in->representation == TR_REG_R_POWER) return (tr_reg_support(r) & ~in->support) == 0;
    return tr_reg_max_degree(r) <= in->degree;
}
static int32_t refuse_reg(tr_reg_explain *ex, int32_t kind, int32_t code) {
    if (ex && ex->refusal.kind == TR_REG_REFUSE_NONE) tr_reg_refuse(&ex->refusal, kind, code, -1, 0, 0, 0, 0, 0);
    return tr_reg_explain_refuse(ex, kind, code);
}
static int32_t kind_of(int32_t code) {
    switch (code) {
    case TR_MATH_INVALID: return TR_REG_REFUSE_INVALID;
    case TR_MATH_MEMORY: return TR_REG_REFUSE_MEMORY;
    case TR_MATH_OVERFLOW: return TR_REG_REFUSE_OVERFLOW;
    case TR_MATH_BUDGET: return TR_REG_REFUSE_BUDGET;
    case TR_MATH_STALE: return TR_REG_REFUSE_STALE;
    case TR_MATH_DEVICE: return TR_REG_REFUSE_DEVICE;
    default: return TR_REG_REFUSE_UNSUPPORTED;
    }
}
/* the checks every CPU execution and state build makes before allocating */
static int32_t precheck(const tr_reg_request *r, int32_t repr, tr_reg_explain *ex) {
    tr_math_refusal kernel;
    int32_t rc = tr_reg_request_check(r, &kernel);
    if (rc) {
        if (ex && rc == TR_MATH_BUDGET) ex->budget = kernel;
        return refuse_reg(ex, kind_of(rc), rc);
    }
    if (repr < TR_REG_R_SCAN || repr > TR_REG_R_EMPTY) return refuse_reg(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
    tr_reg_refusal law;
    rc = tr_reg_check_representation(repr, r, &law);
    if (rc) {
        if (ex) ex->refusal = law;
        return tr_reg_explain_refuse(ex, law.kind, rc);
    }
    return TR_MATH_OK;
}
static int32_t budget_gate(const tr_math_budget *budget, const tr_math_need *need, tr_reg_explain *ex) {
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_refusal refusal;
    int32_t rc = tr_budget_check(budget, need, &refusal);
    if (rc) {
        if (ex) ex->budget = refusal;
        return refuse_reg(ex, TR_REG_REFUSE_BUDGET, rc);
    }
    return TR_MATH_OK;
}

tr_reg_state *tr_reg_state_build(const tr_reg_request *r, int32_t repr, uint32_t support, int32_t D,
                                 const tr_math_budget *budget, tr_reg_explain *ex, int32_t *error) {
    tr_reg_explain_init(ex, repr, TR_REG_DEV_CPU, TR_REG_FINAL_CPU);
    tr_reg_explain_entry(ex, "tr_reg_state_build");
    set_error(error, TR_MATH_OK);
    int32_t rc = precheck(r, repr, ex);
    if (!rc && !is_state_repr(repr)) rc = refuse_reg(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
    if (rc) { set_error(error, rc); return NULL; }
    support = resolve_support(r, support); D = resolve_degree(r, D);
    if (D > TR_REG_DEGREE || (repr == TR_REG_R_POWER && (support >> (TR_REG_DEGREE + 1)))) {
        set_error(error, refuse_reg(ex, TR_REG_REFUSE_INVALID, TR_MATH_INVALID)); return NULL;
    }
    tr_math_need need;
    tr_reg_need(r, repr, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 0, support, D, &need);
    if (r->selected) need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_bitmap_bytes(r->rows));
    rc = budget_gate(budget, &need, ex);
    if (rc) { set_error(error, rc); return NULL; }
    struct tr_reg_state_s *st = calloc(1, sizeof *st);
    if (!st) { set_error(error, refuse_reg(ex, TR_REG_REFUSE_MEMORY, TR_MATH_MEMORY)); return NULL; }
    rc = state_alloc(r, repr, support, D, st);
    tr_reg_result scratch;
    tr_reg_result_init(&scratch, r);
    if (!rc) rc = scan_rows(r, repr, st, &scratch, ex);
    if (!rc && r->selected) {
        st->selection = malloc((size_t)(st->words ? st->words : 1) * 8);
        if (!st->selection) rc = TR_MATH_MEMORY;
        else for (uint64_t w = 0; w < st->words; w++) st->selection[w] = effective(r, w);
        st->info.has_selection = 1;
    }
    if (rc) {
        state_release(st); free(st);
        set_error(error, refuse_reg(ex, kind_of(rc), rc)); return NULL;
    }
    if (ex) { ex->state_elements = st->info.elements; ex->state_bytes = st->info.bytes; ex->native_calls = 1; }
    return st;
}

tr_reg_state *tr_reg_state_import(const tr_reg_request *r, const tr_reg_state_info *in, const void *raw, uint64_t bytes,
                                  int32_t *error) {
    set_error(error, TR_MATH_OK);
    tr_math_refusal kernel;
    if (!r || !in || (bytes && !raw) || tr_reg_request_check(r, &kernel) || !is_state_repr(in->representation) ||
        in->rows != r->rows || in->ncodes != r->ncodes || in->width != r->width || in->offset != r->offset ||
        in->source != r->source || in->revision != r->revision || in->has_validity != (r->validity != NULL) ||
        in->degree < 0 || in->degree > TR_REG_DEGREE || (in->support >> (TR_REG_DEGREE + 1)) || !(in->support & 1) ||
        in->selected != in->valid + in->nulls) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    struct tr_reg_state_s *st = calloc(1, sizeof *st);
    if (!st) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t rc = state_alloc(r, in->representation, in->support, in->degree, st);
    if (!rc && (bytes != st->data_words * 8 || in->elements != st->info.elements)) rc = TR_MATH_INVALID;
    if (!rc) {
        memcpy(st->data, raw, (size_t)bytes);
        uint64_t total = 0, first = st->data_words ? st->data[0] : 0;
        if (in->representation == TR_REG_R_HISTOGRAM) for (uint64_t c = 0; c < r->ncodes; c++) total += st->data[c];
        int consistent = in->representation == TR_REG_R_HISTOGRAM ? total == in->valid
                       : in->representation == TR_REG_R_POWER ? first == in->valid && st->data[1] == 0
                       : first == in->valid;
        if (!consistent || in->selected != tr_reg_selected_count(r)) rc = TR_MATH_INVALID;
    }
    if (!rc && r->selected) {
        st->selection = malloc((size_t)(st->words ? st->words : 1) * 8);
        if (!st->selection) rc = TR_MATH_MEMORY;
        else for (uint64_t w = 0; w < st->words; w++) st->selection[w] = effective(r, w);
        st->info.has_selection = 1;
    }
    if (rc) { state_release(st); free(st); set_error(error, rc); return NULL; }
    st->info.selected = in->selected; st->info.valid = in->valid; st->info.nulls = in->nulls;
    return st;
}
int32_t tr_reg_state_size(const tr_reg_request *r, int32_t repr, uint32_t support, int32_t D, int32_t device,
                          const tr_math_budget *budget, uint64_t *bytes, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!bytes || !is_state_repr(repr) || (device != TR_REG_ON_CPU && device != TR_REG_ON_CUDA)) return TR_MATH_INVALID;
    int32_t rc = tr_reg_request_check(r, refusal);
    if (rc) return rc;
    support = resolve_support(r, support); D = resolve_degree(r, D);
    if (D > TR_REG_DEGREE || (support >> (TR_REG_DEGREE + 1))) return TR_MATH_INVALID;
    tr_math_need need;
    tr_reg_need(r, repr, device, TR_REG_FINAL_CPU, 0, support, D, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    *bytes = tr_reg_state_elements(r, repr, support, D) * (repr == TR_REG_R_POWER ? 16 : 8);
    return TR_MATH_OK;
}
int32_t tr_reg_state_get_info(const tr_reg_state *st, tr_reg_state_info *out) {
    if (!st || !out) return TR_MATH_INVALID;
    *out = st->info;
    return TR_MATH_OK;
}
int32_t tr_reg_state_raw(const tr_reg_state *st, void *out, uint64_t capacity) {
    if (!st || (!out && st->data_words) || capacity < st->data_words * 8) return TR_MATH_INVALID;
    if (st->data_words) memcpy(out, st->data, (size_t)(st->data_words * 8));
    return TR_MATH_OK;
}
int32_t tr_reg_state_matches(const tr_reg_state *st, const tr_reg_request *r) {
    if (!st || !r) return TR_MATH_INVALID;
    const tr_reg_state_info *in = &st->info;
    if (in->rows != r->rows || in->ncodes != r->ncodes || in->width != r->width || in->offset != r->offset ||
        in->source != r->source || in->revision != r->revision || in->has_validity != (r->validity != NULL)) return 0;
    for (uint64_t w = 0; w < st->words; w++) {
        uint64_t mine = st->selection ? st->selection[w] : tr_reg_rowmask(r->rows, w);
        if (mine != effective(r, w)) return 0;
    }
    return 1;
}
void tr_reg_state_free(tr_reg_state *st) {
    if (!st) return;
    state_release(st); free(st);
}

int32_t tr_reg_execute(const tr_reg_request *r, int32_t repr, const tr_reg_state *reuse, const tr_math_budget *budget,
                       tr_reg_result *res, tr_reg_explain *ex) {
    tr_reg_explain_init(ex, repr, TR_REG_DEV_CPU, TR_REG_FINAL_CPU);
    tr_reg_explain_entry(ex, "tr_reg_execute");
    if (!res) return refuse_reg(ex, TR_REG_REFUSE_INVALID, TR_MATH_INVALID);
    int32_t rc = precheck(r, repr, ex);
    if (rc) return rc;
    tr_reg_result_init(res, r);
    if (ex) ex->native_calls = 1;
    if (reuse) {
        const tr_reg_state_info *in = &reuse->info;
        if (in->representation != repr || !is_state_repr(repr)) return refuse_reg(ex, TR_REG_REFUSE_INVALID, TR_MATH_INVALID);
        if (tr_reg_state_matches(reuse, r) != 1) return refuse_reg(ex, TR_REG_REFUSE_STALE, TR_MATH_STALE);
        if (!sufficient(in, r)) return refuse_reg(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
        tr_math_need need;
        tr_reg_need(r, repr, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 1, in->support, in->degree, &need);
        if ((rc = budget_gate(budget, &need, ex))) return rc;
        if (repr != TR_REG_R_HISTOGRAM && !tr_reg_bound_ok(r, in->selected)) {
            if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, -1, TR_OP_CONVERT, TR_REG_C_INT128, 0,
                                  TR_REG_C_MOD128, TR_REG_C_INT128);
            return refuse_reg(ex, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW);
        }
        uint64_t work;
        if (ex) { ex->reused = 1; ex->words_scanned = reuse->words; ex->state_elements = in->elements; ex->state_bytes = in->bytes; }
        rc = contract(r, reuse, res, &work);
        return rc ? refuse_reg(ex, kind_of(rc), rc) : TR_MATH_OK;
    }
    if (repr == TR_REG_R_EMPTY) {
        uint64_t n = tr_reg_selected_count(r);
        if (ex) ex->words_scanned = tr_reg_words(r);
        if (n) {
            if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_PROOF, TR_MATH_UNSUPPORTED, -1, TR_OP_EMPTY, 0, 0, 0, 0);
            return refuse_reg(ex, TR_REG_REFUSE_PROOF, TR_MATH_UNSUPPORTED);
        }
        return TR_MATH_OK;                                   /* zero sums, no rows (MIN/MAX: no value) */
    }
    uint32_t support = resolve_support(r, 0);
    int32_t D = resolve_degree(r, 0);
    tr_math_need need;
    tr_reg_need(r, repr, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 0, support, D, &need);
    if ((rc = budget_gate(budget, &need, ex))) return rc;
    if ((repr == TR_REG_R_POWER || repr == TR_REG_R_BITPLANE) && !tr_reg_bound_ok(r, tr_reg_selected_count(r))) {
        if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, -1, TR_OP_CONVERT, TR_REG_C_INT128, 0,
                              TR_REG_C_MOD128, TR_REG_C_INT128);
        return refuse_reg(ex, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW);
    }
    if (repr == TR_REG_R_SCAN) {
        rc = scan_rows(r, repr, NULL, res, ex);
        return rc ? refuse_reg(ex, kind_of(rc), rc) : TR_MATH_OK;
    }
    struct tr_reg_state_s st;
    rc = state_alloc(r, repr, support, D, &st);
    if (!rc) rc = scan_rows(r, repr, &st, res, ex);
    uint64_t work = 0;
    if (!rc) rc = contract(r, &st, res, &work);
    if (ex) { ex->state_elements = st.info.elements; ex->state_bytes = st.info.bytes; }
    state_release(&st);
    return rc ? refuse_reg(ex, kind_of(rc), rc) : TR_MATH_OK;
}

/* ------------------------------------------------------------------ candidates (the bounded search) */
#define BIT(c) (1u << (c))
#define SHARED_SCAN (BIT(TR_COST_WORDS) | BIT(TR_COST_SCAN) | BIT(TR_COST_FIXED) | BIT(TR_COST_TRANSFER))
#define SHARED_STATE (SHARED_SCAN | BIT(TR_COST_CHAIN) | BIT(TR_COST_CONSTRUCT))
static void candidate_init(tr_reg_candidate *c, int32_t repr, int32_t device, int32_t finalize, int32_t reuse, int32_t batch) {
    memset(c, 0, sizeof *c);
    c->module = repr; c->representation = repr; c->device = device; c->finalize = finalize;
    c->reuse = reuse; c->batch = batch; c->valid = 1;
    tr_reg_refusal_none(&c->refusal); tr_budget_none(&c->budget);
}
static void reject(tr_reg_candidate *c, int32_t kind, int32_t code) {
    if (!c->valid) return;
    c->valid = 0; c->code = code;
    if (c->refusal.kind == TR_REG_REFUSE_NONE) tr_reg_refuse(&c->refusal, kind, code, -1, 0, 0, 0, 0, 0);
}
#define MUL tr_budget_mul
#define ADD tr_budget_add
/* the amounts of a representation's execution (see tr_reg_candidate; S the estimated selected rows) */
static void amounts(tr_reg_candidate *c, const tr_reg_request *r, uint64_t S, uint32_t support, int32_t D) {
    uint64_t W = tr_reg_words(r), k = (uint64_t)r->npolys, V = r->validity ? 2 : 1, T = tr_reg_terms(r);
    uint64_t C = r->ncodes, O = S < C ? S : C, P = tr_reg_plane_sets(r->width, D), NW = S < W ? S : W;
    uint64_t m = (uint64_t)tr_reg_popcount32(support), out = 16 * k + 24, *a = c->amount;
    uint64_t setbits = 0, moebius = 0;
    for (int32_t s = 0; s <= D && s <= r->width; s++)
        setbits = ADD(setbits, MUL((uint64_t)s, tr_budget_binomial((uint64_t)r->width, (uint64_t)s)));
    for (int32_t j = 0; j < r->npolys; j++)
        moebius = ADD(moebius, MUL(tr_reg_plane_terms(r->width, r->degree[j]), (uint64_t)r->degree[j] + 1));
    int cuda = c->device == TR_REG_ON_CUDA;
    uint64_t in = tr_reg_input_bytes(r);
    a[TR_COST_MATERIALIZE] = out;
    if (c->reuse) {
        a[TR_COST_REUSE] = W + 1;
        a[TR_COST_FIXED] = 1;
        if (c->representation == TR_REG_R_HISTOGRAM) a[TR_COST_CPU_FINAL] = ADD(MUL(k, C), MUL(O, T));
        else if (c->representation == TR_REG_R_POWER) a[TR_COST_CPU_FINAL] = T;
        else { a[TR_COST_CPU_FINAL] = MUL(k, P); a[TR_COST_COMBINE] = moebius; }
        return;
    }
    switch (c->representation) {
    case TR_REG_R_EMPTY:
        a[TR_COST_WORDS] = W; a[TR_COST_FIXED] = 1;
        return;
    case TR_REG_R_SCAN:
        a[TR_COST_CHAIN] = MUL(S, T);
        c->shared = SHARED_SCAN;
        if (cuda) { a[TR_COST_GPU_FINAL] = k; a[TR_COST_TRANSFER] = ADD(MUL(k, in), out + 16); }
        else a[TR_COST_CPU_FINAL] = k;
        break;
    case TR_REG_R_HISTOGRAM:
        a[TR_COST_CHAIN] = MUL(k, S); a[TR_COST_CONSTRUCT] = MUL(k, C);
        c->shared = SHARED_STATE;
        if (!cuda) a[TR_COST_CPU_FINAL] = ADD(MUL(k, C), MUL(O, T));
        else if (c->finalize == TR_REG_FINAL_GPU) {
            a[TR_COST_GPU_FINAL] = ADD(MUL(k, C), MUL(O, T)); a[TR_COST_TRANSFER] = ADD(MUL(k, in), out + 16);
        } else {
            a[TR_COST_CPU_FINAL] = ADD(MUL(k, C), MUL(O, T)); a[TR_COST_TRANSFER] = ADD(MUL(k, ADD(in, MUL(C, 8))), 24);
        }
        break;
    case TR_REG_R_POWER:
        a[TR_COST_CHAIN] = MUL(MUL(k, S), (uint64_t)(D > 0 ? D : 1)); a[TR_COST_CONSTRUCT] = MUL(k, m);
        a[TR_COST_CPU_FINAL] = T;
        c->shared = SHARED_STATE;
        if (cuda) a[TR_COST_TRANSFER] = ADD(MUL(k, ADD(in, MUL(m, 16))), 24);
        break;
    case TR_REG_R_BITPLANE:
        a[TR_COST_CHAIN] = MUL(k, ADD(MUL(S, (uint64_t)r->width), MUL(NW, setbits)));
        a[TR_COST_CONSTRUCT] = MUL(k, P); a[TR_COST_COMBINE] = moebius; a[TR_COST_CPU_FINAL] = MUL(k, P);
        c->shared = SHARED_STATE;
        if (cuda) a[TR_COST_TRANSFER] = ADD(MUL(k, ADD(in, MUL(P, 12))), 24);
        break;
    }
    a[TR_COST_WORDS] = MUL(k, W);
    a[TR_COST_SCAN] = MUL(MUL(k, S), V);
    a[TR_COST_FIXED] = cuda && c->representation == TR_REG_R_HISTOGRAM && c->finalize == TR_REG_FINAL_GPU ? 2 * k : k;
    a[TR_COST_FUSION] = k - 1;
}
static void candidate_fill(tr_reg_candidate *c, const tr_reg_request *r, const tr_reg_state *cached, uint64_t S,
                           uint64_t rows_bound, int32_t empty_proven) {
    uint32_t support = resolve_support(r, 0);
    int32_t D = resolve_degree(r, 0);
    if (c->reuse && cached) { support = cached->info.support; D = cached->info.degree; }
    tr_reg_refusal law;
    int32_t rc = tr_reg_check_representation(c->representation, r, &law);
    if (rc) { c->valid = 0; c->code = rc; c->refusal = law; }
    if (c->representation == TR_REG_R_EMPTY && !empty_proven) {
        if (c->valid) tr_reg_refuse(&c->refusal, TR_REG_REFUSE_PROOF, TR_MATH_UNSUPPORTED, -1, TR_OP_EMPTY, 0, 0, 0, 0);
        reject(c, TR_REG_REFUSE_PROOF, TR_MATH_UNSUPPORTED);
    }
    if (c->reuse) {
        if (tr_reg_state_matches(cached, r) != 1) reject(c, TR_REG_REFUSE_STALE, TR_MATH_STALE);
        else if (!sufficient(&cached->info, r)) reject(c, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
        else if (c->representation != TR_REG_R_HISTOGRAM && !tr_reg_bound_ok(r, cached->info.selected)) {
            if (c->valid) tr_reg_refuse(&c->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, -1, TR_OP_CONVERT,
                                        TR_REG_C_INT128, 0, TR_REG_C_MOD128, TR_REG_C_INT128);
            reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW);
        }
    } else if ((c->representation == TR_REG_R_POWER || c->representation == TR_REG_R_BITPLANE) && c->valid &&
               !tr_reg_bound_ok(r, rows_bound)) {
        tr_reg_refuse(&c->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, -1, TR_OP_CONVERT, TR_REG_C_INT128, 0,
                      TR_REG_C_MOD128, TR_REG_C_INT128);
        reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW);
    }
    /* a plan admits SCAN and HISTOGRAM only when no checked evaluation can leave int64 over the code domain */
    if ((c->representation == TR_REG_R_SCAN || c->representation == TR_REG_R_HISTOGRAM) && c->valid &&
        !tr_reg_eval_bound_ok(r)) {
        tr_reg_refuse(&c->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, -1, TR_OP_EVAL, TR_REG_C_INTEGER, 0, 0, 0);
        reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW);
    }
    tr_reg_need(r, c->representation, c->device, c->finalize, c->reuse, support, D, &c->need);
    amounts(c, r, S, support, D);
}
int32_t tr_reg_candidates(const tr_reg_request *r, const tr_reg_state *cached, int32_t devices, uint64_t estimate,
                          uint64_t rows_bound, int32_t empty_proven, const tr_math_budget *budget, tr_reg_candidate *out,
                          int32_t capacity, int32_t *count, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || !count || capacity < 0) return TR_MATH_INVALID;
    *count = 0;
    tr_math_refusal kernel;
    int32_t rc = tr_reg_request_check(r, &kernel);
    if (rc) { if (refusal && rc == TR_MATH_BUDGET) *refusal = kernel; return rc; }
    if (!(devices & TR_REG_ON_CPU) && !(devices & TR_REG_ON_CUDA)) return TR_MATH_INVALID;
    if (cached && !is_state_repr(cached->info.representation)) return TR_MATH_INVALID;
    int32_t plan[TR_REG_MAX_CANDIDATES][4], n = 0;    /* representation, device, finalize, reuse */
#define PLAN(rp, dv, fn, ru) do { plan[n][0] = rp; plan[n][1] = dv; plan[n][2] = fn; plan[n][3] = ru; n++; } while (0)
    if (devices & TR_REG_ON_CPU) {
        PLAN(TR_REG_R_EMPTY, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 0);
        for (int32_t rp = TR_REG_R_SCAN; rp <= TR_REG_R_BITPLANE; rp++) PLAN(rp, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 0);
    }
    if (cached) PLAN(cached->info.representation, TR_REG_ON_CPU, TR_REG_FINAL_CPU, 1);
    if (devices & TR_REG_ON_CUDA) {
        PLAN(TR_REG_R_SCAN, TR_REG_ON_CUDA, TR_REG_FINAL_GPU, 0);
        PLAN(TR_REG_R_HISTOGRAM, TR_REG_ON_CUDA, TR_REG_FINAL_GPU, 0);
        PLAN(TR_REG_R_HISTOGRAM, TR_REG_ON_CUDA, TR_REG_FINAL_CPU, 0);
        PLAN(TR_REG_R_POWER, TR_REG_ON_CUDA, TR_REG_FINAL_CPU, 0);
        PLAN(TR_REG_R_BITPLANE, TR_REG_ON_CUDA, TR_REG_FINAL_CPU, 0);
    }
#undef PLAN
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if ((rc = tr_budget_check_one(budget, TR_BUDGET_ITERATIONS, (uint64_t)n, refusal))) return rc;
    if (n > capacity) return TR_MATH_INVALID;
    uint64_t S = estimate < r->rows ? estimate : r->rows;
    for (int32_t i = 0; i < n; i++) {
        candidate_init(&out[i], plan[i][0], plan[i][1], plan[i][2], plan[i][3], r->npolys);
        candidate_fill(&out[i], r, cached, S, rows_bound, empty_proven);
    }
    *count = n;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ DISC-001 through the registry */
int32_t tr_reg_verify(const tr_disc_view *v, int64_t scale, int64_t bias, tr_disc_evidence *e, tr_reg_explain *ex) {
    tr_reg_explain_init(ex, TR_REG_R_VERIFY, TR_REG_DEV_CPU, TR_REG_FINAL_CPU);
    tr_reg_explain_entry(ex, "tr_disc_verify");
    int32_t rc = tr_disc_verify(v, scale, bias, e);
    if (ex && v) { ex->rows_scanned = v->rows; ex->words_scanned = (v->rows + 63) / 64; ex->native_calls = 1; }
    return rc ? refuse_reg(ex, kind_of(rc), rc) : TR_MATH_OK;
}
int32_t tr_reg_candidates_verify(const tr_disc_view *v, int32_t devices, const tr_math_budget *budget,
                                 tr_reg_candidate *out, int32_t capacity, int32_t *count, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!v || !out || !count || (v->arity != 1 && v->arity != 2) || !(devices & (TR_REG_ON_CPU | TR_REG_ON_CUDA)))
        return TR_MATH_INVALID;
    *count = 0;
    int32_t n = ((devices & TR_REG_ON_CPU) != 0) + ((devices & TR_REG_ON_CUDA) != 0);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_ITERATIONS, (uint64_t)n, refusal);
    if (rc) return rc;
    if (n > capacity) return TR_MATH_INVALID;
    uint64_t columns = (uint64_t)v->arity + 1, W = (v->rows + 63) / 64;
    for (int32_t i = 0; i <= v->arity; i++) if (v->validity[i]) columns++;
    int32_t at = 0;
    for (int32_t dv = TR_REG_ON_CPU; dv <= TR_REG_ON_CUDA; dv <<= 1) {
        if (!(devices & dv)) continue;
        tr_reg_candidate *c = &out[at++];
        candidate_init(c, TR_REG_R_VERIFY, dv, dv == TR_REG_ON_CUDA ? TR_REG_FINAL_GPU : TR_REG_FINAL_CPU, 0, 1);
        for (int k = 0; k < TR_BUDGET_RESOURCES; k++) c->need.need[k] = 0;
        c->need.need[TR_BUDGET_VARIABLES] = (uint64_t)v->arity + 1;
        c->need.need[TR_BUDGET_OUTPUT] = 1;
        c->amount[TR_COST_WORDS] = W;
        c->amount[TR_COST_SCAN] = MUL(v->rows, columns);
        c->amount[TR_COST_CHAIN] = v->used_count;
        c->amount[TR_COST_MATERIALIZE] = sizeof(tr_disc_evidence);
        c->amount[TR_COST_FIXED] = 1;
        if (dv == TR_REG_ON_CUDA) {
            c->need.need[TR_BUDGET_BYTES] = tr_disc_bytes(v);
            c->amount[TR_COST_TRANSFER] = ADD(tr_disc_bytes(v), sizeof(tr_disc_evidence));
            c->amount[TR_COST_GPU_FINAL] = 1;
        } else c->amount[TR_COST_CPU_FINAL] = 1;
    }
    *count = n;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ transforms through the registry */
/* the module's own operation and carrier codes (math_lattice.h) */
static int32_t lattice_op(int32_t op, int32_t variant) {
    if (op == TR_OP_ZETA) return variant == TR_REG_UP ? TR_LATTICE_ZETA_UP : TR_LATTICE_ZETA_DOWN;
    return variant == TR_REG_UP ? TR_LATTICE_MOBIUS_UP : TR_LATTICE_MOBIUS_DOWN;
}
int32_t tr_reg_transform(int32_t op, int32_t variant, int32_t carrier, int32_t input, int32_t k, const tr_math_i128 *in,
                         tr_math_i128 *out, const tr_math_budget *budget, tr_math_refusal *refusal, tr_reg_explain *ex) {
    tr_reg_explain_begin(ex, 0, 0, TR_REG_DEV_CPU, TR_REG_FINAL_CPU);
    tr_budget_none(refusal);
    int32_t module = 0;
    int32_t rc = tr_reg_transform_check(op, variant, carrier, input, k, in, out, TR_REG_ON_CPU, &module, ex);
    if (rc) return rc;
    const int lattice = module == TR_REG_M_LATTICE_TRANSFORM;
    const int32_t fop = lattice ? lattice_op(op, variant) : 0;
    const int32_t fcarrier = carrier == TR_REG_C_INTEGER ? TR_LATTICE_I64 : TR_LATTICE_I128;
    tr_reg_explain_entry(ex, lattice ? "tr_lattice_transform" : "tr_spec_walsh");
    /* the module's budget first, so that nothing is read before it passes (the module checks it again) */
    tr_math_need need;
    if (lattice ? tr_lattice_transform_need(k, fop, fcarrier, &need) : tr_spec_need(k, &need))
        return refuse_reg(ex, TR_REG_REFUSE_INVALID, TR_MATH_INVALID);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_refusal br;
    if ((rc = tr_budget_check(budget, &need, &br))) {
        if (refusal) *refusal = br;
        if (ex) ex->budget = br;
        return refuse_reg(ex, TR_REG_REFUSE_BUDGET, rc);
    }
    if (k > TR_REG_TRANSFORM_MAX_K) return refuse_reg(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
    if ((rc = tr_reg_transform_values(op, carrier, input, k, in, ex))) return rc;
    rc = lattice ? tr_lattice_transform(k, fop, fcarrier, in, out, budget, refusal)
                 : tr_spec_walsh(k, in, out, budget, NULL, refusal);
    if (ex) {
        ex->state_elements = (uint64_t)1 << k; ex->state_bytes = ((uint64_t)1 << k) * 16; ex->native_calls = 1;
        if (rc == TR_MATH_BUDGET && refusal) ex->budget = *refusal;
    }
    return rc ? refuse_reg(ex, kind_of(rc), rc) : TR_MATH_OK;
}
