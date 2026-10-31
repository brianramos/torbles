/* FND-005 on a CUDA device: a request "aggregate f_j(x) over the selected rows" executed through a chosen
 * representation (math_registry.h): SCAN (every selected valid row evaluated on the device, folded exactly),
 * HISTOGRAM (counts per code on the device, then contracted on the device or copied back and contracted on the CPU),
 * POWER (sparse power sums modulo 2^128) and BITPLANE (counts per set of bit planes), both finalized on the CPU.
 * Each thread accumulates its rows locally and adds its counts, sums and extremes with one atomic per counter (sums
 * as 128-bit two-word adds), so results do not depend on thread order; the histogram and plane counts are atomic
 * integer adds. Results equal tr_reg_execute's bit for bit. Also DISC-001's verifier and the LATTICE-002 / SPEC-002
 * transforms through the registry (tr_mgpu_registry_verify, tr_mgpu_registry_transform: the composition check and
 * the implementing module from tr_reg_impls before anything is read, then the family's own kernels).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_registry.h"
#include "math_lattice.h"
#include "math_spec.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_reg_request, 1);          /* the ctypes boundary (math_handles.h) */
TR_MGPU_ABI(tr_reg_result, 1);
TR_MGPU_ABI(tr_reg_explain, 2);
TR_MGPU_ABI(tr_reg_state_info, 1);

/* a full 128-bit value added modulo 2^128 into acc[0] (low word), acc[1] (high word): the carry of the low add goes
 * into the high word (a local equivalent of math_cuda.h's tr_atomic_add_i128 for 128-bit addends) */
TR_DEVICE void tr_reg_atomic_add_u128(uint64_t *acc, tr_u128 v) {
    uint64_t lo = (uint64_t)v, hi = (uint64_t)(v >> 64);
    uint64_t old = tr_atomic_add_u64(&acc[0], lo);
    uint64_t carry = (old + lo) < old;
    if (hi + carry) tr_atomic_add_u64(&acc[1], hi + carry);
}
#ifdef EMULATE
static inline int tr_reg_ctz64(uint64_t x) { return __builtin_ctzll(x); }
#else
__device__ static inline int tr_reg_ctz64(uint64_t x) { return __ffsll((long long)x) - 1; }
#endif
TR_DEVICE uint64_t tr_reg_dev_effective(const tr_reg_request *r, uint64_t w) {
    return (r->selected ? r->selected[w] : ~(uint64_t)0) & tr_reg_rowmask(r->rows, w);
}

/* the accumulator block (TR_REG_ACC_BYTES): sums (lo, hi) x 4, extremes x 4, counters selected / valid / nulls /
 * invalid / overflow */
enum { ACC_SUMS = 0, ACC_EXT = 8, ACC_SELECTED = 12, ACC_VALID = 13, ACC_NULLS = 14, ACC_INVALID = 15,
       ACC_OVERFLOW = 16, ACC_WORDS = 20 };

TR_DEVICE void tr_reg_flush(uint64_t *acc, uint64_t selected, uint64_t valid, uint64_t nulls, uint64_t invalid,
                            uint64_t overflow) {
    if (selected) tr_atomic_add_u64(&acc[ACC_SELECTED], selected);
    if (valid) tr_atomic_add_u64(&acc[ACC_VALID], valid);
    if (nulls) tr_atomic_add_u64(&acc[ACC_NULLS], nulls);
    if (invalid) tr_atomic_or_u64(&acc[ACC_INVALID], 1);
    if (overflow) tr_atomic_or_u64(&acc[ACC_OVERFLOW], 1);
}
TR_DEVICE void tr_reg_fold(const tr_reg_request *r, int64_t x, __int128 *sum, int64_t *ext, uint64_t *overflow) {
    for (int32_t j = 0; j < r->npolys; j++) {
        int64_t v;
        if (!tr_reg_eval(r->coef[j], r->degree[j], x, &v)) { *overflow = 1; continue; }
        if (r->aggregate == TR_REG_SUM) sum[j] += v;
        else if (r->aggregate == TR_REG_MIN) { if (v < ext[j]) ext[j] = v; }
        else if (v > ext[j]) ext[j] = v;
    }
}
TR_DEVICE void tr_reg_flush_fold(const tr_reg_request *r, uint64_t *acc, const __int128 *sum, const int64_t *ext,
                                 uint64_t any) {
    if (!any) return;
    for (int32_t j = 0; j < r->npolys; j++) {
        if (r->aggregate == TR_REG_SUM) { if (sum[j]) tr_reg_atomic_add_u128(&acc[ACC_SUMS + 2 * j], (tr_u128)sum[j]); }
        else if (r->aggregate == TR_REG_MIN) tr_atomic_min_i64((int64_t *)&acc[ACC_EXT + j], ext[j]);
        else tr_atomic_max_i64((int64_t *)&acc[ACC_EXT + j], ext[j]);
    }
}

/* SCAN: one thread per row stride; every selected valid row's values folded locally */
TR_KERNEL void tr_reg_scan_kernel(tr_reg_request r, uint64_t *acc) {
    uint64_t selected = 0, valid = 0, nulls = 0, invalid = 0, overflow = 0;
    __int128 sum[TR_REG_POLYS] = {0, 0, 0, 0};
    int64_t ext[TR_REG_POLYS];
    for (int32_t j = 0; j < TR_REG_POLYS; j++) ext[j] = r.aggregate == TR_REG_MIN ? INT64_MAX : INT64_MIN;
    TR_GRID_LOOP(row, r.rows) {
        if (r.selected && !tr_reg_bit(r.selected, row)) continue;
        selected++;
        if (r.validity && !r.validity[row]) { nulls++; continue; }
        valid++;
        uint32_t c = r.codes[row];
        if ((uint64_t)c >= r.ncodes) { invalid = 1; continue; }
        tr_reg_fold(&r, (int64_t)((__int128)c + r.offset), sum, ext, &overflow);
    }
    tr_reg_flush(acc, selected, valid, nulls, invalid, overflow);
    tr_reg_flush_fold(&r, acc, sum, ext, valid);
}
/* HISTOGRAM: one count per selected valid row, added at its code */
TR_KERNEL void tr_reg_hist_kernel(tr_reg_request r, uint64_t *counts, uint64_t *acc) {
    uint64_t selected = 0, valid = 0, nulls = 0, invalid = 0;
    TR_GRID_LOOP(row, r.rows) {
        if (r.selected && !tr_reg_bit(r.selected, row)) continue;
        selected++;
        if (r.validity && !r.validity[row]) { nulls++; continue; }
        valid++;
        uint32_t c = r.codes[row];
        if ((uint64_t)c >= r.ncodes) { invalid = 1; continue; }
        tr_atomic_add_u64(&counts[c], 1);
    }
    tr_reg_flush(acc, selected, valid, nulls, invalid, 0);
}
/* HISTOGRAM finalized on the device: sum_c count_c f(c + offset), or min/max over the occupied codes */
TR_KERNEL void tr_reg_hist_contract_kernel(tr_reg_request r, const uint64_t *counts, uint64_t *acc) {
    uint64_t overflow = 0, any = 0;
    __int128 sum[TR_REG_POLYS] = {0, 0, 0, 0};
    int64_t ext[TR_REG_POLYS];
    for (int32_t j = 0; j < TR_REG_POLYS; j++) ext[j] = r.aggregate == TR_REG_MIN ? INT64_MAX : INT64_MIN;
    TR_GRID_LOOP(c, r.ncodes) {
        uint64_t n = counts[c];
        if (!n) continue;
        any = 1;
        int64_t x = (int64_t)((__int128)c + r.offset);
        for (int32_t j = 0; j < r.npolys; j++) {
            int64_t v;
            if (!tr_reg_eval(r.coef[j], r.degree[j], x, &v)) { overflow = 1; continue; }
            if (r.aggregate == TR_REG_SUM) sum[j] += (__int128)v * (__int128)n;
            else if (r.aggregate == TR_REG_MIN) { if (v < ext[j]) ext[j] = v; }
            else if (v > ext[j]) ext[j] = v;
        }
    }
    if (overflow) tr_atomic_or_u64(&acc[ACC_OVERFLOW], 1);
    tr_reg_flush_fold(&r, acc, sum, ext, any);
}
/* POWER: S_k = sum x^k modulo 2^128 for the exponents k >= 1 of support (S_0 is the valid count) */
TR_KERNEL void tr_reg_power_kernel(tr_reg_request r, uint32_t support, int32_t top, uint64_t *sums, uint64_t *acc) {
    uint64_t selected = 0, valid = 0, nulls = 0, invalid = 0;
    tr_u128 local[TR_REG_DEGREE + 1];
    for (int32_t k = 0; k <= TR_REG_DEGREE; k++) local[k] = 0;
    TR_GRID_LOOP(row, r.rows) {
        if (r.selected && !tr_reg_bit(r.selected, row)) continue;
        selected++;
        if (r.validity && !r.validity[row]) { nulls++; continue; }
        valid++;
        uint32_t c = r.codes[row];
        if ((uint64_t)c >= r.ncodes) { invalid = 1; continue; }
        tr_u128 xr = (tr_u128)((__int128)c + r.offset), p = 1;
        for (int32_t k = 1; k <= top; k++) { p *= xr; if ((support >> k) & 1) local[k] += p; }
    }
    tr_reg_flush(acc, selected, valid, nulls, invalid, 0);
    int32_t idx = 1;
    for (int32_t k = 1; k <= top; k++) {
        if (!((support >> k) & 1)) continue;
        if (local[k]) tr_reg_atomic_add_u128(&sums[2 * idx], local[k]);
        idx++;
    }
}
/* BITPLANE: one thread per word; the word's selected valid rows as bit planes, then one count per plane set */
TR_KERNEL void tr_reg_plane_kernel(tr_reg_request r, const uint32_t *sets, uint64_t nsets, uint64_t *counts,
                                   uint64_t *acc) {
    uint64_t selected = 0, valid = 0, nulls = 0, invalid = 0, words = (r.rows + 63) / 64;
    TR_GRID_LOOP(w, words) {
        uint64_t m = tr_reg_dev_effective(&r, w), mvalid = 0, planes[32];
        if (!m) continue;
        for (int32_t j = 0; j < r.width; j++) planes[j] = 0;
        while (m) {
            int b = tr_reg_ctz64(m);
            m &= m - 1;
            uint64_t row = w * 64 + (uint64_t)b;
            selected++;
            if (r.validity && !r.validity[row]) { nulls++; continue; }
            valid++;
            uint32_t c = r.codes[row];
            if ((uint64_t)c >= r.ncodes) { invalid = 1; continue; }
            mvalid |= (uint64_t)1 << b;
            for (uint32_t bits = c; bits; bits &= bits - 1) planes[tr_reg_ctz64(bits)] |= (uint64_t)1 << b;
        }
        if (!mvalid) continue;
        for (uint64_t s = 0; s < nsets; s++) {
            uint64_t v = mvalid;
            for (uint32_t S = sets[s]; S && v; S &= S - 1) v &= planes[tr_reg_ctz64(S)];
            if (v) tr_atomic_add_u64(&counts[s], (uint64_t)tr_popc64(v));
        }
    }
    tr_reg_flush(acc, selected, valid, nulls, invalid, 0);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_disc_verify(const tr_disc_view *, int64_t, int64_t, const tr_math_budget *, tr_disc_evidence *,
                        tr_math_refusal *);
#ifdef __cplusplus
}
#endif

/* the device copies of a request, its accumulators and (state representations) its device state */
typedef struct {
    tr_reg_request d;                /* the request with device pointers */
    uint32_t *codes, *validity, *sets;
    uint64_t *selected, *acc, *state, acc_host[ACC_WORDS];
    uint64_t elements, state_words, nsets, copied;
    uint32_t support;
    int32_t D, launches;
    uint32_t *host_sets;
} tr_reg_device;

static void device_free(tr_reg_device *g) {
    tr_cuda_free(g->codes); tr_cuda_free(g->validity); tr_cuda_free(g->sets); tr_cuda_free(g->selected);
    tr_cuda_free(g->acc); tr_cuda_free(g->state); free(g->host_sets);
}
static int32_t refuse(tr_reg_explain *ex, int32_t kind, int32_t code) {
    if (ex && ex->refusal.kind == TR_REG_REFUSE_NONE) tr_reg_refuse(&ex->refusal, kind, code, -1, 0, 0, 0, 0, 0);
    return tr_reg_explain_refuse(ex, kind, code);
}
static int32_t kind_of(int32_t code) {
    switch (code) {
    case TR_REG_E_INVALID: return TR_REG_REFUSE_INVALID;
    case TR_REG_E_MEMORY: return TR_REG_REFUSE_MEMORY;
    case TR_REG_E_OVERFLOW: return TR_REG_REFUSE_OVERFLOW;
    case TR_REG_E_BUDGET: return TR_REG_REFUSE_BUDGET;
    case TR_MATH_DEVICE: return TR_REG_REFUSE_DEVICE;
    case TR_MATH_STALE: return TR_REG_REFUSE_STALE;
    default: return TR_REG_REFUSE_UNSUPPORTED;
    }
}
/* request, representation and laws, finalization placement, budget (device copies included) and the bound, before
 * anything is allocated; then the inputs are copied and the representation's build kernel runs */
static int32_t device_build(const tr_reg_request *r, int32_t repr, int32_t finalize, uint32_t support, int32_t D,
                            const tr_math_budget *budget, tr_reg_device *g, tr_reg_explain *ex) {
    memset(g, 0, sizeof *g);
    tr_math_refusal kernel;
    int32_t rc = tr_reg_request_check(r, &kernel);
    if (rc) { if (ex && rc == TR_REG_E_BUDGET) ex->budget = kernel; return refuse(ex, kind_of(rc), rc); }
    if (repr < TR_REG_R_SCAN || repr > TR_REG_R_BITPLANE) return refuse(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
    int ok_final = repr == TR_REG_R_HISTOGRAM ? (finalize == TR_REG_FINAL_CPU || finalize == TR_REG_FINAL_GPU)
                 : repr == TR_REG_R_SCAN ? finalize == TR_REG_FINAL_GPU : finalize == TR_REG_FINAL_CPU;
    if (!ok_final) return refuse(ex, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED);
    tr_reg_refusal law;
    rc = tr_reg_check_representation(repr, r, &law);
    if (rc) { if (ex) ex->refusal = law; return tr_reg_explain_refuse(ex, law.kind, rc); }
    g->support = (support ? support : tr_reg_support(r)) | 1u;
    g->D = D > 0 ? D : tr_reg_max_degree(r);
    if (g->D > TR_REG_DEGREE || (g->support >> (TR_REG_DEGREE + 1))) return refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    tr_math_need need;
    tr_reg_need(r, repr, TR_REG_ON_CUDA, finalize, 0, g->support, g->D, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_refusal br;
    rc = tr_budget_check(budget, &need, &br);
    if (rc) { if (ex) ex->budget = br; return refuse(ex, TR_REG_REFUSE_BUDGET, rc); }
    if ((repr == TR_REG_R_POWER || repr == TR_REG_R_BITPLANE) && !tr_reg_bound_ok(r, tr_reg_selected_count(r))) {
        if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_BOUND, TR_REG_E_OVERFLOW, -1, TR_OP_CONVERT, TR_REG_C_INT128, 0,
                              TR_REG_C_MOD128, TR_REG_C_INT128);
        return refuse(ex, TR_REG_REFUSE_BOUND, TR_REG_E_OVERFLOW);
    }
    g->elements = tr_reg_state_elements(r, repr, g->support, g->D);
    g->state_words = repr == TR_REG_R_POWER ? 2 * g->elements : g->elements;
    g->d = *r;
    for (int i = 0; i < ACC_WORDS; i++) g->acc_host[i] = 0;
    for (int j = 0; j < TR_REG_POLYS; j++)
        g->acc_host[ACC_EXT + j] = (uint64_t)(r->aggregate == TR_REG_MIN ? INT64_MAX : INT64_MIN);
    uint64_t words = tr_reg_words(r);
    rc = tr_cuda_alloc((void **)&g->acc, sizeof g->acc_host);
    if (!rc) rc = tr_cuda_copy_in(g->acc, g->acc_host, sizeof g->acc_host);
    g->copied = sizeof g->acc_host;
    if (!rc && r->rows) {
        rc = tr_cuda_alloc((void **)&g->codes, (size_t)r->rows * 4);
        if (!rc) rc = tr_cuda_copy_in(g->codes, r->codes, (size_t)r->rows * 4);
        g->d.codes = g->codes; g->copied += r->rows * 4;
        if (!rc && r->validity) {
            rc = tr_cuda_alloc((void **)&g->validity, (size_t)r->rows * 4);
            if (!rc) rc = tr_cuda_copy_in(g->validity, r->validity, (size_t)r->rows * 4);
            g->d.validity = g->validity; g->copied += r->rows * 4;
        }
        if (!rc && r->selected) {
            rc = tr_cuda_alloc((void **)&g->selected, (size_t)words * 8);
            if (!rc) rc = tr_cuda_copy_in(g->selected, r->selected, (size_t)words * 8);
            g->d.selected = g->selected; g->copied += words * 8;
        }
    }
    if (!rc && repr != TR_REG_R_SCAN) rc = tr_cuda_alloc((void **)&g->state, (size_t)(g->state_words ? g->state_words : 1) * 8);
    if (!rc && repr == TR_REG_R_BITPLANE) {
        g->nsets = g->elements;
        g->host_sets = (uint32_t *)malloc((size_t)(g->nsets ? g->nsets : 1) * 4);
        if (!g->host_sets) rc = TR_REG_E_MEMORY;
        else {
            tr_reg_plane_sets_fill(r->width, g->D, g->host_sets, g->nsets);
            rc = tr_cuda_alloc((void **)&g->sets, (size_t)(g->nsets ? g->nsets : 1) * 4);
            if (!rc) rc = tr_cuda_copy_in(g->sets, g->host_sets, (size_t)g->nsets * 4);
            g->copied += g->nsets * 4;
        }
    }
    if (rc) return refuse(ex, kind_of(rc), rc);
    switch (repr) {
    case TR_REG_R_SCAN: TR_LAUNCH(tr_reg_scan_kernel, r->rows, g->d, g->acc); break;
    case TR_REG_R_HISTOGRAM: TR_LAUNCH(tr_reg_hist_kernel, r->rows, g->d, g->state, g->acc); break;
    case TR_REG_R_POWER: {
        int32_t top = 31 - __builtin_clz(g->support);
        TR_LAUNCH(tr_reg_power_kernel, r->rows, g->d, g->support, top, g->state, g->acc);
        break;
    }
    case TR_REG_R_BITPLANE: TR_LAUNCH(tr_reg_plane_kernel, words, g->d, g->sets, g->nsets, g->state, g->acc); break;
    }
    g->launches = 1;
    rc = tr_cuda_sync();
    if (!rc) rc = tr_cuda_copy_out(g->acc_host, g->acc, sizeof g->acc_host);
    g->copied += sizeof g->acc_host;
    if (rc) return refuse(ex, kind_of(rc), rc);
    if (ex) {
        ex->rows_scanned = g->acc_host[ACC_SELECTED]; ex->words_scanned = words;
        ex->state_elements = g->elements;
        ex->state_bytes = g->state_words * 8 + (repr == TR_REG_R_BITPLANE ? g->nsets * 4 : 0);
    }
    if (g->acc_host[ACC_INVALID]) return refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    if (g->acc_host[ACC_OVERFLOW]) return refuse(ex, TR_REG_REFUSE_OVERFLOW, TR_REG_E_OVERFLOW);
    return 0;
}
/* the device state copied to the host (POWER: S_0 is the valid count) */
static int32_t state_out(const tr_reg_request *r, int32_t repr, tr_reg_device *g, uint64_t *host, tr_reg_explain *ex) {
    int32_t rc = tr_cuda_copy_out(host, g->state, (size_t)g->state_words * 8);
    g->copied += g->state_words * 8;
    if (rc) return refuse(ex, kind_of(rc), rc);
    if (repr == TR_REG_R_POWER && g->state_words >= 2) { host[0] = g->acc_host[ACC_VALID]; host[1] = 0; }
    (void)r;
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Executes a representation on the device (finalize: TR_REG_FINAL_GPU for SCAN, either for HISTOGRAM,
 * TR_REG_FINAL_CPU for POWER and BITPLANE). Same contract as tr_reg_execute; the explanation names the device. */
int tr_mgpu_registry_execute(const tr_reg_request *r, int32_t repr, int32_t finalize, const tr_math_budget *budget,
                             tr_reg_result *res, tr_reg_explain *ex) {
    tr_reg_explain_init(ex, repr, tr_cuda_emulated() ? TR_REG_DEV_EMULATION : TR_REG_DEV_CUDA, finalize);
    tr_reg_explain_entry(ex, "tr_mgpu_registry_execute");
    if (!res || !r) return refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    tr_reg_result_init(res, r);
    tr_reg_device g;
    int32_t rc = device_build(r, repr, finalize, 0, 0, budget, &g, ex);
    if (!rc) {
        res->selected = g.acc_host[ACC_SELECTED]; res->valid = g.acc_host[ACC_VALID]; res->nulls = g.acc_host[ACC_NULLS];
        if (repr == TR_REG_R_HISTOGRAM && finalize == TR_REG_FINAL_GPU) {
            TR_LAUNCH(tr_reg_hist_contract_kernel, r->ncodes, g.d, g.state, g.acc);
            g.launches++;
            rc = tr_cuda_sync();
            if (!rc) rc = tr_cuda_copy_out(g.acc_host, g.acc, sizeof g.acc_host);
            g.copied += sizeof g.acc_host;
            if (rc) rc = refuse(ex, kind_of(rc), rc);
            else if (g.acc_host[ACC_OVERFLOW]) rc = refuse(ex, TR_REG_REFUSE_OVERFLOW, TR_REG_E_OVERFLOW);
        }
        if (!rc && (repr == TR_REG_R_SCAN || repr == TR_REG_R_HISTOGRAM) && finalize == TR_REG_FINAL_GPU) {
            for (int32_t j = 0; j < r->npolys; j++) {
                if (r->aggregate == TR_REG_SUM) {
                    tr_u128 v = ((tr_u128)g.acc_host[ACC_SUMS + 2 * j + 1] << 64) | g.acc_host[ACC_SUMS + 2 * j];
                    res->value[j] = tr_reg_i128_of(v);
                } else res->value[j] = tr_reg_i128_of_i64((int64_t)g.acc_host[ACC_EXT + j]);
            }
        } else if (!rc) {
            uint64_t *host = (uint64_t *)malloc((size_t)(g.state_words ? g.state_words : 1) * 8);
            if (!host) rc = refuse(ex, TR_REG_REFUSE_MEMORY, TR_REG_E_MEMORY);
            else {
                rc = state_out(r, repr, &g, host, ex);
                uint64_t work = 0;
                if (!rc && repr == TR_REG_R_HISTOGRAM) {
                    rc = tr_reg_hist_contract(r, host, res, &work);
                    if (rc) rc = refuse(ex, kind_of(rc), rc);
                } else if (!rc && repr == TR_REG_R_POWER) tr_reg_power_contract(r, g.support, host, res);
                else if (!rc) tr_reg_plane_contract(r, g.host_sets, host, g.nsets, res, &work);
                free(host);
            }
        }
    }
    if (ex) { ex->device_bytes = g.copied; ex->native_calls = (uint64_t)g.launches; }
    device_free(&g);
    return rc;
}
/* Builds a state (HISTOGRAM, POWER with support, BITPLANE with D; 0 = the request's) on the device and copies its
 * raw arrays (tr_reg_state_raw's layout) into raw (capacity bytes) with its info, for tr_reg_state_import. */
int tr_mgpu_registry_state(const tr_reg_request *r, int32_t repr, uint32_t support, int32_t D,
                           const tr_math_budget *budget, void *raw, uint64_t capacity, tr_reg_state_info *info,
                           tr_reg_explain *ex) {
    tr_reg_explain_init(ex, repr, tr_cuda_emulated() ? TR_REG_DEV_EMULATION : TR_REG_DEV_CUDA, TR_REG_FINAL_CPU);
    tr_reg_explain_entry(ex, "tr_mgpu_registry_state");
    if (!r || !info || repr == TR_REG_R_SCAN) return refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    tr_reg_device g;
    int32_t rc = device_build(r, repr, TR_REG_FINAL_CPU, support, D, budget, &g, ex);
    if (!rc && (capacity < g.state_words * 8 || (!raw && g.state_words))) rc = refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    if (!rc) rc = state_out(r, repr, &g, (uint64_t *)raw, ex);
    if (!rc) {
        memset(info, 0, sizeof *info);
        info->representation = repr; info->degree = g.D; info->width = r->width; info->has_validity = r->validity != NULL;
        info->support = g.support; info->has_selection = r->selected != NULL;
        info->elements = g.elements;
        info->bytes = g.state_words * 8 + (repr == TR_REG_R_BITPLANE ? g.nsets * 4 : 0);
        info->rows = r->rows; info->ncodes = r->ncodes; info->offset = r->offset;
        info->selected = g.acc_host[ACC_SELECTED]; info->valid = g.acc_host[ACC_VALID]; info->nulls = g.acc_host[ACC_NULLS];
        info->source = r->source; info->revision = r->revision;
    }
    if (ex) { ex->device_bytes = g.copied; ex->native_calls = (uint64_t)g.launches; }
    device_free(&g);
    return rc;
}
/* DISC-001 through the registry on the device: tr_mgpu_disc_verify with an explanation */
int tr_mgpu_registry_verify(const tr_disc_view *v, int64_t scale, int64_t bias, const tr_math_budget *budget,
                            tr_disc_evidence *e, tr_math_refusal *refusal, tr_reg_explain *ex) {
    tr_reg_explain_init(ex, TR_REG_R_VERIFY, tr_cuda_emulated() ? TR_REG_DEV_EMULATION : TR_REG_DEV_CUDA, TR_REG_FINAL_GPU);
    tr_reg_explain_entry(ex, "tr_mgpu_disc_verify");
    int rc = tr_mgpu_disc_verify(v, scale, bias, budget, e, refusal);
    if (ex && v) {
        ex->rows_scanned = v->rows; ex->words_scanned = (v->rows + 63) / 64; ex->native_calls = 1;
        ex->device_bytes = tr_disc_bytes(v);
        if (rc == TR_REG_E_BUDGET && refusal) ex->budget = *refusal;
    }
    return rc ? refuse(ex, kind_of(rc), rc) : 0;
}
/* A transform through the registry on the device (tr_reg_transform's contract): the composition check and the
 * module's implementation on CUDA (tr_reg_impls) before anything is read; LATTICE-002's or SPEC-002's kernels
 * (k launches, one per butterfly pass; the copies: the values in, the overflow flag and the values out). */
int tr_mgpu_registry_transform(int32_t op, int32_t variant, int32_t carrier, int32_t input, int32_t k,
                               const tr_math_i128 *in, tr_math_i128 *out, const tr_math_budget *budget,
                               tr_math_refusal *refusal, tr_reg_explain *ex) {
    tr_reg_explain_begin(ex, 0, 0, tr_cuda_emulated() ? TR_REG_DEV_EMULATION : TR_REG_DEV_CUDA, TR_REG_FINAL_GPU);
    tr_budget_none(refusal);
    int32_t module = 0;
    int32_t rc = tr_reg_transform_check(op, variant, carrier, input, k, in, out, TR_REG_ON_CUDA, &module, ex);
    if (rc) return rc;
    const int lattice = module == TR_REG_M_LATTICE_TRANSFORM;
    const int32_t fop = !lattice ? 0 : op == TR_OP_ZETA ? (variant == TR_REG_UP ? TR_LATTICE_ZETA_UP : TR_LATTICE_ZETA_DOWN)
                                                       : (variant == TR_REG_UP ? TR_LATTICE_MOBIUS_UP : TR_LATTICE_MOBIUS_DOWN);
    const int32_t fcarrier = carrier == TR_REG_C_INTEGER ? TR_LATTICE_I64 : TR_LATTICE_I128;
    tr_reg_explain_entry(ex, lattice ? "tr_mgpu_lattice_transform" : "tr_mgpu_spec_walsh");
    tr_math_need need;
    if (lattice ? tr_lattice_transform_base_need(k, fop, fcarrier, &need) : tr_spec_base_need(k, &need))
        return refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_refusal br;
    if ((rc = tr_budget_check(budget, &need, &br))) {
        if (refusal) *refusal = br;
        if (ex) ex->budget = br;
        return refuse(ex, TR_REG_REFUSE_BUDGET, rc);
    }
    if (k > TR_REG_TRANSFORM_MAX_K) return refuse(ex, TR_REG_REFUSE_UNSUPPORTED, TR_REG_E_UNSUPPORTED);
    if ((rc = tr_reg_transform_values(op, carrier, input, k, in, ex))) return rc;
    rc = lattice ? tr_mgpu_lattice_transform(k, fop, fcarrier, in, out, budget, refusal)
                 : tr_mgpu_spec_walsh(k, in, out, budget, NULL, refusal);
    if (ex) {
        const uint64_t n = (uint64_t)1 << k;
        ex->state_elements = n; ex->state_bytes = n * 16;
        /* the values in and the overflow flag out; on success the values out too */
        ex->device_bytes = rc == 0 ? 2 * n * 16 + 8 : rc == TR_REG_E_OVERFLOW ? n * 16 + 8 : 0;
        ex->native_calls = (uint64_t)k + (lattice && fcarrier == TR_LATTICE_I64 ? 1 : 0);
        if (rc == TR_REG_E_BUDGET && refusal) ex->budget = *refusal;
    }
    return rc ? refuse(ex, kind_of(rc), rc) : 0;
}
#ifdef __cplusplus
}
#endif
