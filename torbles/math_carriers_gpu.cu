/* FND-009 on a CUDA device: the data-parallel carrier kernels (GF(p) arithmetic over arrays, dot products and
 * matrix-vector products mod p, int64 reduction mod p, tight enclosures of int64 rationals, exact 128-bit sums and
 * exact 192-bit dot products). Contracts: math_carriers.h ("CUDA entry points"); every result equals the CPU
 * function's bit for bit (exact residues, exact integers, and enclosures computed with integer arithmetic only).
 *
 * Device arithmetic: 64x64 -> 128-bit products (unsigned __int128 multiplication, mul.lo / mul.hi) and Barrett
 * reduction (tr_gfp_barrett_hd), never a 128-bit division. Order independence: one output slot per element, or
 * per-thread partial sums (residues < 2^62, or exact 128-bit / 192-bit integers) added with exact multiword atomics.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_carriers.h"

/* exact signed 128-bit atomic accumulation of a 128-bit value: lo += x (mod 2^64), hi += high word + carry */
TR_DEVICE void tr_carriers_add128(int64_t *lo, int64_t *hi, __int128 x) {
    uint64_t xl = (uint64_t)x, xh = (uint64_t)((unsigned __int128)x >> 64);
    uint64_t old = tr_atomic_add_u64((uint64_t *)lo, xl);
    uint64_t carry = (old + xl) < old;
    tr_atomic_add_u64((uint64_t *)hi, xh + carry);
}
/* exact 192-bit two's complement atomic accumulation: each word's carry is that of its own atomic addition, so the
 * final words are the sum mod 2^192 whatever the interleaving */
TR_DEVICE void tr_carriers_add192(uint64_t *w, const uint64_t *x) {
    uint64_t old0 = tr_atomic_add_u64(&w[0], x[0]);
    uint64_t c0 = (old0 + x[0]) < old0;
    uint64_t y1 = x[1] + c0, c1a = y1 < c0;
    uint64_t old1 = tr_atomic_add_u64(&w[1], y1);
    uint64_t c1 = (old1 + y1) < old1;
    tr_atomic_add_u64(&w[2], x[2] + c1 + c1a);
}

/* flags: bit 0 an invalid input (a residue >= p, a zero denominator, a group id >= ngroups), bit 1 a zero divisor */
TR_KERNEL void tr_carriers_gfp_vec_kernel(tr_gfp f, int32_t op, const uint64_t *a, const uint64_t *b, uint64_t n,
                                          uint64_t *out, uint64_t *flags) {
    TR_GRID_LOOP(i, n) {
        uint64_t x = a[i], y = b ? b[i] : 0, v;
        if (x >= f.p || y >= f.p) { tr_atomic_or_u64(flags, 1); continue; }
        switch (op) {
        case TR_OP_ADD: v = tr_gfp_add_hd(&f, x, y); break;
        case TR_OP_SUB: v = tr_gfp_sub_hd(&f, x, y); break;
        case TR_OP_MUL: v = tr_gfp_mul_hd(&f, x, y); break;
        case TR_OP_DIV:
            if (!y) { tr_atomic_or_u64(flags, 2); continue; }
            v = tr_gfp_mul_hd(&f, x, tr_gfp_inv_hd(&f, y));
            break;
        case TR_OP_NEG: v = tr_gfp_neg_hd(&f, x); break;
        default:
            if (!x) { tr_atomic_or_u64(flags, 2); continue; }
            v = tr_gfp_inv_hd(&f, x);
            break;
        }
        out[i] = v;
    }
}

TR_KERNEL void tr_carriers_gfp_dot_kernel(tr_gfp f, const uint64_t *a, const uint64_t *b, uint64_t n, int64_t *acc,
                                          uint64_t *flags) {
    uint64_t s = 0;
    int bad = 0;
    TR_GRID_LOOP(i, n) {
        uint64_t x = a[i], y = b[i];
        if (x >= f.p || y >= f.p) { bad = 1; continue; }
        s = tr_gfp_add_hd(&f, s, tr_gfp_mul_hd(&f, x, y));
    }
    if (bad) tr_atomic_or_u64(flags, 1);
    if (s) tr_atomic_add_i128(&acc[0], &acc[1], (int64_t)s);   /* s < 2^62: at most 2^20 threads, no overflow */
}

/* work item w = (row, column chunk): a partial sum < p, added to the row's 128-bit accumulator */
TR_KERNEL void tr_carriers_gfp_matvec_kernel(tr_gfp f, const uint64_t *A, uint64_t rows, uint64_t cols, uint64_t chunks,
                                             const uint64_t *x, int64_t *acc, uint64_t *flags) {
    TR_GRID_LOOP(w, rows * chunks) {
        uint64_t r = w / chunks, c0 = (w % chunks) * TR_CARRIERS_CHUNK;
        uint64_t c1 = c0 + TR_CARRIERS_CHUNK < cols ? c0 + TR_CARRIERS_CHUNK : cols, s = 0;
        const uint64_t *row = A + r * cols;
        int bad = 0;
        for (uint64_t j = c0; j < c1; j++) {
            uint64_t a = row[j], v = x[j];
            if (a >= f.p || v >= f.p) { bad = 1; break; }
            s = tr_gfp_add_hd(&f, s, tr_gfp_mul_hd(&f, a, v));
        }
        if (bad) { tr_atomic_or_u64(flags, 1); continue; }
        if (s) tr_atomic_add_i128(&acc[2 * r], &acc[2 * r + 1], (int64_t)s);
    }
}
TR_KERNEL void tr_carriers_gfp_rows_kernel(tr_gfp f, const int64_t *acc, uint64_t rows, uint64_t *y) {
    TR_GRID_LOOP(r, rows) y[r] = tr_gfp_reduce_u128_hd(&f, (uint64_t)acc[2 * r + 1], (uint64_t)acc[2 * r]);
}

TR_KERNEL void tr_carriers_gfp_reduce_kernel(tr_gfp f, const int64_t *x, uint64_t n, uint64_t *out) {
    TR_GRID_LOOP(i, n) out[i] = tr_gfp_reduce_i64_hd(&f, x[i]);
}

TR_KERNEL void tr_carriers_enclose_kernel(const tr_num_rational *q, uint64_t n, tr_math_interval *out, uint64_t *flags) {
    TR_GRID_LOOP(i, n) {
        if (tr_interval_enclose_q64(q[i].numerator, q[i].denominator, &out[i])) tr_atomic_or_u64(flags, 1);
    }
}

TR_KERNEL void tr_carriers_sum_kernel(const int64_t *x, uint64_t n, const uint32_t *groups, uint64_t ngroups,
                                      int64_t *acc, uint64_t *flags) {
    if (!groups) {                                   /* one group: a local exact sum per thread, then one atomic */
        __int128 s = 0;
        TR_GRID_LOOP(i, n) s += x[i];
        if (s) tr_carriers_add128(&acc[0], &acc[1], s);
        return;
    }
    TR_GRID_LOOP(i, n) {
        uint32_t g = groups[i];
        if ((uint64_t)g >= ngroups) { tr_atomic_or_u64(flags, 1); continue; }
        tr_atomic_add_i128(&acc[2 * (uint64_t)g], &acc[2 * (uint64_t)g + 1], x[i]);
    }
}

TR_KERNEL void tr_carriers_dot_wide_kernel(const int64_t *a, const int64_t *b, uint64_t n, uint64_t *w) {
    uint64_t s[3] = {0, 0, 0};
    TR_GRID_LOOP(i, n) tr_wide_add128(s, (__int128)a[i] * b[i]);
    if (s[0] | s[1] | s[2]) tr_carriers_add192(w, s);
}

#ifdef __cplusplus
extern "C" {
#endif
static const tr_math_budget *tr_carriers_budget(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}

int tr_mgpu_carriers_gfp_vec(const tr_gfp *f, int32_t op, const uint64_t *a, const uint64_t *b, uint64_t n,
                             uint64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int unary = op == TR_OP_NEG || op == TR_OP_INV;
    if (!tr_gfp_consistent(f) || (!unary && op != TR_OP_ADD && op != TR_OP_SUB && op != TR_OP_MUL && op != TR_OP_DIV) ||
        (n && (!a || !out || (!unary && !b)))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_GFP_VEC, n, 0, unary),
                                 refusal);
    if (rc || !n) return rc;
    uint64_t *da = NULL, *db = NULL, *dout = NULL, *dflags = NULL, flags = 0;
    size_t bytes = (size_t)n * sizeof(uint64_t);
    rc = tr_cuda_alloc((void **)&da, bytes);
    if (!rc && !unary) rc = tr_cuda_alloc((void **)&db, bytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, bytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_in(da, a, bytes);
    if (!rc && !unary) rc = tr_cuda_copy_in(db, b, bytes);
    if (!rc) { TR_LAUNCH(tr_carriers_gfp_vec_kernel, n, *f, op, da, db, n, dout, dflags); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, sizeof flags);
    if (!rc && !flags) rc = tr_cuda_copy_out(out, dout, bytes);
    tr_cuda_free(da); tr_cuda_free(db); tr_cuda_free(dout); tr_cuda_free(dflags);
    if (rc) return rc;
    if (flags & 1) return TR_MATH_INVALID;
    return (flags & 2) ? (int)TR_MATH_NO_SOLUTION : (int)TR_MATH_OK;
}

int tr_mgpu_carriers_gfp_dot(const tr_gfp *f, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_gfp_consistent(f) || !out || (n && (!a || !b))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_GFP_DOT, n, 0, 0), refusal);
    if (rc) return rc;
    if (!n) { *out = 0; return TR_MATH_OK; }
    uint64_t *da = NULL, *db = NULL, *dflags = NULL, flags = 0;
    int64_t *dacc = NULL, acc[2] = {0, 0};
    size_t bytes = (size_t)n * sizeof(uint64_t);
    rc = tr_cuda_alloc((void **)&da, bytes);
    if (!rc) rc = tr_cuda_alloc((void **)&db, bytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_in(da, a, bytes);
    if (!rc) rc = tr_cuda_copy_in(db, b, bytes);
    if (!rc) { TR_LAUNCH(tr_carriers_gfp_dot_kernel, n, *f, da, db, n, dacc, dflags); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    tr_cuda_free(da); tr_cuda_free(db); tr_cuda_free(dacc); tr_cuda_free(dflags);
    if (rc) return rc;
    if (flags) return TR_MATH_INVALID;
    *out = (uint64_t)((((unsigned __int128)(uint64_t)acc[1] << 64) | (uint64_t)acc[0]) % f->p);
    return TR_MATH_OK;
}

int tr_mgpu_carriers_gfp_matvec(const tr_gfp *f, const uint64_t *A, uint64_t rows, uint64_t cols, const uint64_t *x,
                                uint64_t *y, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    uint64_t cells;
    if (!tr_gfp_consistent(f) || (rows && !y) || (rows && cols && (!A || !x)) || __builtin_mul_overflow(rows, cols, &cells))
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = cells;
    need.need[TR_BUDGET_BYTES] = tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_GFP_MATVEC, rows, cols, 0);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !rows) return rc;
    uint64_t chunks = cols ? (cols + TR_CARRIERS_CHUNK - 1) / TR_CARRIERS_CHUNK : 1;
    uint64_t *dA = NULL, *dx = NULL, *dy = NULL, *dflags = NULL, flags = 0;
    int64_t *dacc = NULL;
    rc = tr_cuda_alloc((void **)&dA, (size_t)cells * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dx, (size_t)cols * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dy, (size_t)rows * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, (size_t)rows * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc && cells) rc = tr_cuda_copy_in(dA, A, (size_t)cells * 8);
    if (!rc && cols) rc = tr_cuda_copy_in(dx, x, (size_t)cols * 8);
    if (!rc && cols) {
        TR_LAUNCH(tr_carriers_gfp_matvec_kernel, rows * chunks, *f, dA, rows, cols, chunks, dx, dacc, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) { TR_LAUNCH(tr_carriers_gfp_rows_kernel, rows, *f, dacc, rows, dy); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, sizeof flags);
    if (!rc && !flags) rc = tr_cuda_copy_out(y, dy, (size_t)rows * 8);
    tr_cuda_free(dA); tr_cuda_free(dx); tr_cuda_free(dy); tr_cuda_free(dacc); tr_cuda_free(dflags);
    if (rc) return rc;
    return flags ? TR_MATH_INVALID : TR_MATH_OK;
}

int tr_mgpu_carriers_gfp_reduce_i64(const tr_gfp *f, const int64_t *x, uint64_t n, uint64_t *out,
                                    const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_gfp_consistent(f) || (n && (!x || !out))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_GFP_REDUCE, n, 0, 0),
                                 refusal);
    if (rc || !n) return rc;
    int64_t *dx = NULL;
    uint64_t *dout = NULL;
    size_t bytes = (size_t)n * 8;
    rc = tr_cuda_alloc((void **)&dx, bytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, bytes);
    if (!rc) rc = tr_cuda_copy_in(dx, x, bytes);
    if (!rc) { TR_LAUNCH(tr_carriers_gfp_reduce_kernel, n, *f, dx, n, dout); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(out, dout, bytes);
    tr_cuda_free(dx); tr_cuda_free(dout);
    return rc;
}

int tr_mgpu_carriers_enclose_q64(const tr_num_rational *q, uint64_t n, tr_math_interval *out,
                                 const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (n && (!q || !out)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_ENCLOSE_Q64, n, 0, 0),
                                 refusal);
    if (rc || !n) return rc;
    tr_num_rational *dq = NULL;
    tr_math_interval *dout = NULL;
    uint64_t *dflags = NULL, flags = 0;
    rc = tr_cuda_alloc((void **)&dq, (size_t)n * sizeof *dq);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)n * sizeof *dout);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_in(dq, q, (size_t)n * sizeof *dq);
    if (!rc) { TR_LAUNCH(tr_carriers_enclose_kernel, n, dq, n, dout, dflags); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, sizeof flags);
    if (!rc && !flags) rc = tr_cuda_copy_out(out, dout, (size_t)n * sizeof *dout);
    tr_cuda_free(dq); tr_cuda_free(dout); tr_cuda_free(dflags);
    if (rc) return rc;
    return flags ? TR_MATH_INVALID : TR_MATH_OK;
}

int tr_mgpu_carriers_sum_i64(const int64_t *x, uint64_t n, const uint32_t *groups, uint64_t ngroups, tr_math_i128 *out,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || !ngroups || (n && !x) || (!groups && ngroups != 1)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = ngroups;
    need.need[TR_BUDGET_BYTES] = tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_SUM_I64, n, ngroups, groups != NULL);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    int64_t *dx = NULL, *dacc = NULL;
    uint32_t *dg = NULL;
    uint64_t *dflags = NULL, flags = 0;
    int64_t *acc = (int64_t *)malloc((size_t)ngroups * 16);
    if (!acc) return TR_MATH_MEMORY;
    rc = tr_cuda_alloc((void **)&dacc, (size_t)ngroups * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc && n) rc = tr_cuda_alloc((void **)&dx, (size_t)n * 8);
    if (!rc && n && groups) rc = tr_cuda_alloc((void **)&dg, (size_t)n * 4);
    if (!rc && n) rc = tr_cuda_copy_in(dx, x, (size_t)n * 8);
    if (!rc && n && groups) rc = tr_cuda_copy_in(dg, groups, (size_t)n * 4);
    if (!rc && n) { TR_LAUNCH(tr_carriers_sum_kernel, n, dx, n, dg, ngroups, dacc, dflags); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)ngroups * 16);
    tr_cuda_free(dx); tr_cuda_free(dg); tr_cuda_free(dacc); tr_cuda_free(dflags);
    if (!rc && flags) rc = TR_MATH_INVALID;
    if (!rc) for (uint64_t g = 0; g < ngroups; g++) { out[g].lo = acc[2 * g]; out[g].hi = acc[2 * g + 1]; }
    free(acc);
    return rc;
}

int tr_mgpu_carriers_dot_wide(const int64_t *a, const int64_t *b, uint64_t n, uint64_t out[3],
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || (n && (!a || !b))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_carriers_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_carrier_kernel_bytes(TR_CARRIER_KERNEL_DOT_WIDE, n, 0, 0), refusal);
    if (rc) return rc;
    uint64_t w[3] = {0, 0, 0}, *dw = NULL;
    int64_t *da = NULL, *db = NULL;
    rc = tr_cuda_alloc((void **)&dw, sizeof w);
    if (!rc && n) rc = tr_cuda_alloc((void **)&da, (size_t)n * 8);
    if (!rc && n) rc = tr_cuda_alloc((void **)&db, (size_t)n * 8);
    if (!rc && n) rc = tr_cuda_copy_in(da, a, (size_t)n * 8);
    if (!rc && n) rc = tr_cuda_copy_in(db, b, (size_t)n * 8);
    if (!rc && n) { TR_LAUNCH(tr_carriers_dot_wide_kernel, n, da, db, n, dw); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(w, dw, sizeof w);
    tr_cuda_free(dw); tr_cuda_free(da); tr_cuda_free(db);
    if (rc) return rc;
    out[0] = w[0]; out[1] = w[1]; out[2] = w[2];
    return TR_MATH_OK;
}
#ifdef __cplusplus
}
#endif
