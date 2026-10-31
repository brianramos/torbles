/* KERNEL-001: exact finite-feature kernel embeddings. Kernels whose feature map is explicitly finite reduce kernel
 * means, maximum mean discrepancies and kernel sums over selected pairs to retained feature states, the moment and
 * count tensors of TENSOR-001 (math_tensor.h), exactly, on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernels (tr_kernel_spec; values are the scope's logical integers, decimals as scaled integers of one common scale):
 *   POLY      k(x, y) = (x.y + c)^d over n feature columns, d = 0..TR_KERNEL_MAX_DEGREE, |c|^d <= 2^127 - 1. With
 *             x~ = (1, x), its exact finite feature expansion is k(x, y) = sum_J c^{z(J)} prod_k x~_{j_k} y~_{j_k} over
 *             the (n + 1)^d multi-indices J (z(J): the entries of J that are the constant 0), evaluated over the
 *             C(n + d, d) sorted ones with their multiplicities: linear (d = 1, c = 0), homogeneous (c = 0) and
 *             inhomogeneous polynomial kernels.
 *   MONOMIAL  k(x, y) = sum_t w_t x^{a_t} y^{a_t}: 1..TR_KERNEL_MAX_TERMS declared monomials of total degree <= 8 with
 *             int64 weights (equal monomials are merged).
 *   DELTA     k(x, y) = prod_k [x_k = y_k] over n categorical columns, NULL a category of its own (it matches NULL):
 *             the one-hot feature map of the joint category.
 * Any other kind (TR_KERNEL_RBF, _LAPLACE, _OTHER: no finite exact feature map) is refused with TR_MATH_UNSUPPORTED,
 * never approximated (KERNEL-002 provides declared approximations).
 *
 * The feature state (tr_kernel_build) of a scope, n columns, an optional weight w and a declared kernel:
 *   POLY / MONOMIAL: the TENSOR-001 moment tensor of order D (the declared degree, or a larger requested one, and at
 *   least 1 so that its axes read the kernel's columns) whose
 *   every axis lists (1, x_1, ..., x_n), LISTWISE: T_g[J] = sum over the complete rows of group g (every kernel column
 *   and the weight present) of w prod_k x~_{j_k}. Every lower order is its slice with leading constant indices, so
 *   S_a(g) = sum_i w_i x_i^a = T_g[0, .., 0, positions of a] and sum_{i in g, l in h} w_i w_l k(x_i, x_l)
 *   = sum_J c^{z(J)} T_g[0^{D-d}, J] T_h[0^{D-d}, J] = <Phi(X_g), Phi(X_h)> exactly.
 *   DELTA: the TENSOR-001 count tensor of the n columns (weighted: the sums of present weights); <C_g, C_h>.
 *   Per group, besides the tensor's rows and used: W = sum w (int128), Q = sum w^2 and Dg = sum w^2 k(x_i, x_i) of
 *   the declared kernel (exact 1024-bit), over the contributing rows: the self-pair (diagonal) terms.
 * Any kernel the state retains can be evaluated on it (tr_kernel_eval_make): same family (moments or counts), degree
 * <= D, columns a subset of the state's (POLY / MONOMIAL; DELTA: exactly its columns). The self-pair term is retained
 * for the declared kernel only: a request that excludes self-pairs with another kernel is refused (TR_MATH_UNSUPPORTED
 * in its status). An absent degree, dimension or family is refused, never approximated (TR_MATH_UNSUPPORTED).
 *
 * Requests (tr_kernel_request; groups a, b of the scope; weights w_i of group a's rows, v_l of group b's):
 *   SUM       sum over the selected pairs of w_i v_l k(x_i, y_l): a != b every (i, l); a == b ordered pairs (i, l)
 *             with i != l unless TR_KERNEL_SELF (then all), or with TR_KERNEL_UNORDERED the pairs {i, l} once
 *             ((K - Dg) / 2, with self-pairs (K + Dg) / 2). value = num (den = 1), pairs = the weighted pair count
 *             (W_a W_b; within a group W^2 - Q, or W^2 with self-pairs; unordered halves).
 *   MEAN      SUM / pairs (empty when pairs = 0): the kernel mean embedding inner product <mu_a, mu_b>.
 *   MMD2      biased: K_aa / W_a^2 + K_bb / W_b^2 - 2 K_ab / (W_a W_b); TR_KERNEL_UNBIASED: (K_aa - D_a) / (W_a^2 - Q_a)
 *             + (K_bb - D_b) / (W_b^2 - Q_b) - 2 K_ab / (W_a W_b) (empty when a denominator is 0).
 *   WITNESS   mu_a(z) = sum_i w_i k(x_i, z) / W_a at points[point] (logical values of the kernel columns; DELTA: a
 *             category or NULL per column, outside the domain matching no row).
 * Every answer is an exact rational num / den (den > 0, not reduced) in 1024-bit two's complement words; an answer
 * outside that carrier is refused in its status (TR_MATH_OVERFLOW), never wrapped. Each request has its own status
 * (0, TR_MATH_INVALID, _UNSUPPORTED, _OVERFLOW), so a batch never fails as a whole at the native level.
 *
 * Exactness. The tensor's entries are proven by TENSOR-001; the build also proves before reading the rows that the
 * self-pair sums fit (bit lengths: rows + 2 x weight + the kernel's value at the extremes <= 1000), else
 * TR_MATH_OVERFLOW. Inner products accumulate T_a T_b products (256 bits) per coefficient class in 384-bit words and
 * combine the classes in 1024 bits; MMD and means combine with overflow-checked 1024-bit products.
 *
 * Paths giving the same answers bit for bit: the CPU loop (tr_kernel_query) and the CUDA kernel (tr_mgpu_kernel_query:
 * one work item per request, the shared evaluator tr_kernel_eval_request below, on a state uploaded once by
 * tr_mgpu_kernel_upload); the feature state built on the CPU or on the device (tr_mgpu_kernel_build: the TENSOR-001
 * device build, then one work item per row slice for W, Q, Dg with exact multiword atomics; tr_kernel_from_states
 * adopts after consistency checks).
 *
 * Budgets (math_budgets.h), before allocation: the build (the tensor's needs, plus 288 bytes a group), an evaluation
 * plan (support = its items, bytes = 64 per item), a batch (output = requests; CUDA: bytes = the device copies).
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR (columns of different decimal
 * scales), _DEVICE. Objects are immutable; constructors return NULL and set *error. */
#ifndef TORBLES_MATH_KERNEL_H
#define TORBLES_MATH_KERNEL_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_KERNEL_ABI 1
#define TR_KERNEL_WORDS 16                /* 1024-bit two's complement answers */
#define TR_KERNEL_ACC_WORDS 6             /* 384-bit class accumulators of T_a T_b products */
#define TR_KERNEL_MAX_DEGREE 8
#define TR_KERNEL_MAX_COLUMNS 32
#define TR_KERNEL_MAX_TERMS 64
#define TR_KERNEL_MAX_CLASSES (TR_KERNEL_MAX_DEGREE + 1)
#define TR_KERNEL_GROUP_BYTES 288         /* W (16), Q (128), Dg (128) and their rows (16) per group */
#define TR_KERNEL_ITEM_BYTES 64           /* an evaluation item: index, class, coordinates */
#define TR_KERNEL_BITS_LIMIT 1000         /* the self-pair proof's bit budget */
enum { TR_KERNEL_POLY = 1, TR_KERNEL_MONOMIAL = 2, TR_KERNEL_DELTA = 3,
       TR_KERNEL_RBF = 16, TR_KERNEL_LAPLACE = 17, TR_KERNEL_OTHER = 18 };   /* the last three: refused */
enum { TR_KERNEL_OP_SUM = 1, TR_KERNEL_OP_MEAN = 2, TR_KERNEL_OP_MMD2 = 3, TR_KERNEL_OP_WITNESS = 4 };
enum { TR_KERNEL_SELF = 1, TR_KERNEL_UNORDERED = 2, TR_KERNEL_UNBIASED = 4 };

/* ------------------------------------------------------------------ ABI structs */
/* a kernel (2728 bytes); cols are scope columns */
typedef struct {
    int32_t kind, degree, ncols, nterms;     /* degree: POLY d; MONOMIAL max |a_t| (computed); DELTA 0 */
    int64_t c;                               /* POLY */
    int32_t cols[TR_KERNEL_MAX_COLUMNS];
    int64_t weight[TR_KERNEL_MAX_TERMS];     /* MONOMIAL */
    uint8_t exps[TR_KERNEL_MAX_TERMS][TR_KERNEL_MAX_COLUMNS];   /* MONOMIAL: exponents of term t over cols */
    int32_t reserved[4];
} tr_kernel_spec;

typedef struct { uint64_t w[TR_KERNEL_WORDS]; } tr_kernel_wide;   /* 128 bytes, w[0] least significant */

/* a request (40 bytes) and its answer (392 bytes) */
typedef struct {
    int32_t op, flags;
    uint64_t a, b;                           /* groups */
    uint64_t point;                          /* WITNESS: the point's index */
    int64_t reserved;
} tr_kernel_request;
typedef struct {
    int32_t status, empty;
    tr_kernel_wide num, den, pairs;          /* the answer num / den; SUM and MEAN: the pair count */
} tr_kernel_result;

/* The evaluation view (408 bytes): a state and a retained kernel's items, host pointers (tr_kernel_eval_get_view) or
 * device copies. Items: the multi-indices the kernel reads, sorted, with their coefficient class and multiplicity
 * (POLY: one item per sorted multi-index J standing for its d! / prod_c r_c! orderings, r_c the repeats of each
 * position, since the moment tensor is symmetric: C(n + d, d) items instead of (n + 1)^d; MONOMIAL: one item per term,
 * multiplicity 1); DELTA: none (every cell, coefficient 1). */
typedef struct {
    int32_t kind, repr, order, nclass;       /* kernel kind; state representation; tensor order D; classes */
    int32_t has_self, npos, reserved[2];     /* has_self: the self-pair term belongs to this kernel; npos: point columns */
    uint64_t ngroups, cells, nitems, npoints;
    uint64_t dims[TR_TENSOR_MAX_ORDER], stride[TR_TENSOR_MAX_ORDER];
    const uint64_t *item_index;              /* [nitems] linear indices within a group */
    const int32_t *item_class;               /* [nitems] */
    const uint32_t *item_mult;               /* [nitems] the orderings an item stands for (1 .. 8!) */
    const tr_math_i128 *class_coef;          /* [nclass] (MONOMIAL: one class per item) */
    const int32_t *pos_of_axis;              /* [n + 1] point column of a state position (-1: none; 0 is the constant) */
    const tr_math_i128 *value;               /* dense [G x cells] / sparse [nnz] */
    const uint64_t *offset;                  /* sparse [G + 1] */
    const uint32_t *coord;                   /* sparse [nnz x order] */
    const tr_math_i128 *W;                   /* [G] */
    const tr_kernel_wide *Q, *Dg;            /* [G] */
    int64_t lo[TR_TENSOR_MAX_ORDER];         /* DELTA: the value at index 0 of each axis */
    uint64_t span[TR_TENSOR_MAX_ORDER];      /* DELTA: values of each axis (the NULL slot, if any, is index span) */
} tr_kernel_view;

typedef struct {
    uint64_t ngroups, present, rows, used, bytes, cells, nnz;
    int32_t kind, degree, ncols, order, repr, weighted, reserved[2];
    uint64_t digest[2];
} tr_kernel_info;   /* 104 bytes */

typedef struct tr_kernel_state_s tr_kernel_state;
typedef struct tr_kernel_eval_s tr_kernel_eval;

/* ------------------------------------------------------------------ 1024-bit two's complement (CPU and kernels) */
TR_NUM_HD void tr_kernel_w_zero(tr_kernel_wide *r) { for (int i = 0; i < TR_KERNEL_WORDS; i++) r->w[i] = 0; }
TR_NUM_HD int tr_kernel_w_neg(const tr_kernel_wide *a) { return (int)(a->w[TR_KERNEL_WORDS - 1] >> 63); }
TR_NUM_HD int tr_kernel_w_is_zero(const tr_kernel_wide *a) {
    uint64_t o = 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) o |= a->w[i];
    return o == 0;
}
TR_NUM_HD void tr_kernel_w_from_i128(__int128 v, tr_kernel_wide *r) {
    unsigned __int128 u = (unsigned __int128)v;
    r->w[0] = (uint64_t)u; r->w[1] = (uint64_t)(u >> 64);
    for (int i = 2; i < TR_KERNEL_WORDS; i++) r->w[i] = v < 0 ? UINT64_MAX : 0;
}
/* the two's complement negation of the bit pattern (exact for every value but -2^1023) */
TR_NUM_HD void tr_kernel_w_negate(tr_kernel_wide *a) {
    uint64_t c = 1;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) { uint64_t x = ~a->w[i] + c; c = c && x == 0; a->w[i] = x; }
}
/* *r = a + b: 1, or 0 on signed overflow (*r unspecified) */
TR_NUM_HD int tr_kernel_w_add(const tr_kernel_wide *a, const tr_kernel_wide *b, tr_kernel_wide *r) {
    int sa = tr_kernel_w_neg(a), sb = tr_kernel_w_neg(b);
    uint64_t c = 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) {
        uint64_t x = a->w[i] + c, c1 = x < c, y = x + b->w[i], c2 = y < x;
        r->w[i] = y; c = c1 | c2;
    }
    return !(sa == sb && tr_kernel_w_neg(r) != sa);
}
TR_NUM_HD int tr_kernel_w_sub(const tr_kernel_wide *a, const tr_kernel_wide *b, tr_kernel_wide *r) {
    tr_kernel_wide nb = *b;
    if (tr_kernel_w_neg(&nb) && !tr_kernel_w_is_zero(&nb)) {          /* -2^1023 has no negation */
        int top = nb.w[TR_KERNEL_WORDS - 1] == (UINT64_C(1) << 63);
        for (int i = 0; top && i < TR_KERNEL_WORDS - 1; i++) top = nb.w[i] == 0;
        if (top) return 0;
    }
    tr_kernel_w_negate(&nb);
    return tr_kernel_w_add(a, &nb, r);
}
/* *r = a * b: 1, or 0 when |a b| >= 2^1023 */
TR_NUM_HD int tr_kernel_w_mul(const tr_kernel_wide *a, const tr_kernel_wide *b, tr_kernel_wide *r) {
    tr_kernel_wide x = *a, y = *b;
    int na = tr_kernel_w_neg(&x), nb = tr_kernel_w_neg(&y);
    if (na) tr_kernel_w_negate(&x);
    if (nb) tr_kernel_w_negate(&y);
    if (tr_kernel_w_neg(&x) || tr_kernel_w_neg(&y)) return 0;         /* -2^1023 */
    int lx = TR_KERNEL_WORDS, ly = TR_KERNEL_WORDS;
    while (lx > 0 && !x.w[lx - 1]) lx--;
    while (ly > 0 && !y.w[ly - 1]) ly--;
    tr_kernel_wide p;
    tr_kernel_w_zero(&p);
    if (lx + ly > TR_KERNEL_WORDS + 1) return 0;
    for (int i = 0; i < lx; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < ly; j++) {
            if (i + j >= TR_KERNEL_WORDS) { if (x.w[i] && y.w[j]) return 0; continue; }
            unsigned __int128 t = (unsigned __int128)x.w[i] * y.w[j] + p.w[i + j] + carry;
            p.w[i + j] = (uint64_t)t; carry = (uint64_t)(t >> 64);
        }
        if (carry) { if (i + ly >= TR_KERNEL_WORDS) return 0; p.w[i + ly] += carry; }
    }
    if (tr_kernel_w_neg(&p)) return 0;                                /* |a b| >= 2^1023 */
    if (na != nb) tr_kernel_w_negate(&p);
    *r = p;
    return 1;
}
TR_NUM_HD int tr_kernel_w_mul_i128(const tr_kernel_wide *a, __int128 b, tr_kernel_wide *r) {
    tr_kernel_wide y;
    tr_kernel_w_from_i128(b, &y);
    return tr_kernel_w_mul(a, &y, r);
}
/* a / 2 of an even value (an arithmetic shift) */
TR_NUM_HD void tr_kernel_w_half(tr_kernel_wide *a) {
    int s = tr_kernel_w_neg(a);
    for (int i = 0; i < TR_KERNEL_WORDS; i++) {
        uint64_t hi = i + 1 < TR_KERNEL_WORDS ? a->w[i + 1] : (s ? UINT64_MAX : 0);
        a->w[i] = (a->w[i] >> 1) | (hi << 63);
    }
}
/* 384-bit accumulators of signed 256-bit products times a multiplicity m < 2^32 (at most 2^64 additions below 2^286:
 * never overflow) */
TR_NUM_HD void tr_kernel_acc_add_prod(uint64_t *acc, __int128 a, __int128 b, uint32_t mult) {
    unsigned __int128 x = tr_tensor_mag(a), y = tr_tensor_mag(b);
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64), y0 = (uint64_t)y, y1 = (uint64_t)(y >> 64);
    unsigned __int128 p00 = (unsigned __int128)x0 * y0, p01 = (unsigned __int128)x0 * y1;
    unsigned __int128 p10 = (unsigned __int128)x1 * y0, p11 = (unsigned __int128)x1 * y1;
    unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    unsigned __int128 high = (p01 >> 64) + (p10 >> 64) + (uint64_t)p11 + (mid >> 64);
    uint64_t m[TR_KERNEL_ACC_WORDS];
    m[0] = (uint64_t)p00; m[1] = (uint64_t)mid; m[2] = (uint64_t)high;
    m[3] = (uint64_t)(p11 >> 64) + (uint64_t)(high >> 64); m[4] = 0; m[5] = 0;
    if (mult != 1) {                                                   /* < 2^254 x 2^32: five words */
        uint64_t carry = 0;
        for (int i = 0; i < 5; i++) {
            unsigned __int128 t = (unsigned __int128)m[i] * mult + carry;
            m[i] = (uint64_t)t; carry = (uint64_t)(t >> 64);
        }
    }
    if ((a < 0) != (b < 0) && (x && y)) {                              /* two's complement of the magnitude */
        uint64_t c = 1;
        for (int i = 0; i < TR_KERNEL_ACC_WORDS; i++) { uint64_t t = ~m[i] + c; c = c && t == 0; m[i] = t; }
    }
    uint64_t c = 0;
    for (int i = 0; i < TR_KERNEL_ACC_WORDS; i++) {
        uint64_t s = acc[i] + c, c1 = s < c, t = s + m[i], c2 = t < s;
        acc[i] = t; c = c1 | c2;
    }
}
TR_NUM_HD void tr_kernel_w_from_acc(const uint64_t *acc, tr_kernel_wide *r) {
    uint64_t ext = (acc[TR_KERNEL_ACC_WORDS - 1] >> 63) ? UINT64_MAX : 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) r->w[i] = i < TR_KERNEL_ACC_WORDS ? acc[i] : ext;
}

/* ------------------------------------------------------------------ the shared evaluator */
TR_NUM_HD uint64_t tr_kernel_entry_index(const tr_kernel_view *v, const uint32_t *c) {
    uint64_t idx = 0;
    for (int32_t k = 0; k < v->order; k++) idx += (uint64_t)c[k] * v->stride[k];
    return idx;
}
/* the class sums S_c = sum over the items I of class c of mult_I T_a[I] T_b[I] (DELTA: every cell, multiplicity 1):
 * dense by index, sparse by a merge of the sorted items with both groups' sorted entries */
TR_NUM_HD void tr_kernel_classes(const tr_kernel_view *v, uint64_t a, uint64_t b, uint64_t *acc) {
    for (int i = 0; i < TR_KERNEL_MAX_CLASSES * TR_KERNEL_ACC_WORDS; i++) acc[i] = 0;
    if (v->repr == TR_TENSOR_DENSE) {
        const tr_math_i128 *A = v->value + a * v->cells, *B = v->value + b * v->cells;
        if (v->nitems)
            for (uint64_t t = 0; t < v->nitems; t++) {
                uint64_t e = v->item_index[t];
                __int128 x = tr_i128_get(A[e]), y = tr_i128_get(B[e]);
                if (x && y) tr_kernel_acc_add_prod(acc + TR_KERNEL_ACC_WORDS * v->item_class[t], x, y, v->item_mult[t]);
            }
        else
            for (uint64_t e = 0; e < v->cells; e++) {
                __int128 x = tr_i128_get(A[e]), y = tr_i128_get(B[e]);
                if (x && y) tr_kernel_acc_add_prod(acc, x, y, 1);
            }
        return;
    }
    uint64_t ia = v->offset[a], ea = v->offset[a + 1], ib = v->offset[b], eb = v->offset[b + 1];
    if (!v->nitems) {
        while (ia < ea && ib < eb) {
            uint64_t ka = tr_kernel_entry_index(v, v->coord + ia * (uint64_t)v->order);
            uint64_t kb = tr_kernel_entry_index(v, v->coord + ib * (uint64_t)v->order);
            if (ka < kb) ia++;
            else if (kb < ka) ib++;
            else { tr_kernel_acc_add_prod(acc, tr_i128_get(v->value[ia]), tr_i128_get(v->value[ib]), 1); ia++; ib++; }
        }
        return;
    }
    for (uint64_t t = 0; t < v->nitems && ia < ea && ib < eb; t++) {
        uint64_t e = v->item_index[t];
        while (ia < ea && tr_kernel_entry_index(v, v->coord + ia * (uint64_t)v->order) < e) ia++;
        while (ib < eb && tr_kernel_entry_index(v, v->coord + ib * (uint64_t)v->order) < e) ib++;
        if (ia < ea && ib < eb && tr_kernel_entry_index(v, v->coord + ia * (uint64_t)v->order) == e &&
            tr_kernel_entry_index(v, v->coord + ib * (uint64_t)v->order) == e)
            tr_kernel_acc_add_prod(acc + TR_KERNEL_ACC_WORDS * v->item_class[t], tr_i128_get(v->value[ia]),
                                   tr_i128_get(v->value[ib]), v->item_mult[t]);
    }
}
/* K(a, b) = sum_c coef_c S_c: 1, or 0 on overflow */
TR_NUM_HD int tr_kernel_inner(const tr_kernel_view *v, uint64_t a, uint64_t b, tr_kernel_wide *K) {
    tr_kernel_w_zero(K);
    if (v->kind == TR_KERNEL_MONOMIAL) {                               /* one coefficient per item */
        for (uint64_t t = 0; t < v->nitems; t++) {
            __int128 x = 0, y = 0;
            uint64_t e = v->item_index[t];
            if (v->repr == TR_TENSOR_DENSE) { x = tr_i128_get(v->value[a * v->cells + e]); y = tr_i128_get(v->value[b * v->cells + e]); }
            else {
                for (int s = 0; s < 2; s++) {                           /* binary search of the item in the group */
                    uint64_t g = s ? b : a, lo = v->offset[g], hi = v->offset[g + 1];
                    while (lo < hi) {
                        uint64_t mid = lo + (hi - lo) / 2, k = tr_kernel_entry_index(v, v->coord + mid * (uint64_t)v->order);
                        if (k < e) lo = mid + 1; else hi = mid;
                    }
                    if (lo < v->offset[g + 1] && tr_kernel_entry_index(v, v->coord + lo * (uint64_t)v->order) == e) {
                        if (s) y = tr_i128_get(v->value[lo]); else x = tr_i128_get(v->value[lo]);
                    }
                }
            }
            if (!x || !y) continue;
            uint64_t acc[TR_KERNEL_ACC_WORDS] = {0, 0, 0, 0, 0, 0};
            tr_kernel_acc_add_prod(acc, x, y, v->item_mult[t]);
            tr_kernel_wide p, q;
            tr_kernel_w_from_acc(acc, &p);
            if (!tr_kernel_w_mul_i128(&p, tr_i128_get(v->class_coef[v->item_class[t]]), &q) || !tr_kernel_w_add(K, &q, K)) return 0;
        }
        return 1;
    }
    uint64_t acc[TR_KERNEL_MAX_CLASSES * TR_KERNEL_ACC_WORDS];
    tr_kernel_classes(v, a, b, acc);
    for (int32_t c = 0; c < v->nclass; c++) {
        tr_kernel_wide s, q;
        tr_kernel_w_from_acc(acc + TR_KERNEL_ACC_WORDS * c, &s);
        if (tr_kernel_w_is_zero(&s)) continue;
        if (!tr_kernel_w_mul_i128(&s, tr_i128_get(v->class_coef[c]), &q) || !tr_kernel_w_add(K, &q, K)) return 0;
    }
    return 1;
}
/* T_a[e] of a group (dense index or binary search) */
TR_NUM_HD __int128 tr_kernel_lookup(const tr_kernel_view *v, uint64_t a, uint64_t e) {
    if (v->repr == TR_TENSOR_DENSE) return tr_i128_get(v->value[a * v->cells + e]);
    uint64_t lo = v->offset[a], hi = v->offset[a + 1];
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2, k = tr_kernel_entry_index(v, v->coord + mid * (uint64_t)v->order);
        if (k < e) lo = mid + 1; else hi = mid;
    }
    if (lo < v->offset[a + 1] && tr_kernel_entry_index(v, v->coord + lo * (uint64_t)v->order) == e) return tr_i128_get(v->value[lo]);
    return 0;
}
/* sum_i w_i k(x_i, z): 1, or 0 on overflow (points: npos int64 per point; valid: flags, DELTA only) */
TR_NUM_HD int tr_kernel_witness(const tr_kernel_view *v, uint64_t a, const int64_t *z, const uint8_t *zvalid,
                                tr_kernel_wide *out) {
    tr_kernel_w_zero(out);
    if (v->kind == TR_KERNEL_DELTA) {
        uint64_t e = 0;
        for (int32_t k = 0; k < v->order; k++) {
            uint64_t i;
            if (zvalid && !zvalid[k]) {
                if (v->dims[k] == v->span[k]) return 1;               /* no NULL slot: no row matches */
                i = v->span[k];
            } else {
                __int128 d = (__int128)z[k] - v->lo[k];
                if (d < 0 || d >= (__int128)v->span[k]) return 1;      /* outside the domain: no row matches */
                i = (uint64_t)d;
            }
            e += i * v->stride[k];
        }
        tr_kernel_w_from_i128(tr_kernel_lookup(v, a, e), out);
        return 1;
    }
    for (uint64_t t = 0; t < v->nitems; t++) {
        uint64_t e = v->item_index[t];
        __int128 T = tr_kernel_lookup(v, a, e);
        if (!T) continue;
        tr_kernel_wide p, q;
        tr_kernel_w_from_i128(T, &p);
        if (!tr_kernel_w_mul_i128(&p, tr_i128_get(v->class_coef[v->item_class[t]]), &p)) return 0;
        if (v->item_mult[t] != 1 && !tr_kernel_w_mul_i128(&p, (__int128)v->item_mult[t], &p)) return 0;
        for (int32_t k = 0; k < v->order; k++) {                       /* prod_k z~ at the item's coordinates */
            uint64_t j = (e / v->stride[k]) % v->dims[k];
            int32_t col = v->pos_of_axis[j];
            if (col < 0) continue;                                     /* the constant (or a position outside) */
            if (!tr_kernel_w_mul_i128(&p, (__int128)z[col], &p)) return 0;
        }
        q = *out;
        if (!tr_kernel_w_add(&q, &p, out)) return 0;
    }
    return 1;
}
TR_NUM_HD void tr_kernel_result_init(tr_kernel_result *r) {
    r->status = 0; r->empty = 0;
    tr_kernel_w_zero(&r->num); tr_kernel_w_from_i128(1, &r->den); tr_kernel_w_zero(&r->pairs);
}
/* den > 0 (num, den negated together); empty when den = 0 */
TR_NUM_HD void tr_kernel_normalize(tr_kernel_result *r) {
    if (tr_kernel_w_is_zero(&r->den)) { r->empty = 1; tr_kernel_w_zero(&r->num); return; }
    if (tr_kernel_w_neg(&r->den)) { tr_kernel_w_negate(&r->den); tr_kernel_w_negate(&r->num); }
}
/* the selected-pair sum of a request and its weighted pair count: 1, 0 on overflow, -1 when the self-pair term is
 * needed but not retained */
TR_NUM_HD int tr_kernel_pair_sum(const tr_kernel_view *v, uint64_t a, uint64_t b, int32_t flags, tr_kernel_wide *sum,
                                 tr_kernel_wide *pairs) {
    tr_kernel_wide t;
    if (!tr_kernel_inner(v, a, b, sum)) return 0;
    tr_kernel_w_from_i128(tr_i128_get(v->W[a]), &t);
    tr_kernel_w_from_i128(tr_i128_get(v->W[b]), pairs);
    if (!tr_kernel_w_mul(&t, pairs, pairs)) return 0;
    if (a != b) return 1;
    int self = (flags & TR_KERNEL_SELF) != 0, unordered = (flags & TR_KERNEL_UNORDERED) != 0;
    if (self && !unordered) return 1;                                  /* every ordered pair: K, W^2 */
    if (!v->has_self) return -1;                                       /* the self-pair term is not retained */
    if (self) { if (!tr_kernel_w_add(sum, &v->Dg[a], sum) || !tr_kernel_w_add(pairs, &v->Q[a], pairs)) return 0; }
    else { if (!tr_kernel_w_sub(sum, &v->Dg[a], sum) || !tr_kernel_w_sub(pairs, &v->Q[a], pairs)) return 0; }
    if (unordered) { tr_kernel_w_half(sum); tr_kernel_w_half(pairs); }  /* (K -+ Dg) / 2, (W^2 -+ Q) / 2: even */
    return 1;
}
/* one request: the shared evaluation of the CPU loop and the CUDA kernel */
TR_NUM_HD void tr_kernel_eval_request(const tr_kernel_view *v, const tr_kernel_request *q, const int64_t *points,
                                      const uint8_t *valid, tr_kernel_result *r) {
    tr_kernel_result_init(r);
    if (q->a >= v->ngroups || q->reserved || (q->op != TR_KERNEL_OP_WITNESS && q->b >= v->ngroups) ||
        (q->flags & ~(TR_KERNEL_SELF | TR_KERNEL_UNORDERED | TR_KERNEL_UNBIASED))) { r->status = -1; return; }
    switch (q->op) {
    case TR_KERNEL_OP_SUM:
    case TR_KERNEL_OP_MEAN: {
        if (q->flags & TR_KERNEL_UNBIASED) { r->status = -1; return; }
        int rc = tr_kernel_pair_sum(v, q->a, q->b, q->flags, &r->num, &r->pairs);
        if (rc <= 0) { tr_kernel_result_init(r); r->status = rc < 0 ? -12 : -3; return; }
        if (q->op == TR_KERNEL_OP_MEAN) { r->den = r->pairs; tr_kernel_normalize(r); }
        return;
    }
    case TR_KERNEL_OP_MMD2: {
        if (q->flags & ~TR_KERNEL_UNBIASED) { r->status = -1; return; }
        int fa = (q->flags & TR_KERNEL_UNBIASED) ? 0 : TR_KERNEL_SELF;
        tr_kernel_wide Xa, Pa, Xb, Pb, Pab, t;
        int ra = tr_kernel_pair_sum(v, q->a, q->a, fa, &Xa, &Pa);
        int rb = ra < 0 ? 0 : tr_kernel_pair_sum(v, q->b, q->b, fa, &Xb, &Pb);
        if (ra < 0 || rb < 0) { r->status = -12; return; }
        /* (Xa Pb Pab + Xb Pa Pab - 2 Kab Pa Pb) / (Pa Pb Pab), Pab = W_a W_b */
        int ok = ra && rb;
        tr_kernel_w_from_i128(tr_i128_get(v->W[q->a]), &t);
        tr_kernel_w_from_i128(tr_i128_get(v->W[q->b]), &Pab);
        ok = ok && tr_kernel_w_mul(&t, &Pab, &Pab);
        ok = ok && tr_kernel_w_mul(&Xa, &Pb, &t) && tr_kernel_w_mul(&t, &Pab, &r->num);
        ok = ok && tr_kernel_w_mul(&Xb, &Pa, &t) && tr_kernel_w_mul(&t, &Pab, &t) && tr_kernel_w_add(&r->num, &t, &r->num);
        ok = ok && tr_kernel_inner(v, q->a, q->b, &Xa);                 /* Kab */
        ok = ok && tr_kernel_w_mul(&Xa, &Pa, &t) && tr_kernel_w_mul(&t, &Pb, &t) && tr_kernel_w_add(&t, &t, &t) &&
             tr_kernel_w_sub(&r->num, &t, &r->num);
        ok = ok && tr_kernel_w_mul(&Pa, &Pb, &t) && tr_kernel_w_mul(&t, &Pab, &r->den);
        if (!ok) { tr_kernel_result_init(r); r->status = -3; return; }
        if (tr_kernel_w_is_zero(&Pa) || tr_kernel_w_is_zero(&Pb) || tr_kernel_w_is_zero(&Pab)) {
            r->empty = 1; tr_kernel_w_zero(&r->num); tr_kernel_w_zero(&r->den); return;
        }
        tr_kernel_normalize(r);
        return;
    }
    case TR_KERNEL_OP_WITNESS: {
        if (q->flags || q->point >= v->npoints || !points) { r->status = -1; return; }
        const int64_t *z = points + q->point * (uint64_t)v->npos;
        const uint8_t *zv = valid ? valid + q->point * (uint64_t)v->npos : 0;
        if (zv && v->kind != TR_KERNEL_DELTA)
            for (int32_t k = 0; k < v->npos; k++) if (!zv[k]) { r->status = -1; return; }
        if (!tr_kernel_witness(v, q->a, z, zv, &r->num)) { tr_kernel_result_init(r); r->status = -3; return; }
        tr_kernel_w_from_i128(tr_i128_get(v->W[q->a]), &r->den);
        tr_kernel_normalize(r);
        return;
    }
    default:
        r->status = -1;
    }
}

/* ------------------------------------------------------------------ the self-pair term of a row (CPU and kernels) */
/* whether a selected row contributes (weight present; POLY / MONOMIAL: every kernel column present) and its weight */
TR_NUM_HD int tr_kernel_row_ok(const tr_scope_view *v, const tr_kernel_spec *k, int32_t weight, uint64_t row, __int128 *w) {
    if (weight >= 0) { if (!tr_scope_present(v, weight, row)) return 0; *w = tr_scope_value(v, weight, row); }
    else *w = 1;
    if (k->kind != TR_KERNEL_DELTA)
        for (int32_t j = 0; j < k->ncols; j++) if (!tr_scope_present(v, k->cols[j], row)) return 0;
    return 1;
}
/* w^2 k(x, x) of a contributing row: 1, or 0 on overflow (never, after the build's proof) */
TR_NUM_HD int tr_kernel_self_term(const tr_scope_view *v, const tr_kernel_spec *k, uint64_t row, __int128 w,
                                  tr_kernel_wide *out) {
    tr_kernel_wide kv, t;
    if (k->kind == TR_KERNEL_DELTA) tr_kernel_w_from_i128(1, &kv);
    else if (k->kind == TR_KERNEL_POLY) {
        tr_kernel_wide s;
        tr_kernel_w_from_i128(k->c, &s);
        for (int32_t j = 0; j < k->ncols; j++) {
            __int128 x = tr_scope_value(v, k->cols[j], row);
            tr_kernel_w_from_i128(x * x, &t);
            if (!tr_kernel_w_add(&s, &t, &s)) return 0;
        }
        tr_kernel_w_from_i128(1, &kv);
        for (int32_t e = 0; e < k->degree; e++) if (!tr_kernel_w_mul(&kv, &s, &kv)) return 0;
    } else {
        tr_kernel_w_zero(&kv);
        for (int32_t q = 0; q < k->nterms; q++) {
            tr_kernel_wide m;
            tr_kernel_w_from_i128(k->weight[q], &m);
            for (int32_t j = 0; j < k->ncols; j++) {
                __int128 x = tr_scope_value(v, k->cols[j], row);
                for (int32_t e = 0; e < 2 * (int32_t)k->exps[q][j]; e++) if (!tr_kernel_w_mul_i128(&m, x, &m)) return 0;
            }
            if (!tr_kernel_w_add(&kv, &m, &kv)) return 0;
        }
    }
    tr_kernel_w_from_i128(w, &t);
    if (!tr_kernel_w_mul(&kv, &t, &kv) || !tr_kernel_w_mul(&kv, &t, out)) return 0;
    return 1;
}

/* ------------------------------------------------------------------ host helpers (libalg and the CUDA host code) */
static inline uint32_t tr_kernel_bitlen_u128(unsigned __int128 x) { uint32_t n = 0; while (x) { n++; x >>= 1; } return n; }
/* |c|^z for z <= d as int128 (POLY): 1, or 0 when |c|^d > 2^127 - 1 */
static inline int tr_kernel_powers(int64_t c, int32_t d, __int128 *pw) {
    __int128 p = 1;
    for (int32_t z = 0; z <= d; z++) {
        pw[z] = p;
        if (z < d && p) {
            __int128 q;
            if (__builtin_mul_overflow(p, (__int128)c, &q) || tr_tensor_mag(q) > TR_TENSOR_I128_MAX) return 0;
            p = q;
        }
    }
    return 1;
}

/* a canonical kernel over a scope of ncols columns (tr_kernel_check's form): 0 or the refusal's code */
static inline int32_t tr_kernel_spec_valid(const tr_kernel_spec *k, int32_t ncols) {
    if (k->kind >= TR_KERNEL_RBF && k->kind <= TR_KERNEL_OTHER) return -12;      /* no exact finite feature map */
    if (k->kind != TR_KERNEL_POLY && k->kind != TR_KERNEL_MONOMIAL && k->kind != TR_KERNEL_DELTA) return -1;
    if (k->ncols < 1 || k->reserved[0] || k->reserved[1] || k->reserved[2] || k->reserved[3]) return -1;
    if (k->ncols > TR_KERNEL_MAX_COLUMNS || (k->kind == TR_KERNEL_DELTA && k->ncols > TR_TENSOR_MAX_ORDER)) return -12;
    for (int32_t j = 0; j < k->ncols; j++) {
        if (k->cols[j] < 0 || k->cols[j] >= ncols) return -1;
        for (int32_t i = 0; i < j; i++) if (k->cols[i] == k->cols[j]) return -1;
    }
    if (k->kind == TR_KERNEL_POLY) {
        __int128 pw[TR_KERNEL_MAX_DEGREE + 1];
        if (k->nterms || k->degree < 0) return -1;
        if (k->degree > TR_KERNEL_MAX_DEGREE) return -12;
        if (!tr_kernel_powers(k->c, k->degree, pw)) return -3;
    } else if (k->kind == TR_KERNEL_DELTA) {
        if (k->nterms || k->c || k->degree) return -1;
    } else {
        if (k->c || k->nterms < 1) return -1;
        if (k->nterms > TR_KERNEL_MAX_TERMS) return -12;
        int32_t top = 0;
        for (int32_t t = 0; t < k->nterms; t++) {
            int32_t d = 0;
            for (int32_t j = 0; j < TR_KERNEL_MAX_COLUMNS; j++) { if (j >= k->ncols && k->exps[t][j]) return -1; d += k->exps[t][j]; }
            if (d > TR_KERNEL_MAX_DEGREE) return -12;
            if (!k->weight[t]) return -1;
            if (d > top) top = d;
        }
        if (k->degree != top) return -1;
    }
    return 0;
}
/* the TENSOR-001 spec of a kernel's feature state over scope columns cols (POLY / MONOMIAL: order D on
 * (1, cols...); DELTA: the counts of cols) */
static inline int32_t tr_kernel_tensor_spec(const tr_kernel_spec *k, int32_t degree, int32_t weight, int32_t repr,
                                            tr_tensor_spec *s) {
    memset(s, 0, sizeof *s);
    s->weight = weight; s->repr = repr; s->strategy = 0;
    if (k->kind == TR_KERNEL_DELTA) {
        s->kind = TR_TENSOR_COUNT; s->order = k->ncols; s->policy = 0; s->domain = TR_TENSOR_DOMAIN_SCOPE;
        for (int32_t j = 0; j < k->ncols; j++) { s->ncols[j] = 1; s->cols[j][0] = (int8_t)k->cols[j]; }
        return 0;
    }
    if (degree < 0 || degree > TR_KERNEL_MAX_DEGREE) return -12;
    s->kind = TR_TENSOR_MOMENT; s->order = degree; s->policy = TR_TENSOR_LISTWISE;
    for (int32_t a = 0; a < degree; a++) {
        s->ncols[a] = k->ncols + 1;
        s->cols[a][0] = -1;
        for (int32_t j = 0; j < k->ncols; j++) s->cols[a][j + 1] = (int8_t)k->cols[j];
    }
    return 0;
}
/* the self-pair proof: a bit bound on rows x w^2 x max k(x, x) from the scope's extremes (UINT32_MAX: unbounded) */
static inline uint32_t tr_kernel_self_bits(const tr_scope_view *v, const tr_kernel_spec *k, int32_t weight) {
    uint32_t brow = tr_kernel_bitlen_u128(v->selected), bw = weight >= 0 ? tr_kernel_bitlen_u128(tr_tensor_col_mag(v, weight)) : 1;
    uint32_t bm = 1;
    for (int32_t j = 0; j < k->ncols && k->kind != TR_KERNEL_DELTA; j++) {
        uint32_t b = tr_kernel_bitlen_u128(tr_tensor_col_mag(v, k->cols[j]));
        if (b > bm) bm = b;
    }
    uint32_t bk = 1;
    if (k->kind == TR_KERNEL_POLY) {
        uint32_t s = tr_kernel_bitlen_u128((unsigned __int128)k->ncols) + 2 * bm, bc = tr_kernel_bitlen_u128(tr_abs_u64(k->c));
        bk = (uint32_t)k->degree * ((s > bc ? s : bc) + 1) + 1;
    } else if (k->kind == TR_KERNEL_MONOMIAL) {
        unsigned __int128 sw = 0;
        for (int32_t t = 0; t < k->nterms; t++) sw += tr_abs_u64(k->weight[t]);
        bk = tr_kernel_bitlen_u128(sw) + 2u * (uint32_t)k->degree * bm + 1;
    }
    return brow + 2 * bw + bk + 2;
}
/* ------------------------------------------------------------------ libalg */
/* Validates a kernel over a scope's columns: 0, TR_MATH_INVALID (malformed: columns, degree, exponents, a zero
 * monomial weight table), _UNSUPPORTED (a kind without an exact finite feature map; limits), _OVERFLOW (|c|^d),
 * _DOMAIN_ERROR (feature columns of different decimal scales). MONOMIAL: degree is set to the largest |a_t|. */
int32_t tr_kernel_check(const tr_scope *, tr_kernel_spec *kernel);
/* the feature state (degree: the moment order D >= the kernel's degree, 0 = the kernel's; repr: DENSE / SPARSE) */
tr_kernel_state *tr_kernel_build(const tr_scope *, const tr_kernel_spec *kernel, int32_t weight, int32_t degree,
                                 int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_kernel_need(const tr_scope *, const tr_kernel_spec *kernel, int32_t weight, int32_t degree, int32_t repr,
                       int32_t target, tr_math_need *need, uint64_t *capacity);
/* adopts the CUDA readback: the tensor states and the per-group W (int128), Q, Dg (1024-bit) */
tr_kernel_state *tr_kernel_from_states(const tr_scope *origin, const tr_kernel_spec *kernel, int32_t weight,
                                       int32_t degree, int32_t repr, const tr_tensor_states *tensor,
                                       const tr_math_i128 *W, const tr_kernel_wide *Q, const tr_kernel_wide *Dg,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_kernel_get_info(const tr_kernel_state *, tr_kernel_info *);
/* the per-group W, Q, Dg of the declared kernel */
int32_t tr_kernel_get_group(const tr_kernel_state *, uint64_t group, tr_math_i128 *W, tr_kernel_wide *Q,
                            tr_kernel_wide *Dg);
const tr_tensor *tr_kernel_get_tensor(const tr_kernel_state *);   /* borrowed: the feature tensor */
int32_t tr_kernel_get_spec(const tr_kernel_state *, tr_kernel_spec *);
void tr_kernel_free(tr_kernel_state *);
/* An evaluation of the state with a kernel it retains (NULL: the declared one); TR_MATH_UNSUPPORTED (with nothing
 * allocated) when the family, a column or the degree is absent. The view stays valid while both live. */
tr_kernel_eval *tr_kernel_eval_make(const tr_kernel_state *, const tr_kernel_spec *kernel, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error);
int32_t tr_kernel_eval_get_view(const tr_kernel_eval *, tr_kernel_view *);
void tr_kernel_eval_free(tr_kernel_eval *);
/* n requests (output = n); points: npoints x the kernel's columns (valid: NULL, or flags; DELTA only) */
int32_t tr_kernel_query(const tr_kernel_eval *, const tr_kernel_request *requests, uint64_t n, const int64_t *points,
                        const uint8_t *valid, uint64_t npoints, tr_kernel_result *out, const tr_math_budget *budget,
                        tr_math_refusal *refusal);
/* the bytes a device copy of a view takes (tr_mgpu_kernel_upload's need) */
uint64_t tr_kernel_view_device_bytes(const tr_kernel_view *);
static inline uint64_t tr_kernel_view_bytes(const tr_kernel_view *v, uint64_t nnz) {
    uint64_t b = tr_budget_mul(v->ngroups, TR_KERNEL_GROUP_BYTES);
    b = tr_budget_add(b, tr_budget_mul(v->nitems, 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nclass, 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)TR_TENSOR_MAX_FEATURES + 1, 4));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->order, 16));
    if (v->repr == TR_TENSOR_DENSE) return tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->ngroups, v->cells), 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(v->ngroups, 1), 8));
    return tr_budget_add(b, tr_budget_mul(nnz, tr_budget_add(16, tr_budget_mul((uint64_t)v->order, 4))));
}

/* ------------------------------------------------------------------ the CUDA library (math_kernel_gpu.cu)
 * tr_mgpu_kernel_build: the feature state on the device (the TENSOR-001 device build of tr_kernel_tensor_spec, then W,
 * Q and Dg with one work item per row slice and exact multiword atomics); tensor: host arrays as
 * tr_mgpu_tensor_build's out; W, Q, Dg: G entries each. Adopt with tr_kernel_from_states.
 * tr_mgpu_kernel_upload copies a view once (the state and the evaluation's items); tr_mgpu_kernel_query runs one work
 * item per request (tr_kernel_eval_request) and writes exactly tr_kernel_query's results. */
typedef struct tr_mgpu_kernel_s tr_mgpu_kernel;
int tr_mgpu_kernel_build(const tr_scope_view *view, const tr_kernel_spec *kernel, int32_t weight, int32_t degree,
                         int32_t repr, const tr_math_budget *budget, tr_tensor_states *tensor, tr_math_i128 *W,
                         tr_kernel_wide *Q, tr_kernel_wide *Dg, tr_math_refusal *refusal);
tr_mgpu_kernel *tr_mgpu_kernel_upload(const tr_kernel_view *view, uint64_t nnz, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int *error);
int tr_mgpu_kernel_query(const tr_mgpu_kernel *, const tr_kernel_request *requests, uint64_t n, const int64_t *points,
                         const uint8_t *valid, uint64_t npoints, tr_kernel_result *out, const tr_math_budget *budget,
                         tr_math_refusal *refusal);
uint64_t tr_mgpu_kernel_bytes(const tr_mgpu_kernel *);
void tr_mgpu_kernel_free(tr_mgpu_kernel *);
#ifdef __cplusplus
}
#endif
#endif
