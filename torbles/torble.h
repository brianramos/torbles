/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0). */
/* Shared definitions of the torbles engine (alg.c) and its optional GPU executor (gpu.cu). */
#ifndef TORBLES_H
#define TORBLES_H
#define TORBLES_VERSION "0.2.0"
#include <stdint.h>
#define MAXCOL 31                /* columns per torble (the torus has one row per column plus a wrap row) */
#define NB 33
enum { P_TRUE, P_EQ, P_NE, P_LT, P_LE, P_GT, P_GE, P_BETWEEN, P_AND, P_OR, P_NOT, P_PREFIX, P_BIT, P_CUBE, P_CLT, P_CGE, P_XOR };   /* P_XOR: additive (F2) correction */
/* P_CLT / P_CGE(col, a = constant, b = plane mask): the code formed by the planes in mask (most significant id bit first)
   is < a / >= a. The factored comparator polynomial sum_j (1 + x_j) c_j E_j on any set of variables. */   /* P_CUBE(col, a = values, b = plane mask): AND of literals on any planes */   /* P_BIT(col, a = bit): value bit a is set */   /* P_PREFIX(col, a = value, b = nbits): top nbits == a */
enum { A_COUNT, A_LIN, A_BIL, A_GENERAL, A_POLYG, A_DOM, A_MM };   /* A_MM: GROUP BY as one AND-popcount matrix product, see below */   /* A_DOM: dominance sums over x < t_k, see below */   /* A_POLYG: GROUP BY with a degree-2 polynomial measure, by mask splitting */
enum { M_FCOL, M_ATTR, M_CONST, M_ADD, M_SUB, M_MUL };

typedef struct { int32_t op, col; uint32_t a, b; } pinstr_t;
typedef struct { int32_t op, arg; int64_t c; } minstr_t;
typedef struct {
    int32_t nprog; pinstr_t prog[1024];
    int32_t nsemi; int32_t semi_col[8]; const uint64_t *semi_bm[8];
    int32_t agg;
    int32_t nlin; int32_t lin_col[8]; int64_t lin_coef[8];
    int32_t bil_a, bil_b;
    int32_t nmeas; minstr_t meas[48];
    int32_t nattr; int32_t attr_fkcol[8]; const uint32_t *attr_vals[8];
    int32_t nkey; int32_t key_kind[6], key_arg[6]; uint64_t key_radix[6]; int32_t key_nbits[6];   /* kind 2 = top-bit prefix of an fk */
    uint64_t ngroups; int64_t *out_sum; uint64_t *out_cnt;
    int64_t total_sum; uint64_t total_cnt;
    uint64_t semi_n[8], attr_n[8];                    /* dimension sizes: out-of-range foreign keys never match */
    /* degree-2 polynomial measure: p_const + sum pl_coef*x + sum pb_coef*a*b, all fact columns */
    int32_t npl, pl_col[8]; int64_t pl_coef[8];
    int32_t npb, pb_a[8], pb_b[8]; int64_t pb_coef[8]; int64_t p_const;
    uint64_t nranges; const uint64_t *ranges;          /* optional [start, end) unit ranges to scan (physical order as index) */
    uint32_t key_mask[6];                              /* key kind 3: projection of the fk planes in this mask (code = those bits, first plane most significant) */
    int32_t dom_col, ndom; const uint32_t *dom_t;      /* A_DOM: column x and ascending thresholds t_k */
    uint32_t semi_kmask[8], attr_kmask[8];             /* join through a derived surrogate: dimension row = fk & mask (0 = no mask) */
    const uint64_t *xrows; int32_t xmode;              /* delta plane (one bit per fact row): 1 = only these rows, 2 = all but these */
    /* A_MM output is sparse: out_key[i], out_sum[i], out_cnt[i] for i < out_n (the group code is the same mixed-radix
     * key as the dense arrays use). If more than out_cap groups exist, out_n is set to the number needed and the
     * caller retries with larger arrays. mm_stats: [0] tile-slots evaluated dense, [1] sparse, [2] (word, group)
     * pairs past a full tile table (evaluated per word), [3] tiles with groups. */
    uint64_t out_cap, out_n; uint64_t *out_key; int64_t mm_stats[4];
} qplan_t;
#define KM(m, v) ((m) ? ((v) & (m)) : (v))

typedef struct atable_s {
    uint64_t n, nw; int32_t k, width[MAXCOL], msb[MAXCOL];   /* msb: column stored most-significant-bit first */
    uint64_t *V[MAXCOL][NB];          /* V planes, NULL = implicit zero */
    uint64_t *H[MAXCOL + 1][NB];      /* H planes for rows 0..k-1, index k = the wrap row */
    int32_t nfaces; uint64_t *face[(MAXCOL + 1) * NB][4]; int32_t fterms[(MAXCOL + 1) * NB];
    uint64_t bytes;
    /* self-extending columns: column c is virtual when vflag[c] = (flag column) + 1. Its value bits live only for the
     * flagged rows, in a sidecar torble vside[c] (one column, rows in order). The sidecar row of a flagged row is
     * its rank: vrank[c][w] = flagged rows before word w, plus the popcount of the flag bits before it in the word.
     * No pointer is stored; the address is computed from the flag plane. */
    int32_t vflag[MAXCOL]; const struct atable_s *vside[MAXCOL]; const uint32_t *vrank[MAXCOL];
    void *map; uint64_t maplen;       /* loaded from a file: the planes live in this private mapping (at_load) */
} atable_t;

static inline uint64_t tb_pdep(uint64_t src, uint64_t mask) {   /* scatter the low bits of src into the set bits of mask */
#if defined(__BMI2__) && !defined(__CUDACC__)
    return __builtin_ia32_pdep_di(src, mask);
#else
    uint64_t out = 0;
    for (uint64_t bb = 1; mask; bb += bb) { if (src & bb) out |= mask & -mask; mask &= mask - 1; }
    return out;
#endif
}
/* decoded value planes 0..n-1 of a virtual column for word w (flagw = the flag plane's word): the flagged rows'
 * sidecar entries are contiguous from rank vrank[c][w]; gather them from one or two sidecar words and PDEP them into
 * the flagged row positions. Words without flagged rows cost nothing. */
static inline void tb_virtual_planes(const atable_t *t, int c, uint64_t flagw, uint64_t w, uint64_t *out, int n) {
    int cnt = __builtin_popcountll(flagw);
    if (!cnt) { for (int j = 0; j < n; j++) out[j] = 0; return; }
    const atable_t *sd = t->vside[c]; uint64_t a = t->vrank[c][w], sw = a >> 6; int off = (int)(a & 63);
    uint64_t m = cnt == 64 ? ~0ULL : ((1ULL << cnt) - 1), x0 = 0, x1 = 0;
    int two = off + cnt > 64;
    for (int j = 0; j < n; j++) {                       /* sidecar planes decoded on the fly (prefix XOR) */
        x0 ^= sw < sd->nw ? sd->V[0][j][sw] : 0;
        if (two) x1 ^= sw + 1 < sd->nw ? sd->V[0][j][sw + 1] : 0;
        uint64_t chunk = off ? ((x0 >> off) | (two ? x1 << (64 - off) : 0)) : x0;
        out[j] = tb_pdep(chunk & m, flagw);
    }
}

/* GROUP BY as a matrix product. With M[r, g] = 1 when row r passes the filter and has group g (one-hot rows: M is the
 * Khatri-Rao product of the filter and the key-bit indicators [~b, b]) and X[r, p] = measure plane p (value planes of
 * the linear terms, products x_j & y_k of plane pairs for the bilinear ones),
 *     C = M^T X  (an AND-popcount product over the rows),   SUM[g] = p_const COUNT[g] + sum_p w_p C[g, p].
 * The bit planes are already the column-major layout of X, and a group's mask over a tile of rows is a row of M^T.
 * Plane p of the product: X(ca, ja) [& X(cb, jb) when cb >= 0], weight w. */
#define MAXMP 1024
typedef struct { int16_t ca, ja, cb, jb; int64_t w; } mplane_t;
static inline int tb_mm_planes(const int32_t *width, const int32_t *msb, const qplan_t *q, mplane_t *P) {
    int n = 0;
#define TB_PW(c, j) (msb[c] ? width[c] - 1 - (j) : (j))
    for (int i = 0; i < q->npl; i++) { int c = q->pl_col[i];
        for (int j = 0; j < width[c]; j++) { if (n == MAXMP) return -1;
            P[n].ca = (int16_t)c; P[n].ja = (int16_t)j; P[n].cb = -1; P[n].jb = 0; P[n].w = q->pl_coef[i] * ((int64_t)1 << TB_PW(c, j)); n++; } }
    for (int i = 0; i < q->npb; i++) { int a = q->pb_a[i], b = q->pb_b[i];
        for (int j = 0; j < width[a]; j++) for (int k = 0; k < width[b]; k++) { if (n == MAXMP) return -1;
            P[n].ca = (int16_t)a; P[n].ja = (int16_t)j; P[n].cb = (int16_t)b; P[n].jb = (int16_t)k;
            P[n].w = q->pb_coef[i] * ((int64_t)1 << (TB_PW(a, j) + TB_PW(b, k))); n++; } }
#undef TB_PW
    return n;
}

#endif
