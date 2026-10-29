/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0). */
/* Optional GPU executor for algebraic torbles: the math of a query (filter polynomial, popcount sums, GROUP BY by mask
 * splitting, dominance slices) runs on CUDA. The torus planes and their integrity checks stay on the CPU; the GPU holds
 * the value bit planes, decoded once at upload (value bit j = XOR of the column's V planes 0..j, a linear map).
 * One thread evaluates one 64-row word, exactly as alg.c does on the CPU.
 *
 * The same source builds two ways:
 *   nvcc -O3 -arch=native -shared -Xcompiler -fPIC gpu.cu -o libgpu.so            (CUDA)
 *   g++ -O2 -DEMULATE -x c++ -shared -fPIC gpu.cu -o libgpu_emu.so                 (CPU emulation, for testing)
 * The emulation runs the identical per-word code in a loop, so the logic is tested without a GPU.
 * API: gpu_open(atable) -> handle, gpu_query(handle, plan, &kernel_ms) -> 0 or -1 (unsupported: caller uses the CPU),
 *      gpu_name(), gpu_close(handle). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "torble.h"

#ifdef EMULATE
  #define DEV static inline
  #define HD static inline
  static inline int popc(uint64_t x) { return __builtin_popcountll(x); }
  static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
  static inline int clz32(uint32_t x) { return __builtin_clz(x); }
  static inline int ctz32(uint32_t x) { return __builtin_ctz(x); }
  static inline int popc32(uint32_t x) { return __builtin_popcount(x); }
  #define ADD_U64(p, v) (*(unsigned long long *)(p) += (unsigned long long)(v))
#else
  #include <cuda_runtime.h>
  #define DEV __device__ __forceinline__
  #define HD __host__ __device__ __forceinline__
  DEV int popc(uint64_t x) { return __popcll(x); }
  DEV int ctz64(uint64_t x) { return __ffsll((long long)x) - 1; }
  DEV int clz32(uint32_t x) { return __clz((int)x); }
  DEV int ctz32(uint32_t x) { return __ffs((int)x) - 1; }
  DEV int popc32(uint32_t x) { return __popc(x); }
  #define ADD_U64(p, v) atomicAdd((unsigned long long *)(p), (unsigned long long)(v))
#endif

typedef struct { uint64_t n, nw; int32_t k, width[MAXCOL], msb[MAXCOL]; const uint64_t *P[MAXCOL][32]; } gtable_t;
struct avals_t { uint32_t *p[8]; };

DEV uint64_t X(const gtable_t *T, int c, int j, uint64_t w) { return T->P[c][j][w]; }
DEV uint64_t rowmask_g(const gtable_t *T, uint64_t w) { uint64_t left = T->n - w * 64; return left >= 64 ? ~0ULL : ((1ULL << left) - 1); }
DEV int pw_g(const gtable_t *T, int c, int j) { return T->msb[c] ? T->width[c] - 1 - j : j; }

/* comparator polynomial against a constant on all planes of column c (MSB of the value first) */
DEV uint64_t cmp_col(const gtable_t *T, int c, uint64_t w, int op, uint32_t k) {
    int b = T->width[c];
    if (b < 32 && (k >> b)) return (op == P_LT || op == P_LE || op == P_NE) ? ~0ULL : 0;
    uint64_t lt = 0, eq = ~0ULL;
    for (int i = b - 1; i >= 0; i--) {                    /* value bit i */
        uint64_t x = X(T, c, T->msb[c] ? b - 1 - i : i, w);
        if (k >> i & 1) { lt |= eq & ~x; eq &= x; } else eq &= ~x;
    }
    switch (op) { case P_EQ: return eq; case P_NE: return ~eq; case P_LT: return lt; case P_LE: return lt | eq;
                  case P_GT: return ~(lt | eq); default: return ~lt; }
}
DEV uint64_t eval_prog_g(const gtable_t *T, const qplan_t *q, uint64_t w) {
    uint64_t st[48]; int sp = 0;
    for (int i = 0; i < q->nprog; i++) {
        const pinstr_t *I = &q->prog[i];
        switch (I->op) {
        case P_TRUE: st[sp++] = ~0ULL; break;
        case P_AND: sp--; st[sp - 1] &= st[sp]; break;
        case P_OR: sp--; st[sp - 1] |= st[sp]; break;
        case P_XOR: sp--; st[sp - 1] ^= st[sp]; break;
        case P_NOT: st[sp - 1] = ~st[sp - 1]; break;
        case P_BETWEEN: { int b = T->width[I->col];
            st[sp++] = cmp_col(T, I->col, w, P_GE, I->a) & ((b < 32 && (I->b >> b)) ? ~0ULL : cmp_col(T, I->col, w, P_LE, I->b)); break; }
        case P_CUBE: { uint32_t mk = I->b; uint64_t m = ~0ULL;
            while (mk) { int j = ctz32(mk); mk &= mk - 1; uint64_t x = X(T, I->col, j, w); m &= (I->a >> j & 1) ? x : ~x; }
            st[sp++] = m; break; }
        case P_CLT: case P_CGE: {
            uint32_t mk = I->b; int k = popc32(mk) - 1, ms = T->msb[I->col]; uint64_t lt = 0, eq = ~0ULL;
            while (mk) {
                int j = ms ? ctz32(mk) : 31 - clz32(mk); mk &= ~(1u << j);
                uint64_t x = X(T, I->col, j, w);
                if (I->a >> k & 1) { lt |= eq & ~x; eq &= x; } else eq &= ~x;
                k--;
            }
            st[sp++] = I->op == P_CLT ? lt : ~lt; break; }
        case P_BIT: { int b = T->width[I->col], j = T->msb[I->col] ? b - 1 - (int)I->a : (int)I->a;
            st[sp++] = (int)I->a < b ? X(T, I->col, j, w) : 0; break; }
        case P_PREFIX: { int nb = (int)I->b; uint64_t m = ~0ULL;
            for (int j = 0; j < nb; j++) { uint64_t x = X(T, I->col, j, w); m &= (I->a >> (nb - 1 - j) & 1) ? x : ~x; }
            st[sp++] = m; break; }
        default: st[sp++] = cmp_col(T, I->col, w, I->op, I->a); break;
        }
    }
    return sp ? st[0] : ~0ULL;
}
/* the degree-2 measure over the rows in mask M: popcount algebra */
DEV int64_t poly_sum_g(const gtable_t *T, const qplan_t *q, uint64_t w, uint64_t M) {
    int64_t s = q->p_const * popc(M);
    for (int i = 0; i < q->npl; i++) {
        int c = q->pl_col[i]; int64_t a = 0;
        for (int j = 0; j < T->width[c]; j++) a += (int64_t)popc(X(T, c, j, w) & M) << pw_g(T, c, j);
        s += q->pl_coef[i] * a;
    }
    for (int i = 0; i < q->npb; i++) {
        int ca = q->pb_a[i], cb = q->pb_b[i]; int64_t a = 0;
        for (int k = 0; k < T->width[cb]; k++) {
            uint64_t yk = X(T, cb, k, w) & M; if (!yk) continue;
            int64_t r = 0; for (int j = 0; j < T->width[ca]; j++) r += (int64_t)popc(X(T, ca, j, w) & yk) << pw_g(T, ca, j);
            a += r << pw_g(T, cb, k);
        }
        s += q->pb_coef[i] * a;
    }
    return s;
}
DEV uint32_t value_at(const gtable_t *T, int c, uint64_t w, int r) {      /* integer value of row r from its planes */
    uint32_t v = 0; int b = T->width[c];
    for (int j = 0; j < b; j++) v |= (uint32_t)((X(T, c, j, w) >> r) & 1) << (T->msb[c] ? b - 1 - j : j);
    return v;
}
/* filter of one word (program, row mask, delta plane) */
DEV uint64_t filter_g(const gtable_t *T, const qplan_t *q, uint64_t w) {
    uint64_t F = eval_prog_g(T, q, w) & rowmask_g(T, w);
    if (q->xmode) F &= q->xmode == 1 ? q->xrows[w] : ~q->xrows[w];
    return F;
}
/* looked-up keys (at most 2): one gather per live row; rows whose foreign key is outside the dimension are dropped */
DEV uint64_t lookups_g(const gtable_t *T, const qplan_t *q, uint32_t *const *avals, uint64_t w, uint64_t F, uint32_t kl[2][64], int *slot) {
    int ns = 0;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 1) {
        int a = q->key_arg[i], sl = slot[i] = ns++; uint64_t G = F;
        while (G) { int r = ctz64(G); G &= G - 1;
            uint32_t fk = KM(q->attr_kmask[a], value_at(T, q->attr_fkcol[a], w, r));
            if (fk >= q->attr_n[a]) F &= ~(1ULL << r); else kl[sl][r] = avals[a][fk]; }
    }
    return F;
}
/* mask splitting: the rows of F sharing a key are an AND of key-plane literals; emit(key, mask) per distinct key */
template <class Emit> DEV void split_g(const gtable_t *T, const qplan_t *q, uint64_t w, uint64_t F, uint32_t kl[2][64], const int *slot, Emit emit) {
    uint64_t R = F;
    while (R) {
        int r = ctz64(R); uint64_t M = R, key = 0;
        for (int i = 0; i < q->nkey; i++) {
            int c = q->key_arg[i]; uint64_t kv = 0;
            if (q->key_kind[i] == 1) {
                kv = kl[slot[i]][r]; uint64_t G = M;
                while (G) { int rr = ctz64(G); G &= G - 1; if (kl[slot[i]][rr] != kv) M &= ~(1ULL << rr); }
            } else if (q->key_kind[i] == 3) {
                uint32_t mk = q->key_mask[i];
                while (mk) { int j = ctz32(mk); mk &= mk - 1; uint64_t pl = X(T, c, j, w), bit = (pl >> r) & 1; M &= bit ? pl : ~pl; kv = (kv << 1) | bit; }
            } else {
                int kb = q->key_kind[i] == 2 ? q->key_nbits[i] : T->width[c], km = q->key_kind[i] == 2 ? 1 : T->msb[c];
                for (int j = 0; j < kb; j++) { uint64_t pl = X(T, c, j, w), bit = (pl >> r) & 1; M &= bit ? pl : ~pl; kv |= bit << (km ? kb - 1 - j : j); }
            }
            key = key * q->key_radix[i] + kv;
        }
        R &= ~M;
        emit(key, M);
    }
}
/* everything one 64-row word contributes (the per-word kernel); group results go straight to dense output arrays.
 * avals[a]: device copies of the dimension value arrays of lookup keys (key kind 1). */
DEV void word_g(const gtable_t *T, const qplan_t *q, uint32_t *const *avals, uint64_t w, int64_t *gsum, uint64_t *gcnt, int64_t *tot, uint64_t *cnt) {
    uint64_t F = filter_g(T, q, w);
    if (!F) return;
    uint32_t kl[2][64]; int slot[MAXKEY];
    F = lookups_g(T, q, avals, w, F, kl, slot);
    if (!F) return;
    *cnt += popc(F);
    if (q->agg == A_COUNT) return;
    if (q->agg == A_LIN) {
        for (int i = 0; i < q->nlin; i++) {
            int c = q->lin_col[i]; int64_t s = 0;
            for (int j = 0; j < T->width[c]; j++) s += (int64_t)popc(X(T, c, j, w) & F) << pw_g(T, c, j);
            *tot += q->lin_coef[i] * s;
        }
        return;
    }
    if (q->agg == A_BIL) {
        int ca = q->bil_a, cb = q->bil_b;
        for (int j = 0; j < T->width[ca]; j++) { uint64_t xf = X(T, ca, j, w) & F; if (!xf) continue;
            int64_t s = 0; for (int k = 0; k < T->width[cb]; k++) s += (int64_t)popc(xf & X(T, cb, k, w)) << pw_g(T, cb, k);
            *tot += s << pw_g(T, ca, j); }
        return;
    }
    if (q->agg == A_DOM) {                                /* dominance slices t_(k-1) <= x < t_k */
        uint64_t below = 0;
        for (int k = 0; k < q->ndom; k++) {
            uint64_t lt = cmp_col(T, q->dom_col, w, P_LT, q->dom_t[k]) & F, M = lt & ~below; below = lt;
            if (!M) continue;
            ADD_U64(&gsum[k], poly_sum_g(T, q, w, M)); ADD_U64(&gcnt[k], popc(M));
        }
        *tot += poly_sum_g(T, q, w, F);
        return;
    }
    /* A_POLYG: GROUP BY by mask splitting (keys on planes), or a degree-2 total when there are no keys */
    if (!q->nkey) { *tot += poly_sum_g(T, q, w, F); return; }
    split_g(T, q, w, F, kl, slot, [&](uint64_t key, uint64_t M) {
        int64_t v = poly_sum_g(T, q, w, M);
        ADD_U64(&gsum[key], v); ADD_U64(&gcnt[key], popc(M));
        *tot += v;
    });
}

/* ---------------------------------------------------------------- GROUP BY as one matrix product (A_MM)
 * C = M^T X (torble.h). A thread block takes a tile of GTW = 128 words (8192 rows). Phase 1: one thread per word
 * splits its filtered rows by key; each group present gets a slot in a small shared table and its mask words form
 * that slot's row of M^T (128 words). Phase 2: the tile's block of the product, C_tile = M_tile^T X_tile, on the
 * tensor cores: mma.sync.m16n8k256 with b1 operands and AND + popcount accumulation, 16 groups x 8 measure planes
 * x 256 rows per instruction (k-steps where none of the 16 groups has rows are skipped). Without the b1 MMA the
 * same product runs with __popc over each group's nonzero words. Phase 3: SUM[g] = p_const COUNT[g] + sum_p w_p C[g, p]
 * folded once per group and tile into a global hash table: the output has one entry per group present.
 * Slots that do not fit in the shared table (more than GHS groups in a tile) are evaluated per word, directly. */
/* tile shapes (template parameters): GTW words per tile = threads per block, GHS group slots */
#define MMT template <int GTW, int GHS>
#define MMS mmsh_t<GTW, GHS>
#define GEMPTY (~0ULL)
typedef struct { const uint64_t *a, *b; int64_t w; } mmp_t;
typedef struct {
    const mmp_t *P; int np; unsigned long long *hkey, *hsum, *hcnt; uint64_t hcap; unsigned int *overflow;
    const uint64_t *rstart, *rcum; int nr; uint64_t total; int use_mma; unsigned long long *acc;
    unsigned int *nins, *ins;              /* slots inserted by this query, in order: the output is gathered from them */
} mmctx_t;
MMT struct mmsh_t {
    uint64_t M[GHS][GTW];
    unsigned long long key[GHS], nz[GHS][GTW / 64], sum[GHS]; unsigned int cnt[GHS];
    uint64_t wmap[GTW]; int occ[GHS]; int nocc;
};
#ifdef EMULATE
  static inline unsigned long long CAS64(unsigned long long *p, unsigned long long c, unsigned long long v) { unsigned long long o = *p; if (o == c) *p = v; return o; }
  #define ATOR64(p, v) (*(p) |= (v))
  #define ATADD32(p, v) (*(p) += (v))
  #define ATADD64(p, v) (*(p) += (unsigned long long)(v))
  static inline void ATEXCH(unsigned int *p, unsigned int v) { *p = v; }
  static inline unsigned int FETCH_ADD32(unsigned int *p, unsigned int v) { unsigned int o = *p; *p += v; return o; }
  #define VLOAD64(p) (*(p))
#else
  #define CAS64(p, c, v) atomicCAS((p), (c), (v))
  #define ATOR64(p, v) atomicOr((p), (unsigned long long)(v))
  #define ATADD32(p, v) atomicAdd((p), (unsigned int)(v))
  #define ATADD64(p, v) atomicAdd((p), (unsigned long long)(v))
  #define ATEXCH(p, v) atomicExch((p), (v))
  #define FETCH_ADD32(p, v) atomicAdd((p), (unsigned int)(v))
  #define VLOAD64(p) (*(volatile unsigned long long *)(p))
#endif
HD uint64_t mix64(uint64_t k) { k *= 0x9E3779B97F4A7C15ULL; return k ^ (k >> 29); }
/* the global group table: a slot's key is read first and claimed with a CAS only when empty, so groups already
 * present cost two atomic adds; a new slot is appended to the insertion list (the output is gathered from it) */
DEV void ghash_add(const mmctx_t *X, uint64_t key, int64_t s, uint64_t c) {
    uint64_t h = mix64(key) & (X->hcap - 1);
    for (uint64_t i = 0; i < X->hcap; i++) {
        unsigned long long cur = VLOAD64(&X->hkey[h]);
        if (cur == GEMPTY) {
            cur = CAS64(&X->hkey[h], GEMPTY, (unsigned long long)key);
            if (cur == GEMPTY) { if (X->ins) { unsigned int j = FETCH_ADD32(X->nins, 1u); if (j < X->hcap) X->ins[j] = (unsigned int)h; } cur = key; }
        }
        if (cur == key) { ATADD64(&X->hsum[h], s); ATADD64(&X->hcnt[h], c); return; }
        h = (h + 1) & (X->hcap - 1);
    }
    ATEXCH(X->overflow, 1u);
}
/* output entry i: the i-th inserted slot, which is cleared for the next query (the table stays empty between queries) */
DEV void gather_one(const mmctx_t *X, unsigned int i, unsigned long long *ok, unsigned long long *os, unsigned long long *oc) {
    unsigned int h = X->ins[i];
    ok[i] = X->hkey[h]; os[i] = X->hsum[h]; oc[i] = X->hcnt[h];
    X->hkey[h] = GEMPTY; X->hsum[h] = 0; X->hcnt[h] = 0;
}
/* GROUP BY on the per-word kernel, into the group table: one thread per 64-row word splits its rows by key (mask
 * splitting) and sums the measure over each part by popcount algebra */
DEV void word_mm(const gtable_t *T, const qplan_t *q, uint32_t *const *avals, uint64_t w, const mmctx_t *X, int64_t *tot, uint64_t *cnt) {
    uint64_t F = filter_g(T, q, w); if (!F) return;
    uint32_t kl[2][64]; int slot[MAXKEY];
    F = lookups_g(T, q, avals, w, F, kl, slot); if (!F) return;
    *cnt += popc(F);
    split_g(T, q, w, F, kl, slot, [&](uint64_t key, uint64_t M) {
        int64_t v = poly_sum_g(T, q, w, M); ghash_add(X, key, v, popc(M)); *tot += v;
    });
}
MMT DEV int sinsert(MMS *S, uint64_t key) {
    int h = (int)(mix64(key) >> 40) & (GHS - 1);
    for (int i = 0; i < GHS; i++) {
        unsigned long long o = CAS64(&S->key[h], GEMPTY, (unsigned long long)key);
        if (o == GEMPTY || o == key) return h;
        h = (h + 1) & (GHS - 1);
    }
    return -1;
}
DEV uint64_t xword(const mmp_t *P, uint64_t w) { return w == GEMPTY ? 0 : (P->b ? P->a[w] & P->b[w] : P->a[w]); }

/* phase 1, thread tid: its word's groups into the tile's slots */
MMT DEV void mm_phase1(const gtable_t *T, const qplan_t *q, uint32_t *const *avals, const mmctx_t *X, MMS *S, uint64_t tile, int tid, int64_t *tot, uint64_t *cnt) {
    uint64_t vt = tile * GTW + tid, w = vt; int ok = vt < X->total;
    if (ok && X->nr) {                                    /* only the words of the row ranges */
        int lo = 0, hi = X->nr - 1;
        while (lo < hi) { int mid = (lo + hi + 1) / 2; if (X->rcum[mid] <= vt) lo = mid; else hi = mid - 1; }
        w = X->rstart[lo] + (vt - X->rcum[lo]);
    }
    if (ok && w >= T->nw) ok = 0;
    S->wmap[tid] = ok ? w : GEMPTY;
    if (!ok) return;
    uint64_t F = filter_g(T, q, w); if (!F) return;
    uint32_t kl[2][64]; int slot[MAXKEY];
    F = lookups_g(T, q, avals, w, F, kl, slot); if (!F) return;
    *cnt += popc(F);
    split_g(T, q, w, F, kl, slot, [&](uint64_t key, uint64_t M) {
        int pos = sinsert(S, key);
        if (pos < 0) { int64_t v = poly_sum_g(T, q, w, M); ghash_add(X, key, v, popc(M)); *tot += v; return; }
        S->M[pos][tid] = M; ATOR64(&S->nz[pos][tid >> 6], 1ULL << (tid & 63)); ATADD32(&S->cnt[pos], popc(M));
    });
}
MMT DEV void mm_compact(MMS *S) {
    int n = 0; for (int s = 0; s < GHS; s++) if (S->key[s] != GEMPTY) S->occ[n++] = s;
    S->nocc = n;
}
/* phase 2 without tensor cores: (group, plane) pairs over the group's nonzero words */
MMT DEV void mm_phase2_popc(const mmctx_t *X, MMS *S, int tid, int nthr) {
    int np = X->np, n = S->nocc * np;
    for (int idx = tid; idx < n; idx += nthr) {
        int s = S->occ[idx / np]; const mmp_t *P = &X->P[idx % np]; unsigned int c = 0;
        for (int h = 0; h < GTW / 64; h++) { unsigned long long m = S->nz[s][h];
            while (m) { int i = ctz64(m); m &= m - 1; int wi = h * 64 + i; c += popc(S->M[s][wi] & xword(P, S->wmap[wi])); } }
        if (c) ATADD64(&S->sum[s], P->w * (int64_t)c);
    }
}
/* fragments of one lane for m-tile mi (16 groups), n-tile ni (8 planes), k-step ks (words 4ks..4ks+3).
 * PTX m16n8k256 .b1 layout: a0/a2 row groupID, a1/a3 row groupID + 8, k = threadID_in_group*32 + i (+128 for a2/a3);
 * b0/b1: column groupID, same k. Lane t reads the same k-range for A and B, so the AND-popcount pairs bits correctly. */
MMT DEV void mm_frags(const mmctx_t *X, const MMS *S, int mi, int ni, int ks, int lane, uint32_t a[4], uint32_t b[2]) {
    int g = lane >> 2, t = lane & 3, kw = ks * 4, wA = kw + (t >> 1), wB = kw + 2 + (t >> 1), sh = (t & 1) * 32;
    int r0 = mi * 16 + g, r1 = r0 + 8, s0 = r0 < S->nocc ? S->occ[r0] : -1, s1 = r1 < S->nocc ? S->occ[r1] : -1;
    a[0] = s0 >= 0 ? (uint32_t)(S->M[s0][wA] >> sh) : 0; a[1] = s1 >= 0 ? (uint32_t)(S->M[s1][wA] >> sh) : 0;
    a[2] = s0 >= 0 ? (uint32_t)(S->M[s0][wB] >> sh) : 0; a[3] = s1 >= 0 ? (uint32_t)(S->M[s1][wB] >> sh) : 0;
    int p = ni * 8 + g;
    b[0] = p < X->np ? (uint32_t)(xword(&X->P[p], S->wmap[wA]) >> sh) : 0;
    b[1] = p < X->np ? (uint32_t)(xword(&X->P[p], S->wmap[wB]) >> sh) : 0;
}
/* the 4 k-step words in use for m-tile mi: OR of its groups' nonzero-word masks */
MMT DEV unsigned long long mm_used(const MMS *S, int mi, int h) {
    unsigned long long u = 0; for (int r = mi * 16; r < mi * 16 + 16 && r < S->nocc; r++) u |= S->nz[S->occ[r]][h]; return u;
}
/* C fragment -> weighted sums of the lane's two rows (groupID, groupID + 8) over its two columns */
DEV void mm_fold(const mmctx_t *X, int ni, int lane, const int c[4], int64_t *v0, int64_t *v1) {
    int t = lane & 3, p0 = ni * 8 + 2 * t, p1 = p0 + 1;
    int64_t w0 = p0 < X->np ? X->P[p0].w : 0, w1 = p1 < X->np ? X->P[p1].w : 0;
    *v0 = w0 * c[0] + w1 * c[1]; *v1 = w0 * c[2] + w1 * c[3];
}
MMT DEV void mm_flush(const qplan_t *q, const mmctx_t *X, MMS *S, int tid, int nthr, int64_t *tot) {
    if (tid < S->nocc) {
        int s = S->occ[tid]; int64_t v = (int64_t)S->sum[s] + q->p_const * (int64_t)S->cnt[s];
        ghash_add(X, S->key[s], v, S->cnt[s]); *tot += v;
        S->key[s] = GEMPTY; S->sum[s] = 0; S->cnt[s] = 0; for (int h = 0; h < GTW / 64; h++) S->nz[s][h] = 0;
    }
    for (int k = 0; k < S->nocc; k++) for (int wi = tid; wi < GTW; wi += nthr) S->M[S->occ[k]][wi] = 0;
}
MMT DEV void mm_init_sh(MMS *S, int tid, int nthr) {
    for (int i = tid; i < GHS * GTW; i += nthr) S->M[i / GTW][i % GTW] = 0;
    for (int s = tid; s < GHS; s += nthr) { S->key[s] = GEMPTY; S->sum[s] = 0; S->cnt[s] = 0; for (int h = 0; h < GTW / 64; h++) S->nz[s][h] = 0; }
}

#ifdef EMULATE
/* the b1 MMA, as PTX defines it (D = C + A AND-popc B), on the per-lane fragments: used to test the fragment code */
static void emu_mma(uint32_t a[32][4], uint32_t b[32][2], int c[32][4]) {
    static uint8_t A[16][256], B[256][8]; int D[16][8];
    for (int l = 0; l < 32; l++) { int g = l >> 2, t = l & 3;
        for (int i = 0; i < 32; i++) {
            A[g][t * 32 + i] = a[l][0] >> i & 1; A[g + 8][t * 32 + i] = a[l][1] >> i & 1;
            A[g][128 + t * 32 + i] = a[l][2] >> i & 1; A[g + 8][128 + t * 32 + i] = a[l][3] >> i & 1;
            B[t * 32 + i][g] = b[l][0] >> i & 1; B[128 + t * 32 + i][g] = b[l][1] >> i & 1; } }
    for (int r = 0; r < 16; r++) for (int n = 0; n < 8; n++) { int s = 0; for (int k = 0; k < 256; k++) s += A[r][k] & B[k][n]; D[r][n] = s; }
    for (int l = 0; l < 32; l++) { int g = l >> 2, t = l & 3;
        c[l][0] += D[g][2 * t]; c[l][1] += D[g][2 * t + 1]; c[l][2] += D[g + 8][2 * t]; c[l][3] += D[g + 8][2 * t + 1]; }
}
MMT static void mm_phase2_mma_emu(const mmctx_t *X, MMS *S) {
    int mt = (S->nocc + 15) / 16, nt = (X->np + 7) / 8;
    for (int item = 0; item < mt * nt; item++) {
        int mi = item / nt, ni = item % nt; unsigned long long u[GTW / 64]; for (int h = 0; h < GTW / 64; h++) u[h] = mm_used(S, mi, h);
        int c[32][4]; memset(c, 0, sizeof c);
        for (int ks = 0; ks < GTW / 4; ks++) {
            if (!((u[ks >> 4] >> ((ks & 15) * 4)) & 0xF)) continue;
            uint32_t a[32][4], b[32][2];
            for (int l = 0; l < 32; l++) mm_frags(X, S, mi, ni, ks, l, a[l], b[l]);
            emu_mma(a, b, c);
        }
        for (int l = 0; l < 32; l++) { int64_t v0, v1; mm_fold(X, ni, l, c[l], &v0, &v1);
            int r0 = mi * 16 + (l >> 2), r1 = r0 + 8;
            if (r0 < S->nocc) S->sum[S->occ[r0]] += v0; if (r1 < S->nocc) S->sum[S->occ[r1]] += v1; }
    }
}
MMT static int64_t emu_kmm(const gtable_t *T, const qplan_t *q, uint32_t *const *avals, const mmctx_t *X, uint64_t *cnt) {
    static MMS S; mm_init_sh(&S, 0, 1); int64_t tot = 0;
    for (uint64_t tile = 0; tile < (X->total + GTW - 1) / GTW; tile++) {
        for (int tid = 0; tid < GTW; tid++) mm_phase1(T, q, avals, X, &S, tile, tid, &tot, cnt);
        mm_compact(&S);
        if (S.nocc && X->np) { if (X->use_mma) mm_phase2_mma_emu(X, &S); else mm_phase2_popc(X, &S, 0, 1); }
        for (int tid = 0; tid < GTW; tid++) mm_flush(q, X, &S, tid, GTW, &tot);
    }
    return tot;
}
#else
DEV void mma_b1(int c[4], const uint32_t a[4], const uint32_t b[2]) {
#if !defined(NO_B1MMA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
MMT __global__ void __launch_bounds__(GTW) kmm(const gtable_t *T, const qplan_t *q, avals_t av, mmctx_t X) {
    __shared__ MMS S;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5; int64_t tot = 0; uint64_t cnt = 0;
    mm_init_sh(&S, tid, GTW); __syncthreads();
    uint64_t ntiles = (X.total + GTW - 1) / GTW;
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        mm_phase1(T, q, av.p, &X, &S, tile, tid, &tot, &cnt);
        __syncthreads();
        if (tid == 0) mm_compact(&S);
        __syncthreads();
        if (S.nocc && X.np) {
            if (X.use_mma) {
                int mt = (S.nocc + 15) / 16, nt = (X.np + 7) / 8;
                for (int item = warp; item < mt * nt; item += GTW / 32) {
                    int mi = item / nt, ni = item % nt; unsigned long long u[GTW / 64];
                    for (int h = 0; h < GTW / 64; h++) u[h] = mm_used(&S, mi, h);
                    int c[4] = {0, 0, 0, 0};
                    for (int ks = 0; ks < GTW / 4; ks++) {
                        if (!((u[ks >> 4] >> ((ks & 15) * 4)) & 0xF)) continue;   /* warp-uniform */
                        uint32_t a[4], b[2]; mm_frags(&X, &S, mi, ni, ks, lane, a, b); mma_b1(c, a, b);
                    }
                    int64_t v0, v1; mm_fold(&X, ni, lane, c, &v0, &v1);
                    v0 += __shfl_xor_sync(0xffffffffu, v0, 1); v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
                    v1 += __shfl_xor_sync(0xffffffffu, v1, 1); v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
                    if ((lane & 3) == 0) { int r0 = mi * 16 + (lane >> 2), r1 = r0 + 8;
                        if (r0 < S.nocc && v0) ATADD64(&S.sum[S.occ[r0]], v0); if (r1 < S.nocc && v1) ATADD64(&S.sum[S.occ[r1]], v1); }
                }
            } else mm_phase2_popc(&X, &S, tid, GTW);
        }
        __syncthreads();
        mm_flush(q, &X, &S, tid, GTW, &tot);
        __syncthreads();
    }
    for (int o = 16; o; o >>= 1) { tot += __shfl_down_sync(0xffffffffu, tot, o); cnt += __shfl_down_sync(0xffffffffu, cnt, o); }
    if (lane == 0) { if (tot) atomicAdd(&X.acc[0], (unsigned long long)tot); if (cnt) atomicAdd(&X.acc[1], (unsigned long long)cnt); }
}
/* self-test of the b1 MMA: one warp, random 16 x 256 and 256 x 8 bit matrices, compared with popcounts on the host */
__global__ void kmma_test(const uint32_t *A, const uint32_t *B, int *D) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4] = {A[g * 8 + t], A[(g + 8) * 8 + t], A[g * 8 + 4 + t], A[(g + 8) * 8 + 4 + t]}, b[2] = {B[g * 8 + t], B[g * 8 + 4 + t]};
    int c[4] = {0, 0, 0, 0}; mma_b1(c, a, b);
    D[g * 8 + 2 * t] = c[0]; D[g * 8 + 2 * t + 1] = c[1]; D[(g + 8) * 8 + 2 * t] = c[2]; D[(g + 8) * 8 + 2 * t + 1] = c[3];
}
#endif
/* ---------------------------------------------------------------- host side */
typedef struct {
    gtable_t host; gtable_t *dev; uint64_t *buf; size_t bytes;
    int ncache; const void *ck[64]; uint64_t cn[64]; uint32_t *cd[64];
    mmp_t *dP; unsigned long long *hk, *hs, *hc; uint64_t hcap; unsigned int *dovf;    /* A_MM: planes, group hash */
    unsigned int *dins, *dnins; unsigned long long *ok, *os, *oc; int hclean;               /* insertion list, output */
    qplan_t *dq; unsigned long long *dacc; uint64_t *drs, *drc; uint64_t nrcap;
} handle_t;

static int supported(const qplan_t *q) {
    if (q->wide || q->widekey) return 0;                        /* 128-bit sums or group codes: the CPU engine */
    if (q->nsemi) return 0;                                     /* semi-join lookups stay on the CPU */
    if (q->agg == A_GENERAL) return 0;                         /* the caller converts what it can to A_POLYG / A_MM */
    if (q->agg == A_MM && !q->nkey) return 0;
    if (q->nkey > 6) return 0;
    int nl = 0; for (int i = 0; i < q->nkey; i++) nl += q->key_kind[i] == 1;
    return nl <= 2;
}
/* GROUP BY kernel: 0 = per-word (one thread per 64-row word, default), 1 = tile matrix product with __popc,
 * 2 = tile matrix product on the tensor cores (b1 MMA, when its self-test passed; __popc otherwise) */
static int MM_MODE = 0;
static int MMA_OK = -1; static char MM_NOTE[256] = "";
static int mm_planes(const handle_t *h, const qplan_t *q, mmp_t *P) {
    static mplane_t tmp[MAXMP]; int n = tb_mm_planes(h->host.width, h->host.msb, q, tmp);
    for (int i = 0; i < n; i++) { P[i].a = h->host.P[tmp[i].ca][tmp[i].ja]; P[i].b = tmp[i].cb >= 0 ? h->host.P[tmp[i].cb][tmp[i].jb] : NULL; P[i].w = tmp[i].w; }
    return n;
}
/* tile shape: 128 words x 32 group slots, or 64 words x 64 slots when more than 32 groups may share a tile */
static int mm_wide(const qplan_t *q) {
    const char *e = getenv("TORBLES_GPU_TILE");
    if (e) return atoi(e) == 64;
    return q->ngroups > 32;
}
static uint64_t pow2_at_least(uint64_t x) { uint64_t c = 1024; while (c < x) c <<= 1; return c; }
extern "C" const char *gpu_version(void) { return TORBLES_VERSION; }
extern "C" int gpu_mm_mode(int set) { if (set >= 0) MM_MODE = set; return MM_MODE; }
extern "C" const char *gpu_mm_name(void) {
    static char b[400];
    if (MM_MODE == 0) snprintf(b, sizeof b, "per-word mask splitting + popcounts, sparse group table");
    else if (MM_MODE == 1) snprintf(b, sizeof b, "tile matrix product, __popc");
    else if (MMA_OK == 1) snprintf(b, sizeof b, "tile matrix product, tensor-core b1 MMA (mma.sync.m16n8k256 and.popc)%s", MM_NOTE);
    else if (MMA_OK == 0) snprintf(b, sizeof b, "tile matrix product, __popc (b1 MMA unavailable: %s)", MM_NOTE);
    else snprintf(b, sizeof b, "tile matrix product, b1 MMA not yet tested");
    return b;
}

#ifdef EMULATE
extern "C" const char *gpu_name(void) { return "CPU emulation of the GPU kernels"; }
extern "C" void *gpu_open(const atable_t *t) {
    handle_t *h = (handle_t *)calloc(1, sizeof *h); gtable_t *g = &h->host;
    g->n = t->n; g->nw = t->nw; g->k = t->k;
    size_t np = 0; for (int c = 0; c < t->k; c++) np += t->width[c];
    h->bytes = np * t->nw * 8; h->buf = (uint64_t *)malloc(h->bytes ? h->bytes : 8); size_t off = 0;
    for (int c = 0; c < t->k; c++) {
        g->width[c] = t->width[c]; g->msb[c] = t->msb[c];
        if (t->vflag[c]) {                                     /* self-extending column: planes from the sidecar */
            uint64_t tmp[32];
            for (int j = 0; j < t->width[c]; j++) g->P[c][j] = h->buf + off + (size_t)j * t->nw;
            for (uint64_t w = 0; w < t->nw; w++) {
                tb_virtual_planes(t, c, t->V[t->vflag[c] - 1][0][w], w, tmp, t->width[c]);
                for (int j = 0; j < t->width[c]; j++) ((uint64_t *)g->P[c][j])[w] = tmp[j];
            }
            off += (size_t)t->width[c] * t->nw; continue;
        }
        for (int j = 0; j < t->width[c]; j++) {
            uint64_t *p = h->buf + off; const uint64_t *prev = j ? p - t->nw : NULL; off += t->nw;
            for (uint64_t w = 0; w < t->nw; w++) p[w] = (prev ? prev[w] : 0) ^ t->V[c][j][w];   /* prefix XOR = decode */
            g->P[c][j] = p;
        }
    }
    h->dev = g; MMA_OK = 1; snprintf(MM_NOTE, sizeof MM_NOTE, ", emulated on the CPU"); return h;
}
extern "C" int gpu_query(void *hv, qplan_t *q, double *ms) {
    if (!supported(q)) return -1;
    handle_t *h = (handle_t *)hv; int64_t tot = 0; uint64_t cnt = 0;
    uint32_t *av[8]; for (int a = 0; a < q->nattr; a++) av[a] = (uint32_t *)q->attr_vals[a];
    uint64_t total = h->host.nw, *rs = NULL, *rc = NULL; int nr = 0;
    if (q->nranges) {
        rs = (uint64_t *)malloc(q->nranges * 8); rc = (uint64_t *)malloc(q->nranges * 8); total = 0;
        for (uint64_t r = 0; r < q->nranges; r++) {
            uint64_t a = q->ranges[2 * r], b = q->ranges[2 * r + 1] < h->host.nw ? q->ranges[2 * r + 1] : h->host.nw;
            rs[nr] = a; rc[nr] = total; total += b > a ? b - a : 0; nr++;
        }
    }
    if (q->agg == A_MM) {
        static mmp_t P[MAXMP]; int np = mm_planes(h, q, P); if (np < 0) { free(rs); free(rc); return -1; }
        uint64_t cap = pow2_at_least(2 * (q->out_cap ? q->out_cap : 1));
        for (;;) {
            unsigned long long *k = (unsigned long long *)malloc(cap * 8), *s = (unsigned long long *)calloc(cap, 8), *c = (unsigned long long *)calloc(cap, 8);
            unsigned int *ins = (unsigned int *)malloc(cap * 4), nins = 0;
            memset(k, 0xFF, cap * 8); unsigned int ovf = 0; tot = 0; cnt = 0;
            mmctx_t X = {P, np, k, s, c, cap, &ovf, rs, rc, nr, total, MM_MODE == 2, NULL, &nins, ins};
            if (MM_MODE == 0) {
                for (uint64_t vt = 0; vt < total; vt++) {
                    uint64_t w = vt;
                    if (nr) { int lo = 0, hi = nr - 1; while (lo < hi) { int mid = (lo + hi + 1) / 2; if (rc[mid] <= vt) lo = mid; else hi = mid - 1; } w = rs[lo] + (vt - rc[lo]); }
                    if (w < h->host.nw) word_mm(&h->host, q, av, w, &X, &tot, &cnt);
                }
            } else tot = mm_wide(q) ? emu_kmm<64, 64>(&h->host, q, av, &X, &cnt) : emu_kmm<128, 32>(&h->host, q, av, &X, &cnt);
            if (!ovf) {                                         /* gather the output from the insertion list */
                unsigned long long *ok = (unsigned long long *)malloc(cap * 24), *os = ok + cap, *oc = os + cap;
                for (unsigned int i = 0; i < nins; i++) gather_one(&X, i, ok, os, oc);
                for (uint64_t i = 0; i < cap; i++) if (k[i] != GEMPTY || s[i] || c[i]) { fprintf(stderr, "group table not left empty\n"); abort(); }
                q->out_n = nins;
                for (unsigned int i = 0; i < nins && i < q->out_cap; i++) { q->out_key[i] = ok[i]; q->out_sum[i] = (int64_t)os[i]; q->out_cnt[i] = oc[i]; }
                free(ok); free(k); free(s); free(c); free(ins); break;
            }
            free(k); free(s); free(c); free(ins); cap *= 4;
        }
    } else {
        uint64_t ng = q->agg == A_DOM ? (uint64_t)q->ndom : q->nkey ? q->ngroups : 0;
        if (ng) { memset(q->out_sum, 0, ng * 8); memset(q->out_cnt, 0, ng * 8); }
        for (uint64_t vt = 0; vt < total; vt++) {
            uint64_t w = vt;
            if (nr) { int lo = 0, hi = nr - 1; while (lo < hi) { int mid = (lo + hi + 1) / 2; if (rc[mid] <= vt) lo = mid; else hi = mid - 1; } w = rs[lo] + (vt - rc[lo]); }
            if (w < h->host.nw) word_g(&h->host, q, av, w, q->out_sum, q->out_cnt, &tot, &cnt);
        }
    }
    free(rs); free(rc);
    q->total_sum = tot; q->total_cnt = cnt; if (ms) *ms = 0; return 0;
}
extern "C" void gpu_close(void *hv) { handle_t *h = (handle_t *)hv; free(h->buf); free(h); }
#else
__global__ void kdecode(uint64_t *p, uint64_t nw, int b) {            /* prefix XOR over a column's planes, in place */
    uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; if (w >= nw) return;
    uint64_t x = 0; for (int j = 0; j < b; j++) { x ^= p[(uint64_t)j * nw + w]; p[(uint64_t)j * nw + w] = x; }
}
__global__ void kquery(const gtable_t *T, const qplan_t *q, avals_t av, int64_t *gsum, uint64_t *gcnt, unsigned long long *acc,
                       const uint64_t *rstart, const uint64_t *rcum, int nr, uint64_t total) {
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, w = t;
    int64_t tot = 0; uint64_t cnt = 0;
    if (nr && t < total) {                                    /* only the words of the row ranges */
        int lo = 0, hi = nr - 1;
        while (lo < hi) { int mid = (lo + hi + 1) / 2; if (rcum[mid] <= t) lo = mid; else hi = mid - 1; }
        w = rstart[lo] + (t - rcum[lo]);
    }
    if (t < total && w < T->nw) word_g(T, q, av.p, w, gsum, gcnt, &tot, &cnt);
    for (int o = 16; o; o >>= 1) { tot += __shfl_down_sync(0xffffffffu, tot, o); cnt += __shfl_down_sync(0xffffffffu, cnt, o); }
    if ((threadIdx.x & 31) == 0) { if (tot) atomicAdd(&acc[0], (unsigned long long)tot); if (cnt) atomicAdd(&acc[1], (unsigned long long)cnt); }
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 0; } } while (0)
#define CKI(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -2; } } while (0)
static char NAME[256]; static int NSM = 0;
extern "C" const char *gpu_name(void) {
    cudaDeviceProp p; int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&p, d) != cudaSuccess) return "no CUDA device";
    snprintf(NAME, sizeof NAME, "%.200s, %d SMs, %.1f GiB, compute %d.%d", p.name, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, p.major, p.minor);
    return NAME;
}
static void mma_selftest(void) {
#ifdef NO_B1MMA
    MMA_OK = 0; snprintf(MM_NOTE, sizeof MM_NOTE, "built without it (nvcc could not compile b1 mma.sync for this GPU)"); return;
#else
    uint32_t A[128], B[64]; int D[128], *dD; uint32_t *dA, *dB; uint64_t s = 88172645463325252ULL;
    for (int i = 0; i < 128; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; A[i] = (uint32_t)s; }
    for (int i = 0; i < 64; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; B[i] = (uint32_t)(s >> 7); }
    cudaMalloc((void **)&dA, sizeof A); cudaMalloc((void **)&dB, sizeof B); cudaMalloc((void **)&dD, sizeof D);
    cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice); cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice); cudaMemset(dD, 0, sizeof D);
    kmma_test<<<1, 32>>>(dA, dB, dD);
    cudaError_t e = cudaDeviceSynchronize(); if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) { MMA_OK = 0; snprintf(MM_NOTE, sizeof MM_NOTE, "self-test failed to run: %s", cudaGetErrorString(e)); return; }
    cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost); cudaFree(dA); cudaFree(dB); cudaFree(dD);
    int bad = 0;
    for (int r = 0; r < 16; r++) for (int n = 0; n < 8; n++) { int x = 0; for (int j = 0; j < 8; j++) x += __builtin_popcount(A[r * 8 + j] & B[n * 8 + j]); bad += x != D[r * 8 + n]; }
    MMA_OK = bad == 0;
    if (bad) snprintf(MM_NOTE, sizeof MM_NOTE, "self-test: %d of 128 results differ from popcounts", bad);
    else snprintf(MM_NOTE, sizeof MM_NOTE, ", self-test passed");
#endif
}
extern "C" void *gpu_open(const atable_t *t) {
    handle_t *h = (handle_t *)calloc(1, sizeof *h); gtable_t *g = &h->host;
    g->n = t->n; g->nw = t->nw; g->k = t->k;
    cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev); NSM = prop.multiProcessorCount;
    size_t np = 0; for (int c = 0; c < t->k; c++) np += t->width[c];
    h->bytes = np * t->nw * 8; CK(cudaMalloc((void **)&h->buf, h->bytes ? h->bytes : 8)); size_t off = 0;
    for (int c = 0; c < t->k; c++) {
        g->width[c] = t->width[c]; g->msb[c] = t->msb[c];
        if (t->vflag[c]) {                                     /* self-extending column: expanded on the host, uploaded decoded */
            uint64_t *hp = (uint64_t *)malloc((size_t)t->width[c] * t->nw * 8), tmp[32];
            for (uint64_t w = 0; w < t->nw; w++) {
                tb_virtual_planes(t, c, t->V[t->vflag[c] - 1][0][w], w, tmp, t->width[c]);
                for (int j = 0; j < t->width[c]; j++) hp[(size_t)j * t->nw + w] = tmp[j];
            }
            CK(cudaMemcpy(h->buf + off, hp, (size_t)t->width[c] * t->nw * 8, cudaMemcpyHostToDevice)); free(hp);
            for (int j = 0; j < t->width[c]; j++) g->P[c][j] = h->buf + off + (size_t)j * t->nw;
            off += (size_t)t->width[c] * t->nw; continue;
        }
        for (int j = 0; j < t->width[c]; j++) { CK(cudaMemcpy(h->buf + off + (size_t)j * t->nw, t->V[c][j], t->nw * 8, cudaMemcpyHostToDevice)); g->P[c][j] = h->buf + off + (size_t)j * t->nw; }
        if (t->width[c]) kdecode<<<(unsigned)((t->nw + 255) / 256), 256>>>(h->buf + off, t->nw, t->width[c]);
        off += (size_t)t->width[c] * t->nw;
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMalloc((void **)&h->dev, sizeof(gtable_t))); CK(cudaMemcpy(h->dev, g, sizeof(gtable_t), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&h->dq, sizeof(qplan_t))); CK(cudaMalloc((void **)&h->dacc, 16));
    CK(cudaMalloc((void **)&h->dP, MAXMP * sizeof(mmp_t))); CK(cudaMalloc((void **)&h->dovf, 4));
    if (MMA_OK < 0) mma_selftest();
    return h;
}
static int ensure_hash(handle_t *h, uint64_t cap) {
    if (h->hcap >= cap) return 0;
    if (h->hk) { cudaFree(h->hk); cudaFree(h->hs); cudaFree(h->hc); cudaFree(h->dins); cudaFree(h->ok); cudaFree(h->os); cudaFree(h->oc); }
    CKI(cudaMalloc((void **)&h->hk, cap * 8)); CKI(cudaMalloc((void **)&h->hs, cap * 8)); CKI(cudaMalloc((void **)&h->hc, cap * 8));
    CKI(cudaMalloc((void **)&h->dins, cap * 4)); CKI(cudaMalloc((void **)&h->ok, cap * 8)); CKI(cudaMalloc((void **)&h->os, cap * 8)); CKI(cudaMalloc((void **)&h->oc, cap * 8));
    if (!h->dnins) CKI(cudaMalloc((void **)&h->dnins, 4));
    h->hcap = cap; h->hclean = 0; return 0;
}
__global__ void kword(const gtable_t *T, const qplan_t *q, avals_t av, mmctx_t X) {
    uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, w = t;
    int64_t tot = 0; uint64_t cnt = 0;
    if (X.nr && t < X.total) {                               /* only the words of the row ranges */
        int lo = 0, hi = X.nr - 1;
        while (lo < hi) { int mid = (lo + hi + 1) / 2; if (X.rcum[mid] <= t) lo = mid; else hi = mid - 1; }
        w = X.rstart[lo] + (t - X.rcum[lo]);
    }
    if (t < X.total && w < T->nw) word_mm(T, q, av.p, w, &X, &tot, &cnt);
    for (int o = 16; o; o >>= 1) { tot += __shfl_down_sync(0xffffffffu, tot, o); cnt += __shfl_down_sync(0xffffffffu, cnt, o); }
    if ((threadIdx.x & 31) == 0) { if (tot) atomicAdd(&X.acc[0], (unsigned long long)tot); if (cnt) atomicAdd(&X.acc[1], (unsigned long long)cnt); }
}
__global__ void kgather(mmctx_t X, unsigned long long *ok, unsigned long long *os, unsigned long long *oc) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, n = *X.nins;
    if (i < n && i < X.hcap) gather_one(&X, i, ok, os, oc);
}
extern "C" int gpu_query(void *hv, qplan_t *q, double *ms) {
    if (!supported(q)) return -1;
    handle_t *h = (handle_t *)hv;
    static mmp_t P[MAXMP]; int np = 0;
    if (q->agg == A_MM && (np = mm_planes(h, q, P)) < 0) return -1;
    uint64_t ng = q->agg == A_MM ? 0 : q->agg == A_DOM ? (uint64_t)q->ndom : q->nkey ? q->ngroups : 0;
    qplan_t hq = *q; int64_t *gs = 0; uint64_t *gc = 0; uint32_t *dt = 0;
    if (q->agg == A_DOM && q->ndom) { CKI(cudaMalloc((void **)&dt, q->ndom * 4)); CKI(cudaMemcpy(dt, q->dom_t, q->ndom * 4, cudaMemcpyHostToDevice)); hq.dom_t = dt; }
    if (ng) { CKI(cudaMalloc((void **)&gs, ng * 8)); CKI(cudaMalloc((void **)&gc, ng * 8)); CKI(cudaMemset(gs, 0, ng * 8)); CKI(cudaMemset(gc, 0, ng * 8)); }
    if (q->xmode) {                                            /* delta plane: uploaded once, cached */
        int f = -1; for (int i = 0; i < h->ncache; i++) if (h->ck[i] == q->xrows && h->cn[i] == ~0ULL) f = i;
        if (f < 0) {
            if (h->ncache == 64) return -1;
            f = h->ncache++; h->ck[f] = q->xrows; h->cn[f] = ~0ULL;
            CKI(cudaMalloc((void **)&h->cd[f], h->host.nw * 8)); CKI(cudaMemcpy(h->cd[f], q->xrows, h->host.nw * 8, cudaMemcpyHostToDevice));
        }
        hq.xrows = (const uint64_t *)h->cd[f];
    }
    CKI(cudaMemcpy(h->dq, &hq, sizeof(qplan_t), cudaMemcpyHostToDevice)); CKI(cudaMemset(h->dacc, 0, 16));
    uint64_t total = h->host.nw; int nr = 0;
    if (q->nranges) {
        uint64_t *hs = (uint64_t *)malloc(q->nranges * 8), *hc = (uint64_t *)malloc(q->nranges * 8); total = 0;
        for (uint64_t r = 0; r < q->nranges; r++) {
            uint64_t a = q->ranges[2 * r], b = q->ranges[2 * r + 1] < h->host.nw ? q->ranges[2 * r + 1] : h->host.nw;
            hs[nr] = a; hc[nr] = total; total += b > a ? b - a : 0; nr++;
        }
        if ((uint64_t)nr > h->nrcap) { if (h->drs) { cudaFree(h->drs); cudaFree(h->drc); }
            CKI(cudaMalloc((void **)&h->drs, nr * 8)); CKI(cudaMalloc((void **)&h->drc, nr * 8)); h->nrcap = nr; }
        CKI(cudaMemcpy(h->drs, hs, nr * 8, cudaMemcpyHostToDevice)); CKI(cudaMemcpy(h->drc, hc, nr * 8, cudaMemcpyHostToDevice)); free(hs); free(hc);
    }
    avals_t av; memset(&av, 0, sizeof av);
    for (int a = 0; a < q->nattr; a++) {                       /* dimension value arrays: uploaded once, cached */
        int f = -1; for (int i = 0; i < h->ncache; i++) if (h->ck[i] == q->attr_vals[a] && h->cn[i] == q->attr_n[a]) f = i;
        if (f < 0) {
            if (h->ncache == 64) return -1;
            f = h->ncache++; h->ck[f] = q->attr_vals[a]; h->cn[f] = q->attr_n[a];
            CKI(cudaMalloc((void **)&h->cd[f], q->attr_n[a] * 4 + 4)); CKI(cudaMemcpy(h->cd[f], q->attr_vals[a], q->attr_n[a] * 4, cudaMemcpyHostToDevice));
        }
        av.p[a] = h->cd[f];
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float kms = 0;
    if (q->agg == A_MM) {
        if (np) CKI(cudaMemcpy(h->dP, P, np * sizeof(mmp_t), cudaMemcpyHostToDevice));
        uint64_t cap = pow2_at_least(2 * (q->out_cap ? q->out_cap : 1));
        unsigned int nins = 0;
        for (;;) {
            if (ensure_hash(h, cap)) return -2;
            cap = h->hcap;                                     /* the table in place (kept empty between queries) */
            if (!h->hclean) { CKI(cudaMemset(h->hk, 0xFF, cap * 8)); CKI(cudaMemset(h->hs, 0, cap * 8)); CKI(cudaMemset(h->hc, 0, cap * 8)); h->hclean = 1; }
            CKI(cudaMemset(h->dovf, 0, 4)); CKI(cudaMemset(h->dacc, 0, 16)); CKI(cudaMemset(h->dnins, 0, 4));
            mmctx_t X = {h->dP, np, h->hk, h->hs, h->hc, cap, h->dovf, h->drs, h->drc, nr, total, MM_MODE == 2 && MMA_OK == 1, h->dacc, h->dnins, h->dins};
            cudaEventRecord(e0);
            if (MM_MODE == 0) { if (total) kword<<<(unsigned)((total + 255) / 256), 256>>>(h->dev, h->dq, av, X); }
            else {
                int wide = mm_wide(q), tw = wide ? 64 : 128;
                uint64_t ntiles = (total + tw - 1) / tw; unsigned nb = (unsigned)(ntiles < (uint64_t)NSM * 16 ? ntiles : (uint64_t)NSM * 16);
                if (nb) { if (wide) kmm<64, 64><<<nb, 64>>>(h->dev, h->dq, av, X); else kmm<128, 32><<<nb, 128>>>(h->dev, h->dq, av, X); }
            }
            cudaEventRecord(e1); CKI(cudaEventSynchronize(e1)); CKI(cudaGetLastError());
            cudaEventElapsedTime(&kms, e0, e1);
            unsigned int ovf = 0; CKI(cudaMemcpy(&ovf, h->dovf, 4, cudaMemcpyDeviceToHost));
            if (!ovf) {                                          /* gather the groups present (and clear their slots) */
                CKI(cudaMemcpy(&nins, h->dnins, 4, cudaMemcpyDeviceToHost));
                if (nins) kgather<<<(nins + 255) / 256, 256>>>(X, h->ok, h->os, h->oc);
                break;
            }
            h->hclean = 0; cap *= 4;                             /* too many groups for the table: a bigger one */
        }
        q->out_n = nins;
        uint64_t m = nins < q->out_cap ? nins : q->out_cap;
        if (m) {
            CKI(cudaMemcpy(q->out_key, h->ok, m * 8, cudaMemcpyDeviceToHost)); CKI(cudaMemcpy(q->out_sum, h->os, m * 8, cudaMemcpyDeviceToHost));
            CKI(cudaMemcpy(q->out_cnt, h->oc, m * 8, cudaMemcpyDeviceToHost));
        }
        if (nins > q->out_cap) h->hclean = 1;                  /* all gathered and cleared; the caller reruns */
    } else {
        cudaEventRecord(e0);
        if (total) kquery<<<(unsigned)((total + 255) / 256), 256>>>(h->dev, h->dq, av, gs, gc, h->dacc, h->drs, h->drc, nr, total);
        cudaEventRecord(e1); CKI(cudaEventSynchronize(e1)); CKI(cudaGetLastError());
        cudaEventElapsedTime(&kms, e0, e1);
    }
    if (ms) *ms = kms;
    unsigned long long a[2]; CKI(cudaMemcpy(a, h->dacc, 16, cudaMemcpyDeviceToHost));
    q->total_sum = (int64_t)a[0]; q->total_cnt = a[1];
    if (ng) { CKI(cudaMemcpy(q->out_sum, gs, ng * 8, cudaMemcpyDeviceToHost)); CKI(cudaMemcpy(q->out_cnt, gc, ng * 8, cudaMemcpyDeviceToHost)); cudaFree(gs); cudaFree(gc); }
    if (dt) cudaFree(dt);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
extern "C" void gpu_close(void *hv) {
    handle_t *h = (handle_t *)hv; for (int i = 0; i < h->ncache; i++) cudaFree(h->cd[i]);
    cudaFree(h->buf); cudaFree(h->dev); cudaFree(h->dq); cudaFree(h->dacc); cudaFree(h->dP); cudaFree(h->dovf);
    if (h->hk) { cudaFree(h->hk); cudaFree(h->hs); cudaFree(h->hc); cudaFree(h->dins); cudaFree(h->ok); cudaFree(h->os); cudaFree(h->oc); }
    if (h->dnins) cudaFree(h->dnins);
    if (h->drs) { cudaFree(h->drs); cudaFree(h->drc); }
    free(h);
}
#endif
