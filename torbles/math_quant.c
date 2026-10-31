/* QUANT-001 / QUANT-002: quantifier states and coverage states over the child rows of a scope; see math_quant.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_quant.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_quant_counts, TR_QUANT_ABI);          /* the ctypes boundary of QUANT-001 / 002 (math_handles.h) */
TR_MATH_ABI(tr_quant_groups_info, TR_QUANT_ABI);
TR_MATH_ABI(tr_quant_keys, TR_QUANT_ABI);
TR_MATH_ABI(tr_quant_dict_info, TR_QUANT_ABI);
TR_MATH_ABI(tr_quant_cover_arrays, TR_QUANT_ABI);
TR_MATH_ABI(tr_quant_cover_info, TR_QUANT_ABI);

_Static_assert(sizeof(tr_quant_counts) == 48, "tr_quant_counts ABI");
_Static_assert(sizeof(tr_quant_groups_info) == 128, "tr_quant_groups_info ABI");
_Static_assert(sizeof(tr_quant_keys) == 32, "tr_quant_keys ABI");
_Static_assert(sizeof(tr_quant_dict_info) == 64, "tr_quant_dict_info ABI");
_Static_assert(sizeof(tr_quant_cover_arrays) == 104, "tr_quant_cover_arrays ABI");
_Static_assert(sizeof(tr_quant_cover_info) == 184, "tr_quant_cover_info ABI");
_Static_assert(sizeof(tr_bool_condition) == 48 && sizeof(tr_bool_node) == 16, "BOOL-001 layouts in the identity");

#define GROUPS_MAGIC UINT32_C(0x51475250)                 /* "QGRP" */
#define DICT_MAGIC UINT32_C(0x51444354)                   /* "QDCT" */
#define COVER_MAGIC UINT32_C(0x51434f56)                  /* "QCOV" */
#define TAG_GROUPS UINT64_C(0x31535052474e5451)           /* "QTNGRPS1" */
#define TAG_GROUPS_MERGE UINT64_C(0x31474d52474e5451)     /* "QTNGRMG1" */
#define TAG_DICT UINT64_C(0x3154434944544e51)             /* "QNTDICT1" */
#define TAG_COVER UINT64_C(0x3152564f43544e51)            /* "QNTCOVR1" */
#define TAG_COVER_MERGE UINT64_C(0x31474d564f43544e)      /* "NTCOVMG1" */
#define TAG_REQUIRED UINT64_C(0x3151455254544e51)         /* "QNTTREQ1" */

/* ------------------------------------------------------------------ common */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int popc(uint64_t x) { return __builtin_popcountll(x); }
static int ctz(uint64_t x) { return __builtin_ctzll(x); }

/* An identity byte string, written by one function in three modes: counting (p and expect NULL), writing (p) and
 * comparing with stored bytes (expect, limit): the same sequence gives the size, the bytes and the equality. */
typedef struct { uint8_t *p; const uint8_t *expect; uint64_t n, limit; int differ; } idbuf;
static void id_bytes(idbuf *b, const void *src, uint64_t len) {
    if (b->expect) {
        if (!b->differ && len && (b->n + len > b->limit || memcmp(b->expect + b->n, src, (size_t)len))) b->differ = 1;
    } else if (b->p && len) memcpy(b->p + b->n, src, (size_t)len);
    b->n += len;
}
static void id_word(idbuf *b, uint64_t v) { id_bytes(b, &v, 8); }
/* the FND-004 descriptor of a scope (TR_SCOPE_KEY_ALL), length first; a missing scope is the length 0 */
static int32_t id_scope(idbuf *b, const tr_scope *s) {
    if (!s) { id_word(b, 0); return TR_MATH_OK; }
    uint64_t n = tr_scope_key_size(s, TR_SCOPE_KEY_ALL);
    if (!n) return TR_MATH_INVALID;
    id_word(b, n);
    if (b->expect) {
        if (!b->differ && (b->n + n > b->limit || tr_scope_key_equal(s, TR_SCOPE_KEY_ALL, b->expect + b->n, n) != 1))
            b->differ = 1;
    } else if (b->p) {
        int32_t rc = tr_scope_key_write(s, TR_SCOPE_KEY_ALL, b->p + b->n, n);
        if (rc) return rc;
    }
    b->n += n;
    return TR_MATH_OK;
}
static idbuf id_counter(void) { idbuf b; memset(&b, 0, sizeof b); return b; }
static idbuf id_writer(uint8_t *p) { idbuf b = id_counter(); b.p = p; return b; }
static idbuf id_comparer(const uint8_t *expect, uint64_t limit) { idbuf b = id_counter(); b.expect = expect; b.limit = limit; return b; }

/* 128-bit digest of byte strings (two multiply-xorshift lanes): a pre-filter, never an identity */
typedef struct { uint64_t h0, h1, word, len; int nbytes; } hasher;
static void h_mix(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_init(hasher *h) { h->h0 = UINT64_C(0x243F6A8885A308D3); h->h1 = UINT64_C(0x13198A2E03707344); h->word = h->len = 0; h->nbytes = 0; }
static void h_bytes(hasher *h, const uint8_t *p, uint64_t len) {
    h->len += len;
    while (len && h->nbytes) {
        h->word |= (uint64_t)*p++ << (8 * h->nbytes); len--;
        if (++h->nbytes == 8) { h_mix(h, h->word); h->word = 0; h->nbytes = 0; }
    }
    for (; len >= 8; len -= 8, p += 8) { uint64_t w; memcpy(&w, p, 8); h_mix(h, w); }
    while (len) { h->word |= (uint64_t)*p++ << (8 * h->nbytes); h->nbytes++; len--; }
}
static void h_final(hasher *h, uint64_t out[2]) {
    if (h->nbytes) h_mix(h, h->word);
    h_mix(h, h->len);
    out[0] = h->h0 ^ (h->h1 >> 17); out[1] = h->h1 ^ (h->h0 << 13);
}

/* a stable merge sort of uint32 items by cmp(ctx, a, b) (< 0: a before b); tmp holds n items */
typedef int (*cmp_fn)(const void *ctx, uint32_t a, uint32_t b);
static void msort_u32(uint32_t *a, uint32_t *tmp, uint64_t n, cmp_fn cmp, const void *ctx) {
    for (uint64_t width = 1; width < n; width *= 2) {
        for (uint64_t lo = 0; lo < n; lo += 2 * width) {
            uint64_t mid = lo + width < n ? lo + width : n, hi = lo + 2 * width < n ? lo + 2 * width : n;
            uint64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) tmp[k++] = cmp(ctx, a[j], a[i]) < 0 ? a[j++] : a[i++];
            while (i < mid) tmp[k++] = a[i++];
            while (j < hi) tmp[k++] = a[j++];
        }
        memcpy(a, tmp, (size_t)n * 4);
    }
}
static int cmp_value(const void *ctx, uint32_t a, uint32_t b) { (void)ctx; return a < b ? -1 : a > b; }
/* in-place heap sort of uint32 values (the per-group code lists of SETS) */
static void sift_down(uint32_t *a, uint64_t i, uint64_t n) {
    for (;;) {
        uint64_t l = 2 * i + 1, m = i;
        if (l < n && a[l] > a[m]) m = l;
        if (l + 1 < n && a[l + 1] > a[m]) m = l + 1;
        if (m == i) return;
        uint32_t t = a[i]; a[i] = a[m]; a[m] = t;
        i = m;
    }
}
static void heap_sort(uint32_t *a, uint64_t n) {
    if (n < 2) return;
    for (uint64_t i = n / 2; i-- > 0;) sift_down(a, i, n);
    for (uint64_t end = n - 1; end > 0; end--) {
        uint32_t t = a[0]; a[0] = a[end]; a[end] = t;
        sift_down(a, 0, end);
    }
}
static int bsearch_u32(const uint32_t *a, uint64_t n, uint32_t x) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (a[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo < n && a[lo] == x;
}
static int add_u64(uint64_t a, uint64_t b, uint64_t *r) { return !__builtin_add_overflow(a, b, r); }

/* the selected parent rows per group (ids < the parent groups <= the child groups) */
static void count_parents(const tr_scope_view *pv, uint64_t *parents) {
    const uint64_t nw = (pv->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++)
        for (uint64_t bits = pv->selection[w]; bits; bits &= bits - 1) {
            uint64_t r = w * 64 + (uint64_t)ctz(bits);
            parents[pv->groups ? pv->groups[r] : 0]++;
        }
}
/* the domain and parents arguments: 0, TR_MATH_INVALID or TR_MATH_DOMAIN_ERROR (more parent groups than child groups) */
static int32_t domain_check(int32_t domain, const tr_scope *parents, const tr_scope_view *cv, tr_scope_view *pv) {
    if (domain != TR_QUANT_GROUPED && domain != TR_QUANT_IDS && domain != TR_QUANT_PARENTS) return TR_MATH_INVALID;
    if ((domain == TR_QUANT_PARENTS) != (parents != NULL)) return TR_MATH_INVALID;
    if (parents) {
        if (tr_scope_get_view(parents, pv)) return TR_MATH_INVALID;
        if (pv->ngroups > cv->ngroups) return TR_MATH_DOMAIN_ERROR;
    }
    return TR_MATH_OK;
}
/* parents read back from a device: their total and zeros past the parent ids (or everywhere without parents) */
static int32_t parents_consistent(const uint64_t *par, uint64_t G, const tr_scope_view *pv, int has) {
    __int128 total = 0;
    for (uint64_t g = 0; g < G; g++) {
        if (par[g] && (!has || g >= pv->ngroups)) return TR_MATH_INVALID;
        total += par[g];
    }
    return total == (__int128)(has ? pv->selected : 0) ? TR_MATH_OK : TR_MATH_INVALID;
}
static uint8_t *copy_bytes(const uint8_t *src, uint64_t n) {
    uint8_t *p = zalloc(n);
    if (p && n) memcpy(p, src, (size_t)n);
    return p;
}

/* ================================================================== QUANT-001 */
struct tr_quant_groups_s {
    uint32_t magic;
    int32_t domain, k, nnodes, parts, reserved;
    uint64_t ngroups;
    uint64_t *arrays;            /* 5 G words: rows, qualifying, failing, unknown, parents */
    uint8_t *identity, *predicate, *parents_desc;
    uint64_t identity_bytes, predicate_bytes, parents_bytes;
    uint64_t bytes;
    tr_quant_groups_info info;   /* totals, filled when the state is finished */
};

static int groups_ok(const tr_quant_groups *s) { return s && s->magic == GROUPS_MAGIC; }
static void groups_release(tr_quant_groups *s) {
    if (!s) return;
    free(s->arrays); free(s->identity); free(s->predicate); free(s->parents_desc);
    s->magic = 0;
    free(s);
}
void tr_quant_groups_free(tr_quant_groups *s) { if (groups_ok(s)) groups_release(s); }
static tr_quant_groups *groups_fail(tr_quant_groups *s, int32_t *error, int32_t code) {
    groups_release(s); set_error(error, code); return NULL;
}

typedef struct {
    tr_scope_view cv, pv;
    tr_bool_program_view prog;
    uint64_t reads, identity_bytes, predicate_bytes, parents_bytes;
    int has_parents;
} groups_plan;

/* the predicate descriptor: k, the conditions, the node count, the root and the nodes */
static void id_predicate(idbuf *b, const tr_bool_condition *conds, int32_t k, const tr_bool_program_view *p) {
    id_word(b, (uint64_t)k);
    id_bytes(b, conds, (uint64_t)k * sizeof(tr_bool_condition));
    id_word(b, (uint64_t)p->nnodes);
    id_word(b, (uint64_t)p->root);
    id_bytes(b, p->nodes, (uint64_t)p->nnodes * sizeof(tr_bool_node));
}
static int32_t groups_identity(idbuf *b, const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                               const tr_bool_program_view *p, int32_t domain, const tr_scope *parents) {
    id_word(b, TAG_GROUPS);
    id_word(b, (uint64_t)domain);
    int32_t rc = id_scope(b, child);
    if (!rc) rc = id_scope(b, parents);
    if (rc) return rc;
    id_predicate(b, conds, k, p);
    return TR_MATH_OK;
}

static int32_t groups_check(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                            const tr_bool_program *pred, int32_t domain, const tr_scope *parents, groups_plan *P) {
    memset(P, 0, sizeof *P);
    if (!child || tr_scope_get_view(child, &P->cv)) return TR_MATH_INVALID;
    if (!pred || tr_bool_program_get_view(pred, &P->prog)) return TR_MATH_INVALID;
    if (k < 0 || (k && !conds)) return TR_MATH_INVALID;
    if (P->prog.k != k) return TR_MATH_DOMAIN_ERROR;
    for (int32_t i = 0; i < k; i++) if (tr_bool_condition_check(&P->cv, &conds[i])) return TR_MATH_INVALID;
    int32_t rc = domain_check(domain, parents, &P->cv, &P->pv);
    if (rc) return rc;
    P->has_parents = parents != NULL;
    P->reads = P->prog.deps[P->prog.root];
    idbuf c = id_counter();
    rc = groups_identity(&c, child, conds, k, &P->prog, domain, parents);
    if (rc) return rc;
    P->identity_bytes = c.n;
    idbuf q = id_counter();
    id_predicate(&q, conds, k, &P->prog);
    P->predicate_bytes = q.n;
    P->parents_bytes = parents ? tr_scope_key_size(parents, TR_SCOPE_KEY_ALL) : 0;
    return TR_MATH_OK;
}
static uint64_t groups_object_bytes(uint64_t G, uint64_t identity, uint64_t predicate, uint64_t parents) {
    uint64_t b = tr_budget_add(sizeof(tr_quant_groups), tr_budget_mul(G, 40));
    return tr_budget_add(tr_budget_add(b, identity), tr_budget_add(predicate, parents));
}
static void groups_plan_need(const groups_plan *P, const tr_bool_condition *conds, int32_t k, int32_t target,
                             tr_math_need *need) {
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)P->prog.nnodes;
    need->need[TR_BUDGET_STATES] = P->cv.ngroups;
    need->need[TR_BUDGET_OUTPUT] = P->cv.ngroups;
    uint64_t bytes = groups_object_bytes(P->cv.ngroups, P->identity_bytes, P->predicate_bytes, P->parents_bytes);
    if (target == TR_QUANT_NEED_CPU) bytes = tr_budget_add(bytes, tr_budget_mul(tr_quant_item_words(k, P->prog.nnodes), 8));
    else if (target == TR_QUANT_NEED_DEVICE)
        bytes = tr_quant_groups_device_bytes(&P->cv, conds, k, &P->prog, P->has_parents ? &P->pv : NULL);
    need->need[TR_BUDGET_BYTES] = bytes;
}

int32_t tr_quant_groups_need(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                             const tr_bool_program *pred, int32_t domain, const tr_scope *parents, int32_t target,
                             tr_math_need *need) {
    if (!need || target < TR_QUANT_NEED_CPU || target > TR_QUANT_NEED_ADOPT) return TR_MATH_INVALID;
    groups_plan P;
    int32_t rc = groups_check(child, conds, k, pred, domain, parents, &P);
    if (rc) return rc;
    groups_plan_need(&P, conds, k, target, need);
    return TR_MATH_OK;
}

/* the object with its arrays and identity (written), or NULL (*rc set) */
static tr_quant_groups *groups_alloc(const groups_plan *P, const tr_scope *child, const tr_bool_condition *conds,
                                     int32_t k, int32_t domain, const tr_scope *parents, int32_t *rc) {
    tr_quant_groups *s = zalloc(sizeof *s);
    if (!s) { *rc = TR_MATH_MEMORY; return NULL; }
    s->domain = domain; s->k = k; s->nnodes = P->prog.nnodes; s->parts = 1;
    s->ngroups = P->cv.ngroups;
    s->arrays = zalloc(tr_budget_mul(s->ngroups, 40));
    s->identity = zalloc(P->identity_bytes);
    s->predicate = zalloc(P->predicate_bytes);
    s->parents_desc = zalloc(P->parents_bytes);
    if (!s->arrays || !s->identity || !s->predicate || !s->parents_desc) { groups_release(s); *rc = TR_MATH_MEMORY; return NULL; }
    s->identity_bytes = P->identity_bytes; s->predicate_bytes = P->predicate_bytes; s->parents_bytes = P->parents_bytes;
    idbuf w = id_writer(s->identity);
    *rc = groups_identity(&w, child, conds, k, &P->prog, domain, parents);
    idbuf q = id_writer(s->predicate);
    id_predicate(&q, conds, k, &P->prog);
    if (!*rc && parents) *rc = tr_scope_key_write(parents, TR_SCOPE_KEY_ALL, s->parents_desc, s->parents_bytes);
    if (*rc) { groups_release(s); return NULL; }
    return s;
}
/* totals, digest and the bytes retained; the state is valid from here on */
static void groups_finish(tr_quant_groups *s) {
    const uint64_t G = s->ngroups;
    const uint64_t *rows = s->arrays, *q = rows + G, *f = q + G, *u = f + G, *par = u + G;
    tr_quant_groups_info *I = &s->info;
    memset(I, 0, sizeof *I);
    I->ngroups = G; I->domain = s->domain; I->k = s->k; I->nnodes = s->nnodes; I->parts = s->parts;
    for (uint64_t g = 0; g < G; g++) {
        int present = tr_quant_present(s->domain, rows[g], par[g]);
        if (present) { I->present++; if (!rows[g]) I->empty++; }
        else { I->absent++; I->orphans += rows[g]; }
        I->rows += rows[g]; I->qualifying += q[g]; I->failing += f[g]; I->unknown += u[g]; I->parent_rows += par[g];
    }
    hasher h; h_init(&h);
    h_bytes(&h, s->identity, s->identity_bytes);
    h_final(&h, I->digest);
    s->bytes = groups_object_bytes(G, s->identity_bytes, s->predicate_bytes, s->parents_bytes);
    I->bytes = s->bytes; I->identity_bytes = s->identity_bytes;
    s->magic = GROUPS_MAGIC;
}

/* the one pass over the selection words (see Execution) */
static void groups_scan(const tr_scope_view *v, const tr_bool_condition *conds, const tr_bool_program_view *p,
                        uint64_t reads, uint64_t *scratch, uint64_t *rows, uint64_t *q, uint64_t *f, uint64_t *u) {
    const uint64_t nw = (v->rows + 63) / 64, k = (uint64_t)p->k;
    uint64_t *cv = scratch, *ck = scratch + k, *val = scratch + 2 * k;
    for (uint64_t w = 0; w < nw; w++) {
        const uint64_t sel = v->selection[w];
        if (!sel) continue;
        uint64_t T, F, U;
        tr_quant_predicate_word(v, conds, p, reads, w, cv, ck, val, &T, &F, &U);
        int uniform;
        uint32_t g = tr_quant_word_group(v->groups, w, sel, &uniform);
        if (uniform) {
            rows[g] += (uint64_t)popc(sel); q[g] += (uint64_t)popc(T); f[g] += (uint64_t)popc(F); u[g] += (uint64_t)popc(U);
            continue;
        }
        for (uint64_t bits = sel; bits; bits &= bits - 1) {
            const int b = ctz(bits);
            const uint64_t bit = (uint64_t)1 << b;
            const uint32_t h = v->groups[w * 64 + (uint64_t)b];
            rows[h]++;
            if (T & bit) q[h]++; else if (F & bit) f[h]++; else u[h]++;
        }
    }
}

tr_quant_groups *tr_quant_groups_build(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                                       const tr_bool_program *pred, int32_t domain, const tr_scope *parents,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    groups_plan P;
    int32_t rc = groups_check(child, conds, k, pred, domain, parents, &P);
    if (rc) return groups_fail(NULL, error, rc);
    tr_math_need need;
    groups_plan_need(&P, conds, k, TR_QUANT_NEED_CPU, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return groups_fail(NULL, error, rc);
    tr_quant_groups *s = groups_alloc(&P, child, conds, k, domain, parents, &rc);
    if (!s) return groups_fail(NULL, error, rc);
    uint64_t *scratch = zalloc(tr_budget_mul(tr_quant_item_words(k, P.prog.nnodes), 8));
    if (!scratch) return groups_fail(s, error, TR_MATH_MEMORY);
    const uint64_t G = s->ngroups;
    uint64_t *rows = s->arrays;
    groups_scan(&P.cv, conds, &P.prog, P.reads, scratch, rows, rows + G, rows + 2 * G, rows + 3 * G);
    free(scratch);
    if (P.has_parents) count_parents(&P.pv, rows + 4 * G);
    groups_finish(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_quant_groups *tr_quant_groups_adopt(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                                       const tr_bool_program *pred, int32_t domain, const tr_scope *parents,
                                       const tr_quant_counts *counts, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    groups_plan P;
    int32_t rc = groups_check(child, conds, k, pred, domain, parents, &P);
    if (rc) return groups_fail(NULL, error, rc);
    tr_math_need need;
    groups_plan_need(&P, conds, k, TR_QUANT_NEED_ADOPT, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return groups_fail(NULL, error, rc);
    const uint64_t G = P.cv.ngroups;
    if (!counts || counts->ngroups != G || !counts->rows || !counts->qualifying || !counts->failing ||
        !counts->unknown || !counts->parents) return groups_fail(NULL, error, TR_MATH_INVALID);
    __int128 total = 0;
    for (uint64_t g = 0; g < G; g++) {
        __int128 parts = (__int128)counts->qualifying[g] + counts->failing[g] + counts->unknown[g];
        if (parts != (__int128)counts->rows[g]) return groups_fail(NULL, error, TR_MATH_INVALID);
        total += counts->rows[g];
    }
    if (total != (__int128)P.cv.selected) return groups_fail(NULL, error, TR_MATH_INVALID);
    rc = parents_consistent(counts->parents, G, &P.pv, P.has_parents);
    if (rc) return groups_fail(NULL, error, rc);
    tr_quant_groups *s = groups_alloc(&P, child, conds, k, domain, parents, &rc);
    if (!s) return groups_fail(NULL, error, rc);
    const uint64_t *src[5] = {counts->rows, counts->qualifying, counts->failing, counts->unknown, counts->parents};
    for (int a = 0; a < 5; a++) if (G) memcpy(s->arrays + (uint64_t)a * G, src[a], (size_t)(G * 8));
    groups_finish(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_quant_groups *tr_quant_groups_merge(const tr_quant_groups *a, const tr_quant_groups *b,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!groups_ok(a) || !groups_ok(b)) return groups_fail(NULL, error, TR_MATH_INVALID);
    if (a->ngroups != b->ngroups || a->domain != b->domain || a->predicate_bytes != b->predicate_bytes ||
        memcmp(a->predicate, b->predicate, (size_t)a->predicate_bytes) || a->parents_bytes != b->parents_bytes ||
        (a->parents_bytes && memcmp(a->parents_desc, b->parents_desc, (size_t)a->parents_bytes)))
        return groups_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const uint64_t G = a->ngroups;
    const uint64_t id = tr_budget_add(tr_budget_add(24, a->identity_bytes), b->identity_bytes);
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)a->k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)a->nnodes;
    need.need[TR_BUDGET_STATES] = G;
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_BYTES] = groups_object_bytes(G, id, a->predicate_bytes, a->parents_bytes);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return groups_fail(NULL, error, rc);
    tr_quant_groups *s = zalloc(sizeof *s);
    if (!s) return groups_fail(NULL, error, TR_MATH_MEMORY);
    s->domain = a->domain; s->k = a->k; s->nnodes = a->nnodes; s->ngroups = G;
    uint64_t parts;
    if (!add_u64((uint64_t)a->parts, (uint64_t)b->parts, &parts) || parts > INT32_MAX)
        return groups_fail(s, error, TR_MATH_OVERFLOW);
    s->parts = (int32_t)parts;
    s->arrays = zalloc(tr_budget_mul(G, 40));
    s->identity = zalloc(id);
    s->predicate = copy_bytes(a->predicate, a->predicate_bytes);
    s->parents_desc = copy_bytes(a->parents_desc, a->parents_bytes);
    if (!s->arrays || !s->identity || !s->predicate || !s->parents_desc) return groups_fail(s, error, TR_MATH_MEMORY);
    s->identity_bytes = id; s->predicate_bytes = a->predicate_bytes; s->parents_bytes = a->parents_bytes;
    idbuf w = id_writer(s->identity);
    id_word(&w, TAG_GROUPS_MERGE);
    id_word(&w, a->identity_bytes); id_bytes(&w, a->identity, a->identity_bytes);
    id_word(&w, b->identity_bytes); id_bytes(&w, b->identity, b->identity_bytes);
    for (uint64_t i = 0; i < 4 * G; i++)
        if (!add_u64(a->arrays[i], b->arrays[i], &s->arrays[i])) return groups_fail(s, error, TR_MATH_OVERFLOW);
    if (G) memcpy(s->arrays + 4 * G, a->arrays + 4 * G, (size_t)(G * 8));     /* the same parents */
    groups_finish(s);
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_quant_groups_get_info(const tr_quant_groups *s, tr_quant_groups_info *out) {
    if (!groups_ok(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_quant_groups_get_counts(const tr_quant_groups *s, tr_quant_counts *out) {
    if (!groups_ok(s) || !out) return TR_MATH_INVALID;
    const uint64_t G = s->ngroups;
    out->ngroups = G;
    out->rows = s->arrays; out->qualifying = s->arrays + G; out->failing = s->arrays + 2 * G;
    out->unknown = s->arrays + 3 * G; out->parents = s->arrays + 4 * G;
    return TR_MATH_OK;
}
static int quantifier_ok(int32_t q) { return q >= TR_QUANT_SOME && q <= TR_QUANT_BOOL_AND; }
int32_t tr_quant_groups_answer(const tr_quant_groups *s, int32_t q, uint8_t *out, uint64_t capacity) {
    if (!groups_ok(s) || !quantifier_ok(q) || (s->ngroups && (!out || capacity < s->ngroups))) return TR_MATH_INVALID;
    const uint64_t G = s->ngroups, *rows = s->arrays;
    for (uint64_t g = 0; g < G; g++)
        out[g] = (uint8_t)tr_quant_answer_code(q, tr_quant_present(s->domain, rows[g], rows[4 * G + g]), rows[G + g],
                                               rows[2 * G + g], rows[3 * G + g]);
    return TR_MATH_OK;
}
int32_t tr_quant_groups_tally(const tr_quant_groups *s, int32_t q, uint64_t out[4]) {
    if (!groups_ok(s) || !quantifier_ok(q) || !out) return TR_MATH_INVALID;
    const uint64_t G = s->ngroups, *rows = s->arrays;
    out[0] = out[1] = out[2] = out[3] = 0;
    for (uint64_t g = 0; g < G; g++)
        out[tr_quant_answer_code(q, tr_quant_present(s->domain, rows[g], rows[4 * G + g]), rows[G + g], rows[2 * G + g],
                                 rows[3 * G + g])]++;
    return TR_MATH_OK;
}
uint64_t tr_quant_groups_identity_size(const tr_quant_groups *s) { return groups_ok(s) ? s->identity_bytes : 0; }
int32_t tr_quant_groups_identity(const tr_quant_groups *s, uint8_t *out, uint64_t capacity) {
    if (!groups_ok(s) || !out || capacity < s->identity_bytes) return TR_MATH_INVALID;
    memcpy(out, s->identity, (size_t)s->identity_bytes);
    return TR_MATH_OK;
}
int32_t tr_quant_groups_same(const tr_quant_groups *a, const tr_quant_groups *b) {
    if (!groups_ok(a) || !groups_ok(b)) return TR_MATH_INVALID;
    return a->identity_bytes == b->identity_bytes && !memcmp(a->identity, b->identity, (size_t)a->identity_bytes);
}
int32_t tr_quant_groups_matches(const tr_quant_groups *s, const tr_scope *child, const tr_bool_condition *conds,
                                int32_t k, const tr_bool_program *pred, int32_t domain, const tr_scope *parents) {
    if (!groups_ok(s)) return TR_MATH_INVALID;
    groups_plan P;
    int32_t rc = groups_check(child, conds, k, pred, domain, parents, &P);
    if (rc) return rc;
    idbuf c = id_comparer(s->identity, s->identity_bytes);
    rc = groups_identity(&c, child, conds, k, &P.prog, domain, parents);
    if (rc) return rc;
    return !c.differ && c.n == s->identity_bytes;
}

/* ================================================================== QUANT-002: dictionaries */
struct tr_quant_dict_s {
    uint32_t magic;
    int32_t kind, scale;
    uint32_t null_code;
    uint64_t size;
    int64_t offset;
    uint8_t *keys;          /* KEYS: key_bytes */
    uint64_t *offsets;      /* KEYS: size + 1 */
    uint32_t *sorted;       /* KEYS: the codes but the NULL one, by key */
    uint64_t nsorted, key_bytes, bytes;
    uint64_t digest[2];
};

static int dict_ok(const tr_quant_dict *d) { return d && d->magic == DICT_MAGIC; }
static void dict_release(tr_quant_dict *d) {
    if (!d) return;
    free(d->keys); free(d->offsets); free(d->sorted);
    d->magic = 0;
    free(d);
}
void tr_quant_dict_free(tr_quant_dict *d) { if (dict_ok(d)) dict_release(d); }
static tr_quant_dict *dict_fail(tr_quant_dict *d, int32_t *error, int32_t code) {
    dict_release(d); set_error(error, code); return NULL;
}
/* the dictionary's descriptor */
static void id_dict(idbuf *b, const tr_quant_dict *d) {
    id_word(b, TAG_DICT);
    id_word(b, (uint64_t)d->kind); id_word(b, d->size); id_word(b, (uint64_t)d->null_code);
    id_word(b, (uint64_t)d->offset); id_word(b, (uint64_t)(int64_t)d->scale);
    if (d->kind == TR_QUANT_DICT_KEYS) {
        id_word(b, d->key_bytes);
        id_bytes(b, d->offsets, (d->size + 1) * 8);
        id_bytes(b, d->keys, d->key_bytes);
    }
}
static void dict_digest(tr_quant_dict *d) {           /* the digest of id_dict's byte string */
    hasher h; h_init(&h);
    uint64_t head[6] = {TAG_DICT, (uint64_t)d->kind, d->size, (uint64_t)d->null_code, (uint64_t)d->offset,
                        (uint64_t)(int64_t)d->scale};
    h_bytes(&h, (const uint8_t *)head, sizeof head);
    if (d->kind == TR_QUANT_DICT_KEYS) {
        h_bytes(&h, (const uint8_t *)&d->key_bytes, 8);
        h_bytes(&h, (const uint8_t *)d->offsets, (d->size + 1) * 8);
        h_bytes(&h, d->keys, d->key_bytes);
    }
    h_final(&h, d->digest);
}
/* a list of keys is well formed: 0 or TR_MATH_INVALID */
static int32_t keys_check(const tr_quant_keys *k) {
    if (!k || k->reserved || (k->null_key != 0 && k->null_key != 1) || (k->n && !k->offsets)) return TR_MATH_INVALID;
    if (!k->n) return TR_MATH_OK;
    if (k->n > (UINT64_MAX >> 4) || k->offsets[0] != 0) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < k->n; i++) if (k->offsets[i + 1] < k->offsets[i]) return TR_MATH_INVALID;
    if (k->offsets[k->n] && !k->bytes) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static uint64_t keys_total(const tr_quant_keys *k) { return k->n ? k->offsets[k->n] : 0; }

typedef struct { const uint8_t *bytes; const uint64_t *offsets; } key_list;
static int key_cmp_raw(const uint8_t *a, uint64_t na, const uint8_t *b, uint64_t nb) {
    uint64_t n = na < nb ? na : nb;
    int c = n ? memcmp(a, b, (size_t)n) : 0;
    if (c) return c < 0 ? -1 : 1;
    return na < nb ? -1 : na > nb;
}
static int key_cmp(const void *ctx, uint32_t a, uint32_t b) {
    const key_list *L = ctx;
    return key_cmp_raw(L->bytes + L->offsets[a], L->offsets[a + 1] - L->offsets[a], L->bytes + L->offsets[b],
                       L->offsets[b + 1] - L->offsets[b]);
}
static uint64_t dict_object_bytes(int32_t kind, uint64_t size, uint64_t key_bytes) {
    uint64_t b = sizeof(tr_quant_dict);
    if (kind == TR_QUANT_DICT_KEYS)
        b = tr_budget_add(b, tr_budget_add(key_bytes, tr_budget_add(tr_budget_mul(size + 1, 8), tr_budget_mul(size, 4))));
    return b;
}
int32_t tr_quant_dict_need(int32_t kind, uint64_t size, const tr_quant_keys *entries, tr_math_need *need) {
    if (!need || (kind != TR_QUANT_DICT_CODES && kind != TR_QUANT_DICT_KEYS) || size < 1) return TR_MATH_INVALID;
    uint64_t key_bytes = 0;
    if (kind == TR_QUANT_DICT_KEYS) {
        if (keys_check(entries) || entries->n != size || entries->null_key) return TR_MATH_INVALID;
        key_bytes = keys_total(entries);
    }
    zero_need(need);
    need->need[TR_BUDGET_SUPPORT] = size;
    /* KEYS: + the merge-sort scratch of the index (4 per code) */
    need->need[TR_BUDGET_BYTES] = kind == TR_QUANT_DICT_KEYS
        ? tr_budget_add(dict_object_bytes(kind, size, key_bytes), tr_budget_mul(size, 4)) : sizeof(tr_quant_dict);
    return TR_MATH_OK;
}
static int32_t dict_start(int32_t kind, uint64_t size, uint32_t null_code, const tr_quant_keys *entries,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_quant_dict_need(kind, size, entries, &need);
    if (rc) return rc;
    if (null_code != TR_QUANT_NO_NULL && null_code >= size) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (size > TR_SCOPE_MAX_GROUPS) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}

tr_quant_dict *tr_quant_dict_codes(uint64_t size, uint32_t null_code, int64_t offset, int32_t scale,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (scale < 0 || scale > TR_SCOPE_MAX_SCALE) { tr_budget_none(refusal); return dict_fail(NULL, error, TR_MATH_INVALID); }
    int32_t rc = dict_start(TR_QUANT_DICT_CODES, size, null_code, NULL, budget, refusal);
    if (rc) return dict_fail(NULL, error, rc);
    if ((__int128)offset + (__int128)(size - 1) > INT64_MAX) return dict_fail(NULL, error, TR_MATH_OVERFLOW);
    tr_quant_dict *d = zalloc(sizeof *d);
    if (!d) return dict_fail(NULL, error, TR_MATH_MEMORY);
    d->kind = TR_QUANT_DICT_CODES; d->scale = scale; d->null_code = null_code; d->size = size; d->offset = offset;
    d->bytes = sizeof *d;
    dict_digest(d);
    d->magic = DICT_MAGIC;
    set_error(error, TR_MATH_OK);
    return d;
}

tr_quant_dict *tr_quant_dict_keys(uint64_t size, uint32_t null_code, const tr_quant_keys *entries,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    int32_t rc = dict_start(TR_QUANT_DICT_KEYS, size, null_code, entries, budget, refusal);
    if (rc) return dict_fail(NULL, error, rc);
    if (null_code != TR_QUANT_NO_NULL && entries->offsets[null_code + 1] != entries->offsets[null_code])
        return dict_fail(NULL, error, TR_MATH_INVALID);                       /* the NULL code has no key */
    tr_quant_dict *d = zalloc(sizeof *d);
    if (!d) return dict_fail(NULL, error, TR_MATH_MEMORY);
    d->kind = TR_QUANT_DICT_KEYS; d->null_code = null_code; d->size = size;
    d->key_bytes = keys_total(entries);
    d->keys = copy_bytes(entries->bytes, d->key_bytes);
    d->offsets = zalloc((size + 1) * 8);
    d->sorted = zalloc(size * 4);
    uint32_t *tmp = zalloc(size * 4);
    if (!d->keys || !d->offsets || !d->sorted || !tmp) { free(tmp); return dict_fail(d, error, TR_MATH_MEMORY); }
    memcpy(d->offsets, entries->offsets, (size_t)((size + 1) * 8));
    for (uint64_t c = 0; c < size; c++) if ((uint32_t)c != null_code || null_code == TR_QUANT_NO_NULL) d->sorted[d->nsorted++] = (uint32_t)c;
    key_list L = {d->keys, d->offsets};
    msort_u32(d->sorted, tmp, d->nsorted, key_cmp, &L);
    free(tmp);
    for (uint64_t i = 1; i < d->nsorted; i++)
        if (key_cmp(&L, d->sorted[i - 1], d->sorted[i]) == 0) return dict_fail(d, error, TR_MATH_INVALID);   /* duplicate */
    d->bytes = dict_object_bytes(d->kind, size, d->key_bytes);
    dict_digest(d);
    d->magic = DICT_MAGIC;
    set_error(error, TR_MATH_OK);
    return d;
}
/* an independent copy (owned by a coverage state) */
static tr_quant_dict *dict_clone(const tr_quant_dict *s) {
    tr_quant_dict *d = zalloc(sizeof *d);
    if (!d) return NULL;
    *d = *s;
    d->keys = NULL; d->offsets = NULL; d->sorted = NULL; d->magic = 0;
    if (s->kind == TR_QUANT_DICT_KEYS) {
        d->keys = copy_bytes(s->keys, s->key_bytes);
        d->offsets = zalloc((s->size + 1) * 8);
        d->sorted = zalloc(s->size * 4);
        if (!d->keys || !d->offsets || !d->sorted) { dict_release(d); return NULL; }
        memcpy(d->offsets, s->offsets, (size_t)((s->size + 1) * 8));
        if (s->nsorted) memcpy(d->sorted, s->sorted, (size_t)(s->nsorted * 4));
    }
    d->magic = DICT_MAGIC;
    return d;
}
int32_t tr_quant_dict_get_info(const tr_quant_dict *d, tr_quant_dict_info *out) {
    if (!dict_ok(d) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->size = d->size; out->bytes = d->bytes; out->key_bytes = d->key_bytes;
    out->digest[0] = d->digest[0]; out->digest[1] = d->digest[1];
    out->offset = d->offset; out->kind = d->kind; out->scale = d->scale; out->null_code = d->null_code;
    return TR_MATH_OK;
}
static int dict_lookup(const tr_quant_dict *d, const uint8_t *key, uint64_t len, uint32_t *code) {
    if (d->kind == TR_QUANT_DICT_CODES) {
        if (len != 8) return 0;
        int64_t v;
        memcpy(&v, key, 8);
        __int128 c = (__int128)v - d->offset;
        if (c < 0 || c >= (__int128)d->size || (uint64_t)c == (uint64_t)d->null_code) return 0;
        *code = (uint32_t)c;
        return 1;
    }
    uint64_t lo = 0, hi = d->nsorted;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint32_t c = d->sorted[mid];
        int r = key_cmp_raw(d->keys + d->offsets[c], d->offsets[c + 1] - d->offsets[c], key, len);
        if (r == 0) { *code = c; return 1; }
        if (r < 0) lo = mid + 1; else hi = mid;
    }
    return 0;
}
int32_t tr_quant_dict_find(const tr_quant_dict *d, const uint8_t *key, uint64_t len, uint32_t *code) {
    if (!dict_ok(d) || !code || (len && !key)) return TR_MATH_INVALID;
    return dict_lookup(d, key, len, code);
}
int32_t tr_quant_dict_same(const tr_quant_dict *a, const tr_quant_dict *b) {
    if (!dict_ok(a) || !dict_ok(b)) return TR_MATH_INVALID;
    if (a->kind != b->kind || a->size != b->size || a->null_code != b->null_code || a->offset != b->offset ||
        a->scale != b->scale || a->key_bytes != b->key_bytes) return 0;
    if (a->kind == TR_QUANT_DICT_KEYS && (memcmp(a->offsets, b->offsets, (size_t)((a->size + 1) * 8)) ||
                                          (a->key_bytes && memcmp(a->keys, b->keys, (size_t)a->key_bytes)))) return 0;
    return 1;
}

/* The resolution of a required set against a dictionary: codes[0..*ncodes) the distinct required codes (increasing),
 * outside[0..*noutside) the indices of the distinct keys outside the dictionary (sorted by key). codes, outside and
 * tmp hold n items each. */
static void resolve(const tr_quant_dict *d, const tr_quant_keys *k, uint32_t *codes, uint64_t *ncodes,
                    uint32_t *outside, uint64_t *noutside, uint32_t *tmp) {
    uint64_t nc = 0, no = 0;
    for (uint64_t i = 0; i < k->n; i++) {
        uint32_t c;
        if (dict_lookup(d, k->bytes + k->offsets[i], k->offsets[i + 1] - k->offsets[i], &c)) codes[nc++] = c;
        else outside[no++] = (uint32_t)i;
    }
    msort_u32(codes, tmp, nc, cmp_value, NULL);
    uint64_t u = 0;
    for (uint64_t i = 0; i < nc; i++) if (!u || codes[u - 1] != codes[i]) codes[u++] = codes[i];
    key_list L = {k->bytes, k->offsets};
    msort_u32(outside, tmp, no, key_cmp, &L);
    uint64_t v = 0;
    for (uint64_t i = 0; i < no; i++) if (!v || key_cmp(&L, outside[v - 1], outside[i])) outside[v++] = outside[i];
    *ncodes = u; *noutside = v;
}
int32_t tr_quant_dict_mask(const tr_quant_dict *d, const tr_quant_keys *k, uint64_t *mask, uint64_t words,
                           uint64_t *required_codes, uint64_t *constant_missing, const tr_math_budget *budget,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!dict_ok(d) || keys_check(k) || !required_codes || !constant_missing || (mask && words < (d->size + 63) / 64))
        return TR_MATH_INVALID;
    if (k->n > UINT32_MAX) return TR_MATH_UNSUPPORTED;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(k->n, 12);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint32_t *scratch = zalloc(tr_budget_mul(k->n, 12));
    if (!scratch) return TR_MATH_MEMORY;
    uint64_t nc = 0, no = 0;
    resolve(d, k, scratch, &nc, scratch + k->n, &no, scratch + 2 * k->n);
    if (mask) {
        memset(mask, 0, (size_t)(words * 8));
        for (uint64_t i = 0; i < nc; i++) mask[scratch[i] >> 6] |= (uint64_t)1 << (scratch[i] & 63);
    }
    *required_codes = nc;
    *constant_missing = no + (uint64_t)k->null_key;
    free(scratch);
    return TR_MATH_OK;
}

/* ================================================================== QUANT-002: coverage states */
struct tr_quant_cover_s {
    uint32_t magic;
    int32_t domain, repr, reason, null_required, parts;
    uint32_t null_code, reserved;
    uint64_t G, D, W;
    uint64_t *arrays;            /* 7 G words: rows, nulls, parents, distinct, matched, missing, extra */
    uint64_t *masks;             /* MASKS: G W words */
    uint64_t *offsets;           /* SETS: G + 1 */
    uint32_t *codes;             /* SETS: codes_cap entries (offsets[G] used) */
    uint64_t codes_cap;
    uint32_t *req;               /* the required codes, increasing (req_cap allocated) */
    uint64_t nreq, req_cap;
    uint64_t *req_mask;          /* MASKS: W words */
    uint8_t *outside;            /* the distinct outside keys, sorted, concatenated (outside_cap allocated) */
    uint64_t *outside_off;       /* noutside + 1 (outside_off_cap allocated) */
    uint64_t noutside, outside_bytes, outside_cap, outside_off_cap;
    tr_quant_dict *dict;         /* an owned copy of the dictionary */
    uint8_t *presence, *requirement, *parents_desc;
    uint64_t presence_bytes, requirement_bytes, requirement_cap, parents_bytes;
    double masks_cost, sets_cost;
    uint64_t bytes;
    tr_quant_cover_info info;
};
enum { A_ROWS = 0, A_NULLS = 1, A_PARENTS = 2, A_DISTINCT = 3, A_MATCHED = 4, A_MISSING = 5, A_EXTRA = 6 };

static int cover_ok(const tr_quant_cover *c) { return c && c->magic == COVER_MAGIC; }
static uint64_t *col(const tr_quant_cover *c, int a) { return c->arrays + (uint64_t)a * c->G; }
static void cover_release(tr_quant_cover *c) {
    if (!c) return;
    free(c->arrays); free(c->masks); free(c->offsets); free(c->codes); free(c->req); free(c->req_mask);
    free(c->outside); free(c->outside_off); free(c->presence); free(c->requirement); free(c->parents_desc);
    if (c->dict) dict_release(c->dict);
    c->magic = 0;
    free(c);
}
void tr_quant_cover_free(tr_quant_cover *c) { if (cover_ok(c)) cover_release(c); }
static tr_quant_cover *cover_fail(tr_quant_cover *c, int32_t *error, int32_t code) {
    cover_release(c); set_error(error, code); return NULL;
}

/* 1 when two scopes hold the same rows: the same source (table address, identity token, revision and row count),
 * or two array scopes of the same row count, with identical selection bits */
static int same_rows(const tr_scope *a, const tr_scope *b, const tr_scope_view *va, const tr_scope_view *vb) {
    tr_scope_info ia, ib;
    if (tr_scope_get_info(a, &ia) || tr_scope_get_info(b, &ib)) return 0;
    if (ia.source != ib.source || ia.rows != ib.rows || ia.selected != ib.selected) return 0;
    if (ia.source == TR_SCOPE_TABLE &&
        (ia.table != ib.table || ia.revision != ib.revision || memcmp(ia.identity, ib.identity, 16))) return 0;
    uint64_t nw = (ia.rows + 63) / 64;
    return !nw || !memcmp(va->selection, vb->selection, (size_t)(nw * 8));
}

typedef struct {
    tr_scope_view cv, kv, pv;    /* child, categories, parents */
    uint64_t G, D, W, S;         /* groups, categories, mask words, selected child rows */
    uint64_t nkeys, key_bytes;   /* the required set as given */
    uint64_t presence_bytes, requirement_cap, parents_bytes, dict_bytes;
    int has_parents;
} cover_plan;

static int32_t cover_presence(idbuf *b, const tr_scope *child, const tr_scope *categories, const tr_quant_dict *d,
                              int32_t domain, const tr_scope *parents) {
    id_word(b, TAG_COVER);
    id_word(b, (uint64_t)domain);
    int32_t rc = id_scope(b, child);
    if (!rc) rc = id_scope(b, categories);
    if (!rc) rc = id_scope(b, parents);
    if (rc) return rc;
    id_dict(b, d);
    return TR_MATH_OK;
}
/* the requirement part: the required codes, the outside keys (sorted), the NULL flag */
static void cover_requirement(idbuf *b, const uint32_t *req, uint64_t nreq, const uint8_t *outside,
                              const uint64_t *outside_off, uint64_t noutside, int32_t null_required) {
    id_word(b, TAG_REQUIRED);
    id_word(b, nreq);
    id_bytes(b, req, nreq * 4);
    id_word(b, noutside);
    for (uint64_t i = 0; i < noutside; i++) {
        id_word(b, outside_off[i + 1] - outside_off[i]);
        id_bytes(b, outside + outside_off[i], outside_off[i + 1] - outside_off[i]);
    }
    id_word(b, (uint64_t)null_required);
}
static uint64_t requirement_bound(uint64_t nkeys, uint64_t key_bytes) {   /* >= cover_requirement's size */
    return tr_budget_add(40, tr_budget_add(tr_budget_mul(nkeys, 12), key_bytes));
}

static int32_t cover_check(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *d,
                           const tr_quant_keys *required, int32_t domain, const tr_scope *parents, cover_plan *P) {
    memset(P, 0, sizeof *P);
    if (!child || !categories || tr_scope_get_view(child, &P->cv) || tr_scope_get_view(categories, &P->kv))
        return TR_MATH_INVALID;
    if (!dict_ok(d) || keys_check(required)) return TR_MATH_INVALID;
    if (required->n > UINT32_MAX) return TR_MATH_UNSUPPORTED;
    if (!same_rows(child, categories, &P->cv, &P->kv)) return TR_MATH_DOMAIN_ERROR;
    if (d->size != P->kv.ngroups) return TR_MATH_DOMAIN_ERROR;
    int32_t rc = domain_check(domain, parents, &P->cv, &P->pv);
    if (rc) return rc;
    P->has_parents = parents != NULL;
    P->G = P->cv.ngroups; P->D = d->size; P->W = (P->D + 63) / 64; P->S = P->cv.selected;
    P->nkeys = required->n; P->key_bytes = keys_total(required);
    idbuf c = id_counter();
    rc = cover_presence(&c, child, categories, d, domain, parents);
    if (rc) return rc;
    P->presence_bytes = c.n;
    P->requirement_cap = requirement_bound(P->nkeys, P->key_bytes);
    P->parents_bytes = parents ? tr_scope_key_size(parents, TR_SCOPE_KEY_ALL) : 0;
    P->dict_bytes = d->bytes;
    return TR_MATH_OK;
}
/* the bytes of a coverage object (no scan scratch): the arrays, the representation, the required set and the
 * identity parts; required: in-dictionary codes allocated (the given keys, or the codes of a derived state) */
static uint64_t cover_object_bytes(uint64_t G, uint64_t W, int32_t repr, uint64_t set_codes, uint64_t req_codes,
                                   uint64_t outside_cap, uint64_t outside_n, uint64_t presence, uint64_t requirement,
                                   uint64_t parents, uint64_t dict) {
    uint64_t b = tr_budget_add(sizeof(tr_quant_cover), tr_budget_mul(G, 56));
    if (repr == TR_QUANT_REPR_MASKS) b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(G, W), W), 8));
    else b = tr_budget_add(b, tr_budget_add(tr_budget_mul(G + 1, 8), tr_budget_mul(set_codes, 4)));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(req_codes, 4), outside_cap));
    b = tr_budget_add(b, tr_budget_mul(outside_n + 1, 8));
    b = tr_budget_add(b, tr_budget_add(tr_budget_add(presence, requirement), tr_budget_add(parents, dict)));
    return b;
}
static void cover_plan_need(const cover_plan *P, int32_t repr, int32_t target, tr_math_need *need) {
    zero_need(need);
    need->need[TR_BUDGET_SUPPORT] = P->D;
    need->need[TR_BUDGET_OUTPUT] = P->G;
    need->need[TR_BUDGET_STATES] = repr == TR_QUANT_REPR_MASKS ? tr_budget_mul(P->G, P->D) : P->S;
    if (target == TR_QUANT_NEED_DEVICE) {
        need->need[TR_BUDGET_BYTES] = tr_quant_cover_device_bytes(&P->cv, &P->kv, P->has_parents ? &P->pv : NULL);
        return;
    }
    uint64_t bytes = cover_object_bytes(P->G, P->W, repr, P->S, P->nkeys, P->key_bytes, P->nkeys, P->presence_bytes,
                                        P->requirement_cap, P->parents_bytes, P->dict_bytes);
    bytes = tr_budget_add(bytes, tr_budget_mul(P->nkeys, 8));                  /* resolution scratch */
    if (repr == TR_QUANT_REPR_SETS) bytes = tr_budget_add(bytes, tr_budget_mul(P->G, 8));   /* cursors */
    need->need[TR_BUDGET_BYTES] = bytes;
}
int32_t tr_quant_cover_need(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *d,
                            const tr_quant_keys *required, int32_t domain, const tr_scope *parents, int32_t repr,
                            int32_t target, tr_math_need *need) {
    if (!need || (repr != TR_QUANT_REPR_MASKS && repr != TR_QUANT_REPR_SETS) || target < TR_QUANT_NEED_CPU ||
        target > TR_QUANT_NEED_ADOPT || (target != TR_QUANT_NEED_CPU && repr != TR_QUANT_REPR_MASKS))
        return TR_MATH_INVALID;
    cover_plan P;
    int32_t rc = cover_check(child, categories, d, required, domain, parents, &P);
    if (rc) return rc;
    cover_plan_need(&P, repr, target, need);
    return TR_MATH_OK;
}

/* the required set of a state: resolution, its storage and the requirement identity (allocated to their bounds) */
static int32_t cover_set_required(tr_quant_cover *c, const tr_quant_keys *k) {
    const uint64_t n = k->n, kb = keys_total(k);
    c->req_cap = n; c->outside_cap = kb; c->outside_off_cap = n + 1;
    c->req = zalloc(n * 4);
    c->outside = zalloc(kb);
    c->outside_off = zalloc((n + 1) * 8);
    c->requirement_cap = requirement_bound(n, kb);
    c->requirement = zalloc(c->requirement_cap);
    uint32_t *scratch = zalloc(n * 8);
    if (!c->req || !c->outside || !c->outside_off || !c->requirement || !scratch) { free(scratch); return TR_MATH_MEMORY; }
    if (c->repr == TR_QUANT_REPR_MASKS) {
        c->req_mask = zalloc(c->W * 8);
        if (!c->req_mask) { free(scratch); return TR_MATH_MEMORY; }
    }
    uint64_t nout = 0;
    resolve(c->dict, k, c->req, &c->nreq, scratch, &nout, scratch + n);
    uint64_t at = 0;
    for (uint64_t i = 0; i < nout; i++) {
        uint64_t j = scratch[i], len = k->offsets[j + 1] - k->offsets[j];
        c->outside_off[i] = at;
        if (len) memcpy(c->outside + at, k->bytes + k->offsets[j], (size_t)len);
        at += len;
    }
    c->outside_off[nout] = at;
    c->noutside = nout; c->outside_bytes = at;
    free(scratch);
    c->null_required = k->null_key;
    if (c->req_mask) for (uint64_t i = 0; i < c->nreq; i++) c->req_mask[c->req[i] >> 6] |= (uint64_t)1 << (c->req[i] & 63);
    idbuf w = id_writer(c->requirement);
    cover_requirement(&w, c->req, c->nreq, c->outside, c->outside_off, c->noutside, c->null_required);
    c->requirement_bytes = w.n;
    return TR_MATH_OK;
}
/* the object with its arrays and representation (zeroed), its dictionary copy and its presence identity */
static tr_quant_cover *cover_alloc(const cover_plan *P, const tr_scope *child, const tr_scope *categories,
                                   const tr_quant_dict *d, int32_t domain, const tr_scope *parents, int32_t repr,
                                   int32_t *rc) {
    tr_quant_cover *c = zalloc(sizeof *c);
    if (!c) { *rc = TR_MATH_MEMORY; return NULL; }
    c->domain = domain; c->repr = repr; c->reason = -1; c->parts = 1; c->null_code = d->null_code;
    c->G = P->G; c->D = P->D; c->W = P->W;
    c->arrays = zalloc(tr_budget_mul(c->G, 56));
    c->dict = dict_clone(d);
    c->presence = zalloc(P->presence_bytes);
    c->parents_desc = zalloc(P->parents_bytes);
    int ok = c->arrays && c->dict && c->presence && c->parents_desc;
    if (ok && repr == TR_QUANT_REPR_MASKS) ok = (c->masks = zalloc(tr_budget_mul(tr_budget_mul(c->G, c->W), 8))) != NULL;
    if (ok && repr == TR_QUANT_REPR_SETS) {
        c->codes_cap = P->S;
        ok = (c->offsets = zalloc((c->G + 1) * 8)) != NULL && (c->codes = zalloc(P->S * 4)) != NULL;
    }
    if (!ok) { cover_release(c); *rc = TR_MATH_MEMORY; return NULL; }
    c->presence_bytes = P->presence_bytes; c->parents_bytes = P->parents_bytes;
    idbuf w = id_writer(c->presence);
    *rc = cover_presence(&w, child, categories, d, domain, parents);
    if (!*rc && parents) *rc = tr_scope_key_write(parents, TR_SCOPE_KEY_ALL, c->parents_desc, c->parents_bytes);
    if (*rc) { cover_release(c); return NULL; }
    return c;
}

/* the coverage test (see Execution): distinct, matched, missing, extra of every group */
static void cover_test(tr_quant_cover *c) {
    const uint64_t G = c->G, W = c->W, cm = c->noutside + (uint64_t)c->null_required;
    uint64_t *dist = col(c, A_DISTINCT), *mat = col(c, A_MATCHED), *mis = col(c, A_MISSING), *ext = col(c, A_EXTRA);
    for (uint64_t g = 0; g < G; g++) {
        uint64_t d = 0, m = 0, lack = 0, more = 0;
        if (c->repr == TR_QUANT_REPR_MASKS) {                 /* AND-NOT and popcount against the required mask */
            const uint64_t *row = c->masks + g * W;
            for (uint64_t w = 0; w < W; w++) {
                const uint64_t x = row[w], r = c->req_mask[w];
                d += (uint64_t)popc(x); m += (uint64_t)popc(x & r); lack += (uint64_t)popc(r & ~x); more += (uint64_t)popc(x & ~r);
            }
        } else {                                              /* a binary search of the required codes */
            d = c->offsets[g + 1] - c->offsets[g];
            for (uint64_t i = c->offsets[g]; i < c->offsets[g + 1]; i++) m += (uint64_t)bsearch_u32(c->req, c->nreq, c->codes[i]);
            lack = c->nreq - m; more = d - m;
        }
        dist[g] = d; mat[g] = m; mis[g] = lack + cm; ext[g] = more;
    }
}
/* checks of adopted summaries against the presence states: 0 or TR_MATH_INVALID */
static int32_t summaries_consistent(const tr_quant_cover *c, const tr_quant_cover_arrays *a) {
    const uint64_t cm = c->noutside + (uint64_t)c->null_required;
    if (!a->distinct || !a->matched || !a->missing || !a->extra) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < c->G; g++) {
        uint64_t have = c->repr == TR_QUANT_REPR_MASKS ? 0 : c->offsets[g + 1] - c->offsets[g];
        if (c->repr == TR_QUANT_REPR_MASKS) for (uint64_t w = 0; w < c->W; w++) have += (uint64_t)popc(c->masks[g * c->W + w]);
        uint64_t d = a->distinct[g], m = a->matched[g];
        if (d != have || m > d || m > c->nreq || a->extra[g] != d - m || a->missing[g] != c->nreq - m + cm)
            return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
/* totals, digest and bytes; the state is valid from here on */
static void cover_finish(tr_quant_cover *c) {
    const uint64_t G = c->G, *rows = col(c, A_ROWS), *nulls = col(c, A_NULLS), *par = col(c, A_PARENTS);
    const uint64_t *dist = col(c, A_DISTINCT), *mis = col(c, A_MISSING);
    tr_quant_cover_info *I = &c->info;
    memset(I, 0, sizeof *I);
    I->ngroups = G; I->size = c->D; I->words = c->W;
    for (uint64_t g = 0; g < G; g++) {
        int present = tr_quant_present(c->domain, rows[g], par[g]);
        if (present) { I->present++; if (!rows[g]) I->empty++; if (!mis[g]) I->divides++; }
        else { I->absent++; I->orphans += rows[g]; }
        I->rows += rows[g]; I->nulls += nulls[g]; I->parent_rows += par[g]; I->pairs += dist[g];
    }
    I->required = c->nreq; I->outside = c->noutside;
    hasher h; h_init(&h);
    h_bytes(&h, c->presence, c->presence_bytes);
    h_bytes(&h, c->requirement, c->requirement_bytes);
    h_final(&h, I->digest);
    c->bytes = cover_object_bytes(G, c->W, c->repr, c->repr == TR_QUANT_REPR_SETS ? c->codes_cap : 0, c->req_cap,
                                  c->outside_cap, c->outside_off_cap - 1, c->presence_bytes, c->requirement_cap,
                                  c->parents_bytes, c->dict->bytes);
    I->bytes = c->bytes; I->identity_bytes = c->presence_bytes + c->requirement_bytes;
    I->masks_cost = c->masks_cost; I->sets_cost = c->sets_cost;
    I->domain = c->domain; I->repr = c->repr; I->reason = c->reason; I->null_required = c->null_required;
    I->null_code = c->null_code; I->parts = c->parts;
    c->magic = COVER_MAGIC;
}

/* the presence scans (one pass over the selection words) */
static void scan_masks(tr_quant_cover *c, const tr_scope_view *cv, const tr_scope_view *kv) {
    const uint64_t nw = (cv->rows + 63) / 64, W = c->W;
    uint64_t *rows = col(c, A_ROWS), *nulls = col(c, A_NULLS);
    for (uint64_t w = 0; w < nw; w++)
        for (uint64_t bits = cv->selection[w]; bits; bits &= bits - 1) {
            const uint64_t r = w * 64 + (uint64_t)ctz(bits);
            const uint32_t g = cv->groups ? cv->groups[r] : 0, k = kv->groups ? kv->groups[r] : 0;
            rows[g]++;
            if (k == c->null_code) nulls[g]++;
            else c->masks[g * W + (k >> 6)] |= (uint64_t)1 << (k & 63);
        }
}
static int32_t scan_sets(tr_quant_cover *c, const tr_scope_view *cv, const tr_scope_view *kv) {
    const uint64_t nw = (cv->rows + 63) / 64, G = c->G;
    uint64_t *rows = col(c, A_ROWS), *nulls = col(c, A_NULLS), *off = c->offsets;
    uint64_t *cursor = zalloc(G * 8);
    if (!cursor) return TR_MATH_MEMORY;
    for (uint64_t w = 0; w < nw; w++)                         /* pass 1: rows, NULLs and the codes per group */
        for (uint64_t bits = cv->selection[w]; bits; bits &= bits - 1) {
            const uint64_t r = w * 64 + (uint64_t)ctz(bits);
            const uint32_t g = cv->groups ? cv->groups[r] : 0, k = kv->groups ? kv->groups[r] : 0;
            rows[g]++;
            if (k == c->null_code) nulls[g]++; else off[g + 1]++;
        }
    for (uint64_t g = 0; g < G; g++) { off[g + 1] += off[g]; cursor[g] = off[g]; }
    for (uint64_t w = 0; w < nw; w++)                         /* pass 2: the codes, grouped */
        for (uint64_t bits = cv->selection[w]; bits; bits &= bits - 1) {
            const uint64_t r = w * 64 + (uint64_t)ctz(bits);
            const uint32_t g = cv->groups ? cv->groups[r] : 0, k = kv->groups ? kv->groups[r] : 0;
            if (k != c->null_code) c->codes[cursor[g]++] = k;
        }
    free(cursor);
    uint64_t out = 0, start = 0;                              /* sort and deduplicate each group, compacting */
    for (uint64_t g = 0; g < G; g++) {
        const uint64_t end = off[g + 1], first = out;
        heap_sort(c->codes + start, end - start);
        for (uint64_t i = start; i < end; i++) if (out == first || c->codes[out - 1] != c->codes[i]) c->codes[out++] = c->codes[i];
        off[g] = first;
        start = end;
    }
    off[G] = out;
    return TR_MATH_OK;
}

tr_quant_cover *tr_quant_cover_build(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *d,
                                     const tr_quant_keys *required, int32_t domain, const tr_scope *parents,
                                     int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error) {
    tr_budget_none(refusal);
    if (repr != TR_QUANT_REPR_AUTO && repr != TR_QUANT_REPR_MASKS && repr != TR_QUANT_REPR_SETS)
        return cover_fail(NULL, error, TR_MATH_INVALID);
    cover_plan P;
    int32_t rc = cover_check(child, categories, d, required, domain, parents, &P);
    if (rc) return cover_fail(NULL, error, rc);
    tr_math_need nm, ns;
    cover_plan_need(&P, TR_QUANT_REPR_MASKS, TR_QUANT_NEED_CPU, &nm);
    cover_plan_need(&P, TR_QUANT_REPR_SETS, TR_QUANT_NEED_CPU, &ns);
    const double mcost = (double)tr_budget_mul(tr_budget_mul(P.G, P.W), 8);
    const double scost = (double)tr_budget_add(tr_budget_mul(P.S, 4), tr_budget_mul(P.G + 1, 8));
    tr_math_budget dflt;
    const tr_math_budget *B = budget_or_default(budget, &dflt);
    int32_t chosen = repr, reason = -1;
    if (repr == TR_QUANT_REPR_AUTO) {
        tr_math_choice ch;
        rc = tr_math_budget_choose(B, &nm, mcost, &ns, scost, 1, &ch);
        if (rc) {
            if (refusal) *refusal = ch.expanded.refused ? ch.expanded : ch.factored;
            return cover_fail(NULL, error, rc);
        }
        chosen = ch.choice == TR_REPR_EXPANDED ? TR_QUANT_REPR_MASKS : TR_QUANT_REPR_SETS;
        reason = ch.reason;
    } else {
        rc = tr_budget_check(B, repr == TR_QUANT_REPR_MASKS ? &nm : &ns, refusal);
        if (rc) return cover_fail(NULL, error, rc);
    }
    tr_quant_cover *c = cover_alloc(&P, child, categories, d, domain, parents, chosen, &rc);
    if (!c) return cover_fail(NULL, error, rc);
    c->reason = reason; c->masks_cost = mcost; c->sets_cost = scost;
    rc = cover_set_required(c, required);
    if (!rc && chosen == TR_QUANT_REPR_SETS) rc = scan_sets(c, &P.cv, &P.kv);
    if (rc) return cover_fail(c, error, rc);
    if (chosen == TR_QUANT_REPR_MASKS) scan_masks(c, &P.cv, &P.kv);
    if (P.has_parents) count_parents(&P.pv, col(c, A_PARENTS));
    cover_test(c);
    cover_finish(c);
    set_error(error, TR_MATH_OK);
    return c;
}

tr_quant_cover *tr_quant_cover_adopt(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *d,
                                     const tr_quant_keys *required, int32_t domain, const tr_scope *parents,
                                     const tr_quant_cover_arrays *dev, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    cover_plan P;
    int32_t rc = cover_check(child, categories, d, required, domain, parents, &P);
    if (rc) return cover_fail(NULL, error, rc);
    tr_math_need need;
    cover_plan_need(&P, TR_QUANT_REPR_MASKS, TR_QUANT_NEED_ADOPT, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return cover_fail(NULL, error, rc);
    if (!dev || dev->ngroups != P.G || dev->size != P.D || dev->words != P.W || !dev->rows || !dev->nulls ||
        !dev->parents || (P.G && P.W && !dev->masks)) return cover_fail(NULL, error, TR_MATH_INVALID);
    const uint64_t tail = P.D % 64 ? ~(((uint64_t)1 << (P.D % 64)) - 1) : 0;
    __int128 total = 0;
    for (uint64_t g = 0; g < P.G; g++) {                      /* canonical masks, rows and NULL rows */
        const uint64_t *row = dev->masks + g * P.W;
        if (P.W && (row[P.W - 1] & tail)) return cover_fail(NULL, error, TR_MATH_INVALID);
        if (d->null_code != TR_QUANT_NO_NULL && ((row[d->null_code >> 6] >> (d->null_code & 63)) & 1))
            return cover_fail(NULL, error, TR_MATH_INVALID);
        if (dev->nulls[g] > dev->rows[g]) return cover_fail(NULL, error, TR_MATH_INVALID);
        uint64_t bits = 0;
        for (uint64_t w = 0; w < P.W; w++) bits += (uint64_t)popc(row[w]);
        if (bits > dev->rows[g] - dev->nulls[g]) return cover_fail(NULL, error, TR_MATH_INVALID);
        total += dev->rows[g];
    }
    if (total != (__int128)P.S) return cover_fail(NULL, error, TR_MATH_INVALID);
    rc = parents_consistent(dev->parents, P.G, &P.pv, P.has_parents);
    if (rc) return cover_fail(NULL, error, rc);
    tr_quant_cover *c = cover_alloc(&P, child, categories, d, domain, parents, TR_QUANT_REPR_MASKS, &rc);
    if (!c) return cover_fail(NULL, error, rc);
    rc = cover_set_required(c, required);
    if (rc) return cover_fail(c, error, rc);
    const uint64_t G = P.G;
    if (G) {
        memcpy(col(c, A_ROWS), dev->rows, (size_t)(G * 8));
        memcpy(col(c, A_NULLS), dev->nulls, (size_t)(G * 8));
        memcpy(col(c, A_PARENTS), dev->parents, (size_t)(G * 8));
        if (P.W) memcpy(c->masks, dev->masks, (size_t)(G * P.W * 8));
    }
    rc = summaries_consistent(c, dev);
    if (rc) return cover_fail(c, error, rc);
    for (int a = A_DISTINCT; a <= A_EXTRA; a++) {
        const uint64_t *src = a == A_DISTINCT ? dev->distinct : a == A_MATCHED ? dev->matched : a == A_MISSING ? dev->missing : dev->extra;
        if (G) memcpy(col(c, a), src, (size_t)(G * 8));
    }
    cover_finish(c);
    set_error(error, TR_MATH_OK);
    return c;
}

/* a new object holding a copy of the presence states of s (arrays, representation, dictionary, presence identity) */
static tr_quant_cover *cover_copy_presence(const tr_quant_cover *s, int32_t *rc) {
    tr_quant_cover *c = zalloc(sizeof *c);
    if (!c) { *rc = TR_MATH_MEMORY; return NULL; }
    c->domain = s->domain; c->repr = s->repr; c->reason = s->reason; c->parts = s->parts; c->null_code = s->null_code;
    c->G = s->G; c->D = s->D; c->W = s->W; c->masks_cost = s->masks_cost; c->sets_cost = s->sets_cost;
    c->arrays = zalloc(tr_budget_mul(c->G, 56));
    c->dict = dict_clone(s->dict);
    c->presence = copy_bytes(s->presence, s->presence_bytes);
    c->parents_desc = copy_bytes(s->parents_desc, s->parents_bytes);
    int ok = c->arrays && c->dict && c->presence && c->parents_desc;
    if (ok && s->repr == TR_QUANT_REPR_MASKS) ok = (c->masks = zalloc(tr_budget_mul(tr_budget_mul(c->G, c->W), 8))) != NULL;
    if (ok && s->repr == TR_QUANT_REPR_SETS) {
        c->codes_cap = s->offsets[s->G];
        ok = (c->offsets = zalloc((c->G + 1) * 8)) != NULL && (c->codes = zalloc(c->codes_cap * 4)) != NULL;
    }
    if (!ok) { cover_release(c); *rc = TR_MATH_MEMORY; return NULL; }
    c->presence_bytes = s->presence_bytes; c->parents_bytes = s->parents_bytes;
    if (c->G) memcpy(c->arrays, s->arrays, (size_t)(c->G * 24));               /* rows, nulls, parents */
    if (c->masks && c->G && c->W) memcpy(c->masks, s->masks, (size_t)(c->G * c->W * 8));
    if (c->offsets) {
        memcpy(c->offsets, s->offsets, (size_t)((c->G + 1) * 8));
        if (c->codes_cap) memcpy(c->codes, s->codes, (size_t)(c->codes_cap * 4));
    }
    return c;
}

tr_quant_cover *tr_quant_cover_require(const tr_quant_cover *s, const tr_quant_keys *required,
                                       const tr_quant_cover_arrays *summaries, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!cover_ok(s) || keys_check(required)) return cover_fail(NULL, error, TR_MATH_INVALID);
    if (required->n > UINT32_MAX) return cover_fail(NULL, error, TR_MATH_UNSUPPORTED);
    if (summaries && (summaries->ngroups != s->G || summaries->size != s->D || summaries->words != s->W))
        return cover_fail(NULL, error, TR_MATH_INVALID);
    const uint64_t n = required->n, kb = keys_total(required);
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_SUPPORT] = s->D;
    need.need[TR_BUDGET_OUTPUT] = s->G;
    need.need[TR_BUDGET_STATES] = s->repr == TR_QUANT_REPR_MASKS ? tr_budget_mul(s->G, s->D) : s->offsets[s->G];
    need.need[TR_BUDGET_BYTES] = tr_budget_add(cover_object_bytes(s->G, s->W, s->repr, s->repr == TR_QUANT_REPR_SETS ? s->offsets[s->G] : 0,
                                                                  n, kb, n, s->presence_bytes, requirement_bound(n, kb),
                                                                  s->parents_bytes, s->dict->bytes),
                                               tr_budget_mul(n, 8));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return cover_fail(NULL, error, rc);
    tr_quant_cover *c = cover_copy_presence(s, &rc);
    if (!c) return cover_fail(NULL, error, rc);
    rc = cover_set_required(c, required);
    if (rc) return cover_fail(c, error, rc);
    if (summaries) {
        rc = summaries_consistent(c, summaries);
        if (rc) return cover_fail(c, error, rc);
        const uint64_t *src[4] = {summaries->distinct, summaries->matched, summaries->missing, summaries->extra};
        for (int a = 0; a < 4; a++) if (c->G) memcpy(col(c, A_DISTINCT + a), src[a], (size_t)(c->G * 8));
    } else cover_test(c);
    cover_finish(c);
    set_error(error, TR_MATH_OK);
    return c;
}

tr_quant_cover *tr_quant_cover_merge(const tr_quant_cover *a, const tr_quant_cover *b, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!cover_ok(a) || !cover_ok(b)) return cover_fail(NULL, error, TR_MATH_INVALID);
    if (a->G != b->G || a->D != b->D || a->repr != b->repr || a->domain != b->domain ||
        tr_quant_dict_same(a->dict, b->dict) != 1 || a->requirement_bytes != b->requirement_bytes ||
        memcmp(a->requirement, b->requirement, (size_t)a->requirement_bytes) || a->parents_bytes != b->parents_bytes ||
        (a->parents_bytes && memcmp(a->parents_desc, b->parents_desc, (size_t)a->parents_bytes)))
        return cover_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const uint64_t G = a->G, W = a->W;
    const uint64_t pairs = a->repr == TR_QUANT_REPR_SETS ? tr_budget_add(a->offsets[G], b->offsets[G]) : 0;
    const uint64_t presence = tr_budget_add(24, tr_budget_add(a->presence_bytes, b->presence_bytes));
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_SUPPORT] = a->D;
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_STATES] = a->repr == TR_QUANT_REPR_MASKS ? tr_budget_mul(G, a->D) : pairs;
    need.need[TR_BUDGET_BYTES] = cover_object_bytes(G, W, a->repr, pairs, a->nreq, a->outside_bytes, a->noutside, presence,
                                                    a->requirement_bytes, a->parents_bytes, a->dict->bytes);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return cover_fail(NULL, error, rc);
    tr_quant_cover *c = zalloc(sizeof *c);
    if (!c) return cover_fail(NULL, error, TR_MATH_MEMORY);
    c->domain = a->domain; c->repr = a->repr; c->reason = -1; c->null_code = a->null_code; c->G = G; c->D = a->D; c->W = W;
    c->masks_cost = a->masks_cost; c->sets_cost = a->sets_cost;
    uint64_t parts;
    if (!add_u64((uint64_t)a->parts, (uint64_t)b->parts, &parts) || parts > INT32_MAX) return cover_fail(c, error, TR_MATH_OVERFLOW);
    c->parts = (int32_t)parts;
    c->arrays = zalloc(tr_budget_mul(G, 56));
    c->dict = dict_clone(a->dict);
    c->presence = zalloc(presence);
    c->parents_desc = copy_bytes(a->parents_desc, a->parents_bytes);
    c->req_cap = a->nreq; c->outside_cap = a->outside_bytes; c->outside_off_cap = a->noutside + 1;
    c->req = zalloc(a->nreq * 4);
    c->outside = copy_bytes(a->outside, a->outside_bytes);
    c->outside_off = zalloc((a->noutside + 1) * 8);
    c->requirement = copy_bytes(a->requirement, a->requirement_bytes);
    int ok = c->arrays && c->dict && c->presence && c->parents_desc && c->req && c->outside && c->outside_off && c->requirement;
    if (ok && c->repr == TR_QUANT_REPR_MASKS)
        ok = (c->masks = zalloc(tr_budget_mul(tr_budget_mul(G, W), 8))) != NULL && (c->req_mask = zalloc(W * 8)) != NULL;
    if (ok && c->repr == TR_QUANT_REPR_SETS) {
        c->codes_cap = pairs;
        ok = (c->offsets = zalloc((G + 1) * 8)) != NULL && (c->codes = zalloc(pairs * 4)) != NULL;
    }
    if (!ok) return cover_fail(c, error, TR_MATH_MEMORY);
    if (a->nreq) memcpy(c->req, a->req, (size_t)(a->nreq * 4));
    memcpy(c->outside_off, a->outside_off, (size_t)((a->noutside + 1) * 8));
    c->nreq = a->nreq; c->noutside = a->noutside; c->outside_bytes = a->outside_bytes; c->null_required = a->null_required;
    c->requirement_bytes = c->requirement_cap = a->requirement_bytes;
    c->presence_bytes = presence; c->parents_bytes = a->parents_bytes;
    idbuf w = id_writer(c->presence);
    id_word(&w, TAG_COVER_MERGE);
    id_word(&w, a->presence_bytes); id_bytes(&w, a->presence, a->presence_bytes);
    id_word(&w, b->presence_bytes); id_bytes(&w, b->presence, b->presence_bytes);
    for (int x = A_ROWS; x <= A_NULLS; x++)
        for (uint64_t g = 0; g < G; g++)
            if (!add_u64(col(a, x)[g], col(b, x)[g], &col(c, x)[g])) return cover_fail(c, error, TR_MATH_OVERFLOW);
    if (G) memcpy(col(c, A_PARENTS), col(a, A_PARENTS), (size_t)(G * 8));   /* the same parents */
    if (c->repr == TR_QUANT_REPR_MASKS) {
        if (W) memcpy(c->req_mask, a->req_mask, (size_t)(W * 8));
        for (uint64_t i = 0; i < G * W; i++) c->masks[i] = a->masks[i] | b->masks[i];
    } else {
        uint64_t out = 0;
        for (uint64_t g = 0; g < G; g++) {                    /* the union of two sorted distinct lists */
            uint64_t i = a->offsets[g], ie = a->offsets[g + 1], j = b->offsets[g], je = b->offsets[g + 1];
            c->offsets[g] = out;
            while (i < ie || j < je) {
                uint32_t x;
                if (j >= je || (i < ie && a->codes[i] < b->codes[j])) x = a->codes[i++];
                else if (i >= ie || b->codes[j] < a->codes[i]) x = b->codes[j++];
                else { x = a->codes[i++]; j++; }
                c->codes[out++] = x;
            }
        }
        c->offsets[G] = out;
    }
    cover_test(c);
    cover_finish(c);
    set_error(error, TR_MATH_OK);
    return c;
}

int32_t tr_quant_cover_get_info(const tr_quant_cover *c, tr_quant_cover_info *out) {
    if (!cover_ok(c) || !out) return TR_MATH_INVALID;
    *out = c->info;
    return TR_MATH_OK;
}
int32_t tr_quant_cover_get_arrays(const tr_quant_cover *c, tr_quant_cover_arrays *out) {
    if (!cover_ok(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = c->G; out->size = c->D; out->words = c->W;
    out->rows = col(c, A_ROWS); out->nulls = col(c, A_NULLS); out->parents = col(c, A_PARENTS);
    out->distinct = col(c, A_DISTINCT); out->matched = col(c, A_MATCHED); out->missing = col(c, A_MISSING);
    out->extra = col(c, A_EXTRA);
    out->masks = c->masks; out->offsets = c->offsets; out->codes = c->codes;
    return TR_MATH_OK;
}
int32_t tr_quant_cover_required_mask(const tr_quant_cover *c, uint64_t *mask, uint64_t words) {
    if (!cover_ok(c) || (c->W && (!mask || words < c->W))) return TR_MATH_INVALID;
    if (c->repr != TR_QUANT_REPR_MASKS) return TR_MATH_UNSUPPORTED;
    if (c->W) memcpy(mask, c->req_mask, (size_t)(c->W * 8));
    return TR_MATH_OK;
}
int32_t tr_quant_cover_masks(const tr_quant_cover *c, tr_field_matrix *view) {
    if (!cover_ok(c) || !view) return TR_MATH_INVALID;
    if (c->repr != TR_QUANT_REPR_MASKS) return TR_MATH_UNSUPPORTED;
    view->rows = c->G; view->cols = c->D; view->stride = c->W; view->words = c->masks;
    return TR_MATH_OK;
}
static int has_code(const tr_quant_cover *c, uint64_t g, uint32_t k) {
    if (c->repr == TR_QUANT_REPR_MASKS) return (int)((c->masks[g * c->W + (k >> 6)] >> (k & 63)) & 1);
    return bsearch_u32(c->codes + c->offsets[g], c->offsets[g + 1] - c->offsets[g], k);
}
int32_t tr_quant_cover_list(const tr_quant_cover *c, uint64_t g, int32_t which, uint32_t *out, uint64_t capacity,
                            uint64_t *count) {
    if (!cover_ok(c) || g >= c->G || !count || (capacity && !out) || which < TR_QUANT_LIST_PRESENT ||
        which > TR_QUANT_LIST_EXTRA) return TR_MATH_INVALID;
    uint64_t n = 0;
#define EMIT(x) do { if (n < capacity) out[n] = (x); n++; } while (0)
    if (which == TR_QUANT_LIST_MISSING) {
        for (uint64_t i = 0; i < c->nreq; i++) if (!has_code(c, g, c->req[i])) EMIT(c->req[i]);
    } else if (c->repr == TR_QUANT_REPR_MASKS) {
        const uint64_t *row = c->masks + g * c->W;
        for (uint64_t w = 0; w < c->W; w++) {
            uint64_t bits = row[w] & (which == TR_QUANT_LIST_EXTRA ? ~c->req_mask[w] : ~(uint64_t)0);
            for (; bits; bits &= bits - 1) EMIT((uint32_t)(w * 64 + (uint64_t)ctz(bits)));
        }
    } else {
        for (uint64_t i = c->offsets[g]; i < c->offsets[g + 1]; i++)
            if (which == TR_QUANT_LIST_PRESENT || !bsearch_u32(c->req, c->nreq, c->codes[i])) EMIT(c->codes[i]);
    }
#undef EMIT
    *count = n;
    return TR_MATH_OK;
}
int32_t tr_quant_cover_required_codes(const tr_quant_cover *c, uint32_t *out, uint64_t capacity, uint64_t *n) {
    if (!cover_ok(c) || !n || (capacity && !out)) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < c->nreq && i < capacity; i++) out[i] = c->req[i];
    *n = c->nreq;
    return TR_MATH_OK;
}
int32_t tr_quant_cover_outside(const tr_quant_cover *c, uint8_t *bytes, uint64_t bytes_capacity, uint64_t *offsets,
                               uint64_t offsets_capacity, uint64_t *n, uint64_t *total_bytes) {
    if (!cover_ok(c) || !n || !total_bytes || (bytes_capacity && !bytes) || (offsets_capacity && !offsets))
        return TR_MATH_INVALID;
    *n = c->noutside; *total_bytes = c->outside_bytes;
    if (offsets_capacity >= c->noutside + 1 && bytes_capacity >= c->outside_bytes) {
        memcpy(offsets, c->outside_off, (size_t)((c->noutside + 1) * 8));
        if (c->outside_bytes) memcpy(bytes, c->outside, (size_t)c->outside_bytes);
    }
    return TR_MATH_OK;
}
static int question_ok(int32_t q) { return q >= TR_QUANT_DIVIDES && q <= TR_QUANT_DISJOINT; }
int32_t tr_quant_cover_answer(const tr_quant_cover *c, int32_t q, uint8_t *out, uint64_t capacity) {
    if (!cover_ok(c) || !question_ok(q) || (c->G && (!out || capacity < c->G))) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < c->G; g++)
        out[g] = (uint8_t)tr_quant_cover_code(q, tr_quant_present(c->domain, col(c, A_ROWS)[g], col(c, A_PARENTS)[g]),
                                              col(c, A_MATCHED)[g], col(c, A_MISSING)[g], col(c, A_EXTRA)[g]);
    return TR_MATH_OK;
}
int32_t tr_quant_cover_tally(const tr_quant_cover *c, int32_t q, uint64_t out[4]) {
    if (!cover_ok(c) || !question_ok(q) || !out) return TR_MATH_INVALID;
    out[0] = out[1] = out[2] = out[3] = 0;
    for (uint64_t g = 0; g < c->G; g++)
        out[tr_quant_cover_code(q, tr_quant_present(c->domain, col(c, A_ROWS)[g], col(c, A_PARENTS)[g]),
                                col(c, A_MATCHED)[g], col(c, A_MISSING)[g], col(c, A_EXTRA)[g])]++;
    return TR_MATH_OK;
}
uint64_t tr_quant_cover_identity_size(const tr_quant_cover *c) {
    return cover_ok(c) ? c->presence_bytes + c->requirement_bytes : 0;
}
int32_t tr_quant_cover_identity(const tr_quant_cover *c, uint8_t *out, uint64_t capacity) {
    if (!cover_ok(c) || !out || capacity < c->presence_bytes + c->requirement_bytes) return TR_MATH_INVALID;
    memcpy(out, c->presence, (size_t)c->presence_bytes);
    memcpy(out + c->presence_bytes, c->requirement, (size_t)c->requirement_bytes);
    return TR_MATH_OK;
}
int32_t tr_quant_cover_same(const tr_quant_cover *a, const tr_quant_cover *b) {
    if (!cover_ok(a) || !cover_ok(b)) return TR_MATH_INVALID;
    return a->presence_bytes == b->presence_bytes && a->requirement_bytes == b->requirement_bytes &&
           !memcmp(a->presence, b->presence, (size_t)a->presence_bytes) &&
           !memcmp(a->requirement, b->requirement, (size_t)a->requirement_bytes);
}
int32_t tr_quant_cover_matches(const tr_quant_cover *c, const tr_scope *child, const tr_scope *categories,
                               const tr_quant_dict *d, const tr_quant_keys *required, int32_t domain,
                               const tr_scope *parents) {
    if (!cover_ok(c)) return TR_MATH_INVALID;
    cover_plan P;
    int32_t rc = cover_check(child, categories, d, required, domain, parents, &P);
    if (rc) return rc;
    idbuf p = id_comparer(c->presence, c->presence_bytes);
    rc = cover_presence(&p, child, categories, d, domain, parents);
    if (rc) return rc;
    if (p.differ || p.n != c->presence_bytes) return 0;
    /* the requirement part of these keys against this dictionary (equal presence: the same dictionary) */
    const uint64_t n = required->n, kb = keys_total(required);
    uint32_t *scratch = zalloc(n * 12);
    uint8_t *outside = zalloc(kb);
    uint64_t *off = zalloc((n + 1) * 8);
    if (!scratch || !outside || !off) { free(scratch); free(outside); free(off); return TR_MATH_MEMORY; }
    uint64_t nc = 0, no = 0, at = 0;
    resolve(d, required, scratch, &nc, scratch + n, &no, scratch + 2 * n);
    for (uint64_t i = 0; i < no; i++) {
        uint64_t j = scratch[n + i], len = required->offsets[j + 1] - required->offsets[j];
        off[i] = at;
        if (len) memcpy(outside + at, required->bytes + required->offsets[j], (size_t)len);
        at += len;
    }
    off[no] = at;
    idbuf q = id_comparer(c->requirement, c->requirement_bytes);
    cover_requirement(&q, scratch, nc, outside, off, no, required->null_key);
    free(scratch); free(outside); free(off);
    return !q.differ && q.n == c->requirement_bytes;
}
