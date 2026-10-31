/* FND-004: grouped exact moment banks, their rollup and the bank cache; see math_bank.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_bank.h"
#include "mathematics.h"
#include "math_domain.h"
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(tr_bank_term) == 16, "tr_bank_term ABI");
_Static_assert(sizeof(tr_bank_states) == 40, "tr_bank_states ABI");
_Static_assert(sizeof(tr_bank_info) == 72, "tr_bank_info ABI");
_Static_assert(sizeof(tr_bank_cache_stats) == 72, "tr_bank_cache_stats ABI");

#define BANK_MAGIC UINT32_C(0x4b4e4142)
#define BUILDER_MAGIC UINT32_C(0x444c4942)
#define CACHE_MAGIC UINT32_C(0x48434b42)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)

struct tr_bank_s {
    uint32_t magic;
    int32_t ncols, nterms, mode, degree, scoped;
    uint32_t joint;
    uint64_t ngroups, rows, present, bytes;
    uint64_t digest[2];
    uint8_t *exps;              /* nterms * ncols */
    tr_bank_term *terms;        /* nterms */
    uint64_t *grows, *count;    /* ngroups, ngroups * nterms */
    tr_math_i128 *sum;          /* ngroups * nterms */
};

struct tr_bank_builder_s {
    uint32_t magic;
    int32_t ncols, nterms, mode, degree, spent;
    uint32_t joint;
    const tr_scope *scope;
    tr_scope_view view;
    uint64_t digest[2];
    uint8_t *exps, *live;       /* live[j]: term j can be valid on some selected row (else it never accumulates) */
    tr_bank_term *terms;
    uint64_t *grows, *count;
    tr_math_i128 *sum;
    uint64_t *covered, ncovered;
    /* the CPU loop's plan: the basis's columns, those with validity flags, the largest live exponent of each */
    int32_t nused, nnull;
    uint8_t used[TR_SCOPE_MAX_COLUMNS], nullcol[TR_SCOPE_MAX_COLUMNS], maxexp[TR_SCOPE_MAX_COLUMNS];
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void bump(uint64_t *counter) { if (*counter < UINT64_MAX) ++*counter; }
static int bank_valid(const tr_bank *b) { return b && b->magic == BANK_MAGIC; }
static int builder_valid(const tr_bank_builder *b) { return b && b->magic == BUILDER_MAGIC; }
static int mode_ok(int32_t mode) { return mode == TR_BANK_SQL || mode == TR_BANK_JOINT; }
static uint64_t basis_bytes(int32_t nterms, int32_t ncols) {
    return tr_budget_add(tr_budget_mul((uint64_t)nterms, sizeof(tr_bank_term)), tr_budget_mul((uint64_t)nterms, (uint64_t)ncols));
}
static uint64_t bank_alloc_bytes(uint64_t ngroups, int32_t nterms, int32_t ncols) {
    return tr_budget_add(tr_budget_add(sizeof(tr_bank), tr_bank_state_bytes(ngroups, nterms)), basis_bytes(nterms, ncols));
}
static void base_need(uint64_t states, int32_t nterms, int32_t degree, uint32_t joint, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)tr_bank_popc32(joint);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nterms;
    need->need[TR_BUDGET_STATES] = states;
}
/* a build: the builder, its states, coverage bitmap, basis and live flags, and the bank it becomes */
static void cpu_need(const tr_scope_view *v, int32_t nterms, int32_t degree, uint32_t joint, tr_math_need *need) {
    base_need(tr_budget_mul(v->ngroups, (uint64_t)nterms), nterms, degree, joint, need);
    uint64_t bytes = tr_budget_add(sizeof(tr_bank_builder), bank_alloc_bytes(v->ngroups, nterms, v->ncols));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_bitmap_bytes(v->rows), (uint64_t)nterms));
    need->need[TR_BUDGET_BYTES] = bytes;
}

static void bank_release(tr_bank *b) {
    if (!b) return;
    free(b->exps); free(b->terms); free(b->grows); free(b->count); free(b->sum);
    b->magic = 0;
    free(b);
}
void tr_bank_free(tr_bank *b) { if (bank_valid(b)) bank_release(b); }
static void builder_release(tr_bank_builder *b) {
    if (!b) return;
    free(b->exps); free(b->live); free(b->terms); free(b->grows); free(b->count); free(b->sum); free(b->covered);
    b->magic = 0;
    free(b);
}
void tr_bank_builder_free(tr_bank_builder *b) { if (builder_valid(b)) builder_release(b); }

/* an empty bank with its basis copied (states zeroed) */
static tr_bank *bank_alloc(uint64_t ngroups, int32_t ncols, int32_t nterms, int32_t mode, int32_t degree, uint32_t joint,
                           const uint8_t *exps, const tr_bank_term *terms, int with_states) {
    tr_bank *b = zalloc(sizeof *b);
    if (!b) return NULL;
    b->ncols = ncols; b->nterms = nterms; b->mode = mode; b->degree = degree; b->joint = joint; b->ngroups = ngroups;
    uint64_t gm = ngroups * (uint64_t)nterms;
    b->exps = zalloc((uint64_t)nterms * (uint64_t)ncols);
    b->terms = zalloc((uint64_t)nterms * sizeof(tr_bank_term));
    if (with_states) { b->grows = zalloc(ngroups * 8); b->count = zalloc(gm * 8); b->sum = zalloc(gm * 16); }
    if (!b->exps || !b->terms || (with_states && (!b->grows || !b->count || !b->sum))) { bank_release(b); return NULL; }
    if (nterms && ncols) memcpy(b->exps, exps, (size_t)nterms * (size_t)ncols);
    if (nterms) memcpy(b->terms, terms, (size_t)nterms * sizeof(tr_bank_term));
    b->bytes = bank_alloc_bytes(ngroups, nterms, ncols);
    b->magic = BANK_MAGIC;
    return b;
}
static void bank_tally(tr_bank *b) {
    uint64_t rows = 0, present = 0;
    for (uint64_t g = 0; g < b->ngroups; g++) { rows += b->grows[g]; present += b->grows[g] != 0; }
    b->rows = rows; b->present = present;
}

/* ------------------------------------------------------------------ building */
static uint64_t range_mask(uint64_t w, uint64_t begin, uint64_t end) {   /* the bits of word w inside [begin, end) */
    uint64_t lo = w * 64, a = begin > lo ? begin - lo : 0, z = end - lo < 64 ? end - lo : 64;
    uint64_t m = z >= 64 ? UINT64_MAX : (UINT64_C(1) << z) - 1;
    return a ? m & ~((UINT64_C(1) << a) - 1) : m;
}

tr_bank_builder *tr_bank_init(const tr_scope *scope, const uint8_t *exps, int32_t nterms, int32_t mode,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    if (!scope || tr_scope_get_view(scope, &view) || !mode_ok(mode)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank_term terms[TR_BANK_MAX_TERMS];
    uint32_t joint = 0;
    int32_t degree = 0;
    int32_t rc = tr_bank_compile(exps, nterms, view.ncols, terms, &joint, &degree);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    cpu_need(&view, nterms, degree, joint, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    rc = tr_bank_prove(&view, terms, nterms, mode, joint);
    if (rc) { set_error(error, rc); return NULL; }
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    tr_bank_builder *b = zalloc(sizeof *b);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    b->ncols = view.ncols; b->nterms = nterms; b->mode = mode; b->degree = degree; b->joint = joint;
    b->scope = scope; b->view = view; b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1];
    uint64_t gm = view.ngroups * (uint64_t)nterms;
    b->exps = zalloc((uint64_t)nterms * (uint64_t)view.ncols);
    b->live = zalloc((uint64_t)nterms);
    b->terms = zalloc((uint64_t)nterms * sizeof(tr_bank_term));
    b->grows = zalloc(view.ngroups * 8);
    b->count = zalloc(gm * 8);
    b->sum = zalloc(gm * 16);
    b->covered = zalloc(((view.rows + 63) / 64) * 8);
    if (!b->exps || !b->live || !b->terms || !b->grows || !b->count || !b->sum || !b->covered) {
        builder_release(b); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    if (nterms && view.ncols) memcpy(b->exps, exps, (size_t)nterms * (size_t)view.ncols);
    if (nterms) memcpy(b->terms, terms, (size_t)nterms * sizeof(tr_bank_term));
    for (int32_t j = 0; j < nterms; j++) {
        b->live[j] = tr_bank_term_rows(&view, &terms[j], mode, joint) > 0;
        if (!b->live[j]) continue;
        for (int f = 0; f < terms[j].nf; f++)
            if (terms[j].exp[f] > b->maxexp[terms[j].col[f]]) b->maxexp[terms[j].col[f]] = terms[j].exp[f];
    }
    for (int32_t c = 0; c < view.ncols; c++) {
        if (!((joint >> c) & 1)) continue;
        b->used[b->nused++] = (uint8_t)c;
        if (view.col[c].valid) b->nullcol[b->nnull++] = (uint8_t)c;
    }
    b->magic = BUILDER_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}

/* the selected rows of [begin, end): bag multiplicity, then every valid term (sums proven to fit by tr_bank_prove) */
static void accumulate_range(tr_bank_builder *b, uint64_t begin, uint64_t end) {
    const tr_scope_view *v = &b->view;
    const int32_t m = b->nterms;
    const uint32_t all = b->joint;
    __int128 pw[TR_SCOPE_MAX_COLUMNS][TR_BANK_MAX_DEGREE + 1];
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) {
        uint64_t bits = v->selection[w] & range_mask(w, begin, end);
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t g = tr_scope_group(v, r);
            b->grows[g]++;
            if (!m) continue;
            uint32_t pm = all;
            for (int32_t i = 0; i < b->nnull; i++) {
                int c = b->nullcol[i];
                if (!v->col[c].valid[r]) pm &= ~(UINT32_C(1) << c);
            }
            if (b->mode == TR_BANK_JOINT && pm != all) continue;          /* listwise: one NULL voids the row */
            for (int32_t i = 0; i < b->nused; i++) {
                int c = b->used[i];
                if (!((pm >> c) & 1) || !b->maxexp[c]) continue;
                __int128 x = tr_scope_value(v, c, r);
                pw[c][1] = x;
                for (int e = 2; e <= b->maxexp[c]; e++) pw[c][e] = pw[c][e - 1] * x;
            }
            uint64_t *cnt = b->count + (uint64_t)g * (uint64_t)m;
            tr_math_i128 *sm = b->sum + (uint64_t)g * (uint64_t)m;
            for (int32_t j = 0; j < m; j++) {
                const tr_bank_term *t = &b->terms[j];
                if (!b->live[j] || (pm & t->need) != t->need) continue;
                __int128 p = 1;
                for (int f = 0; f < t->nf; f++) p *= pw[t->col[f]][t->exp[f]];
                cnt[j]++;
                sm[j] = tr_i128_set(tr_i128_get(sm[j]) + p);
            }
        }
    }
}

int32_t tr_bank_accumulate(tr_bank_builder *b, uint64_t begin, uint64_t end) {
    if (!builder_valid(b) || b->spent || begin > end || end > b->view.rows) return TR_MATH_INVALID;
    if (begin == end) return TR_MATH_OK;
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++)
        if (b->covered[w] & range_mask(w, begin, end)) return TR_MATH_INVALID;   /* a row is counted once */
    accumulate_range(b, begin, end);
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) b->covered[w] |= range_mask(w, begin, end);
    b->ncovered += end - begin;
    return TR_MATH_OK;
}

int32_t tr_bank_merge(tr_bank_builder *into, const tr_bank_builder *from) {
    if (!builder_valid(into) || !builder_valid(from) || into == from || into->spent || from->spent ||
        into->ncols != from->ncols || into->nterms != from->nterms || into->mode != from->mode ||
        (into->nterms && into->ncols && memcmp(into->exps, from->exps, (size_t)into->nterms * (size_t)into->ncols)))
        return TR_MATH_INVALID;
    if (into->scope != from->scope) {
        int32_t same = tr_scope_same(into->scope, from->scope);
        if (same != 1) return same < 0 ? same : TR_MATH_INVALID;
    }
    uint64_t nw = (into->view.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) if (into->covered[w] & from->covered[w]) return TR_MATH_INVALID;
    uint64_t gm = into->view.ngroups * (uint64_t)into->nterms;
    for (uint64_t g = 0; g < into->view.ngroups; g++) into->grows[g] += from->grows[g];
    for (uint64_t i = 0; i < gm; i++) {
        into->count[i] += from->count[i];
        into->sum[i] = tr_i128_set(tr_i128_get(into->sum[i]) + tr_i128_get(from->sum[i]));   /* disjoint rows: proven */
    }
    for (uint64_t w = 0; w < nw; w++) into->covered[w] |= from->covered[w];
    into->ncovered += from->ncovered;
    return TR_MATH_OK;
}

tr_bank *tr_bank_finish(tr_bank_builder *b, int32_t *error) {
    if (!builder_valid(b) || b->spent || b->ncovered != b->view.rows) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank *k = zalloc(sizeof *k);
    if (!k) { set_error(error, TR_MATH_MEMORY); return NULL; }
    k->ncols = b->ncols; k->nterms = b->nterms; k->mode = b->mode; k->degree = b->degree; k->joint = b->joint;
    k->ngroups = b->view.ngroups; k->scoped = 1; k->digest[0] = b->digest[0]; k->digest[1] = b->digest[1];
    k->exps = b->exps; k->terms = b->terms; k->grows = b->grows; k->count = b->count; k->sum = b->sum;
    b->exps = NULL; b->terms = NULL; b->grows = NULL; b->count = NULL; b->sum = NULL;
    free(b->live); b->live = NULL; free(b->covered); b->covered = NULL;
    b->spent = 1;
    k->bytes = bank_alloc_bytes(k->ngroups, k->nterms, k->ncols);
    k->magic = BANK_MAGIC;
    bank_tally(k);
    set_error(error, TR_MATH_OK);
    return k;
}

tr_bank *tr_bank_build(const tr_scope *scope, const uint8_t *exps, int32_t nterms, int32_t mode,
                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_bank_builder *b = tr_bank_init(scope, exps, nterms, mode, budget, refusal, error);
    if (!b) return NULL;
    int32_t rc = tr_bank_accumulate(b, 0, b->view.rows);
    tr_bank *k = rc ? NULL : tr_bank_finish(b, error);
    if (rc) set_error(error, rc);
    builder_release(b);
    return k;
}

static void adopt_need(uint64_t ngroups, int32_t ncols, int32_t nterms, int32_t degree, uint32_t joint, tr_math_need *need) {
    base_need(tr_budget_mul(ngroups, (uint64_t)nterms), nterms, degree, joint, need);
    need->need[TR_BUDGET_BYTES] = bank_alloc_bytes(ngroups, nterms, ncols);
}

int32_t tr_bank_need(const tr_scope *scope, const uint8_t *exps, int32_t nterms, int32_t mode, int32_t target,
                     tr_math_need *need) {
    tr_scope_view view;
    if (!scope || !need || tr_scope_get_view(scope, &view) || !mode_ok(mode) ||
        (target != TR_BANK_NEED_CPU && target != TR_BANK_NEED_DEVICE && target != TR_BANK_NEED_ADOPT)) return TR_MATH_INVALID;
    tr_bank_term terms[TR_BANK_MAX_TERMS];
    uint32_t joint = 0;
    int32_t degree = 0;
    int32_t rc = tr_bank_compile(exps, nterms, view.ncols, terms, &joint, &degree);
    if (rc) return rc;
    if (target == TR_BANK_NEED_DEVICE) tr_bank_device_need(&view, nterms, mode, joint, degree, need);
    else if (target == TR_BANK_NEED_ADOPT) adopt_need(view.ngroups, view.ncols, nterms, degree, joint, need);
    else cpu_need(&view, nterms, degree, joint, need);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ adoption, rollup, reading */
tr_bank *tr_bank_from_states(const tr_scope *origin, int32_t ncols, const uint8_t *exps, int32_t nterms, int32_t mode,
                             const tr_bank_states *st, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    if (!st || !mode_ok(mode) || st->nterms != nterms || st->ngroups < 1 || st->ngroups > TR_SCOPE_MAX_GROUPS ||
        !st->rows || (nterms && (!st->count || !st->sum)) || (origin && tr_scope_get_view(origin, &view)) ||
        (origin && (view.ncols != ncols || view.ngroups != st->ngroups))) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank_term terms[TR_BANK_MAX_TERMS];
    uint32_t joint = 0;
    int32_t degree = 0;
    int32_t rc = tr_bank_compile(exps, nterms, ncols, terms, &joint, &degree);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    adopt_need(st->ngroups, ncols, nterms, degree, joint, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (origin && (rc = tr_bank_prove(&view, terms, nterms, mode, joint))) { set_error(error, rc); return NULL; }
    int32_t constant = -1;
    for (int32_t j = 0; j < nterms; j++) if (!terms[j].nf) constant = j;
    unsigned __int128 bound[TR_BANK_MAX_TERMS];
    for (int32_t j = 0; origin && j < nterms; j++) bound[j] = tr_bank_term_bound(&view, &terms[j]);
    uint64_t total = 0;
    for (uint64_t g = 0; g < st->ngroups; g++) {
        uint64_t rows = st->rows[g];
        if (__builtin_add_overflow(total, rows, &total)) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t j = 0; j < nterms; j++) {
            uint64_t i = g * (uint64_t)nterms + (uint64_t)j, c = st->count[i];
            __int128 s = tr_i128_get(st->sum[i]);
            unsigned __int128 mag = s < 0 ? (unsigned __int128)0 - (unsigned __int128)s : (unsigned __int128)s;
            int bad = c > rows || (!c && s) || (j == constant && (s != (__int128)c || (mode == TR_BANK_SQL && c != rows)));
            if (!bad && origin) bad = mag > tr_bank_satmul(bound[j], c, I128_LIMIT);
            if (bad) { set_error(error, TR_MATH_INVALID); return NULL; }
        }
    }
    if (origin && total != view.selected) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank *b = bank_alloc(st->ngroups, ncols, nterms, mode, degree, joint, exps, terms, 1);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(b->grows, st->rows, (size_t)st->ngroups * 8);
    if (nterms) {
        memcpy(b->count, st->count, (size_t)(st->ngroups * (uint64_t)nterms) * 8);
        memcpy(b->sum, st->sum, (size_t)(st->ngroups * (uint64_t)nterms) * 16);
    }
    if (origin) {
        tr_scope_info info;
        tr_scope_get_info(origin, &info);
        b->scoped = 1; b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1];
    }
    bank_tally(b);
    set_error(error, TR_MATH_OK);
    return b;
}

int32_t tr_bank_rollup_need(const tr_bank *f, uint64_t coarse_groups, int32_t target, tr_math_need *need) {
    if (!bank_valid(f) || !need || (target != TR_BANK_NEED_CPU && target != TR_BANK_NEED_DEVICE)) return TR_MATH_INVALID;
    if (target == TR_BANK_NEED_DEVICE) tr_bank_rollup_device_need(f->ngroups, f->nterms, coarse_groups, need);
    else {
        adopt_need(coarse_groups, f->ncols, f->nterms, f->degree, f->joint, need);   /* the coarse bank, the bound's scratch */
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_mul(coarse_groups, 16));
    }
    return TR_MATH_OK;
}

tr_bank *tr_bank_rollup(const tr_bank *f, const uint32_t *map, uint64_t coarse_groups, const tr_scope *origin,
                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_info info;
    if (!bank_valid(f) || !map || coarse_groups < 1 || coarse_groups > TR_SCOPE_MAX_GROUPS ||
        (origin && (tr_scope_get_info(origin, &info) || info.ngroups != coarse_groups || info.selected != f->rows ||
                    info.ncols != f->ncols))) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    tr_bank_rollup_need(f, coarse_groups, TR_BANK_NEED_CPU, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t g = 0; g < f->ngroups; g++) if (map[g] >= coarse_groups) { set_error(error, TR_MATH_INVALID); return NULL; }
    const uint64_t m = (uint64_t)f->nterms;
    if (m) {
        unsigned __int128 *scratch = zalloc(coarse_groups * 16);
        if (!scratch) { set_error(error, TR_MATH_MEMORY); return NULL; }
        tr_bank_states fine;
        tr_bank_get_states(f, &fine);
        rc = tr_bank_rollup_bound(&fine, map, coarse_groups, scratch);
        free(scratch);
        if (rc) { set_error(error, rc); return NULL; }
    }
    tr_bank *b = bank_alloc(coarse_groups, f->ncols, f->nterms, f->mode, f->degree, f->joint, f->exps, f->terms, 1);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < f->ngroups; g++) {
        uint64_t q = map[g];
        if (!f->grows[g]) continue;                     /* an absent fine group holds nothing */
        b->grows[q] += f->grows[g];
        for (uint64_t j = 0; j < m; j++) {
            b->count[q * m + j] += f->count[g * m + j];
            b->sum[q * m + j] = tr_i128_set(tr_i128_get(b->sum[q * m + j]) + tr_i128_get(f->sum[g * m + j]));
        }
    }
    if (origin) { b->scoped = 1; b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1]; }
    bank_tally(b);
    set_error(error, TR_MATH_OK);
    return b;
}

tr_bank *tr_bank_clone(const tr_bank *f, int32_t *error) {
    if (!bank_valid(f)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank *b = bank_alloc(f->ngroups, f->ncols, f->nterms, f->mode, f->degree, f->joint, f->exps, f->terms, 1);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t gm = f->ngroups * (uint64_t)f->nterms;
    memcpy(b->grows, f->grows, (size_t)f->ngroups * 8);
    if (gm) { memcpy(b->count, f->count, (size_t)gm * 8); memcpy(b->sum, f->sum, (size_t)gm * 16); }
    b->scoped = f->scoped; b->digest[0] = f->digest[0]; b->digest[1] = f->digest[1];
    b->rows = f->rows; b->present = f->present;
    set_error(error, TR_MATH_OK);
    return b;
}

int32_t tr_bank_equal(const tr_bank *a, const tr_bank *b) {
    if (!bank_valid(a) || !bank_valid(b)) return TR_MATH_INVALID;
    if (a->ncols != b->ncols || a->nterms != b->nterms || a->mode != b->mode || a->ngroups != b->ngroups ||
        a->rows != b->rows || a->present != b->present) return 0;
    uint64_t gm = a->ngroups * (uint64_t)a->nterms;
    if (a->nterms && a->ncols && memcmp(a->exps, b->exps, (size_t)a->nterms * (size_t)a->ncols)) return 0;
    if (memcmp(a->grows, b->grows, (size_t)a->ngroups * 8)) return 0;
    if (gm && (memcmp(a->count, b->count, (size_t)gm * 8) || memcmp(a->sum, b->sum, (size_t)gm * 16))) return 0;
    return 1;
}

int32_t tr_bank_get_info(const tr_bank *b, tr_bank_info *out) {
    if (!bank_valid(b) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = b->ngroups; out->present = b->present; out->rows = b->rows; out->bytes = b->bytes;
    out->digest[0] = b->digest[0]; out->digest[1] = b->digest[1];
    out->ncols = b->ncols; out->nterms = b->nterms; out->mode = b->mode; out->degree = b->degree;
    out->joint = b->joint; out->scoped = b->scoped;
    return TR_MATH_OK;
}
int32_t tr_bank_get_states(const tr_bank *b, tr_bank_states *out) {
    if (!bank_valid(b) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = b->ngroups; out->nterms = b->nterms;
    out->rows = b->grows; out->count = b->count; out->sum = b->sum;
    return TR_MATH_OK;
}
int32_t tr_bank_get_exponents(const tr_bank *b, uint8_t *out, uint64_t capacity) {
    uint64_t need = (uint64_t)b->nterms * (uint64_t)b->ncols;
    if (!bank_valid(b) || (need && (!out || capacity < need))) return TR_MATH_INVALID;
    if (need) memcpy(out, b->exps, (size_t)need);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the cache */
typedef struct {
    tr_bank *bank;                    /* owned; NULL = a free slot */
    unsigned char *key;               /* the scope's whole descriptor */
    uint64_t key_bytes, bytes, touched;
    uint64_t digest[2], version[2], table;
    uint8_t identity[16];
    int32_t source, reserved;
} bank_entry;

struct tr_bank_cache_s {
    uint32_t magic;
    int32_t capacity, entries;
    uint64_t hits, misses, puts, evictions, refreshes, invalidations, bytes, max_bytes, clock;
    bank_entry entry[];
};

static int cache_valid(const tr_bank_cache *c) { return c && c->magic == CACHE_MAGIC; }
static uint64_t cache_fixed(int32_t capacity) { return sizeof(tr_bank_cache) + (uint64_t)capacity * sizeof(bank_entry); }

tr_bank_cache *tr_bank_cache_create(int32_t capacity, uint64_t max_bytes, int32_t *error) {
    if (capacity < 1 || capacity > TR_BANK_CACHE_CAPACITY_MAX) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!max_bytes) max_bytes = TR_BANK_CACHE_BYTES_DEFAULT;
    uint64_t fixed = cache_fixed(capacity);
    if (max_bytes < fixed) { set_error(error, TR_MATH_BUDGET); return NULL; }
    tr_bank_cache *c = zalloc(fixed);
    if (!c) { set_error(error, TR_MATH_MEMORY); return NULL; }
    c->magic = CACHE_MAGIC; c->capacity = capacity; c->bytes = fixed; c->max_bytes = max_bytes;
    set_error(error, TR_MATH_OK);
    return c;
}

static void entry_clear(tr_bank_cache *c, int i) {
    bank_entry *e = &c->entry[i];
    if (!e->bank) return;
    tr_bank_free(e->bank); free(e->key);
    c->bytes -= e->bytes; c->entries--;
    memset(e, 0, sizeof *e);
}
static uint64_t touch(tr_bank_cache *c) {
    if (c->clock == UINT64_MAX) {                 /* renumber 1..k in LRU order instead of wrapping the clock */
        int order[TR_BANK_CACHE_CAPACITY_MAX], n = 0;
        for (int i = 0; i < c->capacity; i++) {
            if (!c->entry[i].bank) continue;
            int at = n;
            while (at && c->entry[order[at - 1]].touched > c->entry[i].touched) { order[at] = order[at - 1]; at--; }
            order[at] = i; n++;
        }
        for (int i = 0; i < n; i++) c->entry[order[i]].touched = (uint64_t)i + 1;
        c->clock = (uint64_t)n;
    }
    return ++c->clock;
}
static int oldest(const tr_bank_cache *c, int except) {
    int found = -1;
    for (int i = 0; i < c->capacity; i++)
        if (i != except && c->entry[i].bank && (found < 0 || c->entry[i].touched < c->entry[found].touched)) found = i;
    return found;
}
/* the entry with exactly this descriptor, basis and mode: its index, -1, or a negative error */
static int find(const tr_bank_cache *c, const tr_scope *scope, const tr_scope_info *info, const uint8_t *exps,
                int32_t nterms, int32_t mode, int *index) {
    *index = -1;
    for (int i = 0; i < c->capacity; i++) {
        const bank_entry *e = &c->entry[i];
        if (!e->bank || e->digest[0] != info->digest[0] || e->digest[1] != info->digest[1]) continue;
        const tr_bank *b = e->bank;
        if (b->ncols != info->ncols || b->nterms != nterms || b->mode != mode ||
            (nterms && info->ncols && memcmp(b->exps, exps, (size_t)nterms * (size_t)info->ncols))) continue;
        int32_t same = tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, e->key, e->key_bytes);
        if (same < 0) return same;
        if (same) { *index = i; return TR_MATH_OK; }
    }
    return TR_MATH_OK;
}

tr_bank *tr_bank_cache_get(tr_bank_cache *c, const tr_scope *scope, const uint8_t *exps, int32_t nterms, int32_t mode,
                           int32_t *error) {
    tr_scope_info info;
    if (!cache_valid(c) || !scope || tr_scope_get_info(scope, &info) || !mode_ok(mode) || nterms < 0 ||
        nterms > TR_BANK_MAX_TERMS || (nterms && info.ncols && !exps)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int index;
    int32_t rc = find(c, scope, &info, exps, nterms, mode, &index);
    if (rc) { set_error(error, rc); return NULL; }
    if (index < 0) { bump(&c->misses); set_error(error, TR_MATH_OK); return NULL; }
    tr_bank *copy = tr_bank_clone(c->entry[index].bank, error);
    if (!copy) return NULL;
    bump(&c->hits);
    c->entry[index].touched = touch(c);
    return copy;
}

int32_t tr_bank_cache_put(tr_bank_cache *c, const tr_scope *scope, const tr_bank *bank, int32_t how,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_info info;
    if (!cache_valid(c) || !scope || tr_scope_get_info(scope, &info) || !bank_valid(bank) ||
        (how != TR_BANK_PUT && how != TR_BANK_PUT_REFRESH)) return TR_MATH_INVALID;
    if (!bank->scoped || bank->digest[0] != info.digest[0] || bank->digest[1] != info.digest[1] ||
        bank->ncols != info.ncols || bank->ngroups != info.ngroups || bank->rows != info.selected)
        return TR_MATH_DOMAIN_ERROR;                    /* not a bank of this scope */
    uint64_t key_bytes = tr_scope_key_size(scope, TR_SCOPE_KEY_ALL);
    if (!key_bytes) return TR_MATH_MEMORY;
    uint64_t entry_bytes = tr_budget_add(key_bytes, bank->bytes), fixed = cache_fixed(c->capacity);
    if (entry_bytes > c->max_bytes - fixed) {
        if (refusal) {
            refusal->resource = TR_BUDGET_BYTES; refusal->refused = 1;
            refusal->need = tr_budget_add(fixed, entry_bytes); refusal->limit = c->max_bytes;
        }
        return TR_MATH_BUDGET;
    }
    /* stage the new entry before anything retained changes */
    unsigned char *key = zalloc(key_bytes);
    if (!key) return TR_MATH_MEMORY;
    int32_t rc = tr_scope_key_write(scope, TR_SCOPE_KEY_ALL, key, key_bytes);
    if (rc) { free(key); return rc; }
    int32_t error;
    tr_bank *copy = tr_bank_clone(bank, &error);
    if (!copy) { free(key); return error; }
    int index;
    rc = find(c, scope, &info, bank->exps, bank->nterms, bank->mode, &index);
    if (rc) { free(key); tr_bank_free(copy); return rc; }
    if (how == TR_BANK_PUT_REFRESH && info.source == TR_SCOPE_TABLE)
        for (int i = 0; i < c->capacity; i++) {
            const bank_entry *e = &c->entry[i];
            if (i == index || !e->bank || e->source != TR_SCOPE_TABLE || e->table != info.table ||
                memcmp(e->identity, info.identity, 16) || (e->version[0] == info.version[0] && e->version[1] == info.version[1]))
                continue;
            entry_clear(c, i);                          /* an older (or other) version of the same table */
            bump(&c->invalidations);
        }
    if (index < 0) {
        for (int i = 0; i < c->capacity; i++) if (!c->entry[i].bank) { index = i; break; }
        if (index < 0) { index = oldest(c, -1); entry_clear(c, index); bump(&c->evictions); }
    }
    uint64_t retained = c->bytes - (c->entry[index].bank ? c->entry[index].bytes : 0) + entry_bytes;
    while (retained > c->max_bytes) {
        int victim = oldest(c, index);
        if (victim < 0) break;                          /* unreachable: entry_bytes fits on its own */
        retained -= c->entry[victim].bytes;
        entry_clear(c, victim);
        bump(&c->evictions);
    }
    entry_clear(c, index);                              /* a replacement of the same identity is not an eviction */
    bank_entry *e = &c->entry[index];
    e->bank = copy; e->key = key; e->key_bytes = key_bytes; e->bytes = entry_bytes;
    e->digest[0] = info.digest[0]; e->digest[1] = info.digest[1];
    e->version[0] = info.version[0]; e->version[1] = info.version[1];
    e->table = info.table; memcpy(e->identity, info.identity, 16); e->source = info.source;
    e->touched = touch(c);
    c->entries++; c->bytes += entry_bytes;
    bump(&c->puts);
    if (how == TR_BANK_PUT_REFRESH) bump(&c->refreshes);
    return TR_MATH_OK;
}

tr_bank *tr_bank_cache_refresh(tr_bank_cache *c, const tr_scope *scope, const uint8_t *exps, int32_t nterms,
                               int32_t mode, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (!cache_valid(c)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank *b = tr_bank_build(scope, exps, nterms, mode, budget, refusal, error);
    if (!b) return NULL;
    int32_t rc = tr_bank_cache_put(c, scope, b, TR_BANK_PUT_REFRESH, refusal);
    if (rc) { tr_bank_free(b); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return b;
}

int32_t tr_bank_cache_get_stats(const tr_bank_cache *c, tr_bank_cache_stats *out) {
    if (!cache_valid(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->hits = c->hits; out->misses = c->misses; out->puts = c->puts; out->evictions = c->evictions;
    out->refreshes = c->refreshes; out->invalidations = c->invalidations; out->bytes = c->bytes;
    out->max_bytes = c->max_bytes; out->capacity = c->capacity; out->entries = c->entries;
    return TR_MATH_OK;
}
int32_t tr_bank_cache_clear(tr_bank_cache *c) {
    if (!cache_valid(c)) return TR_MATH_INVALID;
    for (int i = 0; i < c->capacity; i++) entry_clear(c, i);
    c->hits = c->misses = c->puts = c->evictions = c->refreshes = c->invalidations = c->clock = 0;
    return TR_MATH_OK;
}
void tr_bank_cache_free(tr_bank_cache *c) {
    if (!cache_valid(c)) return;
    for (int i = 0; i < c->capacity; i++) entry_clear(c, i);
    c->magic = 0;
    free(c);
}
