/* COUNT-001: observed-domain constraint counts and witnesses. For a BOOL-001 predicate or a set of constraints
 * (implications A -> B, equivalences A <-> B, exclusions NOT (A AND B)), the exact number of points that satisfy,
 * violate or leave UNKNOWN each constraint, the satisfying points themselves (bounded by the output budget) and the
 * smallest violating points (counterexample witnesses), with a certificate that says exactly which points were covered,
 * on the CPU (math_count.c) and on a CUDA device (math_count_gpu.cu) with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Constraints (tr_count_set, read as a tr_count_set_view) --------------------------------------------------------
 * A set holds one predicate DAG over k conditions (BOOL-001 nodes: CONST, VAR, NOT, AND, OR, XOR; node i refers to
 * nodes < i; operand fields a node does not use are -1) and nc constraints (tr_count_constraint) whose operands a, b
 * are nodes of that DAG (shared subexpressions are evaluated once per word). With SQL's three-valued logic (Kleene's
 * connectives, math_bool.h) a constraint is a truth value at every point:
 *   HOLDS a           a itself (a BOOL-001 predicate)                      b = -1
 *   IMPLIES a b       NOT a OR b       violated where a is TRUE and b FALSE
 *   EQUIV a b         NOT (a XOR b)    violated where both are known and differ
 *   EXCLUDES a b      NOT (a AND b)    violated where both are TRUE
 * At a point a constraint is satisfied (TRUE), violated (FALSE) or UNKNOWN; UNKNOWN is never counted as satisfied or
 * as violated (an implication over a NULL premise is UNKNOWN, as SQL's WHERE NOT (a AND NOT b) would leave it).
 * The set stores per node deps (the conditions read at or below it), the schedule (the nodes some constraint reaches,
 * increasing) and per constraint its cone (the scheduled nodes its operands reach, increasing, closed under operands).
 *
 * ---- Domains (tr_count_domain) ---------------------------------------------------------------------------------------
 * OBSERVED: the points are the physical rows of a data scope (math_scope.h); only its selected rows count, each once.
 *   The k conditions are BOOL-001 row predicates (tr_bool_condition) evaluated once into value and valid bitmaps
 *   (tr_bool_masks_build, or the CUDA words of tr_mgpu_bool_masks adopted with tr_count_observe_adopt). The domain
 *   keeps an FND-007 record of the scope (tr_cert_record_scope over the columns its conditions read, with the exact
 *   selection bits): the table's lifetime token, revision, rows, selected rows, each column's encoding and NULL domain.
 * UNIVERSE: the points are the 2^k assignments of k two-valued conditions (bit i of the assignment a is condition i;
 *   BOOL-001's ASSIGNMENTS domain): a different result, explicitly labelled, for small k only (support budget 2^k).
 *   It is a property of the predicates alone; no data is read.
 *
 * ---- Results (tr_count_run / tr_count_adopt -> tr_count_answer) ------------------------------------------------------
 * Per constraint (tr_count_result): points (selected rows, or 2^k), satisfied, violated, unknown (they sum to points),
 * witness = the smallest violating point (TR_COUNT_NONE without a violation); with TR_COUNT_ROWS every satisfying point
 * in increasing order (points[rows_at .. rows_at + rows_n), rows_n = satisfied); with request.witnesses = w the
 * min(w, violated) smallest violating points in increasing order (witnesses[wit_at .. wit_at + wit_n)). Points are
 * physical row numbers (OBSERVED) or assignments (UNIVERSE). Offsets: constraints in order (tr_count_layout).
 *
 * ---- The certificate of an answer (tr_count_certificate) -------------------------------------------------------------
 * OBSERVED: kind TR_CERT_SNAPSHOT: the counts were taken exhaustively over every selected row of the recorded scope
 *   and over nothing else. points = the selected rows covered (the record keeps their exact bits); support = the
 *   conditions the constraints read (s of them); complete = the covered rows where every support condition is known;
 *   observed = the distinct assignments of the support among the complete rows; universe = 2^s. The rows say nothing
 *   about the universe - observed assignments that never occur: they were not checked, and the certificate never
 *   authorizes a statement about them (TR_COUNT_TARGET_UNIVERSE is refused with TR_COUNT_REFUSE_UNOBSERVED and the
 *   number of unobserved assignments unless observed = universe, i.e. every assignment of the support occurs among
 *   the complete covered rows; the value of a constraint at a point depends only on its support's values there).
 * UNIVERSE: kind TR_CERT_LAW: the counts were taken over all 2^k assignments by enumeration (points = complete = 2^k,
 *   observed = universe = 2^s): always fresh, no scope; it covers no table rows (TR_COUNT_TARGET_SNAPSHOT is refused
 *   with TR_COUNT_REFUSE_DOMAIN).
 * Freshness (tr_count_answer_fresh): the record against the current scope (FND-007 tr_cert_record_fresh, exact match:
 *   identity, revision, rows, selection, null_domain, columns, with their exact reasons in a tr_cert_staleness).
 * Authorization (tr_count_answer_authorize): TR_COUNT_TARGET_SNAPSHOT (a statement about the covered rows of the
 *   current scope) needs an OBSERVED certificate that is fresh (else TR_COUNT_REFUSE_STALE: TR_MATH_STALE with the
 *   reasons); TR_COUNT_TARGET_UNIVERSE (a statement about every assignment of the support) needs a UNIVERSE
 *   certificate, or an OBSERVED one that is fresh and observed every assignment. Other refusals: TR_MATH_DOMAIN_ERROR.
 *   The coverage (complete, observed) is computed on the host from the domain's condition words in both paths.
 *
 * ---- Execution ------------------------------------------------------------------------------------------------------
 * CPU (tr_count_run): one pass over the 64-point words: the scheduled nodes once per word with bitwise operations on a
 * value and a valid word (tr_bool_eval_base), each constraint's words (tr_count_word), popcounts, the first violating
 * point; then, when points or witnesses are requested and the output need is admitted, a second pass that writes them.
 * CUDA: tr_mgpu_count_counts (work items (constraint, word slice) evaluate the constraint's cone per word, count with
 * popcounts, add their counts with one atomic each and their smallest violating point with an atomic minimum), then
 * tr_mgpu_count_extract (per (constraint, chunk of TR_COUNT_CHUNK words): the chunk's satisfying and violating counts
 * in one slot each; their exclusive prefix on the host; then each chunk writes its points at their ranks: one output
 * slot per point, independent of thread order); tr_count_adopt checks the readback and builds the answer.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated ---------------------------------------------------
 *   set: variables = k, circuit = nnodes, output = nc, bytes = the object (tr_count_set_need).
 *   domain OBSERVED: variables = k, bytes = BOOL-001's masks (8 nw (1 + 2k), or the device copies) + the record;
 *     UNIVERSE: variables = k, support = 2^k.
 *   run (tr_count_run_need): variables = k, circuit = nnodes, support = 2^k on UNIVERSE, output = nc, states = 2^s when
 *     the observed assignments are counted with a bitmap (s <= TR_COUNT_PATTERN_BITS), bytes = node words (16 per node),
 *     results, the coverage scratch (the bitmap, or 8 per selected row to sort), the answer and its record. After the
 *     counts the outputs are checked (TR_MATH_BUDGET reports the exact need, nothing is written): output = nc +
 *     sum of satisfied (TR_COUNT_ROWS) + sum of min(w, violated), bytes += 8 per point returned.
 *   CUDA: bytes = the device copies (selection, condition words, nodes, schedule, cones, constraints), the node words of
 *     every work item, the accumulators, and for the extraction the chunk counts, prefixes and outputs.
 * Representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED for k > 64 (OBSERVED) or > 63
 * (UNIVERSE), nnodes > TR_BOOL_MAX_NODES, nc > TR_COUNT_MAX_CONSTRAINTS.
 *
 * Errors: TR_MATH_INVALID (-1, malformed input or a readback that is not this domain's), _MEMORY (-2), _BUDGET (-5),
 * TR_MATH_DOMAIN_ERROR (-6: a set over another number of conditions than the domain; a refused authorization),
 * TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12), TR_MATH_STALE (-13). Objects are immutable once built and
 * individually owned; constructors return NULL and set *error, leaving their inputs unchanged. */
#ifndef TORBLES_MATH_COUNT_H
#define TORBLES_MATH_COUNT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bool.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_COUNT_ABI 1
#define TR_COUNT_NONE UINT64_MAX
#define TR_COUNT_MAX_CONSTRAINTS (1 << 16)
#define TR_COUNT_PATTERN_BITS 24             /* supports up to 24 conditions: observed assignments by a 2^s bitmap */
#define TR_COUNT_CHUNK 16                    /* words per extraction chunk on the device */

enum { TR_COUNT_HOLDS = 1, TR_COUNT_IMPLIES = 2, TR_COUNT_EQUIV = 3, TR_COUNT_EXCLUDES = 4 };   /* constraint kinds */
enum { TR_COUNT_OBSERVED = 1, TR_COUNT_UNIVERSE = 2 };                                       /* domains */
enum { TR_COUNT_ROWS = 1 };                                                                  /* request flags */
enum { TR_COUNT_TARGET_SNAPSHOT = 1, TR_COUNT_TARGET_UNIVERSE = 2 };                         /* authorization targets */
enum { TR_COUNT_REFUSE_NONE = 0, TR_COUNT_REFUSE_STALE = 1, TR_COUNT_REFUSE_UNOBSERVED = 2, TR_COUNT_REFUSE_DOMAIN = 3,
       TR_COUNT_REFUSE_NO_SCOPE = 4 };
enum { TR_COUNT_NEED_CPU = 0, TR_COUNT_NEED_DEVICE = 1, TR_COUNT_NEED_ADOPT = 2 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { int32_t kind, a, b, reserved; } tr_count_constraint;   /* 16 bytes: TR_COUNT_*, operand nodes */

typedef struct {
    int32_t flags, reserved;          /* TR_COUNT_ROWS */
    uint64_t witnesses;               /* the smallest violating points returned per constraint */
} tr_count_request;                   /* 16 bytes */

typedef struct {
    int32_t kind, a, b, reserved;     /* the constraint */
    uint64_t points;                  /* selected rows (OBSERVED) or 2^k (UNIVERSE) */
    uint64_t satisfied, violated, unknown;
    uint64_t witness;                 /* the smallest violating point, or TR_COUNT_NONE */
    uint64_t rows_at, rows_n;         /* its satisfying points in the points output (TR_COUNT_ROWS) */
    uint64_t wit_at, wit_n;           /* its smallest violating points in the witnesses output */
} tr_count_result;                    /* 88 bytes */

/* a set as the CPU code and the kernels read it (pointers into the set; valid while it lives) */
typedef struct {
    const tr_bool_node *nodes;        /* nnodes */
    const uint64_t *deps;             /* nnodes: the conditions read at or below each node */
    const uint32_t *schedule;         /* nsched nodes some constraint reaches, increasing */
    const tr_count_constraint *constraints;   /* nc */
    const uint32_t *cones;            /* the cones of the constraints, concatenated */
    const uint64_t *cone_at;          /* nc + 1 offsets into cones */
    int32_t nnodes, nsched, k, nc;
} tr_count_set_view;                  /* 64 bytes */

typedef struct {
    int32_t k, nnodes, nsched, nc;
    uint64_t support;                 /* the conditions the constraints read */
    uint64_t cones;                   /* total cone length */
    uint64_t bytes;
} tr_count_set_info;                  /* 40 bytes */

typedef struct {
    uint64_t points, nw, selected, bytes;   /* physical rows or 2^k; words; points that count; bytes retained */
    int32_t domain, k;                      /* TR_COUNT_OBSERVED / _UNIVERSE */
    int32_t has_unknown, ncolumns;          /* some condition UNKNOWN on a selected row; columns recorded */
} tr_count_domain_info;                     /* 48 bytes */

typedef struct {
    int32_t kind;                     /* TR_CERT_SNAPSHOT (OBSERVED) or TR_CERT_LAW (UNIVERSE) */
    int32_t domain;                   /* TR_COUNT_OBSERVED / TR_COUNT_UNIVERSE */
    int32_t backend;                  /* TR_CERT_BY_CPU / TR_CERT_BY_DEVICE: who evaluated the constraints */
    int32_t support_size;             /* s */
    uint64_t support;                 /* the conditions the constraints read (bit i: condition i) */
    uint64_t points;                  /* the points covered: the selected rows, or 2^k */
    uint64_t complete;                /* covered points where every support condition is known */
    uint64_t observed;                /* distinct assignments of the support among the complete points */
    uint64_t universe;                /* 2^s (TR_COUNT_NONE for s = 64) */
    uint64_t rows;                    /* the scope's physical rows, or 2^k */
    int32_t nconstraints, exact_selection;   /* constraints; 1: the record keeps the exact selection bits */
} tr_count_certificate;               /* 72 bytes */

typedef struct {
    const tr_count_result *results;   /* nc */
    const uint64_t *points;           /* points_total satisfying points (TR_COUNT_ROWS) */
    const uint64_t *witnesses;        /* witnesses_total smallest violating points */
    uint64_t points_total, witnesses_total;
    int32_t nc, flags;                /* constraints; the request's flags */
} tr_count_answer_view;               /* 48 bytes */

typedef struct {
    int32_t authorized, reason;       /* 1 / 0; TR_COUNT_REFUSE_* */
    int32_t target, domain;           /* the target asked for; the certificate's domain */
    uint64_t unobserved;              /* universe - observed (TR_COUNT_NONE for s = 64 and unobserved assignments) */
    tr_cert_staleness stale;          /* FND-007: the reasons of TR_COUNT_REFUSE_STALE */
} tr_count_decision;                  /* 56 bytes */

typedef struct tr_count_set_s tr_count_set;
typedef struct tr_count_domain_s tr_count_domain;
typedef struct tr_count_answer_s tr_count_answer;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* the truth words of one constraint from its operands' words (see Constraints): SQL three-valued logic */
TR_NUM_HD void tr_count_word(int32_t kind, uint64_t va, uint64_t ka, uint64_t vb, uint64_t kb, uint64_t *v, uint64_t *k) {
    uint64_t x = 0, y = 0;
    switch (kind) {
    case TR_COUNT_IMPLIES:
        tr_bool_apply(TR_BOOL_NOT, va, ka, 0, 0, &x, &y);
        tr_bool_apply(TR_BOOL_OR, x, y, vb, kb, v, k);
        return;
    case TR_COUNT_EQUIV:
        tr_bool_apply(TR_BOOL_XOR, va, ka, vb, kb, &x, &y);
        tr_bool_apply(TR_BOOL_NOT, x, y, 0, 0, v, k);
        return;
    case TR_COUNT_EXCLUDES:
        tr_bool_apply(TR_BOOL_AND, va, ka, vb, kb, &x, &y);
        tr_bool_apply(TR_BOOL_NOT, x, y, 0, 0, v, k);
        return;
    default:
        *v = va; *k = ka;
        return;
    }
}
/* the words of constraint j at word w once its operands' node words are in val (val[2 i], val[2 i + 1]): satisfied,
 * violated and UNKNOWN points among the points that count (sel) */
TR_NUM_HD void tr_count_classes(const tr_count_constraint *c, const uint64_t *val, uint64_t sel, uint64_t *sat,
                                uint64_t *vio, uint64_t *unk) {
    uint64_t va = val[2 * (uint64_t)c->a], ka = val[2 * (uint64_t)c->a + 1];
    uint64_t vb = c->b >= 0 ? val[2 * (uint64_t)c->b] : 0, kb = c->b >= 0 ? val[2 * (uint64_t)c->b + 1] : 0;
    uint64_t v = 0, k = 0;
    tr_count_word(c->kind, va, ka, vb, kb, &v, &k);
    *sat = v & k & sel;
    *vio = k & ~v & sel;
    *unk = sel & ~k;
}
/* the cone of constraint j as a BOOL-001 program view (its schedule; the root is its last node) */
TR_NUM_HD tr_bool_program_view tr_count_cone_view(const tr_count_set_view *s, int32_t j) {
    tr_bool_program_view p;
    p.nodes = s->nodes; p.deps = s->deps;
    p.schedule = s->cones + s->cone_at[j];
    p.nsched = (int32_t)(s->cone_at[j + 1] - s->cone_at[j]);
    p.nnodes = s->nnodes; p.k = s->k;
    p.root = p.nsched ? (int32_t)p.schedule[p.nsched - 1] : 0;
    return p;
}
/* the union schedule as a BOOL-001 program view */
TR_NUM_HD tr_bool_program_view tr_count_schedule_view(const tr_count_set_view *s) {
    tr_bool_program_view p;
    p.nodes = s->nodes; p.deps = s->deps; p.schedule = s->schedule; p.nsched = s->nsched;
    p.nnodes = s->nnodes; p.k = s->k;
    p.root = s->nsched ? (int32_t)s->schedule[s->nsched - 1] : 0;
    return p;
}

/* the output layout of a request: offsets and lengths of every constraint's points and witnesses from its counts,
 * and the totals; 0 or TR_MATH_INVALID (malformed request or counts) */
static inline int32_t tr_count_layout(tr_count_result *r, int32_t nc, const tr_count_request *q, uint64_t *points_total,
                                      uint64_t *witnesses_total) {
    if (!q || (q->flags & ~TR_COUNT_ROWS) || q->reserved || nc < 0 || (nc && !r)) return -1;
    uint64_t pt = 0, wt = 0;
    for (int32_t j = 0; j < nc; j++) {
        if (tr_budget_add(tr_budget_add(r[j].satisfied, r[j].violated), r[j].unknown) != r[j].points) return -1;
        r[j].rows_at = pt;
        r[j].rows_n = (q->flags & TR_COUNT_ROWS) ? r[j].satisfied : 0;
        r[j].wit_at = wt;
        r[j].wit_n = r[j].violated < q->witnesses ? r[j].violated : q->witnesses;
        pt = tr_budget_add(pt, r[j].rows_n);
        wt = tr_budget_add(wt, r[j].wit_n);
    }
    if (points_total) *points_total = pt;
    if (witnesses_total) *witnesses_total = wt;
    return 0;
}

/* 0, or TR_MATH_INVALID / TR_MATH_UNSUPPORTED for a malformed set view: the nodes and the schedule as BOOL-001 checks
 * them, every constraint well formed with its operands scheduled, every cone increasing, inside the schedule, closed
 * under operands and holding its operands. O(nodes + cones x log) without allocating. */
static inline int32_t tr_count_set_check(const tr_count_set_view *s) {
    if (!s || s->nc < 1 || !s->constraints || !s->cone_at || s->k < 0) return -1;
    if (s->nc > TR_COUNT_MAX_CONSTRAINTS) return -12;
    tr_bool_program_view p = tr_count_schedule_view(s);
    int32_t rc = tr_bool_program_check(&p);
    if (rc) return rc;
    if (s->cone_at[0] != 0) return -1;
    for (int32_t j = 0; j < s->nc; j++) {
        const tr_count_constraint *c = &s->constraints[j];
        if (c->reserved || c->kind < TR_COUNT_HOLDS || c->kind > TR_COUNT_EXCLUDES || c->a < 0 || c->a >= s->nnodes ||
            (c->kind == TR_COUNT_HOLDS ? c->b != -1 : (c->b < 0 || c->b >= s->nnodes))) return -1;
        uint64_t lo = s->cone_at[j], hi = s->cone_at[j + 1];
        if (hi < lo || hi - lo > (uint64_t)s->nsched || (hi > lo && !s->cones)) return -1;
        int has_a = 0, has_b = c->b < 0;
        for (uint64_t i = lo; i < hi; i++) {
            uint32_t n = s->cones[i];
            if (n >= (uint32_t)s->nnodes || (i > lo && n <= s->cones[i - 1])) return -1;
            if ((int32_t)n == c->a) has_a = 1;
            if ((int32_t)n == c->b) has_b = 1;
            int32_t lo2 = 0, hi2 = s->nsched - 1, in = 0;       /* inside the schedule */
            while (lo2 <= hi2) {
                int32_t mid = lo2 + (hi2 - lo2) / 2;
                if (s->schedule[mid] == n) { in = 1; break; }
                if (s->schedule[mid] < n) lo2 = mid + 1; else hi2 = mid - 1;
            }
            if (!in) return -1;
            const tr_bool_node *node = &s->nodes[n];          /* closed under operands */
            int32_t ops[2] = {node->op == TR_BOOL_CONST || node->op == TR_BOOL_VAR ? -1 : node->a,
                              node->op >= TR_BOOL_AND ? node->b : -1};
            for (int t = 0; t < 2; t++) {
                if (ops[t] < 0) continue;
                uint64_t a = lo, b = i;                           /* binary search in the cone before i */
                int found = 0;
                while (a < b) {
                    uint64_t mid = a + (b - a) / 2;
                    if (s->cones[mid] == (uint32_t)ops[t]) { found = 1; break; }
                    if (s->cones[mid] < (uint32_t)ops[t]) a = mid + 1; else b = mid;
                }
                if (!found) return -1;
            }
        }
        if (!has_a || !has_b) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ the CUDA work split (shared with the host needs) */
#define TR_COUNT_WORK (UINT64_C(1) << 14)        /* about this many counting work items (constraint, slice) */
#define TR_COUNT_MIN_WORDS 4                      /* at least this many words per counting work item */
#define TR_COUNT_SCRATCH (UINT64_C(1) << 26)      /* at most this many bytes of node words on the device */
/* slices per constraint of the counting kernel: work items = nc x slices, each with 16 bytes per node */
static inline uint64_t tr_count_slices(uint64_t nw, int32_t nc, int32_t nnodes) {
    uint64_t q = nc > 0 ? (uint64_t)nc : 1, per = tr_budget_mul((uint64_t)(nnodes > 0 ? nnodes : 1), 16);
    uint64_t s = TR_COUNT_WORK / q, cap = (nw + TR_COUNT_MIN_WORDS - 1) / TR_COUNT_MIN_WORDS;
    uint64_t room = TR_COUNT_SCRATCH / per / q;
    if (s > cap) s = cap;
    if (s > room) s = room;
    return s ? s : 1;
}
/* words per extraction chunk: TR_COUNT_CHUNK, or more when the node words of nc x chunks work items would pass
 * TR_COUNT_SCRATCH; and the chunks of nw words */
static inline uint64_t tr_count_chunk_words(uint64_t nw, int32_t nc, int32_t nnodes) {
    uint64_t q = nc > 0 ? (uint64_t)nc : 1, per = tr_budget_mul((uint64_t)(nnodes > 0 ? nnodes : 1), 16);
    uint64_t room = TR_COUNT_SCRATCH / per / q, cw = TR_COUNT_CHUNK;
    if (!room) room = 1;
    if ((nw + cw - 1) / cw > room) cw = (nw + room - 1) / room;
    return cw ? cw : 1;
}
static inline uint64_t tr_count_chunks(uint64_t nw, uint64_t cw) { return (nw + cw - 1) / cw; }
/* the device copies shared by both kernels: the domain's words and the set */
static inline uint64_t tr_count_device_base(const tr_bool_domain *d, const tr_count_set_view *s) {
    uint64_t bytes = 0;
    if (d->kind == TR_BOOL_ROWS) {
        bytes = tr_budget_mul(d->nw, 8);
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)d->k, d->nw), d->valid ? 16 : 8));
    }
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)s->nnodes, sizeof(tr_bool_node)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)s->nsched, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)s->nc, sizeof(tr_count_constraint) + 8 + sizeof(tr_count_result)));
    bytes = tr_budget_add(bytes, tr_budget_mul(s->cone_at[s->nc], 4));
    return bytes;
}
/* the device bytes of tr_mgpu_count_counts */
static inline uint64_t tr_count_device_bytes(const tr_bool_domain *d, const tr_count_set_view *s) {
    uint64_t items = tr_budget_mul((uint64_t)s->nc, tr_count_slices(d->nw, s->nc, s->nnodes));
    uint64_t bytes = tr_count_device_base(d, s);
    bytes = tr_budget_add(bytes, tr_budget_mul(items, tr_budget_mul((uint64_t)s->nnodes, 16)));   /* node words */
    return tr_budget_add(bytes, tr_budget_mul((uint64_t)s->nc, 32));                            /* accumulators */
}
/* the device bytes of tr_mgpu_count_extract (outputs of points_total + witnesses_total points) */
static inline uint64_t tr_count_extract_device_bytes(const tr_bool_domain *d, const tr_count_set_view *s,
                                                     uint64_t points_total, uint64_t witnesses_total) {
    uint64_t cw = tr_count_chunk_words(d->nw, s->nc, s->nnodes);
    uint64_t chunks = tr_count_chunks(d->nw, cw), items = tr_budget_mul((uint64_t)s->nc, chunks);
    uint64_t bytes = tr_count_device_base(d, s);
    bytes = tr_budget_add(bytes, tr_budget_mul(items, tr_budget_mul((uint64_t)s->nnodes, 16)));   /* node words */
    bytes = tr_budget_add(bytes, tr_budget_mul(items, 64));     /* chunk counts and prefixes, and their host copies */
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)s->nc, 32));                            /* limits, offsets */
    return tr_budget_add(bytes, tr_budget_mul(tr_budget_add(points_total, witnesses_total), 8));
}

/* ------------------------------------------------------------------ constraint sets (libalg) */
tr_count_set *tr_count_set_create(int32_t k, const tr_bool_node *nodes, int32_t nnodes, const tr_count_constraint *c,
                                  int32_t nc, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_count_set_need(int32_t k, int32_t nnodes, int32_t nc, tr_math_need *);
int32_t tr_count_set_get_view(const tr_count_set *, tr_count_set_view *);
int32_t tr_count_set_get_info(const tr_count_set *, tr_count_set_info *);
void tr_count_set_free(tr_count_set *);

/* ------------------------------------------------------------------ domains (libalg) */
/* OBSERVED: the k conditions over the selected rows of the scope (budget NULL = defaults) */
tr_count_domain *tr_count_observe(const tr_scope *, const tr_bool_condition *, int32_t k, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
/* OBSERVED from condition words computed elsewhere (the readback of tr_mgpu_bool_masks: k * nw words each; valid
 * NULL: every condition known), checked canonical by tr_bool_masks_adopt */
tr_count_domain *tr_count_observe_adopt(const tr_scope *, const tr_bool_condition *, int32_t k, const uint64_t *value,
                                        const uint64_t *valid, const tr_math_budget *, tr_math_refusal *,
                                        int32_t *error);
/* UNIVERSE: the 2^k assignments */
tr_count_domain *tr_count_universe(int32_t k, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the needs of tr_count_observe (TR_COUNT_NEED_CPU), of building the words on the device (_DEVICE) or of
 * tr_count_observe_adopt (_ADOPT); scope NULL and conditions NULL: tr_count_universe */
int32_t tr_count_domain_need(const tr_scope *, const tr_bool_condition *, int32_t k, int32_t target, tr_math_need *);
int32_t tr_count_domain_get_info(const tr_count_domain *, tr_count_domain_info *);
int32_t tr_count_domain_get_domain(const tr_count_domain *, tr_bool_domain *);
const tr_cert_record *tr_count_domain_get_record(const tr_count_domain *);     /* borrowed; NULL for UNIVERSE */
int32_t tr_count_domain_fresh(const tr_count_domain *, const tr_scope *current, tr_cert_staleness *);
void tr_count_domain_free(tr_count_domain *);

/* ------------------------------------------------------------------ counts and answers (libalg) */
/* the needs checked before tr_count_run (TR_COUNT_NEED_CPU) or tr_mgpu_count_counts (TR_COUNT_NEED_DEVICE) */
int32_t tr_count_run_need(const tr_count_domain *, const tr_count_set *, const tr_count_request *, int32_t target,
                          tr_math_need *);
tr_count_answer *tr_count_run(const tr_count_domain *, const tr_count_set *, const tr_count_request *,
                              const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the counts alone on views (the CPU reference of tr_mgpu_count_counts): out[nc] with no points requested */
int32_t tr_count_counts_view(const tr_bool_domain *, const tr_count_set_view *, const tr_math_budget *,
                             tr_count_result *out, tr_math_refusal *);
/* the layout of a request over counts (the exported tr_count_layout; also checks the counts' partitions) */
int32_t tr_count_plan(tr_count_result *results, int32_t nc, const tr_count_request *, uint64_t *points_total,
                      uint64_t *witnesses_total);
/* An answer from results computed elsewhere (the readback of tr_mgpu_count_counts and tr_mgpu_count_extract) after
 * consistency checks: the constraints are the set's, the counts partition the domain's points, the layout is the
 * request's, every point returned is a point that counts, the points of a constraint increase, its first witness is
 * its smallest violating point. backend TR_CERT_BY_*. The coverage of the certificate is computed here. */
tr_count_answer *tr_count_adopt(const tr_count_domain *, const tr_count_set *, const tr_count_request *,
                                const tr_count_result *results, const uint64_t *points, const uint64_t *witnesses,
                                int32_t backend, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_count_answer_get_view(const tr_count_answer *, tr_count_answer_view *);
int32_t tr_count_answer_certificate(const tr_count_answer *, tr_count_certificate *);
const tr_cert_record *tr_count_answer_get_record(const tr_count_answer *);     /* borrowed; NULL for UNIVERSE */
int32_t tr_count_answer_fresh(const tr_count_answer *, const tr_scope *current, tr_cert_staleness *);
int32_t tr_count_answer_authorize(const tr_count_answer *, int32_t target, const tr_scope *current,
                                  tr_count_decision *);
void tr_count_answer_free(tr_count_answer *);

/* ------------------------------------------------------------------ the CUDA library (math_count_gpu.cu, libmathgpu)
 * Views of host memory, copied to the device and budgeted (bytes) before anything is allocated:
 *   tr_mgpu_count_counts    the counts and the smallest violating point of every constraint (results[nc]; no points)
 *   tr_mgpu_count_extract   with the counts of tr_mgpu_count_counts: the layout of the request (results[] filled as
 *                           tr_count_layout does, checked against the device's chunk counts), the satisfying points
 *                           (TR_COUNT_ROWS) and the smallest violating points (points[points_total],
 *                           witnesses[witnesses_total]); output = nc + the points returned */
int tr_mgpu_count_counts(const tr_bool_domain *, const tr_count_set_view *, const tr_math_budget *,
                         tr_count_result *results, tr_math_refusal *);
int tr_mgpu_count_extract(const tr_bool_domain *, const tr_count_set_view *, const tr_count_request *,
                          tr_count_result *results, const tr_math_budget *, uint64_t *points, uint64_t *witnesses,
                          tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
