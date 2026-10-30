/* Native physical choice between an operation chain and a combined algebraic
 * system. SPDX-License-Identifier: Apache-2.0
 *
 * The chain is the original postfix bytecode, executed for each surviving row.
 * The combined system is an exactly normalized degree-2 polynomial executed on
 * bit planes. The hybrid composes checked local regions in a shared scalar DAG.
 * These are alternative runtime plans, not relationship-search paths.
 * Relative-work costs are deliberately explicit, bounded and uncalibrated.
 */
#include "relationships.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int fits64(__int128 x) { return x >= INT64_MIN && x <= INT64_MAX; }

static int add(tr_poly *p, int a, int b, __int128 coefficient) {
    if (!coefficient) return 1;
    if (b >= 0 && a > b) { int x = a; a = b; b = x; }
    for (int i = 0; i < p->n; ++i) if (p->terms[i].a == a && p->terms[i].b == b) {
        coefficient += p->terms[i].coefficient;
        if (!fits64(coefficient)) return 0;
        if (!coefficient) p->terms[i] = p->terms[--p->n];
        else p->terms[i].coefficient = (int64_t)coefficient;
        return 1;
    }
    if (!fits64(coefficient) || p->n == TR_MAX_TERMS) return 0;
    p->terms[p->n++] = (tr_term){a, b, (int64_t)coefficient};
    return 1;
}

static int combine_poly(const tr_poly *left, const tr_poly *right, int op, tr_poly *out) {
    memset(out, 0, sizeof *out);
    if (op == M_ADD || op == M_SUB) {
        for (int i = 0; i < left->n; ++i)
            if (!add(out, left->terms[i].a, left->terms[i].b, left->terms[i].coefficient)) return 0;
        for (int i = 0; i < right->n; ++i)
            if (!add(out, right->terms[i].a, right->terms[i].b,
                     (__int128)right->terms[i].coefficient * (op == M_SUB ? -1 : 1))) return 0;
        return 1;
    }
    for (int i = 0; i < left->n; ++i) for (int j = 0; j < right->n; ++j) {
        const tr_term *a = &left->terms[i], *b = &right->terms[j]; int vars[4], degree = 0;
        if (a->a >= 0) vars[degree++] = a->a;
        if (a->b >= 0) vars[degree++] = a->b;
        if (b->a >= 0) vars[degree++] = b->a;
        if (b->b >= 0) vars[degree++] = b->b;
        if (degree > 2 || !add(out, degree ? vars[0] : -1, degree == 2 ? vars[1] : -1,
                              (__int128)a->coefficient * b->coefficient)) return 0;
    }
    return 1;
}

/* Parse the actual runtime chain independently of the normalized polynomial.
 * Polynomial overflow/degree growth makes combination unavailable, not chaining.
 */
static int analyze_chain(const atable_t *t, const qplan_t *q, tr_poly *poly,
                         int *combined, int *operations, int *peak, uint64_t *inputs) {
    if (q->nmeas < 1 || q->nmeas > 48) return 0;
    if (!tr_measure_valid(q, t->k)) return 0;
    tr_poly stack[24], registers[48]; unsigned char exact[24] = {0}, rexact[48] = {0}; int sp = 0;
    *operations = *peak = 0; *inputs = 0;
    for (int i = 0; i < q->nmeas; ++i) {
        const minstr_t *m = &q->meas[i];
        if (m->op == M_FCOL || m->op == M_CONST) {
            if (sp == 24) return 0;
            memset(&stack[sp], 0, sizeof stack[sp]); exact[sp] = 1;
            if (m->op == M_FCOL) {
                if (m->arg < 0 || m->arg >= t->k) return 0;
                *inputs |= UINT64_C(1) << m->arg;
                add(&stack[sp], m->arg, -1, 1);
            } else add(&stack[sp], -1, -1, m->c);
            ++sp; if (sp > *peak) *peak = sp;
        } else if (m->op == M_SAVE) {
            rexact[m->arg] = exact[sp-1];
            if (exact[sp-1]) registers[m->arg] = stack[sp-1];
        } else if (m->op == M_LOAD) {
            exact[sp] = rexact[m->arg];
            if (rexact[m->arg]) stack[sp] = registers[m->arg];
            ++sp; if (sp > *peak) *peak = sp;
        } else if (m->op == M_ADD || m->op == M_SUB || m->op == M_MUL) {
            if (sp < 2) return 0;
            tr_poly next; int ok = exact[sp-2] && exact[sp-1] &&
                combine_poly(&stack[sp-2], &stack[sp-1], m->op, &next);
            --sp; exact[sp-1] = (unsigned char)ok;
            if (ok) stack[sp-1] = next;
            ++*operations;
        } else return 0; /* attributes require their own algebra module */
    }
    if (sp != 1) return 0;
    *combined = exact[0];
    if (*combined) *poly = stack[0];
    return 1;
}

static int emit(qplan_t *q, int op, int arg, int64_t c) {
    if (q->nmeas == 48) return 0;
    q->meas[q->nmeas++] = (minstr_t){op, arg, c}; return 1;
}

static int synthesize_chain(qplan_t *q) {
    if (q->nmeas) return 1;
    if (q->agg == A_BIL)
        return emit(q, M_FCOL, q->bil_a, 0) && emit(q, M_FCOL, q->bil_b, 0) && emit(q, M_MUL, 0, 0);
    if (q->agg != A_LIN || q->nlin < 1 || q->nlin > 8) return 0;
    for (int i = 0; i < q->nlin; ++i) {
        if (!emit(q, M_FCOL, q->lin_col[i], 0)) return 0;
        if (q->lin_coef[i] != 1 && (!emit(q, M_CONST, 0, q->lin_coef[i]) || !emit(q, M_MUL, 0, 0))) return 0;
        if (i && !emit(q, M_ADD, 0, 0)) return 0;
    }
    return 1;
}

static int write_poly(qplan_t *q, const tr_poly *p) {
    q->npl = q->npb = 0; q->p_const = q->p_const_hi = 0;
    memset(q->pl_coef_hi, 0, sizeof q->pl_coef_hi);
    memset(q->pb_coef_hi, 0, sizeof q->pb_coef_hi);
    for (int i = 0; i < p->n; ++i) {
        const tr_term *term = &p->terms[i];
        if (term->a < 0) q->p_const = term->coefficient;
        else if (term->b < 0) {
            int j = q->npl++; if (j == 8) return 0;
            q->pl_col[j] = term->a; q->pl_coef[j] = term->coefficient;
        } else {
            int j = q->npb++; if (j == 8) return 0;
            q->pb_a[j] = term->a; q->pb_b[j] = term->b; q->pb_coef[j] = term->coefficient;
        }
    }
    return 1;
}

/* Shape validation is independent of selectivity sampling. GPU planning has no
 * alternative to compare, but must still reject malformed source IR/keys. */
static int valid_shape(const atable_t *t, const qplan_t *q) {
    if (t->k < 1 || t->k > MAXCOL || q->nprog < 0 || q->nprog > 1024 ||
        q->nkey < 0 || q->nkey > MAXKEY || q->nsemi < 0 || q->nsemi > 8 ||
        q->nattr < 0 || q->nattr > 8 || q->nranges > 4096 ||
        (q->nranges && !q->ranges) || q->xmode < 0 || q->xmode > 2 ||
        (q->xmode && !q->xrows)) return 0;
    for (int c = 0; c < t->k; ++c) if (t->width[c] < 1 || t->width[c] > 32) return 0;
    int sp = 0;
    for (int i = 0; i < q->nprog; ++i) {
        const pinstr_t *p = &q->prog[i];
        if (p->op == P_AND || p->op == P_OR || p->op == P_XOR) {
            if (sp < 2) return 0;
            --sp;
        } else if (p->op == P_NOT) { if (!sp) return 0; }
        else {
            if (p->op < P_TRUE || p->op > P_XOR ||
                (p->op != P_TRUE && (p->col < 0 || p->col >= t->k))) return 0;
            if ((p->op == P_CUBE || p->op == P_CLT || p->op == P_CGE) &&
                (!p->b || (t->width[p->col] < 32 && (p->b >> t->width[p->col])))) return 0;
            if (p->op == P_PREFIX && p->b > (uint32_t)t->width[p->col]) return 0;
            if (p->op == P_BIT && p->a >= (uint32_t)t->width[p->col]) return 0;
            if (++sp > 48) return 0;
        }
    }
    if (q->nprog && sp != 1) return 0;
    for (int i = 0; i < q->nkey; ++i) {
        int kind = q->key_kind[i], c = q->key_arg[i];
        if ((kind != 0 && kind != 2 && kind != 3) || c < 0 || c >= t->k ||
            (kind == 2 && (q->key_nbits[i] < 1 || q->key_nbits[i] > t->width[c])) ||
            (kind == 3 && (!q->key_mask[i] ||
                (t->width[c] < 32 && (q->key_mask[i] >> t->width[c]))))) return 0;
    }
    return 1;
}

int32_t tr_plan_system(const atable_t *t, qplan_t *q, int32_t backend, int32_t mode,
                       tr_system_trace *trace) {
    if (!t || !q || !trace) return TR_INVALID;
    memset(trace, 0, sizeof *trace);
    trace->hybrid_cost = -1;
    trace->original_agg = trace->selected_agg = q->agg;
    if (mode < TR_SYSTEM_AUTO || mode > TR_SYSTEM_HYBRID || backend < TR_BACKEND_CPU || backend > TR_BACKEND_PLAIN) {
        trace->reason = 7; return TR_INVALID;
    }
    if (backend == TR_BACKEND_GPU) trace->gpu_profile_skipped = 1;
    if (backend == TR_BACKEND_PLAIN) { trace->reason = 5; return TR_OK; }
    if (q->wide) { trace->reason = 4; return TR_OK; }
    if (q->ext_on || q->nattr || (q->agg != A_LIN && q->agg != A_BIL && q->agg != A_GENERAL &&
        q->agg != A_POLYG && q->agg != A_MM)) { trace->reason = 2; return TR_OK; }
    if (backend == TR_BACKEND_GPU &&
        (mode == TR_SYSTEM_CHAIN || q->widekey || q->nsemi || q->nkey > 6)) {
        trace->reason = 3; return TR_OK;
    }
    qplan_t candidate = *q;
    if (!synthesize_chain(&candidate)) { trace->reason = 2; return TR_OK; }
    tr_poly poly; int combined = 0; uint64_t inputs = 0;
    if (!analyze_chain(t, &candidate, &poly, &combined, &trace->chain_ops, &trace->stack_peak, &inputs)) {
        trace->reason = 2; return TR_OK;
    }
    qplan_t fused = candidate;
    if (combined && !write_poly(&fused, &poly)) combined = 0;
    if (backend == TR_BACKEND_GPU) {
        /* GPU has no general-chain executor. Normalize/check the one supported
         * combined representation; do not sample rows to rank a nonexistent
         * competing candidate. CPU fallback remains the ordinary caller's job. */
        trace->chain_cost = trace->combine_cost = -1;
        trace->word_combine_cost = trace->matrix_combine_cost = -1;
        int planes = 0;
        if (combined && valid_shape(t, &candidate)) {
            for (int i = 0; i < poly.n; ++i) {
                const tr_term *term = &poly.terms[i];
                if (term->a >= 0)
                    planes += t->width[term->a] * (term->b >= 0 ? t->width[term->b] : 1);
            }
            if (!candidate.out_key || (candidate.nkey > 0 && planes <= MAXMP)) {
                fused.agg = candidate.out_key ? A_MM : A_POLYG;
                trace->choice = TR_SYSTEM_COMBINE;
                trace->selected_agg = fused.agg;
                *q = fused;
            }
        }
        trace->reason = 3;
        return TR_OK;
    }
    tr_system_sample sample;
    qplan_t hybrid = candidate; tr_dag_stats dag_stats;
    int has_hybrid = tr_dag_optimize(t, &candidate, &hybrid, &dag_stats) == TR_OK;
    if (has_hybrid) {
        trace->dag_nodes = dag_stats.nodes; trace->hybrid_ops = dag_stats.operations;
        trace->hybrid_registers = dag_stats.registers; trace->hybrid_regions = dag_stats.regions;
    }
    int rc = at_relationship_sample(t, &candidate, &sample);
    if (rc != TR_OK) { trace->reason = 2; return TR_OK; }
    trace->sample_words = (int32_t)sample.words; trace->nonempty_words = (int32_t)sample.nonempty;
    trace->sampled_rows = sample.rows; trace->passing_rows = sample.passing;
    trace->groups_per_word = sample.nonempty ? sample.groups/sample.nonempty : 0;

    /* A unit is relative native work, not nanoseconds. The chain includes code
     * decoding, sparse extraction or dense transpose, stack/intermediate traffic,
     * scalar operations and per-row grouping. The combined system includes plane
     * decoding, normalized plane pairs, group-mask construction and group folds.
     * Profiling itself is bounded planning overhead, not hidden execution work. */
    double bits = 0, columns = 0, keybits = 0, combined_bits = 0;
    uint64_t combined_inputs = 0;
    for (int i = 0; i < candidate.nkey; ++i) {
        int c = candidate.key_arg[i]; inputs |= UINT64_C(1)<<c;
        combined_inputs |= UINT64_C(1)<<c;
        keybits += candidate.key_kind[i] == 2 ? candidate.key_nbits[i] :
            candidate.key_kind[i] == 3 ? __builtin_popcount(candidate.key_mask[i]) : t->width[c];
    }
    if (combined) for (int i = 0; i < poly.n; ++i) {
        if (poly.terms[i].a >= 0) combined_inputs |= UINT64_C(1)<<poly.terms[i].a;
        if (poly.terms[i].b >= 0) combined_inputs |= UINT64_C(1)<<poly.terms[i].b;
    }
    for (int c = 0; c < t->k; ++c) if (inputs>>c&1) { bits += t->width[c]; columns++; }
    for (int c = 0; c < t->k; ++c) if (combined_inputs>>c&1) combined_bits += t->width[c];
    double density = sample.nonempty ? (double)sample.passing/sample.nonempty : 0;
    double decode = sample.nonempty * bits * 0.3;
    double unpack = density > 12 ? sample.nonempty * (columns*24.0 + bits*0.15) : sample.passing*bits*0.5;
    double intermediates = sample.passing * (trace->chain_ops*1.5 + trace->stack_peak*0.1 + candidate.nmeas*0.15);
    double grouping = candidate.nkey ? sample.passing * (2.0 + candidate.nkey*0.5) : 0;
    trace->chain_cost = decode + unpack + intermediates + grouping;
    if (has_hybrid) {
        uint64_t hybrid_inputs = dag_stats.inputs;
        for (int i = 0; i < candidate.nkey; ++i) hybrid_inputs |= UINT64_C(1) << candidate.key_arg[i];
        double hybrid_bits = 0, hybrid_columns = 0;
        for (int c = 0; c < t->k; ++c) if (hybrid_inputs >> c & 1) { hybrid_bits += t->width[c]; ++hybrid_columns; }
        double hybrid_unpack = density > 12 ? sample.nonempty*(hybrid_columns*24.0 + hybrid_bits*0.15) : sample.passing*hybrid_bits*0.5;
        trace->hybrid_cost = sample.nonempty*hybrid_bits*0.3 + hybrid_unpack + grouping +
            sample.passing*(dag_stats.operations*1.5 + dag_stats.stack_peak*0.1 + hybrid.nmeas*0.15);
    }
    double plane = combined ? tr_poly_cost(t, &poly) : 0;
    trace->word_combine_cost = combined ? sample.nonempty*combined_bits*0.3 +
        sample.groups*(plane + keybits*0.5 + (candidate.nkey ? 2.0 : 0)) : -1;
    /* Combination has two physical candidates: word-mask popcounts, and the
     * existing 0.2.0 matrix executor with its adaptive packed transpose row kernel.
     * Sparse output buffers are reusable by either executor. Count actual planes
     * separately from relative-work cost: constants consume no matrix slots. */
    int matrix_planes = 0;
    uint64_t measure_inputs = 0;
    if (combined) for (int i = 0; i < poly.n; ++i) {
        const tr_term *term = &poly.terms[i];
        if (term->a >= 0) {
            matrix_planes += t->width[term->a] * (term->b >= 0 ? t->width[term->b] : 1);
            measure_inputs |= UINT64_C(1)<<term->a;
            if (term->b >= 0) measure_inputs |= UINT64_C(1)<<term->b;
        }
    }
    int matrix_compatible = combined && candidate.nkey > 0 && candidate.out_key && matrix_planes <= MAXMP;
    int packed_bits = (int)keybits;
    for (int c = 0; c < t->k; ++c) if (measure_inputs>>c&1) packed_bits += t->width[c];
    int matrix_row_eligible = matrix_compatible && !candidate.widekey && keybits <= 16 && packed_bits <= 64;
    const char *mm_rows = getenv("TORBLES_MM_ROWS");
    if (mm_rows && *mm_rows && atoi(mm_rows) == 0) matrix_row_eligible = 0;
    uint64_t key_domain = 1;
    for (int i = 0; i < candidate.nkey && matrix_row_eligible; ++i) {
        int kind = candidate.key_kind[i]; uint64_t radix = candidate.key_radix[i];
        if ((kind != 0 && kind != 2 && kind != 3) || !radix || key_domain > (UINT64_MAX-1)/radix)
            matrix_row_eligible = 0;
        else key_domain *= radix;
    }
    /* Share 0.2.0's MMC_TRANS=200, MMC_ROW=6, MMC_TERM=1.5 coefficients.
     * Divide rough cycle-like work by six to compare with this prototype's
     * operation units; this remains an uncalibrated estimator, not nanoseconds.
     * The executor retains its own per-tile choice and eligibility checks. */
    trace->matrix_combine_cost = matrix_row_eligible ? sample.nonempty*combined_bits*0.3 +
        (sample.nonempty*200.0 + sample.passing*(6.0+1.5*(fused.npl+2*fused.npb)))/6.0 : -1;
    int prefer_matrix = matrix_row_eligible && trace->matrix_combine_cost < trace->word_combine_cost;
    trace->combine_cost = prefer_matrix ? trace->matrix_combine_cost : trace->word_combine_cost;
    int gpu_combine = combined && !candidate.widekey && !candidate.nsemi && candidate.nkey <= 6 &&
                      (!candidate.out_key || matrix_compatible);
    if (backend == TR_BACKEND_GPU) {
        if (mode == TR_SYSTEM_CHAIN || !gpu_combine) { trace->reason = 3; return TR_OK; }
        trace->choice = TR_SYSTEM_COMBINE; trace->reason = 3;
    } else if (mode == TR_SYSTEM_COMBINE) {
        if (!combined) { trace->reason = 2; return TR_OK; }
        trace->choice = TR_SYSTEM_COMBINE; trace->reason = 1;
    } else if (mode == TR_SYSTEM_CHAIN) {
        trace->choice = TR_SYSTEM_CHAIN; trace->reason = 1;
    } else if (mode == TR_SYSTEM_HYBRID) {
        if (!has_hybrid) { trace->reason = 2; return TR_OK; }
        trace->choice = TR_SYSTEM_HYBRID; trace->reason = 1;
    } else if (!sample.words || !sample.passing) {
        trace->reason = 6; return TR_OK;
    } else {
        trace->choice = combined && trace->combine_cost <= trace->chain_cost ? TR_SYSTEM_COMBINE : TR_SYSTEM_CHAIN;
        double best = trace->choice == TR_SYSTEM_COMBINE ? trace->combine_cost : trace->chain_cost;
        if (has_hybrid && trace->hybrid_cost < best) trace->choice = TR_SYSTEM_HYBRID;
        trace->reason = combined || has_hybrid ? 0 : 2;
    }
    if (trace->choice == TR_SYSTEM_CHAIN) candidate.agg = A_GENERAL;
    else if (trace->choice == TR_SYSTEM_HYBRID) candidate = hybrid;
    else {
        candidate = fused;
        candidate.agg = (backend == TR_BACKEND_GPU && q->out_key) || prefer_matrix ? A_MM : A_POLYG;
    }
    trace->selected_agg = candidate.agg;
    *q = candidate;
    return TR_OK;
}

/* Cache only a selected measure representation. Output storage, result counters,
 * predicates, key arrays and external buffers always belong to the live plan.
 * Mutable external scopes bypass caching instead of scanning/hash-copying large
 * bitmaps on every warm query. Catalog calls are serialized by its Python lock;
 * direct C callers must likewise serialize access to one catalog. */
#define TR_PHYSICAL_ENTRIES 64
#define TR_PHYSICAL_KEY_BYTES 20000
typedef struct {
    int32_t agg, nmeas; minstr_t meas[48];
    int32_t nlin, lin_col[8]; int64_t lin_coef[8];
    int32_t bil_a, bil_b;
    int32_t npl, pl_col[8]; int64_t pl_coef[8], pl_coef_hi[8];
    int32_t npb, pb_a[8], pb_b[8]; int64_t pb_coef[8], pb_coef_hi[8];
    int64_t p_const, p_const_hi;
} tr_measure_patch;
typedef struct {
    size_t length; uint64_t hash; unsigned char *bytes;
} tr_stored_key;
typedef struct {
    tr_stored_key key, alias;
    tr_measure_patch patch;
    tr_system_trace trace;
    uint64_t touched;
} tr_physical_entry;
typedef struct {
    const atable_t *table; uint64_t revision, clock;
    tr_physical_entry entries[TR_PHYSICAL_ENTRIES];
} tr_physical_cache;
typedef struct {
    size_t length; uint64_t hash;
    unsigned char bytes[TR_PHYSICAL_KEY_BYTES];
} tr_key_buffer;

static void clear_entry(tr_physical_entry *entry) {
    free(entry->key.bytes); free(entry->alias.bytes);
    memset(entry, 0, sizeof *entry);
}

void tr_system_cache_free(void *opaque) {
    tr_physical_cache *cache = opaque;
    if (!cache) return;
    for (int i = 0; i < TR_PHYSICAL_ENTRIES; ++i) clear_entry(&cache->entries[i]);
    free(cache);
}

static int append_key(tr_key_buffer *key, const void *p, size_t length) {
    if (length > sizeof key->bytes - key->length) return 0;
    memcpy(key->bytes + key->length, p, length); key->length += length;
    return 1;
}

static int physical_key(const atable_t *t, const qplan_t *q, int backend, int mode,
                         tr_key_buffer *key) {
    if (!valid_shape(t, q) || q->nsemi || q->xmode || q->nranges || q->nattr || q->wide || q->ext_on ||
        backend == TR_BACKEND_PLAIN || backend < TR_BACKEND_CPU || backend > TR_BACKEND_PLAIN ||
        mode < TR_SYSTEM_AUTO || mode > TR_SYSTEM_HYBRID ||
        (q->agg != A_LIN && q->agg != A_BIL && q->agg != A_GENERAL && q->agg != A_POLYG && q->agg != A_MM)) return 0;
    qplan_t canonical = *q;
    if (!synthesize_chain(&canonical) || canonical.nmeas < 1 || canonical.nmeas > 48) return 0;
    key->length = 0;
    uintptr_t owner = (uintptr_t)t;
    int sparse = !!q->out_key;
    const char *env = getenv("TORBLES_MM_ROWS");
    int row_policy = env && *env ? atoi(env) : INT_MIN;
#define KEY(V) do { if (!append_key(key, &(V), sizeof(V))) return 0; } while (0)
    KEY(owner); KEY(t->revision); KEY(t->n); KEY(t->nw); KEY(t->k);
    KEY(backend); KEY(mode); KEY(sparse); KEY(q->widekey); KEY(row_policy);
    if (!append_key(key, t->width, (size_t)t->k*sizeof t->width[0]) ||
        !append_key(key, t->msb, (size_t)t->k*sizeof t->msb[0])) return 0;
    KEY(q->nprog);
    if (!append_key(key, q->prog, (size_t)q->nprog*sizeof q->prog[0])) return 0;
    KEY(canonical.nmeas);
    if (!append_key(key, canonical.meas, (size_t)canonical.nmeas*sizeof canonical.meas[0])) return 0;
    KEY(q->nkey); KEY(q->ngroups);
    for (int i = 0; i < q->nkey; ++i) {
        KEY(q->key_kind[i]); KEY(q->key_arg[i]); KEY(q->key_radix[i]);
        KEY(q->key_nbits[i]); KEY(q->key_mask[i]);
    }
#undef KEY
    key->hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < key->length; ++i) {
        key->hash ^= key->bytes[i]; key->hash *= UINT64_C(1099511628211);
    }
    return 1;
}

static int same_key(const tr_stored_key *stored, const tr_key_buffer *key) {
    return stored->bytes && stored->hash == key->hash && stored->length == key->length &&
        !memcmp(stored->bytes, key->bytes, key->length);
}

static int save_key(tr_stored_key *stored, const tr_key_buffer *key) {
    unsigned char *copy = malloc(key->length ? key->length : 1);
    if (!copy) return 0;
    memcpy(copy, key->bytes, key->length);
    stored->bytes = copy; stored->length = key->length; stored->hash = key->hash;
    return 1;
}

static void save_patch(tr_measure_patch *patch, const qplan_t *q) {
    patch->agg = q->agg; patch->nmeas = q->nmeas;
    memcpy(patch->meas, q->meas, sizeof patch->meas);
    patch->nlin = q->nlin; memcpy(patch->lin_col, q->lin_col, sizeof patch->lin_col);
    memcpy(patch->lin_coef, q->lin_coef, sizeof patch->lin_coef);
    patch->bil_a = q->bil_a; patch->bil_b = q->bil_b;
    patch->npl = q->npl; memcpy(patch->pl_col, q->pl_col, sizeof patch->pl_col);
    memcpy(patch->pl_coef, q->pl_coef, sizeof patch->pl_coef);
    memcpy(patch->pl_coef_hi, q->pl_coef_hi, sizeof patch->pl_coef_hi);
    patch->npb = q->npb; memcpy(patch->pb_a, q->pb_a, sizeof patch->pb_a);
    memcpy(patch->pb_b, q->pb_b, sizeof patch->pb_b);
    memcpy(patch->pb_coef, q->pb_coef, sizeof patch->pb_coef);
    memcpy(patch->pb_coef_hi, q->pb_coef_hi, sizeof patch->pb_coef_hi);
    patch->p_const = q->p_const; patch->p_const_hi = q->p_const_hi;
}

static void apply_patch(qplan_t *q, const tr_measure_patch *patch) {
    q->agg = patch->agg; q->nmeas = patch->nmeas;
    memcpy(q->meas, patch->meas, sizeof patch->meas);
    q->nlin = patch->nlin; memcpy(q->lin_col, patch->lin_col, sizeof patch->lin_col);
    memcpy(q->lin_coef, patch->lin_coef, sizeof patch->lin_coef);
    q->bil_a = patch->bil_a; q->bil_b = patch->bil_b;
    q->npl = patch->npl; memcpy(q->pl_col, patch->pl_col, sizeof patch->pl_col);
    memcpy(q->pl_coef, patch->pl_coef, sizeof patch->pl_coef);
    memcpy(q->pl_coef_hi, patch->pl_coef_hi, sizeof patch->pl_coef_hi);
    q->npb = patch->npb; memcpy(q->pb_a, patch->pb_a, sizeof patch->pb_a);
    memcpy(q->pb_b, patch->pb_b, sizeof patch->pb_b);
    memcpy(q->pb_coef, patch->pb_coef, sizeof patch->pb_coef);
    memcpy(q->pb_coef_hi, patch->pb_coef_hi, sizeof patch->pb_coef_hi);
    q->p_const = patch->p_const; q->p_const_hi = patch->p_const_hi;
}

void *tr_system_source_create(const qplan_t *q) {
    if (!q) return NULL;
    tr_measure_patch *source = malloc(sizeof *source);
    if (source) save_patch(source, q);
    return source;
}

void tr_system_source_free(void *source) { free(source); }

int32_t tr_plan_system_from_source(tr_catalog *catalog, const atable_t *t, qplan_t *live,
                                    const void *source, int32_t backend, int32_t mode, tr_system_trace *trace) {
    if (!live || !source) return TR_INVALID;
    qplan_t candidate = *live; apply_patch(&candidate, source);
    int rc = tr_plan_system_cached(catalog, t, &candidate, backend, mode, trace);
    if (rc == TR_OK) {
        tr_measure_patch selected; save_patch(&selected, &candidate); apply_patch(live, &selected);
    }
    return rc;
}

int32_t tr_plan_system_cached(tr_catalog *catalog, const atable_t *t, qplan_t *q,
                              int32_t backend, int32_t mode, tr_system_trace *trace) {
    if (!catalog || !t || !q || !trace || catalog->table != t) return TR_INVALID;
    tr_key_buffer key;
    if (!physical_key(t, q, backend, mode, &key)) return tr_plan_system(t, q, backend, mode, trace);
    tr_physical_cache *cache = catalog->physical_cache;
    if (cache && (cache->table != t || cache->revision != t->revision)) {
        for (int i = 0; i < TR_PHYSICAL_ENTRIES; ++i) clear_entry(&cache->entries[i]);
        cache->table = t; cache->revision = t->revision; cache->clock = 0;
    }
    if (cache) for (int i = 0; i < TR_PHYSICAL_ENTRIES; ++i) {
        tr_physical_entry *entry = &cache->entries[i];
        if (!same_key(&entry->key, &key) && !same_key(&entry->alias, &key)) continue;
        int original = q->agg;
        /* Preserve is a decision about the live representation. Equivalent
         * source bytecode can arrive in a different aggregate form, so do not
         * transplant an earlier plan's aggregate/measure on a preserve hit. */
        if (entry->trace.choice) apply_patch(q, &entry->patch);
        *trace = entry->trace; trace->original_agg = original; trace->selected_agg = q->agg;
        trace->cache_hit = 1; entry->touched = ++cache->clock;
        return TR_OK;
    }
    int rc = tr_plan_system(t, q, backend, mode, trace);
    if (rc != TR_OK || (!trace->choice && trace->reason != 6 && trace->reason != 3)) return rc;
    if (!cache) {
        cache = calloc(1, sizeof *cache);
        if (!cache) return TR_OK; /* optimization cache allocation is optional */
        cache->table = t; cache->revision = t->revision; catalog->physical_cache = cache;
    }
    int slot = 0;
    for (int i = 0; i < TR_PHYSICAL_ENTRIES; ++i) {
        if (!cache->entries[i].key.bytes) { slot = i; break; }
        if (cache->entries[i].touched < cache->entries[slot].touched) slot = i;
    }
    tr_physical_entry *entry = &cache->entries[slot]; clear_entry(entry);
    if (!save_key(&entry->key, &key)) return TR_OK;
    /* A later DAG planner may replace source IR as well as the aggregate kind.
     * Match that patched IR on the next execution without copying live buffers. */
    tr_key_buffer alias;
    if (physical_key(t, q, backend, mode, &alias) &&
        (alias.length != key.length || memcmp(alias.bytes, key.bytes, key.length)))
        (void)save_key(&entry->alias, &alias);
    save_patch(&entry->patch, q); entry->trace = *trace; entry->touched = ++cache->clock;
    return TR_OK;
}
