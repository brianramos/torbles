/* Experimental, opt-in algebraic relationship modules. SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_RELATIONSHIPS_H
#define TORBLES_RELATIONSHIPS_H
#include "torble.h"
#define TR_MAX_COLUMNS 12
#define TR_MAX_RULES 128
#define TR_MAX_TERMS 32
#define TR_MAX_STEPS 16
enum { TR_AFFINE = 1, TR_PRODUCT = 2, TR_SUM = 4 };
enum { TR_OK = 0, TR_INVALID = -1, TR_MEMORY = -2, TR_OVERFLOW = -3, TR_STALE = -4 };
typedef struct { int32_t col, valid; int64_t offset; } tr_column;
/* Constant: a=b=-1; linear: a>=0,b=-1; quadratic: 0<=a<=b. */
typedef struct { int32_t a, b; int64_t coefficient; } tr_term;
typedef struct { int32_t n; tr_term terms[TR_MAX_TERMS]; } tr_poly;
typedef struct {
    int32_t module, out, left, right, usable;
    int64_t scale, bias;
    uint64_t checked, excluded, violations, witness;
    tr_poly rhs;
} tr_rule;
typedef struct {
    int32_t changed, steps, explored, truncated;
    double before_cost, after_cost;
    int32_t rule[TR_MAX_STEPS], direction[TR_MAX_STEPS]; /* +1 expands out, -1 contracts rhs */
} tr_trace;
typedef struct tr_catalog_s {
    const atable_t *table;
    uint64_t revision, rows, used_count;
    uint64_t *used;
    int32_t ncols, nrules, candidates, truncated, refused;
    tr_column columns[TR_MAX_COLUMNS];
    uint32_t *codes[TR_MAX_COLUMNS], *validity[TR_MAX_COLUMNS];
    tr_rule rules[TR_MAX_RULES];
    void *physical_cache;             /* bounded native physical-decision cache */
} tr_catalog;
typedef struct {
    uint64_t rows, used_rows, decoded_bytes;
    int32_t candidates, rules, truncated, refused;
} tr_catalog_stats;

/* NULL used bitmap means every physical row. max_candidates is a hard proposal budget. */
tr_catalog *tr_discover(const atable_t *, const tr_column *, int32_t ncols,
                        const uint64_t *used, int32_t modules, int32_t max_candidates, int32_t *error);
void tr_free(tr_catalog *);
int32_t tr_rule_count(const tr_catalog *);
int32_t tr_get_rule(const tr_catalog *, int32_t index, tr_rule *);
int32_t tr_stats(const tr_catalog *, tr_catalog_stats *);
int32_t tr_scope_valid(const tr_catalog *, const atable_t *, const uint64_t *used);
/* Explicit hypothesis; failed/partial proofs returned as evidence, never registered as usable rules. */
int32_t tr_probe(tr_catalog *, int32_t module, int32_t out, int32_t left, int32_t right,
                  int64_t scale, int64_t bias, tr_rule *evidence);
int32_t tr_optimize(const tr_catalog *, const atable_t *, const uint64_t *used,
                     const tr_poly *input, tr_poly *output, tr_trace *);
double tr_poly_cost(const atable_t *, const tr_poly *);

/* Physical alternatives: execute the original operation chain on selected rows,
 * or combine its equations and evaluate the normalized polynomial system. */
enum { TR_SYSTEM_AUTO = 0, TR_SYSTEM_CHAIN = 1, TR_SYSTEM_COMBINE = 2, TR_SYSTEM_HYBRID = 3 };
enum { TR_BACKEND_CPU = 0, TR_BACKEND_GPU = 1, TR_BACKEND_PLAIN = 2 };
typedef struct {
    int32_t choice, original_agg, selected_agg, reason;
    int32_t sample_words, nonempty_words, chain_ops, stack_peak;
    uint64_t sampled_rows, passing_rows;
    double groups_per_word, chain_cost, combine_cost;
    double word_combine_cost, matrix_combine_cost;
    int32_t cache_hit, gpu_profile_skipped;
    int32_t dag_nodes, hybrid_ops, hybrid_registers, hybrid_regions;
    double hybrid_cost;
} tr_system_trace;
/* Internal bounded profile: exact predicate/semijoin semantics on sampled words. */
typedef struct {
    uint64_t rows, passing, nonempty, words;
    double groups;
} tr_system_sample;
int32_t at_relationship_sample(const atable_t *, const qplan_t *, tr_system_sample *);
/* Mutates only the aggregate/measure representation; filters/keys/buffers survive.
 * reason: 0=cost,1=forced,2=unsupported,3=GPU capability,4=wide,5=baseline,
 * 6=empty/no sample,7=invalid request. Costs are uncalibrated relative work units. */
int32_t tr_plan_system(const atable_t *, qplan_t *, int32_t backend, int32_t mode, tr_system_trace *);
int32_t tr_plan_system_cached(tr_catalog *, const atable_t *, qplan_t *, int32_t backend, int32_t mode, tr_system_trace *);
/* Original-measure ownership for compiled plans whose selected bytecode changes.
 * Only measure fields are restored; live predicates, keys and output storage stay. */
void *tr_system_source_create(const qplan_t *);
void tr_system_source_free(void *);
int32_t tr_system_source_restore(qplan_t *, const void *);
int32_t tr_plan_system_from_source(tr_catalog *, const atable_t *, qplan_t *, const void *,
                                    int32_t backend, int32_t mode, tr_system_trace *);
/* Internal native DAG compiler and bytecode verifier. */
typedef struct {
    int32_t nodes, operations, registers, regions, stack_peak;
    uint64_t inputs;
} tr_dag_stats;
int32_t tr_dag_optimize(const atable_t *, const qplan_t *, qplan_t *, tr_dag_stats *);
int32_t tr_measure_valid(const qplan_t *, int32_t columns);
void tr_system_cache_free(void *);

/* Experimental whole-stage algebra. Semantic plans own coefficients and small
 * residual buffers; live output addresses are rebound only by finish(). */
enum { TR_STAGE_OFF = 0, TR_STAGE_AUTO = 1, TR_STAGE_COMPOSE = 2 };
typedef struct {
    int32_t choice, reason, cache_hit, scan_count, hidden_key, key_col, key_domain;
    int32_t input_degree, residual_degree, source_ops, residual_terms, coefficient_terms;
    uint64_t workspace_bytes;
    double chain_cost, compose_cost;
    int32_t candidate_count, partition_kind;
} tr_stage_trace;
/* Returns 0=preserve, 1=selected, or a TR_* error for an invalid request. */
int32_t tr_stage_prepare(void **, tr_catalog *, const atable_t *, qplan_t *, const void *,
                          int32_t backend, int32_t mode, tr_stage_trace *);
int32_t tr_stage_scan_count(const void *);
qplan_t *tr_stage_scan(void *, int32_t);
int32_t tr_stage_weights(const void *, int32_t scan_or_minus1, int64_t *out5);
int32_t tr_stage_reset(void *);
int32_t tr_stage_finish(void *, qplan_t *);
void tr_stage_free(void *);
#endif
