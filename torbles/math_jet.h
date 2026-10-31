/* JET-001 / JET-002: exact jets of bounded polynomial circuits. Dual numbers (a value and k first directional
 * derivatives) and truncated multivariate Taylor jets (gradients, Hessians, Hessian-vector products, mixed and third
 * partials) at a declared expansion point, for single points or summed over the selected rows of a data scope
 * (math_scope.h), on the CPU (math_jet.c) and on a CUDA device (math_jet_gpu.cu), with the same states bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Circuit. A tr_jet_circuit is an immutable DAG of nnodes nodes (tr_jet_node); node i refers only to nodes < i:
 *   CONST c        an int64 rational constant (value)
 *   PARAM p        free parameter p < nparams (differentiated)
 *   DATA c         data column c < ncols: a fixed observation, never differentiated
 *   ADD a b, SUB a b, MUL a b
 *   POW a n        a^n for an integer n in 0..TR_JET_MAX_EXPONENT (index); a^0 = 1, also for a = 0
 *   SCALE a c      c * a for an int64 rational c (value)
 *   NEG a          -a
 * Unused fields: a, b = -1; index and value are ignored (stored as 0 and 0/1). The output is the last node. Nodes the
 * output does not reach are kept but never evaluated; the operand of a POW with n = 0 is not evaluated either (its
 * columns still decide validity, see Aggregates). The circuit is the polynomial f(theta, x) it denotes. Its degree
 * bound (CONST 0, PARAM and DATA 1, ADD/SUB max, MUL sum, POW n times, SCALE/NEG unchanged; saturated at
 * TR_JET_DEGREE_CAP) and its parameter degree (the same with DATA 0) describe the reachable nodes.
 *
 * Jets. A plan (tr_jet_plan) fixes the order d (0..TR_JET_MAX_ORDER), m seed directions (0..TR_JET_MAX_DIRECTIONS), the
 * expansion center c (one rational per parameter), the seed matrix U (m x nparams rationals, row j = direction u_j)
 * and the decimal scale of each data column (value = logical / 10^scale). The jet of f at a data point x is the
 * truncated Taylor polynomial in t = (t_0 .. t_{m-1}) of
 *     g(t) = f(c + U^T t, x)   (theta_p = c_p + sum_j U[j][p] t_j),   g(t) = sum_{|alpha| <= d} J[alpha] t^alpha + O(t^(d+1)).
 * Convention: the stored coefficients are the Taylor (Hasse) coefficients
 *     J[alpha] = (1 / alpha!) d^|alpha| g / dt^alpha (0),   alpha! = prod_j alpha_j!,
 * so J[0] = f(c, x), J[e_j] = D_{u_j} f, J[e_i + e_j] = D_{u_i} D_{u_j} f for i != j and J[2 e_i] = D_{u_i}^2 f / 2,
 * J[3 e_i] = D_{u_i}^3 f / 6. Readouts in TR_JET_DERIVATIVE return the derivatives alpha! J[alpha] instead. With U = I
 * (the directions are the parameters) these are the gradient, the Hessian and the third partials at c; with the
 * directions [I; v] (m = nparams + 1) the mixed coefficients J[e_i + e_m] are the Hessian-vector product (H v)_i. At
 * c = 0 with U = I the Taylor coefficients are the coefficients of f's monomials in theta (of degree <= d).
 * Laws, per node: the sum law (a + b)[alpha] = a[alpha] + b[alpha]; the truncated product (a b)[gamma] =
 * sum_{alpha + beta = gamma} a[alpha] b[beta] (order 1: the Leibniz law (a b)' = a' b + a b'); the binomial law for
 * powers (a0 + h)^n = sum_{k <= min(n, d)} C(n, k) a0^(n-k) h^k, h the non-constant part (order 1: the chain rule
 * (a^n)' = n a^(n-1) a'); constants and data have zero derivatives; PARAM p has J[0] = c_p, J[e_j] = U[j][p].
 * Monomial order: graded (|alpha| = 0, 1, ..., d); within a degree the sorted index tuples (i1 <= ... <= ik) in
 * lexicographic order: [1], [t_0 .. t_{m-1}], [t_0^2, t_0 t_1, .., t_0 t_{m-1}, t_1^2, .., t_{m-1}^2], [t_0^3, ...].
 * So index 1 + j is t_j (the dual-number layout) and the support S = C(m + d, d).
 *
 * Exact carrier of the evaluation (fixed denominators; the CPU and the CUDA kernels run the same inline code). Every
 * evaluated node v has a positive int64 denominator D_v fixed by the plan: CONST its canonical denominator; PARAM p the
 * lcm of the denominators of c_p and of U[j][p] (j = 0..m-1); DATA c 10^scale_c; ADD/SUB lcm(D_a, D_b); MUL D_a D_b;
 * POW n D_a^n (n = 0: 1); SCALE by n/d (canonical) d D_a / gcd(|n|, D_a); NEG D_a. A D_v (or a multiplier D_v / D_a)
 * outside int64 refuses the plan with TR_MATH_OVERFLOW. Every coefficient of a node's jet is an int64 numerator over
 * D_v, computed with 128-bit intermediates and checked: a numerator, a product, a binomial coefficient, a power of the
 * constant term or a running sum outside int64 (a 128-bit convolution sum outside int128) refuses the evaluation with
 * TR_MATH_OVERFLOW. Nothing wraps or is rounded. Whether a point is refused depends only on the plan and the point (the
 * evaluation order below), never on the backend, chunking or thread order. Exact results are read canonically
 * (tr_jet_agg_read: reduced numerator over a positive denominator) or converted (tr_jet_values_*).
 * Evaluation order, per node in increasing index: ADD/SUB  ma A[g] +- mb B[g] (ma = D_v / D_a, mb = D_v / D_b, one
 * 128-bit sum, then checked to int64); MUL  for each gamma, the 128-bit sum of A[alpha] B[gamma - alpha] over alpha <=
 * gamma in a fixed table order, checked to int64; POW n on a jet  coef_k = C(n, k) a0^(n-k) for k = 0..K (K = min(n,
 * d); C(n, k) = C(n, k-1) (n-k+1) / k and a0^(n-K) by squaring, each checked), then J = coef_0, P_1 = h, J += coef_1
 * P_1, and for k = 2..K: P_k = P_{k-1} h (as MUL), J += coef_k P_k (each partial sum checked to int64); POW on a scalar
 * a0^n by squaring (checked); SCALE  n' A[g] with n' = n / gcd(|n|, D_a); NEG  -A[g]. Coefficients a node provably
 * cannot have (outside its static support: the directions its parameters are seeded with, and degrees above its
 * parameter degree) are zero and are skipped: they add exactly zero and never decide a refusal.
 *
 * Aggregates. Over a scope whose columns are the circuit's data columns (ncols equal, every scale equal to the plan's;
 * else TR_MATH_DOMAIN_ERROR), the aggregate of group g is the jet of sum_r f(theta, x_r) over the selected rows r of
 * g where every data column the circuit reads (any reachable DATA node) is present: SQL semantics, SUM(f) skips a row
 * where f is NULL. By linearity it is the sum of the rows' jets. States per group (tr_jet_states): rows (bag
 * multiplicity: the selected rows), count (the rows that entered the sum) and S signed 128-bit sums of the rows'
 * numerators over the plan's output denominator D. Each row's numerators are int64, so a sum over fewer than 2^63 rows
 * fits 128 bits: accumulation and merges never overflow, and no component is dropped, truncated or rounded by a merge.
 * Presence is explicit: a group is present iff rows > 0; a present group without a valid row has count 0 and the zero
 * jet (the empirical-functional convention of the first slice, not SQL's NULL); an absent group has rows 0.
 *   tr_jet_agg_init / _accumulate (disjoint physical row ranges [begin, end): a row is counted once, else
 *   TR_MATH_INVALID) / _merge (builders of the same scope and plan) / _finish (every row covered) give exactly
 *   tr_jet_agg_build. A builder whose accumulate refused (TR_MATH_OVERFLOW) is spent. tr_jet_agg_add sums two
 *   aggregates of the same plan and group count: the aggregate of the bag union of their rows (unscoped).
 *   Per-point jets are the aggregate of a scope with one group (and one row) per point.
 *   The CUDA path (tr_mgpu_jet_aggregate) writes the same states, which tr_jet_agg_from_states adopts after checking
 *   their consistency (shape, total rows, count <= rows, zero sums without valid rows, |sum| <= count 2^63, zero
 *   outside the output's static support).
 *
 * Carriers of results (tr_jet_values: immutable, every component of every jet, all or nothing). From an aggregate:
 *   TR_CARRIER_Q64       narrowing: every canonical component fits int64, else TR_MATH_OVERFLOW (nothing is made)
 *   TR_CARRIER_BIGQ      lossless (math_carriers big rationals; each number budgeted by math_carriers)
 *   TR_CARRIER_GFP       reduction mod a prime p < 2^62 (tr_gfp_field): n * D^-1 mod p; TR_MATH_NO_SOLUTION when p
 *                        divides the denominator, TR_MATH_INVALID for a composite p, TR_MATH_UNSUPPORTED for p >= 2^62
 *   TR_CARRIER_INTERVAL  the tight enclosure [RD(x), RU(x)] of each component: APPROXIMATE (certified)
 * Between value carriers (tr_jet_values_convert) the kinds of tr_carrier_conversion (math_carriers.h): Q64 -> BIGQ
 * lossless, BIGQ -> Q64 narrowing, Q64 / BIGQ -> GFP reduction, Q64 / BIGQ -> INTERVAL enclosing, identity; GFP or
 * INTERVAL -> an exact carrier, GFP(p) -> GFP(q != p), anything -> another carrier: TR_MATH_CARRIER_ERROR.
 * tr_jet_values_add is the sum law in one carrier (same plan, convention, carrier, field and shape, else
 * TR_MATH_DOMAIN_ERROR / TR_MATH_CARRIER_ERROR): Q64 checked (TR_MATH_OVERFLOW), BIGQ exact, GFP mod p, INTERVAL
 * outward rounded. With TR_JET_DERIVATIVE the components are alpha! J[alpha].
 *
 * Budgets (math_budgets.h), checked before anything is allocated, refused with TR_MATH_BUDGET and the refusal:
 *   circuit: variables = nparams + ncols, degree = the degree bound, circuit = nnodes, bytes = tr_jet_circuit_bytes.
 *   plan: variables = m (the jet dimension), degree = d (the jet order), support = S = C(m + d, d) (coefficients per
 *     jet), tensor = C(2m + d, d) (the structure constants of the truncated product: the pairs (alpha, beta) with
 *     |alpha| + |beta| <= d), circuit = evaluated nodes, bytes = every allocation (tr_jet_plan_need).
 *   aggregate (CPU): support = S, states = ngroups * S, bytes = the states 16 G (S + 1), the coverage bitmap, the
 *     workspace, the builder and the aggregate (tr_jet_agg_need, TR_JET_NEED_CPU); adoption: TR_JET_NEED_ADOPT; CUDA:
 *     its device copies (TR_JET_NEED_DEVICE, tr_jet_device_need).
 *   values: output = njets * S, bytes = the object, its components and its copy of the plan's descriptor (BIGQ: the
 *     pointer array; math_carriers budgets each big rational with the same budget).
 * Beyond the representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED (nparams > 64, ncols > 32,
 * nnodes > 2^20, m > 64, d > 16, S > 65535, a POW exponent > 255, more than 2^63 selected rows).
 *
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3), _BUDGET (-5), TR_MATH_DOMAIN_ERROR (-6, a scope or values
 * of another plan), TR_MATH_CARRIER_ERROR (-7), TR_MATH_NO_SOLUTION (-10), TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED
 * (-12). No new code. Objects are immutable once built (builders are the one mutable object: one owner, calls
 * serialized) and individually owned; constructors return NULL and set *error, leaving their inputs unchanged; a plan
 * copies what it needs (the circuit may be freed after), a builder reads its plan and scope (both must outlive it), an
 * aggregate or values object owns everything it reads. Read-only calls on one object may run concurrently. */
#ifndef TORBLES_MATH_JET_H
#define TORBLES_MATH_JET_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_JET_ABI 1
#define TR_JET_MAX_NODES (1 << 20)
#define TR_JET_MAX_PARAMS 64
#define TR_JET_MAX_COLUMNS TR_SCOPE_MAX_COLUMNS
#define TR_JET_MAX_DIRECTIONS 64
#define TR_JET_MAX_ORDER 16
#define TR_JET_MAX_SUPPORT 65535
#define TR_JET_MAX_EXPONENT 255
#define TR_JET_DEGREE_CAP (1 << 30)
#define TR_JET_MAX_ROWS (UINT64_C(1) << 63)

enum { TR_JET_CONST = 1, TR_JET_PARAM = 2, TR_JET_DATA = 3, TR_JET_ADD = 4, TR_JET_SUB = 5, TR_JET_MUL = 6,
       TR_JET_POW = 7, TR_JET_SCALE = 8, TR_JET_NEG = 9, TR_JET_OPS = 10 };
enum { TR_JET_TAYLOR = 0, TR_JET_DERIVATIVE = 1 };                         /* readout conventions */
enum { TR_JET_NEED_CPU = 0, TR_JET_NEED_DEVICE = 1, TR_JET_NEED_ADOPT = 2 };   /* tr_jet_agg_need targets */
enum { TR_JET_A = 1, TR_JET_B = 2, TR_JET_R = 4 };                          /* tr_jet_step.jet: which are jets */

/* ------------------------------------------------------------------ ABI structs (registered: math_handles.h) */
typedef struct {
    int32_t op;               /* TR_JET_* */
    int32_t a, b;             /* operand nodes (< this node), -1 when unused */
    int32_t index;            /* PARAM: parameter; DATA: column; POW: exponent; else 0 */
    tr_num_rational value;    /* CONST: the constant; SCALE: the factor; else 0/1 */
} tr_jet_node;                /* 32 bytes */

typedef struct {
    int32_t nodes, nparams, ncols;
    int32_t live;             /* nodes the output reaches (structurally, through every edge) */
    int32_t degree;           /* total degree bound of f (saturated at TR_JET_DEGREE_CAP) */
    int32_t param_degree;     /* degree bound in the parameters */
    uint32_t columns;         /* bit c: a reachable DATA node reads column c (these decide validity) */
    int32_t reserved;
    uint64_t params;          /* bit p: a reachable PARAM node reads parameter p */
    uint64_t bytes;           /* the circuit's allocation */
} tr_jet_circuit_info;        /* 48 bytes */

/* One compiled step (the plan's program, read by the CPU and by the kernels). Scalars (jets with zero derivatives:
 * no seeded parameter below) live in registers, jets in slots of S int64 numerators. */
typedef struct {
    uint8_t op;               /* TR_JET_* */
    uint8_t jet;              /* TR_JET_A / _B / _R: operand a, operand b, the result are jets (else registers) */
    uint8_t pdeg, pdeg_a, pdeg_b;   /* parameter degrees of the result and the operands, capped at the order */
    uint8_t reserved[3];
    int32_t index;            /* PARAM: parameter; DATA: column; POW: exponent */
    uint32_t dst, a, b;       /* register or slot of the result and of the operands */
    uint32_t t0, t1;          /* POW of a jet: two temporary slots */
    uint64_t dep, dep_a, dep_b;   /* bit j: direction j can appear in the result's / an operand's jet */
    int64_t ma, mb;           /* CONST: the numerator (ma); ADD/SUB: D / D_a, D / D_b; SCALE: the reduced factor
                                 numerator (ma); PARAM: unused (the leaf table holds the numerators) */
} tr_jet_step;                /* 72 bytes */

/* The plain-C view of a plan (pointers into the plan; valid while it lives), copied by value into a kernel with its
 * pointers swapped for device copies. */
typedef struct {
    uint32_t support;         /* S */
    uint32_t nsteps, nregs, nslots;
    int32_t order, directions, nparams, ncols;
    uint32_t columns;         /* bit c: column c decides validity */
    uint32_t out;             /* the output's register or slot */
    int32_t out_jet;          /* 1: the output is a jet slot; 0: a register */
    int32_t out_pdeg;         /* the output's parameter degree (capped at the order) */
    int64_t denominator;      /* D of the output */
    uint64_t pairs;           /* entries of conv_pairs: C(2m + d, d) */
    uint64_t workspace;       /* int64 words of one point's registers and slots: nregs + nslots * S */
    const tr_jet_step *steps; /* nsteps */
    const int64_t *leaf;      /* nparams x (m + 1): PARAM p's numerators at 1, t_0 .. t_{m-1} */
    const uint8_t *degree;    /* S: |alpha| */
    const uint64_t *mask;     /* S: bit j when alpha_j > 0 */
    const uint32_t *conv_start;   /* S + 1: pairs of gamma are conv_pairs[conv_start[gamma] .. conv_start[gamma+1]) */
    const uint32_t *conv_pairs;   /* alpha | beta << 16 with alpha + beta = gamma */
    const uint32_t *upto;     /* order + 1: upto[k] = C(m + k, k), the monomials of degree <= k (a prefix: graded) */
    const int32_t *scales;    /* ncols: the decimal scale of each data column (host memory; never read by kernels) */
    uint64_t out_dep;         /* the output's direction mask */
} tr_jet_view;                /* 144 bytes */

typedef struct {
    uint64_t support, pairs, steps, registers, slots;
    uint64_t workspace;       /* int64 words per point (registers and slots) */
    uint64_t bytes;           /* the plan's allocation */
    uint64_t multiplications, additions;   /* per point (static counts of the evaluation order above) */
    uint64_t digest[2];       /* 128-bit digest of the plan's descriptor (a pre-filter, not an identity) */
    int64_t denominator;      /* D of the output */
    int32_t order, directions, nparams, ncols;
    int32_t nodes;            /* of the circuit */
    int32_t jets;             /* steps whose result is a jet */
    int32_t out_jet, out_pdeg;
    uint32_t columns;
    int32_t reserved;
    uint64_t out_dep;         /* the output's direction mask */
} tr_jet_plan_info;           /* 144 bytes */

/* the states of an aggregate (see the header comment); sum[g * support + k] = hi 2^64 + (uint64_t)lo */
typedef struct {
    uint64_t ngroups, support;
    uint64_t *rows;           /* [ngroups] selected rows */
    uint64_t *count;          /* [ngroups] rows that entered the sums */
    tr_math_i128 *sum;        /* [ngroups * support] numerators over the plan's denominator */
} tr_jet_states;              /* 40 bytes */

typedef struct {
    uint64_t ngroups, present, rows, count;   /* groups, groups with rows, selected rows, rows in the sums */
    uint64_t support, bytes;
    uint64_t digest[2];       /* the plan's digest */
    uint64_t scope_digest[2]; /* the scope's descriptor digest (0 when unscoped) */
    int64_t denominator;
    int32_t order, directions, scoped, reserved;
} tr_jet_agg_info;            /* 104 bytes */

typedef struct {
    uint64_t njets, support, bytes;
    uint64_t p;               /* TR_CARRIER_GFP: the prime; else 0 */
    int32_t carrier;          /* TR_CARRIER_Q64 / _BIGQ / _GFP / _INTERVAL */
    int32_t convention;       /* TR_JET_TAYLOR / TR_JET_DERIVATIVE */
    int32_t order, directions;
    int32_t exact;            /* 0 for INTERVAL (approximate, certified) */
    int32_t reserved;
} tr_jet_values_info;         /* 56 bytes */

typedef struct tr_jet_circuit_s tr_jet_circuit;
typedef struct tr_jet_plan_s tr_jet_plan;
typedef struct tr_jet_builder_s tr_jet_builder;
typedef struct tr_jet_agg_s tr_jet_agg;
typedef struct tr_jet_values_s tr_jet_values;

/* ------------------------------------------------------------------ shared inline code (CPU, CUDA host, kernels) */
/* number of terms of the plan's monomial / product tables (saturating, UNKNOWN past 64 bits) */
static inline uint64_t tr_jet_support(uint32_t m, uint32_t d) { return tr_budget_monomials(m, d); }
static inline uint64_t tr_jet_pairs(uint32_t m, uint32_t d) { return tr_budget_monomials(2 * m, d); }

/* *acc += x in 128 bits; 0 (unchanged) when the sum leaves int128 */
TR_NUM_HD int tr_jet_add128(__int128 *acc, __int128 x) {
    __int128 s = (__int128)((unsigned __int128)*acc + (unsigned __int128)x);
    if ((x > 0 && s < *acc) || (x < 0 && s > *acc)) return 0;
    *acc = s;
    return 1;
}
TR_NUM_HD int tr_jet_fit(__int128 x, int64_t *out) {
    if (x > INT64_MAX || x < INT64_MIN) return 0;
    *out = (int64_t)x;
    return 1;
}
/* coefficient idx can be nonzero for a value whose parameters are seeded along dep and whose degrees are lo..hi */
TR_NUM_HD int tr_jet_in(const tr_jet_view *v, uint32_t idx, uint64_t dep, int lo, int hi) {
    int g = v->degree[idx];
    return g >= lo && g <= hi && !(v->mask[idx] & ~dep);
}
/* x^n by squaring (squares only while needed), checked: 0 when |x^n| > int64 */
TR_NUM_HD int tr_jet_pow64(int64_t x, uint32_t n, int64_t *out) {
    int64_t r = 1, b = x;
    while (n) {
        if (n & 1) { if (!tr_mul_i64(r, b, &r)) return 0; }
        n >>= 1;
        if (n && !tr_mul_i64(b, b, &b)) return 0;
    }
    *out = r;
    return 1;
}

/* The workspace of one point: nregs registers, nslots slots of S coefficients, as int64 words w[i * stride]
 * (stride 1 on the CPU; on the device the work items' workspaces are interleaved, so that consecutive threads touch
 * consecutive words). A slot holding the jet of node v is defined on v's static support only (the coefficients g <
 * upto[pdeg_v] with tr_jet_in(g, dep_v, 0, pdeg_v)); every reader tests the support before reading, so the other
 * words may hold anything (they are never read, and are not written: loops stop at upto[pdeg]). */
#define TR_JET_W(ws, stride, i) ((ws)[(uint64_t)(i) * (stride)])

/* J = A * B on the static supports (the MUL law; also the powers of h): every gamma of the result's support (dep,
 * degrees lo..hi) is the 128-bit sum over the table's pairs of A[alpha] B[beta] with alpha in (dep_a, alo..ahi) and
 * beta in (dep_b, blo..bhi), checked to int64. */
TR_NUM_HD int tr_jet_conv(const tr_jet_view *v, int64_t *ws, uint64_t stride, uint64_t A, uint64_t B, uint64_t J,
                          uint64_t dep, int lo, int hi, uint64_t dep_a, int alo, int ahi, uint64_t dep_b, int blo,
                          int bhi) {
    if (hi < lo) return 1;
    const uint32_t g0 = lo > 0 ? v->upto[lo - 1] : 0, g1 = v->upto[hi];
    for (uint32_t g = g0; g < g1; g++) {
        if (v->mask[g] & ~dep) continue;                    /* not in the support (the degree is in [lo, hi]) */
        int64_t r = 0;
        __int128 acc = 0;
        for (uint32_t e = v->conv_start[g]; e < v->conv_start[g + 1]; e++) {
            uint32_t pr = v->conv_pairs[e], al = pr & 0xffffu, be = pr >> 16;
            if (!tr_jet_in(v, al, dep_a, alo, ahi) || !tr_jet_in(v, be, dep_b, blo, bhi)) continue;
            if (!tr_jet_add128(&acc, (__int128)TR_JET_W(ws, stride, A + al) * TR_JET_W(ws, stride, B + be)))
                return 0;
        }
        if (!tr_jet_fit(acc, &r)) return 0;
        TR_JET_W(ws, stride, J + g) = r;
    }
    return 1;
}

/* Evaluates the plan at one point: data[c] is the logical value of column c (every column the circuit reads; the
 * caller checked their validity). Writes the S numerators of the output (over v->denominator) to out[k * ostride].
 * Returns 1, or 0 when the point is refused (TR_MATH_OVERFLOW: see the evaluation order in the header comment). */
TR_NUM_HD int tr_jet_eval(const tr_jet_view *v, const int64_t *data, int64_t *ws, uint64_t stride, int64_t *out,
                          uint64_t ostride) {
    const uint32_t S = v->support, m1 = (uint32_t)v->directions + 1;
    const uint64_t base = v->nregs;                       /* slot s starts at word base + s * S */
    for (uint32_t s = 0; s < v->nsteps; s++) {
        const tr_jet_step *st = &v->steps[s];
        const int ra = st->jet & TR_JET_A, rb = st->jet & TR_JET_B, rr = st->jet & TR_JET_R;
        const uint64_t A = base + (uint64_t)st->a * S, B = base + (uint64_t)st->b * S, J = base + (uint64_t)st->dst * S;
        int64_t r = 0;
        switch (st->op) {
        case TR_JET_CONST: TR_JET_W(ws, stride, st->dst) = st->ma; continue;
        case TR_JET_DATA: TR_JET_W(ws, stride, st->dst) = data[st->index]; continue;
        case TR_JET_PARAM: {
            const int64_t *L = v->leaf + (uint64_t)st->index * m1;
            if (!rr) { TR_JET_W(ws, stride, st->dst) = L[0]; continue; }
            for (uint32_t g = 0; g < m1; g++) TR_JET_W(ws, stride, J + g) = L[g];   /* degrees 0 and 1 */
            continue;
        }
        case TR_JET_ADD: case TR_JET_SUB: {
            const int sub = st->op == TR_JET_SUB;
            if (!rr) {
                __int128 x = (__int128)st->ma * TR_JET_W(ws, stride, st->a), y = (__int128)st->mb * TR_JET_W(ws, stride, st->b);
                if (!tr_jet_add128(&x, sub ? -y : y) || !tr_jet_fit(x, &r)) return 0;
                TR_JET_W(ws, stride, st->dst) = r;
                continue;
            }
            for (uint32_t g = 0, g1 = v->upto[st->pdeg]; g < g1; g++) {
                if (v->mask[g] & ~st->dep) continue;
                int64_t x = ra ? (tr_jet_in(v, g, st->dep_a, 0, st->pdeg_a) ? TR_JET_W(ws, stride, A + g) : 0)
                               : (g ? 0 : TR_JET_W(ws, stride, st->a));
                int64_t y = rb ? (tr_jet_in(v, g, st->dep_b, 0, st->pdeg_b) ? TR_JET_W(ws, stride, B + g) : 0)
                               : (g ? 0 : TR_JET_W(ws, stride, st->b));
                __int128 p = (__int128)st->ma * x, q = (__int128)st->mb * y;
                if (!tr_jet_add128(&p, sub ? -q : q) || !tr_jet_fit(p, &r)) return 0;
                TR_JET_W(ws, stride, J + g) = r;
            }
            continue;
        }
        case TR_JET_MUL:
            if (!rr) {
                if (!tr_mul_i64(TR_JET_W(ws, stride, st->a), TR_JET_W(ws, stride, st->b), &r)) return 0;
                TR_JET_W(ws, stride, st->dst) = r;
                continue;
            }
            if (ra && rb) {
                if (!tr_jet_conv(v, ws, stride, A, B, J, st->dep, 0, st->pdeg, st->dep_a, 0, st->pdeg_a, st->dep_b, 0,
                                 st->pdeg_b)) return 0;
                continue;
            }
            {   /* a jet times a scalar */
                const uint64_t P = ra ? A : B;
                const int64_t c = TR_JET_W(ws, stride, ra ? st->b : st->a);
                for (uint32_t g = 0, g1 = v->upto[st->pdeg]; g < g1; g++) {
                    if (v->mask[g] & ~st->dep) continue;
                    if (!tr_mul_i64(TR_JET_W(ws, stride, P + g), c, &r)) return 0;
                    TR_JET_W(ws, stride, J + g) = r;
                }
            }
            continue;
        case TR_JET_SCALE: case TR_JET_NEG: {
            const int64_t c = st->op == TR_JET_NEG ? -1 : st->ma;
            if (!rr) {
                if (!tr_mul_i64(TR_JET_W(ws, stride, st->a), c, &r)) return 0;
                TR_JET_W(ws, stride, st->dst) = r;
                continue;
            }
            for (uint32_t g = 0, g1 = v->upto[st->pdeg]; g < g1; g++) {
                if (v->mask[g] & ~st->dep) continue;
                if (!tr_mul_i64(TR_JET_W(ws, stride, A + g), c, &r)) return 0;
                TR_JET_W(ws, stride, J + g) = r;
            }
            continue;
        }
        case TR_JET_POW: {
            const uint32_t n = (uint32_t)st->index;
            if (n == 0) { TR_JET_W(ws, stride, st->dst) = 1; continue; }   /* a^0 = 1 (the operand is not evaluated) */
            if (!rr) {
                if (!tr_jet_pow64(TR_JET_W(ws, stride, st->a), n, &r)) return 0;
                TR_JET_W(ws, stride, st->dst) = r;
                continue;
            }
            const int d = v->order, pa = st->pdeg_a;
            const uint32_t K = n < (uint32_t)d ? n : (uint32_t)d;
            const int64_t a0 = TR_JET_W(ws, stride, A);
            int64_t coef[TR_JET_MAX_ORDER + 1], c = 1, pw;
            if (!tr_jet_pow64(a0, n - K, &pw)) return 0;            /* a0^(n-K), then up to a0^n */
            for (uint32_t k = K + 1; k-- > 0;) {
                coef[k] = pw;
                if (k && !tr_mul_i64(pw, a0, &pw)) return 0;
            }
            for (uint32_t k = 1; k <= K; k++) {                      /* C(n, k) = C(n, k-1) (n-k+1) / k */
                if (!tr_mul_i64(c, (int64_t)(n - k + 1), &c)) return 0;
                c /= (int64_t)k;
                if (!tr_mul_i64(coef[k], c, &coef[k])) return 0;
            }
            const uint64_t T0 = base + (uint64_t)st->t0 * S, T1 = base + (uint64_t)st->t1 * S;
            for (uint32_t g = 0, g1 = v->upto[st->pdeg]; g < g1; g++)   /* J = coef_0 on J's support */
                if (!(v->mask[g] & ~st->dep)) TR_JET_W(ws, stride, J + g) = g ? 0 : coef[0];
            for (uint32_t g = 1, g1 = v->upto[pa]; g < g1; g++)         /* P_1 = h on h's support */
                if (!(v->mask[g] & ~st->dep_a)) TR_JET_W(ws, stride, T0 + g) = TR_JET_W(ws, stride, A + g);
            uint64_t P = T0, Q = T1;
            for (uint32_t k = 1; k <= K; k++) {
                int hi = (int)k * pa < d ? (int)k * pa : d;
                if (k >= 2 && !tr_jet_conv(v, ws, stride, P, A, Q, st->dep_a, (int)k, hi, st->dep_a, (int)k - 1,
                                           (int)(k - 1) * pa < d ? (int)(k - 1) * pa : d, st->dep_a, 1, pa)) return 0;
                if (k >= 2) { uint64_t t = P; P = Q; Q = t; }        /* P = h^k */
                for (uint32_t g = v->upto[k - 1], g1 = v->upto[hi]; g < g1; g++) {
                    if (v->mask[g] & ~st->dep_a) continue;
                    __int128 x = (__int128)coef[k] * TR_JET_W(ws, stride, P + g);
                    if (!tr_jet_add128(&x, TR_JET_W(ws, stride, J + g)) || !tr_jet_fit(x, &r)) return 0;
                    TR_JET_W(ws, stride, J + g) = r;
                }
            }
            continue;
        }
        default: return 0;
        }
    }
    if (!v->out_jet) {
        out[0] = TR_JET_W(ws, stride, v->out);
        for (uint32_t g = 1; g < S; g++) out[(uint64_t)g * ostride] = 0;
        return 1;
    }
    const uint64_t O = base + (uint64_t)v->out * S;
    for (uint32_t g = 0; g < S; g++)
        out[(uint64_t)g * ostride] = tr_jet_in(v, g, v->out_dep, 0, v->out_pdeg) ? TR_JET_W(ws, stride, O + g) : 0;
    return 1;
}

/* ------------------------------------------------------------------ the device's work split and needs */
#define TR_JET_WORK (UINT64_C(1) << 16)              /* at most this many work items */
#define TR_JET_MIN_ROWS 16                            /* and at least this many rows per work item */
#define TR_JET_DEVICE_WORKSPACE (UINT64_C(128) << 20) /* the interleaved workspaces stay below this (when W > 1) */
/* int64 words of one work item: registers and slots, the output coefficients, the local sums (two words each) */
static inline uint64_t tr_jet_item_words(const tr_jet_view *p) {
    return tr_budget_add(p->workspace, (uint64_t)p->support * 3);
}
/* W work items for `rows` physical rows: about rows / TR_JET_MIN_ROWS, at most TR_JET_WORK and the workspace cap */
static inline uint64_t tr_jet_device_items(const tr_jet_view *p, uint64_t rows) {
    uint64_t w = (rows + TR_JET_MIN_ROWS - 1) / TR_JET_MIN_ROWS, per = tr_budget_mul(tr_jet_item_words(p), 8);
    if (w > TR_JET_WORK) w = TR_JET_WORK;
    if (per && w > TR_JET_DEVICE_WORKSPACE / per) w = TR_JET_DEVICE_WORKSPACE / per;
    return w ? w : 1;
}
/* the needs of tr_mgpu_jet_aggregate: support = S, states = groups x S, bytes = its device copies (the selection, the
 * group ids, the columns the circuit reads and their validity, the plan's tables, the work items' workspaces, the
 * states, the flag word) */
static inline void tr_jet_device_need(const tr_jet_view *p, const tr_scope_view *v, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    const uint64_t n = v->rows, S = p->support, G = v->ngroups;
    uint64_t bytes = tr_budget_add(tr_budget_bitmap_bytes(n), 8);
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    for (int32_t c = 0; c < v->ncols && c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((p->columns >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, n);
    }
    bytes = tr_budget_add(bytes, tr_budget_mul(p->nsteps, sizeof(tr_jet_step)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)(p->nparams > 0 ? p->nparams : 0),
                                                             (uint64_t)p->directions + 1), 8));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(S, 9), tr_budget_mul(S + 1, 4)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(p->order > 0 ? p->order : 0) + 1, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(p->pairs, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(tr_jet_device_items(p, n), tr_jet_item_words(p)), 8));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(G, 16), tr_budget_mul(tr_budget_mul(G, S), 16)));
    need->need[TR_BUDGET_SUPPORT] = S;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(G, S);
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ circuits (libalg) */
/* bytes a circuit of nnodes nodes allocates (the object and its nodes) */
uint64_t tr_jet_circuit_bytes(int32_t nnodes);
/* the needs tr_jet_circuit_create checks (the nodes validated as there) */
int32_t tr_jet_circuit_need(const tr_jet_node *nodes, int32_t nnodes, int32_t nparams, int32_t ncols, tr_math_need *);
tr_jet_circuit *tr_jet_circuit_create(const tr_jet_node *nodes, int32_t nnodes, int32_t nparams, int32_t ncols,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_jet_circuit_get_info(const tr_jet_circuit *, tr_jet_circuit_info *);
/* the stored (canonical) nodes; TR_MATH_INVALID when capacity < nodes */
int32_t tr_jet_circuit_get_nodes(const tr_jet_circuit *, tr_jet_node *out, int32_t capacity);
void tr_jet_circuit_free(tr_jet_circuit *);

/* ------------------------------------------------------------------ plans */
/* center: nparams rationals (NULL: all 0); seeds: m x nparams rationals, row-major (NULL: the identity, which needs
 * m == nparams); scales: ncols decimal scales 0..18 (NULL: all 0). Denominators must be nonzero (TR_MATH_INVALID);
 * a rational whose canonical form does not fit int64 is TR_MATH_OVERFLOW. */
tr_jet_plan *tr_jet_plan_create(const tr_jet_circuit *, int32_t order, int32_t directions,
                                const tr_num_rational *center, const tr_num_rational *seeds, const int32_t *scales,
                                const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the needs tr_jet_plan_create checks */
int32_t tr_jet_plan_need(const tr_jet_circuit *, int32_t order, int32_t directions, tr_math_need *);
int32_t tr_jet_plan_get_info(const tr_jet_plan *, tr_jet_plan_info *);
int32_t tr_jet_plan_get_view(const tr_jet_plan *, tr_jet_view *);
/* the monomials alpha in index order: S x m exponents, row-major (TR_MATH_INVALID when capacity is smaller) */
int32_t tr_jet_plan_monomials(const tr_jet_plan *, uint8_t *exps, uint64_t capacity);
/* the compiled steps (capacity >= steps) */
int32_t tr_jet_plan_steps(const tr_jet_plan *, tr_jet_step *out, uint64_t capacity);
/* D_v of every circuit node (0 for a node that is not evaluated); capacity >= nodes */
int32_t tr_jet_plan_denominators(const tr_jet_plan *, int64_t *out, uint64_t capacity);
/* 1 when both plans have the same descriptor (circuit, order, directions, center, seeds, scales), else 0 */
int32_t tr_jet_plan_equal(const tr_jet_plan *, const tr_jet_plan *);
void tr_jet_plan_free(tr_jet_plan *);

/* ------------------------------------------------------------------ aggregates over a scope */
tr_jet_builder *tr_jet_agg_init(const tr_jet_plan *, const tr_scope *, const tr_math_budget *, tr_math_refusal *,
                                int32_t *error);
int32_t tr_jet_agg_accumulate(tr_jet_builder *, uint64_t begin, uint64_t end);
int32_t tr_jet_agg_merge(tr_jet_builder *into, const tr_jet_builder *from);
tr_jet_agg *tr_jet_agg_finish(tr_jet_builder *, int32_t *error);
void tr_jet_builder_free(tr_jet_builder *);
tr_jet_agg *tr_jet_agg_build(const tr_jet_plan *, const tr_scope *, const tr_math_budget *, tr_math_refusal *,
                             int32_t *error);
/* the needs of tr_jet_agg_init / _build (TR_JET_NEED_CPU), tr_mgpu_jet_aggregate (TR_JET_NEED_DEVICE: its device
 * copies) or tr_jet_agg_from_states (TR_JET_NEED_ADOPT); the scope is checked against the plan as there */
int32_t tr_jet_agg_need(const tr_jet_plan *, const tr_scope *, int32_t target, tr_math_need *);
/* adopts states computed elsewhere (the readback of tr_mgpu_jet_aggregate) for this plan and scope */
tr_jet_agg *tr_jet_agg_from_states(const tr_jet_plan *, const tr_scope *, const tr_jet_states *,
                                   const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the sum law: rows, counts and every component add (same plan descriptor and group count, else
 * TR_MATH_DOMAIN_ERROR); the result is unscoped; budget: as an adoption of its size */
tr_jet_agg *tr_jet_agg_add(const tr_jet_agg *, const tr_jet_agg *, const tr_math_budget *, tr_math_refusal *,
                           int32_t *error);
int32_t tr_jet_agg_get_info(const tr_jet_agg *, tr_jet_agg_info *);
/* pointers into the aggregate's own states (read only; valid while it lives) */
int32_t tr_jet_agg_get_states(const tr_jet_agg *, tr_jet_states *);
/* the monomials of the aggregate's jets (as tr_jet_plan_monomials) */
int32_t tr_jet_agg_monomials(const tr_jet_agg *, uint8_t *exps, uint64_t capacity);
/* 1 when both have the same plan descriptor, groups and identical states (scopes not compared), else 0 */
int32_t tr_jet_agg_equal(const tr_jet_agg *, const tr_jet_agg *);
/* Canonical components of the jets of groups first .. first + count - 1: num[k] / den[k] in lowest terms, den > 0,
 * S per group in index order; TR_JET_DERIVATIVE scales by alpha!. TR_MATH_OVERFLOW (the outputs then unspecified)
 * only when a scaled numerator leaves int128. */
int32_t tr_jet_agg_read(const tr_jet_agg *, uint64_t first, uint64_t count, int32_t convention, tr_math_i128 *num,
                        int64_t *den);
void tr_jet_agg_free(tr_jet_agg *);

/* ------------------------------------------------------------------ values in a carrier */
/* p: the prime of TR_CARRIER_GFP (ignored otherwise) */
tr_jet_values *tr_jet_values_from_agg(const tr_jet_agg *, int32_t carrier, int32_t convention, uint64_t p,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_jet_values *tr_jet_values_convert(const tr_jet_values *, int32_t carrier, uint64_t p, const tr_math_budget *,
                                     tr_math_refusal *, int32_t *error);
tr_jet_values *tr_jet_values_add(const tr_jet_values *, const tr_jet_values *, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
int32_t tr_jet_values_get_info(const tr_jet_values *, tr_jet_values_info *);
/* every component, in jet-major index order; TR_MATH_CARRIER_ERROR when the values have another carrier */
int32_t tr_jet_values_read_q64(const tr_jet_values *, tr_num_rational *out, uint64_t capacity);
int32_t tr_jet_values_read_gfp(const tr_jet_values *, uint64_t *out, uint64_t capacity);
int32_t tr_jet_values_read_interval(const tr_jet_values *, tr_math_interval *out, uint64_t capacity);
/* component k of a BIGQ values object (borrowed, valid while the values live), NULL otherwise */
const tr_bigq *tr_jet_values_bigq(const tr_jet_values *, uint64_t k);
/* 1 when both have the same plan, convention, carrier, field and identical exact components (INTERVAL: identical
 * endpoints), else 0 */
int32_t tr_jet_values_equal(const tr_jet_values *, const tr_jet_values *);
void tr_jet_values_free(tr_jet_values *);

/* ------------------------------------------------------------------ the CUDA library (math_jet_gpu.cu)
 * plan: tr_jet_plan_get_view; scope: tr_scope_get_view (host pointers, copied to the device and budgeted with
 * tr_jet_device_need); out: host arrays with out->ngroups = scope->ngroups and out->support = plan->support,
 * overwritten with exactly the states tr_jet_agg_build computes (then adopt them with tr_jet_agg_from_states).
 * TR_MATH_OVERFLOW when a valid selected row is refused (as on the CPU; out is then unspecified). budget: NULL =
 * defaults. Returns 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_jet_aggregate(const tr_jet_view *plan, const tr_scope_view *scope, const tr_math_budget *budget,
                          tr_jet_states *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
