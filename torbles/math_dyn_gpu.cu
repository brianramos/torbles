/* DYN-001 / DYN-002 on a CUDA device: the batched exact application of transfer operators to many groups' moment
 * vectors, and modular operator powers per (operator, prime), with exactly the results of tr_dyn_apply_view_eval,
 * tr_dyn_powmod_view_eval and tr_dyn_evolvemod_view_eval (math_dyn.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Application: one work item per (operator, group, basis row) runs tr_dyn_apply_entry (SYM-001's 512-bit row
 * evaluator tr_sym_row_exact on the operator's int128 numerators and the group's int128 moments) and writes its own
 * output slot. Modular powers: the numerators are reduced modulo every prime on the device (one work item per
 * (operator, prime, entry)); the host drives the binary exponentiation, every matrix product one launch with one work
 * item per (operator, prime, entry) running tr_dyn_mulmod_entry; the evolution reduces the moments the same way and
 * applies the powers, one work item per (operator, prime, group, row) running tr_dyn_applymod_entry. Every output slot
 * is written by exactly one work item, so no result depends on the thread order; modular results are unique, so the
 * residues equal the CPU reference's bit for bit. The host (tr_dyn_power_adopt, tr_dyn_evolution_adopt) reconstructs
 * them by Chinese remaindering within the proven bound and checks them modulo the further primes. */
#include "math_cuda.h"
#include "math_dyn.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_dyn_apply_view, TR_DYN_ABI);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_dyn_mod_view, TR_DYN_ABI);

TR_KERNEL void tr_dyn_apply_kernel(tr_dyn_apply_view v, uint64_t total, tr_sym_wide *out, uint64_t *flag) {
    TR_GRID_LOOP(w, total) {
        if (!tr_dyn_apply_entry(&v, w, &out[w])) tr_atomic_or_u64(flag, 1);
    }
}

/* out[(op x np + q) x e2 + x] = num[op x e2 + x] mod p_q */
TR_KERNEL void tr_dyn_reduce_kernel(const tr_math_i128 *num, const tr_gfp *primes, uint64_t np, uint64_t e2, uint64_t nmat,
                                    uint64_t *out) {
    TR_GRID_LOOP(w, nmat * e2) {
        const uint64_t mat = w / e2, x = w % e2, op = mat / np, q = mat % np;
        tr_gfp f = primes[q];
        out[w] = tr_dyn_reduce_i128(&f, num[op * e2 + x]);
    }
}

/* R = I modulo every prime */
TR_KERNEL void tr_dyn_identity_kernel(const tr_gfp *primes, uint64_t np, int32_t nb, uint64_t nmat, uint64_t *R) {
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    TR_GRID_LOOP(w, nmat * e2) {
        const uint64_t x = w % e2, q = (w / e2) % np;
        R[w] = x % (uint64_t)(nb + 1) == 0 ? 1 % primes[q].p : 0;
    }
}

/* C = A B modulo the matrix's prime, every (operator, prime) block */
TR_KERNEL void tr_dyn_mulmod_kernel(const uint64_t *A, const uint64_t *B, const tr_gfp *primes, uint64_t np, int32_t nb,
                                    uint64_t nmat, uint64_t *C) {
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    TR_GRID_LOOP(w, nmat * e2) {
        const uint64_t mat = w / e2, x = w % e2, q = mat % np;
        tr_gfp f = primes[q];
        C[w] = tr_dyn_mulmod_entry(&f, A + mat * e2, B + mat * e2, nb, (int32_t)(x / (uint64_t)nb), (int32_t)(x % (uint64_t)nb));
    }
}

/* mm[(q x G + g) x nb + j] = m_g[j] mod p_q */
TR_KERNEL void tr_dyn_reducemom_kernel(const tr_math_i128 *moments, const tr_gfp *primes, uint64_t np, uint64_t gn, uint64_t *mm) {
    TR_GRID_LOOP(w, np * gn) {
        tr_gfp f = primes[w / gn];
        mm[w] = tr_dyn_reduce_i128(&f, moments[w % gn]);
    }
}

/* out[((op x np + q) x G + g) x nb + i] = (P_(op,q) m_g)_i mod p_q */
TR_KERNEL void tr_dyn_applymod_kernel(const uint64_t *P, const uint64_t *mm, const tr_gfp *primes, uint64_t np, uint64_t G,
                                      int32_t nb, uint64_t nmat, uint64_t *out) {
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb, gn = G * (uint64_t)nb;
    TR_GRID_LOOP(w, nmat * gn) {
        const uint64_t mat = w / gn, r = w % gn, g = r / (uint64_t)nb, i = r % (uint64_t)nb, q = mat % np;
        tr_gfp f = primes[q];
        out[w] = tr_dyn_applymod_entry(&f, P + mat * e2, mm + (q * G + g) * (uint64_t)nb, nb, (int32_t)i);
    }
}

/* the powers N_op^e mod p_q into one of bufs[0..2] (device buffers of nmat x e2 residues); *which: the one holding R */
static int tr_dyn_powers_device(const tr_dyn_mod_view *v, const tr_math_i128 *dnum, const tr_gfp *dprimes, uint64_t *bufs[3],
                                int *which) {
    const uint64_t e2 = (uint64_t)v->nb * (uint64_t)v->nb, nmat = (uint64_t)v->nops * v->nprimes;
    int iR = 0, iB = 1, iT = 2;
    TR_LAUNCH(tr_dyn_identity_kernel, nmat * e2, dprimes, v->nprimes, v->nb, nmat, bufs[iR]);
    TR_LAUNCH(tr_dyn_reduce_kernel, nmat * e2, dnum, dprimes, v->nprimes, e2, nmat, bufs[iB]);
    uint64_t e = v->exponent;
    while (e) {
        if (e & 1) {
            TR_LAUNCH(tr_dyn_mulmod_kernel, nmat * e2, bufs[iR], bufs[iB], dprimes, v->nprimes, v->nb, nmat, bufs[iT]);
            int s = iR; iR = iT; iT = s;
        }
        e >>= 1;
        if (e) {
            TR_LAUNCH(tr_dyn_mulmod_kernel, nmat * e2, bufs[iB], bufs[iB], dprimes, v->nprimes, v->nb, nmat, bufs[iT]);
            int s = iB; iB = iT; iT = s;
        }
    }
    *which = iR;
    return tr_cuda_sync();
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_dyn_apply(const tr_dyn_apply_view *view, tr_sym_wide *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_dyn_apply_view_check(view) || !out) return -1;
    const uint64_t nb = (uint64_t)view->nb, e2 = nb * nb, total = (uint64_t)view->nops * view->ngroups * nb;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = total;
    need.need[TR_BUDGET_BYTES] = tr_dyn_apply_device_bytes(view);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    void *mem[4];
    int nmem = 0;
#define TR_DYN_DEV(ptr, src, bytes) do { if (!rc) { void *p_ = NULL; rc = tr_cuda_alloc(&p_, (size_t)(bytes)); \
        if (!rc) { mem[nmem++] = p_; if (src) rc = tr_cuda_copy_in(p_, (src), (size_t)(bytes)); } (ptr) = p_; } } while (0)
    tr_dyn_apply_view d = *view;
    void *p = NULL;
    TR_DYN_DEV(p, view->num, (uint64_t)view->nops * e2 * 16); d.num = (const tr_math_i128 *)p;
    TR_DYN_DEV(p, view->moments, view->ngroups * nb * 16); d.moments = (const tr_math_i128 *)p;
    tr_sym_wide *dout = NULL;
    uint64_t *dflag = NULL;
    TR_DYN_DEV(p, (const void *)NULL, total * sizeof(tr_sym_wide)); dout = (tr_sym_wide *)p;
    TR_DYN_DEV(p, (const void *)NULL, 8); dflag = (uint64_t *)p;
#undef TR_DYN_DEV
    if (!rc) {
        TR_LAUNCH(tr_dyn_apply_kernel, total, d, total, dout, dflag);
        rc = tr_cuda_sync();
    }
    uint64_t flag = 0;
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)(total * sizeof(tr_sym_wide)));
    if (!rc) rc = tr_cuda_copy_out(&flag, dflag, 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc && flag) rc = TR_MATH_OVERFLOW;
    return rc;
}

static int tr_dyn_mod_entry(const tr_dyn_mod_view *view, const tr_math_i128 *moments, uint64_t G, uint64_t *out,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_dyn_mod_view_check(view) || !out || (G && !moments)) return -1;
    const uint64_t nb = (uint64_t)view->nb, e2 = nb * nb, nmat = (uint64_t)view->nops * view->nprimes;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_TENSOR] = e2;
    need.need[TR_BUDGET_ITERATIONS] = view->exponent;
    need.need[TR_BUDGET_OUTPUT] = G ? tr_budget_mul((uint64_t)view->nops, G * nb) : tr_budget_mul((uint64_t)view->nops, e2);
    need.need[TR_BUDGET_BYTES] = tr_dyn_mod_device_bytes(view, G);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    void *mem[8];
    int nmem = 0;
#define TR_DYN_DEV(ptr, src, bytes) do { if (!rc) { void *p_ = NULL; rc = tr_cuda_alloc(&p_, (size_t)(bytes)); \
        if (!rc) { mem[nmem++] = p_; if (src) rc = tr_cuda_copy_in(p_, (src), (size_t)(bytes)); } (ptr) = p_; } } while (0)
    void *p = NULL;
    const tr_math_i128 *dnum = NULL;
    const tr_gfp *dprimes = NULL;
    uint64_t *bufs[3] = {NULL, NULL, NULL}, *dmm = NULL, *dout = NULL;
    const tr_math_i128 *dmom = NULL;
    TR_DYN_DEV(p, view->num, (uint64_t)view->nops * e2 * 16); dnum = (const tr_math_i128 *)p;
    TR_DYN_DEV(p, view->primes, view->nprimes * sizeof(tr_gfp)); dprimes = (const tr_gfp *)p;
    for (int b = 0; b < 3; b++) { TR_DYN_DEV(p, (const void *)NULL, nmat * e2 * 8); bufs[b] = (uint64_t *)p; }
    if (G) {
        TR_DYN_DEV(p, moments, G * nb * 16); dmom = (const tr_math_i128 *)p;
        TR_DYN_DEV(p, (const void *)NULL, view->nprimes * G * nb * 8); dmm = (uint64_t *)p;
        TR_DYN_DEV(p, (const void *)NULL, nmat * G * nb * 8); dout = (uint64_t *)p;
    }
#undef TR_DYN_DEV
    int which = 0;
    if (!rc) rc = tr_dyn_powers_device(view, dnum, dprimes, bufs, &which);
    if (!rc && !G) rc = tr_cuda_copy_out(out, bufs[which], (size_t)(nmat * e2 * 8));
    if (!rc && G) {
        TR_LAUNCH(tr_dyn_reducemom_kernel, view->nprimes * G * nb, dmom, dprimes, view->nprimes, G * nb, dmm);
        TR_LAUNCH(tr_dyn_applymod_kernel, nmat * G * nb, bufs[which], dmm, dprimes, view->nprimes, G, view->nb, nmat, dout);
        rc = tr_cuda_sync();
        if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)(nmat * G * nb * 8));
    }
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}
int tr_mgpu_dyn_powmod(const tr_dyn_mod_view *view, uint64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    return tr_dyn_mod_entry(view, NULL, 0, out, budget, refusal);
}
int tr_mgpu_dyn_evolvemod(const tr_dyn_mod_view *view, const tr_math_i128 *moments, uint64_t ngroups, uint64_t *out,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    if (!ngroups) return -1;
    return tr_dyn_mod_entry(view, moments, ngroups, out, budget, refusal);
}
#ifdef __cplusplus
}
#endif
