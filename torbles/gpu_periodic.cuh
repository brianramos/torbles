/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#include "periodic.h"

#ifdef EMULATE

extern "C" void *gpu_periodic_open(const atable_t *, uint32_t, uint32_t, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    return NULL;
}
extern "C" int gpu_periodic_reduce(void *, const pinstr_t *, int32_t,
                                   const tp_plan_t *, tp_state_t *, double *) { return -1; }
extern "C" void gpu_periodic_close(void *) {}

#else

#include <stddef.h>

struct gpu_periodic_handle {
    gtable_t host, *table;
    uint64_t *planes;
    qplan_t *prog;
    tp_state_t *states;
    uint32_t *bad_group;
    cudaEvent_t start, stop;
    uint32_t repeats, remainder, tail;
};

DEV uint64_t periodic_valid_g(const gtable_t *t, int col, uint32_t word) {
    return col < 0 ? UINT64_MAX : X(t, col, 0, word);
}

DEV uint64_t periodic_count_g(uint64_t mask, uint32_t repeats, uint64_t tail) {
    return (uint64_t)popc(mask) * repeats + popc(mask & tail);
}

DEV void periodic_measure_g(const gtable_t *t, uint32_t word, uint64_t mask,
                            uint32_t repeats, uint64_t tail, const tp_measure_t &measure,
                            tp_state_t *state, uint32_t index) {
    mask &= periodic_valid_g(t, measure.valid, word);
    if (!mask) return;
    ADD_U64(&state->valid[index], periodic_count_g(mask, repeats, tail));
    if (!measure.flags) return;

    uint64_t sum = 0, low_mask = mask, high_mask = mask;
    uint32_t lo = 0, hi = 0;
    int col = measure.col, width = t->width[col];
    for (int bit = width - 1; bit >= 0; --bit) {
        uint64_t plane = X(t, col, t->msb[col] ? width - 1 - bit : bit, word);
        if (measure.flags & TP_SUM)
            sum += periodic_count_g(mask & plane, repeats, tail) << bit;
        if (measure.flags & TP_MIN) {
            uint64_t lower = low_mask & ~plane;
            if (lower) low_mask = lower;
            else lo |= UINT32_C(1) << bit;
        }
        if (measure.flags & TP_MAX) {
            uint64_t higher = high_mask & plane;
            if (higher) {
                high_mask = higher;
                hi |= UINT32_C(1) << bit;
            }
        }
    }
    if (measure.flags & TP_SUM) ADD_U64(&state->sums[index], sum);
    /* Store the complemented minimum so a zero-filled buffer suffices. */
    if (measure.flags & TP_MIN) atomicMax(&state->mins[index], UINT32_MAX - lo);
    if (measure.flags & TP_MAX) atomicMax(&state->maxs[index], hi);
}

static __global__ void gpu_periodic_words(const gtable_t *t, const qplan_t *prog,
                                          tp_plan_t plan, uint32_t repeats, uint32_t remainder,
                                          uint32_t tail, tp_state_t *states, uint32_t *bad_group) {
    uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= t->nw) return;
    uint64_t pending = eval_prog_g(t, prog, word);
    uint32_t copies = repeats + (word < remainder);
    /* The certificate covers occupied lanes of the original partial word. */
    uint64_t partial = word == remainder && tail ? (UINT64_C(1) << tail) - 1 : 0;
    while (pending) {
        int lane = ctz64(pending);
        uint64_t mask = pending;
        uint32_t group = 0;
        for (uint32_t i = 0; i < plan.nkeys; ++i) {
            const tp_key_t &key = plan.keys[i];
            uint64_t valid = periodic_valid_g(t, key.valid, word);
            uint32_t code = key.radix - 1;
            if (!(valid >> lane & 1)) {
                mask &= ~valid;
            } else {
                mask &= valid;
                code = 0;
                for (int j = 0; j < t->width[key.col]; ++j) {
                    uint64_t plane = X(t, key.col, j, word);
                    uint32_t bit = (uint32_t)(plane >> lane & 1);
                    mask &= bit ? plane : ~plane;
                    code |= bit << pw_g(t, key.col, j);
                }
                if (code >= key.radix - (key.valid >= 0)) {
                    atomicExch(bad_group, 1u);
                    return;
                }
            }
            group = group * key.radix + code;
        }
        pending &= ~mask;
        tp_state_t *state = states + group;
        ADD_U64(&state->count, periodic_count_g(mask, copies, partial));
        for (uint32_t i = 0; i < plan.nmeasures; ++i)
            periodic_measure_g(t, word, mask, copies, partial, plan.measures[i], state, i);
    }
}

static bool gpu_periodic_column(const gtable_t &t, int col) {
    return col >= 0 && col < t.k && t.width[col] > 0;
}

static bool gpu_periodic_validity(const gtable_t &t, int col) {
    return col == -1 || (gpu_periodic_column(t, col) && t.width[col] == 1);
}

static bool gpu_periodic_program(const gtable_t &t, const pinstr_t *prog, int32_t nprog) {
    if (nprog < 0 || nprog > TP_MAX_PROGRAM || (nprog && !prog)) return false;
    int depth = 0;
    for (int32_t i = 0; i < nprog; ++i) {
        const pinstr_t &p = prog[i];
        if (p.op == P_AND || p.op == P_OR || p.op == P_XOR) {
            if (depth < 2) return false;
            --depth;
            continue;
        }
        if (p.op == P_NOT) {
            if (!depth) return false;
            continue;
        }
        if (++depth > TP_MAX_STACK || p.op < P_TRUE || p.op > P_XOR) return false;
        if (p.op == P_TRUE) continue;
        if (!gpu_periodic_column(t, p.col)) return false;
        uint32_t width = (uint32_t)t.width[p.col];
        if (p.op == P_PREFIX && p.b > width) return false;
        if (p.op == P_BIT && p.a >= 32) return false;
        if ((p.op == P_CUBE || p.op == P_CLT || p.op == P_CGE) &&
            (!p.b || (width < 32 && (p.b >> width)))) return false;
    }
    return !nprog || depth == 1;
}

static bool gpu_periodic_plan(const gtable_t &t, const tp_plan_t &plan) {
    if (plan.nkeys > TP_MAX_KEYS || plan.nmeasures > TP_MAX_MEASURES ||
        !plan.groups || plan.groups > TP_MAX_GROUPS) return false;
    uint32_t groups = 1;
    for (uint32_t i = 0; i < plan.nkeys; ++i) {
        const tp_key_t &key = plan.keys[i];
        if (!gpu_periodic_column(t, key.col) || !gpu_periodic_validity(t, key.valid) ||
            !key.radix || groups > TP_MAX_GROUPS / key.radix) return false;
        groups *= key.radix;
    }
    if (groups != plan.groups) return false;
    for (uint32_t i = 0; i < plan.nmeasures; ++i) {
        const tp_measure_t &measure = plan.measures[i];
        if ((measure.flags & ~(TP_SUM | TP_MIN | TP_MAX)) ||
            !gpu_periodic_validity(t, measure.valid) ||
            ((measure.flags || measure.col != -1) && !gpu_periodic_column(t, measure.col))) return false;
    }
    return true;
}

extern "C" void gpu_periodic_close(void *handle) {
    auto *h = (gpu_periodic_handle *)handle;
    if (!h) return;
    if (h->stop) cudaEventDestroy(h->stop);
    if (h->start) cudaEventDestroy(h->start);
    if (h->bad_group) cudaFree(h->bad_group);
    if (h->states) cudaFree(h->states);
    if (h->prog) cudaFree(h->prog);
    if (h->table) cudaFree(h->table);
    if (h->planes) cudaFree(h->planes);
    free(h);
}

extern "C" void *gpu_periodic_open(const atable_t *t, uint32_t cycle_words,
                                   uint32_t column_mask, uint64_t *plane_bytes) {
    if (plane_bytes) *plane_bytes = 0;
    if (!t || t->k < 1 || t->k > MAXCOL || t->n > UINT32_MAX ||
        !cycle_words || cycle_words > TP_MAX_CYCLE_WORDS || cycle_words > t->n / 64 ||
        t->nw != t->n / 64 + (t->n % 64 != 0) || (column_mask >> t->k)) return NULL;

    size_t columns_width = 0;
    for (int c = 0; c < t->k; ++c) {
        if (!(column_mask >> c & 1)) continue;
        if (t->vflag[c] || t->width[c] < 1 || t->width[c] > 32) return NULL;
        for (int j = 0; j < t->width[c]; ++j)
            if (!t->V[c][j]) return NULL;
        columns_width += t->width[c];
    }

    auto *h = (gpu_periodic_handle *)calloc(1, sizeof(gpu_periodic_handle));
    if (!h) return NULL;
    size_t bytes = columns_width * cycle_words * sizeof(uint64_t);
    if ((bytes && cudaMalloc((void **)&h->planes, bytes) != cudaSuccess) ||
        cudaMalloc((void **)&h->table, sizeof(gtable_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->prog, sizeof(qplan_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->states, TP_MAX_GROUPS * sizeof(tp_state_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->bad_group, sizeof(uint32_t)) != cudaSuccess ||
        cudaEventCreate(&h->start) != cudaSuccess || cudaEventCreate(&h->stop) != cudaSuccess) {
        gpu_periodic_close(h);
        return NULL;
    }

    h->repeats = (uint32_t)(t->n / 64 / cycle_words);
    h->remainder = (uint32_t)(t->n / 64 % cycle_words);
    h->tail = (uint32_t)(t->n % 64);
    h->host.k = t->k;
    h->host.nw = cycle_words;
    h->host.n = (uint64_t)cycle_words * 64;
    if (bytes) {
        auto *decoded = (uint64_t *)malloc(bytes);
        if (!decoded) {
            gpu_periodic_close(h);
            return NULL;
        }
        size_t offset = 0;
        for (int c = 0; c < t->k; ++c) {
            if (!(column_mask >> c & 1)) continue;
            h->host.width[c] = t->width[c];
            h->host.msb[c] = t->msb[c];
            for (int j = 0; j < t->width[c]; ++j) {
                h->host.P[c][j] = h->planes + offset;
                for (uint32_t word = 0; word < cycle_words; ++word) {
                    uint64_t prev = j ? decoded[offset - cycle_words + word] : 0;
                    decoded[offset + word] = prev ^ t->V[c][j][word];
                }
                offset += cycle_words;
            }
        }
        cudaError_t status = cudaMemcpy(h->planes, decoded, bytes, cudaMemcpyHostToDevice);
        free(decoded);
        if (status != cudaSuccess) {
            gpu_periodic_close(h);
            return NULL;
        }
    }
    if (cudaMemcpy(h->table, &h->host, sizeof(h->host), cudaMemcpyHostToDevice) != cudaSuccess) {
        gpu_periodic_close(h);
        return NULL;
    }
    if (plane_bytes) *plane_bytes = bytes;
    return h;
}

extern "C" int gpu_periodic_reduce(void *handle, const pinstr_t *prog, int32_t nprog,
                                   const tp_plan_t *plan, tp_state_t *out, double *ms) {
    auto *h = (gpu_periodic_handle *)handle;
    if (ms) *ms = 0;
    if (!h || !plan || !out || !gpu_periodic_plan(h->host, *plan) ||
        !gpu_periodic_program(h->host, prog, nprog)) return -1;

    qplan_t filter;
    filter.nprog = nprog;
    size_t prog_bytes = (size_t)nprog * sizeof(pinstr_t);
    if (prog_bytes) memcpy(filter.prog, prog, prog_bytes);
    size_t state_bytes = plan->groups * sizeof(tp_state_t);
    if (cudaMemcpy(h->prog, &filter, offsetof(qplan_t, prog) + prog_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(h->states, 0, state_bytes) != cudaSuccess ||
        cudaMemset(h->bad_group, 0, sizeof(uint32_t)) != cudaSuccess ||
        cudaEventRecord(h->start) != cudaSuccess) return -2;

    constexpr uint32_t threads = 64;
    uint32_t blocks = ((uint32_t)h->host.nw + threads - 1) / threads;
    gpu_periodic_words<<<blocks, threads>>>(h->table, h->prog, *plan, h->repeats, h->remainder,
                                           h->tail, h->states, h->bad_group);
    uint32_t bad_group = 0;
    if (cudaGetLastError() != cudaSuccess || cudaEventRecord(h->stop) != cudaSuccess ||
        cudaMemcpy(&bad_group, h->bad_group, sizeof(bad_group), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if (bad_group) return -1;
    if (cudaMemcpy(out, h->states, state_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    if (ms) {
        float elapsed = 0;
        if (cudaEventElapsedTime(&elapsed, h->start, h->stop) != cudaSuccess) return -2;
        *ms = elapsed;
    }
    for (uint32_t group = 0; group < plan->groups; ++group)
        for (uint32_t i = 0; i < TP_MAX_MEASURES; ++i)
            out[group].mins[i] = UINT32_MAX - out[group].mins[i];
    return 0;
}

#endif
