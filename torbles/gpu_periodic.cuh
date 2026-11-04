/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#include "periodic.h"

#ifdef EMULATE

extern "C" void *gpu_periodic_open(const atable_t *, uint32_t, uint32_t, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    return NULL;
}
extern "C" void *gpu_periodic_open_sectors(const atable_t *, const tp_sector_t *, uint32_t,
                                            uint32_t, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    return NULL;
}
extern "C" int gpu_periodic_reduce(void *, const pinstr_t *, int32_t,
                                   const tp_plan_t *, tp_state_t *, double *) { return -1; }
extern "C" void gpu_periodic_close(void *) {}

#else

#include <stddef.h>

/* n <= UINT32_MAX bounds complete-word multiplicity to 26 bits. */
enum { GPU_PERIODIC_WEIGHT_BITS = 26 };
static constexpr uint32_t GPU_PERIODIC_WEIGHT_MASK = (1u << GPU_PERIODIC_WEIGHT_BITS) - 1;

struct gpu_periodic_handle {
    gtable_t host, *table;
    uint64_t *planes;
    uint32_t *weights;
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

template <bool Sectors>
static __global__ void gpu_periodic_words(const gtable_t *t, const qplan_t *prog,
                                          tp_plan_t plan, uint32_t repeats, uint32_t remainder,
                                          uint32_t tail, const uint32_t *weights,
                                          tp_state_t *states, uint32_t *bad_group) {
    uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= t->nw) return;
    uint64_t pending = eval_prog_g(t, prog, word);
    uint32_t copies, partial_rows;
    if (Sectors) {
        uint32_t weight = weights[word];
        copies = weight & GPU_PERIODIC_WEIGHT_MASK;
        partial_rows = weight >> GPU_PERIODIC_WEIGHT_BITS;
    } else {
        copies = repeats + (word < remainder);
        partial_rows = word == remainder ? tail : 0;
    }
    uint64_t partial = partial_rows ? (UINT64_C(1) << partial_rows) - 1 : 0;
    /* An ordinary tail word has no full copies. Exclude padding from extrema and keys. */
    if (Sectors && !copies) pending &= partial;
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
    if (h->weights) cudaFree(h->weights);
    if (h->planes) cudaFree(h->planes);
    free(h);
}

static bool gpu_periodic_source(const atable_t *t, uint32_t column_mask) {
    return t && t->k >= 1 && t->k <= MAXCOL && t->n <= UINT32_MAX &&
           t->nw == t->n / 64 + (t->n % 64 != 0) && !(column_mask >> t->k);
}

static gpu_periodic_handle *gpu_periodic_alloc(const atable_t *t, uint32_t word_count,
                                               uint32_t column_mask, bool sectors, size_t *bytes) {
    size_t columns_width = 0;
    for (int c = 0; c < t->k; ++c) {
        if (!(column_mask >> c & 1)) continue;
        if (t->vflag[c] || t->width[c] < 1 || t->width[c] > 32) return NULL;
        for (int j = 0; j < t->width[c]; ++j)
            if (!t->V[c][j]) return NULL;
        columns_width += t->width[c];
    }
    if (columns_width > SIZE_MAX / sizeof(uint64_t) / word_count) return NULL;
    *bytes = columns_width * word_count * sizeof(uint64_t);

    auto *h = (gpu_periodic_handle *)calloc(1, sizeof(gpu_periodic_handle));
    if (!h) return NULL;
    if ((*bytes && cudaMalloc((void **)&h->planes, *bytes) != cudaSuccess) ||
        (sectors && cudaMalloc((void **)&h->weights, (size_t)word_count * sizeof(uint32_t)) != cudaSuccess) ||
        cudaMalloc((void **)&h->table, sizeof(gtable_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->prog, sizeof(qplan_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->states, TP_MAX_GROUPS * sizeof(tp_state_t)) != cudaSuccess ||
        cudaMalloc((void **)&h->bad_group, sizeof(uint32_t)) != cudaSuccess ||
        cudaEventCreate(&h->start) != cudaSuccess || cudaEventCreate(&h->stop) != cudaSuccess) {
        gpu_periodic_close(h);
        return NULL;
    }

    h->host.k = t->k;
    h->host.nw = word_count;
    h->host.n = (uint64_t)word_count * 64;
    size_t offset = 0;
    for (int c = 0; c < t->k; ++c) {
        if (!(column_mask >> c & 1)) continue;
        h->host.width[c] = t->width[c];
        h->host.msb[c] = t->msb[c];
        for (int j = 0; j < t->width[c]; ++j) {
            h->host.P[c][j] = h->planes + offset;
            offset += word_count;
        }
    }
    return h;
}

extern "C" void *gpu_periodic_open(const atable_t *t, uint32_t cycle_words,
                                   uint32_t column_mask, uint64_t *plane_bytes) {
    if (plane_bytes) *plane_bytes = 0;
    if (!gpu_periodic_source(t, column_mask) || !cycle_words ||
        cycle_words > TP_MAX_CYCLE_WORDS || cycle_words > t->n / 64) return NULL;

    size_t bytes = 0;
    auto *h = gpu_periodic_alloc(t, cycle_words, column_mask, false, &bytes);
    if (!h) return NULL;
    h->repeats = (uint32_t)(t->n / 64 / cycle_words);
    h->remainder = (uint32_t)(t->n / 64 % cycle_words);
    h->tail = (uint32_t)(t->n % 64);
    if (bytes) {
        auto *decoded = (uint64_t *)malloc(bytes);
        if (!decoded) {
            gpu_periodic_close(h);
            return NULL;
        }
        size_t offset = 0;
        for (int c = 0; c < t->k; ++c) {
            for (int j = 0; j < h->host.width[c]; ++j) {
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

static uint32_t gpu_periodic_sector_words(const tp_sector_t &s) {
    return s.cycle_words ? s.cycle_words : s.full_words + (s.tail_rows != 0);
}

static bool gpu_periodic_sector_weights(gpu_periodic_handle *h, const tp_sector_t *sectors,
                                        uint32_t nsectors, uint32_t word_count) {
    size_t bytes = (size_t)word_count * sizeof(uint32_t);
    auto *weights = (uint32_t *)malloc(bytes);
    if (!weights) return false;
    size_t offset = 0;
    for (uint32_t i = 0; i < nsectors; ++i) {
        const tp_sector_t &s = sectors[i];
        if (s.cycle_words) {
            uint32_t copies = s.full_words / s.cycle_words, remainder = s.full_words % s.cycle_words;
            for (uint32_t word = 0; word < s.cycle_words; ++word) {
                uint32_t tail = word == remainder ? s.tail_rows : 0;
                weights[offset++] = (tail << GPU_PERIODIC_WEIGHT_BITS) | (copies + (word < remainder));
            }
        } else {
            for (uint32_t word = 0; word < s.full_words; ++word) weights[offset++] = 1;
            if (s.tail_rows) weights[offset++] = s.tail_rows << GPU_PERIODIC_WEIGHT_BITS;
        }
    }
    cudaError_t status = cudaMemcpy(h->weights, weights, bytes, cudaMemcpyHostToDevice);
    free(weights);
    return status == cudaSuccess;
}

static bool gpu_periodic_sector_planes(gpu_periodic_handle *h, const atable_t *t,
                                       const tp_sector_t *sectors, uint32_t nsectors) {
    uint32_t words = (uint32_t)h->host.nw;
    size_t bytes = (size_t)words * sizeof(uint64_t);
    /* Bound staging to one packed plane. Decode XOR planes in place on CUDA. */
    auto *packed = (uint64_t *)malloc(bytes);
    if (!packed) return false;
    bool ok = true;
    size_t plane_offset = 0;
    for (int c = 0; ok && c < t->k; ++c) {
        int width = h->host.width[c];
        if (!width) continue;
        uint64_t *column = h->planes + plane_offset;
        for (int j = 0; j < width; ++j) {
            size_t offset = 0;
            for (uint32_t i = 0; i < nsectors; ++i) {
                const tp_sector_t &s = sectors[i];
                uint32_t n = gpu_periodic_sector_words(s);
                memcpy(packed + offset, t->V[c][j] + s.first_word, (size_t)n * sizeof(uint64_t));
                offset += n;
            }
            if (cudaMemcpy(h->planes + plane_offset, packed, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                ok = false;
                break;
            }
            plane_offset += words;
        }
        if (ok) {
            kdecode<<<(words + 255) / 256, 256>>>(column, words, width);
            ok = cudaGetLastError() == cudaSuccess;
        }
    }
    free(packed);
    return ok;
}

extern "C" void *gpu_periodic_open_sectors(const atable_t *t, const tp_sector_t *sectors,
                                            uint32_t nsectors, uint32_t column_mask,
                                            uint64_t *plane_bytes) {
    if (plane_bytes) *plane_bytes = 0;
    if (!gpu_periodic_source(t, column_mask) || !sectors || !nsectors || nsectors > t->nw) return NULL;
    uint64_t covered = 0;
    uint32_t words = 0;
    for (uint32_t i = 0; i < nsectors; ++i) {
        const tp_sector_t &s = sectors[i];
        if (s.first_word != covered / 64 || (!s.full_words && !s.tail_rows) ||
            s.cycle_words > s.full_words || s.cycle_words > TP_MAX_CYCLE_WORDS ||
            s.tail_rows >= 64 || (s.tail_rows && i + 1 != nsectors)) return NULL;
        covered += (uint64_t)s.full_words * 64 + s.tail_rows;
        if (covered > t->n) return NULL;
        words += gpu_periodic_sector_words(s);
    }
    if (covered != t->n) return NULL;

    size_t bytes = 0;
    auto *h = gpu_periodic_alloc(t, words, column_mask, true, &bytes);
    if (!h) return NULL;
    if (!gpu_periodic_sector_weights(h, sectors, nsectors, words) ||
        (bytes && !gpu_periodic_sector_planes(h, t, sectors, nsectors)) ||
        cudaMemcpy(h->table, &h->host, sizeof(h->host), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaStreamSynchronize(0) != cudaSuccess) {
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
    if (h->weights)
        gpu_periodic_words<true><<<blocks, threads>>>(h->table, h->prog, *plan, 0, 0, 0,
                                                      h->weights, h->states, h->bad_group);
    else
        gpu_periodic_words<false><<<blocks, threads>>>(h->table, h->prog, *plan, h->repeats,
                                                       h->remainder, h->tail, NULL, h->states,
                                                       h->bad_group);
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
