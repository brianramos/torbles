/* Exact packed UTF-8 referrer projection and counted moments.
 * SPDX-License-Identifier: Apache-2.0 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define REFERRER_DOMAIN_LIMIT (UINT32_C(1) << 27)
#define REFERRER_HASH_TAG UINT64_C(0xffffffff00000000)
#define REFERRER_START_SHIFT 32
#define REFERRER_LF_SHIFT 40

static uint32_t referrer_start(uint64_t span)
{
    return (uint8_t)(span >> REFERRER_START_SHIFT);
}

static uint8_t referrer_lf(uint64_t span)
{
    return (uint8_t)((span >> REFERRER_LF_SHIFT) & 1);
}

static uint64_t referrer_project(const uint8_t *s, uint32_t length)
{
    /* Python's default ^https?://(?:www\\.)?([^/]+)/.*$ with replacement \\1. */
    uint32_t start;
    if (length >= 7 && memcmp(s, "http://", 7) == 0) start = 7;
    else if (length >= 8 && memcmp(s, "https://", 8) == 0) start = 8;
    else return length;

    const uint8_t *slash = memchr(s + start, '/', length - start);
    if (!slash || slash == s + start) return length;
    const uint32_t end = (uint32_t)(slash - s);
    const uint32_t tail = length - end - 1;
    /* A final LF may remain unmatched by $, whereas an interior path LF
     * prevents a match. LF inside the captured host is an ordinary byte. */
    if (tail > 1 && memchr(slash + 1, '\n', tail - 1)) return length;
    if (end - start > 4 && memcmp(s + start, "www.", 4) == 0) start += 4;
    return (end - start) | ((uint64_t)start << REFERRER_START_SHIFT)
                         | ((uint64_t)(tail && s[length - 1] == '\n') << REFERRER_LF_SHIFT);
}

static uint64_t referrer_hash(const uint8_t *s, uint64_t span)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint32_t length = (uint32_t)span;
    for (uint32_t i = 0; i < length; ++i)
        hash = (hash ^ s[i]) * UINT64_C(1099511628211);
    if (referrer_lf(span)) hash = (hash ^ '\n') * UINT64_C(1099511628211);
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    return hash ^ (hash >> 33);
}

static int referrer_equal(const uint8_t *a, uint64_t a_span, const uint8_t *b, uint64_t b_span)
{
    const uint32_t a_len = (uint32_t)a_span, b_len = (uint32_t)b_span;
    if ((uint64_t)a_len + referrer_lf(a_span) != (uint64_t)b_len + referrer_lf(b_span)) return 0;
    const uint32_t common = a_len < b_len ? a_len : b_len;
    if (memcmp(a, b, common) != 0) return 0;
    return a_len == b_len || (a_len > common ? a[common] : b[common]) == '\n';
}

/* Map every original dictionary code to the lowest code with the same exact
 * projected bytes. lengths are original byte lengths, while spans describe
 * the capture (or the entire unmatched input) and optional remaining LF. */
int at_referrer_prepare(const uint8_t *blob, uint64_t blob_bytes, const int64_t *offsets,
                        uint32_t domain, uint32_t *mapping, uint32_t *lengths, uint64_t *spans)
{
    if ((!blob && blob_bytes) || blob_bytes > SIZE_MAX || !offsets || !domain ||
        domain > REFERRER_DOMAIN_LIMIT || !mapping || !lengths || !spans ||
        offsets[0] != 0 || offsets[domain] < 0 || (uint64_t)offsets[domain] != blob_bytes) return -1;

    size_t capacity = 16;
    while (domain > capacity - capacity / 4) {
        if (capacity > SIZE_MAX / sizeof(uint64_t) / 2) return -2;
        capacity *= 2;
    }
    uint64_t *slots = calloc(capacity, sizeof(*slots));
    if (!slots) return -2;
    const size_t mask = capacity - 1;
    const uint8_t *bytes = blob ? blob : (const uint8_t *)"";
    int64_t begin = 0;
    int status = 0;
    for (uint32_t code = 0; code < domain; ++code) {
        const int64_t end = offsets[code + 1];
        if (end < begin || (uint64_t)end > blob_bytes || (uint64_t)(end - begin) > UINT32_MAX) {
            status = -1;
            break;
        }
        const uint32_t length = (uint32_t)(end - begin);
        const uint64_t span = referrer_project(bytes + (size_t)begin, length);
        lengths[code] = length;
        spans[code] = span;
        const uint8_t *text = bytes + (size_t)begin + referrer_start(span);
        const uint64_t hash = referrer_hash(text, span);
        const uint64_t tag = hash & REFERRER_HASH_TAG;
        size_t slot = (size_t)hash & mask;
        for (; slots[slot]; slot = (slot + 1) & mask) {
            const uint64_t entry = slots[slot];
            if ((entry & REFERRER_HASH_TAG) != tag) continue;
            const uint32_t rep = (uint32_t)entry - 1;
            const uint64_t other_span = spans[rep];
            const uint8_t *other = bytes + (size_t)offsets[rep] + referrer_start(other_span);
            if (referrer_equal(text, span, other, other_span)) break;
        }
        if (!slots[slot]) slots[slot] = tag | ((uint64_t)code + 1);
        mapping[code] = (uint32_t)slots[slot] - 1;
        begin = end;
    }
    free(slots);
    return status;
}

typedef struct {
    uint64_t sum;
    uint32_t count, min;
} referrer_group;

/* Push selected dictionary multiplicities through a prepared projection.
 * Results contain the representative span and the minimum original code. */
int at_referrer_counted_moments(uint32_t domain, const uint32_t *mapping, const uint32_t *lengths,
                                const uint64_t *spans, const uint32_t *counts, int32_t exclude_empty,
                                uint32_t min_count, uint32_t cap, uint32_t *out_reps, uint32_t *out_counts,
                                uint64_t *out_sums, uint32_t *out_min, uint32_t *out_start,
                                uint32_t *out_length, uint8_t *out_lf, uint32_t *out_n)
{
    if (!out_n) return -1;
    *out_n = 0;
    if (!domain || domain > REFERRER_DOMAIN_LIMIT || !mapping || !lengths || !spans || !counts ||
        (exclude_empty != 0 && exclude_empty != 1) || !cap || !out_reps || !out_counts ||
        !out_sums || !out_min || !out_start || !out_length || !out_lf) return -1;

    referrer_group *groups = calloc(domain, sizeof(*groups));
    if (!groups) return -2;
    int status = -1;
    uint32_t emitted = 0;
    for (uint32_t code = 0; code < domain; ++code) {
        const uint32_t count = counts[code];
        if (!count || (exclude_empty && !lengths[code])) continue;
        const uint32_t rep = mapping[code];
        if (rep >= domain || mapping[rep] != rep) goto done;
        referrer_group *group = &groups[rep];
        if (count > UINT32_MAX - group->count) goto done;
        if (!group->count) group->min = code;
        group->count += count;
        /* The bounded uint32 total count times the maximum uint32 length
         * fits uint64, including after adding all contributions. */
        group->sum += (uint64_t)count * lengths[code];
    }
    for (uint32_t rep = 0; rep < domain; ++rep) {
        const referrer_group *group = &groups[rep];
        if (!group->count || group->count < min_count) continue;
        if (emitted == cap) goto done;
        const uint64_t span = spans[rep];
        const uint32_t start = referrer_start(span), length = (uint32_t)span;
        const uint8_t lf = referrer_lf(span);
        if ((span >> (REFERRER_LF_SHIFT + 1)) || (uint64_t)start + length + lf > lengths[rep]) goto done;
        out_reps[emitted] = rep;
        out_counts[emitted] = group->count;
        out_sums[emitted] = group->sum;
        out_min[emitted] = group->min;
        out_start[emitted] = start;
        out_length[emitted] = length;
        out_lf[emitted] = lf;
        ++emitted;
    }
    *out_n = emitted;
    status = 0;
done:
    free(groups);
    return status;
}
