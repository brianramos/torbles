# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU group interchanges over physical equality codes."""

import os

import numpy as np

from torbles import algq, sqlinterchanges
from torbles.sqlencodednative import BY_COUNT, BY_DISTINCT, encoded_groups
from torbles.sqltexthist import code_counts



def enabled():
    return os.environ.get("TORBLES_INTERCHANGES", "auto") != "0"


def count_topk(fact, keys, cap, scope, stats):
    """Use one physical histogram instead of a hash for bounded COUNT ranking."""
    if len(keys) != 1:
        return None
    low, high, _, transform, _ = keys[0]
    if high is not None or transform:
        return None
    domain = fact.maxcode(low) + 1
    if domain > min(1 << 20, fact.n // 8):
        return None
    active = fact.n if scope is fact.used_bm else sum(int(word).bit_count() for word in scope)
    if domain * 8 > active:
        return None
    histogram = code_counts(fact, low, scope)
    if histogram is None:
        return None
    live = np.flatnonzero(histogram)
    scores = histogram[live]
    if len(live) > cap:
        pick = np.argpartition(scores, -cap)[-cap:]
        live, scores = live[pick], scores[pick]
    order = np.lexsort((live, -scores.astype(np.int64)))
    stats["cpu_interchange_histogram_codes"] = domain
    return live[order].astype(np.uint64).reshape(-1, 1), scores[order].astype(np.uint64), None


def _group(fact, keys, measures, cap, scope, distinct, rank):
    return encoded_groups(fact, [key[:-1] for key in keys], measures, cap, scope,
                          distinct=distinct, rank=rank)


def _ranked(fact, keys, measures, cap, scope, distinct, rank, stats):
    # Unlike CUDA, the CPU histogram and candidate mask both scan host planes.
    # Automatically do that extra work only for COUNT rank on large inputs.
    mode = os.environ.get("TORBLES_INTERCHANGES", "auto")
    if (len(keys) != 1 or distinct is None or rank not in (BY_COUNT, BY_DISTINCT)
            or cap > 64 or (mode != "1" and (rank != BY_COUNT or fact.n < 2000000))):
        return None
    low, high, shift, transform, width, _ = keys[0]
    # Bound the histogram footprint and avoid extra full passes on small inputs.
    physical_width = fact.width(low)
    if high is not None or transform not in (0, 4) or physical_width > 17 or fact.n < 262144:
        return None
    histogram = code_counts(fact, low, scope)
    if histogram is None:
        return None
    coordinates = None
    if transform == 4:
        coordinates = (np.arange(len(histogram), dtype=np.uint32) >> shift) & ((1 << width) - 1)
        counts = np.zeros(1 << width, dtype=np.uint64)
        np.add.at(counts, coordinates, histogram)
    else:
        counts = histogram.astype(np.uint64)
    live = np.flatnonzero(counts)
    if len(live) <= cap:
        return None
    # The native comparator breaks rank ties by ascending physical key code.
    order = live[np.lexsort((live, -counts[live].astype(np.int64)))]
    candidates = order[:cap]
    omitted = int(counts[order[cap]])
    if rank == BY_DISTINCT and int(counts[candidates[-1]]) <= omitted:
        # Even the last candidate's upper bound cannot prove a strict win.
        return None
    selected = int(counts[candidates].sum())
    if selected * 2 > int(counts.sum()):
        return None

    keep = np.zeros(len(counts), dtype=bool)
    keep[candidates] = True
    pred = algq.CodeSet(algq.Col(fact, low), keep if coordinates is None else keep[coordinates])
    query = algq.Query(fact).where(pred)
    query._rows = query._rowbits = True
    try:
        narrowed = query._run_codes("alg") & scope
    except (TypeError, ValueError, OverflowError):
        return None
    result = _group(fact, keys, measures, cap, narrowed, distinct, rank)
    if result is None or len(result[1]) != cap:
        return None
    if rank == BY_DISTINCT and int(result[3][-1]) <= omitted:
        return None
    stats["cpu_interchange_rank_rows"] = selected
    if rank == BY_DISTINCT:
        stats["cpu_interchange_distinct_bound"] = omitted
    return result


def run(fact, keys, measures, cap, scope, stats, *, distinct, rank, projection, group):
    """Return native outputs and their decoders, or defer to ordinary grouping."""
    if projection is None:
        result = _ranked(fact, keys, measures, cap, scope, distinct, rank, stats)
        return None if result is None else (result, keys)

    pos = next(i for i, key in enumerate(keys) if key[3] == 3)
    low, _, _, _, _, decode = keys[pos]
    proof = sqlinterchanges.prepare(fact, low, group[pos].name, projection)
    if not isinstance(proof, sqlinterchanges.Coordinate):
        return None
    projected = list(keys)
    projected[pos] = (low, None, proof.shift, 4, proof.width,
                      lambda codes: decode(proof.labels[codes]))
    result = _ranked(fact, projected, measures, cap, scope, distinct, rank, stats)
    if result is None:
        result = _group(fact, projected, measures, cap, scope, distinct, rank)
    if result is None:
        return None
    stats["cpu_interchange_coordinate_bits"] = proof.width
    return result, projected
