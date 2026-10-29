# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""8-dimension star schema with coordinate ids (id = grp | sub | tag | seq) in two encodings.
"plain":  seq is a row counter inside each (grp, sub, tag) cell; rnd and rnd2 are ordinary attributes.
"bounds": seq is ordered by rnd with global offsets: seq = O[rnd] + rank within (cell, rnd), with
          O[v+1] = O[v] + max over cells of count(cell, v). rnd is then a function of the seq bits alone and every
          range predicate on rnd is one interval of seq: rnd in [a, b)  <=>  O[a] <= seq < O[b].
The random draws are identical in both encodings, so both describe the same rows; only the ids differ."""
import numpy as np
D = 8
SIZES = [3650, 50_000, 20_000, 100_000, 10_000, 200_000, 5_000, 80_000]

def make_data(N, MODE="plain", SIZES=SIZES):
    """returns dims {name: columns over the id space}, used {name: bool mask}, fact columns, seq bits per dim"""
    rng = np.random.default_rng(11); rng2 = np.random.default_rng(12)
    dims, used, fks, seqbits = {}, {}, {}, {}
    for i, n in enumerate(SIZES):
        grp, sub, tag = rng.integers(0, 8, n), rng.integers(0, 8, n), rng.integers(0, 10, n)
        order = np.lexsort((tag, sub, grp)); grp, sub, tag = grp[order], sub[order], tag[order]
        cell = (grp * 8 + sub) * 16 + tag
        rnd = rng.integers(0, 10, n); rnd2 = rng2.integers(0, 10, n)        # same draws in both encodings
        if MODE == "bounds":
            cnt = np.zeros((1024, 10), np.int64); np.add.at(cnt, (cell, rnd), 1)
            O = np.concatenate([[0], np.cumsum(cnt.max(0))])                  # global offset of each rnd value
            key = cell * 10 + rnd; o = np.argsort(key, kind="stable"); ks = key[o]
            first = np.r_[True, ks[1:] != ks[:-1]]; pos = np.arange(n)
            rank = np.empty(n, np.int64); rank[o] = pos - np.maximum.accumulate(np.where(first, pos, 0))
            seq = O[rnd] + rank; S = max(1, int(O[-1] - 1).bit_length())
        else:
            first = np.r_[True, cell[1:] != cell[:-1]]
            seq = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0)); S = max(1, int(seq.max()).bit_length())
        ids = (cell << S) | seq; size = int(ids.max()) + 1
        a = np.arange(size)
        cols = dict(grp=(a >> (S + 7)).astype(np.uint32), sub=((a >> (S + 4)) & 7).astype(np.uint32),
                    tag=((a >> S) & 15).astype(np.uint32), rnd=np.zeros(size, np.uint32), rnd2=np.zeros(size, np.uint32))
        if MODE == "bounds":    # rnd is defined on the whole id space by the seq field: the value whose interval holds it
            cols["rnd"] = np.minimum(np.searchsorted(O, a & ((1 << S) - 1), "right") - 1, 9).astype(np.uint32)
            assert np.array_equal(cols["rnd"][ids], rnd)
        else: cols["rnd"][ids] = rnd
        cols["rnd2"][ids] = rnd2
        u = np.zeros(size, bool); u[ids] = True
        dims[f"dim{i}"], used[f"dim{i}"], fks[f"dim{i}"], seqbits[f"dim{i}"] = cols, u, ids, S
    fact = {f"fk{i}": fks[f"dim{i}"][rng.integers(0, SIZES[i], N)].astype(np.uint32) for i in range(D)}
    fact["price"] = rng.integers(1, 100_001, N, dtype=np.uint32); fact["disc"] = rng.integers(0, 11, N, dtype=np.uint32)
    order = np.argsort(fact["fk0"], kind="stable"); fact = {k: v[order] for k, v in fact.items()}     # time order
    return dims, used, fact, seqbits
