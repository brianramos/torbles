# Changelog

## 0.1.0 (2026-10-29)

First public release of torbles (a torble is a toroidal table; torbles is the engine and package): a research
prototype. The API may change; see the README's known limitations.

Engine
- Torus bit-plane storage with face (XOR) integrity checks and single-bit repair; only planes that can be nonzero
  are stored.
- Algebraic query execution: filters as Boolean polynomials, sums as popcount algebra (linear and bilinear measures).
- Equi-joins as a substitution in the dimension filter's polynomial (`JoinPoly`), planned by a calibrated cost
  model: factored comparators, expanded cubes, complement, or lookup; join polynomials planned in C (`jp_terms`).
- Order-encoded bounds, range / band / inequality joins with dominance sums (groups or slices).
- Derived surrogates (official keys kept as a projection), additive deltas for moved dimension rows,
  self-extending records (rank-addressed sidecars, PDEP), large objects (`BlobStore`).
- GROUP BY as one AND-popcount matrix product `C = M^T X` with sparse output (no dense key domain; the former
  2^22-group limit is gone); AVX-512 VPOPCNTDQ / AVX2 / scalar kernels chosen at run time.
- Persistent worker pool for every parallel call; compiled plans reused for repeated queries (same query, same data
  version); results decoded in bulk.
- Native, parallel ingestion: the bit-plane build on all cores, foreign-key reference checks and re-encoding in C.

GPU (CUDA, optional)
- One thread per 64-row word: filters, popcount sums, GROUP BY by mask splitting into a sparse group table on the
  GPU (default), dominance slices, row ranges, delta planes, self-extending columns.
- Tile matrix-product GROUP BY kernels with `__popc` and with tensor-core b1 MMA (`mma.sync.m16n8k256 .and.popc`),
  self-tested against popcounts at start.
- Builds with nvcc older than the GPU (PTX fallback) and without the b1 MMA if it is rejected; a CPU emulation of
  every kernel for testing without a GPU.

Benchmarks
- Official SSB scaling benchmark, SF1 to SF50 (6M-300M rows) and 1-24 threads (`wsl/run_scaling.sh`, graphic in
  `docs/img/ssb_scaling.png`, from `bench/ssb/poster_scaling.py`). DuckDB runs on an in-memory copy of its tables with its default memory limit, so both
  engines answer from memory; first runs are reported next to the best of 3.

Fixes since the development snapshots
- Values wider than 32 bits are refused, never truncated. A derived surrogate whose hierarchy fields no longer fit
  beside the official key (part, from SF50 on) had wrapped silently. Such a hierarchy is now stored as one dense,
  order-preserving cell rank.
- Planner caches keyed by Python object id could return a freed space's answer to a new one; keyed by object now.
- The plain engine no longer scans a unit twice when given many row ranges.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
