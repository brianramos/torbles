# Algebraic torbles

Torbles stored as torus **bit planes**: plane j of a torus lane holds bit j of that lane for 64 rows per word.
Only planes that can be nonzero are stored (277 of the 1024 planes of a 9-column 16x32 torus at 30M rows),
so a torble is smaller than plain uint32 columns, and every row still validates against all of its faces.

Queries are built as algebraic expressions (`algq.py`) and executed algebraically:
- decode: value bit j = XOR of V planes 0..j (64 rows per instruction)
- filters: Boolean polynomials over value planes (bit-sliced comparisons, AND/OR/NOT)
- equi-joins as one polynomial (`JoinPoly`): the primary key determines every attribute, so a dimension filter is
  a Boolean polynomial f in the pk's bits, and the equi-join is the substitution pk := fk: the filtered join is
  f evaluated on the fk's bit planes (foreign keys are stored most-significant-bit first). The planner chooses the
  variables: the id bits the filtered attributes depend on when they are coordinates of the id (discovered: which
  bits change the value, then a check that the attribute is a function of them), else all id bits. In those
  variables f is a sum of intervals over the codes that occur; unused codes are don't-cares, so each interval
  widens into the gaps around it to get bounds with trailing zeros (fewer planes). Every interval is the comparator
  polynomial [code >= A][code < B], evaluated per term in the cheaper of two forms:
    factored  one pass over the code's planes per bound (P_CGE / P_CLT on any set of planes)
    expanded  its aligned blocks, one cube (AND of literals) each (P_CUBE)
  The complement may be cheaper (NOT). The whole f can instead be evaluated by lookup (bitmap semi-join), which is
  what an f with no structure in these variables costs: a random attribute on a 200k-row dimension is ~77k cubes
  or ~262k ANF monomials, and a table lookup is the cheapest way to evaluate an incompressible polynomial.
  Costs are measured, not guessed (`calibrate.py`: ~0.65 ns per literal, ~2.7 ns per instruction, ~9.5 ns per
  looked-up row, per 64-row word), so a polynomial wins up to roughly 60 medium cubes.
  `join_mode` forces a form for testing: auto, factored, expanded, lookup (old names prefix/between/bitmap work).
- GROUP BY on coordinate attributes reads them from the fk planes (a projection of the fk bits)
- joins through a derived surrogate (`Surrogate`, `refs={fk: (dim, Surrogate(w, s_of_row))}`): the fact stores
  s = (coordinates << w) | official key, and the dimension stays indexed by its official key, which is the
  projection s mod 2^w. The join substitutes pk := s mod 2^w; the planner works in s's own variables (`IdSpace`),
  so filters on hierarchy attributes are cubes on the high planes and filters on the official key are comparators on
  the low planes; lookups (when chosen) mask the fk by 2^w - 1 in C. Integrity: a reference is valid only if its key
  exists and its high bits equal the derivation for that key (`_refok`). Coordinate discovery also tests directly for
  the smallest id prefix that determines an attribute, since single-bit flips find nothing in a sparse surrogate
  space (`check_surrogate.py`). Where the derivation is pure arithmetic (yyyymmdd -> year|month|day), no table is
  involved in either direction.
- order-encoded attributes (bounds): the seq field that tells rows of one coordinate cell apart has no meaning of
  its own, so its order is free. Ordering it by one more attribute with global offsets,
  seq = O[v] + rank within (cell, v) with O[v+1] = O[v] + max over cells of count(cell, v), makes that attribute a
  function of the seq bits alone (the same in every cell). Every range predicate on it is one interval of seq,
  v in [a, b) <=> O[a] <= seq < O[b], so the planner discovers it like any coordinate and evaluates it as a few
  comparator (one bound: rnd < 7 is seq < O[7]) on the fk's low planes (equality, IN, <>, BETWEEN, OR included). It costs the
  spread of the per-cell counts (61 vs 53 seq bits across 8 dimensions here; an aligned 4-bit field would take 66).
  One attribute per seq field can own the order; others stay bitmap joins (`star8.py`, `check_bounds.py`).
- SUM(x) = sum_j 2^j popcount(x_j & F); SUM(x*y) = sum_jk 2^(j+k) popcount(x_j & y_k & F); linear forms likewise
- dimension changes as additive deltas (`Surrogate.move`): when dimension rows change (a customer moves city, so
  nation and region), nothing in the fact is rewritten. The old surrogates become stale versions; a delta plane (one
  bit per fact row, found once per change by a scan of the fk planes) marks the rows that reference a moved key.
  A query that filters or groups on that dimension is split additively: pass A = every row but the delta plane, on
  the stored surrogates (cubes, comparators, plane keys); pass B = only the delta rows (only their words are scanned),
  with the moved dimension read by lookup of current values. Result A + B, exactly what an UPDATE would give.
  Compaction (rewriting the surrogates) is optional and its timing a choice; `check_moves.py` checks both against
  DuckDB after UPDATE, with facts written before and after the move. (An earlier variant XORed the predicate value
  of each stale point into the filter, over F2; the delta plane does the same job at a fraction of the cost.)
- self-extending records (`extend={col: B}`): a value of up to B + 32 bits is a B-bit base for every row, a flag plane,
  and the high part only for flagged rows, in a sidecar torble with its own integrity faces. The sidecar address of
  a flagged row is its rank (flagged rows before its word, from a small directory, plus a popcount within the word):
  computed, never stored. Per 64-row word, the flagged rows' high bits are gathered from one or two sidecar words and
  scattered into place with PDEP; words without flagged rows cost nothing. Queries see base + 2^B hi: measures become
  polynomials in the parts, comparisons with constants beyond 32 bits become lexicographic comparisons on
  (hi, base), GROUP BY keys become two keys (`check_extend.py`, vs DuckDB BIGINT). GROUP BY on the full wide value
  uses the sparse group output (default GROUP BY mode); the `split` mode needs a dense key domain and refuses it.
- large objects (`blobcol.py`, `blob_bench.py`): a CLOB / BLOB is a self-extending record. Its first 16 bytes are 4
  big-endian 32-bit limbs in planes (limb k only for objects longer than 4k bytes: a virtual column), its length is a
  column, and the remaining bytes go to one contiguous payload heap. The heap offset of an object is derived, not
  stored: one number per 64 rows plus the tail lengths of the rows before it in its word, from the length planes.
  Prefix / type / length filters and groupings on leading bytes run on the planes and never touch the payload;
  fetching assembles limbs and tail; substring search scans the payload; replacing objects is additive (a delta
  plane over the base + a small delta store). Against DuckDB (VARCHAR / BLOB), LMDB and a filesystem object store
  (one file per object), all answers equal: questions about prefixes, types and sizes are answered 10-1000x faster
  than the object stores and 2-1000x faster than DuckDB; point fetches are about 2x slower than LMDB (fetch is
  assembled in Python); substring search over text is ~6x slower than DuckDB (a single-threaded scan of the heap).
  The payload heap is not covered by the torus integrity planes (the heads and lengths are).
- optional GPU executor (`gpu.cu`, `engine="gpu"`): the math of a query runs on CUDA, one thread per 64-row word, the
  same per-word code as the CPU engine: the filter polynomial, popcount sums, GROUP BY by mask splitting (plane keys
  and up to two looked-up keys) into a sparse group table on the GPU (a slot's key is read before it is claimed by
  CAS; the groups present are gathered from an insertion list and their slots cleared, so the table stays empty
  between queries), dominance slices, row ranges, delta planes, self-extending columns. The torus planes and integrity
  checks stay on the CPU; the GPU holds the value planes, decoded once at upload. Semi-join lookups stay on the CPU
  (the query falls back). `gpu_mm("word" | "popc" | "mma")` selects the GROUP BY kernel: per-word (default), or the
  tile matrix product with `__popc` or on the tensor cores. The same source builds as plain C++ (`-DEMULATE`) so the
  kernels are tested without a GPU (`tests/check_gpu.py`). Results on an RTX 5090 Laptop GPU: `docs/SSB.md`.
- GROUP BY with a degree-2 polynomial measure (e.g. revenue = 100*price - price*discount) on keys that live on
  planes (the clustering key, fact columns, coordinate fields of fks): mask splitting. The rows sharing a key are
  an AND of key-plane literals; the measure over them is popcount algebra; remove and repeat. Chosen by cost: one
  step per distinct key in a 64-row word (the clustering key rarely changes; a coordinate has few values).
- GROUP BY as one matrix product (default; `algq.set_groupby("split")` or `TORBLES_GROUPBY=split` for the per-word
  path above). With M[r, g] = 1 when row r passes the filter and has group g (one-hot rows: the Khatri-Rao product of
  the filter and the key-bit indicators [~b, b]) and X[r, p] = measure plane p (value planes of linear terms, plane
  pairs x_j & y_k of bilinear ones), C = M^T X is an AND-popcount product over the rows and
  SUM[g] = c0 COUNT[g] + sum_p w_p C[g, p] (w_p = coefficient x 2^(bit weight)). The bit planes are already X in
  column-major form; a group's mask over a tile is a row of M^T. Per tile (CPU: 4096 rows, GPU: 4096-8192 rows) the
  groups present get slots (found by mask splitting each word), their masks form M_tile, and C_tile = M_tile^T X_tile
  is a dense AND-popcount GEMM: on the CPU with AVX-512 VPOPCNTDQ (8 words per instruction) when the CPU has it, AVX2
  nibble-table popcounts otherwise; on the GPU on the tensor cores, `mma.sync.m16n8k256 .b1 .and.popc`
  (16 groups x 8 planes x 256 rows per instruction; self-tested against popcounts at start), or `__popc`. Groups
  present in few words of a tile are evaluated sparsely; pair planes whose factors vanish on a group are dropped.
  The output is sparse (one entry per group present): no dense key-domain arrays, so the 2^22 group limit is gone.
  Cost: dense AND-popcount work is G x B / 64 word operations per row against ~B for a row-at-a-time evaluation, so
  it pays when a tile holds few groups with many rows each (unfiltered or lightly filtered GROUP BY on low-cardinality
  keys: M1 in `ssb/`), or with wide popcount units (VPOPCNTDQ, tensor cores); on selective queries M is sparse and
  the tile path is about even with the row path.
- range, band and inequality joins (`join_range`, `join_ineq`): an inequality is the same kind of polynomial as
  equality, [x < t] = sum_j (1 + x_j) t_j E_j with E_j = prod_{k > j} (1 + x_k + t_k) (MSB first); equality is its
  last term (no differing bit), and a range is two inequalities: [lo <= x <= hi] = [x < hi + 1] - [x < lo].
  With aggregation, SUM over pairs of m(x) [lo(y) <= x <= hi(y)] = sum_{x < t} M[k, x] H[g, t], where M is the
  measure aggregated by (fact group, x) and H holds +1 at hi(y) + 1 and -1 at lo(y) per dimension group. Two
  evaluations of that sum, chosen by cost:
    groups  aggregate the fact by (fact keys, x) on the bit-plane engine (mask splitting when x is the clustering
            key), rank-transform the coordinates (order-preserving, so the comparator is unchanged) and apply the
            comparator polynomial level by level: at level j, x and t agree above bit j and differ at it, so the
            even blocks of size 2^j of M pair with the odd blocks of H (`dominance()`, a few small matrix products)
    slices  the comparator on x's own planes, one slice per distinct bound (A_DOM): rows with t_(k-1) <= x < t_k,
            measure by popcount algebra, prefix sums give D(t). For a high-cardinality x with few bounds.
  This is aggregation pushed below the join (eager aggregation is a known technique), derived from the polynomial.
  `sql_preagg()` renders the same rewrite as SQL, so DuckDB can be measured on the same algebra.
- physical order as an index: when the fact is sorted by a foreign key, a filter through that dimension becomes
  contiguous row ranges (binary search), and only those words are evaluated
- other GROUP BY / general measures: integers are extracted only for surviving rows
- window analytics: the aggregated series as a polynomial S(t): moving sum = S(t)(1+...+t^6),
  year-over-year = S(t)(1 - t^365), cumulative = S(t)/(1 - t)

- multithreaded: 64-row words are independent; work is handed out from an atomic counter, each thread keeps its own
  totals and a small hash of its groups, and integer sums merge exactly (`set_threads(n)`)
- full 32-bit unsigned values; inputs that do not fit uint32 are rejected
- referential integrity: when a fact table is built with `refs=`, whether each foreign key points at an existing
  dimension row is recorded in a hidden `_refok` column (one bit per foreign key, torus-protected like any column).
  Every join requires its bit, so orphaned references never join (inner-join semantics); lookups are also
  bounds-checked.

    make && make test                  # from the repository root: build, then every correctness test (see README.md)
    pip install duckdb pandas
    # individual tests and benchmarks (PYTHONPATH=torbles:tests, as `make` sets it):
    python3 tests/check_small.py 300000      # every query: algebraic == plain columnar == DuckDB
    python3 tests/test_fixes.py              # full-width uint32 values and orphaned foreign keys vs DuckDB
    python3 torbles/calibrate.py               # measure the join cost model (literal, instruction, lookup) on this machine
    python3 tests/check_bounds.py            # order-encoded attribute: 12 predicate shapes x engines x join forms vs DuckDB
    python3 tests/check_surrogate.py         # joins through derived surrogates (incl. inconsistent / missing keys) vs DuckDB
    python3 tests/check_moves.py 300000      # dimension changes as additive deltas, vs DuckDB after UPDATE and vs compaction
    python3 tests/check_extend.py            # self-extending 64-bit columns (sparse sidecar, rank + PDEP) vs DuckDB BIGINT
    python3 bench/blob_bench.py clob 500000  # CLOB workload: torbles vs DuckDB vs LMDB vs filesystem object store
    python3 bench/blob_bench.py blob 50000   # BLOB workload (binary objects, 8 KiB median, up to 4 MiB)
    make && python3 tests/check_gpu.py                                              # GPU kernels (emulated)
    make gpu                                                                        # GPU kernels (CUDA)
    python3 tests/check_ineq.py              # range / band / inequality joins: 15 cases x both paths x forms x engines vs DuckDB
    python3 bench/ineq_bench.py 10000000     # range / band / inequality joins vs DuckDB (as written, and pre-aggregated)
    bench/ssb/                         # the official Star Schema Benchmark (official generator and queries), docs/SSB.md
    python3 bench/scale_tables.py 10000000 plain    # 1..8 joined dimensions, coordinate vs non-coordinate filters
    python3 bench/scale_tables.py 10000000 bounds   # same rows, ids ordered by rnd: rnd filters become bounds

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
