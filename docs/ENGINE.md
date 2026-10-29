# Algebraic torbles

Torbles stored as torus **bit planes**: plane j of a torus lane holds bit j of that lane for 64 rows per word.
The torus has one row per column plus a wrap row (up to 31 columns since 0.2.0; 16 rows in 0.1.0) and one lane per bit.
Only planes that can be nonzero are stored (277 of the 1024 planes of a 9-column 16x32 torus at 30M rows), and every
row still validates against all of its faces.

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


## Typed data, loaders and persistence (0.2.0)

**Column types** (`coltypes.py`). Every type is stored as an order-preserving unsigned code: integers, decimals,
dates and timestamps as their exact integer representation minus the column's minimum (no offset for non-negative
values when it would not narrow the column), booleans as 0 / 1, strings and doubles as their rank in the sorted
distinct values. A comparison with a constant is turned into an interval of codes, computed exactly (`Fraction`), so it
has SQL's meaning: `x > 10.005` on DECIMAL(10, 2) is `code >= 1001 - min`; a string absent from the dictionary gives
an empty interval (FALSE), or the interval between its neighbours for `<` and `>`. The interval is simplified against
the codes that exist (TRUE, FALSE, `=`, `<=`, `>=`, BETWEEN). Codes wider than 32 bits go to self-extending columns
in a fact; in a referenced table they are ranked (`RankCodec`), which keeps dimension codes within 32 bits.
Measures are lowered once per query: a column becomes `code + offset` at scale s (DECIMAL(p, s)), sums and differences
are brought to a common scale, products add scales; the engine sums the integer polynomial and the result is divided
by 10^scale (`Decimal`), every digit kept. How wide the sums must be is decided when the plan is made (next section).
Group keys come back decoded.

**Keys** (`database.py`). The loader reads PRIMARY KEY / FOREIGN KEY declarations through the standard
INFORMATION_SCHEMA views (TABLE_CONSTRAINTS, KEY_COLUMN_USAGE, REFERENTIAL_CONSTRAINTS), from a DDL script parsed in a
scratch DuckDB connection, or from Frictionless Table Schemas. A referenced table is stored sorted by its primary key,
so its row is the key's rank; a foreign key value is mapped to that row natively (a lookup table when the keys are
dense, a binary search when not, a DuckDB join for strings), and a value without a match gets row n, which no
reference bit accepts.

**Key design** (`keydesign.py`). Candidates are a dimension's columns with 2 to n/2 distinct values. A -> B holds when B
is constant on every run of A (sorted once per A, checked on all rows). Equivalent columns (A <-> B) collapse; for
each column f, the longest chain of dependencies ending at f is found by dynamic programming over increasing
cardinality, and the chain whose finest level determines the most columns wins. Levels are ranked within their parent
and concatenated into a cell code; the surrogate is `cell << w | row`. For every column the finest level determines,
the bits it depends on are found by flipping cell bits (cells are dense, so flips land on other cells) and verified;
they are given to the planner as coordinate hints, which `attr_coords` checks against the data before using.
Without the hints, a sparse surrogate (one row per cell) hides its coordinates from single-bit flips, and the planner
falls back to prefixes of the key: at SSB SF1 that made the date's year a 256-value key instead of a 7-value one and
M3 four times slower.

**Persistence.** `at_save` writes a header (magic, format, shape, widths, the directory of stored planes, an FNV hash
of all of it) and the planes, 64-byte aligned, to a temporary file that is renamed into place. `at_load` maps the file
with `MAP_PRIVATE` (repairs and fault injection stay in memory) and rebuilds the face list from the planes present.
`torbles.load` validates the faces of every torble and its sidecars before anything is decoded from them.

**Fuzzing** (`fuzz.py`). Each case is reproducible from (seed, case); queries are generated as data and materialized
on whichever database is being tested (built, reloaded, reloaded from a copy with one flipped bit), so one replay
command reproduces a failure.

## Exact sums: modular 64-bit, 128-bit, and past 128 bits (0.2.0)

**Modular arithmetic.** Every sum in the engine is computed modulo 2^64, in unsigned arithmetic (no undefined
overflow): the popcount folds, the row kernel, per-row evaluation, the merges of the threads. A measure's expansion may
have terms much larger than its value (an offset of 2^62 times a column, less the same offset times another), and they
may wrap: the sum is exact when its true value is below 2^63 in magnitude. The planner bounds it by interval arithmetic
on the measure as written, not on its expansion: each group's sum, and the total, is at most rows x max |one row's
value|. Below 2^63 the plan is the 64-bit one, as before.

**128 bits: the 64 x 64 plane pairs.** Up to 2^127 the plan is marked wide. The popcounts do not change: they are exact
counts of rows. Only their weighted fold needs more than 64 bits. A 64-bit value is stored as two 32-bit columns (the
low and high words of a self-extending column), so a product of two 64-bit values is the 64 x 64 matrix of plane pairs,
weights 2^(j+k) from 2^0 to 2^126, and coefficients have a high word too (an offset of 2^63, squared, is 2^126). The fold
is sum_p W_p C_p, with W_p = coefficient x 2^(j+k) a 128-bit integer and C_p the plane pair's popcount: one 128-bit
multiply-add per plane pair, per group, per tile of 4096 rows. The matrix product (A_MM) folds its tiles this way.
Measured at SSB SF1 on one thread (`bench/ssb/wide_sums.py`, DuckDB computing in HUGEINT): `SUM(lo_extendedprice *
lo_revenue)` needs 128 bits (6M rows of up to 2^47) and takes 61 ms against DuckDB's 41; by year, 76 ms against 120
(the row kernel, with 128-bit accumulators); with a third factor, `lo_quantity`, summed per value by the bridge below,
98 ms against 133.
Products of two 64-bit columns (4096 plane pairs, more than the 1024 measure planes of the matrix product) go through
mask splitting, with the same 128-bit fold per (word, group) mask. The general path (GROUP BY keys read by lookup) and
the plain baseline evaluate the polynomial per row in 128 bits, and so does the row kernel when the cost model picks it
for a dense tile (its per-group accumulators become 128-bit). Sums come back as (low, high) words. The Moebius bridge
sums in 64 bits and is not used for wide plans; neither is the GPU executor (the query runs on the CPU). The refusal
threshold moved from an expanded bound of 2^62 to an interval bound of 2^127.

**128-bit group codes.** A group's code is the mixed-radix number of its keys. When the radix product passes 2^64 (two
64-bit columns, or one and a few more keys), the codes are 128-bit in the tiles, the per-thread tables and the output
(the high words in `out_key_hi`); the planner allows up to 2^128 - 2 groups and 12 key columns (a 64-bit fact column
takes two). The splitting tree, the row kernel and the Moebius bridge index by 64-bit codes and step aside.

**Past 128 bits, and columns without an affine code: the histogram bridge.** SUM(m) is the sum, over the values v
taken by some columns H, of m restricted to the rows where H = v, and on those rows the columns of H are constants.
The measure is expanded into monomials with exact rational coefficients; the columns of H leave each monomial as
values: H joins the GROUP BY as hidden keys, the engine sums each monomial's residual (fact columns, degree <= 2) per
group, and the parts are put together in exact arithmetic,

    SUM[k] = sum over the groups (k, v) of  sum_m  c_m * prod_(h in m) v_h * SUM_r(m)[k, v]        (SUM_() = COUNT).

H holds the columns without an affine code (a referenced table's column whose values span more than 32 bits is
rank-coded), the referenced tables' columns (the engine's 128-bit sums are polynomials of fact columns), and fact
columns past degree 2, or past degree 1 when the degree-2 residuals could still pass 128 bits. The result has no width
limit. The cost is one engine query per distinct residual (and a COUNT when the measure has a constant term), over as
many groups as the hidden values take; of the candidate columns, the one with the fewest values is hidden first. It
runs only for queries that would otherwise be refused. DOUBLE columns are not bridged: SQL sums them in
floating point, in an order the engine does not reproduce.

**What is still refused** (each with the reason): arithmetic on DOUBLE columns, strings, dates, timestamps and
booleans; GROUP BY keys needing more than 128 bits together, or more than 12 key columns; constants with more than 18
decimal places; filter programs longer than 1024 instructions. In the fuzzer, torbles now refuses fewer queries than
DuckDB rejects (DuckDB stops on overflow in its own BIGINT and DECIMAL(18) arithmetic, which torbles evaluates exactly).

## Dense GROUP BY: bridges and their ablation (0.2.0)

An unfiltered GROUP BY with every group in every tile was the CPU's weak spot: SSB M3 (`GROUP BY d_year, lo_discount`,
`SUM(lo_extendedprice * lo_discount)`, 77 groups) took 244 ms on one thread against DuckDB's 68. Each tile split every
word into its ~77 groups and then multiplied 24 x 4 plane pairs per group: far more than one multiply-add per row.
Four bridges between the plane algebra and per-row arithmetic were implemented behind switches and measured on SSB SF1
(2 vCPUs, three interleaved rounds, best per query; only answers equal to DuckDB's are timed; `bench/ssb/ablation.sh`,
`results/ablation_sf1_2vcpu`):

- **Factor through the key** (`TORBLES_FACTOR`, on). A monomial is split into the part that depends only on the GROUP
  BY keys and the rest: SUM = P_R(key) SUM(R) + P_1(key) COUNT, with P_R and P_1 evaluated per group in Python. In M3,
  `lo_discount` is a key, so the plane-pair product becomes the plain sum of one column: 244 -> 116 ms.
- **Row kernel** (`TORBLES_MM_ROWS`, chosen per tile). Every passing row's key fields and measure columns packed in
  one word by one 64 x 64 bit transpose per 64 rows (a staged, vectorizable transpose: 88 ns instead of 356), and the
  sums accumulated in an array indexed by group code. M2 212 -> 71 ms, M3 -> 60. The cost model's choice needs a margin:
  at a ratio of 0.33 (M2, M3) the row kernel is 3x faster as predicted, at 0.70 (M1) and 0.88 (Q3.1) it is slower
  (M1 60 ms against 48, Q3.1 33 against 28). The row kernel is chosen when the model puts it below half the popcount
  product's cost, smoothed over the tiles (`TORBLES_RK_MARGIN`, default 0.5); the same margin applies to folding a
  tile per row. A popcount tile measures the model again every 32 tiles, every 256 when the row kernel is far ahead
  (a popcount tile with more groups than its 256 slots evaluates the rest word by word, which is expensive).
- **Shannon splitting tree and direct tile slots** (`TORBLES_SPLIT=tree`, `TORBLES_DIRECT=1`, off). Splitting a word by
  its key bits (two ANDs per bit, shared prefixes) and using the code as the slot when it fits: slower (13 queries
  240 ms against 212) and neutral.
- **Moebius inversion over the key bits** (`TORBLES_MOBIUS=1` or `-1`, off). The tile in the monomial basis of the key
  bits (one AND per monomial), turned into groups by inclusion-exclusion: no gain when chosen by cost, 382 ms for the
  13 queries when forced.

| SF1, ms | 13 SSB, 1 thread | 13 SSB, 2 threads | M1-M3, 1 thread | M1-M3, 2 threads |
|---|---|---|---|---|
| DuckDB (in memory) | 478.0 | 310.8 | 414.1 | 228.3 |
| 0.2.0 before the bridges | 208.2 | 116.7 | 505.4 | 269.5 |
| defaults: factor + row kernel at margin 0.5 | 213.1 | 114.8 | 179.2 | 96.1 |

The 13 queries are where they were (within the VM's noise); M1-M3 go from 505 to 179 ms, 2.3x faster than DuckDB, and
M3 alone takes 59.9 ms against DuckDB's 68.4. Every configuration of the ablation, and every forced mode, passes the tests
and the fuzzer with no mismatch. The full table is in docs/SSB.md.

