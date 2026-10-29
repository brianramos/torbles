Differential fuzzing of 0.2.0 against DuckDB 2.0.0, 2 vCPUs.

**Final record** (`fuzz_0.2.0_final_seeds11-16.txt`): `python -m torbles.fuzz --seed S --cases 800 --keep-going` for
S = 11 to 16, on the release with 128-bit sums, 128-bit group codes and the histogram bridge. 4,806 random typed
schemas, 162,325 queries; 149,176 answers compared with DuckDB's, no mismatch, no unexpected error. torbles refused
119 queries (0.07%: all GROUP BY keys needing more than 128 bits together); DuckDB rejected 13,030 (8.0%, overflow in its
own BIGINT, UBIGINT and DECIMAL(18) arithmetic, which torbles evaluates exactly). Seed 14's first run found one
mismatch, a GROUP BY with exactly 2^64 groups on the GPU path (fixed in commit 0997b80); seed 14 was run again on the
fix, the other seeds ran on the commit before it.

**Every engine mode** (`modes_0.2.0.txt`): the quick test suite and 100 fuzz cases under each environment switch of
the CPU planner (bridges off, Moebius forced, row kernel forced, per-row flushes, splitting tree, direct slots,
Moebius by cost, no factoring): all pass, no mismatch.

**Before the 128-bit work** (`fuzz_0.2.0_seeds11-16.txt`): `--minutes 6` per seed, 5,016 schemas, 169,600 queries,
124,364 answers compared, no mismatch; torbles refused 31,595 queries (sums that could pass 64 bits, GROUP BY keys
wider than 64 bits, columns that could not be summed), 2.3x the 13,641 DuckDB rejected. "refused" counts documented
refusals; "duckdb-error" counts queries DuckDB rejected, which are not compared.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
