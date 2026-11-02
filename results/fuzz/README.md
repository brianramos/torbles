**0.3.0** was fuzzed on the RTX 5090 laptop, an hour on 24 threads each time, by the SQL fuzzer (`torbles.sqlfuzz`,
seven workers on the alg, GPU and plain engines) and the engine fuzzer (`torbles.fuzz`, three workers):
`results/release_run_2026-11-02/` (1,355,495 statements and 1,455,975 queries; 50 reports, nine from five faults,
classified one by one in its `fuzz_triage.json`) and, on the fixes and new seeds, `results/release_fuzz_rerun_2026-11-02/`
(1,548,007 statements and 1,482,976 queries; two reports, one a sixth fault). Every fault is fixed and is a regression
test in `tests/check_campaign_sql.py`.

The records below are 0.2.0's: differential fuzzing against DuckDB 2.0.0, 2 vCPUs.

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
