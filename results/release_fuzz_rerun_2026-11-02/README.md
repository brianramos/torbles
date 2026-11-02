# The fuzzing rerun of 0.3.0 (2026-11-02)

Once the release run's fuzzing reports were fixed (`results/release_run_2026-11-02/`), the fixes (605bb4f) ran again
on the same laptop (Core Ultra 9 290HX Plus with 24 logical CPUs, 54 GiB of memory for WSL2, RTX 5090 Laptop GPU;
Python 3.14.4, NumPy 2.5.4, pandas 3.1.0, DuckDB 2.0.0). `torbles_fuzz_rerun.sh` (here) ran, one step after another:
the build of the CPU and CUDA libraries; `torbles_sql_devices.py` (here), which runs the 13 SSB queries as SQL at SF30
and SF50 and records where each engine call ran; the full test suite with the GPU tests on the GPU; and an hour of
fuzzing against DuckDB on ten workers at the lowest priority, on seeds the release run had not used. `summary.txt`
gives every step's outcome.

**Where SQL's engine calls ran** (`sql_devices.log`, `sql_devices.json`): with the GPU engine, every engine call of
the 13 queries ran on the GPU at both scales, one call per query. Best of three runs after a first one, without
checking the answers: Q3.1 took 4.0 ms at SF30 and 5.3 ms at SF50, and the 13 queries 36.1 and 60.1 ms (on the CPU
engine, 254.7 and 411.3 ms). The release run's sweep had measured Q3.1 at 43.8 ms at SF50 through SQL on the GPU;
that did not come back here.

**Suite**: 120 checks, all passed.

**Fuzzing** (`fuzz/`, 60 minutes per worker): the SQL fuzzer ran 1,548,007 statements on seven workers (seeds 601 to
607: the alg, GPU and plain engines, and the alg and GPU engines with the algebraic paths forced onto its small
tables) and the engine fuzzer 1,482,976 queries on three (seeds 41 to 43), with the fuzzers' checks of 605bb4f. Eight
SQL cases generated a fact table wider than a torble, which the loader refused (counted as refusals). Two statements
were reported, and both are fixed:

- `sql_forced_gpu_607` case 3144, query 4: an ERROR. The general path had pushed the WHERE terms on the fact,
  `abs(f_m0) < 4 AND f_d0 IS NULL`, down to the fact, where abs failed on a row holding the least BIGINT. A LEFT JOIN
  pads that row, and a WHERE term on the dimension then drops it; DuckDB evaluated that term first and answered. The
  general path now checks such a term again after the joins, where an error counts only for a row the other terms
  leave undecided, as the star planner already did (C07 in `tests/check_campaign_sql.py`).
- `sql_gpu_603` case 1410, query 4: a MISMATCH. A ROLLUP's two rows with a NULL key were ordered by CORR, NULL in
  torbles and NaN in DuckDB, under ORDER BY ... LIMIT 1: the documented difference. The fuzzer's check for it skipped
  statements with a window function; it now looks at the statement's last clauses only (C06).

Each replays with the command its log prints, and both are answered and counted as equal after the fixes.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
