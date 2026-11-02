# The release run of 0.3.0 (2026-11-02)

The release candidate's laptop run, at 99cf300, on the laptop of the earlier runs (Core Ultra 9 290HX Plus with
24 logical CPUs, 54 GiB of memory for WSL2, RTX 5090 Laptop GPU; Python 3.14.4, NumPy 2.5.4, pandas 3.1.0, DuckDB
2.0.0). `torbles_release_run.sh` (here) ran, one step after another: the build of the CPU and CUDA libraries; the
scaling sweep of the 13 official SSB queries from 6M to 300M fact rows (SF1, 3, 10, 30 and 50) on all 24 threads,
DuckDB on an in-memory copy of its tables, torbles through the Python API and through SQL on the CPU and the GPU,
every answer checked against DuckDB's (`scaling/`); the cross-engine benchmark's two window queries that output all
of 1M rows (`window_output.json`); the full test suite with the GPU tests on the GPU (`full_suite.log`); and an hour
of fuzzing against DuckDB on ten workers at the lowest priority (`fuzz/`). `summary.txt` gives every step's outcome.

**Scaling** (`scaling/scaling_summary.txt`): every answer of every engine at every scale equals DuckDB's. The 13
queries, total ms, best of 3, end to end:

| scale | fact rows | DuckDB, 24 threads | Python API, CPU | Python API, GPU | SQL, CPU | SQL, GPU |
|---|---|---|---|---|---|---|
| SF1 | 6M | 56.5 | 13.1 | 3.5 | 27.2 | 16.4 |
| SF3 | 18M | 92.3 | 27.6 | 4.6 | 45.1 | 17.6 |
| SF10 | 60M | 217.2 | 70.8 | 9.7 | 94.1 | 23.2 |
| SF30 | 180M | 582.1 | 206.9 | 17.4 | 237.2 | 33.3 |
| SF50 | 300M | 887.2 | 344.5 | 30.0 | 423.9 | 108.2 |

Through SQL on the GPU at SF50, Q3.1 took 43.8 ms against 3.5 ms at SF30 (through the Python API, 4.0 and 2.3 ms);
the other twelve queries grew with the data. The fuzzing rerun recorded where each engine call ran
(`results/release_fuzz_rerun_2026-11-02/`): every call on the GPU, and Q3.1 at 5.3 ms at SF50; the 43.8 ms did not
come back.

**Full-output windows** (`window_output.json`, 1M rows, median of 5): the running SUM took 252 ms on the CPU engine
and 247 ms with the GPU, against 583 ms for DuckDB; the global RANK 158 and 171 ms, against 149 ms; every answer
equal to DuckDB's, in order.

**Suite**: 119 checks, all passed.

**Fuzzing** (`fuzz/`, 60 minutes per worker): the SQL fuzzer ran 1,355,495 statements on seven workers (seeds 501
to 507: the alg, GPU and plain engines, and the alg and GPU engines with the algebraic paths forced onto its small
tables) and the engine fuzzer 1,455,975 queries on three (seeds 31 to 33). 50 failures were reported (the summary's
count of 45 missed five that failed while building a case). `fuzz_triage.json` classifies each one; every statement
can be replayed with the command its log prints. After the fixes, all 50 replay as answered, refused or counted
among DuckDB's by the fuzzers' own checks (`docs/SQL.md`, Tests).

**torbles' faults (fixed; `tests/check_campaign_sql.py`)** (9):

- C01, an aggregate's FILTER computed before the WHERE: an overflow on a row the WHERE drops failed the query: sql_alg_501 5834/15, sql_alg_502 6254/7, sql_gpu_503 968/12
- C02, a LEFT JOIN's ON term on the dimension alone failed the query for an error on an unmatched dimension row: sql_plain_505 668/22
- C03, a LEFT JOIN's padded rows read values cached while its ON terms ran as NULL: rows lost: sql_alg_501 4579/16, sql_alg_502 2113/21
- C04, SUM / AVG of values holding -2^63 wrapped modulo 2^64 (np.abs(-2^63) is negative): sql_alg_501 4869/0, sql_gpu_503 2778/24
- C05, comma joins on the general path formed every pair before the WHERE's equalities (120 GiB asked): sql_alg_501 5514/26

**tables wider than a torble (now a refusal at load)** (5):

- sql_alg_501 case 1594 (building), sql_plain_505 case 2682 (building), sql_plain_505 case 4951 (building), sql_gpu_504 case 5716 (building), sql_forced_gpu_507 case 4983 (building)

**DuckDB without its statistics propagation gives torbles' answer** (14):

- engine_33 4659/21, sql_alg_501 3064/28, sql_alg_501 3085/15, sql_alg_501 5982/13, sql_alg_502 4015/27, sql_alg_502 6481/4, sql_forced_alg_506 3110/4, sql_forced_alg_506 4130/9, sql_forced_alg_506 5798/1, sql_forced_gpu_507 3989/28, sql_forced_gpu_507 5349/15, sql_gpu_503 4747/20, sql_gpu_504 5553/23, sql_plain_505 4533/10

**DuckDB without its optimizer gives torbles' answer** (2):

- sql_forced_gpu_507 5495/18, sql_gpu_503 4351/28

**errors DuckDB's optimizer skipped (it raises them without its optimizer)** (8):

- sql_alg_501 4783/14, sql_forced_alg_506 94/29, sql_forced_alg_506 6475/4, sql_forced_gpu_507 1750/11, sql_gpu_503 378/15, sql_gpu_503 4942/7, sql_gpu_503 5012/19, sql_gpu_504 4662/3

**orders under ORDER BY ... LIMIT that differ through documented differences** (11):

- sql_alg_501 1779/26, sql_alg_501 2585/21, sql_alg_501 3239/19, sql_alg_502 481/26, sql_alg_502 884/4, sql_alg_502 4760/7, sql_alg_502 6876/12, sql_forced_alg_506 1920/23, sql_forced_gpu_507 1033/5, sql_gpu_503 2399/11, sql_gpu_503 5343/2

**the fuzzer's comparison** (1):

- sql_forced_gpu_507 5058/15
---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
