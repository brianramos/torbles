Final differential fuzzing run of 0.2.0 against DuckDB 2.0.0: `python -m torbles.fuzz --seed S --minutes 6 --keep-going`
for S = 11 to 16 (one summary per seed, in order), 2 vCPUs. 5,016 random typed schemas, 169,600 queries; 124,364 answers
compared with DuckDB's, no mismatch, no unexpected error. "refused" counts documented refusals; "duckdb-error" counts
queries DuckDB rejected (mostly integer overflow in its own arithmetic), which are not compared.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
