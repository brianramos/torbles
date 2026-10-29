First scaling run (superseded by `results/scaling/`). DuckDB ran on its database file, so the first run of each
query loaded and decompressed its columns into DuckDB's buffer pool. First runs were 1.3-2x slower than the best of 3,
and up to 8x slower for single queries at SF30/SF50. DuckDB also ran under a 5 GB memory limit, a leftover from the
first 7 GB test machine. The SF50 torbles results here are from before the 32-bit surrogate fix; four queries
mismatched. The rerun copies DuckDB's tables into an in-memory database, so both engines answer from memory, and uses
DuckDB's default memory limit.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
