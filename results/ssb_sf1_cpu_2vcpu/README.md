CPU comparison at SSB SF1 on a 2-vCPU cloud VM (AVX2), four interleaved rounds (round r = DuckDB, torbles 0.1.0 with
hand-designed keys, torbles 0.2.0 with hand-designed keys, torbles 0.2.0 with automatic keys), each run best of 3, at 1
and 2 threads. DuckDB on an in-memory copy of its tables. The tables in docs/SSB.md and the README keep, per query, the
best time over the rounds. Files: duckdb_r*.json (ssb_duck.py), torbles_derived_r010_*.json / _r020_*.json
(ssb_torbles.py derived, from v0.1.0 and from 0.2.0), torbles_auto_r*.json (ssb_auto.py).

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
