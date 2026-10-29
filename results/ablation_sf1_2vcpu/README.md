Ablation of the GROUP BY bridges (docs/ENGINE.md, "Dense GROUP BY"), SSB SF1, a 2-vCPU cloud VM (AVX2), DuckDB 2.0.0 in
memory. `bash bench/ssb/ablation.sh 3` (with `BASELINE=` a checkout of commit a190692, "0.2.0 before the bridges"),
then `python3 bench/ssb/ablation_table.py <out dir>`: every configuration runs `ssb_auto.py 1 cpu@1,cpu@2` once per
round, rounds interleaved; per query, the best time over the rounds (each run is best of 3), counting a run only when
its answer equals DuckDB's (all did). DuckDB: best over four rounds of `ssb_duck.py`. `table.md` is the printed table,
`summary.json` the per-query numbers.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
