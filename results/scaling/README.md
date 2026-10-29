Scaling run with both engines in memory: SF1 to SF50, and 1 to 24 CPU threads at SF10. `scaling.json` and
`scaling_summary.txt` are collected by `bench/ssb/ssb_scaling.py`; the poster is `docs/img/ssb_scaling.png`.

The `cp: cannot stat ... torbles_derived_*.json` lines and the traceback in `scaling_20261029_112815.log` are from the
engine rename: during that run the benchmark still wrote its results as `torble_derived_*.json`. The benchmarks
themselves completed, and every result matched DuckDB. The files were then copied by hand under their new names, and
`ssb_torbles.py` now writes the new names.

---
Copyright 2026 Brian Richard RAMOS. Licensed under the Apache License, Version 2.0 (see LICENSE).
