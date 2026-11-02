# The release review's protocol, run again with windows read from their series (2026-11-02)

The second run of the release review's protocol on the same laptop (`../release_rerun_2026-11-02/` has the first and
the method), at 245ff75: the first run's code with window functions read from their aggregated series (one engine
GROUP BY of each window's partition and order key, the frame as prefix sums on it; docs/SQL.md, Window functions) and
the fixed gate test. `torbles_review_rerun.sh` (here) ran the same steps: the device validation of FND-012, whose gate
test had changed, which left all 81 capabilities of the mathematics inventory validated; the full test suite with the
GPU tests on the GPU (118 checks, all passed); the review's reproducers; the device memory probe; the installed wheel's
checks; and the review's two benchmark harnesses, unchanged, with every answer equal to DuckDB's.

`summary.txt` gives each step's outcome and `comparison.md` the review's medians beside this run's. The mixed SQL
workload's two window queries at 1M rows took 17.7 ms together on the CPU engine and 12.4 ms with the GPU engine,
against 62.6 ms for DuckDB (the first run: 375 and 398 ms; the review, on its older snapshot: 7.7 and 7.3 s), and the
whole workload of eight queries 36.8 and 24.5 ms against 74.8 ms. The SSB numbers are those of the first run within
the machine's noise.
