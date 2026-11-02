# The release review's protocol, run on 0.2.0 (2026-11-02)

The independent release review of 2026-11-01 measured an older snapshot of this release (a8b42fe). Once its findings
were fixed, its protocol was run again on the release itself, at 7486dba, on the same laptop (Core Ultra 9 290HX Plus
with 24 logical CPUs, RTX 5090 Laptop GPU, WSL2, Python 3.14.4, NumPy 2.5.4, DuckDB 2.0.0). `torbles_review_rerun.sh`
(here) ran, one step after another: the device validation of the last 21 capabilities of the mathematics inventory,
the full test suite with the GPU tests on the GPU, the review's own reproducers, a probe of device memory over 120
rounds of tables uploaded, queried and released, the wheel installed in a new environment outside the source tree with
its SQL tests on the three engines, and the review's two benchmark harnesses unchanged (`benchmark_release.py` and
`benchmark_native_ssb.py`), one fresh process per workload and scale.

`summary.txt` gives each step's outcome and `comparison.md` the review's medians beside this run's. The JSON reports
are the harnesses' own, with every round's timings, the answer digests and the routes taken; `source_manifest.json`
hashes the tree that was measured. Every benchmark answer equals DuckDB's. The suite's one failure
(`test_math_gates`, in `full_suite.log`) came from the device validation earlier in the same run: with every
capability validated, the gate test had no capability left on which to forge a validated claim. The test was fixed
afterwards (8888721), and the capability it belongs to, FND-012, was validated again.
