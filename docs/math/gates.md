# Validation gates (FND-012)

How a mathematical capability becomes `validated` (docs/MATH_CONVENTIONS.md): `bench/validate_math.py CAP` runs the
capability's tests (the test methods whose names contain its id, e.g. `lin001`), the same tests under ASAN/UBSAN
(`tests/sanitize_math.sh`), `nvcc` on its kernels and its benchmark, and writes the receipts to `results/math/<ID>/`
with the SHA-256 of every covered source (the planned files plus the local headers they include and the
`torbles.math*` modules they import). The status is then `in_progress`; `bench/validate_math.py --device CAP` (or
`--device --all-ready`) on a machine with an NVIDIA GPU, after `make gpu`, adds the actual-device receipts and marks it
`validated`. It refuses the emulation, and refuses when the CPU receipts are older than the sources.

`python bench/render_algebra_inventory.py --check` refuses a validated claim without test receipts, benchmark
receipts, actual-device CUDA receipts, or with a source that differs from the measured one.

Native error codes map to exceptions in `torbles/math_errors.py`: -1 `ValueError`, -2 `MemoryError`,
-3 `OverflowError`, -5 `AlgebraBudgetError`, -6 `DomainError`, -7 `CarrierError`, -8 `OperatorError`,
-9 `SingularError`, -10 `NoSolutionError`, -11 `DeviceError`, -12 `UnsupportedError`, -13 `StaleStateError`,
-14 `NotConvergedError`, -15 `InfeasibleError`, -16 `UnboundedError`.

`torbles.math_gpu.device()` names the device the kernels ran on, or `"CPU emulation of the CUDA kernels"`;
`TORBLES_MATH_REQUIRE_DEVICE=1` makes every CUDA entry point refuse the emulation.
