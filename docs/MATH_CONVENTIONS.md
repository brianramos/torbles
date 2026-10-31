# Implementing a mathematical capability

The canonical backlog is `docs/algebra_inventory.json` (readable view `ALGEBRA_INVENTORY.md`, rendered and checked by
`python bench/render_algebra_inventory.py [--check]`). Every capability there has a scope, exactness contract,
budgets, acceptance tests and benchmark cases. This file says how one is built, tested and validated so that all 81
are consistent. It applies to new work; the first slice (`mathematics.c`, `math_domain.c`, `math_modules.c`,
`math_state.c`) keeps its own interfaces, which new code extends rather than replaces.

## What "validated" means

A capability is `validated` only when all of these exist and pass:

1. a native CPU implementation in C (Python marshals arguments and formats results; it never computes the result);
2. a CUDA kernel for the capability's data-parallel part, compiled by `nvcc` and run on an actual NVIDIA device;
3. its acceptance tests from the inventory, against an independent oracle (exact Python arithmetic, `sympy`,
   `networkx`, DuckDB SQL: never the implementation under test), including every budget refusal;
4. the same tests with the native library built under AddressSanitizer and UndefinedBehaviorSanitizer;
5. a benchmark over a grid derived from the inventory's benchmark cases, with all raw samples kept.

Until the device run exists the capability is `in_progress`, with the CPU receipts, the `nvcc` compile receipt and
the emulated-kernel receipt recorded. `bench/validate_math.py` produces the receipts and updates the inventory entry
(receipt paths, SHA-256 of every source file it covers, the runtime commit measured); `bench/validate_math.py --device`
runs on a machine with a GPU and adds the device receipts. Editing a covered source file invalidates the claim until it
is measured again (the renderer's `--check` refuses a stale hash).

## Files of one family

| What | Where |
|---|---|
| native CPU code | `torbles/math_<fam>.c`, `torbles/math_<fam>.h` (`<fam>` lower case, e.g. `math_lin.c`) |
| CUDA kernels | `torbles/math_<fam>_gpu.cu` (built into `libmathgpu.so` by nvcc, and as C++ with `-DEMULATE` into `libmathgpu_emu.so`) |
| Python API | `torbles/math_<fam>.py` (opt-in; `import torbles` must not load it) |
| tests | `tests/test_math_<fam>.py` (unittest; run by `tests/run_all.sh`) |
| benchmark | `bench/benchmark_math_<fam>.py` |
| receipts | `results/math/<CAPABILITY-ID>/` |

The build compiles every `torbles/math*.c` into `libalg.so` and every `torbles/math_*_gpu.cu` into the GPU library, so
adding a family needs no build-file edit.

## Native rules (CPU)

- C11, no mutable globals (constant tables only), no threads of its own. Objects are immutable once built and
  individually owned; a constructor that fails returns NULL, sets `*error` and leaves its inputs unchanged.
- Error codes are those of `mathematics.h` and `math_domain.h` (`TR_MATH_INVALID`, `_MEMORY`, `_OVERFLOW`,
  `_BUDGET`, `_DOMAIN_ERROR`, ...). New codes are added to `math_num.h`, never reused with another meaning.
- Budgets (dimensions, support, states, output, iterations) are checked before allocation and refused with
  `TR_MATH_BUDGET`; a request is never silently truncated.
- Arithmetic is exact or carries an explicit error bound. Integer and rational arithmetic uses the checked helpers of
  `math_num.h` (64-bit with 128-bit intermediates, overflow reported, never wrapped). A floating result is returned
  only together with a rigorous bound (`tr_math_interval`: lower and upper doubles from directed rounding, or an
  explicit absolute/relative error); the contract is stated in the header.
- NULL validity, bag multiplicity (row counts) and group presence are separate components of every data state.
- Data reach a capability as a *scope*: the rows the engine selected (its row bitmap), the requested columns as
  logical int64 values (decimals as scaled integers with their scale), a validity flag per value, and a dense group
  id per row (`math_scope.h`). States are accumulated chunk by chunk (`init`, `accumulate`, `merge`, `finish`), so a
  scan streams, runs in parallel on CPU threads, and maps directly onto a CUDA kernel.

## CUDA rules

- Include `math_cuda.h` and use its macros only: `TR_KERNEL` for kernels, `TR_DEVICE` for device functions,
  `TR_LAUNCH(kernel, blocks, threads, args...)` to launch, `tr_cuda_alloc/copy_in/copy_out/free` for memory, and the
  atomics it defines. Compiled with `-DEMULATE` these become plain C++: the launch loops over blocks and threads on
  the CPU, so the same source is tested here and runs on the GPU.
- A kernel's result must not depend on thread order: accumulate with atomics (integer adds, min, max, or), or give
  each thread its own output slot. No `__syncthreads` and no shared-memory barriers (they cannot be emulated
  serially). The emulator runs threads in a scrambled order when `TORBLES_EMU_SCRAMBLE=1` so that order dependence
  shows up in tests.
- Exact integer accumulation on the device uses 64-bit atomics on values whose sums are proven to fit (the host checks
  the bound from the column ranges and row count before launching and refuses with `TR_MATH_BUDGET` otherwise), or a
  two-word 128-bit accumulator from `math_cuda.h`.
- Each kernel has a host entry point `tr_mgpu_<fam>_<op>(...)` returning an error code; it reports whether it ran on
  a device or in emulation (`tr_mgpu_device_name`). Python never claims a CUDA run for an emulated one.
- Every kernel compiles with `nvcc -gencode arch=compute_80,code=sm_80` here (receipt `nvcc.json`), and its emulated
  results equal the CPU implementation's bit for bit (receipt `emulated.json`).

## Python rules

- `ctypes` bindings, argument checks and result formatting only. Exact results come back as `int` or `Fraction`,
  intervals as a pair with their contract, never as an unqualified float.
- A request outside the budgets raises `AlgebraBudgetError`; an unsupported type or domain raises `DomainError`;
  nothing falls back to a Python implementation.
- The public entry points are documented in `docs/MATHEMATICS.md` with their exact scope and refusals.

## Tests

A capability's tests are the methods of its test module whose names contain its ID in lower case without the dash
(`test_lin001_matches_matrix_products` for LIN-001); `bench/validate_math.py` selects them with `-k lin001`. Each
acceptance test of the inventory entry becomes at least one test. Every test that has a native result also
checks the CUDA path (emulated here, on the device with `--device`) for equality with the CPU result. Budget limits
are tested at the limit and one past it. Random tests use fixed seeds and record them.

## Benchmarks

`bench/benchmark_math_<fam>.py --capability ID --out FILE [--device]` writes one JSON receipt: the machine, the
library hashes, the seed and every raw sample; `--device` times the CUDA path on an actual device (it refuses to run
on the emulation).

Derived from the inventory's benchmark cases, kept small enough to run in minutes on a laptop: CPU native, CUDA
(device runs only), and the baseline the case names (complete reconstruction, DuckDB SQL, or a NumPy computation),
separating state construction, first execution and warm reuse; every raw sample is written to the receipt with the
machine, library hashes and seed.
