# State ownership, concurrency and the native ABI (FND-013)

`torbles/math_handles.h`, `math_handles.c`, `math_handles_gpu.cu`, `math_handles.py`. A caller-owned native table of
opaque handles to immutable (and serialized mutable) mathematical states, and the size/version check of every struct
that crosses a ctypes boundary.

```python
from torbles import math_handles as mh
with mh.HandleTable(capacity=1024) as table:                  # created and owned by the caller; no global table
    p = table.polynomial({(2, 0): 3, (1, 1): -5, (0, 0): 7})    # immutable 3*x0^2 - 5*x0*x1 + 7
    p.evaluate([[1, 2], [10**18, 3]])                          # {"values": [0, 2999...7], "overflow": [...], ...}
    p.evaluate(points, backend="cuda")                         # "cuda", or "cuda-emulated" with the emulation
    with p.retained():                                         # a native reference for the block
        p.close()                                              # "deferred": freed when the block ends
    table.domain("snapshot", b"rows", nvars=1, ndata=1, universe_rows=10)   # first-slice states, owned by the table
    table.bank({(0,): 3, (1,): 6, (2,): 14})
    cache = table.state_cache(capacity=64)                     # mutable; put / get / stats serialized natively
    table.stats()                                              # live handles, payload bytes, native allocations
mh.abi_check()                                                 # every ctypes Structure of torbles.math* vs native
```

## Handles

- A handle is a 64-bit id: slot index in bits 0..31, the slot's generation (>= 1) in bits 32..63; 0 and every value
  below 2^32 are never issued. A freed slot is reused (last freed first) with the next generation; a slot whose
  generation reaches `generation_limit` (default 2^32 - 1) is retired, so no (slot, generation) is issued twice.
- Errors are decided from one atomic read of the slot and never depend on timing: a closed handle raises
  `ClosedHandleError` (native `TR_MATH_CLOSED` = -17) whether or not its payload was freed yet; a value never issued,
  or a handle of another kind, raises `ValueError` (`TR_MATH_INVALID`). A closed table refuses everything with
  `ClosedHandleError`. `close()` returns `"freed"`, `"deferred"` or `"closed"` (already closed).
- Every native operation retains the handles it uses (a C11 atomic reference count in the slot's state word) and
  releases them when it returns; the CUDA path keeps the polynomial retained around the device call
  (`tr_math_handles_ipoly_view` ... `tr_math_handles_release`). A close marks the handle closed at once and frees the
  payload only when no operation holds it, otherwise at the last release: a payload is never freed under a live
  operation. `HandleTable.close()` waits for the operations in flight in other threads and then frees every live
  payload; it refuses to run inside one of its own operations (`RuntimeError`), and the native
  `tr_math_handles_free` refuses (-1) while any handle is retained.
- Owned handles (returned by the table's constructors) are closed when garbage-collected; `table.handle(value)`
  returns a borrowed wrapper of a raw id.

## Locking

`retain` / `release` are lock-free (one compare-and-swap); immutable payloads (polynomials, domains, banks) are read
concurrently by any number of operations. The table's mutex serializes insert, close, the final free of a payload (the
destructor runs with the mutex held), `stats` / `info` and free. A mutable payload (`TR_HANDLE_MUTABLE`, e.g. the state
cache) has its own error-checking mutex: `tr_math_handles_lock` = retain + lock, `tr_math_handles_unlock` = unlock +
release, so every call on one cache is serialized natively. A payload mutex may be held while taking the table mutex,
never the reverse. Python native calls release the GIL; the Python wrapper adds one short lock per call for the
table's own lifetime only.

## The polynomial state

An immutable sparse polynomial in `nvars` coordinates with int64 coefficients and exponents 0..255, stored
canonically: equal exponent vectors merged exactly (a merged coefficient outside int64 raises `OverflowError`), zero
terms dropped, rows sorted. `terms`, `term_count`, `degree` read the native state. At an int64 point the value is
returned exactly as a 128-bit integer; it is flagged as overflow (`None`, never wrapped) exactly when a canonical term
`c * prod x_v^e_v` or the sum lies outside [-2^127, 2^127 - 1]. Terms are unsigned magnitudes with overflow detection
summed in 192 bits, so the flag does not depend on the order of the terms; `x^0 = 1` also at `x = 0`, and a zero
factor makes a term exactly 0. The CUDA kernel copies the terms once per call into read-only device buffers shared by
every thread and writes one output slot per point (an atomic counts the flagged points); its results equal the CPU's
bit for bit. `evaluate_raw` returns the (lo, hi) words and flags as arrays.

## Budgets and refusals (`AlgebraBudgetError` with `.resource`, `.need`, `.limit`)

- Table: `states` = capacity (1..2^24), `bytes` = its fixed allocation (`table_bytes()`). A full table (every slot live
  or retired) refuses the next state with `states`, need capacity + 1, before the state is built.
- Polynomial: `variables` = nvars, `degree` = the largest total degree of an input term, `support` = input terms,
  `bytes` = payload + sorting scratch (`tr_math_handles_ipoly_bytes`), checked before allocation; more than 65535
  variables is `UnsupportedError`.
- Evaluation: `output` = points, on both backends; the CUDA path also budgets its device copies (`bytes` =
  `device_bytes(npoints)`). Defaults are those of `math_budgets.Budget()`.
- `stats()`: `live`, `open`, `deferred`, `retired`, `free_slots`, `inserted`, `closes`, `deferred_closes`,
  `destroyed`, `retains` / `releases` (equal when nothing is in flight), `refused_closed` / `refused_invalid`,
  `payload_bytes` (exact for polynomials, the domain's bytes, a cache's `max_bytes` bound, 0 for banks),
  `peak_payload_bytes`, `allocations` (this module's live blocks; 1 = the table alone), `live_kind`.

## The ABI registry

Each struct that crosses a ctypes boundary is registered once, in the C file of its family:
`TR_MATH_ABI(type, version);` in a `torbles/math*.c` file (libalg.so) and `TR_MGPU_ABI(type, version);` in a
`torbles/math_*_gpu.cu` file (the CUDA library). Entries are constants in a linker section, so a family adds its line
without editing any shared file; `tr_math_handles_abi` / `tr_mgpu_handles_abi` report them all (name, size,
alignment, version as compiled). In Python the mirror declares its C type: `_native_ = "tr_lin_matrix"` (and
`_abi_version_ = 2` when not 1) on the `ctypes.Structure`. `abi_check()` compares every module-level ctypes Structure
of every `torbles.math*` module with every registry that has its struct and raises `AbiError` on a size, alignment or
version mismatch, an unregistered Structure, a Structure no registry knows, or libraries that disagree;
`math_handles` also checks the loaded modules against libalg.so (and the CUDA library) at first use. Registered now:
the first slice, FND-010, DISC-001, `tr_math_i128` / `tr_math_interval` and this module's structs.

Not covered: the first-slice Python wrappers (`Domain`, `MomentCache`, expressions) keep their own ownership; their
close is still externally serialized with their operations. Only the states above have table kinds.
