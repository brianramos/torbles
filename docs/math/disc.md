# Relationship verification (DISC-001)

The bounded affine, sum and product discovery of `torbles/relationships.c` (`torbles.relationships.discover`), plus
`math_disc.c` / `math_disc_gpu.cu` / `math_disc.py`: the exhaustive verification of one hypothesis on the CPU or with
a CUDA kernel, with the catalog's semantics and without registering anything.

```python
from torbles.relationships import discover
from torbles import math_disc
cat = discover(db, "t", ["x", "y", "s"])
math_disc.verify(cat, "sum", "s", "x", "y", backend="cuda")
# {"checked", "excluded", "violations", "witness", "usable", "varied_left", "varied_right", "backend", "device"}
math_disc.propose_affine(cat, "a", "x")       # (scale, bias) the discovery would verify, or None
```

- A used row with a NULL in one of the hypothesis's columns is excluded; every other used row is checked against
  `out = scale*left + bias`, `left*right` or `left + right`, exactly (128-bit), on logical values (code + codec offset,
  which must fit int64; otherwise `OverflowError`). The witness is the smallest violating stored row.
- `usable` (an exact rewrite) needs every physical row in scope, at least two checked rows, no exclusion, no violation,
  varied inputs and a non-zero affine scale. Partial, nullable and failed proofs are evidence only.
- A catalog whose table changed since discovery (revision, rows or used rows) raises `StaleStateError`.
- `backend="cuda"` reports `"cuda"` with the device name on an actual device and `"cuda-emulated"` otherwise; the
  device copies of the columns are budgeted (`budget=`, resource `bytes`; `math_disc.device_bytes`).
