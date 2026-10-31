# State serialization and interoperability (FND-011)

`torbles/math_persist.h` (the format and every contract; the checksum weight shared with CUDA), `math_persist.c`,
`math_persist_gpu.cu` (the payload checksum on the device), `math_persist.py` (opt-in). Versioned binary records of
typed mathematical states; no live address is ever written, every pointer is rebuilt on load.

```python
from torbles import math_persist as mp
data = mp.dumps(bank, metadata={"note": "q3"})   # FND-004 bank (+ its scope), DIST-001 histogram, LIN-001 moments,
mp.save(state, path); rec = mp.load(path)        # FND-009 BigInt / BigRational / GFElement, FND-007 certificate
rec = mp.loads(data, expected="bank")             # PersistError subclasses on refusal (below)
rec.type, rec.header, rec.source, rec.metadata, rec.bank_states()
rec.fresh(scope)                                  # Freshness of the recorded source against the current scope
rec.bank(scope); rec.bank()                       # adopted as the scope's bank (StaleSourceError if it changed) / unscoped
rec.histogram(scope); rec.moments(scope); rec.value(); rec.certificate()
mp.validate(data, backend="cuda")                 # header + checksum (computed on the device)
mp.checksum(words, backend="cuda"); mp.need("save" | "load" | "checksum", record_bytes)
```

## Format (version 1)

A 128-byte header (magic `TORBMATH`, format version, header size, endianness marker `0x0102030405060708`, type tag
and type version, carrier, exactness, flags, the four section sizes, the total, the checksum) and four sections, each
padded to 8 bytes: **source** (the FND-007 record of the scope the state summarizes: lifetime token, revision, rows,
selected rows, version digest, selection digest, every column's encoding and NULL domain, the exact grouping
descriptor; absent for carriers and unscoped states), **basis** (the shape: bank columns, terms, mode, groups and
exponents; histogram spec and dense domain; moment spec, marginals, slots), **metadata** (the caller's bytes; Python
stores JSON with the names it needs to rebuild its objects), **payload** (bank rows / counts / 128-bit sums;
histogram rows, CSR offsets, coordinates, NULL masks, counts, weighted counts and sums; moment rows / counts / 192-bit
sums; big integer sign and limbs; rational numerator and denominator; GF(p) modulus and residue; a certificate's
canonical bytes). Every word is written little-endian byte by byte, so a record does not depend on the host.
Writing the same state twice gives the same bytes; a reloaded state written again gives the original bytes.

**Checksum**: two lanes, lane L = sum over every 64-bit word w_i of the record but words 12 and 13 (the checksum) of
w_i m_L(i) mod 2^64, m_L(i) an odd splitmix multiplier of the position. Order-independent (the CUDA kernel adds
per-slice partial sums with atomics) and any single-word change is detected.

## Refusals (distinct, in this order)

| reason | exception | when |
|---|---|---|
| truncated | `TruncatedStateError` | fewer bytes than the header or its total |
| magic | `NotAStateError` | not a torbles mathematical state |
| byte order | `ByteOrderError` | the endianness marker reads swapped |
| version | `UnknownVersionError` | an unknown format version or header size (before the checksum), an unknown type version (after it) |
| malformed | `MalformedStateError` | inconsistent sizes or trailing bytes (before the checksum), invalid sections (after it) |
| corrupt | `CorruptStateError` | the checksum differs (`.status` has stored and computed) |
| type | `WrongTypeError` | another state type than `expected` |

## Immutability and the source

A record keeps the source identity it was saved with; nothing re-stamps it. `fresh(scope)` is the FND-007 exact
comparison (identity, revision, rows, selection, NULL domain, columns, grouping) and every rehydration into a scope
requires it: a reloaded state cannot be adopted as the state of a changed source (`StaleSourceError` with
`.reasons`). The lifetime token and the version digest are process-specific, so a record of another process is
never fresh (its states can still be read, and a bank rehydrated unscoped; it cannot enter a FND-004 cache).

## Budgets

Saving: bytes = the record + the blob object (`mp.need("save", n)`); loading: bytes = twice the record + the record
object (`mp.need("load", n)`), then the rehydrated object's own checks (FND-004 / DIST-001 / LIN-001 / FND-009 budgets);
the device checksum: bytes = 8 per word + 16. `AlgebraBudgetError` before anything is allocated.

## Limitations

A moment state records the accumulation width it used (128 or 192 bits), not the requested policy: a state built
with width `auto` reloads with that width declared (identical states; LIN-001's native equality also compares the
requested width, so compare the states). Histograms and moments rehydrate only into their recorded source (their
adoption needs the scope); unscoped ones can be read natively (`tr_persist_histogram_view` /
`tr_persist_moments_view`; in Python only `Record.bank_states()` is wrapped) but not rebuilt. Only little-endian
records are written (a foreign byte order is refused, not converted). Metadata is opaque to the native code.
