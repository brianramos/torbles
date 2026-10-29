# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Values wider than 32 bits are refused, never truncated: derived surrogates, moves and key maps (a surrogate that
outgrew 32 bits once wrapped silently at load and gave wrong joins)."""
import numpy as np, algq
from algq import Surrogate, Table
fails = 0
def refused(name, f):
    global fails
    try: f(); fails += 1; print("  FAIL (accepted):", name)
    except ValueError: print("  refused:", name)
refused("surrogate wider than 32 bits", lambda: Surrogate(21, np.array([0, (1 << 33) | 5], np.int64)))
sg = Surrogate(4, np.array([0, 1, 2], np.int64))
refused("move to a surrogate wider than 32 bits", lambda: sg.move([1], [1 << 32]))
refused("key map to values wider than 32 bits", lambda: algq.par_map(np.array([0, 1], np.uint32), np.array([5, 1 << 32], np.int64)))
refused("column values wider than 32 bits", lambda: Table("t", {"a": np.array([1 << 32], np.int64)}))
ok = algq.par_map(np.array([0, 1, 2], np.uint32), np.array([7, 0xFFFFFFFF, -1], np.int64)).tolist() == [7, 0xFFFFFFFF, 0xFFFFFFFF]
if not ok: fails += 1; print("  FAIL: 32-bit values and absent keys must map exactly")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
