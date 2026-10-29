# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""torbles: an algebraic relational datastore on torus-encoded bit planes.

    import torbles                      # or, with torbles/ on sys.path: import algq
    from torbles import Table, Query
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path: _sys.path.insert(0, _here)       # the engine modules import each other as top-level modules
from algq import *                                          # noqa: F401,F403
from algq import __version__, Table, Query, Col, Surrogate, PlaneKey, set_threads, set_build_threads, set_groupby, gpu_name, gpu_mm, mm_kernel  # noqa: F401
from blobcol import BlobStore                               # noqa: F401
