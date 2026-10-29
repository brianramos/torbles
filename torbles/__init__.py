# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""torbles: an algebraic relational datastore on torus-encoded bit planes.

    import torbles
    db = torbles.from_duckdb("my.duckdb")            # keys from the declared PRIMARY KEY / FOREIGN KEY constraints
    db = torbles.from_ddl("schema.sql", {"sales": "sales.parquet", "customer": "customer.csv"})
    lo, c = db["sales"], db["customer"]
    db.query(lo).join(c).where(c.region == "EUROPE").group_by(c.nation).sum(lo.amount).run()
    db.save("my.torbles");  db = torbles.load("my.torbles")

The engine-level API (uint32 columns, hand-made keys) is torbles.Table / torbles.Query.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path: _sys.path.insert(0, _here)       # the engine modules import each other as top-level modules
from algq import *                                          # noqa: F401,F403
from algq import __version__, Table, Query, Col, Surrogate, PlaneKey, set_threads, set_build_threads, set_groupby, gpu_name, gpu_mm, mm_kernel  # noqa: F401
from blobcol import BlobStore                               # noqa: F401
from database import Database, from_duckdb, from_ddl, from_frictionless, from_frames, keys_from_ddl, load, LoadError   # noqa: F401
import algq, blobcol, coltypes, database, keydesign         # noqa: F401,E402
for _m in ("algq", "blobcol", "coltypes", "database", "keydesign"):   # one module object, however it is imported
    _sys.modules.setdefault("torbles." + _m, _sys.modules[_m])
