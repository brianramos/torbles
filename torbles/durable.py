# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Opt-in durable SQL over writable bitplane tables."""

import struct
from threading import Condition, RLock, current_thread

from . import sqlfuncs, sqltypes
from .database import Database
from .sqlfront import _TOK, split_statements, tokenize
from .sqltypes import SqlError


_LENGTH = struct.Struct("<I")
_SQL_EXECUTION = RLock()
_BEGIN = {("BEGIN",), ("BEGIN", "TRANSACTION"), ("START", "TRANSACTION")}
_COMMIT = {("COMMIT",), ("COMMIT", "TRANSACTION")}
_ROLLBACK = {("ROLLBACK",), ("ROLLBACK", "TRANSACTION")}
_TRANSACTION_VERBS = {command[0] for command in _BEGIN | _COMMIT | _ROLLBACK}


def _changes(payload):
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < _LENGTH.size:
            raise ValueError("truncated transaction record")
        size, = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        if size == 0 or size > len(payload) - offset:
            raise ValueError("invalid transaction record length")
        yield payload[offset:offset + size]
        offset += size


def open_durable(path, *, tables=None):
    """Create with {table: (columns, primary_key)}, or reopen with just path."""
    from .transaction_log import Journal

    return DurableDatabase(Journal(path, tables=tables))


class DurableDatabase:
    """A directory-backed SQL database with serial execution across its clients."""

    def __init__(self, journal):
        self._journal = journal
        self._lock = Condition()
        self._owner = None
        self._pending = []
        self._db = None
        try:
            self._db = self._recover()
        except BaseException:
            journal.close()
            raise

    def _recover(self):
        from . import transaction_redo

        db = Database()
        try:
            for name, (columns, key) in self._journal.load_base().items():
                db.add_writable_table(name, columns, primary_key=key)
            for record in self._journal.records():
                for change in _changes(record):
                    transaction_redo.replay(db, change)
            transaction_redo.attach(db, self._pending.append)
        except BaseException:
            db.close()
            raise
        return db

    def _check(self):
        if self._db is None:
            raise RuntimeError("durable database is closed or requires recovery")

    def connect(self):
        with self._lock:
            self._check()
            return DurableConnection(self)

    @property
    def wal_bytes(self):
        with self._lock:
            self._check()
            return self._journal.wal_bytes

    def _restore(self):
        old, self._db = self._db, None
        self._pending.clear()
        old.close()
        self._db = self._recover()

    def _commit(self):
        if not self._pending:
            return
        try:
            payload = b"".join(_LENGTH.pack(len(change)) + change for change in self._pending)
        except BaseException:
            self._restore()
            raise
        try:
            self._journal.append(payload)
        except BaseException:
            old, self._db = self._db, None
            old.close()
            raise
        finally:
            self._pending.clear()

    def close(self):
        with self._lock:
            if self._owner is not None:
                raise RuntimeError("finish the active transaction before closing the database")
            try:
                if self._db is not None:
                    old, self._db = self._db, None
                    old.close()
            finally:
                self._journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DurableConnection:
    """A SQL client. An explicit transaction belongs to its opening thread."""

    def __init__(self, database):
        self._database = database
        self._transaction_thread = None
        self._closed = False
        self._zone = None

    def _finish(self):
        self._transaction_thread = None
        self._database._owner = None
        self._database._lock.notify_all()

    def sql(self, text, engine="alg"):
        """Run one SQL statement. Mutations outside BEGIN commit automatically."""
        from .sqldmlparse import is_dml

        statements = split_statements(text)
        if len(statements) != 1:
            raise SqlError("durable SQL requires exactly one statement per call")
        statement = statements[0]
        first = next((m.group(0).upper() for m in _TOK.finditer(statement)
                      if m.lastgroup not in ("ws", "lc", "bc")), "")
        command = tuple(t.up() for t in tokenize(statement)) if first in _TRANSACTION_VERBS else ()
        db = self._database
        with db._lock:
            while db._owner is not None:
                owner = db._owner
                thread = owner._transaction_thread
                if thread is current_thread():
                    if owner is not self:
                        raise SqlError("another connection on this thread owns the transaction")
                    break
                if not thread.is_alive():
                    owner.close()
                    break
                if owner is self:
                    raise RuntimeError("use the transaction from the thread that began it")
                if self._closed:
                    raise RuntimeError("connection is closed")
                db._lock.wait(timeout=0.1)
            if self._closed:
                raise RuntimeError("connection is closed")
            db._check()
            if command in _BEGIN:
                if db._owner is not None:
                    raise SqlError("a transaction is already active")
                db._owner = self
                self._transaction_thread = current_thread()
                return None
            if command in _COMMIT or command in _ROLLBACK:
                if db._owner is None:
                    raise SqlError("no active transaction")
                try:
                    if command in _COMMIT:
                        db._commit()
                    elif db._pending:
                        db._restore()
                finally:
                    self._finish()
                return None

            mutates = is_dml(statement)
            try:
                # The SQL runtime also has process-wide evaluation settings.
                with _SQL_EXECUTION:
                    zone = sqltypes.SESSION["zone"]
                    now = sqlfuncs.CLOCK["now"]
                    db._db._sql_zone = self._zone
                    try:
                        result = db._db.sql(statement, engine)
                        self._zone = db._db._sql_zone
                    finally:
                        sqltypes.SESSION["zone"] = zone
                        sqlfuncs.CLOCK["now"] = now
            except BaseException:
                try:
                    if mutates or db._pending:
                        db._restore()
                finally:
                    if db._owner is self:
                        self._finish()
                raise
            if mutates and db._owner is None:
                db._commit()
            return result

    def close(self):
        db = self._database
        with db._lock:
            try:
                if db._owner is self:
                    try:
                        if db._pending:
                            db._restore()
                    finally:
                        self._finish()
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
