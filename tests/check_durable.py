# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Durable SQL recovery, visibility, and commit failure tests."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

import torbles


TABLES = {
    "fact": ({"id": [1, 2], "v": [10, 20]}, "id"),
    "stage": ({"id": [2, 5], "v": [22, 55]}, "id"),
}
ROWS = "SELECT id, v FROM fact ORDER BY id"
MERGE = (
    "MERGE INTO fact f USING stage s ON f.id = s.id "
    "WHEN MATCHED THEN UPDATE SET v = s.v "
    "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)"
)


class DurableChecks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "database"

    def test_four_mutations_replay_and_rollback(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as con:
            con.sql("/* test */ BEGIN TRANSACTION;")
            con.sql("INSERT INTO fact VALUES (3, 30)")
            con.sql("UPDATE fact SET v = v + 10 WHERE id = 1")
            con.sql("DELETE FROM fact WHERE id = 3")
            con.sql(MERGE)
            con.sql("UPDATE stage SET v = 0 WHERE id = 2")
            expected = [(1, 20), (2, 22), (5, 55)]
            self.assertEqual(con.sql(ROWS).rows, expected)
            con.sql("COMMIT TRANSACTION")

            n = db.wal_bytes
            con.sql("BEGIN")
            con.sql("UPDATE fact SET v = 0")
            con.sql("ROLLBACK")
            self.assertEqual(db.wal_bytes, n)
            self.assertEqual(con.sql(ROWS).rows, expected)

            con.sql("BEGIN")
            con.sql("DELETE FROM fact")
            with self.assertRaises(torbles.SqlError):
                con.sql("INSERT INTO fact VALUES (2, NULL)")
            self.assertEqual(con.sql(ROWS).rows, expected)
            self.assertEqual(db.wal_bytes, n)

        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(con.sql(ROWS).rows, expected)
            self.assertEqual(con.sql("SELECT v FROM stage WHERE id = 2").rows, [(0,)])

    def test_clients_wait_for_commit(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as writer:
            with db.connect() as reader, ThreadPoolExecutor(1) as pool:
                entered = Event()

                def read():
                    entered.set()
                    return reader.sql(ROWS).rows

                writer.sql("BEGIN")
                writer.sql("DELETE FROM fact")
                future = pool.submit(read)
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertFalse(future.done())
                    with self.assertRaises(torbles.SqlError):
                        reader.sql(ROWS)
                    writer.sql("INSERT INTO fact VALUES (1, 100)")
                finally:
                    writer.sql("COMMIT")
                self.assertEqual(future.result(timeout=5), [(1, 100)])

    def test_abandoned_transaction_rolls_back(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as reader:
            writer = db.connect()
            changed = Event()

            def abandon():
                writer.sql("BEGIN")
                writer.sql("UPDATE fact SET v = 99 WHERE id = 1")
                changed.set()

            thread = Thread(target=abandon)
            thread.start()
            thread.join(timeout=5)
            self.assertTrue(changed.is_set())
            self.assertEqual(reader.sql(ROWS).rows, [(1, 10), (2, 20)])
            with self.assertRaises(RuntimeError):
                writer.sql(ROWS)

    def test_process_crash_before_and_after_commit(self):
        with torbles.open_durable(self.path, tables=TABLES):
            pass
        code = """
import os, signal, sys, torbles
db = torbles.open_durable(sys.argv[1])
c = db.connect()
c.sql("BEGIN")
c.sql("UPDATE fact SET v = 100 WHERE id = 1")
c.sql("INSERT INTO fact VALUES (3, 30)")
if sys.argv[2] == "committed":
    c.sql("COMMIT")
os.kill(os.getpid(), signal.SIGKILL)
"""
        for state, expected in [
            ("uncommitted", [(1, 10), (2, 20)]),
            ("committed", [(1, 100), (2, 20), (3, 30)]),
        ]:
            child = subprocess.run([sys.executable, "-c", code, os.fspath(self.path), state],
                                   check=False, timeout=30)
            self.assertEqual(child.returncode, -signal.SIGKILL)
            with torbles.open_durable(self.path) as db, db.connect() as con:
                self.assertEqual(con.sql(ROWS).rows, expected)

    def test_process_crash_mid_frame(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as con:
            con.sql("UPDATE fact SET v = 11 WHERE id = 1")
            committed = db.wal_bytes
        code = """
import os, signal, sys, torbles
from torbles import transaction_log
db = torbles.open_durable(sys.argv[1])
c = db.connect()
c.sql("BEGIN")
c.sql("UPDATE fact SET v = 99 WHERE id = 1")
c.sql("DELETE FROM fact WHERE id = 2")
write = transaction_log._write_all
writes = 0
def crash_during_payload(fd, data):
    global writes
    writes += 1
    if writes == 2:
        write(fd, data[:len(data) // 2])
        os.fsync(fd)
        os.kill(os.getpid(), signal.SIGKILL)
    write(fd, data)
transaction_log._write_all = crash_during_payload
c.sql("COMMIT")
"""
        child = subprocess.run([sys.executable, "-c", code, os.fspath(self.path)],
                               check=False, timeout=30)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertGreater((self.path / "wal").stat().st_size, committed)
        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(db.wal_bytes, committed)
            self.assertEqual(con.sql(ROWS).rows, [(1, 11), (2, 20)])
            con.sql("INSERT INTO fact VALUES (4, 40)")
        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(con.sql(ROWS).rows, [(1, 11), (2, 20), (4, 40)])

    def test_incomplete_tail_removed_and_corruption_detected(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as con:
            con.sql("INSERT INTO fact VALUES (3, 30)")
            committed = db.wal_bytes
        wal = self.path / "wal"
        with wal.open("ab") as stream:
            stream.write(b"\x00\x01")
        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(db.wal_bytes, committed)
            self.assertEqual(con.sql(ROWS).rows, [(1, 10), (2, 20), (3, 30)])
            con.sql("UPDATE fact SET v = 31 WHERE id = 3")
        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(con.sql(ROWS).rows[-1], (3, 31))
        with wal.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            old = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([old[0] ^ 1]))
        with self.assertRaises((OSError, ValueError)):
            torbles.open_durable(self.path)

    def test_seed_validation_can_be_retried(self):
        duplicate = {"fact": ({"id": [1, 1], "v": [10, 20]}, "id")}
        with self.assertRaises(ValueError):
            torbles.open_durable(self.path, tables=duplicate)
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as con:
            n = db.wal_bytes
            con.sql("BEGIN")
            self.assertEqual(con.sql(ROWS).rows, [(1, 10), (2, 20)])
            con.sql("COMMIT")
            self.assertEqual(db.wal_bytes, n)

    def test_exclusive_open_and_uncertain_commit(self):
        with torbles.open_durable(self.path, tables=TABLES) as db, db.connect() as con:
            with self.assertRaises(BlockingIOError):
                torbles.open_durable(self.path)
            append = db._journal.append

            def uncertain(payload):
                append(payload)
                raise OSError("injected failure after WAL write")

            with patch.object(db._journal, "append", uncertain):
                with self.assertRaises(OSError):
                    con.sql("INSERT INTO fact VALUES (3, 30)")
            with self.assertRaises(RuntimeError):
                con.sql(ROWS)
        with torbles.open_durable(self.path) as db, db.connect() as con:
            self.assertEqual(con.sql(ROWS).rows, [(1, 10), (2, 20), (3, 30)])


if __name__ == "__main__":
    unittest.main()
