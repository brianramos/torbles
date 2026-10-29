# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""CLOB / BLOB workloads: torbles (self-extending records: head limbs in planes, payload heap with derived offsets)
vs DuckDB (VARCHAR / BLOB columns) vs two native object stores: LMDB (memory-mapped key-value store) and a filesystem
object store (one file per object, sharded directories). Same seeded data in every engine; each engine runs in its
own process; every answer is cross-checked.
  python3 blob_bench.py clob|blob [N]                        runs all engines and prints the comparison
  python3 blob_bench.py clob|blob N engine                   one engine (internal)
Workload: load; fetch 10k random objects; fetch a 1k id range; prefix filter (CLOB: 'svc-17|', BLOB: '%PDF' magic);
length statistics; GROUP BY leading bytes (service / file type) with count and total size; substring search
(CLOB: 'ERROR', BLOB: a marker planted in 1% of objects); replace 1% of objects, then fetch and filter again."""
import sys, os, time, json, zlib, shutil, subprocess, resource, numpy as np

KIND = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else (500_000 if KIND == "clob" else 50_000)
ENGINE = sys.argv[3] if len(sys.argv) > 3 else None
WORK = os.environ.get("BLOB_WORK", os.path.expanduser("~/torbles_data/blobwork")); os.makedirs(WORK, exist_ok=True)

def generate(kind, n, seed=7):
    rng = np.random.default_rng(seed)
    if kind == "clob":
        words = [b"GET", b"POST", b"user", b"id", b"ok", b"time", b"ms", b"cache", b"miss", b"hit", b"db", b"query", b"retry",
                 b"timeout", b"ERROR", b"WARN", b"INFO", b"node", b"shard", b"commit", b"json", b"payload", b"region", b"eu-west"]
        p = np.array([6, 4, 8, 8, 8, 6, 6, 4, 3, 3, 4, 4, 2, 1, 0.15, 1, 6, 4, 3, 2, 3, 3, 2, 2], float); p /= p.sum()
        corpus = b" ".join(words[i] for i in rng.choice(len(words), 4_000_000, p=p))
        sizes = np.clip(rng.lognormal(5.7, 1.0, n), 20, 256 * 1024).astype(np.int64)
        starts = rng.integers(0, len(corpus) - 300_000, n); svc = rng.integers(0, 100, n); lvl = rng.choice([b"INFO", b"WARN", b"DEBUG"], n)
        out = []
        for i in range(n):
            pre = b"svc-%02d|%s|" % (svc[i], lvl[i]); s = int(sizes[i]) - len(pre)
            out.append(pre + corpus[starts[i]: starts[i] + max(s, 0)])
        return out
    magics = [b"\x89PNG", b"%PDF", b"PK\x03\x04", b"\xff\xd8\xff\xe0"]
    kinds = rng.choice(4, n, p=[0.35, 0.2, 0.15, 0.3]); sizes = np.clip(rng.lognormal(9.0, 1.2, n), 64, 4 << 20).astype(np.int64)
    mark = rng.random(n) < 0.01; buf = rng.integers(0, 256, int(sizes.sum()), dtype=np.uint8).tobytes()
    out, pos = [], 0
    for i in range(n):
        s = int(sizes[i]); o = bytearray(buf[pos: pos + s]); pos += s
        o[:4] = magics[kinds[i]]
        if mark[i]: at = int(rng.integers(4, s - 7)); o[at: at + 7] = b"TB-MARK"
        out.append(bytes(o))
    return out

if KIND == "clob": PREFIX, PAT, BIG, GROUP = b"svc-17|", b"ERROR", 4096, ("svc", 6)
else: PREFIX, PAT, BIG, GROUP = b"%PDF", b"TB-MARK", 1 << 20, ("magic", 2)       # the file types differ in 2 bytes
def checksum(objs): return [len(objs), sum(len(o) for o in objs), sum(zlib.crc32(o) for o in objs) & 0xFFFFFFFFFFFF]
def best(f, n=3):
    b, r = 1e9, None
    for _ in range(n): t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r

def run_engine(name):
    objs = generate(KIND, N)
    rng = np.random.default_rng(3); ids = rng.integers(0, N, 10_000); rng_ids = np.arange(N // 2, N // 2 + 1000)
    upd_ids = np.sort(rng.choice(N, N // 100, replace=False)); upd = generate(KIND, len(upd_ids), seed=99)
    R, T = {}, {}
    t0 = time.perf_counter()
    if name == "torbles":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "torbles"))
        from blobcol import BlobStore
        from algq import set_threads
        set_threads(2); st = BlobStore(objs); del objs
        T["load"] = time.perf_counter() - t0; size = sum(st.bytes_used().values()); R["bytes_detail"] = st.bytes_used()
        get = lambda ids: st.get(ids)
        prefix = lambda: st.count_prefix(PREFIX)
        lens = lambda: list(st.length_stats(BIG))
        grp = lambda: {k.hex(): list(v) for k, v in st.group_leading(0, 2).items()}
        if KIND == "clob":                       # the service number: bytes 5-6 = the top 2 bytes of limb 1
            grp = lambda: {(b"svc-" + k).hex(): list(v) for k, v in st.group_leading(1, 2).items()}
        contains = lambda: st.count_contains(PAT)
        replace = lambda: st.replace(upd_ids, upd)
    elif name == "duckdb":
        import duckdb, pandas as pd
        path = os.path.join(WORK, f"{KIND}.duckdb")
        for f in (path, path + ".wal"):
            if os.path.exists(f): os.remove(f)
        con = duckdb.connect(path); con.execute("SET threads = 2"); con.execute("SET memory_limit = '4GB'")
        col = (lambda o: o.decode()) if KIND == "clob" else (lambda o: o)
        for i in range(0, N, 100_000):
            con.register("t0", pd.DataFrame({"id": np.arange(i, min(N, i + 100_000)), "o": [col(o) for o in objs[i: i + 100_000]]}))
            con.execute("CREATE TABLE t AS SELECT * FROM t0" if i == 0 else "INSERT INTO t SELECT * FROM t0"); con.unregister("t0")
        con.execute("CHECKPOINT"); del objs
        T["load"] = time.perf_counter() - t0; size = os.path.getsize(path)
        enc = (lambda v: v.encode()) if KIND == "clob" else (lambda v: bytes(v))
        def get(ids):
            con.register("q", pd.DataFrame({"id": np.asarray(ids)}))
            rows = dict(con.execute("SELECT t.id, t.o FROM t JOIN (SELECT DISTINCT id FROM q) q USING (id)").fetchall()); con.unregister("q")
            return [enc(rows[int(i)]) for i in ids]
        L = "length(o)" if KIND == "clob" else "octet_length(o)"
        if KIND == "clob":
            prefix = lambda: con.execute(f"SELECT count(*) FROM t WHERE starts_with(o, '{PREFIX.decode()}')").fetchall()[0][0]
            grp = lambda: {k.encode().hex(): [c, s] for k, c, s in con.execute(f"SELECT substr(o, 1, 6), count(*), sum({L}) FROM t GROUP BY 1").fetchall()}
            contains = lambda: con.execute(f"SELECT count(*) FROM t WHERE contains(o, '{PAT.decode()}')").fetchall()[0][0]
        else:
            prefix = lambda: con.execute("SELECT count(*) FROM t WHERE o[1:4] = '%PDF'::BLOB").fetchall()[0][0]
            grp = lambda: {bytes(k).hex(): [c, s] for k, c, s in con.execute(f"SELECT o[1:2], count(*), sum({L}) FROM t GROUP BY 1").fetchall()}
            contains = lambda: con.execute(f"SELECT count(*) FROM t WHERE contains(o::VARCHAR, '{PAT.decode()}')").fetchall()[0][0]
        lens = lambda: [int(x) for x in con.execute(f"SELECT sum({L}), count(*) FILTER (WHERE {L} > {BIG}) FROM t").fetchall()[0]]
        def replace():
            con.register("u0", pd.DataFrame({"id": upd_ids, "o": [col(o) for o in upd]}))
            con.execute("UPDATE t SET o = u0.o FROM u0 WHERE t.id = u0.id"); con.unregister("u0")
    elif name == "lmdb":
        import lmdb
        path = os.path.join(WORK, f"{KIND}.lmdb"); shutil.rmtree(path, ignore_errors=True)
        env = lmdb.open(path, map_size=1 << 36, writemap=True, sync=False, metasync=False)
        with env.begin(write=True) as txn:
            for i, o in enumerate(objs): txn.put(i.to_bytes(8, "big"), o, append=True)
        env.sync(); del objs
        T["load"] = time.perf_counter() - t0; size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
        size = env.info()["last_pgno"] * env.stat()["psize"]
        def get(ids):
            with env.begin() as txn: return [txn.get(int(i).to_bytes(8, "big")) for i in ids]
        def scan(f):
            with env.begin(buffers=True) as txn:
                for _, v in txn.cursor(): f(v)
        def prefix():
            c = [0]; m = len(PREFIX)
            def f(v): c[0] += bytes(v[:m]) == PREFIX
            scan(f); return c[0]
        def lens():
            s = [0, 0]
            def f(v): l = len(v); s[0] += l; s[1] += l > BIG
            scan(f); return s
        def grp():
            g = {}
            def f(v):
                k = bytes(v[:GROUP[1]]).hex(); c = g.setdefault(k, [0, 0]); c[0] += 1; c[1] += len(v)
            scan(f); return g
        def contains():
            c = [0]
            def f(v): c[0] += PAT in v.tobytes() if len(v) > 4096 else PAT in bytes(v)
            scan(f); return c[0]
        def replace():
            with env.begin(write=True) as txn:
                for i, o in zip(upd_ids.tolist(), upd): txn.put(int(i).to_bytes(8, "big"), o)
    elif name == "fs":
        root = os.path.join(WORK, f"{KIND}_fs"); shutil.rmtree(root, ignore_errors=True)
        def path(i): return os.path.join(root, f"{i & 0xFF:02x}", str(i))     # 256 shard directories
        for a in range(256): os.makedirs(os.path.join(root, f"{a:02x}"), exist_ok=True)
        for i, o in enumerate(objs):
            with open(path(i), "wb") as fh: fh.write(o)
        del objs
        T["load"] = time.perf_counter() - t0
        size = int(subprocess.run(["du", "-sb", root], capture_output=True, text=True).stdout.split()[0])
        def get(ids):
            out = []
            for i in ids:
                with open(path(int(i)), "rb") as fh: out.append(fh.read())
            return out
        def files():
            for a in range(256):
                with os.scandir(os.path.join(root, f"{a:02x}")) as it:
                    for e in it: yield e
        def prefix():
            c = 0; m = len(PREFIX)
            for e in files():
                with open(e.path, "rb") as fh: c += fh.read(m) == PREFIX
            return c
        def lens():
            s = [0, 0]
            for e in files(): l = e.stat().st_size; s[0] += l; s[1] += l > BIG
            return s
        def grp():
            g = {}
            for e in files():
                with open(e.path, "rb") as fh: k = fh.read(GROUP[1]).hex()
                c = g.setdefault(k, [0, 0]); c[0] += 1; c[1] += e.stat().st_size
            return g
        def contains():
            c = 0
            for e in files():
                with open(e.path, "rb") as fh: c += PAT in fh.read()
            return c
        def replace():
            for i, o in zip(upd_ids.tolist(), upd):
                with open(path(int(i)), "wb") as fh: fh.write(o)
    T["get_10k"], r = best(lambda: get(ids)); R["get_10k"] = checksum(r)
    T["get_range_1k"], r = best(lambda: get(rng_ids)); R["get_range_1k"] = checksum(r)
    T["prefix"], R["prefix"] = best(prefix)
    T["lengths"], R["lengths"] = best(lens)
    T["group_leading"], R["group_leading"] = best(grp)
    T["contains"], R["contains"] = best(contains, 1 if name == "fs" else 3)
    t0 = time.perf_counter(); replace(); T["replace_1pct"] = time.perf_counter() - t0
    T["get_10k_after"], r = best(lambda: get(ids)); R["get_10k_after"] = checksum(r)
    T["prefix_after"], R["prefix_after"] = best(prefix)
    T["contains_after"], R["contains_after"] = best(contains, 1 if name == "fs" else 3)
    R["size_bytes"] = int(size); R["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    json.dump(dict(engine=name, kind=KIND, n=N, times=T, results=R), open(os.path.join(WORK, f"{KIND}_{name}.json"), "w"))

if ENGINE: run_engine(ENGINE); sys.exit(0)
ENGINES = ["torbles", "duckdb", "lmdb", "fs"]
raw = sum(len(o) for o in generate(KIND, N))
print(f"{KIND.upper()}: {N:,} objects, {raw / 2**20:.0f} MiB of payload")
for e in ENGINES:
    t = time.perf_counter(); rc = subprocess.run([sys.executable, os.path.abspath(__file__), KIND, str(N), e]).returncode
    print(f"  ran {e:7s} in {time.perf_counter() - t:6.1f} s" + ("" if rc == 0 else f"  FAILED rc={rc}"), flush=True)
J = {e: json.load(open(os.path.join(WORK, f"{KIND}_{e}.json"))) for e in ENGINES}
keys = ["get_10k", "get_range_1k", "prefix", "lengths", "group_leading", "contains", "get_10k_after", "prefix_after", "contains_after"]
ok = all(J[e]["results"][k] == J["torbles"]["results"][k] for e in ENGINES for k in keys)
print(f"{'operation':20s}" + "".join(f"{e:>12s}" for e in ENGINES))
for k in ["load"] + keys[:6] + ["replace_1pct"] + keys[6:]:
    print(f"{k:20s}" + "".join(f"{J[e]['times'][k] * 1e3:10.1f}ms" for e in ENGINES))
print(f"{'size':20s}" + "".join(f"{J[e]['results']['size_bytes'] / 2**20:10.0f}MiB" for e in ENGINES))
print(f"{'peak memory':20s}" + "".join(f"{J[e]['results']['peak_rss_gib']:10.2f}GiB" for e in ENGINES))
print("torble storage:", {k: f"{v / 2**20:.1f} MiB" for k, v in J["torbles"]["results"]["bytes_detail"].items()})
print("all engines agree" if ok else "MISMATCH: " + str([(e, k) for e in ENGINES for k in keys if J[e]["results"][k] != J["torbles"]["results"][k]]))
