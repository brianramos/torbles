"""Reproducible native DAG property check against a Python integer oracle.

Run from the repository root after ``make``:
    python tests/validate_relationship_dag_random.py --attempts 2000 --seed 80492

This exercises the direct C executor's modular-64 contract, including values
whose sums exceed the public query API's normal exact-result width guards.
Oversized source expressions are counted and skipped at the 48-op ABI boundary.
"""
import argparse
import ctypes
import json
import random

import numpy as np

from torbles.algq import A_GENERAL, M, MInstr, QPlan, Table, lib


class DAGStats(ctypes.Structure):
    _fields_ = [("nodes", ctypes.c_int32), ("operations", ctypes.c_int32),
                ("registers", ctypes.c_int32), ("regions", ctypes.c_int32),
                ("stack_peak", ctypes.c_int32), ("inputs", ctypes.c_uint64)]


def validate(attempts=2000, seed=80492):
    lib.tr_dag_optimize.argtypes = [ctypes.c_void_p, ctypes.POINTER(QPlan),
                                    ctypes.POINTER(QPlan), ctypes.POINTER(DAGStats)]
    lib.tr_dag_optimize.restype = ctypes.c_int32
    rng = random.Random(seed)
    rows = 129
    values = np.array([[rng.randrange(32) for _ in range(rows)] for _ in range(4)], dtype=np.uint32)
    table = Table("dag_property", {str(c): values[c] for c in range(4)})
    constants = [0, 1, -1, 2, -3, (1 << 63) - 1, -(1 << 63)]

    def tree(depth):
        if depth == 0 or rng.random() < .27:
            return ("c", rng.choice(constants)) if rng.random() < .3 else ("x", rng.randrange(4))
        return (rng.choice(["+", "-", "*"]), tree(depth - 1), tree(depth - 1))

    def bytecode(node):
        if node[0] == "x":
            return [MInstr(M["FCOL"], node[1], 0)]
        if node[0] == "c":
            return [MInstr(M["CONST"], 0, node[1])]
        operation = M[{"+": "ADD", "-": "SUB", "*": "MUL"}[node[0]]]
        return bytecode(node[1]) + bytecode(node[2]) + [MInstr(operation, 0, 0)]

    def value(node, row):
        if node[0] == "x":
            return int(values[node[1], row])
        if node[0] == "c":
            return node[1]
        a, b = value(node[1], row), value(node[2], row)
        return a + b if node[0] == "+" else a - b if node[0] == "-" else a * b

    executed = skipped = normalized = shared = 0
    for attempt in range(attempts):
        expression = tree(3)
        if rng.random() < .4:
            expression = (rng.choice(["+", "*"]), expression, expression)
        code = bytecode(expression)
        if len(code) > 48:
            skipped += 1
            continue
        source = QPlan()
        source.agg, source.nmeas = A_GENERAL, len(code)
        for index, instruction in enumerate(code):
            source.meas[index] = instruction
        optimized, stats = QPlan(), DAGStats()
        rc = lib.tr_dag_optimize(table.at, ctypes.byref(source), ctypes.byref(optimized), ctypes.byref(stats))
        assert rc == 0, (seed, attempt, rc, expression)
        assert lib.at_query(table.at, ctypes.byref(source)) == 0, (seed, attempt, "source")
        assert lib.at_query(table.at, ctypes.byref(optimized)) == 0, (seed, attempt, "optimized")
        expected = sum(value(expression, row) for row in range(rows)) & ((1 << 64) - 1)
        if expected >> 63:
            expected -= 1 << 64
        assert source.total_sum == expected == optimized.total_sum, (
            seed, attempt, expression, expected, source.total_sum, optimized.total_sum)
        assert source.total_cnt == rows == optimized.total_cnt, (seed, attempt, "count")
        executed += 1
        normalized += stats.regions > 0
        shared += stats.registers > 0
    return dict(seed=seed, attempts=attempts, executed=executed, skipped_over_48=skipped,
                rows=rows, normalized=normalized, shared=shared, correct=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=80492)
    args = parser.parse_args()
    print(json.dumps(validate(args.attempts, args.seed), sort_keys=True))
