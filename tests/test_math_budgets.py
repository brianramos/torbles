# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-010: search, representation and output budgets (torbles/math_budgets.c, math_budgets_gpu.cu).

Oracles: exact Python integers (math.comb, math.prod) for the size formulas, numpy.unique for the distinct-cell
pre-pass, and the existing polynomial kernel's own refusals for its fixed limits. Seeds are fixed.
"""
import math
import unittest

import numpy as np

from torbles import math_budgets as mb, math_gpu
from torbles.mathematics import AlgebraBudgetError, Context

U64 = (1 << 64) - 1


def sat(v):
    return v if v <= U64 else U64


class Budgets(unittest.TestCase):
    def test_fnd010_size_formulas_match_exact_integers(self):
        for n in range(0, 40):
            for d in range(0, 40):
                self.assertEqual(mb.monomials(n, d), sat(math.comb(n + d, d)), (n, d))
                self.assertEqual(mb.monomials(n, d, homogeneous=True),
                                 1 if d == 0 else (0 if n == 0 else sat(math.comb(n + d - 1, d))), (n, d))
        for n, d in ((1000, 30), (64, 64), (200, 200), (2**31, 3), (3, 2**31)):
            self.assertEqual(mb.monomials(n, d), sat(math.comb(n + d, d)), (n, d))
        for n, k in ((67, 33), (68, 34), (2**63, 1), (2**63, 2), (2**64 - 1, 0), (5, 9)):
            self.assertEqual(mb.binomial(n, k), sat(math.comb(n, k)), (n, k))
        rng = np.random.default_rng(10)
        for _ in range(300):
            dims = [int(x) for x in rng.integers(0, 2**rng.integers(1, 40), size=rng.integers(0, 6))]
            self.assertEqual(mb.product(dims), sat(math.prod(dims)), dims)
        for k in range(0, 70):
            self.assertEqual(mb.power2(k), sat(2**k))

    def test_fnd010_check_refuses_first_resource_in_fixed_order(self):
        b = mb.Budget(variables=8, degree=8, support=128, output=1000, bytes=1 << 20)
        mb.check(b, variables=8, degree=8, support=128, output=1000, bytes=1 << 20)      # at every limit
        for name, limit in (("variables", 8), ("degree", 8), ("support", 128), ("output", 1000), ("bytes", 1 << 20)):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mb.check(b, **{name: limit + 1})
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), (name, limit + 1, limit))
        # several over: the first in RESOURCES order is reported, the same way every time
        for _ in range(3):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mb.check(b, bytes=1 << 30, support=129, degree=9)
            self.assertEqual(ctx.exception.resource, "degree")
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.check(b, support=mb.monomials(10**6, 10**6))
        self.assertEqual(ctx.exception.need, U64)
        self.assertEqual(mb.Budget().limits, dict(variables=64, degree=64, support=1 << 20, tensor=1 << 26,
                                                  circuit=1 << 20, iterations=1 << 20, output=1 << 24,
                                                  states=1 << 24, bytes=1 << 30))
        with self.assertRaises(ValueError):
            mb.Budget(colours=3)
        with self.assertRaises(ValueError):
            mb.check(b, support=-1)

    def test_fnd010_choose_expanded_or_factored_is_explicit(self):
        b = mb.Budget(support=1000)
        both = mb.choose(b, dict(support=500), 10.0, dict(support=40), 20.0)
        self.assertEqual((both["choice"], both["reason"]), ("expanded", "cheaper"))
        cheaper_factored = mb.choose(b, dict(support=500), 30.0, dict(support=40), 20.0)
        self.assertEqual((cheaper_factored["choice"], cheaper_factored["reason"]), ("factored", "cheaper"))
        tie = mb.choose(b, dict(support=500), 20.0, dict(support=40), 20.0)
        self.assertEqual(tie["choice"], "expanded")
        fallback = mb.choose(b, dict(support=5000), 1.0, dict(support=40), 20.0)
        self.assertEqual((fallback["choice"], fallback["reason"]), ("factored", "only fit"))
        self.assertEqual(fallback["expanded_refusal"], dict(resource="support", need=5000, limit=1000))
        only_expanded = mb.choose(b, dict(support=900), 50.0, dict(support=2000), 1.0)
        self.assertEqual((only_expanded["choice"], only_expanded["reason"]), ("expanded", "only fit"))
        with self.assertRaises(AlgebraBudgetError) as ctx:   # the fallback disallowed: refused, not replaced
            mb.choose(b, dict(support=5000), 1.0, dict(support=40), 20.0, allow_factored=False)
        self.assertEqual(ctx.exception.resource, "support")
        with self.assertRaises(AlgebraBudgetError):          # neither fits
            mb.choose(b, dict(support=5000), 1.0, dict(support=4000), 20.0)
        with self.assertRaises(ValueError):
            mb.choose(b, dict(support=5), float("nan"), dict(support=4), 1.0)

    def _cells(self, seed, n, ncells, fraction=0.5):
        rng = np.random.default_rng(seed)
        cells = rng.integers(0, ncells, size=n, dtype=np.uint32)
        rows = rng.random(n) < fraction
        return cells, rows

    def test_fnd010_cell_count_prepass_refuses_before_output(self):
        for seed, n, ncells in ((1, 0, 10), (2, 1, 1), (3, 1000, 64), (4, 5000, 65), (5, 100_000, 3000),
                                (6, 70_000, 1 << 20)):
            cells, rows = self._cells(seed, n, ncells)
            exact_all = len(np.unique(cells))
            exact_sel = len(np.unique(cells[rows]))
            for backend in ("cpu", "cuda"):
                self.assertEqual(mb.count_cells(cells, ncells, backend=backend), exact_all, (seed, backend))
                self.assertEqual(mb.count_cells(cells, ncells, rows=rows, backend=backend), exact_sel, (seed, backend))
                if exact_sel:
                    at = mb.Budget(output=exact_sel)
                    self.assertEqual(mb.count_cells(cells, ncells, rows=rows, budget=at, backend=backend), exact_sel)
                    with self.assertRaises(AlgebraBudgetError) as ctx:
                        mb.count_cells(cells, ncells, rows=rows, budget=mb.Budget(output=exact_sel - 1), backend=backend)
                    self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", exact_sel))
        # the pre-pass bitmap itself is budgeted before it is allocated
        cells, _ = self._cells(7, 100, 1 << 30)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.count_cells(cells, 1 << 30, budget=mb.Budget(states=1 << 24))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", 1 << 30))
        need_bytes = ((1 << 20) + 63) // 64 * 8 + 8
        mb.count_cells(cells % (1 << 20), 1 << 20, budget=mb.Budget(bytes=need_bytes))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.count_cells(cells % (1 << 20), 1 << 20, budget=mb.Budget(bytes=need_bytes - 1))
        self.assertEqual(ctx.exception.resource, "bytes")
        # the device path also budgets its input copies: the same request needs more bytes there
        with self.assertRaises(AlgebraBudgetError):
            mb.count_cells(cells % (1 << 20), 1 << 20, budget=mb.Budget(bytes=need_bytes), backend="cuda")
        # an id outside the declared domain is invalid input, with no result
        bad = np.array([0, 5, 10], dtype=np.uint32)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(ValueError):
                mb.count_cells(bad, 10, backend=backend)
            self.assertEqual(mb.count_cells(bad, 10, rows=np.array([True, True, False]), backend=backend), 2)

    def test_fnd010_cuda_pass_equals_cpu_on_scrambled_threads(self):
        rng = np.random.default_rng(11)
        for trial in range(20):
            n = int(rng.integers(1, 20_000))
            ncells = int(rng.integers(1, 5000))
            cells = rng.integers(0, ncells, size=n, dtype=np.uint32)
            rows = rng.random(n) < rng.random()
            self.assertEqual(mb.count_cells(cells, ncells, rows=rows, backend="cuda"),
                             mb.count_cells(cells, ncells, rows=rows, backend="cpu"), trial)
        info = math_gpu.device()
        self.assertIn("emulated", info)

    def test_fnd010_existing_polynomial_limits_preserved_at_boundaries(self):
        kernel = mb.Budget.polynomial_kernel()
        self.assertEqual({k: kernel.limits[k] for k in ("variables", "degree", "support")},
                         dict(variables=8, degree=8, support=128))
        context = Context()
        xs = context.symbols(*(f"x{i}" for i in range(8)))
        # degree: x^8 is exact at the limit, x^9 is refused, as the budget module predicts
        self.assertEqual(((xs[0] ** 8).degree, (xs[0] ** 8).terms), (8, (((8,) + (0,) * 7, 1),)))
        self.assertEqual((xs[0] ** 8)(x0=3), 3 ** 8)
        mb.check(kernel, degree=8)
        with self.assertRaises(AlgebraBudgetError):
            xs[0] ** 9
        with self.assertRaises(AlgebraBudgetError):
            mb.check(kernel, degree=9)
        # support: prod(1 + x_i) over k variables has 2^k terms; k = 7 is exactly the limit, k = 8 is refused
        e = context.constant(1)
        for i in range(7):
            e = e * (1 + xs[i])
        self.assertEqual(e.term_count, 128)
        import itertools
        oracle = {tuple(bits) + (0,) for bits in itertools.product((0, 1), repeat=7)}
        self.assertEqual({exps for exps, coef in e.terms}, oracle)
        self.assertTrue(all(coef == 1 for _, coef in e.terms))
        mb.check(kernel, support=mb.power2(7))
        point = {f"x{i}": i + 2 for i in range(8)}
        self.assertEqual(e(**point), math.prod(i + 3 for i in range(7)))
        with self.assertRaises(AlgebraBudgetError):
            e * (1 + xs[7])
        with self.assertRaises(AlgebraBudgetError):
            mb.check(kernel, support=mb.power2(8))
        self.assertEqual(e.term_count, 128)                     # the refused product left its input unchanged
        # variables: 8 are allowed, the ninth is refused
        mb.check(kernel, variables=8)
        with self.assertRaises(AlgebraBudgetError):
            context.symbol("x8")
        with self.assertRaises(AlgebraBudgetError):
            mb.check(kernel, variables=9)

    def test_fnd010_histogram_and_boolean_domain_boundaries(self):
        b = mb.Budget(states=1 << 16)
        dims = [16, 64, 64]                                      # exactly 2^16 dense cells
        mb.check(b, states=mb.product(dims))
        with self.assertRaises(AlgebraBudgetError):
            mb.check(b, states=mb.product(dims + [2]))
        mb.check(b, states=mb.power2(16))                        # 16 Boolean predicates: 2^16 assignments
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.check(b, states=mb.power2(17))
        self.assertEqual(ctx.exception.need, 1 << 17)
        # a dense joint histogram past the budget can still run as its observed cells, decided explicitly
        rng = np.random.default_rng(12)
        cols = [rng.integers(0, d, size=50_000) for d in (200, 300, 400)]
        dense = mb.product([200, 300, 400])
        cell = (cols[0] * 300 + cols[1]) * 400 + cols[2]
        _, codes = np.unique(cell, return_inverse=True)
        observed = mb.count_cells(codes.astype(np.uint32), int(codes.max()) + 1, budget=mb.Budget(output=1 << 16))
        self.assertEqual(observed, len(np.unique(cell)))
        decision = mb.choose(mb.Budget(states=1 << 20), dict(states=dense), float(dense), dict(states=observed),
                             float(observed) * 4)
        self.assertEqual((decision["choice"], decision["reason"]), ("factored", "only fit"))


if __name__ == "__main__":
    unittest.main()
