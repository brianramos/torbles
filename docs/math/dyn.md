# Closed-observable dynamics and operator powers (DYN-001, DYN-002)

`torbles/math_dyn.h` (every contract; the residue products, the batched row evaluator and the device needs are inline
and shared with CUDA), `math_dyn.c`, `math_dyn_gpu.cu`, `math_dyn.py` (opt-in). DYN-001 represents declared dynamics
x_{t+1} = F(x_t) as an exact matrix action on a polynomial basis proven closed under F, and evolves retained moment
states without touching rows. DYN-002 takes many steps from that finite operator: exact powers, semigroup checks,
recurrences, parameterized evolution, and floating powers under a rigorous contract. Built on SYM-001 (`sym.md`): its
declared geometry, bases, exact affine maps, action matrices, retained states and 512-bit request evaluator.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_sym as ms, math_dyn as dyn
geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})            # SYM-001: the only coordinates
B = ms.basis(geo, degree=2)                                         # or terms=["1", "x", "x^2", "y"]
F = dyn.affine(geo, ms.compose(ms.rotate("x", "y", cos=Fraction(3, 5), sin=Fraction(4, 5)), ms.translate(x=1)))
P = dyn.polynomial(geo, {"x": {"x": Fraction(1, 2)}, "y": {"y": Fraction(1, 3), "x^2": 1}})   # value units
K = dyn.operator(B, F)                    # exact K (SYM-001's action); dyn.NotClosedError(.term, .missing)
Kt = dyn.operator(B, P, truncate=dyn.box(geo, {"x": (-3, 3), "y": (-3, 3)}))   # certified approximate
K.matrix(); K.numerators(); K.info; K.epsilon(); K.box; K.authorize_exact(); K.at(theta); K.apply(st, backend="cuda")
st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)  # or ms.state(mt.moments(...), geo)
ev = dyn.evolve(st, K, 1000, method="auto")   # "stepping" | "squaring" | "modular"; backend="cuda" (modular)
ev.moments(g); ev.errors(g) (truncated); ev.at(theta) (parametric); ev.group(g); ev.info; ev.authorize_exact()
dyn.evolve_rows(scope, F, B, 10, weight=None)  # the reference: every row evolved exactly, then its moments
Kn = K.power(n, method="auto", backend="cpu" | "cuda"); Kn.matrix(); Kn.compose(Km); Kn.equal(Km)
dyn.semigroup(K, m, n); K.charpoly().coefficients
rec = K.recurrence({"x^2": 1}, state=st, group=0, kind="minimal"); rec.coefficients; rec.initial; rec.term(10**6)
dyn.mixture([(Fraction(1, 3), K1), (Fraction(2, 3), K2)])
K.float_power(1000, tolerance=1e-9, norm="abs" | "rel", state=None)   # FloatPower(values, bounds) or UnstableError
dyn.need("power" | "evolve" | "float" | "charpoly" | "recurrence" | "apply", K, steps=n, state=st, method=, backend=)
```

## DYN-001: dynamics, closure, transfer operators

- Dynamics act on the n declared coordinates of a SYM-001 geometry, in logical units (value = logical / 10^scale).
  `affine(geo, T)`: an exact SYM-001 map (translation, scaling, reflection, permutation, affine, rotation by an exact
  rational point of the unit circle or a multiple of 90 degrees, compositions); interval rotations and value
  permutations are refused (`UnsupportedError`). `polynomial(geo, {c: {term: coefficient}})`: x'_c = sum of the terms
  (exact int64 rationals in value units, total degree <= 8, undeclared coordinates fixed), converted exactly to
  logical units (c 10^(s_c - sum_j s_j e_j)), merged, one int128 denominator per coordinate; with `parameter="a"`
  terms may carry a^p (p <= 8). Only numeric coordinates are read or written (`DomainError`, as SYM-001's metric
  operations).
- Closure, decided exactly and completely: the basis is closed when every phi_a(F(x)) = prod_i F_i(x)^(a_i) lies in its
  span as a polynomial. Affine maps take SYM-001's `tr_sym_action_make` (its closure decision, witness and canonical
  matrix); polynomial maps are expanded here in int128 (every product and sum checked, `OverflowError` past it). The
  first basis term whose image leaves the span is refused with `NotClosedError` (`.term`, `.missing`, image degrees up
  to 64). Nonlinear maps may have closed bases (x' = x / 2, y' = y / 3 + x^2 on {1, x, x^2, y}): exact.
- The transfer operator K: phi(F(x)) = K phi(x) for every point x, hence moments(F(rows)) = K moments(rows) and
  moments_t = K^t moments_0 for any weights. Exact operators are int128 numerators over one canonical denominator D
  (parametric: K(a) = sum_j a^j N_j / D). `authorize_exact` returns the FND-007 decision of a law (a polynomial
  identity over Q).
- Truncation (only on request, with a contract): `truncate=box(geo, {c: (lo, hi)})`, every coordinate the basis'
  evolution reads bounded (else `DomainError`). K^ keeps the terms of each image that lie in the basis; for every x in
  the box, |phi_a(F(x)) - (K^ phi(x))_a| <= eps_a = sum over the dropped terms |c| prod_i max(|lo_i|, |hi_i|)^(b_i)
  (exact FND-009 rationals, `epsilon()`). The box is checked F-invariant by exact rational interval arithmetic
  (`info["invariant"]`). Truncated operators are approximate (label `certified approximate`, UNC-002's label value 5):
  `authorize_exact` raises `NotAuthorizedError` (reason `approximate`), and every exact use (powers, charpoly,
  recurrences, mixtures, substitution, application, CUDA, semigroups) is refused (`DomainError`): uncertainty is never
  dropped. A truncation of a closed basis is exact. Parametric maps cannot be truncated (`UnsupportedError`).
- Evolution of retained states (SYM-001: FND-004 banks, TENSOR-001 moment tensors; the state of the operator's scope,
  basis and geometry, else `DomainError`; an abstract geometry has no state): m_t = K^t m_0 per group as integer
  numerators N^t m_0 over D^t (the same integers by every method): `stepping` (t big-integer products v <- N v),
  `squaring` (N^t, then N^t m_0), `modular` (residues modulo primes of LIN-002 in [2^61, 2^62), Chinese remaindering
  within the proven bound |N^t m_0| <= ||N||_inf^t max|m_0| and a check modulo one more prime), `auto` (the smallest
  estimated cost: each method's structural work -- loop steps, the nonzero products of N's pattern, limb products from
  the proven sizes, residues, Garner steps, primes -- times unit costs fitted to timed runs; the estimates, in 1/16 ns,
  are in `info`). A truncated evolution also returns the certified
  componentwise bound E_t (E_0 = 0, E_t = |K^| E_(t-1) + W eps, W the group's contributing rows):
  |moments_t(rows) - m^_t| <= E_t, under checked premises: the state unweighted (`UnsupportedError`), the state's
  scope given with every bounded coordinate's selected extremes inside the box (`DomainError`), and at most one step
  unless the box is F-invariant (`DomainError`). Truncated evolutions step or square (`modular`: `UnsupportedError`).
- `evolve_rows` (the reference): every selected row whose evolved coordinates (and weight) are present is evolved
  exactly t steps with one common denominator (x_t = X_t / Q_t, Q_(t+1) = L Q_t^d, d the map's degree, L = lcm of the
  denominators), then its moments summed per group (numerators over Q_t^|a|).

## DYN-002: powers, semigroups, recurrences, parameters, floating powers

- `power(n)`: K^n as integer numerators N^n over D^n (K^0 = I over 1), by exact repeated squaring (the binary method,
  every intermediate bounded first) or modularly (N^n mod p per prime, CRT within |N^n|_max <= ||N||_inf^n, the check
  prime). `compose` multiplies (numerators and denominators), `equal` compares exactly by cross products;
  `semigroup(K, m, n)` verifies K^0 = I, K^1 = K and K^m K^n = K^(m+n).
- `charpoly()`: det(x I - K) = sum_k a_k D^(k - nb) x^k from chi_N(y) = sum_k a_k y^k computed modulo primes (Hessenberg
  reduction and its recurrence, O(nb^3) per prime), reconstructed within |a_(nb-k)| <= C(nb, k) ||N||_inf^k (Hadamard)
  and checked modulo a further prime.
- `recurrence(c, state=, group= | vector=, kind=)`: s_t = c^T K^t v (c: int64 over the basis, v: a group's moments or
  int128 values) satisfies s_(t+d) = sum_k r_k s_(t+k). `charpoly`: d = nb (Cayley-Hamilton). `minimal`: Berlekamp-
  Massey over Q (FND-009 rationals) on u_t = c^T N^t v for t < 2 nb (the linear complexity is <= nb, so these terms
  determine it); its coefficients are integers (a monic divisor of chi_N, Gauss), checked, and the recurrence is
  verified on every computed term. `term(T)`: s_T = u_T / D^T with u_T = sum_k w_k u_k, w = x^T mod the recurrence's
  polynomial by square and multiply in Z[x] (O(d^2 log T)).
- Parameterized evolution: `mixture` (sum_j w_j K_j, exact weights >= 0 summing to 1, one basis and geometry: the
  expected moments of a dynamics choosing F_j with probability w_j independently at each step); a parametric operator
  evolves the moments as polynomials in the parameter (degree t p, stepping), `ev.at(a)` substitutes an exact rational
  (equal to `evolve(st, K.at(a), t)`).
- `float_power(n, tolerance, norm)` (APPROXIMATE, certified): K^n (or K^n m_g with `state`) by repeated squaring in
  outward-rounded binary64 interval arithmetic (FND-009) from tight enclosures of N / D; every value comes with a
  rigorous bound |exact - value| <= bound. The tolerance (`abs`: the largest bound; `rel`: over the largest |value|) is
  the contract: a bound past it, or a non-finite one, is unstable and refused (`UnstableError`, a `NotConvergedError`,
  with `.info`: the exponent reached within the tolerance, the error, the magnitude), never returned.

## CUDA

- `K.apply(st, backend="cuda")` (`tr_mgpu_dyn_apply`): one work item per (operator, group, basis row) runs
  `tr_dyn_apply_entry` (SYM-001's 512-bit `tr_sym_row_exact` on int128 numerators and moments), one output slot each.
- `K.power(n, backend="cuda")` (`tr_mgpu_dyn_powmod`) and `evolve(..., backend="cuda")` (`tr_mgpu_dyn_evolvemod`): the
  numerators reduced on the device (one work item per (operator, prime, entry)), the binary exponentiation driven by
  the host with one launch per product (one work item per (operator, prime, entry) running `tr_dyn_mulmod_entry`),
  then the moments reduced and multiplied (one work item per (operator, prime, group, row)). The host adopts the
  residues only after Chinese remaindering within the proven bound and the check prime (`tr_dyn_power_adopt`,
  `tr_dyn_evolution_adopt`: a corrupted residue is `ValueError`). Every slot has one writer: the results equal the
  CPU references (`tr_dyn_apply_view_eval`, `tr_dyn_powmod_view_eval`, `tr_dyn_evolvemod_view_eval`) bit for bit;
  views batch several operators.

## Budgets and refusals

| request | needs |
|---|---|
| dynamics | variables, degree, support = terms, bytes |
| operator | variables, degree = the image degree, support = nb, tensor = (p + 1) nb^2, states = the largest expansion, bytes |
| power | support, tensor = nb^2, iterations = n (the step count), output = nb^2, bytes = every number from its proven bound (n bitlen(norm) bits per numerator, n bitlen(D) for D^n) and the scratch (modular: the residues) |
| evolution | iterations = t, output = groups x nb (x (t p + 1) parametric, x 2 truncated), states, degree = t p, bytes as powers |
| row evolution | iterations = t, output, bytes from the a priori bounds of X_t and Q_t |
| charpoly, recurrence, term | support, iterations = terms (term: T), output, bytes |
| floating power | tensor, iterations = n, output, bytes |
| CUDA | bytes = every device copy (`tr_dyn_apply_device_bytes`, `tr_dyn_mod_device_bytes`), checked before any copy |

Refusals raise before allocation: `AlgebraBudgetError` (`.resource`, `.need`, `.limit`; `dyn.need` reports every
need, so a request is admitted at its need and refused one below it), `OverflowError` (an int128 map, operator or
mixture entry), `UnsupportedError` (representation limits: 32 coordinates, map degree 8, parameter degree 8, 4,096
terms, 64 mixture operators, 4,096 primes; interval or value maps; weighted or parametric truncation; parametric
powers; truncated modular evolution), `DomainError` (`NotClosedError`; undeclared or categorical coordinates; a state
of another scope, basis or geometry; an approximate operator in an exact use; a contract that does not hold),
`NotAuthorizedError` (an approximate operator or evolution asked to authorize an exact rewrite), `UnstableError`,
`ValueError` (malformed input, inconsistent residues), `TypeError` (a float parameter).

## Limitations

Exact operators are int128 numerators over an int128 denominator (SYM-001's limit); powers and evolutions use big
integers (schoolbook products, no Karatsuba) bounded a priori by ||N||_inf^n, which can be far above the actual sizes
(cancellations are not foreseen); the native row evolution of a nonlinear map shares one denominator Q_(t+1) = L Q_t^d,
so its proven bound grows doubly exponentially and long nonlinear row evolutions are refused (the closed operator's
grows linearly); the truncation bound is a worst case over the box (often far above the actual error) and needs an
unweighted state; minimal recurrences use Berlekamp-Massey over FND-009 rationals (quadratic in nb, single-threaded);
parametric evolutions step; floating powers are binary64 intervals (no multiprecision); the CRT, the
characteristic polynomial and every big-integer product run on the CPU, single-threaded. On the CPU, Garner's
reconstruction (O(k^2) per value for k primes) and the residue products' loop over every (i, j, k) per prime make the
modular method the slower one except for dense operators at large exponents with few groups; `auto`'s unit costs were
fitted on one machine (2.1 GHz Xeon) and are estimates: over the calibration runs (the benchmark's operators and four
dense ones, 10 to 6,000 steps) it took the fastest method or one within 1.7x of it.
