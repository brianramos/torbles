# Certified sparse ideal reduction (GEOM-001)

`torbles/math_geom.h` (every contract; the dense graded rank of the checks and the work-item expansion are inline and
shared with CUDA), `math_geom.c`, `math_geom_gpu.cu`, `math_geom.py` (opt-in). Relations proven on data (FND-007
certificates, e.g. DISC-002 identities) form an ideal; a query expression f is reduced to a normal form g with
f - g = sum h_i r_i exactly, the certificate of that equation is derived and checked by FND-007, and the cheaper of
the equivalent forms is chosen on measured costs. g equals f only where the relations hold.

```python
from torbles import math_ident as mi, math_geom as mg
rels = [i["certificate"] for i in mi.discover(scope, ["a", "s", "z", "x", "y"], degree=2).discoveries()]
I = mg.ideal(rels, order="lex", variables=["a", "s", "z"])     # reduced Groebner basis (complete=False: the relations)
I.basis; I.cofactors(0); I.info                                 # basis[j] = sum_i cofactors(j)[i] r_i, exactly
f = mg.poly({(1, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): -3}, ["a", "s", "z", "x", "y"])     # a s - 3 z
r = I.reduce(f)                      # r.g = 3x^2 - 7x - 7y, r.cofactors, r.info (steps, denominator D, claimable)
c = r.certify()                      # FND-007 derived certificate: P = D (f - g) = sum D h_i r_i in Z[x]
c.authorize_exact(scope)             # the relations' own scope only (a hypothesis relation never authorizes)
P, cofactors = r.claims()            # the certificate's parts as math_certificates claims
mg.check([r, (P, rels, cofactors)], backend="cuda")             # batched exact checks: holds, first nonzero monomial
mg.agree(scope, [(f, r.g)], backend="cuda")                     # f and g on every selected row: disagreements, bits
mg.cost(scope, r.g); mg.choose(scope, [f, r.g], priority="bits", measure=True)
mg.derivative(f, "x"); mg.derivative(f, "x", through=True, dependencies={"y": mg.poly({(2,): 1}, ["x"])})
mg.dependency_check(scope, {"y": mg.poly({(2,): 1}, ["x"])})   # a declaration verified on every row
```

## Relations, orders, ideals

- Relations are FND-007 certificates of POLYNOMIAL claims (DISC-002) or AFFINE / SUM / PRODUCT claims (DISC-001),
  read as FND-007 reads them (out - scale x - bias, out - x - y, out - x y): snapshot, hypothesis or derived
  certificates of one NULL policy, at most 16 (FND-007's premises) over at most 8 columns, recorded on one table (or
  array scope) through one column layout (checked natively on their FND-007 records). Laws and schema facts,
  tolerance or GF(p) claims, mixed policies, and relations of different tables or layouts are `DomainError`.
- Orders: `lex`, `deglex`, `degrevlex` over `variables` (most significant first); the other columns follow in scope
  order. `lex` with the defined columns first eliminates them (expresses f in the free columns); a degree order
  rewrites high-degree monomials through the relations (the generator form).
- `complete=True`: Buchberger's algorithm (normal selection, product criterion) then interreduction: the reduced
  Groebner basis (monic, increasing leading monomial), each element with its cofactors over the relations. The normal
  form is then unique. `complete=False`: division by the monic relations in their order.

## Exactness

- Coefficients are canonical int64 rationals throughout; a value past int64 is `OverflowError`, never rounded.
- `reduce`: f = sum q_j g_j + g with no term of g divisible by a leading monomial; h_i = sum_j q_j A_ji, so
  f - g = sum h_i r_i holds in Q[x]. The certificate P = D (f - g), cofactors D h_i (D the lcm of every denominator)
  over the claim columns, is checked by FND-007's `tr_cert_derive` (P - sum D h_i r_i = 0 in Z[x]); its kind is the
  weakest relation's and its domain theirs: a snapshot relation makes g equal to f only on that snapshot's rows,
  stale with its table, refused on any other scope. A certificate past int64 or 4096 terms has no FND-007 claim
  (`info["claimable"]` false, `certify` raises).
- `check`: certificates P = sum q_i G_i decided by expanding every product into a dense graded index with signed
  128-bit sums; the host proves sum |P| + sum |q| |G| <= 2^127 - 1 per certificate, else it is undecided
  (`decided` false, `holds` false), never wrapped. `first` is the smallest nonzero monomial in the graded order
  (`unrank`). It catches a wrong coefficient, a wrong or missing premise, a swapped premise order.
- `agree`: F = D_f f, G = D_g g and P = D (f - g) evaluated exactly on every selected row (DISC-002's proven W-word
  evaluation; a row where a column of f or g is NULL is excluded): checked, excluded, disagreements with the smallest
  disagreeing row, and the largest bit lengths of |F| and |G| (the actual widths).
- `cost`: terms, columns read, degree, naive multiplications, the bound bits of D p on the scope's extremes, D;
  `choose` orders by the declared priority (columns, bits or terms first, then the others in that order, ties to the
  earliest) on measured bits with `measure=True` (which also reports whether each candidate agrees with the first on
  every row). A reduction is not always cheaper: (x + 1)^6 from y^6 is a term explosion and is not chosen.

## Derivatives and declared dependencies

An identity on the rows (y = x^2 there) does not make y a function of x: `derivative(f, "x")` is the formal partial
derivative of f as written (no declarations may be passed), so equivalent forms y and x^2 have different formal
derivatives. A derivative through relations (`through=True`) needs declared dependencies {v: phi}: the total
derivative df/du + sum df/dv dv/du through the declared chain, which equivalent forms agree on; none declared is
`DomainError` with `.reason == "undeclared"`, d/dv of a declared dependent column `"dependent"`, a cyclic declaration
`"cycle"`. `dependency_check` verifies each declaration D v = D phi on every selected row (FND-007 evidence).

## CUDA

`tr_mgpu_geom_agree` (work items over pair chunks and row slices; local counters, one atomic each, atomic minimum of
the witness and maxima of the bit lengths) and `tr_mgpu_geom_check` (one work item per product into the
certificate's dense slots with the two-word atomic, then a scan counting nonzero slots and their minimum) give the
CPU's results bit for bit. Results name `cuda` or `cuda-emulated` and the device.

## Budgets and refusals

Checked before allocation or, inside Buchberger, as reached: iterations (S-pairs plus division steps), support (terms
of any polynomial or cofactor), degree, output (basis elements; pairs; certificates), states (dense slots of a check
batch), bytes. Past a limit the request is refused (`AlgebraBudgetError` naming the resource, the need reached and the
limit), never truncated. More than 128 basis elements, 16 relations, 8 variables, 256 pairs or 256 certificates per
batch, an exponent past 255 or a dense index past 2^40 slots is `UnsupportedError`.

## Limitations

- Coefficients are int64 rationals (no arbitrary precision): a reduction whose coefficients grow past int64 is
  refused. Completion is Buchberger's with the product criterion only (no F4/F5); lex bases can be large.
- The CUDA kernels cover agreement and certificate checks; completion and reduction run on the CPU.
- Costs are bounds and measured widths on one scope; they are not a query planner's cost model.
