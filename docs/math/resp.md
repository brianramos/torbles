# Piecewise absolute and hinge loss curves (RESP-002)

`torbles/math_resp.h` (every contract; the evaluator is inline and shared with CUDA), `math_resp.c`,
`math_resp_gpu.cu`, `math_resp.py` (opt-in). The complete loss functions of a parameter theta over the sorted
support and prefix moments of a DIST-002 prefix state ([dist.md](dist.md)): exact values and one-sided slopes at
and between every breakpoint, the piecewise polynomial pieces, and the minimizer set with a stated tie convention.

```python
from fractions import Fraction
from torbles import math_bank as mbk, math_dist as md, math_resp as mr
p = md.histogram(scope, ["x"], weight="w").prefix("weight", degree=2)   # or prefix("count", 2)
loss = mr.Loss(p, "abs")         # "hinge", "hinge_down", "sqhinge", "sqhinge_down", "squared", "pinball" (tau=)
loss.evaluate([0, Fraction(7, 2)], group=g)            # [{"value", "left", "right", "at_breakpoint", "index"}]
loss.evaluate(thetas, group=g, backend="cuda")         # one CUDA work item per theta, the same answers
loss.pieces(group=g)             # [{"lo", "hi", "coefficients": (c0, c1, c2)}]: the complete function
loss.minimizers(group=g)         # {"kind", "lo", "hi", "point", "value", "lo_index", "hi_index"}
mr.evaluate_arrays(p, mr.requests("hinge", thetas, group=g), backend="cpu")    # raw RESULT_DTYPE answers
```

## Losses

Over the support values x with masses m (rows, or nonnegative weight sums; NULL values are not in the support):

| loss | value | left slope | right slope |
|---|---|---|---|
| `abs` | sum m abs(x - theta) | M_< - M_>= | M_<= - M_> |
| `hinge` | sum m max(0, x - theta) | -M_>= | -M_> |
| `hinge_down` | sum m max(0, theta - x) | M_< | M_<= |
| `sqhinge` | sum m max(0, x - theta)^2 | -2 sum m max(0, x - theta) | (equal) |
| `sqhinge_down` | sum m max(0, theta - x)^2 | 2 sum m max(0, theta - x) | (equal) |
| `squared` | sum m (x - theta)^2 | 2 (theta M - S1) | (equal) |
| `pinball` (tau) | tau hinge + (1 - tau) hinge_down | (1 - tau) M_< - tau M_>= | (1 - tau) M_<= - tau M_> |

M_<, M_<=, M_>, M_>= are the masses below, at most, above and at least theta. `abs`, the hinges and `pinball` need a
prefix of degree >= 1, the squared forms degree >= 2 (else `UnsupportedError`). theta is an exact rational (int,
Fraction, Decimal; floats are refused) in the column's logical units (a DECIMAL(p, s) value is logical / 10^s);
values and slopes are exact Fractions in logical units times the mass's units.

## Exactness

theta = num / den with int64 parts. Each answer is computed natively from two binary searches and prefix
differences: numerators in 256-bit two's complement (exact products of 128-bit moments with theta's parts),
denominators den (abs, hinges), den^2 (squared forms), den x tau's denominator (pinball). The prefix proof of
DIST-002 bounds every moment of a subset by 2^127 - 1; a numerator that would not fit 256 bits is refused for its
request (`OverflowError`); for states built from at most 2^32 - 1 rows this cannot occur. CPU and CUDA run the same
inline evaluator and agree bit for bit.

## The complete function

`pieces(group)`: d + 1 closed pieces (-inf, x_1], [x_1, x_2], ..., [x_d, +inf) for d support values (one piece
(-inf, +inf) for `squared` or an empty support), each with loss(theta) = c0 + c1 theta + c2 theta^2 exactly on the
piece (pinball coefficients are rationals with tau's denominator). Adjacent pieces agree at their shared breakpoint
(the losses are continuous). Output budget: the pieces.

## Minimizers and the tie convention

The whole minimizer set is reported as a closed interval [lo, hi] (None for an unbounded end); its canonical point
is lo when bounded below, else hi; `value` is the minimum (the loss at the canonical point). With the expanded order
statistics y_1 <= ... <= y_M (each x repeated m times):

- `abs`: [y_ceil(M/2), y_floor(M/2)+1], the median interval: a `point` when M is odd or the middle values tie, an
  `interval` when the bottom is flat (e.g. values 1, 3: every theta in [1, 3]).
- `pinball`: [y_ceil(tau M), y_floor(tau M)+1] with y_0 = -inf and y_{M+1} = +inf: an interval exactly when tau M is
  an integer between two distinct values; tau = 0 gives the ray (-inf, x_1] (`down`), tau = 1 the ray [x_d, +inf)
  (`up`).
- `hinge`, `sqhinge`: [x_d, +inf) (`up`, minimum 0); `hinge_down`, `sqhinge_down`: (-inf, x_1] (`down`, minimum 0).
- `squared`: the point S1 / M (the mean), minimum (M S2 - S1^2) / M.
- Empty input (an absent group, a group of NULL values only, only zero weights): the loss is 0 everywhere; the set
  is (-inf, +inf) (`all`) without a canonical point. Duplicate observations are one support value with their mass.

## Budgets and refusals

Batches: output = thetas; CUDA also bytes = the device copies of the requests and answers (48 + 144 bytes each), and
the prefix state's upload once (`Prefix.device`). Pieces: output = pieces. `AlgebraBudgetError` before anything is
computed; `ValueError` (an invalid loss, tau outside [0, 1] or given to another loss, a group out of range, den 0
in a raw request: that request's status), `TypeError` (a float theta, a state that is not a `Prefix`),
`OverflowError` (a part outside int64), `UnsupportedError` (a squared form over a degree-1 prefix).
