# CUDA window temporaries

The window pipeline starts with the fact's raw bitplanes and visible-row mask.
CUDA orders the active rows, calculates the rank or prefix sum, then scatters
the values into physical row order. The temporary is private to the statement.

| Consumer | Current lowering | Boundary |
|---|---|---|
| One comparison to a signed integer literal | Compare the device window values and compact a packed mask | Download only selected IDs and values |
| Two comparisons joined by AND | Rebase the signed range, transpose into packed value planes, and intersect the two comparison masks | Download only selected IDs and values |
| ORDER BY the window value | Stable device sort, preserving physical order within ties | Download the requested slice |
| Literal LIMIT and OFFSET | Device slice in the chosen order | Download the requested slice |
| Simple filter above an unshared derived window | Move the comparison to the window block as QUALIFY | Materialize only survivors |
| Scalar COUNT, MIN, MAX and SUM of an eligible derived window | Device reduction over the active window values and any proven qualifier | Download one scalar |
| Full primitive integer projection | Download typed columns with lazy row conversion | Use `fetch_columns()` for NumPy arrays or `fetchall()` for Python tuples |
| Base payload | Existing gather by the final row IDs | Host decode of selected payload |
| Computed projections and general QUALIFY | Existing SQL expression evaluator | Host |
| Grouped reductions, joins, set operations, shared CTEs | Existing relational executor | Host |
| Result reuse across SQL blocks | Simple derived predicates and scalar aggregates can fuse into the producer | General reuse needs an owned device relation with snapshot and lifetime rules |

A single comparison can avoid allocating the value planes. Set
`TORBLES_WINDOW_PLANES=1` to exercise the packed comparator with one predicate.
The two-comparison path can use the same planes for both tests. Signed prefixes
use a checked range, and wider or unsupported SQL shapes retain the existing
executor. The packed path may use a scalar CUDA predicate when the comparison
is constant over the proven range or its allocation would exceed the budget.

`cuda_window_download_bytes` counts transferred window values and row IDs.
It excludes source upload, payload gathering and scalar control counters.
