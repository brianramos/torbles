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
| Base payload | Gather supported narrow, non-null integer and Boolean codes from the source GPU bitplanes after the final slice | Download the selected codes and decode SQL values |
| Computed projections and general QUALIFY | Existing SQL expression evaluator | Host |
| Derived grouped reductions | COUNT and SUM over one narrow non-null stored grouping key | Download group states |
| Shared-CTE self-join | Count an inner equality join over a common unique global rank by intersecting two comparison masks | Download one count |
| Shared-CTE set operations | UNION DISTINCT, INTERSECT and EXCEPT over a common unique global rank via packed masks | Download only selected ranks |
| Shared CTE reuse | Two filtered COUNT branches reuse one resident window build | Download two counts |
| General result reuse across SQL blocks | Supported consumers retain a statement-owned CUDA temporary while reading its source | Other grouped, join, set and CTE shapes use the existing executor |

A single comparison can avoid allocating the value planes. Set
`TORBLES_WINDOW_PLANES=1` to exercise the packed comparator with one predicate.
The two-comparison path can use the same planes for both tests. Signed prefixes
use a checked range, and wider or unsupported SQL shapes retain the existing
executor. The packed path may use a scalar CUDA predicate when the comparison
is constant over the proven range or its allocation would exceed the budget.

`cuda_window_download_bytes` counts transferred window values and row IDs.
It excludes source upload, payload gathering and scalar control counters.


## Measurements

Warm end-to-end medians at 4M input rows on an RTX 5090 Laptop GPU.
Each result was verified against DuckDB. The relational comparison uses
`21e40d7` as the baseline and `226b196` as the candidate.

| Stage | Prior Torbles | Resident CUDA | DuckDB |
|---|---:|---:|---:|
| Gather three payload columns for 17 rows | 6.94 ms | 6.82 ms | 48.04 ms |
| Grouped rank COUNT and SUM | 1,059.22 ms | 7.57 ms | 33.66 ms |
| Shared unique-rank self-join count | 99.73 ms | 6.85 ms | 35.15 ms |
| Shared unique-rank intersection | 108.03 ms | 7.88 ms | 34.43 ms |
| Two counts from one shared rank CTE | 101.73 ms | 6.66 ms | 33.35 ms |

The dense payload case took 73.39 ms on the GPU gather path and 71.35 ms
with the prior host gather. Each transferred a full result. The resident
join and shared counts transferred 8 and 16 bytes of result data.
