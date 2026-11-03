# In-memory bitplane refresh prototype

torbles.refresh.RefreshTable is an experimental API for fixed-schema nonnullable uint32 columns. Each table starts with implicit logical IDs from zero. Append assigns new IDs and returns their range. Update replaces specified columns and preserves the logical IDs. Delete clears visibility. Each operation holds the table lock and rejects invalid IDs before making visible changes.

Appends and replacements build immutable native bitplane segments. A filtered read intersects the predicate mask with the segment's visibility mask. Exact count, sum and grouped sum combine the visible reductions across segments. Unfiltered count reads maintained live-row metadata. The CUDA executor uploads a changed visibility bitmap and a new segment when a query first needs them.

Compaction rebuilds live rows into one native segment and preserves logical IDs. If deleted IDs leave holes, the compacted segment currently retains an eight-byte ID per live row. The stats property reports native torus bytes, visible and physical rows, visibility bitmap bytes, ID-array bytes, sparse override entries and the uncompressed value bytes rewritten by the most recent compaction. These counters cover named structures rather than total process memory.

The separate native at_xor_words primitive validates a batch and applies masked XOR deltas to stored value and parity planes. The benchmark invokes it on an isolated raw table. Segment refresh uses visibility and replacements, so direct XOR edits do not share its snapshot, lookup, or GPU-cache semantics.

The prototype runs in memory through the Python API. Each table serializes its own operations. SQL DML, durable commits, crash recovery and atomic transactions across tables remain outside its scope.

## Benchmark

Run from the repository root after building the CPU and CUDA libraries.

    PYTHONPATH=. python bench/benchmark_refresh.py --output /tmp/torbles-refresh.json

The workload models a TPC-DS-style refresh and includes explicit fact and dimension updates. A cycle appends a fact batch, replaces fact measures, deletes a different fact batch, and updates a separate dimension store. DuckDB uses in-memory indexed IDs and the same data and mutations. The join check groups fact sums by dimension ID and applies the dimension filter to those results. It also verifies a filtered count and a selective sum after every cycle, then compares before and after compaction. The JSON records first calls and repeated warm reads separately. Results describe this workload and fixture.