# P4: shared, register and tuned in one comparison

Source experiment: `autotune-20261021-011927-125653/kernel-comparison`, 2026-10-21. RTX 4060 Laptop, M/N/K=16/256/128, strict FP32, Release, seed 42, 100 warmups, 10000 iterations, five rounds per pass. Two passes reversed implementation order; the figures below use the median of all ten round means. All eight original CSVs, including paired cuBLAS and correctness fields, are retained.

| Implementation | Parameters | Median microseconds | GFLOP/s |
|---|---|---:|---:|
| shared | 32 by 32 tile | 8.670 | 120.94 |
| register | 64 by 64 tile, TM=TN=4 | 8.402 | 124.80 |
| tuned baseline | BM/BN/BK/TM/TN=64/64/64/4/4 | 7.891 | 132.88 |
| tuned small tile | BM/BN/BK/TM/TN=16/32/32/2/2 | 7.355 | 142.57 |

The small-tile tuned version was 7.30% faster than the fixed tuned baseline in this separate comparison; the autotuning confirmation reported 13.27%. Shared and register were close, and their order changed in the second pass. These measurements do not establish a guaranteed ordering for short kernels. See [summary.json](summary.json) for both pass medians and all CSV references.

Shared, register and small-tile tuned used the [P4 winner source snapshot](../source-snapshot.zip), compiled with tuned parameters 16/32/32/2/2. The tuned baseline used the [earlier snapshot](../../p1-p3/source-snapshot.zip), compiled with 64/64/64/4/4. The historical register implementation differs from the current repository's register kernel; use the archives for a historical comparison. Source hashes remain in each CSV.

Follow the [P4 rebuild commands](../README.md), then run the resulting executable with `--kernel shared`, `--kernel register`, or `--kernel tuned`, retaining the same dimensions and timing options. Build the historical tuned baseline separately with its own snapshot and five parameters. Repeat the four measurements in reverse order for the second pass. This reproduces the method, not the exact reported values.
