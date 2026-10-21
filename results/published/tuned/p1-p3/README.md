# P1-P3 tuned search, 2026-10-21

Source experiment: `autotune-20261021-010747-921122`. This was the initial 37-candidate search. Parameter order: **BM / BN / BK / TM / TN**. GPU: RTX 4060 Laptop, SM89; CUDA compiler NVIDIA 13.2.78; host compiler MSVC 19.44.35217.0. Runtime/driver/cuBLAS version fields are retained exactly in [manifest.json](manifest.json), rather than inferred from the compiler version.

| Case | M / N / K | Winner | Median ms | GFLOP/s | Paired cuBLAS median ms |
|---|---|---|---:|---:|---:|
| P1 | 2048 / 2048 / 2048 | 64 / 128 / 32 / 4 / 4 | 3.059012 | 5616.15 | 2.477637 |
| P2 | 16 / 4096 / 4096 | 32 / 64 / 32 / 4 / 4 | 0.403416 | 1330.81 | 0.315721 |
| P3 | 1001 / 769 / 513 | 64 / 64 / 16 / 4 / 4 | 0.192593 | 4100.78 | 0.151913 |

Release, strict FP32, seed 42. Screening used three rounds. Confirmation used two passes in reversed task order, 100 warmups, five rounds per pass, and 44 / 335 / 800 iterations per round for P1 / P2 / P3. The copied [matrix.csv](matrix.csv) contains the original requested warmup=20; the confirmation policy raises it to 100, as the retained CSVs show. Rankings pool ten round means and take their median, not the median of two pass medians.

Every candidate passed the 32-case correctness suite plus wide, tall and tile-tail checks before timing. Every timed shape was validated against full-output cuBLAS and the CPU double reference (sampled for large shapes; see `cpu_scope`). [rankings.json](rankings.json) links all 36 retained confirmation CSVs for six finalists across three shapes. [winners.json](winners.json) contains the winner source fingerprints and CMake parameters. [manifest.json](manifest.json) retains the confirmed variants' compiler register/spill counts and source identity.

The workspace's matrix, search space and script changed during the original run; these results belong to the initial immutable snapshot. The current tuned kernel matched that snapshot at publication, while the current register kernel and build/CLI infrastructure did not. See [provenance.json](provenance.json).

To rebuild the historical P1 winner from the repository root in an x64 Visual Studio developer terminal:

```bat
python -m zipfile -e results\published\tuned\p1-p3\source-snapshot.zip results\reproduce-tuned-p1
cmake -S results\reproduce-tuned-p1 -B build\reproduce-tuned-p1 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_TUNED_BM=64 -DGEMM_TUNED_BN=128 -DGEMM_TUNED_BK=32 -DGEMM_TUNED_TM=4 -DGEMM_TUNED_TN=4
cmake --build build\reproduce-tuned-p1 --target gemm
build\reproduce-tuned-p1\gemm.exe --mode bench --kernel tuned --m 2048 --n 2048 --k 2048 --seed 42 --warmup 100 --iters 44 --rounds 5
```

For P2/P3, use their five parameters and dimensions above, with 335/800 iterations. A single invocation produces five new round means, not the original ten-round aggregate. To repeat the search on historical source, use the extracted `scripts/autotune.py --cases P1 P2 P3 --arch 89`. Recompiled fingerprints may change with compiler version or build options; check the preserved individual source hashes before comparing builds.
