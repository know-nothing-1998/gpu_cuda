# P4 small-matrix tuned search, 2026-10-21

Source experiment: `autotune-20261021-011927-125653`. P4 is **M=16, N=256, K=128**. Parameter order: **BM / BN / BK / TM / TN**. The 43-candidate search added six small-tile configurations to the previous 37; P1-P3 were not rerun for those six additions.

The winner **16 / 32 / 32 / 2 / 2** reached **141.64 GFLOP/s**, median **0.007402898 ms**. The fixed tuned baseline reached **125.05 GFLOP/s**; winner throughput was 1.133 times that baseline. Paired cuBLAS median time was **0.006704157 ms**, making winner throughput 90.56% of paired cuBLAS. See [ranking.csv](ranking.csv) and [winners.json](winners.json).

RTX 4060 Laptop, SM89, Release, strict FP32 cuBLAS, seed 42. Confirmation used 100 warmups, 10000 iterations per round, five rounds in each of two passes with reversed task order. Rankings take the median of all ten round means. All six confirmation CSVs for three finalists are retained, with validation results and paired cuBLAS. The original [matrix.csv](matrix.csv) lists four cases and warmup=20; this run selected only P4 and raised confirmation warmup to 100.

Thirty-seven binaries were reused from the preceding search after source compatibility checks, then checked and timed again; six were built from this run's new snapshot. Reused variants preserve the original binary/source fingerprints. Their portable source references point to [the preceding snapshot](../p1-p3/source-snapshot.zip), while the small-tile winner uses [this snapshot](source-snapshot.zip). [manifest.json](manifest.json) and per-variant `build_info.h` identify each source. No executables are distributed.

The [additional same-run kernel comparison](kernel-comparison/README.md) is a separate measurement from this search's confirmation passes; its slightly different timings must not be substituted into this ranking.

The current tuned kernel matched the historical snapshot at publication; current register and harness/build files differ. To rebuild the historical winner from the repository root in an x64 Visual Studio developer terminal:

```bat
python -m zipfile -e results\published\tuned\p4\source-snapshot.zip results\reproduce-tuned-p4
cmake -S results\reproduce-tuned-p4 -B build\reproduce-tuned-p4 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_TUNED_BM=16 -DGEMM_TUNED_BN=32 -DGEMM_TUNED_BK=32 -DGEMM_TUNED_TM=2 -DGEMM_TUNED_TN=2
cmake --build build\reproduce-tuned-p4 --target gemm
build\reproduce-tuned-p4\gemm.exe --mode bench --kernel tuned --m 16 --n 256 --k 128 --seed 42 --warmup 100 --iters 10000 --rounds 5
```

To rerun the historical search, use the extracted `scripts/autotune.py --cases P4 --arch 89` without `--reuse-builds`; this rebuilds all candidates instead of requiring omitted historical binaries. A new build or measurement is not expected to reproduce exact timings or binary hashes.
