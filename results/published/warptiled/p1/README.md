# P1 warp-tiling search, 2026-10-22 00:50

Source experiment: `autotune-warptiled-20261022-005002-742164`. Shape **M/N/K=2048/2048/2048**. Parameter order: **BM / BN / BK / WM / WN / TM / TN / WNITER**. Fourteen configurations were screened; three warp finalists plus two tuned references were confirmed. All ten confirmation CSVs are retained through [summary.json](summary.json).

The winner **64 / 64 / 16 / 32 / 32 / 4 / 4 / 1** achieved median **2.589180 ms**, **6635.25 GFLOP/s**, versus the same-run prior tuned winner's **6048.59 GFLOP/s** and paired cuBLAS **7589.67 GFLOP/s**. Its two pass medians were 2.606196 and 2.582551 ms; the reported aggregate is the median of all ten round means. The runner-up reached 6552.88 GFLOP/s and was flagged within 3% of best; that flag is not a confidence interval.

RTX 4060 Laptop, SM89, Release, strict FP32, seed 42. Each confirmation invocation used **warmup=100, iters=44, rounds=5**; second-pass task order was reversed. The copied matrix's requested warmup=20 was raised to 100 by the confirmation policy. The winning configuration derives **128 threads/block**, **8192 shared bytes/block**, **66 registers/thread**, and zero compiler-reported spill loads/stores. See [manifest.json](manifest.json) for all retained resource reports and exact compiler/runtime versions.

The prior tuned winner is BM/BN/BK/TM/TN=64/128/32/4/4; it was recompiled from this experiment's snapshot, not represented by an older timing. [winners.json](winners.json) records the winning source fingerprint and portable build arguments. This is a whole-kernel comparison: changes to warp mapping, shared layout and generated instructions are not isolated causes.

The archived `src/**` matched the current code at publication. CMake and the candidate catalog have since gained two 64-thread configurations, so a new ordinary build can have a different fingerprint. To rebuild historical source from the repository root in an x64 Visual Studio developer terminal:

```bat
python -m zipfile -e results\published\warptiled\p1\source-snapshot.zip results\reproduce-warp-p1
cmake -S results\reproduce-warp-p1 -B build\reproduce-warp-p1 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_WARP_RUNTIME_CONFIGS=OFF -DGEMM_WARP_BM=64 -DGEMM_WARP_BN=64 -DGEMM_WARP_BK=16 -DGEMM_WARP_WM=32 -DGEMM_WARP_WN=32 -DGEMM_WARP_TM=4 -DGEMM_WARP_TN=4 -DGEMM_WARP_WNITER=1
cmake --build build\reproduce-warp-p1 --target gemm
build\reproduce-warp-p1\gemm.exe --mode bench --kernel warptiled --m 2048 --n 2048 --k 2048 --seed 42 --warmup 100 --iters 44 --rounds 5
```

To repeat the whole search and same-run tuned comparisons, run the extracted `scripts/autotune.py --kernel warptiled --cases P1 --arch 89`. For the current precompiled executable, the same eight values can instead be passed as `--bm 64 --bn 64 --bk 16 --wm 32 --wn 32 --tm 4 --tn 4 --wniter 1`. Check `--mode list --kernel warptiled` for compiled configurations.
