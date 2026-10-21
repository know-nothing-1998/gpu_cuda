# P4 warp-tiling search, 2026-10-22 00:12

Source experiment: `autotune-warptiled-20261022-001257-922837`. Shape **M/N/K=16/256/128**. Parameter order: **BM / BN / BK / WM / WN / TM / TN / WNITER**. Fourteen configurations were screened; four warp finalists plus two tuned references were confirmed. All twelve confirmation CSVs are retained through [summary.json](summary.json).

The winner **16 / 32 / 32 / 8 / 16 / 2 / 2 / 1** achieved median **0.007366298 ms**, **142.35 GFLOP/s**. The previous tuned winner, recompiled and measured in this experiment, achieved **142.57 GFLOP/s**; paired cuBLAS achieved **158.86 GFLOP/s**. The 0.998 warp/tuned ratio is effectively a tie at this timescale, not evidence that warp tiling always accelerates small matrices.

RTX 4060 Laptop, SM89, Release, strict FP32, seed 42. Each confirmation invocation used **warmup=100, iters=10000, rounds=5**. Two passes reversed task order, and rankings take the median of the combined ten round means. The copied matrix's requested warmup=20 was raised to 100. The winner's pass medians were 0.007367783 and 0.007240467 ms. Its block has **128 derived threads**. [manifest.json](manifest.json) preserves compiler resources, correctness status and version fields; [winners.json](winners.json) records its source fingerprint.

The same-run prior tuned winner uses BM/BN/BK/TM/TN=16/32/32/2/2. Its measurements belong to this run, rather than the earlier tuned search. Comparisons include both shared layout and instruction-generation effects in addition to warp ownership.

This historical snapshot predates the executable's eight-parameter runtime dispatch. At publication, the kernel body was unchanged; the current `.cu` appends dispatch functions, and CLI/header/build support has evolved. Rebuild the archived tree to retain its exact source provenance. From the repository root in an x64 Visual Studio developer terminal:

```bat
python -m zipfile -e results\published\warptiled\p4\source-snapshot.zip results\reproduce-warp-p4
cmake -S results\reproduce-warp-p4 -B build\reproduce-warp-p4 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_WARP_BM=16 -DGEMM_WARP_BN=32 -DGEMM_WARP_BK=32 -DGEMM_WARP_WM=8 -DGEMM_WARP_WN=16 -DGEMM_WARP_TM=2 -DGEMM_WARP_TN=2 -DGEMM_WARP_WNITER=1
cmake --build build\reproduce-warp-p4 --target gemm
build\reproduce-warp-p4\gemm.exe --mode bench --kernel warptiled --m 16 --n 256 --k 128 --seed 42 --warmup 100 --iters 10000 --rounds 5
```

To repeat the entire historical search, run the extracted `scripts/autotune.py --kernel warptiled --cases P4 --arch 89`. For the current precompiled executable, select the same configuration with `--bm 16 --bn 32 --bk 32 --wm 8 --wn 16 --tm 2 --tn 2 --wniter 1`; a new build's fingerprint may differ.
