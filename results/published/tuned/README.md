# Tuned SGEMM: preserved autotuning results

These are historical measurements from 2026-10-21 on an RTX 4060 Laptop GPU (24 SMs, compute capability 8.9). The parameter order is **BM / BN / BK / TM / TN**. All timings use the CUDA Event harness in Release mode and strict FP32 cuBLAS (`CUBLAS_COMPUTE_32F_PEDANTIC`, alpha=1, beta=0); they are not Nsight Compute timings.

| Case | M / N / K | Best measured parameters | GFLOP/s | Same-run fixed tuned baseline / winner time | Paired cuBLAS / winner time |
|---|---|---|---:|---:|---:|
| [P1](p1-p3/README.md) | 2048 / 2048 / 2048 | 64 / 128 / 32 / 4 / 4 | 5616.15 | 1.057 | 0.810 |
| [P2](p1-p3/README.md) | 16 / 4096 / 4096 | 32 / 64 / 32 / 4 / 4 | 1330.81 | 1.178 | 0.783 |
| [P3](p1-p3/README.md) | 1001 / 769 / 513 | 64 / 64 / 16 / 4 / 4 | 4100.78 | 1.061 | 0.789 |
| [P4](p4/README.md) | 16 / 256 / 128 | 16 / 32 / 32 / 2 / 2 | 141.64 | 1.133 | 0.906 |

The fixed tuned baseline is 64 / 64 / 64 / 4 / 4. P1-P3 used 37 candidates; P4 was a separate 43-candidate search with additional small tiles. These runs do not form a complete 43-candidate by four-shape experiment. The winner is the best measured candidate in that search, not a global optimum.

The [separate P4 shared/register/tuned comparison](p4/kernel-comparison/README.md) preserves the small-matrix observation: fixed large tiles do not always help, and nearby kernels can change order between passes. For the latest same-run comparison against the user's warp-tiling kernel, use [the warp-tiling results](../warptiled/README.md); do not compute a speedup by dividing measurements from different dates.

## What is retained

Each run retains every confirmation CSV behind its published rankings, winner parameters, resource metadata, source fingerprints, and a small source-only ZIP. Every CSV contains both the candidate and its paired cuBLAS result, validation status, timing settings, and all five per-round means. Combine both passes (ten means) before taking the median. `cublas_over_self = time_cublas / time_candidate`, also equal to candidate GFLOP/s divided by cuBLAS GFLOP/s for the same shape.

Screening samples, build directories, executables, profiler reports, and unrelated logs are deliberately omitted. `provenance.json` records the original experiment identifier and checksums of byte-identical retained files. `manifest.json` and `winners.json` preserve the measurements but replace local executable paths with portable source references. The source ZIPs and their file-hash maps are unchanged.

## Reproduction

For current code, run from the repository root in an x64 Visual Studio developer terminal with CUDA, CMake, NMake and Python available:

```bat
python scripts\autotune.py --kernel tuned --cases P1 P2 P3 --space results\published\tuned\p1-p3\search-space.json --arch 89
python scripts\autotune.py --kernel tuned --cases P4 --space results\published\tuned\p4\search-space.json --arch 89
```

These commands create new measurements. For historical source, extract the corresponding `source-snapshot.zip` into a new directory and use its CMake project or scripts, as shown in each run's README. Do not extract over the current checkout. The current `src/tuned.cu` matched these historical snapshots at publication, but the CLI/build infrastructure and `src/register.cu` have changed; the compiled source fingerprint of a new build can therefore differ. Performance also depends on GPU clocks, power, thermals, driver and compiler versions.
