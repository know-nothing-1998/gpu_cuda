# User warp-tiling implementation: preserved results

These measurements describe the user's eight-parameter warp-tiling kernel on 2026-10-22. They do not describe the earlier assistant-written `warptiling-20261021-122319` experiment. Parameter order is **BM / BN / BK / WM / WN / TM / TN / WNITER**; thread count is derived, not an independent parameter.

| Case | M / N / K | Best measured warp parameters | Warp GFLOP/s | Same-run prior tuned winner GFLOP/s | Paired cuBLAS GFLOP/s | Warp / prior tuned |
|---|---|---|---:|---:|---:|---:|
| [P1](p1/README.md) | 2048 / 2048 / 2048 | 64 / 64 / 16 / 32 / 32 / 4 / 4 / 1 | 6635.25 | 6048.59 | 7589.67 | 1.097 |
| [P4](p4/README.md) | 16 / 256 / 128 | 16 / 32 / 32 / 8 / 16 / 2 / 2 / 1 | 142.35 | 142.57 | 158.86 | 0.998 |

P1 improved about 9.70% over the prior tuned winner recompiled and remeasured in the same experiment, and reached 87.42% of paired cuBLAS throughput. P4 was effectively tied with the small-tile tuned version. These are two separate 14-candidate searches; a winner on one shape is not a universal configuration. P1's second-ranked candidate is within 3% of first place, so the ranking should be read alongside the complete rounds. No conclusion about the later 64-thread experiment is recorded here because its measured data were not archived in these runs.

## Measurement and evidence

RTX 4060 Laptop, 24 SMs, SM89, NVIDIA CUDA compiler 13.2.78, MSVC 19.44.35217.0. Release, FP32 inputs/accumulation/output, strict FP32 cuBLAS, alpha=1, beta=0, seed 42. Screening used three rounds; confirmation used two passes in reversed task order, 100 warmups and five rounds per pass. The table aggregates the median of ten CUDA Event round means, not profiler timings or the median of two medians.

Each compiled variant passed 32 fixed correctness cases plus wide/tall/tile-tail checks. Each timed shape was checked against the full cuBLAS output and a CPU double reference (sampled for large outputs). The retained CSVs include these validation results, actual kernel, timing settings, and paired cuBLAS rounds. Both the default tuned version and the previous tuned winner were recompiled from the same snapshot and timed during confirmation.

Each run keeps [rankings and original confirmation CSVs](p1/rankings.json), `summary.json` including tuned references, winners, resources and compiler register/spill counts in `manifest.json`, per-variant build/config headers, original source ZIP/hash maps, and `provenance.json`. All confirmation candidates are preserved, not just winning rounds. Screening logs, binaries, build trees and NCU reports are omitted. JSON metadata has local executable paths removed; the timing CSVs and source ZIPs are byte-identical to the original experiment.

## Reproduction

From the repository root with the documented Windows CUDA/Visual Studio toolchain:

```bat
python scripts\autotune.py --kernel warptiled --cases P1 --space results\published\warptiled\p1\search-space.json --arch 89
python scripts\autotune.py --kernel warptiled --cases P4 --space results\published\warptiled\p4\search-space.json --arch 89
```

The archived spaces preserve the 14 tested candidates even if the main candidate catalog grows. The P1 snapshot's `src/**` matched the current source at publication; P4 preceded the executable's runtime configuration dispatch, while its kernel body was unchanged. CMake/catalog differences can change compiled fingerprints. To rebuild the exact historical source tree, extract that run's source ZIP to a new directory and follow its README. This preserves the source and procedure; identical numerical performance is not promised across GPU clocks, power limits, compiler or driver changes.
