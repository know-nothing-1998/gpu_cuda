# TM sweep：更多累加器有最优点

来源：`results/register-tm-sweep-20261018-232909`。固定 TILED=32、每线程单列，测试 TM=1/2/4/8/16/32 与同轮 shared。两遍相反顺序测试在 P1/P2/P3 上都由 TM=8 胜出；不表示其他 tile 或形状也有同样最优值。

| TM | P1 GFLOP/s | P2 GFLOP/s | P3 GFLOP/s |
|---|---:|---:|---:|
| 1 | 1211.72 | 485.63 | 1051.51 |
| 2 | 2032.00 | 850.55 | 1794.42 |
| 4 | 2679.33 | 1097.25 | 2288.23 |
| 8 | 3328.49 | 1391.93 | 2893.89 |
| 16 | 2586.16 | 1087.15 | 2251.31 |
| 32 | 1679.51 | 652.91 | 1399.70 |

精确数值以 [summary.json](summary.json) / [summary.csv](summary.csv) 为准；[benchmarks](benchmarks) 保存全部 42 次 benchmark 的 CSV（6 个 TM 与 shared × 3 个形状 × 2 遍），各自带配对 cuBLAS、五轮平均耗时和正确性结果。

[profile-metrics.json](profile-metrics.json) 保留 P1 的资源与指令统计。TM=8/16/32 对应寄存器 50/56/127 个/线程，线程数 128/64/32；A 计算加载分别出现 LDS.128/LDS.64/LDS。全部版本的编译资源记录无 local allocation；更大的 TM 没有继续提高吞吐。各版本的 suite/tall 检查计数与编译资源原文摘要见 [validation-summary.json](validation-summary.json)，包含原始日志路径与哈希；历史总体信息见 [manifest.json](manifest.json)。

TM=8 复现（从项目根目录）：

```bat
python -m zipfile -e results\published\register\tm-sweep\source-snapshot.zip build\repro-tm8-source
copy results\published\register\tm-sweep\variants\tm8\register.cu build\repro-tm8-source\src\register.cu
cmake -S build\repro-tm8-source -B build\repro-tm8 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF
cmake --build build\repro-tm8 --target gemm
build\repro-tm8\gemm.exe --mode check --kernel register
build\repro-tm8\gemm.exe --mode bench --kernel register --m 2048 --n 2048 --k 2048 --warmup 20 --iters 44 --rounds 5
```

其他 TM 使用 `variants/tm*/register.cu`；同轮 shared 用 `--kernel shared`。每个版本应先 check，再按 [matrix.csv](matrix.csv) 重测，并以正序和倒序各做一遍。通用口径与限制见 [上一层说明](../README.md)。
