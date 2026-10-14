# 初始化版本验证

日期：2026-10-14（Asia/Shanghai）。本次验证包含用户实现的 naive FP32 SGEMM；与 `results/m0*` 中 naive 加入前的 cuBLAS 历史测量分开记录。

环境：Windows、RTX 4060 Laptop、SM 8.9、CUDA Toolkit 13.2、MSVC 19.44、Release。完整信息见 [info.txt](benchmark/info.txt) 和 [environment.txt](benchmark/environment.txt)。

## 检查结果

| 检查 | 结果 | 日志 |
|---|---|---|
| CMake Release 构建 | 通过 | [build.log](build.log) |
| CTest | 4/4 通过 | [ctest.log](ctest.log) |
| naive 固定正确性套件 | 32/32 通过；对 cuBLAS 全量检查，并用 CPU double 复核 | [naive-check.log](naive-check.log) |
| naive 非方阵边界 `(65,97,33)` | 通过，6305 个输出均有效 | [naive-boundary.log](naive-boundary.log) |
| 上述边界的 Compute Sanitizer memcheck | 0 errors | [naive-memcheck.log](naive-memcheck.log) |
| P1/P2/P3 成对 benchmark | naive 和 cuBLAS 共 6 行结果均 valid | [summary.csv](benchmark/summary.csv) |

验证命令（项目根目录，CMD）：

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1
ctest --test-dir build --output-on-failure
build\gemm.exe --mode check --kernel naive
build\gemm.exe --mode check --kernel naive --m 65 --n 97 --k 33
compute-sanitizer --tool memcheck --error-exitcode 3 build\gemm.exe --mode check --kernel naive --m 65 --n 97 --k 33
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Kernel naive -Iters 10 -OutputDir results\initial-20261014\benchmark
```

复现时为 benchmark 指定新的输出目录。脚本拒绝覆盖现有目录。

## 本次成对计时

FP32、seed=42、warmup=20、iters=10、rounds=5；同一 shape 的 naive 和 cuBLAS 使用相同输入和重复次数，轮间交替执行。

| ID | M × N × K | naive median ms | cuBLAS median ms | `t_cublas/t_naive` |
|---|---|---:|---:|---:|
| P1 | 2048 × 2048 × 2048 | 18.511769 | 1.933619 | 0.104454 |
| P2 | 16 × 4096 × 4096 | 0.799619 | 0.298394 | 0.373170 |
| P3 | 1001 × 769 × 513 | 0.864640 | 0.119296 | 0.137972 |

R=10 用于本次初始化运行验证，矩阵文件的默认重复次数没有修改。部分批次短于 50 ms，程序已给出 `timing_note`；正式优化对比应复核重复次数、时钟和后台负载，保留各轮均值。这里的时间不能与 profiler 重放条件下的时间直接比较，也不用于从耗时单独推断瓶颈。

原始轮均值、误差及前后 GPU 状态位于 `benchmark/`。源码摘要为 `fecb259c6699e29afe0c57490f36373e65941bf6f150f2b5d36f949c95893d50`；源码和可执行文件哈希见 [sha256.csv](benchmark/sha256.csv)。

## Profiling 使用约定

开发者此前已采集本地 `results/naive-p1.ncu-rep`，并确认 Nsight Compute 图形界面在纯英文用户名账户下可启动。后续使用该账户打开报告。本次初始化验证未重新采集 profiler 数据，原始报告默认由 `.gitignore` 排除。
