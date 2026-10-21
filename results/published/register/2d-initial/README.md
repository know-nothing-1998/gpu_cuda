# 初始 2D 4×4：保留失败实验

来源：`results/register-2d-review-20261019-114039`。这是 2026-10-19 11:40 开始捕获的 **TILED=32、TM=TN=4** 版本，不是后续修正后的实现。原始 README 有编码问题，此处依据原始 CSV、manifest 和检查日志重新整理。

2D 的 A 加载检查错误地使用输出的 `col+j<n`，应检查被读取 A 的 K 坐标。P1/P2 数值检查通过；P3、N 尾部及其他边界检查失败。K 尾部 32×32×33 虽可通过数值检查，memcheck 仍报告越界。因此 **不应使用此版本作为正确 GEMM 或宣传其通用性能**。

| Case | 1D 8×1 GFLOP/s | 初始 2D 4×4 GFLOP/s |
|---|---:|---:|
| P1 | 3168.30 | 3167.59（仅该尺寸有效） |
| P2 | 1236.59 | 1248.17（仅该尺寸有效） |
| P3 | 2663.76 | 无：正确性失败 |

两版 P1/P2 差距很小，不能从这轮宣称二维展开加速。[summary.json](summary.json) 保留 invalid 状态；各目录 `P*-pass*.csv` 是所有成功计时的原始数据，[check-P3.log](2d-4x4/check-P3.log) 和 [memcheck-a-k-tail.log](2d-4x4/memcheck-a-k-tail.log) 保留关键失败证据。完整 suite 的通过/失败计数与资源信息见 [validation-summary.json](validation-summary.json)，并记录原始日志哈希。

[profile-metrics.json](profile-metrics.json) 与 [profile-source-evidence.json](profile-source-evidence.json) 显示 P1 的 2D 版本 shared 写入 actual/ideal wavefront 为 4，而计算 shared load 的 actual=ideal。这里需要优化的是协作搬运与线程分工，不能将所有 shared 访问都称为 4-way conflict。

如需研究这个历史错误，可在独立副本重现（下面的 check 预期失败；不要将它当通过测试的教程实现）：

```bat
python -m zipfile -e results\published\register\2d-initial\source-snapshot.zip build\repro-2d-initial-source
copy results\published\register\2d-initial\2d-4x4\register.cu build\repro-2d-initial-source\src\register.cu
copy results\published\register\2d-initial\2d-4x4\register.cuh build\repro-2d-initial-source\src\register.cuh
cmake -S build\repro-2d-initial-source -B build\repro-2d-initial -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF
cmake --build build\repro-2d-initial --target gemm
build\repro-2d-initial\gemm.exe --mode check --kernel register --m 32 --n 1 --k 32
```

正确的后续二维实现及对照见 [cooperative-load](../cooperative-load/README.md)。历史 source manifest 记录测试期间工作区另外发生编辑；实验始终使用归档快照，而非后续编辑中的源码。
