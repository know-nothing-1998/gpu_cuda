# FP32 / DRAM Roofline

设备为 RTX 4060 Laptop、SM 8.9。最新图来自 **2026-10-22** 的同一当前构建，覆盖 naive、coalesced、shared、register、tuned、warptiled 和 cuBLAS，矩阵均为 P1=`2048×2048×2048`，见 [采集与配置说明](results/published/roofline-20261022/README.md)。各实现使用默认参数，不是 autotune 优胜配置。

![七版本 FP32 / DRAM Roofline，P1 2048³](results/published/roofline-20261022/roofline.svg)

[model.json](results/published/roofline-20261022/model.json) 保留指标、报告哈希和计算结果。模型中的原始本机路径用于溯源，离线重画不需要这些路径或原始 `.ncu-rep`。大型报告仍只保存在开发机。

2026-10-18 的四版本结果继续保留为 [历史基线](results/published/baseline/README.md)：[旧图](results/published/baseline/roofline/roofline.svg)、[旧模型](results/published/baseline/roofline/model.json)、[其他关键 counters](results/published/baseline/profiling-metrics.json)。新旧图采集日期和构建不同，不能把差值直接归因于某一项优化。

## 模型与口径

```text
P(I) = min(P_compute, B_DRAM * I)
I = FP32 运算数 / DRAM 字节数
```

I 的单位为 FLOP/Byte，P 为 GFLOP/s，B 为十进制 GB/s。斜线是显存带宽上限，水平线是计算上限，交点为 P_compute/B_DRAM。上限按每份报告的实际频率计算，尚未用独立微基准校准可持续性能；cuBLAS 实测值不等于机器峰值。

脚本沿用 NCU 的 scalar FP32 口径：

```text
P_compute = 2 * FFMA_peak_inst_per_cycle * SM_frequency
B_DRAM = DRAM_peak_bytes_per_cycle * DRAM_frequency
P_HW = (FADD_per_cycle + FMUL_per_cycle + 2*FFMA_per_cycle) * SMSP_frequency
I_HW = P_HW / measured_DRAM_bytes_per_second
```

FFMA 按 2 FLOP。JSON 另存有效 GEMM 工作量 `2MNK` 对应的吞吐与强度；额外计算会使硬件执行量不同。不能把两套坐标混在一起。

采集使用 kernel replay、cache-control=all、clock-control=boost，跳过 20 次 warmup 后采集一次 launch。清缓存重放和频率条件不同，NCU 时间不能代替普通 Event benchmark 的性能比较。显存点在斜线下方也不能单凭位置判断 DRAM 带宽饱和；需要检查访存指令、cache 和 warp 等待。本模型不包含 PCIe、L1/L2/shared 的独立上限。

## 离线重画

clone 后可直接读取已发布的模型；只需 Python 和 Matplotlib，不需要 GPU 或 Nsight Compute：

```bat
python -m pip install matplotlib
python scripts/roofline.py --model results/published/roofline-20261022/model.json --output results/roofline-redraw
```

这只重画已有测量，不重新测试 kernel；输出位于新的本地 results 目录。

## 重新采集和绘图

先按根 README 编译。以下命令从仓库根目录运行，需要 GPU、Nsight Compute 及 performance counters 权限；文件名须为新的报告：

```bat
build\gemm.exe --mode check --kernel shared --m 2048 --n 2048 --k 2048
build\gemm.exe --mode check --kernel cublas --m 2048 --n 2048 --k 2048
ncu --set full --replay-mode kernel --cache-control all --clock-control boost --launch-skip 20 --launch-count 1 --export results\shared-p1-new build\gemm.exe --mode profile --kernel shared --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1
ncu --set full --replay-mode kernel --cache-control all --clock-control boost --launch-skip 20 --launch-count 1 --export results\cublas-p1-new build\gemm.exe --mode profile --kernel cublas --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1
python -m pip install matplotlib
python scripts\roofline.py --report shared=results\shared-p1-new.ncu-rep --report cublas=results\cublas-p1-new.ncu-rep --m 2048 --n 2048 --k 2048 --output results\roofline-new
```

绘图脚本只读取报告，不启动 GPU 工作负载。每份报告必须恰好包含一个完成整次 GEMM 的 kernel；多 kernel 的操作需要另行汇总，不能把完整 2MNK 重复分配给每一个 kernel。目前模型限定 SM 8.9 scalar FP32；换设备或 Tensor Core 路径需调整模型。

脚本的旧无参数默认输入是开发机的 2026-10-17 报告，**clone 后不可用**；重画精选结果时使用 `--model/--output`，导入新报告时使用 `--report/--output`。也可通过 `--ncu PATH` 指定 NCU。已有 SVG/JSON 可直接阅读，不需要安装 profiler。

参考：[NVIDIA Roofline](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)、[缓存控制](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)、[提取公式](https://forums.developer.nvidia.com/t/extract-data-from-roofline-plot/322219/2)。
