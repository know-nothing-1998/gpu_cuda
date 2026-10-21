# FP32 / DRAM Roofline

设备为 RTX 4060 Laptop、SM 8.9。精选图和指标来自 2026-10-18 的 naive/coalesced/shared/cuBLAS 同轮 profiling，见 [基线实验](results/published/baseline/README.md)。

![FP32 / DRAM Roofline](results/published/baseline/roofline/roofline.svg)

[model.json](results/published/baseline/roofline/model.json) 保留原始指标、报告哈希和计算结果；[profiling-metrics.json](results/published/baseline/profiling-metrics.json) 用于核对其他关键 counters。模型中的原始本机路径是历史溯源信息，不是 clone 后所需输入。大型 `.ncu-rep` 没有上传。

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

脚本的旧无参数默认输入是开发机的 2026-10-17 报告，**clone 后不可用**；使用上面的显式 `--report/--output`。也可通过 `--ncu PATH` 指定 NCU。已有 SVG/JSON 可直接阅读，不需要安装 profiler。

参考：[NVIDIA Roofline](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)、[缓存控制](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)、[提取公式](https://forums.developer.nvidia.com/t/extract-data-from-roofline-plot/322219/2)。
