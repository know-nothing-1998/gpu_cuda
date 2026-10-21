# 本机 GPU 架构与 GEMM 资源速查

2026-10-18，Asia/Shanghai。本机 CUDA 设备 0 为 **NVIDIA GeForce RTX 4060 Laptop GPU**，计算能力 **8.9（Ada Lovelace，编译目标 `sm_89`）**。CUDA 查询到 **24 个 SM**；结合 Ada 每 SM 128 个 FP32 CUDA core 的架构规格，全卡有 **3072 个 FP32 执行通道**。这里的 8.9 是硬件计算能力，不是 CUDA Toolkit 版本。[Ada 架构说明](https://docs.nvidia.com/cuda/ada-tuning-guide/)、[Ada 白皮书](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)

## 查询入口和证据

```bat
build\gemm.exe --mode info
```

已扩展 [src/gemm.cu](src/gemm.cu) 中的 `info()`，以后仍用这一条命令查看；修改源码后先 `cmake --build build --target gemm`。

- [原始 CUDA 输出](results/published/device/device-info.log)：保留 bytes、kHz、寄存器个数等原始单位。
- [结构化快照与计算公式](results/published/device/device-info.json)：原始字段、派生值、采集时间和可执行文件哈希。原始本机路径是历史元数据，不是 clone 后的输入依赖。
- 来源为本地 `results/device-info-20261018/`；当时 Release 构建及 40 次 CLI 检查通过，大型/重复日志不收录。
- [同轮 benchmark / profiling](results/published/baseline/README.md)：已测性能，不把设备属性当作实测吞吐。

`cudaGetDevice` 只返回当前设备编号，硬件属性由 `cudaGetDeviceProperties` 读取；频率由 `cudaDeviceGetAttribute` 补充；可用显存由 `cudaMemGetInfo` 读取。本机 CUDA 13.2 的 `cudaDeviceProp` 已移除旧教程中的 `clockRate/memoryClockRate` 字段，代码使用其替代 API。[官方字段迁移说明](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html)

本次 info 扩展后的源码摘要为 `e0eb76316a14bbf9c40ccd316203a794c2d23c902ba190e47a6ab2f2c05b94e4`；之前四实现性能测量仍对应其原始摘要 `25d0db3ee00e7f864653347a07994404840658c34c6d432eb4491099f7960dd4`。本次只扩充查询输出，没有修改 GEMM kernel，也没有重测性能或替换旧报告。

## 优先关注：线程与 SM

| 项目 | 本机值 | CUDA 字段 / 来源 | 对 GEMM 的意义 |
|---|---:|---|---|
| CUDA 设备数量 | 1 | `cudaGetDeviceCount` | 本项目选择设备 0 |
| SM 数 | 24 | `multiProcessorCount` | block 分配到这些 SM 执行 |
| Warp 大小 | 32 threads | `warpSize` | 合并访存和 bank conflict 都按 warp 分析 |
| 单 block 线程上限 | 1024 | `maxThreadsPerBlock` | `32×32` 已用满单 block 上限 |
| 单 SM 驻留线程上限 | 1536 | `maxThreadsPerMultiProcessor` | 不是单周期执行线程数 |
| 单 SM 驻留 warp 上限 | 48 | `1536 / 32` 推导 | 用于计算理论 occupancy |
| 单 SM 驻留 block 上限 | 24 | `maxBlocksPerMultiProcessor` | 线程、寄存器、shared 可能先成为限制 |
| Block 各维上限 | 1024×1024×64 | `maxThreadsDim` | 各维限制与总线程数 ≤1024 必须同时满足 |
| Grid 各维上限 | 2147483647×65535×65535 | `maxGridSize` | 限制一次 launch 的网格尺寸 |

**对当前 kernel 的直接结论：** naive/coalesced/shared 都是 `32×32=1024` threads/block。一个 SM 无法放下两个这样的 block，因为 `2×1024 > 1536`。所以仅由线程数决定的驻留上限为 **1 block/SM、32 warps/SM、66.7% occupancy**。它不表示 FP32 算力只能达到 66.7%，也不表示改成小 block 就必然更快；每线程工作量和实际资源分配也会影响速度。

SM 是调度和执行资源的集合，CUDA core 是其中的算术执行通道。驻留的 48 个 warp 不会在同一周期全部执行；硬件在就绪 warp 间发射指令，以隐藏访存和指令依赖的等待。

## 优先关注：寄存器、shared、缓存、显存

以下容量中 KiB=`1024` bytes、MiB=`1024²` bytes、GiB=`1024³` bytes。容量作用域不同，不能直接混加。

| 资源 | 本机查询值 | CUDA 字段 | 使用时要注意 |
|---|---:|---|---|
| 每 SM 寄存器 | 65536 个 32-bit = **256 KiB** | `regsPerMultiprocessor` | 同 SM 的驻留线程共享这个寄存器预算 |
| 单 block 寄存器上限 | 65536 个 32-bit | `regsPerBlock` | 这是总量，不是每线程可用量 |
| 每 SM shared 上限 | **100 KiB** | `sharedMemPerMultiprocessor` | 实际 kernel 的 carveout/分配可能更小 |
| 单 block 默认 shared 上限 | **48 KiB** | `sharedMemPerBlock` | 当前静态 shared 分配处于这一范围内 |
| 单 block opt-in shared 上限 | **99 KiB** | `sharedMemPerBlockOptin` | 超过默认上限需 dynamic shared 和显式 opt-in |
| 每 block driver 保留 shared | **1 KiB** | `reservedSharedMemPerBlock` | NCU 总分配可比源码数组大小多这一项 |
| 全卡 L2 | **32 MiB** | `l2CacheSize` | 所有 SM 共享；会影响重复读取是否到达显存 |
| 可配置 persisting L2 上限 | **22 MiB** | `persistingL2CacheMaxSize` | 不是默认已经保留了 22 MiB，也不是额外缓存 |
| Constant 地址空间 | **64 KiB** | `totalConstMem` | 这是地址空间容量，不是 constant cache 大小 |
| CUDA 可见总显存 | **8,585,216,000 bytes ≈7.996 GiB** | `totalGlobalMem` | 约 8 GiB；不把 API 原始值改写成恰好 8 GiB |
| 查询时空闲显存 | **7,429,160,960 bytes ≈6.919 GiB** | `cudaMemGetInfo` | 会随桌面程序、上下文和内存分配变化 |
| 显存总线宽度 | **128 bit** | `memoryBusWidth` | 用于计算理论显存带宽 |

字段定义以 [CUDA Runtime API](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/structcudaDeviceProp.html) 为准。

**寄存器预算：** 当前 shared 编译结果是 37 registers/thread，粗算一个 block 为 `37×1024=37888` 个寄存器。实际分配还受寄存器分配粒度约束，不能只用整数除法预测所有 occupancy。Ada 的每线程架构上限是 255 个寄存器，但 1024 线程 block 只有 65536 个总预算；做 register tiling 时需同时观察 ptxas/NCU 的寄存器数、spill 和驻留情况。[Ada 资源限制](https://docs.nvidia.com/cuda/ada-tuning-guide/#occupancy)

**Shared 预算：** 当前 `As[32][32] + Bs[32][32]` 为 `2×32×32×4=8192 bytes`，即 8 KiB。加上 driver 的 1 KiB，本轮 NCU 报告总 shared 为 9 KiB/block。这是不同口径的同一资源，不是报告矛盾。

**L2 对当前矩阵的意义：** 2048² 个 float 为 16 MiB，A+B 共 32 MiB，A+B+C 共 48 MiB。输入规模与 L2 相近，输出和其他流量也会占用缓存；因此 coalesced 的重复读取可以大量命中缓存，却不能假定三份矩阵一直完整留在 L2。

## 架构文档补充：API 没有直接提供的细节

| Ada SM 8.9 特性 | 值 | 在本项目中的用途 |
|---|---:|---|
| 每 SM FP32 CUDA cores | 128 | 结合 24 个 SM，推导全卡 3072 个 FP32 通道 |
| 每 SM warp scheduler | 4 | 阅读 NCU 每 scheduler 的 eligible/issued warp 指标 |
| 每 SM Tensor Core | 4 个，第 4 代 | 未来单独研究 Tensor Core 路径；当前严格 FP32 使用普通 FP32 对照 |
| 每 SM L1/texture/shared 统一资源池 | 128 KiB | **不是**额外独立的“128 KiB L1 + 100 KiB shared” |
| Shared banks | 32 | 连续 float 映射到连续 bank，同地址读取可广播 |

SM 内部由四个分区组成，每个分区有 16 个 FP32 专用通道及 16 个可执行 FP32/INT32 的通道。纯 FP32 峰值使用全部 128 个通道；整数寻址、访存、同步和依赖等会影响真实 kernel 的效率。[Ada 白皮书](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)

L1/texture/shared 共用物理资源，可为 shared 选择不同 carveout；100 KiB 是 shared 的设备上限，并不表示每个 kernel 都使用这个设置。[统一缓存与 shared 说明](https://docs.nvidia.com/cuda/ada-tuning-guide/#unified-shared-memory-l1-texture-cache)

对于标量 float，bank 可按 `(byte_address / 4) % 32` 理解；同一 warp 的不同线程访问同 bank 的不同地址会产生冲突，读取同一地址则可广播。原布局 `Bs[i][tx]` 跨 lane 连续；转置成 `Bs[tx][i]` 且行宽为 32 时会集中到同一个 bank。[Shared 访问说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns)

## 频率、理论上限与实测性能

本次 `cudaDeviceGetAttribute` 返回：

| 属性 | 原始值 | 换算 |
|---|---:|---:|
| `cudaDevAttrClockRate` | 1,890,000 kHz | 1.890 GHz |
| `cudaDevAttrMemoryClockRate` | 8,001,000 kHz | 8.001 GHz（API 报告口径） |

按该核心频率和 FP32 FMA 计算：

```text
FP32 理论吞吐 = SM 数 × FP32 通道/SM × 每 FMA 的 FLOP × 核心频率
              = 24 × 128 × 2 × 1.890 GHz
              = 11612.16 GFLOP/s = 11.61216 TFLOP/s
```

不需要再乘 warpSize。FMA 完成一次乘法和一次加法，按 2 FLOP 计；这是 **1.890 GHz 下纯 FP32 FMA 的理论值**，不是本机所有动态 Boost 状态下的最高性能，也不是 SGEMM 可持续实测吞吐。

按 CUDA 报告显存频率的 DDR 口径计算：

```text
理论带宽 = 2 × 8.001e9 × (128 / 8)
         = 256.032 GB/s

该频率下 Roofline 拐点 = 11612.16 / 256.032 ≈45.35 FLOP/byte
```

这里 GB/s 使用十进制。公式中的 2 已计入双倍数据率，不能把已经翻倍的有效数据率再乘一次 2；理论总线带宽也不是已经测得的持续带宽。[NVIDIA 理论带宽计算](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation)

另存的 [nvidia-smi 频率快照](results/published/device/nvidia-smi-clocks.csv) 显示空闲时 SM 为 210 MHz、memory 为 405 MHz，`clocks.max.sm` 报告 3105 MHz。它与 CUDA 报告的 1890 MHz 是不同口径；最大频率字段不保证 GEMM 能持续运行在该频率。实际运行受功耗、温度、负载等影响，分析某个 kernel 的效率应采用对应 NCU 报告中的频率。

2026-10-18 的 2048³ 普通 benchmark：naive **117.10**、coalesced **959.37**、shared **1209.10**、与 shared 配对的 cuBLAS **7604.82 GFLOP/s**。这是普通 CUDA Event 计时；不要直接把它们除以静态 1.890 GHz 下的理论值就当作准确的运行利用率。按各自 NCU 采集时钟计算的 FP32 峰值占比分别为约 **0.74%、6.14%、7.81%、59.83%**，见 [四实现 Roofline](results/published/baseline/README.md)。

## 次要但需要记住的运行特性

| CUDA 属性 | 本机值 | 含义 |
|---|---:|---|
| `asyncEngineCount` | 1 | CUDA 报告的异步引擎数；拷贝与计算重叠还需合适的 pinned buffer、stream 和资源条件 |
| `concurrentKernels` | 1 | 支持并发 kernel，不保证任意两个 kernel 都能同时驻留 |
| `streamPrioritiesSupported` | 1 | 支持 stream 优先级 |
| `unifiedAddressing` | 1 | 支持统一虚拟地址；不意味着 CPU/GPU 访问成本相同 |
| `managedMemory` / `concurrentManagedAccess` | 1 / 0 | 支持 managed memory，但不应假设 CPU/GPU 可随意同时访问 |
| `canMapHostMemory` | 1 | 支持映射 host memory，不代表 PCIe 访问等同显存速度 |
| `integrated` | 0 | 独立 GPU |
| `ECCEnabled` | 0 | 当前未启用 ECC；该字段不单独证明硬件是否具有 ECC 能力 |
| `tccDriver` | 0 | 本机 Windows 使用 WDDM 路径；桌面活动会影响测量稳定性 |

当前项目 benchmark 的数据已驻留 GPU，计时不包含 host/device 拷贝，因此先关注 **warp 地址布局 → shared/寄存器复用 → LSU 指令量与调度 → tile 和资源预算**。设备属性给出硬件边界，真正的瓶颈仍需结合 kernel 的编译资源和 NCU 指标判断。
