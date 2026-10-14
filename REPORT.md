# M0 基础设施与 cuBLAS 历史记录

本文保留 naive 加入前的 M0 原始测量，表格与日志对应当时的源码摘要，不代表当前 naive 的验证状态。2026-10-14 后续已加入用户编写的 naive SGEMM，并成功采集本地 `results/naive-p1.ncu-rep`；该二进制默认不提交。开发者确认 Nsight Compute 图形界面可在纯英文用户名账户下启动，后续使用该账户查看报告。当前实现、构建和使用方式见 [README](README.md)。

日期：2026-10-14，Asia/Shanghai。范围：任务书第 10 节的 M0 基础设施，按用户要求保留 naive 供其自行实现；本轮不进入 M1/M3。

## M0 阶段状态

已建立可编译运行的 CMake/C++17/CUDA 工程、CPU double oracle、严格 FP32 cuBLAS 基线、四种 CLI 模式、数值错误检测、Event 计时、CSV 和固定 workload 批量脚本。naive 只有不可用的接入接口，不含计算实现，也不回退到 cuBLAS。其正确性、双实现交替计时、性能比值和 profiler 采集均待用户实现后验证。

首次检查项目只有 `GEMM_PROJECT_BRIEF.md`，没有已有实现、构建文件或 git 仓库；项目内没有 `AGENTS.md` 或任务书提及的 `DAILY_SOURCE_READING_GUIDE.md`。原任务书未修改。实现证据采用本地安装头文件、NVIDIA 文档及本次实测，不推测 cuBLAS 闭源实现。

## 实际环境

| 项目 | 本次读到的值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU；设备 0；WDDM |
| compute capability / 编译目标 | 8.9 / `CMAKE_CUDA_ARCHITECTURES=89` |
| SM / warp | 24 / 32 |
| 显存 / L2 | 8,585,216,000 bytes / 33,554,432 bytes |
| threads/block、threads/SM | 1024、1536 |
| registers/block、registers/SM | 65,536、65,536（设备上限，不是某 kernel 实耗） |
| shared/block、shared/SM | 49,152、102,400 bytes；opt-in block 上限 101,376 |
| NVCC / Toolkit | 13.2.78 / 13.2 |
| host compiler | MSVC 19.44.35217.0，Visual Studio 2022 17.14 |
| CMake / generator / 配置 | 4.0.2 / NMake Makefiles / Release |
| NVIDIA driver | 581.42；`cudaDriverGetVersion` 原始值 13000 |
| Runtime API | `cudaRuntimeGetVersion` 原始值 13000；安装头文件 `CUDART_VERSION=13020`，DLL 文件版本 `6,14,11,13020`，分别原样记录 |
| cuBLAS | `cublasGetVersion=130400`；本地头文件 13.4.0 build 1 |
| 数值模式 | math mode=2 (`CUBLAS_PEDANTIC_MATH`)，compute=`CUBLAS_COMPUTE_32F_PEDANTIC` |
| NSYS | 2025.6.3.541 |
| NCU / Compute Sanitizer | 2026.1.1.0 / 2026.1.1.0 |
| 电源计划 | Windows 高性能；是否接 AC 未单独采集 |

API 返回值、安装 Toolkit 版本和文件版本是不同的观测字段，没有将它们替换为推测的版本号。GPU 资源来自 `cudaGetDeviceProperties`。完整程序输出：[info.txt](results/m0-baseline/info.txt)，电源/工具记录：[environment.txt](results/m0-baseline/environment.txt)，完整 GPU 快照：[nvidia-smi-before.txt](results/m0-baseline/nvidia-smi-before.txt)。

本机 CUDA 13.2 头文件确认了 `cublasGemmEx` 签名和 pedantic 枚举；布局、计算类型语义对照 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)。NSYS/NCU 命令参数按已安装版本的 `--help` 和 section 列表核对。

## 验证结果

| 验证 | 实测结果 | 证据 |
|---|---|---|
| Release 构建 | 通过，无自写 GPU kernel，因此 ptxas 暂无用户 kernel 资源数据 | [build.log](results/m0/build.log) |
| 32 个结构化/随机正确性用例 | cuBLAS 对 CPU double 全量通过 | [ctest-details.log](results/m0/ctest-details.log) |
| 独立边界 `(31,37,29)` | 通过 | 同上 |
| 验证框架自身 | 手算非方阵、错误/转置结果、NaN/Inf、零范数、容限、溢出、RNG、统计通过 | 同上 |
| CLI 集成 | 23 次调用，非法参数/未实现路径非零退出，CSV 保护及 profile 隔离通过 | 同上 |
| P1/P2/P3 | 全输出有限；CPU 固定点积抽样通过 | [summary.csv](results/m0-baseline/summary.csv) |
| memcheck / racecheck / synccheck | cuBLAS `(65,97,33)` 均通过，0 error/hazard | [memcheck](results/m0/sanitizer-memcheck.log)、[racecheck](results/m0/sanitizer-racecheck.log)、[synccheck](results/m0/sanitizer-synccheck.log) |

CTest 总计 4/4 通过。小输入全量比较，性能输入分别检查 51/46/50 个 CPU double 点积，所有输出另做有限性扫描。大矩阵的 max_abs 和 relative_frob 是**所抽样位置**的误差，不声称完成整矩阵 CPU double 计算。没有生成 naive 的正确性/性能占位结果。

## 固定 benchmark 与结果

矩阵已锁定在 [benchmarks/matrix.csv](benchmarks/matrix.csv)。三个 shape 未调整，仅将初始 R=100 按 `ceil(100/pilot_median_ms)` 校准到 44/335/800，过程见 [矩阵说明](benchmarks/README.md)。两个实现存在后应使用同样输入、同样 R 和交替顺序；本轮只有库基线。

正式命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1
ctest --test-dir build --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\benchmark.ps1 -OutputDir results\m0-baseline
```

复现时换一个新的输出目录；脚本拒绝覆盖现有证据。逐 case 展开的实际命令在 [commands.txt](results/m0-baseline/commands.txt)。

下表是同工作集、预分配 GPU 内存、20 次预热、5 轮的 CUDA Event 时间；每个样本为 R 次操作的平均耗时。

| ID / M×N×K | R | median ms | min–max ms | GFLOP/s | CPU sample max_abs |
|---|---:|---:|---:|---:|---:|
| P1 / 2048×2048×2048 | 44 | 2.576868 | 2.532375–2.750324 | 6666.96 | 5.332e-5 |
| P2 / 16×4096×4096 | 335 | 0.339124 | 0.326839–0.342383 | 1583.11 | 1.905e-5 |
| P3 / 1001×769×513 | 800 | 0.178458 | 0.161981–0.188212 | 4425.61 | 4.092e-6 |

全部 valid、失败元素=0、非有限元素=0。每组 CSV 保留 5 个原始轮均值和误差范围；比值列=1 是库与自身的约定值，不是 naive 的性能。

正式测量前后快照记录温度约 57–69°C、graphics clock 1890–2610 MHz，开始为 P3 状态随后为 P0；显示桌面和多项后台图形进程仍存在。采样点不是整个计时区间的连续监控，也不是独占 GPU 测量。三组轮间波动仍可见，尤其 P3。预跑与正式跑的时延差别没有对应算法改动，不能归因于优化；计时次数、频率/负载状态和主机提交行为都是竞争解释。当前仅建立可复现基线，不下硬件瓶颈结论。

源码事实：FP32 pedantic、row-major 到 column-major 参数重解释、同 stream Event 区间，以及排除分配/传输/验证。数学计算：`2MNK/(1e6*t_ms)`。本次实测：上表、温度/频率快照和 timeline。尚无 NCU counters，不能据此声称内存、计算、occupancy 中哪项限制了性能。

## Profiling 实际状态

NSYS 原始调用曾报 `std::range_error: bad conversion`；改用项目内 ASCII TEMP/TMP 后成功。NCU 初始调用无输出并以 `-1073740791` 退出；使用 ASCII TEMP/TMP、禁用 stock 文件部署、显式 section 路径及 `--config-file off` 后可列出 sections 并连接目标。此过程说明工作区配置可避开启动异常，尚未精确定位工具内部根因。

NSYS 取得本地文件 `results/m0/cublas-nsys.nsys-rep`（二进制未入库），文本统计见 [nsys-stats.txt](results/m0/nsys-stats.txt)。该次 P1 profile 只运行 cuBLAS，实际记录 **21 次 `ampere_sgemm_64x64_nn`**，符合 20 warmup + 1 target。名称是观察到的库符号，不代表设备是 Ampere，也不能从名称推断完整内部实现。profile 前独立 check 的结果在 [profile-check.log](results/m0/profile-check.log)。timeline 时间包含 profiling 条件及预热，不能替代上面的独立 Event benchmark。

NCU 最小 `LaunchStats` 采集返回 `ERR_NVGPUCTRPERM`，退出码 1；证据在 [ncu-ascii-temp.log](results/m0/ncu-ascii-temp.log)。**没有生成可用 NCU 硬件报告，M2 硬件分析尚未完成**。本轮未修改 GPU 计数器权限；NVIDIA 提供了 [权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)。naive 尚未实现，因此本次 NSYS/NCU 尝试仅为工具可用性检查，不能替代任务书中的 naive 专项采集。

可复现命令及环境修正见 [README 的 Profiling 部分](README.md#profiling) 和 [commands.txt](results/m0/commands.txt)。原始 profiler 二进制在本地保留，默认被 `.gitignore` 排除；提交仓库时可以保留本节索引和文本摘要，按需显式收录关键二进制。

## 版本与证据索引

本轮最终源码摘要：`0ed11bacc29211db971d6781cdf100e613b289d859ad325ae2463952d902fc26`。构建时由 CMake 对自身、gemm/naive 源码、构建信息模板、性能矩阵的 SHA256 清单再求摘要。源码修改会触发重新配置，摘要写入程序和 CSV。

| 路径 | 内容 |
|---|---|
| `results/m0/` | 构建、CTest、设备、sanitizer、profile 尝试及 timeline |
| `results/m0-pilot/` | 原始 R=100 预跑、环境、命令、源码清单及相对最终版本的差异 |
| `results/m0-baseline/` | 锁定 R 的正式基线、各轮 CSV、环境和前后遥测 |
| `results/m0-baseline/source-snapshot.zip` | 对应最终可执行文件的源码/构建/矩阵/脚本/测试快照 |
| `results/m0-pilot/source-snapshot.zip` | 预跑对应的源码/构建/原始矩阵快照 |
| `results/m0-baseline/sha256.csv` | 源码与实际 executable SHA256，作为当前无 git commit 时的关联 |

## M0 结束时的后续任务

用户在 `src/naive.cu` 实现自己的 kernel 并开启接口，先跑 32 个正确性用例和 `(65,97,33)` 的 sanitizer，再测 P1/P2/P3。保留其原始线程映射与资源/计时基线后，才决定 M1：如果 naive 的线程列映射已经合并访存，就直接以 shared 分块复用为下一项单变量改动；不强行制造一个 mapped 升级。没有 naive 的实测、资源信息和可用 NCU counters，目前没有证据支持选择寄存器 tile、双缓冲或宣称瓶颈已经确认。
