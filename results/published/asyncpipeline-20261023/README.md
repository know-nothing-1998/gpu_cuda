# 2026-10-23：从 shared 冲突优化到混合异步流水线

RTX 4060 Laptop / Ada SM 8.9，P1=`M=N=K=2048`，FP32 输入/累加/输出，与严格 FP32 cuBLAS 配对。默认八参数均为 `BM/BN/BK/WM/WN/TM/TN/WNITER = 64/64/16/32/32/4/4/1`，128 threads/block，两阶段 shared 为 16 KiB。

本次最有用的结果是：A 改用 `float4` 普通预取、B 保留硬件异步拷贝后，吞吐比前一版标量异步 A 提高约 **32%～33%**。最新两轮与同窗口 `doublebuffer` 相比，耗时仍高 **2.14% / 2.34%**，目前只能说接近基线，不能据此认定硬件异步一定更快。另一个 `doublebuffer_opt` 实验说明，消除 bank conflict、改善 C 写回，并不保证总性能提升。

当前实现见 [asyncpipeline.cu](../../../src/asyncpipeline.cu)、[doublebuffer.cu](../../../src/doublebuffer.cu)。双缓冲的上一阶段四形状结果见 [2026-10-22 实验](../doublebuffer-20261022/README.md)。性能部分整理已有测量；本次发布另从独立源码副本重新构建，执行完整 CTest 和边界 sanitizer，见 [发布验证](validation/README.md)。

## 保留了哪些证据

| 文件 | 用途 |
|---|---|
| [benchmark-summary.json](benchmark-summary.json) | 两个实验窗口的逐轮时间、GFLOP/s、配对 cuBLAS、构建指纹和顺序 |
| [benchmarks/](benchmarks/) | 12 份原始 CSV、24 行；每份同时保留自写 kernel 和同轮 cuBLAS |
| [ncu-summary.json](ncu-summary.json) | 六份历史采样的资源、动态指令、global sectors、shared wavefront 摘要及报告 SHA256 |
| [global-store-evidence.json](global-store-evidence.json) | 原 doublebuffer 的 C 写回 actual/ideal sector 证据 |
| [sources/](sources/) | 三个阶段的精确 `.cu` 快照，便于用现有项目框架重建 |
| [provenance.json](provenance.json) | 原始输入路径/哈希、逐字节复制关系、三种哈希的区别与复现边界 |
| [validation.json](validation.json)、[historical-validation/](historical-validation/) | 早先构建状态的正确性与 sanitizer 检查摘要及关键原始日志 |
| [validation/README.md](validation/README.md) | 本次独立 Release 构建、20 项 CTest 和新增两版的六次 sanitizer 检查 |

原始 `.ncu-rep`、EXE、完整 SASS、整项目源码 ZIP 和重复 autotuning 输出不收录。JSON 中的 `report_path_local_only` / `original_*` 是开发机历史来源，不是 clone 后的依赖；它们的 SHA256 用于识别原证据。精选文件的完整性由上级发布索引维护。

## 1. doublebuffer_opt：局部指标改善，额外指令增加代价

原 `doublebuffer` 的 A 在 shared 中转置，P1 标量 STS 的 actual/ideal wavefront 比为 4。C 每线程逐元素写回，32 条静态标量 STG 的聚合 actual sectors 为 2,097,152，ideal 为 524,288。这里的 sector 是请求层指标；C 的有效输出仍只有 `2048×2048×4 = 16 MiB`，不能把重复 sector 数当作同倍数 DRAM 写入量。

`doublebuffer_opt` 在同一文件中单独注册：A 使用 swizzle，C 尝试 `float4` 写回。NCU 结构对照如下：

| P1 指标 | doublebuffer | doublebuffer_opt |
|---|---:|---:|
| A 标量 shared store：actual / ideal wavefront | 16,777,216 / 4,194,304 | 4,194,304 / 4,194,304 |
| C 动态 warp store 指令 | 131,072 个 STG | 32,768 个 STG.128 |
| Global store sectors | 2,097,152 | 524,288 |
| 全部动态 warp 指令 | 367,136,768 | 411,312,128（+12.03%） |
| 寄存器/thread；分配等效值 | 105；112 | 119；120 |
| 理论 occupancy | 33.33% | 33.33% |
| 实际 occupancy | 32.654% | 32.614% |
| Local load/store spill 指令 | 0 / 0 | 0 / 0 |

LOP3 从 2,695,168 增至 27,369,472，IMAD 从 6,496,256 增至 19,267,584。增加的布局/寻址工作说明“冲突归零”需要付出代价；不能把观察到的退化归因于理论 occupancy 下降，因为两版仍是 4 blocks/SM。

**计时边界：** 用户报告了 P1 负优化，但该对照没有保留下可审计的普通 benchmark CSV，因此本页不给出速度退化百分比。两份 NCU 的 SM 频率分别为约 0.748 / 1.717 GHz，其 Duration 不能直接比较。这里确认的是机制与指令代价，不是对负优化原因的单变量证明。

## 2. A 的三种搬运方式

| 名称与快照 | A 路径 | B 路径 |
|---|---|---|
| [scalar-async](sources/asyncpipeline-scalar-async.cu) | 从 `element=tid` 开始，逐个 4 B async copy 到转置 As | 连续时 16 B async copy |
| [a4-scalar-async](sources/asyncpipeline-a4-scalar-async.cu) | 从 `element=4*tid` 开始，每线程负责相邻四个元素，仍是四次独立 4 B async copy | 同上 |
| [float4-hybrid](sources/asyncpipeline-float4-hybrid.cu) | 普通 `LDG.128` 预取到 nextA；算当前块后分四次 STS 发布到转置 As | 保留 `LDGSTS...128` 硬件异步拷贝 |

三版都保留两份 shared、相同的 warp tiling 与边界补零。`float4-hybrid` 的 A 属于软件预取，B 属于硬件异步；四个连续 global 元素不能被一次连续 16 B 拷贝自动散写到转置 As。

### 实验窗口 A：只改 A 的线程分工

每次 `warmup=50, iters=50, rounds=5, seed=42`；表中每个数来自一次调用的 `median_ms`，不是最快轮。两次顺序分别是 scalar→A4→doublebuffer，以及 doublebuffer→A4→scalar。

| 版本 | 第1轮 ms / GFLOP/s | 第2轮 ms / GFLOP/s |
|---|---:|---:|
| scalar-async | 3.654817 / 4700.61 | 3.726232 / 4610.52 |
| a4-scalar-async | 3.361689 / 5110.49 | 3.491082 / 4921.07 |
| 同窗口 doublebuffer | 2.516214 / 6827.67 | 2.550272 / 6736.48 |

A4 吞吐提高 **8.72% / 6.74%**。但 A4 的标量 global 指令跨 lane 变为 stride-4，global load sectors 从 33,554,432 增为 83,886,080；shared async copy 的 excessive wavefront 从 67,108,864 降为 41,943,040。它改善一侧并增加另一侧的请求工作，仍明显慢于 doublebuffer。Excessive wavefront 的变化不是显存字节变化，也不直接等于某个标量 bank-conflict 倍数。

### 实验窗口 B：将 A 换成普通 float4 预取

同样的计时参数；顺序分别是 A4→hybrid→doublebuffer，以及 doublebuffer→hybrid→A4。这里重测了 A4，不能把两个窗口各自最好的数拼成累计加速曲线。

| 版本 | 第1轮 ms / GFLOP/s | 第2轮 ms / GFLOP/s |
|---|---:|---:|
| a4-scalar-async | 3.386915 / 5072.42 | 3.516355 / 4885.70 |
| float4-hybrid | **2.558095 / 6715.88** | **2.635550 / 6518.51** |
| 同窗口 doublebuffer | 2.504458 / 6859.71 | 2.575319 / 6670.97 |
| 与 hybrid 配对的 cuBLAS | 2.297443 / 7477.82 | 2.322328 / 7397.69 |

hybrid 相对 A4 的吞吐提高 **32.40% / 33.42%**；相对 doublebuffer 耗时高 **2.14% / 2.34%**。两轮小差距不足以证明普遍或稳定的优劣。每个版本都有自己的配对 cuBLAS 行，上表没有用另一版本的 cuBLAS 时间替换它。

## 3. 最终 SASS / NCU 告诉了我们什么

以下动态指令数量以 P1 的一次采样为范围，时间不参与这张跨采样结构对照：

| 指标 | doublebuffer | float4-hybrid |
|---|---:|---:|
| LDG.128 | 2,097,152 | 1,048,576 |
| STS.128 | 1,048,576 | 0 |
| LDGSTS...128 | 0 | 1,048,576 |
| A 标量 STS | 4,194,304 | 4,194,304 |
| FFMA | 268,435,456 | 268,435,456 |
| 有效 LDS.128 | 25,165,824 | 25,165,824 |
| LDGDEPBAR / DEPBAR | 0 / 0 | 524,288 / 524,288 |
| BAR / WARPSYNC | 524,288 / 520,192 | 524,288 / 520,192 |
| 全部动态 warp 指令 | 367,136,768 | 394,256,384（+7.39%） |
| 寄存器/thread | 105 | 79 |
| 理论 occupancy | 33.33% | 41.67% |

B 的 `LDG.128 + STS.128` 被一条 `LDGSTS...128` 替换，但 commit/wait、状态维护及控制流也增加。两版 global load sectors 都是 33,554,432，计算阶段的 shared actual/ideal wavefront 都是 67,108,864；A 的标量 shared 写入均为测得的 4-way。hybrid 改变搬运机制和寄存器压力，没有减少 GEMM 的计算量或这些请求总量。

hybrid 另有 3,145,728 次计入 issued 统计的 `@!PT LDS RZ,[RZ]`，对应实际/理想 wavefront 均为零，**不是新增的 shared 读取流量**。同样，动态指令增加 7.39% 不等于耗时必然增加 7.39%。这里只能提出有证据的解释：硬件异步节省的搬运工作，可能被控制开销、原有软件预取已经隐藏的延迟和其他资源限制抵消。

## 4. 复现当前版与历史变体

从仓库根目录，在已初始化 MSVC/CUDA 的终端中，用 CUDA 13.2、Release、`sm_89` 构建；下例使用新 build 目录，采用本页八参数默认值。不同 GPU、工具链、功耗和时钟下绝对成绩会变。

```bat
cmake -S . -B build/replay-20261023 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_WARP_RUNTIME_CONFIGS=ON -DBUILD_TESTING=ON
cmake --build build/replay-20261023 --target gemm
ctest --test-dir build/replay-20261023 -R "asyncpipeline_|doublebuffer_.*correctness|doublebuffer_.*boundary|cli_contract" --output-on-failure -j 1
build\replay-20261023\gemm.exe --mode bench --kernel asyncpipeline --m 2048 --n 2048 --k 2048 --seed 42 --warmup 50 --iters 50 --rounds 5 --csv results\replay-asyncpipeline.csv
build\replay-20261023\gemm.exe --mode bench --kernel doublebuffer --m 2048 --n 2048 --k 2048 --seed 42 --warmup 50 --iters 50 --rounds 5 --csv results\replay-doublebuffer.csv
build\replay-20261023\gemm.exe --mode bench --kernel doublebuffer_opt --m 2048 --n 2048 --k 2048 --seed 42 --warmup 50 --iters 50 --rounds 5 --csv results\replay-doublebuffer-opt.csv
```

复测应保留正反顺序的多次运行。`doublebuffer_opt` 的普通 benchmark 命令是补做可量化对照的入口，不代表本文已经有该命令的历史计时结果。

历史变体仅需在**单独的 clone/实验工作副本**中，用对应快照替换 `src/asyncpipeline.cu`，其余当前 CMake、`.cuh`、生成配置和注册入口可复用。例如在 PowerShell 中：

```powershell
Copy-Item results/published/asyncpipeline-20261023/sources/asyncpipeline-a4-scalar-async.cu src/asyncpipeline.cu
cmake --build build/replay-20261023 --target gemm
build/replay-20261023/gemm.exe --mode check --kernel asyncpipeline
```

替换为 `asyncpipeline-scalar-async.cu` 可重建最初版本；替换为 `asyncpipeline-float4-hybrid.cu` 可恢复最终版本。继续使用同一套 bench 命令并换 CSV 文件名保存各版。八参数可通过 `GEMM_ASYNC_*` / `GEMM_DB_*` CMake 配置显式设置；本页不增加重复调参工具。

快照保证单个 kernel 文件的历史字节内容，**没有归档每次实验的整个 harness**。CSV 的 `source_sha256` 是当时 CMake 所有输入和参数的组合指纹，不是 `.cu` 文件哈希；当前框架更新、换行和工具链变化会使重建的组合指纹不同。精确对应关系见 [provenance.json](provenance.json)。

历史检查：最终 hybrid 的三项 CTest 通过，129×193×67 的 memcheck/racecheck 为零错误/零 hazard；`doublebuffer_opt` 当时18项 CTest及同形状memcheck/racecheck/synccheck通过。测试覆盖并不证明任意模板组合都正确，修改参数后仍应重新 check。
