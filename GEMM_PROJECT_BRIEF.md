# CUDA GEMM 工程任务总览与 Codex 启动说明

日期：2026-10-14。项目代号：`cuda-gemm-lab`。预计周期：7–10 天。

## 1 项目目标

开发一个单 GPU 的 CUDA GEMM 工程，完成多版 kernel 实现、正确性验证、同精度 cuBLAS 对照、可靠性能测量和基于 GPU 架构的瓶颈分析。最终交付可复现代码与性能研究记录，支持算子开发岗位的项目展示和技术讨论。

本文件定义工程范围、接口和交付标准。项目沿连续版本推进；优化是否有效由真实测量决定。没有目标 GPU 或工具时，静态检查、源码分析和待运行事项分别记录。

## 2 已确定的技术范围

| 项目 | 约定 |
|---|---|
| 开发语言 | C++17、CUDA C++ |
| 主要依赖 | CUDA Runtime、cuBLAS；正式构建入口采用 CMake |
| 初始目标环境 | Windows 原生、MSVC、RTX 4060 Laptop 约 8 GB；来自既有 env 记录，首轮重新核验 |
| 执行方式 | 单 GPU、单 stream；显式设备内存分配 |
| 算子 | $C_{M\times N}=A_{M\times K}B_{K\times N}$ |
| 布局 | 连续 row-major，无转置；`lda=K, ldb=N, ldc=N` 为自写接口的行跨度 |
| 标量参数 | 固定 $\alpha=1,\beta=0$，覆盖输出 C，不读取其旧值 |
| 尺寸与边界 | 支持正整数 M/N/K 和非整 tile 尾部；输入合法、设备内存足够，分配字节数与索引计算检查溢出 |
| 内存约束 | A、B、C 为有效且不重叠的设备缓冲区；生命周期覆盖异步操作 |
| 核心数值路径 | FP32 输入、FP32 累加、FP32 输出 |
| 10 天版扩展 | FP16 输入、FP32 累加和输出的 WMMA/Tensor Core 路径 |

非法尺寸、未知实现、暂不支持的 dtype/实现组合，应明确报错并以非零退出码结束。NaN/Inf 传播、特殊舍入、零维 GEMM 和完整 BLAS 兼容性不属于本轮契约；规定的测试输入均为有限数值。

本轮不包含：多 GPU、通信、任意 stride/transpose/batched GEMM、PyTorch/Triton 接入、完整自动调参平台、跨 GPU 架构性能适配、完整 CUTLASS 移植。

## 3 工程交付物

建立一个仓库和一个可执行程序 `gemm`，使用统一入口选择不同实现。优先复用已有仓库结构，避免无关重构；新仓库的最小文件组织如下。

| 路径 | 职责 |
|---|---|
| `CMakeLists.txt` | CUDA/C++ 构建、目标架构参数、cuBLAS 链接、Release 配置 |
| `src/gemm.cu` | 初期集中放参数处理、设备查询、kernel、检查与计时 |
| `src/kernels.cuh` | 实现增多后按需拆出 kernel；首轮不要求拆分 |
| `README.md` | 契约、依赖、构建和运行命令、结果入口 |
| `REPORT.md` | 环境、版本演进、性能证据、瓶颈结论与局限 |
| `results/` | 关键计时记录、资源报告和必要 profiler 文件 |

无需为本项目新增通用插件架构、复杂配置系统或报告生成平台。简单 CSV 和 Markdown 表即可保存结果；大体积 profiler 文件保留关键案例及其索引，不默认全部提交进 git。

## 4 实现路线和里程碑

### M0 可运行基线

交付设备/工具检查、naive GEMM、CPU double 参考、FP32 cuBLAS 基线、正确性检查、CUDA Event 计时和最小命令行。

完成条件：在目标机器实际编译和运行；小型用例与参考结果一致；性能计时边界清楚；保存设备、工具链与构建信息。尽早尝试最小 NCU 采集，明确工具是否真正可用。

### M1 FP32 SIMT 实现链

| 实现名 | 改动范围 | 保留的证据 |
|---|---|---|
| `naive` | 每线程计算一个输出元素 | 索引、边界、基线正确性和耗时 |
| `mapped` | 线程到行列的映射对照 | warp 地址分布与访问行为；若 naive 已有合适映射，不强制制造一个升级版 |
| `shared` | block 级 A/B 分块复用 | cooperative load、同步、尾部、shared 用量 |
| `register` | 每线程多个输出与累加器 | 映射、寄存器/shared 预算和实际资源报告 |
| `tuned` | 根据瓶颈证据选择一次布局或资源优化 | 改动假设、计时、目标指标、副作用与适用边界 |

所有版本复用相同的数据生成、参考比较和 benchmark 代码。中间实验可保留为可回溯提交；最终验收列出的 naive/shared/register/优化实现必须同时能通过 CLI 独立选择，不覆盖掉用于比较的基线。每次优化只改变一个主要变量。

### M2 Profiling 和瓶颈分析

与 M1 并行推进，不等所有实现完成后才采集。

交付至少一份 Nsight Systems 时间线、一份 Nsight Compute 硬件报告，以及两条完整分析：问题/竞争解释 → 代码改动 → 正确性 → 独立计时 → 硬件或源码证据 → 结论。

至少保留一次无收益/退化尝试，或一个清楚的适用边界。分析覆盖当前实际相关的访存、复用、寄存器/shared 资源、调度、指令依赖或工作量不足；不要求采集所有指标。

### M3 Tensor Core 扩展

10 天版进入此阶段。实现 `wmma`，先从小矩阵验证，再运行合适的代表性 workload。自写与 cuBLAS 使用完全相同的 FP16 输入、FP32 累加和输出；精度组单独报告。

fast path 必须检查 shape、对齐和布局条件。不能直接处理的输入走明确的 SIMT fallback：读取同一份量化后的 FP16 数据、转 FP32 计算并输出 FP32。打印请求实现、实际实现及 fallback 原因；禁止静默改用原始 FP32 输入。

用实际指令或 profiler 证据确认矩阵计算路径。双缓冲/异步拷贝与小 M 特化按剩余瓶颈二选一；核心实现或 profiling 尚有缺口时优先补缺。

### M4 复现和收敛

完成构建入口、固定 workload 的结果表、关键报告索引和最终分析。以干净构建复现代表性结果，检查一个此前未用过的小型边界输入或自主修改的 tile 配置。

7 天版以 M0、M1、M2、M4 收尾；10 天版加入 M3。M2 的工具受阻状态单列，不把文档阅读算作硬件采集完成。性能目标以可解释且可复现的结果为准，不预设达到 cuBLAS 的固定比例。

## 5 程序入口约定

以下是待实现的 CLI 设计，当前不是已有可运行程序。

| 模式 | 行为 |
|---|---|
| `info` | 打印设备、运行时、库版本与可获取的资源限制 |
| `check` | 运行所选实现并比较参考结果；失败非零退出 |
| `bench` | 在计时区间外验证正确性，再做独立 Event 计时；失败不进入性能排名 |
| `profile` | 只运行所选目标实现、固定预热和目标 launch；参考计算不混入采集 |

公共参数：`--mode`、`--kernel`、`--dtype`、`--m`、`--n`、`--k`、`--seed`。计时参数：`--warmup`、`--iters`、`--rounds`。初始仅实现已存在的 kernel 和 FP32 路径；未实现的功能报错，不创建虚假占位结果。

`check` 未指定 M/N/K 时运行固定正确性集合，同时指定三项时运行该单例；部分指定报错。`bench/profile` 要求完整指定 M/N/K。允许 `--kernel cublas`；`bench` 默认自动测量所选实现和同条件 cuBLAS，输出两条结果及比值；若选择 cublas 本身则只输出基线行，不重复测量。检查 cuBLAS 本身时以 CPU double 为独立参考，小尺寸全量、大尺寸按固定位置抽样，不与自身结果比较。

```powershell
.\gemm.exe --mode info
.\gemm.exe --mode check --kernel naive --dtype fp32 --m 65 --n 97 --k 33 --seed 42
.\gemm.exe --mode bench --kernel naive --dtype fp32 --m 2048 --n 2048 --k 2048 --warmup 20 --iters 100 --rounds 5 --seed 42
.\gemm.exe --mode profile --kernel naive --dtype fp32 --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1 --seed 42
```

`profile` 默认只有一轮；上述配置发出 20 次匹配目标的预热和 1 次目标 launch。采集前以同一代码、shape、dtype 和 seed 单独通过 `check`。自写 kernel 可统一使用 `sgemm_`/`hgemm_` 前缀，便于过滤，最终命令以真实符号为准。

## 6 Workload 和正确性要求

### 固定性能集合

| 编号 | M | N | K | 用途 |
|---|---:|---:|---:|---|
| P1 | 2048 | 2048 | 2048 | 规则吞吐与复用 |
| P2 | 16 | 4096 | 4096 | 小 M、并行供给与 tile 浪费 |
| P3 | 1001 | 769 | 513 | 尾部和 fast/fallback 边界 |

默认 seed=42，均匀生成 $[-1,1]$ 输入，生成方式纳入代码版本。M0 可根据目标机器的实际限制调整同类规模一次，记录原因并锁定；后续不扩大性能集合。

### 正确性集合与判据

小型用例至少覆盖 `(1,1,1)`、`(2,2,3)`、`(17,19,23)`、`(65,97,33)`；再按当前 tile 加入少量边界检查。包含零矩阵、单位矩阵、小整数和带正负号输入。小矩阵全量使用 CPU double 累加参考；性能输入全量与同精度 cuBLAS 比较，抽取固定输出位置用 CPU double 点积复核布局。

对项目规定的输入范围且 $K\le4096$，先固定如下工程验收容限：

| 路径 | 参考输入 | 元素级误差判据 |
|---|---|---|
| FP32 | 实际 FP32 输入值 | `abs(error) <= 1e-3 + 1e-4*abs(reference)` |
| FP16/FP32 | 已量化的 FP16 输入值转 double | `abs(error) <= 2e-3 + 2e-4*abs(reference)` |

这些是本项目受控测试的初始标准，不能推广为任意输入的数值保证。改精度或扩大 K/数值范围时先修订数值契约并说明理由，不为掩盖错误临时放宽容限。FP16 输入量化误差另与原始 FP32 结果比较，不能混入 kernel 计算误差。

所有输出必须为有限值；`check` 可先用 NaN 填充 C 检测漏写。报告最大绝对误差、相对 Frobenius 范数误差和失败元素数；参考范数为零时明确定义分母保护。检查失败以非零退出码结束，性能行标记 invalid，不输出有效加速比。

CUDA/cuBLAS 返回值、launch 错误和异步执行错误均检查。涉及 shared/sync 改动，在小型输入上运行 Compute Sanitizer 的 memcheck、racecheck、synccheck；记录通过、失败或工具不可用，不用它替代数值比较。

## 7 cuBLAS 和性能比较契约

FP32 基线固定使用 `cublasGemmEx`，A/B/C 为 `CUDA_R_32F`，计算类型为 `CUBLAS_COMPUTE_32F_PEDANTIC`。记录 math mode、库版本和参数，不把允许不同输入精度的快速路径混入同一基准组。

经典 cuBLAS 采用 column-major。对本项目连续 row-major 内存，调用计算 $C^T=B^TA^T$：`m=N,n=M,k=K`，两个 op 为 N，第一个指针 B、leading dimension N；第二个指针 A、leading dimension K；输出 C、leading dimension N。此处是布局解释，不插入真实转置 kernel。

FP16 组采用 FP16 A/B、FP32 C、FP32 compute 的对应 GemmEx 配置。输入转换在主计时区间外；若报告应用端到端代价，转换、padding、fallback 等成本另行明确。基线 handle 的 stream 与目标实现一致。

主指标为预分配设备内存、数据已在 GPU 上、同一工作集重复使用的 Event 时间：

- Release 优化构建；保留源码定位信息，不用 `-G` 构建声称性能结果。
- 同 stream 的 start Event → R 次目标操作 → stop Event；等待 stop 后读取毫秒并除以 R。
- 分配、复制、转换、初始化、参考计算、验证、handle 创建不计入主区间。
- 初始 20 次预热、100 次重复、5 轮；如批量过短或过长，可按同一规则调整到约 50–200 ms 并记录 R。首版支持手工调 R 即可，不必先做自动校准系统。
- 汇报各轮均值样本的中位数与 min/max；不把它标成请求 p95。
- 记录电源模式、可得频率/温度与明显后台干扰；对照交替运行。收益被波动覆盖时，不宣称稳定加速。

统一公式：$GFLOP/s=2MNK/(10^6t_{ms})$；相对库性能为 $t_{cublas}/t_{self}$。它不等于理论峰值利用率。极小 workload 的提交间隙、稳态缓存和 profiler 的 replay/cache/clock 设置要在分析中注明。

Nsight Systems 用于进程时间线，Nsight Compute 用于 kernel 硬件诊断；二者与正常 Event 计时分别保存。比较 NCU 结果时保持采集条件一致，并依据本机帮助核对 section 和过滤规则。

## 8 结果记录的最小内容

环境、编译与版本元数据每次关键运行记录一次；每个性能行记录 kernel、实际路径、M/N/K、输入/累加/输出 dtype、seed、正确性误差、warmup/iters/rounds、时间统计、GFLOP/s 和同条件 cuBLAS 比例。不存在或未测的字段写 `NA/未测`，不填推测数字。

关键优化另记录 grid/block、tile 参数、registers/thread、shared/block，以及支持结论的少量指标。原始报告与源码 commit 或明确的未提交 diff 关联；理论估算、源码事实、实际测量、推断分别标注。

每个性能结论必须回答：改了什么；为什么预期有效；保持哪些条件相同；实际测得什么；证据支持到哪里；其他输入上有什么代价。

## 9 Codex 协作方式

1. 先阅读仓库已有 `AGENTS.md`、构建文件与相关代码，保留用户改动。已有功能优先复用，不从空工程假设出发覆盖文件。
2. 按当前里程碑交付一个小闭环。可自主完成已授权范围内的实现、修复和验证；不要把所有优化一次性写完。
3. 每次优化先保存基线，再实施可归因改动。正确性失败优先修复；性能未改善可以保留为负面结果。
4. 每轮汇报改动文件、实现动机、运行命令、真实验证状态、结果解释和下一项具体任务。简明报告，不扩成教学课件。
5. 以实际设备确定架构和编译目标。既有 env 中的 Windows/RTX 4060 Laptop 信息不是本次已验证事实；不把其他 GPU 的 SM/L2/指令能力代入。
6. 如果 Codex 执行环境不能访问目标 GPU，继续完成可做的代码、构建配置与静态检查，给出最少的目标机命令；GPU 正确性、性能和 counters 保持“待运行”。仅在需要目标输出才能继续判断时请求该输出。
7. 本地实现或 API 细节遵循 `DAILY_SOURCE_READING_GUIDE.md` 的证据原则：文档用于语义，固定版本源码用于实现，运行结果用于本次执行。没有读到的闭源库内部实现不补写为事实。

## 10 给 Codex 的首次执行任务

**本轮范围：完成 M0，建立后续优化共用的工程基础。**

请先检查当前仓库和实际执行环境，然后实施：

1. 核对 GPU、compute capability、NVCC、host compiler、CUDA Runtime/Driver、cuBLAS 与 profiler 可用性；结果写入项目记录。
2. 建立最小 CMake 构建和 `gemm` 可执行程序，架构目标可配置，按已核验设备设置；仅用必要依赖。
3. 实现 FP32 row-major naive GEMM，覆盖非整除边界，配备 CUDA/cuBLAS 错误检查和资源释放。
4. 实现 CPU double 参考、固定正确性用例和 FP32 cuBLAS 基线，核对 row-major 参数转换。
5. 实现 `info/check/bench/profile` 四种模式及本阶段所需参数，使用统一数据生成和 Event 计时。
6. 从小输入开始实际编译、检查，再运行代表性 benchmark；若目标 GPU 可访问，尝试一次仅针对 naive 的最小 NCU 采集。
7. 更新 README 与 REPORT，记录实际命令、结果和阻塞。若无目标 GPU，交付可用代码及待运行命令，不编造实测记录。

M0 暂不实现 shared/register/WMMA 优化。交付时给出当前基线状态、重要代码入口和下一轮 M1 的第一项具体改动依据。

## 11 最终完成条件

- [ ] 目标机器可以按文档构建并运行工程。
- [ ] FP32 naive/shared/register 及一项优化实现能独立选择，并通过统一正确性检查。
- [ ] 三个固定性能输入完成同精度 cuBLAS 对照，计时边界、统计和波动清楚。
- [ ] 实际取得并解释 NSYS 时间线与 NCU 硬件数据；受阻部分单列为尚未完成。
- [ ] 至少两条可追溯的优化分析，包含一次失败尝试或明确适用边界。
- [ ] 10 天版包含已运行的 WMMA 实现、数值检查及真实路径证据；条件扩展如实记录。
- [ ] 代码、版本、命令、原始证据与报告可以相互对应；独立修改与边界处理得到验证。

工程技术依据沿用已核对的 NVIDIA [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)、[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)、[Nsight Compute 指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)与 [Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。正式实现时以目标机工具版本和相应文档核对；本任务书不代表代码已实现或环境已验证。
