# CUDA GEMM Lab

从 naive 到 warp tiling、double buffer 和异步流水线的手写 CUDA SGEMM 学习项目。各实现共用 CPU double 正确性检查、严格 FP32 cuBLAS 对照、CUDA Event 计时和固定 benchmark 矩阵。

计算契约：`C[M,N] = A[M,K] * B[K,N]`，连续 row-major，FP32 输入/累加/输出，`alpha=1,beta=0`。当前验证范围为正整数尺寸、`K <= 4096`。

## 实现路线

| 实现 | 主要变化 | 源码 |
|---|---|---|
| naive | 每线程一个输出；x 对应行，访存跨行 | [naive.cu](src/naive.cu) |
| coalesced | x 对应列，使相邻 lane 访问连续 B/C | [coalesced.cu](src/coalesced.cu) |
| shared | 32×32 shared-memory tile，复用 A/B | [shared.cu](src/shared.cu) |
| register | 每线程多个输出、寄存器复用、float4 加载 | [register.cu](src/register.cu) |
| tuned | BM/BN/BK/TM/TN 五参数模板与自动搜索 | [tuned.cu](src/tuned.cu) |
| warptiled | 显式 warp 分工、转置 As、八参数搜索和选择已编译配置 | [warptiled.cu](src/warptiled.cu) |
| doublebuffer | 下一 tile 的寄存器预取、shared 双缓冲与软件流水线 | [doublebuffer.cu](src/doublebuffer.cu) |
| doublebuffer_opt | doublebuffer 的独立实验核函数：As swizzle 与 C 向量化写回 | [doublebuffer.cu](src/doublebuffer.cu) |
| asyncpipeline | A 的 float4 预取 + B 的硬件异步拷贝 + warp tiling | [asyncpipeline.cu](src/asyncpipeline.cu) |

截至 **2026-10-23** 的关键结果见 [实验索引](results/published/README.md)，包含同轮 cuBLAS 对照、原始计时数据、参数和精选历史源码。性能对应特定机器与当时源码，不保证在其他机器重现相同数值；不同日期的计时不能直接解释成单项优化收益。

最新发布的 [异步流水线实验](results/published/asyncpipeline-20261023/README.md) 记录 `doublebuffer_opt` 的资源代价、A 加载方式的两次调整，以及当前 hybrid 与 doublebuffer、同轮 cuBLAS 的 P1 对照。当前 hybrid 相比前一版约快 33%，两遍测量仍比 doublebuffer 慢约 2%；这不是所有形状上的结论。[GPU 架构讲解](GPU_ARCHITECTURE_GUIDE.md) 结合本项目说明内存路径、SASS、调度、同步与流水线。

此前的 [double buffer 实验](results/published/doublebuffer-20261022/README.md) 保留 `doublebuffer` 基线的 P1～P4 benchmark、warptiled/cuBLAS 对照、流水原理与 SASS 证据。

此前的 [七版本 Roofline 对比](results/published/roofline-20261022/README.md) 是引入 doublebuffer 前的快照：在同一构建、P1=`2048³` 下采集 naive、coalesced、shared、register、tuned、warptiled 和 cuBLAS，使用默认参数，不代表 autotune 优胜配置，也不包含 doublebuffer。可直接查看 [图](results/published/roofline-20261022/roofline.svg) 和 [指标](results/published/roofline-20261022/model.json)，或按 [Roofline 说明](ROOFLINE.md) 离线重画。

## 从 clone 开始

已验证环境：Windows、RTX 4060 Laptop（SM 8.9）、CUDA Toolkit 13.2、Visual Studio 2022/MSVC 19.44。需要 CMake 3.24+、CUDA Toolkit/cuBLAS，以及 Visual Studio C++ x64 工具。Python 3.10+ 用于测试和自动调参，仅 Roofline 绘图额外需要 Matplotlib。Linux 构建路径尚未在本项目验证。

在 **x64 Native Tools Command Prompt for VS 2022** 中执行：

```bat
git clone https://github.com/know-nothing-1998/gpu_cuda.git
cd gpu_cuda
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
build\gemm.exe --mode list
build\gemm.exe --mode info
build\gemm.exe --mode check --kernel doublebuffer
build\gemm.exe --mode bench --kernel doublebuffer --m 2048 --n 2048 --k 2048 --warmup 100 --iters 100 --rounds 5
```

其他 GPU 请调整 `CMAKE_CUDA_ARCHITECTURES`。从普通 PowerShell 构建，也可执行 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1`，脚本通过 `vswhere` 查找 MSVC；CUDA 运行时 DLL 需在 PATH 中。建议使用不含中文的项目/临时目录，避免本机曾遇到的 CUDA 工具路径问题。

修改 `.cu` 后只需增量构建：

```bat
cmake --build build --target gemm
```

`bench` 先检查输出，再交替测量选定实现与 cuBLAS；计时不含分配、传输和校验。cuBLAS 使用 `CUBLAS_COMPUTE_32F_PEDANTIC` 和 `CUBLAS_PEDANTIC_MATH`。

## 测试与 benchmark

| mode | 用途 |
|---|---|
| `list` | 列出实现及已编译 warp 配置，不初始化 CUDA |
| `info` | GPU 属性、构建信息、参数 |
| `check` | 未传 MNK 时运行 32 个固定用例；传入时检查一个 shape |
| `bench` | 检查、预热、Event 计时并与 cuBLAS 对比 |
| `profile` | 仅运行指定实现，供 NCU 采集；需单独先做 check |

```bat
cmake --build build
ctest --test-dir build --output-on-failure
build\gemm.exe --mode check --kernel register --m 65 --n 97 --k 33
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Kernel warptiled
```

完整 CTest 包含 GPU 工作负载。只检查 CPU 工具和命令行：

```bat
ctest --test-dir build -R "^(numerical_harness|runtime_warp_cli_contract|autotune_contract|autotune_cli_contract|bench_tuned_contract)$" --output-on-failure
```

[固定矩阵](benchmarks/matrix.csv)：P1=`2048×2048×2048`，P2=`16×4096×4096`，P3=`1001×769×513`，P4=`16×256×128`。用途与次数见 [矩阵说明](benchmarks/README.md)。

`bench --csv NEW_FILE` 可保存结果。先看 `status/failed/nonfinite`，再看 `median_ms/gflops/round_mean_ms`。`cublas_over_self=t_cublas/t_self`，大于 1 表示自写实现更快；GFLOP/s=`2MNK/(1e6×ms)`。配置和源码 SHA256 随结果保存。

## 手动参数与自动调优

warptiled 的八个 tile 参数必须全填或全省略；MNK 是矩阵尺寸。下面选择 P4 的一组已编译配置：

```bat
build\gemm.exe --mode bench --kernel warptiled --bm 16 --bn 32 --bk 32 --wm 8 --wn 16 --tm 2 --tn 2 --wniter 1 --m 16 --n 256 --k 128 --warmup 100 --iters 10000 --rounds 5
build\gemm.exe --mode list --kernel warptiled
```

默认配置为 `64/64/16/32/32/4/4/1`。普通构建预编译 [warptiling.json](benchmarks/warptiling.json) 中的 16 组候选与 CMake 默认配置。新组合需加入候选表或设置 `GEMM_WARP_*` 后重新编译；没有 JIT，也不会静默回退。CSV 的 `warptiled_config` 记录实际选择。线程数由 `32*(BM/WM)*(BN/WN)` 推导，不是独立参数。

tuned 的五参数通过 Python 入口传给 CMake 编译：

```bat
python scripts\bench_tuned.py --bm 16 --bn 32 --bk 32 --tm 2 --tn 2 --m 16 --n 256 --k 128 --iters 10000
```

统一自动调优入口：

```bat
python scripts\autotune.py --kernel warptiled --m 2048 --n 2048 --k 2048 --dry-run
python scripts\autotune.py --kernel warptiled --m 2048 --n 2048 --k 2048
python scripts\autotune.py --kernel tuned --cases P1 P2 P3 P4
```

脚本在独立目录编译、检查、筛选和复测，不替换普通 `build/gemm.exe`。结果写入新的 `results/autotune-*`，先读 README、ranking.csv 和 winners.json。候选范围有限，优胜不代表全局最优，也不会按 MNK 自动派发。`compare_warptiled.py` 是统一入口使用的 warp 搜索后端，通常无需单独调用。Python 独立构建关闭完整 warp 候选表，避免重复编译。

## Double buffer：寄存器预取与 shared 双缓冲

`src/doublebuffer.cu` 已实现普通 global load（SASS 的 `LDG`）配合寄存器预取的流水线。先将 tile 0 写入 shared；随后提前把下一 tile 读到 `nextA/nextB`，用当前 shared tile 完成计算，再把预取值发布到另一份 shared tile，经过 block 同步后交换读写阶段。A 在 shared 中保持转置布局，B 保持 row-major。

这里没有使用 `cp.async`，也没有专门负责搬运的 warp；同一组线程发起加载并执行独立计算。双缓冲保证计算与发布使用不同的 shared 区域，真正能隐藏多少访存延迟仍取决于指令调度、资源占用和矩阵形状。

设 `S = ceil(K/BK)`，原 warptiled 每 tile 两次 block 同步，共 `2S` 次；当前实现为预热后一次，加后续阶段切换的 `S-1` 次，共 `S` 次。最后一批不再发布下一 tile，也不需要该次同步。因此性能差异同时包含预取、双缓冲和同步减少，不能全部归因于加载与计算重叠。

默认 `BM/BN/BK/WM/WN/TM/TN/WNITER = 64/64/16/32/32/4/4/1`，128 线程、block 为 `16×8`，两份 shared tile 共 16 KiB。线程数由参数计算。修改参数使用独立的 `GEMM_DB_*` CMake 配置，例如 `cmake -S . -B build -DGEMM_DB_BK=32`；doublebuffer 尚未接入 exe 的 `--bm/--bn/...` 运行时配置选择或自动调优入口。

检查与计时：

```bat
cmake --build build --target gemm
build\gemm.exe --mode check --kernel doublebuffer
build\gemm.exe --mode check --kernel doublebuffer --m 129 --n 193 --k 67
build\gemm.exe --mode bench --kernel doublebuffer --m 2048 --n 2048 --k 2048 --warmup 100 --iters 100 --rounds 5
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Kernel doublebuffer
```

`doublebuffer_correctness` 和 `doublebuffer_boundary` 是对应的 CTest 入口；CLI 测试覆盖正确性、计时 CSV 与 profile 接口。`scripts/benchmark.ps1 -Kernel doublebuffer` 使用项目固定矩阵，逐个与 cuBLAS 配对。

[double buffer 实验报告](results/published/doublebuffer-20261022/README.md) 对应当前用户编写的正式源码，包含同轮多形状计时、原始数据及复现步骤。此前助手参考预实验保留在开发机，不作为当前实现的成绩。

同一文件中的 `sgemm_doublebuffer_opt` 是单独注册的对比版本。它在满足 `BM % 32 == 0`、`BK == 16`、`TM % 4 == 0` 且 `TM <= 8` 时对 As 使用 swizzle；其他参数保留原 As 布局。C 写回按四个相邻结果尝试 `float4` 存储，地址未对齐、输出边界或不足四个元素时使用标量写回。两版共用 `GEMM_DB_*` 参数。用户在默认配置的 P1 观察到退化；已归档的 NCU 数据显示寄存器和动态指令增加，理论 occupancy 仍同为 33.33%，不能把退化直接归因于 occupancy 降低。该对照未保留可审计的普通 benchmark CSV，因此不报告退化百分比。详见 [资源与指令分析](results/published/asyncpipeline-20261023/README.md)，该实验版本未替换 doublebuffer 基线。

分别运行下面两条 bench 即可比较，两次都会与 cuBLAS 配对：

```bat
cmake --build build --target gemm
build\gemm.exe --mode check --kernel doublebuffer_opt
build\gemm.exe --mode bench --kernel doublebuffer --m 2048 --n 2048 --k 2048 --warmup 100 --iters 100 --rounds 5
build\gemm.exe --mode bench --kernel doublebuffer_opt --m 2048 --n 2048 --k 2048 --warmup 100 --iters 100 --rounds 5
```

`doublebuffer_opt_correctness` 和 `doublebuffer_opt_boundary` 是优化版的 CTest 入口。`--mode info` 中的 `doublebuffer_opt_config` 及其计时 CSV 的 `doublebuffer_config` 列记录 `SWIZZLE_A` 和 `VECTOR_STORE_C` 编译能力开关；默认参数下均为 `1`，具体边界仍可走标量路径。使用 NCU 时将 `--kernel` 改成 `doublebuffer_opt`，并使用新的报告文件名。

## 硬件异步流水线

[asyncpipeline.cu](src/asyncpipeline.cu) 的 `sgemm_asyncpipeline` 使用混合流水线：A 通过普通 `float4` 全局读取预取到寄存器；B 使用 `cuda::memcpy_async` 和线程作用域的 `cuda::pipeline` 搬入下一份 shared buffer。计算当前块后，再将预取的 A 逐元素写入下一份 As。线程数按八个 tile 参数推导，默认 `BM/BN/BK/WM/WN/TM/TN/WNITER = 64/64/16/32/32/4/4/1`；独立的 CMake 参数前缀为 `GEMM_ASYNC_*`，两阶段 shared 预算为 16 KiB。

保留转置的 `As[2][BK][BM]`：A 对齐且完整时一次读取四个 float，再通过四次标量 shared store 完成转置；未对齐或尾部走标量读取并补零。B 对齐且完整时使用 16 字节硬件异步拷贝，否则逐元素拷贝或补零。`consumer_wait()` 等待本线程的 B 拷贝，随后 `__syncthreads()` 同时保证 A/B 数据就绪及旧 buffer 的读取结束。配置字符串的 `A_FLOAT4_PREFETCH=1;B_ASYNC=1` 标明这条混合路径；A 的预取属于软件流水线，B 的硬件异步拷贝需要 SM 80+。

实现已启用，可以直接执行：

```bat
cmake --build build --target gemm
build\gemm.exe --mode check --kernel asyncpipeline
build\gemm.exe --mode check --kernel asyncpipeline --m 129 --n 193 --k 67
build\gemm.exe --mode bench --kernel asyncpipeline --m 2048 --n 2048 --k 2048 --warmup 100 --iters 100 --rounds 5
```

`asyncpipeline_correctness` / `asyncpipeline_boundary` 以及 CLI 检查已接入。八参数通过 CMake 设置，exe 的 `--bm/...` 参数仍仅用于 warptiled。

## Profiling

安装 Nsight Compute 并启用 GPU performance counters 权限后：

```bat
build\gemm.exe --mode check --kernel doublebuffer --m 2048 --n 2048 --k 2048
ncu --set full --launch-skip 20 --launch-count 1 --export results\doublebuffer-p1-new build\gemm.exe --mode profile --kernel doublebuffer --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1
ncu-ui
```

先看 Launch Statistics、Occupancy、Memory Workload Analysis，再结合指令和 warp 等待。用普通 `bench` 判断性能变化，NCU 用于解释原因。本机 NCU 2026.1.1 GUI 在中文用户名下启动异常，使用纯英文用户名账户可以启动；这不是所有环境的通用要求。

## 仓库与证据

- [实验汇总](REPORT.md) / [精选结果](results/published/README.md)：从 naive 到异步流水线的关键记录。
- [GPU 资源速查](GPU_DEVICE.md) / [Roofline](ROOFLINE.md)：本机资源、公式和采样口径。
- [从手写 GEMM 理解 GPU 架构](GPU_ARCHITECTURE_GUIDE.md)：内存路径、指令、bank conflict、warp 调度、驻留、流水线和官方阅读路线。
- [项目任务书](GEMM_PROJECT_BRIEF.md) / [分支工作流](CONTRIBUTING.md)：阶段目标、计算契约与 Git Flow。
- `src/`：kernel 与公共 harness；`scripts/`：构建、测量和调参工具；`tests/`：数值及接口检查。
- `results/published/`：精选实验及各采样版本的 Roofline 证据。其余 `results/`、build、NCU 二进制、临时文件默认忽略；运行产生的新结果不会自动进入 Git。
- `.clang-format` 和 `.editorconfig`：四空格缩进。VS Code 使用 C/C++ 扩展，按需将 [settings.example.json](.vscode/settings.example.json) 复制为 `settings.json`；[IntelliSense 示例](.vscode/c_cpp_properties.example.json) 依赖从 MSVC x64 开发环境启动的 VS Code，或自行填入本机编译器路径。个人 IDE 配置保留本地。

实现学习参考 Simon Boehm 的 SGEMM 优化系列；第三方许可见 [third_party](third_party/siboehm-SGEMM-CUDA-LICENSE.txt)。历史参考实现与当前用户编写的 kernel 在精选报告中分别说明。
