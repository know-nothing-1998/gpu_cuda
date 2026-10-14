# CUDA GEMM Lab

用于学习和优化 CUDA 矩阵乘法的实验项目：从手写 naive SGEMM 出发，在统一的正确性检查、cuBLAS 对照、固定 benchmark 矩阵和 profiling 入口下，逐步研究数据复用、线程布局与 GPU 资源利用。

当前包含手写 FP32 naive kernel、严格 FP32 cuBLAS 基线、CPU double 参考、CUDA Event 计时和结果导出。后续计划加入 shared memory 分块、寄存器分块和 Tensor Core 实现；这些优化版本尚未实现。路线见 [项目任务书](GEMM_PROJECT_BRIEF.md)。

初始化版本验证：Release 构建、CTest 4/4、naive 32 个用例和边界 memcheck 均通过，P1/P2/P3 已完成成对运行。命令、日志和测量条件见 [2026-10-14 验证记录](results/initial-20261014/README.md)。

## 快速开始

已验证环境：Windows、RTX 4060 Laptop（SM 8.9）、CUDA Toolkit 13.2、Visual Studio 2022 / MSVC 19.44。需要 CMake 3.24+ 和 MSVC x64 工具；Python 用于 CLI 集成测试。CUDA 工具和运行时 DLL 需要在 PATH 中。

在 **x64 Native Tools Command Prompt** 或本项目已配置的 **MSVC x64** VS Code 终端中进入项目根目录。以下示例使用 Windows CMD 语法。

首次配置并编译：

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

以后修改 `.cu` 后，只需执行 `cmake --build build`。CMake 将 `src/gemm.cu` 和 `src/naive.cu` 编译、链接为 `build\gemm.exe`，并链接 CUDA Runtime 和 cuBLAS。

查看环境、检查 naive、比较性能：

```bat
build\gemm.exe --mode info
build\gemm.exe --mode check --kernel naive
build\gemm.exe --mode bench --kernel naive --m 2048 --n 2048 --k 2048 --iters 10
```

`bench` 先检查数值，再分别测量 naive 和 cuBLAS。`--iters 10` 表示每轮每个实现重复 10 次，默认预热 20 次、测量 5 轮。

其他 GPU 需调整 `CMAKE_CUDA_ARCHITECTURES`。构建采用 Release、`-lineinfo` 和 ptxas 资源报告，不启用 `-G` 或 `--use_fast_math`。也可使用自动寻找 MSVC 的脚本：

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1
```

`.vscode/msvc-terminal.cmd` 中的 Visual Studio 路径是开发机配置，在其他机器上使用该终端前需调整。构建脚本会通过 `vswhere` 查找安装位置。

## 实现与程序入口

计算契约为 `C[M,N] = A[M,K] * B[K,N]`：连续 row-major、无转置、FP32 输入/累加/输出、固定 `alpha=1,beta=0`。当前验收范围为正整数尺寸且 `K <= 4096`。

| 实现 | 当前行为 |
|---|---|
| `naive` | 每线程计算一个 C 元素；block 为 `(32,32)`，网格向上取整并检查行列边界 |
| `cublas` | `cublasGemmEx`，使用 `CUBLAS_COMPUTE_32F_PEDANTIC` 和 `CUBLAS_PEDANTIC_MATH` |

naive kernel 位于 [src/naive.cu](src/naive.cu)。`launch_naive` 在传入的 CUDA stream 上启动 kernel 并返回 launch 状态；分配、复制、校验和计时由 [src/gemm.cu](src/gemm.cu) 负责。

CLI（命令行接口）通过参数选择操作：

| `--mode` | 用途 |
|---|---|
| `info` | 设备、资源限制、工具链、源码摘要和可用实现 |
| `check` | 不传尺寸时运行 32 个固定用例；传入 M/N/K 时检查单个随机输入 |
| `bench` | 正确性检查、预热、Event 计时；naive 自动与 cuBLAS 成对比较 |
| `profile` | 只运行选定实现，默认 20 次预热和 1 次目标调用；不执行正确性检查 |

常用参数为 `--kernel naive|cublas`、`--m`、`--n`、`--k`、`--seed`、`--warmup`、`--iters`、`--rounds`。目前 `--dtype` 仅支持 `fp32`；`profile` 仅允许一轮。`bench` 可通过 `--csv NEW_FILE` 保存单次结果，已有文件不会被覆盖。

## 测试

```bat
ctest --test-dir build --output-on-failure
build\gemm.exe --mode check --kernel naive
build\gemm.exe --mode check --kernel naive --m 65 --n 97 --k 33
compute-sanitizer --tool memcheck build\gemm.exe --mode check --kernel naive --m 65 --n 97 --k 33
```

CTest 注册集中在 [CMakeLists.txt](CMakeLists.txt) 的 `BUILD_TESTING` 块：

| 测试 | 检查内容 |
|---|---|
| `numerical_harness` | 手算矩阵、错误/转置结果、NaN/Inf、容限、溢出、随机数和统计，见 `tests/harness_tests.cu` |
| `cublas_correctness` | cuBLAS 对 CPU double 的 32 个固定用例 |
| `cublas_unseen_boundary` | 独立非方阵边界 `(31,37,29)` |
| `cli_contract` | 非法参数、CSV 防覆盖、CPU 参考和 profile 隔离，见 `tests/cli_tests.py`；找到 Python 时启用 |

**当前 CTest 不包含 naive 完整正确性套件，需另行执行 `--mode check --kernel naive`。** 固定套件覆盖 8 个 shape，每个检查随机、零 A、单位/矩形单位 A、小整数四种输入。

输入采用固定 seed 的 `mt19937_u24_v1`，范围为 FP32 `[-1,1)`。当 `M*N <= 65536` 且 `M*N*K <= 8,000,000` 时使用 CPU double 全量参考；否则使用固定位置抽样。naive 另与 cuBLAS 全量比较。所有输出检查有限性，并通过预填 NaN 检测漏写。

容限为 `abs(error) <= 1e-3 + 1e-4*abs(reference)`。退出码：0 成功、2 数值不通过、1 参数或运行错误。失败用例不进入性能计时。

## 固定 benchmark 与结果阅读

性能矩阵定义在 [benchmarks/matrix.csv](benchmarks/matrix.csv)，校准记录见 [矩阵说明](benchmarks/README.md)。

| ID | M | N | K | 默认 iters | 用途 |
|---|---:|---:|---:|---:|---|
| P1 | 2048 | 2048 | 2048 | 44 | 规则吞吐和数据复用 |
| P2 | 16 | 4096 | 4096 | 335 | 小 M、并行供给和 tile 利用 |
| P3 | 1001 | 769 | 513 | 800 | 非整 tile 尾部 |

统一 FP32、seed=42、warmup=20、rounds=5。默认重复次数按初始 cuBLAS 预跑校准；naive 可覆盖次数以缩短运行时间，每次比较的两个实现使用相同次数。

运行整套矩阵并保存结果：

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Kernel naive -Iters 10
```

脚本创建新的 `results/run-*` 目录，保存各 case 的 CSV、日志、汇总表、命令、矩阵、环境、GPU 状态和源码/程序哈希。也可用 `-OutputDir` 指定新的输出目录，已有目录不会被覆盖。

| 字段 | 含义 |
|---|---|
| `status`、`failed`、`nonfinite` | 先确认结果有效、失败和非有限值计数为 0 |
| `median_ms` | 各轮单次平均耗时的中位数，越小越快 |
| `min_ms`、`max_ms`、`round_mean_ms` | 轮间波动及各轮原始均值 |
| `gflops` | `2*M*N*K/(1e6*median_ms)`，越大越快 |
| `cublas_over_self` | `t_cublas/t_self`；naive 为 0.1 表示耗时约为 cuBLAS 的 10 倍 |
| `reference_scope`、`cpu_scope` | 区分参考来源、CPU 全量或抽样；抽样误差不能视为整矩阵误差 |
| `source_sha256` | 将结果与实际编译源码关联 |

CUDA Event 区间只包围同一 stream 上的重复 GEMM 调用，不包含分配、拷贝、参考计算和校验。两种实现交替测量，使用同一工作集。时间是每轮平均后取中位数，不是逐次延迟的 p95。

## Profiling

先对相同 shape 执行 `check`，再采集选定 kernel：

```bat
build\gemm.exe --mode check --kernel naive --m 2048 --n 2048 --k 2048
ncu --set full --launch-skip 20 --launch-count 1 --export results\naive-p1-new build\gemm.exe --mode profile --kernel naive --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1
ncu-ui
```

命令适用于已配置好 PATH 和 GPU 性能计数器权限的环境。`--launch-skip 20` 跳过预热，只采集之后的一次 kernel；输出文件名应保持唯一。`--set full` 会通过重放采集多个指标，运行时间比普通 benchmark 长。

**本机 Nsight Compute 2026.1.1 图形界面在中文用户名账户下启动崩溃，开发者已确认使用纯英文用户名账户可以正常启动。后续在该英文账户中运行 `ncu-ui`，通过 File → Open 打开报告。** 这是本机验证过的使用方式，不表示已确定工具内部根因。无需继续修改原账户的 Nsight 用户配置。

CLI 在中文临时目录下启动异常时，本机曾使用以下仅影响当前 CMD 会话的设置完成采集；英文账户环境优先使用上面的直接命令：

```bat
mkdir build\profiler-temp
set "TEMP=D:\gpu_cuda\build\profiler-temp"
set "TMP=%TEMP%"
set "NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1"
ncu --config-file off --section-folder "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\sections" --set full --launch-skip 20 --launch-count 1 --export results\naive-p1-new build\gemm.exe --mode profile --kernel naive --m 2048 --n 2048 --k 2048 --warmup 20 --iters 1
```

路径需按本机安装位置调整，临时目录已存在时跳过 `mkdir`。若出现 `ERR_NVGPUCTRPERM`，按 [NVIDIA 权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM) 配置，或从管理员 CMD 采集。

报告先看 Speed Of Light、Launch Statistics、Occupancy、Memory Workload Analysis，再结合 Scheduler/Warp State 判断等待原因。Memory Throughput 不等于 DRAM 带宽，occupancy 也不是算力利用率。优化效果使用正常 `bench` 时间评估，profiling 指标用于解释原因，不混合两种采集条件下的耗时。

## 文件与阶段记录

| 路径 | 内容 |
|---|---|
| `src/naive.cu`、`src/naive.cuh` | 手写 naive kernel 和启动接口 |
| `src/gemm.cu` | CLI、资源管理、数据生成、参考检查、cuBLAS、计时和 CSV |
| `CMakeLists.txt` | 编译、链接、源码摘要和 CTest 注册 |
| `tests/` | 数值检查框架与 CLI 集成测试 |
| `benchmarks/` | 固定性能矩阵和重复次数校准说明 |
| `scripts/` | 可选构建脚本和批量 benchmark 脚本 |
| `results/` | 可复核的文本日志和 CSV；大型 profiler 二进制默认不提交 |
| [REPORT.md](REPORT.md) | naive 加入前的 M0 基础设施与 cuBLAS 历史测量 |

构建目录和 profiler 二进制由 `.gitignore` 排除。源码版本通过 Git commit 管理，运行结果同时保存 CMake 生成的源码 SHA256。性能报告应注明矩阵、编译配置、工具版本和实际测量条件。
