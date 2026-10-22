# Double buffer：当前实现、流水证据与四组 benchmark

2026-10-22，RTX 4060 Laptop / SM 8.9。比较当前正式 `doublebuffer` 与同次构建的 `warptiled`，两者使用相同的默认八参数，**不是各自 autotune 最优配置的比较**。P2/P3/P4 的吞吐量分别提高约 42%/8%/46%；P1 复测有快有慢，尚不能确认稳定收益。

当前源码：[doublebuffer.cu](../../../src/doublebuffer.cu)、[接口](../../../src/doublebuffer.cuh)、[CMake](../../../CMakeLists.txt)。测试对应的精确源码保存在 [source-snapshot.zip](source-snapshot.zip)，不是此前助手参考预实验。旧预实验保留在开发机，不混入本次成绩。

## 1. 两份缓冲区、软件流水、硬件异步分别是什么

**Double buffer 指两份交替使用的缓冲区。** 本实现是 `As[2][BK][BM]`、`Bs[2][BK][BN]`：一份供当前 tile 计算，另一份接收下一 tile。它提供安全的读写分离，单凭数组大小不能证明实际的访存与计算重叠。

**软件流水指把相邻迭代的工作穿插安排。** 这里先用普通 global load 把下一 tile 读到每线程的 `nextA/nextB` 寄存器，再计算当前 shared tile，最后把寄存器中的下一 tile 发布到另一份 shared。顺序如下，`publish` 是源码中的 lambda，并不是 CUDA 特殊指令：

```cpp
prefetch(0);                    // global -> nextA/nextB
publish(0);                     // nextA/nextB -> shared[0]
__syncthreads();                 // 首批可供整个 block 读取

for (每个 K tile p) {
    if (有下一批) prefetch(p + BK);
    compute(shared[read_stage]); // 当前批；不依赖下一批的寄存器结果
    if (有下一批) {
        publish(read_stage ^ 1);
        __syncthreads();
        read_stage ^= 1;
    }
}
```

**显式硬件异步拷贝是另一种机制。** 当前实现没有 `cp.async` / `LDGSTS`，也没有把 warp 分成独立搬运组和计算组；同一组线程执行上述工作。普通 LDG 发出后，只要后面的计算不消费它的目标寄存器，机器码可以把结果等待延后。这是当前软件流水可隐藏部分访存延迟的依据，不等于已经测得完全重叠。显式异步接口另见 [NVIDIA Async Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

A 的 global 读取仍尽量用 `float4`，存入 shared 时转置为 `As[stage][k][m]`，因此 A 的发布是分散的标量 store；B 保持连续的 `float4` store。尾部补零，地址未对齐或不足四个元素时回退到标量读取。

## 2. 为什么 block 同步从两次变为一次

原 `warptiled` 每批有两次 `__syncthreads()`：加载结束后等待数据齐备，计算结束后等待所有线程读完，再允许覆盖唯一的 shared 缓冲区。

双缓冲允许快线程把下一批写入另一份缓冲区，而慢线程继续读取当前缓冲区。切换前的同一道 barrier 同时保证：下一批已经写完、当前批已经读完。进入再下一次循环时，才可以重用旧缓冲区。

设 `S = ceil(K/BK)`，`K > 0`：

| 实现 | 实际执行的 block barrier 次数 |
|---|---:|
| warptiled | 每批两次，共 `2S` |
| doublebuffer | 首批一次，加 `S-1` 次切换，共 `S` |

例如 P1 的 `K=2048, BK=16`：256 次变为 128 次。这是**同一个 block 内所有线程的同步**，不是两个 block 之间同步，也不是单纯 warp 内同步。次数减半不能推出等待耗时减半；当前 benchmark 同时包含预取、同步减少、指令和资源占用变化，没有单独隔离每项贡献。[NVIDIA 对 `__syncthreads()` 的定义](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#thread-block-synchronization-functions)

## 3. 机器码能证明到哪一步

干净构建产生的完整 kernel 有 928 条指令，指令编码与当天先前审核快照完全一致。以下地址是**本次二进制的 kernel 相对偏移**；重新编译后的地址可能改变。

| 下一批数据 | 普通 global load | 目标寄存器 | 发布到 shared |
|---|---|---|---|
| A0 | `0x17f0 LDG.E.128` | R28–R31 | `0x31a0`–`0x31d0` |
| A1 | `0x1a20 LDG.E.128` | R24–R27 | `0x31e0`–`0x3210` |
| B0 | `0x1c60 LDG.E.128` | R36–R39 | `0x3220 STS.128` |
| B1 | `0x1e80 LDG.E.128` | R32–R35 | `0x3230 STS.128` |

当前批计算位于 `0x1ed0`–`0x30a0`：包含 256 条静态 FFMA，每个 BK=16 的 tile 循环两遍，期间不读取 R24–R39。上述 LDG 使用 write scoreboard 5；沿对齐向量路径，首次等待该结果依赖出现在计算之后的 `0x31a0 STS`（wait mask `0x20`）。LDG 后的 `BSYNC B0` 有 wait mask `0x01`，对应地址/源寄存器的 read scoreboard 0，不能将其误读为等待加载结果。`B0` 本身是控制流汇合 token，与 scoreboard 编号不是同一概念。

这提供了“下一批 load 的结果等待被推迟到当前批计算之后”的静态证据。它**没有测量实际重叠的周期数**，也不证明所有尾部路径都同样调度。解码依据是第三方逆向实现 [CuSMVersion](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuSMVersion.py)、[CuControlCode](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py) 和 [依赖说明](https://github.com/cloudcores/CuAssembler/blob/master/UserGuide.md)，不是 NVIDIA 官方公开的控制字规范。

可核查 [带注释的分析](sass-review.txt)、[主循环原始指令和编码](sass-main-loop.txt)、[资源查询](metadata/resources.log)。

| 资源，默认八参数 | warptiled | doublebuffer |
|---|---:|---:|
| 线程/block | 128，block=(16,8) | 128，block=(16,8) |
| 寄存器/thread | 66 | 105 |
| shared/block | 8 KiB | 16 KiB |
| stack / local / spill loads / spill stores | 0 | 0 |

双缓冲的代价包括更大的 shared 占用和预取寄存器的存活区间。本次没有测 achieved occupancy，不能据此给出具体 occupancy 降幅或把性能变化全部归因于同步。

## 4. 四组形状的实测结果

两个 kernel 均为 `BM=64, BN=64, BK=16, WM=32, WN=32, TM=4, TN=4, WNITER=1`，即每 block 四个 warp、`WMITER=2`。Release，CUDA 13.2.78、MSVC 19.44.35217.0，驱动 581.42。cuBLAS 使用严格 FP32（pedantic），不是 TF32/Tensor Core 对照。

每种形状各测两个 kernel、各四遍。每次调用 `warmup=100, rounds=5, seed=42`，每轮计时批量迭代后除以次数。第 1/3 遍按 P1→P4、warptiled→doublebuffer；第 2/4 遍逆序。单次调用内部交替计时自写 kernel 与 cuBLAS。

汇总先取**四次运行各自 median 时间的中位数**，再按 `2MNK/(ms × 10^6)` 换算有效 GFLOP/s；不是取最快成绩，也不是平均 GFLOP/s。吞吐提升为 `t_warptiled / t_doublebuffer - 1`。表中 cuBLAS 采用四次 **doublebuffer 配对调用** 的时间汇总；warptiled 配对的 cuBLAS 同样保留在 CSV 和 summary 中。

| case / M×N×K | iters/轮 | warptiled ms | doublebuffer ms | warptiled GFLOP/s | doublebuffer GFLOP/s | 配对 cuBLAS GFLOP/s | 吞吐提升 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 / 2048×2048×2048 | 44 | 2.765277 | 2.728305 | 6212.71 | 6296.90 | 6869.35 | +1.36%，未证实稳定收益 |
| P2 / 16×4096×4096 | 335 | 0.521318 | 0.367850 | 1029.83 | 1459.48 | 1670.11 | +41.72% |
| P3 / 1001×769×513 | 800 | 0.182206 | 0.169199 | 4334.56 | 4667.77 | 5188.62 | +7.69% |
| P4 / 16×256×128 | 10000 | 0.012310 | 0.008443 | 85.18 | 124.19 | 156.16 | +45.80% |

四遍各自的吞吐提升，按同一遍的 warptiled 与 doublebuffer 比值：

| case | 第1遍 | 第2遍 | 第3遍 | 第4遍 |
|---|---:|---:|---:|---:|
| P1 | -11.15% | -2.01% | +1.66% | +4.91% |
| P2 | +280.34% | +40.80% | +43.25% | +40.22% |
| P3 | +12.33% | +7.63% | +4.91% | +8.34% |
| P4 | +46.11% | +41.46% | +45.58% | +46.15% |

**波动与解释边界：** 最初两遍后发现 P1/P2 首遍波动，才补做两遍相同设置的正反序复测。没有剔除任何遍次。P2 首遍 warptiled 及其 cuBLAS 同时明显变慢，原因尚未确定，因此不能将该遍的 +280% 当作稳定收益。P1 首遍 doublebuffer 对应 cuBLAS 也异常慢；不能据此宣称胜过 cuBLAS。P1 四遍方向不一致，其汇总 +1.36% 只能描述这组样本。P2 后三遍约 +40%～43%，P3/P4 四遍均有收益。本次没有锁频，前后 [设备快照](metadata/telemetry-before.log) 也不是 kernel 运行期间的频率采样。

P4 的耗时只有约 8～12 微秒，有效吞吐受启动、调度及很小的网格限制；P2/P4 的 M=16 也没有填满 BM=64。此处验证的是固定配置下的改进，不能替代按形状调参。四组汇总均仍低于配对的 cuBLAS。

原始证据：[32 份计时 CSV](benchmarks/)，共 64 行（自写 kernel + 配对 cuBLAS），每行保留五轮均值、正确性、源码指纹和参数；[机器可读汇总](summary.json)、[简表 CSV](comparison.csv)、[采集元信息](capture.json)。旧的 [七版本 Roofline](../roofline-20261022/README.md) 是引入 doublebuffer 前的历史采样；本次没有新增 NCU Roofline 点，不能用普通 benchmark 的时间替换旧 NCU 时间。

## 5. 正确性与复现

精确测试源码的 CMake 指纹为 `88a8b45c9bc32b3087b4ba86de29c0d5c1b52977073edf742c7e73293f0995b5`。源码快照包含 CMake、src、benchmarks、scripts、tests、third_party，不含 EXE、build 或原始 profiler 二进制。逐文件哈希见 [source-files.json](source-files.json)，整个归档的哈希见 [capture.json](capture.json)。其他操作系统或换行方式下重新生成的源码指纹可能不同。

干净构建后 16 项 CTest 全部通过、无跳过，包含 doublebuffer 的 32 个固定正确性用例及尾部输入；额外检查 64×64×16/32/48，覆盖单/双/三批；129×193×67 上 memcheck、racecheck、synccheck 均为零错误/零 hazard。所有 benchmark 行均 valid，无失败元素或非有限结果。[构建及验证日志](validation/)

Windows 在已初始化 MSVC/CUDA 环境的终端中，从仓库根目录执行。复现本次精确版本时，先用 PowerShell 将快照展开，再用独立新 build 目录保留本次默认配置：

```powershell
Expand-Archive results/published/doublebuffer-20261022/source-snapshot.zip build/db-replay-source
```

```bat
cmake -S build/db-replay-source -B build/db-replay -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGEMM_WARP_RUNTIME_CONFIGS=ON -DBUILD_TESTING=ON
cmake --build build/db-replay
ctest --test-dir build/db-replay --output-on-failure -j 1
build\db-replay\gemm.exe --mode check --kernel doublebuffer
```

以下四条每条都会同时测试 doublebuffer 和严格 FP32 cuBLAS。用同样命令把 `--kernel doublebuffer` 换为 `--kernel warptiled` 即得到基线；将 CSV 文件名中的 doublebuffer 也换成 warptiled。

```bat
build\db-replay\gemm.exe --mode bench --kernel doublebuffer --m 2048 --n 2048 --k 2048 --seed 42 --warmup 100 --iters 44 --rounds 5 --csv results/P1-doublebuffer.csv
build\db-replay\gemm.exe --mode bench --kernel doublebuffer --m 16 --n 4096 --k 4096 --seed 42 --warmup 100 --iters 335 --rounds 5 --csv results/P2-doublebuffer.csv
build\db-replay\gemm.exe --mode bench --kernel doublebuffer --m 1001 --n 769 --k 513 --seed 42 --warmup 100 --iters 800 --rounds 5 --csv results/P3-doublebuffer.csv
build\db-replay\gemm.exe --mode bench --kernel doublebuffer --m 16 --n 256 --k 128 --seed 42 --warmup 100 --iters 10000 --rounds 5 --csv results/P4-doublebuffer.csv
```

复现四遍汇总时按上述正反顺序重复，给 CSV 加 `-r1`～`-r4` 后缀保留每遍。绝对时间会随机器状态、工具链和驱动变化。

```bat
compute-sanitizer --tool memcheck build\db-replay\gemm.exe --mode check --kernel doublebuffer --m 129 --n 193 --k 67
compute-sanitizer --tool racecheck build\db-replay\gemm.exe --mode check --kernel doublebuffer --m 129 --n 193 --k 67
compute-sanitizer --tool synccheck build\db-replay\gemm.exe --mode check --kernel doublebuffer --m 129 --n 193 --k 67
```

doublebuffer 的参数目前通过 CMake 的 `GEMM_DB_BM/BN/BK/WM/WN/TM/TN/WNITER` 设置后重新编译；`gemm.exe --bm ...` 的运行时选择仍仅用于 warptiled，自动调优入口目前支持 tuned/warptiled。本次不添加重复调参工具，也不宣称已完成 doublebuffer autotuning。
