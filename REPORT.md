# 实验进展：naive → 异步流水线

截至 2026-10-23，九种 FP32 自写 kernel 已接入统一检查、cuBLAS 对照、benchmark 和 profiling，其中 doublebuffer_opt 为独立优化实验，asyncpipeline 为 A 软件预取与 B 硬件异步拷贝的混合流水线。自动调参支持 tuned 五参数和 warptiled 八参数；普通 exe 可选择已编译的 warp 配置。doublebuffer / asyncpipeline 分别使用 `GEMM_DB_*` / `GEMM_ASYNC_*` 编译参数，当前未接入自动调参。

**完整精选入口：[results/published/README.md](results/published/README.md)。**

| 阶段 | 要回答的问题 | 证据 |
|---|---|---|
| naive / coalesced / shared | lane 与矩阵布局、合并访存、shared 复用带来什么变化？ | [同轮对照与 Roofline](results/published/baseline/README.md) |
| register | 每线程多个输出、线程分工、float4 是否改善复用和指令开销？ | [寄存器阶段](results/published/register/README.md) |
| tuned | BM/BN/BK/TM/TN 如何适配不同 MNK？ | [参数搜索](results/published/tuned/README.md) |
| warptiled | 显式 warp tile 与转置 As 是否进一步改善性能？ | [用户实现的 P1/P4 搜索](results/published/warptiled/README.md) |
| 六实现 + cuBLAS 的历史快照 | 同一构建的各默认实现，在 P1 上处于 Roofline 的什么位置？ | [2026-10-22 对照说明](results/published/roofline-20261022/README.md) / [图](results/published/roofline-20261022/roofline.svg) / [指标](results/published/roofline-20261022/model.json)；不含 doublebuffer |
| doublebuffer | 下一 tile 的寄存器预取、shared 双缓冲和减少同步能否受益？ | [当前用户实现的 P1～P4 对照与 SASS 证据](results/published/doublebuffer-20261022/README.md) |
| doublebuffer_opt / asyncpipeline | bank conflict 改善为何可能退化？普通预取和硬件异步的指令/资源代价有何不同？ | [2026-10-23 精选实验](results/published/asyncpipeline-20261023/README.md) |

两组保持 64×64 输出 tile 的 64 线程候选已加入搜索空间。用户反馈没有改善；尚无归档的本轮 GPU 计时数据，因此不将它们列为已验证的性能提升。

## 异步流水线阶段（2026-10-23）

当前 asyncpipeline 对 A 使用普通 `float4` 全局读取，在寄存器暂存后转置写入 As；B 使用 `cuda::memcpy_async` 直接搬入下一份 shared buffer。两阶段 shared 仍为 16 KiB，八参数与 doublebuffer 默认值相同。它不是 A/B 全部走硬件异步的实现。

P1=`2048³` 两遍正反顺序对照中，当前 hybrid 达到约 6716 / 6519 GFLOP/s，比前一版标量 A4 加载快约 32.4% / 33.4%；同轮 doublebuffer 为 6860 / 6671 GFLOP/s，hybrid 耗时仍高约 2.1% / 2.3%。每遍使用 warmup=50、iters=50、rounds=5，并保留各自配对 cuBLAS。两遍和一个 shape 不足以证明所有形状的稳定优劣。

NCU 显示 hybrid 的每线程寄存器由 doublebuffer 的 105 降至 79，理论 occupancy 提高，但动态 warp 指令增加约 7.39%；FFMA 数量不变。减少中转寄存器并不保证更快。两份 NCU 的时钟条件不同，速度结论来自普通 Event benchmark，NCU 用于解释路径和指令差异。

独立的 doublebuffer_opt 改善 As 写入 bank conflict 和 C 写回指令，但增加寄存器和布局/寻址指令；当前参数下理论 occupancy 仍同为 33.33%，不是已证实的退化原因。用户观察到默认 P1 负优化；没有归档可审计的普通计时 CSV，不将异时钟 NCU duration 当作退化百分比。

原始 CSV、历史源码变体、资源/指令摘要与复现步骤见 [本轮报告](results/published/asyncpipeline-20261023/README.md)。[GPU 架构讲解](GPU_ARCHITECTURE_GUIDE.md) 系统整理本项目涉及的内存层级、SASS、warp 调度、驻留、同步和流水线，并给出 NVIDIA 官方参考资料。

发布前从独立源码副本重新完成 Release 构建，20 项 CTest 全部通过；doublebuffer_opt 与 asyncpipeline 的 `129×193×67` 各通过 memcheck、racecheck、synccheck，共六次检查。记录与命令见 [发布验证](results/published/asyncpipeline-20261023/validation/README.md)。

## Double buffer 阶段（2026-10-22）

当前实现使用普通 `LDG` 将下一 tile 预取到寄存器，再计算当前 shared tile，最后把预取值发布到另一份 shared 缓冲区并同步切换。A 的 shared 布局保持转置；没有使用 `cp.async`，也没有另分专门搬运的 warp。

默认参数沿用 warptiled 的 `BM/BN/BK/WM/WN/TM/TN/WNITER = 64/64/16/32/32/4/4/1`。两套 A/B shared tile 共 16 KiB；令 `S=ceil(K/BK)`，block 同步从原来的 `2S` 次减少到首批一次加后续 `S-1` 次，共 `S` 次。收益需同时考虑预取、同步减少及增加的寄存器/shared 占用，不能只看是否声明了两份数组。

本轮使用 P1 大方阵、P2 矮宽矩阵、P3 非对齐矩阵、P4 小矩阵，对当前 doublebuffer、warptiled 和配对 cuBLAS 进行比较。方法、原始计时及验证记录见 [double buffer 报告](results/published/doublebuffer-20261022/README.md)。之前助手参考预实验只保留在本地，不混入当前实现的结果。

固定默认参数的四遍复测中，P1 波动明显，未证实稳定收益；P2/P3/P4 的汇总吞吐较 warptiled 分别提高约 **42% / 8% / 46%**。每遍包含五轮计时，第二、四遍反转 shape 和 kernel 顺序；汇总取四遍各自中位耗时的中位数，再换算 GFLOP/s。P2 首遍出现伴随 cuBLAS 变慢的异常大增益，全部原始数据和解释保留在报告中。这些结果不是 autotune 最优配置的比较。

该历史构建完整 16 项 CTest 实际执行通过；`129×193×67` 尾部输入的三种 Compute Sanitizer 检查均无错误。

## 结果口径

普通 benchmark 使用同 stream 的 CUDA Event，区间只包含重复 GEMM；每轮求单次平均，再取各轮中位数。每个自写实现与同轮严格 FP32 cuBLAS 配对。GFLOP/s 为有效工作量 `2MNK/t`，不等于实际执行的所有硬件浮点指令。

NCU 的 replay、缓存、频率条件与普通 benchmark 不同，Roofline 点不能与 Event 时间混算。不同日期的表格用于记录路线，只有同轮对照可以支持该轮加速比。

2026-10-22 的七版本 Roofline 使用引入 doublebuffer 前的同一 Release 构建，包含当时六种自写实现和 cuBLAS，MNK 均为 `2048³`，各实现使用默认参数；它不是自动调参优胜结果的排名，也未包含本次 doublebuffer 的新采样。所有目标先单独通过正确性检查，再按相同的 NCU 设置采集。历史四版本基线仍单独保留。[ROOFLINE.md](ROOFLINE.md) 说明公式、采样限制和无需 GPU 的重画方法。

历史实验对应当时的源码；精选目录保留来源路径、原始数据、参数和可用源码快照。重跑当前源码验证当前版本，重跑归档快照复核历史版本；都不保证得到相同的绝对速度。当前开发机参数见 [GPU_DEVICE.md](GPU_DEVICE.md)。

## 本次发布整理

完整实验目录继续保存在开发机，远程仅收录 `results/published/`。早期 M0 临时日志、重复构建、所有候选二进制和大型 NCU 报告不进入当前版本；旧 Git 提交历史不重写。复制下来的原始数据不改写成当前源码结果。

构建、检查、benchmark、自动调优、profiling 的可执行命令集中在 [README](README.md)。此前的归档整理、七版本 Roofline、double buffer 各有独立的历史验证记录，分别见 [VALIDATION.md](results/published/VALIDATION.md)、[Roofline 报告](results/published/roofline-20261022/README.md) 和 [double buffer 报告](results/published/doublebuffer-20261022/README.md)。2026-10-23 的代码、精选证据和新验证见 [异步流水线报告](results/published/asyncpipeline-20261023/README.md)，不将旧验证记录冒充当前构建的测试。
