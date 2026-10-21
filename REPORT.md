# 实验进展：naive → warp tiling

截至 2026-10-22，六种 FP32 自写 kernel 已接入统一检查、cuBLAS 对照、benchmark 和 profiling。自动调参支持 tuned 五参数和 warptiled 八参数；普通 exe 可选择已编译的 warp 配置。

**完整精选入口：[results/published/README.md](results/published/README.md)。**

| 阶段 | 要回答的问题 | 证据 |
|---|---|---|
| naive / coalesced / shared | lane 与矩阵布局、合并访存、shared 复用带来什么变化？ | [同轮对照与 Roofline](results/published/baseline/README.md) |
| register | 每线程多个输出、线程分工、float4 是否改善复用和指令开销？ | [寄存器阶段](results/published/register/README.md) |
| tuned | BM/BN/BK/TM/TN 如何适配不同 MNK？ | [参数搜索](results/published/tuned/README.md) |
| warptiled | 显式 warp tile 与转置 As 是否进一步改善性能？ | [用户实现的 P1/P4 搜索](results/published/warptiled/README.md) |

两组保持 64×64 输出 tile 的 64 线程候选已加入搜索空间。用户反馈没有改善；尚无归档的本轮 GPU 计时数据，因此不将它们列为已验证的性能提升。

## 结果口径

普通 benchmark 使用同 stream 的 CUDA Event，区间只包含重复 GEMM；每轮求单次平均，再取各轮中位数。每个自写实现与同轮严格 FP32 cuBLAS 配对。GFLOP/s 为有效工作量 `2MNK/t`，不等于实际执行的所有硬件浮点指令。

NCU 的 replay、缓存、频率条件与普通 benchmark 不同，Roofline 点不能与 Event 时间混算。不同日期的表格用于记录路线，只有同轮对照可以支持该轮加速比。

历史实验对应当时的源码；精选目录保留来源路径、原始数据、参数和可用源码快照。重跑当前源码验证当前版本，重跑归档快照复核历史版本；都不保证得到相同的绝对速度。当前开发机参数见 [GPU_DEVICE.md](GPU_DEVICE.md)。

## 本次发布整理

完整实验目录继续保存在开发机，远程仅收录 `results/published/`。早期 M0 临时日志、重复构建、所有候选二进制和大型 NCU 报告不进入当前版本；旧 Git 提交历史不重写。复制下来的原始数据不改写成当前源码结果。

构建、检查、benchmark、自动调优、profiling 的可执行命令集中在 [README](README.md)。本次整理不修改核函数或重新测量 GPU 性能；发布前验证从待提交文件集合独立配置构建及 CPU 测试。
