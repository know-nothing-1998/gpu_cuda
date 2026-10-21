# Register tiling：四个关键实验

以下目录保存 2026-10-18 至 2026-10-20 的历史快照与原始逐轮 CSV。设备均为 RTX 4060 Laptop GPU，目标 SM89，Release，严格 FP32 cuBLAS。它们是不同时间的独立实验，**只在各实验内部计算优化收益和 cuBLAS 比例**，不把跨天的吞吐数值拼成受控加速曲线。

| 阶段 | 保留的内容 | 主要观察 |
|---|---|---|
| [TM sweep](tm-sweep/README.md) | TILED=32，TM=1/2/4/8/16/32，以及同轮 shared | 这些形状里 TM=8 最快；P1 为 3328.49 GFLOP/s，不能靠增加 TM 一直提速 |
| [初始二维版本](2d-initial/README.md) | TILED=32，1D 8×1 / 2D 4×4 | P1 性能相近，2D 有 A 加载边界错误；保留失败证据，不作为通用正确版本 |
| [合作加载](cooperative-load/README.md) | TILED=64、TM=TN=4；strided / coalesced | P1 从 4869.48 到 5361.83 GFLOP/s，P2 则回退约 9.33% |
| [float4 加载](vector-load/README.md) | 同样 64/4/4；scalar / vector，含补测 | 搬运指令减少；P1 初测 6001.17 GFLOP/s，补测 5667.46，保留波动而不挑最快结果 |

每个子目录包含历史 `source-snapshot.zip`、版本覆盖用的 `.cu/.cuh`、汇总、逐轮 CSV、`validation-summary.json` 验证/资源摘要及 `publication.json` 文件哈希表。重复的成功检查、设备和全库资源日志不发布；仅保留初始 2D 的两个关键失败日志。原始 `manifest.json` 和某些 JSON 里的本机绝对路径只记录当时的构建位置，不是复现入口。

整理时已离线按原 CMake 的顺序重算构建指纹：六个 TM、两个初始 2D 对照、两个合作加载和两个 float4 对照的快照加覆盖源码，都与对应原始 CSV 的 `source_sha256` 一致。汇总中位数也已由归档 `round_mean_ms` 重新核对，包括 float4 补测。这不等于使用当前工具链重新编译或重测。

## 数据怎么算

P1/P2/P3 分别为 2048×2048×2048、16×4096×4096、1001×769×513。通常 seed=42，warmup=20，iters=44/335/800，rounds=5；每个版本跑两遍相反顺序，共 10 个每轮平均耗时，合并后取中位数，再用 `2MNK/(ms×10^6)` 换算吞吐。float4 的 P1 补测参数不同，单独保存。

每份原始 CSV 同时包含目标 kernel 与它同次配对的 cuBLAS；`round_mean_ms` 是原始轮次，正确性字段区分完整 cuBLAS 对照和 CPU double 抽样。NCU 指令统计用于解释机制，不用 NCU duration 替代普通 Event 计时。

## 历史源码复现

子目录的命令都从仓库根目录执行。需 CUDA、CMake、Ninja 和已加载 MSVC 的终端；其他 GPU 修改 `CMAKE_CUDA_ARCHITECTURES`。这些 register 快照仅含完整构建输入，没有历史 `tests/`，所以必须配置 `-DBUILD_TESTING=OFF`；生成的 `gemm.exe --mode check` 仍可用。

先解压某阶段快照到新的独立目录，再用该阶段选定版本的 `register.cu`（如有 `.cuh` 也复制）覆盖解压副本的 `src/`。最后用该快照自己的 CMake 编译。这不修改仓库当前的 kernel。子目录给出一个具体版本的命令，其他版本按相同方式切换。

复现命令统一使用 NMake，不要求额外安装 Ninja。生成器不是这些历史 CMake 的源码指纹输入；更换生成器不影响上述源码哈希核对，但编译器版本、架构和编译选项改变后仍需重新测量性能。

由于未发布 `.ncu-rep`、EXE、构建目录或完整 SASS 日志，clone 后可以重建源码和重测 benchmark，但不能直接打开原始 NCU GUI。已导出的关键 profile 指标可离线核查；需要新 GUI 报告时请重新采集。这里发布的是历史实验，当前源码后续可能已有修正。本次整理没有新增 GPU 测量。
