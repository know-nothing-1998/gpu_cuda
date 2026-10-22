# 精选实验：naive → double buffer

发布截止：2026-10-22。设备为 RTX 4060 Laptop、SM 8.9，使用严格 FP32 cuBLAS。普通 benchmark 为 Release CUDA Event 计时，Roofline 单独使用 NCU 采样数据。这里是实验筛选后的可复核记录，完整原始目录仍保存在开发机。

| 时间 / 阶段 | 关键观察 | 入口 |
|---|---|---|
| 10-18：naive / coalesced / shared | P1 为 117.10 / 959.37 / 1209.10 GFLOP/s；P2 shared 不如 coalesced | [基线与 Roofline](baseline/README.md) |
| 10-18～20：register | TM 有最优范围；二维展开不自动保证加速；合作加载和 float4 需结合形状验证 | [寄存器阶段](register/README.md) |
| 10-21：tuned | 五参数按 MNK 搜索；P4 的小 tile 与大矩阵配置不同 | [tuned 搜索](tuned/README.md) |
| 10-22：用户 warp tiling | P1 6635.25 GFLOP/s，同轮 prior tuned 6048.59、cuBLAS 7589.67；P4 与 tuned 基本持平 | [warp 搜索](warptiled/README.md) |
| 10-22：六实现 + cuBLAS 历史快照 | P1=`2048³`、同一构建、默认参数的 NCU 采样；不代表 autotune 最佳成绩，不含 doublebuffer | [七版本 Roofline](roofline-20261022/README.md) / [图](roofline-20261022/roofline.svg) / [指标](roofline-20261022/model.json) |
| 10-22：用户 double buffer | 固定默认参数四遍复测：P1 波动明显，未证实稳定收益；P2/P3/P4 较 warptiled 约 +42%/+8%/+46%，非 autotune 最优配置比较 | [当前实现的 benchmark、原理与 SASS 证据](doublebuffer-20261022/README.md) |

这是不同时间的实验路线，不是一条严格控制所有变量的加速曲线。比较收益时使用各实验内部的成对结果；不得用别次 cuBLAS 作统一分母。最新两组保持 64×64 tile 的 64 线程候选没有归档性能数据，不计入上述结果。

## 保存什么

- 原始计时 CSV，保留各轮均值、正确性字段和同次 cuBLAS；关键参数、排名与资源摘要。
- 必要的历史源码 ZIP 和变体覆盖源码；ZIP 仅包含源码，不含 EXE 或构建产物。
- 历史基线、10-22 六种自写实现加 cuBLAS 的 Roofline SVG 及其 JSON 指标；不上传 NCU 二进制或全量 SASS。
- 子目录中的来源清单与 SHA256。部分历史 JSON 中的本机路径仅作溯源，运行入口以 README 的相对路径命令为准。
- [设备原始查询](device/device-info.log)、[结构化设备快照](device/device-info.json)、[时钟快照](device/nvidia-smi-clocks.csv)；解释见 [GPU_DEVICE.md](../../GPU_DEVICE.md)。

初始二维 register 的错误版本明确标记为失败案例，不当作通用正确实现；初测与补测有差异时同时保留，不只挑最快轮次。先前由助手编写、后来移除的 warp 实验不作为当前用户实现的成绩。

doublebuffer 目录现在对应用户编写的正式实现，替换此前尚未发布的助手参考预实验；旧预实验原件仍在开发机保留。当前流水使用普通 global load 提前读取下一 tile，计算结束后发布到另一份 shared 缓冲区，不使用 `cp.async`。当 `S=ceil(K/BK)` 时，block 同步从 warptiled 的 `2S` 次变为首批一次加后续 `S-1` 次；性能变化包含预取、同步减少及资源占用的共同影响。

## 如何复现

当前源码的构建、check、benchmark、autotune 和 profiling 见 [根 README](../../README.md)。不需要任何未上传的本地 results 或 build 文件。

重现历史实现时，按子目录说明将 source-snapshot.zip 解压到新的 build 子目录，应用对应版本覆盖，再以快照自己的 CMake 编译；不要覆盖当前 src。多数旧快照没有完整 tests，说明中会要求 BUILD_TESTING=OFF，但 exe 的 check 仍可用。

源码、参数和测量方法可以复现；GPU 频率、功耗、温度、驱动、编译器与后台负载会改变绝对时间。没有发布原始 NCU 报告，因此不能在 clone 后直接打开旧 GUI 报告；需要按根 README 重新采集。

已发布的七版本 Roofline 可直接离线重画，无需 GPU 或原始 NCU 报告（需安装 Matplotlib）：

```bat
python scripts/roofline.py --model results/published/roofline-20261022/model.json --output results/roofline-redraw
```

命令从仓库根目录运行，模型与采样口径见 [ROOFLINE.md](../../ROOFLINE.md)。该图是引入 doublebuffer 前的历史快照，不会因新增实现而自动增加数据点。[VALIDATION.md](VALIDATION.md) 记录此前归档整理的独立构建与 CPU 测试；七版本的 GPU 检查和 NCU 采样证据见 [Roofline 实验说明](roofline-20261022/README.md)，当前流水实现的新验证及计时见 [double buffer 报告](doublebuffer-20261022/README.md)。

新增实验继续写到被忽略的 results 目录。只有筛选后的独立结论和必要证据进入此目录，避免累计重复调参、二进制和中间日志。来源/文件完整性可用 [SHA256SUMS.txt](SHA256SUMS.txt) 校验。
