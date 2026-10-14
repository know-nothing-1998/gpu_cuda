# 固定 benchmark 矩阵 v1

锁定日期：2026-10-14（Asia/Shanghai）。唯一批量入口为 [matrix.csv](matrix.csv)，实现路线推进时不扩大性能 shape 集合。正确性测试集合独立维护在 `src/gemm.cu`，不能用性能矩阵替代边界/结构化输入检查。

| ID | M × N × K | A+B+C FP32 字节数 | 目的 |
|---|---|---:|---|
| P1 | 2048 × 2048 × 2048 | 50,331,648 | 规则吞吐和数据复用 |
| P2 | 16 × 4096 × 4096 | 67,633,152 | 小 M 的工作量供给和 tile 利用 |
| P3 | 1001 × 769 × 513 | 6,711,116 | M/N/K 尾部及后续 fast/fallback 边界 |

内存数为源码公式 `4*(M*K+K*N+M*N)` 的计算值，不是运行时显存实测；自写实现检查/对照还需要一份 C，cuBLAS 内部 workspace 不包含在表中。

共同设置：row-major、FP32 输入/累加/输出、alpha=1、beta=0、seed=42、`mt19937_u24_v1`、预热 20 次、5 轮。输入生成和 CPU 抽样规则见项目 README。

首次预跑使用任务书的 100 次重复，结果保存在 [m0-pilot](../results/m0-pilot/summary.csv)。统一按 `ceil(100 ms / pilot_median_ms)` 手动校准：

| ID | 预跑 median ms | 初始轮时长约 ms | 锁定 iters |
|---|---:|---:|---:|
| P1 | 2.315192261 | 232 | 44 |
| P2 | 0.298677120 | 30 | 335 |
| P3 | 0.125132799 | 13 | 800 |

仅改变重复次数，三个 shape 均未缩放。正式基线 [m0-baseline/summary.csv](../results/m0-baseline/summary.csv) 的轮区间分别约 111–121、109–115、130–151 ms。后续若 naive 很慢，可手工覆盖 iters，将耗时控制在合理范围；同一次比较对两个实现使用同一 R，并保存实际 R 和全部轮均值。CLI 的单例默认仍为 100 次；批量脚本显式读取本 CSV。

以下为 naive 加入前的 M0 执行记录；当前实现和使用方式见 [项目 README](../README.md)。性能 shape 和默认重复次数保持不变。

| 路径 | 正确性集合 | P1/P2/P3 Event | profile |
|---|---|---|---|
| cuBLAS FP32 pedantic | 已通过 | 已保存 | NSYS 已采集；NCU 计数器权限受阻 |
| naive FP32 | 等待用户实现 | 待运行，自动与 cuBLAS 成对比较 | 待实现并 check 后运行 |
| shared/register/tuned/WMMA | 不在本轮范围 | 未测 | 未测 |
