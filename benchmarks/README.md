# 固定 benchmark 矩阵

唯一批量性能矩阵为 [matrix.csv](matrix.csv)。P1/P2/P3 于 2026-10-14 锁定，P4 于 2026-10-21 加入。正确性集合独立维护在 `src/gemm.cu`；性能矩阵不能替代边界和结构化输入测试。

| ID | M × N × K | A+B+C FP32 字节数 | iters | 目的 |
|---|---|---:|---:|---|
| P1 | 2048 × 2048 × 2048 | 50,331,648 | 44 | 规则吞吐和数据复用 |
| P2 | 16 × 4096 × 4096 | 67,633,152 | 335 | 小 M 的工作供给和 tile 利用 |
| P3 | 1001 × 769 × 513 | 6,711,116 | 800 | 非整 tile 尾部 |
| P4 | 16 × 256 × 128 | 155,648 | 10000 | 小 GEMM 的启动成本、block 数和 tile 浪费 |

字节数为 `4*(M*K+K*N+M*N)`，不是运行时显存实测；校验还需另一份 C，cuBLAS workspace 未计入。

统一 row-major、FP32 输入/累加/输出、alpha=1、beta=0、seed=42、`mt19937_u24_v1`。CSV 默认 warmup=20、rounds=5；自动调参的确认轮使用至少 100 次预热，具体以每份原始结果为准。

P1/P2/P3 的 iters 最初根据 cuBLAS 预跑单次 2.315192261 / 0.298677120 / 0.125132799 ms，按 `ceil(100ms / pilot_ms)` 校准；这只是历史选择次数的方法，不代表当前性能。P4 的 10000 次用于降低短 kernel 计时噪声。不要将早期 pilot 当作当前优化对照，精选实验见 [results/published](../results/published/README.md)。

不同实现速度相差很大时，同样 iters 无法使双方的 batch 都落在相同时间范围。可显式覆盖次数，但同一次自写实现/cuBLAS 比较应使用相同次数，并保留实际 warmup/iters/rounds 和全部轮均值。

## 调参空间

- [tuning.json](tuning.json)：tuned 的 43 组 BM/BN/BK/TM/TN 候选。
- [warptiling.json](warptiling.json)：warptiled 的 16 组 BM/BN/BK/WM/WN/TM/TN/WNITER 候选，线程数由 warp tile 推导。
- 统一入口：`python scripts/autotune.py --kernel tuned|warptiled --cases P1 P2 P3 P4`，或传入 `--m/--n/--k`。
- `--space PATH` 可选择其他搜索表。历史精选目录保存当时的参数空间，避免当前新增候选被误当作当年的实验范围。
- 这是有限候选搜索，结果只表示测量范围内的优胜，不代表全局最优。重复实验用新输出目录，不复制一套调参脚本。
