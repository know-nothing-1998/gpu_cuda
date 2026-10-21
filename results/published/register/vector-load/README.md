# float4：减少搬运与索引指令

来源：`results/register-vector-comparison-20261020-151759`。两版均 **TILED=64、TM=TN=4**，block=16×16。scalar 是之前的合作加载版，本轮重新构建；vector 为每线程 float4 加载，并保留未对齐/边界的标量回退。[load-vectorization.diff](load-vectorization.diff) 显示两版只改搬运，计算、launch 和写回相同。

| Case | scalar GFLOP/s | vector GFLOP/s | 吞吐变化 |
|---|---:|---:|---:|
| P1 初测 | 4852.64 | 6001.17 | +23.67% |
| P2 | 1107.43 | 1278.88 | +15.48% |
| P3 | 4082.13 | 4328.07 | +6.02% |
| P1 补测 | 5141.22 | 5667.46 | +10.24% |

初测 [summary.json](summary.json) 与两版 `P*-pass*.csv` 保存全部记录。P1 初测存在明显漂移，因此保留 [p1-confirmation/summary.json](p1-confirmation/summary.json)、[records.json](p1-confirmation/records.json) 和四份补测 CSV，**不只选最快结果**。补测为 warmup=100、iters=88，两遍各五轮；初测为 warmup=20、iters=44。cuBLAS 同样随机器状态变化，不能把 +23.67% 当稳定、精确的增益。

P1 的 LDG 和 STS 指令均减少 75%；global load 总 sector 数不变，shared store 总 ideal wavefront 数不变，FFMA 与计算 LDS 数量也不变。总动态 warp 指令下降约 16.395%。这支持减少搬运/索引指令是优化的重要原因，而不是数据传输字节数减为四分之一。

`STS.128` 每条 warp 指令服务更多字节，理想需要四个 wavefront；actual=ideal，不能把这四个 wavefront 本身当成 bank conflict。P3 的 N/K 非四的倍数，频繁走标量回退，P1 的指令比例不适用于 P3。证据分别见 [profile-summary.json](profile-summary.json)、[profile-source-evidence.json](profile-source-evidence.json)、[P2](P2-analysis/profile-summary.json)、[P3](P3-analysis/profile-summary.json)。

两版在本轮均通过 suite、wide/tall 边界检查和 tail memcheck。计数、终态原文与编译资源摘要见 [validation-summary.json](validation-summary.json)，附原始日志路径和哈希，省去重复成功日志。原始实现的 float4 对齐健壮性和很大矩阵下 int 索引溢出仍应在后续实现中审视；此处原样保留历史源码，不以归档名义修改代码。

vector 版重测（从项目根目录）：

```bat
python -m zipfile -e results\published\register\vector-load\source-snapshot.zip build\repro-vector-source
copy results\published\register\vector-load\vector-load\register.cu build\repro-vector-source\src\register.cu
copy results\published\register\vector-load\vector-load\register.cuh build\repro-vector-source\src\register.cuh
cmake -S build\repro-vector-source -B build\repro-vector -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF
cmake --build build\repro-vector --target gemm
build\repro-vector\gemm.exe --mode check --kernel register
build\repro-vector\gemm.exe --mode bench --kernel register --m 2048 --n 2048 --k 2048 --warmup 20 --iters 44 --rounds 5
```

对照时改用 `scalar-load` 的 `.cu/.cuh` 重建；按 [matrix.csv](matrix.csv) 测三个形状，并交替两版测试顺序。补测另用 warmup=100、iters=88，其余相同。本目录不含 NCU 二进制；导出的指标用于历史复核，重新打开 GUI 需要重新采集。
