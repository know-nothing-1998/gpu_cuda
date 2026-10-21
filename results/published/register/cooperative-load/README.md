# 2D register：分开计算与合作搬运

来源：`results/register-memory-comparison-20261019-191833`。两版固定 **TILED=64、TM=TN=4**，block=16×16，计算阶段与输出映射相同；比较原来每线程相邻 4×4 元素加载和线性 tid 分工的合并加载。两版的准确差异见 [load-mapping.diff](load-mapping.diff)。

| Case | strided GFLOP/s | coalesced GFLOP/s | 吞吐变化 |
|---|---:|---:|---:|
| P1 | 4869.48 | 5361.83 | +10.11% |
| P2 | 1219.33 | 1105.58 | −9.33% |
| P3 | 3772.81 | 4142.10 | +9.79% |

来源为 [summary.json](summary.json)；两版各 P1/P2/P3 的 pass1/pass2 CSV 保存全部十轮结果和各自的 cuBLAS。两遍都给出同样的快慢方向，P2 回退同样完整保留。

P1 profiling 的 global load sectors/request 从 16 降到 4，shared store actual/ideal wavefront 从 4 降到 1；计算阶段 shared load 两版均 actual=ideal。总动态指令反而增加约 18.34%，包含更多索引、比较和控制指令，所以访存效率提高不等于等比例提速。详见 [profile-summary.json](profile-summary.json)、[profile-metrics.json](profile-metrics.json)、[profile-source-evidence.json](profile-source-evidence.json)。

两版的数值 suite、wide/tall 边界检查计数、tail memcheck 结论及编译资源原文摘要见 [validation-summary.json](validation-summary.json)，每项附原始日志路径和哈希。重复成功日志不发布；`manifest.json` 保留历史构建/报告哈希；`.ncu-rep` 本身未发布。

合并加载版重测（从项目根目录）：

```bat
python -m zipfile -e results\published\register\cooperative-load\source-snapshot.zip build\repro-cooperative-source
copy results\published\register\cooperative-load\coalesced-load\register.cu build\repro-cooperative-source\src\register.cu
copy results\published\register\cooperative-load\coalesced-load\register.cuh build\repro-cooperative-source\src\register.cuh
cmake -S build\repro-cooperative-source -B build\repro-cooperative -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF
cmake --build build\repro-cooperative --target gemm
build\repro-cooperative\gemm.exe --mode check --kernel register
build\repro-cooperative\gemm.exe --mode bench --kernel register --m 2048 --n 2048 --k 2048 --warmup 20 --iters 44 --rounds 5
```

改用 `strided-load` 的 `.cu/.cuh` 重建即可得到另一版。P2 为 `--m 16 --n 4096 --k 4096 --iters 335`，其余参数不变。该目录没有单独控制 tile=32→64 的效应，不应将它与前一天 TM sweep 直接相除来归因。
