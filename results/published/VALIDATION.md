# 2026-10-22 发布验证

本次只整理仓库与工具，不新增 GPU 性能测量，也不修改核函数。验证环境为根 README 记录的 Windows / MSVC / CUDA 开发机。

从准备发布的文件集合复制出独立源码目录，不携带原 build、未公开 results 或个人 IDE 设置；以 NMake、Release、SM89 从头配置并编译全部目标。

```bat
cmake -S SOURCE -B BUILD -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build BUILD
ctest --test-dir BUILD -R "^(numerical_harness|runtime_warp_cli_contract|autotune_contract|autotune_cli_contract|bench_tuned_contract)$" --output-on-failure
```

结果：构建成功，5/5 CPU CTest 通过，其中包含 92 次无 GPU 命令行调用，以及 49 项 Python 测试。另验证 PowerShell benchmark 脚本语法，tuned/warptiled/手动 tuned 的 dry-run 均成功。这里没有运行 GPU correctness、benchmark 或 profiling；历史 GPU 检查证据仍属于各实验的原始时点。

发布数据已核查相对链接、复制文件哈希、源码 ZIP 内容、原始 round_mean_ms 与表格中位数。源码快照中未包含 EXE、DLL、构建对象或 GPU profiler 二进制。所有精选文件的字节哈希见 SHA256SUMS.txt；.gitattributes 禁止 Git 对这些证据转换换行。
