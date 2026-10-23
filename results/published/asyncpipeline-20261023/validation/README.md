# 2026-10-23 发布验证

从待发布源码复制出独立工作副本，排除 `results/`、已有 build 和本机 `.vscode` 配置，使用空的构建目录重新配置、编译和测试。没有依赖旧 EXE 或旧 generated 头文件；完整采集记录中的绝对路径只表示本机验证位置，复现入口使用下方相对路径。

环境：Windows、RTX 4060 Laptop / SM 8.9、CUDA 13.2.78、MSVC 19.44.35217.0、Release。默认八参数为 `64/64/16/32/32/4/4/1`；doublebuffer 与 asyncpipeline 各使用 128 threads/block、16 KiB shared。组合源码指纹：

```text
fc67a738011924342699686422ff136209c81d424fe575ce984ad6e213d3826b
```

此指纹与本页归档的最终 float4 hybrid benchmark 相同。它表示 CMake 中约定的输入文件和配置，不代替每个发布文件的 SHA256 校验。

| 检查 | 结果 | 原始记录 |
|---|---|---|
| 独立 Release 配置与完整编译 | 通过 | [clean-build.log](clean-build.log) |
| 全部 CTest，串行执行 | **20/20 通过，无跳过** | [ctest.log](ctest.log) |
| doublebuffer_opt，129×193×67 | memcheck / racecheck / synccheck 全部通过 | [memcheck](doublebuffer_opt-memcheck.log)、[racecheck](doublebuffer_opt-racecheck.log)、[synccheck](doublebuffer_opt-synccheck.log) |
| asyncpipeline，129×193×67 | memcheck / racecheck / synccheck 全部通过 | [memcheck](asyncpipeline-memcheck.log)、[racecheck](asyncpipeline-racecheck.log)、[synccheck](asyncpipeline-synccheck.log) |
| 参数、设备、实现注册 | 当前两实现均可用 | [device-info.log](device-info.log)、[kernel-list.log](kernel-list.log) |

内存/同步检查均为 0 errors，racecheck 为 0 hazards、0 warnings。边界用例同时检查 cuBLAS 与完整 CPU double 参考。完整 CTest 包括新增两实现的固定 correctness 用例、边界，以及所有可用实现的 CLI 正确性、CSV 和 profile 隔离契约。

[commands.json](commands.json) 保存此次实际命令和返回码；[source-checksums.json](source-checksums.json) 保存构建输入副本的各文件 SHA256。检查结束后再次核对工作区中的源码、CMake、脚本、测试和 benchmark 配置，未在验证期间发生变化。文档随后补入本次验证结果，不影响组合源码指纹。

从仓库根目录复现默认配置验证：

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -BuildDir build\release-check
ctest --test-dir build/release-check --output-on-failure -j 1
"%CUDA_PATH%\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --error-exitcode 1 build\release-check\gemm.exe --mode check --kernel asyncpipeline --m 129 --n 193 --k 67
"%CUDA_PATH%\compute-sanitizer\compute-sanitizer.exe" --tool racecheck --error-exitcode 1 build\release-check\gemm.exe --mode check --kernel asyncpipeline --m 129 --n 193 --k 67
"%CUDA_PATH%\compute-sanitizer\compute-sanitizer.exe" --tool synccheck --error-exitcode 1 build\release-check\gemm.exe --mode check --kernel asyncpipeline --m 129 --n 193 --k 67
```

将上述三条命令的 `--kernel` 换成 `doublebuffer_opt` 即为另一组检查。工具路径示例适用于本次 CUDA 13.2 安装；也可以使用已加入 PATH 的 `compute-sanitizer`。测试覆盖本次默认参数，修改模板参数后应重新检查。

本次发布验证没有重新进行性能排名或 NCU 采样。性能结论对应上级报告保留的两组历史实验窗口，不能将 CTest 的短计时作为 benchmark 数据。
