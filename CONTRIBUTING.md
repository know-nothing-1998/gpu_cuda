# 分支与实验工作流

采用 Git Flow 的分支分工，使用普通 Git 命令，不依赖额外 git-flow 插件。

| 分支 | 用途 |
|---|---|
| `main` | 已验证的阶段发布；不直接开发 |
| `develop` | 日常集成分支，发布后回到这里 |
| `feature/<topic>` | 从 develop 创建，完成后合回 develop |
| `release/<milestone>` | 从 develop 创建，整理验证后合入 main，再回合 develop |
| `hotfix/<topic>` | 必要时从 main 修复，再合入 main/develop |

当前阶段为 `doublebuffer-20261022`。提交源码、复现命令和精选证据；未完成的新性能实验留在本地。

## 下次开始

```bat
git switch develop
git switch -c feature/next-gemm-optimization
```

完成一个可检查的改动后，在 feature 分支提交，再合回 develop：

```bat
git add src tests benchmarks scripts README.md
git commit -m "perf: describe the measured GEMM change"
git switch develop
git merge --no-ff feature/next-gemm-optimization
```

有远程 develop 时先同步。多用户协作建议通过 PR 合并；不要对共享分支 force-push。阶段发布从 develop 拉 `release/<milestone>`，构建/检查后合入 main，再合回 develop；完成发布后继续在 develop/feature 上开发。

## 验证与证据

- 改核函数：先 `check`，再测同轮 cuBLAS；涉及 shared/sync 时检查边界输入和 Compute Sanitizer。
- 改命令行、构建或调参脚本：跑对应 CPU 测试，再验证干净构建。完整 CTest 含 GPU 工作负载。
- 统一自动调优入口为 `scripts/autotune.py --kernel tuned|warptiled`。不要为每次实验复制一套新的脚本；不同 shape 使用参数或候选 JSON。
- 生成的 `results/*` 默认忽略。只有经过检查的摘要、原始 CSV、参数、必要历史源码及来源索引进入 `results/published/`。
- 不提交 build、exe、GPU profiler 二进制、重复中间日志或本机 IDE 路径。个人 `.vscode` 配置保留本地，通用示例可以跟踪。
- 历史实验不改名冒充新源码结果。记录来源 SHA256、测试形状、计时次数和实际配置；跨日期性能不要直接合算加速比。
