# trajectory_autoresearch

面向 `PPO_project` 的自治实验壳层，目标是把当前轨迹平滑任务整理成类似 [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) 的研究仓库。

当前版本保留三层职责：

- `prepare.py`：固定工具层。负责工作区初始化、训练调用、统一评测、结果聚合、轨迹导出与结果表维护。
- `train.py`：自治循环入口。这里定义候选策略、基于历史结果的候选选择、保留/丢弃规则，以及无限实验循环。
- `program.md`：给代理的运行说明，约束自治研究流程。

与原版 `autoresearch` 的对应关系：

- 原版固定 `prepare.py` + 单文件 `train.py` + `program.md`
- 本项目固定 `prepare.py` + 主要编辑 `train.py` + `program.md`
- 固定评测从 `val_bpb` 改为多路径聚合得分与轨迹归档

## 当前依赖

- 现有 `PPO_project/`
- `D:\Anaconda\envs\PPO` 环境

## 先做初始化

```powershell
D:\Anaconda\Scripts\conda.exe run -n PPO python trajectory_autoresearch\train.py --setup-only
```

这一步会创建：

- `trajectory_autoresearch\workspace\base_config.yaml`
- `trajectory_autoresearch\workspace\current_best.yaml`
- `trajectory_autoresearch\workspace\leaderboard.md`
- `trajectory_autoresearch\results.tsv`

## 跑一轮实验

```powershell
D:\Anaconda\Scripts\conda.exe run -n PPO python trajectory_autoresearch\train.py --max-experiments 1
```

## 持续自治迭代

```powershell
D:\Anaconda\Scripts\conda.exe run -n PPO python trajectory_autoresearch\train.py --max-experiments 0
```

其中：

- `--max-experiments 0` 表示无限循环，直到手工中断
- 每轮实验都会：
  1. 以当前最优配置为父代
  2. 基于历史结果选择下一个候选方向并生成配置
  3. 调用 `PPO_project/main.py` 训练
  4. 调用 `acceptance_suite.py` 做多路径统一评测
  5. 调用 `phase32_export_best_trajectories.py` 导出每条路径的最佳轨迹图与 CSV
  6. 写入 `results.tsv`
  7. 刷新 `workspace/leaderboard.{md,json}`
  8. 若得分提升，则晋升为新的当前最优，并封存到 `archives/promoted/<experiment_id>/`

## 当前阶段说明

这一版先把“自治实验骨架”搭起来，重点是：

- 固定评测
- 自动训练
- keep/discard 机制
- 轨迹归档
- 面向 GitHub 仓库整理的清晰入口

后续如果你要进一步贴近 `autoresearch`，下一步就是把 `PPO_project` 中真正允许代理编辑的核心优化面进一步收束成更小、更稳定的一组文件。
