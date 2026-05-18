# trajectory_autoresearch

这是一个面向轨迹平滑强化学习任务的自治研究项目，工作方式参考 `karpathy/autoresearch`，但优化目标从语言模型验证损失改成了多路径轨迹平滑综合得分。

## Setup

开始一轮新的自治研究时，先和用户完成以下动作：

1. 约定一个运行标签，例如 `traj-mar26`。
2. 为这轮研究创建独立分支，例如 `trajectory-autoresearch/<tag>`。
3. 阅读以下文件：
   - `trajectory_autoresearch/README.md`
   - `trajectory_autoresearch/prepare.py`
   - `trajectory_autoresearch/train.py`
   - `PPO_project/configs/default.yaml`
   - `PPO_project/tools/acceptance_suite.py`
   - `PPO_project/tools/phase32_export_best_trajectories.py`
4. 确认 `PPO` conda 环境可用。
5. 执行：

```powershell
D:\Anaconda\Scripts\conda.exe run -n PPO python trajectory_autoresearch\train.py --setup-only
```

6. 确认 `trajectory_autoresearch/_local_results/results.tsv` 已初始化。

## Ground Rules

你优先修改 `trajectory_autoresearch/train.py`，因为它是自治策略入口。

默认不要修改：

- `trajectory_autoresearch/prepare.py`
- `PPO_project/tools/acceptance_suite.py`
- `PPO_project/tools/phase32_export_best_trajectories.py`

这些文件构成了固定评测和固定归档层。只有在你明确发现评测或运行链路有 bug 时，才允许对它们做小范围修复。

可以修改的对象：

- `trajectory_autoresearch/train.py`
- 生成出的候选配置
- 在确有依据时，`PPO_project/main.py` 或 `PPO_project/src/**` 中与训练稳定性、奖励设计、策略结构直接相关的实现

## Objective

目标不是单路径刷分，而是在统一路径集合上取得更高的聚合得分，同时保留每轮的最佳轨迹图和 CSV。
当前实验是两阶段的：

- `stage1`：粗筛，低成本评测
- `stage2`：只对 top 候选做全量复评估

候选幅度 `amp` 不再依赖 keep-rate 启发式，而是基于最近 N 次完整评测的真实得分增益自动缩放。
同时，`stage1` 和 `stage2` 使用不同的聚合权重，避免用粗筛分数直接替代最终决策。

当前聚合评测包含：

- `pass_count`
- `mean_success_rate`
- `mean_progress_final`
- `mean_stall_rate`
- `mean_error_ratio`
- `max_error_ratio`
- `score`

更高的 `score` 更好。

## Experiment Loop

每轮实验遵循以下流程：

1. 读取当前最优状态与 `_local_results/results.tsv`
2. 基于当前最优配置提出一批候选，并根据最近 N 次完整评测的真实得分增益自动调节每个候选的调参幅度
3. 运行：

```powershell
D:\Anaconda\Scripts\conda.exe run -n PPO python trajectory_autoresearch\train.py --max-experiments 1
```

4. 训练完成后检查：
   - `trajectory_autoresearch/_local_results/runs/<experiment_id>/train.log`
   - `trajectory_autoresearch/_local_results/runs/<experiment_id>/evaluation/stage1/summary.json`
   - `trajectory_autoresearch/_local_results/runs/<experiment_id>/evaluation/stage2/summary.json`
   - `trajectory_autoresearch/_local_results/runs/<experiment_id>/best_rollouts/summary.json`
5. 将结果写入 `_local_results/results.tsv`
6. 如果得分变好，则晋升为新的当前最优；否则丢弃并继续下一轮

## Never Stop Rule

当用户明确要求“持续跑”“一直优化”时，不要在每一轮后停下来等待确认。你应该持续迭代，直到用户手动中断。

## Practical Guidance

- 优先做小而可解释的改动
- 先从配置层搜索开始，再逐步进入算法层
- 对每次保留的改动，确保能解释它为什么改善了成功率、进度、轨迹误差或 stall 行为
- 复杂改动必须通过固定评测，不允许只看训练回报
