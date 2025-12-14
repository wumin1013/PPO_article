# P1：恢复旧版动作语义 + 修 action mismatch + parity_check（必做）
> 前置条件：P0 已通过。  
> 本阶段只做“让新 Env 行为回到旧版”。**不要做内切、不要加新奖励、不要扩状态维度**。

## 目标（Scope）
1) 新 Env 的动作语义恢复为旧版：相对路径切向的微调（而非累计航向积分）；  
2) 修复 action mismatch（执行动作与 PPO 更新用动作一致）；  
3) 新增 `tests/parity_check.py`：旧版 Env vs 新版 Env 行为一致性回归测试。

## 允许改动的文件
- `src/environment/cnc_env.py`（动作语义/step 内执行动作）
- `src/algorithms/ppo.py`（若需要修 log_prob 与 action 的一致性）
- 新增：`legacy_env.py`（从旧版脚本抽取 Env 核心）
- 新增：`tests/parity_check.py`

## 禁止改动（本阶段不要碰）
- observation/state 维度（lookahead_points 等先不动）
- “内切走廊”或任何拐角新策略
- 速度目标、停滞惩罚等 P4 内容

## 任务 1：恢复动作语义（关键）
必须实现旧版核心：
- `path_angle = _get_path_direction(current_position)`
- `effective_angle = path_angle + theta_prime * dt`
- `displacement = length_prime * dt`

要求：
- 删除/禁用“累计航向积分”式替代实现；
- dt 必须使用 YAML 生效的 `interpolation_period`。

## 任务 2：修复 action mismatch（policy_action / exec_action / log_prob 对齐）
若环境内部对 action 做了 clip/KCM/安全过滤，但 PPO 更新用的 log_prob 却对应另一套动作，就会出现系统性“梯度错配”，训练会像在优化一个并未真正执行的世界。

**关键原则（强制）：PPO 的 log_prob 必须对应“策略采样并送入环境的动作”（action_policy）。**  
环境内部经 KCM/安全过滤得到的 `action_exec` 只能用于执行与诊断，不允许偷偷替换成 PPO 更新用动作。

要求（强制落地）：
1) 在 `env.step()` 中显式记录两套动作（通过 `info` 返回或日志打印均可）：
   - `action_policy`：策略输出经缩放/裁剪到 action_space 后、送入 env 的动作
   - `action_exec`：经过 KCM/安全过滤后，实际用于更新状态的动作
2) rollout/buffer **只存 `action_policy`**，并用它计算/存 `log_prob`（PPO 更新只看这一套）。
3) 额外输出 `|action_exec - action_policy|` 的统计（mean/max 或分位数），用来判断 KCM 是否过强或配置漂移。

实现路线二选一（优先 A）：
A) **最推荐**：把“硬边界裁剪”前移到策略输出侧（只做 action_space 边界裁剪），env 内只做 KCM/soft constraint 产出 `action_exec`；PPO 始终用 `action_policy` 做更新。  
B) 若你要用 tanh-squash 等有界分布：在 policy 中实现 squashed Gaussian，并用“有界后的 `action_policy`”计算 log_prob；env 仍可在 KCM 后得到 `action_exec`，但**禁止**用 `action_exec` 反算/替换 log_prob。

## 任务 3：新增 parity_check（行为一致性回归测试）
### legacy_env.py
- 从 `PPO最终版_改进.py` 抽取 Env 核心（reset/step/reward/done/辅助函数）。
- 不要带训练代码；保证可被测试脚本 import。

### tests/parity_check.py
- 固定 seed；同一路径 Pm；同一动作序列（可随机但固定随机种子）；
- 对比每一步：position/progress/reward；
- 输出：max error、mean error、分位数（p50/p90/p99）。

## 自验证/验收标准（你将这样验证）
### 验证 1：parity_check 必须能跑
运行 `tests/parity_check.py`：
- **验收：** 脚本运行成功并输出误差统计；误差不随步数呈爆炸增长。

### 验证 2：训练冒烟（可选但建议）
用直线/简单路径训练 50~100 episode：
- **验收：** reward 不再长期贴“坏平台”，progress 有明显上升趋势。

## 交付物（提交时必须包含）
1) `legacy_env.py` + `tests/parity_check.py`  
2) 动作语义改动说明  
3) action mismatch 修复说明  
4) parity_check 输出样例（误差统计）
