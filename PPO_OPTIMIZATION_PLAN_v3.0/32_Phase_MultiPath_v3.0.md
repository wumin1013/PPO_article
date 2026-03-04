# Phase 32：多路径验证（v3.0, 更新版）
版本日期：2026-03-03  
依赖：Phase 31 主线模型

---

## 目标

验证策略在多几何形状上的泛化能力，并输出每条路径的最佳轨迹可视化。

---

## 路径集合（本阶段执行标准）

| 路径类型 | 结构特征 | 目标尺度 |
|----------|----------|----------|
| 方形（square） | 90°直角拐弯 | scale=22 |
| S形（s_shape） | 连续曲率 + 曲率反转 | scale=26 |
| 蝴蝶形（butterfly, academic） | 长短直线交叉 + 两侧弧线（非简单8字） | scale=40 |
| 梯形（trapezoid） | 非等角折线（含锐/钝角） | scale=24 |
| 圆形（circle） | 全程连续曲率 | scale=24 |

说明：  
`butterfly` 默认采用 `style=academic`，用于学术验证常见的“长短直线交叉弧线”场景，不再使用简单 8 字作为主验证路径。

---

## 训练配置

使用：

- `PPO_project/configs/archive/legacy/train_square_v32_learnable_lookahead.yaml`

关键约束：

- `training.num_episodes >= 200`（当前设为 220）
- `training.path_curriculum.enabled = true`
- `training.path_curriculum.paths` 包含上述 5 种路径
- `environment.max_steps = 8000`（修复复杂路径回合长度不足风险）
- `reward_weights.p4.stall_*` 与 `reward_weights.p7_3.stall_cap_low` 已重标定，避免低速可行运动被误判 stall
- `reward_weights.p8.use_recovery_cap = false`、`ang_cap_min_ratio = 0.12`，避免转弯限速过低导致“卡住”
- `path_curriculum.episodes_per_path = 2`，降低每回合切换路径带来的学习震荡

---

## 执行命令

```powershell
cd PPO_project
conda run -n PPO python main.py --mode train --config configs/archive/legacy/train_square_v32_learnable_lookahead.yaml
```

---

## 输出要求

1. 每条路径输出最佳轨迹图（PNG）  
2. 每条路径输出轨迹点与速度（CSV）  
3. 汇总 JSON（每条路径的 reward/progress/steps 与最优模型来源）

---

## 验收标准（Phase 32）

| 指标 | 条件 |
|------|------|
| 训练回合数 | `num_episodes >= 200` |
| 路径覆盖 | 覆盖 square/s_shape/butterfly/trapezoid/circle |
| 可视化产物 | 每条路径至少 1 张最佳轨迹图 |
| 结果追踪 | 提供 summary JSON |

---

## 问题诊断基线（本轮改动依据）

在旧配置下，多路径 rollout 的主要终止原因为 `stall`（非 `max_steps`）：

- 蝴蝶路径：`turn_angle_max≈140.8°`，`v_ratio_cap_ang` 长时间压低至约 `0.083`
- 多路径均出现 `done_reason=stall`

因此本阶段优先修复：
1. 蝴蝶几何中中心重叠段与过尖折角  
2. stall 误判阈值与过严转弯限速配置
